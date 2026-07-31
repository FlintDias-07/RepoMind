import json
from pathlib import Path

from ai.ai_service import AIService, IntelligenceReport, RepositoryHealth, RepositoryProfile
from ai.citation_manager import Citation
from git import Repo
from github_loader import RepositoryManager
from ui.insight_cards import get_insight_card_specs
from ui.module_explorer import filter_module_options
from ui.report_exporter import ReportExporter
from ui.report_viewer import format_citation, report_to_sections
from ui.repository_dashboard import (
    RETRIEVAL_MODE_OPTIONS,
    build_custom_question_report_key,
    build_health_tables,
    build_overview_metrics,
    get_or_create_report_bucket,
    is_repository_indexed,
)


def make_report() -> IntelligenceReport:
    citation = Citation(
        repository_name="sample-repo",
        file_path="src/auth/AuthService.java",
        file_name="AuthService.java",
        class_name="AuthService",
        function_name="authenticate",
        start_line=12,
        end_line=38,
    )
    return IntelligenceReport(
        title="Repository Summary",
        action_key="repository_summary",
        repository_name="sample-repo",
        summary="AuthService coordinates authentication.",
        detailed_explanation="AuthService validates credentials and delegates token creation.",
        repository_insights=("Evidence spans security and auth modules.",),
        referenced_files=("src/auth/AuthService.java",),
        referenced_classes=("AuthService",),
        referenced_functions=("authenticate",),
        confidence_level="high",
        warnings=(),
        citations=(citation,),
        rendered_markdown="## Summary\nAuthService coordinates authentication.",
        raw_answer="AuthService coordinates authentication.",
    )


def test_report_exporter_supports_markdown_json_and_pdf():
    report = make_report()
    exporter = ReportExporter()

    markdown = exporter.to_markdown(report)
    json_payload = exporter.to_json(report)
    pdf_payload = exporter.to_pdf_bytes(report)

    assert "# Repository Summary" in markdown
    assert json.loads(json_payload)["title"] == "Repository Summary"
    assert pdf_payload.startswith(b"%PDF-1.4")


def test_report_viewer_sections_are_structured():
    report = make_report()
    sections = report_to_sections(report)

    assert sections["summary"] == report.summary
    assert sections["confidence"] == "high"
    assert sections["files"] == ["src/auth/AuthService.java"]


def test_report_viewer_falls_back_when_summary_is_heading_only():
    report = IntelligenceReport(
        title="Question: How is face recognition implemented?",
        action_key="ask_repository",
        repository_name="sample-repo",
        summary="1. Summary",
        detailed_explanation="Face recognition uses detection and embedding comparison.",
        repository_insights=(),
        referenced_files=("backend/recognition.py",),
        referenced_classes=(),
        referenced_functions=("recognize_face",),
        confidence_level="high",
        warnings=(),
        citations=(),
        rendered_markdown="",
        raw_answer="1. Summary\nFace recognition uses detection and embedding comparison.",
    )

    sections = report_to_sections(report)

    assert sections["summary"] == "Face recognition uses detection and embedding comparison."
    assert sections["summary"] != "1. Summary"


def test_format_citation_uses_repository_relative_path_and_line_range():
    citation = Citation(
        repository_name="sample-repo",
        file_path="backend/recognition.py",
        file_name="recognition.py",
        class_name="",
        function_name="recognize_face",
        start_line=42,
        end_line=86,
    )

    assert format_citation(citation) == "backend/recognition.py — lines 42-86"


def test_overview_and_health_helpers_shape_dashboard_data():
    profile = RepositoryProfile(
        repository_name="sample-repo",
        repo_url="https://github.com/example/sample-repo.git",
        local_path="repositories/sample-repo",
        indexed_at="2026-07-10T20:00:00",
        total_files=120,
        total_chunks=980,
        embeddings_stored=980,
        repository_size="12 MB",
        language_distribution={"PY": 45, "JAVA": 30},
        largest_files=({"path": "src/auth/AuthService.java", "size": "24 KB"},),
        documentation_files=("README.md",),
    )
    health = RepositoryHealth(
        language_distribution={"PY": 45, "JAVA": 30},
        largest_modules=({"path": "src/auth/AuthService.java", "size": "24 KB"},),
        most_referenced_files=({"file_path": "src/auth/AuthService.java", "references": 4},),
        most_connected_classes=({"class_name": "AuthService", "connections": 3},),
        configuration_files=("config/application.yml",),
        documentation_coverage={"documentation_files": 3, "total_files": 120, "coverage_ratio": 0.025},
    )

    metrics = build_overview_metrics(profile)
    tables = build_health_tables(health)

    assert any(item["label"] == "Files" for item in metrics)
    assert any(item["label"] == "Primary Language" for item in metrics)
    assert not tables["largest_modules"].empty
    assert not tables["documentation_coverage"].empty


def test_insight_specs_and_module_filter_support_ui_interactions():
    specs = get_insight_card_specs()
    filtered = filter_module_options(
        ("src/auth/AuthService.java", "src/security/JwtUtil.java", "README.md"),
        "auth",
    )

    assert any(spec.key == "repository_summary" for spec in specs)
    assert filtered == ("src/auth/AuthService.java",)


def test_repository_intelligence_defaults_to_fast_retrieval_mode():
    assert list(RETRIEVAL_MODE_OPTIONS.keys())[0] == "fast"
    assert RETRIEVAL_MODE_OPTIONS["deep"] == "Deep Analysis"


def test_custom_question_helpers_validate_indexed_state_and_stable_keys():
    indexed = {"indexed_at": "2026-07-13T14:00:00", "index_stats": {"embeddings_stored": 12}}
    not_indexed = {"indexed_at": "", "index_stats": {"embeddings_stored": 0}}

    assert is_repository_indexed(indexed) is True
    assert is_repository_indexed(not_indexed) is False
    assert build_custom_question_report_key("How is face recognition implemented?").startswith("ask_repository:")
    assert " " not in build_custom_question_report_key("How is face recognition implemented?")


def test_report_bucket_state_is_repository_scoped():
    state = {}

    repo_a = get_or_create_report_bucket(state, "repo-a")
    repo_b = get_or_create_report_bucket(state, "repo-b")
    repo_a["repository_summary"] = make_report()

    assert "repo-a" in state and "repo-b" in state
    assert "repository_summary" in state["repo-a"]
    assert state["repo-b"] == {}


def test_custom_question_answers_are_scoped_by_repository_bucket():
    state = {}
    repo_a = get_or_create_report_bucket(state, "SmartAttendanceAI")
    repo_b = get_or_create_report_bucket(state, "OtherRepo")
    key = build_custom_question_report_key("How is face recognition implemented?")

    repo_a[key] = make_report()

    assert key in state["SmartAttendanceAI"]
    assert repo_b == {}


def test_repository_health_initial_render_path_does_not_trigger_retrieval(monkeypatch):
    class ExplodingVectorService:
        def search(self, query, top_k=5):
            raise AssertionError("initial repository health should not run retrieval")

    monkeypatch.setattr(
        AIService,
        "_load_repository_metadata",
        lambda self: {
            "repo_name": "sample-repo",
            "stats": {
                "total_files": 2,
                "total_size": "10 KB",
                "files_by_extension": {".py": 1, ".md": 1},
                "largest_files": [{"path": "src/app.py", "size": "8 KB"}],
            },
            "index_stats": {
                "chunks_generated": 3,
                "embeddings_stored": 3,
                "chunks_by_language": {"python": 3},
            },
        },
    )
    monkeypatch.setattr(AIService, "_configuration_files_from_metadata", lambda self: [])

    service = AIService(
        repository_name="sample-repo",
        vector_service=ExplodingVectorService(),
        llm_provider=object(),
        pipeline=object(),
    )

    health = service.generate_repository_health()

    assert health.language_distribution == {"python": 3}
    assert health.most_referenced_files == ()
    assert health.most_connected_classes == ()


def test_about_page_copy_does_not_mark_completed_features_as_coming_soon():
    text = Path("app.py").read_text(encoding="utf-8")

    assert "RepoMind AI" in text
    assert "Coming Soon" not in text
    assert "Vector + BM25 Retrieval" in text


def test_primary_ui_modules_do_not_contain_obvious_mojibake():
    suspicious = ("ð", "â", "Ã", "¢", "Å")
    paths = [
        Path("app.py"),
        Path("config.py"),
        Path("ui/repository_dashboard.py"),
        Path("ui/insight_cards.py"),
        Path("ui/module_explorer.py"),
        Path("evaluation/dashboard.py"),
    ]

    offenders = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in suspicious):
            offenders.append(str(path))

    assert offenders == []


def test_repository_manager_recovers_valid_git_repository_metadata(tmp_path):
    repo_path = tmp_path / "RecoveredRepo"
    repo_path.mkdir()
    Repo.init(str(repo_path))
    (repo_path / "README.md").write_text("# RecoveredRepo", encoding="utf-8")

    manager = RepositoryManager(base_path=str(tmp_path))
    unmanaged = manager.get_unmanaged_git_repositories()

    assert [repo["repo_name"] for repo in unmanaged] == ["RecoveredRepo"]

    success, message = manager.recover_repository_metadata("RecoveredRepo")

    assert success is True
    assert "RecoveredRepo" in message
    assert (repo_path / ".rag_metadata.json").exists()
    assert manager.get_unmanaged_git_repositories() == []
