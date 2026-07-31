"""Primary Repository Intelligence page."""

from __future__ import annotations

from html import escape
from typing import Dict, Sequence

import pandas as pd
import streamlit as st

from ai.ai_service import AIService, IntelligenceReport, RepositoryHealth, RepositoryProfile

from .insight_cards import render_insight_cards
from .module_explorer import render_module_explorer
from .components import (
    DEFAULT_PROGRESS_MESSAGES,
    empty_state,
    metric_card,
    page_header,
    panel_note,
    run_with_progress_messages,
    section_header,
)
from .report_exporter import ReportExporter
from .report_viewer import render_report_sections


ACTION_SPECS = (
    ("repository_summary", "Generate Repository Summary"),
    ("architecture_overview", "Generate Architecture Overview"),
    ("onboarding_guide", "Generate Developer Onboarding Guide"),
    ("explain_main_modules", "Explain Main Modules"),
    ("folder_structure", "Explain Folder Structure"),
    ("core_components", "Identify Core Components"),
    ("learning_path", "Generate Learning Path for New Developers"),
)


RETRIEVAL_MODE_OPTIONS = {
    "fast": "Fast",
    "deep": "Deep Analysis",
}


FIRST_QUESTION_SUGGESTIONS = (
    "What are the main modules?",
    "Explain the architecture.",
    "What should a new developer understand first?",
    "Where is authentication implemented?",
)


def navigate_to_page(page: str, repository_name: str | None = None) -> None:
    st.session_state["active_page"] = page
    if repository_name:
        st.session_state["selected_repository_name"] = repository_name
    st.rerun()


def ensure_dashboard_state(repository_name: str) -> Dict[str, IntelligenceReport]:
    state = st.session_state.setdefault("repository_intelligence_reports", {})
    return get_or_create_report_bucket(state, repository_name)


def get_or_create_report_bucket(
    state: Dict[str, Dict[str, IntelligenceReport]],
    repository_name: str,
) -> Dict[str, IntelligenceReport]:
    return state.setdefault(repository_name, {})


def is_repository_indexed(repository: Dict[str, object]) -> bool:
    index_stats = repository.get("index_stats", {}) if isinstance(repository.get("index_stats"), dict) else {}
    return bool(repository.get("indexed_at")) and int(index_stats.get("embeddings_stored", 0) or 0) > 0


def build_custom_question_report_key(question: str) -> str:
    normalized = "_".join((question or "").strip().lower().split())
    safe = "".join(char for char in normalized if char.isalnum() or char in {"_", "-"})
    return f"ask_repository:{safe[:48] or 'question'}"


def build_overview_metrics(profile: RepositoryProfile) -> Sequence[Dict[str, object]]:
    primary_language = "Unknown"
    if profile.language_distribution:
        primary_language = next(iter(profile.language_distribution.keys()))
    return (
        {"label": "Files", "value": profile.total_files, "help": "Supported source files"},
        {"label": "Chunks", "value": profile.total_chunks, "help": "Parsed retrieval units"},
        {"label": "Embeddings", "value": profile.embeddings_stored, "help": "Vectors stored"},
        {"label": "Primary Language", "value": primary_language, "help": "Largest indexed language"},
        {"label": "Repository Size", "value": profile.repository_size, "help": "Source footprint"},
        {"label": "Last Indexed", "value": profile.indexed_at or "Not indexed", "help": "Metadata timestamp"},
    )


def build_health_tables(health: RepositoryHealth) -> Dict[str, pd.DataFrame]:
    return {
        "largest_modules": pd.DataFrame(list(health.largest_modules)),
        "most_referenced_files": pd.DataFrame(list(health.most_referenced_files)),
        "most_connected_classes": pd.DataFrame(list(health.most_connected_classes)),
        "configuration_files": pd.DataFrame([{"file_path": item} for item in health.configuration_files]),
        "documentation_coverage": pd.DataFrame([health.documentation_coverage]),
    }


def render_repository_intelligence_page() -> None:
    page_header(
        "Repository Intelligence",
        "Explore architecture, modules, technologies, and developer insights from an indexed repository.",
    )
    repositories = AIService.list_available_repositories()
    if not repositories:
        empty_state(
            "No repositories available yet.",
            "Add a repository, index it, then return here to ask questions and generate reports.",
            "Use Add Repository, then Index Repository.",
        )
        if st.button("Add Repository", type="primary", use_container_width=True):
            navigate_to_page("Add Repository")
        return

    repo_options = {repo["repo_name"]: repo for repo in repositories if repo.get("repo_name")}
    section_header("Repository Selector", "Choose the indexed repository you want to explore.")
    repo_names = list(repo_options.keys())
    carried_repo = st.session_state.get("selected_repository_name")
    selected_index = repo_names.index(carried_repo) if carried_repo in repo_names else 0
    selected_name = st.selectbox(
        "Repository",
        options=repo_names,
        index=selected_index,
        key="repository_intelligence_selected_name",
        label_visibility="collapsed",
        help="Switch repositories without carrying over reports from another repository.",
    )
    st.session_state["selected_repository_name"] = selected_name
    selected_repo = repo_options[selected_name]
    previous_repo = st.session_state.get("repository_intelligence_active_repo")
    if previous_repo != selected_name:
        st.session_state["repository_intelligence_active_repo"] = selected_name
        st.session_state["repository_intelligence_selected_report"] = None
    ai_service = AIService.for_repository(selected_name)
    reports_state = ensure_dashboard_state(selected_name)
    session_id = f"repository-intelligence:{selected_name}"
    retrieval_mode = render_retrieval_mode_control(selected_name)
    repository_is_indexed = is_repository_indexed(selected_repo)

    profile = ai_service.generate_repository_profile()
    health = ai_service.generate_repository_health()

    render_repository_overview(ai_service, profile, reports_state, session_id, retrieval_mode)

    if not repository_is_indexed:
        empty_state(
            "Repository not indexed yet.",
            "RepoMind can show basic metadata, but repository questions need embeddings and retrieval indexes first.",
            "Open Index Repository to prepare this codebase.",
        )
        if st.button("Index Repository", type="primary", use_container_width=True):
            navigate_to_page("Index Repository", selected_name)

    render_ask_repository(
        ai_service=ai_service,
        repository_name=selected_name,
        repository_is_indexed=repository_is_indexed,
        session_id=session_id,
        reports_state=reports_state,
        retrieval_mode=retrieval_mode,
    )

    section_header("Quick AI Actions", "Choose an action to generate grounded repository insights.")
    selected_report = render_insight_cards(
        ai_service,
        session_id,
        reports_state,
        key_prefix=f"insights_{selected_name}",
        retrieval_mode=retrieval_mode,
    )
    if selected_report:
        st.session_state["repository_intelligence_selected_report"] = selected_report.action_key

    section_header("Developer Actions", "Run focused reports for common onboarding and architecture questions.")
    render_developer_actions(ai_service, session_id, reports_state, selected_name, retrieval_mode)

    section_header("Module Explorer", "Search for a file, folder, class, or module and analyze it on demand.")
    render_module_explorer(
        ai_service,
        session_id,
        reports_state,
        key_prefix=f"module_explorer_{selected_name}",
        retrieval_mode=retrieval_mode,
    )

    section_header("Repository Health", "Fast metadata diagnostics. Deep retrieval analysis is only run from explicit actions.")
    render_repository_health(health)

    section_header("Generated Report", "Review generated reports and export them when ready.")
    render_generated_report(reports_state, selected_name)


def render_retrieval_mode_control(repository_name: str) -> str:
    section_header("Retrieval Mode", "Choose speed or depth for generated answers.")
    mode_cols = st.columns(2)
    mode_descriptions = {
        "fast": ("Fast", "Hybrid retrieval", "Recommended for interactive questions."),
        "deep": ("Deep Analysis", "Hybrid + CrossEncoder", "Higher quality. Slower on CPU."),
    }
    for column, mode_key in zip(mode_cols, ("fast", "deep")):
        title, method, caption = mode_descriptions[mode_key]
        with column:
            st.markdown(
                f"""
                <div class="repomind-mode-card">
                    <strong>{escape(title)}</strong>
                    <p>{escape(method)}</p>
                    <span>{escape(caption)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    selected = st.radio(
        "Select retrieval mode",
        options=list(RETRIEVAL_MODE_OPTIONS.keys()),
        format_func=lambda value: RETRIEVAL_MODE_OPTIONS[value],
        index=0,
        horizontal=True,
        key=f"retrieval_mode_{repository_name}",
        help="Fast uses hybrid retrieval without CrossEncoder reranking. Deep Analysis adds CrossEncoder reranking and is slower on CPU.",
    )
    if selected == "deep":
        st.caption("Deep Analysis uses CrossEncoder reranking and may take noticeably longer on CPU.")
    else:
        st.caption("Fast mode uses hybrid retrieval and is recommended for interactive analysis.")
    return selected


def render_repository_overview(
    ai_service: AIService,
    profile: RepositoryProfile,
    reports_state: Dict[str, IntelligenceReport],
    session_id: str,
    retrieval_mode: str = "fast",
) -> None:
    section_header("Repository Overview", "Index coverage, language balance, and quick architecture context.")
    metric_rows = build_overview_metrics(profile)
    columns = st.columns(3)
    for index, metric in enumerate(metric_rows):
        with columns[index % 3]:
            metric_card(metric["label"], metric["value"], str(metric.get("help", "")))

    chart_col, summary_col = st.columns([0.52, 0.48])
    with chart_col:
        if profile.language_distribution:
            st.markdown("**Language Distribution**")
            st.bar_chart(profile.language_distribution)
        else:
            empty_state("No language chart available.", "Index metadata does not include a language distribution yet.")

    with summary_col:
        architecture_report = reports_state.get("architecture_overview")
        st.markdown("**Architecture Summary**")
        if architecture_report:
            st.write(architecture_report.summary)
        else:
            panel_note(
                "No architecture summary yet.",
                "Generate an overview to pin a high-level reading of this repository here.",
            )
        if st.button(
            "Generate",
            key=f"overview_architecture_{profile.repository_name}",
            use_container_width=True,
            help="Generate an architecture overview report for the selected repository.",
        ):
            reports_state["architecture_overview"] = run_with_progress_messages(
                "Generating architecture summary...",
                DEFAULT_PROGRESS_MESSAGES,
                lambda: ai_service.generate_architecture_overview(
                    session_id=session_id,
                    retrieval_mode=retrieval_mode,
                ),
                "Architecture summary generated.",
            )


def render_developer_actions(
    ai_service: AIService,
    session_id: str,
    reports_state: Dict[str, IntelligenceReport],
    repository_name: str,
    retrieval_mode: str = "fast",
) -> None:
    for row_start in range(0, len(ACTION_SPECS), 2):
        columns = st.columns(2)
        for column, (action_key, label) in zip(columns, ACTION_SPECS[row_start : row_start + 2]):
            with column:
                if st.button(
                    label,
                    key=f"action_{repository_name}_{action_key}",
                    use_container_width=True,
                    help=f"Run: {label}",
                ):
                    reports_state[action_key] = run_with_progress_messages(
                        f"{label}...",
                        DEFAULT_PROGRESS_MESSAGES,
                        lambda action_key=action_key: ai_service.run_action(
                            action_key,
                            session_id=session_id,
                            retrieval_mode=retrieval_mode,
                        ),
                        "Report generated.",
                    )


def render_ask_repository(
    ai_service: AIService,
    repository_name: str,
    repository_is_indexed: bool,
    session_id: str,
    reports_state: Dict[str, IntelligenceReport],
    retrieval_mode: str = "fast",
) -> None:
    section_header("Ask Repository", "The fastest way to interrogate the selected codebase with grounded evidence.")
    st.markdown(
        """
        <div class="repomind-ask-intro">
            <h3>Ask RepoMind anything about this repository</h3>
            <p>Use natural language to ask about architecture, behavior, files, classes, functions, or implementation details.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    question_key = f"ask_repository_question_{repository_name}"
    with st.container(border=True):
        if repository_is_indexed and not reports_state:
            render_first_question_suggestions(question_key, repository_name)

        question = st.text_area(
            "Question for RepoMind",
            key=question_key,
            placeholder="Ask about architecture, data flow, key modules, or a specific implementation detail...",
            height=126,
            help="RepoMind answers using indexed repository evidence.",
            label_visibility="collapsed",
        )
        stripped_question = (question or "").strip()
        disabled = not stripped_question or not repository_name or not repository_is_indexed
        if not repository_is_indexed:
            st.warning("Index this repository before asking RepoMind custom questions.")
        elif not stripped_question:
            empty_state(
                "No question asked yet.",
                "Type a repository question above to generate a grounded answer with citations.",
            )

        button_col, hint_col = st.columns([0.32, 0.68])
        with button_col:
            ask_clicked = st.button(
                "Ask RepoMind",
                key=f"ask_repository_button_{repository_name}",
                disabled=disabled,
                type="primary",
                use_container_width=True,
                help="Generate a grounded answer from the selected repository.",
            )
        with hint_col:
            st.caption("Answers use the selected retrieval mode and appear in Generated Report.")

        if ask_clicked:
            report = run_with_progress_messages(
                "RepoMind is reading the indexed repository evidence...",
                DEFAULT_PROGRESS_MESSAGES,
                lambda: ai_service.ask_repository_question(
                    stripped_question,
                    session_id=session_id,
                    retrieval_mode=retrieval_mode,
                ),
                "Answer generated.",
            )
            report_key = build_custom_question_report_key(stripped_question)
            reports_state[report_key] = report
            st.session_state["repository_intelligence_selected_report"] = report_key
            st.success("Answer generated.")


def render_first_question_suggestions(question_key: str, repository_name: str) -> None:
    st.markdown('<div class="suggestion-label">Suggested first questions</div>', unsafe_allow_html=True)
    columns = st.columns(2)
    for index, suggestion in enumerate(FIRST_QUESTION_SUGGESTIONS):
        with columns[index % 2]:
            if st.button(
                suggestion,
                key=f"first_question_{repository_name}_{index}",
                use_container_width=True,
                help="Populate the Ask RepoMind question box.",
            ):
                st.session_state[question_key] = suggestion
                st.rerun()


def render_repository_health(health: RepositoryHealth) -> None:
    tabs = st.tabs(["Languages", "Files", "Configuration", "Documentation", "Deep Analysis"])
    tables = build_health_tables(health)

    with tabs[0]:
        if health.language_distribution:
            st.bar_chart(health.language_distribution)
        else:
            st.caption("No language distribution available.")
    with tabs[1]:
        if not tables["largest_modules"].empty:
            st.dataframe(tables["largest_modules"], use_container_width=True)
        else:
            st.caption("No module size information available.")
    with tabs[2]:
        if not tables["configuration_files"].empty:
            st.dataframe(tables["configuration_files"], use_container_width=True)
        else:
            st.caption("No configuration files detected.")
    with tabs[3]:
        st.dataframe(tables["documentation_coverage"], use_container_width=True)
    with tabs[4]:
        st.caption("Retrieval-enriched health analysis is intentionally not run during initial page load.")
        if not tables["most_referenced_files"].empty:
            st.dataframe(tables["most_referenced_files"], use_container_width=True)
        if not tables["most_connected_classes"].empty:
            st.dataframe(tables["most_connected_classes"], use_container_width=True)


def render_generated_report(reports_state: Dict[str, IntelligenceReport], repository_name: str) -> None:
    if not reports_state:
        empty_state(
            "No report generated yet.",
            "Ask RepoMind or choose an AI action above to generate grounded repository insights.",
            "Generated reports, citations, and downloads will appear here.",
        )
        return

    report_keys = list(reports_state.keys())
    preferred_key = st.session_state.get("repository_intelligence_selected_report")
    selected_index = report_keys.index(preferred_key) if preferred_key in report_keys else 0
    selected_key = st.selectbox(
        "Generated Reports",
        options=report_keys,
        index=selected_index,
        format_func=lambda key: reports_state[key].title,
        key=f"generated_report_{repository_name}",
        help="Choose which generated report to review or export.",
    )
    report = reports_state[selected_key]
    render_report_sections(report, key_prefix=f"report_{repository_name}_{selected_key}")

    exporter = ReportExporter()
    export_cols = st.columns(3)
    with export_cols[0]:
        st.download_button(
            "Download Markdown",
            data=exporter.to_markdown(report),
            file_name=f"{repository_name}_{selected_key}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with export_cols[1]:
        st.download_button(
            "Download JSON",
            data=exporter.to_json(report),
            file_name=f"{repository_name}_{selected_key}.json",
            mime="application/json",
            use_container_width=True,
        )
    with export_cols[2]:
        st.download_button(
            "Download PDF",
            data=exporter.to_pdf_bytes(report),
            file_name=f"{repository_name}_{selected_key}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
