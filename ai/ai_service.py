"""High-level entry points for AI repository intelligence features."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from config import REPOSITORIES_DIR
from github_loader import RepositoryManager
from utils import load_metadata
from vectorstore.vector_service import VectorService

from .ai_pipeline import AIPipeline, AIPipelineResult
from .llm_provider import BaseLLMProvider, LLMProviderFactory


@dataclass(frozen=True)
class RepositoryProfile:
    repository_name: str
    repo_url: str
    local_path: str
    indexed_at: str
    total_files: int
    total_chunks: int
    embeddings_stored: int
    repository_size: str
    language_distribution: Dict[str, int] = field(default_factory=dict)
    largest_files: Sequence[Dict[str, object]] = field(default_factory=tuple)
    documentation_files: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class RepositoryHealth:
    language_distribution: Dict[str, int]
    largest_modules: Sequence[Dict[str, object]]
    most_referenced_files: Sequence[Dict[str, object]]
    most_connected_classes: Sequence[Dict[str, object]]
    configuration_files: Sequence[str]
    documentation_coverage: Dict[str, object]


@dataclass(frozen=True)
class IntelligenceReport:
    title: str
    action_key: str
    repository_name: str
    summary: str
    detailed_explanation: str
    repository_insights: Sequence[str]
    referenced_files: Sequence[str]
    referenced_classes: Sequence[str]
    referenced_functions: Sequence[str]
    confidence_level: str
    warnings: Sequence[str]
    citations: Sequence[object]
    rendered_markdown: str
    raw_answer: str
    diagnostics: Dict[str, object] = field(default_factory=dict)


def _report_from_pipeline_result(title: str, action_key: str, result: AIPipelineResult) -> IntelligenceReport:
    formatted = result.formatted_response
    diagnostics = {
        "retrieval_mode": result.diagnostics.retrieval_mode,
        "strategy_name": result.diagnostics.strategy_name,
        "reranking_used": result.diagnostics.reranking_used,
        "retrieval_time_seconds": result.diagnostics.retrieval_time_seconds,
        "reranking_time_seconds": result.diagnostics.reranking_time_seconds,
        "llm_generation_time_seconds": result.diagnostics.llm_generation_time_seconds,
        "total_response_time_seconds": result.diagnostics.total_response_time_seconds,
        "bm25_rebuilt": result.diagnostics.bm25_rebuilt,
        "bm25_document_count": result.diagnostics.bm25_document_count,
    }
    return IntelligenceReport(
        title=title,
        action_key=action_key,
        repository_name=result.context.repository.repository_name,
        summary=formatted.summary,
        detailed_explanation=formatted.detailed_explanation,
        repository_insights=formatted.repository_insights,
        referenced_files=formatted.referenced_files,
        referenced_classes=formatted.referenced_classes,
        referenced_functions=formatted.referenced_functions,
        confidence_level=formatted.confidence_level,
        warnings=formatted.warnings,
        citations=formatted.citations,
        rendered_markdown=formatted.rendered_markdown,
        raw_answer=formatted.raw_answer,
        diagnostics=diagnostics,
    )


def _question_title(question: str, max_length: int = 80) -> str:
    cleaned = " ".join((question or "").strip().split())
    if len(cleaned) > max_length:
        cleaned = cleaned[: max_length - 3].rstrip() + "..."
    return f"Question: {cleaned}" if cleaned else "Question"


class AIService:
    """Stable service surface for future UI pages and workflows."""

    def __init__(
        self,
        repository_name: str,
        vector_service: Optional[object] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        pipeline: Optional[AIPipeline] = None,
        provider_factory: Optional[LLMProviderFactory] = None,
    ):
        self.repository_name = repository_name
        vector_repository = getattr(vector_service, "repository_name", repository_name) if vector_service is not None else repository_name
        if vector_repository != repository_name:
            raise ValueError(
                f"Vector service repository mismatch: expected '{repository_name}', got '{vector_repository}'"
            )
        self.vector_service = vector_service or VectorService(repository_name=repository_name)
        self.provider_factory = provider_factory or LLMProviderFactory()
        self.llm_provider = llm_provider or self.provider_factory.create()
        self.pipeline = pipeline or AIPipeline(vector_service=self.vector_service, llm_provider=self.llm_provider)

    @classmethod
    def list_available_repositories(cls) -> List[Dict[str, object]]:
        manager = RepositoryManager()
        repositories = manager.get_cloned_repositories()
        return sorted(repositories, key=lambda item: str(item.get("repo_name", "")).lower())

    @classmethod
    def for_repository(
        cls,
        repository_name: str,
        llm_provider: Optional[BaseLLMProvider] = None,
        provider_factory: Optional[LLMProviderFactory] = None,
    ) -> "AIService":
        return cls(
            repository_name=repository_name,
            vector_service=VectorService(repository_name=repository_name),
            llm_provider=llm_provider,
            provider_factory=provider_factory,
        )

    def answer_question(
        self,
        question: str,
        session_id: str = "default",
        top_k: int = 5,
        retrieval_mode: str = "fast",
    ) -> IntelligenceReport:
        result = self.pipeline.run(
            question=question,
            repository_metadata=self._load_repository_metadata(),
            session_id=session_id,
            role="software_architect",
            task_hint="Answer the user's repository question using retrieved evidence.",
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        return _report_from_pipeline_result("Question Answer", "answer_question", result)

    def ask_repository_question(
        self,
        question: str,
        session_id: str = "default",
        top_k: int = 5,
        retrieval_mode: str = "fast",
    ) -> IntelligenceReport:
        if not question or not question.strip():
            raise ValueError("question cannot be empty")

        result = self.pipeline.run(
            question=question.strip(),
            repository_metadata=self._load_repository_metadata(),
            session_id=session_id,
            role="software_architect",
            task_hint=(
                "Answer the user's custom repository question using only retrieved repository evidence. "
                "If the indexed repository does not contain evidence, state that clearly."
            ),
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        return _report_from_pipeline_result(_question_title(question), "ask_repository", result)

    def generate_summary(self, session_id: str = "default", top_k: int = 6, retrieval_mode: str = "fast") -> IntelligenceReport:
        result = self.pipeline.run(
            question="Provide a high-level summary of this repository.",
            repository_metadata=self._load_repository_metadata(),
            session_id=session_id,
            role="technical_mentor",
            task_hint="Summarize the repository purpose, core modules, and main responsibilities.",
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        return _report_from_pipeline_result("Repository Summary", "repository_summary", result)

    def generate_architecture_overview(self, session_id: str = "default", top_k: int = 8, retrieval_mode: str = "fast") -> IntelligenceReport:
        result = self.pipeline.run(
            question="Explain the repository architecture and how the major modules interact.",
            repository_metadata=self._load_repository_metadata(),
            session_id=session_id,
            role="software_architect",
            task_hint="Describe architecture, boundaries, and interactions between important components.",
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        return _report_from_pipeline_result("Architecture Overview", "architecture_overview", result)

    def generate_onboarding_guide(self, session_id: str = "default", top_k: int = 8, retrieval_mode: str = "fast") -> IntelligenceReport:
        result = self.pipeline.run(
            question="Create an onboarding guide for a new engineer joining this repository.",
            repository_metadata=self._load_repository_metadata(),
            session_id=session_id,
            role="technical_mentor",
            task_hint="Explain where a new engineer should start, key modules to learn, and likely workflows.",
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        return _report_from_pipeline_result("Developer Onboarding Guide", "onboarding_guide", result)

    def generate_module_summary(
        self,
        module_name: str,
        session_id: str = "default",
        top_k: int = 6,
        retrieval_mode: str = "fast",
    ) -> IntelligenceReport:
        result = self.pipeline.run(
            question=f"Explain the module or component named {module_name}.",
            repository_metadata=self._load_repository_metadata(),
            session_id=session_id,
            role="backend_engineer",
            task_hint=f"Focus on the module '{module_name}', its responsibilities, and how it interacts with the repository.",
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        return _report_from_pipeline_result("Module Summary", "module_summary", result)

    def generate_folder_structure_explanation(self, session_id: str = "default", top_k: int = 8, retrieval_mode: str = "fast") -> IntelligenceReport:
        result = self.pipeline.run(
            question="Explain the repository folder structure and what each major folder is responsible for.",
            repository_metadata=self._load_repository_metadata(),
            session_id=session_id,
            role="technical_mentor",
            task_hint="Describe the folder structure, responsibilities, and likely developer workflows.",
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        return _report_from_pipeline_result("Folder Structure", "folder_structure", result)

    def identify_core_components(self, session_id: str = "default", top_k: int = 8, retrieval_mode: str = "fast") -> IntelligenceReport:
        result = self.pipeline.run(
            question="Identify the core components in this repository and explain why they are central.",
            repository_metadata=self._load_repository_metadata(),
            session_id=session_id,
            role="software_architect",
            task_hint="Highlight central components, boundaries, and why they matter.",
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        return _report_from_pipeline_result("Core Components", "core_components", result)

    def generate_learning_path(self, session_id: str = "default", top_k: int = 8, retrieval_mode: str = "fast") -> IntelligenceReport:
        result = self.pipeline.run(
            question="Generate a learning path for a new developer to understand this repository.",
            repository_metadata=self._load_repository_metadata(),
            session_id=session_id,
            role="technical_mentor",
            task_hint="Provide a step-by-step learning path for a developer onboarding to this codebase.",
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        return _report_from_pipeline_result("Learning Path", "learning_path", result)

    def identify_important_modules(self, session_id: str = "default", top_k: int = 8, retrieval_mode: str = "fast") -> IntelligenceReport:
        result = self.pipeline.run(
            question="Identify the most important modules in this repository and explain their responsibilities.",
            repository_metadata=self._load_repository_metadata(),
            session_id=session_id,
            role="backend_engineer",
            task_hint="Focus on the modules that appear central to the repository.",
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        return _report_from_pipeline_result("Important Modules", "important_modules", result)

    def identify_potential_entry_points(self, session_id: str = "default", top_k: int = 8, retrieval_mode: str = "fast") -> IntelligenceReport:
        result = self.pipeline.run(
            question="Identify potential entry points in this repository for understanding runtime flow or developer workflows.",
            repository_metadata=self._load_repository_metadata(),
            session_id=session_id,
            role="software_architect",
            task_hint="Point to files, classes, or functions that are good starting points for understanding execution flow.",
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        return _report_from_pipeline_result("Potential Entry Points", "entry_points", result)

    def identify_main_technologies(self, session_id: str = "default", top_k: int = 8, retrieval_mode: str = "fast") -> IntelligenceReport:
        result = self.pipeline.run(
            question="Identify the main technologies, frameworks, and implementation patterns used in this repository.",
            repository_metadata=self._load_repository_metadata(),
            session_id=session_id,
            role="backend_engineer",
            task_hint="Summarize the main technologies and implementation patterns visible in the codebase.",
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        return _report_from_pipeline_result("Main Technologies", "main_technologies", result)

    def identify_dependencies(self, session_id: str = "default", top_k: int = 8, retrieval_mode: str = "fast") -> IntelligenceReport:
        result = self.pipeline.run(
            question="Explain the important dependencies and external integrations used by this repository.",
            repository_metadata=self._load_repository_metadata(),
            session_id=session_id,
            role="backend_engineer",
            task_hint="Focus on dependencies, libraries, and integrations that matter for understanding the repository.",
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        return _report_from_pipeline_result("Dependencies", "dependencies", result)

    def explain_main_modules(self, session_id: str = "default", top_k: int = 8, retrieval_mode: str = "fast") -> IntelligenceReport:
        return self.identify_important_modules(session_id=session_id, top_k=top_k, retrieval_mode=retrieval_mode)

    def generate_repository_profile(self) -> RepositoryProfile:
        metadata = self._load_repository_metadata()
        stats = metadata.get("stats", {}) if isinstance(metadata.get("stats"), dict) else {}
        index_stats = metadata.get("index_stats", {}) if isinstance(metadata.get("index_stats"), dict) else {}
        files_by_extension = stats.get("files_by_extension", {}) if isinstance(stats.get("files_by_extension"), dict) else {}
        language_distribution = self._normalize_language_distribution(
            index_stats.get("chunks_by_language") if isinstance(index_stats.get("chunks_by_language"), dict) else files_by_extension
        )
        documentation_files = tuple(
            item.get("path", "")
            for item in (stats.get("largest_files") or [])
            if isinstance(item, dict) and str(item.get("path", "")).lower().endswith(".md")
        )
        return RepositoryProfile(
            repository_name=str(metadata.get("repo_name") or metadata.get("repository_name") or self.repository_name),
            repo_url=str(metadata.get("repo_url", "") or ""),
            local_path=str(metadata.get("local_path", "") or ""),
            indexed_at=str(metadata.get("indexed_at", "") or "Not indexed"),
            total_files=int(stats.get("total_files", 0) or 0),
            total_chunks=int(index_stats.get("chunks_generated", 0) or 0),
            embeddings_stored=int(index_stats.get("embeddings_stored", 0) or 0),
            repository_size=str(stats.get("total_size", "Unknown") or "Unknown"),
            language_distribution=language_distribution,
            largest_files=tuple(stats.get("largest_files", []) or []),
            documentation_files=documentation_files,
        )

    def generate_repository_health(self, include_retrieval: bool = False) -> RepositoryHealth:
        profile = self.generate_repository_profile()
        metadata = self._load_repository_metadata()
        stats = metadata.get("stats", {}) if isinstance(metadata.get("stats"), dict) else {}
        files_by_extension = stats.get("files_by_extension", {}) if isinstance(stats.get("files_by_extension"), dict) else {}
        configuration_files = self._configuration_files_from_metadata()
        largest_modules = list(profile.largest_files[:5])
        most_referenced_files = self._most_referenced_files() if include_retrieval else []
        most_connected_classes = self._most_connected_classes() if include_retrieval else []
        documentation_count = sum(count for ext, count in files_by_extension.items() if str(ext).lower() in {".md", ".rst", ".txt"})
        return RepositoryHealth(
            language_distribution=profile.language_distribution,
            largest_modules=tuple(largest_modules),
            most_referenced_files=tuple(most_referenced_files),
            most_connected_classes=tuple(most_connected_classes),
            configuration_files=tuple(configuration_files),
            documentation_coverage={
                "documentation_files": documentation_count,
                "total_files": profile.total_files,
                "coverage_ratio": round((documentation_count / profile.total_files), 4) if profile.total_files else 0.0,
            },
        )

    def list_module_options(self) -> Dict[str, Sequence[str]]:
        metadata = self._load_repository_metadata()
        repo_path = Path(metadata.get("local_path", "")) if metadata.get("local_path") else Path(REPOSITORIES_DIR) / self.repository_name
        files: List[str] = []
        folders: set[str] = set()
        modules: set[str] = set()
        classes: set[str] = set()

        if repo_path.exists():
            for path in repo_path.rglob("*"):
                if path.is_file() and ".git" not in path.parts:
                    relative = str(path.relative_to(repo_path))
                    files.append(relative)
                    modules.add(path.stem)
                    if path.parent != repo_path:
                        folders.add(str(path.parent.relative_to(repo_path)))
                    if path.stem and path.suffix in {".py", ".java", ".js", ".ts", ".tsx", ".jsx"}:
                        classes.add(path.stem)

        return {
            "folders": tuple(sorted(folders)),
            "files": tuple(sorted(files)),
            "modules": tuple(sorted(modules)),
            "classes": tuple(sorted(classes)),
        }

    def run_action(self, action_key: str, session_id: str = "default", retrieval_mode: str = "fast") -> IntelligenceReport:
        action_map = {
            "repository_summary": self.generate_summary,
            "architecture_overview": self.generate_architecture_overview,
            "onboarding_guide": self.generate_onboarding_guide,
            "important_modules": self.identify_important_modules,
            "entry_points": self.identify_potential_entry_points,
            "main_technologies": self.identify_main_technologies,
            "dependencies": self.identify_dependencies,
            "explain_main_modules": self.explain_main_modules,
            "folder_structure": self.generate_folder_structure_explanation,
            "core_components": self.identify_core_components,
            "learning_path": self.generate_learning_path,
        }
        if action_key not in action_map:
            raise ValueError(f"Unsupported action: {action_key}")
        return action_map[action_key](session_id=session_id, retrieval_mode=retrieval_mode)

    def _load_repository_metadata(self) -> Dict[str, object]:
        metadata_path = Path(REPOSITORIES_DIR) / self.repository_name / ".rag_metadata.json"
        return load_metadata(str(metadata_path)) or {"repo_name": self.repository_name}

    def _normalize_language_distribution(self, raw: Dict[str, int]) -> Dict[str, int]:
        if not isinstance(raw, dict):
            return {}
        distribution: Dict[str, int] = {}
        for key, value in raw.items():
            label = str(key).lstrip(".").upper() if str(key).startswith(".") else str(key)
            distribution[label] = int(value or 0)
        return dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True))

    def _configuration_files_from_metadata(self) -> List[str]:
        metadata = self._load_repository_metadata()
        repo_root = Path(metadata.get("local_path", "")) if metadata.get("local_path") else Path(REPOSITORIES_DIR) / self.repository_name
        config_files: List[str] = []
        if repo_root.exists():
            for path in repo_root.rglob("*"):
                if not path.is_file():
                    continue
                lowered = path.name.lower()
                if lowered.endswith((".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".env", ".xml")):
                    config_files.append(str(path.relative_to(repo_root)))
        return sorted(config_files)[:20]

    def _most_referenced_files(self) -> List[Dict[str, object]]:
        search_results = self.vector_service.search_hybrid("core architecture repository components", top_k=10, enable_rerank=False)
        counts: Dict[str, int] = {}
        for result in search_results:
            counts[result.file_path] = counts.get(result.file_path, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [{"file_path": path, "references": count} for path, count in ranked[:10]]

    def _most_connected_classes(self) -> List[Dict[str, object]]:
        search_results = self.vector_service.search_hybrid("main classes services controllers managers architecture", top_k=12, enable_rerank=False)
        counts: Dict[str, int] = {}
        for result in search_results:
            class_name = result.class_name or result.file_name
            counts[class_name] = counts.get(class_name, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [{"class_name": name, "connections": count} for name, count in ranked[:10] if name]
