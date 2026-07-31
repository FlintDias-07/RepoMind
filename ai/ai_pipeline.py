"""End-to-end AI orchestration built on top of retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Optional, Sequence

from config import AI_CONTEXT_TOKEN_LIMIT, AI_MEMORY_MAX_TURNS
from utils import get_logger, load_metadata

from .citation_manager import Citation, CitationManager
from .context_builder import AIContext, AIContextBuilder
from .conversation_memory import ConversationMemory
from .llm_provider import BaseLLMProvider, LLMResponse
from .prompt_builder import PromptBuilder, PromptPackage
from .response_formatter import FormattedResponse, ResponseFormatter


@dataclass(frozen=True)
class AIPipelineDiagnostics:
    retrieval_mode: str
    reranking_used: bool
    retrieval_time_seconds: float = 0.0
    reranking_time_seconds: float = 0.0
    llm_generation_time_seconds: float = 0.0
    total_response_time_seconds: float = 0.0
    strategy_name: str = ""
    bm25_rebuilt: bool = False
    bm25_document_count: int = 0


@dataclass(frozen=True)
class AIPipelineResult:
    question: str
    context: AIContext
    prompt: PromptPackage
    provider_response: LLMResponse
    formatted_response: FormattedResponse
    citations: Sequence[Citation]
    retrieval_results: Sequence[object]
    diagnostics: AIPipelineDiagnostics = field(
        default_factory=lambda: AIPipelineDiagnostics(retrieval_mode="fast", reranking_used=False)
    )


class AIPipeline:
    """Compose retrieval, context building, prompting, LLM calls, and formatting."""

    def __init__(
        self,
        vector_service: object,
        llm_provider: BaseLLMProvider,
        context_builder: Optional[AIContextBuilder] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        citation_manager: Optional[CitationManager] = None,
        response_formatter: Optional[ResponseFormatter] = None,
        conversation_memory: Optional[ConversationMemory] = None,
    ):
        self.vector_service = vector_service
        self.llm_provider = llm_provider
        self.context_builder = context_builder or AIContextBuilder(token_limit=AI_CONTEXT_TOKEN_LIMIT)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.citation_manager = citation_manager or CitationManager()
        self.response_formatter = response_formatter or ResponseFormatter()
        self.conversation_memory = conversation_memory or ConversationMemory(max_turns=AI_MEMORY_MAX_TURNS)
        self.logger = get_logger(self.__class__.__name__)

    def run(
        self,
        question: str,
        repository_metadata: Optional[Dict[str, object]] = None,
        session_id: str = "default",
        role: str = "software_architect",
        task_hint: str = "",
        top_k: int = 5,
        retrieval_mode: str = "fast",
    ) -> AIPipelineResult:
        if not question or not question.strip():
            raise ValueError("question cannot be empty")

        total_started = perf_counter()
        normalized_mode = self._normalize_retrieval_mode(retrieval_mode)
        enable_rerank = normalized_mode == "deep"
        retrieval_execution = None
        if hasattr(self.vector_service, "search_hybrid_with_diagnostics"):
            retrieval_execution = self.vector_service.search_hybrid_with_diagnostics(
                question,
                top_k=top_k,
                enable_rerank=enable_rerank,
            )
            retrieved_results = retrieval_execution.results
        elif hasattr(self.vector_service, "search_hybrid"):
            retrieved_results = self.vector_service.search_hybrid(
                question,
                top_k=top_k,
                enable_rerank=enable_rerank,
            )
        else:
            retrieved_results = self.vector_service.search(question, top_k=top_k)

        repository_name = (
            str((repository_metadata or {}).get("repo_name") or (repository_metadata or {}).get("repository_name") or "")
            or str(getattr(self.vector_service, "repository_name", "") or "")
        )
        history = self.conversation_memory.render_history(repository_name or "unknown-repository", session_id)
        context = self.context_builder.build(question, retrieved_results, repository_metadata)
        prompt = self.prompt_builder.build(context, role=role, follow_up_history=history, task_hint=task_hint)
        llm_started = perf_counter()
        provider_response = self.llm_provider.generate(prompt)
        llm_generation_time = perf_counter() - llm_started
        citations = self.citation_manager.from_context(context)
        formatted_response = self.response_formatter.format(provider_response.content, context, citations)
        self.conversation_memory.append_turn(
            repository_name=context.repository.repository_name,
            session_id=session_id,
            question=question,
            answer=formatted_response.summary,
        )
        self.logger.info(
            "AI pipeline completed for repository '%s' with %s citations (mode=%s rerank=%s)",
            context.repository.repository_name,
            len(citations),
            normalized_mode,
            enable_rerank,
        )
        diagnostics = self._build_diagnostics(
            retrieval_mode=normalized_mode,
            reranking_used=enable_rerank,
            retrieval_execution=retrieval_execution,
            llm_generation_time_seconds=llm_generation_time,
            total_response_time_seconds=perf_counter() - total_started,
        )
        return AIPipelineResult(
            question=question,
            context=context,
            prompt=prompt,
            provider_response=provider_response,
            formatted_response=formatted_response,
            citations=citations,
            retrieval_results=tuple(retrieved_results),
            diagnostics=diagnostics,
        )

    def _normalize_retrieval_mode(self, retrieval_mode: str) -> str:
        normalized = (retrieval_mode or "fast").strip().lower().replace("_", "-")
        if normalized in {"deep", "deep-analysis", "deep analysis", "rerank", "reranked"}:
            return "deep"
        return "fast"

    def _build_diagnostics(
        self,
        retrieval_mode: str,
        reranking_used: bool,
        retrieval_execution: Optional[object],
        llm_generation_time_seconds: float,
        total_response_time_seconds: float,
    ) -> AIPipelineDiagnostics:
        search_diagnostics = getattr(retrieval_execution, "diagnostics", None)
        return AIPipelineDiagnostics(
            retrieval_mode=retrieval_mode,
            reranking_used=reranking_used,
            retrieval_time_seconds=float(getattr(search_diagnostics, "retrieval_time_seconds", 0.0) or 0.0),
            reranking_time_seconds=float(getattr(search_diagnostics, "reranking_time_seconds", 0.0) or 0.0),
            llm_generation_time_seconds=llm_generation_time_seconds,
            total_response_time_seconds=total_response_time_seconds,
            strategy_name=str(getattr(search_diagnostics, "strategy_name", "") or ("hybrid_reranked" if reranking_used else "hybrid")),
            bm25_rebuilt=bool(getattr(search_diagnostics, "bm25_rebuilt", False)),
            bm25_document_count=int(getattr(search_diagnostics, "bm25_document_count", 0) or 0),
        )
