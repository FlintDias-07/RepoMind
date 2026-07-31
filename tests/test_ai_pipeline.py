from types import SimpleNamespace

from ai.ai_pipeline import AIPipeline
from ai.ai_service import AIService
from ai.llm_provider import BaseLLMProvider, LLMProviderFactory, LLMResponse, ProviderSettings
from ai.prompt_builder import PromptPackage
from vectorstore.vector_service import SearchResult


class FakeProvider(BaseLLMProvider):
    def __init__(self, settings: ProviderSettings):
        super().__init__(settings)
        self.prompts = []

    def generate(self, prompt: PromptPackage) -> LLMResponse:
        self.prompts.append(prompt)
        if "Kubernetes cluster" in prompt.user_prompt:
            content = "I could not find enough evidence in the indexed repository."
        else:
            content = (
                "Authentication is handled by AuthService.\n"
                "JwtUtil is used for token generation and validation."
            )
        return LLMResponse(
            content=content,
            provider_name=self.settings.provider_name,
            model=self.settings.model,
            finish_reason="stop",
            usage={"total_tokens": 42},
        )


class FakeVectorService:
    def __init__(self, repository_name: str = "sample-repo"):
        self.repository_name = repository_name
        self.queries = []
        self.hybrid_queries = []
        self.rerank_calls = []

    def search(self, query: str, top_k: int = 5):
        self.queries.append((query, top_k))
        return self._results()

    def search_hybrid_with_diagnostics(self, query: str, top_k: int = 5, enable_rerank: bool = True):
        self.hybrid_queries.append((query, top_k, enable_rerank))
        self.rerank_calls.append(enable_rerank)
        diagnostics = SimpleNamespace(
            strategy_name="hybrid_reranked" if enable_rerank else "hybrid",
            retrieval_time_seconds=0.02,
            reranking_time_seconds=0.5 if enable_rerank else 0.0,
            total_query_time_seconds=0.52 if enable_rerank else 0.02,
            bm25_rebuilt=False,
            bm25_document_count=2,
        )
        return SimpleNamespace(results=self._results(), diagnostics=diagnostics)

    def _results(self):
        return [
            SearchResult(
                chunk_id="auth-1",
                similarity_score=0.91,
                distance=0.0,
                file_name="AuthService.java",
                file_path="src/auth/AuthService.java",
                language="java",
                class_name="AuthService",
                function_name="authenticate",
                start_line=12,
                end_line=38,
                document="class AuthService { void authenticate() { validate(); } }",
                repository_name=self.repository_name,
                metadata={"repository_name": self.repository_name},
            ),
            SearchResult(
                chunk_id="jwt-1",
                similarity_score=0.88,
                distance=0.0,
                file_name="JwtUtil.java",
                file_path="src/security/JwtUtil.java",
                language="java",
                class_name="JwtUtil",
                function_name="generateToken",
                start_line=41,
                end_line=63,
                document="class JwtUtil { String generateToken(User user) { return jwt; } }",
                repository_name=self.repository_name,
                metadata={"repository_name": self.repository_name},
            ),
        ]


def test_provider_factory_selects_registered_provider():
    factory = LLMProviderFactory(registry={"openai": FakeProvider, "groq": FakeProvider})

    provider = factory.create("groq", overrides={"api_key": "test-key", "model": "fake-model"})

    assert isinstance(provider, FakeProvider)
    assert provider.settings.provider_name == "groq"
    assert provider.settings.model == "fake-model"


def test_ai_pipeline_uses_retrieval_context_and_updates_memory():
    vector_service = FakeVectorService()
    provider = FakeProvider(
        ProviderSettings(
            provider_name="fake",
            model="fake-model",
            api_key="unused",
        )
    )
    pipeline = AIPipeline(vector_service=vector_service, llm_provider=provider)

    result = pipeline.run(
        question="Explain authentication and JWT handling.",
        repository_metadata={"repo_name": "sample-repo", "stats": {"total_files": 24}},
        session_id="session-1",
        task_hint="Answer a repository question.",
    )

    assert vector_service.hybrid_queries == [("Explain authentication and JWT handling.", 5, False)]
    assert result.context.repository.repository_name == "sample-repo"
    assert result.citations
    assert result.diagnostics.retrieval_mode == "fast"
    assert result.diagnostics.reranking_used is False
    assert result.formatted_response.summary.startswith("Authentication is handled")
    assert provider.prompts

    follow_up = pipeline.run(
        question="Now explain token generation.",
        repository_metadata={"repo_name": "sample-repo", "stats": {"total_files": 24}},
        session_id="session-1",
        task_hint="Answer a follow-up repository question.",
    )
    assert "Conversation History:" in provider.prompts[-1].user_prompt
    assert "Explain authentication and JWT handling." in provider.prompts[-1].user_prompt
    assert follow_up.formatted_response.referenced_functions == ("authenticate", "generateToken")


def test_fast_mode_does_not_call_cross_encoder_reranking():
    vector_service = FakeVectorService()
    provider = FakeProvider(ProviderSettings(provider_name="fake", model="fake-model", api_key="unused"))
    pipeline = AIPipeline(vector_service=vector_service, llm_provider=provider)

    result = pipeline.run("Explain authentication", repository_metadata={"repo_name": "sample-repo"})

    assert vector_service.rerank_calls == [False]
    assert result.context.chunks
    assert result.citations
    assert result.diagnostics.strategy_name == "hybrid"


def test_deep_analysis_mode_calls_cross_encoder_reranking():
    vector_service = FakeVectorService()
    provider = FakeProvider(ProviderSettings(provider_name="fake", model="fake-model", api_key="unused"))
    pipeline = AIPipeline(vector_service=vector_service, llm_provider=provider)

    result = pipeline.run("Explain authentication", repository_metadata={"repo_name": "sample-repo"}, retrieval_mode="deep")

    assert vector_service.rerank_calls == [True]
    assert result.diagnostics.retrieval_mode == "deep"
    assert result.diagnostics.reranking_used is True
    assert result.diagnostics.reranking_time_seconds == 0.5


def test_ask_repository_question_returns_grounded_custom_report():
    vector_service = FakeVectorService(repository_name="selected-repo")
    provider = FakeProvider(ProviderSettings(provider_name="fake", model="fake-model", api_key="unused"))
    service = AIService(repository_name="selected-repo", vector_service=vector_service, llm_provider=provider)

    report = service.ask_repository_question("Where is authentication handled?", retrieval_mode="fast")

    assert report.title == "Question: Where is authentication handled?"
    assert report.action_key == "ask_repository"
    assert report.repository_name == "selected-repo"
    assert report.citations
    assert report.diagnostics["retrieval_mode"] == "fast"
    assert vector_service.hybrid_queries[-1] == ("Where is authentication handled?", 5, False)


def test_ask_repository_question_rejects_empty_question():
    service = AIService(
        repository_name="sample-repo",
        vector_service=FakeVectorService(),
        llm_provider=FakeProvider(ProviderSettings(provider_name="fake", model="fake-model", api_key="unused")),
    )

    try:
        service.ask_repository_question("   ")
    except ValueError as exc:
        assert "question cannot be empty" in str(exc)
    else:
        raise AssertionError("empty custom question should fail validation")


def test_ask_repository_question_respects_deep_mode():
    vector_service = FakeVectorService()
    provider = FakeProvider(ProviderSettings(provider_name="fake", model="fake-model", api_key="unused"))
    service = AIService(repository_name="sample-repo", vector_service=vector_service, llm_provider=provider)

    report = service.ask_repository_question("Where is authentication handled?", retrieval_mode="deep")

    assert vector_service.rerank_calls[-1] is True
    assert report.diagnostics["reranking_used"] is True


def test_ai_service_rejects_stale_vector_service_repository():
    vector_service = FakeVectorService(repository_name="repo-a")
    provider = FakeProvider(ProviderSettings(provider_name="fake", model="fake-model", api_key="unused"))

    try:
        AIService(repository_name="repo-b", vector_service=vector_service, llm_provider=provider)
    except ValueError as exc:
        assert "Vector service repository mismatch" in str(exc)
    else:
        raise AssertionError("stale vector service should not be accepted for a different repository")


def test_unanswerable_custom_question_returns_insufficient_evidence():
    vector_service = FakeVectorService()
    provider = FakeProvider(ProviderSettings(provider_name="fake", model="fake-model", api_key="unused"))
    service = AIService(repository_name="sample-repo", vector_service=vector_service, llm_provider=provider)

    report = service.ask_repository_question("What Kubernetes cluster does this repository deploy to?")

    assert report.confidence_level == "low"
    assert "I could not find enough evidence in the indexed repository." in report.raw_answer


def test_ai_service_exposes_high_level_methods():
    vector_service = FakeVectorService()
    provider = FakeProvider(
        ProviderSettings(
            provider_name="fake",
            model="fake-model",
            api_key="unused",
        )
    )
    service = AIService(repository_name="sample-repo", vector_service=vector_service, llm_provider=provider)

    answer = service.answer_question("Explain authentication")
    deep_answer = service.answer_question("Explain authentication", retrieval_mode="deep")
    overview = service.generate_architecture_overview()
    module = service.generate_module_summary("AuthService")
    health = service.generate_repository_health()
    profile = service.generate_repository_profile()

    assert answer.summary
    assert vector_service.rerank_calls[0] is False
    assert deep_answer.diagnostics["reranking_used"] is True
    assert overview.title == "Architecture Overview"
    assert "AuthService" in module.summary or module.title == "Module Summary"
    assert isinstance(health.language_distribution, dict)
    assert profile.repository_name == "sample-repo"
