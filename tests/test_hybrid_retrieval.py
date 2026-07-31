from pathlib import Path

from embedding.embedding_manager import EmbeddingManager
from embedding.embedding_service import EmbeddingService
from embedding.model_config import EmbeddingModelConfig
from parser.chunk_model import CodeChunk
from retrieval.bm25_retriever import BM25Retriever
from retrieval.evaluation import EvaluationQuery, RetrievalEvaluator
from retrieval.file_classifier import FileCategory
from vectorstore.chroma_manager import ChromaManager, ChromaManagerConfig
from vectorstore.vector_service import VectorService


class FlatVectorSentenceTransformer:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if any(term in lowered for term in {"jwt", "token", "auth", "login", "config", "database"}):
                vectors.append([1.0, 0.0, 0.0])
            else:
                vectors.append([0.5, 0.25, 0.25])
        return vectors


class FakeCrossEncoder:
    def predict(self, pairs):
        scores = []
        for query, document in pairs:
            lowered_query = query.lower()
            lowered_doc = document.lower()
            score = 0.2
            if "jwt" in lowered_query and "jwtutil" in lowered_doc:
                score = 0.95
            elif "database" in lowered_query and "application.yml" in lowered_doc:
                score = 0.9
            elif "login" in lowered_query and "logincontroller" in lowered_doc:
                score = 0.88
            elif "auth" in lowered_query and "authservice" in lowered_doc:
                score = 0.86
            scores.append(score)
        return scores


def make_chunk(chunk_id, file_name, file_path, content, category, function_name="", class_name="", language="java"):
    chunk = CodeChunk(
        chunk_id=chunk_id,
        file_name=file_name,
        file_path=file_path,
        language=language,
        repository_name="sample-repo",
        content=content,
        chunk_type="method" if class_name else "function",
        class_name=class_name or None,
        function_name=function_name or None,
        start_line=1,
        end_line=5,
    )
    chunk.add_metadata("file_category", category)
    return chunk


def build_service(tmp_path):
    return build_service_for_repo(tmp_path, "sample-repo")


def build_service_for_repo(tmp_path, repository_name):
    embedding_service = EmbeddingService(
        config=EmbeddingModelConfig(cache_enabled=False, cache_dir=Path(tmp_path) / f"cache-{repository_name}"),
        model=FlatVectorSentenceTransformer(),
    )
    embedding_manager = EmbeddingManager(embedding_service=embedding_service)
    chroma_manager = ChromaManager(ChromaManagerConfig(persist_directory=Path(tmp_path) / "chroma"))
    service = VectorService(
        repository_name=repository_name,
        chroma_manager=chroma_manager,
        embedding_manager=embedding_manager,
    )
    service.hybrid_pipeline.reranker.model = FakeCrossEncoder()
    return service


def sample_chunks(repository_name="sample-repo"):
    return [
        make_chunk(
            "generic-auth",
            "SecurityHelper.java",
            "src/utils/SecurityHelper.java",
            "class SecurityHelper { void verifyUser() {} }",
            FileCategory.SOURCE_CODE,
            function_name="verifyUser",
            class_name="SecurityHelper",
        ),
        make_chunk(
            "jwt-util",
            "JwtUtil.java",
            "src/security/JwtUtil.java",
            "class JwtUtil { String verifyToken(String jwt) { return jwt; } }",
            FileCategory.SOURCE_CODE,
            function_name="verifyToken",
            class_name="JwtUtil",
        ),
        make_chunk(
            "config",
            "application.yml",
            "config/application.yml",
            "application.yml\ndatabase:\n  host: localhost",
            FileCategory.CONFIGURATION,
            function_name="database_config",
            language="yaml",
        ),
        make_chunk(
            "login-controller",
            "LoginController.java",
            "src/controllers/LoginController.java",
            "@RestController class LoginController { void login() {} }",
            FileCategory.SOURCE_CODE,
            function_name="login",
            class_name="LoginController",
        ),
    ]


def with_repository(chunks, repository_name):
    copied = []
    for chunk in chunks:
        cloned = CodeChunk(
            chunk_id=chunk.chunk_id,
            file_name=chunk.file_name,
            file_path=chunk.file_path,
            language=chunk.language,
            repository_name=repository_name,
            content=chunk.content,
            chunk_type=chunk.chunk_type,
            class_name=chunk.class_name,
            function_name=chunk.function_name,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            metadata=dict(chunk.metadata),
        )
        copied.append(cloned)
    return copied


def test_bm25_retriever_builds_and_searches_repository_index():
    retriever = BM25Retriever()
    chunks = sample_chunks()

    retriever.rebuild_index("sample-repo", chunks)
    results = retriever.search("sample-repo", "jwt validation", top_k=5)

    assert results
    assert results[0].chunk_id == "jwt-util"
    assert results[0].bm25_score > 0


def test_hybrid_search_combines_vector_bm25_and_reranker(tmp_path):
    service = build_service(tmp_path)
    service.reindex_chunks(sample_chunks())

    vector_results = service.search_vector_only("Which file handles JWT validation?", top_k=5)
    hybrid_results = service.search("Which file handles JWT validation?", top_k=5)

    assert vector_results
    assert hybrid_results
    assert hybrid_results[0].chunk_id == "jwt-util"
    assert hybrid_results[0].bm25_score > 0
    assert hybrid_results[0].cross_encoder_score > 0
    assert hybrid_results[0].reranked_position == 1
    assert hybrid_results[0].ranking_reasons


def test_hybrid_search_prefers_config_after_fusion_and_rerank(tmp_path):
    service = build_service(tmp_path)
    service.reindex_chunks(sample_chunks())

    results = service.search("Where is database configuration?", top_k=5)

    assert results
    assert results[0].chunk_id == "config"
    assert results[0].bm25_score > 0
    assert results[0].cross_encoder_score > 0


def test_retrieval_evaluator_compares_vector_and_hybrid(tmp_path):
    service = build_service(tmp_path)
    service.reindex_chunks(sample_chunks())
    evaluator = RetrievalEvaluator()
    queries = [
        EvaluationQuery("Which file handles JWT validation?", ["jwt-util"]),
        EvaluationQuery("Where is database configuration?", ["config"]),
    ]

    comparison = evaluator.compare(service.search_vector_only, service.search, queries, top_k=5)

    assert comparison["vector_only"].top_5_accuracy >= 0.0
    assert comparison["hybrid"].top_5_accuracy >= comparison["vector_only"].top_5_accuracy
    assert comparison["hybrid"].mrr >= comparison["vector_only"].mrr


def test_hybrid_search_lazily_recovers_bm25_from_chroma_for_fresh_service(tmp_path):
    repo = "fresh-recovery-repo"
    original_service = build_service_for_repo(tmp_path, repo)
    original_service.reindex_chunks(with_repository(sample_chunks(), repo))
    original_service.bm25_retriever.clear_all()

    fresh_service = build_service_for_repo(tmp_path, repo)
    assert fresh_service.bm25_retriever.document_count(repo) == 0

    execution = fresh_service.search_hybrid_with_diagnostics(
        "Which file handles JWT validation?",
        top_k=5,
        enable_rerank=False,
    )

    assert execution.results
    assert execution.diagnostics.bm25_available is True
    assert execution.diagnostics.bm25_rebuilt is True
    assert execution.diagnostics.bm25_rebuild_source == "chroma"
    assert execution.diagnostics.bm25_document_count == 4
    assert execution.diagnostics.bm25_candidates > 0
    assert fresh_service.bm25_retriever.document_count(repo) == 4


def test_lazy_bm25_rebuild_is_reused_on_second_query(tmp_path, monkeypatch):
    repo = "reuse-repo"
    service = build_service_for_repo(tmp_path, repo)
    service.reindex_chunks(with_repository(sample_chunks(), repo))
    service.bm25_retriever.clear_all()

    load_calls = 0
    original_loader = service._load_chunks_from_chroma

    def counting_loader():
        nonlocal load_calls
        load_calls += 1
        return original_loader()

    monkeypatch.setattr(service, "_load_chunks_from_chroma", counting_loader)

    first = service.search_hybrid_with_diagnostics("Which file handles JWT validation?", top_k=5, enable_rerank=False)
    second = service.search_hybrid_with_diagnostics("Where is database configuration?", top_k=5, enable_rerank=False)

    assert first.diagnostics.bm25_rebuilt is True
    assert second.diagnostics.bm25_rebuilt is False
    assert second.diagnostics.bm25_rebuild_source == "memory"
    assert load_calls == 1


def test_lazy_bm25_indexes_are_repository_isolated(tmp_path):
    repo_a = "isolated-a"
    repo_b = "isolated-b"
    service_a = build_service_for_repo(tmp_path, repo_a)
    service_b = build_service_for_repo(tmp_path, repo_b)
    service_a.reindex_chunks(
        with_repository(
            [
                make_chunk(
                    "a-auth",
                    "AuthService.java",
                    "src/a/AuthService.java",
                    "class AuthService { void authenticate() {} }",
                    FileCategory.SOURCE_CODE,
                    function_name="authenticate",
                    class_name="AuthService",
                )
            ],
            repo_a,
        )
    )
    service_b.reindex_chunks(
        with_repository(
            [
                make_chunk(
                    "b-jwt",
                    "JwtUtil.java",
                    "src/b/JwtUtil.java",
                    "class JwtUtil { void verifyToken() {} }",
                    FileCategory.SOURCE_CODE,
                    function_name="verifyToken",
                    class_name="JwtUtil",
                )
            ],
            repo_b,
        )
    )
    service_a.bm25_retriever.clear_all()

    fresh_a = build_service_for_repo(tmp_path, repo_a)
    fresh_b = build_service_for_repo(tmp_path, repo_b)
    result_a = fresh_a.search_hybrid_with_diagnostics("authentication", top_k=5, enable_rerank=False)
    result_b = fresh_b.search_hybrid_with_diagnostics("jwt token", top_k=5, enable_rerank=False)

    assert result_a.diagnostics.bm25_document_count == 1
    assert result_b.diagnostics.bm25_document_count == 1
    assert all(item.repository_name in {"", repo_a} or item.metadata.get("repository_name") == repo_a for item in result_a.results)
    assert all(item.repository_name in {"", repo_b} or item.metadata.get("repository_name") == repo_b for item in result_b.results)
    assert result_a.results[0].chunk_id == "a-auth"
    assert result_b.results[0].chunk_id == "b-jwt"


def test_vector_and_hybrid_results_filter_mixed_repository_metadata(tmp_path):
    repo_a = "mixed-a"
    repo_b = "mixed-b"
    service = build_service_for_repo(tmp_path, repo_a)
    mixed_chunks = [
        CodeChunk(
            chunk_id="a-face",
            file_name="FaceRecognition.py",
            file_path="backend/FaceRecognition.py",
            language="python",
            repository_name=repo_a,
            content="def recognize_face(): return 'face recognition implementation'",
            chunk_type="function",
            function_name="recognize_face",
            start_line=1,
            end_line=3,
            metadata={"file_category": FileCategory.SOURCE_CODE},
        ),
        CodeChunk(
            chunk_id="b-rag",
            file_name="ranking_pipeline.py",
            file_path="retrieval/ranking_pipeline.py",
            language="python",
            repository_name=repo_b,
            content="class RankingPipeline: pass # hybrid retrieval implementation",
            chunk_type="class",
            class_name="RankingPipeline",
            start_line=1,
            end_line=3,
            metadata={"file_category": FileCategory.SOURCE_CODE},
        ),
    ]
    service.reindex_chunks(mixed_chunks)

    vector_results = service.search_vector_only("hybrid retrieval implementation", top_k=5)
    hybrid_results = service.search_hybrid("hybrid retrieval implementation", top_k=5, enable_rerank=False)

    assert vector_results
    assert hybrid_results
    assert all(result.repository_name == repo_a for result in vector_results)
    assert all((getattr(result, "repository_name", "") or result.metadata.get("repository_name")) == repo_a for result in hybrid_results)
    assert all(result.chunk_id != "b-rag" for result in vector_results)
    assert all(result.chunk_id != "b-rag" for result in hybrid_results)


def test_bm25_search_filters_documents_with_wrong_repository_metadata(tmp_path):
    repo_a = "bm25-a"
    repo_b = "bm25-b"
    service = build_service_for_repo(tmp_path, repo_a)
    chunks = [
        CodeChunk(
            chunk_id="a-face",
            file_name="FaceRecognition.py",
            file_path="backend/FaceRecognition.py",
            language="python",
            repository_name=repo_a,
            content="def recognize_face(): return 'face recognition implementation'",
            chunk_type="function",
            function_name="recognize_face",
            start_line=1,
            end_line=3,
            metadata={"file_category": FileCategory.SOURCE_CODE},
        ),
        CodeChunk(
            chunk_id="b-rag",
            file_name="ranking_pipeline.py",
            file_path="retrieval/ranking_pipeline.py",
            language="python",
            repository_name=repo_b,
            content="class RankingPipeline: pass # hybrid retrieval implementation",
            chunk_type="class",
            class_name="RankingPipeline",
            start_line=1,
            end_line=3,
            metadata={"file_category": FileCategory.SOURCE_CODE},
        ),
    ]
    service.reindex_chunks(chunks)

    results = service.bm25_retriever.search(repo_a, "hybrid retrieval implementation", top_k=5)

    assert results == []


def test_reindex_replaces_bm25_state_for_repository(tmp_path):
    repo = "reindex-repo"
    service = build_service_for_repo(tmp_path, repo)
    old_chunks = with_repository(
        [
            make_chunk(
                "old-auth",
                "OldAuth.java",
                "src/OldAuth.java",
                "class OldAuth { void legacyAuth() {} }",
                FileCategory.SOURCE_CODE,
                function_name="legacyAuth",
                class_name="OldAuth",
            )
        ],
        repo,
    )
    new_chunks = with_repository(
        [
            make_chunk(
                "new-auth",
                "NewAuth.java",
                "src/NewAuth.java",
                "class NewAuth { void modernAuth() {} }",
                FileCategory.SOURCE_CODE,
                function_name="modernAuth",
                class_name="NewAuth",
            )
        ],
        repo,
    )

    service.reindex_chunks(old_chunks)
    assert [document.chunk_id for document in service.bm25_retriever.indexes[repo].documents] == ["old-auth"]

    service.reindex_chunks(new_chunks)
    indexed_chunk_ids = [document.chunk_id for document in service.bm25_retriever.indexes[repo].documents]

    assert service.bm25_retriever.document_count(repo) == 1
    assert indexed_chunk_ids == ["new-auth"]


def test_hybrid_diagnostics_show_graceful_bm25_degradation(tmp_path, monkeypatch):
    repo = "degraded-repo"
    service = build_service_for_repo(tmp_path, repo)
    service.bm25_retriever.clear_all()

    def broken_loader():
        raise RuntimeError("simulated Chroma load failure")

    monkeypatch.setattr(service, "_load_chunks_from_chroma", broken_loader)

    execution = service.search_hybrid_with_diagnostics("authentication", top_k=5, enable_rerank=False)

    assert execution.results == []
    assert execution.diagnostics.bm25_available is False
    assert execution.diagnostics.bm25_rebuilt is False
    assert execution.diagnostics.bm25_document_count == 0
    assert "simulated Chroma load failure" in execution.diagnostics.degradation_reason
