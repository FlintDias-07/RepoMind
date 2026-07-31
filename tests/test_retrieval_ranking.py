from pathlib import Path

from embedding.embedding_manager import EmbeddingManager
from embedding.embedding_service import EmbeddingService
from embedding.model_config import EmbeddingModelConfig
from parser.chunk_model import CodeChunk
from retrieval.file_classifier import FileCategory
from vectorstore.chroma_manager import ChromaManager, ChromaManagerConfig
from vectorstore.vector_service import VectorService


class TieHeavySentenceTransformer:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if any(term in lowered for term in {"jwt", "token", "auth", "login", "controller"}):
                vectors.append([1.0, 0.0, 0.0])
            elif any(term in lowered for term in {"config", "database", "yaml", "properties"}):
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.4, 0.4, 0.2])
        return vectors


class FakeCrossEncoder:
    def predict(self, pairs):
        scores = []
        for query, document in pairs:
            lowered_query = query.lower()
            lowered_doc = document.lower()
            score = 0.2
            if "jwt" in lowered_query and "jwt" in lowered_doc:
                score = 0.96
            elif "config" in lowered_query and "application.yml" in lowered_doc:
                score = 0.94
            elif "database" in lowered_query and "application.yml" in lowered_doc:
                score = 0.92
            elif "controller" in lowered_query and "logincontroller" in lowered_doc:
                score = 0.9
            elif "auth" in lowered_query and "authservice" in lowered_doc:
                score = 0.91
            scores.append(score)
        return scores


def build_vector_service(tmp_path):
    embedding_service = EmbeddingService(
        config=EmbeddingModelConfig(cache_enabled=False, cache_dir=Path(tmp_path) / "cache"),
        model=TieHeavySentenceTransformer(),
    )
    embedding_manager = EmbeddingManager(embedding_service=embedding_service)
    chroma_manager = ChromaManager(ChromaManagerConfig(persist_directory=Path(tmp_path) / "chroma"))
    return VectorService(
        repository_name="sample-repo",
        chroma_manager=chroma_manager,
        embedding_manager=embedding_manager,
    )


def make_chunk(
    chunk_id,
    file_name,
    file_path,
    content,
    category,
    function_name="handler",
    class_name="",
    language="python",
):
    chunk = CodeChunk(
        chunk_id=chunk_id,
        file_name=file_name,
        file_path=file_path,
        language=language,
        repository_name="sample-repo",
        content=content,
        chunk_type="method" if class_name else "function",
        class_name=class_name or None,
        function_name=function_name,
        start_line=1,
        end_line=5,
    )
    chunk.add_metadata("file_category", category)
    return chunk


def test_authentication_queries_never_rank_tests_above_production_code(tmp_path):
    service = build_vector_service(tmp_path)
    service.hybrid_pipeline.reranker.model = FakeCrossEncoder()
    service.reindex_chunks(
        [
            make_chunk(
                "prod-auth",
                "AuthService.java",
                "src/services/AuthService.java",
                "class AuthService { boolean authenticate(String token) { return true; } }",
                FileCategory.SOURCE_CODE,
                function_name="authenticate",
                class_name="AuthService",
                language="java",
            ),
            make_chunk(
                "test-auth",
                "test_auth.py",
                "tests/test_auth.py",
                "def test_authenticate(): assert authenticate('jwt')",
                FileCategory.TESTS,
                function_name="test_authenticate",
            ),
        ]
    )

    results = service.search("How authentication works?", top_k=5)

    assert results
    assert all(item.metadata.get("file_category") != FileCategory.TESTS for item in results)
    assert results[0].file_path == "src/services/AuthService.java"


def test_readme_is_ranked_below_controller_implementation(tmp_path):
    service = build_vector_service(tmp_path)
    service.hybrid_pipeline.reranker.model = FakeCrossEncoder()
    service.reindex_chunks(
        [
            make_chunk(
                "controller",
                "LoginController.java",
                "src/controllers/LoginController.java",
                "@RestController class LoginController { void login() {} }",
                FileCategory.SOURCE_CODE,
                function_name="login",
                class_name="LoginController",
                language="java",
            ),
            make_chunk(
                "readme",
                "README.md",
                "README.md",
                "This document explains the login controller API architecture.",
                FileCategory.DOCUMENTATION,
                function_name="readme_section",
                language="markdown",
            ),
        ]
    )

    results = service.search("API controller architecture", top_k=5)

    assert results
    assert results[0].file_path == "src/controllers/LoginController.java"
    assert results[0].folder_weight > results[1].folder_weight


def test_jwt_queries_rank_jwt_util_above_unrelated_utilities(tmp_path):
    service = build_vector_service(tmp_path)
    service.hybrid_pipeline.reranker.model = FakeCrossEncoder()
    service.reindex_chunks(
        [
            make_chunk(
                "jwt-util",
                "JwtUtil.java",
                "src/security/JwtUtil.java",
                "class JwtUtil { String verifyToken(String jwt) { return jwt; } }",
                FileCategory.SOURCE_CODE,
                function_name="verifyToken",
                class_name="JwtUtil",
                language="java",
            ),
            make_chunk(
                "string-util",
                "StringUtil.java",
                "src/utils/StringUtil.java",
                "class StringUtil { String normalize(String value) { return value; } }",
                FileCategory.SOURCE_CODE,
                function_name="normalize",
                class_name="StringUtil",
                language="java",
            ),
        ]
    )

    results = service.search("Which file handles JWT validation?", top_k=5)

    assert results
    assert results[0].file_name == "JwtUtil.java"
    assert results[0].symbol_boost > results[1].symbol_boost


def test_configuration_queries_prefer_config_files(tmp_path):
    service = build_vector_service(tmp_path)
    service.hybrid_pipeline.reranker.model = FakeCrossEncoder()
    service.reindex_chunks(
        [
            make_chunk(
                "config-file",
                "application.yml",
                "config/application.yml",
                "database:\n  host: localhost",
                FileCategory.CONFIGURATION,
                function_name="database_config",
                language="yaml",
            ),
            make_chunk(
                "db-service",
                "DatabaseService.java",
                "src/services/DatabaseService.java",
                "class DatabaseService { void connectDatabase() {} }",
                FileCategory.SOURCE_CODE,
                function_name="connectDatabase",
                class_name="DatabaseService",
                language="java",
            ),
        ]
    )

    results = service.search("Where is database configured?", top_k=5)

    assert results
    assert results[0].file_name == "application.yml"
    assert results[0].metadata.get("file_category") == FileCategory.CONFIGURATION


def test_folder_weighting_changes_ranking(tmp_path):
    service = build_vector_service(tmp_path)
    service.hybrid_pipeline.reranker.model = FakeCrossEncoder()
    service.reindex_chunks(
        [
            make_chunk(
                "src-auth",
                "AuthService.java",
                "src/services/AuthService.java",
                "class AuthService { void authenticate() {} }",
                FileCategory.SOURCE_CODE,
                function_name="authenticate",
                class_name="AuthService",
                language="java",
            ),
            make_chunk(
                "misc-auth",
                "AuthService.java",
                "misc/AuthService.java",
                "class AuthService { void authenticate() {} }",
                FileCategory.SOURCE_CODE,
                function_name="authenticate",
                class_name="AuthService",
                language="java",
            ),
        ]
    )

    results = service.search("authentication service", top_k=5)

    assert results[0].file_path == "src/services/AuthService.java"
    assert results[0].folder_weight > results[1].folder_weight


def test_metadata_filtering_by_language_and_category(tmp_path):
    service = build_vector_service(tmp_path)
    service.hybrid_pipeline.reranker.model = FakeCrossEncoder()
    service.reindex_chunks(
        [
            make_chunk(
                "java-auth",
                "AuthService.java",
                "src/services/AuthService.java",
                "class AuthService { void authenticate() {} }",
                FileCategory.SOURCE_CODE,
                function_name="authenticate",
                class_name="AuthService",
                language="java",
            ),
            make_chunk(
                "python-auth",
                "auth_service.py",
                "src/auth_service.py",
                "def authenticate(): return True",
                FileCategory.SOURCE_CODE,
                function_name="authenticate",
                language="python",
            ),
            make_chunk(
                "doc-auth",
                "README.md",
                "README.md",
                "Authentication guide",
                FileCategory.DOCUMENTATION,
                function_name="readme_section",
                language="markdown",
            ),
        ]
    )

    results = service.search(
        "authentication service in java",
        top_k=5,
        metadata_filters={"language": "java", "category": FileCategory.SOURCE_CODE},
    )

    assert results
    assert all(item.language == "java" for item in results)
    assert all(item.metadata.get("file_category") == FileCategory.SOURCE_CODE for item in results)
