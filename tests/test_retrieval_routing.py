from pathlib import Path

from embedding.embedding_manager import EmbeddingManager
from embedding.embedding_service import EmbeddingService
from embedding.model_config import EmbeddingModelConfig
from parser.chunk_model import CodeChunk
from retrieval.collection_router import CollectionRouter
from retrieval.file_classifier import FileCategory, FileClassifier
from retrieval.query_classifier import QueryClassifier, QueryIntent
from vectorstore.chroma_manager import ChromaManager, ChromaManagerConfig
from vectorstore.vector_service import VectorService


class FakeSentenceTransformer:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "jwt" in lowered or "auth" in lowered or "token" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "install" in lowered or "readme" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


def build_vector_service(tmp_path):
    embedding_service = EmbeddingService(
        config=EmbeddingModelConfig(cache_enabled=False, cache_dir=Path(tmp_path) / "cache"),
        model=FakeSentenceTransformer(),
    )
    embedding_manager = EmbeddingManager(embedding_service=embedding_service)
    chroma_manager = ChromaManager(ChromaManagerConfig(persist_directory=Path(tmp_path) / "chroma"))
    return VectorService(
        repository_name="sample-repo",
        chroma_manager=chroma_manager,
        embedding_manager=embedding_manager,
    )


def build_chunk(chunk_id, file_name, file_path, content, file_category, function_name=None):
    chunk = CodeChunk(
        chunk_id=chunk_id,
        file_name=file_name,
        file_path=file_path,
        language="python",
        repository_name="sample-repo",
        content=content,
        chunk_type="function",
        function_name=function_name,
        start_line=1,
        end_line=3,
    )
    chunk.add_metadata("file_category", file_category)
    return chunk


def test_file_classifier_assigns_expected_categories():
    classifier = FileClassifier()

    assert classifier.classify("README.md").category == FileCategory.DOCUMENTATION
    assert classifier.classify("src/auth/AuthService.java").category == FileCategory.SOURCE_CODE
    assert classifier.classify("tests/test_auth.py").category == FileCategory.TESTS
    assert classifier.classify("config/application.yml").category == FileCategory.CONFIGURATION


def test_query_classifier_assigns_expected_intents():
    classifier = QueryClassifier()

    assert classifier.classify("How authentication works?").intent == QueryIntent.AUTHENTICATION
    assert classifier.classify("How to install this project?").intent == QueryIntent.INSTALLATION
    assert classifier.classify("Where is the database configured?").intent == QueryIntent.CONFIGURATION


def test_vector_service_routes_chunks_and_queries_by_collection(tmp_path):
    service = build_vector_service(tmp_path)
    router = CollectionRouter("sample-repo")
    chunks = [
        build_chunk(
            "code-auth",
            "auth_service.py",
            "src/auth_service.py",
            "def validate_jwt(token): return verify_token(token)",
            FileCategory.SOURCE_CODE,
            function_name="validate_jwt",
        ),
        build_chunk(
            "doc-readme",
            "README.md",
            "README.md",
            "JWT validation setup guide and installation notes",
            FileCategory.DOCUMENTATION,
            function_name="readme_section",
        ),
        build_chunk(
            "test-auth",
            "test_auth.py",
            "tests/test_auth.py",
            "def test_validate_jwt(): assert True",
            FileCategory.TESTS,
            function_name="test_validate_jwt",
        ),
    ]

    result = service.reindex_chunks(chunks)

    assert result.stored_chunks == 3
    assert service._get_collection_manager(router.collection_name_for_category(FileCategory.SOURCE_CODE)).get_collection().count() == 1
    assert service._get_collection_manager(router.collection_name_for_category(FileCategory.DOCUMENTATION)).get_collection().count() == 1
    assert service._get_collection_manager(router.collection_name_for_category(FileCategory.TESTS)).get_collection().count() == 1

    auth_results = service.search("Which file handles JWT validation?", top_k=5)
    assert auth_results
    assert auth_results[0].file_path == "src/auth_service.py"
    assert all(item.metadata.get("file_category") == FileCategory.SOURCE_CODE for item in auth_results)

    doc_results = service.search("How to install this project?", top_k=5)
    assert doc_results
    assert doc_results[0].file_path == "README.md"
    assert doc_results[0].metadata.get("file_category") == FileCategory.DOCUMENTATION
