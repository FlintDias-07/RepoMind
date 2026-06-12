from pathlib import Path

import pytest

from embedding.embedding_manager import EmbeddingManager
from embedding.embedding_service import EmbeddingService
from embedding.model_config import EmbeddingModelConfig
from parser.chunk_model import CodeChunk
from vectorstore.chroma_manager import ChromaManager, ChromaManagerConfig
from vectorstore.vector_service import VectorService


class FakeSentenceTransformer:
    def __init__(self):
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "auth" in lowered or "jwt" in lowered or "login" in lowered:
                vectors.append([1.0, 0.0, 0.0])
            elif "database" in lowered or "config" in lowered:
                vectors.append([0.0, 1.0, 0.0])
            elif "user" in lowered:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.3, 0.3, 0.4])
        return vectors


def sample_chunks():
    return [
        CodeChunk(
            chunk_id="auth-1",
            file_name="auth_service.py",
            file_path="src/auth_service.py",
            language="python",
            repository_name="sample-repo",
            content="def authenticate_user():\n    return validate_jwt(token)",
            chunk_type="function",
            function_name="authenticate_user",
            start_line=1,
            end_line=2,
        ),
        CodeChunk(
            chunk_id="jwt-1",
            file_name="jwt_utils.py",
            file_path="src/jwt_utils.py",
            language="python",
            repository_name="sample-repo",
            content="def validate_jwt(token):\n    return decode_signature(token)",
            chunk_type="function",
            function_name="validate_jwt",
            start_line=1,
            end_line=2,
        ),
        CodeChunk(
            chunk_id="db-1",
            file_name="database_config.py",
            file_path="src/database_config.py",
            language="python",
            repository_name="sample-repo",
            content="def load_database_config():\n    return {'host': 'localhost'}",
            chunk_type="function",
            function_name="load_database_config",
            start_line=1,
            end_line=2,
        ),
        CodeChunk(
            chunk_id="login-1",
            file_name="login_flow.py",
            file_path="src/login_flow.py",
            language="python",
            repository_name="sample-repo",
            content="def user_login_flow():\n    return authenticate_user()",
            chunk_type="function",
            function_name="user_login_flow",
            start_line=1,
            end_line=2,
        ),
    ]


@pytest.fixture()
def vector_service(tmp_path):
    fake_model = FakeSentenceTransformer()
    embedding_service = EmbeddingService(
        config=EmbeddingModelConfig(cache_enabled=False, cache_dir=Path(tmp_path) / "cache"),
        model=fake_model,
    )
    embedding_manager = EmbeddingManager(embedding_service=embedding_service)
    chroma_manager = ChromaManager(ChromaManagerConfig(persist_directory=Path(tmp_path) / "chroma"))
    service = VectorService(
        repository_name="sample-repo",
        chroma_manager=chroma_manager,
        embedding_manager=embedding_manager,
    )
    service.index_chunks(sample_chunks())
    return service


@pytest.mark.parametrize(
    "query, expected_chunk_id, expected_display_name",
    [
        ("authentication logic", "auth-1", "authenticate_user()"),
        ("JWT validation", "jwt-1", "validate_jwt()"),
        ("database configuration", "db-1", "load_database_config()"),
        ("user login flow", "login-1", "user_login_flow()"),
    ],
)
def test_search_returns_expected_top_result(vector_service, query, expected_chunk_id, expected_display_name):
    results = vector_service.search(query, top_k=3)

    assert results
    assert results[0].chunk_id == expected_chunk_id
    assert results[0].display_name == expected_display_name
    assert results[0].similarity_score > 0.5


def demo_search_queries() -> None:
    """Run the sample search queries and print a compact report."""

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        fake_model = FakeSentenceTransformer()
        embedding_service = EmbeddingService(
            config=EmbeddingModelConfig(cache_enabled=False, cache_dir=Path(temp_dir) / "cache"),
            model=fake_model,
        )
        embedding_manager = EmbeddingManager(embedding_service=embedding_service)
        chroma_manager = ChromaManager(ChromaManagerConfig(persist_directory=Path(temp_dir) / "chroma"))
        service = VectorService(
            repository_name="sample-repo",
            chroma_manager=chroma_manager,
            embedding_manager=embedding_manager,
        )
        service.index_chunks(sample_chunks())

        queries = [
            "authentication logic",
            "JWT validation",
            "database configuration",
            "user login flow",
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            for result in service.search(query, top_k=3):
                print(
                    f"  score={result.similarity_score:.3f} | file={result.file_name} | "
                    f"symbol={result.display_name} | lines={result.start_line}-{result.end_line}"
                )


if __name__ == "__main__":
    demo_search_queries()
