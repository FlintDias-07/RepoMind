from pathlib import Path

from embedding.embedding_manager import EmbeddingManager
from embedding.embedding_service import EmbeddingService
from embedding.model_config import EmbeddingModelConfig
from parser.chunk_model import CodeChunk


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
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


def sample_chunk() -> CodeChunk:
    return CodeChunk(
        chunk_id="chunk-1",
        file_name="auth.py",
        file_path="src/auth.py",
        language="python",
        repository_name="sample-repo",
        content="def login_user(token):\n    return validate_token(token)",
        chunk_type="function",
        class_name=None,
        function_name="login_user",
        start_line=1,
        end_line=2,
    )


def test_embedding_service_batches_and_caches_duplicates(tmp_path):
    fake_model = FakeSentenceTransformer()
    config = EmbeddingModelConfig(cache_enabled=True, cache_dir=Path(tmp_path))
    service = EmbeddingService(config=config, model=fake_model)

    vectors = service.embed_texts(["authentication logic", "authentication logic", "database configuration"])

    assert vectors[0] == [1.0, 0.0, 0.0]
    assert vectors[1] == [1.0, 0.0, 0.0]
    assert vectors[2] == [0.0, 1.0, 0.0]
    assert len(fake_model.calls) == 1
    assert fake_model.calls[0] == ["authentication logic", "database configuration"]


def test_embedding_manager_builds_required_metadata(tmp_path):
    fake_model = FakeSentenceTransformer()
    config = EmbeddingModelConfig(cache_enabled=False, cache_dir=Path(tmp_path))
    manager = EmbeddingManager(embedding_service=EmbeddingService(config=config, model=fake_model))

    embedded = manager.embed_chunks([sample_chunk()])

    assert len(embedded) == 1
    assert embedded[0].metadata == {
        "chunk_id": "chunk-1",
        "file_name": "auth.py",
        "file_path": "src/auth.py",
        "language": "python",
        "class_name": "",
        "function_name": "login_user",
        "start_line": 1,
        "end_line": 2,
    }
