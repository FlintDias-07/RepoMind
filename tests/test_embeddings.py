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
        "repository_name": "sample-repo",
        "chunk_type": "function",
        "class_name": "",
        "function_name": "login_user",
        "start_line": 1,
        "end_line": 2,
        "file_category": "source_code",
    }


def test_embedding_service_detects_incomplete_model_cache(tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    model_dir = cache_dir / "sentence-transformers_all-MiniLM-L6-v2"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_text("weights", encoding="utf-8")

    service = EmbeddingService(config=EmbeddingModelConfig(cache_dir=cache_dir))

    statuses = service.inspect_model_cache()

    assert len(statuses) == 1
    assert statuses[0].valid is False
    assert "tokenizer asset" in statuses[0].missing_files


def test_embedding_service_quarantines_invalid_cache_dir(tmp_path):
    cache_dir = Path(tmp_path) / "cache"
    model_dir = cache_dir / "sentence-transformers_all-MiniLM-L6-v2"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    service = EmbeddingService(config=EmbeddingModelConfig(cache_dir=cache_dir))

    quarantine_path = service._quarantine_invalid_cache_dir(model_dir, "test")

    assert quarantine_path is not None
    assert quarantine_path.exists()
    assert not model_dir.exists()


def test_embedding_model_is_reused_for_same_configuration(tmp_path, monkeypatch):
    EmbeddingService.reset_model_cache()
    load_calls = []

    class LoadTrackedSentenceTransformer(FakeSentenceTransformer):
        def __init__(self, model_source, **kwargs):
            super().__init__()
            load_calls.append((model_source, kwargs))

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", LoadTrackedSentenceTransformer)
    monkeypatch.setattr(EmbeddingService, "_resolve_model_source", lambda self: self.config.model_name)

    config = EmbeddingModelConfig(model_name="same-model", cache_enabled=False, cache_dir=Path(tmp_path) / "cache")
    first = EmbeddingService(config=config)
    second = EmbeddingService(config=config)

    assert first.embed_text("authentication") == [1.0, 0.0, 0.0]
    assert second.embed_text("database config") == [0.0, 1.0, 0.0]
    assert len(load_calls) == 1
    assert first.embedding_model_cache_hit is False
    assert second.embedding_model_cache_hit is True

    EmbeddingService.reset_model_cache()


def test_embedding_model_cache_isolated_by_configuration(tmp_path, monkeypatch):
    EmbeddingService.reset_model_cache()
    load_calls = []

    class LoadTrackedSentenceTransformer(FakeSentenceTransformer):
        def __init__(self, model_source, **kwargs):
            super().__init__()
            load_calls.append(model_source)

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", LoadTrackedSentenceTransformer)
    monkeypatch.setattr(EmbeddingService, "_resolve_model_source", lambda self: self.config.model_name)

    first = EmbeddingService(
        config=EmbeddingModelConfig(model_name="model-a", cache_enabled=False, cache_dir=Path(tmp_path) / "cache")
    )
    second = EmbeddingService(
        config=EmbeddingModelConfig(model_name="model-b", cache_enabled=False, cache_dir=Path(tmp_path) / "cache")
    )

    first.embed_text("authentication")
    second.embed_text("authentication")

    assert load_calls == ["model-a", "model-b"]
    assert EmbeddingService.model_cache_size() == 2

    EmbeddingService.reset_model_cache()
