from pathlib import Path
from types import SimpleNamespace

from retrieval.ranking_config import RankingConfig
from retrieval.reranker import CrossEncoderReranker
from vectorstore.chroma_manager import ChromaManager, ChromaManagerConfig


def make_candidate(chunk_id: str, document: str):
    return SimpleNamespace(
        chunk_id=chunk_id,
        document=document,
        adjusted_score=0.5,
        similarity_score=0.5,
        cross_encoder_score=0.0,
        reranked_position=0,
    )


def test_cross_encoder_is_lazy_loaded(monkeypatch):
    CrossEncoderReranker.reset_model_cache()
    load_calls = []

    class FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            load_calls.append(model_name)

        def predict(self, pairs):
            return [0.7 for _ in pairs]

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    reranker = CrossEncoderReranker(RankingConfig(cross_encoder_model_name="fake-reranker"))

    assert load_calls == []
    assert reranker.model is None

    reranker.rerank("query", [make_candidate("one", "document")], top_k=1)

    assert load_calls == ["fake-reranker"]
    assert reranker.reranker_model_load_count == 1

    CrossEncoderReranker.reset_model_cache()


def test_cross_encoder_model_is_reused(monkeypatch):
    CrossEncoderReranker.reset_model_cache()
    load_calls = []

    class FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            load_calls.append(model_name)

        def predict(self, pairs):
            return [0.8 for _ in pairs]

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    config = RankingConfig(cross_encoder_model_name="shared-reranker")
    first = CrossEncoderReranker(config)
    second = CrossEncoderReranker(config)

    first.rerank("query", [make_candidate("one", "document")], top_k=1)
    second.rerank("query", [make_candidate("two", "document")], top_k=1)

    assert load_calls == ["shared-reranker"]
    assert first.reranker_model_cache_hit is False
    assert second.reranker_model_cache_hit is True

    CrossEncoderReranker.reset_model_cache()


def test_chroma_client_reuses_same_persistent_path(tmp_path):
    ChromaManager.reset_client_cache()
    config = ChromaManagerConfig(persist_directory=Path(tmp_path) / "chroma")
    first = ChromaManager(config)
    second = ChromaManager(config)

    assert first.get_client() is second.get_client()
    assert first.client_cache_hit is False
    assert second.client_cache_hit is True
    assert ChromaManager.client_cache_size() == 1

    ChromaManager.reset_client_cache()


def test_chroma_client_cache_isolated_by_persistent_path(tmp_path):
    ChromaManager.reset_client_cache()
    first = ChromaManager(ChromaManagerConfig(persist_directory=Path(tmp_path) / "chroma-a"))
    second = ChromaManager(ChromaManagerConfig(persist_directory=Path(tmp_path) / "chroma-b"))

    assert first.get_client() is not second.get_client()
    assert ChromaManager.client_cache_size() == 2

    ChromaManager.reset_client_cache()
