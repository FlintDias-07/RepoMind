"""Cross-encoder reranker for hybrid retrieval."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple
import inspect
import os

from utils import get_logger

from .ranking_config import RankingConfig


class CrossEncoderReranker:
    """Rerank candidates using a sentence-transformers CrossEncoder."""

    _model_registry: Dict[Tuple[str, str], object] = {}
    _model_load_counts: Dict[Tuple[str, str], int] = {}
    _model_load_times: Dict[Tuple[str, str], float] = {}
    _model_lock = RLock()

    def __init__(self, config: Optional[RankingConfig] = None, model: object = None):
        self.config = config or RankingConfig()
        self.model = model
        self.logger = get_logger(self.__class__.__name__)
        self.reranker_model_cache_hit = model is not None
        self.reranker_model_load_count = 0
        self.reranker_model_load_time_seconds = 0.0
        self.reranker_inference_time_seconds = 0.0

    @classmethod
    def reset_model_cache(cls) -> None:
        with cls._model_lock:
            cls._model_registry.clear()
            cls._model_load_counts.clear()
            cls._model_load_times.clear()

    @classmethod
    def model_cache_size(cls) -> int:
        return len(cls._model_registry)

    def rerank(self, query: str, candidates: List[object], top_k: int) -> List[object]:
        if not candidates:
            return []

        if self.model is None:
            self._ensure_model()

        if self.model is None:
            for position, candidate in enumerate(candidates[:top_k], start=1):
                candidate.cross_encoder_score = 0.0
                candidate.reranked_position = position
            return candidates[:top_k]

        pairs = [(query, getattr(candidate, "document", "")) for candidate in candidates]
        inference_started = perf_counter()
        scores = self.model.predict(pairs)
        self.reranker_inference_time_seconds = perf_counter() - inference_started
        reranked = []
        for candidate, score in zip(candidates, scores):
            candidate.cross_encoder_score = float(score)
            reranked.append(candidate)

        reranked.sort(
            key=lambda item: (
                float(getattr(item, "cross_encoder_score", 0.0)),
                float(getattr(item, "adjusted_score", getattr(item, "similarity_score", 0.0))),
            ),
            reverse=True,
        )
        for position, candidate in enumerate(reranked[:top_k], start=1):
            candidate.reranked_position = position
            candidate.similarity_score = float(getattr(candidate, "cross_encoder_score", 0.0))
            candidate.adjusted_score = float(getattr(candidate, "cross_encoder_score", 0.0))
        return reranked[:top_k]

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except Exception:
            self.logger.warning("CrossEncoder is unavailable; returning fusion-ranked results without reranking")
            self.model = None
            return

        cache_root = Path(os.getenv("HF_HOME", "")) if os.getenv("HF_HOME") else None
        init_kwargs = {}
        if cache_root and "cache_dir" in inspect.signature(CrossEncoder.__init__).parameters:
            cache_root.mkdir(parents=True, exist_ok=True)
            init_kwargs["cache_dir"] = str(cache_root)
        cache_key = (self.config.cross_encoder_model_name, str(cache_root.resolve()) if cache_root else "")
        try:
            with self._model_lock:
                cached_model = self._model_registry.get(cache_key)
                if cached_model is not None:
                    self.model = cached_model
                    self.reranker_model_cache_hit = True
                    self.reranker_model_load_count = self._model_load_counts.get(cache_key, 0)
                    self.reranker_model_load_time_seconds = self._model_load_times.get(cache_key, 0.0)
                    return

                started = perf_counter()
                self.logger.info("Loading cross-encoder reranker: %s", self.config.cross_encoder_model_name)
                loaded_model = CrossEncoder(self.config.cross_encoder_model_name, **init_kwargs)
                load_time = perf_counter() - started
                self._model_registry[cache_key] = loaded_model
                self._model_load_counts[cache_key] = self._model_load_counts.get(cache_key, 0) + 1
                self._model_load_times[cache_key] = load_time
                self.model = loaded_model
                self.reranker_model_cache_hit = False
                self.reranker_model_load_count = self._model_load_counts[cache_key]
                self.reranker_model_load_time_seconds = load_time
        except Exception as exc:
            self.logger.warning("Unable to load cross-encoder reranker: %s", exc)
            self.model = None
