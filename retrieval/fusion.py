"""Weighted score fusion for hybrid retrieval."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .ranking_config import RankingConfig


class WeightedScoreFusion:
    """Fuse normalized vector and BM25 scores with configurable weights."""

    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()

    def fuse(self, candidates: List[object]) -> List[object]:
        vector_scores = [float(getattr(item, "original_vector_score", 0.0)) for item in candidates]
        bm25_scores = [float(getattr(item, "bm25_score", 0.0)) for item in candidates]
        normalized_vector = self._normalize(vector_scores)
        normalized_bm25 = self._normalize(bm25_scores)

        fused: List[object] = []
        for item, vector_score, bm25_score in zip(candidates, normalized_vector, normalized_bm25):
            fusion_score = (
                self.config.vector_weight * vector_score
                + self.config.bm25_weight * bm25_score
            )
            item.original_vector_score = float(getattr(item, "original_vector_score", 0.0))
            item.bm25_score = float(getattr(item, "bm25_score", 0.0))
            item.fusion_score = fusion_score
            fused.append(item)
        return fused

    def _normalize(self, values: List[float]) -> List[float]:
        if not values:
            return []
        maximum = max(values)
        minimum = min(values)
        if maximum == minimum:
            return [1.0 if value > 0 else 0.0 for value in values]
        return [(value - minimum) / (maximum - minimum) for value in values]
