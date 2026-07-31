"""Sequential ranking pipeline for semantic retrieval candidates."""

from __future__ import annotations

from typing import Dict, List, Optional

from .folder_ranker import FolderRanker
from .metadata_filter import MetadataFilter, RetrievalFilters
from .query_classifier import QueryClassification
from .ranking_config import RankingConfig
from .symbol_ranker import SymbolRanker


class RankingPipeline:
    """Apply metadata filtering and score adjustments to retrieved candidates."""

    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()
        self.metadata_filter = MetadataFilter(self.config)
        self.folder_ranker = FolderRanker(self.config)
        self.symbol_ranker = SymbolRanker(self.config)

    def infer_filters(
        self,
        query: str,
        classification: QueryClassification,
        explicit_filters: Optional[Dict[str, object]] = None,
    ) -> RetrievalFilters:
        return self.metadata_filter.infer_filters(query, classification, explicit_filters)

    def rerank(
        self,
        query: str,
        classification: QueryClassification,
        candidates: List[object],
        top_k: int,
        explicit_filters: Optional[Dict[str, object]] = None,
    ) -> List[object]:
        filters = self.infer_filters(query, classification, explicit_filters)
        ranked: List[object] = []

        for candidate in candidates:
            metadata = dict(getattr(candidate, "metadata", {}) or {})
            if not self.metadata_filter.matches(metadata, filters):
                continue

            original_score = float(
                getattr(
                    candidate,
                    "fusion_score",
                    getattr(candidate, "original_vector_score", getattr(candidate, "similarity_score", 0.0)),
                )
            )
            folder_weight, folder_reasons = self.folder_ranker.weight_for(getattr(candidate, "file_path", ""))
            metadata_bonus, metadata_reasons = self.metadata_filter.metadata_bonus(metadata, filters)
            preference_bonus, preference_reasons = self._intent_preference_bonus(
                classification.intent,
                str(metadata.get("file_category", "")),
            )
            metadata_bonus += preference_bonus
            symbol_boost, symbol_reasons = self.symbol_ranker.score(
                query=query,
                class_name=getattr(candidate, "class_name", ""),
                function_name=getattr(candidate, "function_name", ""),
                file_name=getattr(candidate, "file_name", ""),
                document=getattr(candidate, "document", ""),
            )

            adjusted_score = original_score * folder_weight * (1.0 + metadata_bonus) * (1.0 + symbol_boost)
            reasons = [f"intent={classification.intent}"] + folder_reasons + metadata_reasons + preference_reasons + symbol_reasons
            candidate.similarity_score = adjusted_score
            candidate.original_vector_score = float(getattr(candidate, "original_vector_score", 0.0))
            candidate.folder_weight = folder_weight
            candidate.metadata_bonus = metadata_bonus
            candidate.symbol_boost = symbol_boost
            candidate.adjusted_score = adjusted_score
            candidate.ranking_reasons = reasons
            candidate.query_intent = classification.intent
            ranked.append(candidate)

        ranked.sort(key=lambda item: getattr(item, "adjusted_score", getattr(item, "similarity_score", 0.0)), reverse=True)
        return ranked[:top_k]

    def _intent_preference_bonus(self, intent: str, category: str) -> tuple[float, list[str]]:
        preferences = self.config.intent_category_preferences.get(intent, [])
        if not preferences or category not in preferences:
            return 0.0, []

        index = preferences.index(category)
        if index == 0:
            return 0.4, [f"preferred_category={category}"]
        return max(0.0, 0.12 - (0.03 * index)), [f"allowed_category={category}"]
