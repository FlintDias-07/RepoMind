"""Hybrid retrieval pipeline combining vector, BM25, ranking, and reranking."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional
from types import SimpleNamespace

from .bm25_retriever import BM25Result
from .fusion import WeightedScoreFusion
from .query_classifier import QueryClassification
from .ranking_config import RankingConfig
from .ranking_pipeline import RankingPipeline
from .reranker import CrossEncoderReranker


@dataclass(frozen=True)
class HybridPipelineResult:
    results: List[object]
    retrieval_time_seconds: float
    reranking_time_seconds: float
    merged_candidates: int


class HybridRetrievalPipeline:
    """Wrap vector retrieval with lexical retrieval, fusion, and reranking."""

    def __init__(
        self,
        config: Optional[RankingConfig] = None,
        ranking_pipeline: Optional[RankingPipeline] = None,
        fusion: Optional[WeightedScoreFusion] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.config = config or RankingConfig()
        self.ranking_pipeline = ranking_pipeline or RankingPipeline(self.config)
        self.fusion = fusion or WeightedScoreFusion(self.config)
        self.reranker = reranker or CrossEncoderReranker(self.config)

    def run(
        self,
        query: str,
        classification: QueryClassification,
        vector_candidates: List[object],
        bm25_candidates: List[BM25Result],
        top_k: int,
        explicit_filters: Optional[Dict[str, object]] = None,
    ) -> List[object]:
        return self.run_with_diagnostics(
            query=query,
            classification=classification,
            vector_candidates=vector_candidates,
            bm25_candidates=bm25_candidates,
            top_k=top_k,
            explicit_filters=explicit_filters,
            enable_rerank=True,
        ).results

    def run_with_diagnostics(
        self,
        query: str,
        classification: QueryClassification,
        vector_candidates: List[object],
        bm25_candidates: List[BM25Result],
        top_k: int,
        explicit_filters: Optional[Dict[str, object]] = None,
        enable_rerank: bool = True,
    ) -> HybridPipelineResult:
        retrieval_started = perf_counter()
        merged = self._merge_candidates(vector_candidates, bm25_candidates)
        fused = self.fusion.fuse(merged)
        ranked = self.ranking_pipeline.rerank(
            query=query,
            classification=classification,
            candidates=fused,
            top_k=min(self.config.rerank_candidate_limit, max(top_k, self.config.rerank_candidate_limit)),
            explicit_filters=explicit_filters,
        )
        for candidate in ranked:
            candidate.ranking_reasons.append(
                f"fusion={self.config.vector_weight:.2f}*vector+{self.config.bm25_weight:.2f}*bm25"
            )
        retrieval_time_seconds = perf_counter() - retrieval_started

        reranking_started = perf_counter()
        if enable_rerank:
            reranked = self.reranker.rerank(
                query,
                ranked[: self.config.rerank_candidate_limit],
                top_k=min(top_k, self.config.rerank_top_k),
            )
        else:
            reranked = ranked[:top_k]
            for position, candidate in enumerate(reranked, start=1):
                candidate.cross_encoder_score = 0.0
                candidate.reranked_position = 0
        reranking_time_seconds = perf_counter() - reranking_started if enable_rerank else 0.0

        return HybridPipelineResult(
            results=reranked[:top_k],
            retrieval_time_seconds=retrieval_time_seconds,
            reranking_time_seconds=reranking_time_seconds,
            merged_candidates=len(merged),
        )

    def _merge_candidates(self, vector_candidates: List[object], bm25_candidates: List[BM25Result]) -> List[object]:
        merged: Dict[str, object] = {}

        for candidate in vector_candidates:
            candidate.bm25_score = 0.0
            candidate.fusion_score = float(getattr(candidate, "original_vector_score", getattr(candidate, "similarity_score", 0.0)))
            candidate.cross_encoder_score = 0.0
            candidate.reranked_position = 0
            merged[candidate.chunk_id] = candidate

        for bm25 in bm25_candidates:
            if bm25.chunk_id in merged:
                merged[bm25.chunk_id].bm25_score = bm25.bm25_score
                continue

            metadata = dict(bm25.metadata)
            merged[bm25.chunk_id] = SimpleNamespace(
                chunk_id=bm25.chunk_id,
                similarity_score=0.0,
                distance=0.0,
                file_name=bm25.file_name,
                file_path=bm25.file_path,
                language=bm25.language,
                class_name=bm25.class_name,
                function_name=bm25.function_name,
                start_line=int(metadata.get("start_line", 0) or 0),
                end_line=int(metadata.get("end_line", 0) or 0),
                document=bm25.document,
                collection_name="",
                repository_name=bm25.repository_name,
                original_vector_score=0.0,
                folder_weight=1.0,
                metadata_bonus=0.0,
                symbol_boost=0.0,
                adjusted_score=0.0,
                ranking_reasons=[],
                query_intent="",
                metadata=metadata,
            )
            merged[bm25.chunk_id].bm25_score = bm25.bm25_score
            merged[bm25.chunk_id].fusion_score = 0.0
            merged[bm25.chunk_id].cross_encoder_score = 0.0
            merged[bm25.chunk_id].reranked_position = 0

        return list(merged.values())
