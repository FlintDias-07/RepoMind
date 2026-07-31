"""Evaluation helpers for vector-only vs hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class EvaluationQuery:
    query: str
    relevant_chunk_ids: Sequence[str]


@dataclass(frozen=True)
class EvaluationMetrics:
    top_1_accuracy: float
    top_5_accuracy: float
    precision_at_5: float
    recall_at_5: float
    mrr: float
    average_search_time_seconds: float


class RetrievalEvaluator:
    """Compare retrieval strategies against labeled relevance expectations."""

    def evaluate(
        self,
        search_fn: Callable[[str, int], Sequence[object]],
        queries: Sequence[EvaluationQuery],
        top_k: int = 5,
    ) -> EvaluationMetrics:
        if not queries:
            return EvaluationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        top1_hits: List[float] = []
        top5_hits: List[float] = []
        precision_scores: List[float] = []
        recall_scores: List[float] = []
        reciprocal_ranks: List[float] = []
        search_times: List[float] = []

        for item in queries:
            started = perf_counter()
            results = list(search_fn(item.query, top_k))
            search_times.append(perf_counter() - started)
            returned_ids = [getattr(result, "chunk_id", "") for result in results]
            relevant = set(item.relevant_chunk_ids)
            hits = [chunk_id for chunk_id in returned_ids if chunk_id in relevant]

            top1_hits.append(1.0 if returned_ids[:1] and returned_ids[0] in relevant else 0.0)
            top5_hits.append(1.0 if hits else 0.0)
            precision_scores.append(len(hits) / max(top_k, 1))
            recall_scores.append(len(hits) / max(len(relevant), 1))

            reciprocal_rank = 0.0
            for index, chunk_id in enumerate(returned_ids, start=1):
                if chunk_id in relevant:
                    reciprocal_rank = 1.0 / index
                    break
            reciprocal_ranks.append(reciprocal_rank)

        return EvaluationMetrics(
            top_1_accuracy=mean(top1_hits),
            top_5_accuracy=mean(top5_hits),
            precision_at_5=mean(precision_scores),
            recall_at_5=mean(recall_scores),
            mrr=mean(reciprocal_ranks),
            average_search_time_seconds=mean(search_times),
        )

    def compare(
        self,
        vector_search_fn: Callable[[str, int], Sequence[object]],
        hybrid_search_fn: Callable[[str, int], Sequence[object]],
        queries: Sequence[EvaluationQuery],
        top_k: int = 5,
    ) -> Dict[str, EvaluationMetrics]:
        return {
            "vector_only": self.evaluate(vector_search_fn, queries, top_k=top_k),
            "hybrid": self.evaluate(hybrid_search_fn, queries, top_k=top_k),
        }
