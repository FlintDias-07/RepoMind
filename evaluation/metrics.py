"""Metrics for retrieval benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log2
from statistics import mean
from typing import Dict, List, Sequence

from .benchmark_loader import BenchmarkCase


@dataclass(frozen=True)
class QueryMetrics:
    precision_at_1: float
    precision_at_5: float
    recall_at_5: float
    mrr: float
    ndcg: float
    relevant_results_at_5: int
    matched_expected_targets: int
    total_expected_targets: int


@dataclass(frozen=True)
class AggregateMetrics:
    precision_at_1: float
    precision_at_5: float
    recall_at_5: float
    mrr: float
    ndcg: float
    average_retrieval_time_seconds: float
    average_reranking_time_seconds: float
    average_total_query_time_seconds: float
    total_queries: int


@dataclass(frozen=True)
class ResultMatch:
    is_relevant: bool
    graded_relevance: int
    matched_targets: Sequence[str] = field(default_factory=tuple)


class MetricsCalculator:
    """Compute per-query and aggregate retrieval metrics."""

    def evaluate_query(self, benchmark: BenchmarkCase, results: Sequence[object], top_k: int = 5) -> QueryMetrics:
        limited_results = list(results[:top_k])
        matches = [self.match_result(benchmark, result) for result in limited_results]
        relevant_flags = [1.0 if item.is_relevant else 0.0 for item in matches]
        expected_targets = self._expected_targets(benchmark)

        reciprocal_rank = 0.0
        for position, item in enumerate(matches, start=1):
            if item.is_relevant:
                reciprocal_rank = 1.0 / position
                break

        matched_target_names = set()
        for item in matches:
            matched_target_names.update(item.matched_targets)

        dcg = 0.0
        for position, item in enumerate(matches, start=1):
            if item.graded_relevance > 0:
                dcg += item.graded_relevance / log2(position + 1)

        ideal_relevances = sorted(
            [self._maximum_result_relevance(benchmark)] * min(top_k, len(expected_targets)),
            reverse=True,
        )
        idcg = sum(score / log2(index + 2) for index, score in enumerate(ideal_relevances))

        return QueryMetrics(
            precision_at_1=relevant_flags[0] if relevant_flags else 0.0,
            precision_at_5=sum(relevant_flags) / float(max(top_k, 1)),
            recall_at_5=len(matched_target_names) / float(max(len(expected_targets), 1)),
            mrr=reciprocal_rank,
            ndcg=(dcg / idcg) if idcg > 0 else 0.0,
            relevant_results_at_5=int(sum(relevant_flags)),
            matched_expected_targets=len(matched_target_names),
            total_expected_targets=len(expected_targets),
        )

    def aggregate(self, query_metrics: Sequence[QueryMetrics], timings: Sequence[Dict[str, float]]) -> AggregateMetrics:
        if not query_metrics:
            return AggregateMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        return AggregateMetrics(
            precision_at_1=mean(item.precision_at_1 for item in query_metrics),
            precision_at_5=mean(item.precision_at_5 for item in query_metrics),
            recall_at_5=mean(item.recall_at_5 for item in query_metrics),
            mrr=mean(item.mrr for item in query_metrics),
            ndcg=mean(item.ndcg for item in query_metrics),
            average_retrieval_time_seconds=mean(item.get("retrieval_time_seconds", 0.0) for item in timings),
            average_reranking_time_seconds=mean(item.get("reranking_time_seconds", 0.0) for item in timings),
            average_total_query_time_seconds=mean(item.get("total_query_time_seconds", 0.0) for item in timings),
            total_queries=len(query_metrics),
        )

    def match_result(self, benchmark: BenchmarkCase, result: object) -> ResultMatch:
        matched_targets: List[str] = []
        file_name = str(getattr(result, "file_name", "") or "")
        class_name = str(getattr(result, "class_name", "") or "")
        function_name = str(getattr(result, "function_name", "") or "")

        if file_name in set(benchmark.expected_files):
            matched_targets.append(f"file:{file_name}")
        if class_name in set(benchmark.expected_classes):
            matched_targets.append(f"class:{class_name}")
        if function_name in set(benchmark.expected_functions):
            matched_targets.append(f"function:{function_name}")

        return ResultMatch(
            is_relevant=bool(matched_targets),
            graded_relevance=len(matched_targets),
            matched_targets=tuple(matched_targets),
        )

    def _expected_targets(self, benchmark: BenchmarkCase) -> Sequence[str]:
        targets = set()
        targets.update(f"file:{name}" for name in benchmark.expected_files)
        targets.update(f"class:{name}" for name in benchmark.expected_classes)
        targets.update(f"function:{name}" for name in benchmark.expected_functions)
        return tuple(sorted(targets))

    def _maximum_result_relevance(self, benchmark: BenchmarkCase) -> int:
        score = 0
        if benchmark.expected_files:
            score += 1
        if benchmark.expected_classes:
            score += 1
        if benchmark.expected_functions:
            score += 1
        return max(score, 1)
