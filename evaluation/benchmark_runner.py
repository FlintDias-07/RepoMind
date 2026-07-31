"""Automatic benchmark runner for retrieval strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Sequence

from utils import get_logger

from .benchmark_loader import BenchmarkCase, BenchmarkSuite
from .metrics import AggregateMetrics, MetricsCalculator, QueryMetrics


@dataclass(frozen=True)
class StrategyExecution:
    results: Sequence[object]
    retrieval_time_seconds: float
    reranking_time_seconds: float
    total_query_time_seconds: float


@dataclass(frozen=True)
class SearchStrategy:
    name: str
    execute: Callable[[str, int], StrategyExecution]
    display_name: str = ""


@dataclass(frozen=True)
class RetrievedResultRecord:
    rank: int
    chunk_id: str
    file_name: str
    file_path: str
    class_name: str
    function_name: str
    start_line: int
    end_line: int
    vector_score: float
    bm25_score: float
    fusion_score: float
    cross_encoder_score: float
    final_score: float
    ranking_reasons: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class QueryEvaluationRecord:
    repository_name: str
    strategy_name: str
    benchmark: BenchmarkCase
    metrics: QueryMetrics
    expected_results: Mapping[str, Sequence[str]]
    retrieved_results: Sequence[RetrievedResultRecord]
    missed_results: Mapping[str, Sequence[str]]
    retrieval_time_seconds: float
    reranking_time_seconds: float
    total_query_time_seconds: float


@dataclass(frozen=True)
class StrategyEvaluationReport:
    strategy_name: str
    display_name: str
    overall_metrics: AggregateMetrics
    repository_metrics: Mapping[str, AggregateMetrics]
    query_records: Sequence[QueryEvaluationRecord]


class BenchmarkRunner:
    """Run benchmark suites against one or more retrieval strategies."""

    def __init__(self, metrics_calculator: MetricsCalculator | None = None):
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        self.logger = get_logger(self.__class__.__name__)

    def run(
        self,
        suites: Sequence[BenchmarkSuite],
        strategies: Sequence[SearchStrategy],
        top_k: int = 5,
    ) -> Dict[str, StrategyEvaluationReport]:
        reports: Dict[str, StrategyEvaluationReport] = {}

        for strategy in strategies:
            query_records: List[QueryEvaluationRecord] = []
            repository_metrics: Dict[str, AggregateMetrics] = {}

            for suite in suites:
                suite_records = self._run_suite_for_strategy(suite, strategy, top_k=top_k)
                query_records.extend(suite_records)
                repository_metrics[suite.repository_name] = self.metrics_calculator.aggregate(
                    [record.metrics for record in suite_records],
                    [
                        {
                            "retrieval_time_seconds": record.retrieval_time_seconds,
                            "reranking_time_seconds": record.reranking_time_seconds,
                            "total_query_time_seconds": record.total_query_time_seconds,
                        }
                        for record in suite_records
                    ],
                )

            reports[strategy.name] = StrategyEvaluationReport(
                strategy_name=strategy.name,
                display_name=strategy.display_name or strategy.name,
                overall_metrics=self.metrics_calculator.aggregate(
                    [record.metrics for record in query_records],
                    [
                        {
                            "retrieval_time_seconds": record.retrieval_time_seconds,
                            "reranking_time_seconds": record.reranking_time_seconds,
                            "total_query_time_seconds": record.total_query_time_seconds,
                        }
                        for record in query_records
                    ],
                ),
                repository_metrics=repository_metrics,
                query_records=query_records,
            )
            self.logger.info(
                "Completed benchmark run for strategy '%s' across %s repositories and %s queries",
                strategy.name,
                len(suites),
                len(query_records),
            )

        return reports

    def _run_suite_for_strategy(
        self,
        suite: BenchmarkSuite,
        strategy: SearchStrategy,
        top_k: int,
    ) -> List[QueryEvaluationRecord]:
        records: List[QueryEvaluationRecord] = []
        for benchmark in suite.benchmarks:
            execution = strategy.execute(benchmark.query, top_k)
            query_metrics = self.metrics_calculator.evaluate_query(benchmark, execution.results, top_k=top_k)
            expected_results = {
                "files": tuple(benchmark.expected_files),
                "classes": tuple(benchmark.expected_classes),
                "functions": tuple(benchmark.expected_functions),
            }
            records.append(
                QueryEvaluationRecord(
                    repository_name=suite.repository_name,
                    strategy_name=strategy.name,
                    benchmark=benchmark,
                    metrics=query_metrics,
                    expected_results=expected_results,
                    retrieved_results=self._serialize_results(execution.results),
                    missed_results=self._missed_results(benchmark, execution.results),
                    retrieval_time_seconds=execution.retrieval_time_seconds,
                    reranking_time_seconds=execution.reranking_time_seconds,
                    total_query_time_seconds=execution.total_query_time_seconds,
                )
            )
        return records

    def _serialize_results(self, results: Sequence[object]) -> Sequence[RetrievedResultRecord]:
        records: List[RetrievedResultRecord] = []
        for index, result in enumerate(results, start=1):
            records.append(
                RetrievedResultRecord(
                    rank=index,
                    chunk_id=str(getattr(result, "chunk_id", "")),
                    file_name=str(getattr(result, "file_name", "")),
                    file_path=str(getattr(result, "file_path", "")),
                    class_name=str(getattr(result, "class_name", "")),
                    function_name=str(getattr(result, "function_name", "")),
                    start_line=int(getattr(result, "start_line", 0) or 0),
                    end_line=int(getattr(result, "end_line", 0) or 0),
                    vector_score=float(getattr(result, "original_vector_score", 0.0)),
                    bm25_score=float(getattr(result, "bm25_score", 0.0)),
                    fusion_score=float(getattr(result, "fusion_score", 0.0)),
                    cross_encoder_score=float(getattr(result, "cross_encoder_score", 0.0)),
                    final_score=float(getattr(result, "adjusted_score", getattr(result, "similarity_score", 0.0))),
                    ranking_reasons=tuple(getattr(result, "ranking_reasons", []) or []),
                )
            )
        return records

    def _missed_results(self, benchmark: BenchmarkCase, results: Sequence[object]) -> Mapping[str, Sequence[str]]:
        returned_files = {str(getattr(result, "file_name", "")) for result in results}
        returned_classes = {str(getattr(result, "class_name", "")) for result in results}
        returned_functions = {str(getattr(result, "function_name", "")) for result in results}
        return {
            "files": tuple(item for item in benchmark.expected_files if item not in returned_files),
            "classes": tuple(item for item in benchmark.expected_classes if item not in returned_classes),
            "functions": tuple(item for item in benchmark.expected_functions if item not in returned_functions),
        }
