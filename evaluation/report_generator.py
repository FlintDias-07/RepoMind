"""Comparison report generation for retrieval evaluation."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from .benchmark_runner import StrategyEvaluationReport


class EvaluationReportGenerator:
    """Build comparison tables and chart-friendly payloads."""

    METRIC_FIELDS = {
        "Precision@1": "precision_at_1",
        "Precision@5": "precision_at_5",
        "Recall@5": "recall_at_5",
        "MRR": "mrr",
        "NDCG": "ndcg",
        "Avg Retrieval Time (s)": "average_retrieval_time_seconds",
        "Avg Reranking Time (s)": "average_reranking_time_seconds",
        "Avg Total Query Time (s)": "average_total_query_time_seconds",
    }

    def build_comparison_rows(
        self,
        reports: Mapping[str, StrategyEvaluationReport],
        baseline: str = "vector",
    ) -> List[Dict[str, object]]:
        baseline_report = reports.get(baseline)
        rows: List[Dict[str, object]] = []
        for metric_name, attribute in self.METRIC_FIELDS.items():
            row: Dict[str, object] = {"metric": metric_name}
            baseline_value = getattr(baseline_report.overall_metrics, attribute, 0.0) if baseline_report else 0.0
            for strategy_name, report in reports.items():
                row[strategy_name] = getattr(report.overall_metrics, attribute, 0.0)
                if strategy_name != baseline:
                    row[f"{strategy_name}_improvement"] = row[strategy_name] - baseline_value
            rows.append(row)
        return rows

    def build_repository_rows(self, reports: Mapping[str, StrategyEvaluationReport]) -> List[Dict[str, object]]:
        repositories = sorted(
            {
                repository_name
                for report in reports.values()
                for repository_name in report.repository_metrics.keys()
            }
        )
        rows: List[Dict[str, object]] = []
        for repository_name in repositories:
            row: Dict[str, object] = {"repository": repository_name}
            for strategy_name, report in reports.items():
                metrics = report.repository_metrics.get(repository_name)
                if metrics:
                    row[f"{strategy_name}_precision_at_5"] = metrics.precision_at_5
                    row[f"{strategy_name}_mrr"] = metrics.mrr
                    row[f"{strategy_name}_avg_total_query_time_seconds"] = metrics.average_total_query_time_seconds
            rows.append(row)
        return rows

    def build_query_rows(self, reports: Mapping[str, StrategyEvaluationReport]) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for strategy_name, report in reports.items():
            for record in report.query_records:
                rows.append(
                    {
                        "repository": record.repository_name,
                        "strategy": strategy_name,
                        "query": record.benchmark.query,
                        "precision_at_1": record.metrics.precision_at_1,
                        "precision_at_5": record.metrics.precision_at_5,
                        "recall_at_5": record.metrics.recall_at_5,
                        "mrr": record.metrics.mrr,
                        "ndcg": record.metrics.ndcg,
                        "retrieval_time_seconds": record.retrieval_time_seconds,
                        "reranking_time_seconds": record.reranking_time_seconds,
                        "total_query_time_seconds": record.total_query_time_seconds,
                    }
                )
        return rows
