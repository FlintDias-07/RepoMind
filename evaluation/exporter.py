"""Export retrieval evaluation reports."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from typing import Any, Dict, Iterable, Mapping, Sequence

import pandas as pd

from .benchmark_runner import StrategyEvaluationReport
from .report_generator import EvaluationReportGenerator


class EvaluationExporter:
    """Convert evaluation reports into download-ready formats."""

    def __init__(self, report_generator: EvaluationReportGenerator | None = None):
        self.report_generator = report_generator or EvaluationReportGenerator()

    def to_json(self, reports: Mapping[str, StrategyEvaluationReport]) -> str:
        serializable = {
            strategy_name: self._serialize(report)
            for strategy_name, report in reports.items()
        }
        return json.dumps(serializable, indent=2)

    def to_markdown(self, reports: Mapping[str, StrategyEvaluationReport]) -> str:
        rows = self.report_generator.build_comparison_rows(reports)
        markdown = [
            "# Retrieval Evaluation Report",
            "",
            "## Overall Metrics",
            "",
            self._rows_to_markdown(rows),
        ]
        repository_rows = self.report_generator.build_repository_rows(reports)
        if repository_rows:
            markdown.extend(["", "## Repository Comparison", "", self._rows_to_markdown(repository_rows)])
        return "\n".join(markdown)

    def to_csv(self, reports: Mapping[str, StrategyEvaluationReport], section: str = "comparison") -> str:
        if section == "comparison":
            rows = self.report_generator.build_comparison_rows(reports)
        elif section == "queries":
            rows = self.report_generator.build_query_rows(reports)
        elif section == "repositories":
            rows = self.report_generator.build_repository_rows(reports)
        else:
            raise ValueError(f"Unsupported CSV export section: {section}")
        return pd.DataFrame(rows).to_csv(index=False)

    def _serialize(self, value: Any) -> Any:
        if is_dataclass(value):
            return {key: self._serialize(item) for key, item in asdict(value).items()}
        if isinstance(value, dict):
            return {str(key): self._serialize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize(item) for item in value]
        return value

    def _rows_to_markdown(self, rows: Sequence[Mapping[str, Any]]) -> str:
        if not rows:
            return "_No data_"

        columns = list(rows[0].keys())
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        body = []
        for row in rows:
            body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
        return "\n".join([header, separator, *body])
