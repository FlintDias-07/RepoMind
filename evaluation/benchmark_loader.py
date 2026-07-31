"""JSON benchmark loading for retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from config import BENCHMARKS_DIR
from utils import get_logger


@dataclass(frozen=True)
class BenchmarkCase:
    query: str
    expected_files: Sequence[str] = field(default_factory=tuple)
    expected_classes: Sequence[str] = field(default_factory=tuple)
    expected_functions: Sequence[str] = field(default_factory=tuple)

    @property
    def expected_target_count(self) -> int:
        return len(set(self.expected_files)) + len(set(self.expected_classes)) + len(set(self.expected_functions))


@dataclass(frozen=True)
class BenchmarkSuite:
    repository_name: str
    benchmarks: Sequence[BenchmarkCase]
    source_path: str = ""


class BenchmarkLoader:
    """Load and persist repository-level benchmark suites."""

    def __init__(self, benchmark_dir: Path | None = None):
        self.benchmark_dir = Path(benchmark_dir or BENCHMARKS_DIR)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(self.__class__.__name__)

    def list_benchmark_files(self) -> List[Path]:
        return sorted(self.benchmark_dir.glob("*.json"))

    def load_all(self) -> List[BenchmarkSuite]:
        return [self.load_file(path) for path in self.list_benchmark_files()]

    def load_repository(self, repository_name: str) -> BenchmarkSuite:
        benchmark_path = self.benchmark_dir / f"{repository_name}.json"
        return self.load_file(benchmark_path)

    def load_file(self, path: str | Path) -> BenchmarkSuite:
        file_path = Path(path)
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        repository_name = payload.get("repository") or payload.get("repository_name") or file_path.stem
        raw_cases = payload.get("benchmarks")
        if not isinstance(raw_cases, list):
            raise ValueError(f"Benchmark file must contain a 'benchmarks' list: {file_path}")

        benchmarks = [self._parse_case(case) for case in raw_cases]
        suite = BenchmarkSuite(
            repository_name=str(repository_name),
            benchmarks=benchmarks,
            source_path=str(file_path),
        )
        self.logger.info("Loaded %s benchmarks for repository '%s'", len(benchmarks), repository_name)
        return suite

    def save_suite(self, suite: BenchmarkSuite, path: str | Path | None = None) -> Path:
        file_path = Path(path) if path else self.benchmark_dir / f"{suite.repository_name}.json"
        payload = {
            "repository": suite.repository_name,
            "benchmarks": [
                {
                    "query": case.query,
                    "expected_files": list(case.expected_files),
                    "expected_classes": list(case.expected_classes),
                    "expected_functions": list(case.expected_functions),
                }
                for case in suite.benchmarks
            ],
        }
        file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.logger.info("Saved %s benchmarks to %s", len(suite.benchmarks), file_path)
        return file_path

    def suite_summary(self, suites: Sequence[BenchmarkSuite]) -> List[Dict[str, object]]:
        return [
            {
                "repository_name": suite.repository_name,
                "benchmark_count": len(suite.benchmarks),
                "source_path": suite.source_path,
            }
            for suite in suites
        ]

    def _parse_case(self, payload: Dict[str, object]) -> BenchmarkCase:
        query = str(payload.get("query", "")).strip()
        if not query:
            raise ValueError("Benchmark query cannot be empty")
        return BenchmarkCase(
            query=query,
            expected_files=self._ensure_list(payload.get("expected_files")),
            expected_classes=self._ensure_list(payload.get("expected_classes")),
            expected_functions=self._ensure_list(payload.get("expected_functions")),
        )

    def _ensure_list(self, value: object) -> Sequence[str]:
        if value is None:
            return tuple()
        if not isinstance(value, list):
            raise ValueError("Benchmark expected fields must be lists")
        return tuple(str(item) for item in value if str(item).strip())
