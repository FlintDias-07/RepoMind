"""Streamlit dashboard for retrieval evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pandas as pd
import streamlit as st

from config import BENCHMARKS_DIR
from github_loader import RepositoryManager
from ui.components import empty_state, page_header, section_header
from vectorstore.vector_service import SearchExecution, VectorService

from .benchmark_loader import BenchmarkLoader, BenchmarkSuite
from .benchmark_runner import BenchmarkRunner, SearchStrategy, StrategyExecution, StrategyEvaluationReport
from .exporter import EvaluationExporter
from .report_generator import EvaluationReportGenerator


def render_retrieval_evaluation_page() -> None:
    page_header(
        "Retrieval Evaluation",
        "Benchmark Vector, Hybrid, and Hybrid + Reranker strategies against repository-specific queries.",
    )

    loader = BenchmarkLoader()
    manager = RepositoryManager()
    repositories = manager.get_cloned_repositories()
    indexed_repositories = [
        repo for repo in repositories
        if repo.get("indexed_at") and repo.get("index_stats", {}).get("embeddings_stored", 0) > 0
    ]

    if not indexed_repositories:
        empty_state("No indexed repositories available.", "Index a repository before running retrieval evaluation.")
        return

    benchmark_files = loader.list_benchmark_files()
    if not benchmark_files:
        st.warning(f"No benchmark JSON files found in `{BENCHMARKS_DIR}`.")
        st.code(example_benchmark_json(), language="json")
        return

    section_header("Evaluation Controls", "Select a repository and benchmark dataset, then run the comparison explicitly.")
    repo_names = [repo["repo_name"] for repo in indexed_repositories]
    selected_repo = st.selectbox("Repository", options=repo_names, key="evaluation_repo")
    repo_benchmark_files = [path for path in benchmark_files if path.stem == selected_repo]
    benchmark_options = repo_benchmark_files or benchmark_files
    selected_benchmark = st.selectbox(
        "Benchmark Dataset",
        options=benchmark_options,
        format_func=lambda path: Path(path).name,
        key="evaluation_benchmark_file",
    )
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5, help="Controls Precision@K and Recall@K calculations.")

    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        run_button = st.button("Run Evaluation", type="primary", use_container_width=True)
    with col2:
        st.caption("Runs Vector, Hybrid, and Hybrid + Reranker for every benchmark query. Deep reranking may take longer on CPU.")

    if run_button:
        with st.spinner("Running retrieval evaluation..."):
            suite = loader.load_file(selected_benchmark)
            service = VectorService(repository_name=selected_repo)
            reports = run_repository_benchmark_suite(service, [suite], top_k=top_k)
            st.session_state["evaluation_reports"] = reports
            st.session_state["evaluation_suite"] = suite
            st.session_state["evaluation_repo_name"] = selected_repo

    reports = st.session_state.get("evaluation_reports")
    suite = st.session_state.get("evaluation_suite")
    if reports and suite and st.session_state.get("evaluation_repo_name") == selected_repo:
        render_evaluation_results(reports=reports, suite=suite)


def run_repository_benchmark_suite(
    service: VectorService,
    suites: Sequence[BenchmarkSuite],
    top_k: int = 5,
) -> Mapping[str, StrategyEvaluationReport]:
    runner = BenchmarkRunner()
    strategies = [
        SearchStrategy("vector", lambda query, k: _vector_execution(service, query, k), display_name="Vector Only"),
        SearchStrategy("hybrid", lambda query, k: _hybrid_execution(service, query, k, False), display_name="Hybrid"),
        SearchStrategy("hybrid_reranked", lambda query, k: _hybrid_execution(service, query, k, True), display_name="Hybrid + Re-ranking"),
    ]
    return runner.run(suites=suites, strategies=strategies, top_k=top_k)


def render_evaluation_results(
    reports: Mapping[str, StrategyEvaluationReport],
    suite: BenchmarkSuite,
) -> None:
    report_generator = EvaluationReportGenerator()
    exporter = EvaluationExporter(report_generator)

    section_header("Overall Metrics")
    metrics_frame = pd.DataFrame(
        [
            {
                "Strategy": report.display_name,
                "Precision@1": report.overall_metrics.precision_at_1,
                "Precision@5": report.overall_metrics.precision_at_5,
                "Recall@5": report.overall_metrics.recall_at_5,
                "MRR": report.overall_metrics.mrr,
                "NDCG": report.overall_metrics.ndcg,
                "Avg Retrieval Time (s)": report.overall_metrics.average_retrieval_time_seconds,
                "Avg Reranking Time (s)": report.overall_metrics.average_reranking_time_seconds,
                "Avg Total Query Time (s)": report.overall_metrics.average_total_query_time_seconds,
            }
            for report in reports.values()
        ]
    )
    st.dataframe(metrics_frame, use_container_width=True)

    metric_cols = st.columns(4)
    hybrid_reranked = reports.get("hybrid_reranked")
    if hybrid_reranked:
        with metric_cols[0]:
            st.metric("Benchmarks", len(suite.benchmarks))
        with metric_cols[1]:
            st.metric("Precision@5", f"{hybrid_reranked.overall_metrics.precision_at_5:.3f}")
        with metric_cols[2]:
            st.metric("MRR", f"{hybrid_reranked.overall_metrics.mrr:.3f}")
        with metric_cols[3]:
            st.metric("Avg Total Time", f"{hybrid_reranked.overall_metrics.average_total_query_time_seconds:.3f}s")

    section_header("Comparison")
    comparison_frame = pd.DataFrame(report_generator.build_comparison_rows(reports))
    st.dataframe(comparison_frame, use_container_width=True)

    section_header("Charts")
    chart_frame = metrics_frame.set_index("Strategy")[["Precision@5", "Recall@5", "MRR", "NDCG"]]
    st.bar_chart(chart_frame)
    latency_frame = metrics_frame.set_index("Strategy")[["Avg Retrieval Time (s)", "Avg Reranking Time (s)", "Avg Total Query Time (s)"]]
    st.line_chart(latency_frame)

    repository_frame = pd.DataFrame(report_generator.build_repository_rows(reports))
    if not repository_frame.empty:
        section_header("Repository Comparison")
        st.dataframe(repository_frame, use_container_width=True)

    section_header("Query Inspector")
    render_query_inspector(reports)

    section_header("Export")
    st.download_button(
        "Download JSON Report",
        data=exporter.to_json(reports),
        file_name=f"{suite.repository_name}_retrieval_report.json",
        mime="application/json",
        use_container_width=True,
    )
    export_cols = st.columns(2)
    with export_cols[0]:
        st.download_button(
            "Download CSV Report",
            data=exporter.to_csv(reports, section="comparison"),
            file_name=f"{suite.repository_name}_retrieval_report.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_cols[1]:
        st.download_button(
            "Download Markdown Report",
            data=exporter.to_markdown(reports),
            file_name=f"{suite.repository_name}_retrieval_report.md",
            mime="text/markdown",
            use_container_width=True,
        )


def render_query_inspector(reports: Mapping[str, StrategyEvaluationReport]) -> None:
    strategy_options = list(reports.keys())
    selected_strategy = st.selectbox("Strategy", options=strategy_options, format_func=lambda name: reports[name].display_name)
    report = reports[selected_strategy]
    query_options = [record.benchmark.query for record in report.query_records]
    selected_query = st.selectbox("Benchmark Query", options=query_options)
    selected_record = next(record for record in report.query_records if record.benchmark.query == selected_query)

    st.markdown(f"**Expected Files:** {', '.join(selected_record.expected_results['files']) or 'None'}")
    st.markdown(f"**Expected Classes:** {', '.join(selected_record.expected_results['classes']) or 'None'}")
    st.markdown(f"**Expected Functions:** {', '.join(selected_record.expected_results['functions']) or 'None'}")
    st.markdown(f"**Missed Results:** {format_missed_results(selected_record.missed_results)}")

    details_frame = pd.DataFrame(
        [
            {
                "Rank": result.rank,
                "File": result.file_name,
                "Class": result.class_name,
                "Function": result.function_name,
                "Lines": f"{result.start_line}-{result.end_line}" if result.start_line or result.end_line else "",
                "Vector Score": result.vector_score,
                "BM25 Score": result.bm25_score,
                "Fusion Score": result.fusion_score,
                "CrossEncoder Score": result.cross_encoder_score,
                "Final Score": result.final_score,
                "Ranking Reason": " | ".join(result.ranking_reasons),
            }
            for result in selected_record.retrieved_results
        ]
    )
    st.dataframe(details_frame, use_container_width=True)

    latency_cols = st.columns(3)
    with latency_cols[0]:
        st.metric("Retrieval Time", f"{selected_record.retrieval_time_seconds:.4f}s")
    with latency_cols[1]:
        st.metric("Reranking Time", f"{selected_record.reranking_time_seconds:.4f}s")
    with latency_cols[2]:
        st.metric("Total Query Time", f"{selected_record.total_query_time_seconds:.4f}s")


def format_missed_results(missed_results: Mapping[str, Sequence[str]]) -> str:
    parts: List[str] = []
    for key in ("files", "classes", "functions"):
        values = missed_results.get(key, ())
        if values:
            parts.append(f"{key}: {', '.join(values)}")
    return "; ".join(parts) if parts else "None"


def example_benchmark_json() -> str:
    return """{
  "repository": "sample-repo",
  "benchmarks": [
    {
      "query": "Which file handles JWT validation?",
      "expected_files": ["JwtUtil.java"],
      "expected_classes": ["JwtUtil"],
      "expected_functions": ["verifyToken"]
    }
  ]
}"""


def _vector_execution(service: VectorService, query: str, top_k: int) -> StrategyExecution:
    execution = service.search_vector_only_with_diagnostics(query, top_k=top_k)
    return _to_strategy_execution(execution)


def _hybrid_execution(service: VectorService, query: str, top_k: int, enable_rerank: bool) -> StrategyExecution:
    execution = service.search_hybrid_with_diagnostics(query, top_k=top_k, enable_rerank=enable_rerank)
    return _to_strategy_execution(execution)


def _to_strategy_execution(execution: SearchExecution) -> StrategyExecution:
    return StrategyExecution(
        results=execution.results,
        retrieval_time_seconds=execution.diagnostics.retrieval_time_seconds,
        reranking_time_seconds=execution.diagnostics.reranking_time_seconds,
        total_query_time_seconds=execution.diagnostics.total_query_time_seconds,
    )
