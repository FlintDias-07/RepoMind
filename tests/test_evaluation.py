import json
from pathlib import Path

from embedding.embedding_manager import EmbeddingManager
from embedding.embedding_service import EmbeddingService
from embedding.model_config import EmbeddingModelConfig
from evaluation.benchmark_loader import BenchmarkCase, BenchmarkLoader, BenchmarkSuite
from evaluation.dashboard import format_missed_results, run_repository_benchmark_suite
from evaluation.exporter import EvaluationExporter
from evaluation.metrics import MetricsCalculator
from evaluation.report_generator import EvaluationReportGenerator
from parser.chunk_model import CodeChunk
from retrieval.file_classifier import FileCategory
from vectorstore.chroma_manager import ChromaManager, ChromaManagerConfig
from vectorstore.vector_service import VectorService


class FlatVectorSentenceTransformer:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if any(term in lowered for term in {"jwt", "token", "auth", "login", "config", "database"}):
                vectors.append([1.0, 0.0, 0.0])
            else:
                vectors.append([0.4, 0.3, 0.3])
        return vectors


class FakeCrossEncoder:
    def predict(self, pairs):
        scores = []
        for query, document in pairs:
            lowered_query = query.lower()
            lowered_doc = document.lower()
            score = 0.1
            if "jwt" in lowered_query and "jwtutil" in lowered_doc:
                score = 0.95
            elif "database" in lowered_query and "application.yml" in lowered_doc:
                score = 0.9
            elif "login" in lowered_query and "logincontroller" in lowered_doc:
                score = 0.88
            scores.append(score)
        return scores


def make_chunk(chunk_id, file_name, file_path, content, category, function_name="", class_name="", language="java"):
    chunk = CodeChunk(
        chunk_id=chunk_id,
        file_name=file_name,
        file_path=file_path,
        language=language,
        repository_name="sample-repo",
        content=content,
        chunk_type="method" if class_name else "function",
        class_name=class_name or None,
        function_name=function_name or None,
        start_line=1,
        end_line=5,
    )
    chunk.add_metadata("file_category", category)
    return chunk


def build_service(tmp_path):
    embedding_service = EmbeddingService(
        config=EmbeddingModelConfig(cache_enabled=False, cache_dir=Path(tmp_path) / "cache"),
        model=FlatVectorSentenceTransformer(),
    )
    embedding_manager = EmbeddingManager(embedding_service=embedding_service)
    chroma_manager = ChromaManager(ChromaManagerConfig(persist_directory=Path(tmp_path) / "chroma"))
    service = VectorService(
        repository_name="sample-repo",
        chroma_manager=chroma_manager,
        embedding_manager=embedding_manager,
    )
    service.hybrid_pipeline.reranker.model = FakeCrossEncoder()
    return service


def sample_chunks():
    return [
        make_chunk(
            "jwt-util",
            "JwtUtil.java",
            "src/security/JwtUtil.java",
            "class JwtUtil { String verifyToken(String jwt) { return jwt; } }",
            FileCategory.SOURCE_CODE,
            function_name="verifyToken",
            class_name="JwtUtil",
        ),
        make_chunk(
            "config",
            "application.yml",
            "config/application.yml",
            "application.yml\ndatabase:\n  host: localhost",
            FileCategory.CONFIGURATION,
            function_name="database_config",
            language="yaml",
        ),
        make_chunk(
            "login-controller",
            "LoginController.java",
            "src/controllers/LoginController.java",
            "@RestController class LoginController { void login() {} }",
            FileCategory.SOURCE_CODE,
            function_name="login",
            class_name="LoginController",
        ),
    ]


def test_benchmark_loader_round_trips_suite(tmp_path):
    loader = BenchmarkLoader(tmp_path)
    suite = BenchmarkSuite(
        repository_name="sample-repo",
        benchmarks=[
            BenchmarkCase(
                query="Which file handles JWT validation?",
                expected_files=("JwtUtil.java",),
                expected_classes=("JwtUtil",),
                expected_functions=("verifyToken",),
            )
        ],
    )

    output_path = loader.save_suite(suite)
    loaded = loader.load_file(output_path)

    assert output_path.exists()
    assert loaded.repository_name == "sample-repo"
    assert loaded.benchmarks[0].query == suite.benchmarks[0].query
    assert loaded.benchmarks[0].expected_files == ("JwtUtil.java",)


def test_metrics_and_exporter_generate_expected_report_shapes(tmp_path):
    loader = BenchmarkLoader(tmp_path)
    suite = BenchmarkSuite(
        repository_name="sample-repo",
        benchmarks=[
            BenchmarkCase(
                query="Where is database configuration?",
                expected_files=("application.yml",),
                expected_functions=("database_config",),
            )
        ],
    )
    loader.save_suite(suite)
    service = build_service(tmp_path)
    service.reindex_chunks(sample_chunks())

    reports = run_repository_benchmark_suite(service, [suite], top_k=5)
    generator = EvaluationReportGenerator()
    exporter = EvaluationExporter(generator)

    comparison_rows = generator.build_comparison_rows(reports)
    repository_rows = generator.build_repository_rows(reports)
    query_rows = generator.build_query_rows(reports)

    assert {"vector", "hybrid", "hybrid_reranked"} <= set(reports.keys())
    assert comparison_rows
    assert repository_rows
    assert query_rows

    json_payload = json.loads(exporter.to_json(reports))
    markdown_payload = exporter.to_markdown(reports)
    csv_payload = exporter.to_csv(reports, section="queries")

    assert "hybrid_reranked" in json_payload
    assert "# Retrieval Evaluation Report" in markdown_payload
    assert "strategy,query" in csv_payload


def test_benchmark_runner_produces_per_strategy_metrics_and_timings(tmp_path):
    suite = BenchmarkSuite(
        repository_name="sample-repo",
        benchmarks=[
            BenchmarkCase(
                query="Which file handles JWT validation?",
                expected_files=("JwtUtil.java",),
                expected_classes=("JwtUtil",),
                expected_functions=("verifyToken",),
            ),
            BenchmarkCase(
                query="Where is database configuration?",
                expected_files=("application.yml",),
                expected_functions=("database_config",),
            ),
        ],
    )
    service = build_service(tmp_path)
    service.reindex_chunks(sample_chunks())

    reports = run_repository_benchmark_suite(service, [suite], top_k=5)

    assert reports["vector"].overall_metrics.total_queries == 2
    assert reports["hybrid"].overall_metrics.average_reranking_time_seconds == 0.0
    assert reports["hybrid_reranked"].overall_metrics.average_total_query_time_seconds >= 0.0
    assert reports["hybrid_reranked"].overall_metrics.mrr >= reports["vector"].overall_metrics.mrr
    assert any(
        result.cross_encoder_score > 0
        for record in reports["hybrid_reranked"].query_records
        for result in record.retrieved_results
    )

    hybrid_query = reports["hybrid_reranked"].query_records[0]
    assert hybrid_query.retrieved_results
    assert hybrid_query.retrieved_results[0].file_name
    assert format_missed_results(hybrid_query.missed_results) in {"None", "files: JwtUtil.java", "classes: JwtUtil", "functions: verifyToken"}
