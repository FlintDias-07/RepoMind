from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from embedding.embedding_manager import EmbeddingManager
from embedding.embedding_service import EmbeddingError, EmbeddingService
from embedding.model_config import EmbeddingModelConfig
from parser.parser_manager import ParserManager
from vectorstore.vector_service import VectorService


QUERIES = [
    "How authentication works?",
    "Which file handles JWT validation?",
    "Where is database configuration?",
    "Explain login flow.",
    "Which service talks to the database?",
]

QUERY_EXPECTATIONS = {
    "How authentication works?": {"auth", "authenticate", "login", "token", "security"},
    "Which file handles JWT validation?": {"jwt", "token", "validate", "auth"},
    "Where is database configuration?": {"database", "config", "db", "connection"},
    "Explain login flow.": {"login", "auth", "session", "token", "user"},
    "Which service talks to the database?": {"database", "db", "repository", "service", "query"},
}


def stage(message: str) -> None:
    print(f"[stage] {message}", flush=True)


def build_vector_service(repository_name: str) -> VectorService:
    config = EmbeddingModelConfig.from_env()
    config.show_progress_bar = True
    embedding_service = EmbeddingService(config=config)
    embedding_manager = EmbeddingManager(embedding_service=embedding_service)
    return VectorService(repository_name=repository_name, embedding_manager=embedding_manager)


def validate_search_results(repo_path: Path, query: str, results: List[Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    normalized_results: List[Dict[str, Any]] = []
    relevant_non_test_result = False
    expected_terms = QUERY_EXPECTATIONS.get(query, set())

    if not results:
        warnings.append("No results found")

    seen_chunk_ids = set()
    for result in results:
        source_path = repo_path / result.file_path
        metadata_ok = bool(
            result.metadata.get("chunk_id") == result.chunk_id
            and result.metadata.get("file_path") == result.file_path
            and result.metadata.get("file_name") == result.file_name
        )
        if result.chunk_id in seen_chunk_ids:
            warnings.append(f"Duplicate chunk returned: {result.chunk_id}")
        seen_chunk_ids.add(result.chunk_id)
        if not source_path.exists():
            warnings.append(f"Missing source file: {result.file_path}")
        if not result.document.strip():
            warnings.append(f"Empty chunk content: {result.chunk_id}")
        if not metadata_ok:
            warnings.append(f"Metadata mismatch: {result.chunk_id}")

        searchable_text = " ".join(
            [
                result.file_name,
                result.file_path,
                result.display_name,
                result.document,
            ]
        ).lower()
        is_test_or_doc = result.file_path.startswith("tests\\") or result.file_path.lower().endswith(".md")
        if expected_terms and any(term in searchable_text for term in expected_terms) and not is_test_or_doc:
            relevant_non_test_result = True

        normalized_results.append(
            {
                "chunk_id": result.chunk_id,
                "original_vector_score": round(result.original_vector_score, 4),
                "bm25_score": round(result.bm25_score, 4),
                "fusion_score": round(result.fusion_score, 4),
                "folder_weight": round(result.folder_weight, 4),
                "metadata_bonus": round(result.metadata_bonus, 4),
                "symbol_boost": round(result.symbol_boost, 4),
                "cross_encoder_score": round(result.cross_encoder_score, 4),
                "reranked_position": result.reranked_position,
                "score": round(result.similarity_score, 4),
                "adjusted_final_score": round(result.adjusted_score, 4),
                "file_name": result.file_name,
                "file_path": result.file_path,
                "symbol": result.display_name,
                "start_line": result.start_line,
                "end_line": result.end_line,
                "content_preview": result.document[:160].strip(),
                "query_intent": result.query_intent,
                "collection_name": result.collection_name,
                "ranking_reasons": result.ranking_reasons,
            }
        )

    if results and not relevant_non_test_result:
        warnings.append("No relevant non-test code result found in top-k results")

    return {
        "query": query,
        "result_count": len(results),
        "warnings": warnings,
        "results": normalized_results,
    }


def main() -> None:
    repo_path = PROJECT_ROOT
    repo_name = "runtime-smoke-rag-app"
    started_at = time.perf_counter()
    status = {
        "Embedding Model": "FAIL",
        "Repository Parsing": "FAIL",
        "Chunk Generation": "FAIL",
        "Embedding Generation": "FAIL",
        "Vector Database": "FAIL",
        "Semantic Retrieval": "FAIL",
    }

    stage("Inspecting embedding model cache")
    vector_service = build_vector_service(repo_name)
    embedding_service = vector_service.embedding_manager.embedding_service
    cache_status = [
        {
            "path": str(item.path),
            "valid": item.valid,
            "missing_files": item.missing_files,
        }
        for item in embedding_service.inspect_model_cache()
    ]

    stage("Loading embedding model")
    model_started = time.perf_counter()
    try:
        model_info = embedding_service.get_model_info()
        status["Embedding Model"] = "PASS"
    except EmbeddingError as exc:
        report = {
            "system_status": status,
            "overall_status": "BLOCKED",
            "error": str(exc),
            "cache_status": cache_status,
        }
        print(json.dumps(report, indent=2))
        raise
    model_load_seconds = round(time.perf_counter() - model_started, 2)

    stage("Parsing repository")
    parse_started = time.perf_counter()
    parser = ParserManager(str(repo_path), repo_name)
    chunks, parse_report = parser.parse_repository()
    parse_seconds = round(time.perf_counter() - parse_started, 2)
    if parse_report.get("error"):
        raise RuntimeError(parse_report["error"])
    status["Repository Parsing"] = "PASS"
    if chunks:
        status["Chunk Generation"] = "PASS"

    stage("Generating embeddings and storing vectors")
    index_started = time.perf_counter()
    index_result = vector_service.reindex_chunks(chunks)
    index_seconds = round(time.perf_counter() - index_started, 2)
    if index_result.embedded_chunks > 0:
        status["Embedding Generation"] = "PASS"

    integrity = vector_service.collection_manager.inspect_integrity()
    collection_stats = vector_service.get_statistics()
    collection_exists = integrity.collection_name in vector_service.chroma_manager.list_collections()
    vector_db_warnings: List[str] = []
    if not collection_exists:
        vector_db_warnings.append("Collection not found in Chroma client")
    if integrity.total_vectors <= 0:
        vector_db_warnings.append("Collection count is zero")
    if integrity.duplicate_chunk_ids:
        vector_db_warnings.append(f"Duplicate IDs: {integrity.duplicate_chunk_ids}")
    if integrity.missing_metadata_ids:
        vector_db_warnings.append(f"Missing metadata IDs: {integrity.missing_metadata_ids}")
    if integrity.empty_document_ids:
        vector_db_warnings.append(f"Empty documents: {integrity.empty_document_ids}")
    if integrity.empty_embedding_ids:
        vector_db_warnings.append(f"Empty embeddings: {integrity.empty_embedding_ids}")
    if not vector_db_warnings:
        status["Vector Database"] = "PASS"

    stage("Running semantic retrieval checks")
    retrieval_reports: List[Dict[str, Any]] = []
    retrieval_times: List[float] = []
    retrieval_warnings: List[str] = []
    for query in QUERIES:
        search_started = time.perf_counter()
        results = vector_service.search(query, top_k=3)
        retrieval_times.append(time.perf_counter() - search_started)
        result_report = validate_search_results(repo_path, query, results)
        retrieval_reports.append(result_report)
        retrieval_warnings.extend([f"{query}: {warning}" for warning in result_report["warnings"]])

    if not retrieval_warnings and any(report["result_count"] > 0 for report in retrieval_reports):
        status["Semantic Retrieval"] = "PASS"

    overall_ready = all(value == "PASS" for value in status.values())
    total_seconds = round(time.perf_counter() - started_at, 2)
    report = {
        "system_status": status,
        "overall_status": "READY FOR PHASE 5 (LLM/RAG PIPELINE)" if overall_ready else "NOT READY",
        "model": {
            "name": model_info.model_name,
            "dimension": model_info.dimension,
            "device": model_info.device,
            "cache_enabled": model_info.cache_enabled,
            "load_time_seconds": model_load_seconds,
            "cache_status": cache_status,
        },
        "indexing": {
            "repositories_indexed": index_result.total_repositories_indexed,
            "files_scanned": parse_report.get("files_processed", 0),
            "chunks_created": len(chunks),
            "embeddings_generated": index_result.embedded_chunks,
            "vectors_stored": integrity.total_vectors,
            "duplicates_detected": index_result.duplicate_chunks,
            "average_embedding_time_seconds": round(
                index_seconds / max(index_result.embedded_chunks, 1),
                4,
            ),
            "processing_time_seconds": index_seconds,
            "parse_time_seconds": parse_seconds,
        },
        "vector_database": {
            "collection_name": integrity.collection_name,
            "collection_exists": collection_exists,
            "collection_count": integrity.total_vectors,
            "insertion_time_seconds": round(index_result.duration_seconds, 2),
            "metadata_integrity_ok": not integrity.missing_metadata_ids,
            "empty_embeddings": integrity.empty_embedding_ids,
            "empty_documents": integrity.empty_document_ids,
            "duplicates": integrity.duplicate_chunk_ids,
            "warnings": vector_db_warnings,
            "stats": collection_stats,
        },
        "retrieval": {
            "queries": retrieval_reports,
            "average_search_time_seconds": round(sum(retrieval_times) / max(len(retrieval_times), 1), 4),
            "warnings": retrieval_warnings,
        },
        "timing": {
            "total_time_seconds": total_seconds,
        },
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
