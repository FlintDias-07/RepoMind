from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from parser.parser_manager import ParserManager
from vectorstore.vector_service import VectorService


def main() -> None:
    repo_path = PROJECT_ROOT
    repo_name = "runtime-smoke-rag-app"

    parser = ParserManager(str(repo_path), repo_name)
    chunks, report = parser.parse_repository()
    if report.get("error"):
        raise RuntimeError(report["error"])
    if not chunks:
        raise RuntimeError("No chunks were generated during runtime smoke test.")

    vector_service = VectorService(repository_name=repo_name)
    index_result = vector_service.reindex_chunks(chunks)

    queries = [
        "authentication",
        "Where is database config?",
        "How authentication works?",
        "Which file handles JWT?",
    ]

    search_payload = {}
    for query in queries:
        results = vector_service.search(query, top_k=3)
        search_payload[query] = [
            {
                "file_name": item.file_name,
                "file_path": item.file_path,
                "display_name": item.display_name,
                "score": round(item.similarity_score, 4),
            }
            for item in results
        ]

    print(
        json.dumps(
            {
                "files_processed": report.get("files_processed", 0),
                "chunks_generated": len(chunks),
                "embeddings_stored": index_result.stored_chunks,
                "embeddings_generated": index_result.embedded_chunks,
                "duplicates": index_result.duplicate_chunks,
                "indexing_time_seconds": round(index_result.duration_seconds, 2),
                "search_results": search_payload,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
