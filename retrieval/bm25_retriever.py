"""Repository-level BM25 retrieval manager."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence

from parser.chunk_model import CodeChunk
from utils import get_logger

from .bm25_index import BM25Index


@dataclass(frozen=True)
class BM25Result:
    chunk_id: str
    repository_name: str
    file_name: str
    file_path: str
    language: str
    class_name: str
    function_name: str
    document: str
    metadata: Dict[str, Any]
    bm25_score: float


class BM25Retriever:
    """Maintain one BM25 index per repository and execute lexical search."""

    _shared_indexes: Dict[str, BM25Index] = {}
    _shared_lock = RLock()

    def __init__(self, index_factory: Optional[type[BM25Index]] = None):
        self.logger = get_logger(self.__class__.__name__)
        self.index_factory = index_factory or BM25Index
        self.indexes = self._shared_indexes

    def document_count(self, repository_name: str) -> int:
        index = self.indexes.get(repository_name)
        return len(index.documents) if index else 0

    def rebuild_index(self, repository_name: str, chunks: Sequence[CodeChunk]) -> None:
        index = self.index_factory(repository_name)
        index.build(chunks)
        with self._shared_lock:
            self.indexes[repository_name] = index

    def invalidate_index(self, repository_name: str) -> None:
        with self._shared_lock:
            self.indexes.pop(repository_name, None)

    def clear_all(self) -> None:
        with self._shared_lock:
            self.indexes.clear()

    def has_index(self, repository_name: str) -> bool:
        return self.document_count(repository_name) > 0

    def search(self, repository_name: str, query: str, top_k: int = 20) -> List[BM25Result]:
        index = self.indexes.get(repository_name)
        if index is None:
            return []

        results: List[BM25Result] = []
        for document, score in index.search(query, top_k=top_k):
            if document.repository_name != repository_name:
                continue
            results.append(
                BM25Result(
                    chunk_id=document.chunk_id,
                    repository_name=document.repository_name,
                    file_name=document.file_name,
                    file_path=document.file_path,
                    language=document.language,
                    class_name=document.class_name,
                    function_name=document.function_name,
                    document=document.content,
                    metadata=document.metadata,
                    bm25_score=score,
                )
            )
        self.logger.info(
            "BM25 search for repository '%s' returned %s results",
            repository_name,
            len(results),
        )
        return results
