"""High-level indexing and semantic search service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from parser.chunk_model import CodeChunk
from utils import get_logger

from embedding.embedding_manager import EmbeddingManager, EmbeddedChunk
from embedding.embedding_service import EmbeddingService

from .chroma_manager import ChromaManager
from .collection_manager import CollectionManager, CollectionStatistics


@dataclass
class SearchResult:
    """Normalized search result returned from semantic retrieval."""

    chunk_id: str
    similarity_score: float
    distance: float
    file_name: str
    file_path: str
    language: str
    class_name: str
    function_name: str
    start_line: int
    end_line: int
    document: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        if self.class_name and self.function_name:
            return f"{self.class_name}.{self.function_name}()"
        if self.function_name:
            return f"{self.function_name}()"
        if self.class_name:
            return f"class {self.class_name}"
        return self.file_path


@dataclass
class IndexResult:
    """Summary of an indexing run."""

    repository_name: str
    total_chunks: int
    embedded_chunks: int
    stored_chunks: int
    skipped_chunks: int
    duplicate_chunks: int
    total_files_processed: int
    total_repositories_indexed: int
    chunks_by_language: Dict[str, int] = field(default_factory=dict)
    average_chunk_size: int = 0
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class VectorService:
    """Coordinate embedding generation, persistence, and retrieval."""

    def __init__(
        self,
        repository_name: str,
        collection_name: Optional[str] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        chroma_manager: Optional[ChromaManager] = None,
    ):
        self.repository_name = repository_name
        self.logger = get_logger(self.__class__.__name__)
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.chroma_manager = chroma_manager or ChromaManager()
        self.collection_manager = CollectionManager(
            repository_name=repository_name,
            chroma_manager=self.chroma_manager,
            collection_name=collection_name or repository_name,
        )

    def index_chunks(self, chunks: Sequence[CodeChunk], reindex: bool = False) -> IndexResult:
        """Embed and store chunks in ChromaDB."""

        start_time = datetime.now()
        valid_chunks = [chunk for chunk in chunks if chunk and chunk.content and chunk.content.strip()]
        embedded_chunks = self.embedding_manager.embed_chunks(valid_chunks)
        storage_result = self.collection_manager.store_embedded_chunks(embedded_chunks, reindex=reindex)
        collection_stats = self.collection_manager.get_statistics()
        duration = (datetime.now() - start_time).total_seconds()

        return IndexResult(
            repository_name=self.repository_name,
            total_chunks=len(valid_chunks),
            embedded_chunks=len(embedded_chunks),
            stored_chunks=storage_result.get("stored", 0),
            skipped_chunks=storage_result.get("skipped", 0),
            duplicate_chunks=storage_result.get("duplicates", 0),
            total_files_processed=collection_stats.total_files_processed,
            total_repositories_indexed=collection_stats.total_repositories_indexed,
            chunks_by_language=collection_stats.chunks_by_language,
            average_chunk_size=collection_stats.average_chunk_size,
            duration_seconds=duration,
        )

    def reindex_chunks(self, chunks: Sequence[CodeChunk]) -> IndexResult:
        """Force a full re-index of the active repository collection."""

        return self.index_chunks(chunks, reindex=True)

    def _distance_to_similarity(self, distance: float) -> float:
        """Convert Chroma distance values into a bounded similarity score."""

        return 1.0 / (1.0 + max(0.0, float(distance)))

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Run semantic search over the indexed chunks."""

        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        embedding = self.embedding_manager.embedding_service.embed_text(query)
        results = self.collection_manager.query(embedding, top_k=top_k)

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]

        search_results: List[SearchResult] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    similarity_score=self._distance_to_similarity(distance),
                    distance=float(distance),
                    file_name=metadata.get("file_name", ""),
                    file_path=metadata.get("file_path", ""),
                    language=metadata.get("language", ""),
                    class_name=metadata.get("class_name", ""),
                    function_name=metadata.get("function_name", ""),
                    start_line=int(metadata.get("start_line", 0) or 0),
                    end_line=int(metadata.get("end_line", 0) or 0),
                    document=document or "",
                    metadata=metadata,
                )
            )

        return search_results

    def get_statistics(self) -> Dict[str, Any]:
        """Return repository-level indexing statistics."""

        stats: CollectionStatistics = self.collection_manager.get_statistics()
        return {
            "repository_name": self.repository_name,
            "total_repositories_indexed": stats.total_repositories_indexed,
            "total_files_processed": stats.total_files_processed,
            "total_chunks_created": stats.total_chunks_created,
            "total_embeddings_stored": stats.total_embeddings_stored,
            "chunks_by_language": stats.chunks_by_language,
            "average_chunk_size": stats.average_chunk_size,
            "collection_name": stats.collection_name,
        }
