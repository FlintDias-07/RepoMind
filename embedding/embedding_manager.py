"""High-level orchestration for chunk embedding generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime

from parser.chunk_model import CodeChunk, ChunkStatistics
from utils import get_logger

from .embedding_service import EmbeddingService


@dataclass
class EmbeddedChunk:
    """Container for a chunk, its embedding, and its normalized metadata."""

    chunk: CodeChunk
    embedding: List[float]
    metadata: Dict[str, Any]


@dataclass
class EmbeddingBatchResult:
    """Summary of a batch embedding run."""

    repository_name: str
    total_chunks: int
    embedded_chunks: int
    skipped_chunks: int
    average_vector_size: int
    chunks_by_language: Dict[str, int] = field(default_factory=dict)
    average_chunk_size: int = 0
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class EmbeddingManager:
    """Create embeddings and metadata records from CodeChunk inputs."""

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.logger = get_logger(self.__class__.__name__)

    def _validate_chunk(self, chunk: CodeChunk) -> bool:
        return bool(chunk and chunk.content and chunk.content.strip())

    def build_metadata(self, chunk: CodeChunk) -> Dict[str, Any]:
        """Normalize the metadata schema expected by the vector store."""

        return {
            "chunk_id": chunk.chunk_id,
            "file_name": chunk.file_name,
            "file_path": chunk.file_path,
            "language": chunk.language,
            "class_name": chunk.class_name or "",
            "function_name": chunk.function_name or "",
            "start_line": int(chunk.start_line or 0),
            "end_line": int(chunk.end_line or 0),
        }

    def embed_chunks(self, chunks: Sequence[CodeChunk]) -> List[EmbeddedChunk]:
        """Generate embeddings for all valid chunks in a single batch."""

        valid_chunks = [chunk for chunk in chunks if self._validate_chunk(chunk)]
        if not valid_chunks:
            return []

        self.logger.info("Generating embeddings for %s chunks", len(valid_chunks))
        embeddings = self.embedding_service.embed_chunks(valid_chunks)

        embedded_chunks: List[EmbeddedChunk] = []
        for chunk, embedding in zip(valid_chunks, embeddings):
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk=chunk,
                    embedding=embedding,
                    metadata=self.build_metadata(chunk),
                )
            )

        return embedded_chunks

    def build_statistics(
        self,
        repository_name: str,
        chunks: Sequence[CodeChunk],
        embedded_chunks: Sequence[EmbeddedChunk],
        duration_seconds: float,
    ) -> EmbeddingBatchResult:
        """Create a summary for embedding operations."""

        statistics = ChunkStatistics(list(chunks)) if chunks else None
        vector_sizes = [len(record.embedding) for record in embedded_chunks]
        return EmbeddingBatchResult(
            repository_name=repository_name,
            total_chunks=len(chunks),
            embedded_chunks=len(embedded_chunks),
            skipped_chunks=max(0, len(chunks) - len(embedded_chunks)),
            average_vector_size=(sum(vector_sizes) // len(vector_sizes)) if vector_sizes else 0,
            chunks_by_language=(statistics.by_language if statistics else {}),
            average_chunk_size=(statistics.to_dict()["avg_chunk_size"] if statistics else 0),
            duration_seconds=duration_seconds,
        )
