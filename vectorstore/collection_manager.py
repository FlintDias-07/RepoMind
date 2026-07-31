"""Collection-level helpers for storing and querying embedded chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import math

from parser.chunk_model import CodeChunk
from utils import get_logger

from embedding.embedding_manager import EmbeddedChunk
from embedding.embedding_service import EmbeddingService

from .chroma_manager import ChromaManager, _sanitize_collection_name


@dataclass
class CollectionStatistics:
    """Aggregate statistics for a Chroma collection."""

    collection_name: str
    total_documents: int = 0
    total_embeddings: int = 0
    total_repositories_indexed: int = 0
    total_files_processed: int = 0
    total_chunks_created: int = 0
    total_embeddings_stored: int = 0
    chunks_by_language: Dict[str, int] = field(default_factory=dict)
    average_chunk_size: int = 0


@dataclass
class CollectionIntegrityReport:
    """Integrity checks for a stored Chroma collection."""

    collection_name: str
    total_vectors: int
    duplicate_chunk_ids: List[str] = field(default_factory=list)
    missing_metadata_ids: List[str] = field(default_factory=list)
    empty_document_ids: List[str] = field(default_factory=list)
    empty_embedding_ids: List[str] = field(default_factory=list)


class CollectionManager:
    """Wrap a Chroma collection with indexing and retrieval helpers."""

    def __init__(
        self,
        repository_name: str,
        chroma_manager: Optional[ChromaManager] = None,
        collection_name: Optional[str] = None,
    ):
        self.repository_name = repository_name
        self.chroma_manager = chroma_manager or ChromaManager()
        self.collection_name = collection_name or repository_name
        self.logger = get_logger(self.__class__.__name__)
        self._collection = None

    @property
    def normalized_collection_name(self) -> str:
        return _sanitize_collection_name(self.collection_name)

    def get_collection(self):
        if self._collection is None:
            self._collection = self.chroma_manager.get_or_create_collection(
                self.normalized_collection_name,
                metadata={"repository_name": self.repository_name},
            )
        return self._collection

    def reset_collection(self) -> None:
        """Delete and recreate the active collection."""

        self.chroma_manager.delete_collection(self.normalized_collection_name)
        self._collection = None
        self.get_collection()

    def build_record(self, embedded_chunk: EmbeddedChunk) -> Dict[str, Any]:
        """Build a Chroma insert record from an embedded chunk."""

        chunk = embedded_chunk.chunk
        metadata = dict(embedded_chunk.metadata)
        metadata.setdefault("chunk_id", chunk.chunk_id)
        metadata.setdefault("file_name", chunk.file_name)
        metadata.setdefault("file_path", chunk.file_path)
        metadata.setdefault("language", chunk.language)
        metadata.setdefault("repository_name", chunk.repository_name)
        metadata.setdefault("chunk_type", chunk.chunk_type)
        metadata.setdefault("class_name", chunk.class_name or "")
        metadata.setdefault("function_name", chunk.function_name or "")
        metadata.setdefault("start_line", int(chunk.start_line or 0))
        metadata.setdefault("end_line", int(chunk.end_line or 0))
        metadata.setdefault("file_category", chunk.metadata.get("file_category", "source_code"))

        return {
            "id": chunk.chunk_id,
            "document": chunk.content,
            "embedding": embedded_chunk.embedding,
            "metadata": metadata,
        }

    def get_existing_ids(self, chunk_ids: Sequence[str]) -> set:
        """Return the subset of chunk ids already stored in the collection."""

        if not chunk_ids:
            return set()

        collection = self.get_collection()
        try:
            existing = collection.get(ids=list(chunk_ids), include=[])
            return set(existing.get("ids", []))
        except Exception:
            return set()

    def store_embedded_chunks(self, embedded_chunks: Sequence[EmbeddedChunk], reindex: bool = False) -> Dict[str, Any]:
        """Insert embedded chunks into Chroma, skipping duplicates by default."""

        if reindex:
            self.reset_collection()

        collection = self.get_collection()
        records = [self.build_record(item) for item in embedded_chunks if item and item.chunk.content.strip()]
        if not records:
            return {"stored": 0, "skipped": 0, "duplicates": 0}

        if reindex or collection.count() == 0:
            existing_ids = set()
        else:
            existing_ids = self.get_existing_ids([record["id"] for record in records])
        to_store = [record for record in records if record["id"] not in existing_ids]
        duplicates = len(records) - len(to_store)

        if not to_store:
            return {"stored": 0, "skipped": len(records), "duplicates": duplicates}

        collection.upsert(
            ids=[record["id"] for record in to_store],
            documents=[record["document"] for record in to_store],
            metadatas=[record["metadata"] for record in to_store],
            embeddings=[record["embedding"] for record in to_store],
        )

        return {"stored": len(to_store), "skipped": len(records) - len(to_store), "duplicates": duplicates}

    def query(self, embedding: List[float], top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the collection using a vector embedding."""

        collection = self.get_collection()
        query_args = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_args["where"] = where
        return collection.query(**query_args)

    def get_statistics(self) -> CollectionStatistics:
        """Compute statistics from the current collection contents."""

        collection = self.get_collection()
        count = collection.count()
        stats = CollectionStatistics(
            collection_name=self.normalized_collection_name,
            total_documents=count,
            total_embeddings=count,
            total_repositories_indexed=len(self.chroma_manager.list_collections()),
            total_chunks_created=count,
            total_embeddings_stored=count,
        )

        if count == 0:
            return stats

        data = collection.get(include=["documents", "metadatas"])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        total_size = 0
        file_paths = set()
        by_language: Dict[str, int] = {}

        for document, metadata in zip(documents, metadatas):
            total_size += len(document or "")
            if isinstance(metadata, dict):
                file_paths.add(metadata.get("file_path", ""))
                language = metadata.get("language", "unknown") or "unknown"
                by_language[language] = by_language.get(language, 0) + 1

        stats.total_files_processed = len({path for path in file_paths if path})
        stats.chunks_by_language = by_language
        stats.average_chunk_size = math.floor(total_size / count) if count else 0
        return stats

    def inspect_integrity(self) -> CollectionIntegrityReport:
        """Inspect stored vectors and metadata for consistency issues."""

        collection = self.get_collection()
        count = collection.count()
        if count == 0:
            return CollectionIntegrityReport(
                collection_name=self.normalized_collection_name,
                total_vectors=0,
            )

        data = collection.get(include=["documents", "metadatas", "embeddings"])
        ids = data.get("ids", []) or []
        documents = data.get("documents", []) or []
        metadatas = data.get("metadatas", []) or []
        embeddings = data.get("embeddings", []) or []

        seen_ids = set()
        duplicate_ids: List[str] = []
        missing_metadata_ids: List[str] = []
        empty_document_ids: List[str] = []
        empty_embedding_ids: List[str] = []

        for chunk_id, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            if chunk_id in seen_ids:
                duplicate_ids.append(chunk_id)
            seen_ids.add(chunk_id)

            if not isinstance(metadata, dict):
                missing_metadata_ids.append(chunk_id)
            else:
                required_keys = {
                    "chunk_id",
                    "file_name",
                    "file_path",
                    "language",
                    "file_category",
                    "start_line",
                    "end_line",
                }
                if not required_keys.issubset(metadata.keys()):
                    missing_metadata_ids.append(chunk_id)

            if not document or not str(document).strip():
                empty_document_ids.append(chunk_id)

            if embedding is None or len(embedding) == 0:
                empty_embedding_ids.append(chunk_id)

        return CollectionIntegrityReport(
            collection_name=self.normalized_collection_name,
            total_vectors=count,
            duplicate_chunk_ids=duplicate_ids,
            missing_metadata_ids=missing_metadata_ids,
            empty_document_ids=empty_document_ids,
            empty_embedding_ids=empty_embedding_ids,
        )
