"""ChromaDB client and collection lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Dict, List, Optional
import re

from config import CHROMA_DB_PATH
from utils import get_logger


def _sanitize_collection_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "_", name)
    name = name.strip("._-")
    if not name:
        return "rag_repository"
    return name[:63]


@dataclass
class ChromaManagerConfig:
    """Configuration used to initialize the Chroma client."""

    persist_directory: Path = Path(CHROMA_DB_PATH)
    telemetry_enabled: bool = False


class ChromaManager:
    """Own the ChromaDB client and collection creation logic."""

    _client_registry: Dict[str, object] = {}
    _client_creation_counts: Dict[str, int] = {}
    _client_creation_times: Dict[str, float] = {}
    _client_lock = RLock()

    def __init__(self, config: Optional[ChromaManagerConfig] = None):
        self.config = config or ChromaManagerConfig()
        self.config.persist_directory = Path(self.config.persist_directory)
        self.config.persist_directory.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(self.__class__.__name__)
        self._client = None
        self.client_cache_hit = False
        self.client_creation_count = 0
        self.client_creation_time_seconds = 0.0

    @classmethod
    def reset_client_cache(cls) -> None:
        with cls._client_lock:
            cls._client_registry.clear()
            cls._client_creation_counts.clear()
            cls._client_creation_times.clear()

    @classmethod
    def client_cache_size(cls) -> int:
        return len(cls._client_registry)

    def get_client(self):
        """Return a lazily initialized persistent Chroma client."""

        if self._client is None:
            cache_key = str(self.config.persist_directory.resolve())
            try:
                import chromadb
                from chromadb.config import Settings

                with self._client_lock:
                    cached_client = self._client_registry.get(cache_key)
                    if cached_client is not None:
                        self._client = cached_client
                        self.client_cache_hit = True
                        self.client_creation_count = self._client_creation_counts.get(cache_key, 0)
                        self.client_creation_time_seconds = self._client_creation_times.get(cache_key, 0.0)
                        return self._client

                    started = perf_counter()
                    self._client = chromadb.PersistentClient(
                        path=str(self.config.persist_directory),
                        settings=Settings(anonymized_telemetry=False),
                    )
                    creation_time = perf_counter() - started
                    self._client_registry[cache_key] = self._client
                    self._client_creation_counts[cache_key] = self._client_creation_counts.get(cache_key, 0) + 1
                    self._client_creation_times[cache_key] = creation_time
                    self.client_cache_hit = False
                    self.client_creation_count = self._client_creation_counts[cache_key]
                    self.client_creation_time_seconds = creation_time
            except Exception as exc:
                raise RuntimeError(f"Unable to connect to ChromaDB: {exc}") from exc
        return self._client

    def get_or_create_collection(self, collection_name: str, metadata: Optional[Dict] = None):
        """Get or create a collection with the requested name."""

        collection_name = _sanitize_collection_name(collection_name)
        client = self.get_client()
        collection_metadata = {"hnsw:space": "cosine"}
        if metadata:
            collection_metadata.update(metadata)
        return client.get_or_create_collection(
            name=collection_name,
            metadata=collection_metadata,
            embedding_function=None,
        )

    def list_collections(self) -> List[str]:
        """Return all known collection names."""

        client = self.get_client()
        collections = client.list_collections()
        return [collection.name for collection in collections]

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection if it exists."""

        collection_name = _sanitize_collection_name(collection_name)
        client = self.get_client()
        try:
            client.delete_collection(collection_name)
            return True
        except Exception:
            return False
