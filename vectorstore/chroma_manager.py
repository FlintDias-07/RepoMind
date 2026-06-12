"""ChromaDB client and collection lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import re

import chromadb
from chromadb.config import Settings

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

    def __init__(self, config: Optional[ChromaManagerConfig] = None):
        self.config = config or ChromaManagerConfig()
        self.config.persist_directory = Path(self.config.persist_directory)
        self.config.persist_directory.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(self.__class__.__name__)
        self._client = None

    def get_client(self):
        """Return a lazily initialized persistent Chroma client."""

        if self._client is None:
            try:
                self._client = chromadb.PersistentClient(
                    path=str(self.config.persist_directory),
                    settings=Settings(anonymized_telemetry=False),
                )
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
