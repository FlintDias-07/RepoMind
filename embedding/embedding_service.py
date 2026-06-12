"""SentenceTransformer-backed embedding generation service."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from parser.chunk_model import CodeChunk
from utils import get_logger

from .model_config import EmbeddingModelConfig


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


@dataclass
class ModelInfo:
    """Lightweight description of the active embedding model."""

    model_name: str
    device: Optional[str]
    dimension: Optional[int]
    cache_enabled: bool


class EmbeddingService:
    """Generate and cache vector embeddings for text and CodeChunk objects."""

    def __init__(self, config: Optional[EmbeddingModelConfig] = None, model: Any = None):
        self.config = config or EmbeddingModelConfig.from_env()
        self.model = model
        self.logger = get_logger(self.__class__.__name__)
        self._cache: Dict[str, List[float]] = {}
        self._dimension: Optional[int] = None
        self._load_cache()

    def _ensure_model(self) -> None:
        if self.model is not None:
            if self._dimension is None and hasattr(self.model, "get_sentence_embedding_dimension"):
                try:
                    self._dimension = int(self.model.get_sentence_embedding_dimension())
                except Exception:
                    self._dimension = None
            return

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - import guard
            raise EmbeddingError("sentence-transformers is not installed") from exc

        try:
            self.logger.info("Loading embedding model: %s", self.config.model_name)
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
                trust_remote_code=self.config.trust_remote_code,
            )
            if self.config.max_seq_length:
                self.model.max_seq_length = self.config.max_seq_length
            if hasattr(self.model, "get_sentence_embedding_dimension"):
                self._dimension = int(self.model.get_sentence_embedding_dimension())
        except Exception as exc:
            raise EmbeddingError(f"Failed to load embedding model '{self.config.model_name}': {exc}") from exc

    def _cache_key(self, text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if not self.config.cache_enabled:
            return

        cache_file = self.config.cache_file
        if not cache_file.exists():
            return

        try:
            with cache_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                self._cache = {
                    key: [float(value) for value in vector]
                    for key, vector in payload.items()
                    if isinstance(vector, list)
                }
                self.logger.info("Loaded %s cached embeddings", len(self._cache))
        except Exception as exc:
            self.logger.warning("Unable to load embedding cache: %s", exc)

    def save_cache(self) -> None:
        """Persist the in-memory embedding cache to disk."""

        if not self.config.cache_enabled or not self._cache:
            return

        try:
            with self.config.cache_file.open("w", encoding="utf-8") as handle:
                json.dump(self._cache, handle)
        except Exception as exc:
            self.logger.warning("Unable to save embedding cache: %s", exc)

    def get_model_info(self) -> ModelInfo:
        """Return model metadata for logging and diagnostics."""

        self._ensure_model()
        return ModelInfo(
            model_name=self.config.model_name,
            device=self.config.device,
            dimension=self._dimension,
            cache_enabled=self.config.cache_enabled,
        )

    def _encode(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []

        self._ensure_model()

        try:
            embeddings = self.model.encode(  # type: ignore[union-attr]
                list(texts),
                batch_size=self.config.batch_size,
                show_progress_bar=self.config.show_progress_bar,
                normalize_embeddings=self.config.normalize_embeddings,
            )
            array = np.asarray(embeddings, dtype=float)
            array = np.atleast_2d(array)
            return array.tolist()
        except Exception as exc:
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""

        if not text or not text.strip():
            raise ValueError("text cannot be empty")

        key = self._cache_key(text)
        if self.config.cache_enabled and key in self._cache:
            return list(self._cache[key])

        embedding = self._encode([text])[0]
        if self.config.cache_enabled:
            self._cache[key] = embedding
        return embedding

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a list of text strings while using caching and batch calls."""

        if not texts:
            return []

        normalized_texts = [text if text is not None else "" for text in texts]
        for text in normalized_texts:
            if not text.strip():
                raise ValueError("texts cannot contain empty entries")

        results: List[Optional[List[float]]] = [None] * len(normalized_texts)
        pending: List[Tuple[int, str, str]] = []
        seen_pending: Dict[str, int] = {}

        for index, text in enumerate(normalized_texts):
            key = self._cache_key(text)
            if self.config.cache_enabled and key in self._cache:
                results[index] = list(self._cache[key])
                continue

            if key not in seen_pending:
                seen_pending[key] = len(pending)
                pending.append((index, key, text))
            else:
                original_index, _, _ = pending[seen_pending[key]]
                pending.append((index, key, text))

        if pending:
            unique_pairs: List[Tuple[str, str]] = []
            unique_indices: Dict[str, List[int]] = {}
            for index, key, text in pending:
                if key not in unique_indices:
                    unique_pairs.append((key, text))
                    unique_indices[key] = []
                unique_indices[key].append(index)

            generated = self._encode([text for _, text in unique_pairs])
            for (key, _), embedding in zip(unique_pairs, generated):
                if self.config.cache_enabled:
                    self._cache[key] = embedding
                for index in unique_indices[key]:
                    results[index] = list(embedding)

        return [embedding for embedding in results if embedding is not None]

    def embed_chunk(self, chunk: CodeChunk) -> List[float]:
        """Embed a single CodeChunk."""

        return self.embed_text(chunk.content)

    def embed_chunks(self, chunks: Sequence[CodeChunk]) -> List[List[float]]:
        """Embed multiple CodeChunks."""

        return self.embed_texts([chunk.content for chunk in chunks])

    def close(self) -> None:
        """Flush the embedding cache to disk."""

        self.save_cache()
