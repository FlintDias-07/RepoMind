"""SentenceTransformer-backed embedding generation service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import inspect
import json
from pathlib import Path
import shutil
from threading import RLock
from time import perf_counter
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


@dataclass
class ModelCacheStatus:
    """Describe the state of a local embedding-model snapshot."""

    path: Path
    valid: bool
    missing_files: List[str]
    source: str


class EmbeddingService:
    """Generate and cache vector embeddings for text and CodeChunk objects."""

    REQUIRED_MODEL_FILES = ("config.json", "model.safetensors")
    TOKENIZER_FILES = (
        "tokenizer.json",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
        "vocab.txt",
    )
    _model_registry: Dict[Tuple[str, Optional[str], str, Optional[int], bool], Any] = {}
    _model_load_counts: Dict[Tuple[str, Optional[str], str, Optional[int], bool], int] = {}
    _model_load_times: Dict[Tuple[str, Optional[str], str, Optional[int], bool], float] = {}
    _model_lock = RLock()

    def __init__(self, config: Optional[EmbeddingModelConfig] = None, model: Any = None):
        self.config = config or EmbeddingModelConfig.from_env()
        self.model = model
        self.logger = get_logger(self.__class__.__name__)
        self._cache: Dict[str, List[float]] = {}
        self._dimension: Optional[int] = None
        self.embedding_model_cache_hit = model is not None
        self.embedding_model_load_count = 0
        self.embedding_model_load_time_seconds = 0.0
        self._load_cache()

    @classmethod
    def reset_model_cache(cls) -> None:
        with cls._model_lock:
            cls._model_registry.clear()
            cls._model_load_counts.clear()
            cls._model_load_times.clear()

    @classmethod
    def model_cache_size(cls) -> int:
        return len(cls._model_registry)

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

        model_source = self._resolve_model_source()
        init_kwargs = {
            "device": self.config.device,
            "cache_folder": str(self.config.cache_dir),
        }
        if "trust_remote_code" in inspect.signature(SentenceTransformer.__init__).parameters:
            init_kwargs["trust_remote_code"] = self.config.trust_remote_code

        cache_key = self._model_registry_key()
        with self._model_lock:
            cached_model = self._model_registry.get(cache_key)
            if cached_model is not None:
                self.model = cached_model
                self.embedding_model_cache_hit = True
                self.embedding_model_load_count = self._model_load_counts.get(cache_key, 0)
                self.embedding_model_load_time_seconds = self._model_load_times.get(cache_key, 0.0)
                if self._dimension is None and hasattr(self.model, "get_sentence_embedding_dimension"):
                    self._dimension = int(self.model.get_sentence_embedding_dimension())
                return

            try:
                started = perf_counter()
                self.logger.info("Loading embedding model: %s", model_source)
                loaded_model = SentenceTransformer(model_source, **init_kwargs)
                if self.config.max_seq_length:
                    loaded_model.max_seq_length = self.config.max_seq_length
                load_time = perf_counter() - started
                self._model_registry[cache_key] = loaded_model
                self._model_load_counts[cache_key] = self._model_load_counts.get(cache_key, 0) + 1
                self._model_load_times[cache_key] = load_time
                self.model = loaded_model
                self.embedding_model_cache_hit = False
                self.embedding_model_load_count = self._model_load_counts[cache_key]
                self.embedding_model_load_time_seconds = load_time
                if hasattr(self.model, "get_sentence_embedding_dimension"):
                    self._dimension = int(self.model.get_sentence_embedding_dimension())
            except Exception as exc:
                if self._looks_like_local_model_path(model_source):
                    self.logger.warning("Local embedding model is invalid: %s", model_source)
                    self._quarantine_invalid_cache_dir(Path(model_source), str(exc))
                    fallback_source = self._download_model_snapshot(force_download=True)
                    try:
                        started = perf_counter()
                        self.logger.info("Retrying embedding model load from %s", fallback_source)
                        loaded_model = SentenceTransformer(fallback_source, **init_kwargs)
                        if self.config.max_seq_length:
                            loaded_model.max_seq_length = self.config.max_seq_length
                        load_time = perf_counter() - started
                        self._model_registry[cache_key] = loaded_model
                        self._model_load_counts[cache_key] = self._model_load_counts.get(cache_key, 0) + 1
                        self._model_load_times[cache_key] = load_time
                        self.model = loaded_model
                        self.embedding_model_cache_hit = False
                        self.embedding_model_load_count = self._model_load_counts[cache_key]
                        self.embedding_model_load_time_seconds = load_time
                        if hasattr(self.model, "get_sentence_embedding_dimension"):
                            self._dimension = int(self.model.get_sentence_embedding_dimension())
                        return
                    except Exception as retry_exc:
                        raise EmbeddingError(
                            f"Failed to recover embedding model '{self.config.model_name}': {retry_exc}"
                        ) from retry_exc
                raise EmbeddingError(f"Failed to load embedding model '{self.config.model_name}': {exc}") from exc

    def _model_registry_key(self) -> Tuple[str, Optional[str], str, Optional[int], bool]:
        return (
            self.config.model_name,
            self.config.device,
            str(self.config.cache_dir.resolve()),
            self.config.max_seq_length,
            self.config.trust_remote_code,
        )

    def _cache_key(self, text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()

    def inspect_model_cache(self) -> List[ModelCacheStatus]:
        """Inspect known cache directories for the active model."""

        statuses: List[ModelCacheStatus] = []
        for candidate in self._candidate_model_dirs():
            if not candidate.exists():
                continue
            missing_files = self._missing_model_files(candidate)
            statuses.append(
                ModelCacheStatus(
                    path=candidate,
                    valid=not missing_files,
                    missing_files=missing_files,
                    source="local_cache",
                )
            )
        return statuses

    def _resolve_model_source(self) -> str:
        """Prefer a valid local model snapshot and recover from invalid cache copies."""

        for status in self.inspect_model_cache():
            if status.valid:
                return str(status.path)
            self.logger.warning(
                "Ignoring invalid cached embedding model at %s; missing files: %s",
                status.path,
                ", ".join(status.missing_files),
            )
            self._quarantine_invalid_cache_dir(status.path, "missing required model files")

        return self._download_model_snapshot(force_download=False)

    def _candidate_model_dirs(self) -> List[Path]:
        safe_name = self.config.model_name.replace("/", "_")
        return [
            self.config.cache_dir / safe_name,
            self.config.cache_dir / f"sentence-transformers_{safe_name}",
        ]

    def _missing_model_files(self, cached_model_dir: Path) -> List[str]:
        if not cached_model_dir.exists() or not cached_model_dir.is_dir():
            return list(self.REQUIRED_MODEL_FILES) + ["tokenizer asset"]

        missing = [
            file_name for file_name in self.REQUIRED_MODEL_FILES if not (cached_model_dir / file_name).exists()
        ]
        has_tokenizer = any((cached_model_dir / file_name).exists() for file_name in self.TOKENIZER_FILES)
        if not has_tokenizer:
            missing.append("tokenizer asset")
        return missing

    def _download_model_snapshot(self, force_download: bool) -> str:
        """Download the model snapshot into the writable cache if needed."""

        target_dir = self._candidate_model_dirs()[-1]
        if not force_download and not self._missing_model_files(target_dir):
            self.logger.info("Using cached embedding model from %s", target_dir)
            return str(target_dir)

        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            if force_download:
                raise EmbeddingError("huggingface_hub is required to recover the embedding model cache") from exc
            self.logger.info("huggingface_hub is unavailable; falling back to model name resolution")
            return self.config.model_name

        repo_id = self._resolve_repo_id()
        self.logger.info("Downloading embedding model %s to %s", repo_id, target_dir)
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                resume_download=not force_download,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to download embedding model '{repo_id}'. Check network access and cache permissions: {exc}"
            ) from exc

        missing_files = self._missing_model_files(target_dir)
        if missing_files:
            raise EmbeddingError(
                f"Downloaded model cache at '{target_dir}' is incomplete. Missing files: {', '.join(missing_files)}"
            )

        self.logger.info("Embedding model ready at %s", target_dir)
        return str(target_dir)

    def _resolve_repo_id(self) -> str:
        if "/" in self.config.model_name:
            return self.config.model_name
        return f"sentence-transformers/{self.config.model_name}"

    def _looks_like_local_model_path(self, model_source: str) -> bool:
        try:
            return Path(model_source).exists()
        except OSError:
            return False

    def _quarantine_invalid_cache_dir(self, path: Path, reason: str) -> Optional[Path]:
        if not path.exists():
            return None

        quarantine_path = path.with_name(f"{path.name}.invalid.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        try:
            shutil.move(str(path), str(quarantine_path))
            self.logger.warning("Moved invalid embedding cache from %s to %s (%s)", path, quarantine_path, reason)
            return quarantine_path
        except Exception as exc:
            self.logger.warning("Unable to quarantine invalid embedding cache %s: %s", path, exc)
            return None

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
