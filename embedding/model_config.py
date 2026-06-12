"""Configuration for embedding models and caching."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
import re

from config import EMBEDDING_MODEL, VECTOR_DB_DIR


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value.strip("._-") or "embedding_model"


@dataclass
class EmbeddingModelConfig:
    """Runtime configuration for embedding generation."""

    model_name: str = EMBEDDING_MODEL
    device: Optional[str] = None
    batch_size: int = 32
    normalize_embeddings: bool = True
    show_progress_bar: bool = False
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: VECTOR_DB_DIR / "embedding_cache")
    max_seq_length: Optional[int] = None
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "EmbeddingModelConfig":
        """Build configuration from environment variables."""

        return cls(
            model_name=os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL),
            device=os.getenv("EMBEDDING_DEVICE") or None,
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            normalize_embeddings=_env_bool("EMBEDDING_NORMALIZE", True),
            show_progress_bar=_env_bool("EMBEDDING_PROGRESS_BAR", False),
            cache_enabled=_env_bool("EMBEDDING_CACHE_ENABLED", True),
            cache_dir=Path(os.getenv("EMBEDDING_CACHE_DIR", str(VECTOR_DB_DIR / "embedding_cache"))),
            max_seq_length=(int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH")) if os.getenv("EMBEDDING_MAX_SEQ_LENGTH") else None),
            trust_remote_code=_env_bool("EMBEDDING_TRUST_REMOTE_CODE", False),
        )

    @property
    def cache_file(self) -> Path:
        """Return the cache file path for the active model."""

        return self.cache_dir / f"{_safe_name(self.model_name)}.json"
