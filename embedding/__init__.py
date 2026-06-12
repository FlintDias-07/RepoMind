"""Embedding package for Phase 3."""

from .model_config import EmbeddingModelConfig
from .embedding_service import EmbeddingService, EmbeddingError
from .embedding_manager import EmbeddingManager, EmbeddedChunk, EmbeddingBatchResult

__all__ = [
    "EmbeddingModelConfig",
    "EmbeddingService",
    "EmbeddingError",
    "EmbeddingManager",
    "EmbeddedChunk",
    "EmbeddingBatchResult",
]
