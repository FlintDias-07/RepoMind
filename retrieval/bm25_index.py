"""Repository-scoped BM25 index for lexical retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from parser.chunk_model import CodeChunk
from utils import get_logger


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_@.]+")
CAMEL_SPLIT_PATTERN = re.compile(r"(?<!^)(?=[A-Z])")


@dataclass(frozen=True)
class BM25Document:
    """Lexical retrieval document derived from a CodeChunk."""

    chunk_id: str
    repository_name: str
    file_name: str
    file_path: str
    language: str
    class_name: str
    function_name: str
    content: str
    metadata: Dict[str, Any]
    tokens: List[str]


class BM25Index:
    """In-memory BM25 index for a single repository."""

    def __init__(self, repository_name: str, model: Any = None):
        self.repository_name = repository_name
        self.logger = get_logger(self.__class__.__name__)
        self.model = model
        self.documents: List[BM25Document] = []

    def build(self, chunks: Sequence[CodeChunk]) -> None:
        """Rebuild the index from repository chunks."""

        documents: List[BM25Document] = []
        for chunk in chunks:
            if not chunk or not chunk.content.strip():
                continue
            metadata = dict(chunk.metadata)
            metadata.setdefault("chunk_id", chunk.chunk_id)
            metadata.setdefault("file_name", chunk.file_name)
            metadata.setdefault("file_path", chunk.file_path)
            metadata.setdefault("language", chunk.language)
            metadata.setdefault("repository_name", chunk.repository_name)
            metadata.setdefault("class_name", chunk.class_name or "")
            metadata.setdefault("function_name", chunk.function_name or "")
            metadata.setdefault("start_line", int(chunk.start_line or 0))
            metadata.setdefault("end_line", int(chunk.end_line or 0))
            metadata.setdefault("chunk_type", chunk.chunk_type)
            metadata.setdefault("file_category", chunk.metadata.get("file_category", "source_code"))

            lexical_text = " ".join(
                [
                    chunk.file_name,
                    chunk.file_path,
                    chunk.class_name or "",
                    chunk.function_name or "",
                    chunk.content,
                ]
            )
            documents.append(
                BM25Document(
                    chunk_id=chunk.chunk_id,
                    repository_name=chunk.repository_name,
                    file_name=chunk.file_name,
                    file_path=chunk.file_path,
                    language=chunk.language,
                    class_name=chunk.class_name or "",
                    function_name=chunk.function_name or "",
                    content=chunk.content,
                    metadata=metadata,
                    tokens=self.tokenize(lexical_text),
                )
            )

        self.documents = documents
        self.model = self._build_model([document.tokens for document in documents])
        self.logger.info("Built BM25 index for repository '%s' with %s documents", self.repository_name, len(documents))

    def search(self, query: str, top_k: int = 20) -> List[tuple[BM25Document, float]]:
        """Run lexical retrieval against the repository index."""

        if not self.model or not self.documents:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = list(self.model.get_scores(query_tokens))
        ranked = sorted(
            zip(self.documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [(document, float(score)) for document, score in ranked[:top_k] if float(score) > 0.0]

    def _build_model(self, corpus: Sequence[Sequence[str]]) -> Any:
        try:
            from rank_bm25 import BM25Okapi
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("rank_bm25 is required for lexical retrieval") from exc
        return BM25Okapi(list(corpus))

    @staticmethod
    def tokenize(text: str) -> List[str]:
        tokens: List[str] = []
        for raw in TOKEN_PATTERN.findall(text or ""):
            lowered = raw.lower()
            tokens.append(lowered)
            split_parts = [part.lower() for part in CAMEL_SPLIT_PATTERN.split(raw) if part]
            tokens.extend(split_parts)
            collapsed = "".join(split_parts)
            if collapsed:
                tokens.append(collapsed)
            if "." in lowered:
                tokens.extend([part for part in lowered.split(".") if part])
            if "/" in lowered or "\\" in lowered:
                normalized = lowered.replace("\\", "/")
                tokens.extend([part for part in normalized.split("/") if part])
        return [token for token in tokens if token]
