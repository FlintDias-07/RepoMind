"""Build LLM-ready context objects from retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from utils import get_logger


@dataclass(frozen=True)
class RepositorySnapshot:
    """Repository metadata carried into the AI layer."""

    repository_name: str
    repo_url: str = ""
    local_path: str = ""
    indexed_at: str = ""
    total_files: int = 0
    total_chunks: int = 0
    embeddings_stored: int = 0
    extra_metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextChunk:
    """Normalized chunk passed to the LLM context window."""

    chunk_id: str
    repository_name: str
    file_name: str
    file_path: str
    class_name: str
    function_name: str
    language: str
    start_line: int
    end_line: int
    content: str
    similarity_score: float = 0.0
    source_rank: int = 0
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def line_span(self) -> str:
        if self.start_line and self.end_line:
            return f"{self.start_line}-{self.end_line}"
        if self.start_line:
            return str(self.start_line)
        return ""

    @property
    def symbol_name(self) -> str:
        if self.class_name and self.function_name:
            return f"{self.class_name}.{self.function_name}"
        return self.function_name or self.class_name or ""


@dataclass(frozen=True)
class AIContext:
    """Structured context object consumed by prompt builders and formatters."""

    question: str
    repository: RepositorySnapshot
    chunks: Sequence[ContextChunk]
    estimated_tokens: int
    truncated: bool
    warnings: Sequence[str] = field(default_factory=tuple)
    source_files: Sequence[str] = field(default_factory=tuple)
    source_classes: Sequence[str] = field(default_factory=tuple)
    source_functions: Sequence[str] = field(default_factory=tuple)


class AIContextBuilder:
    """Prepare ranked retrieval chunks for LLM consumption."""

    def __init__(
        self,
        token_limit: int = 2800,
        merge_line_gap: int = 3,
        min_similarity_score: float = 0.0,
    ):
        self.token_limit = token_limit
        self.merge_line_gap = merge_line_gap
        self.min_similarity_score = min_similarity_score
        self.logger = get_logger(self.__class__.__name__)

    def build(
        self,
        question: str,
        retrieved_chunks: Sequence[object],
        repository_metadata: Optional[Dict[str, object]] = None,
    ) -> AIContext:
        if not question or not question.strip():
            raise ValueError("question cannot be empty")

        snapshot = self._repository_snapshot(retrieved_chunks, repository_metadata or {})
        normalized_chunks = self._normalize_chunks(retrieved_chunks)
        deduplicated_chunks = self._deduplicate(normalized_chunks)
        merged_chunks = self._merge_overlapping(deduplicated_chunks)
        limited_chunks, estimated_tokens, truncated = self._apply_token_limit(merged_chunks)

        warnings: List[str] = []
        if not limited_chunks:
            warnings.append("No retrieved context was available for the question.")
        if truncated:
            warnings.append("Context was truncated to respect token limits.")

        return AIContext(
            question=question.strip(),
            repository=snapshot,
            chunks=tuple(limited_chunks),
            estimated_tokens=estimated_tokens,
            truncated=truncated,
            warnings=tuple(warnings),
            source_files=tuple(sorted({chunk.file_path for chunk in limited_chunks if chunk.file_path})),
            source_classes=tuple(sorted({chunk.class_name for chunk in limited_chunks if chunk.class_name})),
            source_functions=tuple(sorted({chunk.function_name for chunk in limited_chunks if chunk.function_name})),
        )

    def _repository_snapshot(
        self,
        retrieved_chunks: Sequence[object],
        metadata: Dict[str, object],
    ) -> RepositorySnapshot:
        repository_name = str(metadata.get("repo_name") or metadata.get("repository_name") or "").strip()
        if not repository_name and retrieved_chunks:
            repository_name = str(getattr(retrieved_chunks[0], "repository_name", "") or "").strip()
        index_stats = metadata.get("index_stats") if isinstance(metadata.get("index_stats"), dict) else {}
        return RepositorySnapshot(
            repository_name=repository_name or "unknown-repository",
            repo_url=str(metadata.get("repo_url", "") or ""),
            local_path=str(metadata.get("local_path", "") or ""),
            indexed_at=str(metadata.get("indexed_at", "") or ""),
            total_files=int(metadata.get("stats", {}).get("total_files", 0) if isinstance(metadata.get("stats"), dict) else 0),
            total_chunks=int(index_stats.get("chunks_generated", 0) if isinstance(index_stats, dict) else 0),
            embeddings_stored=int(index_stats.get("embeddings_stored", 0) if isinstance(index_stats, dict) else 0),
            extra_metadata={
                key: value
                for key, value in metadata.items()
                if key not in {"repo_name", "repository_name", "repo_url", "local_path", "indexed_at", "stats", "index_stats"}
            },
        )

    def _normalize_chunks(self, retrieved_chunks: Sequence[object]) -> List[ContextChunk]:
        normalized: List[ContextChunk] = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            score = float(getattr(chunk, "similarity_score", 0.0) or 0.0)
            if self.min_similarity_score > 0 and score < self.min_similarity_score:
                continue
            content = str(getattr(chunk, "document", "") or "").strip()
            if not content:
                continue
            metadata = dict(getattr(chunk, "metadata", {}) or {})
            normalized.append(
                ContextChunk(
                    chunk_id=str(getattr(chunk, "chunk_id", "") or f"chunk-{index}"),
                    repository_name=str(getattr(chunk, "repository_name", metadata.get("repository_name", "")) or ""),
                    file_name=str(getattr(chunk, "file_name", metadata.get("file_name", "")) or ""),
                    file_path=str(getattr(chunk, "file_path", metadata.get("file_path", "")) or ""),
                    class_name=str(getattr(chunk, "class_name", metadata.get("class_name", "")) or ""),
                    function_name=str(getattr(chunk, "function_name", metadata.get("function_name", "")) or ""),
                    language=str(getattr(chunk, "language", metadata.get("language", "")) or ""),
                    start_line=int(getattr(chunk, "start_line", metadata.get("start_line", 0)) or 0),
                    end_line=int(getattr(chunk, "end_line", metadata.get("end_line", 0)) or 0),
                    content=content,
                    similarity_score=score,
                    source_rank=index,
                    metadata=metadata,
                )
            )
        return normalized

    def _deduplicate(self, chunks: Sequence[ContextChunk]) -> List[ContextChunk]:
        unique_by_chunk_id: Dict[str, ContextChunk] = {}
        unique_by_fingerprint: Dict[tuple[str, int, int, str], ContextChunk] = {}

        for chunk in chunks:
            if chunk.chunk_id and chunk.chunk_id not in unique_by_chunk_id:
                unique_by_chunk_id[chunk.chunk_id] = chunk
                continue

            fingerprint = (
                chunk.file_path,
                chunk.start_line,
                chunk.end_line,
                chunk.content.strip(),
            )
            if fingerprint not in unique_by_fingerprint:
                unique_by_fingerprint[fingerprint] = chunk

        ordered = list(unique_by_chunk_id.values())
        ordered.extend(
            chunk
            for fingerprint, chunk in unique_by_fingerprint.items()
            if chunk not in ordered
        )
        ordered.sort(key=lambda item: (item.source_rank, -item.similarity_score))
        return ordered

    def _merge_overlapping(self, chunks: Sequence[ContextChunk]) -> List[ContextChunk]:
        if not chunks:
            return []

        sorted_chunks = sorted(
            chunks,
            key=lambda item: (
                item.file_path,
                item.class_name,
                item.function_name,
                item.start_line or 0,
                item.end_line or 0,
            ),
        )

        merged: List[ContextChunk] = []
        current = sorted_chunks[0]
        for candidate in sorted_chunks[1:]:
            if self._can_merge(current, candidate):
                current = self._merge_pair(current, candidate)
            else:
                merged.append(current)
                current = candidate
        merged.append(current)

        merged.sort(key=lambda item: item.source_rank)
        return merged

    def _can_merge(self, left: ContextChunk, right: ContextChunk) -> bool:
        if left.file_path != right.file_path:
            return False
        if left.class_name != right.class_name:
            return False
        if left.function_name != right.function_name:
            return False
        if not left.end_line or not right.start_line:
            return False
        return right.start_line <= left.end_line + self.merge_line_gap

    def _merge_pair(self, left: ContextChunk, right: ContextChunk) -> ContextChunk:
        merged_content = self._merge_content(left.content, right.content)
        merged_metadata = dict(left.metadata)
        merged_metadata.update(right.metadata)
        return ContextChunk(
            chunk_id=left.chunk_id or right.chunk_id,
            repository_name=left.repository_name or right.repository_name,
            file_name=left.file_name or right.file_name,
            file_path=left.file_path or right.file_path,
            class_name=left.class_name or right.class_name,
            function_name=left.function_name or right.function_name,
            language=left.language or right.language,
            start_line=min(line for line in [left.start_line, right.start_line] if line > 0) if (left.start_line or right.start_line) else 0,
            end_line=max(left.end_line, right.end_line),
            content=merged_content,
            similarity_score=max(left.similarity_score, right.similarity_score),
            source_rank=min(left.source_rank, right.source_rank),
            metadata=merged_metadata,
        )

    def _merge_content(self, left: str, right: str) -> str:
        left_clean = left.rstrip()
        right_clean = right.lstrip()
        if not left_clean:
            return right_clean
        if not right_clean:
            return left_clean
        if right_clean in left_clean:
            return left_clean
        if left_clean in right_clean:
            return right_clean

        left_lines = left_clean.splitlines()
        right_lines = right_clean.splitlines()
        overlap = 0
        max_overlap = min(len(left_lines), len(right_lines))
        for size in range(max_overlap, 0, -1):
            if left_lines[-size:] == right_lines[:size]:
                overlap = size
                break

        if overlap:
            return "\n".join(left_lines + right_lines[overlap:])
        return f"{left_clean}\n{right_clean}"

    def _apply_token_limit(self, chunks: Sequence[ContextChunk]) -> tuple[List[ContextChunk], int, bool]:
        selected: List[ContextChunk] = []
        current_tokens = 0
        truncated = False

        for chunk in chunks:
            chunk_tokens = self._estimate_tokens(chunk.content) + self._estimate_tokens(chunk.file_path) + 24
            if selected and current_tokens + chunk_tokens > self.token_limit:
                truncated = True
                break
            if not selected and chunk_tokens > self.token_limit:
                selected.append(self._truncate_single_chunk(chunk, self.token_limit))
                current_tokens = self._estimate_tokens(selected[0].content) + self._estimate_tokens(selected[0].file_path) + 24
                truncated = True
                break
            selected.append(chunk)
            current_tokens += chunk_tokens

        return selected, current_tokens, truncated

    def _truncate_single_chunk(self, chunk: ContextChunk, token_limit: int) -> ContextChunk:
        max_chars = max((token_limit - 24) * 4, 120)
        truncated_content = chunk.content[:max_chars].rstrip()
        if truncated_content != chunk.content:
            truncated_content = f"{truncated_content}\n... [truncated]"
        return ContextChunk(
            chunk_id=chunk.chunk_id,
            repository_name=chunk.repository_name,
            file_name=chunk.file_name,
            file_path=chunk.file_path,
            class_name=chunk.class_name,
            function_name=chunk.function_name,
            language=chunk.language,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            content=truncated_content,
            similarity_score=chunk.similarity_score,
            source_rank=chunk.source_rank,
            metadata=chunk.metadata,
        )

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)
