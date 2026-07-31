"""Citation extraction from repository context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .context_builder import AIContext, ContextChunk


@dataclass(frozen=True)
class Citation:
    """Evidence reference shown in AI responses."""

    repository_name: str
    file_path: str
    file_name: str
    class_name: str
    function_name: str
    start_line: int
    end_line: int

    @property
    def line_label(self) -> str:
        if self.start_line and self.end_line:
            return f"Lines {self.start_line}-{self.end_line}"
        if self.start_line:
            return f"Line {self.start_line}"
        return "Lines unknown"


class CitationManager:
    """Deduplicate and normalize citations from LLM context."""

    def from_context(self, context: AIContext) -> List[Citation]:
        citations: List[Citation] = []
        seen = set()
        for chunk in context.chunks:
            citation = Citation(
                repository_name=chunk.repository_name or context.repository.repository_name,
                file_path=chunk.file_path,
                file_name=chunk.file_name,
                class_name=chunk.class_name,
                function_name=chunk.function_name,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )
            fingerprint = (
                citation.file_path,
                citation.class_name,
                citation.function_name,
                citation.start_line,
                citation.end_line,
            )
            if fingerprint not in seen:
                seen.add(fingerprint)
                citations.append(citation)
        return citations

    def summarize_files(self, citations: Sequence[Citation]) -> List[str]:
        return sorted({citation.file_path for citation in citations if citation.file_path})

    def summarize_classes(self, citations: Sequence[Citation]) -> List[str]:
        return sorted({citation.class_name for citation in citations if citation.class_name})

    def summarize_functions(self, citations: Sequence[Citation]) -> List[str]:
        return sorted({citation.function_name for citation in citations if citation.function_name})
