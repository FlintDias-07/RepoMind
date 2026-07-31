"""Structured response formatting for the AI intelligence layer."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Sequence

from .citation_manager import Citation
from .context_builder import AIContext


@dataclass(frozen=True)
class FormattedResponse:
    """Final structured answer returned by the AI service."""

    summary: str
    detailed_explanation: str
    repository_insights: Sequence[str]
    referenced_files: Sequence[str]
    referenced_classes: Sequence[str]
    referenced_functions: Sequence[str]
    confidence_level: str
    warnings: Sequence[str]
    citations: Sequence[Citation]
    rendered_markdown: str
    raw_answer: str


class ResponseFormatter:
    """Render provider output into a stable product-facing structure."""

    INSUFFICIENT_EVIDENCE_MESSAGE = "I could not find enough evidence in the indexed repository."
    INSUFFICIENT_EVIDENCE_MARKERS = (
        INSUFFICIENT_EVIDENCE_MESSAGE.lower(),
        "does not contain explicit information",
        "does not contain information",
        "doesn't contain explicit information",
        "doesn't contain information",
        "no explicit information",
        "no evidence",
        "not enough evidence",
        "insufficient evidence",
    )
    SECTION_ALIASES = {
        "summary": "summary",
        "detailed explanation": "details",
        "details": "details",
        "implementation flow": "details",
        "key components": "details",
        "repository insights": "insights",
        "insights": "insights",
        "referenced evidence": "evidence",
        "evidence": "evidence",
        "sources": "evidence",
        "source citations": "evidence",
        "confidence": "confidence",
        "warnings": "warnings",
    }
    SECTION_HEADING_PATTERN = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:\d+[\.)]\s*)?(summary|detailed explanation|details|implementation flow|key components|repository insights|insights|referenced evidence|evidence|sources|source citations|confidence|warnings)\s*:?\s*(.*)$",
        re.IGNORECASE,
    )

    def format(
        self,
        raw_answer: str,
        context: AIContext,
        citations: Sequence[Citation],
    ) -> FormattedResponse:
        cleaned = (raw_answer or "").strip() or self.INSUFFICIENT_EVIDENCE_MESSAGE
        warnings = list(context.warnings)
        confidence = self._confidence_level(context, cleaned)
        if self._is_insufficient_answer(cleaned) and self.INSUFFICIENT_EVIDENCE_MESSAGE not in warnings:
            warnings.append(self.INSUFFICIENT_EVIDENCE_MESSAGE)

        sections = self._split_sections(cleaned)
        repository_insights = self._derive_insights(context, citations)
        rendered_markdown = self._render_markdown(
            summary=sections["summary"],
            detailed_explanation=sections["details"],
            repository_insights=repository_insights,
            citations=citations,
            confidence_level=confidence,
            warnings=warnings,
        )
        return FormattedResponse(
            summary=sections["summary"],
            detailed_explanation=sections["details"],
            repository_insights=tuple(repository_insights),
            referenced_files=tuple(sorted({citation.file_path for citation in citations if citation.file_path})),
            referenced_classes=tuple(sorted({citation.class_name for citation in citations if citation.class_name})),
            referenced_functions=tuple(sorted({citation.function_name for citation in citations if citation.function_name})),
            confidence_level=confidence,
            warnings=tuple(warnings),
            citations=tuple(citations),
            rendered_markdown=rendered_markdown,
            raw_answer=cleaned,
        )

    def _confidence_level(self, context: AIContext, raw_answer: str) -> str:
        if self._is_insufficient_answer(raw_answer) or not context.chunks:
            return "low"
        if context.truncated or len(context.chunks) <= 1:
            return "medium"
        return "high"

    def _is_insufficient_answer(self, raw_answer: str) -> bool:
        lowered = (raw_answer or "").strip().lower()
        return any(marker in lowered for marker in self.INSUFFICIENT_EVIDENCE_MARKERS)

    def _split_sections(self, raw_answer: str) -> dict[str, str]:
        lines = [line.strip() for line in raw_answer.splitlines() if line.strip()]
        if not lines:
            return {"summary": self.INSUFFICIENT_EVIDENCE_MESSAGE, "details": ""}
        if len(lines) == 1:
            summary = self._strip_heading_prefix(lines[0])
            return {"summary": summary or lines[0], "details": raw_answer.strip()}

        parsed = self._parse_labeled_sections(lines)
        if parsed.get("summary"):
            summary = self._compact_summary(parsed["summary"])
            details = self._build_details(raw_answer, parsed)
            return {
                "summary": summary or self._fallback_summary(raw_answer),
                "details": details or raw_answer.strip(),
            }

        summary = self._fallback_summary(raw_answer)
        return {
            "summary": summary or self.INSUFFICIENT_EVIDENCE_MESSAGE,
            "details": raw_answer.strip(),
        }

    def _parse_labeled_sections(self, lines: Sequence[str]) -> dict[str, str]:
        sections: dict[str, list[str]] = {}
        current_key = ""

        for line in lines:
            match = self.SECTION_HEADING_PATTERN.match(line)
            if match:
                label = match.group(1).lower()
                remainder = match.group(2).strip()
                current_key = self.SECTION_ALIASES.get(label, "")
                if current_key:
                    sections.setdefault(current_key, [])
                    if remainder:
                        sections[current_key].append(remainder)
                continue

            if current_key:
                sections.setdefault(current_key, []).append(line)

        return {key: "\n".join(value).strip() for key, value in sections.items() if "\n".join(value).strip()}

    def _build_details(self, raw_answer: str, parsed_sections: dict[str, str]) -> str:
        detail_parts: List[str] = []
        if parsed_sections.get("details"):
            detail_parts.append(parsed_sections["details"])
        if parsed_sections.get("insights"):
            detail_parts.append("## Repository Insights\n" + parsed_sections["insights"])
        if parsed_sections.get("evidence"):
            detail_parts.append("## Sources\n" + parsed_sections["evidence"])

        if detail_parts:
            return "\n\n".join(part.strip() for part in detail_parts if part.strip())

        cleaned = raw_answer.strip()
        summary = parsed_sections.get("summary", "").strip()
        if summary and cleaned.startswith(summary):
            return cleaned
        return cleaned

    def _compact_summary(self, text: str, max_sentences: int = 5) -> str:
        cleaned = self._strip_heading_prefix(text)
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
        source = paragraphs[0] if paragraphs else cleaned
        sentences = re.split(r"(?<=[.!?])\s+", source)
        summary = " ".join(sentence.strip() for sentence in sentences[:max_sentences] if sentence.strip())
        return summary or source.strip()

    def _fallback_summary(self, raw_answer: str) -> str:
        lines = [self._strip_heading_prefix(line.strip()) for line in raw_answer.splitlines() if line.strip()]
        meaningful = [
            line
            for line in lines
            if line
            and not self._is_section_heading_only(line)
            and not line.startswith("#")
            and not line.startswith("- ")
            and not re.match(r"^\d+[\.)]\s*$", line)
        ]
        if not meaningful:
            return ""
        return self._compact_summary(meaningful[0])

    def _strip_heading_prefix(self, text: str) -> str:
        match = self.SECTION_HEADING_PATTERN.match(text.strip())
        if match and match.group(2).strip():
            return match.group(2).strip()
        return text.strip()

    def _is_section_heading_only(self, text: str) -> bool:
        match = self.SECTION_HEADING_PATTERN.match(text.strip())
        return bool(match and not match.group(2).strip())

    def _derive_insights(self, context: AIContext, citations: Sequence[Citation]) -> List[str]:
        insights: List[str] = []
        if context.repository.total_files:
            insights.append(f"Repository index covers {context.repository.total_files} source files.")
        if context.repository.total_chunks:
            insights.append(f"Indexed chunk count: {context.repository.total_chunks}.")
        if citations:
            insights.append(f"Evidence spans {len({citation.file_path for citation in citations})} files.")
        if context.truncated:
            insights.append("Only a subset of the retrieved evidence fit within the context window.")
        return insights

    def _render_markdown(
        self,
        summary: str,
        detailed_explanation: str,
        repository_insights: Sequence[str],
        citations: Sequence[Citation],
        confidence_level: str,
        warnings: Sequence[str],
    ) -> str:
        lines = [
            "## Summary",
            summary,
            "",
            "## Detailed Explanation",
            detailed_explanation or self.INSUFFICIENT_EVIDENCE_MESSAGE,
            "",
            "## Repository Insights",
        ]
        lines.extend(f"- {item}" for item in (repository_insights or ["No repository-level insights available."]))
        lines.extend(["", "## Referenced Evidence"])
        if citations:
            for citation in citations:
                symbol = citation.function_name or citation.class_name or "file"
                lines.append(
                    f"- `{citation.file_path}` | {symbol} | {citation.line_label}"
                )
        else:
            lines.append("- No source citations available.")
        lines.extend(["", "## Confidence", confidence_level])
        if warnings:
            lines.extend(["", "## Warnings"])
            lines.extend(f"- {warning}" for warning in warnings)
        return "\n".join(lines)
