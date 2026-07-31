"""Export helpers for Repository Intelligence reports."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from typing import Any, Mapping, Sequence

from ai.ai_service import IntelligenceReport


class ReportExporter:
    """Serialize intelligence reports into download-ready formats."""

    def to_markdown(self, report: IntelligenceReport) -> str:
        lines = [
            f"# {report.title}",
            "",
            f"**Repository:** {report.repository_name}",
            f"**Confidence:** {report.confidence_level}",
            "",
            "## Summary",
            report.summary,
            "",
            "## Detailed Explanation",
            report.detailed_explanation,
            "",
            "## Repository Insights",
        ]
        lines.extend(f"- {item}" for item in (report.repository_insights or ["No repository insights available."]))
        lines.extend(["", "## Referenced Files"])
        lines.extend(f"- `{item}`" for item in (report.referenced_files or ["No referenced files."]))
        lines.extend(["", "## Referenced Classes"])
        lines.extend(f"- `{item}`" for item in (report.referenced_classes or ["No referenced classes."]))
        lines.extend(["", "## Referenced Functions"])
        lines.extend(f"- `{item}`" for item in (report.referenced_functions or ["No referenced functions."]))
        lines.extend(["", "## Citations"])
        if report.citations:
            for citation in report.citations:
                lines.append(
                    f"- `{citation.file_path}` | {citation.function_name or citation.class_name or 'file'} | "
                    f"{citation.start_line}-{citation.end_line if citation.end_line else citation.start_line}"
                )
        else:
            lines.append("- No citations available.")
        if report.warnings:
            lines.extend(["", "## Warnings"])
            lines.extend(f"- {warning}" for warning in report.warnings)
        return "\n".join(lines)

    def to_json(self, report: IntelligenceReport) -> str:
        return json.dumps(self._serialize(report), indent=2)

    def to_pdf_bytes(self, report: IntelligenceReport) -> bytes:
        text = self.to_markdown(report)
        return self._simple_pdf(text)

    def _serialize(self, value: Any) -> Any:
        if is_dataclass(value):
            return {key: self._serialize(item) for key, item in asdict(value).items()}
        if isinstance(value, dict):
            return {str(key): self._serialize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize(item) for item in value]
        return value

    def _simple_pdf(self, text: str) -> bytes:
        lines = [line[:100] for line in text.splitlines() if line.strip()]
        if not lines:
            lines = ["Repository Intelligence Report"]

        content_lines = ["BT", "/F1 10 Tf", "50 780 Td"]
        first = True
        for line in lines[:60]:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            if first:
                content_lines.append(f"({escaped}) Tj")
                first = False
            else:
                content_lines.append(f"0 -14 Td ({escaped}) Tj")
        content_lines.append("ET")
        stream = "\n".join(content_lines).encode("latin-1", errors="replace")

        objects = []
        objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
        objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
        objects.append(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n")
        objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
        objects.append(
            f"5 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1")
            + stream
            + b"\nendstream endobj\n"
        )

        pdf = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for obj in objects:
            offsets.append(len(pdf))
            pdf.extend(obj)
        xref_offset = len(pdf)
        pdf.extend(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
        pdf.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
        pdf.extend(
            f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode("latin-1")
        )
        return bytes(pdf)
