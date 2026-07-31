"""Shared report rendering helpers for the Repository Intelligence page."""

from __future__ import annotations

from typing import Dict
from html import escape
import re

import streamlit as st

from ai.ai_service import IntelligenceReport


SUMMARY_HEADING_ONLY = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\d+[\.)]\s*)?summary\s*:?\s*$", re.IGNORECASE)


def _safe_summary(report: IntelligenceReport) -> str:
    summary = (report.summary or "").strip()
    if summary and not SUMMARY_HEADING_ONLY.match(summary):
        return summary

    for line in (report.raw_answer or report.detailed_explanation or "").splitlines():
        cleaned = line.strip()
        if not cleaned or SUMMARY_HEADING_ONLY.match(cleaned):
            continue
        if cleaned.startswith("#"):
            continue
        return cleaned
    return summary


def format_citation(citation: object) -> str:
    file_path = str(getattr(citation, "file_path", "") or "unknown source")
    start_line = int(getattr(citation, "start_line", 0) or 0)
    end_line = int(getattr(citation, "end_line", 0) or 0)
    if start_line and end_line:
        line_label = f"lines {start_line}-{end_line}"
    elif start_line:
        line_label = f"line {start_line}"
    else:
        line_label = "lines unknown"
    return f"{file_path} — {line_label}"


def report_to_sections(report: IntelligenceReport) -> Dict[str, object]:
    return {
        "summary": _safe_summary(report),
        "details": report.detailed_explanation,
        "insights": list(report.repository_insights),
        "files": list(report.referenced_files),
        "classes": list(report.referenced_classes),
        "functions": list(report.referenced_functions),
        "confidence": report.confidence_level,
        "warnings": list(report.warnings),
        "citations": list(report.citations),
        "diagnostics": dict(report.diagnostics or {}),
    }


def _render_evidence_group(title: str, items: object, empty_message: str) -> None:
    rendered_items = list(items or [])
    if rendered_items:
        items_html = "\n".join(
            f"<div class='evidence-item'><code>{escape(str(item))}</code></div>"
            for item in rendered_items
        )
    else:
        items_html = f"<div class='evidence-item'>{escape(empty_message)}</div>"
    st.markdown(
        f"""
        <div class="evidence-card">
            <h4>{escape(title)}</h4>
            {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_diagnostics(diagnostics: Dict[str, object]) -> None:
    if not diagnostics:
        return

    with st.expander("Retrieval Diagnostics", expanded=False):
        metric_cols = st.columns(4)
        metric_cols[0].metric("Mode", diagnostics.get("retrieval_mode", "fast"))
        metric_cols[1].metric("Strategy", diagnostics.get("strategy_name", "hybrid"))
        metric_cols[2].metric("Reranking", "Yes" if diagnostics.get("reranking_used") else "No")
        metric_cols[3].metric("Total Time", f"{float(diagnostics.get('total_response_time_seconds', 0.0) or 0.0):.4f}s")

        timing_cols = st.columns(3)
        timing_cols[0].metric(
            "Retrieval",
            f"{float(diagnostics.get('retrieval_time_seconds', 0.0) or 0.0):.4f}s",
        )
        timing_cols[1].metric(
            "Reranking",
            f"{float(diagnostics.get('reranking_time_seconds', 0.0) or 0.0):.4f}s",
        )
        timing_cols[2].metric(
            "LLM Generation",
            f"{float(diagnostics.get('llm_generation_time_seconds', 0.0) or 0.0):.4f}s",
        )


def render_report_sections(report: IntelligenceReport, key_prefix: str = "report") -> None:
    sections = report_to_sections(report)
    summary_tab, details_tab, evidence_tab = st.tabs(["Summary", "Details", "Evidence"])

    with summary_tab:
        st.markdown("### Summary")
        st.markdown(
            f"""
            <div class="report-summary">
                {escape(str(sections["summary"] or "No summary generated."))}
            </div>
            <div class="report-confidence">
                <strong>Confidence</strong>
                <span>{escape(str(sections["confidence"]))}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if sections["warnings"]:
            st.markdown("#### Warnings")
            for warning in sections["warnings"]:
                st.warning(warning)

    with details_tab:
        st.markdown("### Details")
        st.markdown(sections["details"] or "No detailed explanation generated.")
        if sections["insights"]:
            st.markdown("### Repository Insights")
            for item in sections["insights"]:
                st.write(f"- {item}")

    with evidence_tab:
        st.markdown("### Evidence")
        cols = st.columns(3)
        with cols[0]:
            _render_evidence_group("Files", sections["files"], "No referenced files.")
        with cols[1]:
            _render_evidence_group("Classes", sections["classes"], "No referenced classes.")
        with cols[2]:
            _render_evidence_group("Functions", sections["functions"], "No referenced functions.")

        with st.expander("Source Citations", expanded=False):
            if sections["citations"]:
                for citation in sections["citations"]:
                    st.write(f"- `{format_citation(citation)}`")
                    symbol = citation.function_name or citation.class_name
                    if symbol:
                        label = "Function" if citation.function_name else "Class"
                        st.caption(f"{label}: {symbol}")
            else:
                st.caption("No citations available.")

        _render_diagnostics(sections["diagnostics"])
