"""Module explorer for repository-scoped AI explanations."""

from __future__ import annotations

from typing import Dict, Sequence

import streamlit as st

from ai.ai_service import AIService, IntelligenceReport
from .components import DEFAULT_PROGRESS_MESSAGES, empty_state, run_with_progress_messages


def filter_module_options(options: Sequence[str], query: str) -> Sequence[str]:
    if not query.strip():
        return options
    lowered = query.strip().lower()
    return tuple(item for item in options if lowered in item.lower())


def render_module_explorer(
    ai_service: AIService,
    session_id: str,
    reports_state: Dict[str, IntelligenceReport],
    key_prefix: str = "module_explorer",
    retrieval_mode: str = "fast",
) -> None:
    options = ai_service.list_module_options()
    search_text = st.text_input("Search modules", key=f"{key_prefix}_search", placeholder="Filter files, folders, classes, or modules")
    tab_files, tab_folders, tab_classes, tab_modules = st.tabs(["Files", "Folders", "Classes", "Modules"])

    with tab_files:
        _render_scope_selector(
            label="File",
            scope_key="files",
            options=filter_module_options(options.get("files", ()), search_text),
            ai_service=ai_service,
            session_id=session_id,
            reports_state=reports_state,
            key_prefix=key_prefix,
            retrieval_mode=retrieval_mode,
        )
    with tab_folders:
        _render_scope_selector(
            label="Folder",
            scope_key="folders",
            options=filter_module_options(options.get("folders", ()), search_text),
            ai_service=ai_service,
            session_id=session_id,
            reports_state=reports_state,
            key_prefix=key_prefix,
            retrieval_mode=retrieval_mode,
        )
    with tab_classes:
        _render_scope_selector(
            label="Class",
            scope_key="classes",
            options=filter_module_options(options.get("classes", ()), search_text),
            ai_service=ai_service,
            session_id=session_id,
            reports_state=reports_state,
            key_prefix=key_prefix,
            retrieval_mode=retrieval_mode,
        )
    with tab_modules:
        _render_scope_selector(
            label="Module",
            scope_key="modules",
            options=filter_module_options(options.get("modules", ()), search_text),
            ai_service=ai_service,
            session_id=session_id,
            reports_state=reports_state,
            key_prefix=key_prefix,
            retrieval_mode=retrieval_mode,
        )


def _render_scope_selector(
    label: str,
    scope_key: str,
    options: Sequence[str],
    ai_service: AIService,
    session_id: str,
    reports_state: Dict[str, IntelligenceReport],
    key_prefix: str,
    retrieval_mode: str = "fast",
) -> None:
    if not options:
        empty_state(
            f"No {label.lower()} options available.",
            "Try a different search term or confirm the repository metadata includes this scope.",
        )
        return

    selected = st.selectbox(
        label,
        options=list(options),
        key=f"{key_prefix}_{scope_key}_select",
    )
    st.caption(f"Selected {label.lower()}: {selected}")
    if st.button(
        "Analyze Module",
        key=f"{key_prefix}_{scope_key}_button",
        use_container_width=True,
        help=f"Generate a grounded explanation for the selected {label.lower()}.",
    ):
        report = run_with_progress_messages(
            f"Generating {label.lower()} explanation...",
            DEFAULT_PROGRESS_MESSAGES,
            lambda: ai_service.generate_module_summary(
                selected,
                session_id=session_id,
                retrieval_mode=retrieval_mode,
            ),
            "Module analysis generated.",
        )
        reports_state[f"{scope_key}:{selected}"] = report

    report_key = f"{scope_key}:{selected}"
    if report_key in reports_state:
        report = reports_state[report_key]
        st.markdown("**Purpose**")
        st.write(report.summary)
        st.markdown("**Responsibilities**")
        st.write(report.detailed_explanation)
        if report.repository_insights:
            st.markdown("**Related Components**")
            for item in report.repository_insights:
                st.write(f"- {item}")
