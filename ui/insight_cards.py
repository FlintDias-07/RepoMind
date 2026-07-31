"""Insight cards for the Repository Intelligence experience."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import streamlit as st

from ai.ai_service import AIService, IntelligenceReport
from .components import DEFAULT_PROGRESS_MESSAGES, run_with_progress_messages


@dataclass(frozen=True)
class InsightCardSpec:
    key: str
    title: str
    description: str
    action_key: str


def get_insight_card_specs() -> Sequence[InsightCardSpec]:
    return (
        InsightCardSpec("repository_summary", "Repository Summary", "High-level purpose and structure.", "repository_summary"),
        InsightCardSpec("architecture_overview", "Architecture Overview", "How major modules interact.", "architecture_overview"),
        InsightCardSpec("important_modules", "Main Modules", "Modules that carry the most responsibility.", "important_modules"),
        InsightCardSpec("onboarding_guide", "Developer Onboarding Guide", "Where new engineers should start.", "onboarding_guide"),
        InsightCardSpec("learning_path", "Learning Path", "A practical sequence for new developers.", "learning_path"),
        InsightCardSpec("entry_points", "Potential Entry Points", "Best files or flows to start reading.", "entry_points"),
        InsightCardSpec("core_components", "Core Components", "Central parts of the system.", "core_components"),
        InsightCardSpec("main_technologies", "Main Technologies", "Frameworks and technical patterns in use.", "main_technologies"),
        InsightCardSpec("dependencies", "Dependencies", "Important libraries and integrations.", "dependencies"),
        InsightCardSpec("folder_structure", "Folder Structure", "How the repository is organized.", "folder_structure"),
    )


def render_insight_cards(
    ai_service: AIService,
    session_id: str,
    reports_state: Dict[str, IntelligenceReport],
    key_prefix: str = "insight",
    retrieval_mode: str = "fast",
) -> Optional[IntelligenceReport]:
    selected_report: Optional[IntelligenceReport] = None
    groups = {
        "Understand": ("repository_summary", "architecture_overview", "important_modules"),
        "Onboard": ("onboarding_guide", "learning_path", "entry_points"),
        "Analyze": ("core_components", "main_technologies", "dependencies", "folder_structure"),
    }
    specs_by_key = {spec.key: spec for spec in get_insight_card_specs()}
    tabs = st.tabs(list(groups.keys()))
    for tab, (group_name, keys) in zip(tabs, groups.items()):
        with tab:
            for row_start in range(0, len(keys), 3):
                columns = st.columns(3)
                for column, key in zip(columns, keys[row_start : row_start + 3]):
                    spec = specs_by_key[key]
                    with column:
                        with st.container(border=True):
                            st.markdown(f"**{spec.title}**")
                            st.caption(spec.description)
                            if st.button(
                                "Generate",
                                key=f"{key_prefix}_{spec.key}",
                                use_container_width=True,
                                help=f"Generate {spec.title.lower()} insights.",
                            ):
                                report = run_with_progress_messages(
                                    f"Generating {spec.title.lower()}...",
                                    DEFAULT_PROGRESS_MESSAGES,
                                    lambda action_key=spec.action_key: ai_service.run_action(
                                        action_key,
                                        session_id=session_id,
                                        retrieval_mode=retrieval_mode,
                                    ),
                                    "Report generated.",
                                )
                                reports_state[spec.key] = report
                                selected_report = report
                            if spec.key in reports_state:
                                st.success("Generated")
                                st.caption(reports_state[spec.key].summary[:140])
    return selected_report
