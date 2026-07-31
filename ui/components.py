"""Shared Streamlit presentation helpers for RepoMind AI."""

from __future__ import annotations

from html import escape
from typing import Callable, Iterable, Optional, TypeVar

import streamlit as st


T = TypeVar("T")


DEFAULT_PROGRESS_MESSAGES = (
    "Retrieving repository context...",
    "Ranking relevant code...",
    "Generating grounded answer...",
    "Formatting report...",
)


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --repomind-bg: #0f172a;
            --repomind-panel: rgba(15, 23, 42, 0.62);
            --repomind-panel-strong: rgba(17, 24, 39, 0.92);
            --repomind-border: rgba(148, 163, 184, 0.22);
            --repomind-border-strong: rgba(148, 163, 184, 0.34);
            --repomind-text-muted: #94a3b8;
            --repomind-text-soft: #cbd5e1;
            --repomind-accent: #38bdf8;
        }
        .stApp,
        [data-testid="stAppViewContainer"] {
            background: #0b1120;
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid var(--repomind-border);
        }
        [data-testid="stHeader"] { background: rgba(11, 17, 32, 0.88); }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        label,
        .stRadio span,
        .stSelectbox span,
        .stTextInput span {
            color: inherit;
        }
        .main .block-container {
            padding-top: 1.15rem;
            padding-bottom: 2rem;
            width: min(100%, 1080px);
            max-width: 1080px;
        }
        h1, h2, h3 { letter-spacing: 0; }
        h2, h3 { margin-top: 1.15rem; }
        [data-testid="stVerticalBlock"] { gap: .72rem; }
        .repomind-hero {
            padding: 1.1rem 1.25rem;
            border: 1px solid var(--repomind-border);
            border-radius: 8px;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(17, 24, 39, 0.92));
            margin-bottom: 1rem;
            box-shadow: 0 14px 32px rgba(2, 6, 23, 0.18);
        }
        .repomind-hero h1 { margin: 0; font-size: 1.85rem; }
        .repomind-hero p { margin: .35rem 0 0 0; color: var(--repomind-text-soft); max-width: 860px; }
        .section-kicker { color: var(--repomind-text-muted); font-size: .92rem; margin-top: -.25rem; }
        .info-card {
            border: 1px solid var(--repomind-border);
            border-radius: 8px;
            padding: .9rem .95rem;
            background: var(--repomind-panel);
            min-height: 104px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025);
        }
        .info-card-title {
            color: var(--repomind-text-muted);
            font-size: .74rem;
            font-weight: 700;
            letter-spacing: .04em;
            line-height: 1.2;
            text-transform: uppercase;
        }
        .info-card-value {
            color: #f8fafc;
            font-size: 1.18rem;
            font-weight: 750;
            line-height: 1.25;
            margin-top: .38rem;
            overflow-wrap: anywhere;
        }
        .info-card-caption {
            color: var(--repomind-text-muted);
            font-size: .82rem;
            line-height: 1.35;
            margin-top: .45rem;
        }
        .status-badge {
            display: inline-block;
            padding: .18rem .5rem;
            border-radius: 999px;
            border: 1px solid var(--repomind-border-strong);
            font-size: .78rem;
            color: #e5e7eb;
            background: rgba(31, 41, 55, 0.8);
        }
        .empty-state {
            border: 1px dashed var(--repomind-border-strong);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            background: rgba(15, 23, 42, 0.34);
            color: var(--repomind-text-soft);
            margin: .35rem 0;
        }
        .empty-state strong { color: #f8fafc; }
        .onboarding-card {
            border: 1px solid rgba(56, 189, 248, 0.28);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            background: linear-gradient(135deg, rgba(8, 47, 73, 0.34), rgba(15, 23, 42, 0.72));
            color: var(--repomind-text-soft);
            margin: .5rem 0 .8rem;
        }
        .onboarding-card h3 {
            color: #f8fafc;
            font-size: 1.12rem;
            line-height: 1.3;
            margin: 0 0 .35rem;
        }
        .onboarding-card p { margin: 0; }
        .repomind-panel-note {
            border: 1px solid var(--repomind-border);
            border-radius: 8px;
            padding: .95rem 1rem;
            background: var(--repomind-panel);
            color: var(--repomind-text-soft);
        }
        .repomind-panel-note strong { color: #f8fafc; }
        .repomind-ask-intro {
            border: 1px solid rgba(56, 189, 248, 0.26);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            background: linear-gradient(135deg, rgba(8, 47, 73, 0.42), rgba(15, 23, 42, 0.76));
            margin-bottom: .25rem;
        }
        .repomind-ask-intro h3 {
            margin: 0;
            font-size: 1.22rem;
            line-height: 1.28;
        }
        .repomind-ask-intro p {
            color: var(--repomind-text-soft);
            margin: .35rem 0 0 0;
        }
        .repomind-mode-card {
            border: 1px solid var(--repomind-border);
            border-radius: 8px;
            padding: .82rem .9rem;
            min-height: 108px;
            background: rgba(15, 23, 42, 0.44);
        }
        .repomind-mode-card strong { color: #f8fafc; font-size: 1rem; }
        .repomind-mode-card p {
            margin: .28rem 0 0;
            color: var(--repomind-text-soft);
            font-size: .9rem;
        }
        .repomind-mode-card span {
            display: inline-block;
            margin-top: .42rem;
            color: var(--repomind-text-muted);
            font-size: .82rem;
        }
        .progress-step-list {
            border: 1px solid var(--repomind-border);
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.45);
            padding: .85rem .95rem;
        }
        .progress-step {
            display: flex;
            align-items: center;
            gap: .6rem;
            color: var(--repomind-text-muted);
            font-size: .92rem;
            margin: .42rem 0;
        }
        .progress-step strong { color: #f8fafc; }
        .progress-dot {
            width: .58rem;
            height: .58rem;
            border-radius: 999px;
            border: 1px solid var(--repomind-border-strong);
            flex: 0 0 auto;
        }
        .progress-step.done .progress-dot { background: #38bdf8; border-color: #38bdf8; }
        .progress-step.active .progress-dot {
            background: #f59e0b;
            border-color: #f59e0b;
            box-shadow: 0 0 0 4px rgba(245, 158, 11, 0.14);
        }
        .suggestion-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: .5rem;
            margin: .4rem 0 .2rem;
        }
        .suggestion-label {
            color: var(--repomind-text-muted);
            font-size: .86rem;
            margin-top: .7rem;
        }
        .report-summary {
            border: 1px solid var(--repomind-border);
            border-radius: 8px;
            padding: 1rem;
            background: var(--repomind-panel);
            font-size: 1rem;
            line-height: 1.65;
        }
        .report-confidence {
            display: inline-flex;
            align-items: center;
            gap: .4rem;
            margin-top: .6rem;
            padding: .3rem .58rem;
            border: 1px solid var(--repomind-border-strong);
            border-radius: 999px;
            background: rgba(30, 41, 59, 0.72);
            color: #e2e8f0;
            font-size: .84rem;
        }
        .evidence-card {
            border: 1px solid var(--repomind-border);
            border-radius: 8px;
            padding: .8rem .9rem;
            background: rgba(15, 23, 42, 0.42);
            min-height: 132px;
        }
        .evidence-card h4 {
            margin: 0 0 .45rem 0;
            font-size: .95rem;
            color: #f8fafc;
        }
        .evidence-item {
            margin: .28rem 0;
            color: var(--repomind-text-soft);
            overflow-wrap: anywhere;
        }
        .stTextArea textarea {
            min-height: 126px !important;
            border-radius: 8px !important;
            border-color: rgba(56, 189, 248, 0.28) !important;
            background: rgba(2, 6, 23, 0.28) !important;
            font-size: .98rem !important;
            line-height: 1.5 !important;
        }
        .stTextArea textarea:focus {
            border-color: rgba(56, 189, 248, 0.72) !important;
            box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.32) !important;
        }
        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            border-radius: 7px;
            min-height: 2.55rem;
            font-weight: 650;
        }
        div[data-testid="stButton"] button[kind="primary"] {
            border: 1px solid rgba(56, 189, 248, 0.36);
        }
        div[data-testid="stDownloadButton"] button {
            border: 1px solid var(--repomind-border-strong);
            color: #e2e8f0;
        }
        code { white-space: pre-wrap; word-break: break-word; }
        @media (max-width: 900px) {
            .main .block-container { padding-left: .85rem; padding-right: .85rem; }
            .repomind-hero h1 { font-size: 1.55rem; }
            .info-card { min-height: 96px; }
            .suggestion-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", eyebrow: str = "RepoMind AI") -> None:
    st.markdown(
        f"""
        <div class="repomind-hero">
            <div class="section-kicker">{escape(eyebrow)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, description: str = "") -> None:
    st.subheader(title)
    if description:
        st.markdown(f'<div class="section-kicker">{escape(description)}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: object, help_text: str = "") -> None:
    help_html = f"<div class='info-card-caption'>{escape(help_text)}</div>" if help_text else ""
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-title">{escape(label)}</div>
            <div class="info-card-value">{escape(str(value))}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str, action: Optional[str] = None) -> None:
    action_html = f"<p><strong>{escape(action)}</strong></p>" if action else ""
    st.markdown(
        f"""
        <div class="empty-state">
            <strong>{escape(title)}</strong>
            <p>{escape(body)}</p>
            {action_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_note(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="repomind-panel-note">
            <strong>{escape(title)}</strong>
            <div>{escape(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def onboarding_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="onboarding-card">
            <h3>{escape(title)}</h3>
            <p>{escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def progress_step_list(steps: Iterable[str], active_index: int) -> str:
    rendered_steps = []
    for index, step in enumerate(steps):
        if index < active_index:
            state = "done"
            label = "Complete"
        elif index == active_index:
            state = "active"
            label = "In progress"
        else:
            state = "pending"
            label = "Pending"
        rendered_steps.append(
            f"""
            <div class="progress-step {state}">
                <span class="progress-dot"></span>
                <span><strong>{escape(step)}</strong> <span>{escape(label)}</span></span>
            </div>
            """
        )
    return f"<div class='progress-step-list'>{''.join(rendered_steps)}</div>"


def status_badge(label: str) -> None:
    st.markdown(f'<span class="status-badge">{escape(label)}</span>', unsafe_allow_html=True)


def safe_error(message: str, details: Optional[BaseException] = None) -> None:
    st.error(message)
    if details is not None:
        with st.expander("Technical details"):
            st.code(str(details), language="text")


def bullet_list(items: Iterable[str]) -> None:
    for item in items:
        st.write(f"- {item}")


def run_with_progress_messages(
    title: str,
    messages: Iterable[str],
    action: Callable[[], T],
    complete_message: str = "Complete.",
) -> T:
    with st.status(title, expanded=True) as status:
        for message in messages:
            st.write(message)
        result = action()
        status.update(label=complete_message, state="complete", expanded=False)
        return result
