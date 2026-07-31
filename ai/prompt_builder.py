"""Prompt construction for repository intelligence tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .context_builder import AIContext, ContextChunk


PromptRole = str


@dataclass(frozen=True)
class PromptPackage:
    """Provider-agnostic prompt payload."""

    system_prompt: str
    user_prompt: str
    context_summary: str
    role: PromptRole
    warnings: Sequence[str] = field(default_factory=tuple)


class PromptBuilder:
    """Build strict prompts that force grounded repository answers."""

    DEFAULT_ROLE: PromptRole = "software_architect"

    ROLE_DESCRIPTIONS = {
        "software_architect": "Senior Software Architect",
        "backend_engineer": "Senior Backend Engineer",
        "code_reviewer": "Code Reviewer",
        "technical_mentor": "Technical Mentor",
    }

    def build(
        self,
        context: AIContext,
        role: PromptRole | None = None,
        follow_up_history: Sequence[str] | None = None,
        task_hint: str = "",
    ) -> PromptPackage:
        selected_role = role or self.DEFAULT_ROLE
        role_description = self.ROLE_DESCRIPTIONS.get(selected_role, self.ROLE_DESCRIPTIONS[self.DEFAULT_ROLE])
        history = tuple(item.strip() for item in (follow_up_history or ()) if item and item.strip())
        context_summary = self._render_context(context)

        system_prompt = (
            f"You are acting as a {role_description} for a code intelligence platform.\n"
            "Use ONLY the supplied repository context.\n"
            "Never invent code, files, classes, functions, or architecture.\n"
            "If the context is insufficient, say exactly: "
            '"I could not find enough evidence in the indexed repository."\n'
            "Always cite source files, classes, functions, and line numbers when evidence exists.\n"
            "Be precise, technical, and evidence-driven."
        )

        user_sections = [
            f"Repository: {context.repository.repository_name}",
        ]
        if task_hint:
            user_sections.append(f"Task: {task_hint}")
        if history:
            user_sections.append("Conversation History:\n" + "\n".join(f"- {item}" for item in history))
        user_sections.extend(
            [
                f"Question: {context.question}",
                "Repository Context:",
                context_summary,
                (
                    "Answer format:\n"
                    "1. Summary: provide a direct 2-5 sentence answer, not just a heading.\n"
                    "2. Detailed Explanation: use readable Markdown sections and numbered flow steps when useful.\n"
                    "3. Repository Insights: include concise implementation observations.\n"
                    "4. Referenced Evidence: list repository-relative files and line ranges from the supplied context."
                ),
            ]
        )

        return PromptPackage(
            system_prompt=system_prompt,
            user_prompt="\n\n".join(user_sections),
            context_summary=context_summary,
            role=selected_role,
            warnings=context.warnings,
        )

    def _render_context(self, context: AIContext) -> str:
        if not context.chunks:
            return "No repository context was retrieved."

        rendered_chunks = []
        for chunk in context.chunks:
            header_parts = [
                f"file={chunk.file_path}",
                f"lines={chunk.line_span or 'unknown'}",
            ]
            if chunk.class_name:
                header_parts.append(f"class={chunk.class_name}")
            if chunk.function_name:
                header_parts.append(f"function={chunk.function_name}")
            rendered_chunks.append(
                "[Context]\n"
                + ", ".join(header_parts)
                + "\n"
                + chunk.content.strip()
            )
        return "\n\n".join(rendered_chunks)
