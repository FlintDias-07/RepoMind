"""Repository-scoped session memory for AI question answering."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class MemoryTurn:
    question: str
    answer: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RepositoryConversation:
    repository_name: str
    session_id: str
    turns: List[MemoryTurn] = field(default_factory=list)


class ConversationMemory:
    """Keep repository-specific conversation state for follow-up questions."""

    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self._sessions: Dict[tuple[str, str], RepositoryConversation] = {}

    def append_turn(self, repository_name: str, session_id: str, question: str, answer: str) -> RepositoryConversation:
        key = (repository_name, session_id)
        conversation = self._sessions.setdefault(
            key,
            RepositoryConversation(repository_name=repository_name, session_id=session_id),
        )
        conversation.turns.append(MemoryTurn(question=question, answer=answer))
        if len(conversation.turns) > self.max_turns:
            conversation.turns = conversation.turns[-self.max_turns :]
        return conversation

    def get_history(self, repository_name: str, session_id: str) -> Sequence[MemoryTurn]:
        return tuple(self._sessions.get((repository_name, session_id), RepositoryConversation(repository_name, session_id)).turns)

    def render_history(self, repository_name: str, session_id: str) -> Sequence[str]:
        rendered = []
        for turn in self.get_history(repository_name, session_id):
            rendered.append(f"Q: {turn.question}")
            rendered.append(f"A: {turn.answer}")
        return tuple(rendered)

    def clear(self, repository_name: str, session_id: str) -> None:
        self._sessions.pop((repository_name, session_id), None)
