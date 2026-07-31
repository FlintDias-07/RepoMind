"""Heuristic query intent classifier for retrieval routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set


class QueryIntent:
    CODE_UNDERSTANDING = "code_understanding"
    ARCHITECTURE = "architecture"
    AUTHENTICATION = "authentication"
    CONFIGURATION = "configuration"
    DATABASE = "database"
    API = "api"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    INSTALLATION = "installation"
    GENERAL = "general"


@dataclass(frozen=True)
class QueryClassification:
    intent: str
    matched_terms: Set[str]


class QueryClassifier:
    """Classify developer questions into routing intents."""

    INTENT_TERMS: Dict[str, Set[str]] = {
        QueryIntent.AUTHENTICATION: {"auth", "authentication", "jwt", "token", "login", "signin", "security", "oauth", "session"},
        QueryIntent.CONFIGURATION: {"config", "configuration", "configured", "env", "setting", "yaml", "yml", "properties", "toml"},
        QueryIntent.DATABASE: {"database", "db", "sql", "query", "repository", "storage", "connection", "orm"},
        QueryIntent.API: {"api", "endpoint", "route", "controller", "request", "response", "http"},
        QueryIntent.DOCUMENTATION: {"readme", "docs", "document", "guide", "usage", "how to"},
        QueryIntent.INSTALLATION: {"install", "setup", "run", "start", "quickstart", "requirements"},
        QueryIntent.TESTING: {"test", "testing", "spec", "coverage", "fixture", "assert"},
        QueryIntent.ARCHITECTURE: {"architecture", "flow", "design", "structure", "component", "service"},
        QueryIntent.CODE_UNDERSTANDING: {"function", "class", "method", "file", "module", "implement", "handles", "works"},
    }

    def classify(self, query: str) -> QueryClassification:
        normalized = (query or "").strip().lower()
        if not normalized:
            return QueryClassification(QueryIntent.GENERAL, set())

        scores: Dict[str, Set[str]] = {}
        for intent, terms in self.INTENT_TERMS.items():
            matches = {term for term in terms if term in normalized}
            if matches:
                scores[intent] = matches

        if not scores:
            return QueryClassification(QueryIntent.GENERAL, set())

        best_intent = max(
            scores.items(),
            key=lambda item: (len(item[1]), self._intent_priority(item[0])),
        )[0]
        return QueryClassification(best_intent, scores[best_intent])

    def _intent_priority(self, intent: str) -> int:
        ordered = [
            QueryIntent.AUTHENTICATION,
            QueryIntent.CONFIGURATION,
            QueryIntent.DATABASE,
            QueryIntent.API,
            QueryIntent.TESTING,
            QueryIntent.INSTALLATION,
            QueryIntent.DOCUMENTATION,
            QueryIntent.ARCHITECTURE,
            QueryIntent.CODE_UNDERSTANDING,
            QueryIntent.GENERAL,
        ]
        return len(ordered) - ordered.index(intent) if intent in ordered else 0
