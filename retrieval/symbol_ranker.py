"""Symbol-aware ranking boosts for code retrieval."""

from __future__ import annotations

import re
from typing import Optional, Set

from .ranking_config import RankingConfig


class SymbolRanker:
    """Boost code chunks whose symbols align with the query intent."""

    TOKEN_PATTERN = re.compile(r"[a-zA-Z_@][a-zA-Z0-9_@]*")
    CAMEL_SPLIT_PATTERN = re.compile(r"(?<!^)(?=[A-Z])")

    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()

    def score(
        self,
        query: str,
        class_name: str,
        function_name: str,
        file_name: str,
        document: str,
    ) -> tuple[float, list[str]]:
        query_tokens = self._normalize_tokens(query)
        symbol_tokens = self._normalize_tokens(" ".join([class_name, function_name, file_name, document[:400]]))
        reasons: list[str] = []
        bonus = 0.0

        direct_matches = query_tokens & symbol_tokens
        if direct_matches:
            bonus += min(self.config.strong_symbol_boost_value, 0.15 * len(direct_matches))
            reasons.append(f"direct_symbol_match={','.join(sorted(direct_matches)[:4])}")

        for domain, domain_terms in self.config.symbol_boost_terms.items():
            if query_tokens & domain_terms:
                domain_matches = symbol_tokens & domain_terms
                if domain_matches:
                    bonus += min(self.config.symbol_boost_value, 0.12 * len(domain_matches))
                    reasons.append(f"{domain}_symbols={','.join(sorted(domain_matches)[:4])}")

        return bonus, reasons

    def _normalize_tokens(self, value: str) -> Set[str]:
        tokens: Set[str] = set()
        for raw in self.TOKEN_PATTERN.findall(value or ""):
            raw = raw.strip()
            if not raw:
                continue
            fragments = [part.lower() for part in self.CAMEL_SPLIT_PATTERN.split(raw) if part]
            for fragment in fragments:
                cleaned = fragment.strip("(){}[],:;\"'`").lower()
                if len(cleaned) >= 3:
                    tokens.add(cleaned)
            collapsed = "".join(fragments)
            if len(collapsed) >= 3:
                tokens.add(collapsed)
        return tokens
