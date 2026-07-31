"""Metadata-aware retrieval filtering and bonuses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Set

from .query_classifier import QueryClassification
from .ranking_config import RankingConfig


@dataclass(frozen=True)
class RetrievalFilters:
    include_categories: Set[str] = field(default_factory=set)
    exclude_categories: Set[str] = field(default_factory=set)
    languages: Set[str] = field(default_factory=set)
    file_extensions: Set[str] = field(default_factory=set)
    repositories: Set[str] = field(default_factory=set)
    folders: Set[str] = field(default_factory=set)
    file_names: Set[str] = field(default_factory=set)


class MetadataFilter:
    """Infer and apply metadata filters to retrieval candidates."""

    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()

    def infer_filters(
        self,
        query: str,
        classification: QueryClassification,
        explicit_filters: Optional[Dict[str, Sequence[str] | str]] = None,
    ) -> RetrievalFilters:
        normalized_query = (query or "").lower()
        include_categories = set(self.config.intent_category_preferences.get(classification.intent, []))
        exclude_categories = set(self.config.intent_excluded_categories.get(classification.intent, set()))
        languages: Set[str] = set()
        file_extensions: Set[str] = set()
        repositories: Set[str] = set()
        folders: Set[str] = set()
        file_names: Set[str] = set()

        for language, terms in self.config.language_terms.items():
            if any(term in normalized_query for term in terms):
                languages.add(language)
                for term in terms:
                    if term.startswith("."):
                        file_extensions.add(term)

        explicit = explicit_filters or {}
        include_categories.update(self._to_set(explicit.get("category")))
        exclude_categories.update(self._to_set(explicit.get("exclude_category")))
        languages.update(self._to_set(explicit.get("language")))
        file_extensions.update(self._to_set(explicit.get("file_extension")))
        repositories.update(self._to_set(explicit.get("repository")))
        folders.update(self._to_set(explicit.get("folder")))
        file_names.update(self._to_set(explicit.get("file_name")))

        return RetrievalFilters(
            include_categories=include_categories,
            exclude_categories=exclude_categories,
            languages={value.lower() for value in languages},
            file_extensions={value.lower() for value in file_extensions},
            repositories={value.lower() for value in repositories},
            folders={value.lower() for value in folders},
            file_names={value.lower() for value in file_names},
        )

    def matches(self, metadata: Dict[str, Any], filters: RetrievalFilters) -> bool:
        category = str(metadata.get("file_category", "")).lower()
        language = str(metadata.get("language", "")).lower()
        file_path = str(metadata.get("file_path", "")).lower().replace("\\", "/")
        file_name = str(metadata.get("file_name", "")).lower()
        repository = str(metadata.get("repository_name", "")).lower()

        if filters.include_categories and category not in {item.lower() for item in filters.include_categories}:
            return False
        if filters.exclude_categories and category in {item.lower() for item in filters.exclude_categories}:
            return False
        if filters.languages and language not in filters.languages:
            return False
        if filters.file_extensions and not any(file_name.endswith(ext) for ext in filters.file_extensions):
            return False
        if filters.repositories and repository not in filters.repositories:
            return False
        if filters.file_names and file_name not in filters.file_names:
            return False
        if filters.folders and not any(folder in file_path for folder in filters.folders):
            return False
        return True

    def metadata_bonus(self, metadata: Dict[str, Any], filters: RetrievalFilters) -> tuple[float, list[str]]:
        category = str(metadata.get("file_category", "")).lower()
        language = str(metadata.get("language", "")).lower()
        file_path = str(metadata.get("file_path", "")).lower().replace("\\", "/")
        file_name = str(metadata.get("file_name", "")).lower()
        repository = str(metadata.get("repository_name", "")).lower()
        bonus = self.config.metadata_bonuses.get(category, 0.0)
        reasons = [f"category={category}"]

        if filters.languages and language in filters.languages:
            bonus += 0.1
            reasons.append(f"language={language}")
        if filters.file_extensions and any(file_name.endswith(ext) for ext in filters.file_extensions):
            bonus += 0.1
            reasons.append(f"extension={file_name.split('.')[-1] if '.' in file_name else file_name}")
        if filters.repositories and repository in filters.repositories:
            bonus += 0.1
            reasons.append(f"repository={repository}")
        if filters.folders and any(folder in file_path for folder in filters.folders):
            bonus += 0.1
            reasons.append("folder match")
        if filters.file_names and file_name in filters.file_names:
            bonus += 0.15
            reasons.append(f"file={file_name}")

        return bonus, reasons

    def _to_set(self, value: Sequence[str] | str | None) -> Set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            return {value}
        return {item for item in value if item}
