"""Route indexed chunks and search queries to the right vector collections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .file_classifier import FileCategory
from .query_classifier import QueryIntent


@dataclass(frozen=True)
class CollectionRoute:
    collection_key: str
    collection_name: str
    categories: List[str]


class CollectionRouter:
    """Map file categories and query intents to collection names."""

    CATEGORY_TO_COLLECTION = {
        FileCategory.SOURCE_CODE: "code_collection",
        FileCategory.DOCUMENTATION: "documentation_collection",
        FileCategory.TESTS: "test_collection",
        FileCategory.CONFIGURATION: "config_collection",
        FileCategory.ASSETS: "asset_collection",
        FileCategory.GENERATED_CODE: "generated_collection",
    }

    QUERY_TO_COLLECTIONS = {
        QueryIntent.AUTHENTICATION: ["code_collection"],
        QueryIntent.CODE_UNDERSTANDING: ["code_collection"],
        QueryIntent.ARCHITECTURE: ["code_collection", "documentation_collection"],
        QueryIntent.API: ["code_collection", "documentation_collection"],
        QueryIntent.CONFIGURATION: ["config_collection", "code_collection"],
        QueryIntent.DATABASE: ["code_collection", "config_collection"],
        QueryIntent.DOCUMENTATION: ["documentation_collection"],
        QueryIntent.INSTALLATION: ["documentation_collection", "config_collection"],
        QueryIntent.TESTING: ["test_collection", "code_collection"],
        QueryIntent.GENERAL: ["code_collection", "documentation_collection"],
    }

    def __init__(self, repository_name: str):
        self.repository_name = repository_name

    def collection_name_for_category(self, category: str) -> str:
        collection_key = self.CATEGORY_TO_COLLECTION.get(category, "code_collection")
        return self._qualified_name(collection_key)

    def build_routes(self) -> Dict[str, CollectionRoute]:
        routes: Dict[str, CollectionRoute] = {}
        for category, collection_key in self.CATEGORY_TO_COLLECTION.items():
            routes[collection_key] = CollectionRoute(
                collection_key=collection_key,
                collection_name=self._qualified_name(collection_key),
                categories=[cat for cat, key in self.CATEGORY_TO_COLLECTION.items() if key == collection_key],
            )
        return routes

    def collections_for_intent(self, intent: str) -> List[str]:
        collection_keys = self.QUERY_TO_COLLECTIONS.get(intent, self.QUERY_TO_COLLECTIONS[QueryIntent.GENERAL])
        return [self._qualified_name(key) for key in collection_keys]

    def _qualified_name(self, collection_key: str) -> str:
        return f"{self.repository_name}_{collection_key}"
