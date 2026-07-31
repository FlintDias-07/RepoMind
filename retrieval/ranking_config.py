"""Configuration objects for retrieval ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from .file_classifier import FileCategory
from .query_classifier import QueryIntent


@dataclass(frozen=True)
class RankingConfig:
    """Central ranking configuration with conservative defaults."""

    candidate_multiplier: int = 5
    vector_weight: float = 0.65
    bm25_weight: float = 0.35
    rerank_candidate_limit: int = 20
    rerank_top_k: int = 5
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    folder_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "src/": 2.0,
            "controllers/": 1.8,
            "controller/": 1.8,
            "services/": 1.8,
            "service/": 1.8,
            "repository/": 1.7,
            "repositories/": 1.7,
            "security/": 1.8,
            "config/": 1.5,
            "readme.md": 0.3,
            "docs/": 0.3,
            "examples/": 0.2,
            "tests/": 0.2,
            "generated/": 0.1,
        }
    )
    default_folder_weight: float = 1.0
    metadata_bonuses: Dict[str, float] = field(
        default_factory=lambda: {
            FileCategory.SOURCE_CODE: 0.2,
            FileCategory.CONFIGURATION: 0.35,
            FileCategory.DOCUMENTATION: 0.05,
            FileCategory.TESTS: -0.4,
            FileCategory.GENERATED_CODE: -0.6,
            FileCategory.ASSETS: -0.8,
        }
    )
    intent_category_preferences: Dict[str, List[str]] = field(
        default_factory=lambda: {
            QueryIntent.AUTHENTICATION: [FileCategory.SOURCE_CODE],
            QueryIntent.CODE_UNDERSTANDING: [FileCategory.SOURCE_CODE],
            QueryIntent.ARCHITECTURE: [FileCategory.SOURCE_CODE, FileCategory.DOCUMENTATION],
            QueryIntent.API: [FileCategory.SOURCE_CODE, FileCategory.DOCUMENTATION],
            QueryIntent.CONFIGURATION: [FileCategory.CONFIGURATION, FileCategory.SOURCE_CODE],
            QueryIntent.DATABASE: [FileCategory.SOURCE_CODE, FileCategory.CONFIGURATION],
            QueryIntent.DOCUMENTATION: [FileCategory.DOCUMENTATION],
            QueryIntent.INSTALLATION: [FileCategory.DOCUMENTATION, FileCategory.CONFIGURATION],
            QueryIntent.TESTING: [FileCategory.TESTS, FileCategory.SOURCE_CODE],
            QueryIntent.GENERAL: [FileCategory.SOURCE_CODE, FileCategory.DOCUMENTATION],
        }
    )
    intent_excluded_categories: Dict[str, Set[str]] = field(
        default_factory=lambda: {
            QueryIntent.AUTHENTICATION: {FileCategory.TESTS, FileCategory.DOCUMENTATION, FileCategory.GENERATED_CODE, FileCategory.ASSETS},
            QueryIntent.CODE_UNDERSTANDING: {FileCategory.TESTS, FileCategory.DOCUMENTATION, FileCategory.GENERATED_CODE, FileCategory.ASSETS},
            QueryIntent.API: {FileCategory.TESTS, FileCategory.GENERATED_CODE, FileCategory.ASSETS},
            QueryIntent.CONFIGURATION: {FileCategory.TESTS, FileCategory.GENERATED_CODE, FileCategory.ASSETS},
            QueryIntent.DATABASE: {FileCategory.TESTS, FileCategory.GENERATED_CODE, FileCategory.ASSETS},
            QueryIntent.ARCHITECTURE: {FileCategory.TESTS, FileCategory.GENERATED_CODE, FileCategory.ASSETS},
            QueryIntent.INSTALLATION: {FileCategory.TESTS, FileCategory.GENERATED_CODE, FileCategory.ASSETS},
            QueryIntent.DOCUMENTATION: {FileCategory.TESTS, FileCategory.GENERATED_CODE, FileCategory.ASSETS},
            QueryIntent.TESTING: {FileCategory.GENERATED_CODE, FileCategory.ASSETS},
            QueryIntent.GENERAL: {FileCategory.GENERATED_CODE, FileCategory.ASSETS},
        }
    )
    language_terms: Dict[str, Set[str]] = field(
        default_factory=lambda: {
            "python": {"python", ".py"},
            "java": {"java", ".java"},
            "javascript": {"javascript", ".js", "node"},
            "typescript": {"typescript", ".ts", ".tsx"},
            "yaml": {"yaml", "yml", ".yaml", ".yml"},
            "sql": {"sql", ".sql"},
        }
    )
    symbol_boost_terms: Dict[str, Set[str]] = field(
        default_factory=lambda: {
            "authentication": {"auth", "authenticate", "authentication", "login", "signin", "security", "password", "user", "role"},
            "jwt": {"jwt", "token", "verifytoken", "generatetoken", "tokenprovider", "jwtu", "jwtutil"},
            "api": {"controller", "endpoint", "request", "response", "route"},
            "database": {"database", "repository", "query", "connection", "config", "sql"},
            "config": {"config", "configuration", "properties", "yaml", "env"},
        }
    )
    symbol_boost_value: float = 0.35
    strong_symbol_boost_value: float = 0.55
