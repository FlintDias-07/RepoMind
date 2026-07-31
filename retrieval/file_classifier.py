"""Classify repository files into retrieval-aware categories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class FileCategory:
    SOURCE_CODE = "source_code"
    DOCUMENTATION = "documentation"
    TESTS = "tests"
    CONFIGURATION = "configuration"
    ASSETS = "assets"
    GENERATED_CODE = "generated_code"


@dataclass(frozen=True)
class FileClassification:
    category: str
    reason: str


class FileClassifier:
    """Apply conservative file categorization heuristics."""

    DOCUMENTATION_EXTENSIONS = {".md", ".rst", ".adoc", ".txt"}
    CONFIG_EXTENSIONS = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".env", ".properties", ".xml"}
    ASSET_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".pdf",
        ".mp4",
        ".mp3",
        ".woff",
        ".woff2",
        ".ttf",
    }
    SOURCE_CODE_EXTENSIONS = {
        ".py",
        ".java",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".go",
        ".rb",
        ".php",
        ".cs",
        ".cpp",
        ".c",
        ".h",
        ".rs",
        ".kt",
        ".swift",
        ".sql",
        ".css",
        ".html",
    }

    GENERATED_MARKERS = {
        "generated",
        "__generated__",
        ".generated",
        "gen",
        "dist",
        "build",
        "target",
        "coverage",
        ".next",
        ".nuxt",
    }
    TEST_MARKERS = {
        "test",
        "tests",
        "__tests__",
        "spec",
        "specs",
        "fixtures",
        "integration",
        "e2e",
    }
    DOC_MARKERS = {"docs", "doc", "documentation", "wiki", "guides", "examples"}
    CONFIG_NAMES = {
        "dockerfile",
        "makefile",
        "compose.yml",
        "compose.yaml",
        "package-lock.json",
        "package.json",
        "requirements.txt",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "pom.xml",
        "build.gradle",
        "application.yml",
        "application.yaml",
    }

    def classify(self, file_path: str) -> FileClassification:
        normalized = file_path.replace("\\", "/").lower()
        path = Path(normalized)
        name = path.name
        parts = set(path.parts)
        extension = path.suffix

        if name in {"readme.md", "changelog.md", "contributing.md", "license"} or extension in self.DOCUMENTATION_EXTENSIONS:
            return FileClassification(FileCategory.DOCUMENTATION, "documentation extension or common document name")

        if any(marker in parts for marker in self.TEST_MARKERS) or name.startswith("test_") or name.endswith("_test.py") or name.endswith(".spec.js") or name.endswith(".test.js"):
            return FileClassification(FileCategory.TESTS, "test directory or filename pattern")

        if any(marker in parts for marker in self.GENERATED_MARKERS) or ".min." in name or name.endswith(".generated.py") or name.endswith(".pb.go"):
            return FileClassification(FileCategory.GENERATED_CODE, "generated/build artifact pattern")

        if extension in self.ASSET_EXTENSIONS:
            return FileClassification(FileCategory.ASSETS, "asset extension")

        if name in self.CONFIG_NAMES or extension in self.CONFIG_EXTENSIONS or any(part in {"config", "configs", ".github"} for part in parts):
            return FileClassification(FileCategory.CONFIGURATION, "configuration filename or extension")

        if any(marker in parts for marker in self.DOC_MARKERS):
            return FileClassification(FileCategory.DOCUMENTATION, "documentation directory")

        if extension in self.SOURCE_CODE_EXTENSIONS:
            return FileClassification(FileCategory.SOURCE_CODE, "source code extension")

        return FileClassification(FileCategory.ASSETS, "fallback non-code asset")
