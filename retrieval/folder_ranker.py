"""Folder-aware ranking adjustments."""

from __future__ import annotations

from typing import Optional

from .ranking_config import RankingConfig


class FolderRanker:
    """Apply configurable folder/file path multipliers."""

    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()

    def weight_for(self, file_path: str) -> tuple[float, list[str]]:
        normalized = (file_path or "").lower().replace("\\", "/")
        best_pattern = None
        best_weight = self.config.default_folder_weight
        for pattern, weight in self.config.folder_weights.items():
            normalized_pattern = pattern.lower().replace("\\", "/")
            if normalized.endswith(normalized_pattern) or normalized_pattern in normalized:
                if abs(weight - 1.0) > abs(best_weight - 1.0):
                    best_weight = weight
                    best_pattern = pattern

        if best_pattern is None:
            return self.config.default_folder_weight, []
        return best_weight, [f"path_weight={best_pattern}:{best_weight}"]
