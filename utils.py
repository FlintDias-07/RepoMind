"""
Utility functions for RAG Application
Includes logging, file operations, and helper functions
"""

import logging
import os
from pathlib import Path
from datetime import datetime
import shutil
import json
from typing import List, Dict, Optional

from config import LOGS_DIR, LOG_LEVEL, LOG_FORMAT

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOGS_DIR / f"rag_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module
    
    Args:
        name: Module name for logging context
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def sanitize_repo_name(url: str) -> str:
    """
    Convert GitHub URL to a safe directory name
    
    Args:
        url: GitHub repository URL
        
    Returns:
        Sanitized directory name
        
    Examples:
        >>> sanitize_repo_name("https://github.com/user/repo.git")
        "user-repo"
        >>> sanitize_repo_name("https://github.com/user/repo")
        "user-repo"
    """
    # Extract repo name from URL
    repo_name = url.rstrip("/").split("/")[-1]
    repo_name = repo_name.replace(".git", "").replace(" ", "_")
    return repo_name


def validate_github_url(url: str) -> bool:
    """
    Validate GitHub repository URL format
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid GitHub URL, False otherwise
    """
    valid_patterns = [
        "github.com",
        "https://",
        "git@github.com",
    ]
    return any(pattern in url for pattern in valid_patterns)


def get_file_extension(filepath: str) -> str:
    """
    Get file extension with dot prefix
    
    Args:
        filepath: Path to file
        
    Returns:
        File extension (e.g., '.py')
    """
    return Path(filepath).suffix


def is_supported_file(filepath: str, supported_extensions: set) -> bool:
    """
    Check if file extension is supported
    
    Args:
        filepath: Path to file
        supported_extensions: Set of supported extensions (e.g., {'.py', '.java'})
        
    Returns:
        True if supported, False otherwise
    """
    return get_file_extension(filepath) in supported_extensions


def is_ignored_directory(dirpath: str, ignore_dirs: set) -> bool:
    """
    Check if directory should be ignored during traversal
    
    Args:
        dirpath: Path to directory
        ignore_dirs: Set of directory names to ignore
        
    Returns:
        True if directory should be ignored, False otherwise
    """
    path_parts = Path(dirpath).parts
    return any(part in ignore_dirs for part in path_parts)


def safe_delete_directory(dirpath: str) -> bool:
    """
    Safely delete a directory and all its contents
    
    Args:
        dirpath: Path to directory
        
    Returns:
        True if deletion successful, False otherwise
    """
    try:
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)
            logger = get_logger(__name__)
            logger.info(f"Deleted directory: {dirpath}")
            return True
        return False
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error deleting directory {dirpath}: {str(e)}")
        return False


def count_files(dirpath: str, supported_extensions: set) -> Dict[str, int]:
    """
    Count files by extension in a directory
    
    Args:
        dirpath: Path to directory
        supported_extensions: Set of supported extensions
        
    Returns:
        Dictionary with extension counts
    """
    counts = {}
    try:
        for filepath in Path(dirpath).rglob("*"):
            if filepath.is_file():
                ext = get_file_extension(str(filepath))
                if ext in supported_extensions:
                    counts[ext] = counts.get(ext, 0) + 1
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error counting files in {dirpath}: {str(e)}")
    
    return counts


def format_file_size(size_bytes: int) -> str:
    """
    Format bytes to human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
        
    Examples:
        >>> format_file_size(1024)
        "1.0 KB"
        >>> format_file_size(1048576)
        "1.0 MB"
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def save_metadata(metadata: Dict, filepath: str) -> bool:
    """
    Save metadata to JSON file
    
    Args:
        metadata: Dictionary of metadata
        filepath: Path to save JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        return True
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error saving metadata to {filepath}: {str(e)}")
        return False


def load_metadata(filepath: str) -> Optional[Dict]:
    """
    Load metadata from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary of metadata or None if file doesn't exist
    """
    try:
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error loading metadata from {filepath}: {str(e)}")
    
    return None


def get_repo_stats(repo_path: str, supported_extensions: set) -> Dict:
    """
    Generate statistics about a repository
    
    Args:
        repo_path: Path to repository
        supported_extensions: Set of supported extensions
        
    Returns:
        Dictionary with repository statistics
    """
    stats = {
        "total_files": 0,
        "total_size_bytes": 0,
        "files_by_extension": {},
        "largest_files": [],
    }
    
    try:
        files = []
        for filepath in Path(repo_path).rglob("*"):
            if filepath.is_file():
                ext = get_file_extension(str(filepath))
                if ext in supported_extensions:
                    size = filepath.stat().st_size
                    stats["total_files"] += 1
                    stats["total_size_bytes"] += size
                    stats["files_by_extension"][ext] = stats["files_by_extension"].get(ext, 0) + 1
                    files.append((str(filepath), size))
        
        # Get top 5 largest files
        files.sort(key=lambda x: x[1], reverse=True)
        stats["largest_files"] = [
            {"path": f[0], "size": format_file_size(f[1])} for f in files[:5]
        ]
        
        stats["total_size"] = format_file_size(stats["total_size_bytes"])
    
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error generating stats for {repo_path}: {str(e)}")
    
    return stats


class RepositoryMetadata:
    """
    Class to manage repository metadata
    """
    
    def __init__(self, repo_name: str, repo_url: str, local_path: str):
        self.repo_name = repo_name
        self.repo_url = repo_url
        self.local_path = local_path
        self.indexed_at = None
        self.stats = {}
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary"""
        return {
            "repo_name": self.repo_name,
            "repo_url": self.repo_url,
            "local_path": self.local_path,
            "indexed_at": str(self.indexed_at),
            "stats": self.stats,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "RepositoryMetadata":
        """Create instance from dictionary"""
        instance = cls(data["repo_name"], data["repo_url"], data["local_path"])
        instance.indexed_at = data.get("indexed_at")
        instance.stats = data.get("stats", {})
        return instance
