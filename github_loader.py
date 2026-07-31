"""
GitHub Repository Loader
Handles cloning, validation, and management of GitHub repositories
"""

import os
import subprocess
from pathlib import Path
from typing import Tuple, Optional, List
from datetime import datetime
import json

from git import Repo, GitCommandError
from config import REPOSITORIES_DIR, SUPPORTED_EXTENSIONS, IGNORE_DIRS
from utils import (
    get_logger,
    sanitize_repo_name,
    validate_github_url,
    safe_delete_directory,
    get_repo_stats,
    save_metadata,
    RepositoryMetadata,
    load_metadata,
)

logger = get_logger(__name__)


class GitHubRepositoryLoader:
    """
    Handles cloning and management of GitHub repositories
    """
    
    def __init__(self, repo_url: str, base_path: str = str(REPOSITORIES_DIR)):
        """
        Initialize repository loader
        
        Args:
            repo_url: GitHub repository URL
            base_path: Base directory to clone repositories
            
        Raises:
            ValueError: If URL is not valid
        """
        if not validate_github_url(repo_url):
            raise ValueError(f"Invalid GitHub URL: {repo_url}")
        
        self.repo_url = repo_url
        self.base_path = Path(base_path)
        self.repo_name = sanitize_repo_name(repo_url)
        self.local_path = self.base_path / self.repo_name
        self.repo = None
        self.logger = get_logger(self.__class__.__name__)
    
    def clone_repository(self, overwrite: bool = False) -> Tuple[bool, str]:
        """
        Clone GitHub repository to local path
        
        Args:
            overwrite: Whether to overwrite existing repository
            
        Returns:
            Tuple of (success: bool, message: str)
            
        Examples:
            >>> loader = GitHubRepositoryLoader("https://github.com/user/repo.git")
            >>> success, message = loader.clone_repository()
            >>> print(message)
            "Repository cloned successfully to /path/to/repo"
        """
        try:
            # Check if repository already exists
            if self.local_path.exists():
                if not overwrite:
                    msg = f"Repository already exists at {self.local_path}. Use overwrite=True to replace."
                    self.logger.warning(msg)
                    return False, msg
                else:
                    self.logger.info(f"Removing existing repository at {self.local_path}")
                    safe_delete_directory(str(self.local_path))
            
            # Create base directory if it doesn't exist
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Clone repository
            self.logger.info(f"Cloning repository from {self.repo_url}")
            self.repo = Repo.clone_from(self.repo_url, str(self.local_path))
            
            msg = f"Repository cloned successfully to {self.local_path}"
            self.logger.info(msg)
            return True, msg
        
        except GitCommandError as e:
            msg = f"Git error while cloning: {str(e)}"
            self.logger.error(msg)
            return False, msg
        except Exception as e:
            msg = f"Error cloning repository: {str(e)}"
            self.logger.error(msg)
            return False, msg
    
    def verify_repository(self) -> Tuple[bool, str]:
        """
        Verify that repository exists and is valid
        
        Returns:
            Tuple of (valid: bool, message: str)
        """
        try:
            if not self.local_path.exists():
                msg = f"Repository path does not exist: {self.local_path}"
                self.logger.warning(msg)
                return False, msg
            
            # Try to open repository
            self.repo = Repo(str(self.local_path))
            
            # Check if it's a valid Git repository
            if self.repo.git_dir:
                msg = f"Repository verified at {self.local_path}"
                self.logger.info(msg)
                return True, msg
        
        except Exception as e:
            msg = f"Invalid repository: {str(e)}"
            self.logger.error(msg)
            return False, msg
    
    def get_repository_info(self) -> Optional[dict]:
        """
        Get information about the cloned repository
        
        Returns:
            Dictionary with repository information or None if not valid
        """
        try:
            if not self.verify_repository()[0]:
                return None
            
            info = {
                "name": self.repo_name,
                "url": self.repo_url,
                "local_path": str(self.local_path),
                "branch": self.repo.active_branch.name,
                "commit": self.repo.head.commit.hexsha[:7],
                "last_modified": datetime.fromtimestamp(self.repo.head.commit.committed_date).isoformat(),
                "remote_url": self.repo.remotes.origin.url if self.repo.remotes else None,
            }
            
            self.logger.info(f"Repository info retrieved: {info['name']}")
            return info
        
        except Exception as e:
            self.logger.error(f"Error getting repository info: {str(e)}")
            return None
    
    def get_file_list(self, relative: bool = True) -> List[str]:
        """
        Get list of supported code files in repository
        
        Args:
            relative: If True, return relative paths from repo root
            
        Returns:
            List of file paths
        """
        files = []
        try:
            for filepath in self.local_path.rglob("*"):
                if filepath.is_file():
                    # Skip ignored directories
                    if self._is_ignored(filepath):
                        continue
                    
                    # Check if file extension is supported
                    if filepath.suffix in SUPPORTED_EXTENSIONS:
                        if relative:
                            files.append(str(filepath.relative_to(self.local_path)))
                        else:
                            files.append(str(filepath))
            
            self.logger.info(f"Found {len(files)} supported files in repository")
            return sorted(files)
        
        except Exception as e:
            self.logger.error(f"Error getting file list: {str(e)}")
            return []
    
    def read_file(self, relative_path: str) -> Optional[str]:
        """
        Read content of a file in the repository
        
        Args:
            relative_path: Path relative to repository root
            
        Returns:
            File content or None if file doesn't exist
        """
        try:
            filepath = self.local_path / relative_path
            
            if not filepath.exists():
                self.logger.warning(f"File not found: {relative_path}")
                return None
            
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            return content
        
        except Exception as e:
            self.logger.error(f"Error reading file {relative_path}: {str(e)}")
            return None
    
    def get_repository_stats(self) -> dict:
        """
        Get statistics about the repository
        
        Returns:
            Dictionary with repository statistics
        """
        try:
            stats = get_repo_stats(str(self.local_path), SUPPORTED_EXTENSIONS)
            stats["repo_name"] = self.repo_name
            stats["repo_url"] = self.repo_url
            self.logger.info(f"Generated stats for {self.repo_name}")
            return stats
        
        except Exception as e:
            self.logger.error(f"Error generating repository stats: {str(e)}")
            return {}
    
    def save_repository_metadata(self) -> bool:
        """
        Save repository metadata to JSON file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            metadata = RepositoryMetadata(
                self.repo_name,
                self.repo_url,
                str(self.local_path)
            )
            metadata.indexed_at = datetime.now().isoformat()
            metadata.stats = self.get_repository_stats()
            
            metadata_path = self.local_path / ".rag_metadata.json"
            return save_metadata(metadata.to_dict(), str(metadata_path))
        
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")
            return False
    
    def _is_ignored(self, filepath: Path) -> bool:
        """
        Check if file path should be ignored
        
        Args:
            filepath: Path to check
            
        Returns:
            True if should be ignored, False otherwise
        """
        # Check if any part of the path matches ignore list
        for part in filepath.parts:
            if part in IGNORE_DIRS:
                return True
        return False
    
    def cleanup(self) -> bool:
        """
        Delete the cloned repository
        
        Returns:
            True if successful, False otherwise
        """
        return safe_delete_directory(str(self.local_path))


class RepositoryManager:
    """
    Manages multiple repositories
    """
    
    def __init__(self, base_path: str = str(REPOSITORIES_DIR)):
        self.base_path = Path(base_path)
        self.logger = get_logger(self.__class__.__name__)
    
    def get_cloned_repositories(self) -> List[dict]:
        """
        Get list of all cloned repositories
        
        Returns:
            List of repository metadata dictionaries
        """
        repos = []
        try:
            for repo_dir in self.base_path.iterdir():
                if repo_dir.is_dir() and (repo_dir / ".git").exists():
                    metadata_file = repo_dir / ".rag_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, "r") as f:
                            repos.append(json.load(f))
        
        except Exception as e:
            self.logger.error(f"Error getting repositories: {str(e)}")
        
        return repos

    def get_unmanaged_git_repositories(self) -> List[dict]:
        """Return valid Git folders that are missing application metadata."""

        unmanaged = []
        try:
            for repo_dir in self.base_path.iterdir():
                if not repo_dir.is_dir() or not (repo_dir / ".git").exists():
                    continue
                if (repo_dir / ".rag_metadata.json").exists():
                    continue
                try:
                    repo = Repo(str(repo_dir))
                    unmanaged.append(
                        {
                            "repo_name": repo_dir.name,
                            "local_path": str(repo_dir),
                            "repo_url": repo.remotes.origin.url if repo.remotes else "",
                        }
                    )
                except Exception as exc:
                    self.logger.warning("Skipping invalid unmanaged repository %s: %s", repo_dir, exc)
        except Exception as exc:
            self.logger.error("Error discovering unmanaged repositories: %s", exc)
        return unmanaged

    def recover_repository_metadata(self, repo_name: str) -> Tuple[bool, str]:
        """Create application metadata for a valid cloned repository."""

        repo_path = self.base_path / repo_name
        metadata_path = repo_path / ".rag_metadata.json"
        if metadata_path.exists():
            return True, f"{repo_name} is already managed."
        if not (repo_path / ".git").exists():
            return False, f"{repo_name} is not a valid Git repository."

        try:
            repo = Repo(str(repo_path))
            repo_url = repo.remotes.origin.url if repo.remotes else ""
            metadata = RepositoryMetadata(repo_name, repo_url, str(repo_path))
            metadata.indexed_at = None
            metadata.stats = get_repo_stats(str(repo_path), SUPPORTED_EXTENSIONS)
            if save_metadata(metadata.to_dict(), str(metadata_path)):
                return True, f"Recovered metadata for {repo_name}."
            return False, f"Failed to save metadata for {repo_name}."
        except Exception as exc:
            self.logger.error("Error recovering repository metadata: %s", exc)
            return False, f"Failed to recover metadata: {exc}"
    
    def delete_repository(self, repo_name: str) -> Tuple[bool, str]:
        """
        Delete a cloned repository
        
        Args:
            repo_name: Name of repository to delete
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            repo_path = self.base_path / repo_name
            if repo_path.exists():
                if safe_delete_directory(str(repo_path)):
                    msg = f"Repository {repo_name} deleted successfully"
                    self.logger.info(msg)
                    return True, msg
                else:
                    msg = f"Failed to delete repository {repo_name}"
                    self.logger.error(msg)
                    return False, msg
            else:
                msg = f"Repository {repo_name} not found"
                self.logger.warning(msg)
                return False, msg
        
        except Exception as e:
            msg = f"Error deleting repository: {str(e)}"
            self.logger.error(msg)
            return False, msg
