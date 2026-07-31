"""
File Scanner Module

Recursively discovers source code files in a repository and extracts metadata.
Handles directory filtering, file validation, and metadata extraction.

Key Concepts:
- Recursive traversal: Walk through all directories
- Intelligent filtering: Skip ignored directories
- Metadata extraction: Extract file properties for context
- Language detection: Determine language from extension
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import os

from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_logger, is_ignored_directory, is_supported_file, get_file_extension
from config import SUPPORTED_EXTENSIONS, IGNORE_DIRS
from retrieval.file_classifier import FileCategory, FileClassifier

logger = get_logger(__name__)


@dataclass
class FileMetadata:
    """
    Metadata about a source file
    
    Attributes:
        file_name: Name of the file (e.g., "auth.py")
        file_path: Relative path from repo root (e.g., "src/utils/auth.py")
        absolute_path: Absolute path on filesystem
        extension: File extension (e.g., ".py")
        language: Programming language (e.g., "python")
        size_bytes: File size in bytes
        line_count: Number of lines (estimated)
    """
    file_name: str
    file_path: str
    absolute_path: str
    extension: str
    language: str
    size_bytes: int
    line_count: int = 0
    category: str = FileCategory.SOURCE_CODE


class FileScanner:
    """
    Scans a repository directory for source code files.
    
    Features:
    - Recursive directory traversal
    - Intelligent directory filtering (skip .git, node_modules, etc.)
    - File type validation
    - Metadata extraction
    - Language detection
    
    Why this design is important:
    - Efficiency: Avoids scanning unnecessary directories
    - Accuracy: Only processes supported file types
    - Context: Extracts metadata needed for proper parsing
    - Robustness: Handles missing/unreadable files gracefully
    """
    
    # Mapping of extensions to language names
    LANGUAGE_MAP = {
        ".py": "python",
        ".java": "java",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".md": "markdown",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
        ".sql": "sql",
    }
    
    def __init__(self, repository_path: str):
        """
        Initialize file scanner
        
        Args:
            repository_path: Path to the repository root
            
        Raises:
            ValueError: If repository path doesn't exist
        """
        repo_path = Path(repository_path)
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repository_path}")
        
        self.repository_path = repo_path
        self.logger = get_logger(self.__class__.__name__)
        self.files: List[FileMetadata] = []
        self.file_classifier = FileClassifier()
    
    def scan(self) -> List[FileMetadata]:
        """
        Scan repository for source files
        
        Returns:
            List of FileMetadata objects
        """
        self.files = []
        self.logger.info(f"Scanning repository: {self.repository_path}")
        
        try:
            self._traverse_directory(self.repository_path)
            self.logger.info(f"Found {len(self.files)} source files")
        except Exception as e:
            self.logger.error(f"Error scanning repository: {str(e)}")
        
        return self.files
    
    def _traverse_directory(self, directory: Path) -> None:
        """
        Recursively traverse directory and collect files
        
        Args:
            directory: Directory to traverse
        """
        try:
            for item in directory.iterdir():
                # Check if directory should be ignored
                if item.is_dir():
                    if self._should_ignore_directory(item):
                        self.logger.debug(f"Ignoring directory: {item}")
                        continue
                    
                    # Recursively traverse subdirectory
                    self._traverse_directory(item)
                
                # Check if file should be processed
                elif item.is_file():
                    if self._is_valid_file(item):
                        metadata = self._extract_metadata(item)
                        self.files.append(metadata)
        
        except PermissionError as e:
            self.logger.warning(f"Permission denied: {directory}")
        except Exception as e:
            self.logger.error(f"Error traversing {directory}: {str(e)}")
    
    def _should_ignore_directory(self, path: Path) -> bool:
        """
        Check if directory should be ignored
        
        Args:
            path: Directory path
            
        Returns:
            True if directory should be ignored
        """
        directory_name = path.name
        return directory_name in IGNORE_DIRS or is_ignored_directory(str(path), IGNORE_DIRS)
    
    def _is_valid_file(self, path: Path) -> bool:
        """
        Check if file is valid and supported
        
        Args:
            path: File path
            
        Returns:
            True if file should be processed
        """
        return is_supported_file(str(path), SUPPORTED_EXTENSIONS)
    
    def _extract_metadata(self, file_path: Path) -> FileMetadata:
        """
        Extract metadata from a file
        
        Args:
            file_path: Path to file
            
        Returns:
            FileMetadata object
        """
        try:
            extension = get_file_extension(str(file_path))
            language = self.LANGUAGE_MAP.get(extension, "unknown")
            
            # Calculate relative path
            relative_path = file_path.relative_to(self.repository_path)
            
            # Get file size
            size_bytes = file_path.stat().st_size
            
            # Estimate line count (read file)
            line_count = self._count_lines(file_path)
            
            metadata = FileMetadata(
                file_name=file_path.name,
                file_path=str(relative_path),
                absolute_path=str(file_path),
                extension=extension,
                language=language,
                size_bytes=size_bytes,
                line_count=line_count,
                category=self.file_classifier.classify(str(relative_path)).category,
            )
            
            return metadata
        
        except Exception as e:
            self.logger.error(f"Error extracting metadata for {file_path}: {str(e)}")
            # Return minimal metadata on error
            return FileMetadata(
                file_name=file_path.name,
                file_path=str(file_path),
                absolute_path=str(file_path),
                extension=get_file_extension(str(file_path)),
                language="unknown",
                size_bytes=0,
                line_count=0,
                category=self.file_classifier.classify(str(file_path)).category,
            )
    
    def _count_lines(self, file_path: Path) -> int:
        """
        Count lines in a file
        
        Args:
            file_path: Path to file
            
        Returns:
            Number of lines
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return sum(1 for _ in f)
        except Exception as e:
            self.logger.warning(f"Error counting lines in {file_path}: {str(e)}")
            return 0
    
    def get_files_by_language(self) -> Dict[str, List[FileMetadata]]:
        """
        Group files by language
        
        Returns:
            Dictionary mapping language to list of files
        """
        by_language: Dict[str, List[FileMetadata]] = {}
        for file_meta in self.files:
            if file_meta.language not in by_language:
                by_language[file_meta.language] = []
            by_language[file_meta.language].append(file_meta)
        
        return by_language
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about scanned files
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_files": len(self.files),
            "total_lines": sum(f.line_count for f in self.files),
            "total_size_bytes": sum(f.size_bytes for f in self.files),
            "files_by_language": {},
            "files_by_extension": {},
        }
        
        for file_meta in self.files:
            # Count by language
            lang = file_meta.language
            if lang not in stats["files_by_language"]:
                stats["files_by_language"][lang] = 0
            stats["files_by_language"][lang] += 1
            
            # Count by extension
            ext = file_meta.extension
            if ext not in stats["files_by_extension"]:
                stats["files_by_extension"][ext] = 0
            stats["files_by_extension"][ext] += 1
        
        return stats
    
    def filter_by_language(self, language: str) -> List[FileMetadata]:
        """
        Get files for a specific language
        
        Args:
            language: Language name (e.g., "python", "java")
            
        Returns:
            List of files for that language
        """
        return [f for f in self.files if f.language == language]
    
    def filter_by_extension(self, extension: str) -> List[FileMetadata]:
        """
        Get files with a specific extension
        
        Args:
            extension: File extension (e.g., ".py")
            
        Returns:
            List of files with that extension
        """
        return [f for f in self.files if f.extension == extension]


class DirectoryAnalyzer:
    """
    Analyzes a directory structure and provides insights
    
    Useful for understanding repository layout
    """
    
    def __init__(self, repository_path: str):
        """Initialize analyzer"""
        self.repository_path = Path(repository_path)
        self.logger = get_logger(self.__class__.__name__)
    
    def get_directory_structure(self, max_depth: int = 3) -> Dict:
        """
        Get directory structure
        
        Args:
            max_depth: Maximum depth to traverse
            
        Returns:
            Dictionary representing directory tree
        """
        return self._build_tree(self.repository_path, 0, max_depth)
    
    def _build_tree(self, directory: Path, current_depth: int, max_depth: int) -> Dict:
        """Build directory tree recursively"""
        if current_depth >= max_depth:
            return {"name": directory.name, "type": "dir", "children": []}
        
        result = {
            "name": directory.name,
            "type": "dir",
            "children": []
        }
        
        try:
            for item in sorted(directory.iterdir()):
                # Skip ignored directories
                if item.is_dir() and item.name in IGNORE_DIRS:
                    continue
                
                if item.is_dir():
                    result["children"].append(
                        self._build_tree(item, current_depth + 1, max_depth)
                    )
                else:
                    result["children"].append({
                        "name": item.name,
                        "type": "file",
                        "size": item.stat().st_size
                    })
        
        except PermissionError:
            pass
        except Exception as e:
            self.logger.error(f"Error building tree: {str(e)}")
        
        return result
    
    def find_main_directories(self) -> List[str]:
        """
        Find main source directories (src, lib, app, etc.)
        
        Returns:
            List of common source directory names found
        """
        common_names = {"src", "lib", "app", "source", "code", "main"}
        found = []
        
        try:
            for item in self.repository_path.iterdir():
                if item.is_dir() and item.name in common_names:
                    found.append(item.name)
        except Exception as e:
            self.logger.error(f"Error finding main directories: {str(e)}")
        
        return found
