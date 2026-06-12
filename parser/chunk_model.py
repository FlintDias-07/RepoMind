"""
Code Chunk Data Model

Defines the structure and validation for code chunks extracted from source files.
Each chunk represents a semantic unit (class, function, method, etc.) from the codebase.

Why this model matters:
- Enables semantic search (retrieve related functions, classes, etc.)
- Preserves context (class_name helps understand method relationships)
- Tracks location (start_line, end_line for navigation)
- Standardizes across languages
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import hashlib
import json


@dataclass
class CodeChunk:
    """
    Represents a semantic unit of code (class, function, method, etc.)
    
    This is the core data model for the RAG system. Each chunk is a meaningful
    piece of code that can be embedded and searched independently.
    
    Attributes:
        chunk_id: Unique identifier for this chunk (auto-generated)
        file_name: Name of the source file (e.g., "AuthService.java")
        file_path: Relative path from repository root (e.g., "src/service/AuthService.java")
        language: Programming language (python, java, javascript, etc.)
        repository_name: Name of the repository this chunk comes from
        content: The actual source code content
        chunk_type: Type of chunk - "class", "function", "method", "module", "section"
        class_name: Parent class name (if chunk is a method or nested function)
        function_name: Function/method name (if chunk is a function or method)
        start_line: Starting line number in original file (1-indexed)
        end_line: Ending line number in original file (1-indexed)
        docstring: Documentation/docstring if available
        dependencies: List of other chunks this depends on
        metadata: Additional metadata (complexity, size, etc.)
        created_at: Timestamp when chunk was created
    
    Why each field matters for RAG:
    - chunk_id: Enable precise retrieval and updates
    - file_path: Help users find the original code
    - language: Guide LLM on code-specific context
    - class_name/function_name: Enable hierarchical search
    - docstring: Rich context for semantic search
    - dependencies: Understand code relationships
    - start_line/end_line: Direct user to location
    """
    
    chunk_id: str
    file_name: str
    file_path: str
    language: str
    repository_name: str
    content: str
    chunk_type: str  # "class", "function", "method", "module", "section"
    class_name: Optional[str] = None
    function_name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    docstring: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Validate chunk data after initialization"""
        self._validate()
    
    def _validate(self) -> None:
        """
        Validate chunk integrity
        
        Raises:
            ValueError: If chunk data is invalid
        """
        if not self.chunk_id:
            raise ValueError("chunk_id is required")
        if not self.file_path:
            raise ValueError("file_path is required")
        if not self.content or not self.content.strip():
            raise ValueError("content cannot be empty")
        if self.start_line and self.end_line and self.start_line > self.end_line:
            raise ValueError("start_line cannot be greater than end_line")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chunk to dictionary
        
        Returns:
            Dictionary representation of the chunk
        """
        return asdict(self)
    
    def to_json(self) -> str:
        """
        Convert chunk to JSON string
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeChunk":
        """
        Create chunk from dictionary
        
        Args:
            data: Dictionary with chunk data
            
        Returns:
            CodeChunk instance
        """
        return cls(**data)
    
    def get_context_key(self) -> str:
        """
        Get unique context key for this chunk
        
        Useful for deduplication and indexing
        
        Returns:
            Unique context string (file_path:class_name:function_name)
        """
        parts = [self.file_path]
        if self.class_name:
            parts.append(self.class_name)
        if self.function_name:
            parts.append(self.function_name)
        return ":".join(parts)
    
    def get_display_name(self) -> str:
        """
        Get human-readable display name for the chunk
        
        Returns:
            Display name (e.g., "AuthService.validateToken()")
        """
        if self.class_name and self.function_name:
            return f"{self.class_name}.{self.function_name}()"
        elif self.function_name:
            return f"{self.function_name}()"
        elif self.class_name:
            return f"class {self.class_name}"
        else:
            return self.file_path
    
    def get_size_info(self) -> Dict[str, int]:
        """
        Get size information about the chunk
        
        Returns:
            Dictionary with size metrics
        """
        return {
            "characters": len(self.content),
            "lines": self.end_line - self.start_line + 1 if self.start_line and self.end_line else 0,
            "words": len(self.content.split()),
        }
    
    def add_dependency(self, dep_chunk_id: str) -> None:
        """
        Add a dependency to this chunk
        
        Args:
            dep_chunk_id: ID of the chunk this depends on
        """
        if dep_chunk_id not in self.dependencies:
            self.dependencies.append(dep_chunk_id)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add or update metadata
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value


class ChunkStatistics:
    """
    Statistics about a set of code chunks
    
    Useful for reporting and quality assurance
    """
    
    def __init__(self, chunks: List[CodeChunk]):
        """
        Initialize statistics from chunks
        
        Args:
            chunks: List of CodeChunk instances
        """
        self.chunks = chunks
        self._compute_stats()
    
    def _compute_stats(self) -> None:
        """Compute statistics"""
        self.total_chunks = len(self.chunks)
        self.total_lines = sum(
            (chunk.end_line - chunk.start_line + 1) 
            for chunk in self.chunks 
            if chunk.start_line and chunk.end_line
        )
        self.total_characters = sum(len(chunk.content) for chunk in self.chunks)
        
        # Count by type
        self.by_type: Dict[str, int] = {}
        self.by_language: Dict[str, int] = {}
        
        for chunk in self.chunks:
            self.by_type[chunk.chunk_type] = self.by_type.get(chunk.chunk_type, 0) + 1
            self.by_language[chunk.language] = self.by_language.get(chunk.language, 0) + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary"""
        return {
            "total_chunks": self.total_chunks,
            "total_lines": self.total_lines,
            "total_characters": self.total_characters,
            "chunks_by_type": self.by_type,
            "chunks_by_language": self.by_language,
            "avg_chunk_size": self.total_characters // self.total_chunks if self.total_chunks > 0 else 0,
        }
    
    def __str__(self) -> str:
        """String representation"""
        stats = self.to_dict()
        lines = [
            f"Total Chunks: {stats['total_chunks']}",
            f"Total Lines: {stats['total_lines']}",
            f"Total Characters: {stats['total_characters']}",
            f"Average Chunk Size: {stats['avg_chunk_size']} chars",
            "\nChunks by Type:",
        ]
        for type_name, count in stats['chunks_by_type'].items():
            lines.append(f"  {type_name}: {count}")
        
        lines.append("\nChunks by Language:")
        for lang, count in stats['chunks_by_language'].items():
            lines.append(f"  {lang}: {count}")
        
        return "\n".join(lines)


def generate_chunk_id(
    file_path: str,
    chunk_type: str,
    name: Optional[str] = None,
    index: int = 0
) -> str:
    """
    Generate a unique chunk ID
    
    Strategy: Hash-based ID combining file path, type, and name
    
    Args:
        file_path: Path to source file
        chunk_type: Type of chunk (class, function, etc.)
        name: Name of the chunk (class/function name)
        index: Index if multiple chunks of same type
        
    Returns:
        Unique chunk ID
        
    Examples:
        >>> generate_chunk_id("src/auth.py", "function", "login", 0)
        "py_auth_f_login_0"
    """
    # Create a readable ID format: language_filename_type_name_index
    file_stem = file_path.split("/")[-1].split(".")[0][:10]  # First 10 chars of filename
    type_abbr = chunk_type[0].lower()  # First letter of type (c, f, m)
    name_abbr = (name[:10] if name else "anon").lower()  # First 10 chars of name
    
    # Also include a hash for uniqueness
    hash_input = f"{file_path}:{chunk_type}:{name}:{index}"
    hash_val = hashlib.md5(hash_input.encode()).hexdigest()[:6]
    
    return f"{file_stem}_{type_abbr}{name_abbr}_{hash_val}"
