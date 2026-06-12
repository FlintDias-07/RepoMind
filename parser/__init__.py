"""
Parser Package

Modular parsing system for multiple programming languages.

Modules:
- chunk_model: Data structures for code chunks
- file_scanner: File discovery and metadata
- python_parser: Python code parsing
- java_parser: Java code parsing
- js_parser: JavaScript/TypeScript parsing
- markdown_parser: Markdown parsing
- yaml_parser: YAML parsing
- parser_manager: Orchestrates all parsers

Usage:
    from parser.parser_manager import ParserManager
    
    manager = ParserManager("/path/to/repo", "repo_name")
    chunks, stats = manager.parse_repository()
    
    for chunk in chunks:
        print(chunk.get_display_name())
"""

from .chunk_model import CodeChunk, ChunkStatistics, generate_chunk_id
from .file_scanner import FileScanner, FileMetadata
from .python_parser import PythonParser
from .java_parser import JavaParser
from .js_parser import JavaScriptParser
from .markdown_parser import MarkdownParser
from .yaml_parser import YAMLParser
from .parser_manager import ParserManager, ChunkAggregator

__all__ = [
    # Models
    "CodeChunk",
    "ChunkStatistics",
    "generate_chunk_id",
    "FileMetadata",
    # Scanners
    "FileScanner",
    # Parsers
    "PythonParser",
    "JavaParser",
    "JavaScriptParser",
    "MarkdownParser",
    "YAMLParser",
    # Manager
    "ParserManager",
    "ChunkAggregator",
]

__version__ = "1.0.0"
