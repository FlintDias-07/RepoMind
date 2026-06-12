"""
Parser Manager

Orchestrates all language-specific parsers and provides unified interface.
Handles file discovery, language detection, and chunk aggregation.

Architecture:
1. FileScanner: Discovers all source files
2. Language-specific parsers: Extract chunks
3. ParserManager: Coordinates everything
4. ChunkAggregator: Collects and validates chunks
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.file_scanner import FileScanner, FileMetadata
from parser.chunk_model import CodeChunk, ChunkStatistics, generate_chunk_id
from parser.python_parser import PythonParser
from parser.java_parser import JavaParser
from parser.js_parser import JavaScriptParser
from parser.markdown_parser import MarkdownParser
from parser.yaml_parser import YAMLParser

from utils import get_logger

logger = get_logger(__name__)


class ParserManager:
    """
    Orchestrates code parsing across multiple languages.
    
    Features:
    - Discovers all supported files
    - Routes files to appropriate parsers
    - Aggregates chunks from all parsers
    - Generates statistics and reports
    - Deduplicates chunks
    - Handles errors gracefully
    
    Design rationale:
    - Modular: Each language has dedicated parser
    - Scalable: Easy to add new languages
    - Robust: Graceful fallback on parse errors
    - Observable: Detailed logging and statistics
    """
    
    # Map of language to parser class
    PARSER_MAP = {
        "python": PythonParser,
        "java": JavaParser,
        "javascript": JavaScriptParser,
        "typescript": JavaScriptParser,  # Same parser handles both
        "markdown": MarkdownParser,
        "yaml": YAMLParser,
    }
    
    def __init__(self, repository_path: str, repository_name: str = "unknown"):
        """
        Initialize parser manager
        
        Args:
            repository_path: Path to repository root
            repository_name: Name of repository
            
        Raises:
            ValueError: If repository path doesn't exist
        """
        self.repository_path = repository_path
        self.repository_name = repository_name
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize file scanner
        self.file_scanner = FileScanner(repository_path)
        
        # Initialize parsers
        self.parsers = {
            lang: parser_class(repository_name)
            for lang, parser_class in self.PARSER_MAP.items()
        }
        
        # Storage for results
        self.chunks: List[CodeChunk] = []
        self.statistics: Optional[ChunkStatistics] = None
        self.parse_report: Dict = {}
    
    def parse_repository(self) -> Tuple[List[CodeChunk], Dict]:
        """
        Parse entire repository and extract chunks
        
        Returns:
            Tuple of (chunks: List[CodeChunk], statistics: Dict)
            
        Example:
            >>> manager = ParserManager("/path/to/repo", "my_repo")
            >>> chunks, stats = manager.parse_repository()
            >>> print(f"Found {len(chunks)} chunks")
            >>> print(f"Python: {stats['chunks_by_language'].get('python', 0)} chunks")
        """
        self.logger.info(f"Starting repository parsing for {self.repository_name}")
        start_time = datetime.now()
        
        try:
            # Step 1: Scan files
            self.logger.info("Step 1: Scanning files...")
            files = self.file_scanner.scan()
            self.logger.info(f"Found {len(files)} source files")
            
            # Step 2: Parse files by language
            self.logger.info("Step 2: Parsing files...")
            self.chunks = []
            self._parse_files(files)
            
            # Step 3: Post-processing
            self.logger.info("Step 3: Post-processing chunks...")
            self._post_process_chunks()
            
            # Step 4: Generate statistics
            self.logger.info("Step 4: Generating statistics...")
            self.statistics = ChunkStatistics(self.chunks)
            
            # Step 5: Generate report
            self.parse_report = self._generate_report(start_time)
            
            self.logger.info(f"Parsing complete. Extracted {len(self.chunks)} chunks")
            
            return self.chunks, self.parse_report
        
        except Exception as e:
            self.logger.error(f"Error parsing repository: {str(e)}", exc_info=True)
            return [], {"error": str(e)}
    
    def _parse_files(self, files: List[FileMetadata]) -> None:
        """
        Parse all files by language
        
        Args:
            files: List of files to parse
        """
        files_by_language = {}
        for file_meta in files:
            if file_meta.language not in files_by_language:
                files_by_language[file_meta.language] = []
            files_by_language[file_meta.language].append(file_meta)
        
        # Parse each language
        for language, language_files in files_by_language.items():
            self.logger.info(f"Parsing {len(language_files)} {language} files...")
            
            if language not in self.PARSER_MAP:
                self.logger.warning(f"No parser for language: {language}")
                continue
            
            parser = self.parsers[language]
            
            for file_meta in language_files:
                try:
                    # Read file
                    with open(file_meta.absolute_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_content = f.read()
                    
                    # Parse file
                    file_chunks = parser.parse_file(
                        file_meta.absolute_path,
                        file_content,
                        file_meta.file_path
                    )
                    
                    self.chunks.extend(file_chunks)
                    self.logger.debug(f"Parsed {file_meta.file_path}: {len(file_chunks)} chunks")
                
                except Exception as e:
                    self.logger.error(f"Error parsing {file_meta.file_path}: {str(e)}")
    
    def _post_process_chunks(self) -> None:
        """
        Post-process chunks for quality assurance
        
        Tasks:
        - Validate chunk integrity
        - Remove duplicates
        - Add cross-references
        """
        # Remove empty chunks
        self.chunks = [c for c in self.chunks if c.content.strip()]
        
        # Deduplicate (optional)
        seen = set()
        unique_chunks = []
        
        for chunk in self.chunks:
            context_key = chunk.get_context_key()
            if context_key not in seen:
                seen.add(context_key)
                unique_chunks.append(chunk)
        
        self.chunks = unique_chunks
        self.logger.info(f"After deduplication: {len(self.chunks)} chunks")
    
    def _generate_report(self, start_time: datetime) -> Dict:
        """
        Generate parsing report
        
        Args:
            start_time: When parsing started
            
        Returns:
            Dictionary with detailed report
        """
        elapsed = (datetime.now() - start_time).total_seconds()
        
        stats_dict = self.statistics.to_dict() if self.statistics else {}
        
        report = {
            "repository_name": self.repository_name,
            "repository_path": self.repository_path,
            "parsing_time_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
            **stats_dict,
            "files_processed": len(self.file_scanner.files),
            "parse_errors": 0,  # Track errors
        }
        
        return report
    
    def get_chunks_by_language(self, language: str) -> List[CodeChunk]:
        """
        Get chunks for a specific language
        
        Args:
            language: Language name
            
        Returns:
            List of chunks
        """
        return [c for c in self.chunks if c.language == language]
    
    def get_chunks_by_class(self, class_name: str) -> List[CodeChunk]:
        """
        Get all chunks related to a class
        
        Args:
            class_name: Class name
            
        Returns:
            List of chunks (class definition and methods)
        """
        return [c for c in self.chunks if c.class_name == class_name]
    
    def get_chunks_by_file(self, file_path: str) -> List[CodeChunk]:
        """
        Get all chunks from a file
        
        Args:
            file_path: Relative file path
            
        Returns:
            List of chunks
        """
        return [c for c in self.chunks if c.file_path == file_path]
    
    def search_chunks(self, keyword: str) -> List[CodeChunk]:
        """
        Search chunks by keyword
        
        Args:
            keyword: Keyword to search for
            
        Returns:
            List of matching chunks
        """
        keyword_lower = keyword.lower()
        matching = []
        
        for chunk in self.chunks:
            if (keyword_lower in chunk.content.lower() or
                (chunk.function_name and keyword_lower in chunk.function_name.lower()) or
                (chunk.class_name and keyword_lower in chunk.class_name.lower())):
                matching.append(chunk)
        
        return matching
    
    def save_chunks(self, output_path: str) -> bool:
        """
        Save chunks to JSON file
        
        Args:
            output_path: Path to save JSON
            
        Returns:
            True if successful
        """
        try:
            import json
            
            chunks_data = [c.to_dict() for c in self.chunks]
            
            with open(output_path, 'w') as f:
                json.dump({
                    "metadata": self.parse_report,
                    "chunks": chunks_data,
                }, f, indent=2, default=str)
            
            self.logger.info(f"Saved {len(self.chunks)} chunks to {output_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving chunks: {str(e)}")
            return False
    
    def print_report(self) -> None:
        """Print parsing report to console"""
        if not self.parse_report:
            print("No parsing report available")
            return
        
        print("\n" + "="*60)
        print("PARSING REPORT")
        print("="*60)
        print(f"Repository: {self.parse_report.get('repository_name')}")
        print(f"Total Chunks: {self.parse_report.get('total_chunks')}")
        print(f"Total Lines: {self.parse_report.get('total_lines')}")
        print(f"Parsing Time: {self.parse_report.get('parsing_time_seconds'):.2f}s")
        print()
        print("Chunks by Language:")
        for lang, count in self.parse_report.get('chunks_by_language', {}).items():
            print(f"  {lang}: {count}")
        print()
        print("Chunks by Type:")
        for chunk_type, count in self.parse_report.get('chunks_by_type', {}).items():
            print(f"  {chunk_type}: {count}")
        print("="*60 + "\n")


class ChunkAggregator:
    """
    Aggregates and organizes chunks from parsing
    
    Useful for:
    - Organizing chunks by various criteria
    - Building chunk indexes
    - Finding related chunks
    - Exporting chunk data
    """
    
    def __init__(self, chunks: List[CodeChunk]):
        """
        Initialize aggregator
        
        Args:
            chunks: List of chunks to aggregate
        """
        self.chunks = chunks
        self.logger = get_logger(self.__class__.__name__)
    
    def build_class_index(self) -> Dict[str, List[CodeChunk]]:
        """
        Build index of chunks by class
        
        Returns:
            Dictionary mapping class name to chunks
        """
        index = {}
        for chunk in self.chunks:
            if chunk.class_name:
                if chunk.class_name not in index:
                    index[chunk.class_name] = []
                index[chunk.class_name].append(chunk)
        
        return index
    
    def build_file_index(self) -> Dict[str, List[CodeChunk]]:
        """
        Build index of chunks by file
        
        Returns:
            Dictionary mapping file path to chunks
        """
        index = {}
        for chunk in self.chunks:
            if chunk.file_path not in index:
                index[chunk.file_path] = []
            index[chunk.file_path].append(chunk)
        
        return index
    
    def get_related_chunks(self, chunk: CodeChunk, include_same_file: bool = True) -> List[CodeChunk]:
        """
        Find chunks related to a given chunk
        
        Args:
            chunk: Reference chunk
            include_same_file: Whether to include other chunks from same file
            
        Returns:
            List of related chunks
        """
        related = []
        
        for other in self.chunks:
            if other.chunk_id == chunk.chunk_id:
                continue
            
            # Same class
            if chunk.class_name and chunk.class_name == other.class_name:
                related.append(other)
            # Same file
            elif include_same_file and chunk.file_path == other.file_path:
                related.append(other)
        
        return related
