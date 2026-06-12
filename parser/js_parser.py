"""
JavaScript/TypeScript Code Parser

Uses regex-based parsing to extract functions, classes, and exports.
Handles both JavaScript and TypeScript syntax.

Features:
- Extract classes and methods
- Extract functions (arrow and regular)
- Extract exports
- Parse async/await
- Handle TypeScript types
- Support for modern ES6+ syntax

Design approach:
- Regex for reliable pattern matching
- Handles most common patterns
- Graceful fallback for complex cases
"""

import re
from typing import List, Optional, Dict
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.chunk_model import CodeChunk, generate_chunk_id
from utils import get_logger

logger = get_logger(__name__)


class JavaScriptParser:
    """
    Parses JavaScript/TypeScript source code.
    
    Extracts:
    - Classes and methods
    - Functions (arrow functions and regular)
    - Exports (default and named)
    - Async functions
    
    Supports:
    - ES6 class syntax
    - Arrow functions
    - Async/await
    - TypeScript type annotations
    - Comments and JSDoc
    """
    
    def __init__(self, repository_name: str = "unknown", language: str = "javascript"):
        """
        Initialize JavaScript/TypeScript parser
        
        Args:
            repository_name: Name of repository
            language: "javascript", "typescript", or detect from extension
        """
        self.repository_name = repository_name
        self.language = language
        self.logger = get_logger(self.__class__.__name__)
        
        # Regex patterns
        self.class_pattern = re.compile(
            r'(?:export\s+)?(class|interface)\s+(\w+)',
            re.MULTILINE
        )
        
        self.function_pattern = re.compile(
            r'(?:export\s+)?(?:async\s+)?(?:function|const|let|var)\s+(\w+)\s*(?::\s*\w+)?\s*=\s*(?:async\s+)?\(',
            re.MULTILINE
        )
        
        self.method_pattern = re.compile(
            r'^\s*(?:async\s+)?(?:get|set)?\s*(\w+)\s*\(.*?\)\s*(?::\s*[\w<>,\s|\(\)]+)?\s*\{',
            re.MULTILINE
        )
    
    def parse_file(
        self,
        file_path: str,
        file_content: str,
        relative_path: str
    ) -> List[CodeChunk]:
        """
        Parse a JavaScript/TypeScript file
        
        Args:
            file_path: Absolute path to file
            file_content: File content
            relative_path: Relative path
            
        Returns:
            List of CodeChunk objects
        """
        chunks = []
        
        try:
            # Extract classes
            class_matches = list(self.class_pattern.finditer(file_content))
            
            for class_match in class_matches:
                class_type = class_match.group(1)
                class_name = class_match.group(2)
                class_start = class_match.start()
                class_end = self._find_block_end(file_content, class_start)
                class_content = file_content[class_start:class_end]
                
                class_lines = self._get_line_numbers(file_content, class_start, class_end)
                docstring = self._extract_jsdoc_before(file_content, class_start)
                
                class_chunk = CodeChunk(
                    chunk_id=generate_chunk_id(relative_path, "class", class_name),
                    file_name=Path(file_path).name,
                    file_path=relative_path,
                    language=self.language,
                    repository_name=self.repository_name,
                    content=class_content,
                    chunk_type="class",
                    class_name=class_name,
                    function_name=None,
                    start_line=class_lines[0],
                    end_line=class_lines[1],
                    docstring=docstring,
                )
                
                chunk_metadata = self._extract_class_metadata(class_content)
                for key, value in chunk_metadata.items():
                    class_chunk.add_metadata(key, value)
                
                chunks.append(class_chunk)
                
                # Extract methods from class
                method_chunks = self._extract_methods_from_class(
                    class_content,
                    class_name,
                    file_path,
                    relative_path,
                    class_start
                )
                chunks.extend(method_chunks)
            
            # Extract standalone functions
            function_chunks = self._extract_functions(
                file_content,
                file_path,
                relative_path
            )
            chunks.extend(function_chunks)
            
            self.logger.info(f"Extracted {len(chunks)} chunks from {relative_path}")
        
        except Exception as e:
            self.logger.error(f"Error parsing {relative_path}: {str(e)}")
        
        return chunks
    
    def _extract_functions(
        self,
        file_content: str,
        file_path: str,
        relative_path: str
    ) -> List[CodeChunk]:
        """
        Extract standalone functions
        
        Args:
            file_content: File content
            file_path: Absolute path
            relative_path: Relative path
            
        Returns:
            List of function chunks
        """
        chunks = []
        
        # Find standalone functions (not inside classes)
        func_matches = list(self.function_pattern.finditer(file_content))
        
        for func_match in func_matches:
            func_name = func_match.group(1)
            func_start = func_match.start()
            
            # Skip if inside a class (simple heuristic)
            if self._is_inside_class(file_content, func_start):
                continue
            
            # Find function end
            func_end = self._find_block_end(file_content, func_start)
            func_content = file_content[func_start:func_end]
            
            func_lines = self._get_line_numbers(file_content, func_start, func_end)
            docstring = self._extract_jsdoc_before(file_content, func_start)
            
            is_async = 'async' in file_content[max(0, func_start - 20):func_start]
            is_export = 'export' in file_content[max(0, func_start - 20):func_start]
            
            func_chunk = CodeChunk(
                chunk_id=generate_chunk_id(relative_path, "function", func_name),
                file_name=Path(file_path).name,
                file_path=relative_path,
                language=self.language,
                repository_name=self.repository_name,
                content=func_content,
                chunk_type="function",
                class_name=None,
                function_name=func_name,
                start_line=func_lines[0],
                end_line=func_lines[1],
                docstring=docstring,
            )
            
            func_chunk.add_metadata("is_async", is_async)
            func_chunk.add_metadata("is_export", is_export)
            
            chunks.append(func_chunk)
        
        return chunks
    
    def _extract_methods_from_class(
        self,
        class_content: str,
        class_name: str,
        file_path: str,
        relative_path: str,
        class_offset: int
    ) -> List[CodeChunk]:
        """
        Extract methods from class
        
        Args:
            class_content: Class content
            class_name: Class name
            file_path: Absolute path
            relative_path: Relative path
            class_offset: Offset in file
            
        Returns:
            List of method chunks
        """
        method_chunks = []
        
        # Find methods in class body
        # Skip the class declaration line
        class_body_start = class_content.find('{') + 1
        if class_body_start == 0:
            return method_chunks
        
        class_body = class_content[class_body_start:]
        
        # Find all method-like patterns
        for match in re.finditer(r'(\w+)\s*\([^)]*\)\s*(?::\s*[\w<>,\s|\(\)]+)?\s*\{', class_body):
            method_name = match.group(1)
            
            # Skip common non-method patterns
            if method_name in ['if', 'for', 'while', 'switch', 'try', 'catch']:
                continue
            
            method_start_in_body = match.start()
            method_start = class_offset + class_body_start + method_start_in_body
            method_end = class_offset + class_body_start + self._find_block_end(
                class_body,
                method_start_in_body
            )
            
            method_content = file_content[method_start:method_end]
            method_lines = self._get_line_numbers(file_path, method_start, method_end)
            
            method_chunk = CodeChunk(
                chunk_id=generate_chunk_id(relative_path, "method", f"{class_name}.{method_name}"),
                file_name=Path(file_path).name,
                file_path=relative_path,
                language=self.language,
                repository_name=self.repository_name,
                content=method_content,
                chunk_type="method",
                class_name=class_name,
                function_name=method_name,
                start_line=method_lines[0],
                end_line=method_lines[1],
            )
            
            method_chunks.append(method_chunk)
        
        return method_chunks
    
    def _find_block_end(self, content: str, start_pos: int) -> int:
        """
        Find the end of a block (opening brace to closing brace)
        
        Args:
            content: Content
            start_pos: Start position
            
        Returns:
            End position
        """
        brace_count = 0
        found_opening = False
        in_string = False
        string_char = None
        
        for i in range(start_pos, len(content)):
            char = content[i]
            
            # Handle strings
            if char in ['"', "'", '`'] and (i == 0 or content[i - 1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
            
            if in_string:
                continue
            
            if char == '{':
                brace_count += 1
                found_opening = True
            elif char == '}':
                brace_count -= 1
                if found_opening and brace_count == 0:
                    return i + 1
        
        return len(content)
    
    def _get_line_numbers(self, content: str, start_pos: int, end_pos: int) -> tuple:
        """Get line numbers for position range"""
        start_line = content[:start_pos].count('\n') + 1
        end_line = content[:end_pos].count('\n') + 1
        return (start_line, end_line)
    
    def _extract_jsdoc_before(self, content: str, position: int) -> Optional[str]:
        """Extract JSDoc comment before position"""
        search_start = max(0, position - 500)
        search_content = content[search_start:position]
        
        jsdoc_match = re.search(
            r'/\*\*.*?\*/',
            search_content,
            re.DOTALL
        )
        
        if jsdoc_match:
            return jsdoc_match.group(0)
        
        return None
    
    def _is_inside_class(self, content: str, position: int) -> bool:
        """Check if position is inside a class definition"""
        # Count opening and closing braces before position
        before = content[:position]
        
        # Find last class keyword
        last_class = before.rfind('class ')
        if last_class == -1:
            return False
        
        # Count braces between class and position
        section = before[last_class:]
        opening = section.count('{')
        closing = section.count('}')
        
        return opening > closing
    
    def _extract_class_metadata(self, class_content: str) -> Dict[str, any]:
        """Extract metadata from class"""
        metadata = {
            "has_constructor": "constructor(" in class_content,
            "is_abstract": "abstract" in class_content,
        }
        
        return metadata
    
    def extract_exports(self, file_content: str) -> List[str]:
        """
        Extract exported symbols
        
        Args:
            file_content: File content
            
        Returns:
            List of exported names
        """
        exports = []
        
        # Named exports
        export_pattern = re.compile(r'export\s+(?:async\s+)?(?:class|function|const|let|var)\s+(\w+)')
        for match in export_pattern.finditer(file_content):
            exports.append(match.group(1))
        
        # Default export
        if 'export default' in file_content:
            exports.append('default')
        
        return exports
