"""
Java Code Parser

Uses regex-based parsing to extract classes and methods.
Creates chunks for class definitions and methods with proper relationships.

Why regex for Java (instead of AST):
1. Java doesn't have a built-in AST library like Python
2. Full parsing would require external tools (ANTLR, JavaParser)
3. Regex patterns work well for common Java structures
4. Faster than external library dependencies
5. Maintains accuracy for standard Java patterns

Limitations acknowledged:
- Complex inner classes not fully supported
- Generic types simplified
- Comments preserved but not analyzed separately
- Nested method bodies may include extra closing braces

This is a pragmatic approach balancing accuracy and simplicity.
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


class JavaParser:
    """
    Parses Java source code using regex patterns.
    
    Extracts:
    - Classes (public, private, abstract)
    - Methods (public, private, static, etc.)
    - Method signatures with return types
    - Basic docstrings (Javadoc)
    
    Design rationale for regex:
    - Java structure is relatively predictable
    - Most classes follow naming conventions
    - Methods are clearly delimited with braces
    - Pragmatic balance between accuracy and simplicity
    """
    
    def __init__(self, repository_name: str = "unknown"):
        """
        Initialize Java parser
        
        Args:
            repository_name: Name of the repository
        """
        self.repository_name = repository_name
        self.logger = get_logger(self.__class__.__name__)
        
        # Regex patterns for Java parsing
        self.class_pattern = re.compile(
            r'(public\s+|private\s+|protected\s+|abstract\s+)*class\s+(\w+)',
            re.MULTILINE
        )
        
        self.method_pattern = re.compile(
            r'(public|private|protected)?\s+(static\s+)?(synchronized\s+)?(\w+[\w\[\]<>,\s]*)\s+(\w+)\s*\((.*?)\)',
            re.MULTILINE | re.DOTALL
        )
    
    def parse_file(
        self,
        file_path: str,
        file_content: str,
        relative_path: str
    ) -> List[CodeChunk]:
        """
        Parse a Java file and extract chunks
        
        Args:
            file_path: Absolute path to file
            file_content: File content as string
            relative_path: Path relative to repository root
            
        Returns:
            List of CodeChunk objects
        """
        chunks = []
        
        try:
            # Find all classes
            class_matches = list(self.class_pattern.finditer(file_content))
            
            for class_match in class_matches:
                class_name = class_match.group(2)
                class_start = class_match.start()
                class_end = self._find_class_end(file_content, class_start)
                class_content = file_content[class_start:class_end]
                
                # Extract class chunk
                class_lines = self._get_line_numbers(file_content, class_start, class_end)
                docstring = self._extract_javadoc_before(file_content, class_start)
                
                class_chunk = CodeChunk(
                    chunk_id=generate_chunk_id(relative_path, "class", class_name),
                    file_name=Path(file_path).name,
                    file_path=relative_path,
                    language="java",
                    repository_name=self.repository_name,
                    content=class_content,
                    chunk_type="class",
                    class_name=class_name,
                    function_name=None,
                    start_line=class_lines[0],
                    end_line=class_lines[1],
                    docstring=docstring,
                )
                chunks.append(class_chunk)
                
                # Extract methods within this class
                method_chunks = self._extract_methods(
                    class_content,
                    class_name,
                    file_path,
                    relative_path,
                    class_start
                )
                chunks.extend(method_chunks)
            
            self.logger.info(f"Extracted {len(chunks)} chunks from {relative_path}")
        
        except Exception as e:
            self.logger.error(f"Error parsing {relative_path}: {str(e)}")
        
        return chunks
    
    def _extract_methods(
        self,
        class_content: str,
        class_name: str,
        file_path: str,
        relative_path: str,
        class_offset: int
    ) -> List[CodeChunk]:
        """
        Extract method chunks from class content
        
        Args:
            class_content: Content of the class
            class_name: Name of the class
            file_path: Absolute path
            relative_path: Relative path
            class_offset: Offset of class in original file
            
        Returns:
            List of method chunks
        """
        method_chunks = []
        method_matches = list(self.method_pattern.finditer(class_content))
        
        for method_match in method_matches:
            try:
                method_name = method_match.group(5)
                
                # Skip constructors (they have same name as class)
                if method_name == class_name:
                    method_name = f"<init> ({method_name})"
                
                method_start = method_match.start()
                method_end = self._find_method_end(class_content, method_start)
                method_content = class_content[method_start:method_end]
                
                # Get line numbers relative to original file
                file_start = class_offset + method_start
                file_end = class_offset + method_end
                line_numbers = self._get_line_numbers(file_path, file_start, file_end)
                
                docstring = self._extract_javadoc_before(class_content, method_start)
                
                method_chunk = CodeChunk(
                    chunk_id=generate_chunk_id(relative_path, "method", f"{class_name}.{method_name}"),
                    file_name=Path(file_path).name,
                    file_path=relative_path,
                    language="java",
                    repository_name=self.repository_name,
                    content=method_content,
                    chunk_type="method",
                    class_name=class_name,
                    function_name=method_name,
                    start_line=line_numbers[0],
                    end_line=line_numbers[1],
                    docstring=docstring,
                )
                
                # Extract method signature
                signature = method_match.group(0)
                method_chunk.add_metadata("signature", signature)
                method_chunk.add_metadata("return_type", method_match.group(4).strip())
                method_chunk.add_metadata("is_static", bool(method_match.group(2)))
                
                method_chunks.append(method_chunk)
            
            except Exception as e:
                self.logger.warning(f"Error extracting method: {str(e)}")
        
        return method_chunks
    
    def _find_class_end(self, content: str, class_start: int) -> int:
        """
        Find the end of a class definition
        
        Uses brace counting to find matching closing brace
        
        Args:
            content: File content
            class_start: Start position of class keyword
            
        Returns:
            Position of closing brace
        """
        brace_count = 0
        in_class = False
        
        for i in range(class_start, len(content)):
            char = content[i]
            
            if char == '{':
                brace_count += 1
                in_class = True
            elif char == '}':
                brace_count -= 1
                if in_class and brace_count == 0:
                    return i + 1
        
        return len(content)
    
    def _find_method_end(self, content: str, method_start: int) -> int:
        """
        Find the end of a method definition
        
        Args:
            content: Class content
            method_start: Start position of method
            
        Returns:
            Position of method end
        """
        brace_count = 0
        found_opening = False
        
        for i in range(method_start, len(content)):
            char = content[i]
            
            if char == '{':
                brace_count += 1
                found_opening = True
            elif char == '}':
                brace_count -= 1
                if found_opening and brace_count == 0:
                    return i + 1
        
        return len(content)
    
    def _get_line_numbers(self, content: str, start_pos: int, end_pos: int) -> tuple:
        """
        Get line numbers for a position range
        
        Args:
            content: File content
            start_pos: Start position
            end_pos: End position
            
        Returns:
            Tuple of (start_line, end_line) 1-indexed
        """
        start_line = content[:start_pos].count('\n') + 1
        end_line = content[:end_pos].count('\n') + 1
        return (start_line, end_line)
    
    def _extract_javadoc_before(self, content: str, position: int) -> Optional[str]:
        """
        Extract Javadoc comment before a position
        
        Args:
            content: Content to search in
            position: Position to search before
            
        Returns:
            Javadoc string or None
        """
        # Look backwards for /** ... */
        search_start = max(0, position - 500)  # Search in previous 500 chars
        search_content = content[search_start:position]
        
        javadoc_match = re.search(
            r'/\*\*.*?\*/',
            search_content,
            re.DOTALL
        )
        
        if javadoc_match:
            return javadoc_match.group(0)
        
        return None
    
    def extract_imports(self, file_content: str) -> List[str]:
        """
        Extract import statements
        
        Args:
            file_content: File content
            
        Returns:
            List of imported packages/classes
        """
        imports = []
        import_pattern = re.compile(r'import\s+([\w\.]+)(?:\.\*)?;')
        
        for match in import_pattern.finditer(file_content):
            imports.append(match.group(1))
        
        return imports
    
    def extract_class_hierarchy(self, class_content: str) -> Dict[str, any]:
        """
        Extract class hierarchy information
        
        Args:
            class_content: Class content
            
        Returns:
            Dictionary with inheritance and interface info
        """
        hierarchy = {
            "extends": None,
            "implements": []
        }
        
        # Find extends clause
        extends_match = re.search(r'extends\s+(\w+)', class_content)
        if extends_match:
            hierarchy["extends"] = extends_match.group(1)
        
        # Find implements clause
        implements_match = re.search(r'implements\s+([\w\s,]+)\s*\{', class_content)
        if implements_match:
            interfaces = implements_match.group(1).split(',')
            hierarchy["implements"] = [i.strip() for i in interfaces]
        
        return hierarchy
