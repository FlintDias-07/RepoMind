"""
Python Code Parser

Uses Abstract Syntax Tree (AST) for intelligent code parsing.
Extracts classes, functions, methods, and their documentation.

Why AST-based parsing is better than regex:
1. Understands code structure (nested functions, decorators, etc.)
2. Handles complex syntax (f-strings, type hints, etc.)
3. Preserves accurate line numbers
4. Can extract docstrings reliably
5. Detects actual definitions vs. string references
6. Supports all Python syntax variations

Why AST > regex for RAG:
- Regex cannot understand nesting (def inside class vs. at module level)
- Regex misses decorators, type hints, complex signatures
- AST gives us semantic structure, not just pattern matching
- AST enables relationship extraction (methods know their classes)
"""

import ast
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.chunk_model import CodeChunk, generate_chunk_id
from utils import get_logger

logger = get_logger(__name__)


class PythonParser:
    """
    Parses Python source code using AST.
    
    Creates chunks for:
    - Classes (and their methods)
    - Functions (module-level and nested)
    - Methods (as separate chunks)
    
    Benefits of AST approach:
    - Accurate line number tracking
    - Proper nesting understanding
    - Docstring extraction
    - Decorator information
    - Type hints preservation
    """
    
    def __init__(self, repository_name: str = "unknown"):
        """
        Initialize Python parser
        
        Args:
            repository_name: Name of the repository (for metadata)
        """
        self.repository_name = repository_name
        self.logger = get_logger(self.__class__.__name__)
    
    def parse_file(
        self,
        file_path: str,
        file_content: str,
        relative_path: str
    ) -> List[CodeChunk]:
        """
        Parse a Python file and extract chunks
        
        Args:
            file_path: Absolute path to file
            file_content: File content as string
            relative_path: Path relative to repository root
            
        Returns:
            List of CodeChunk objects
            
        Example:
            >>> parser = PythonParser("my_repo")
            >>> chunks = parser.parse_file("/path/to/auth.py", content, "src/auth.py")
            >>> for chunk in chunks:
            ...     print(chunk.get_display_name())
            # class User
            # User.__init__()
            # User.authenticate()
            # validate_token()
        """
        chunks = []
        
        try:
            # Parse the file into an AST
            tree = ast.parse(file_content)
            
            # Extract module-level docstring
            module_docstring = ast.get_docstring(tree)
            
            # Process all top-level nodes
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    # Process class and its methods
                    class_chunks = self._parse_class(
                        node,
                        file_content,
                        file_path,
                        relative_path
                    )
                    chunks.extend(class_chunks)
                
                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # Process module-level function
                    func_chunk = self._parse_function(
                        node,
                        file_content,
                        file_path,
                        relative_path,
                        class_name=None
                    )
                    if func_chunk:
                        chunks.append(func_chunk)
            
            self.logger.info(f"Extracted {len(chunks)} chunks from {relative_path}")
        
        except SyntaxError as e:
            self.logger.error(f"Syntax error in {relative_path}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error parsing {relative_path}: {str(e)}")
        
        return chunks
    
    def _parse_class(
        self,
        class_node: ast.ClassDef,
        file_content: str,
        file_path: str,
        relative_path: str
    ) -> List[CodeChunk]:
        """
        Extract chunks from a class
        
        Args:
            class_node: AST ClassDef node
            file_content: Original file content
            file_path: Absolute path
            relative_path: Relative path
            
        Returns:
            List of chunks (class and its methods)
        """
        chunks = []
        
        # Extract class chunk
        class_content = self._extract_node_content(class_node, file_content)
        class_docstring = ast.get_docstring(class_node)
        
        class_chunk = CodeChunk(
            chunk_id=generate_chunk_id(relative_path, "class", class_node.name, class_node.lineno),
            file_name=Path(file_path).name,
            file_path=relative_path,
            language="python",
            repository_name=self.repository_name,
            content=class_content,
            chunk_type="class",
            class_name=class_node.name,
            function_name=None,
            start_line=class_node.lineno,
            end_line=class_node.end_lineno,
            docstring=class_docstring,
        )
        chunks.append(class_chunk)
        
        # Extract methods
        for item in class_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_chunk = self._parse_function(
                    item,
                    file_content,
                    file_path,
                    relative_path,
                    class_name=class_node.name
                )
                if method_chunk:
                    chunks.append(method_chunk)
        
        return chunks
    
    def _parse_function(
        self,
        func_node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_content: str,
        file_path: str,
        relative_path: str,
        class_name: Optional[str] = None
    ) -> Optional[CodeChunk]:
        """
        Extract a function or method chunk
        
        Args:
            func_node: AST FunctionDef node
            file_content: Original file content
            file_path: Absolute path
            relative_path: Relative path
            class_name: Parent class name (if method)
            
        Returns:
            CodeChunk object or None
        """
        try:
            content = self._extract_node_content(func_node, file_content)
            docstring = ast.get_docstring(func_node)
            
            # Determine if it's a special method (skip __init__, etc. unless documented)
            is_private = func_node.name.startswith("_")
            
            # Get function signature with type hints
            signature = self._get_function_signature(func_node)
            
            chunk_type = "method" if class_name else "function"
            
            chunk = CodeChunk(
                chunk_id=generate_chunk_id(relative_path, chunk_type, func_node.name, func_node.lineno),
                file_name=Path(file_path).name,
                file_path=relative_path,
                language="python",
                repository_name=self.repository_name,
                content=content,
                chunk_type=chunk_type,
                class_name=class_name,
                function_name=func_node.name,
                start_line=func_node.lineno,
                end_line=func_node.end_lineno,
                docstring=docstring,
            )
            
            # Add signature to metadata
            chunk.add_metadata("signature", signature)
            chunk.add_metadata("is_async", isinstance(func_node, ast.AsyncFunctionDef))
            chunk.add_metadata("is_private", is_private)
            chunk.add_metadata("has_docstring", bool(docstring))
            
            return chunk
        
        except Exception as e:
            self.logger.error(f"Error parsing function {func_node.name}: {str(e)}")
            return None
    
    def _extract_node_content(self, node: ast.AST, file_content: str) -> str:
        """
        Extract source code for an AST node
        
        Args:
            node: AST node
            file_content: Original file content
            
        Returns:
            Source code for the node
        """
        lines = file_content.split("\n")
        
        start_line = node.lineno - 1  # Convert to 0-indexed
        end_line = node.end_lineno    # end_lineno is already inclusive
        
        if start_line < 0 or end_line > len(lines):
            return ""
        
        extracted_lines = lines[start_line:end_line]
        return "\n".join(extracted_lines)
    
    def _get_function_signature(self, func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """
        Get function signature with type hints
        
        Args:
            func_node: AST FunctionDef node
            
        Returns:
            Function signature string
            
        Example:
            >>> "def authenticate(self, username: str, password: str) -> bool:"
        """
        try:
            args = func_node.args
            arg_strs = []
            
            # Process arguments
            for arg in args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += ": " + ast.unparse(arg.annotation)
                arg_strs.append(arg_str)
            
            # Process *args
            if args.vararg:
                arg_strs.append(f"*{args.vararg.arg}")
            
            # Process keyword-only arguments
            for arg in args.kwonlyargs:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += ": " + ast.unparse(arg.annotation)
                arg_strs.append(arg_str)
            
            # Process **kwargs
            if args.kwarg:
                arg_strs.append(f"**{args.kwarg.arg}")
            
            signature = f"def {func_node.name}({', '.join(arg_strs)})"
            
            # Add return type if present
            if func_node.returns:
                return_type = ast.unparse(func_node.returns)
                signature += f" -> {return_type}"
            
            signature += ":"
            return signature
        
        except Exception as e:
            self.logger.warning(f"Error getting signature: {str(e)}")
            return f"def {func_node.name}(...):"
    
    def extract_imports(self, file_content: str) -> List[str]:
        """
        Extract import statements from a file
        
        Useful for understanding dependencies
        
        Args:
            file_content: File content
            
        Returns:
            List of import statements
        """
        imports = []
        
        try:
            tree = ast.parse(file_content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}" if module else alias.name)
        
        except Exception as e:
            self.logger.error(f"Error extracting imports: {str(e)}")
        
        return imports
    
    def extract_dependencies(self, chunks: List[CodeChunk]) -> Dict[str, List[str]]:
        """
        Identify dependencies between chunks
        
        This is advanced analysis that helps understand code relationships
        
        Args:
            chunks: List of extracted chunks
            
        Returns:
            Dictionary mapping chunk_id to list of dependent chunk_ids
        """
        # This is a simplified version - full implementation would analyze
        # function calls, class inheritance, etc.
        dependencies: Dict[str, List[str]] = {}
        
        for chunk in chunks:
            dependencies[chunk.chunk_id] = []
            
            # Simple heuristic: if chunk content mentions another chunk's name
            for other_chunk in chunks:
                if chunk.chunk_id != other_chunk.chunk_id:
                    if other_chunk.function_name:
                        if other_chunk.function_name in chunk.content:
                            dependencies[chunk.chunk_id].append(other_chunk.chunk_id)
                    if other_chunk.class_name:
                        if other_chunk.class_name in chunk.content:
                            dependencies[chunk.chunk_id].append(other_chunk.chunk_id)
        
        return dependencies
