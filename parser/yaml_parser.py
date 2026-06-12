"""
YAML/YML Parser

Parses YAML configuration files by top-level sections.
Creates chunks for each configuration section.

Why section-based chunking for YAML:
1. YAML files are configuration/metadata
2. Top-level keys represent logical sections
3. Preserves structure and relationships
4. Easier to understand in context
5. Better for configuration queries
"""

import yaml
import re
import sys
import os
from typing import List, Optional, Dict
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.chunk_model import CodeChunk, generate_chunk_id
from utils import get_logger

logger = get_logger(__name__)


class YAMLParser:
    """
    Parses YAML/YML files by top-level sections.
    
    Creates chunks for:
    - Each top-level key and its content
    - Preserves YAML structure
    - Groups related configuration together
    """
    
    def __init__(self, repository_name: str = "unknown"):
        """
        Initialize YAML parser
        
        Args:
            repository_name: Name of repository
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
        Parse a YAML file
        
        Args:
            file_path: Absolute path
            file_content: File content
            relative_path: Relative path
            
        Returns:
            List of CodeChunk objects
        """
        chunks = []
        
        try:
            # Try to parse YAML
            try:
                data = yaml.safe_load(file_content)
            except yaml.YAMLError:
                # If parsing fails, treat as text and chunk by sections
                return self._parse_as_text(file_path, file_content, relative_path)
            
            if not data or not isinstance(data, dict):
                # Not a key-value YAML, treat as text
                return self._parse_as_text(file_path, file_content, relative_path)
            
            # Extract sections based on top-level keys
            sections = self._extract_sections(file_content, data)
            
            for section in sections:
                chunk = CodeChunk(
                    chunk_id=generate_chunk_id(relative_path, "section", section["key"]),
                    file_name=Path(file_path).name,
                    file_path=relative_path,
                    language="yaml",
                    repository_name=self.repository_name,
                    content=section["content"],
                    chunk_type="section",
                    class_name=None,
                    function_name=section["key"],
                    start_line=section["start_line"],
                    end_line=section["end_line"],
                )
                
                chunk.add_metadata("is_root_key", True)
                
                chunks.append(chunk)
            
            self.logger.info(f"Extracted {len(chunks)} sections from {relative_path}")
        
        except Exception as e:
            self.logger.error(f"Error parsing {relative_path}: {str(e)}")
        
        return chunks
    
    def _extract_sections(self, file_content: str, data: dict) -> List[Dict]:
        """
        Extract sections from YAML content
        
        Args:
            file_content: YAML content
            data: Parsed YAML data
            
        Returns:
            List of section dictionaries
        """
        sections = []
        lines = file_content.split('\n')
        
        # Find line numbers for each top-level key
        key_positions = {}
        
        for i, line in enumerate(lines):
            # Look for top-level keys (no leading spaces)
            if line and not line[0].isspace() and ':' in line:
                key = line.split(':')[0].strip()
                if key and key in data:
                    key_positions[key] = i
        
        # Sort by position
        sorted_keys = sorted(key_positions.items(), key=lambda x: x[1])
        
        # Extract content for each key
        for idx, (key, start_pos) in enumerate(sorted_keys):
            # Find end position (start of next key)
            if idx + 1 < len(sorted_keys):
                end_pos = sorted_keys[idx + 1][1] - 1
            else:
                end_pos = len(lines) - 1
            
            # Get section content
            section_lines = lines[start_pos:end_pos + 1]
            section_content = '\n'.join(section_lines)
            
            sections.append({
                "key": key,
                "content": section_content,
                "start_line": start_pos + 1,
                "end_line": end_pos + 1,
            })
        
        return sections
    
    def _parse_as_text(
        self,
        file_path: str,
        file_content: str,
        relative_path: str
    ) -> List[CodeChunk]:
        """
        Parse YAML file as plain text if YAML parsing fails
        
        Args:
            file_path: Absolute path
            file_content: File content
            relative_path: Relative path
            
        Returns:
            List with single chunk
        """
        chunk = CodeChunk(
            chunk_id=generate_chunk_id(relative_path, "file", Path(file_path).stem),
            file_name=Path(file_path).name,
            file_path=relative_path,
            language="yaml",
            repository_name=self.repository_name,
            content=file_content,
            chunk_type="file",
            class_name=None,
            function_name=None,
            start_line=1,
            end_line=len(file_content.split('\n')),
        )
        
        return [chunk]
    
    def extract_keys(self, data: dict, prefix: str = "") -> List[str]:
        """
        Recursively extract all keys from YAML data
        
        Args:
            data: Parsed YAML data
            prefix: Current key prefix
            
        Returns:
            List of all keys with full paths
        """
        keys = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                keys.append(full_key)
                
                # Recursively extract nested keys
                if isinstance(value, dict):
                    keys.extend(self.extract_keys(value, full_key))
        
        return keys
    
    def validate_yaml_syntax(self, file_content: str) -> tuple:
        """
        Validate YAML file syntax
        
        Args:
            file_content: File content
            
        Returns:
            Tuple of (is_valid: bool, error_message: str or None)
        """
        try:
            yaml.safe_load(file_content)
            return (True, None)
        except yaml.YAMLError as e:
            return (False, str(e))
