"""
Markdown Parser

Parses Markdown files by heading hierarchy.
Creates chunks for each section (heading + content until next heading).

Why heading-based chunking for Markdown:
1. Headings define natural semantic units
2. Hierarchy (H1, H2, H3) shows relationships
3. Preserves document structure
4. Humans read Markdown by sections
5. Better context for embedding and retrieval
"""

import re
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.chunk_model import CodeChunk, generate_chunk_id
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class MarkdownSection:
    """Represents a section in Markdown"""
    level: int  # 1-6 for H1-H6
    title: str
    content: str
    start_line: int
    end_line: int


class MarkdownParser:
    """
    Parses Markdown files by heading hierarchy.
    
    Creates chunks for:
    - Each heading with its content
    - Respects heading hierarchy (H1, H2, etc.)
    - Groups content under headings
    
    This is better than line-based splitting because:
    - Semantic units (sections) are preserved
    - Heading hierarchy provides context
    - Related content stays together
    - Matches how humans read Markdown
    """
    
    def __init__(self, repository_name: str = "unknown"):
        """
        Initialize Markdown parser
        
        Args:
            repository_name: Name of repository
        """
        self.repository_name = repository_name
        self.logger = get_logger(self.__class__.__name__)
        
        # Regex for Markdown headings
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def parse_file(
        self,
        file_path: str,
        file_content: str,
        relative_path: str
    ) -> List[CodeChunk]:
        """
        Parse a Markdown file
        
        Args:
            file_path: Absolute path
            file_content: File content
            relative_path: Relative path
            
        Returns:
            List of CodeChunk objects
        """
        chunks = []
        
        try:
            sections = self._extract_sections(file_content)
            
            for section in sections:
                chunk = CodeChunk(
                    chunk_id=generate_chunk_id(relative_path, f"h{section.level}", section.title),
                    file_name=Path(file_path).name,
                    file_path=relative_path,
                    language="markdown",
                    repository_name=self.repository_name,
                    content=section.content,
                    chunk_type=f"section_h{section.level}",
                    class_name=None,
                    function_name=section.title,
                    start_line=section.start_line,
                    end_line=section.end_line,
                )
                
                chunk.add_metadata("heading_level", section.level)
                chunk.add_metadata("is_main_section", section.level == 1)
                
                chunks.append(chunk)
            
            self.logger.info(f"Extracted {len(chunks)} sections from {relative_path}")
        
        except Exception as e:
            self.logger.error(f"Error parsing {relative_path}: {str(e)}")
        
        return chunks
    
    def _extract_sections(self, content: str) -> List[MarkdownSection]:
        """
        Extract sections from Markdown content
        
        Args:
            content: Markdown content
            
        Returns:
            List of MarkdownSection objects
        """
        sections = []
        lines = content.split('\n')
        
        # Find all headings
        headings: List[Tuple[int, str, int]] = []  # (level, title, line_num)
        
        for i, line in enumerate(lines):
            match = self.heading_pattern.match(line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                headings.append((level, title, i))
        
        # Create sections
        for idx, (level, title, line_num) in enumerate(headings):
            start_line = line_num
            
            # Find content until next heading
            if idx + 1 < len(headings):
                end_line = headings[idx + 1][2] - 1
            else:
                end_line = len(lines) - 1
            
            # Get section content
            section_lines = lines[start_line:end_line + 1]
            section_content = '\n'.join(section_lines)
            
            section = MarkdownSection(
                level=level,
                title=title,
                content=section_content,
                start_line=start_line + 1,  # Convert to 1-indexed
                end_line=end_line + 1,
            )
            
            sections.append(section)
        
        # If no headings, treat entire file as one section
        if not sections:
            sections.append(MarkdownSection(
                level=1,
                title=Path(lines[0]).stem if lines else "Document",
                content=content,
                start_line=1,
                end_line=len(lines),
            ))
        
        return sections
    
    def extract_table_of_contents(self, content: str) -> Dict[int, List[str]]:
        """
        Extract table of contents from Markdown
        
        Args:
            content: Markdown content
            
        Returns:
            Dictionary mapping heading level to titles
        """
        toc = {}
        
        for match in self.heading_pattern.finditer(content):
            level = len(match.group(1))
            title = match.group(2).strip()
            
            if level not in toc:
                toc[level] = []
            toc[level].append(title)
        
        return toc
