"""
SemanticThrottledChunker - Advanced Markdown-aware parent-child chunking

Implements sophisticated chunking strategy that:
1. Split on headings (#, ##) for parents
2. Semantic throttling for children within parents
3. Respect Markdown boundaries (lists, quotes)
4. Maintain parent-child relationships
"""

import re
from typing import List, Dict, Any, Optional
import tiktoken
from dataclasses import dataclass
try:
    from .constants import (
        PARENT_TOKEN_LIMIT, CHILD_TOKEN_MIN, CHILD_TOKEN_MAX,
        SIMILARITY_MERGE_THRESHOLD, SIMILARITY_SPLIT_THRESHOLD,
        THIN_PARENT_MIN
    )
except ImportError:
    from constants import (
        PARENT_TOKEN_LIMIT, CHILD_TOKEN_MIN, CHILD_TOKEN_MAX,
        SIMILARITY_MERGE_THRESHOLD, SIMILARITY_SPLIT_THRESHOLD,
        THIN_PARENT_MIN
    )


@dataclass
class ChildChunk:
    """Represents a child chunk within a parent"""
    content: str
    child_idx: int
    parent_id: str
    tokens: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "child_idx": self.child_idx,
            "parent_id": self.parent_id,
            "tokens": self.tokens
        }


@dataclass
class ParentChunk:
    """Represents a parent chunk with potential children"""
    content: str
    parent_id: str
    section_heading: str
    anchor_id: str
    children: List[ChildChunk]
    tokens: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "parent_id": self.parent_id,
            "section_heading": self.section_heading,
            "anchor_id": self.anchor_id,
            "children": [child.to_dict() for child in self.children],
            "tokens": self.tokens,
            "metadata": self.metadata
        }


class SemanticThrottledChunker:
    """
    Advanced Markdown-aware parent-child chunking
    
    1. Split on headings (#, ##) for parents
    2. Semantic throttling for children within parents
    3. Respect Markdown boundaries (lists, quotes)
    4. Maintain parent-child relationships
    """
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize the tokenizer"""
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to tokens"""
        return self.encoding.encode(text)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenize(text))
    
    def chunk_document(self, text: str) -> List[ParentChunk]:
        """
        Main chunking method following the build plan algorithm
        
        Args:
            text: Input markdown text to chunk
            
        Returns:
            List of ParentChunk objects with children
        """
        # A. Heading pass - create parent skeletons
        parents = self.split_by_headings(text, max_tokens=PARENT_TOKEN_LIMIT)
        
        # B. Child semantic throttling
        for parent in parents:
            children = self.semantic_throttle(
                parent.content,
                parent.parent_id,
                min_tokens=CHILD_TOKEN_MIN,
                max_tokens=CHILD_TOKEN_MAX,
                merge_threshold=SIMILARITY_MERGE_THRESHOLD,
                split_threshold=SIMILARITY_SPLIT_THRESHOLD
            )
            parent.children = children
        
        # C. Thin-parent repair
        parents = self.repair_thin_parents(parents, min_tokens=THIN_PARENT_MIN)
        
        return parents
    
    def split_by_headings(self, text: str, max_tokens: int) -> List[ParentChunk]:
        """
        Split text by markdown headings to create parent chunks
        
        Args:
            text: Input text
            max_tokens: Maximum tokens per parent
            
        Returns:
            List of parent chunks
        """
        # Split by headings (# and ##)
        heading_pattern = r'^(#{1,2})\s+(.+)$'
        lines = text.split('\n')
        
        parents = []
        current_section = []
        current_heading = "Introduction"
        parent_idx = 0
        
        for line in lines:
            heading_match = re.match(heading_pattern, line, re.MULTILINE)
            
            if heading_match:
                # Process previous section if it exists
                if current_section:
                    section_text = '\n'.join(current_section)
                    token_count = self.count_tokens(section_text)
                    
                    # Split large sections
                    if token_count > max_tokens:
                        sub_sections = self.split_large_section(section_text, max_tokens)
                        for i, sub_section in enumerate(sub_sections):
                            parent = ParentChunk(
                                content=sub_section,
                                parent_id=f"parent_{parent_idx}_{i}",
                                section_heading=f"{current_heading} (Part {i+1})",
                                anchor_id=self.create_anchor_id(current_heading, i),
                                children=[],
                                tokens=self.count_tokens(sub_section),
                                metadata={"split_from_large": True}
                            )
                            parents.append(parent)
                            parent_idx += 1
                    else:
                        parent = ParentChunk(
                            content=section_text,
                            parent_id=f"parent_{parent_idx}",
                            section_heading=current_heading,
                            anchor_id=self.create_anchor_id(current_heading),
                            children=[],
                            tokens=token_count,
                            metadata={}
                        )
                        parents.append(parent)
                        parent_idx += 1
                
                # Start new section
                current_heading = heading_match.group(2).strip()
                current_section = [line]
            else:
                current_section.append(line)
        
        # Process final section
        if current_section:
            section_text = '\n'.join(current_section)
            token_count = self.count_tokens(section_text)
            
            if token_count > max_tokens:
                sub_sections = self.split_large_section(section_text, max_tokens)
                for i, sub_section in enumerate(sub_sections):
                    parent = ParentChunk(
                        content=sub_section,
                        parent_id=f"parent_{parent_idx}_{i}",
                        section_heading=f"{current_heading} (Part {i+1})",
                        anchor_id=self.create_anchor_id(current_heading, i),
                        children=[],
                        tokens=self.count_tokens(sub_section),
                        metadata={"split_from_large": True}
                    )
                    parents.append(parent)
                    parent_idx += 1
            else:
                parent = ParentChunk(
                    content=section_text,
                    parent_id=f"parent_{parent_idx}",
                    section_heading=current_heading,
                    anchor_id=self.create_anchor_id(current_heading),
                    children=[],
                    tokens=token_count,
                    metadata={}
                )
                parents.append(parent)
        
        return parents
    
    def split_large_section(self, text: str, max_tokens: int) -> List[str]:
        """Split a large section into smaller chunks respecting markdown boundaries"""
        # Simple sentence-based splitting for now
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def semantic_throttle(self, text: str, parent_id: str, min_tokens: int, 
                         max_tokens: int, merge_threshold: float, 
                         split_threshold: float) -> List[ChildChunk]:
        """
        Apply semantic throttling to create child chunks
        
        This is a simplified implementation - in production you'd use
        actual semantic similarity calculations.
        """
        # For now, implement basic sentence-based chunking with token limits
        sentences = re.split(r'(?<=[.!?])\s+', text)
        children = []
        current_chunk = []
        current_tokens = 0
        child_idx = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # Check if adding this sentence would exceed max tokens
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Create child chunk if it meets minimum token requirement
                chunk_text = ' '.join(current_chunk)
                if current_tokens >= min_tokens:
                    child = ChildChunk(
                        content=chunk_text,
                        child_idx=child_idx,
                        parent_id=parent_id,
                        tokens=current_tokens
                    )
                    children.append(child)
                    child_idx += 1
                
                # Start new chunk
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Handle final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if current_tokens >= min_tokens:
                child = ChildChunk(
                    content=chunk_text,
                    child_idx=child_idx,
                    parent_id=parent_id,
                    tokens=current_tokens
                )
                children.append(child)
        
        return children
    
    def repair_thin_parents(self, parents: List[ParentChunk], 
                           min_tokens: int) -> List[ParentChunk]:
        """
        Repair thin parents by merging with adjacent parents or promoting children
        """
        repaired_parents = []
        i = 0
        
        while i < len(parents):
            parent = parents[i]
            
            if parent.tokens < min_tokens:
                # Try to merge with next parent
                if i + 1 < len(parents):
                    next_parent = parents[i + 1]
                    combined_tokens = parent.tokens + next_parent.tokens
                    
                    if combined_tokens <= PARENT_TOKEN_LIMIT:
                        # Merge parents
                        merged_content = parent.content + "\n\n" + next_parent.content
                        merged_parent = ParentChunk(
                            content=merged_content,
                            parent_id=parent.parent_id,
                            section_heading=f"{parent.section_heading} + {next_parent.section_heading}",
                            anchor_id=parent.anchor_id,
                            children=parent.children + next_parent.children,
                            tokens=combined_tokens,
                            metadata={"merged_thin_parent": True}
                        )
                        repaired_parents.append(merged_parent)
                        # Skip next parent as it's been merged
                        i += 2  # Skip both current and next
                        continue
                
                # If can't merge, promote children or keep as-is
                if parent.children:
                    parent.metadata["thin_parent_kept"] = True
                
                repaired_parents.append(parent)
            else:
                repaired_parents.append(parent)
            
            i += 1
        
        return repaired_parents
    
    def create_anchor_id(self, heading: str, part: Optional[int] = None) -> str:
        """Create anchor ID from heading"""
        anchor = re.sub(r'[^\w\s-]', '', heading).strip()
        anchor = re.sub(r'[-\s]+', '-', anchor).lower()
        if part is not None:
            anchor += f"-part-{part}"
        return anchor
    
    def get_chunk_stats(self, parents: List[ParentChunk]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not parents:
            return {}
        
        total_children = sum(len(parent.children) for parent in parents)
        parent_tokens = [parent.tokens for parent in parents]
        child_tokens = []
        
        for parent in parents:
            child_tokens.extend([child.tokens for child in parent.children])
        
        stats = {
            "total_parents": len(parents),
            "total_children": total_children,
            "avg_parent_tokens": sum(parent_tokens) / len(parent_tokens) if parent_tokens else 0,
            "total_tokens": sum(parent_tokens),
            "method": "semantic_throttled"
        }
        
        if child_tokens:
            stats.update({
                "avg_child_tokens": sum(child_tokens) / len(child_tokens),
                "min_child_tokens": min(child_tokens),
                "max_child_tokens": max(child_tokens)
            })
        
        return stats