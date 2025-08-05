"""Advanced Parent-Child Chunking Strategy Implementation"""
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import tiktoken
from langchain_core.documents import Document
from rich.console import Console

console = Console()


@dataclass
class ParentChunk:
    """Parent chunk containing document section"""
    content: str
    parent_id: str
    section_heading: str
    anchor_id: str
    tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: List['ChildChunk'] = field(default_factory=list)


@dataclass
class ChildChunk:
    """Child chunk within a parent"""
    content: str
    child_idx: int
    parent_id: str
    tokens: int
    parent_content: str = ""
    section_heading: str = ""


class ParentChildChunker:
    """
    Implements the three-phase Parent-Child chunking algorithm:
    - Phase A: Heading Pass (Parent Creation)
    - Phase B: Semantic Throttling (Child Creation)
    - Phase C: Thin-Parent Repair
    """
    
    def __init__(
        self,
        parent_max_tokens: int = 1200,
        parent_min_tokens: int = 300,
        child_min_tokens: int = 100,
        child_max_tokens: int = 140
    ):
        self.parent_max_tokens = parent_max_tokens
        self.parent_min_tokens = parent_min_tokens
        self.child_min_tokens = child_min_tokens
        self.child_max_tokens = child_max_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def chunk_documents(self, documents: List[Document]) -> List[ChildChunk]:
        """
        Main entry point: convert documents to child chunks
        """
        all_children = []
        
        for doc_idx, doc in enumerate(documents):
            console.print(f"[dim]Processing document {doc_idx + 1}/{len(documents)}...[/dim]")
            
            # Phase A: Create parents from document structure
            parents = self._create_parent_chunks(doc.page_content)
            
            # Phase B: Create children within each parent
            for parent in parents:
                children = self._create_child_chunks(parent)
                parent.children = children
            
            # Phase C: Repair thin parents
            parents = self._repair_thin_parents(parents)
            
            # Collect all children with parent metadata
            for parent in parents:
                for child in parent.children:
                    child.parent_content = parent.content
                    child.section_heading = parent.section_heading
                    all_children.append(child)
        
        console.print(f"[green]Created {len(all_children)} child chunks from {len(documents)} documents[/green]")
        return all_children
    
    def _create_parent_chunks(self, text: str) -> List[ParentChunk]:
        """
        Phase A: Split document by headings to create parent chunks
        """
        parents = []
        
        # Split by markdown headings (# and ##)
        heading_pattern = r'^(#{1,2})\s+(.+)$'
        lines = text.split('\n')
        
        current_section = []
        current_heading = "Introduction"
        current_level = 0
        
        for i, line in enumerate(lines):
            match = re.match(heading_pattern, line, re.MULTILINE)
            
            if match:
                # Save previous section if exists
                if current_section:
                    content = '\n'.join(current_section).strip()
                    if content:
                        parent = self._create_parent(content, current_heading)
                        
                        # Handle oversized parents
                        if parent.tokens > self.parent_max_tokens:
                            sub_parents = self._split_large_parent(parent)
                            parents.extend(sub_parents)
                        else:
                            parents.append(parent)
                
                # Start new section
                level = len(match.group(1))
                heading_text = match.group(2).strip()
                current_heading = heading_text
                current_level = level
                current_section = [line]
            else:
                current_section.append(line)
        
        # Don't forget the last section
        if current_section:
            content = '\n'.join(current_section).strip()
            if content:
                parent = self._create_parent(content, current_heading)
                if parent.tokens > self.parent_max_tokens:
                    sub_parents = self._split_large_parent(parent)
                    parents.extend(sub_parents)
                else:
                    parents.append(parent)
        
        return parents
    
    def _create_parent(self, content: str, heading: str) -> ParentChunk:
        """Create a parent chunk with metadata"""
        parent_id = str(uuid.uuid4())[:8]
        anchor_id = re.sub(r'[^a-zA-Z0-9-]', '-', heading.lower())
        
        return ParentChunk(
            content=content,
            parent_id=parent_id,
            section_heading=heading,
            anchor_id=anchor_id,
            tokens=self.count_tokens(content),
            metadata={'heading_level': 1 if content.startswith('#') else 2}
        )
    
    def _split_large_parent(self, parent: ParentChunk) -> List[ParentChunk]:
        """Split oversized parent into smaller chunks"""
        sentences = self._split_into_sentences(parent.content)
        sub_parents = []
        current_content = []
        current_tokens = 0
        part_num = 1
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > self.parent_max_tokens and current_content:
                # Create sub-parent
                sub_content = ' '.join(current_content)
                sub_parent = ParentChunk(
                    content=sub_content,
                    parent_id=f"{parent.parent_id}-{part_num}",
                    section_heading=f"{parent.section_heading} (Part {part_num})",
                    anchor_id=f"{parent.anchor_id}-part-{part_num}",
                    tokens=current_tokens,
                    metadata=parent.metadata.copy()
                )
                sub_parents.append(sub_parent)
                
                # Reset for next chunk
                current_content = [sentence]
                current_tokens = sentence_tokens
                part_num += 1
            else:
                current_content.append(sentence)
                current_tokens += sentence_tokens
        
        # Handle remaining content
        if current_content:
            sub_content = ' '.join(current_content)
            sub_parent = ParentChunk(
                content=sub_content,
                parent_id=f"{parent.parent_id}-{part_num}",
                section_heading=f"{parent.section_heading} (Part {part_num})",
                anchor_id=f"{parent.anchor_id}-part-{part_num}",
                tokens=current_tokens,
                metadata=parent.metadata.copy()
            )
            sub_parents.append(sub_parent)
        
        return sub_parents
    
    def _create_child_chunks(self, parent: ParentChunk) -> List[ChildChunk]:
        """
        Phase B: Apply semantic throttling to create child chunks
        """
        sentences = self._split_into_sentences(parent.content)
        children = []
        current_chunk = []
        current_tokens = 0
        child_idx = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # Check if adding sentence exceeds max tokens
            if current_tokens + sentence_tokens > self.child_max_tokens and current_chunk:
                # Save current chunk if it meets minimum
                if current_tokens >= self.child_min_tokens:
                    child = ChildChunk(
                        content=' '.join(current_chunk),
                        child_idx=child_idx,
                        parent_id=parent.parent_id,
                        tokens=current_tokens
                    )
                    children.append(child)
                    child_idx += 1
                    
                    # Start new chunk
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                else:
                    # Current chunk too small, keep adding
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Handle remaining content
        if current_chunk:
            # If too small, merge with previous child if possible
            if current_tokens < self.child_min_tokens and children:
                last_child = children[-1]
                combined_tokens = last_child.tokens + current_tokens
                
                if combined_tokens <= self.child_max_tokens * 1.2:  # Allow 20% overflow
                    # Merge with previous
                    last_child.content += ' ' + ' '.join(current_chunk)
                    last_child.tokens = combined_tokens
                else:
                    # Keep as separate small chunk
                    child = ChildChunk(
                        content=' '.join(current_chunk),
                        child_idx=child_idx,
                        parent_id=parent.parent_id,
                        tokens=current_tokens
                    )
                    children.append(child)
            else:
                # Create final chunk
                child = ChildChunk(
                    content=' '.join(current_chunk),
                    child_idx=child_idx,
                    parent_id=parent.parent_id,
                    tokens=current_tokens
                )
                children.append(child)
        
        return children
    
    def _repair_thin_parents(self, parents: List[ParentChunk]) -> List[ParentChunk]:
        """
        Phase C: Merge or adjust parents that are too small
        """
        if not parents:
            return parents
        
        repaired = []
        i = 0
        
        while i < len(parents):
            parent = parents[i]
            
            if parent.tokens < self.parent_min_tokens:
                # Try merging with next parent
                if i + 1 < len(parents):
                    next_parent = parents[i + 1]
                    combined_tokens = parent.tokens + next_parent.tokens
                    
                    if combined_tokens <= self.parent_max_tokens:
                        # Merge parents
                        merged = self._merge_parents(parent, next_parent)
                        repaired.append(merged)
                        i += 2  # Skip next parent
                        continue
                
                # If can't merge forward, try merging backward
                if repaired and parent.tokens + repaired[-1].tokens <= self.parent_max_tokens:
                    # Merge with previous
                    prev_parent = repaired.pop()
                    merged = self._merge_parents(prev_parent, parent)
                    repaired.append(merged)
                    i += 1
                    continue
                
                # Can't merge, mark as thin
                parent.metadata['thin_parent'] = True
            
            repaired.append(parent)
            i += 1
        
        return repaired
    
    def _merge_parents(self, parent1: ParentChunk, parent2: ParentChunk) -> ParentChunk:
        """Merge two parent chunks"""
        merged_content = parent1.content + '\n\n' + parent2.content
        merged_heading = f"{parent1.section_heading} & {parent2.section_heading}"
        
        merged = ParentChunk(
            content=merged_content,
            parent_id=f"{parent1.parent_id}-{parent2.parent_id}",
            section_heading=merged_heading,
            anchor_id=f"{parent1.anchor_id}-{parent2.anchor_id}",
            tokens=parent1.tokens + parent2.tokens,
            metadata={
                'merged': True,
                'original_parents': [parent1.parent_id, parent2.parent_id]
            }
        )
        
        # Re-chunk children for merged parent
        merged.children = self._create_child_chunks(merged)
        
        return merged
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter - can be enhanced
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Handle special cases (e.g., "Dr.", "Mr.", etc.)
        merged_sentences = []
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            
            # Check if sentence ends with common abbreviation
            if (i + 1 < len(sentences) and 
                re.search(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.$', sentence)):
                # Merge with next sentence
                sentence = sentence + ' ' + sentences[i + 1]
                i += 1
            
            merged_sentences.append(sentence)
            i += 1
        
        return merged_sentences