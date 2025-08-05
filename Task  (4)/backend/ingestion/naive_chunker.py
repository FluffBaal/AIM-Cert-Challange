"""
NaiveChunker - Simple fixed-size chunking strategy

Implements basic fixed-size chunking with overlap as specified in the build plan.
"""

from typing import List, Dict, Any
import tiktoken
try:
    from .constants import NAIVE_CHUNK_SIZE, NAIVE_OVERLAP
except ImportError:
    from constants import NAIVE_CHUNK_SIZE, NAIVE_OVERLAP


class Chunk:
    """Represents a single chunk of text with metadata"""
    
    def __init__(self, content: str, chunk_id: str, method: str, 
                 chunk_index: int = 0, overlap_start: int = 0, overlap_end: int = 0):
        self.content = content
        self.chunk_id = chunk_id
        self.method = method
        self.chunk_index = chunk_index
        self.overlap_start = overlap_start
        self.overlap_end = overlap_end
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage"""
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "method": self.method,
            "chunk_index": self.chunk_index,
            "overlap_start": self.overlap_start,
            "overlap_end": self.overlap_end
        }


class NaiveChunker:
    """
    Simple fixed-size chunking strategy
    
    Uses fixed token count with overlap to create chunks.
    This serves as the baseline for comparison with advanced chunking.
    """
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize the tokenizer"""
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to tokens"""
        return self.encoding.encode(text)
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert tokens back to text"""
        return self.encoding.decode(tokens)
    
    def chunk_document(self, text: str) -> List[Chunk]:
        """
        Chunk document using fixed-size strategy with overlap
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of Chunk objects
        """
        # Fixed-size chunking with overlap
        chunk_size = NAIVE_CHUNK_SIZE  # 500 tokens
        overlap = NAIVE_OVERLAP        # 50 tokens
        
        chunks = []
        tokens = self.tokenize(text)
        
        chunk_index = 0
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            
            # Calculate overlap boundaries
            overlap_start = max(0, i - overlap) if i > 0 else 0
            overlap_end = min(len(tokens), i + chunk_size + overlap)
            
            chunk = Chunk(
                content=self.detokenize(chunk_tokens),
                chunk_id=f"naive_{i}",
                method="naive",
                chunk_index=chunk_index,
                overlap_start=overlap_start,
                overlap_end=overlap_end
            )
            
            chunks.append(chunk)
            chunk_index += 1
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not chunks:
            return {}
        
        token_counts = [len(self.tokenize(chunk.content)) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "total_tokens": sum(token_counts),
            "method": "naive"
        }