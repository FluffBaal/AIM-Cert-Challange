"""
Ingestion Package - Dual Chunking and RAG Implementation

This package implements the complete dual chunking and retrieval system
as specified in the build plan, including:

- Chunking constants and configuration
- NaiveChunker for fixed-size chunking
- SemanticThrottledChunker for advanced markdown-aware chunking
- Dual Qdrant collections setup and management
- DualRAGRetriever with context corridor and Cohere reranking
- DualChunkingPipeline for orchestrating the entire process
"""

from .constants import (
    SIMILARITY_MERGE_THRESHOLD,
    SIMILARITY_SPLIT_THRESHOLD,
    PARENT_TOKEN_LIMIT,
    CHILD_TOKEN_MIN,
    CHILD_TOKEN_MAX,
    THIN_PARENT_MIN,
    CONTEXT_CORRIDOR_THRESHOLD,
    NAIVE_COLLECTION,
    ADVANCED_COLLECTION,
    NAIVE_CHUNK_SIZE,
    NAIVE_OVERLAP,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
    COHERE_MODEL
)

from .naive_chunker import NaiveChunker, Chunk
from .semantic_chunker import SemanticThrottledChunker, ParentChunk, ChildChunk
from .qdrant_collections import QdrantCollectionManager
from .dual_rag_retriever import DualRAGRetriever, RetrievalResult, EmbeddingService
from .dual_chunking_pipeline import DualChunkingPipeline, PDFProcessor

__version__ = "1.0.0"
__author__ = "AIM Cert Challenge Task 4"

__all__ = [
    # Constants
    "SIMILARITY_MERGE_THRESHOLD",
    "SIMILARITY_SPLIT_THRESHOLD", 
    "PARENT_TOKEN_LIMIT",
    "CHILD_TOKEN_MIN",
    "CHILD_TOKEN_MAX",
    "THIN_PARENT_MIN",
    "CONTEXT_CORRIDOR_THRESHOLD",
    "NAIVE_COLLECTION",
    "ADVANCED_COLLECTION",
    "NAIVE_CHUNK_SIZE",
    "NAIVE_OVERLAP",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "COHERE_MODEL",
    
    # Chunking classes
    "NaiveChunker",
    "Chunk",
    "SemanticThrottledChunker", 
    "ParentChunk",
    "ChildChunk",
    
    # Storage and retrieval
    "QdrantCollectionManager",
    "DualRAGRetriever",
    "RetrievalResult",
    "EmbeddingService",
    
    # Pipeline orchestration
    "DualChunkingPipeline",
    "PDFProcessor"
]