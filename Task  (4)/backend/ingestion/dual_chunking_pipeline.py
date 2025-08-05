"""
DualChunkingPipeline - Orchestrates both naive and advanced chunking strategies

Implements the main pipeline that processes documents using both chunking approaches
for comparison and dual-collection storage.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import asyncio
from pathlib import Path

try:
    from .naive_chunker import NaiveChunker, Chunk
    from .semantic_chunker import SemanticThrottledChunker, ParentChunk
    from .qdrant_collections import QdrantCollectionManager
    from .dual_rag_retriever import EmbeddingService
    from .constants import EMBEDDING_MODEL
except ImportError:
    # Handle direct execution without package structure
    from naive_chunker import NaiveChunker, Chunk
    from semantic_chunker import SemanticThrottledChunker, ParentChunk
    from qdrant_collections import QdrantCollectionManager
    from dual_rag_retriever import EmbeddingService
    from constants import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class MarkdownProcessor:
    """Markdown file processing for text extraction"""
    
    @staticmethod
    def extract_text(md_path: str) -> str:
        """
        Extract text from Markdown file
        
        Args:
            md_path: Path to the markdown file
            
        Returns:
            The raw text content of the markdown file
        """
        try:
            with open(md_path, 'r', encoding='utf-8') as file:
                text = file.read()
                logger.info(f"Successfully extracted {len(text)} characters from {md_path}")
                return text
        except Exception as e:
            logger.error(f"Failed to read markdown file {md_path}: {e}")
            return ""

class PDFProcessor:
    """Simple PDF text extraction - placeholder for actual PDF processing"""
    
    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """
        Extract text from PDF file
        
        This is a placeholder implementation. In production, you would use
        libraries like PyMuPDF, pdfplumber, or similar.
        """
        # For now, return a placeholder
        # In real implementation:
        # import fitz  # PyMuPDF
        # doc = fitz.open(pdf_path)
        # text = ""
        # for page in doc:
        #     text += page.get_text()
        # return text
        
        logger.warning("PDF extraction not implemented - using placeholder text")
        return """
# Sample Negotiation Guide

## Introduction
This is a sample negotiation document that would be extracted from a PDF.

## Key Principles
- Always prepare thoroughly before any negotiation
- Listen actively to understand the other party's needs
- Look for win-win solutions that benefit everyone

## Advanced Techniques
- Use tactical empathy to build rapport
- Ask calibrated questions to gather information
- Practice active listening to uncover hidden motivations

## Conclusion
Successful negotiation requires preparation, patience, and practice.
"""


class DualChunkingPipeline:
    """
    Implements both naive and advanced chunking for comparison
    
    This is the main orchestration class that:
    1. Processes documents with both chunking strategies
    2. Generates embeddings for all chunks
    3. Stores chunks in appropriate Qdrant collections
    4. Provides comparison metrics
    """
    
    def __init__(self, qdrant_host: str = "qdrant", qdrant_port: int = 6333, openai_api_key: Optional[str] = None):
        self.naive_chunker = NaiveChunker()
        self.advanced_chunker = SemanticThrottledChunker()
        self.collection_manager = QdrantCollectionManager(qdrant_host, qdrant_port)
        self.embedder = EmbeddingService(EMBEDDING_MODEL, openai_api_key=openai_api_key)
        
    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document with both chunking strategies
        
        Args:
            file_path: Path to document file to process (PDF or Markdown)
            
        Returns:
            Dictionary with processing results and statistics
        """
        try:
            # Extract text once - handle both PDF and Markdown
            text = self.extract_document_text(file_path)
            
            if not text.strip():
                return {
                    "success": False,
                    "error": "No text extracted from document",
                    "file_path": file_path
                }
            
            # Process with both strategies
            logger.info(f"Processing document with dual chunking strategies: {file_path}")
            
            naive_chunks = self.naive_chunker.chunk_document(text)
            advanced_chunks = self.advanced_chunker.chunk_document(text)
            
            # Generate embeddings for both chunk types
            naive_embeddings = await self.generate_naive_embeddings(naive_chunks)
            advanced_embeddings = await self.generate_advanced_embeddings(advanced_chunks)
            
            # Store in collections
            naive_stored = await self.store_naive_chunks(naive_chunks, naive_embeddings)
            advanced_stored = await self.store_advanced_chunks(advanced_chunks, advanced_embeddings)
            
            # Calculate statistics
            naive_stats = self.naive_chunker.get_chunk_stats(naive_chunks)
            advanced_stats = self.advanced_chunker.get_chunk_stats(advanced_chunks)
            
            return {
                "success": True,
                "file_path": file_path,
                "text_length": len(text),
                "naive_chunks": naive_chunks,
                "advanced_chunks": advanced_chunks,
                "naive": {
                    "chunks_created": len(naive_chunks),
                    "embeddings_generated": len(naive_embeddings),
                    "stored_successfully": naive_stored,
                    "stats": naive_stats
                },
                "advanced": {
                    "parents_created": len(advanced_chunks),
                    "total_children": sum(len(parent.children) for parent in advanced_chunks),
                    "embeddings_generated": len(advanced_embeddings),
                    "stored_successfully": advanced_stored,
                    "stats": advanced_stats
                },
                "comparison": self.compare_chunking_strategies(naive_stats, advanced_stats)
            }
            
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    def extract_document_text(self, file_path: str) -> str:
        """Extract text from document file (PDF or Markdown)"""
        if file_path.lower().endswith('.md'):
            return MarkdownProcessor.extract_text(file_path)
        elif file_path.lower().endswith('.pdf'):
            return PDFProcessor.extract_text(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return ""
    
    async def generate_naive_embeddings(self, chunks: List[Chunk]) -> List[List[float]]:
        """Generate embeddings for naive chunks"""
        embeddings = []
        
        logger.info(f"Generating embeddings for {len(chunks)} naive chunks")
        
        for chunk in chunks:
            try:
                embedding = await self.embedder.embed(chunk.content)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk {chunk.chunk_id}: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 1536)
        
        return embeddings
    
    async def generate_advanced_embeddings(self, parents: List[ParentChunk]) -> List[List[float]]:
        """Generate embeddings for advanced chunks (children)"""
        embeddings = []
        
        # Count total children for logging
        total_children = sum(len(parent.children) for parent in parents)
        logger.info(f"Generating embeddings for {total_children} advanced child chunks")
        
        for parent in parents:
            for child in parent.children:
                try:
                    embedding = await self.embedder.embed(child.content)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Failed to generate embedding for child {child.parent_id}_{child.child_idx}: {e}")
                    # Use zero vector as fallback
                    embeddings.append([0.0] * 1536)
        
        return embeddings
    
    async def store_naive_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> bool:
        """Store naive chunks in Qdrant collection"""
        try:
            # Convert chunks to dictionaries
            chunk_dicts = [chunk.to_dict() for chunk in chunks]
            
            # Add token count to each chunk
            for i, chunk_dict in enumerate(chunk_dicts):
                chunk_dict["tokens"] = len(self.naive_chunker.tokenize(chunk_dict["content"]))
            
            success = await self.collection_manager.upsert_naive_chunks(chunk_dicts, embeddings)
            
            if success:
                logger.info(f"Successfully stored {len(chunks)} naive chunks")
            else:
                logger.error("Failed to store naive chunks")
                
            return success
            
        except Exception as e:
            logger.error(f"Error storing naive chunks: {e}")
            return False
    
    async def store_advanced_chunks(self, parents: List[ParentChunk], embeddings: List[List[float]]) -> bool:
        """Store advanced chunks in Qdrant collection"""
        try:
            # Convert parents to dictionaries
            parent_dicts = [parent.to_dict() for parent in parents]
            
            success = await self.collection_manager.upsert_advanced_chunks(parent_dicts, embeddings)
            
            if success:
                total_children = sum(len(parent.children) for parent in parents)
                logger.info(f"Successfully stored {len(parents)} parents with {total_children} children")
            else:
                logger.error("Failed to store advanced chunks")
                
            return success
            
        except Exception as e:
            logger.error(f"Error storing advanced chunks: {e}")
            return False
    
    def compare_chunking_strategies(self, naive_stats: Dict[str, Any], 
                                  advanced_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Compare the two chunking strategies"""
        comparison = {
            "chunk_count_difference": 0,
            "token_efficiency": {},
            "strategy_recommendations": []
        }
        
        try:
            # Compare chunk counts
            naive_chunks = naive_stats.get("total_chunks", 0)
            advanced_children = advanced_stats.get("total_children", 0)
            
            comparison["chunk_count_difference"] = naive_chunks - advanced_children
            
            # Compare token efficiency
            naive_avg_tokens = naive_stats.get("avg_tokens_per_chunk", 0)
            advanced_avg_tokens = advanced_stats.get("avg_child_tokens", 0)
            
            comparison["token_efficiency"] = {
                "naive_avg_tokens": naive_avg_tokens,
                "advanced_avg_tokens": advanced_avg_tokens,
                "efficiency_ratio": advanced_avg_tokens / naive_avg_tokens if naive_avg_tokens > 0 else 0
            }
            
            # Generate recommendations
            if advanced_children < naive_chunks:
                comparison["strategy_recommendations"].append(
                    "Advanced chunking created fewer, potentially more meaningful chunks"
                )
            
            if advanced_avg_tokens > naive_avg_tokens:
                comparison["strategy_recommendations"].append(
                    "Advanced chunking maintains larger context windows per chunk"
                )
            
            # Check for parent-child benefits
            if advanced_stats.get("total_parents", 0) > 0:
                comparison["strategy_recommendations"].append(
                    "Advanced chunking enables hierarchical context retrieval"
                )
                
        except Exception as e:
            logger.error(f"Error comparing chunking strategies: {e}")
            comparison["error"] = str(e)
        
        return comparison
    
    async def setup_collections(self) -> Dict[str, Any]:
        """Initialize both Qdrant collections"""
        try:
            results = await self.collection_manager.create_collections()
            logger.info(f"Collection setup results: {results}")
            return {
                "success": True,
                "results": results
            }
        except Exception as e:
            logger.error(f"Failed to setup collections: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the pipeline"""
        health = {
            "pipeline_ready": False,
            "components": {}
        }
        
        try:
            # Check collection manager
            collection_health = await self.collection_manager.health_check()
            health["components"]["collections"] = collection_health
            
            # Check embedding service
            try:
                test_embedding = await self.embedder.embed("test query")
                health["components"]["embeddings"] = {
                    "available": True,
                    "dimensions": len(test_embedding)
                }
            except Exception as e:
                health["components"]["embeddings"] = {
                    "available": False,
                    "error": str(e)
                }
            
            # Check chunkers
            health["components"]["chunkers"] = {
                "naive_ready": self.naive_chunker is not None,
                "advanced_ready": self.advanced_chunker is not None
            }
            
            # Overall pipeline readiness
            pipeline_components_ready = (
                collection_health.get("client_connected", False) and
                health["components"]["embeddings"]["available"] and
                health["components"]["chunkers"]["naive_ready"] and
                health["components"]["chunkers"]["advanced_ready"]
            )
            
            health["pipeline_ready"] = pipeline_components_ready
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health["error"] = str(e)
        
        return health
    
    async def process_multiple_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process multiple documents in batch"""
        results = {
            "total_documents": len(file_paths),
            "successful": 0,
            "failed": 0,
            "details": []
        }
        
        for file_path in file_paths:
            try:
                result = await self.process_document(file_path)
                results["details"].append(result)
                
                if result.get("success", False):
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results["failed"] += 1
                results["details"].append({
                    "success": False,
                    "file_path": file_path,
                    "error": str(e)
                })
        
        return results