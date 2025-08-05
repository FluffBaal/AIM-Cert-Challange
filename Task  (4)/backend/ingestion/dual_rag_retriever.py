"""
DualRAGRetriever with Context Corridor and Cohere Reranking

Implements the advanced retrieval system with:
- Dual collection support (naive vs advanced)
- Context corridor logic for advanced mode
- Cohere reranking for improved relevance
- Graceful fallback handling
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.models import ScoredPoint
import cohere
from openai import OpenAI

try:
    from .constants import (
        NAIVE_COLLECTION, ADVANCED_COLLECTION,
        CONTEXT_CORRIDOR_THRESHOLD, NAIVE_RETRIEVAL_LIMIT,
        ADVANCED_RETRIEVAL_LIMIT, FINAL_RESULTS_LIMIT,
        ADVANCED_RESULTS_LIMIT, COHERE_MODEL
    )
except ImportError:
    from constants import (
        NAIVE_COLLECTION, ADVANCED_COLLECTION,
        CONTEXT_CORRIDOR_THRESHOLD, NAIVE_RETRIEVAL_LIMIT,
        ADVANCED_RETRIEVAL_LIMIT, FINAL_RESULTS_LIMIT,
        ADVANCED_RESULTS_LIMIT, COHERE_MODEL
    )

logger = logging.getLogger(__name__)


@dataclass
class DetailedContext:
    """Detailed information about a retrieved context"""
    content: str
    source: str
    chapter: Optional[str] = None
    relevance_score: float = 0.0
    technique: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class RetrievalResult:
    """Result from retrieval operation"""
    success: bool
    retrieved_contexts: List[str]
    retrieval_time: float
    avg_similarity_score: Optional[float]
    context_corridor_used: bool = False
    detailed_contexts: List[DetailedContext] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EmbeddingService:
    """Service for generating embeddings using OpenAI"""
    
    def __init__(self, model: str = "text-embedding-3-small", openai_api_key: Optional[str] = None):
        # Use provided API key or fall back to environment variable for backward compatibility
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided either as parameter or environment variable")
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise


class DualRAGRetriever:
    """
    Retrieval with Context Corridor and Reranking Implementation
    
    Supports both naive and advanced retrieval modes with Cohere reranking
    """
    
    def __init__(self, qdrant_host: str = "qdrant", qdrant_port: int = 6333, 
                 openai_api_key: Optional[str] = None, cohere_api_key: Optional[str] = None):
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.embedder = EmbeddingService(openai_api_key=openai_api_key)
        
        # Initialize Cohere reranker with provided key only
        cohere_key = cohere_api_key
        try:
            # Try ClientV2 first (for newer versions)
            if cohere_key:
                try:
                    self.cohere_client = cohere.ClientV2(
                        api_key=cohere_key,
                        timeout=30
                    )
                    logger.info("Initialized Cohere ClientV2 successfully")
                except AttributeError:
                    # Fall back to Client for older versions
                    self.cohere_client = cohere.Client(
                        api_key=cohere_key,
                        timeout=30
                    )
                    logger.info("Initialized Cohere Client (v1) successfully")
            else:
                self.cohere_client = None
                logger.warning("No Cohere API key provided, reranking will be disabled")
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")
            self.cohere_client = None
    
    async def retrieve(self, query: str, mode: str) -> RetrievalResult:
        """
        Main retrieval method supporting both naive and advanced modes
        
        Args:
            query: Search query
            mode: "naive" or "advanced"
            
        Returns:
            RetrievalResult with contexts and metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            if mode == "naive":
                result = await self._retrieve_naive(query)
            elif mode == "advanced":
                result = await self._retrieve_advanced(query)
            else:
                return RetrievalResult(
                    success=False,
                    retrieved_contexts=[],
                    retrieval_time=0,
                    avg_similarity_score=None,
                    error_message=f"Unknown mode: {mode}"
                )
            
            end_time = asyncio.get_event_loop().time()
            result.retrieval_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            return result
            
        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            logger.error(f"Retrieval failed for mode {mode}: {e}")
            return RetrievalResult(
                success=False,
                retrieved_contexts=[],
                retrieval_time=(end_time - start_time) * 1000,
                avg_similarity_score=None,
                error_message=str(e)
            )
    
    async def _retrieve_naive(self, query: str) -> RetrievalResult:
        """Naive retrieval with simple similarity search and reranking"""
        try:
            # Generate query embedding
            query_vector = await self.embedder.embed(query)
            
            # Simple similarity search with more candidates for reranking
            results = self.qdrant.search(
                collection_name=NAIVE_COLLECTION,
                query_vector=query_vector,
                limit=NAIVE_RETRIEVAL_LIMIT  # Retrieve more candidates
            )
            
            if not results:
                return RetrievalResult(
                    success=True,
                    retrieved_contexts=[],
                    retrieval_time=0,
                    avg_similarity_score=None
                )
            
            # Rerank results
            reranked_results = await self.rerank_results(query, results)
            final_results = reranked_results[:FINAL_RESULTS_LIMIT]  # Top 5 after reranking
            
            # Extract contexts and calculate metrics
            contexts = [result.payload.get("content", "") for result in final_results]
            scores = [result.score for result in final_results if hasattr(result, 'score')]
            avg_score = sum(scores) / len(scores) if scores else None
            
            # Create detailed contexts
            detailed_contexts = []
            for i, result in enumerate(final_results):
                payload = result.payload if hasattr(result, 'payload') else {}
                score = scores[i] if i < len(scores) else 0.0
                
                # Extract technique from content if possible
                content = payload.get("content", "")
                technique = None
                for tech in ["mirroring", "labeling", "tactical_empathy", "calibrated_questions", "anchoring", "loss_aversion"]:
                    if tech.replace("_", " ") in content.lower():
                        technique = tech
                        break
                
                detailed_ctx = DetailedContext(
                    content=content,
                    source=payload.get("source", "Never Split the Difference"),
                    chapter=payload.get("chapter"),
                    relevance_score=score,
                    technique=technique,
                    metadata={
                        "chunk_id": payload.get("chunk_id"),
                        "page": payload.get("page")
                    }
                )
                detailed_contexts.append(detailed_ctx)
            
            return RetrievalResult(
                success=True,
                retrieved_contexts=contexts,
                retrieval_time=0,  # Will be set by parent method
                avg_similarity_score=avg_score,
                context_corridor_used=False,
                detailed_contexts=detailed_contexts,
                metadata={
                    "total_candidates": len(results),
                    "reranked_results": len(reranked_results),
                    "final_results": len(final_results)
                }
            )
            
        except Exception as e:
            logger.error(f"Naive retrieval failed: {e}")
            raise
    
    async def _retrieve_advanced(self, query: str) -> RetrievalResult:
        """Advanced retrieval with parent-child structure and context corridor"""
        try:
            query_vector = await self.embedder.embed(query)
            
            # Retrieve more candidates for reranking from advanced collection
            initial_results = self.qdrant.search(
                collection_name=ADVANCED_COLLECTION,
                query_vector=query_vector,
                limit=ADVANCED_RETRIEVAL_LIMIT * 3  # Get more results to dedupe manually
            )
            
            logger.info(f"Advanced search returned {len(initial_results)} results")
            
            if not initial_results:
                return RetrievalResult(
                    success=True,
                    retrieved_contexts=[],
                    retrieval_time=0,
                    avg_similarity_score=None,
                    context_corridor_used=False
                )
            
            # Manually deduplicate by parent_id
            seen_parents = set()
            deduplicated_results = []
            for result in initial_results:
                parent_id = result.payload.get("parent_id") if hasattr(result, 'payload') else None
                if parent_id and parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    deduplicated_results.append(result)
                    if len(deduplicated_results) >= ADVANCED_RETRIEVAL_LIMIT:
                        break
            
            logger.info(f"After deduplication: {len(deduplicated_results)} unique parent results")
            
            # Rerank parent documents
            reranked_parents = await self.rerank_results(query, deduplicated_results)
            
            # Apply context corridor logic on reranked results
            enhanced_results = []
            context_corridor_used = False
            
            logger.info(f"Processing top {ADVANCED_RESULTS_LIMIT} reranked results")
            
            for i, result in enumerate(reranked_parents[:ADVANCED_RESULTS_LIMIT]):  # Top 3 reranked
                enhanced_results.append(result)
                
                # Check if preceding parent's last child has high similarity
                if i > 0:
                    prev_parent_id = reranked_parents[i-1].payload.get("parent_id")
                    if prev_parent_id:
                        try:
                            last_child = await self.get_last_child_of_parent(prev_parent_id)
                            
                            if last_child:
                                similarity = self.cosine_similarity(
                                    last_child.vector if hasattr(last_child, 'vector') else [],
                                    query_vector
                                )
                                
                                if similarity >= CONTEXT_CORRIDOR_THRESHOLD:
                                    # Prepend the preceding parent for context
                                    preceding_parent = await self.get_parent_document(prev_parent_id)
                                    if preceding_parent:
                                        enhanced_results.insert(-1, preceding_parent)
                                        context_corridor_used = True
                                        
                        except Exception as e:
                            logger.warning(f"Context corridor failed for parent {prev_parent_id}: {e}")
            
            # Extract contexts and calculate metrics
            contexts = []
            scores = []
            detailed_contexts = []
            
            logger.info(f"Extracting contexts from {len(enhanced_results)} enhanced results")
            
            for i, result in enumerate(enhanced_results):
                payload = result.payload if hasattr(result, 'payload') else {}
                content = payload.get("content", "")
                section_heading = payload.get("section_heading", "")
                
                # Combine section heading with content for better context
                if section_heading:
                    full_content = f"Section: {section_heading}\n\n{content}"
                else:
                    full_content = content
                    
                contexts.append(full_content)
                
                score = result.score if hasattr(result, 'score') else 0.0
                scores.append(score)
                
                # Extract technique from content if possible
                technique = None
                for tech in ["mirroring", "labeling", "tactical_empathy", "calibrated_questions", "anchoring", "loss_aversion"]:
                    if tech.replace("_", " ") in content.lower():
                        technique = tech
                        break
                
                detailed_ctx = DetailedContext(
                    content=full_content,
                    source=payload.get("source", "Never Split the Difference"),
                    chapter=payload.get("chapter") or section_heading,
                    relevance_score=score,
                    technique=technique,
                    metadata={
                        "parent_id": payload.get("parent_id"),
                        "chunk_id": payload.get("chunk_id"),
                        "page": payload.get("page"),
                        "section": section_heading,
                        "context_corridor": i > 0 and context_corridor_used  # Mark if added via corridor
                    }
                )
                detailed_contexts.append(detailed_ctx)
            
            avg_score = sum(scores) / len(scores) if scores else None
            
            return RetrievalResult(
                success=True,
                retrieved_contexts=contexts,
                retrieval_time=0,  # Will be set by parent method
                avg_similarity_score=avg_score,
                context_corridor_used=context_corridor_used,
                detailed_contexts=detailed_contexts,
                metadata={
                    "total_candidates": len(initial_results),
                    "reranked_parents": len(reranked_parents),
                    "enhanced_results": len(enhanced_results),
                    "context_corridor_triggered": context_corridor_used
                }
            )
            
        except Exception as e:
            logger.error(f"Advanced retrieval failed: {e}")
            raise
    
    async def rerank_results(self, query: str, results: List[ScoredPoint]) -> List[ScoredPoint]:
        """
        Rerank results using Cohere's rerank API for better relevance
        """
        if not results:
            return results
        
        # If Cohere client is not available, return original results
        if not self.cohere_client:
            logger.warning("Cohere reranking unavailable, returning original order")
            return results
            
        # Extract texts from results - include section heading for context
        documents = []
        for result in results:
            content = result.payload.get("content", "")
            section_heading = result.payload.get("section_heading", "")
            
            # Combine section heading with content for better reranking context
            if section_heading:
                document = f"Section: {section_heading}\n\n{content}"
            else:
                document = content
                
            documents.append(document)
        
        try:
            # Use Cohere rerank with the latest model
            rerank_response = self.cohere_client.rerank(
                query=query,
                documents=documents,
                top_n=len(documents),
                model=COHERE_MODEL  # rerank-v3.5
                # Note: return_documents parameter not supported in older client versions
            )
            
            # Sort results based on reranking scores
            # Create list of (result, relevance_score) tuples
            scored_results = []
            for hit in rerank_response.results:
                result = results[hit.index]
                scored_results.append((result, hit.relevance_score))
            
            # Sort by relevance score (highest first)
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # Extract just the results in reranked order
            reranked_results = [result for result, _ in scored_results]
            
            # Log reranking impact for monitoring
            original_top = results[0].payload.get('parent_id', 'unknown')
            reranked_top = reranked_results[0].payload.get('parent_id', 'unknown')
            
            logger.info(f"Reranking impact - Top result changed: {original_top != reranked_top}")
            
            return reranked_results
            
        except Exception as e:
            logger.warning(f"Cohere reranking failed: {type(e).__name__}: {e}")
            logger.debug(f"Cohere client type: {type(self.cohere_client)}")
            logger.debug(f"Model used: {COHERE_MODEL}")
            return results
    
    async def get_last_child_of_parent(self, parent_id: str) -> Optional[ScoredPoint]:
        """Get the last child of a specific parent"""
        try:
            results = self.qdrant.search(
                collection_name=ADVANCED_COLLECTION,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_id",
                            match=MatchValue(value=parent_id)
                        )
                    ]
                ),
                limit=100,  # Get all children
                query_vector=[0.0] * 1536  # Dummy vector since we're filtering, not searching
            )
            
            if not results:
                return None
            
            # Sort by child_idx to get the last child
            sorted_children = sorted(results, key=lambda x: x.payload.get("child_idx", 0))
            return sorted_children[-1] if sorted_children else None
            
        except Exception as e:
            logger.error(f"Failed to get last child of parent {parent_id}: {e}")
            return None
    
    async def get_parent_document(self, parent_id: str) -> Optional[ScoredPoint]:
        """Get the full parent document by parent_id"""
        try:
            results = self.qdrant.search(
                collection_name=ADVANCED_COLLECTION,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_id",
                            match=MatchValue(value=parent_id)
                        )
                    ]
                ),
                limit=1,
                query_vector=[0.0] * 1536  # Dummy vector since we're filtering
            )
            
            return results[0] if results else None
            
        except Exception as e:
            logger.error(f"Failed to get parent document {parent_id}: {e}")
            return None
    
    def cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vector1 or not vector2:
            return 0.0
            
        try:
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of retrieval system"""
        health = {
            "qdrant_connected": False,
            "cohere_available": False,
            "openai_available": False,
            "collections": {}
        }
        
        try:
            # Test Qdrant connection
            collections = self.qdrant.get_collections()
            health["qdrant_connected"] = True
            
            # Check collections exist
            collection_names = [col.name for col in collections.collections]
            health["collections"]["naive_exists"] = NAIVE_COLLECTION in collection_names
            health["collections"]["advanced_exists"] = ADVANCED_COLLECTION in collection_names
            
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            health["qdrant_error"] = str(e)
        
        try:
            # Test Cohere
            if self.cohere_client:
                # Simple test query - this will fail gracefully if API key is missing
                test_docs = ["test document"]
                self.cohere_client.rerank(
                    query="test",
                    documents=test_docs,
                    top_n=1,
                    model=COHERE_MODEL
                )
                health["cohere_available"] = True
            else:
                health["cohere_available"] = False
                health["cohere_error"] = "Cohere client not initialized (no API key provided)"
        except Exception as e:
            logger.warning(f"Cohere health check failed: {e}")
            health["cohere_error"] = str(e)
        
        try:
            # Test OpenAI embedding
            await self.embedder.embed("test query")
            health["openai_available"] = True
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            health["openai_error"] = str(e)
        
        return health