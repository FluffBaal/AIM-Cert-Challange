"""
Dual Collection Vector Storage Schema

Implements the configuration and setup for both naive and advanced
Qdrant collections as specified in the build plan.
"""

from typing import Dict, Any, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionConfig, VectorParams, Distance, PointStruct,
    CreateCollection, UpdateCollection
)
import logging
try:
    from .constants import (
        NAIVE_COLLECTION, ADVANCED_COLLECTION, 
        EMBEDDING_DIMENSIONS
    )
except ImportError:
    from constants import (
        NAIVE_COLLECTION, ADVANCED_COLLECTION, 
        EMBEDDING_DIMENSIONS
    )

logger = logging.getLogger(__name__)


class QdrantCollectionManager:
    """Manages the dual collection setup for naive and advanced chunking strategies"""
    
    def __init__(self, host: str = "qdrant", port: int = 6333):
        """Initialize Qdrant client"""
        self.client = QdrantClient(host=host, port=port)
        self.naive_collection = NAIVE_COLLECTION
        self.advanced_collection = ADVANCED_COLLECTION
    
    def get_naive_collection_config(self) -> Dict[str, Any]:
        """Get naive collection configuration"""
        return {
            "collection_name": self.naive_collection,
            "vectors_config": VectorParams(
                size=EMBEDDING_DIMENSIONS,  # text-embedding-3-small
                distance=Distance.COSINE
            ),
            "payload_schema": {
                "chunk_id": "keyword",
                "content": "text",
                "chunk_index": "integer",
                "overlap_start": "integer",
                "overlap_end": "integer",
                "method": "keyword",
                "tokens": "integer"
            }
        }
    
    def get_advanced_collection_config(self) -> Dict[str, Any]:
        """Get advanced collection configuration with parent-child structure"""
        return {
            "collection_name": self.advanced_collection,
            "vectors_config": VectorParams(
                size=EMBEDDING_DIMENSIONS,  # text-embedding-3-small
                distance=Distance.COSINE
            ),
            "payload_schema": {
                "parent_id": "keyword",
                "section_heading": "text",
                "anchor_id": "keyword",
                "child_idx": "integer",
                "content": "text",
                "parent_text": "text",  # Full parent content
                "metadata": "json",
                "tokens": "integer",
                "method": "keyword"
            }
        }
    
    async def create_collections(self) -> Dict[str, bool]:
        """
        Create both collections if they don't exist
        
        Returns:
            Dict with creation status for each collection
        """
        results = {}
        
        # Create naive collection
        try:
            naive_config = self.get_naive_collection_config()
            
            # Check if collection exists
            collections = self.client.get_collections()
            naive_exists = any(col.name == self.naive_collection for col in collections.collections)
            
            if not naive_exists:
                self.client.create_collection(
                    collection_name=self.naive_collection,
                    vectors_config=naive_config["vectors_config"]
                )
                logger.info(f"Created naive collection: {self.naive_collection}")
                results[self.naive_collection] = True
            else:
                logger.info(f"Naive collection already exists: {self.naive_collection}")
                results[self.naive_collection] = False
                
        except Exception as e:
            logger.error(f"Failed to create naive collection: {e}")
            results[self.naive_collection] = False
        
        # Create advanced collection
        try:
            advanced_config = self.get_advanced_collection_config()
            
            # Check if collection exists
            collections = self.client.get_collections()
            advanced_exists = any(col.name == self.advanced_collection for col in collections.collections)
            
            if not advanced_exists:
                self.client.create_collection(
                    collection_name=self.advanced_collection,
                    vectors_config=advanced_config["vectors_config"]
                )
                logger.info(f"Created advanced collection: {self.advanced_collection}")
                results[self.advanced_collection] = True
            else:
                logger.info(f"Advanced collection already exists: {self.advanced_collection}")
                results[self.advanced_collection] = False
                
        except Exception as e:
            logger.error(f"Failed to create advanced collection: {e}")
            results[self.advanced_collection] = False
        
        return results
    
    async def delete_collections(self) -> Dict[str, bool]:
        """Delete both collections - useful for testing/reset"""
        results = {}
        
        try:
            self.client.delete_collection(self.naive_collection)
            logger.info(f"Deleted naive collection: {self.naive_collection}")
            results[self.naive_collection] = True
        except Exception as e:
            logger.warning(f"Failed to delete naive collection: {e}")
            results[self.naive_collection] = False
        
        try:
            self.client.delete_collection(self.advanced_collection)
            logger.info(f"Deleted advanced collection: {self.advanced_collection}")
            results[self.advanced_collection] = True
        except Exception as e:
            logger.warning(f"Failed to delete advanced collection: {e}")
            results[self.advanced_collection] = False
        
        return results
    
    async def upsert_naive_chunks(self, chunks: List[Dict[str, Any]], 
                                 vectors: List[List[float]]) -> bool:
        """
        Upsert naive chunks to the naive collection
        
        Args:
            chunks: List of chunk dictionaries with metadata
            vectors: Corresponding embedding vectors
            
        Returns:
            Success status
        """
        try:
            points = []
            # Get current point count to continue numbering
            try:
                collection_info = self.client.get_collection(self.naive_collection)
                start_id = collection_info.points_count
            except:
                start_id = 0
                
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                point = PointStruct(
                    id=start_id + i,  # Continue from existing points
                    vector=vector,
                    payload={
                        "chunk_id": chunk.get("chunk_id"),
                        "content": chunk.get("content"),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "overlap_start": chunk.get("overlap_start", 0),
                        "overlap_end": chunk.get("overlap_end", 0),
                        "method": "naive",
                        "tokens": chunk.get("tokens", 0)
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.naive_collection,
                points=points
            )
            
            logger.info(f"Upserted {len(points)} naive chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert naive chunks: {e}")
            return False
    
    async def upsert_advanced_chunks(self, parent_chunks: List[Dict[str, Any]], 
                                   child_vectors: List[List[float]]) -> bool:
        """
        Upsert advanced parent-child chunks to the advanced collection
        
        Args:
            parent_chunks: List of parent chunk dictionaries with children
            child_vectors: Embedding vectors for child chunks
            
        Returns:
            Success status
        """
        try:
            points = []
            vector_idx = 0
            # Get current point count to continue numbering
            try:
                collection_info = self.client.get_collection(self.advanced_collection)
                point_id = collection_info.points_count
            except:
                point_id = 0
            
            for parent in parent_chunks:
                for child in parent.get("children", []):
                    if vector_idx < len(child_vectors):
                        point = PointStruct(
                            id=point_id,  # Use integer ID
                            vector=child_vectors[vector_idx],
                            payload={
                                "parent_id": parent["parent_id"],
                                "section_heading": parent["section_heading"],
                                "anchor_id": parent["anchor_id"],
                                "child_idx": child["child_idx"],
                                "content": child["content"],
                                "parent_text": parent["content"],
                                "metadata": parent.get("metadata", {}),
                                "tokens": child.get("tokens", 0),
                                "method": "advanced",
                                "unique_id": f"{parent['parent_id']}_{child['child_idx']}"  # Store original ID as metadata
                            }
                        )
                        points.append(point)
                        vector_idx += 1
                        point_id += 1  # Increment point ID
            
            self.client.upsert(
                collection_name=self.advanced_collection,
                points=points
            )
            
            logger.info(f"Upserted {len(points)} advanced chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert advanced chunks: {e}")
            return False
    
    async def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a collection"""
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance
                }
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for {collection_name}: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of both collections"""
        health = {
            "client_connected": False,
            "collections": {}
        }
        
        try:
            # Test client connection
            collections = self.client.get_collections()
            health["client_connected"] = True
            
            # Check each collection
            for collection_name in [self.naive_collection, self.advanced_collection]:
                info = await self.get_collection_info(collection_name)
                health["collections"][collection_name] = {
                    "exists": info is not None,
                    "info": info
                }
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health["error"] = str(e)
        
        return health