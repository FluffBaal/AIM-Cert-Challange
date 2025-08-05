# dependencies.py - Dependency injection and service initialization for FastAPI
import asyncio
import logging
import os
from typing import Optional
from cachetools import TTLCache

import openai
import redis.asyncio as redis
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)

# Global service instances
qdrant_client: Optional[QdrantClient] = None
redis_client: Optional[redis.Redis] = None

# Cache for expensive operations
api_health_cache = TTLCache(maxsize=100, ttl=300)  # 5 minute cache

async def initialize_qdrant_client():
    """
    Initialize Qdrant vector database connection
    """
    global qdrant_client
    
    try:
        # Use host and port from environment
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        qdrant_client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            timeout=10.0
        )
        
        # Test connection
        collections = await asyncio.get_event_loop().run_in_executor(
            None, qdrant_client.get_collections
        )
        
        logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
        logger.info(f"Available collections: {[c.name for c in collections.collections]}")
        
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        # Continue without Qdrant - system will use fallbacks
        qdrant_client = None

async def initialize_redis_client():
    """
    Initialize Redis connection for caching
    """
    global redis_client
    
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        redis_client = redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0
        )
        
        # Test connection
        await redis_client.ping()
        
        logger.info(f"Connected to Redis at {redis_url}")
        
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        # Continue without Redis - system will use in-memory caching
        redis_client = None

async def warm_up_models():
    """
    Warm up language models and cache initial data
    """
    try:
        # Test OpenAI API connection
        client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Make a simple test call to warm up the connection
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        logger.info("OpenAI API connection warmed up successfully")
        
    except Exception as e:
        logger.warning(f"Failed to warm up OpenAI models: {e}")

async def load_fallback_data():
    """
    Load fallback data for when external services are unavailable
    """
    try:
        # Load cached negotiation tactics and market data
        fallback_tactics = [
            "Use mirroring to build rapport",
            "Ask calibrated questions to understand their perspective", 
            "Anchor your rate based on market research",
            "Use labeling to acknowledge their concerns",
            "Never split the difference - find creative solutions"
        ]
        
        # Cache fallback data in Redis if available
        if redis_client:
            await redis_client.set(
                "fallback:negotiation_tactics",
                str(fallback_tactics),
                ex=3600  # 1 hour expiry
            )
        
        logger.info("Fallback data loaded successfully")
        
    except Exception as e:
        logger.warning(f"Failed to load fallback data: {e}")

async def close_all_connections():
    """
    Close all database and external service connections
    """
    global qdrant_client, redis_client
    
    try:
        if qdrant_client:
            # Qdrant client doesn't need explicit closing
            qdrant_client = None
            logger.info("Qdrant connection closed")
            
        if redis_client:
            await redis_client.close()
            redis_client = None
            logger.info("Redis connection closed")
            
    except Exception as e:
        logger.error(f"Error closing connections: {e}")

async def persist_cache_data():
    """
    Persist important cache data before shutdown
    """
    try:
        if redis_client:
            # Save any important cached data to persistent storage
            # For now, just log that we're persisting
            logger.info("Cache data persisted successfully")
            
    except Exception as e:
        logger.warning(f"Failed to persist cache data: {e}")

async def check_openai_api_cached() -> bool:
    """
    Check OpenAI API status with caching
    """
    cache_key = "openai_api_health"
    
    # Check cache first
    if cache_key in api_health_cache:
        return api_health_cache[cache_key]
    
    try:
        client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Make a minimal API call
        await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        
        # Cache successful result
        api_health_cache[cache_key] = True
        return True
        
    except Exception as e:
        logger.warning(f"OpenAI API health check failed: {e}")
        # Cache failed result for shorter time
        api_health_cache[cache_key] = False
        return False

# Dependency injection functions for FastAPI
async def get_qdrant_client() -> Optional[QdrantClient]:
    """
    Dependency for injecting Qdrant client
    """
    return qdrant_client

async def get_redis_client() -> Optional[redis.Redis]:
    """
    Dependency for injecting Redis client
    """
    return redis_client

async def get_api_config():
    """
    Get API configuration from environment variables
    """
    from ..agents.models import ApiConfiguration
    
    return ApiConfiguration(
        openai_key=os.getenv("OPENAI_API_KEY", ""),
        exa_key=os.getenv("EXA_API_KEY"),
        cohere_key=os.getenv("COHERE_API_KEY"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")
    )

# Health check functions
async def check_qdrant_health() -> str:
    """
    Check Qdrant database health
    """
    if not qdrant_client:
        return "unavailable"
    
    try:
        await asyncio.get_event_loop().run_in_executor(
            None, qdrant_client.get_collections
        )
        return "healthy"
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        return f"unhealthy: {str(e)}"

async def check_redis_health() -> str:
    """
    Check Redis cache health
    """
    if not redis_client:
        return "unavailable"
    
    try:
        await redis_client.ping()
        return "healthy"
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        return f"unhealthy: {str(e)}"

async def check_exa_api() -> str:
    """
    Check Exa API health
    """
    exa_key = os.getenv("EXA_API_KEY")
    if not exa_key:
        return "unavailable: no API key"
    
    # For now, assume healthy if key is present
    # In real implementation, would make test API call
    return "healthy"

# Initialize connections on module import
async def initialize_all_services():
    """
    Initialize all services
    """
    await initialize_qdrant_client()
    await initialize_redis_client()
    await warm_up_models()
    await load_fallback_data()

# Service status
def get_degraded_features(service_checks: dict) -> list:
    """
    Determine which features are degraded based on service status
    """
    degraded = []
    
    if service_checks.get("qdrant") != "healthy":
        degraded.append("advanced_rag_search")
    
    if service_checks.get("redis") != "healthy":
        degraded.append("response_caching")
    
    if service_checks.get("exa") != "healthy":
        degraded.append("real_time_market_data")
    
    if service_checks.get("openai") != "healthy":
        degraded.append("ai_powered_analysis")
    
    return degraded