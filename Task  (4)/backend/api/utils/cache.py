"""Simple caching utilities for negotiation responses"""

import hashlib
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import redis
import logging

logger = logging.getLogger(__name__)

class ResponseCache:
    """Cache negotiation responses to improve performance"""
    
    def __init__(self, redis_url: str = "redis://redis:6379"):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.enabled = True
            self.ttl = 3600  # 1 hour cache
        except Exception as e:
            logger.warning(f"Redis not available, caching disabled: {e}")
            self.redis_client = None
            self.enabled = False
    
    def _get_cache_key(self, request_text: str, context: Dict[str, Any], mode: str) -> str:
        """Generate a cache key from request parameters"""
        cache_data = {
            "text": request_text,
            "context": context,
            "mode": mode
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return f"negotiation:v1:{hashlib.md5(cache_str.encode()).hexdigest()}"
    
    async def get(self, request_text: str, context: Dict[str, Any], mode: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available"""
        if not self.enabled:
            return None
            
        try:
            key = self._get_cache_key(request_text, context, mode)
            cached = self.redis_client.get(key)
            
            if cached:
                logger.info(f"Cache hit for negotiation request")
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    async def set(self, request_text: str, context: Dict[str, Any], mode: str, response: Dict[str, Any]):
        """Cache a response"""
        if not self.enabled:
            return
            
        try:
            key = self._get_cache_key(request_text, context, mode)
            self.redis_client.setex(
                key,
                self.ttl,
                json.dumps(response)
            )
            logger.info(f"Cached negotiation response")
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def clear_pattern(self, pattern: str = "negotiation:*"):
        """Clear cached responses matching pattern"""
        if not self.enabled:
            return
            
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cached responses")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

# Global cache instance
response_cache = ResponseCache()