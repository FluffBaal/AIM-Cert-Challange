# error_handler.py - Comprehensive error handling for agent system
import asyncio
import time
import random
import re
import json
import hashlib
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Callable, List
from cachetools import TTLCache

import openai

from .models import (
    NegotiationRequest, 
    NegotiationResponse, 
    AgentExecutionError,
    RateLimitExceeded,
    CircuitBreakerOpenError
)

logger = logging.getLogger(__name__)

class ErrorHandler:
    """
    Centralized error handling with multiple fallback strategies
    """
    
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.circuit_breakers = {}
        self.fallback_cache = TTLCache(maxsize=1000, ttl=900)  # 15 min cache
        self.model_fallbacks = {
            "gpt-4": ["gpt-3.5-turbo", "gpt-4o-mini"],
            "gpt-4o-mini": ["gpt-3.5-turbo", "gpt-4"]
        }
    
    async def handle_supervisor_failure(self, error: Exception, request: NegotiationRequest, mode: str = "fallback") -> NegotiationResponse:
        """
        Handle complete supervisor agent failure
        """
        logger.error(f"Supervisor failure: {error}", exc_info=True)
        
        # Try simplified sequential execution
        try:
            # Direct RAG search only
            rag_result = await self.execute_single_agent_fallback("rag_search", request)
            
            return NegotiationResponse(
                success=True,
                degraded=True,
                partial_failure=True,
                rewritten_text=self.create_template_rewrite(request.negotiation_text),
                insights=rag_result.get("insights", []),
                strategy={"techniques": ["basic_negotiation"], "confidence": 0.5},
                error_recovery="supervisor_fallback",
                request_id=str(hash(request.negotiation_text)),
                confidence=0.5,
                mode=f"{mode}_fallback" if mode != "fallback" else "fallback"
            )
        except Exception as e:
            # Last resort - return template response
            return self.create_emergency_response(request, mode)
    
    async def handle_llm_failure(self, error: Exception, agent_name: str, retry_context: dict) -> dict:
        """
        Comprehensive LLM failure handling with multiple strategies
        """
        error_type = type(error).__name__
        self.error_counts[f"{agent_name}_{error_type}"] += 1
        
        # Check circuit breaker
        if self.is_circuit_open(agent_name):
            return await self.get_cached_or_default(agent_name, retry_context)
        
        if isinstance(error, openai.RateLimitError):
            # Strategy 1: Use fallback model
            fallback_model = await self.get_fallback_model(retry_context.get("model"))
            if fallback_model:
                retry_context["model"] = fallback_model
                return await self.retry_with_fallback_model(retry_context)
            
            # Strategy 2: Add delay and retry
            await asyncio.sleep(self.calculate_backoff_delay(agent_name))
            return await self.retry_with_backoff(retry_context)
            
        elif isinstance(error, openai.APITimeoutError):
            # Strategy 1: Retry with increased timeout
            retry_context["timeout"] = retry_context.get("timeout", 30) * 1.5
            
            # Strategy 2: Simplify prompt if possible
            if "prompt" in retry_context:
                retry_context["prompt"] = self.simplify_prompt(retry_context["prompt"])
            
            return await self.retry_with_backoff(retry_context, max_attempts=2)
            
        elif isinstance(error, openai.APIConnectionError):
            # Check if API is reachable
            if not await self.check_api_health():
                # Use offline fallback
                return await self.get_offline_fallback(agent_name, retry_context)
            
            # Retry with new connection
            return await self.retry_with_new_client(retry_context)
            
        elif isinstance(error, openai.AuthenticationError):
            # Critical error - notify and use cached results
            await self.notify_auth_failure(agent_name)
            return await self.get_cached_or_default(agent_name, retry_context)
            
        else:
            # Unknown error - log and use fallback
            logger.error(f"Unknown LLM error in {agent_name}: {error}")
            return await self.get_generic_fallback(agent_name, retry_context)
    
    async def handle_vector_db_failure(self, error: Exception, operation: str) -> Any:
        """
        Handle Qdrant/vector database failures
        """
        logger.warning(f"Vector DB {operation} failed: {error}")
        
        # Strategy 1: Retry with connection pool refresh
        if "connection" in str(error).lower():
            await self.refresh_qdrant_connection()
            try:
                return await self.retry_vector_operation(operation)
            except Exception as e:
                logger.error(f"Retry failed: {e}")
        
        # Strategy 2: Use backup vector store (if configured)
        if hasattr(self, 'backup_vector_store'):
            return await self.query_backup_store(operation)
        
        # Strategy 3: Return embedded fallback contexts
        return self.get_embedded_contexts(operation)
    
    async def handle_external_api_failure(self, error: Exception, api_name: str, context: dict) -> Any:
        """
        Handle external API failures (Exa, Cohere, etc.)
        """
        self.error_counts[f"{api_name}_error"] += 1
        
        if api_name == "exa":
            # Fallback to alternative search
            if self.error_counts["exa_error"] > 3:
                return await self.use_alternative_search(context)
            
            # Retry with reduced scope
            context["num_results"] = min(context.get("num_results", 10) // 2, 3)
            return await self.retry_with_backoff(context, max_attempts=2)
            
        elif api_name == "cohere":
            # Skip reranking if Cohere fails
            logger.warning("Cohere reranking failed, using original order")
            return context.get("original_results", [])
            
        else:
            # Generic external API fallback
            cached = self.fallback_cache.get(f"{api_name}_{hash(str(context))}")
            if cached:
                return cached
            
            return self.get_default_external_response(api_name)
    
    def is_circuit_open(self, service: str) -> bool:
        """
        Check if circuit breaker is open for a service
        """
        breaker = self.circuit_breakers.get(service)
        if not breaker:
            self.circuit_breakers[service] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                expected_exception=Exception
            )
            breaker = self.circuit_breakers[service]
        
        return breaker.is_open
    
    async def get_fallback_model(self, current_model: str) -> Optional[str]:
        """
        Get fallback model for failed primary model
        """
        fallbacks = self.model_fallbacks.get(current_model, [])
        
        for fallback in fallbacks:
            # Check if fallback model is available
            if await self.check_model_availability(fallback):
                logger.info(f"Switching from {current_model} to {fallback}")
                return fallback
        
        return None
    
    def calculate_backoff_delay(self, agent_name: str) -> float:
        """
        Calculate exponential backoff delay
        """
        error_count = self.error_counts.get(f"{agent_name}_RateLimitError", 0)
        base_delay = 1.0
        max_delay = 60.0
        
        delay = min(base_delay * (2 ** error_count), max_delay)
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0, delay * 0.1)
        
        return delay + jitter
    
    def simplify_prompt(self, prompt: str) -> str:
        """
        Simplify prompt to reduce token usage and complexity
        """
        # Remove examples if present
        simplified = re.sub(r'Example:.*?(?=\n\n|\Z)', '', prompt, flags=re.DOTALL)
        
        # Truncate if too long
        max_length = 1000
        if len(simplified) > max_length:
            simplified = simplified[:max_length] + "\n\nProvide a concise response."
        
        return simplified
    
    async def get_cached_or_default(self, agent_name: str, context: dict) -> dict:
        """
        Get cached result or default response
        """
        # Try cache first
        cache_key = f"{agent_name}_{self.hash_context(context)}"
        cached = self.fallback_cache.get(cache_key)
        
        if cached:
            logger.info(f"Using cached result for {agent_name}")
            return cached
        
        # Return agent-specific default
        defaults = {
            "web_search": {
                "market_data": {"min": 50, "max": 150, "median": 85, "currency": "USD"},
                "source": "default",
                "confidence": 0.3
            },
            "rag_search": {
                "insights": [
                    "Use mirroring to build rapport",
                    "Ask calibrated questions", 
                    "Never split the difference"
                ],
                "source": "default"
            },
            "synthesis": {
                "rewritten_text": self.get_template_rewrite(context),
                "strategy": {"techniques": ["basic"], "confidence": 0.4}
            }
        }
        
        return defaults.get(agent_name, {"error": "No fallback available"})
    
    def create_emergency_response(self, request: NegotiationRequest, mode: str = "fallback") -> NegotiationResponse:
        """
        Create last-resort emergency response
        """
        return NegotiationResponse(
            success=False,
            degraded=True,
            rewritten_text="Thank you for your message. I'd like to understand your position better. What are the main constraints driving your decision?",
            strategy={
                "techniques": ["acknowledge", "pause"],
                "confidence": 0.2
            },
            insights=[
                "Acknowledge their position",
                "Take time to consider", 
                "Research market rates",
                "Prepare counter-offer"
            ],
            confidence=0.2,
            mode=f"{mode}_emergency" if mode != "fallback" else "emergency",
            request_id=str(hash(request.negotiation_text)),
            error_recovery="emergency_mode"
        )
    
    async def notify_auth_failure(self, agent_name: str):
        """
        Notify about authentication failures
        """
        logger.critical(f"Authentication failed for {agent_name}")
        # Could send alerts, update monitoring, etc.
    
    def hash_context(self, context: dict) -> str:
        """
        Create hash of context for caching
        """
        return hashlib.md5(json.dumps(context, sort_keys=True).encode()).hexdigest()
    
    def create_template_rewrite(self, original_text: str) -> str:
        """
        Create a basic template rewrite when full analysis fails
        """
        return f"I appreciate your offer. Based on my experience and current market conditions, I'd like to discuss the rate further. {original_text}"
    
    def get_template_rewrite(self, context: dict) -> str:
        """
        Get template rewrite from context
        """
        original = context.get("original_text", "")
        return self.create_template_rewrite(original)
    
    # Placeholder methods for external dependencies
    async def execute_single_agent_fallback(self, agent_name: str, request: NegotiationRequest) -> dict:
        """Placeholder for single agent fallback execution"""
        return {"insights": ["Basic negotiation advice"], "source": "fallback"}
    
    async def check_model_availability(self, model: str) -> bool:
        """Placeholder for model availability check"""
        return True
    
    async def check_api_health(self) -> bool:
        """Placeholder for API health check"""
        return True
    
    async def retry_with_fallback_model(self, context: dict) -> dict:
        """Placeholder for model fallback retry"""
        return await self.get_cached_or_default("generic", context)
    
    async def retry_with_backoff(self, context: dict, max_attempts: int = 3) -> dict:
        """Placeholder for backoff retry"""
        return await self.get_cached_or_default("generic", context)
    
    async def get_offline_fallback(self, agent_name: str, context: dict) -> dict:
        """Placeholder for offline fallback"""
        return await self.get_cached_or_default(agent_name, context)
    
    async def retry_with_new_client(self, context: dict) -> dict:
        """Placeholder for new client retry"""
        return await self.get_cached_or_default("generic", context)
    
    async def get_generic_fallback(self, agent_name: str, context: dict) -> dict:
        """Placeholder for generic fallback"""
        return await self.get_cached_or_default(agent_name, context)
    
    async def refresh_qdrant_connection(self):
        """Placeholder for Qdrant connection refresh"""
        pass
    
    async def retry_vector_operation(self, operation: str) -> Any:
        """Placeholder for vector operation retry"""
        return []
    
    async def query_backup_store(self, operation: str) -> Any:
        """Placeholder for backup store query"""
        return []
    
    def get_embedded_contexts(self, operation: str) -> Any:
        """Placeholder for embedded contexts"""
        return []
    
    async def use_alternative_search(self, context: dict) -> Any:
        """Placeholder for alternative search"""
        return {"results": [], "source": "fallback"}
    
    def get_default_external_response(self, api_name: str) -> Any:
        """Placeholder for default external API response"""
        return {"status": "fallback", "data": None}


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance
    """
    def __init__(self, failure_threshold: int, recovery_timeout: int, expected_exception: type):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    @property
    def is_open(self) -> bool:
        if self.state == "open":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return False
            return True
        return False
    
    async def call(self, func: Callable, *args, **kwargs):
        if self.is_open:
            raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            # Success - reset failure count
            if self.state == "half-open":
                self.state = "closed"
            self.failure_count = 0
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e


class RateLimiter:
    """
    Rate limiter for external API calls
    """
    def __init__(self, max_requests_per_minute: int, max_requests_per_hour: int):
        self.max_per_minute = max_requests_per_minute
        self.max_per_hour = max_requests_per_hour
        self.requests = deque()
    
    async def check_rate_limit(self):
        now = datetime.now()
        
        # Remove old requests
        self.requests = deque(
            r for r in self.requests 
            if now - r < timedelta(hours=1)
        )
        
        # Check limits
        recent_minute = sum(
            1 for r in self.requests 
            if now - r < timedelta(minutes=1)
        )
        
        if recent_minute >= self.max_per_minute:
            wait_time = 60 - (now - self.requests[-1]).seconds
            await asyncio.sleep(wait_time)
        
        if len(self.requests) >= self.max_per_hour:
            raise RateLimitExceeded("Hourly rate limit reached")
        
        self.requests.append(now)


# Retry logic with exponential backoff
async def retry_with_backoff(
    func: Callable,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
) -> Any:
    """
    Retry failed operations with exponential backoff
    """
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            
            if attempt == max_attempts - 1:
                # Last attempt failed
                raise e
            
            # Calculate delay with jitter
            delay = min(initial_delay * (exponential_base ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay + jitter:.2f}s: {e}")
            await asyncio.sleep(delay + jitter)
    
    raise last_exception