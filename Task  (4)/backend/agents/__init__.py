# agents/__init__.py - Agent system package initialization
"""
Freelancer Negotiation Helper Agent System

This package contains the multi-agent system for negotiation analysis:
- SupervisorAgent: Main orchestrator
- WebSearchAgent: Market data retrieval
- RAGSearchAgent: Negotiation tactics retrieval
- SynthesisAgent: Response generation
- ErrorHandler: Comprehensive error handling
"""

from .models import (
    MODEL_CONFIG,
    ApiConfiguration,
    NegotiationRequest,
    NegotiationResponse,
    MarketRateReport,
    SynthesisInput,
    NegotiationInsights,
    CachedNegotiationInsights,
    ComparisonResponse,
    ComparisonMetrics,
    AgentTask,
    AgentState,
    HealthStatus,
    AgentExecutionError,
    RateLimitExceeded,
    ExaAPIError,
    CircuitBreakerOpenError
)

from .error_handler import (
    ErrorHandler,
    CircuitBreaker,
    RateLimiter,
    retry_with_backoff
)

from .supervisor_agent import SupervisorAgent
from .web_search_agent import WebSearchAgent
from .rag_search_agent import RAGSearchAgent
from .synthesis_agent import SynthesisAgent

__version__ = "1.0.0"
__author__ = "Freelancer Negotiation Helper Team"
__description__ = "Multi-agent system for negotiation analysis and improvement"

# Export main classes
__all__ = [
    # Core agents
    "SupervisorAgent",
    "WebSearchAgent",
    
    # Models and data classes
    "MODEL_CONFIG",
    "ApiConfiguration",
    "NegotiationRequest", 
    "NegotiationResponse",
    "MarketRateReport",
    "SynthesisInput",
    "NegotiationInsights",
    "CachedNegotiationInsights",
    "ComparisonResponse",
    "ComparisonMetrics",
    "AgentTask",
    "AgentState",
    "HealthStatus",
    
    # Error handling
    "ErrorHandler",
    "CircuitBreaker", 
    "RateLimiter",
    "retry_with_backoff",
    
    # Exceptions
    "AgentExecutionError",
    "RateLimitExceeded", 
    "ExaAPIError",
    "CircuitBreakerOpenError"
]