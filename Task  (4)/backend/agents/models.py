# models.py - Data models and configuration for agent system
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Callable, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import os

# Standardized model configuration for all agents
MODEL_CONFIG = {
    "supervisor": "gpt-4",        # Advanced reasoning and orchestration
    "synthesis": "gpt-4",         # High-quality creative writing
    "web_search": "gpt-4o-mini",   # Fast, cost-effective for search
    "rag_search": "gpt-4o-mini",   # Fast, cost-effective for retrieval
}

# Benefits of standardizing on GPT-4 models:
# 1. Consistent performance across agents
# 2. Simplified model management
# 3. Better token context handling
# 4. Cost-effective with gpt-4o-mini for simple tasks
# 5. No migration needed before July 2025
# 6. Cost-effective with superior performance

# API Configuration
@dataclass
class ApiConfiguration:
    """Configuration for external API services"""
    openai_key: str
    exa_key: Optional[str] = None
    cohere_key: Optional[str] = None
    qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379")
    
    @property
    def qdrant_host(self) -> str:
        """Extract host from qdrant_url"""
        if "://" in self.qdrant_url:
            return self.qdrant_url.split("://")[1].split(":")[0]
        return self.qdrant_url.split(":")[0]
    
    @property
    def qdrant_port(self) -> int:
        """Extract port from qdrant_url"""
        if ":" in self.qdrant_url:
            port_part = self.qdrant_url.split(":")[-1]
            try:
                return int(port_part)
            except ValueError:
                pass
        return 6333

# Request Models
class NegotiationRequest(BaseModel):
    """Request model for negotiation analysis"""
    id: str = Field(default_factory=lambda: str(datetime.now().timestamp()), description="Unique request identifier")
    negotiation_text: str = Field(..., description="The text to analyze and improve")
    user_context: Dict[str, Any] = Field(default_factory=dict, description="User's professional context")
    mode: str = Field(default="advanced", description="Analysis mode: naive or advanced")
    
    class Config:
        schema_extra = {
            "example": {
                "negotiation_text": "I'm willing to work for $50/hour on this project.",
                "user_context": {
                    "skill": "UX Designer",
                    "location": "Berlin",
                    "experience_years": 5
                },
                "mode": "advanced"
            }
        }

# Response Models
class NegotiationResponse(BaseModel):
    """Response model for negotiation analysis"""
    success: bool = True
    partial_failure: bool = False
    degraded: bool = False
    rewritten_text: str = Field(..., description="Improved negotiation message")
    strategy: Dict[str, Any] = Field(..., description="Negotiation strategy and techniques")
    market_data: Optional['MarketRateReport'] = Field(None, description="Market rate information")
    insights: List[str] = Field(default_factory=list, description="Key negotiation insights")
    confidence: float = Field(..., description="Confidence score 0-1")
    mode: str = Field(..., description="Analysis mode used")
    agents_used: List[str] = Field(default_factory=list, description="List of agents involved")
    retrieval_time: float = Field(default=0.0, description="Time spent on retrieval")
    error_recovery: Optional[str] = Field(None, description="Error recovery method used")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    retrieved_contexts: List[str] = Field(default_factory=list, description="Retrieved context chunks")
    avg_similarity_score: Optional[float] = Field(None, description="Average similarity score of retrieved chunks")
    context_corridor_used: bool = Field(default=False, description="Whether context corridor was used in advanced mode")
    retrieval_details: Optional[Dict[str, Any]] = Field(None, description="Detailed retrieval information")
    processing_steps: Optional[List[Dict[str, Any]]] = Field(None, description="Processing step details")
    detailed_contexts: List['RetrievedContext'] = Field(default_factory=list, description="Detailed context information")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "rewritten_text": "Based on my 5 years of UX design experience and current market rates in Berlin, my rate for this project would be $120/hour.",
                "strategy": {
                    "techniques": ["anchoring", "mirroring", "value_demonstration"],
                    "confidence": 0.85
                },
                "confidence": 0.85,
                "mode": "advanced"
            }
        }

class MarketRateReport(BaseModel):
    """Market rate analysis report"""
    min_rate: float = Field(..., description="Minimum rate found")
    max_rate: float = Field(..., description="Maximum rate found")
    median_rate: float = Field(..., description="Median rate")
    average_rate: Optional[float] = Field(None, description="Average rate")
    currency: str = Field(default="USD", description="Currency code")
    sources: List[str] = Field(default_factory=list, description="Data sources")
    data_points: int = Field(default=0, description="Number of data points")
    location: Optional[str] = Field(None, description="Geographic location")
    skill: Optional[str] = Field(None, description="Skill/profession")
    confidence: float = Field(default=0.8, description="Confidence in data quality")
    note: Optional[str] = Field(None, description="Additional notes")
    
    class Config:
        schema_extra = {
            "example": {
                "min_rate": 85,
                "max_rate": 150,
                "median_rate": 120,
                "currency": "USD",
                "data_points": 15,
                "location": "Berlin",
                "skill": "UX Designer"
            }
        }

class SynthesisInput(BaseModel):
    """Input for the synthesis agent"""
    original_request: NegotiationRequest
    context: Dict[str, Any]
    rag_insights: Optional[Any] = None
    market_data: Optional[MarketRateReport] = None
    mode: str = "advanced"

class RetrievedContext(BaseModel):
    """Detailed information about a retrieved context chunk"""
    content: str = Field(..., description="The text content of the chunk")
    source: str = Field(..., description="Source document/chapter")
    chapter: Optional[str] = Field(None, description="Chapter number if applicable")
    page: Optional[int] = Field(None, description="Page number if applicable")
    relevance_score: float = Field(..., description="Similarity/relevance score")
    technique: Optional[str] = Field(None, description="Negotiation technique associated")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class NegotiationInsights(BaseModel):
    """Structured negotiation insights from RAG"""
    tactics: List[str] = Field(default_factory=list, description="Recommended tactics")
    insights: List[str] = Field(default_factory=list, description="Key insights")
    source: str = Field(..., description="Source of insights")
    retrieval_time: float = Field(default=0.0, description="Time spent retrieving")
    avg_similarity_score: Optional[float] = Field(None, description="Average similarity score of retrieved chunks")
    context_corridor_used: bool = Field(default=False, description="Whether context corridor was used")
    retrieved_contexts: List[str] = Field(default_factory=list, description="Retrieved context chunks")
    detailed_contexts: List[RetrievedContext] = Field(default_factory=list, description="Detailed context information")

class CachedNegotiationInsights(BaseModel):
    """Cached negotiation insights for fallback"""
    tactics: List[str]
    insights: List[str]
    source: str = "cached"
    cached_at: datetime = Field(default_factory=datetime.now)

# Comparison Models
class ComparisonResponse(BaseModel):
    """Response model for RAG comparison"""
    request_id: str
    naive_result: Optional[NegotiationResponse]
    advanced_result: Optional[NegotiationResponse]
    comparison_metrics: Optional[Union[Dict[str, Any], 'ComparisonMetrics']]
    recommendation: Optional[Union[str, Dict[str, Any]]]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    error_message: Optional[str] = None

class ComparisonMetrics(BaseModel):
    """Metrics for comparing naive vs advanced RAG"""
    total_processing_time: float
    naive_retrieval_time: Optional[float]
    advanced_retrieval_time: Optional[float]
    naive_chunks_retrieved: int
    advanced_chunks_retrieved: int
    naive_avg_similarity: Optional[float]
    advanced_avg_similarity: Optional[float]
    context_corridor_used: bool
    reranking_impact: float
    quality_improvement: float

# Agent Task Configuration
@dataclass
class AgentTask:
    """Configuration for parallel agent execution"""
    agent_name: str
    agent_method: Callable
    args: tuple
    required: bool = True
    timeout: float = 10.0

# Agent State Management
class AgentState:
    """Maintains conversation state across agent interactions"""
    def __init__(self):
        self.conversation_history = []
        self.extracted_entities = {}
        self.agent_responses = {}
        self.error_count = 0
    
    def update(self, agent_name: str, response: Any):
        self.agent_responses[agent_name] = response
        self.conversation_history.append({
            "agent": agent_name,
            "timestamp": datetime.now(),
            "response_summary": str(response)[:200]
        })

# Health Status Models
class HealthStatus(BaseModel):
    """API health status"""
    status: str = Field(..., description="Overall health status")
    services: Dict[str, str] = Field(..., description="Individual service statuses")
    timestamp: str = Field(..., description="Health check timestamp")
    degraded_features: List[str] = Field(default_factory=list, description="Features running in degraded mode")

# Exception Classes
class AgentExecutionError(Exception):
    """Custom exception for agent execution failures"""
    pass

class RateLimitExceeded(Exception):
    """Exception for rate limit violations"""
    pass

class ExaAPIError(Exception):
    """Exception for Exa API errors"""
    pass

class CircuitBreakerOpenError(Exception):
    """Exception when circuit breaker is open"""
    pass

# Placeholder classes for external API results
class ExaResult:
    """Placeholder for Exa search result"""
    def __init__(self, text: str, url: str):
        self.text = text
        self.url = url

# Update forward references
NegotiationResponse.model_rebuild()
ComparisonResponse.model_rebuild()