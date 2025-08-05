# routers/negotiation.py - Main negotiation analysis endpoints
import logging
import asyncio
import time
import os
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Header
from typing import Dict, Any, Optional, Union

try:
    # Try relative imports first (when running as module)
    from ...agents.supervisor_agent import SupervisorAgent
    from ...agents.models import (
        NegotiationRequest,
        NegotiationResponse, 
        ApiConfiguration,
        ComparisonResponse,
        ComparisonMetrics
    )
    from ..dependencies import get_api_config
except ImportError:
    # Fallback to absolute imports (when running standalone)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from agents.supervisor_agent import SupervisorAgent
    from agents.models import (
        NegotiationRequest,
        NegotiationResponse, 
        ApiConfiguration,
        ComparisonResponse,
        ComparisonMetrics
    )
    from api.dependencies import get_api_config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/negotiate", tags=["negotiation"])

@router.post("/", response_model=NegotiationResponse)
async def negotiate(
    request: NegotiationRequest,
    mode: str = "advanced",
    openai_api_key: str = Header(..., alias="X-OpenAI-API-Key"),
    cohere_api_key: Optional[str] = Header(None, alias="X-Cohere-API-Key"),
    exa_api_key: Optional[str] = Header(None, alias="X-Exa-API-Key")
) -> NegotiationResponse:
    """
    Main negotiation analysis endpoint
    
    Analyzes negotiation text and provides improved response with strategy
    """
    try:
        # Configure API clients with user-provided keys, falling back to env vars
        api_config = ApiConfiguration(
            openai_key=openai_api_key,
            cohere_key=cohere_api_key or os.getenv("COHERE_API_KEY"),
            exa_key=exa_api_key or os.getenv("EXA_API_KEY")
        )
        
        # Initialize supervisor agent
        supervisor = SupervisorAgent(api_config)
        
        # Process the request
        result = await supervisor.analyze(request, mode)
        
        return result
        
    except Exception as e:
        logger.error(f"Negotiation analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@router.post("/compare")
async def compare_rag_modes(
    request: NegotiationRequest,
    openai_api_key: str = Header(..., alias="X-OpenAI-API-Key"),
    cohere_api_key: Optional[str] = Header(None, alias="X-Cohere-API-Key"),
    exa_api_key: Optional[str] = Header(None, alias="X-Exa-API-Key")
) -> ComparisonResponse:
    """
    Process negotiation through both RAG modes for side-by-side comparison
    Returns results from both naive and advanced RAG with performance metrics
    """
    try:
        # Configure API clients with user-provided keys, falling back to env vars
        api_config = ApiConfiguration(
            openai_key=openai_api_key,
            cohere_key=cohere_api_key or os.getenv("COHERE_API_KEY"),
            exa_key=exa_api_key or os.getenv("EXA_API_KEY")
        )
        
        # Create supervisor agent with API configuration
        supervisor_agent = SupervisorAgent(api_config)
        
        # Process through both modes in parallel for efficiency
        naive_task = asyncio.create_task(
            supervisor_agent.analyze(request, mode="naive")
        )
        advanced_task = asyncio.create_task(
            supervisor_agent.analyze(request, mode="advanced")
        )
        
        # Wait for both results
        start_time = time.time()
        naive_result, advanced_result = await asyncio.gather(
            naive_task, 
            advanced_task,
            return_exceptions=True
        )
        
        # Handle partial failures
        if isinstance(naive_result, Exception):
            logger.error(f"Naive RAG failed: {naive_result}")
            naive_result = NegotiationResponse(
                success=False,
                rewritten_text="Processing failed",
                strategy={"error": "naive_processing_failed"},
                confidence=0.0,
                mode="naive",
                request_id=request.id if hasattr(request, 'id') else "unknown"
            )
            
        if isinstance(advanced_result, Exception):
            logger.error(f"Advanced RAG failed: {advanced_result}")
            advanced_result = NegotiationResponse(
                success=False,
                rewritten_text="Processing failed",
                strategy={"error": "advanced_processing_failed"},
                confidence=0.0,
                mode="advanced",
                request_id=request.id if hasattr(request, 'id') else "unknown"
            )
        
        # Calculate comparison metrics
        total_time = time.time() - start_time
        
        # Calculate scores for frontend compatibility
        naive_accuracy = calculate_accuracy_score(naive_result)
        advanced_accuracy = calculate_accuracy_score(advanced_result)
        naive_relevance = (getattr(naive_result, 'avg_similarity_score', 0.0) or 0.0) * 100
        advanced_relevance = (getattr(advanced_result, 'avg_similarity_score', 0.0) or 0.0) * 100
        
        comparison_metrics = {
            "total_processing_time": total_time,
            "naive_retrieval_time": naive_result.retrieval_time if naive_result.success else None,
            "advanced_retrieval_time": advanced_result.retrieval_time if advanced_result.success else None,
            "naive_chunks_retrieved": len(getattr(naive_result, 'retrieved_contexts', [])) if naive_result.success else 0,
            "advanced_chunks_retrieved": len(getattr(advanced_result, 'retrieved_contexts', [])) if advanced_result.success else 0,
            "naive_avg_similarity": getattr(naive_result, 'avg_similarity_score', None) if naive_result.success else None,
            "advanced_avg_similarity": getattr(advanced_result, 'avg_similarity_score', None) if advanced_result.success else None,
            "context_corridor_used": getattr(advanced_result, 'context_corridor_used', False) if advanced_result.success else False,
            "reranking_impact": calculate_reranking_impact(naive_result, advanced_result),
            "quality_improvement": calculate_quality_improvement(naive_result, advanced_result),
            # Frontend expected fields
            "response_time_naive": naive_result.retrieval_time if naive_result.success else 0,
            "response_time_advanced": advanced_result.retrieval_time if advanced_result.success else 0,
            "accuracy_score_naive": naive_accuracy,
            "accuracy_score_advanced": advanced_accuracy,
            "relevance_score_naive": naive_relevance,
            "relevance_score_advanced": advanced_relevance
        }
        
        return ComparisonResponse(
            request_id=getattr(request, 'id', "unknown"),
            naive_result=naive_result,
            advanced_result=advanced_result,
            comparison_metrics=comparison_metrics,
            recommendation=determine_best_mode(comparison_metrics),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Comparison analysis failed: {e}", exc_info=True)
        return ComparisonResponse(
            success=False,
            error_message="Comparison analysis failed. Please try again.",
            request_id=getattr(request, 'id', "unknown")
        )

@router.post("/test-api-key")
async def test_api_key(
    openai_api_key: str = Header(..., alias="X-OpenAI-API-Key")
):
    """
    Test OpenAI API key validity
    """
    try:
        from langchain_openai import ChatOpenAI
        
        # Test the API key with a simple request
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            api_key=openai_api_key,
            max_tokens=5
        )
        
        # Make a simple test call
        response = await llm.ainvoke("Test")
        
        return {
            "valid": True,
            "message": "API key is valid"
        }
        
    except Exception as e:
        logger.error(f"API key test failed: {e}")
        return {
            "valid": False,
            "message": f"API key test failed: {str(e)}"
        }

@router.get("/health")
async def negotiation_health():
    """
    Health check for negotiation service
    """
    return {
        "status": "healthy",
        "service": "negotiation_analysis",
        "version": "1.0.0"
    }


# Comparison Helper Functions
def calculate_reranking_impact(naive_result: NegotiationResponse, advanced_result: NegotiationResponse) -> float:
    """
    Calculate the impact of reranking on retrieval quality
    """
    if not (naive_result.success and advanced_result.success):
        return 0.0
    
    # Compare average similarity scores
    naive_avg = getattr(naive_result, 'avg_similarity_score', 0.0) or 0.0
    advanced_avg = getattr(advanced_result, 'avg_similarity_score', 0.0) or 0.0
    
    return advanced_avg - naive_avg


def calculate_quality_improvement(naive_result: NegotiationResponse, advanced_result: NegotiationResponse) -> float:
    """
    Calculate overall quality improvement percentage
    """
    if not (naive_result.success and advanced_result.success):
        return 0.0
    
    # Factors to consider:
    # 1. Retrieval precision (avg similarity)
    # 2. Chunk efficiency (fewer chunks with higher relevance)
    # 3. Response time
    
    naive_score = calculate_quality_score(naive_result)
    advanced_score = calculate_quality_score(advanced_result)
    
    if naive_score == 0:
        return 0.0
    
    return ((advanced_score - naive_score) / naive_score) * 100


def calculate_accuracy_score(result: NegotiationResponse) -> float:
    """
    Calculate accuracy score based on result quality metrics
    """
    if not result.success:
        return 0.0
    
    # Base score from confidence
    score = result.confidence * 100
    
    # Bonus for having market data
    if result.market_data:
        score += 10
    
    # Bonus for having insights
    if result.insights:
        score += len(result.insights) * 2
    
    # Bonus for successful strategy generation
    if result.strategy and isinstance(result.strategy, dict):
        if result.strategy.get('techniques'):
            score += 5
    
    return min(score, 100.0)


def calculate_quality_score(result: NegotiationResponse) -> float:
    """
    Calculate a quality score for a negotiation result
    """
    if not result.success:
        return 0.0
    
    # Weighted scoring
    similarity_weight = 0.4
    efficiency_weight = 0.3
    speed_weight = 0.3
    
    # Similarity score (0-1)
    similarity_score = getattr(result, 'avg_similarity_score', 0.0) or 0.0
    
    # Efficiency score (fewer chunks is better)
    chunk_count = len(getattr(result, 'retrieved_contexts', []))
    efficiency_score = 1.0 / (1.0 + (chunk_count / 10))  # Normalize
    
    # Speed score (faster is better, normalized to 0-1)
    retrieval_time = result.retrieval_time or 1000
    speed_score = 1.0 / (1.0 + (retrieval_time / 1000))  # Normalize
    
    return (
        similarity_score * similarity_weight +
        efficiency_score * efficiency_weight +
        speed_score * speed_weight
    )


def determine_best_mode(metrics: Union[Dict[str, Any], ComparisonMetrics]) -> dict:
    """
    Determine which RAG mode performed better based on metrics
    """
    improvements = []
    
    # Handle both dict and object access
    quality_improvement = metrics.get('quality_improvement', 0) if isinstance(metrics, dict) else metrics.quality_improvement
    reranking_impact = metrics.get('reranking_impact', 0) if isinstance(metrics, dict) else metrics.reranking_impact
    advanced_chunks = metrics.get('advanced_chunks_retrieved', 0) if isinstance(metrics, dict) else metrics.advanced_chunks_retrieved
    naive_chunks = metrics.get('naive_chunks_retrieved', 0) if isinstance(metrics, dict) else metrics.naive_chunks_retrieved
    context_corridor = metrics.get('context_corridor_used', False) if isinstance(metrics, dict) else metrics.context_corridor_used
    advanced_time = metrics.get('advanced_retrieval_time') if isinstance(metrics, dict) else metrics.advanced_retrieval_time
    naive_time = metrics.get('naive_retrieval_time') if isinstance(metrics, dict) else metrics.naive_retrieval_time
    advanced_similarity = metrics.get('advanced_avg_similarity') if isinstance(metrics, dict) else metrics.advanced_avg_similarity
    naive_similarity = metrics.get('naive_avg_similarity') if isinstance(metrics, dict) else metrics.naive_avg_similarity
    
    if quality_improvement > 10:
        preferred = "advanced"
        reasoning = f"Advanced RAG shows {quality_improvement:.1f}% quality improvement with better context relevance"
        
        if reranking_impact > 0:
            improvements.append(f"Reranking improved relevance by {reranking_impact:.2%}")
        if advanced_chunks < naive_chunks:
            improvements.append(f"More efficient retrieval with {advanced_chunks} vs {naive_chunks} chunks")
        if context_corridor:
            improvements.append("Context corridor optimization was utilized")
            
    elif quality_improvement < -10:
        preferred = "naive"
        reasoning = f"Naive RAG performs better for this query with {abs(quality_improvement):.1f}% higher quality"
        improvements.append("Consider simpler retrieval for straightforward queries")
        improvements.append("Advanced features may be over-complicating the response")
        
    else:
        # Consider other factors if quality is similar
        if advanced_time and naive_time:
            if advanced_time < naive_time * 1.5:
                preferred = "advanced"
                reasoning = "Advanced RAG provides similar quality with acceptable performance overhead"
                improvements.append("Advanced features provide marginal gains for complex negotiations")
            else:
                preferred = "naive"
                reasoning = f"Naive RAG is {naive_time/advanced_time:.1f}x faster with comparable quality"
                improvements.append("Speed advantage makes naive approach more practical")
        else:
            preferred = "advanced"
            reasoning = "Advanced RAG recommended for comprehensive negotiation analysis"
            improvements.append("Enable all API keys for full advanced RAG capabilities")
    
    # Add general improvements based on metrics
    if advanced_similarity and naive_similarity:
        if advanced_similarity > naive_similarity:
            improvements.append(f"Advanced RAG shows {((advanced_similarity - naive_similarity) / naive_similarity * 100):.1f}% better semantic matching")
    
    # Ensure we always have at least one improvement suggestion
    if not improvements:
        improvements.append("Both approaches perform similarly for this scenario")
    
    return {
        "preferred_approach": preferred,
        "reasoning": reasoning,
        "improvements": improvements[:3]  # Limit to top 3 improvements
    }