"""
RAG Search Agent for retrieving negotiation insights
"""
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import sys
import os

from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

# Import the DualRAGRetriever
try:
    from ..ingestion.dual_rag_retriever import DualRAGRetriever
except ImportError:
    # Fallback for different import paths
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from ingestion.dual_rag_retriever import DualRAGRetriever

from .models import ApiConfiguration, NegotiationInsights

logger = logging.getLogger(__name__)

class RAGSearchAgent:
    """
    Agent responsible for retrieving negotiation tactics and insights from vector database
    Uses either naive or advanced retrieval based on mode
    """
    
    def __init__(self, api_config: ApiConfiguration):
        self.api_config = api_config
        # Use the DualRAGRetriever with API keys
        self.retriever = DualRAGRetriever(
            qdrant_host=api_config.qdrant_host,
            qdrant_port=api_config.qdrant_port,
            openai_api_key=api_config.openai_key,
            cohere_api_key=api_config.cohere_key
        )
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=api_config.openai_key
        )
        
    async def retrieve_insights(self, negotiation_text: str, mode: str = "advanced") -> NegotiationInsights:
        """
        Retrieve relevant negotiation insights based on the scenario using DualRAGRetriever
        """
        try:
            start_time = datetime.now()
            
            # Use the DualRAGRetriever to get relevant contexts
            retrieval_result = await self.retriever.retrieve(negotiation_text, mode)
            
            if not retrieval_result.success:
                logger.warning(f"Retrieval failed: {retrieval_result.error_message}")
                return self._get_fallback_insights()
            
            # Extract tactics and insights from retrieved contexts
            tactics = []
            insights = []
            
            for context in retrieval_result.retrieved_contexts:
                if context.strip():
                    extracted = await self._extract_tactics(context)
                    tactics.extend(extracted["tactics"])
                    insights.extend(extracted["insights"])
            
            # Deduplicate and prioritize
            tactics = list(dict.fromkeys(tactics))[:5]  # Top 5 unique tactics
            insights = list(dict.fromkeys(insights))[:5]  # Top 5 unique insights
            
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            # Convert detailed contexts if available
            detailed_contexts = []
            if retrieval_result.detailed_contexts:
                for ctx in retrieval_result.detailed_contexts:
                    from .models import RetrievedContext
                    detailed_contexts.append(RetrievedContext(
                        content=ctx.content,
                        source=ctx.source,
                        chapter=ctx.chapter,
                        relevance_score=ctx.relevance_score,
                        technique=ctx.technique,
                        metadata=ctx.metadata or {}
                    ))
            
            return NegotiationInsights(
                tactics=tactics,
                insights=insights,
                source=f"{mode}_retrieval",
                retrieval_time=retrieval_time,
                avg_similarity_score=retrieval_result.avg_similarity_score,
                context_corridor_used=retrieval_result.context_corridor_used,
                retrieved_contexts=retrieval_result.retrieved_contexts,
                detailed_contexts=detailed_contexts
            )
            
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return self._get_fallback_insights()
    
    def _get_fallback_insights(self) -> NegotiationInsights:
        """Return fallback insights when retrieval fails"""
        return NegotiationInsights(
            tactics=["mirroring", "labeling", "calibrated_questions"],
            insights=[
                "Focus on understanding their position",
                "Use tactical empathy",
                "Ask how/what questions instead of why"
            ],
            source="fallback",
            retrieval_time=0.0
        )
    
    
    async def _extract_tactics(self, content: str) -> Dict[str, List[str]]:
        """Extract negotiation tactics and insights from content"""
        prompt = f"""
        Extract negotiation tactics and insights from this content:
        
        {content[:1500]}  # Limit content length
        
        Return JSON with:
        - tactics: List of specific negotiation tactics mentioned
        - insights: List of key insights or principles
        
        Focus on actionable tactics from Chris Voss's methodology.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            import json
            return json.loads(response.content)
        except:
            return {"tactics": [], "insights": []}