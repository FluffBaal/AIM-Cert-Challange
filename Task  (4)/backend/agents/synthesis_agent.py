"""
Synthesis Agent for creating final negotiation responses
"""
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json

from langchain_openai import ChatOpenAI

from .models import (
    ApiConfiguration, 
    NegotiationRequest, 
    NegotiationResponse,
    SynthesisInput,
    MarketRateReport
)

logger = logging.getLogger(__name__)

class SynthesisAgent:
    """
    Agent responsible for synthesizing all gathered information into
    a coherent negotiation response with rewritten text and strategy
    """
    
    def __init__(self, api_config: ApiConfiguration):
        self.api_config = api_config
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            api_key=api_config.openai_key
        )
        
    async def synthesize(
        self, 
        synthesis_input: SynthesisInput
    ) -> NegotiationResponse:
        """
        Synthesize all agent outputs into final response
        """
        try:
            # Prepare context for synthesis
            context = self._prepare_context(synthesis_input)
            
            # Generate improved negotiation text
            original_text = synthesis_input.original_request.negotiation_text
            rewritten_text = await self._generate_improved_text(
                original_text,
                context,
                synthesis_input.mode
            )
            
            # Generate negotiation strategy
            strategy = await self._generate_strategy(
                original_text,
                rewritten_text,
                context,
                synthesis_input.mode
            )
            
            # Extract key insights
            insights = self._extract_insights(synthesis_input)
            
            # Add metadata
            metadata = {
                "mode": synthesis_input.mode,
                "agents_used": self._get_agents_used(synthesis_input),
                "processing_time": datetime.now().isoformat()
            }
            
            return NegotiationResponse(
                success=True,
                rewritten_text=rewritten_text,
                strategy=strategy,
                insights=insights,
                market_data=synthesis_input.market_data,
                confidence=self._calculate_confidence(synthesis_input),
                mode=synthesis_input.mode,
                request_id=synthesis_input.original_request.id,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return NegotiationResponse(
                success=False,
                rewritten_text=synthesis_input.original_request.negotiation_text,
                strategy={
                    "techniques": ["error_recovery"],
                    "key_points": ["Unable to generate strategy"],
                    "confidence": 0.0,
                    "approach": "fallback"
                },
                insights=[],
                confidence=0.0,
                request_id=synthesis_input.original_request.id,
                mode=synthesis_input.mode
            )
    
    def _prepare_context(self, synthesis_input: SynthesisInput) -> Dict[str, Any]:
        """Prepare context from all agent inputs"""
        context = {
            "user_context": synthesis_input.original_request.user_context,
            "negotiation_scenario": synthesis_input.original_request.negotiation_text
        }
        
        # Add RAG insights if available
        if synthesis_input.rag_insights:
            context["tactics"] = synthesis_input.rag_insights.tactics
            context["insights"] = synthesis_input.rag_insights.insights
        
        # Add market data if available
        if synthesis_input.market_data:
            context["market_rates"] = {
                "min": synthesis_input.market_data.min_rate,
                "max": synthesis_input.market_data.max_rate,
                "median": synthesis_input.market_data.median_rate
            }
        
        return context
    
    async def _generate_improved_text(
        self, 
        original_text: str,
        context: Dict[str, Any],
        mode: str
    ) -> str:
        """Generate improved negotiation text"""
        
        tactics_str = ", ".join(context.get("tactics", ["tactical empathy", "mirroring"]))
        
        prompt = f"""
        You are an expert negotiator trained in Chris Voss's methods from "Never Split the Difference".
        
        SCENARIO: {original_text}
        
        CONTEXT: {json.dumps(context, indent=2)}
        
        MODE: {mode}
        
        Create a professional response using these negotiation tactics: {tactics_str}
        
        IMPORTANT GUIDELINES:
        1. Start with tactical empathy - acknowledge their position without immediately agreeing
        2. Use calibrated questions (How/What) to understand their constraints
        3. Apply the specific tactics from Chris Voss that are relevant to this situation
        4. Be collaborative but maintain your position
        5. If discussing rates/prices, use appropriate anchoring based on market data
        6. End with an open-ended question that moves the conversation forward
        
        The response should:
        - Sound natural and professional
        - Apply specific negotiation techniques subtly
        - Address their concerns while advancing your position
        - Be 2-4 sentences maximum
        
        Return ONLY the response text, nothing else. Do not explain the tactics being used.
        """
        
        response = await self.llm.ainvoke(prompt)
        return response.content.strip()
    
    async def _generate_strategy(
        self,
        original_text: str,
        rewritten_text: str,
        context: Dict[str, Any],
        mode: str
    ) -> Dict[str, Any]:
        """Generate negotiation strategy"""
        
        prompt = f"""
        Based on this negotiation scenario and the improved response, provide a negotiation strategy in JSON format.
        
        Original: {original_text}
        Improved: {rewritten_text}
        Context: {json.dumps(context, indent=2)}
        
        Return a JSON object with:
        {{
            "techniques": ["technique1", "technique2"],  // List of 2-3 negotiation techniques being used
            "key_points": ["point1", "point2"],  // 2-3 key strategic points
            "confidence": 0.8,  // Confidence score between 0-1
            "approach": "collaborative"  // Overall approach: collaborative, assertive, or balanced
        }}
        
        Only return the JSON object, no additional text.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            strategy_json = response.content.strip()
            # Try to extract JSON if wrapped in markdown
            if "```json" in strategy_json:
                strategy_json = strategy_json.split("```json")[1].split("```")[0].strip()
            elif "```" in strategy_json:
                strategy_json = strategy_json.split("```")[1].strip()
            
            return json.loads(strategy_json)
        except Exception as e:
            logger.error(f"Failed to parse strategy JSON: {e}")
            # Return default strategy on error
            return {
                "techniques": ["mirroring", "calibrated_questions"],
                "key_points": ["Build rapport", "Understand their position"],
                "confidence": 0.6,
                "approach": "collaborative"
            }
    
    def _extract_insights(self, synthesis_input: SynthesisInput) -> List[str]:
        """Extract key insights from all sources"""
        insights = []
        
        # Add RAG insights
        if synthesis_input.rag_insights:
            insights.extend(synthesis_input.rag_insights.insights[:3])
        
        # Add market insights
        if synthesis_input.market_data and synthesis_input.market_data.median_rate:
            insights.append(
                f"Market rate range: ${synthesis_input.market_data.min_rate}-"
                f"${synthesis_input.market_data.max_rate}/hour"
            )
        
        return insights[:5]  # Limit to 5 insights
    
    def _calculate_confidence(self, synthesis_input: SynthesisInput) -> float:
        """Calculate confidence score based on available data"""
        confidence = 0.5  # Base confidence
        
        # Add confidence for each data source
        if synthesis_input.rag_insights:
            confidence += 0.2
        
        if synthesis_input.market_data:
            confidence += 0.2
            
        # Mode bonus
        if synthesis_input.mode == "advanced":
            confidence += 0.1
            
        return min(confidence, 1.0)
    
    def _get_agents_used(self, synthesis_input: SynthesisInput) -> List[str]:
        """Get list of agents that provided data"""
        agents = ["synthesis"]
        
        if synthesis_input.rag_insights:
            agents.append("rag_search")
            
        if synthesis_input.market_data:
            agents.append("web_search")
            
        return agents