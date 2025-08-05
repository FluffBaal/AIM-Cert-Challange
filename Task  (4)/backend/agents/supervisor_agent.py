# supervisor_agent.py - Main orchestrator for negotiation analysis workflow
import asyncio
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

from langchain_openai import ChatOpenAI

from .models import (
    ApiConfiguration, 
    NegotiationRequest, 
    NegotiationResponse,
    AgentTask,
    AgentState,
    SynthesisInput,
    AgentExecutionError
)
from .error_handler import ErrorHandler

# Import other agents
from .web_search_agent import WebSearchAgent
from .rag_search_agent import RAGSearchAgent  
from .synthesis_agent import SynthesisAgent

logger = logging.getLogger(__name__)

class SupervisorAgent:
    """
    Orchestrates the negotiation analysis workflow with parallel execution
    - Parses user input and determines required agents
    - Manages parallel execution of search agents
    - Maintains conversation state
    - Routes results to synthesis agent
    - Comprehensive error handling with fallbacks
    """
    
    def __init__(self, api_config: ApiConfiguration):
        self.api_config = api_config
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            api_key=api_config.openai_key
        )
        # Initialize agent instances
        self.web_search_agent = WebSearchAgent(api_config)
        self.rag_search_agent = RAGSearchAgent(api_config)
        self.synthesis_agent = SynthesisAgent(api_config)
        
        self.state = AgentState()
        self.error_handler = ErrorHandler()
        
    async def analyze(self, request: NegotiationRequest, mode: str = "advanced", progress_emitter=None) -> NegotiationResponse:
        """
        Main orchestration method with parallel agent execution
        """
        processing_steps = []
        
        try:
            # Use progress emitter if provided, otherwise track locally
            if progress_emitter:
                processing_steps = progress_emitter.processing_steps
            else:
                # Track processing steps
                steps_config = [
                    ("retrieval", "Knowledge Retrieval", "Searching negotiation tactics database"),
                    ("context_analysis", "Context Analysis", "Understanding your negotiation scenario"),
                    ("market_search", "Market Research", "Fetching real-time rate data"),
                    ("strategy", "Strategy Formation", "Applying Chris Voss techniques"),
                    ("response", "Response Generation", "Crafting your negotiation response")
                ]
                
                for step_id, name, desc in steps_config:
                    processing_steps.append({
                        "id": step_id,
                        "name": name,
                        "description": desc,
                        "status": "pending",
                        "startTime": None,
                        "endTime": None
                    })
            
            # Parse and extract context from request
            if progress_emitter:
                await progress_emitter.update_step("context_analysis", "active", datetime.now().timestamp())
            else:
                self._update_step(processing_steps, "context_analysis", "active", datetime.now().timestamp())
            
            context = await self.extract_context(request)
            
            if progress_emitter:
                await progress_emitter.update_step("context_analysis", "completed", end_time=datetime.now().timestamp())
            else:
                self._update_step(processing_steps, "context_analysis", "completed", end_time=datetime.now().timestamp())
            
            # Determine which agents to invoke based on request
            agent_tasks = self.plan_agent_execution(request, context)
            
            # Execute agents in parallel with error handling
            results = await self.execute_agents_parallel(agent_tasks, mode, processing_steps)
            
            # Route to synthesis agent
            self._update_step(processing_steps, "response", "active", datetime.now().timestamp())
            synthesis_result = await self.synthesize_results(
                request=request,
                context=context,
                agent_results=results,
                mode=mode,
                processing_steps=processing_steps
            )
            self._update_step(processing_steps, "response", "completed", end_time=datetime.now().timestamp())
            
            # Add processing steps to response
            synthesis_result.processing_steps = processing_steps
            
            return synthesis_result
            
        except Exception as e:
            logger.error(f"Supervisor agent failed: {e}")
            # Graceful degradation
            return await self.error_handler.handle_supervisor_failure(e, request, mode)
    
    async def extract_context(self, request: NegotiationRequest) -> dict:
        """
        Extract structured context from negotiation request
        """
        prompt = f"""
        Extract key context from this negotiation scenario:
        
        Text: {request.negotiation_text}
        User Context: {request.user_context}
        
        Extract:
        1. Freelancer skill/profession
        2. Location (or remote)
        3. Years of experience
        4. Project type (if mentioned)
        5. Current rate discussion
        6. Client objections (if any)
        
        Return as JSON.
        """
        
        try:
            response = await self.llm.ainvoke(prompt)
            return json.loads(response.content)
        except Exception as e:
            logger.warning(f"Context extraction failed: {e}")
            # Fallback to basic context
            return {
                "skill": request.user_context.get("skill", "freelancer"),
                "location": request.user_context.get("location", "remote"),
                "experience_years": request.user_context.get("experience_years", 5)
            }
    
    def plan_agent_execution(self, request: NegotiationRequest, context: dict) -> List[AgentTask]:
        """
        Determine which agents to invoke based on request analysis
        """
        tasks = []
        
        # Always include RAG search for negotiation tactics
        if self.rag_search_agent:
            tasks.append(AgentTask(
                agent_name="rag_search",
                agent_method=self.rag_search_agent.retrieve_insights,
                args=(request.negotiation_text,),
                required=True,
                timeout=5.0
            ))
        
        # Include web search if market data would be helpful
        if self.should_search_market_data(request, context) and self.web_search_agent:
            tasks.append(AgentTask(
                agent_name="web_search",
                agent_method=self.web_search_agent.search_market_data,
                args=(context,),
                required=False,  # Not required, can fail gracefully
                timeout=30.0  # Increased timeout for web search
            ))
        
        return tasks
    
    def _update_step(self, steps: List[Dict], step_id: str, status: str, start_time: float = None, end_time: float = None):
        """Helper to update processing step status"""
        for step in steps:
            if step["id"] == step_id:
                step["status"] = status
                if start_time:
                    step["startTime"] = start_time
                if end_time:
                    step["endTime"] = end_time
                break
    
    async def execute_agents_parallel(self, tasks: List[AgentTask], mode: str, processing_steps: List[Dict] = None) -> Dict[str, Any]:
        """
        Execute multiple agents in parallel with comprehensive error handling
        """
        results = {}
        
        # Create async tasks with timeout and error handling
        async_tasks = []
        for task in tasks:
            # Update processing steps
            if processing_steps:
                if task.agent_name == "rag_search":
                    self._update_step(processing_steps, "retrieval", "active", datetime.now().timestamp())
                elif task.agent_name == "web_search":
                    self._update_step(processing_steps, "market_search", "active", datetime.now().timestamp())
            
            wrapped_task = self.create_wrapped_task(task, mode)
            async_tasks.append(wrapped_task)
        
        # Execute all tasks in parallel
        task_results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # Process results and handle failures
        for task, result in zip(tasks, task_results):
            if isinstance(result, Exception):
                logger.error(f"Agent {task.agent_name} failed: {result}")
                if task.required:
                    # Try fallback for required agents
                    fallback_result = await self.get_fallback_result(task.agent_name, mode)
                    results[task.agent_name] = fallback_result
                else:
                    # Optional agents can fail silently
                    results[task.agent_name] = None
                    # Mark as skipped in processing steps
                    if processing_steps:
                        if task.agent_name == "web_search":
                            self._update_step(processing_steps, "market_search", "skipped")
            else:
                results[task.agent_name] = result
                # Mark as completed in processing steps
                if processing_steps:
                    if task.agent_name == "rag_search":
                        self._update_step(processing_steps, "retrieval", "completed", end_time=datetime.now().timestamp())
                    elif task.agent_name == "web_search":
                        self._update_step(processing_steps, "market_search", "completed", end_time=datetime.now().timestamp())
        
        return results
    
    async def create_wrapped_task(self, task: AgentTask, mode: str) -> Any:
        """
        Wrap agent task with timeout and error handling
        """
        try:
            # Add mode to args if applicable
            args = task.args
            if task.agent_name == "rag_search":
                args = task.args + (mode,)
            
            # Execute with timeout
            return await asyncio.wait_for(
                task.agent_method(*args),
                timeout=task.timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Agent {task.agent_name} timed out after {task.timeout}s")
        except Exception as e:
            raise AgentExecutionError(f"Agent {task.agent_name} failed: {str(e)}")
    
    def should_search_market_data(self, request: NegotiationRequest, context: dict) -> bool:
        """
        Determine if market data search would be beneficial
        """
        # Search if rate is being discussed
        rate_keywords = ["rate", "price", "cost", "budget", "fee", "charge", "hourly", "daily", "project"]
        text_lower = request.negotiation_text.lower()
        
        return any(keyword in text_lower for keyword in rate_keywords)
    
    async def get_fallback_result(self, agent_name: str, mode: str) -> Any:
        """
        Get fallback result for failed agent
        """
        if agent_name == "rag_search":
            # Return cached negotiation tactics
            return CachedNegotiationInsights(
                tactics=["mirroring", "labeling", "calibrated_questions"],
                insights=self.get_default_negotiation_advice(),
                source="cached"
            )
        elif agent_name == "web_search":
            # Return industry average rates
            return MarketRateReport(
                rates={"min": 50, "max": 150, "median": 85},
                currency="USD",
                sources=["industry_average"],
                confidence=0.5
            )
        
        return None
    
    def get_default_negotiation_advice(self) -> List[str]:
        """
        Get default negotiation advice when RAG fails
        """
        return [
            "Use tactical empathy - understand their constraints",
            "Ask calibrated questions like 'How am I supposed to do that?'",
            "Mirror their key concerns to show understanding",
            "Use labels like 'It seems like budget is your main concern'",
            "Always aim for a win-win outcome"
        ]
    
    async def create_fallback_response(
        self, 
        request: NegotiationRequest, 
        partial_results: Dict[str, Any],
        mode: str = "fallback"
    ) -> NegotiationResponse:
        """
        Create best possible response from partial results
        """
        # Use any successful results
        insights = partial_results.get("rag_search")
        market_data = partial_results.get("web_search")
        
        # Basic response construction
        response = NegotiationResponse(
            success=True,
            partial_failure=True,
            rewritten_text=self.create_basic_rewrite(request.negotiation_text),
            strategy=self.create_basic_strategy(insights),
            market_data=market_data,
            insights=insights.insights if insights else [],
            confidence=0.6  # Lower confidence for fallback
        )
        
        return response
    
    def create_basic_rewrite(self, text: str) -> str:
        """
        Create basic rewrite without LLM
        """
        # Simple improvements based on common patterns
        improvements = {
            "I think": "Based on my experience",
            "maybe": "I believe",
            "could": "would",
            "try to": "will"
        }
        
        result = text
        for old, new in improvements.items():
            result = result.replace(old, new)
        
        return result
    
    def create_basic_strategy(self, insights: Any) -> Dict[str, Any]:
        """
        Create basic strategy without synthesis - returns proper dict structure
        """
        techniques = ["tactical_empathy", "anchoring", "calibrated_questions"]
        key_points = [
            "Acknowledge their position",
            "Ask open-ended questions",
            "Focus on mutual benefit"
        ]
        
        if insights and hasattr(insights, 'tactics'):
            techniques = insights.tactics[:3]
            if hasattr(insights, 'insights'):
                key_points = insights.insights[:3]
        
        return {
            "techniques": techniques,
            "key_points": key_points,
            "confidence": 0.5,
            "approach": "collaborative"
        }
    
    async def synthesize_results(
        self, 
        request: NegotiationRequest,
        context: dict,
        agent_results: Dict[str, Any],
        mode: str,
        processing_steps: List[Dict] = None
    ) -> NegotiationResponse:
        """
        Send all agent results to synthesis agent for final response
        """
        try:
            # Update strategy step
            if processing_steps:
                self._update_step(processing_steps, "strategy", "active", datetime.now().timestamp())
            
            # Prepare synthesis input
            
            synthesis_input = SynthesisInput(
                original_request=request,
                context=context,
                rag_insights=agent_results.get("rag_search"),
                market_data=agent_results.get("web_search"),
                mode=mode
            )
            
            # Get synthesized response (placeholder implementation)
            if self.synthesis_agent:
                response = await self.synthesis_agent.synthesize(synthesis_input)
            else:
                # Fallback synthesis when agent not available
                response = await self.create_fallback_synthesis(synthesis_input)
            
            # Update strategy step as completed
            if processing_steps:
                self._update_step(processing_steps, "strategy", "completed", end_time=datetime.now().timestamp())
            
            # Add metadata
            response.mode = mode
            response.agents_used = list(agent_results.keys())
            response.retrieval_time = sum(
                r.retrieval_time for r in agent_results.values() 
                if hasattr(r, 'retrieval_time')
            )
            
            # Add retrieved contexts from RAG search
            rag_result = agent_results.get("rag_search")
            if rag_result and hasattr(rag_result, 'retrieved_contexts'):
                response.retrieved_contexts = rag_result.retrieved_contexts
            elif rag_result and hasattr(rag_result, 'contexts'):
                response.retrieved_contexts = rag_result.contexts
            
            # Add detailed contexts if available
            if rag_result and hasattr(rag_result, 'detailed_contexts'):
                response.detailed_contexts = rag_result.detailed_contexts
            
            # Add similarity score and context corridor flag
            if rag_result:
                response.avg_similarity_score = getattr(rag_result, 'avg_similarity_score', None)
                response.context_corridor_used = getattr(rag_result, 'context_corridor_used', False)
                
                # Add retrieval details
                response.retrieval_details = {
                    "documents_searched": getattr(rag_result, 'source', 'unknown'),
                    "chunks_retrieved": len(rag_result.retrieved_contexts) if hasattr(rag_result, 'retrieved_contexts') else 0,
                    "relevance_threshold": 0.7,  # Default threshold
                    "mode": mode,
                    "context_corridor_used": getattr(rag_result, 'context_corridor_used', False)
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Fallback to best available result
            return await self.create_fallback_response(request, agent_results, mode)
    
    def should_search_market_data(self, request: NegotiationRequest, context: dict) -> bool:
        """
        Determine if market data search would be beneficial
        """
        # Search if rate is being discussed
        rate_keywords = ["rate", "price", "cost", "budget", "fee", "charge", "hourly", "daily", "project"]
        text_lower = request.negotiation_text.lower()
        
        return any(keyword in text_lower for keyword in rate_keywords)
    
    async def get_fallback_result(self, agent_name: str, mode: str) -> Any:
        """
        Get fallback result for failed agent
        """
        if agent_name == "rag_search":
            # Return cached negotiation tactics
            from .models import CachedNegotiationInsights
            return CachedNegotiationInsights(
                tactics=["mirroring", "labeling", "calibrated_questions"],
                insights=self.get_default_negotiation_advice(),
                source="cached"
            )
        elif agent_name == "web_search":
            # Return industry average rates
            from .models import MarketRateReport
            return MarketRateReport(
                min_rate=50,
                max_rate=150,
                median_rate=85,
                currency="USD",
                sources=["industry_average"],
                confidence=0.5
            )
        
        return None
    
    async def create_fallback_response(
        self, 
        request: NegotiationRequest, 
        partial_results: Dict[str, Any],
        mode: str = "fallback"
    ) -> NegotiationResponse:
        """
        Create best possible response from partial results
        """
        # Use any successful results
        insights = partial_results.get("rag_search")
        market_data = partial_results.get("web_search")
        
        # Basic response construction
        response = NegotiationResponse(
            success=True,
            partial_failure=True,
            rewritten_text=self.create_basic_rewrite(request.negotiation_text),
            strategy=self.create_basic_strategy(insights),
            market_data=market_data,
            insights=insights.insights if insights else [],
            confidence=0.6,  # Lower confidence for fallback
            request_id=str(hash(request.negotiation_text)),
            mode=mode
        )
        
        # Add retrieved contexts from RAG search
        if insights and hasattr(insights, 'retrieved_contexts'):
            response.retrieved_contexts = insights.retrieved_contexts
            response.avg_similarity_score = getattr(insights, 'avg_similarity_score', None)
            response.context_corridor_used = getattr(insights, 'context_corridor_used', False)
        
        return response
    
    def get_default_negotiation_advice(self) -> List[str]:
        """
        Get default negotiation advice when agents fail
        """
        return [
            "Use mirroring to build rapport with the client",
            "Ask calibrated questions to understand their perspective",
            "Anchor your rate based on market research",
            "Use labeling to acknowledge their concerns",
            "Never split the difference - find creative solutions"
        ]
    
    def create_basic_rewrite(self, original_text: str) -> str:
        """
        Create basic rewrite when synthesis agent fails - using Chris Voss techniques
        """
        # Check if it's about rates/pricing
        if any(word in original_text.lower() for word in ['rate', 'hour', 'pay', 'budget', 'cost', 'price', '$']):
            return "I understand you're working with a specific budget. Help me understand - what are the key constraints you're facing that led to this figure? I want to ensure we find an arrangement that works for both of us."
        
        # Check if it's about timeline
        if any(word in original_text.lower() for word in ['deadline', 'timeline', 'when', 'date', 'rush', 'urgent']):
            return "I hear the timeline is important to you. What's driving this particular deadline? Understanding your constraints will help me propose the best solution."
        
        # Check if it's about scope/requirements
        if any(word in original_text.lower() for word in ['requirement', 'feature', 'need', 'want', 'scope']):
            return "It sounds like you have specific needs in mind. How did you arrive at these particular requirements? I want to make sure I fully understand your priorities."
        
        # Default response using tactical empathy and calibrated questions
        return "I appreciate you bringing this to my attention. How can we work together to find a solution that addresses your needs while ensuring a successful outcome for both of us?"
    
    def create_basic_strategy(self, insights: Any) -> Dict[str, Any]:
        """
        Create basic strategy from available insights using Chris Voss techniques
        """
        # Default Chris Voss techniques
        techniques = ["tactical_empathy", "calibrated_questions", "mirroring"]
        key_points = [
            "Understand their constraints",
            "Build rapport through empathy",
            "Use open-ended questions"
        ]
        
        if insights and hasattr(insights, 'tactics'):
            # Use tactics from RAG search if available
            techniques = insights.tactics[:3]
            if hasattr(insights, 'insights'):
                key_points = insights.insights[:3]
        
        return {
            "techniques": techniques,
            "key_points": key_points,
            "confidence": 0.6,
            "approach": "collaborative"
        }
    
    async def create_fallback_synthesis(self, synthesis_input: SynthesisInput) -> NegotiationResponse:
        """
        Create fallback synthesis when synthesis agent is unavailable
        """
        # Extract key information
        request = synthesis_input.original_request
        context = synthesis_input.context
        
        # Create basic improved text
        improved_text = self.create_basic_rewrite(request.negotiation_text)
        
        # Create strategy from available data
        strategy = {
            "techniques": ["acknowledge", "research", "counter_offer"],
            "confidence": 0.5
        }
        
        # Extract insights
        insights = []
        if synthesis_input.rag_insights:
            insights = getattr(synthesis_input.rag_insights, 'insights', [])
        
        return NegotiationResponse(
            success=True,
            degraded=True,
            rewritten_text=improved_text,
            strategy=strategy,
            market_data=synthesis_input.market_data,
            insights=insights,
            confidence=0.5,
            request_id=str(hash(request.negotiation_text)),
            mode=synthesis_input.mode,
            error_recovery="fallback_synthesis"
        )