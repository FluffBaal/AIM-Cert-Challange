# web_search_agent.py - Web search agent with self-correcting Exa AI integration
import asyncio
import statistics
import re
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

from langchain_openai import ChatOpenAI

from .models import (
    ApiConfiguration,
    MarketRateReport,
    ExaAPIError,
    ExaResult
)
from .error_handler import RateLimiter

logger = logging.getLogger(__name__)

# Import real Exa client
try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    logger.warning("Exa API not available, using mock implementation")
    EXA_AVAILABLE = False

class ExaClientWrapper:
    """Wrapper for Exa API client with fallback to mock"""
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        if EXA_AVAILABLE and api_key:
            try:
                self.client = Exa(api_key)
                self.use_real_api = True
                logger.info("Initialized real Exa API client")
            except Exception as e:
                logger.error(f"Failed to initialize Exa client: {e}")
                self.client = None
                self.use_real_api = False
        else:
            self.client = None
            self.use_real_api = False
            if not api_key:
                logger.info("No Exa API key provided, using fallback")
    
    async def search(self, query: str, num_results: int = 10, type: str = "neural", 
                    include_domains: List[str] = None, start_published_date: str = None):
        """Search with real Exa API or fallback"""
        if self.use_real_api and self.client:
            try:
                # Use real Exa API
                kwargs = {
                    "query": query,
                    "num_results": num_results,
                    "type": type
                }
                if include_domains:
                    kwargs["include_domains"] = include_domains
                if start_published_date:
                    kwargs["start_published_date"] = start_published_date
                
                logger.info(f"Calling Exa search with query: {query}")
                logger.info(f"Exa search kwargs: {kwargs}")
                    
                # Exa client is synchronous, so run in executor
                import asyncio
                loop = asyncio.get_event_loop()
                # Create a wrapper function for the search
                def do_search():
                    # First search to get IDs
                    search_response = self.client.search(**kwargs)
                    if search_response and hasattr(search_response, 'results') and search_response.results:
                        # Get the IDs
                        ids = [r.id for r in search_response.results]
                        # Get contents for those IDs
                        contents_response = self.client.get_contents(ids)
                        # Merge content into results
                        for i, result in enumerate(search_response.results):
                            if i < len(contents_response.results):
                                result.text = contents_response.results[i].text
                    return search_response
                response = await loop.run_in_executor(None, do_search)
                
                logger.info(f"Exa search returned: {response}")
                logger.info(f"Exa results count: {len(response.results) if hasattr(response, 'results') else 0}")
                
                # Convert Exa response to our format
                results = []
                for result in response.results:
                    text = getattr(result, 'text', '') or getattr(result, 'title', '')
                    url = getattr(result, 'url', '')
                    results.append(ExaResult(text, url))
                
                # Create a simple response object
                class Response:
                    def __init__(self):
                        self.results = results
                return Response()
                
            except Exception as e:
                import traceback
                logger.error(f"Exa API search failed: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Fall back to mock
        
        # Fallback to mock response
        return MockExaResponse()

class MockExaResponse:
    """Mock response for fallback implementation"""
    def __init__(self):
        self.results = [
            ExaResult("Freelance developer rates range from $75-150/hour in 2024", "https://example.com/rates1"),
            ExaResult("UX designer hourly rates: Junior $60-85, Senior $100-150", "https://example.com/rates2"),
            ExaResult("Market survey: Average freelance rates by skill and location", "https://example.com/rates3")
        ]

class WebSearchAgent:
    """
    Fetches real-time market data via Exa AI with self-correcting capabilities
    - Constructs targeted queries based on freelancer details
    - Parses and structures market rate information
    - Self-corrects with follow-up queries if initial results are insufficient
    - Returns MarketRateReport with sources
    """
    
    def __init__(self, api_config: ApiConfiguration):
        self.api_config = api_config
        self.exa_client = ExaClientWrapper(api_key=api_config.exa_key if api_config.exa_key else None)
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=60,
            max_requests_per_hour=1000
        )
        self.max_search_iterations = 3
        self.grader_llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.1,
            api_key=api_config.openai_key
        )
        
    async def search_market_data(self, context: dict) -> MarketRateReport:
        """
        Construct and execute market data queries with self-correction
        """
        search_iterations = 0
        all_results = []
        search_queries = self.construct_queries(context)
        
        while search_iterations < self.max_search_iterations:
            try:
                # Rate limiting check
                await self.rate_limiter.check_rate_limit()
                
                # Execute searches with Exa AI
                iteration_results = []
                for query in search_queries:
                    response = await self.exa_client.search(
                        query=query,
                        num_results=10,
                        type="neural",
                        include_domains=["glassdoor.com", "indeed.com", "upwork.com", "freelancer.com"],
                        start_published_date="2024-01-01"
                    )
                    iteration_results.extend(response.results)
                
                all_results.extend(iteration_results)
                
                # Parse and structure results
                market_report = await self.parse_market_data(all_results, context)
                
                # Grade the results
                grade_result = await self.grade_market_data(market_report, context)
                
                if grade_result["grade"] == "pass":
                    return market_report
                
                # Generate follow-up queries if needed
                search_queries = grade_result["follow_up_queries"]
                search_iterations += 1
                
                logger.info(f"Market data incomplete, iteration {search_iterations} with queries: {search_queries}")
                
            except ExaAPIError as e:
                # Fallback strategy when Exa is unavailable
                return await self.fallback_market_search(context)
        
        # Return best available results after max iterations
        return market_report
    
    def construct_queries(self, context: dict) -> List[str]:
        """
        Build targeted search queries
        """
        skill = context.get("skill", "freelancer")
        location = context.get("location", "remote")
        experience = context.get("experience_years", 5)
        
        queries = [
            f"{skill} hourly rate {location} {datetime.now().year}",
            f"freelance {skill} day rate {location} {experience} years experience",
            f"{skill} contractor rates market data {location}"
        ]
        
        # Add project-specific queries if available
        if context.get("project_type"):
            queries.append(f"{skill} {context['project_type']} project rates")
            
        return queries
    
    async def parse_market_data(self, results: List[ExaResult], context: dict) -> MarketRateReport:
        """
        Extract structured rate information from search results
        """
        rates = []
        
        for result in results:
            # Use Exa's content extraction
            content = result.text
            
            # Extract rate mentions using regex patterns
            # Updated patterns to be more specific and avoid false matches
            rate_patterns = [
                r"\$([1-9]\d{0,2}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per\s*)?(?:hour|hr|/hr)",  # $X per hour
                r"\$([1-9]\d{0,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per\s*)?(?:day|/day)",     # $X per day (higher range)
                r"([1-9]\d{1,2})\s*(?:USD|EUR|GBP)\s*(?:per\s*)?(?:hour|hr)"              # X EUR/USD/GBP per hour
            ]
            
            for i, pattern in enumerate(rate_patterns):
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    rate = self.normalize_rate(match)
                    if rate is not None:
                        # Convert day rates to hourly (assuming 8 hour day)
                        if i == 1 and '/day' in pattern:
                            rate = rate / 8
                        rates.append(rate)
        
        # Calculate statistics
        if rates:
            # Filter out outliers (rates below $20/hr are likely errors for tech roles)
            valid_rates = [r for r in rates if r >= 20]
            if not valid_rates:
                # If no rates meet threshold, use a higher percentile of all rates
                valid_rates = sorted(rates)[len(rates)//4:]  # Use top 75% of rates
                
            return MarketRateReport(
                min_rate=min(valid_rates),
                max_rate=max(valid_rates),
                median_rate=statistics.median(valid_rates),
                average_rate=statistics.mean(valid_rates),
                sources=[r.url for r in results[:5]],
                currency=self.detect_currency(context),
                data_points=len(valid_rates),
                location=context.get("location"),
                skill=context.get("skill")
            )
        else:
            # No rates found
            return await self.fallback_market_search(context)
    
    def normalize_rate(self, rate_string: str) -> float:
        """
        Convert rate string to normalized float value
        """
        try:
            # Remove commas and convert to float
            cleaned = re.sub(r'[,\s]', '', rate_string)
            rate = float(cleaned)
            # Filter out invalid rates
            if rate <= 0 or rate > 1000:  # Reasonable bounds for hourly rates
                logger.warning(f"Skipping invalid rate: {rate}")
                return None
            return rate
        except (ValueError, TypeError):
            logger.warning(f"Could not parse rate: {rate_string}")
            return None
    
    def detect_currency(self, context: dict) -> str:
        """
        Detect currency based on location context
        """
        location = context.get("location", "").lower()
        
        if any(country in location for country in ["uk", "britain", "england", "scotland", "wales"]):
            return "GBP"
        elif any(country in location for country in ["germany", "france", "spain", "italy", "netherlands"]):
            return "EUR"
        else:
            return "USD"  # Default
    
    async def grade_market_data(self, market_report: MarketRateReport, context: dict) -> dict:
        """
        Grade the quality of market data and generate follow-up queries if needed
        """
        grading_prompt = f"""
        Evaluate the market rate data for a {context.get('skill', 'freelancer')} in {context.get('location', 'remote')}.
        
        Current data found:
        - Data points: {market_report.data_points}
        - Rate range: ${market_report.min_rate} - ${market_report.max_rate}
        - Median: ${market_report.median_rate}
        - Sources: {len(market_report.sources)}
        
        Requirements for PASS grade:
        1. At least 5 data points from different sources
        2. Rates should be from the last 12 months
        3. Include both hourly and day rates if possible
        4. Cover the specific skill level ({context.get('experience_years', 5)} years)
        
        Grade this as 'pass' or 'fail'. If 'fail', provide 2-3 specific follow-up queries to gather missing information.
        
        Return JSON: {{"grade": "pass/fail", "follow_up_queries": ["query1", "query2"]}}
        """
        
        try:
            response = await self.grader_llm.ainvoke(grading_prompt)
            result = json.loads(response.content)
            return {
                "grade": result.get("grade", "fail"),
                "follow_up_queries": result.get("follow_up_queries", [])
            }
        except Exception as e:
            logger.warning(f"Grading failed: {e}")
            # Default to pass if parsing fails
            return {"grade": "pass", "follow_up_queries": []}
    
    async def fallback_market_search(self, context: dict) -> MarketRateReport:
        """
        Fallback when Exa AI is unavailable or returns no results
        """
        # Use cached historical data
        cached_data = await self.get_cached_market_data(context)
        
        if cached_data:
            return MarketRateReport(
                **cached_data,
                sources=["cached_data"],
                note="Using cached market data due to API unavailability"
            )
        
        # Ultimate fallback: industry standard estimates
        skill_rates = {
            "ux designer": {"min": 75, "max": 150, "median": 100},
            "developer": {"min": 85, "max": 175, "median": 120},
            "data scientist": {"min": 100, "max": 200, "median": 140},
            "graphic designer": {"min": 50, "max": 120, "median": 75},
            "copywriter": {"min": 40, "max": 100, "median": 65},
            "project manager": {"min": 70, "max": 140, "median": 95},
            "consultant": {"min": 80, "max": 200, "median": 125}
        }
        
        base_rates = skill_rates.get(
            context.get("skill", "").lower(), 
            {"min": 50, "max": 125, "median": 85}
        )
        
        # Adjust for experience
        experience_multiplier = 1 + (context.get("experience_years", 5) * 0.05)
        
        return MarketRateReport(
            min_rate=base_rates["min"] * experience_multiplier,
            max_rate=base_rates["max"] * experience_multiplier,
            median_rate=base_rates["median"] * experience_multiplier,
            sources=["fallback_estimates"],
            note="Estimated rates based on industry standards",
            data_points=3,
            location=context.get("location"),
            skill=context.get("skill"),
            confidence=0.5
        )
    
    async def get_cached_market_data(self, context: dict) -> Dict[str, Any]:
        """
        Retrieve cached market data if available
        """
        # In a real implementation, this would query a cache/database
        # For now, return None to trigger fallback estimates
        return None
    
