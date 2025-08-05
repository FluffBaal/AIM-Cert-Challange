"""
RAGAS Evaluation Service for Freelancer Negotiation Helper

This service provides continuous evaluation of the RAG system using RAGAS metrics.
"""

import os
import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from qdrant_client import QdrantClient
from langchain.embeddings import OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Freelancer Negotiation Evaluation API",
    description="RAGAS evaluation service for RAG quality monitoring",
    version="1.0.0"
)

# Global variables
qdrant_client = None
embeddings = None

# Configuration from environment
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
EVALUATION_MODE = os.getenv("EVALUATION_MODE", "batch")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class EvaluationRequest(BaseModel):
    """Request model for evaluation"""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str = ""
    mode: str = "advanced"


class EvaluationResponse(BaseModel):
    """Response model for evaluation results"""
    metrics: Dict[str, float]
    timestamp: str
    mode: str
    status: str


@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    global qdrant_client, embeddings
    
    try:
        # Initialize Qdrant client
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        
        # Initialize embeddings (requires API key from request headers in production)
        if OPENAI_API_KEY:
            embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
            logger.info("OpenAI embeddings initialized")
        else:
            logger.warning("No OpenAI API key in environment - evaluation will require API key in requests")
            
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("Shutting down evaluation service")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "qdrant": "connected" if qdrant_client else "disconnected",
            "mode": EVALUATION_MODE
        }
    }
    return health_status


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_rag(request: EvaluationRequest):
    """
    Evaluate RAG response using RAGAS metrics
    """
    try:
        # Prepare data for RAGAS
        eval_data = {
            "question": [request.question],
            "answer": [request.answer],
            "contexts": [request.contexts],
            "ground_truth": [request.ground_truth] if request.ground_truth else [""]
        }
        
        # Convert to dataset
        dataset = Dataset.from_dict(eval_data)
        
        # Define metrics to evaluate
        metrics = [answer_relevancy, context_precision]
        if request.ground_truth:
            metrics.extend([faithfulness, context_recall])
        
        # Run evaluation
        logger.info(f"Running RAGAS evaluation for mode: {request.mode}")
        results = evaluate(
            dataset,
            metrics=metrics,
            llm=None,  # Will use default
            embeddings=embeddings
        )
        
        # Extract scores
        metric_scores = {
            metric.name: float(results[metric.name])
            for metric in metrics
        }
        
        return EvaluationResponse(
            metrics=metric_scores,
            timestamp=datetime.utcnow().isoformat(),
            mode=request.mode,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_evaluate")
async def batch_evaluate():
    """
    Run batch evaluation on stored test cases
    """
    try:
        # This would load test cases from a file or database
        test_cases_path = "/app/test_cases.json"
        
        if os.path.exists(test_cases_path):
            with open(test_cases_path, 'r') as f:
                test_cases = json.load(f)
        else:
            # Default test cases
            test_cases = [
                {
                    "question": "What is mirroring in negotiation?",
                    "ground_truth": "Mirroring is repeating the last three words or critical words of what someone said."
                },
                {
                    "question": "How do you use tactical empathy?",
                    "ground_truth": "Tactical empathy involves understanding and acknowledging the other person's feelings and perspective."
                }
            ]
        
        results = []
        for test_case in test_cases:
            # Here you would call your RAG system to get answers and contexts
            # For now, returning placeholder
            result = {
                "test_case": test_case,
                "evaluation": "pending",
                "timestamp": datetime.utcnow().isoformat()
            }
            results.append(result)
        
        return {
            "batch_id": datetime.utcnow().isoformat(),
            "total_cases": len(test_cases),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/summary")
async def get_metrics_summary():
    """
    Get summary of evaluation metrics over time
    """
    try:
        # This would retrieve historical metrics from storage
        # For now, returning sample data
        return {
            "summary": {
                "average_relevancy": 0.85,
                "average_precision": 0.78,
                "evaluations_count": 42,
                "last_evaluation": datetime.utcnow().isoformat()
            },
            "trends": {
                "relevancy_trend": "improving",
                "precision_trend": "stable"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )