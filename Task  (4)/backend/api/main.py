# main.py - FastAPI application with comprehensive error handling
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

app = FastAPI(
    title="Freelancer Negotiation Helper API",
    version="1.0.0",
    docs_url="/api/docs"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID Middleware
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_var.set(request_id)
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

# Timing Middleware
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log slow requests
        if process_time > 5.0:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {process_time:.2f}s"
            )
        
        return response

# Error Handling Middleware
class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
            
        except HTTPException:
            # Let FastAPI handle HTTP exceptions
            raise
            
        except ValueError as e:
            # Handle validation errors
            logger.error(f"Validation error: {e}", exc_info=True)
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid request data",
                    "detail": str(e),
                    "request_id": request_id_var.get()
                }
            )
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error: {e}", exc_info=True)
            
            # Don't expose internal errors in production
            DEBUG = os.getenv("DEBUG", "false").lower() == "true"
            if DEBUG:
                error_detail = str(e)
            else:
                error_detail = "An internal error occurred"
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": error_detail,
                    "request_id": request_id_var.get()
                }
            )

# Add middleware in order
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)

# Global exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid value",
            "detail": str(exc),
            "request_id": request_id_var.get()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "request_id": request_id_var.get()
        }
    )

# Import dependencies with error handling
try:
    from .dependencies import (
        initialize_qdrant_client,
        warm_up_models,
        load_fallback_data,
        close_all_connections,
        persist_cache_data,
        qdrant_client,
        redis_client,
        check_openai_api_cached
    )
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False
    
    # Create stub functions
    async def initialize_qdrant_client(): pass
    async def warm_up_models(): pass
    async def load_fallback_data(): pass
    async def close_all_connections(): pass
    async def persist_cache_data(): pass
    qdrant_client = None
    redis_client = None
    async def check_openai_api_cached(): return False
try:
    from .routers import negotiation_router, websocket_router
except ImportError:
    # Fallback imports
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from routers import negotiation_router, websocket_router

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Freelancer Negotiation Helper API")
    
    # Initialize vector database connection
    await initialize_qdrant_client()
    
    # Warm up models
    await warm_up_models()
    
    # Load fallback data
    await load_fallback_data()
    
    logger.info("API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API")
    
    # Close database connections
    await close_all_connections()
    
    # Save cache data
    await persist_cache_data()
    
    logger.info("API shutdown complete")

# Health check endpoint with detailed status
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": app.version,
        "services": {}
    }
    
    # Check Qdrant
    try:
        if qdrant_client and DEPENDENCIES_AVAILABLE:
            await qdrant_client.health()
            health_status["services"]["qdrant"] = "healthy"
        else:
            health_status["services"]["qdrant"] = "unavailable"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["qdrant"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        if redis_client and DEPENDENCIES_AVAILABLE:
            await redis_client.ping()
            health_status["services"]["redis"] = "healthy"
        else:
            health_status["services"]["redis"] = "unavailable"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check OpenAI API (with cached result)
    try:
        if DEPENDENCIES_AVAILABLE and await check_openai_api_cached():
            health_status["services"]["openai"] = "healthy"
        else:
            health_status["services"]["openai"] = "unavailable"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["openai"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)

# Include routers
app.include_router(negotiation_router, prefix="/api")
app.include_router(websocket_router, prefix="/ws")