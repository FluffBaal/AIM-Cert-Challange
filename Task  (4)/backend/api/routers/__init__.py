# routers/__init__.py - API routers package
"""
API routers for the Freelancer Negotiation Helper

This package contains FastAPI routers for:
- negotiation_router: Main negotiation analysis endpoints
- websocket_router: WebSocket endpoints for real-time communication
"""

from .negotiation import router as negotiation_router
from .websocket import router as websocket_router

__all__ = ["negotiation_router", "websocket_router"]