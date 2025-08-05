"""Progress tracking and WebSocket emission utilities"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ProgressEmitter:
    """Emit progress updates via WebSocket during negotiation analysis"""
    
    def __init__(self, websocket_manager=None, client_id: Optional[str] = None):
        self.websocket_manager = websocket_manager
        self.client_id = client_id
        self.processing_steps: List[Dict[str, Any]] = []
        self._initialize_steps()
    
    def _initialize_steps(self):
        """Initialize the processing steps"""
        steps_config = [
            ("retrieval", "Knowledge Retrieval", "Searching negotiation tactics database"),
            ("context_analysis", "Context Analysis", "Understanding your negotiation scenario"),
            ("market_search", "Market Research", "Fetching real-time rate data"),
            ("strategy", "Strategy Formation", "Applying Chris Voss techniques"),
            ("response", "Response Generation", "Crafting your negotiation response")
        ]
        
        for step_id, name, desc in steps_config:
            self.processing_steps.append({
                "id": step_id,
                "name": name,
                "description": desc,
                "status": "pending",
                "startTime": None,
                "endTime": None
            })
    
    async def update_step(self, step_id: str, status: str, start_time: float = None, end_time: float = None):
        """Update a processing step and emit via WebSocket"""
        for step in self.processing_steps:
            if step["id"] == step_id:
                step["status"] = status
                if start_time:
                    step["startTime"] = start_time
                if end_time:
                    step["endTime"] = end_time
                break
        
        # Emit update via WebSocket if available
        if self.websocket_manager and self.client_id:
            try:
                await self.websocket_manager.send_personal_message({
                    "type": "processing_update",
                    "processing_steps": self.processing_steps,
                    "current_step": step_id,
                    "status": status,
                    "timestamp": datetime.now().isoformat()
                }, self.client_id)
            except Exception as e:
                logger.warning(f"Failed to emit progress update: {e}")
    
    def get_steps(self) -> List[Dict[str, Any]]:
        """Get the current processing steps"""
        return self.processing_steps

# Global progress emitter registry
progress_emitters: Dict[str, ProgressEmitter] = {}

def get_progress_emitter(request_id: str, websocket_manager=None, client_id: Optional[str] = None) -> ProgressEmitter:
    """Get or create a progress emitter for a request"""
    if request_id not in progress_emitters:
        progress_emitters[request_id] = ProgressEmitter(websocket_manager, client_id)
    return progress_emitters[request_id]

def cleanup_progress_emitter(request_id: str):
    """Clean up a progress emitter after request completion"""
    if request_id in progress_emitters:
        del progress_emitters[request_id]