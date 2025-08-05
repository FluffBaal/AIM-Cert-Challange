# routers/websocket.py - WebSocket endpoints for real-time communication
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_json(message)

# Global connection manager
manager = ConnectionManager()

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time negotiation analysis
    """
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Process the message
            try:
                response = await process_websocket_message(data, client_id)
                await manager.send_personal_message(response, client_id)
                
            except Exception as e:
                # Send error response
                error_response = {
                    "type": "error",
                    "message": f"Processing error: {str(e)}",
                    "client_id": client_id
                }
                await manager.send_personal_message(error_response, client_id)
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(client_id)

async def process_websocket_message(data: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    """
    Process incoming WebSocket messages
    """
    message_type = data.get("type", "unknown")
    
    if message_type == "negotiate":
        # Handle negotiation request
        request_data = data.get("request", {})
        
        # Create placeholder response
        response = {
            "type": "negotiation_result",
            "client_id": client_id,
            "result": {
                "success": True,
                "rewritten_text": "Thank you for your message. I'd like to discuss this further.",
                "strategy": {"techniques": ["acknowledge", "research"], "confidence": 0.7},
                "confidence": 0.7,
                "mode": "realtime"
            }
        }
        
        return response
        
    elif message_type == "ping":
        # Handle ping
        return {
            "type": "pong",
            "client_id": client_id,
            "timestamp": data.get("timestamp")
        }
        
    else:
        # Unknown message type
        return {
            "type": "error",
            "message": f"Unknown message type: {message_type}",
            "client_id": client_id
        }