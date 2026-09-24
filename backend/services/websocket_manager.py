"""
WebSocket Connection Manager
Manages active WebSocket connections for real-time alert pushing.
Each streaming session has a unique session_id mapping to a WebSocket.
"""
import asyncio
from datetime import datetime
from fastapi import WebSocket

class ConnectionManager:
    """
    Manages WebSocket connections keyed by session_id.
    Supports per-session messaging and broadcast to all sessions.
    """
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        """
        Accept a WebSocket connection and register it under session_id.
        Sends a 'connected' confirmation message.
        """
        await websocket.accept()
        async with self._lock:
            self.active_connections[session_id] = websocket
        await self.send_json(session_id, {
            "type": "connected",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"message": "VoiceShield stream connected. Send PCM audio chunks."}
        })
    
    async def disconnect(self, session_id: str) -> None:
        """Remove a disconnected session."""
        async with self._lock:
            if session_id in self.active_connections:
                del self.active_connections[session_id]
    
    async def send_json(self, session_id: str, data: dict) -> None:
        """
        Send JSON data to a specific session.
        Silently handles disconnected clients.
        """
        async with self._lock:
            websocket = self.active_connections.get(session_id)
        if websocket:
            try:
                await websocket.send_json(data)
            except Exception:
                await self.disconnect(session_id)
    
    async def broadcast(self, data: dict) -> None:
        """
        Send JSON data to ALL connected sessions.
        Removes disconnected clients automatically.
        """
        async with self._lock:
            sessions = list(self.active_connections.keys())
            
        for session_id in sessions:
            try:
                websocket = self.active_connections.get(session_id)
                if websocket:
                    await websocket.send_json(data)
            except Exception:
                await self.disconnect(session_id)
    
    def get_active_sessions(self) -> list[str]:
        """Return list of currently active session IDs."""
        return list(self.active_connections.keys())
    
    def session_count(self) -> int:
        """Return number of active connections."""
        return len(self.active_connections)

# Module-level singleton used by all routers
manager = ConnectionManager()
