"""
Modern Web Interface for Aider Hive Architecture.

This module provides a comprehensive web-based interface for monitoring,
interacting with, and managing the Hive system. It includes real-time
updates, interactive debugging, session management, and visualization
of agent activities.

Key Features:
- Real-time WebSocket communication for live updates
- Interactive dashboard for system monitoring
- Session management and history viewing
- Code diff visualization and editing
- Chat interface with streaming responses
- Agent activity visualization
- Performance metrics and health monitoring
- Interactive debugging workflows
- RESTful API for external integrations
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from collections import defaultdict, deque

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..hive_coordinator import HiveCoordinator
from ..session.session_manager import SessionManager, SessionStatus
from ..session.session_events import EventType
from ..agents.base_agent import AgentMessage, MessagePriority


# Pydantic models for API requests/responses
class SessionCreateRequest(BaseModel):
    user_id: str
    project_path: str
    session_type: str = "default"
    initial_context: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    session_id: str
    status: str
    created_at: str
    last_activity: Optional[str] = None
    user_id: str
    project_path: str


class MessageRequest(BaseModel):
    content: str
    message_type: str = "user_input"
    context: Optional[Dict[str, Any]] = None


class MessageResponse(BaseModel):
    message_id: str
    content: str
    sender: str
    timestamp: str
    message_type: str = "response"


class AgentStatusResponse(BaseModel):
    agent_id: str
    agent_type: str
    status: str
    current_task: Optional[str] = None
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)


class SystemStatusResponse(BaseModel):
    hive_state: str
    active_sessions: int
    total_agents: int
    active_agents: int
    system_health: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class WebSocketManager:
    """Manages WebSocket connections for real-time communication."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, Set[str]] = defaultdict(set)
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        self.logger = structlog.get_logger().bind(component="websocket_manager")

    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket

        if session_id:
            self.session_connections[session_id].add(connection_id)

        self.connection_metadata[connection_id] = {
            'session_id': session_id,
            'user_id': user_id,
            'connected_at': datetime.utcnow(),
            'last_activity': datetime.utcnow()
        }

        self.logger.info(f"WebSocket connected: {connection_id}")

    def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]

        # Remove from session connections
        metadata = self.connection_metadata.get(connection_id, {})
        session_id = metadata.get('session_id')
        if session_id and connection_id in self.session_connections[session_id]:
            self.session_connections[session_id].remove(connection_id)

        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]

        self.logger.info(f"WebSocket disconnected: {connection_id}")

    async def send_personal_message(self, message: Dict[str, Any], connection_id: str):
        """Send a message to a specific connection."""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.send_json(message)

                # Update last activity
                if connection_id in self.connection_metadata:
                    self.connection_metadata[connection_id]['last_activity'] = datetime.utcnow()

            except Exception as e:
                self.logger.error(f"Failed to send message to {connection_id}: {e}")
                self.disconnect(connection_id)

    async def broadcast_to_session(self, message: Dict[str, Any], session_id: str):
        """Broadcast a message to all connections in a session."""
        connection_ids = list(self.session_connections.get(session_id, set()))

        for connection_id in connection_ids:
            await self.send_personal_message(message, connection_id)

    async def broadcast_system_message(self, message: Dict[str, Any]):
        """Broadcast a system message to all connections."""
        connection_ids = list(self.active_connections.keys())

        for connection_id in connection_ids:
            await self.send_personal_message(message, connection_id)

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        now = datetime.utcnow()
        active_sessions = len([s for s in self.session_connections.keys() if self.session_connections[s]])

        connection_ages = []
        for metadata in self.connection_metadata.values():
            age = (now - metadata['connected_at']).total_seconds()
            connection_ages.append(age)

        return {
            'total_connections': len(self.active_connections),
            'active_sessions': active_sessions,
            'average_connection_age_seconds': sum(connection_ages) / len(connection_ages) if connection_ages else 0,
            'connections_by_session': {k: len(v) for k, v in self.session_connections.items()}
        }


class HiveWebApp:
    """
    Main web application class for the Hive system interface.

    Provides REST API endpoints, WebSocket support, and serves the web interface
    for monitoring and interacting with the Hive system.
    """

    def __init__(
        self,
        hive_coordinator: HiveCoordinator,
        host: str = "localhost",
        port: int = 8080,
        debug: bool = False,
        enable_cors: bool = True,
        static_directory: Optional[str] = None,
        template_directory: Optional[str] = None
    ):
        """Initialize the web application."""
        self.hive_coordinator = hive_coordinator
        self.host = host
        self.port = port
        self.debug = debug

        self.logger = structlog.get_logger().bind(component="hive_webapp")

        # Initialize FastAPI app
        self.app = FastAPI(
            title="Aider Hive System",
            description="Web interface for the Aider Multi-Agent Hive Architecture",
            version="1.0.0",
            debug=debug
        )

        # WebSocket manager
        self.websocket_manager = WebSocketManager()

        # Message queues for streaming responses
        self.streaming_queues: Dict[str, asyncio.Queue] = {}

        # Active request tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}

        # Setup middleware
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"] if debug else ["http://localhost:3000", "http://127.0.0.1:3000"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"]
            )

        self.app.add_middleware(GZipMiddleware, minimum_size=1000)

        # Setup static files and templates
        self.static_dir = static_directory or str(Path(__file__).parent / "static")
        self.template_dir = template_directory or str(Path(__file__).parent / "templates")

        if Path(self.static_dir).exists():
            self.app.mount("/static", StaticFiles(directory=self.static_dir), name="static")

        if Path(self.template_dir).exists():
            self.templates = Jinja2Templates(directory=self.template_dir)
        else:
            self.templates = None

        # Setup routes
        self._setup_routes()

        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()

    def _setup_routes(self):
        """Setup all API routes and WebSocket endpoints."""

        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """Basic health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

        # System status endpoint
        @self.app.get("/api/system/status", response_model=SystemStatusResponse)
        async def get_system_status():
            """Get comprehensive system status."""
            try:
                status = await self.hive_coordinator.get_system_status()

                return SystemStatusResponse(
                    hive_state=status.get('hive_state', 'unknown'),
                    active_sessions=len(status.get('agents', {})),
                    total_agents=status.get('configuration', {}).get('agent_count', 0),
                    active_agents=status.get('metrics', {}).get('active_agents', 0),
                    system_health=status.get('health_status', {}),
                    performance_metrics=status.get('metrics', {})
                )
            except Exception as e:
                self.logger.error(f"Failed to get system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Session management endpoints
        @self.app.post("/api/sessions", response_model=SessionResponse)
        async def create_session(request: SessionCreateRequest):
            """Create a new session."""
            try:
                session_manager = self._get_session_manager()

                session_id = await session_manager.create_session(
                    user_id=request.user_id,
                    project_path=request.project_path,
                    session_type=request.session_type,
                    initial_context=request.initial_context
                )

                session_data = await session_manager.get_session(session_id)
                if not session_data:
                    raise HTTPException(status_code=500, detail="Failed to retrieve created session")

                return SessionResponse(
                    session_id=session_id,
                    status=session_data['status'].value,
                    created_at=session_data['created_at'].isoformat(),
                    last_activity=session_data['statistics'].last_activity.isoformat()
                        if session_data['statistics'].last_activity else None,
                    user_id=session_data['metadata'].user_id,
                    project_path=session_data['metadata'].project_path
                )

            except Exception as e:
                self.logger.error(f"Failed to create session: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/sessions/{session_id}", response_model=SessionResponse)
        async def get_session(session_id: str):
            """Get session details."""
            try:
                session_manager = self._get_session_manager()
                session_data = await session_manager.get_session(session_id)

                if not session_data:
                    raise HTTPException(status_code=404, detail="Session not found")

                return SessionResponse(
                    session_id=session_id,
                    status=session_data['status'].value,
                    created_at=session_data['created_at'].isoformat(),
                    last_activity=session_data['statistics'].last_activity.isoformat()
                        if session_data['statistics'].last_activity else None,
                    user_id=session_data['metadata'].user_id,
                    project_path=session_data['metadata'].project_path
                )

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get session: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/sessions")
        async def list_sessions(user_id: Optional[str] = None):
            """List active sessions."""
            try:
                session_manager = self._get_session_manager()
                sessions = await session_manager.get_active_sessions(user_id=user_id)
                return {"sessions": sessions}

            except Exception as e:
                self.logger.error(f"Failed to list sessions: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Message/Chat endpoints
        @self.app.post("/api/sessions/{session_id}/messages")
        async def send_message(session_id: str, request: MessageRequest, background_tasks: BackgroundTasks):
            """Send a message to a session."""
            try:
                request_id = str(uuid.uuid4())

                # Track active request
                self.active_requests[request_id] = {
                    'session_id': session_id,
                    'started_at': datetime.utcnow(),
                    'status': 'processing'
                }

                # Process message through hive coordinator
                result = await self.hive_coordinator.process_request(
                    request=request.content,
                    context=request.context,
                    user_id=None,  # Could be extracted from authentication
                    request_id=request_id
                )

                # Broadcast to WebSocket connections
                await self.websocket_manager.broadcast_to_session(
                    {
                        "type": "message_response",
                        "data": {
                            "request_id": request_id,
                            "response": result,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    },
                    session_id
                )

                # Cleanup active request
                if request_id in self.active_requests:
                    del self.active_requests[request_id]

                return result

            except Exception as e:
                self.logger.error(f"Failed to send message: {e}")
                if request_id in self.active_requests:
                    del self.active_requests[request_id]
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/sessions/{session_id}/messages")
        async def get_session_messages(session_id: str, limit: int = 50, offset: int = 0):
            """Get message history for a session."""
            try:
                session_manager = self._get_session_manager()
                session_data = await session_manager.get_session(session_id)

                if not session_data:
                    raise HTTPException(status_code=404, detail="Session not found")

                messages = session_data.get('conversation_history', [])

                # Apply pagination
                paginated_messages = messages[offset:offset + limit]

                return {
                    "messages": paginated_messages,
                    "total": len(messages),
                    "offset": offset,
                    "limit": limit
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get session messages: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Streaming endpoint for real-time responses
        @self.app.get("/api/sessions/{session_id}/stream")
        async def stream_session_updates(session_id: str, request: Request):
            """Server-sent events endpoint for real-time session updates."""

            async def event_generator():
                stream_id = str(uuid.uuid4())
                queue = asyncio.Queue()
                self.streaming_queues[stream_id] = queue

                try:
                    while True:
                        # Check if client is still connected
                        if await request.is_disconnected():
                            break

                        try:
                            # Wait for updates with timeout
                            update = await asyncio.wait_for(queue.get(), timeout=30.0)
                            yield {
                                "event": "update",
                                "data": json.dumps(update)
                            }
                        except asyncio.TimeoutError:
                            # Send heartbeat
                            yield {
                                "event": "heartbeat",
                                "data": json.dumps({"timestamp": datetime.utcnow().isoformat()})
                            }

                except asyncio.CancelledError:
                    pass
                finally:
                    # Cleanup
                    if stream_id in self.streaming_queues:
                        del self.streaming_queues[stream_id]

            return EventSourceResponse(event_generator())

        # Agent management endpoints
        @self.app.get("/api/agents")
        async def list_agents():
            """List all agents and their status."""
            try:
                status = await self.hive_coordinator.get_system_status()
                agents = status.get('agents', {})

                agent_list = []
                for agent_id, agent_status in agents.items():
                    agent_list.append(AgentStatusResponse(
                        agent_id=agent_id,
                        agent_type=agent_status.get('type', 'unknown'),
                        status=agent_status.get('status', 'unknown'),
                        current_task=agent_status.get('current_task'),
                        performance_metrics=agent_status.get('metrics', {})
                    ))

                return {"agents": agent_list}

            except Exception as e:
                self.logger.error(f"Failed to list agents: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/agents/{agent_id}")
        async def get_agent_details(agent_id: str):
            """Get detailed information about a specific agent."""
            try:
                status = await self.hive_coordinator.get_system_status()
                agents = status.get('agents', {})

                if agent_id not in agents:
                    raise HTTPException(status_code=404, detail="Agent not found")

                return agents[agent_id]

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get agent details: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Metrics and monitoring endpoints
        @self.app.get("/api/metrics")
        async def get_system_metrics():
            """Get comprehensive system metrics."""
            try:
                status = await self.hive_coordinator.get_system_status()
                metrics = status.get('metrics', {})

                # Add WebSocket connection metrics
                ws_stats = self.websocket_manager.get_connection_stats()
                metrics['websocket_connections'] = ws_stats

                # Add active request metrics
                metrics['active_requests'] = len(self.active_requests)

                return {"metrics": metrics, "timestamp": datetime.utcnow().isoformat()}

            except Exception as e:
                self.logger.error(f"Failed to get system metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # WebSocket endpoint
        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            """WebSocket endpoint for real-time communication."""
            connection_id = str(uuid.uuid4())

            try:
                await self.websocket_manager.connect(
                    websocket,
                    connection_id,
                    session_id=session_id
                )

                # Send initial connection acknowledgment
                await self.websocket_manager.send_personal_message(
                    {
                        "type": "connection_established",
                        "data": {
                            "connection_id": connection_id,
                            "session_id": session_id,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    },
                    connection_id
                )

                while True:
                    # Receive messages from client
                    data = await websocket.receive_json()

                    # Handle different message types
                    message_type = data.get("type", "unknown")

                    if message_type == "ping":
                        await self.websocket_manager.send_personal_message(
                            {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                            connection_id
                        )

                    elif message_type == "chat_message":
                        # Process chat message through the system
                        content = data.get("content", "")
                        context = data.get("context", {})

                        try:
                            result = await self.hive_coordinator.process_request(
                                request=content,
                                context=context,
                                user_id=None,
                                request_id=str(uuid.uuid4())
                            )

                            await self.websocket_manager.send_personal_message(
                                {
                                    "type": "chat_response",
                                    "data": result
                                },
                                connection_id
                            )

                        except Exception as e:
                            await self.websocket_manager.send_personal_message(
                                {
                                    "type": "error",
                                    "data": {"message": str(e)}
                                },
                                connection_id
                            )

                    elif message_type == "request_status":
                        # Send current system status
                        status = await self.hive_coordinator.get_system_status()
                        await self.websocket_manager.send_personal_message(
                            {
                                "type": "system_status",
                                "data": status
                            },
                            connection_id
                        )

            except WebSocketDisconnect:
                pass
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_manager.disconnect(connection_id)

        # Main dashboard page (if templates are available)
        if self.templates:
            @self.app.get("/", response_class=HTMLResponse)
            async def dashboard(request: Request):
                """Main dashboard page."""
                return self.templates.TemplateResponse("dashboard.html", {"request": request})

            @self.app.get("/sessions/{session_id}", response_class=HTMLResponse)
            async def session_page(request: Request, session_id: str):
                """Individual session page."""
                return self.templates.TemplateResponse(
                    "session.html",
                    {"request": request, "session_id": session_id}
                )

    def _get_session_manager(self) -> Optional[SessionManager]:
        """Get the session manager from the hive coordinator."""
        # This would need to be implemented based on the actual hive coordinator structure
        # For now, return None as a placeholder
        return None

    async def start_background_tasks(self):
        """Start background tasks for the web application."""
        # System status broadcasting task
        async def broadcast_system_status():
            while True:
                try:
                    await asyncio.sleep(5)  # Broadcast every 5 seconds

                    if self.websocket_manager.active_connections:
                        status = await self.hive_coordinator.get_system_status()

                        await self.websocket_manager.broadcast_system_message({
                            "type": "system_status_update",
                            "data": status
                        })

                except Exception as e:
                    self.logger.error(f"System status broadcast error: {e}")

        # Active request cleanup task
        async def cleanup_stale_requests():
            while True:
                try:
                    await asyncio.sleep(60)  # Check every minute

                    now = datetime.utcnow()
                    stale_requests = []

                    for request_id, request_data in self.active_requests.items():
                        age = now - request_data['started_at']
                        if age > timedelta(minutes=10):  # 10 minute timeout
                            stale_requests.append(request_id)

                    for request_id in stale_requests:
                        del self.active_requests[request_id]

                    if stale_requests:
                        self.logger.info(f"Cleaned up {len(stale_requests)} stale requests")

                except Exception as e:
                    self.logger.error(f"Request cleanup error: {e}")

        # Start background tasks
        tasks = [
            asyncio.create_task(broadcast_system_status()),
            asyncio.create_task(cleanup_stale_requests())
        ]

        for task in tasks:
            self.background_tasks.add(task)

        self.logger.info("Background tasks started")

    async def stop_background_tasks(self):
        """Stop all background tasks."""
        for task in self.background_tasks:
            task.cancel()

        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()

        self.logger.info("Background tasks stopped")

    async def startup(self):
        """Application startup tasks."""
        self.logger.info("Starting Hive Web Application")
        await self.start_background_tasks()

    async def shutdown(self):
        """Application shutdown tasks."""
        self.logger.info("Shutting down Hive Web Application")
        await self.stop_background_tasks()

    def run(self):
        """Run the web application server."""
        import uvicorn

        # Add startup and shutdown events
        self.app.add_event_handler("startup", self.startup)
        self.app.add_event_handler("shutdown", self.shutdown)

        self.logger.info(f"Starting web server on {self.host}:{self.port}")

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="debug" if self.debug else "info",
            reload=self.debug,
            access_log=True
        )


# Factory function for easy integration
async def create_hive_webapp(
    hive_coordinator: HiveCoordinator,
    host: str = "localhost",
    port: int = 8080,
    debug: bool = False,
    **kwargs
) -> HiveWebApp:
    """Create and configure a HiveWebApp instance."""
    webapp = HiveWebApp(
        hive_coordinator=hive_coordinator,
        host=host,
        port=port,
        debug=debug,
        **kwargs
    )

    return webapp


# CLI integration
if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Run Hive Web Interface")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--config", help="Path to hive configuration file")

    args = parser.parse_args()

    async def main():
        # This would need to be integrated with the actual hive coordinator
        # For now, this is a placeholder
        print(f"Starting Hive Web Interface on {args.host}:{args.port}")
        print("Note: This requires integration with a running HiveCoordinator instance")

    asyncio.run(main())
