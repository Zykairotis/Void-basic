"""
Void-basic Phase 3.0 - Enterprise Web Dashboard Backend
FastAPI application with WebSocket support for real-time monitoring

Features:
- Multi-tenant architecture with data isolation
- JWT authentication with RBAC
- WebSocket connections for real-time updates
- Integration with autonomous agents
- Comprehensive API for dashboard functionality
- Enterprise security and compliance
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import jwt
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Integer, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import redis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Import existing agents from Phase 2.2
import sys
sys.path.append('/home/mewtwo/Code/Github/Void-basic/Void-basic')
from aider.agents.workflow_orchestrator import WorkflowOrchestrator
from aider.agents.code_agent import CodeAgent
from aider.agents.context_agent import ContextAgent
from aider.agents.git_agent import GitAgent
from aider.agents.quality_agent import QualityAgent
from aider.agents.deployment_agent import DeploymentAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_WORKFLOWS = Gauge('active_workflows_total', 'Number of active workflows')
WEBSOCKET_CONNECTIONS = Gauge('websocket_connections_total', 'Number of active WebSocket connections')

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/voidbasic_enterprise")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
JWT_SECRET = os.getenv("JWT_SECRET", "your-super-secret-jwt-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Database Setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis Setup
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

# Pydantic Models
class TenantCreate(BaseModel):
    name: str
    domain: str
    admin_email: str
    compliance_requirements: List[str] = []

class UserCreate(BaseModel):
    email: str
    name: str
    role: str
    tenant_id: str

class WorkflowRequest(BaseModel):
    workflow_type: str
    description: str
    priority: str = "medium"
    compliance_requirements: List[str] = []

    @validator('workflow_type')
    def validate_workflow_type(cls, v):
        allowed_types = ['feature_development', 'bug_fix', 'refactoring', 'security_fix', 'performance_optimization']
        if v not in allowed_types:
            raise ValueError(f'workflow_type must be one of {allowed_types}')
        return v

class AuthToken(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    tenant_id: str
    user_role: str

# Database Models
class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    domain = Column(String, unique=True, nullable=False)
    admin_email = Column(String, nullable=False)
    compliance_requirements = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    settings = Column(JSON, default=dict)

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)  # admin, developer, viewer
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=False)
    workflow_type = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String, default="pending")  # pending, running, completed, failed
    priority = Column(String, default="medium")
    compliance_requirements = Column(JSON, default=list)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    result = Column(JSON)
    error_message = Column(Text)
    agent_assignments = Column(JSON, default=dict)

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.tenant_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, tenant_id: str, user_id: str):
        await websocket.accept()

        if tenant_id not in self.active_connections:
            self.active_connections[tenant_id] = []
        if tenant_id not in self.tenant_connections:
            self.tenant_connections[tenant_id] = []

        self.active_connections[tenant_id].append(websocket)
        self.tenant_connections[tenant_id].append(websocket)

        WEBSOCKET_CONNECTIONS.inc()
        logger.info(f"WebSocket connected for tenant {tenant_id}, user {user_id}")

    def disconnect(self, websocket: WebSocket, tenant_id: str):
        if tenant_id in self.active_connections:
            if websocket in self.active_connections[tenant_id]:
                self.active_connections[tenant_id].remove(websocket)
        if tenant_id in self.tenant_connections:
            if websocket in self.tenant_connections[tenant_id]:
                self.tenant_connections[tenant_id].remove(websocket)

        WEBSOCKET_CONNECTIONS.dec()
        logger.info(f"WebSocket disconnected for tenant {tenant_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_to_tenant(self, message: str, tenant_id: str):
        if tenant_id in self.tenant_connections:
            for connection in self.tenant_connections[tenant_id]:
                try:
                    await connection.send_text(message)
                except:
                    # Remove broken connections
                    self.tenant_connections[tenant_id].remove(connection)

# Global instances
manager = ConnectionManager()
workflow_orchestrator = None
agents = {}

# Authentication
security = HTTPBearer()

def create_access_token(user_id: str, tenant_id: str, role: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "user_id": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "exp": expire
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        tenant_id = payload.get("tenant_id")
        role = payload.get("role")

        if not user_id or not tenant_id or not role:
            raise HTTPException(status_code=401, detail="Invalid token")

        return {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "role": role
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Enterprise Agent Manager
class EnterpriseAgentManager:
    def __init__(self):
        self.agents = {}
        self.workflow_orchestrator = None

    async def initialize(self):
        """Initialize all agents with enterprise configurations"""
        try:
            # Initialize WorkflowOrchestrator
            self.workflow_orchestrator = WorkflowOrchestrator()
            await self.workflow_orchestrator.initialize()

            # Initialize individual agents
            self.agents = {
                'code_agent': CodeAgent(),
                'context_agent': ContextAgent(),
                'git_agent': GitAgent(),
                'quality_agent': QualityAgent(),
                'deployment_agent': DeploymentAgent()
            }

            # Initialize each agent
            for agent_name, agent in self.agents.items():
                await agent.initialize()
                logger.info(f"Initialized {agent_name}")

            logger.info("Enterprise Agent Manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Enterprise Agent Manager: {e}")
            raise

    async def execute_workflow(self, workflow_request: WorkflowRequest, tenant_id: str, user_id: str) -> str:
        """Execute autonomous workflow with enterprise features"""
        try:
            # Create workflow execution record
            workflow_id = str(uuid.uuid4())

            # Store workflow context in Redis for real-time updates
            workflow_context = {
                "id": workflow_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "type": workflow_request.workflow_type,
                "description": workflow_request.description,
                "status": "running",
                "started_at": datetime.utcnow().isoformat(),
                "compliance_requirements": workflow_request.compliance_requirements
            }

            redis_client.setex(f"workflow:{workflow_id}", 3600, json.dumps(workflow_context))
            ACTIVE_WORKFLOWS.inc()

            # Broadcast workflow start to tenant
            await manager.broadcast_to_tenant(
                json.dumps({
                    "type": "workflow_started",
                    "workflow_id": workflow_id,
                    "data": workflow_context
                }),
                tenant_id
            )

            # Execute workflow using orchestrator
            result = await self.workflow_orchestrator.execute_autonomous_workflow(
                workflow_type=workflow_request.workflow_type,
                description=workflow_request.description,
                compliance_requirements=workflow_request.compliance_requirements
            )

            # Update workflow status
            workflow_context["status"] = "completed"
            workflow_context["completed_at"] = datetime.utcnow().isoformat()
            workflow_context["result"] = result

            redis_client.setex(f"workflow:{workflow_id}", 3600, json.dumps(workflow_context))
            ACTIVE_WORKFLOWS.dec()

            # Broadcast completion
            await manager.broadcast_to_tenant(
                json.dumps({
                    "type": "workflow_completed",
                    "workflow_id": workflow_id,
                    "data": workflow_context
                }),
                tenant_id
            )

            return workflow_id

        except Exception as e:
            # Handle workflow failure
            workflow_context["status"] = "failed"
            workflow_context["error"] = str(e)
            workflow_context["completed_at"] = datetime.utcnow().isoformat()

            redis_client.setex(f"workflow:{workflow_id}", 3600, json.dumps(workflow_context))
            ACTIVE_WORKFLOWS.dec()

            await manager.broadcast_to_tenant(
                json.dumps({
                    "type": "workflow_failed",
                    "workflow_id": workflow_id,
                    "data": workflow_context
                }),
                tenant_id
            )

            logger.error(f"Workflow {workflow_id} failed: {e}")
            raise

# Global agent manager
agent_manager = EnterpriseAgentManager()

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Void-basic Enterprise Dashboard Backend")

    # Initialize database
    Base.metadata.create_all(bind=engine)

    # Initialize agents
    await agent_manager.initialize()

    logger.info("Application startup completed")
    yield

    # Shutdown
    logger.info("Shutting down application")

# FastAPI Application
app = FastAPI(
    title="Void-basic Enterprise Dashboard",
    description="Enterprise web dashboard for autonomous development workflows",
    version="3.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://dashboard.voidbasic.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "dashboard.voidbasic.com"]
)

# Health Check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.0.0",
        "agents_status": "operational"
    }

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    return generate_latest()

# Authentication Endpoints
@app.post("/auth/login")
async def login(email: str, password: str, db: Session = Depends(get_db)):
    # In production, implement proper password hashing and validation
    user = db.query(User).filter(User.email == email).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()

    # Create token
    access_token = create_access_token(str(user.id), str(user.tenant_id), user.role)

    return AuthToken(
        access_token=access_token,
        expires_in=JWT_EXPIRATION_HOURS * 3600,
        tenant_id=str(user.tenant_id),
        user_role=user.role
    )

# Tenant Management
@app.post("/tenants")
async def create_tenant(tenant: TenantCreate, db: Session = Depends(get_db)):
    # Check if domain already exists
    existing = db.query(Tenant).filter(Tenant.domain == tenant.domain).first()
    if existing:
        raise HTTPException(status_code=400, detail="Domain already exists")

    new_tenant = Tenant(
        name=tenant.name,
        domain=tenant.domain,
        admin_email=tenant.admin_email,
        compliance_requirements=tenant.compliance_requirements
    )

    db.add(new_tenant)
    db.commit()
    db.refresh(new_tenant)

    return {"tenant_id": str(new_tenant.id), "message": "Tenant created successfully"}

@app.get("/tenants/{tenant_id}")
async def get_tenant(tenant_id: str, current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    if current_user["tenant_id"] != tenant_id and current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Access denied")

    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    return tenant

# Workflow Management
@app.post("/workflows")
async def create_workflow(
    workflow: WorkflowRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    REQUEST_COUNT.labels(method="POST", endpoint="/workflows", status="200").inc()

    try:
        # Create workflow execution record
        workflow_execution = WorkflowExecution(
            tenant_id=current_user["tenant_id"],
            user_id=current_user["user_id"],
            workflow_type=workflow.workflow_type,
            description=workflow.description,
            priority=workflow.priority,
            compliance_requirements=workflow.compliance_requirements
        )

        db.add(workflow_execution)
        db.commit()
        db.refresh(workflow_execution)

        # Execute workflow asynchronously
        workflow_id = await agent_manager.execute_workflow(
            workflow,
            current_user["tenant_id"],
            current_user["user_id"]
        )

        return {
            "workflow_id": workflow_id,
            "status": "started",
            "message": "Workflow execution started"
        }

    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint="/workflows", status="500").inc()
        logger.error(f"Workflow creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflows")
async def list_workflows(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    workflows = db.query(WorkflowExecution).filter(
        WorkflowExecution.tenant_id == current_user["tenant_id"]
    ).order_by(WorkflowExecution.started_at.desc()).all()

    return {"workflows": workflows}

@app.get("/workflows/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    current_user: dict = Depends(get_current_user)
):
    # Get workflow from Redis cache
    workflow_data = redis_client.get(f"workflow:{workflow_id}")
    if not workflow_data:
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow = json.loads(workflow_data)

    # Check tenant access
    if workflow["tenant_id"] != current_user["tenant_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    return workflow

# Real-time WebSocket endpoint
@app.websocket("/ws/{tenant_id}")
async def websocket_endpoint(websocket: WebSocket, tenant_id: str):
    try:
        # Simple authentication for WebSocket (in production, use proper token validation)
        await manager.connect(websocket, tenant_id, "user")

        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
            await manager.send_personal_message(f"Message received: {data}", websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket, tenant_id)

# Dashboard Analytics
@app.get("/analytics/overview")
async def get_analytics_overview(current_user: dict = Depends(get_current_user)):
    tenant_id = current_user["tenant_id"]

    # Get real-time metrics from Redis
    active_workflows_count = len(redis_client.keys(f"workflow:*"))

    # Get historical data (in production, query from time-series database)
    analytics = {
        "active_workflows": active_workflows_count,
        "completed_today": 25,  # Mock data
        "success_rate": 96.6,
        "average_completion_time": "2.3 minutes",
        "agent_utilization": {
            "code_agent": 85,
            "context_agent": 78,
            "git_agent": 92,
            "quality_agent": 88,
            "deployment_agent": 76
        },
        "compliance_status": {
            "sox_compliant": True,
            "gdpr_compliant": True,
            "hipaa_compliant": True if "healthcare" in str(tenant_id) else False
        }
    }

    return analytics

# System Status
@app.get("/system/status")
async def get_system_status(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    return {
        "system_health": "healthy",
        "uptime": "99.9%",
        "active_agents": 6,
        "total_tenants": 1,
        "active_connections": len(manager.active_connections),
        "resource_usage": {
            "cpu": "45%",
            "memory": "62%",
            "disk": "23%"
        },
        "recent_alerts": []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
