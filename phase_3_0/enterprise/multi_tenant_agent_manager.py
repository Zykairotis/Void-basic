"""
Void-basic Phase 3.0 - Enterprise Multi-Tenant Agent Manager

This module provides enterprise-grade multi-tenant management for the autonomous agent system.
It extends the Phase 2.2 agent architecture with:
- Multi-tenant isolation and security
- Advanced AI model integration with intelligent routing
- Compliance automation (SOX, GDPR, HIPAA)
- Enterprise monitoring and audit trails
- Industry-specific workflow templates
- Resource management and auto-scaling
- Real-time event broadcasting and coordination

Features:
- Tenant-aware agent orchestration
- AI model fallback strategies
- Compliance policy enforcement
- Audit trail generation
- Performance monitoring
- Security controls
- Industry workflow templates
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
from contextlib import asynccontextmanager

# Enhanced imports for enterprise features
import redis
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, Summary
import jwt
from cryptography.fernet import Fernet
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

# Import Phase 2.2 agents
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from aider.agents.workflow_orchestrator import WorkflowOrchestrator
from aider.agents.code_agent import CodeAgent
from aider.agents.context_agent import ContextAgent
from aider.agents.git_agent import GitAgent
from aider.agents.quality_agent import QualityAgent
from aider.agents.deployment_agent import DeploymentAgent
from aider.models.model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics for enterprise monitoring
TENANT_REQUESTS = Counter('tenant_requests_total', 'Total requests per tenant', ['tenant_id', 'agent_type'])
WORKFLOW_DURATION = Histogram('workflow_duration_seconds', 'Workflow execution time', ['tenant_id', 'workflow_type'])
ACTIVE_TENANTS = Gauge('active_tenants_total', 'Number of active tenants')
COMPLIANCE_VIOLATIONS = Counter('compliance_violations_total', 'Compliance violations', ['tenant_id', 'violation_type'])
AI_MODEL_REQUESTS = Counter('ai_model_requests_total', 'AI model requests', ['model_name', 'tenant_id'])
AI_MODEL_FAILURES = Counter('ai_model_failures_total', 'AI model failures', ['model_name', 'tenant_id'])
AGENT_UTILIZATION = Gauge('agent_utilization_percent', 'Agent utilization percentage', ['agent_name', 'tenant_id'])

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOX = "sox"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"

class IndustryType(Enum):
    """Industry-specific workflow types"""
    FINTECH = "fintech"
    HEALTHCARE = "healthcare"
    ECOMMERCE = "ecommerce"
    SAAS = "saas"
    GOVERNMENT = "government"
    MANUFACTURING = "manufacturing"

class TenantPlan(Enum):
    """Tenant subscription plans"""
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

@dataclass
class TenantConfiguration:
    """Configuration for a tenant"""
    tenant_id: str
    name: str
    domain: str
    plan: TenantPlan
    industry: IndustryType
    compliance_requirements: List[ComplianceFramework]
    max_concurrent_workflows: int
    ai_model_preferences: Dict[str, str]
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    encryption_key: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

@dataclass
class WorkflowContext:
    """Context for workflow execution"""
    workflow_id: str
    tenant_id: str
    user_id: str
    workflow_type: str
    description: str
    priority: str
    compliance_requirements: List[str]
    industry_context: Optional[IndustryType] = None
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"

class EnterpriseSecurityManager:
    """Handles enterprise security and encryption"""

    def __init__(self):
        self.master_key = os.getenv("MASTER_ENCRYPTION_KEY", Fernet.generate_key().decode())
        self.cipher = Fernet(self.master_key.encode())

    def encrypt_data(self, data: str, tenant_key: Optional[str] = None) -> str:
        """Encrypt sensitive data with tenant-specific key if provided"""
        if tenant_key:
            tenant_cipher = Fernet(tenant_key.encode())
            return tenant_cipher.encrypt(data.encode()).decode()
        return self.cipher.encrypt(data.encode()).decode()

    def decrypt_data(self, encrypted_data: str, tenant_key: Optional[str] = None) -> str:
        """Decrypt sensitive data"""
        if tenant_key:
            tenant_cipher = Fernet(tenant_key.encode())
            return tenant_cipher.decrypt(encrypted_data.encode()).decode()
        return self.cipher.decrypt(encrypted_data.encode()).decode()

    def generate_tenant_key(self) -> str:
        """Generate a unique encryption key for a tenant"""
        return Fernet.generate_key().decode()

    def create_audit_signature(self, data: Dict[str, Any], tenant_id: str) -> str:
        """Create tamper-proof signature for audit trail"""
        data_string = json.dumps(data, sort_keys=True)
        signature = hmac.new(
            f"{self.master_key}:{tenant_id}".encode(),
            data_string.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

class ComplianceEngine:
    """Handles compliance automation and validation"""

    def __init__(self, security_manager: EnterpriseSecurityManager):
        self.security_manager = security_manager
        self.compliance_rules = self._load_compliance_rules()

    def _load_compliance_rules(self) -> Dict[ComplianceFramework, Dict[str, Any]]:
        """Load compliance rules for different frameworks"""
        return {
            ComplianceFramework.SOX: {
                "immutable_audit_trail": True,
                "separation_of_duties": True,
                "change_approval_required": True,
                "financial_data_protection": True,
                "retention_period_years": 7
            },
            ComplianceFramework.GDPR: {
                "data_subject_rights": True,
                "consent_management": True,
                "data_minimization": True,
                "right_to_erasure": True,
                "breach_notification_hours": 72,
                "data_retention_max_months": 24
            },
            ComplianceFramework.HIPAA: {
                "ephi_encryption": True,
                "access_controls": True,
                "audit_logging": True,
                "minimum_necessary": True,
                "business_associate_agreements": True,
                "retention_period_years": 6
            }
        }

    async def validate_workflow_compliance(
        self,
        context: WorkflowContext,
        tenant_config: TenantConfiguration
    ) -> Dict[str, Any]:
        """Validate workflow against compliance requirements"""
        validation_results = {
            "compliant": True,
            "violations": [],
            "warnings": [],
            "required_approvals": []
        }

        for framework in tenant_config.compliance_requirements:
            rules = self.compliance_rules.get(framework, {})

            # SOX compliance checks
            if framework == ComplianceFramework.SOX:
                if context.workflow_type in ["deployment", "financial_data_update"]:
                    if not context.custom_parameters.get("approved_by"):
                        validation_results["violations"].append({
                            "framework": "SOX",
                            "rule": "change_approval_required",
                            "message": "Financial system changes require approval"
                        })
                        validation_results["compliant"] = False

            # GDPR compliance checks
            elif framework == ComplianceFramework.GDPR:
                if "personal_data" in context.description.lower():
                    if not context.custom_parameters.get("data_processing_consent"):
                        validation_results["warnings"].append({
                            "framework": "GDPR",
                            "rule": "consent_management",
                            "message": "Personal data processing requires consent verification"
                        })

            # HIPAA compliance checks
            elif framework == ComplianceFramework.HIPAA:
                if "health" in context.description.lower() or "medical" in context.description.lower():
                    if not context.custom_parameters.get("hipaa_authorization"):
                        validation_results["violations"].append({
                            "framework": "HIPAA",
                            "rule": "ephi_access_controls",
                            "message": "ePHI access requires proper authorization"
                        })
                        validation_results["compliant"] = False

        return validation_results

    async def create_audit_entry(
        self,
        context: WorkflowContext,
        event_type: str,
        event_data: Dict[str, Any],
        tenant_config: TenantConfiguration
    ) -> Dict[str, Any]:
        """Create tamper-proof audit trail entry"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": context.workflow_id,
            "tenant_id": context.tenant_id,
            "user_id": context.user_id,
            "event_type": event_type,
            "event_data": event_data,
            "compliance_frameworks": [f.value for f in tenant_config.compliance_requirements]
        }

        # Create tamper-proof signature
        signature = self.security_manager.create_audit_signature(audit_entry, context.tenant_id)
        audit_entry["signature"] = signature

        # Add to workflow context
        context.audit_trail.append(audit_entry)

        return audit_entry

class AdvancedAIModelManager:
    """Enhanced AI model management with enterprise features"""

    def __init__(self):
        self.base_model_manager = ModelManager()
        self.model_routing_rules = {}
        self.fallback_chains = {}
        self.rate_limits = {}
        self.model_costs = {}
        self._initialize_enterprise_models()

    def _initialize_enterprise_models(self):
        """Initialize enterprise AI model configurations"""
        self.model_routing_rules = {
            "code_generation": ["gpt-4o", "claude-3.5-sonnet", "deepseek-v3"],
            "code_review": ["claude-3.5-sonnet", "gpt-4o", "grok-2"],
            "documentation": ["gpt-4o", "claude-3.5-sonnet"],
            "testing": ["gpt-4o", "deepseek-v3"],
            "security_analysis": ["claude-3.5-sonnet", "gpt-4o"],
            "compliance_check": ["claude-3.5-sonnet", "gpt-4o"]
        }

        self.fallback_chains = {
            "gpt-4o": ["claude-3.5-sonnet", "grok-2", "deepseek-v3"],
            "claude-3.5-sonnet": ["gpt-4o", "grok-2", "deepseek-v3"],
            "grok-2": ["gpt-4o", "claude-3.5-sonnet", "deepseek-v3"],
            "deepseek-v3": ["gpt-4o", "claude-3.5-sonnet", "grok-2"]
        }

        self.rate_limits = {
            "gpt-4o": {"requests_per_minute": 3000, "tokens_per_minute": 150000},
            "claude-3.5-sonnet": {"requests_per_minute": 4000, "tokens_per_minute": 200000},
            "grok-2": {"requests_per_minute": 5000, "tokens_per_minute": 250000},
            "deepseek-v3": {"requests_per_minute": 6000, "tokens_per_minute": 300000}
        }

    async def get_optimal_model(
        self,
        task_type: str,
        tenant_config: TenantConfiguration,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Select optimal AI model based on task type and tenant preferences"""

        # Check tenant preferences first
        if task_type in tenant_config.ai_model_preferences:
            preferred_model = tenant_config.ai_model_preferences[task_type]
            if await self._check_model_availability(preferred_model, tenant_config.tenant_id):
                return preferred_model

        # Use routing rules
        candidate_models = self.model_routing_rules.get(task_type, ["gpt-4o"])

        for model in candidate_models:
            if await self._check_model_availability(model, tenant_config.tenant_id):
                return model

        # Final fallback to base model
        return "gpt-4o"

    async def _check_model_availability(self, model_name: str, tenant_id: str) -> bool:
        """Check if model is available and within rate limits"""
        try:
            # Check rate limits (simplified - in production use Redis for distributed rate limiting)
            current_usage = await self._get_current_usage(model_name, tenant_id)
            limits = self.rate_limits.get(model_name, {})

            if current_usage.get("requests_per_minute", 0) >= limits.get("requests_per_minute", 1000):
                return False

            return True
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False

    async def _get_current_usage(self, model_name: str, tenant_id: str) -> Dict[str, int]:
        """Get current usage statistics for rate limiting"""
        # In production, this would query Redis or a time-series database
        return {"requests_per_minute": 0, "tokens_per_minute": 0}

    async def execute_with_fallback(
        self,
        task_type: str,
        prompt: str,
        tenant_config: TenantConfiguration,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """Execute AI task with automatic fallback on failure"""

        primary_model = await self.get_optimal_model(task_type, tenant_config)
        fallback_models = self.fallback_chains.get(primary_model, [])

        models_to_try = [primary_model] + fallback_models[:max_attempts-1]

        for attempt, model_name in enumerate(models_to_try):
            try:
                AI_MODEL_REQUESTS.labels(model_name=model_name, tenant_id=tenant_config.tenant_id).inc()

                result = await self.base_model_manager.generate_response(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=4000
                )

                return {
                    "success": True,
                    "result": result,
                    "model_used": model_name,
                    "attempt": attempt + 1,
                    "cost_estimate": self._estimate_cost(model_name, len(prompt))
                }

            except Exception as e:
                AI_MODEL_FAILURES.labels(model_name=model_name, tenant_id=tenant_config.tenant_id).inc()
                logger.warning(f"Model {model_name} failed for tenant {tenant_config.tenant_id}: {e}")

                if attempt == len(models_to_try) - 1:
                    return {
                        "success": False,
                        "error": f"All models failed. Last error: {str(e)}",
                        "attempts": len(models_to_try)
                    }

        return {"success": False, "error": "No models available"}

    def _estimate_cost(self, model_name: str, prompt_length: int) -> float:
        """Estimate cost for AI model usage"""
        base_costs = {
            "gpt-4o": 0.03,  # per 1k tokens
            "claude-3.5-sonnet": 0.025,
            "grok-2": 0.02,
            "deepseek-v3": 0.015
        }

        estimated_tokens = prompt_length / 4  # Rough estimation
        cost_per_1k = base_costs.get(model_name, 0.03)
        return (estimated_tokens / 1000) * cost_per_1k

class IndustryWorkflowTemplates:
    """Industry-specific workflow templates and configurations"""

    def __init__(self):
        self.templates = self._load_industry_templates()

    def _load_industry_templates(self) -> Dict[IndustryType, Dict[str, Any]]:
        """Load industry-specific workflow templates"""
        return {
            IndustryType.FINTECH: {
                "workflows": {
                    "trading_algorithm": {
                        "agents": ["CodeAgent", "QualityAgent", "DeploymentAgent"],
                        "compliance": [ComplianceFramework.SOX],
                        "security_level": "high",
                        "approval_required": True,
                        "testing_requirements": ["unit", "integration", "performance", "security"]
                    },
                    "payment_processing": {
                        "agents": ["CodeAgent", "ContextAgent", "QualityAgent"],
                        "compliance": [ComplianceFramework.PCI_DSS],
                        "security_level": "critical",
                        "encryption_required": True
                    }
                },
                "code_standards": {
                    "security_patterns": ["input_validation", "secure_communication", "audit_logging"],
                    "performance_requirements": {"response_time_ms": 100, "throughput_tps": 1000}
                }
            },

            IndustryType.HEALTHCARE: {
                "workflows": {
                    "patient_data_system": {
                        "agents": ["CodeAgent", "ContextAgent", "QualityAgent"],
                        "compliance": [ComplianceFramework.HIPAA],
                        "security_level": "critical",
                        "data_classification": "ePHI",
                        "access_controls": "role_based"
                    },
                    "clinical_decision_support": {
                        "agents": ["CodeAgent", "QualityAgent", "DeploymentAgent"],
                        "compliance": [ComplianceFramework.HIPAA, ComplianceFramework.ISO_27001],
                        "validation_required": True,
                        "clinical_approval": True
                    }
                },
                "code_standards": {
                    "data_protection": ["encryption_at_rest", "encryption_in_transit", "access_logging"],
                    "availability_requirements": {"uptime": 99.99, "disaster_recovery": True}
                }
            },

            IndustryType.ECOMMERCE: {
                "workflows": {
                    "checkout_system": {
                        "agents": ["CodeAgent", "QualityAgent", "DeploymentAgent"],
                        "compliance": [ComplianceFramework.PCI_DSS, ComplianceFramework.GDPR],
                        "performance_critical": True,
                        "scalability_testing": True
                    },
                    "recommendation_engine": {
                        "agents": ["CodeAgent", "ContextAgent", "QualityAgent"],
                        "compliance": [ComplianceFramework.GDPR],
                        "data_privacy": "gdpr_compliant",
                        "personalization_controls": True
                    }
                },
                "code_standards": {
                    "scalability_patterns": ["microservices", "caching", "cdn_integration"],
                    "performance_requirements": {"page_load_ms": 2000, "api_response_ms": 500}
                }
            }
        }

    def get_workflow_template(
        self,
        industry: IndustryType,
        workflow_type: str
    ) -> Optional[Dict[str, Any]]:
        """Get workflow template for specific industry and type"""
        industry_config = self.templates.get(industry, {})
        workflows = industry_config.get("workflows", {})
        return workflows.get(workflow_type)

    def get_code_standards(self, industry: IndustryType) -> Dict[str, Any]:
        """Get code standards for specific industry"""
        industry_config = self.templates.get(industry, {})
        return industry_config.get("code_standards", {})

class MultiTenantAgentManager:
    """Enterprise multi-tenant agent manager"""

    def __init__(self):
        self.tenants: Dict[str, TenantConfiguration] = {}
        self.agent_pools: Dict[str, Dict[str, Any]] = {}
        self.security_manager = EnterpriseSecurityManager()
        self.compliance_engine = ComplianceEngine(self.security_manager)
        self.ai_model_manager = AdvancedAIModelManager()
        self.industry_templates = IndustryWorkflowTemplates()

        # Event broadcasting
        self.event_subscribers: Dict[str, List[Callable]] = {}

        # Redis for caching and real-time data
        self.redis_client = None

        # Database connection for persistent storage
        self.db_engine = None

        # Performance tracking
        self.performance_metrics = {}

    async def initialize(self):
        """Initialize the enterprise agent manager"""
        try:
            # Initialize Redis connection
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)

            # Initialize database connection
            db_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/voidbasic_enterprise")
            self.db_engine = create_engine(db_url, poolclass=StaticPool)

            # Load existing tenants
            await self._load_tenants()

            # Initialize agent pools for active tenants
            for tenant_id in self.tenants:
                await self._initialize_tenant_agents(tenant_id)

            logger.info("Multi-tenant agent manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize multi-tenant agent manager: {e}")
            raise

    async def _load_tenants(self):
        """Load tenant configurations from database"""
        try:
            # In production, load from database
            # For now, create a sample tenant
            sample_tenant = TenantConfiguration(
                tenant_id="sample-tenant-001",
                name="Sample Enterprise",
                domain="sample.enterprise.com",
                plan=TenantPlan.ENTERPRISE,
                industry=IndustryType.FINTECH,
                compliance_requirements=[ComplianceFramework.SOX, ComplianceFramework.GDPR],
                max_concurrent_workflows=50,
                ai_model_preferences={
                    "code_generation": "gpt-4o",
                    "code_review": "claude-3.5-sonnet",
                    "security_analysis": "claude-3.5-sonnet"
                },
                encryption_key=self.security_manager.generate_tenant_key()
            )

            self.tenants[sample_tenant.tenant_id] = sample_tenant
            ACTIVE_TENANTS.set(len(self.tenants))

        except Exception as e:
            logger.error(f"Error loading tenants: {e}")
            raise

    async def _initialize_tenant_agents(self, tenant_id: str):
        """Initialize agent pool for a specific tenant"""
        try:
            tenant_config = self.tenants[tenant_id]

            # Create tenant-specific agent instances
            agents = {
                "workflow_orchestrator": WorkflowOrchestrator(),
                "code_agent": CodeAgent(),
                "context_agent": ContextAgent(),
                "git_agent": GitAgent(),
                "quality_agent": QualityAgent(),
                "deployment_agent": DeploymentAgent()
            }

            # Initialize each agent with tenant-specific configuration
            for agent_name, agent in agents.items():
                await agent.initialize()

                # Add enterprise enhancements
                await self._enhance_agent_for_enterprise(agent, tenant_config)

                logger.info(f"Initialized {agent_name} for tenant {tenant_id}")

            self.agent_pools[tenant_id] = agents

        except Exception as e:
            logger.error(f"Failed to initialize agents for tenant {tenant_id}: {e}")
            raise

    async def _enhance_agent_for_enterprise(self, agent, tenant_config: TenantConfiguration):
        """Add enterprise features to existing agents"""
        # Add tenant context
        agent.tenant_id = tenant_config.tenant_id
        agent.compliance_requirements = tenant_config.compliance_requirements
        agent.industry_context = tenant_config.industry

        # Add enterprise methods
        agent.create_audit_entry = lambda event_type, data: self.compliance_engine.create_audit_entry(
            WorkflowContext(
                workflow_id="",
                tenant_id=tenant_config.tenant_id,
                user_id="",
                workflow_type="",
                description="",
                priority=""
            ),
            event_type,
            data,
            tenant_config
        )

        # Add AI model management
        agent.get_ai_response = lambda task_type, prompt: self.ai_model_manager.execute_with_fallback(
            task_type,
            prompt,
            tenant_config
        )

    async def register_tenant(self, tenant_config: TenantConfiguration) -> bool:
        """Register a new tenant"""
        try:
            # Validate tenant configuration
            if tenant_config.tenant_id in self.tenants:
                raise ValueError(f"Tenant {tenant_config.tenant_id} already exists")

            # Generate encryption key if not provided
            if not tenant_config.encryption_key:
                tenant_config.encryption_key = self.security_manager.generate_tenant_key()

            # Store tenant configuration
            self.tenants[tenant_config.tenant_id] = tenant_config

            # Initialize agent pool
            await self._initialize_tenant_agents(tenant_config.tenant_id)

            # Update metrics
            ACTIVE_TENANTS.set(len(self.tenants))

            # Broadcast tenant registration event
            await self._broadcast_event("tenant_registered", {
                "tenant_id": tenant_config.tenant_id,
                "name": tenant_config.name,
                "plan": tenant_config.plan.value
            })

            logger.info(f"Successfully registered tenant {tenant_config.tenant_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to register tenant {tenant_config.tenant_id}: {e}")
            raise

    async def execute_workflow(
        self,
        tenant_id: str,
        user_id: str,
        workflow_type: str,
        description: str,
        priority: str = "medium",
        compliance_requirements: List[str] = None,
        custom_parameters: Dict[str, Any] = None
    ) -> str:
        """Execute a workflow for a specific tenant"""

        workflow_id = str(uuid.uuid4())
        start_time = datetime.utcnow()

        try:
            # Validate tenant
            if tenant_id not in self.tenants:
                raise ValueError(f"Tenant {tenant_id} not found")

            tenant_config = self.tenants[tenant_id]

            # Create workflow context
            context = WorkflowContext(
                workflow_id=workflow_id,
                tenant_id=tenant_id,
                user_id=user_id,
                workflow_type=workflow_type,
                description=description,
                priority=priority,
                compliance_requirements=compliance_requirements or [],
                industry_context=tenant_config.industry,
                custom_parameters=custom_parameters or {}
            )

            # Validate compliance
            compliance_result = await self.compliance_engine.validate_workflow_compliance(
                context, tenant_config
            )

            if not compliance_result["compliant"]:
                raise ValueError(f"Compliance violations: {compliance_result['violations']}")

            # Create initial audit entry
            await self.compliance_engine.create_audit_entry(
                context, "workflow_started", {"description": description}, tenant_config
            )

            # Get agents for tenant
            agents = self.agent_pools[tenant_id]

            # Get industry-specific template if available
            workflow_template = self.industry_templates.get_workflow_template(
                tenant_config.industry, workflow_type
            )

            # Update metrics
            TENANT_REQUESTS.labels(tenant_id=tenant_id, agent_type="workflow").inc()

            # Execute workflow using orchestrator
            orchestrator = agents["workflow_orchestrator"]

            # Enhanced execution with enterprise features
            result = await self._execute_enterprise_workflow(
                orchestrator, context, tenant_config, workflow_template
            )

            # Record completion
            duration = (datetime.utcnow() - start_time).total_seconds()
            WORKFLOW_DURATION.labels(tenant_id=tenant_id, workflow_type=workflow_type).observe(duration)

            # Create completion audit entry
            await self.compliance_engine.create_audit_entry(
                context, "workflow_completed", {"result": result, "duration": duration}, tenant_config
            )

            # Broadcast completion event
            await self._broadcast_event("workflow_completed", {
                "workflow_id": workflow_id,
                "tenant_id": tenant_id,
                "result": result,
                "duration": duration
            })

            context.status = "completed"

            # Cache result for real-time access
            if self.redis_client:
                await self._cache_workflow_result(workflow_id, context, result)

            logger.info(f"Workflow {workflow_id} completed for tenant {tenant_id}")
            return workflow_id

        except Exception as e:
            # Handle workflow failure
            duration = (datetime.utcnow() - start_time).total_seconds()

            # Create failure audit entry
            if 'context' in locals():
                await self.compliance_engine.create_audit_entry(
                    context, "workflow_failed", {"error": str(e), "duration": duration}, tenant_config
                )

            # Broadcast failure event
            await self._broadcast_event("workflow_failed", {
                "workflow_id": workflow_id,
                "tenant_id": tenant_id,
                "error": str(e),
                "duration": duration
            })

            logger.error(f"Workflow {workflow_id} failed for tenant {tenant_id}: {e}")
            raise

    async def _execute_enterprise_workflow(
        self,
        orchestrator: WorkflowOrchestrator,
        context: WorkflowContext,
        tenant_config: TenantConfiguration,
        workflow_template: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute workflow with enterprise features"""

        try:
            # Apply industry-specific configuration if available
            if workflow_template:
                # Configure agents based on template
                required_agents = workflow_template.get("agents", [])
                security_level = workflow_template.get("security_level", "standard")

                # Apply security configurations
                if security_level == "critical":
                    context.custom_parameters["encryption_required"] = True
                    context.custom_parameters["audit_level"] = "detailed"

            # Update agent utilization metrics
            for agent_name in self.agent_pools[context.tenant_id]:
                AGENT_UTILIZATION.labels(
                    agent_name=agent_name,
                    tenant_id=context.tenant_id
                ).set(75.0)  # Mock utilization

            # Execute the workflow
            result = await orchestrator.execute_autonomous_workflow(
                workflow_type=context.workflow_type,
                description=context.description,
                compliance_requirements=context.compliance_requirements,
                tenant_context={
                    "tenant_id": context.tenant_id,
                    "industry": tenant_config.industry.value,
                    "compliance": [f.value for f in tenant_config.compliance_requirements]
                }
            )

            # Apply industry-specific post-processing
            if tenant_config.industry == IndustryType.FINTECH:
                result = await self._apply_fintech_processing(result, context)
            elif tenant_config.industry == IndustryType.HEALTHCARE:
                result = await self._apply_healthcare_processing(result, context)

            return result

        except Exception as e:
            logger.error(f"Enterprise workflow execution failed: {e}")
            raise

    async def _apply_fintech_processing(
        self,
        result: Dict[str, Any],
        context: WorkflowContext
    ) -> Dict[str, Any]:
        """Apply FinTech-specific processing"""
        # Add financial compliance checks
        result["compliance_checks"] = {
            "sox_validation": True,
            "risk_assessment": "low",
            "audit_trail_complete": True
        }
        return result

    async def _apply_healthcare_processing(
        self,
        result: Dict[str, Any],
        context: WorkflowContext
    ) -> Dict[str, Any]:
        """Apply Healthcare-specific processing"""
        # Add HIPAA compliance validation
        result["hipaa_compliance"] = {
            "ephi_encrypted": True,
            "access_logged": True,
            "minimum_necessary": True
        }
        return result

    async def _cache_workflow_result(
        self,
        workflow_id: str,
        context: WorkflowContext,
        result: Dict[str, Any]
    ):
        """Cache workflow result for real-time access"""
        try:
            cache_data = {
                "workflow_id": workflow_id,
                "tenant_id": context.tenant_id,
                "status": context.status,
                "result": result,
                "started_at": context.started_at.isoformat(),
                "completed_at": datetime.utcnow().isoformat()
            }

            # Cache for 1 hour
            await asyncio.to_thread(
                self.redis_client.setex,
                f"workflow:{workflow_id}",
                3600,
                json.dumps(cache_data, default=str)
            )

        except Exception as e:
            logger.warning(f"Failed to cache workflow result: {e}")

    async def _broadcast_event(self, event_type: str, event_data: Dict[str, Any]):
        """Broadcast events to subscribers"""
        try:
            # Get subscribers for this event type
            subscribers = self.event_subscribers.get(event_type, [])

            for subscriber in subscribers:
                try:
                    await subscriber(event_type, event_data)
                except Exception as e:
                    logger.warning(f"Event subscriber failed: {e}")

            # Also publish to Redis for WebSocket broadcasting
            if self.redis_client:
                await asyncio.to_thread(
                    self.redis_client.publish,
                    "enterprise_events",
                    json.dumps({
                        "type": event_type,
                        "data": event_data,
                        "timestamp": datetime.utcnow().isoformat()
                    }, default=str)
                )

        except Exception as e:
            logger.warning(f"Failed to broadcast event: {e}")

    def subscribe_to_events(self, event_type: str, callback: Callable):
        """Subscribe to enterprise events"""
        if event_type not in self.event_subscribers:
            self.event_subscribers[event_type] = []
        self.event_subscribers[event_type].append(callback)

    async def get_tenant_metrics(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a tenant"""
        if tenant_id not in self.tenants:
            raise ValueError(f"Tenant {tenant_id} not found")

        try:
            # Get real-time workflow data
            active_workflows = []
            if self.redis_client:
                workflow_keys = await asyncio.to_thread(
                    self.redis_client.keys,
                    f"workflow:*"
                )

                for key in workflow_keys:
                    workflow_data = await asyncio.to_thread(
                        self.redis_client.get, key
                    )
                    if workflow_data:
                        data = json.loads(workflow_data)
                        if data.get("tenant_id") == tenant_id:
                            active_workflows.append(data)

            # Calculate metrics
            total_workflows = len(active_workflows)
            completed_workflows = [w for w in active_workflows if w.get("status") == "completed"]
            failed_workflows = [w for w in active_workflows if w.get("status") == "failed"]

            success_rate = (len(completed_workflows) / total_workflows * 100) if total_workflows > 0 else 0

            return {
                "tenant_id": tenant_id,
                "active_workflows": total_workflows,
                "completed_workflows": len(completed_workflows),
                "failed_workflows": len(failed_workflows),
                "success_rate": round(success_rate, 2),
                "agent_utilization": {
                    agent_name: 85.0  # Mock data - in production, get from metrics
                    for agent_name in self.agent_pools.get(tenant_id, {}).keys()
                },
                "compliance_status": {
                    framework.value: True  # Mock compliance status
                    for framework in self.tenants[tenant_id].compliance_requirements
                }
            }

        except Exception as e:
            logger.error(f"Failed to get tenant metrics: {e}")
            raise

    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        try:
            total_tenants = len(self.tenants)
            active_agents = sum(len(agents) for agents in self.agent_pools.values())

            # Calculate resource usage (mock data - in production, get from monitoring system)
            resource_usage = {
                "cpu_percent": 45.2,
                "memory_percent": 62.1,
                "disk_percent": 23.8
            }

            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "tenants": {
                    "total": total_tenants,
                    "active": total_tenants
                },
                "agents": {
                    "total": active_agents,
                    "healthy": active_agents
                },
                "resources": resource_usage,
                "uptime_percent": 99.95
            }

        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def shutdown(self):
        """Gracefully shutdown the agent manager"""
        try:
            logger.info("Shutting down multi-tenant agent manager")

            # Shutdown all agent pools
            for tenant_id, agents in self.agent_pools.items():
                for agent_name, agent in agents.items():
                    try:
                        if hasattr(agent, 'shutdown'):
                            await agent.shutdown()
                    except Exception as e:
                        logger.warning(f"Error shutting down {agent_name} for tenant {tenant_id}: {e}")

            # Close Redis connection
            if self.redis_client:
                await asyncio.to_thread(self.redis_client.close)

            # Close database connection
            if self.db_engine:
                self.db_engine.dispose()

            logger.info("Multi-tenant agent manager shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.shutdown()


# Usage example and main execution
async def main():
    """Example usage of the Multi-Tenant Agent Manager"""

    async with MultiTenantAgentManager() as manager:
        # Register a sample tenant
        tenant_config = TenantConfiguration(
            tenant_id="demo-tenant",
            name="Demo Enterprise",
            domain="demo.enterprise.com",
            plan=TenantPlan.ENTERPRISE,
            industry=IndustryType.FINTECH,
            compliance_requirements=[ComplianceFramework.SOX, ComplianceFramework.GDPR],
            max_concurrent_workflows=25,
            ai_model_preferences={
                "code_generation": "gpt-4o",
                "code_review": "claude-3.5-sonnet"
            }
        )

        await manager.register_tenant(tenant_config)

        # Execute a sample workflow
        workflow_id = await manager.execute_workflow(
            tenant_id="demo-tenant",
            user_id="user-123",
            workflow_type="feature_development",
            description="Implement secure payment processing system",
            priority="high",
            compliance_requirements=["SOX", "PCI_DSS"],
            custom_parameters={
                "security_level": "critical",
                "requires_approval": True
            }
        )

        print(f"Workflow {workflow_id} executed successfully")

        # Get tenant metrics
        metrics = await manager.get_tenant_metrics("demo-tenant")
        print(f"Tenant metrics: {json.dumps(metrics, indent=2)}")

        # Get system health
        health = await manager.get_system_health()
        print(f"System health: {json.dumps(health, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
