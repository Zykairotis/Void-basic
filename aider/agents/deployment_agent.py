"""
Deployment Agent: Autonomous CI/CD and Deployment Pipeline Management

This agent implements comprehensive deployment capabilities for autonomous workflows,
including CI/CD pipeline automation, deployment strategies, infrastructure management,
and intelligent monitoring with rollback capabilities.

Key Capabilities:
- Automated CI/CD pipeline execution
- Multiple deployment strategies (canary, blue-green, rolling)
- Infrastructure provisioning and management
- Health monitoring and validation
- Automatic rollback on failure detection
- Environment management and configuration
- Security scanning and compliance checks
- Performance monitoring and optimization
- Integration with version control systems
- AI-powered deployment intelligence
"""

import asyncio
import json
import logging
import time
import uuid
import subprocess
import os
import yaml
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

import structlog

from .base_agent import BaseAgent, AgentMessage, MessagePriority, AgentState, AgentCapability
from ..task_management.task_queue import Task, TaskPriority, TaskState
from ..models.model_manager import get_model_manager, ModelRequest, TaskType, ComplexityLevel, Priority


class DeploymentStrategy(Enum):
    """Deployment strategies supported."""
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"


class DeploymentEnvironment(Enum):
    """Target deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    QA = "qa"
    UAT = "uat"
    PREVIEW = "preview"


class DeploymentStatus(Enum):
    """Status of deployment execution."""
    PENDING = "pending"
    BUILDING = "building"
    TESTING = "testing"
    DEPLOYING = "deploying"
    VALIDATING = "validating"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class HealthCheckType(Enum):
    """Types of health checks."""
    HTTP = "http"
    TCP = "tcp"
    COMMAND = "command"
    CUSTOM = "custom"


@dataclass
class HealthCheck:
    """Health check configuration."""
    id: str
    name: str
    type: HealthCheckType
    endpoint: Optional[str] = None
    port: Optional[int] = None
    command: Optional[str] = None
    expected_status: int = 200
    timeout: int = 30
    interval: int = 10
    retries: int = 3
    enabled: bool = True


@dataclass
class DeploymentTarget:
    """Deployment target configuration."""
    id: str
    name: str
    environment: DeploymentEnvironment
    platform: str  # kubernetes, docker, aws, azure, etc.
    configuration: Dict[str, Any] = field(default_factory=dict)
    health_checks: List[HealthCheck] = field(default_factory=list)
    capacity: Optional[int] = None  # number of instances
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)


@dataclass
class BuildConfiguration:
    """Build configuration for deployment."""
    id: str
    dockerfile_path: Optional[str] = None
    build_context: str = "."
    build_args: Dict[str, str] = field(default_factory=dict)
    registry: Optional[str] = None
    image_name: Optional[str] = None
    tag_strategy: str = "git_commit"  # git_commit, semantic, timestamp
    multi_stage: bool = False
    cache_enabled: bool = True


@dataclass
class SecurityScan:
    """Security scan configuration and results."""
    id: str
    scanner_type: str  # trivy, snyk, clair, etc.
    severity_threshold: str = "HIGH"
    fail_on_critical: bool = True
    scan_results: Optional[Dict[str, Any]] = None
    vulnerabilities_found: int = 0
    passed: bool = False
    timestamp: Optional[datetime] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for deployment validation."""
    response_time_p95: Optional[float] = None
    response_time_p99: Optional[float] = None
    throughput: Optional[float] = None
    error_rate: Optional[float] = None
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    disk_usage: Optional[float] = None
    network_io: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class DeploymentPlan:
    """Complete deployment plan."""
    id: str
    name: str
    description: str
    strategy: DeploymentStrategy
    source_commit: str
    target: DeploymentTarget
    build_config: BuildConfiguration
    health_checks: List[HealthCheck] = field(default_factory=list)
    security_scans: List[SecurityScan] = field(default_factory=list)
    rollback_strategy: Dict[str, Any] = field(default_factory=dict)
    notification_config: Dict[str, Any] = field(default_factory=dict)
    approval_required: bool = False
    estimated_duration: int = 1800  # 30 minutes default
    timeout: int = 3600  # 1 hour default
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DeploymentExecution:
    """Runtime execution state of a deployment."""
    id: str
    plan: DeploymentPlan
    status: DeploymentStatus
    current_stage: str
    progress: float = 0.0  # 0-100%
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    build_logs: List[str] = field(default_factory=list)
    deployment_logs: List[str] = field(default_factory=list)
    health_check_results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Optional[PerformanceMetrics] = None
    security_results: Dict[str, SecurityScan] = field(default_factory=dict)
    rollback_data: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DeploymentAgent(BaseAgent):
    """
    Deployment Agent for Autonomous CI/CD and Deployment Pipeline Management.

    Provides comprehensive deployment capabilities including build automation,
    multiple deployment strategies, health monitoring, and intelligent rollback.
    """

    def __init__(self, agent_id: str = "deployment_agent", config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, config)

        self.agent_type = "deployment"
        self.capabilities = {
            AgentCapability.DEPLOYMENT_AUTOMATION,
            AgentCapability.CI_CD_PIPELINE,
            AgentCapability.INFRASTRUCTURE_MANAGEMENT,
            AgentCapability.MONITORING,
            AgentCapability.ROLLBACK_MANAGEMENT,
            AgentCapability.AI_INTEGRATION
        }

        # Core components
        self.model_manager = None
        self.active_deployments: Dict[str, DeploymentExecution] = {}
        self.deployment_targets: Dict[str, DeploymentTarget] = {}
        self.deployment_history: List[DeploymentExecution] = []

        # Configuration
        self.config = config or {}
        self.project_root = Path(self.config.get('project_root', '.'))
        self.build_output_dir = Path(self.config.get('build_output_dir', './build'))
        self.deployment_configs_dir = Path(self.config.get('deployment_configs_dir', './deploy'))
        self.docker_enabled = self.config.get('docker_enabled', True)
        self.kubernetes_enabled = self.config.get('kubernetes_enabled', False)
        self.aws_enabled = self.config.get('aws_enabled', False)
        self.azure_enabled = self.config.get('azure_enabled', False)

        # CI/CD Configuration
        self.registry_url = self.config.get('registry_url', 'localhost:5000')
        self.default_strategy = DeploymentStrategy(self.config.get('default_strategy', 'rolling'))
        self.auto_rollback_enabled = self.config.get('auto_rollback_enabled', True)
        self.health_check_timeout = self.config.get('health_check_timeout', 300)  # 5 minutes

        # Security and compliance
        self.security_scanning_enabled = self.config.get('security_scanning_enabled', True)
        self.compliance_checks_enabled = self.config.get('compliance_checks_enabled', True)
        self.vulnerability_threshold = self.config.get('vulnerability_threshold', 'HIGH')

        # Monitoring and metrics
        self.monitoring_enabled = self.config.get('monitoring_enabled', True)
        self.metrics_retention_days = self.config.get('metrics_retention_days', 30)
        self.performance_thresholds = {
            'response_time_p95': self.config.get('max_response_time_p95', 1000),  # ms
            'error_rate': self.config.get('max_error_rate', 0.05),  # 5%
            'cpu_usage': self.config.get('max_cpu_usage', 80),  # 80%
            'memory_usage': self.config.get('max_memory_usage', 85)  # 85%
        }

        # Metrics tracking
        self.deployment_metrics = {
            'total_deployments': 0,
            'successful_deployments': 0,
            'failed_deployments': 0,
            'rollbacks': 0,
            'average_deployment_time': 0.0,
            'deployment_frequency': 0.0,
            'mttr': 0.0,  # Mean Time To Recovery
            'success_rate': 0.0
        }

        # Create necessary directories
        self.build_output_dir.mkdir(parents=True, exist_ok=True)
        self.deployment_configs_dir.mkdir(parents=True, exist_ok=True)

        self.logger = structlog.get_logger(__name__).bind(agent_id=agent_id)

    async def initialize(self):
        """Initialize the Deployment Agent."""
        await super().initialize()

        try:
            # Initialize model manager for AI capabilities
            self.model_manager = await get_model_manager()
            if self.model_manager:
                await self.model_manager.initialize()
                self.logger.info("AI models initialized for intelligent deployment")

            # Initialize deployment platforms
            await self._initialize_deployment_platforms()

            # Load deployment targets
            await self._load_deployment_targets()

            # Validate tools and dependencies
            await self._validate_deployment_tools()

            self.state = AgentState.READY
            self.logger.info("Deployment Agent initialized successfully")

        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Failed to initialize Deployment Agent: {e}")
            raise

    async def create_deployment_plan(
        self,
        name: str,
        description: str,
        target_environment: DeploymentEnvironment,
        source_commit: str,
        strategy: Optional[DeploymentStrategy] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> DeploymentPlan:
        """
        Create an intelligent deployment plan using AI analysis.

        Args:
            name: Deployment name
            description: Description of what's being deployed
            target_environment: Target environment
            source_commit: Git commit hash to deploy
            strategy: Deployment strategy (auto-selected if None)
            context: Additional context (project info, requirements, etc.)

        Returns:
            Generated deployment plan
        """
        plan_id = str(uuid.uuid4())

        self.logger.info(
            "Creating deployment plan",
            plan_id=plan_id,
            name=name,
            target_environment=target_environment.value,
            source_commit=source_commit
        )

        try:
            # Get target configuration
            target = await self._get_deployment_target(target_environment)

            # Auto-select strategy if not provided
            if not strategy:
                strategy = await self._select_deployment_strategy(
                    target_environment, context
                )

            # Create build configuration
            build_config = await self._create_build_configuration(
                source_commit, target, context
            )

            # Configure health checks
            health_checks = await self._configure_health_checks(target, context)

            # Configure security scans
            security_scans = await self._configure_security_scans(context)

            # Create rollback strategy
            rollback_strategy = await self._create_rollback_strategy(
                strategy, target, context
            )

            # Estimate deployment duration
            estimated_duration = await self._estimate_deployment_duration(
                strategy, target, context
            )

            plan = DeploymentPlan(
                id=plan_id,
                name=name,
                description=description,
                strategy=strategy,
                source_commit=source_commit,
                target=target,
                build_config=build_config,
                health_checks=health_checks,
                security_scans=security_scans,
                rollback_strategy=rollback_strategy,
                estimated_duration=estimated_duration,
                approval_required=target_environment == DeploymentEnvironment.PRODUCTION
            )

            self.logger.info(
                "Deployment plan created",
                plan_id=plan_id,
                strategy=strategy.value,
                estimated_duration=estimated_duration
            )

            return plan

        except Exception as e:
            self.logger.error(
                "Failed to create deployment plan",
                plan_id=plan_id,
                error=str(e)
            )
            raise

    async def execute_deployment(
        self,
        plan: DeploymentPlan,
        auto_approve: bool = False
    ) -> str:
        """
        Execute a deployment plan with comprehensive monitoring.

        Args:
            plan: Deployment plan to execute
            auto_approve: Skip approval for production deployments

        Returns:
            Deployment execution ID
        """
        execution_id = str(uuid.uuid4())

        self.logger.info(
            "Starting deployment execution",
            execution_id=execution_id,
            plan_id=plan.id,
            strategy=plan.strategy.value,
            environment=plan.target.environment.value
        )

        try:
            # Check for approval requirement
            if plan.approval_required and not auto_approve:
                await self._request_deployment_approval(plan)

            # Initialize execution tracking
            execution = DeploymentExecution(
                id=execution_id,
                plan=plan,
                status=DeploymentStatus.PENDING,
                current_stage="initialization"
            )

            self.active_deployments[execution_id] = execution

            # Start deployment pipeline asynchronously
            asyncio.create_task(self._execute_deployment_pipeline(execution))

            self.deployment_metrics['total_deployments'] += 1

            return execution_id

        except Exception as e:
            self.logger.error(
                "Failed to start deployment execution",
                execution_id=execution_id,
                error=str(e)
            )
            raise

    async def _execute_deployment_pipeline(self, execution: DeploymentExecution):
        """Execute the complete deployment pipeline."""

        execution_id = execution.id
        plan = execution.plan

        self.logger.info(
            "Executing deployment pipeline",
            execution_id=execution_id,
            plan_id=plan.id
        )

        start_time = time.time()

        try:
            execution.status = DeploymentStatus.BUILDING

            # Stage 1: Build
            await self._execute_build_stage(execution)
            execution.progress = 20.0

            # Stage 2: Security Scanning
            if self.security_scanning_enabled:
                await self._execute_security_scan_stage(execution)
                execution.progress = 40.0

            # Stage 3: Pre-deployment Testing
            await self._execute_testing_stage(execution)
            execution.progress = 60.0

            # Stage 4: Deployment
            execution.status = DeploymentStatus.DEPLOYING
            await self._execute_deployment_stage(execution)
            execution.progress = 80.0

            # Stage 5: Health Validation
            execution.status = DeploymentStatus.VALIDATING
            await self._execute_validation_stage(execution)
            execution.progress = 90.0

            # Stage 6: Monitoring
            execution.status = DeploymentStatus.MONITORING
            await self._execute_monitoring_stage(execution)
            execution.progress = 100.0

            # Mark as completed
            execution.status = DeploymentStatus.COMPLETED
            execution.completed_at = datetime.now()

            duration = time.time() - start_time
            self._update_deployment_metrics(execution, duration, True)

            self.logger.info(
                "Deployment completed successfully",
                execution_id=execution_id,
                duration=duration
            )

        except Exception as e:
            execution.status = DeploymentStatus.FAILED
            execution.errors.append(str(e))

            duration = time.time() - start_time
            self._update_deployment_metrics(execution, duration, False)

            self.logger.error(
                "Deployment failed",
                execution_id=execution_id,
                error=str(e),
                duration=duration
            )

            # Attempt automatic rollback if enabled
            if self.auto_rollback_enabled:
                await self._initiate_rollback(execution)

    async def _execute_build_stage(self, execution: DeploymentExecution):
        """Execute the build stage."""

        plan = execution.plan
        build_config = plan.build_config

        execution.current_stage = "building"
        self.logger.info(
            "Executing build stage",
            execution_id=execution.id,
            build_context=build_config.build_context
        )

        try:
            if self.docker_enabled and build_config.dockerfile_path:
                await self._build_docker_image(execution, build_config)
            else:
                await self._build_application(execution, build_config)

            execution.build_logs.append("Build completed successfully")

        except Exception as e:
            execution.build_logs.append(f"Build failed: {str(e)}")
            raise

    async def _build_docker_image(self, execution: DeploymentExecution, build_config: BuildConfiguration):
        """Build Docker image."""

        plan = execution.plan
        image_name = build_config.image_name or f"{plan.name}:{plan.source_commit[:8]}"

        # Prepare build command
        cmd = [
            'docker', 'build',
            '-t', image_name,
            '-f', build_config.dockerfile_path or 'Dockerfile',
            build_config.build_context
        ]

        # Add build args
        for key, value in build_config.build_args.items():
            cmd.extend(['--build-arg', f'{key}={value}'])

        if build_config.cache_enabled:
            cmd.append('--cache-from')
            cmd.append(image_name)

        self.logger.info(
            "Building Docker image",
            execution_id=execution.id,
            image_name=image_name,
            command=' '.join(cmd)
        )

        # Execute build
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.project_root
        )

        # Stream build output
        while True:
            line = await process.stdout.readline()
            if not line:
                break

            log_line = line.decode().strip()
            execution.build_logs.append(log_line)

            # Log important build steps
            if any(keyword in log_line.lower() for keyword in ['step', 'error', 'warning']):
                self.logger.debug("Build output", line=log_line)

        await process.wait()

        if process.returncode != 0:
            raise Exception(f"Docker build failed with exit code {process.returncode}")

        # Tag and push to registry if configured
        if build_config.registry:
            await self._push_to_registry(execution, image_name, build_config.registry)

    async def _build_application(self, execution: DeploymentExecution, build_config: BuildConfiguration):
        """Build application using detected build system."""

        # Detect build system and execute appropriate build command
        if (self.project_root / 'package.json').exists():
            await self._build_node_application(execution)
        elif (self.project_root / 'requirements.txt').exists() or (self.project_root / 'pyproject.toml').exists():
            await self._build_python_application(execution)
        elif (self.project_root / 'pom.xml').exists():
            await self._build_java_application(execution)
        else:
            execution.build_logs.append("No recognized build system found")

    async def _build_node_application(self, execution: DeploymentExecution):
        """Build Node.js application."""

        commands = [
            ['npm', 'ci'],
            ['npm', 'run', 'build']
        ]

        for cmd in commands:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.project_root
            )

            stdout, _ = await process.communicate()
            execution.build_logs.append(f"Command: {' '.join(cmd)}")
            execution.build_logs.append(stdout.decode())

            if process.returncode != 0:
                raise Exception(f"Build command failed: {' '.join(cmd)}")

    async def _build_python_application(self, execution: DeploymentExecution):
        """Build Python application."""

        commands = [
            ['pip', 'install', '-r', 'requirements.txt'],
            ['python', '-m', 'pytest', '--tb=short']  # Run tests as part of build
        ]

        for cmd in commands:
            # Skip if requirements file doesn't exist
            if 'requirements.txt' in cmd and not (self.project_root / 'requirements.txt').exists():
                continue

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.project_root
            )

            stdout, _ = await process.communicate()
            execution.build_logs.append(f"Command: {' '.join(cmd)}")
            execution.build_logs.append(stdout.decode())

            if process.returncode != 0 and 'pytest' not in cmd:  # Don't fail on test failures
                raise Exception(f"Build command failed: {' '.join(cmd)}")

    async def _build_java_application(self, execution: DeploymentExecution):
        """Build Java application."""

        commands = [
            ['mvn', 'clean', 'compile'],
            ['mvn', 'test'],
            ['mvn', 'package']
        ]

        for cmd in commands:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.project_root
            )

            stdout, _ = await process.communicate()
            execution.build_logs.append(f"Command: {' '.join(cmd)}")
            execution.build_logs.append(stdout.decode())

            if process.returncode != 0:
                raise Exception(f"Build command failed: {' '.join(cmd)}")

    async def _push_to_registry(self, execution: DeploymentExecution, image_name: str, registry: str):
        """Push image to container registry."""

        registry_image = f"{registry}/{image_name}"

        # Tag for registry
        tag_cmd = ['docker', 'tag', image_name, registry_image]
        process = await asyncio.create_subprocess_exec(*tag_cmd)
        await process.wait()

        if process.returncode != 0:
            raise Exception("Failed to tag image for registry")

        # Push to registry
        push_cmd = ['docker', 'push', registry_image]
        process = await asyncio.create_subprocess_exec(
            *push_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )

        stdout, _ = await process.communicate()
        execution.build_logs.append(f"Registry push: {stdout.decode()}")

        if process.returncode != 0:
            raise Exception("Failed to push image to registry")

    async def _execute_security_scan_stage(self, execution: DeploymentExecution):
        """Execute security scanning stage."""

        execution.current_stage = "security_scanning"
        plan = execution.plan

        for scan_config in plan.security_scans:
            scan_result = await self._run_security_scan(execution, scan_config)
            execution.security_results[scan_config.id] = scan_result

            if scan_config.fail_on_critical and not scan_result.passed:
                raise Exception(f"Security scan failed: {scan_config.scanner_type}")

    async def _run_security_scan(self, execution: DeploymentExecution, scan_config: SecurityScan) -> SecurityScan:
        """Run a security scan."""

        self.logger.info(
            "Running security scan",
            execution_id=execution.id,
            scanner_type=scan_config.scanner_type
        )

        # Simulate security scan (in real implementation, would use actual scanners)
        scan_config.timestamp = datetime.now()
        scan_config.vulnerabilities_found = 0  # Placeholder
        scan_config.passed = True  # Assume pass for demo
        scan_config.scan_results = {
            'total_vulnerabilities': 0,
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }

        return scan_config

    async def _execute_testing_stage(self, execution: DeploymentExecution):
        """Execute pre-deployment testing stage."""

        execution.current_stage = "testing"
        # Integration with quality agent would happen here
        # For now, simulate testing
        execution.deployment_logs.append("Pre-deployment tests completed")

    async def _execute_deployment_stage(self, execution: DeploymentExecution):
        """Execute the actual deployment."""

        execution.current_stage = "deploying"
        plan = execution.plan

        if plan.strategy == DeploymentStrategy.ROLLING:
            await self._execute_rolling_deployment(execution)
        elif plan.strategy == DeploymentStrategy.BLUE_GREEN:
            await self._execute_blue_green_deployment(execution)
        elif plan.strategy == DeploymentStrategy.CANARY:
            await self._execute_canary_deployment(execution)
        else:
            await self._execute_recreate_deployment(execution)

    async def _execute_rolling_deployment(self, execution: DeploymentExecution):
        """Execute rolling deployment strategy."""

        self.logger.info(
            "Executing rolling deployment",
            execution_id=execution.id
        )

        # Simulate rolling deployment
        instances = execution.plan.target.capacity or 3

        for i in range(instances):
            execution.deployment_logs.append(f"Updating instance {i+1}/{instances}")
            await asyncio.sleep(2)  # Simulate deployment time

            # Simulate health check
            await self._check_instance_health(execution, f"instance-{i+1}")

        execution.deployment_logs.append("Rolling deployment completed")

    async def _execute_blue_green_deployment(self, execution: DeploymentExecution):
        """Execute blue-green deployment strategy."""

        self.logger.info(
            "Executing blue-green deployment",
            execution_id=execution.id
        )

        # Simulate blue-green deployment
        execution.deployment_logs.append("Creating green environment")
        await asyncio.sleep(5)  # Simulate environment creation

        execution.deployment_logs.append("Deploying to green environment")
        await asyncio.sleep(3)  # Simulate deployment

        execution.deployment_logs.append("Switching traffic to green environment")
        await asyncio.sleep(2)  # Simulate traffic switch

        execution.deployment_logs.append("Blue-green deployment completed")

    async def _execute_canary_deployment(self, execution: DeploymentExecution):
        """Execute canary deployment strategy."""

        self.logger.info(
            "Executing canary deployment",
            execution_id=execution.id
        )

        # Simulate canary deployment phases
        phases = [10, 25, 50, 100]  # Traffic percentages

        for phase in phases:
            execution.deployment_logs.append(f"Deploying to {phase}% of traffic")
            await asyncio.sleep(3)  # Simulate deployment and monitoring time

            # Check metrics at each phase
            await self._monitor_canary_metrics(execution, phase)

        execution.deployment_logs.append("Canary deployment completed")

    async def _execute_recreate_deployment(self, execution: DeploymentExecution):
        """Execute recreate deployment strategy."""

        self.logger.info(
            "Executing recreate deployment",
            execution_id=execution.id
        )

        execution.deployment_logs.append("Stopping old version")
        await asyncio.sleep(2)  # Simulate stop time

        execution.deployment_logs.append("Starting new version")
        await asyncio.sleep(3)  # Simulate start time

        execution.deployment_logs.append("Recreate deployment completed")

    async def _execute_validation_stage(self, execution: DeploymentExecution):
        """Execute post-deployment validation."""

        execution.current_stage = "validating"
        plan = execution.plan

        # Run health checks
        for health_check in plan.health_checks:
            await self._run_health_check(execution, health_check)

        # Validate performance metrics
        await self._validate_performance_metrics(execution)

    async def _run_health_check(self, execution: DeploymentExecution, health_check: HealthCheck):
        """Run a health check."""

        self.logger.info(
            "Running health check",
            execution_id=execution.id,
            health_check_name=health_check.name
        )

        # Simulate health check
        if health_check.type == HealthCheckType.HTTP:
            # Simulate HTTP health check
            result = {
                'status': 'healthy',
                'response_time': 150,  # ms
                'status_code': 200
            }
        else:
            # Simulate other health check types
            result = {
                'status': 'healthy',
                'details': f'{health_check.type.value} check passed'
            }

        execution.health_check_results[health_check.id] = result

    async def _validate_performance_metrics(self, execution: DeploymentExecution):
        """Validate performance metrics after deployment."""

        # Simulate performance metrics collection
        metrics = PerformanceMetrics(
            response_time_p95=250.0,  # ms
            response_time_p99=450.0,  # ms
            throughput=1200.0,  # requests/sec
            error_rate=0.02,  # 2%
            cpu_usage=45.0,  # 45%
            memory_usage=60.0,  # 60%
            custom_metrics={'database_connections': 85.0}
        )

        execution.performance_metrics = metrics

        # Validate against thresholds
        issues = []
        if metrics.response_time_p95 > self.performance_thresholds['response_time_p95']:
            issues.append(f"Response time P95 ({metrics.response_time_p95}ms) exceeds threshold")

        if metrics.error_rate > self.performance_thresholds['error_rate']:
            issues.append(f"Error rate ({metrics.error_rate}) exceeds threshold")

        if metrics.cpu_usage > self.performance_thresholds['cpu_usage']:
            issues.append(f"CPU usage ({metrics.cpu_usage}%) exceeds threshold")

        if metrics.memory_usage > self.performance_thresholds['memory_usage']:
            issues.append(f"Memory usage ({metrics.memory_usage}%) exceeds threshold")

        if issues:
            execution.warnings.extend(issues)
            if len(issues) > 2:  # Too many performance issues
                raise Exception(f"Performance validation failed: {'; '.join(issues)}")

    async def _execute_monitoring_stage(self, execution: DeploymentExecution):
        """Execute post-deployment monitoring stage."""

        execution.current_stage = "monitoring"

        # Start monitoring for a brief period
        monitoring_duration = 60  # 1 minute for demo

        self.logger.info(
            "Starting post-deployment monitoring",
            execution_id=execution.id,
            duration=monitoring_duration
        )

        # Simulate monitoring checks
        for i in range(3):  # 3 monitoring cycles
            await asyncio.sleep(monitoring_duration / 3)

            # Check system health
            health_status = await self._check_system_health(execution)
            execution.deployment_logs.append(f"Monitoring cycle {i+1}: {health_status}")

            if health_status != "healthy":
                execution.warnings.append(f"Health issue detected in monitoring cycle {i+1}")

    async def _check_system_health(self, execution: DeploymentExecution) -> str:
        """Check overall system health."""

        # Simulate health check
        import random

        health_score = random.uniform(0.85, 1.0)  # 85-100% health

        if health_score > 0.95:
            return "healthy"
        elif health_score > 0.85:
            return "warning"
        else:
            return "unhealthy"

    async def _check_instance_health(self, execution: DeploymentExecution, instance_id: str):
        """Check health of a specific instance."""

        # Simulate instance health check
        await asyncio.sleep(1)  # Simulate check time

        execution.deployment_logs.append(f"Instance {instance_id}: healthy")

    async def _monitor_canary_metrics(self, execution: DeploymentExecution, traffic_percentage: int):
        """Monitor metrics during canary deployment."""

        # Simulate canary monitoring
        await asyncio.sleep(2)  # Simulate monitoring time

        # Check for anomalies (simulate)
        anomaly_detected = traffic_percentage == 25  # Simulate issue at 25%

        if anomaly_detected:
            execution.warnings.append(f"Anomaly detected at {traffic_percentage}% traffic")
            # In real implementation, this might trigger rollback

        execution.deployment_logs.append(f"Metrics at {traffic_percentage}%: OK")

    async def _initiate_rollback(self, execution: DeploymentExecution):
        """Initiate automatic rollback on deployment failure."""

        execution.status = DeploymentStatus.ROLLING_BACK

        self.logger.info(
            "Initiating automatic rollback",
            execution_id=execution.id
        )

        try:
            # Execute rollback strategy
            rollback_strategy = execution.plan.rollback_strategy

            if rollback_strategy.get('type') == 'previous_version':
                await self._rollback_to_previous_version(execution)
            elif rollback_strategy.get('type') == 'snapshot':
                await self._rollback_to_snapshot(execution)
            else:
                await self._rollback_to_previous_version(execution)  # Default

            execution.status = DeploymentStatus.ROLLED_BACK
            execution.deployment_logs.append("Automatic rollback completed")

            self.deployment_metrics['rollbacks'] += 1

        except Exception as e:
            execution.deployment_logs.append(f"Rollback failed: {str(e)}")
            self.logger.error(f"Rollback failed: {e}")

    async def _rollback_to_previous_version(self, execution: DeploymentExecution):
        """Rollback to previous version."""

        # Simulate rollback process
        execution.deployment_logs.append("Rolling back to previous version")
        await asyncio.sleep(3)  # Simulate rollback time

        # Restore previous configuration
        execution.deployment_logs.append("Restoring previous configuration")
        await asyncio.sleep(2)

        # Verify rollback success
        execution.deployment_logs.append("Verifying rollback success")
        await asyncio.sleep(1)

    async def _rollback_to_snapshot(self, execution: DeploymentExecution):
        """Rollback to infrastructure snapshot."""

        # Simulate snapshot rollback
        execution.deployment_logs.append("Rolling back to infrastructure snapshot")
        await asyncio.sleep(4)  # Simulate snapshot restore time

    async def _request_deployment_approval(self, plan: DeploymentPlan):
        """Request human approval for deployment."""

        self.logger.info(
            "Requesting deployment approval",
            plan_id=plan.id,
            environment=plan.target.environment.value
        )

        # In real implementation, this would:
        # 1. Send notification to approvers
        # 2. Wait for approval response
        # 3. Timeout after specified period

        # Simulate approval wait
        await asyncio.sleep(5)  # Simulate approval time

    async def _get_deployment_target(self, environment: DeploymentEnvironment) -> DeploymentTarget:
        """Get deployment target configuration for environment."""

        # Check if target exists
        target_id = f"{environment.value}_target"
        if target_id in self.deployment_targets:
            return self.deployment_targets[target_id]

        # Create default target
        target = DeploymentTarget(
            id=target_id,
            name=f"{environment.value.title()} Environment",
            environment=environment,
            platform="docker",
            capacity=3 if environment == DeploymentEnvironment.PRODUCTION else 1,
            health_checks=[
                HealthCheck(
                    id=f"{target_id}_http_check",
                    name="HTTP Health Check",
                    type=HealthCheckType.HTTP,
                    endpoint="/health",
                    expected_status=200
                )
            ]
        )

        self.deployment_targets[target_id] = target
        return target

    async def _select_deployment_strategy(
        self,
        environment: DeploymentEnvironment,
        context: Optional[Dict[str, Any]]
    ) -> DeploymentStrategy:
        """Select optimal deployment strategy using AI analysis."""

        if self.model_manager and context:
            strategy = await self._ai_select_deployment_strategy(environment, context)
            if strategy:
                return strategy

        # Fallback to rule-based selection
        if environment == DeploymentEnvironment.PRODUCTION:
            return DeploymentStrategy.BLUE_GREEN
        elif environment == DeploymentEnvironment.STAGING:
            return DeploymentStrategy.ROLLING
        else:
            return DeploymentStrategy.RECREATE

    async def _ai_select_deployment_strategy(
        self,
        environment: DeploymentEnvironment,
        context: Dict[str, Any]
    ) -> Optional[DeploymentStrategy]:
        """Use AI to select optimal deployment strategy."""

        strategy_prompt = f"""
        Select the optimal deployment strategy for this scenario:

        ENVIRONMENT: {environment.value}
        CONTEXT: {json.dumps(context, indent=2)}

        Available strategies:
        - rolling: Gradual replacement of instances
        - blue_green: Deploy to separate environment, switch traffic
        - canary: Gradual traffic shift with monitoring
        - recreate: Stop old, start new (downtime)

        Consider:
        1. Risk tolerance for the environment
        2. Downtime requirements
        3. Rollback speed requirements
        4. Resource availability
        5. Application characteristics

        Return only the strategy name (e.g., "blue_green").
        """

        try:
            model_request = ModelRequest(
                prompt=strategy_prompt,
                task_type=TaskType.ANALYSIS,
                complexity=ComplexityLevel.MEDIUM,
                priority=Priority.QUALITY,
                max_tokens=100,
                temperature=0.1
            )

            response = await self.model_manager.generate_response(model_request)

            if response and response.content:
                strategy_name = response.content.strip().lower()

                # Map to enum
                strategy_mapping = {
                    'rolling': DeploymentStrategy.ROLLING,
                    'blue_green': DeploymentStrategy.BLUE_GREEN,
                    'canary': DeploymentStrategy.CANARY,
                    'recreate': DeploymentStrategy.RECREATE
                }

                return strategy_mapping.get(strategy_name)

        except Exception as e:
            self.logger.warning(f"AI strategy selection failed: {e}")

        return None

    async def _create_build_configuration(
        self,
        source_commit: str,
        target: DeploymentTarget,
        context: Optional[Dict[str, Any]]
    ) -> BuildConfiguration:
        """Create build configuration."""

        config_id = str(uuid.uuid4())

        # Detect Dockerfile
        dockerfile_path = None
        if (self.project_root / 'Dockerfile').exists():
            dockerfile_path = 'Dockerfile'
        elif (self.project_root / 'docker' / 'Dockerfile').exists():
            dockerfile_path = 'docker/Dockerfile'

        return BuildConfiguration(
            id=config_id,
            dockerfile_path=dockerfile_path,
            build_context=".",
            registry=self.registry_url,
            image_name=f"app-{target.environment.value}",
            tag_strategy="git_commit",
            cache_enabled=True
        )

    async def _configure_health_checks(
        self,
        target: DeploymentTarget,
        context: Optional[Dict[str, Any]]
    ) -> List[HealthCheck]:
        """Configure health checks for deployment."""

        health_checks = target.health_checks.copy()

        # Add default health checks if none configured
        if not health_checks:
            health_checks.append(
                HealthCheck(
                    id=str(uuid.uuid4()),
                    name="Default HTTP Check",
                    type=HealthCheckType.HTTP,
                    endpoint="/health",
                    expected_status=200
                )
            )

        return health_checks

    async def _configure_security_scans(self, context: Optional[Dict[str, Any]]) -> List[SecurityScan]:
        """Configure security scans for deployment."""

        scans = []

        if self.security_scanning_enabled:
            scans.append(
                SecurityScan(
                    id=str(uuid.uuid4()),
                    scanner_type="trivy",
                    severity_threshold=self.vulnerability_threshold,
                    fail_on_critical=True
                )
            )

        return scans

    async def _create_rollback_strategy(
        self,
        strategy: DeploymentStrategy,
        target: DeploymentTarget,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create rollback strategy configuration."""

        if strategy == DeploymentStrategy.BLUE_GREEN:
            return {
                'type': 'environment_switch',
                'preserve_data': True,
                'timeout': 300
            }
        elif strategy == DeploymentStrategy.CANARY:
            return {
                'type': 'traffic_revert',
                'preserve_data': True,
                'timeout': 180
            }
        else:
            return {
                'type': 'previous_version',
                'preserve_data': True,
                'timeout': 600
            }

    async def _estimate_deployment_duration(
        self,
        strategy: DeploymentStrategy,
        target: DeploymentTarget,
        context: Optional[Dict[str, Any]]
    ) -> int:
        """Estimate deployment duration in seconds."""

        base_duration = 600  # 10 minutes base

        # Adjust based on strategy
        if strategy == DeploymentStrategy.BLUE_GREEN:
            base_duration += 300  # Extra time for environment setup
        elif strategy == DeploymentStrategy.CANARY:
            base_duration += 600  # Extra time for gradual rollout

        # Adjust based on target capacity
        if target.capacity:
            base_duration += target.capacity * 60  # 1 minute per instance

        return base_duration

    def _update_deployment_metrics(self, execution: DeploymentExecution, duration: float, success: bool):
        """Update deployment metrics."""

        if success:
            self.deployment_metrics['successful_deployments'] += 1
        else:
            self.deployment_metrics['failed_deployments'] += 1

        # Update average deployment time
        total_deployments = self.deployment_metrics['total_deployments']
        if total_deployments > 1:
            current_avg = self.deployment_metrics['average_deployment_time']
            self.deployment_metrics['average_deployment_time'] = (
                (current_avg * (total_deployments - 1) + duration) / total_deployments
            )
        else:
            self.deployment_metrics['average_deployment_time'] = duration

        # Update success rate
        successful = self.deployment_metrics['successful_deployments']
        self.deployment_metrics['success_rate'] = (successful / total_deployments) * 100

    async def _initialize_deployment_platforms(self):
        """Initialize deployment platform integrations."""

        self.logger.info("Initializing deployment platforms")

        # Initialize Docker if enabled
        if self.docker_enabled:
            try:
                process = await asyncio.create_subprocess_exec(
                    'docker', 'version',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()

                if process.returncode == 0:
                    self.logger.info("Docker platform initialized")
                else:
                    self.logger.warning("Docker not available")
                    self.docker_enabled = False
            except FileNotFoundError:
                self.logger.warning("Docker not installed")
                self.docker_enabled = False

        # Initialize Kubernetes if enabled
        if self.kubernetes_enabled:
            try:
                process = await asyncio.create_subprocess_exec(
                    'kubectl', 'version', '--client',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()

                if process.returncode == 0:
                    self.logger.info("Kubernetes platform initialized")
                else:
                    self.logger.warning("Kubernetes not available")
                    self.kubernetes_enabled = False
            except FileNotFoundError:
                self.logger.warning("kubectl not installed")
                self.kubernetes_enabled = False

    async def _load_deployment_targets(self):
        """Load deployment target configurations."""

        # Load from configuration files if they exist
        config_file = self.deployment_configs_dir / 'targets.yaml'

        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)

                for target_data in config_data.get('targets', []):
                    target = DeploymentTarget(**target_data)
                    self.deployment_targets[target.id] = target

        self.logger.info(f"Loaded {len(self.deployment_targets)} deployment targets")

    async def _validate_deployment_tools(self):
        """Validate required deployment tools are available."""

        required_tools = ['git']

        if self.docker_enabled:
            required_tools.append('docker')

        if self.kubernetes_enabled:
            required_tools.append('kubectl')

        for tool in required_tools:
            try:
                process = await asyncio.create_subprocess_exec(
                    tool, '--version',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()

                if process.returncode != 0:
                    self.logger.warning(f"Tool not available: {tool}")
            except FileNotFoundError:
                self.logger.warning(f"Tool not installed: {tool}")

    async def process_autonomous_task(self, task_request: Dict[str, Any]) -> Dict[str, Any]:
        """Process autonomous task for workflow integration."""

        task_description = task_request.get('description', '')
        context = task_request.get('context', {})

        if 'deploy' in task_description.lower():
            # Create and execute deployment
            plan = await self.create_deployment_plan(
                name=context.get('name', 'Autonomous Deployment'),
                description=task_description,
                target_environment=DeploymentEnvironment.STAGING,  # Default to staging
                source_commit=context.get('commit', 'HEAD'),
                context=context
            )

            execution_id = await self.execute_deployment(plan, auto_approve=True)

            return {
                'status': 'completed',
                'result': {
                    'execution_id': execution_id,
                    'plan_id': plan.id,
                    'strategy': plan.strategy.value,
                    'estimated_duration': plan.estimated_duration
                }
            }

        else:
            return {
                'status': 'completed',
                'result': {
                    'message': 'Deployment task processed',
                    'metrics': self.deployment_metrics
                }
            }

    def get_deployment_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment execution status."""

        execution = self.active_deployments.get(execution_id)
        if not execution:
            return None

        return {
            'id': execution.id,
            'status': execution.status.value,
            'current_stage': execution.current_stage,
            'progress': execution.progress,
            'started_at': execution.started_at.isoformat(),
            'completed_at': execution.completed_at.isoformat() if execution.completed_at else None,
            'errors': execution.errors,
            'warnings': execution.warnings
        }

    async def cancel_deployment(self, execution_id: str) -> bool:
        """Cancel a running deployment."""

        execution = self.active_deployments.get(execution_id)
        if not execution:
            return False

        execution.status = DeploymentStatus.CANCELLED
        self.logger.info("Deployment cancelled", execution_id=execution_id)
        return True

    def get_metrics(self) -> Dict[str, Any]:
        """Get deployment agent metrics."""
        return self.deployment_metrics.copy()

    def list_active_deployments(self) -> List[Dict[str, Any]]:
        """List all active deployments."""

        return [
            {
                'id': execution.id,
                'plan_name': execution.plan.name,
                'status': execution.status.value,
                'environment': execution.plan.target.environment.value,
                'progress': execution.progress,
                'started_at': execution.started_at.isoformat()
            }
            for execution in self.active_deployments.values()
        ]

    def get_capabilities(self) -> Set[AgentCapability]:
        """Get agent capabilities."""
        return self.capabilities

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the deployment agent."""
        return {
            'status': 'healthy',
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'state': self.state.value if hasattr(self.state, 'value') else str(self.state),
            'active_deployments': len(self.active_deployments),
            'deployment_targets': len(self.deployment_targets),
            'model_manager_available': self.model_manager is not None,
            'docker_enabled': self.docker_enabled,
            'kubernetes_enabled': self.kubernetes_enabled,
            'metrics': self.deployment_metrics
        }

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages from other agents."""
        try:
            if message.message_type == 'deployment_plan_request':
                name = message.payload.get('name', 'Deployment')
                description = message.payload.get('description', '')
                target_environment = DeploymentEnvironment(message.payload.get('target_environment', 'staging'))
                source_commit = message.payload.get('source_commit', 'HEAD')
                context = message.payload.get('context', {})

                plan = await self.create_deployment_plan(
                    name=name,
                    description=description,
                    target_environment=target_environment,
                    source_commit=source_commit,
                    context=context
                )

                return AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type='deployment_plan_response',
                    payload={'plan_id': plan.id, 'strategy': plan.strategy.value},
                    correlation_id=message.correlation_id
                )

            elif message.message_type == 'deployment_execution_request':
                plan_data = message.payload.get('plan')
                auto_approve = message.payload.get('auto_approve', False)

                # Reconstruct plan from payload (simplified for demo)
                execution_id = await self.execute_deployment(plan_data, auto_approve)

                return AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type='deployment_execution_response',
                    payload={'execution_id': execution_id},
                    correlation_id=message.correlation_id
                )

            else:
                self.logger.warning(
                    "Unknown message type received",
                    message_type=message.message_type,
                    sender=message.sender_id
                )
                return None

        except Exception as e:
            self.logger.error(
                "Error processing message",
                error=str(e),
                message_type=message.message_type,
                sender=message.sender_id
            )
            return None
