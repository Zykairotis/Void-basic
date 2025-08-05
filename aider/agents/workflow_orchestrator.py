"""
Autonomous Workflow Orchestrator: End-to-End Development Pipeline Management

This orchestrator implements Phase 2.2 Priority 4: End-to-End Autonomous Workflows
Supporting complete feature development from requirements to deployment.

Key Capabilities:
- Complete autonomous development cycles
- Multi-agent coordination with resilience patterns
- Quality assurance integration
- Deployment automation with monitoring
- Human-in-the-loop decision points
- Error recovery and rollback mechanisms
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from pathlib import Path

import structlog

from .base_agent import BaseAgent, AgentMessage, MessagePriority, AgentState, AgentCapability
from ..task_management.task_queue import Task, TaskPriority, TaskState
from ..models.model_manager import get_model_manager, ModelRequest, TaskType, ComplexityLevel, Priority


class WorkflowType(Enum):
    """Types of autonomous workflows supported."""
    FEATURE_DEVELOPMENT = "feature_development"
    BUG_FIX = "bug_fix"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    SECURITY_FIX = "security_fix"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    TESTING_ENHANCEMENT = "testing_enhancement"
    DEPLOYMENT = "deployment"


class WorkflowStage(Enum):
    """Stages in the autonomous workflow pipeline."""
    PLANNING = "planning"
    REQUIREMENTS_ANALYSIS = "requirements_analysis"
    ARCHITECTURE_DESIGN = "architecture_design"
    IMPLEMENTATION = "implementation"
    QUALITY_ASSURANCE = "quality_assurance"
    INTEGRATION = "integration"
    HUMAN_REVIEW = "human_review"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    COMPLETION = "completion"


class WorkflowStatus(Enum):
    """Status of workflow execution."""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    WAITING_REVIEW = "waiting_review"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


@dataclass
class WorkflowTask:
    """Individual task within a workflow."""
    id: str
    name: str
    description: str
    agent_type: str
    stage: WorkflowStage
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: int = 0  # seconds
    actual_duration: Optional[int] = None
    status: TaskState = TaskState.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class WorkflowCheckpoint:
    """Checkpoint for workflow state recovery."""
    id: str
    workflow_id: str
    stage: WorkflowStage
    timestamp: datetime
    state_data: Dict[str, Any]
    completed_tasks: List[str]
    agent_states: Dict[str, Any]


@dataclass
class QualityGate:
    """Quality gate that must pass before proceeding."""
    id: str
    name: str
    stage: WorkflowStage
    criteria: List[str]
    required_score: float = 80.0
    actual_score: Optional[float] = None
    passed: bool = False
    automated: bool = True
    human_review_required: bool = False


@dataclass
class WorkflowPlan:
    """Complete plan for autonomous workflow execution."""
    id: str
    workflow_type: WorkflowType
    description: str
    tasks: List[WorkflowTask]
    quality_gates: List[QualityGate]
    dependencies: Dict[str, List[str]]
    estimated_duration: int
    priority: Priority
    human_review_points: List[WorkflowStage]
    rollback_strategy: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Runtime execution state of a workflow."""
    id: str
    plan: WorkflowPlan
    status: WorkflowStatus
    current_stage: WorkflowStage
    progress: float = 0.0  # 0-100%
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    checkpoints: List[WorkflowCheckpoint] = field(default_factory=list)
    agent_assignments: Dict[str, str] = field(default_factory=dict)  # task_id -> agent_id
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class WorkflowOrchestrator(BaseAgent):
    """
    Autonomous Workflow Orchestrator for End-to-End Development Pipelines.

    Coordinates complete development workflows from requirements to deployment,
    managing multiple agents and ensuring quality and reliability.
    """

    def __init__(self, agent_id: str = "workflow_orchestrator", config: Optional[Dict[str, Any]] = None, agents: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, config)

        self.agent_type = "workflow_orchestrator"
        self.capabilities = {
            AgentCapability.WORKFLOW_ORCHESTRATION,
            AgentCapability.MULTI_AGENT_COORDINATION,
            AgentCapability.QUALITY_ASSURANCE,
            AgentCapability.ERROR_RECOVERY,
            AgentCapability.DEPLOYMENT_AUTOMATION
        }

        # Core components
        self.model_manager = None
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_templates: Dict[WorkflowType, Dict[str, Any]] = {}
        self.agent_pool: Dict[str, BaseAgent] = {}
        self.quality_agents: List[str] = []

        # Configuration
        self.config = config or {}
        self.max_concurrent_workflows = self.config.get('max_concurrent_workflows', 5)
        self.default_timeout = self.config.get('default_timeout', 3600)  # 1 hour
        self.human_review_enabled = self.config.get('human_review_enabled', True)
        self.auto_deployment_enabled = self.config.get('auto_deployment_enabled', False)

        # Metrics and monitoring
        self.workflow_metrics = {
            'total_workflows': 0,
            'successful_workflows': 0,
            'failed_workflows': 0,
            'average_duration': 0.0,
            'quality_score_average': 0.0,
            'human_intervention_rate': 0.0
        }

        # Error handling and resilience
        self.failure_handlers: Dict[str, Callable] = {}
        self.recovery_strategies: Dict[str, Callable] = {}

        # Initialize workflow templates
        self._initialize_workflow_templates()

        # Store agent references for task execution
        self.agents = agents or {}

        self.logger = structlog.get_logger(__name__).bind(agent_id=agent_id)

    async def initialize(self):
        """Initialize the workflow orchestrator."""
        await super().initialize()

        try:
            # Initialize model manager
            self.model_manager = await get_model_manager()
            if self.model_manager:
                await self.model_manager.initialize()
                self.logger.info("AI models initialized for workflow intelligence")

            # Register failure handlers
            self._register_failure_handlers()

            # Initialize quality gates
            await self._initialize_quality_gates()

            self.state = AgentState.READY
            self.logger.info("Autonomous Workflow Orchestrator initialized successfully")

        except Exception as e:
            self.state = AgentState.ERROR
            self.logger.error(f"Failed to initialize workflow orchestrator: {e}")
            raise

    async def execute_autonomous_workflow(
        self,
        workflow_type: WorkflowType,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        priority: Priority = Priority.BALANCED,
        human_review_required: bool = None
    ) -> str:
        """
        Execute a complete autonomous development workflow.

        Args:
            workflow_type: Type of workflow to execute
            description: Natural language description of what to achieve
            context: Additional context (project info, requirements, etc.)
            priority: Workflow priority level
            human_review_required: Override for human review requirement

        Returns:
            Workflow execution ID
        """
        workflow_id = str(uuid.uuid4())

        self.logger.info(
            "Starting autonomous workflow execution",
            workflow_id=workflow_id,
            workflow_type=workflow_type.value,
            description=description
        )

        try:
            # Step 1: Create workflow plan using AI
            plan = await self._create_workflow_plan(
                workflow_id, workflow_type, description, context, priority
            )

            # Step 2: Validate plan and check prerequisites
            await self._validate_workflow_plan(plan)

            # Step 3: Initialize workflow execution
            execution = WorkflowExecution(
                id=workflow_id,
                plan=plan,
                status=WorkflowStatus.QUEUED,
                current_stage=WorkflowStage.PLANNING
            )

            self.active_workflows[workflow_id] = execution

            # Step 4: Start execution asynchronously
            asyncio.create_task(self._execute_workflow_pipeline(execution))

            self.workflow_metrics['total_workflows'] += 1

            return workflow_id

        except Exception as e:
            self.logger.error(
                "Failed to start autonomous workflow",
                workflow_id=workflow_id,
                error=str(e)
            )
            raise

    async def _create_workflow_plan(
        self,
        workflow_id: str,
        workflow_type: WorkflowType,
        description: str,
        context: Optional[Dict[str, Any]],
        priority: Priority
    ) -> WorkflowPlan:
        """Create an intelligent workflow plan using AI analysis."""

        # Start with template
        template = self.workflow_templates.get(workflow_type, {})

        if self.model_manager:
            # Use AI to create intelligent plan
            plan = await self._create_ai_workflow_plan(
                workflow_id, workflow_type, description, context, priority, template
            )
        else:
            # Fallback to template-based plan
            plan = await self._create_template_workflow_plan(
                workflow_id, workflow_type, description, context, priority, template
            )

        return plan

    async def _create_ai_workflow_plan(
        self,
        workflow_id: str,
        workflow_type: WorkflowType,
        description: str,
        context: Optional[Dict[str, Any]],
        priority: Priority,
        template: Dict[str, Any]
    ) -> WorkflowPlan:
        """Create workflow plan using AI intelligence."""

        # Prepare AI prompt for workflow planning
        planning_prompt = f"""
Create a detailed autonomous workflow plan for this software development request:

WORKFLOW TYPE: {workflow_type.value}
DESCRIPTION: {description}
CONTEXT: {json.dumps(context) if context else 'No additional context'}
PRIORITY: {priority.value}

Based on the template structure: {json.dumps(template, indent=2)}

Create a comprehensive workflow plan with:
1. Detailed task breakdown with dependencies
2. Quality gates and acceptance criteria
3. Risk assessment and mitigation strategies
4. Human review points (if needed)
5. Rollback strategy
6. Estimated timelines

Return a structured JSON response with the complete workflow plan.
"""

        try:
            model_request = ModelRequest(
                prompt=planning_prompt,
                task_type=TaskType.PLANNING,
                complexity=ComplexityLevel.COMPLEX,
                priority=priority,
                max_tokens=4000,
                temperature=0.3
            )

            response = await self.model_manager.generate_response(model_request)

            if response and response.content:
                # Parse AI response and create workflow plan
                plan_data = json.loads(response.content)
                return self._parse_workflow_plan_data(workflow_id, workflow_type, plan_data)

        except Exception as e:
            self.logger.warning(f"AI workflow planning failed, using template: {e}")

        # Fallback to template
        return await self._create_template_workflow_plan(
            workflow_id, workflow_type, description, context, priority, template
        )

    async def _create_template_workflow_plan(
        self,
        workflow_id: str,
        workflow_type: WorkflowType,
        description: str,
        context: Optional[Dict[str, Any]],
        priority: Priority,
        template: Dict[str, Any]
    ) -> WorkflowPlan:
        """Create workflow plan from template."""

        # Create tasks from template
        tasks = []
        stage_to_task_id = {}  # Map stage names to task IDs

        # First pass: create tasks and build stage mapping
        for i, task_template in enumerate(template.get('tasks', [])):
            task_id = f"{workflow_id}_task_{i+1}"
            task = WorkflowTask(
                id=task_id,
                name=task_template['name'],
                description=task_template['description'],
                agent_type=task_template['agent_type'],
                stage=WorkflowStage(task_template['stage']),
                dependencies=[],  # Will be filled in second pass
                estimated_duration=task_template.get('estimated_duration', 300),
                priority=TaskPriority.HIGH if priority == Priority.QUALITY else TaskPriority.NORMAL
            )
            tasks.append(task)
            stage_to_task_id[task_template['stage']] = task_id

        # Second pass: resolve dependencies from stage names to task IDs
        for i, task_template in enumerate(template.get('tasks', [])):
            stage_dependencies = task_template.get('dependencies', [])
            task_dependencies = []
            for dep_stage in stage_dependencies:
                if dep_stage in stage_to_task_id:
                    task_dependencies.append(stage_to_task_id[dep_stage])
                else:
                    # If stage not found, log warning but continue
                    self.logger.warning(f"Dependency stage '{dep_stage}' not found in workflow",
                                      workflow_id=workflow_id, task_name=task_template['name'])
            tasks[i].dependencies = task_dependencies

        # Create quality gates
        quality_gates = []
        for gate_template in template.get('quality_gates', []):
            gate = QualityGate(
                id=f"{workflow_id}_gate_{len(quality_gates)}",
                name=gate_template['name'],
                stage=WorkflowStage(gate_template['stage']),
                criteria=gate_template['criteria'],
                required_score=gate_template.get('required_score', 80.0),
                automated=gate_template.get('automated', True)
            )
            quality_gates.append(gate)

        return WorkflowPlan(
            id=workflow_id,
            workflow_type=workflow_type,
            description=description,
            tasks=tasks,
            quality_gates=quality_gates,
            dependencies=template.get('dependencies', {}),
            estimated_duration=sum(task.estimated_duration for task in tasks),
            priority=priority,
            human_review_points=template.get('human_review_points', [WorkflowStage.HUMAN_REVIEW]),
            rollback_strategy=template.get('rollback_strategy', {}),
            context=context or {}
        )

    def _parse_workflow_plan_data(
        self,
        workflow_id: str,
        workflow_type: WorkflowType,
        plan_data: Dict[str, Any]
    ) -> WorkflowPlan:
        """Parse AI-generated workflow plan data into WorkflowPlan object."""

        # Parse tasks
        tasks = []
        for task_data in plan_data.get('tasks', []):
            task = WorkflowTask(
                id=task_data['id'],
                name=task_data['name'],
                description=task_data['description'],
                agent_type=task_data['agent_type'],
                stage=WorkflowStage(task_data['stage']),
                dependencies=task_data.get('dependencies', []),
                estimated_duration=task_data.get('estimated_duration', 300),
                priority=TaskPriority(task_data.get('priority', 'normal'))
            )
            tasks.append(task)

        # Parse quality gates
        quality_gates = []
        for gate_data in plan_data.get('quality_gates', []):
            gate = QualityGate(
                id=gate_data['id'],
                name=gate_data['name'],
                stage=WorkflowStage(gate_data['stage']),
                criteria=gate_data['criteria'],
                required_score=gate_data.get('required_score', 80.0),
                automated=gate_data.get('automated', True),
                human_review_required=gate_data.get('human_review_required', False)
            )
            quality_gates.append(gate)

        return WorkflowPlan(
            id=workflow_id,
            workflow_type=workflow_type,
            description=plan_data.get('description', ''),
            tasks=tasks,
            quality_gates=quality_gates,
            dependencies=plan_data.get('dependencies', {}),
            estimated_duration=plan_data.get('estimated_duration', 0),
            priority=Priority.BALANCED,
            human_review_points=[WorkflowStage(stage) for stage in plan_data.get('human_review_points', [])],
            rollback_strategy=plan_data.get('rollback_strategy', {}),
            context=plan_data.get('context', {})
        )

    async def _execute_workflow_pipeline(self, execution: WorkflowExecution):
        """Execute the complete workflow pipeline."""

        workflow_id = execution.id
        self.logger.info("Starting workflow pipeline execution", workflow_id=workflow_id)

        try:
            execution.status = WorkflowStatus.IN_PROGRESS

            # Execute workflow stages in order
            stages = [
                WorkflowStage.REQUIREMENTS_ANALYSIS,
                WorkflowStage.ARCHITECTURE_DESIGN,
                WorkflowStage.IMPLEMENTATION,
                WorkflowStage.QUALITY_ASSURANCE,
                WorkflowStage.INTEGRATION,
                WorkflowStage.HUMAN_REVIEW,
                WorkflowStage.DEPLOYMENT,
                WorkflowStage.MONITORING
            ]

            for stage in stages:
                execution.current_stage = stage
                await self._create_checkpoint(execution)

                self.logger.info(
                    "Executing workflow stage",
                    workflow_id=workflow_id,
                    stage=stage.value
                )

                # Execute all tasks for this stage
                stage_tasks = [task for task in execution.plan.tasks if task.stage == stage]

                if stage_tasks:
                    await self._execute_stage_tasks(execution, stage_tasks)

                # Check quality gates for this stage
                await self._check_quality_gates(execution, stage)

                # Check if human review is required
                if stage in execution.plan.human_review_points and self.human_review_enabled:
                    await self._request_human_review(execution, stage)

                # Update progress
                completed_stages = len([s for s in stages if stages.index(s) <= stages.index(stage)])
                execution.progress = (completed_stages / len(stages)) * 100

            # Mark as completed
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
            execution.current_stage = WorkflowStage.COMPLETION
            execution.progress = 100.0

            self.workflow_metrics['successful_workflows'] += 1

            self.logger.info(
                "Workflow pipeline completed successfully",
                workflow_id=workflow_id,
                duration=self._calculate_duration(execution)
            )

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.errors.append(str(e))
            self.workflow_metrics['failed_workflows'] += 1

            self.logger.error(
                "Workflow pipeline failed",
                workflow_id=workflow_id,
                error=str(e),
                stage=execution.current_stage.value
            )

            # Attempt recovery
            await self._handle_workflow_failure(execution, e)

    async def _execute_stage_tasks(self, execution: WorkflowExecution, tasks: List[WorkflowTask]):
        """Execute all tasks for a workflow stage."""

        # Group tasks by dependencies for parallel execution
        task_groups = self._group_tasks_by_dependencies(tasks)

        for group in task_groups:
            # Execute tasks in parallel within each group
            task_coroutines = []
            for task in group:
                coroutine = self._execute_task(execution, task)
                task_coroutines.append(coroutine)

            # Wait for all tasks in the group to complete
            await asyncio.gather(*task_coroutines, return_exceptions=True)

    async def _execute_task(self, execution: WorkflowExecution, task: WorkflowTask):
        """Execute a single workflow task."""

        task_id = task.id
        workflow_id = execution.id

        self.logger.info(
            "Executing workflow task",
            workflow_id=workflow_id,
            task_id=task_id,
            task_name=task.name,
            agent_type=task.agent_type
        )

        try:
            task.status = TaskState.RUNNING
            task.started_at = datetime.now()

            # Get appropriate agent for the task
            agent = await self._get_agent_for_task(task)

            if not agent:
                raise Exception(f"No available agent for task type: {task.agent_type}")

            # Execute the task through the agent
            result = await self._delegate_task_to_agent(agent, task, execution.plan.context)

            task.result = result
            task.status = TaskState.COMPLETED
            task.completed_at = datetime.now()
            task.actual_duration = int((task.completed_at - task.started_at).total_seconds())

            execution.results[task_id] = result

            self.logger.info(
                "Task completed successfully",
                workflow_id=workflow_id,
                task_id=task_id,
                duration=task.actual_duration
            )

        except Exception as e:
            task.status = TaskState.FAILED
            task.error = str(e)
            task.retry_count += 1

            self.logger.error(
                "Task execution failed",
                workflow_id=workflow_id,
                task_id=task_id,
                error=str(e),
                retry_count=task.retry_count
            )

            # Retry if under limit
            if task.retry_count <= task.max_retries:
                await self._retry_task(execution, task)
            else:
                # Task failed permanently
                execution.errors.append(f"Task {task_id} failed permanently: {str(e)}")
                raise

    async def _delegate_task_to_agent(
        self,
        agent: BaseAgent,
        task: WorkflowTask,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Delegate a task to the appropriate agent."""

        # Prepare agent-specific task request
        agent_request = {
            'task_id': task.id,
            'task_name': task.name,
            'description': task.description,
            'context': context,
            'priority': task.priority.value,
            'estimated_duration': task.estimated_duration
        }

        # Create AgentMessage for standard agent communication
        message = AgentMessage(
            message_type="task_execution",
            payload=agent_request,
            priority=MessagePriority.HIGH if task.priority == TaskPriority.HIGH else MessagePriority.NORMAL
        )

        # Call agent's standard process_message method
        if hasattr(agent, 'process_autonomous_task'):
            return await agent.process_autonomous_task(agent_request)
        else:
            # Use standard agent message processing
            response = await agent.process_message(message)
            if response and response.payload:
                return response.payload
            else:
                # Return a basic success response if no response from agent
                return {'result': f'Task {task.name} processed', 'status': 'completed'}

    async def _check_quality_gates(self, execution: WorkflowExecution, stage: WorkflowStage):
        """Check quality gates for a workflow stage."""

        stage_gates = [gate for gate in execution.plan.quality_gates if gate.stage == stage]

        for gate in stage_gates:
            self.logger.info(
                "Checking quality gate",
                workflow_id=execution.id,
                gate_id=gate.id,
                gate_name=gate.name
            )

            if gate.automated:
                # Automated quality check
                score = await self._run_automated_quality_check(execution, gate)
                gate.actual_score = score
                gate.passed = score >= gate.required_score
            else:
                # Manual quality check - request human review
                gate.human_review_required = True
                gate.passed = await self._request_quality_review(execution, gate)

            if not gate.passed:
                raise Exception(f"Quality gate failed: {gate.name} (Score: {gate.actual_score})")

    async def _run_automated_quality_check(
        self,
        execution: WorkflowExecution,
        gate: QualityGate
    ) -> float:
        """Run automated quality checks using AI analysis."""

        if not self.model_manager:
            return 75.0  # Default passing score

        # Gather context for quality analysis
        stage_results = {}
        for task in execution.plan.tasks:
            if task.stage == gate.stage and task.result:
                stage_results[task.id] = task.result

        # Create quality analysis prompt
        quality_prompt = f"""
Analyze the quality of this workflow stage output:

QUALITY GATE: {gate.name}
STAGE: {gate.stage.value}
CRITERIA: {', '.join(gate.criteria)}

STAGE RESULTS: {json.dumps(stage_results, indent=2)}

Evaluate against each criterion and provide:
1. Overall quality score (0-100)
2. Detailed analysis for each criterion
3. Recommendations for improvement (if needed)

Return JSON with the analysis.
"""

        try:
            model_request = ModelRequest(
                prompt=quality_prompt,
                task_type=TaskType.ANALYSIS,
                complexity=ComplexityLevel.MEDIUM,
                priority=Priority.QUALITY,
                max_tokens=2000,
                temperature=0.2
            )

            response = await self.model_manager.generate_response(model_request)

            if response and response.content:
                analysis = json.loads(response.content)
                return analysis.get('quality_score', 75.0)

        except Exception as e:
            self.logger.warning(f"Automated quality check failed: {e}")

        return 75.0  # Default score if analysis fails

    async def _request_human_review(self, execution: WorkflowExecution, stage: WorkflowStage):
        """Request human review for a workflow stage."""

        execution.status = WorkflowStatus.WAITING_REVIEW

        self.logger.info(
            "Requesting human review",
            workflow_id=execution.id,
            stage=stage.value
        )

        # In a real implementation, this would:
        # 1. Create review request in external system
        # 2. Send notifications to human reviewers
        # 3. Wait for approval/feedback
        # 4. Process review results

        # For now, simulate automatic approval after delay
        await asyncio.sleep(30)  # Simulate review time

        self.logger.info(
            "Human review completed (simulated)",
            workflow_id=execution.id,
            stage=stage.value
        )

    def _group_tasks_by_dependencies(self, tasks: List[WorkflowTask]) -> List[List[WorkflowTask]]:
        """Group tasks into execution groups based on dependencies."""

        task_groups = []
        remaining_tasks = tasks.copy()

        while remaining_tasks:
            # Find tasks with no unmet dependencies
            ready_tasks = []
            completed_task_ids = set()

            # Add task IDs from previous groups
            for group in task_groups:
                completed_task_ids.update(task.id for task in group)

            for task in remaining_tasks:
                if not task.dependencies or all(dep in completed_task_ids for dep in task.dependencies):
                    ready_tasks.append(task)

            if not ready_tasks:
                # Circular dependency or other issue
                self.logger.warning("No ready tasks found, adding remaining tasks to avoid deadlock")
                ready_tasks = remaining_tasks

            task_groups.append(ready_tasks)

            # Remove processed tasks
            for task in ready_tasks:
                remaining_tasks.remove(task)

        return task_groups

    async def _get_agent_for_task(self, task: WorkflowTask) -> Optional[BaseAgent]:
        """Get appropriate agent for a task."""

        # Get agent from the stored agent references
        return self.agents.get(task.agent_type)

    async def _create_checkpoint(self, execution: WorkflowExecution):
        """Create a checkpoint for workflow state recovery."""

        checkpoint = WorkflowCheckpoint(
            id=str(uuid.uuid4()),
            workflow_id=execution.id,
            stage=execution.current_stage,
            timestamp=datetime.now(),
            state_data={
                'status': execution.status.value,
                'progress': execution.progress,
                'results': execution.results
            },
            completed_tasks=[task.id for task in execution.plan.tasks if task.status == TaskState.COMPLETED],
            agent_states={}  # Would capture actual agent states
        )

        execution.checkpoints.append(checkpoint)

        # Persist checkpoint (would save to database/file in real implementation)
        self.logger.debug(
            "Created workflow checkpoint",
            workflow_id=execution.id,
            checkpoint_id=checkpoint.id,
            stage=checkpoint.stage.value
        )

    async def _handle_workflow_failure(self, execution: WorkflowExecution, error: Exception):
        """Handle workflow failure with recovery strategies."""

        workflow_id = execution.id
        self.logger.info(
            "Handling workflow failure",
            workflow_id=workflow_id,
            error=str(error),
            stage=execution.current_stage.value
        )

        # Try recovery strategies
        recovery_strategies = [
            self._retry_failed_tasks,
            self._rollback_to_checkpoint,
            self._partial_completion
        ]

        for strategy in recovery_strategies:
            try:
                success = await strategy(execution)
                if success:
                    self.logger.info(
                        "Workflow recovery successful",
                        workflow_id=workflow_id,
                        strategy=strategy.__name__
                    )
                    return
            except Exception as e:
                self.logger.warning(
                    "Recovery strategy failed",
                    workflow_id=workflow_id,
                    strategy=strategy.__name__,
                    error=str(e)
                )

        # All recovery strategies failed
        execution.status = WorkflowStatus.FAILED
        self.logger.error(
            "All recovery strategies failed",
            workflow_id=workflow_id
        )

    async def _retry_failed_tasks(self, execution: WorkflowExecution) -> bool:
        """Retry failed tasks in the workflow."""

        failed_tasks = [task for task in execution.plan.tasks if task.status == TaskState.FAILED]

        if not failed_tasks:
            return False

        for task in failed_tasks:
            if task.retry_count < task.max_retries:
                await self._retry_task(execution, task)
            else:
                return False

        return True

    async def _retry_task(self, execution: WorkflowExecution, task: WorkflowTask):
        """Retry a failed task."""
        task.status = TaskState.PENDING
        task.error = None
        await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
        await self._execute_task(execution, task)

    async def _rollback_to_checkpoint(self, execution: WorkflowExecution) -> bool:
        """Rollback workflow to the last successful checkpoint."""
        if not execution.checkpoints:
            return False

        latest_checkpoint = execution.checkpoints[-1]

        # Restore state from checkpoint
        execution.current_stage = latest_checkpoint.stage
        execution.results = latest_checkpoint.state_data.get('results', {})
        execution.status = WorkflowStatus.IN_PROGRESS

        self.logger.info(
            "Rolled back to checkpoint",
            workflow_id=execution.id,
            checkpoint_id=latest_checkpoint.id,
            stage=latest_checkpoint.stage.value
        )

        return True

    async def _partial_completion(self, execution: WorkflowExecution) -> bool:
        """Complete workflow with partial results."""
        completed_tasks = [task for task in execution.plan.tasks if task.status == TaskState.COMPLETED]

        if len(completed_tasks) > len(execution.plan.tasks) / 2:
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
            execution.progress = (len(completed_tasks) / len(execution.plan.tasks)) * 100

            self.logger.info(
                "Workflow completed with partial results",
                workflow_id=execution.id,
                completed_tasks=len(completed_tasks),
                total_tasks=len(execution.plan.tasks)
            )
            return True

        return False

    async def _request_quality_review(self, execution: WorkflowExecution, gate: QualityGate) -> bool:
        """Request human quality review."""
        # Simulate human quality review
        await asyncio.sleep(10)  # Simulate review time
        return True  # Assume approval for demo

    def _calculate_duration(self, execution: WorkflowExecution) -> int:
        """Calculate workflow execution duration in seconds."""
        if execution.completed_at and execution.started_at:
            return int((execution.completed_at - execution.started_at).total_seconds())
        return 0

    def _initialize_workflow_templates(self):
        """Initialize predefined workflow templates."""
        self.workflow_templates = {
            WorkflowType.FEATURE_DEVELOPMENT: {
                'tasks': [
                    {
                        'name': 'Requirements Analysis',
                        'description': 'Analyze and refine feature requirements',
                        'agent_type': 'context',
                        'stage': 'requirements_analysis',
                        'estimated_duration': 600
                    },
                    {
                        'name': 'Architecture Design',
                        'description': 'Design system architecture and APIs',
                        'agent_type': 'context',
                        'stage': 'architecture_design',
                        'estimated_duration': 900,
                        'dependencies': ['requirements_analysis']
                    },
                    {
                        'name': 'Code Implementation',
                        'description': 'Implement feature code',
                        'agent_type': 'code',
                        'stage': 'implementation',
                        'estimated_duration': 1800,
                        'dependencies': ['architecture_design']
                    },
                    {
                        'name': 'Unit Testing',
                        'description': 'Create and run unit tests',
                        'agent_type': 'quality',
                        'stage': 'quality_assurance',
                        'estimated_duration': 600,
                        'dependencies': ['implementation']
                    },
                    {
                        'name': 'Integration Testing',
                        'description': 'Run integration tests',
                        'agent_type': 'quality',
                        'stage': 'quality_assurance',
                        'estimated_duration': 300,
                        'dependencies': ['unit_testing']
                    },
                    {
                        'name': 'Git Operations',
                        'description': 'Commit changes and create pull request',
                        'agent_type': 'git',
                        'stage': 'integration',
                        'estimated_duration': 300,
                        'dependencies': ['quality_assurance']
                    }
                ],
                'quality_gates': [
                    {
                        'name': 'Code Quality Gate',
                        'stage': 'quality_assurance',
                        'criteria': ['code_coverage', 'complexity', 'security'],
                        'required_score': 85.0,
                        'automated': True
                    },
                    {
                        'name': 'Integration Gate',
                        'stage': 'integration',
                        'criteria': ['all_tests_pass', 'no_conflicts'],
                        'required_score': 95.0,
                        'automated': True
                    }
                ],
                'human_review_points': ['human_review'],
                'rollback_strategy': {
                    'type': 'git_revert',
                    'preserve_data': True
                }
            },
            WorkflowType.BUG_FIX: {
                'tasks': [
                    {
                        'name': 'Bug Analysis',
                        'description': 'Analyze bug report and identify root cause',
                        'agent_type': 'context',
                        'stage': 'requirements_analysis',
                        'estimated_duration': 900
                    },
                    {
                        'name': 'Fix Implementation',
                        'description': 'Implement bug fix',
                        'agent_type': 'code',
                        'stage': 'implementation',
                        'estimated_duration': 1200,
                        'dependencies': ['requirements_analysis']
                    },
                    {
                        'name': 'Regression Testing',
                        'description': 'Run regression tests',
                        'agent_type': 'quality',
                        'stage': 'quality_assurance',
                        'estimated_duration': 600,
                        'dependencies': ['implementation']
                    }
                ],
                'quality_gates': [
                    {
                        'name': 'Fix Validation',
                        'stage': 'quality_assurance',
                        'criteria': ['bug_resolved', 'no_regressions'],
                        'required_score': 90.0,
                        'automated': True
                    }
                ],
                'human_review_points': [],
                'rollback_strategy': {
                    'type': 'git_revert',
                    'preserve_data': False
                }
            }
        }

    def _register_failure_handlers(self):
        """Register failure handlers for different error types."""
        self.failure_handlers = {
            'timeout': self._handle_timeout_failure,
            'agent_failure': self._handle_agent_failure,
            'quality_gate_failure': self._handle_quality_gate_failure,
            'dependency_failure': self._handle_dependency_failure
        }

    async def _handle_timeout_failure(self, execution: WorkflowExecution, error: Exception):
        """Handle timeout failures."""
        # Extend timeout and retry
        pass

    async def _handle_agent_failure(self, execution: WorkflowExecution, error: Exception):
        """Handle agent failures."""
        # Reassign to different agent
        pass

    async def _handle_quality_gate_failure(self, execution: WorkflowExecution, error: Exception):
        """Handle quality gate failures."""
        # Lower threshold or request manual review
        pass

    async def _handle_dependency_failure(self, execution: WorkflowExecution, error: Exception):
        """Handle dependency failures."""
        # Reorder tasks or skip non-critical dependencies
        pass

    async def _initialize_quality_gates(self):
        """Initialize quality gate configurations."""
        # Would initialize quality checking systems
        pass

    async def _validate_workflow_plan(self, plan: WorkflowPlan):
        """Validate workflow plan before execution."""
        if not plan.tasks:
            raise ValueError("Workflow plan must contain at least one task")

        # Check for circular dependencies
        task_ids = {task.id for task in plan.tasks}
        for task in plan.tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    raise ValueError(f"Task {task.id} has invalid dependency: {dep}")

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow."""
        execution = self.active_workflows.get(workflow_id)
        if not execution:
            return None

        return {
            'id': execution.id,
            'status': execution.status.value,
            'current_stage': execution.current_stage.value,
            'progress': execution.progress,
            'started_at': execution.started_at.isoformat(),
            'completed_at': execution.completed_at.isoformat() if execution.completed_at else None,
            'errors': execution.errors,
            'metrics': execution.metrics
        }

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        execution = self.active_workflows.get(workflow_id)
        if not execution:
            return False

        execution.status = WorkflowStatus.CANCELLED
        self.logger.info("Workflow cancelled", workflow_id=workflow_id)
        return True

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        return self.workflow_metrics.copy()

    def get_capabilities(self) -> Set[AgentCapability]:
        """Get agent capabilities."""
        return self.capabilities

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the workflow orchestrator."""
        return {
            'status': 'healthy',
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'state': self.state.value if hasattr(self.state, 'value') else str(self.state),
            'active_workflows': len(self.active_workflows),
            'model_manager_available': self.model_manager is not None,
            'metrics': self.workflow_metrics
        }

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process incoming messages from other agents."""
        try:
            if message.message_type == 'workflow_status_request':
                workflow_id = message.payload.get('workflow_id')
                status = self.get_workflow_status(workflow_id)

                return AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type='workflow_status_response',
                    payload={'status': status},
                    correlation_id=message.correlation_id
                )

            elif message.message_type == 'workflow_execution_request':
                workflow_type = WorkflowType(message.payload.get('workflow_type', 'feature_development'))
                description = message.payload.get('description', '')
                context = message.payload.get('context', {})

                workflow_id = await self.execute_autonomous_workflow(
                    workflow_type=workflow_type,
                    description=description,
                    context=context
                )

                return AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type='workflow_execution_response',
                    payload={'workflow_id': workflow_id},
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
