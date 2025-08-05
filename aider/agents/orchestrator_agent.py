"""
OrchestratorAgent: Central coordinator for the Aider Multi-Agent Hive Architecture.

This agent serves as the primary entry point for user requests, responsible for:
- Analyzing and decomposing complex user requests
- Delegating tasks to specialized agents
- Coordinating multi-agent workflows
- Synthesizing results from multiple agents
- Managing the overall execution flow
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import structlog

from .base_agent import BaseAgent, AgentMessage, MessagePriority, AgentState, AgentCapability
from ..task_management.task_queue import Task, TaskPriority, TaskState
from ..models.model_manager import get_model_manager, ModelRequest, TaskType, ComplexityLevel, Priority


class RequestType(Enum):
    """Classification of user request types."""
    CODE_GENERATION = "code_generation"
    CODE_MODIFICATION = "code_modification"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    GIT_OPERATION = "git_operation"
    PROJECT_ANALYSIS = "project_analysis"
    MIXED_WORKFLOW = "mixed_workflow"
    UNKNOWN = "unknown"


class RequestComplexity(Enum):
    """Complexity levels for request analysis."""
    SIMPLE = "simple"       # Single agent, single task
    MODERATE = "moderate"   # Single agent, multiple tasks
    COMPLEX = "complex"     # Multiple agents, coordinated tasks
    CRITICAL = "critical"   # Multiple agents, complex dependencies


@dataclass
class SubTask:
    """Individual subtask within a request analysis."""
    id: str
    description: str
    agent_type: str
    estimated_duration: int
    dependencies: List[str]
    priority: TaskPriority


@dataclass
class RequestAnalysis:
    """Results of request analysis."""
    request_id: str
    request_type: RequestType
    complexity: RequestComplexity
    required_agents: Set[str]
    subtasks: List[SubTask]
    dependencies: Dict[str, List[str]]
    estimated_duration: Optional[float] = None
    confidence_score: float = 0.0
    context_requirements: Optional[List[str]] = None
    risk_factors: Optional[List[str]] = None


@dataclass
class WorkflowStep:
    """Individual step in a workflow execution."""
    step_id: str
    agent_type: str
    task_data: Dict[str, Any]
    dependencies: List[str]
    status: TaskState = TaskState.PENDING
    result: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class OrchestratorAgent(BaseAgent):
    """
    Central orchestrator agent that coordinates all other agents in the hive.

    Responsibilities:
    - Request analysis and decomposition
    - Task delegation and workflow management
    - Agent coordination and communication
    - Result synthesis and response generation
    - Error handling and recovery
    """

    def __init__(
        self,
        agent_id: str = "orchestrator",
        config: Optional[Dict[str, Any]] = None,
        message_bus=None,
    ):
        """Initialize the orchestrator agent."""
        super().__init__(
            agent_id=agent_id,
            agent_type="orchestrator",
            config=config,
            message_bus=message_bus,
        )

        # Orchestrator-specific configuration
        self.max_concurrent_workflows = self.config.get('max_concurrent_workflows', 5)
        self.default_timeout = self.config.get('default_timeout', 300.0)  # 5 minutes
        self.analysis_timeout = self.config.get('analysis_timeout', 30.0)

        # Active workflows tracking
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_lock = asyncio.Lock()

        # AI Model Manager
        self.model_manager = None

        # Agent registry and capabilities
        self.available_agents: Dict[str, Set[str]] = {
            'code': {'generate_code', 'modify_code', 'review_code', 'debug_code'},
            'context': {'build_context', 'search_context', 'analyze_project'},
            'git': {'commit_changes', 'create_branch', 'merge_branch', 'resolve_conflicts'},
        }

        # Request analysis patterns
        self.request_patterns = {
            'code_generation': [
                'create', 'generate', 'write', 'implement', 'build', 'make'
            ],
            'code_modification': [
                'modify', 'change', 'update', 'edit', 'fix', 'alter', 'refactor'
            ],
            'code_review': [
                'review', 'check', 'analyze', 'examine', 'audit', 'validate'
            ],
            'debugging': [
                'debug', 'troubleshoot', 'find bug', 'error', 'issue', 'problem'
            ],
            'git_operation': [
                'commit', 'push', 'pull', 'merge', 'branch', 'checkout', 'git'
            ],
            'documentation': [
                'document', 'explain', 'describe', 'comment', 'readme', 'docs'
            ]
        }

        # Performance metrics
        self.workflow_metrics = {
            'total_requests': 0,
            'successful_workflows': 0,
            'failed_workflows': 0,
            'average_duration': 0.0,
            'agent_utilization': {}
        }

    async def initialize(self) -> None:
        """Initialize the orchestrator agent."""
        try:
            await super().initialize()

            # Initialize AI Model Manager
            self.model_manager = await get_model_manager()
            self.logger.info("ModelManager initialized successfully")

            # Register orchestrator-specific message handlers
            self.register_message_handler('user_request', self._handle_user_request)
            self.register_message_handler('agent_response', self._handle_agent_response)
            self.register_message_handler('workflow_status', self._handle_workflow_status)

            self.logger.info("OrchestratorAgent initialized successfully")
            pass

        except Exception as e:
            self.logger.error(f"Failed to initialize OrchestratorAgent: {e}")
            raise

    async def process_user_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> Dict[str, Any]:
        """
        Process a user request through the multi-agent workflow.

        Args:
            request: The user's natural language request
            context: Optional context information
            user_id: Optional user identifier
            priority: Request priority level

        Returns:
            Dict containing the workflow results and metadata
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        self.logger.info(
            "Processing user request",
            request_id=request_id,
            user_id=user_id,
            priority=priority.value
        )

        try:
            # Step 1: Analyze the request
            analysis = await self.analyze_request(request, context)

            # Step 2: Create workflow plan
            workflow_plan = await self._create_workflow_plan(analysis, priority)

            # Step 3: Execute workflow
            results = await self._execute_workflow(request_id, workflow_plan)

            # Step 4: Synthesize final response
            final_response = await self._synthesize_response(request_id, results)

            # Update metrics
            duration = time.time() - start_time
            self.workflow_metrics['total_requests'] += 1
            self.workflow_metrics['successful_workflows'] += 1
            self._update_average_duration(duration)

            return {
                'request_id': request_id,
                'status': 'completed',
                'response': final_response,
                'metadata': {
                    'analysis': analysis,
                    'duration': duration,
                    'agents_used': list(analysis.required_agents),
                    'complexity': analysis.complexity.value
                }
            }

        except Exception as e:
            self.logger.error(
                "Failed to process user request",
                request_id=request_id,
                error=str(e),
                exc_info=True
            )

            self.workflow_metrics['failed_workflows'] += 1

            return {
                'request_id': request_id,
                'status': 'failed',
                'error': str(e),
                'metadata': {
                    'duration': time.time() - start_time
                }
            }

    async def _analyze_request_with_ai(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Use AI to analyze and understand the user request.

        Args:
            request: The user's natural language request
            context: Optional context information

        Returns:
            Dict with AI analysis results
        """
        if not self.model_manager:
            raise RuntimeError("ModelManager not initialized")

        # Create analysis prompt
        analysis_prompt = f"""
Analyze this software development request and provide a structured analysis:

REQUEST: {request}

PROJECT CONTEXT: {json.dumps(context) if context else 'No context provided'}

Provide analysis in this JSON format:
{{
    "request_type": "code_generation|code_modification|code_review|debugging|refactoring|documentation|git_operation|project_analysis|mixed_workflow",
    "complexity": "simple|moderate|complex|critical",
    "required_agents": ["code", "context", "git"],
    "subtasks": [
        {{
            "id": "task_1",
            "description": "Detailed task description",
            "agent": "code|context|git",
            "estimated_duration": 30,
            "dependencies": []
        }}
    ],
    "confidence_score": 0.85,
    "context_requirements": ["project_structure", "dependencies", "git_history"],
    "risk_assessment": "low|medium|high",
    "recommendations": ["Specific recommendations for execution"]
}}

Focus on:
1. What type of development task this is
2. Which agents are needed (code, context, git)
3. Breaking down into specific subtasks
4. Identifying dependencies between tasks
5. Assessing complexity and risk
"""

        # Generate AI analysis
        ai_request = ModelRequest(
            prompt=analysis_prompt,
            task_type=TaskType.ANALYSIS,
            complexity=ComplexityLevel.MEDIUM,
            priority=Priority.QUALITY,
            max_tokens=2000,
            temperature=0.1
        )

        try:
            response = await self.model_manager.generate_response(ai_request)

            try:
                # Parse AI response as JSON
                analysis = json.loads(response.content)
                self.logger.info("AI request analysis completed successfully")
                return analysis
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse AI analysis response: {e}")
                # Use mock response when parsing fails
                return self._get_mock_analysis(request, context)

        except Exception as e:
            self.logger.warning(f"AI model unavailable ({e}), using mock analysis for demo")
            # Use mock analysis when AI models aren't available
            return self._get_mock_analysis(request, context)

    def _get_mock_analysis(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate realistic mock analysis for demo purposes"""
        request_lower = request.lower()

        # Analyze request to provide realistic mock responses
        if any(word in request_lower for word in ['create', 'generate', 'write', 'implement', 'build']):
            return {
                "request_type": "code_generation",
                "complexity": "moderate",
                "required_agents": ["context", "code", "git"],
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": "Analyze project context and requirements",
                        "agent": "context",
                        "estimated_duration": 15,
                        "dependencies": []
                    },
                    {
                        "id": "task_2",
                        "description": "Generate code based on requirements",
                        "agent": "code",
                        "estimated_duration": 45,
                        "dependencies": ["task_1"]
                    },
                    {
                        "id": "task_3",
                        "description": "Create unit tests and documentation",
                        "agent": "code",
                        "estimated_duration": 30,
                        "dependencies": ["task_2"]
                    },
                    {
                        "id": "task_4",
                        "description": "Commit changes with descriptive message",
                        "agent": "git",
                        "estimated_duration": 10,
                        "dependencies": ["task_3"]
                    }
                ],
                "confidence_score": 0.85,
                "context_requirements": ["project_structure", "dependencies", "coding_standards"],
                "risk_assessment": "low",
                "recommendations": ["Ensure comprehensive testing", "Follow coding standards", "Add error handling"]
            }
        elif any(word in request_lower for word in ['refactor', 'modify', 'change', 'update', 'improve']):
            return {
                "request_type": "code_modification",
                "complexity": "complex",
                "required_agents": ["context", "code", "git"],
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": "Analyze existing code structure and dependencies",
                        "agent": "context",
                        "estimated_duration": 20,
                        "dependencies": []
                    },
                    {
                        "id": "task_2",
                        "description": "Plan refactoring approach and identify risks",
                        "agent": "code",
                        "estimated_duration": 25,
                        "dependencies": ["task_1"]
                    },
                    {
                        "id": "task_3",
                        "description": "Execute refactoring with safety checks",
                        "agent": "code",
                        "estimated_duration": 60,
                        "dependencies": ["task_2"]
                    },
                    {
                        "id": "task_4",
                        "description": "Update tests and validate changes",
                        "agent": "code",
                        "estimated_duration": 35,
                        "dependencies": ["task_3"]
                    },
                    {
                        "id": "task_5",
                        "description": "Commit refactored code with detailed message",
                        "agent": "git",
                        "estimated_duration": 10,
                        "dependencies": ["task_4"]
                    }
                ],
                "confidence_score": 0.75,
                "context_requirements": ["project_structure", "dependencies", "test_coverage", "git_history"],
                "risk_assessment": "medium",
                "recommendations": ["Backup before refactoring", "Run comprehensive tests", "Review dependencies"]
            }
        elif any(word in request_lower for word in ['review', 'analyze', 'check', 'audit']):
            return {
                "request_type": "code_review",
                "complexity": "moderate",
                "required_agents": ["context", "code"],
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": "Load and analyze code structure",
                        "agent": "context",
                        "estimated_duration": 15,
                        "dependencies": []
                    },
                    {
                        "id": "task_2",
                        "description": "Perform comprehensive code review",
                        "agent": "code",
                        "estimated_duration": 40,
                        "dependencies": ["task_1"]
                    },
                    {
                        "id": "task_3",
                        "description": "Generate review report with recommendations",
                        "agent": "code",
                        "estimated_duration": 20,
                        "dependencies": ["task_2"]
                    }
                ],
                "confidence_score": 0.90,
                "context_requirements": ["project_structure", "coding_standards", "test_coverage"],
                "risk_assessment": "low",
                "recommendations": ["Focus on security", "Check performance", "Validate error handling"]
            }
        else:
            # Default general analysis
            return {
                "request_type": "general",
                "complexity": "moderate",
                "required_agents": ["context", "code"],
                "subtasks": [
                    {
                        "id": "task_1",
                        "description": "Understand request context and requirements",
                        "agent": "context",
                        "estimated_duration": 20,
                        "dependencies": []
                    },
                    {
                        "id": "task_2",
                        "description": f"Execute requested task: {request[:50]}...",
                        "agent": "code",
                        "estimated_duration": 45,
                        "dependencies": ["task_1"]
                    }
                ],
                "confidence_score": 0.70,
                "context_requirements": ["project_structure"],
                "risk_assessment": "medium",
                "recommendations": ["Clarify requirements", "Consider edge cases"]
            }

    async def _fallback_analysis(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Fallback analysis when AI fails"""
        return {
            "request_type": "general",
            "complexity": "moderate",
            "required_agents": ["code", "context"],
            "subtasks": [
                {
                    "id": "task_1",
                    "description": request,
                    "agent": "code",
                    "estimated_duration": 60,
                    "dependencies": []
                }
            ],
            "confidence_score": 0.5,
            "context_requirements": ["project_structure"],
            "risk_assessment": "medium",
            "recommendations": ["Review request manually"]
        }

    async def analyze_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RequestAnalysis:
        """
        Analyze a user request to determine the required workflow using AI.

        Args:
            request: The user's natural language request
            context: Optional context information

        Returns:
            RequestAnalysis with workflow planning information
        """
        request_id = str(uuid.uuid4())

        self.logger.debug("Analyzing request with AI", request_id=request_id)

        try:
            # Get AI analysis
            ai_analysis = await self._analyze_request_with_ai(request, context)

            # Convert AI analysis to RequestAnalysis object
            request_type = RequestType(ai_analysis.get("request_type", "unknown"))
            complexity = RequestComplexity(ai_analysis.get("complexity", "moderate"))
            required_agents = set(ai_analysis.get("required_agents", ["code"]))

            # Process subtasks
            subtasks = []
            for task_data in ai_analysis.get("subtasks", []):
                subtask = SubTask(
                    id=task_data.get("id", str(uuid.uuid4())),
                    description=task_data.get("description", ""),
                    agent_type=task_data.get("agent", "code"),
                    estimated_duration=task_data.get("estimated_duration", 60),
                    dependencies=task_data.get("dependencies", []),
                    priority=TaskPriority.NORMAL
                )
                subtasks.append(subtask)

            # Create analysis result
            analysis = RequestAnalysis(
                request_id=request_id,
                request_type=request_type,
                complexity=complexity,
                required_agents=required_agents,
                subtasks=subtasks,
                dependencies=self._extract_dependencies(subtasks),
                estimated_duration=sum(task.estimated_duration for task in subtasks),
                confidence_score=ai_analysis.get("confidence_score", 0.7),
                context_requirements=ai_analysis.get("context_requirements", []),
                risk_factors=ai_analysis.get("recommendations", [])
            )

            self.logger.info(
                "Request analysis completed",
                request_id=request_id,
                request_type=request_type.value,
                complexity=complexity.value,
                agents_needed=len(required_agents),
                subtasks_count=len(subtasks)
            )

            return analysis

        except Exception as e:
            self.logger.error(f"Request analysis failed: {e}", exc_info=True)
            # Return basic fallback analysis
            return await self._create_fallback_analysis(request_id, request)

    def _extract_dependencies(self, subtasks: List) -> Dict[str, List[str]]:
        """Extract task dependencies from subtasks"""
        dependencies = {}
        for task in subtasks:
            if hasattr(task, 'dependencies') and task.dependencies:
                dependencies[task.id] = task.dependencies
        return dependencies

    async def _create_fallback_analysis(self, request_id: str, request: str) -> RequestAnalysis:
        """Create a basic fallback analysis when AI analysis fails"""
        return RequestAnalysis(
            request_id=request_id,
            request_type=RequestType.UNKNOWN,
            complexity=RequestComplexity.MODERATE,
            required_agents={"code"},
            subtasks=[
                SubTask(
                    id="fallback_task",
                    description=request,
                    agent_type="code",
                    estimated_duration=60,
                    dependencies=[],
                    priority=TaskPriority.NORMAL
                )
            ],
            dependencies={},
            estimated_duration=60,
            confidence_score=0.3,
            context_requirements=["project_structure"],
            risk_factors=["AI analysis failed - manual review recommended"]
        )

    def _classify_request_type(self, request: str) -> RequestType:
        """Classify the type of user request."""
        request_lower = request.lower()

        # Score each request type based on keyword matches
        type_scores = {}

        for req_type, keywords in self.request_patterns.items():
            score = sum(1 for keyword in keywords if keyword in request_lower)
            if score > 0:
                type_scores[req_type] = score

        if not type_scores:
            return RequestType.UNKNOWN

        # Return the type with the highest score
        best_type = max(type_scores.items(), key=lambda x: x[1])[0]

        # Map string keys to enum values
        type_mapping = {
            'code_generation': RequestType.CODE_GENERATION,
            'code_modification': RequestType.CODE_MODIFICATION,
            'code_review': RequestType.CODE_REVIEW,
            'debugging': RequestType.DEBUGGING,
            'git_operation': RequestType.GIT_OPERATION,
            'documentation': RequestType.DOCUMENTATION
        }

        return type_mapping.get(best_type, RequestType.UNKNOWN)

    def _assess_complexity(self, request: str, context: Optional[Dict[str, Any]]) -> RequestComplexity:
        """Assess the complexity level of the request."""
        complexity_indicators = {
            'simple': ['simple', 'quick', 'small', 'single'],
            'moderate': ['multiple', 'several', 'few', 'moderate'],
            'complex': ['complex', 'large', 'many', 'comprehensive', 'full'],
            'critical': ['critical', 'urgent', 'production', 'enterprise']
        }

        request_lower = request.lower()
        word_count = len(request.split())

        # Base complexity on word count
        if word_count < 10:
            base_complexity = RequestComplexity.SIMPLE
        elif word_count < 30:
            base_complexity = RequestComplexity.MODERATE
        elif word_count < 60:
            base_complexity = RequestComplexity.COMPLEX
        else:
            base_complexity = RequestComplexity.CRITICAL

        # Adjust based on complexity indicators
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in request_lower for indicator in indicators):
                if complexity == 'critical':
                    return RequestComplexity.CRITICAL
                elif complexity == 'complex' and base_complexity != RequestComplexity.CRITICAL:
                    base_complexity = RequestComplexity.COMPLEX
                elif complexity == 'moderate' and base_complexity == RequestComplexity.SIMPLE:
                    base_complexity = RequestComplexity.MODERATE

        return base_complexity

    def _identify_required_agents(self, request_type: RequestType, complexity: RequestComplexity) -> Set[str]:
        """Identify which agents are required for this request."""
        required_agents = set()

        # Always need context for project understanding
        required_agents.add('context')

        # Add agents based on request type
        if request_type in [RequestType.CODE_GENERATION, RequestType.CODE_MODIFICATION,
                           RequestType.CODE_REVIEW, RequestType.DEBUGGING]:
            required_agents.add('code')

        if request_type == RequestType.GIT_OPERATION:
            required_agents.add('git')

        # For complex requests, might need additional agents
        if complexity in [RequestComplexity.COMPLEX, RequestComplexity.CRITICAL]:
            if request_type != RequestType.GIT_OPERATION:
                required_agents.add('git')  # For version control of complex changes

        return required_agents

    async def _decompose_into_subtasks(
        self,
        request: str,
        request_type: RequestType,
        complexity: RequestComplexity
    ) -> List[Dict[str, Any]]:
        """Break down the request into manageable subtasks."""
        subtasks = []

        # Base subtask: context gathering
        subtasks.append({
            'id': f"context_{uuid.uuid4().hex[:8]}",
            'agent_type': 'context',
            'action': 'build_project_context',
            'description': 'Gather relevant project context',
            'priority': TaskPriority.HIGH.value,
            'estimated_duration': 30.0
        })

        # Add request-specific subtasks
        if request_type == RequestType.CODE_GENERATION:
            subtasks.extend([
                {
                    'id': f"code_{uuid.uuid4().hex[:8]}",
                    'agent_type': 'code',
                    'action': 'generate_code',
                    'description': f'Generate code based on request: {request[:100]}...',
                    'priority': TaskPriority.NORMAL.value,
                    'estimated_duration': 120.0
                }
            ])
        elif request_type == RequestType.CODE_MODIFICATION:
            subtasks.extend([
                {
                    'id': f"code_{uuid.uuid4().hex[:8]}",
                    'agent_type': 'code',
                    'action': 'modify_code',
                    'description': f'Modify code based on request: {request[:100]}...',
                    'priority': TaskPriority.NORMAL.value,
                    'estimated_duration': 90.0
                }
            ])
        elif request_type == RequestType.GIT_OPERATION:
            subtasks.extend([
                {
                    'id': f"git_{uuid.uuid4().hex[:8]}",
                    'agent_type': 'git',
                    'action': 'perform_git_operation',
                    'description': f'Perform git operation: {request[:100]}...',
                    'priority': TaskPriority.NORMAL.value,
                    'estimated_duration': 60.0
                }
            ])

        # Add commit subtask for code changes
        if request_type in [RequestType.CODE_GENERATION, RequestType.CODE_MODIFICATION]:
            subtasks.append({
                'id': f"git_{uuid.uuid4().hex[:8]}",
                'agent_type': 'git',
                'action': 'intelligent_commit',
                'description': 'Create intelligent commit for changes',
                'priority': TaskPriority.LOW.value,
                'estimated_duration': 30.0
            })

        return subtasks

    def _analyze_dependencies(self, subtasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze dependencies between subtasks."""
        dependencies = {}

        # Create dependency mapping
        context_tasks = [t['id'] for t in subtasks if t['agent_type'] == 'context']
        code_tasks = [t['id'] for t in subtasks if t['agent_type'] == 'code']
        git_tasks = [t['id'] for t in subtasks if t['agent_type'] == 'git']

        # Code tasks depend on context
        for code_task_id in code_tasks:
            dependencies[code_task_id] = context_tasks.copy()

        # Git tasks depend on code tasks (if any)
        for git_task_id in git_tasks:
            if code_tasks:
                dependencies[git_task_id] = code_tasks.copy()
            else:
                dependencies[git_task_id] = context_tasks.copy()

        return dependencies

    def _estimate_duration(self, subtasks: List[Dict[str, Any]], complexity: RequestComplexity) -> float:
        """Estimate the total duration for the workflow."""
        base_duration = sum(task.get('estimated_duration', 60.0) for task in subtasks)

        # Apply complexity multiplier
        complexity_multipliers = {
            RequestComplexity.SIMPLE: 1.0,
            RequestComplexity.MODERATE: 1.3,
            RequestComplexity.COMPLEX: 1.7,
            RequestComplexity.CRITICAL: 2.2
        }

        return base_duration * complexity_multipliers[complexity]

    def _calculate_confidence(self, request: str, subtasks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the analysis."""
        # Base confidence on request clarity
        word_count = len(request.split())
        clarity_score = min(1.0, word_count / 20)  # Normalize to 1.0

        # Adjust based on subtask count (more subtasks = lower confidence)
        subtask_penalty = max(0.0, (len(subtasks) - 3) * 0.1)

        confidence = max(0.1, clarity_score - subtask_penalty)
        return round(confidence, 2)

    def _determine_context_requirements(self, request_type: RequestType, subtasks: List[Dict[str, Any]]) -> List[str]:
        """Determine what context information is needed."""
        requirements = ['project_structure', 'recent_changes']

        if request_type in [RequestType.CODE_GENERATION, RequestType.CODE_MODIFICATION]:
            requirements.extend(['existing_code', 'dependencies', 'coding_standards'])

        if request_type == RequestType.GIT_OPERATION:
            requirements.extend(['git_status', 'branch_info', 'commit_history'])

        return requirements

    async def _create_workflow_plan(self, analysis: RequestAnalysis, priority: TaskPriority) -> Dict[str, Any]:
        """Create a detailed execution plan for the workflow."""
        return {
            'workflow_id': str(uuid.uuid4()),
            'analysis': analysis,
            'priority': priority,
            'steps': [
                WorkflowStep(
                    step_id=subtask.id,
                    agent_type=subtask.agent_type,
                    task_data=subtask,
                    dependencies=analysis.dependencies.get(subtask.id, [])
                )
                for subtask in analysis.subtasks
            ],
            'created_at': datetime.utcnow(),
            'timeout': (analysis.estimated_duration or 300.0) + 60.0  # Add buffer
        }

    async def _execute_workflow(self, request_id: str, workflow_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow plan."""
        workflow_id = workflow_plan['workflow_id']

        async with self.workflow_lock:
            self.active_workflows[workflow_id] = {
                'request_id': request_id,
                'plan': workflow_plan,
                'status': 'running',
                'results': {},
                'started_at': datetime.utcnow()
            }

        try:
            # Execute steps in dependency order
            completed_steps = set()
            results = {}

            while len(completed_steps) < len(workflow_plan['steps']):
                ready_steps = [
                    step for step in workflow_plan['steps']
                    if step.step_id not in completed_steps and
                    all(dep in completed_steps for dep in step.dependencies)
                ]

                if not ready_steps:
                    raise Exception("Workflow deadlock detected - no ready steps")

                # Execute ready steps in parallel
                tasks = []
                for step in ready_steps:
                    task = asyncio.create_task(self._execute_step(step, results))
                    tasks.append((step.step_id, task))

                # Wait for all tasks to complete
                for step_id, task in tasks:
                    try:
                        step_result = await task
                        results[step_id] = step_result
                        completed_steps.add(step_id)
                    except Exception as e:
                        self.logger.error(f"Step {step_id} failed: {e}")
                        results[step_id] = {'error': str(e)}
                        completed_steps.add(step_id)  # Mark as completed even if failed

            # Update workflow status
            async with self.workflow_lock:
                if workflow_id in self.active_workflows:
                    self.active_workflows[workflow_id]['status'] = 'completed'
                    self.active_workflows[workflow_id]['results'] = results
                    self.active_workflows[workflow_id]['completed_at'] = datetime.utcnow()

            return results

        except Exception as e:
            async with self.workflow_lock:
                if workflow_id in self.active_workflows:
                    self.active_workflows[workflow_id]['status'] = 'failed'
                    self.active_workflows[workflow_id]['error'] = str(e)
            raise

    async def _execute_step(self, step: WorkflowStep, context_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step."""
        step.started_at = datetime.utcnow()
        step.status = TaskState.RUNNING

        try:
            # Prepare message for the target agent
            message_data = {
                'action': step.task_data['action'],
                'task_data': step.task_data,
                'context': context_results,
                'request_id': step.step_id
            }

            message = AgentMessage(
                sender_id=self.agent_id,
                recipient_id=step.agent_type,
                message_type='task_request',
                payload=message_data,
                priority=MessagePriority.NORMAL,
                correlation_id=step.step_id
            )

            # Send message and wait for response
            response = await self.request_response(
                message,
                timeout=step.task_data.get('estimated_duration', 120.0)
            )

            step.status = TaskState.COMPLETED
            step.completed_at = datetime.utcnow()
            step.result = response.data if response else {'error': 'No response received'}

            return step.result

        except Exception as e:
            step.status = TaskState.FAILED
            step.error = str(e)
            step.completed_at = datetime.utcnow()
            raise

    async def _synthesize_response(self, request_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize the final response from all step results."""
        # Extract successful results
        successful_results = {
            step_id: result for step_id, result in results.items()
            if 'error' not in result
        }

        # Extract errors
        errors = {
            step_id: result['error'] for step_id, result in results.items()
            if 'error' in result
        }

        # Create synthesized response
        synthesis = {
            'summary': f"Workflow completed with {len(successful_results)} successful steps",
            'successful_steps': len(successful_results),
            'failed_steps': len(errors),
            'results': successful_results,
            'errors': errors if errors else None,
            'recommendations': self._generate_recommendations(results)
        }

        return synthesis

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on workflow results."""
        recommendations = []

        # Check for errors and suggest remediation
        errors = [r for r in results.values() if 'error' in r]
        if errors:
            recommendations.append("Some steps failed - review errors and retry if needed")

        # Check for code generation results
        code_results = [r for r in results.values() if 'generated_code' in r]
        if code_results:
            recommendations.append("Review generated code before committing")
            recommendations.append("Run tests to verify code functionality")

        # Check for git operations
        git_results = [r for r in results.values() if 'git_operation' in r]
        if git_results:
            recommendations.append("Verify git operations completed successfully")

        return recommendations

    def _update_average_duration(self, duration: float):
        """Update the running average duration metric."""
        current_avg = self.workflow_metrics['average_duration']
        total_requests = self.workflow_metrics['total_requests']

        if total_requests > 1:
            new_avg = ((current_avg * (total_requests - 1)) + duration) / total_requests
            self.workflow_metrics['average_duration'] = new_avg
        else:
            self.workflow_metrics['average_duration'] = duration

    async def _handle_user_request(self, message: AgentMessage) -> None:
        """Handle incoming user requests."""
        try:
            request_data = message.payload
            response = await self.process_user_request(
                request=request_data.get('request', ''),
                context=request_data.get('context'),
                user_id=request_data.get('user_id'),
                priority=TaskPriority(request_data.get('priority', 'normal'))
            )

            # Send response back
            response_message = AgentMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type='user_response',
                payload=response,
                correlation_id=message.correlation_id
            )

            await self.send_message(response_message)

        except Exception as e:
            self.logger.error(f"Error handling user request: {e}", exc_info=True)

    async def _handle_agent_response(self, message: AgentMessage) -> None:
        """Handle responses from other agents."""
        # This is handled by the request_response mechanism
        pass

    async def _handle_workflow_status(self, message: AgentMessage) -> None:
        """Handle workflow status updates."""
        try:
            status_data = message.payload
            workflow_id = status_data.get('workflow_id')

            if workflow_id in self.active_workflows:
                async with self.workflow_lock:
                    self.active_workflows[workflow_id].update(status_data)

        except Exception as e:
            self.logger.error(f"Error handling workflow status: {e}", exc_info=True)

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a workflow."""
        return self.active_workflows.get(workflow_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics."""
        return {
            **self.workflow_metrics,
            'active_workflows': len(self.active_workflows),
            'agent_status': self.get_status()
        }

    # Abstract method implementations
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message and route it to appropriate handlers.

        Args:
            message: The message to process

        Returns:
            Optional response message
        """
        try:
            self.logger.debug(f"Processing message: {message.message_type} from {message.sender_id}")

            # Handle different message types
            if message.message_type == "user_request":
                # Process user request and return response
                response_data = await self.process_user_request(
                    message.payload.get('request_text', ''),
                    message.payload.get('context', {}),
                    message.correlation_id
                )

                return AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="user_response",
                    payload=response_data,
                    correlation_id=message.correlation_id
                )

            elif message.message_type == "workflow_status":
                await self._handle_workflow_status(message)
                return None

            elif message.message_type == "agent_response":
                await self._handle_agent_response(message)
                return None

            else:
                # Return None for unhandled message types to avoid recursion
                self.logger.debug(f"Unhandled message type: {message.message_type}")
                return None

        except Exception as e:
            self.logger.error(f"Error processing message: {e}", exc_info=True)

            # Return error response if this was a request expecting a response
            if message.message_type in ["user_request"]:
                return AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="error_response",
                    payload={
                        'error': str(e),
                        'error_type': type(e).__name__
                    },
                    correlation_id=message.correlation_id
                )

            return None

    def get_capabilities(self) -> List[AgentCapability]:
        """
        Return list of capabilities this orchestrator provides.

        Returns:
            List of agent capabilities
        """


        return [
            AgentCapability(
                name="request_orchestration",
                description="Analyze and orchestrate complex user requests across multiple agents",
                input_types=["text", "dict"],
                output_types=["dict", "text"],
                cost_estimate=2.0
            ),
            AgentCapability(
                name="workflow_management",
                description="Manage multi-agent workflows and task coordination",
                input_types=["dict"],
                output_types=["dict"],
                cost_estimate=1.5
            ),
            AgentCapability(
                name="agent_coordination",
                description="Coordinate communication and task delegation between agents",
                input_types=["dict"],
                output_types=["dict"],
                cost_estimate=1.0
            ),
            AgentCapability(
                name="request_analysis",
                description="Analyze and decompose complex requests into actionable subtasks",
                input_types=["text"],
                output_types=["dict"],
                cost_estimate=1.0
            ),
            AgentCapability(
                name="result_synthesis",
                description="Synthesize results from multiple agents into coherent responses",
                input_types=["dict"],
                output_types=["text", "dict"],
                cost_estimate=1.5
            )
        ]

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return status.

        Returns:
            Dictionary containing health status information
        """
        try:
            # Initialize base health status
            base_health = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "state": self.state.value if hasattr(self, 'state') else "unknown"
            }

            # Check orchestrator-specific health
            current_time = datetime.utcnow()

            # Check workflow management
            active_workflow_count = len(self.active_workflows)
            workflow_health = "healthy" if active_workflow_count <= self.max_concurrent_workflows else "overloaded"

            # Check agent availability
            available_agent_types = list(self.available_agents.keys())
            agent_availability = len(available_agent_types) > 0

            # Calculate success rate
            total_requests = self.workflow_metrics['total_requests']
            success_rate = 0.0
            if total_requests > 0:
                success_rate = (self.workflow_metrics['successful_workflows'] / total_requests) * 100

            # Determine overall health status
            is_healthy = (
                workflow_health == "healthy" and
                agent_availability and
                success_rate >= 70.0  # Consider healthy if success rate >= 70%
            )

            health_status = {
                **base_health,
                "status": "healthy" if is_healthy else "degraded",
                "timestamp": current_time.isoformat(),
                "orchestrator_specific": {
                    "active_workflows": active_workflow_count,
                    "max_workflows": self.max_concurrent_workflows,
                    "workflow_health": workflow_health,
                    "available_agents": available_agent_types,
                    "agent_availability": agent_availability,
                    "success_rate": success_rate,
                    "total_requests": total_requests,
                    "average_duration": self.workflow_metrics['average_duration']
                },
                "capabilities": len(self.get_capabilities()),
                "uptime": (current_time - self.created_at).total_seconds() if hasattr(self, 'created_at') else 0
            }

            # Add any critical issues
            issues = []
            if not agent_availability:
                issues.append("No agents available for delegation")
            if workflow_health == "overloaded":
                issues.append(f"Too many active workflows: {active_workflow_count}/{self.max_concurrent_workflows}")
            if success_rate < 70.0 and total_requests > 0:
                issues.append(f"Low success rate: {success_rate:.1f}%")

            if issues:
                health_status["issues"] = issues

            return health_status

        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "error_type": type(e).__name__
            }
