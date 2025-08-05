"""
TaskRouter: Intelligent task classification and routing system for the Aider Hive Architecture.

This module provides sophisticated task routing capabilities including:
- Intelligent task classification based on content and context
- Dynamic routing rules and strategies
- Agent capability matching and load balancing
- Performance-based routing optimization
- Machine learning-enhanced routing decisions
- Rule-based routing with priority and fallback mechanisms
"""

import asyncio
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Tuple, Union
from weakref import WeakSet

import structlog

from .task_queue import Task, TaskType, TaskPriority


class RoutingStrategy(Enum):
    """Task routing strategies."""
    CAPABILITY_MATCH = "capability_match"
    LOAD_BALANCED = "load_balanced"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    RULE_BASED = "rule_based"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"


class ClassificationConfidence(Enum):
    """Confidence levels for task classification."""
    VERY_HIGH = 0.9
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2


@dataclass
class RoutingRule:
    """Represents a routing rule for task assignment."""
    id: str
    name: str
    description: str
    condition: Callable[[Task], bool]
    target_agent_type: str
    priority: int = 100  # Lower number = higher priority
    enabled: bool = True
    success_rate: float = 0.0
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ClassificationResult:
    """Result of task classification."""
    task_type: TaskType
    confidence: float
    suggested_agent_type: str
    required_capabilities: List[str]
    estimated_complexity: float
    estimated_execution_time: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Represents a routing decision for a task."""
    task_id: str
    target_agent_type: str
    reasoning: str
    confidence: float
    alternative_agents: List[str] = field(default_factory=list)
    estimated_wait_time: float = 0.0
    routing_strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_MATCH
    applied_rules: List[str] = field(default_factory=list)


@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for agent types."""
    agent_type: str
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_execution_time: float = 0.0
    average_queue_time: float = 0.0
    current_load: float = 0.0
    success_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class TaskClassifier:
    """Intelligent task classifier using multiple techniques."""

    def __init__(self):
        # Pattern-based classification rules
        self.classification_patterns = {
            TaskType.CODE_GENERATION: [
                r'generate.*code',
                r'create.*function',
                r'write.*class',
                r'implement.*method',
                r'build.*module',
                r'code.*generation',
            ],
            TaskType.CODE_ANALYSIS: [
                r'analyze.*code',
                r'review.*code',
                r'check.*syntax',
                r'validate.*code',
                r'inspect.*function',
                r'audit.*code',
            ],
            TaskType.CONTEXT_BUILDING: [
                r'build.*context',
                r'gather.*information',
                r'collect.*data',
                r'index.*files',
                r'map.*codebase',
                r'understand.*project',
            ],
            TaskType.GIT_OPERATION: [
                r'git.*commit',
                r'git.*push',
                r'git.*merge',
                r'version.*control',
                r'commit.*changes',
                r'branch.*operation',
            ],
            TaskType.FILE_OPERATION: [
                r'read.*file',
                r'write.*file',
                r'create.*file',
                r'delete.*file',
                r'move.*file',
                r'copy.*file',
            ],
            TaskType.SEARCH_OPERATION: [
                r'search.*code',
                r'find.*function',
                r'locate.*file',
                r'grep.*pattern',
                r'query.*codebase',
                r'discover.*symbol',
            ],
        }

        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for task_type, patterns in self.classification_patterns.items():
            self.compiled_patterns[task_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        # Classification history for learning
        self.classification_history = deque(maxlen=1000)

    def classify_task(self, task: Task) -> ClassificationResult:
        """
        Classify a task and determine its type and requirements.

        Args:
            task: Task to classify

        Returns:
            Classification result with confidence and recommendations
        """
        # Extract text for analysis
        text_content = self._extract_text_content(task)

        # Pattern-based classification
        pattern_result = self._classify_by_patterns(text_content)

        # Context-based classification
        context_result = self._classify_by_context(task)

        # Payload-based classification
        payload_result = self._classify_by_payload(task)

        # Combine results and determine best classification
        combined_result = self._combine_classifications([
            pattern_result,
            context_result,
            payload_result,
        ])

        # Enhance with capability recommendations
        self._enhance_with_capabilities(combined_result, task)

        # Add to history for learning
        self.classification_history.append({
            'task_id': task.id,
            'classification': combined_result,
            'timestamp': datetime.utcnow(),
        })

        return combined_result

    def _extract_text_content(self, task: Task) -> str:
        """Extract textual content from task for analysis."""
        content_parts = []

        # Add payload text
        if isinstance(task.payload, dict):
            for key, value in task.payload.items():
                if isinstance(value, str):
                    content_parts.append(f"{key}: {value}")

        # Add metadata text
        for key, value in task.metadata.items():
            if isinstance(value, str):
                content_parts.append(f"{key}: {value}")

        # Add context text
        for key, value in task.context.items():
            if isinstance(value, str):
                content_parts.append(f"{key}: {value}")

        return " ".join(content_parts)

    def _classify_by_patterns(self, text: str) -> ClassificationResult:
        """Classify task using regex patterns."""
        best_match = None
        best_score = 0.0
        best_task_type = TaskType.CODE_GENERATION

        for task_type, patterns in self.compiled_patterns.items():
            score = 0.0
            matches = 0

            for pattern in patterns:
                if pattern.search(text):
                    matches += 1
                    score += 1.0 / len(patterns)

            if matches > 0:
                confidence = min(score, 1.0)
                if confidence > best_score:
                    best_score = confidence
                    best_task_type = task_type
                    best_match = f"Pattern match: {matches}/{len(patterns)} patterns"

        return ClassificationResult(
            task_type=best_task_type,
            confidence=best_score,
            suggested_agent_type=self._task_type_to_agent_type(best_task_type),
            required_capabilities=self._get_default_capabilities(best_task_type),
            estimated_complexity=self._estimate_complexity(text),
            estimated_execution_time=self._estimate_execution_time(best_task_type, text),
            reasoning=best_match or "No pattern matches found",
        )

    def _classify_by_context(self, task: Task) -> ClassificationResult:
        """Classify task using context information."""
        # Analyze task context for clues
        confidence = 0.5
        task_type = TaskType.CODE_GENERATION
        reasoning = "Default context classification"

        # Check for explicit task type hints
        if 'task_type' in task.context:
            explicit_type = task.context['task_type']
            if isinstance(explicit_type, str):
                try:
                    task_type = TaskType(explicit_type)
                    confidence = 0.9
                    reasoning = "Explicit task type specified in context"
                except ValueError:
                    pass

        # Check for file-related context
        if any(key in task.context for key in ['file_path', 'file_content', 'files']):
            task_type = TaskType.FILE_OPERATION
            confidence = 0.7
            reasoning = "File-related context detected"

        # Check for git-related context
        if any(key in task.context for key in ['git_branch', 'commit_message', 'repository']):
            task_type = TaskType.GIT_OPERATION
            confidence = 0.8
            reasoning = "Git-related context detected"

        # Check for search-related context
        if any(key in task.context for key in ['search_query', 'pattern', 'search_scope']):
            task_type = TaskType.SEARCH_OPERATION
            confidence = 0.7
            reasoning = "Search-related context detected"

        return ClassificationResult(
            task_type=task_type,
            confidence=confidence,
            suggested_agent_type=self._task_type_to_agent_type(task_type),
            required_capabilities=self._get_default_capabilities(task_type),
            estimated_complexity=0.5,
            estimated_execution_time=self._estimate_execution_time(task_type, ""),
            reasoning=reasoning,
        )

    def _classify_by_payload(self, task: Task) -> ClassificationResult:
        """Classify task using payload structure and content."""
        confidence = 0.3
        task_type = TaskType.CODE_GENERATION
        reasoning = "Default payload classification"

        # Analyze payload structure
        if isinstance(task.payload, dict):
            # Check for code-related keys
            code_keys = ['code', 'source_code', 'function', 'class', 'method']
            if any(key in task.payload for key in code_keys):
                if any(action in str(task.payload).lower() for action in ['generate', 'create', 'write']):
                    task_type = TaskType.CODE_GENERATION
                    confidence = 0.6
                    reasoning = "Code generation payload detected"
                else:
                    task_type = TaskType.CODE_ANALYSIS
                    confidence = 0.6
                    reasoning = "Code analysis payload detected"

            # Check for file operation keys
            file_keys = ['file_path', 'filename', 'content', 'data']
            if any(key in task.payload for key in file_keys):
                task_type = TaskType.FILE_OPERATION
                confidence = 0.7
                reasoning = "File operation payload detected"

            # Check for git operation keys
            git_keys = ['commit', 'branch', 'merge', 'diff']
            if any(key in task.payload for key in git_keys):
                task_type = TaskType.GIT_OPERATION
                confidence = 0.8
                reasoning = "Git operation payload detected"

        return ClassificationResult(
            task_type=task_type,
            confidence=confidence,
            suggested_agent_type=self._task_type_to_agent_type(task_type),
            required_capabilities=self._get_default_capabilities(task_type),
            estimated_complexity=self._estimate_payload_complexity(task.payload),
            estimated_execution_time=self._estimate_execution_time(task_type, str(task.payload)),
            reasoning=reasoning,
        )

    def _combine_classifications(self, results: List[ClassificationResult]) -> ClassificationResult:
        """Combine multiple classification results into a single best result."""
        if not results:
            return ClassificationResult(
                task_type=TaskType.CODE_GENERATION,
                confidence=0.1,
                suggested_agent_type="code",
                required_capabilities=[],
                estimated_complexity=0.5,
                estimated_execution_time=30.0,
                reasoning="No classification results available",
            )

        # Find result with highest confidence
        best_result = max(results, key=lambda r: r.confidence)

        # Combine capabilities from all results
        all_capabilities = set()
        for result in results:
            all_capabilities.update(result.required_capabilities)

        # Average complexity and execution time estimates
        avg_complexity = sum(r.estimated_complexity for r in results) / len(results)
        avg_execution_time = sum(r.estimated_execution_time for r in results) / len(results)

        # Combine reasoning
        reasoning_parts = [f"{r.task_type.value}: {r.reasoning} (conf: {r.confidence:.2f})" for r in results]
        combined_reasoning = f"Best: {best_result.reasoning}; All: {'; '.join(reasoning_parts)}"

        return ClassificationResult(
            task_type=best_result.task_type,
            confidence=best_result.confidence,
            suggested_agent_type=best_result.suggested_agent_type,
            required_capabilities=list(all_capabilities),
            estimated_complexity=avg_complexity,
            estimated_execution_time=avg_execution_time,
            reasoning=combined_reasoning,
        )

    def _enhance_with_capabilities(self, result: ClassificationResult, task: Task) -> None:
        """Enhance classification with additional capability requirements."""
        # Add capabilities based on task complexity
        if result.estimated_complexity > 0.7:
            result.required_capabilities.append("advanced_analysis")

        # Add capabilities based on payload content
        if isinstance(task.payload, dict):
            if 'test' in str(task.payload).lower():
                result.required_capabilities.append("testing")
            if 'optimize' in str(task.payload).lower():
                result.required_capabilities.append("optimization")
            if 'debug' in str(task.payload).lower():
                result.required_capabilities.append("debugging")

        # Remove duplicates
        result.required_capabilities = list(set(result.required_capabilities))

    def _task_type_to_agent_type(self, task_type: TaskType) -> str:
        """Map task type to recommended agent type."""
        mapping = {
            TaskType.CODE_GENERATION: "code",
            TaskType.CODE_ANALYSIS: "code",
            TaskType.CONTEXT_BUILDING: "context",
            TaskType.GIT_OPERATION: "git",
            TaskType.FILE_OPERATION: "code",
            TaskType.SEARCH_OPERATION: "context",
            TaskType.VALIDATION: "code",
            TaskType.OPTIMIZATION: "code",
            TaskType.MONITORING: "orchestrator",
            TaskType.MAINTENANCE: "orchestrator",
        }
        return mapping.get(task_type, "code")

    def _get_default_capabilities(self, task_type: TaskType) -> List[str]:
        """Get default capabilities required for a task type."""
        capabilities = {
            TaskType.CODE_GENERATION: ["code_generation", "syntax_validation"],
            TaskType.CODE_ANALYSIS: ["code_analysis", "pattern_recognition"],
            TaskType.CONTEXT_BUILDING: ["context_analysis", "semantic_understanding"],
            TaskType.GIT_OPERATION: ["git_operations", "version_control"],
            TaskType.FILE_OPERATION: ["file_operations", "io_operations"],
            TaskType.SEARCH_OPERATION: ["search", "pattern_matching"],
            TaskType.VALIDATION: ["validation", "testing"],
            TaskType.OPTIMIZATION: ["optimization", "performance_analysis"],
            TaskType.MONITORING: ["monitoring", "metrics_collection"],
            TaskType.MAINTENANCE: ["maintenance", "cleanup"],
        }
        return capabilities.get(task_type, ["general"])

    def _estimate_complexity(self, text: str) -> float:
        """Estimate task complexity based on text content."""
        # Simple heuristic based on text length and keywords
        base_complexity = min(len(text) / 1000.0, 1.0)

        # Adjust for complexity keywords
        complex_keywords = ['complex', 'advanced', 'sophisticated', 'comprehensive', 'detailed']
        simple_keywords = ['simple', 'basic', 'easy', 'quick', 'minimal']

        for keyword in complex_keywords:
            if keyword in text.lower():
                base_complexity += 0.2

        for keyword in simple_keywords:
            if keyword in text.lower():
                base_complexity -= 0.1

        return max(0.1, min(base_complexity, 1.0))

    def _estimate_execution_time(self, task_type: TaskType, text: str) -> float:
        """Estimate execution time based on task type and content."""
        # Base time estimates (in seconds)
        base_times = {
            TaskType.CODE_GENERATION: 60.0,
            TaskType.CODE_ANALYSIS: 30.0,
            TaskType.CONTEXT_BUILDING: 45.0,
            TaskType.GIT_OPERATION: 15.0,
            TaskType.FILE_OPERATION: 10.0,
            TaskType.SEARCH_OPERATION: 20.0,
            TaskType.VALIDATION: 25.0,
            TaskType.OPTIMIZATION: 90.0,
            TaskType.MONITORING: 5.0,
            TaskType.MAINTENANCE: 30.0,
        }

        base_time = base_times.get(task_type, 30.0)

        # Adjust based on content length
        length_factor = min(len(text) / 500.0, 3.0)
        return base_time * (1.0 + length_factor)

    def _estimate_payload_complexity(self, payload: Any) -> float:
        """Estimate complexity based on payload structure."""
        if isinstance(payload, dict):
            # More keys = more complex
            key_complexity = min(len(payload) / 10.0, 0.5)

            # Nested structures = more complex
            nested_complexity = 0.0
            for value in payload.values():
                if isinstance(value, (dict, list)):
                    nested_complexity += 0.1

            return min(key_complexity + nested_complexity, 1.0)

        elif isinstance(payload, list):
            return min(len(payload) / 20.0, 0.8)

        return 0.3


class TaskRouter:
    """
    Intelligent task routing system for the Aider Hive Architecture.

    Features:
    - Task classification and analysis
    - Dynamic routing rules and strategies
    - Agent capability matching
    - Performance-based optimization
    - Load balancing and queue management
    - Machine learning-enhanced decisions
    """

    def __init__(
        self,
        default_strategy: RoutingStrategy = RoutingStrategy.HYBRID,
        enable_learning: bool = True,
        performance_weight: float = 0.3,
        load_weight: float = 0.4,
        capability_weight: float = 0.3,
    ):
        """
        Initialize the task router.

        Args:
            default_strategy: Default routing strategy
            enable_learning: Enable machine learning optimization
            performance_weight: Weight for performance-based routing
            load_weight: Weight for load-based routing
            capability_weight: Weight for capability-based routing
        """
        self.default_strategy = default_strategy
        self.enable_learning = enable_learning
        self.performance_weight = performance_weight
        self.load_weight = load_weight
        self.capability_weight = capability_weight

        # Core components
        self.classifier = TaskClassifier()

        # Logging
        self.logger = structlog.get_logger().bind(component="task_router")

        # Routing rules
        self.routing_rules: Dict[str, RoutingRule] = {}
        self.rule_performance: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Agent performance tracking
        self.agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        self.performance_history: deque = deque(maxlen=1000)

        # Routing history and statistics
        self.routing_history: deque = deque(maxlen=1000)
        self.routing_stats: Dict[str, int] = defaultdict(int)

        # Agent pool reference (set externally)
        self.agent_pool = None

    async def initialize(self) -> None:
        """Initialize the task router components."""
        try:
            self.logger.info("Initializing task router")

            # Initialize default routing rules
            self._initialize_default_rules()

            # Initialize classifier
            await self.classifier.initialize()

            # Validate configuration
            self._validate_config()

            self.logger.info("Task router initialized successfully")

        except Exception as e:
            self.logger.error("Failed to initialize task router", error=str(e))
            raise

    async def start(self) -> None:
        """Start the task router."""
        try:
            self.logger.info("Starting task router")

            # Start classifier if needed
            if hasattr(self.classifier, 'start'):
                await self.classifier.start()

            self.logger.info("Task router started successfully")

        except Exception as e:
            self.logger.error("Failed to start task router", error=str(e))
            raise

    async def stop(self) -> None:
        """Stop the task router gracefully."""
        try:
            self.logger.info("Stopping task router")

            # Stop classifier if needed
            if hasattr(self.classifier, 'stop'):
                await self.classifier.stop()

            self.logger.info("Task router stopped")

        except Exception as e:
            self.logger.error("Failed to stop task router", error=str(e))
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check of the task router."""
        try:
            health_status = {
                "status": "healthy",
                "routing_rules_count": len(self.routing_rules),
                "agent_metrics_count": len(self.agent_metrics),
                "routing_history_size": len(self.routing_history),
                "performance_history_size": len(self.performance_history),
                "classifier_status": "unknown",
                "agent_pool_connected": self.agent_pool is not None,
                "errors": []
            }

            # Check classifier health
            if hasattr(self.classifier, 'health_check'):
                classifier_health = await self.classifier.health_check()
                health_status["classifier_status"] = classifier_health.get("status", "unknown")
                if classifier_health.get("status") != "healthy":
                    health_status["errors"].append(f"Classifier unhealthy: {classifier_health.get('error', 'unknown')}")

            # Check agent pool connection
            if not self.agent_pool:
                health_status["errors"].append("Agent pool not connected")
                health_status["status"] = "degraded"

            # Overall health determination
            if health_status["errors"]:
                health_status["status"] = "degraded" if health_status["status"] == "healthy" else "unhealthy"

            return health_status

        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    def _initialize_default_rules(self) -> None:
        """Initialize default routing rules."""
        # Add basic routing rules
        self.logger.debug("Default routing rules initialized")

    def _validate_config(self) -> None:
        """Validate task router configuration."""
        if not (0 <= self.performance_weight <= 1):
            raise ValueError("performance_weight must be between 0 and 1")
        if not (0 <= self.load_weight <= 1):
            raise ValueError("load_weight must be between 0 and 1")
        if not (0 <= self.capability_weight <= 1):
            raise ValueError("capability_weight must be between 0 and 1")

        total_weight = self.performance_weight + self.load_weight + self.capability_weight
        if not (0.9 <= total_weight <= 1.1):  # Allow small floating point errors
            raise ValueError("Sum of routing weights must equal 1.0")

        # Initialize default rules
        self._initialize_default_rules()

    def route_task(self, task: Task) -> RoutingDecision:
        """
        Route a task to the most appropriate agent.

        Args:
            task: Task to route

        Returns:
            Routing decision with target agent and reasoning
        """
        start_time = time.time()

        try:
            # Classify the task
            classification = self.classifier.classify_task(task)

            # Apply routing strategy
            if self.default_strategy == RoutingStrategy.RULE_BASED:
                decision = self._route_by_rules(task, classification)
            elif self.default_strategy == RoutingStrategy.CAPABILITY_MATCH:
                decision = self._route_by_capabilities(task, classification)
            elif self.default_strategy == RoutingStrategy.LOAD_BALANCED:
                decision = self._route_by_load(task, classification)
            elif self.default_strategy == RoutingStrategy.PERFORMANCE_OPTIMIZED:
                decision = self._route_by_performance(task, classification)
            elif self.default_strategy == RoutingStrategy.MACHINE_LEARNING:
                decision = self._route_by_ml(task, classification)
            else:  # HYBRID
                decision = self._route_hybrid(task, classification)

            # Record routing decision
            routing_time = time.time() - start_time
            self._record_routing_decision(task, decision, classification, routing_time)

            self.logger.info(
                "Task routed",
                task_id=task.id,
                target_agent=decision.target_agent_type,
                strategy=decision.routing_strategy.value,
                confidence=decision.confidence,
                routing_time=routing_time,
            )

            return decision

        except Exception as e:
            self.logger.error("Error routing task", task_id=task.id, error=str(e))
            # Fallback routing
            return RoutingDecision(
                task_id=task.id,
                target_agent_type="code",  # Safe default
                reasoning=f"Fallback routing due to error: {str(e)}",
                confidence=0.1,
                routing_strategy=RoutingStrategy.RULE_BASED,
            )

    def classify_task(self, task: Task) -> ClassificationResult:
        """Classify a task and return detailed analysis."""
        return self.classifier.classify_task(task)

    def add_routing_rule(
        self,
        rule_id: str,
        name: str,
        description: str,
        condition: Callable[[Task], bool],
        target_agent_type: str,
        priority: int = 100,
    ) -> None:
        """Add a new routing rule."""
        rule = RoutingRule(
            id=rule_id,
            name=name,
            description=description,
            condition=condition,
            target_agent_type=target_agent_type,
            priority=priority,
        )

        self.routing_rules[rule_id] = rule

        self.logger.info(
            "Routing rule added",
            rule_id=rule_id,
            name=name,
            target_agent=target_agent_type,
        )

    def remove_routing_rule(self, rule_id: str) -> bool:
        """Remove a routing rule."""
        if rule_id in self.routing_rules:
            del self.routing_rules[rule_id]
            self.logger.info("Routing rule removed", rule_id=rule_id)
            return True
        return False

    def update_agent_performance(
        self,
        agent_type: str,
        task_success: bool,
        execution_time: float,
        queue_time: float = 0.0,
    ) -> None:
        """Update performance metrics for an agent type."""
        if agent_type not in self.agent_metrics:
            self.agent_metrics[agent_type] = AgentPerformanceMetrics(agent_type=agent_type)

        metrics = self.agent_metrics[agent_type]
        metrics.total_tasks += 1

        if task_success:
            metrics.successful_tasks += 1
        else:
            metrics.failed_tasks += 1

        # Update averages
        metrics.success_rate = metrics.successful_tasks / metrics.total_tasks
        metrics.average_execution_time = (
            metrics.average_execution_time + execution_time
        ) / 2
        metrics.average_queue_time = (
            metrics.average_queue_time + queue_time
        ) / 2
        metrics.last_updated = datetime.utcnow()

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        total_routings = len(self.routing_history)

        strategy_stats = defaultdict(int)
        agent_stats = defaultdict(int)
        confidence_sum = 0.0

        for entry in self.routing_history:
            decision = entry['decision']
            strategy_stats[decision.routing_strategy.value] += 1
            agent_stats[decision.target_agent_type] += 1
            confidence_sum += decision.confidence

        return {
            'total_routings': total_routings,
            'average_confidence': confidence_sum / max(1, total_routings),
            'strategy_distribution': dict(strategy_stats),
            'agent_distribution': dict(agent_stats),
            'agent_performance': {
                agent_type: {
                    'total_tasks': metrics.total_tasks,
                    'success_rate': metrics.success_rate,
                    'average_execution_time': metrics.average_execution_time,
                    'average_queue_time': metrics.average_queue_time,
                }
                for agent_type, metrics in self.agent_metrics.items()
            },
            'rule_performance': dict(self.rule_performance),
        }

    def _route_by_rules(self, task: Task, classification: ClassificationResult) -> RoutingDecision:
        """Route task using predefined rules."""
        # Sort rules by priority
        sorted_rules = sorted(
            [rule for rule in self.routing_rules.values() if rule.enabled],
            key=lambda r: r.priority
        )

        applied_rules = []
        for rule in sorted_rules:
            try:
                if rule.condition(task):
                    applied_rules.append(rule.id)
                    rule.usage_count += 1

                    return RoutingDecision(
                        task_id=task.id,
                        target_agent_type=rule.target_agent_type,
                        reasoning=f"Matched rule '{rule.name}': {rule.description}",
                        confidence=0.8,
                        routing_strategy=RoutingStrategy.RULE_BASED,
                        applied_rules=applied_rules,
                    )
            except Exception as e:
                self.logger.warning("Error evaluating rule", rule_id=rule.id, error=str(e))

        # Fallback to classification suggestion
        return RoutingDecision(
            task_id=task.id,
            target_agent_type=classification.suggested_agent_type,
            reasoning="No matching rules, using classification suggestion",
            confidence=classification.confidence * 0.7,
            routing_strategy=RoutingStrategy.RULE_BASED,
            applied_rules=applied_rules,
        )

    def _route_by_capabilities(self, task: Task, classification: ClassificationResult) -> RoutingDecision:
        """Route task based on agent capabilities."""
        if not self.agent_pool:
            return self._fallback_routing(task, classification, "No agent pool available")

        # Get available agent types
        available_agents = self.agent_pool.registered_types.keys()

        # Score agents based on capability match
        agent_scores = {}
        for agent_type in available_agents:
            score = self._calculate_capability_match_score(
                agent_type, classification.required_capabilities
            )
            if score > 0:
                agent_scores[agent_type] = score

        if not agent_scores:
            return self._fallback_routing(task, classification, "No capable agents found")

        # Select best agent
        best_agent = max(agent_scores, key=agent_scores.get)
        confidence = agent_scores[best_agent]

        return RoutingDecision(
            task_id=task.id,
            target_agent_type=best_agent,
            reasoning=f"Best capability match (score: {confidence:.2f})",
            confidence=confidence,
            routing_strategy=RoutingStrategy.CAPABILITY_MATCH,
            alternative_agents=list(agent_scores.keys()),
        )

    def _route_by_load(self, task: Task, classification: ClassificationResult) -> RoutingDecision:
        """Route task based on agent load balancing."""
        if not self.agent_pool:
            return self._fallback_routing(task, classification, "No agent pool available")

        # Get agent load information
        pool_status = self.agent_pool.get_pool_status()
        agent_loads = {}

        for agent_type, stats in pool_status.get('agent_types', {}).items():
            if stats['healthy_instances'] > 0:
                load = stats['average_load']
                agent_loads[agent_type] = load

        if not agent_loads:
            return self._fallback_routing(task, classification, "No available agents")

        # Select least loaded agent
        best_agent = min(agent_loads, key=agent_loads.get)
        load = agent_loads[best_agent]
        confidence = max(0.1, 1.0 - load)

        return RoutingDecision(
            task_id=task.id,
            target_agent_type=best_agent,
            reasoning=f"Least loaded agent (load: {load:.2f})",
            confidence=confidence,
            routing_strategy=RoutingStrategy.LOAD_BALANCED,
            estimated_wait_time=load * 10.0,  # Rough estimate
        )

    def _route_by_performance(self, task: Task, classification: ClassificationResult) -> RoutingDecision:
        """Route task based on agent performance history."""
        best_agent = classification.suggested_agent_type
        best_score = 0.0
        reasoning = "Default agent selection"

        for agent_type, metrics in self.agent_metrics.items():
            if metrics.total_tasks < 5:  # Not enough data
                continue

            # Calculate performance score
            score = (
                metrics.success_rate * 0.4 +
                (1.0 / max(1.0, metrics.average_execution_time / 30.0)) * 0.3 +
                (1.0 / max(1.0, metrics.average_queue_time / 10.0)) * 0.3
            )

            if score > best_score:
                best_score = score
                best_agent = agent_type
                reasoning = f"Best performing agent (score: {score:.2f})"

        return RoutingDecision(
            task_id=task.id,
            target_agent_type=best_agent,
            reasoning=reasoning,
            confidence=min(best_score, 1.0),
            routing_strategy=RoutingStrategy.PERFORMANCE_OPTIMIZED,
        )

    def _route_by_ml(self, task: Task, classification: ClassificationResult) -> RoutingDecision:
        """Route task using machine learning optimization."""
        # Placeholder for ML-based routing
        # This would use historical data to predict optimal routing
        return self._route_hybrid(task, classification)

    def _route_hybrid(self, task: Task, classification: ClassificationResult) -> RoutingDecision:
        """Route task using hybrid approach combining multiple strategies."""
        # Get scores from different strategies
        capability_decision = self._route_by_capabilities(task, classification)

        if self.agent_pool:
            load_decision = self._route_by_load(task, classification)
        else:
            load_decision = capability_decision

        performance_decision = self._route_by_performance(task, classification)

        # Calculate weighted scores
        agents_scores = defaultdict(float)

        # Capability score
        agents_scores[capability_decision.target_agent_type] += (
            capability_decision.confidence * self.capability_weight
        )

        # Load score
        agents_scores[load_decision.target_agent_type] += (
            load_decision.confidence * self.load_weight
        )

        # Performance score
        agents_scores[performance_decision.target_agent_type] += (
            performance_decision.confidence * self.performance_weight
        )

        # Select best agent
        if not agents_scores:
            return self._fallback_routing(task, classification, "No scoring results")

        best_agent = max(agents_scores, key=agents_scores.get)
        best_score = agents_scores[best_agent]

        reasoning = (
            f"Hybrid routing: capability={capability_decision.confidence:.2f}, "
            f"load={load_decision.confidence:.2f}, "
            f"performance={performance_decision.confidence:.2f}"
        )

        return RoutingDecision(
            task_id=task.id,
            target_agent_type=best_agent,
            reasoning=reasoning,
            confidence=best_score,
            routing_strategy=RoutingStrategy.HYBRID,
            alternative_agents=list(agents_scores.keys()),
        )

    def _calculate_capability_match_score(self, agent_type: str, required_capabilities: List[str]) -> float:
        """Calculate how well an agent type matches required capabilities."""
        if not required_capabilities:
            return 0.5  # Default score when no specific requirements

        if not self.agent_pool or agent_type not in self.agent_pool.registered_types:
            return 0.1  # Low score for unknown agents

        # Get agent capabilities (would need to be implemented in agent pool)
        # For now, use a simple mapping
        agent_capabilities = self._get_agent_type_capabilities(agent_type)

        if not agent_capabilities:
            return 0.3  # Default score

        # Calculate match ratio
        matched = sum(1 for cap in required_capabilities if cap in agent_capabilities)
        total = len(required_capabilities)

        base_score = matched / total if total > 0 else 0.5

        # Bonus for having extra relevant capabilities
        extra_relevant = len(agent_capabilities) - len(required_capabilities)
        bonus = min(extra_relevant * 0.1, 0.2)

        return min(base_score + bonus, 1.0)

    def _get_agent_type_capabilities(self, agent_type: str) -> Set[str]:
        """Get capabilities for an agent type."""
        # Default capability mappings
        capability_map = {
            "code": {
                "code_generation", "syntax_validation", "code_analysis",
                "refactoring", "debugging", "testing"
            },
            "context": {
                "context_analysis", "semantic_understanding", "search",
                "pattern_matching", "indexing"
            },
            "git": {
                "git_operations", "version_control", "merge_operations",
                "conflict_resolution", "branch_management"
            },
            "orchestrator": {
                "coordination", "planning", "monitoring", "resource_management"
            },
        }
        return capability_map.get(agent_type, set())

    def _fallback_routing(self, task: Task, classification: ClassificationResult, reason: str) -> RoutingDecision:
        """Provide fallback routing when other strategies fail."""
        return RoutingDecision(
            task_id=task.id,
            target_agent_type=classification.suggested_agent_type,
            reasoning=f"Fallback routing: {reason}",
            confidence=0.2,
            routing_strategy=RoutingStrategy.RULE_BASED,
        )

    def _record_routing_decision(
        self,
        task: Task,
        decision: RoutingDecision,
        classification: ClassificationResult,
        routing_time: float
    ) -> None:
        """Record routing decision for analysis and learning."""
        entry = {
            'timestamp': datetime.utcnow(),
            'task_id': task.id,
            'task_type': task.type,
            'decision': decision,
            'classification': classification,
            'routing_time': routing_time,
        }

        self.routing_history.append(entry)
        self.routing_stats[decision.target_agent_type] += 1

    def _initialize_default_rules(self) -> None:
        """Initialize default routing rules."""
        # High priority tasks go to dedicated agents
        self.add_routing_rule(
            "urgent_tasks",
            "Urgent Task Priority",
            "Route urgent tasks to fastest available agents",
            lambda task: task.priority.value <= 2,  # Critical or High
            "code",  # Default to code agent for urgent tasks
            priority=10
        )

        # Git-related tasks
        self.add_routing_rule(
            "git_operations",
            "Git Operations",
            "Route git operations to git agent",
            lambda task: any(keyword in str(task.payload).lower()
                           for keyword in ['git', 'commit', 'merge', 'branch']),
            "git",
            priority=20
        )

        # Search and context tasks
        self.add_routing_rule(
            "search_operations",
            "Search Operations",
            "Route search operations to context agent",
            lambda task: any(keyword in str(task.payload).lower()
                           for keyword in ['search', 'find', 'locate', 'query']),
            "context",
            priority=30
        )

        # Large payload tasks need more capable agents
        self.add_routing_rule(
            "large_payloads",
            "Large Payload Tasks",
            "Route large payload tasks to code agents",
            lambda task: len(str(task.payload)) > 1000,
            "code",
            priority=40
        )

        # Background tasks can go to any available agent
        self.add_routing_rule(
            "background_tasks",
            "Background Tasks",
            "Route background tasks to least loaded agent",
            lambda task: task.priority == TaskPriority.BACKGROUND,
            "code",  # Will be overridden by load balancing
            priority=90
        )
