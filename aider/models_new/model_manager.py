"""
AI Model Manager - 2025 Enterprise Implementation
Provides intelligent routing, multi-provider support, and production-grade features.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib
import os
from contextlib import asynccontextmanager

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Try to import enhanced providers, fallback to basic ones if not available
try:
    from .openai_enhanced_provider import EnhancedOpenAIProvider
    ENHANCED_OPENAI_AVAILABLE = True
except ImportError:
    ENHANCED_OPENAI_AVAILABLE = False

try:
    from .anthropic_enhanced_provider import EnhancedAnthropicProvider
    ENHANCED_ANTHROPIC_AVAILABLE = True
except ImportError:
    ENHANCED_ANTHROPIC_AVAILABLE = False

try:
    from .xai_enhanced_provider import EnhancedXAIProvider
    ENHANCED_XAI_AVAILABLE = True
except ImportError:
    ENHANCED_XAI_AVAILABLE = False

try:
    from .gemini_enhanced_provider import EnhancedGeminiProvider
    ENHANCED_GEMINI_AVAILABLE = True
except ImportError:
    ENHANCED_GEMINI_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Supported task types for intelligent routing"""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    GENERAL = "general"

class ComplexityLevel(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXPERT = "expert"

class Priority(Enum):
    """Request priority levels"""
    SPEED = "speed"
    QUALITY = "quality"
    COST = "cost"
    BALANCED = "balanced"

@dataclass
class ModelConfig:
    """Configuration for a specific AI model"""
    provider: str
    model_name: str
    api_base: str
    cost_per_1m_input: float
    cost_per_1m_output: float
    context_window: int
    rate_limit_rpm: int
    rate_limit_tpm: int
    strengths: List[str]
    supported_tasks: List[TaskType]
    max_retries: int = 3
    timeout_seconds: int = 60

@dataclass
class ModelRequest:
    """Standardized request format"""
    prompt: str
    task_type: TaskType
    complexity: ComplexityLevel
    priority: Priority
    max_tokens: Optional[int] = None
    temperature: float = 0.1
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ModelResponse:
    """Standardized response format"""
    content: str
    model_used: str
    provider: str
    tokens_used: Dict[str, int]
    cost_estimate: float
    latency_ms: int
    confidence_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    request_id: str = ""

class RateLimiter:
    """Token bucket rate limiter for API calls"""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.rpm_limit = requests_per_minute
        self.tpm_limit = tokens_per_minute
        self.request_tokens = []
        self.token_usage = []
        self.lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 1000) -> bool:
        """Acquire rate limit permission"""
        async with self.lock:
            now = time.time()
            minute_ago = now - 60

            # Clean old entries
            self.request_tokens = [t for t in self.request_tokens if t > minute_ago]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > minute_ago]

            # Check limits
            current_requests = len(self.request_tokens)
            current_tokens = sum(tokens for _, tokens in self.token_usage)

            if (current_requests >= self.rpm_limit or
                current_tokens + estimated_tokens > self.tpm_limit):
                return False

            # Record usage
            self.request_tokens.append(now)
            self.token_usage.append((now, estimated_tokens))
            return True

class ModelProvider:
    """Base class for AI model providers"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.rate_limiter = RateLimiter(
            config.rate_limit_rpm,
            config.rate_limit_tpm
        )
        self.client = None

    async def initialize(self):
        """Initialize the provider client"""
        raise NotImplementedError

    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate response from the model"""
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Check if the provider is healthy"""
        try:
            test_request = ModelRequest(
                prompt="Test",
                task_type=TaskType.GENERAL,
                complexity=ComplexityLevel.SIMPLE,
                priority=Priority.SPEED,
                max_tokens=10
            )
            response = await self.generate_response(test_request)
            return len(response.content) > 0
        except Exception as e:
            logger.error(f"Health check failed for {self.config.provider}: {e}")
            return False

class OpenAIProvider(ModelProvider):
    """OpenAI GPT provider implementation"""

    async def initialize(self):
        """Initialize OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = httpx.AsyncClient(
            base_url=self.config.api_base,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=self.config.timeout_seconds
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate response using OpenAI API"""
        start_time = time.time()
        request_id = hashlib.md5(f"{request.prompt[:100]}{start_time}".encode()).hexdigest()

        # Wait for rate limit
        estimated_tokens = len(request.prompt.split()) * 1.3
        while not await self.rate_limiter.acquire(int(estimated_tokens)):
            await asyncio.sleep(1)

        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens or 2000,
            "temperature": request.temperature
        }

        try:
            response = await self.client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            usage = data["usage"]

            latency_ms = int((time.time() - start_time) * 1000)
            cost = (usage["prompt_tokens"] * self.config.cost_per_1m_input / 1_000_000 +
                   usage["completion_tokens"] * self.config.cost_per_1m_output / 1_000_000)

            return ModelResponse(
                content=content,
                model_used=self.config.model_name,
                provider=self.config.provider,
                tokens_used={
                    "input": usage["prompt_tokens"],
                    "output": usage["completion_tokens"],
                    "total": usage["total_tokens"]
                },
                cost_estimate=cost,
                latency_ms=latency_ms,
                request_id=request_id
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

class AnthropicProvider(ModelProvider):
    """Anthropic Claude provider implementation"""

    async def initialize(self):
        """Initialize Anthropic client"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = httpx.AsyncClient(
            base_url=self.config.api_base,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            timeout=self.config.timeout_seconds
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate response using Anthropic API"""
        start_time = time.time()
        request_id = hashlib.md5(f"{request.prompt[:100]}{start_time}".encode()).hexdigest()

        # Wait for rate limit
        estimated_tokens = len(request.prompt.split()) * 1.3
        while not await self.rate_limiter.acquire(int(estimated_tokens)):
            await asyncio.sleep(1)

        payload = {
            "model": self.config.model_name,
            "max_tokens": request.max_tokens or 2000,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature
        }

        try:
            response = await self.client.post("/v1/messages", json=payload)
            response.raise_for_status()
            data = response.json()

            content = data["content"][0]["text"]
            usage = data["usage"]

            latency_ms = int((time.time() - start_time) * 1000)
            cost = (usage["input_tokens"] * self.config.cost_per_1m_input / 1_000_000 +
                   usage["output_tokens"] * self.config.cost_per_1m_output / 1_000_000)

            return ModelResponse(
                content=content,
                model_used=self.config.model_name,
                provider=self.config.provider,
                tokens_used={
                    "input": usage["input_tokens"],
                    "output": usage["output_tokens"],
                    "total": usage["input_tokens"] + usage["output_tokens"]
                },
                cost_estimate=cost,
                latency_ms=latency_ms,
                request_id=request_id
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

class XAIProvider(ModelProvider):
    """xAI Grok provider implementation"""

    async def initialize(self):
        """Initialize xAI client"""
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable not set")

        self.client = httpx.AsyncClient(
            base_url=self.config.api_base,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=self.config.timeout_seconds
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate response using xAI API"""
        start_time = time.time()
        request_id = hashlib.md5(f"{request.prompt[:100]}{start_time}".encode()).hexdigest()

        # Wait for rate limit
        estimated_tokens = len(request.prompt.split()) * 1.3
        while not await self.rate_limiter.acquire(int(estimated_tokens)):
            await asyncio.sleep(1)

        payload = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens or 2000,
            "temperature": request.temperature
        }

        try:
            response = await self.client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            usage = data["usage"]

            latency_ms = int((time.time() - start_time) * 1000)
            cost = (usage["prompt_tokens"] * self.config.cost_per_1m_input / 1_000_000 +
                   usage["completion_tokens"] * self.config.cost_per_1m_output / 1_000_000)

            return ModelResponse(
                content=content,
                model_used=self.config.model_name,
                provider=self.config.provider,
                tokens_used={
                    "input": usage["prompt_tokens"],
                    "output": usage["completion_tokens"],
                    "total": usage["total_tokens"]
                },
                cost_estimate=cost,
                latency_ms=latency_ms,
                request_id=request_id
            )

        except Exception as e:
            logger.error(f"xAI API error: {e}")
            raise

class ModelManager:
    """Enterprise AI Model Manager with intelligent routing and fallback support"""

    def __init__(self):
        self.providers: Dict[str, ModelProvider] = {}
        self.model_configs = self._load_model_configurations()
        self.routing_rules = self._initialize_routing_rules()
        self.performance_metrics = {}
        self.initialized = False

    def _load_model_configurations(self) -> Dict[str, ModelConfig]:
        """Load model configurations based on 2025 best practices"""
        return {
            "claude": ModelConfig(
                provider="anthropic",
                model_name="claude-3-5-sonnet-20241022",
                api_base="https://api.anthropic.com",
                cost_per_1m_input=3.00,
                cost_per_1m_output=15.00,
                context_window=200000,
                rate_limit_rpm=50,
                rate_limit_tpm=40000,
                strengths=["code_quality", "architecture", "complex_reasoning"],
                supported_tasks=[
                    TaskType.CODE_GENERATION,
                    TaskType.CODE_REVIEW,
                    TaskType.CODE_REFACTORING,
                    TaskType.ANALYSIS
                ]
            ),
            "gpt4": ModelConfig(
                provider="openai",
                model_name="gpt-4-1106-preview",
                api_base="https://api.openai.com",
                cost_per_1m_input=10.00,
                cost_per_1m_output=30.00,
                context_window=128000,
                rate_limit_rpm=60,
                rate_limit_tpm=60000,
                strengths=["versatility", "debugging", "general_reasoning"],
                supported_tasks=[
                    TaskType.DEBUGGING,
                    TaskType.ANALYSIS,
                    TaskType.PLANNING,
                    TaskType.GENERAL
                ]
            ),
            "grok": ModelConfig(
                provider="xai",
                model_name="grok-beta",
                api_base="https://api.x.ai",
                cost_per_1m_input=3.00,
                cost_per_1m_output=15.00,
                context_window=128000,
                rate_limit_rpm=100,
                rate_limit_tpm=80000,
                strengths=["speed", "reasoning", "real_time_data"],
                supported_tasks=[
                    TaskType.CODE_GENERATION,
                    TaskType.DOCUMENTATION,
                    TaskType.TESTING,
                    TaskType.GENERAL
                ]
            ),
            "gemini": ModelConfig(
                provider="google",
                model_name="gemini-2.5-pro",
                api_base="https://generativelanguage.googleapis.com/v1beta",
                cost_per_1m_input=1.25,
                cost_per_1m_output=5.00,
                context_window=1000000,
                rate_limit_rpm=60,
                rate_limit_tpm=100000,
                strengths=["multimodal", "large_context", "code_execution"],
                supported_tasks=[
                    TaskType.CODE_GENERATION,
                    TaskType.ANALYSIS,
                    TaskType.DOCUMENTATION,
                    TaskType.GENERAL
                ]
            )
        }

    def _initialize_routing_rules(self) -> Dict[str, List[str]]:
        """Initialize intelligent routing rules based on research"""
        return {
            # Code generation: Claude first (best quality), Gemini (large context), Grok (speed), GPT-4 fallback
            TaskType.CODE_GENERATION.value: ["claude", "gemini", "grok", "gpt4"],
            TaskType.CODE_REVIEW.value: ["claude", "gpt4", "gemini", "grok"],
            TaskType.CODE_REFACTORING.value: ["claude", "gpt4", "gemini", "grok"],

            # Analysis and debugging: GPT-4 first (best reasoning), Claude fallback, Gemini for large context
            TaskType.DEBUGGING.value: ["gpt4", "claude", "gemini", "grok"],
            TaskType.ANALYSIS.value: ["gemini", "gpt4", "claude", "grok"],  # Gemini first for large context analysis
            TaskType.PLANNING.value: ["gpt4", "claude", "gemini", "grok"],

            # Fast tasks: Grok first (speed), others as fallback
            TaskType.DOCUMENTATION.value: ["gemini", "grok", "claude", "gpt4"],  # Gemini first for multimodal docs
            TaskType.TESTING.value: ["grok", "claude", "gemini", "gpt4"],

            # General: All models available with Gemini prioritized for multimodal
            TaskType.GENERAL.value: ["gemini", "claude", "gpt4", "grok"]
        }

    async def initialize(self):
        """Initialize all model providers"""
        provider_classes = {
            "openai": EnhancedOpenAIProvider if ENHANCED_OPENAI_AVAILABLE else OpenAIProvider,
            "anthropic": EnhancedAnthropicProvider if ENHANCED_ANTHROPIC_AVAILABLE else AnthropicProvider,
            "xai": EnhancedXAIProvider if ENHANCED_XAI_AVAILABLE else XAIProvider,
            "google": EnhancedGeminiProvider if ENHANCED_GEMINI_AVAILABLE else None
        }

        for model_key, config in self.model_configs.items():
            provider_class = provider_classes.get(config.provider)
            if provider_class is None:
                logger.warning(f"Provider {config.provider} not available, skipping {model_key}")
                continue

            provider = provider_class(config)

            try:
                await provider.initialize()
                self.providers[model_key] = provider
                logger.info(f"Initialized {model_key} provider successfully")
            except Exception as e:
                logger.error(f"Failed to initialize {model_key}: {e}")

        self.initialized = True
        logger.info(f"ModelManager initialized with {len(self.providers)} providers")

    def select_model(self, request: ModelRequest) -> str:
        """Intelligent model selection based on task characteristics"""
        task_type = request.task_type.value
        complexity = request.complexity
        priority = request.priority

        # Get routing options for task type
        routing_options = self.routing_rules.get(task_type, ["gemini", "claude", "gpt4", "grok"])

        # Apply priority-based adjustments
        if priority == Priority.SPEED:
            # Prefer faster models
            speed_priority = ["grok", "gemini", "claude", "gpt4"]
            routing_options = sorted(routing_options,
                                   key=lambda x: speed_priority.index(x) if x in speed_priority else 999)

        elif priority == Priority.QUALITY:
            # Prefer highest quality models
            quality_priority = ["claude", "gpt4", "gemini", "grok"]
            routing_options = sorted(routing_options,
                                   key=lambda x: quality_priority.index(x) if x in quality_priority else 999)

        elif priority == Priority.COST:
            # Prefer lower cost models (Gemini is most cost-effective)
            cost_priority = ["gemini", "grok", "claude", "gpt4"]
            routing_options = sorted(routing_options,
                                   key=lambda x: cost_priority.index(x) if x in cost_priority else 999)

        # Check for multimodal requirements
        if hasattr(request, 'images') or hasattr(request, 'files') or hasattr(request, 'videos'):
            # Prioritize multimodal-capable models
            multimodal_priority = ["gemini", "gpt4", "claude", "grok"]
            routing_options = sorted(routing_options,
                                   key=lambda x: multimodal_priority.index(x) if x in multimodal_priority else 999)

        # Filter to available providers
        available_options = [model for model in routing_options if model in self.providers]

        if not available_options:
            raise ValueError(f"No available models for task type: {task_type}")

        return available_options[0]

    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate response with intelligent routing and fallback"""
        if not self.initialized:
            await self.initialize()

        # Select primary model
        selected_model = self.select_model(request)
        routing_options = self.routing_rules.get(request.task_type.value, [selected_model])

        # Try primary model and fallbacks
        last_error = None
        for model_key in routing_options:
            if model_key not in self.providers:
                continue

            try:
                provider = self.providers[model_key]
                response = await provider.generate_response(request)

                # Update performance metrics
                self._update_metrics(model_key, response, success=True)

                logger.info(f"Successfully generated response using {model_key}")
                return response

            except Exception as e:
                logger.warning(f"Model {model_key} failed: {e}")
                self._update_metrics(model_key, None, success=False)
                last_error = e
                continue

        # All models failed
        raise Exception(f"All models failed. Last error: {last_error}")

    def _update_metrics(self, model_key: str, response: Optional[ModelResponse], success: bool):
        """Update performance metrics for monitoring"""
        if model_key not in self.performance_metrics:
            self.performance_metrics[model_key] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_cost": 0.0,
                "total_latency": 0,
                "last_updated": datetime.now()
            }

        metrics = self.performance_metrics[model_key]
        metrics["total_requests"] += 1

        if success and response:
            metrics["successful_requests"] += 1
            metrics["total_cost"] += response.cost_estimate
            metrics["total_latency"] += response.latency_ms

        metrics["last_updated"] = datetime.now()

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all providers"""
        health_status = {}

        for model_key, provider in self.providers.items():
            try:
                health_status[model_key] = await provider.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {model_key}: {e}")
                health_status[model_key] = False

        return health_status

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        metrics_summary = {}

        for model_key, metrics in self.performance_metrics.items():
            if metrics["total_requests"] > 0:
                success_rate = metrics["successful_requests"] / metrics["total_requests"]
                avg_latency = metrics["total_latency"] / metrics["successful_requests"] if metrics["successful_requests"] > 0 else 0

                metrics_summary[model_key] = {
                    "success_rate": success_rate,
                    "total_requests": metrics["total_requests"],
                    "total_cost": metrics["total_cost"],
                    "average_latency_ms": avg_latency,
                    "last_updated": metrics["last_updated"].isoformat()
                }

        return metrics_summary

    async def close(self):
        """Close all provider connections"""
        for provider in self.providers.values():
            if hasattr(provider, 'client') and provider.client:
                await provider.client.aclose()

        logger.info("ModelManager connections closed")

# Convenience functions for common use cases
async def generate_code(description: str, language: str = "python",
                       complexity: ComplexityLevel = ComplexityLevel.MEDIUM) -> str:
    """Generate code using the best available model"""
    manager = ModelManager()

    prompt = f"""Generate high-quality {language} code for the following requirement:

{description}

Requirements:
- Write clean, well-documented code
- Include proper error handling
- Follow best practices for {language}
- Add docstrings and comments where appropriate

Code:"""

    request = ModelRequest(
        prompt=prompt,
        task_type=TaskType.CODE_GENERATION,
        complexity=complexity,
        priority=Priority.QUALITY,
        temperature=0.1
    )

    try:
        response = await manager.generate_response(request)
        return response.content
    finally:
        await manager.close()

async def analyze_code(code: str, analysis_type: str = "quality") -> str:
    """Analyze code using the best available model"""
    manager = ModelManager()

    prompt = f"""Analyze the following code for {analysis_type}:

{code}

Provide a detailed analysis covering:
- Code quality and best practices
- Potential issues or bugs
- Performance considerations
- Suggested improvements
- Security considerations (if applicable)

Analysis:"""

    request = ModelRequest(
        prompt=prompt,
        task_type=TaskType.ANALYSIS,
        complexity=ComplexityLevel.MEDIUM,
        priority=Priority.QUALITY,
        temperature=0.0
    )

    try:
        response = await manager.generate_response(request)
        return response.content
    finally:
        await manager.close()

# Global instance for singleton pattern
_model_manager_instance = None

async def get_model_manager() -> ModelManager:
    """Get or create the global ModelManager instance"""
    global _model_manager_instance

    if _model_manager_instance is None:
        _model_manager_instance = ModelManager()
        await _model_manager_instance.initialize()

    return _model_manager_instance
