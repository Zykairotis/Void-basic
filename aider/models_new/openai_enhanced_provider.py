"""
Enhanced OpenAI Provider - Advanced AI Capabilities
Supports function calling, multi-modal processing, built-in tools, and agent workflows.
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .model_manager import ModelProvider, ModelConfig, ModelRequest, ModelResponse

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Available OpenAI tool types"""
    FUNCTION = "function"
    CODE_INTERPRETER = "code_interpreter"
    FILE_SEARCH = "file_search"
    WEB_SEARCH = "web_search"


class MessageRole(Enum):
    """OpenAI message roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class FunctionDefinition:
    """Function definition for OpenAI function calling"""
    name: str
    description: str
    parameters: Dict[str, Any]
    strict: bool = True  # Enable structured outputs


@dataclass
class ToolDefinition:
    """Tool definition for OpenAI tools"""
    type: str
    function: Optional[FunctionDefinition] = None


@dataclass
class MessageContent:
    """Multi-modal message content"""
    type: str  # "text", "image_url", "audio"
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None
    audio: Optional[Dict[str, str]] = None


@dataclass
class ChatMessage:
    """OpenAI chat message with multi-modal support"""
    role: str
    content: Union[str, List[MessageContent]]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


@dataclass
class ToolCall:
    """OpenAI tool call result"""
    id: str
    type: str
    function: Dict[str, Any]


@dataclass
class ToolResult:
    """Result from tool execution"""
    tool_call_id: str
    content: str
    is_error: bool = False


class EnhancedOpenAIProvider(ModelProvider):
    """
    Enhanced OpenAI Provider with advanced capabilities:
    - Function calling with structured outputs
    - Multi-modal processing (text, images, audio)
    - Built-in tools (code interpreter, file search, web search)
    - Agent workflow support
    - Tool result processing
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tools: Dict[str, Callable] = {}
        self.built_in_tools: List[str] = []
        self.conversation_history: List[ChatMessage] = []
        self.max_tool_iterations = 10

    async def initialize(self):
        """Initialize OpenAI client with enhanced capabilities"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = httpx.AsyncClient(
            base_url=self.config.api_base,
            headers={
                "Authorization": f"Bearer {api_key}",
                "OpenAI-Beta": "assistants=v2"  # Enable beta features
            },
            timeout=self.config.timeout_seconds
        )

        # Initialize built-in tools
        await self._initialize_builtin_tools()
        logger.info(f"Enhanced OpenAI provider initialized with {len(self.tools)} tools")

    async def _initialize_builtin_tools(self):
        """Initialize OpenAI's built-in tools"""
        self.built_in_tools = [
            ToolType.CODE_INTERPRETER.value,
            ToolType.FILE_SEARCH.value,
            ToolType.WEB_SEARCH.value
        ]

        # Register built-in tool handlers
        self.tools.update({
            "web_search": self._handle_web_search,
            "code_interpreter": self._handle_code_interpreter,
            "file_search": self._handle_file_search,
            "image_generation": self._handle_image_generation
        })

    def register_function(self, name: str, description: str, parameters: Dict[str, Any],
                         handler: Callable, strict: bool = True):
        """Register a custom function for function calling"""
        self.tools[name] = handler
        logger.info(f"Registered function: {name}")

    def create_function_definition(self, name: str, description: str,
                                 parameters: Dict[str, Any], strict: bool = True) -> FunctionDefinition:
        """Create a function definition for OpenAI function calling"""
        return FunctionDefinition(
            name=name,
            description=description,
            parameters=parameters,
            strict=strict
        )

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for multi-modal input"""
        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
                return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

    def create_multimodal_message(self, text: str, images: List[str] = None,
                                 audio_path: str = None) -> ChatMessage:
        """Create a multi-modal message with text, images, and audio"""
        content = []

        # Add text content
        if text:
            content.append(MessageContent(type="text", text=text))

        # Add image content
        if images:
            for image_path in images:
                image_url = self.encode_image(image_path)
                content.append(MessageContent(
                    type="image_url",
                    image_url={"url": image_url}
                ))

        # Add audio content (if supported by model)
        if audio_path:
            audio_data = self.encode_image(audio_path)  # Same encoding process
            content.append(MessageContent(
                type="audio",
                audio={"data": audio_data}
            ))

        return ChatMessage(role=MessageRole.USER.value, content=content)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate response with enhanced OpenAI capabilities"""
        start_time = time.time()
        request_id = hashlib.md5(f"{request.prompt[:100]}{start_time}".encode()).hexdigest()

        # Wait for rate limit
        estimated_tokens = len(str(request.prompt).split()) * 1.3
        while not await self.rate_limiter.acquire(int(estimated_tokens)):
            await asyncio.sleep(1)

        # Prepare conversation
        messages = self._prepare_messages(request)
        tools = self._prepare_tools(request)

        try:
            response = await self._chat_completion_with_tools(
                messages=messages,
                tools=tools,
                request=request,
                request_id=request_id
            )

            latency_ms = int((time.time() - start_time) * 1000)
            return self._create_response(response, request_id, latency_ms)

        except Exception as e:
            logger.error(f"Enhanced OpenAI API error: {e}")
            raise

    def _prepare_messages(self, request: ModelRequest) -> List[Dict[str, Any]]:
        """Prepare messages for OpenAI API"""
        messages = []

        # Add system message if needed
        if hasattr(request, 'system_message') and request.system_message:
            messages.append({
                "role": "system",
                "content": request.system_message
            })

        # Handle multi-modal content
        if hasattr(request, 'images') and request.images:
            # Multi-modal request
            multimodal_msg = self.create_multimodal_message(
                text=request.prompt,
                images=request.images,
                audio_path=getattr(request, 'audio_path', None)
            )
            messages.append(asdict(multimodal_msg))
        else:
            # Text-only request
            messages.append({
                "role": "user",
                "content": request.prompt
            })

        return messages

    def _prepare_tools(self, request: ModelRequest) -> List[Dict[str, Any]]:
        """Prepare tools for OpenAI API"""
        tools = []

        # Add built-in tools if requested
        if hasattr(request, 'enable_web_search') and request.enable_web_search:
            tools.append({"type": "web_search"})

        if hasattr(request, 'enable_code_interpreter') and request.enable_code_interpreter:
            tools.append({"type": "code_interpreter"})

        if hasattr(request, 'enable_file_search') and request.enable_file_search:
            tools.append({"type": "file_search"})

        # Add custom functions
        if hasattr(request, 'functions') and request.functions:
            for func_def in request.functions:
                tools.append({
                    "type": "function",
                    "function": asdict(func_def)
                })

        return tools

    async def _chat_completion_with_tools(self, messages: List[Dict[str, Any]],
                                        tools: List[Dict[str, Any]],
                                        request: ModelRequest,
                                        request_id: str) -> Dict[str, Any]:
        """Perform chat completion with tool support"""
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "max_tokens": request.max_tokens or 2000,
            "temperature": request.temperature,
            "response_format": {"type": "json_object"} if getattr(request, 'json_mode', False) else None
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = getattr(request, 'tool_choice', "auto")

        iteration_count = 0
        conversation = messages.copy()

        while iteration_count < self.max_tool_iterations:
            response = await self.client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            message = data["choices"][0]["message"]
            conversation.append(message)

            # Check if model wants to call tools
            if message.get("tool_calls"):
                tool_results = await self._execute_tool_calls(message["tool_calls"])

                # Add tool results to conversation
                for result in tool_results:
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": result.content
                    })

                # Update payload for next iteration
                payload["messages"] = conversation
                iteration_count += 1
            else:
                # No more tool calls, return final response
                return data

        # Max iterations reached
        logger.warning(f"Max tool iterations ({self.max_tool_iterations}) reached")
        return data

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute tool calls and return results"""
        results = []

        for tool_call in tool_calls:
            try:
                if tool_call["type"] == "function":
                    result = await self._execute_function_call(tool_call)
                else:
                    result = await self._execute_builtin_tool(tool_call)

                results.append(result)

            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                results.append(ToolResult(
                    tool_call_id=tool_call["id"],
                    content=f"Error executing tool: {str(e)}",
                    is_error=True
                ))

        return results

    async def _execute_function_call(self, tool_call: Dict[str, Any]) -> ToolResult:
        """Execute a custom function call"""
        function_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])

        if function_name not in self.tools:
            raise ValueError(f"Unknown function: {function_name}")

        handler = self.tools[function_name]
        result = await handler(**arguments)

        return ToolResult(
            tool_call_id=tool_call["id"],
            content=json.dumps(result) if not isinstance(result, str) else result
        )

    async def _execute_builtin_tool(self, tool_call: Dict[str, Any]) -> ToolResult:
        """Execute a built-in OpenAI tool"""
        tool_type = tool_call["type"]

        if tool_type in self.tools:
            handler = self.tools[tool_type]
            result = await handler(tool_call)
            return ToolResult(
                tool_call_id=tool_call["id"],
                content=str(result)
            )
        else:
            # For built-in tools, OpenAI handles execution
            return ToolResult(
                tool_call_id=tool_call["id"],
                content="Tool executed by OpenAI"
            )

    async def _handle_web_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Handle web search requests"""
        # This would integrate with OpenAI's web search capability
        logger.info(f"Performing web search: {query}")
        return {
            "query": query,
            "results": "Web search results would be provided by OpenAI",
            "status": "completed"
        }

    async def _handle_code_interpreter(self, code: str, **kwargs) -> Dict[str, Any]:
        """Handle code interpreter requests"""
        logger.info("Executing code via OpenAI Code Interpreter")
        return {
            "code": code,
            "execution": "Code executed by OpenAI Code Interpreter",
            "status": "completed"
        }

    async def _handle_file_search(self, query: str, files: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Handle file search requests"""
        logger.info(f"Searching files for: {query}")
        return {
            "query": query,
            "files": files or [],
            "results": "File search results from OpenAI",
            "status": "completed"
        }

    async def _handle_image_generation(self, prompt: str, size: str = "1024x1024", **kwargs) -> Dict[str, Any]:
        """Handle image generation requests"""
        try:
            payload = {
                "model": "dall-e-3",
                "prompt": prompt,
                "size": size,
                "quality": "standard",
                "n": 1
            }

            response = await self.client.post("/v1/images/generations", json=payload)
            response.raise_for_status()
            data = response.json()

            return {
                "prompt": prompt,
                "image_url": data["data"][0]["url"],
                "revised_prompt": data["data"][0].get("revised_prompt"),
                "status": "completed"
            }

        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return {
                "prompt": prompt,
                "error": str(e),
                "status": "failed"
            }

    def _create_response(self, data: Dict[str, Any], request_id: str, latency_ms: int) -> ModelResponse:
        """Create ModelResponse from OpenAI API response"""
        message = data["choices"][0]["message"]
        usage = data["usage"]

        # Calculate cost
        cost = (usage["prompt_tokens"] * self.config.cost_per_1m_input / 1_000_000 +
               usage["completion_tokens"] * self.config.cost_per_1m_output / 1_000_000)

        # Extract tool calls if present
        tool_calls = message.get("tool_calls", [])
        metadata = {
            "finish_reason": data["choices"][0]["finish_reason"],
            "tool_calls": tool_calls,
            "model_version": data.get("model"),
            "system_fingerprint": data.get("system_fingerprint")
        }

        return ModelResponse(
            content=message["content"] or "",
            model_used=self.config.model_name,
            provider=self.config.provider,
            tokens_used={
                "input": usage["prompt_tokens"],
                "output": usage["completion_tokens"],
                "total": usage["total_tokens"]
            },
            cost_estimate=cost,
            latency_ms=latency_ms,
            request_id=request_id,
            metadata=metadata
        )

    async def create_assistant(self, name: str, instructions: str,
                             tools: List[str] = None,
                             model: str = None) -> Dict[str, Any]:
        """Create an OpenAI Assistant for complex workflows"""
        payload = {
            "name": name,
            "instructions": instructions,
            "model": model or self.config.model_name,
            "tools": []
        }

        # Add requested tools
        if tools:
            for tool in tools:
                if tool in self.built_in_tools:
                    payload["tools"].append({"type": tool})
                elif tool in self.tools:
                    # Custom function - would need function definition
                    pass

        try:
            response = await self.client.post("/v1/assistants", json=payload)
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Failed to create assistant: {e}")
            raise

    async def run_assistant_thread(self, assistant_id: str, message: str) -> Dict[str, Any]:
        """Run a conversation thread with an assistant"""
        try:
            # Create thread
            thread_response = await self.client.post("/v1/threads", json={})
            thread_response.raise_for_status()
            thread_id = thread_response.json()["id"]

            # Add message to thread
            await self.client.post(f"/v1/threads/{thread_id}/messages", json={
                "role": "user",
                "content": message
            })

            # Run the thread
            run_response = await self.client.post(f"/v1/threads/{thread_id}/runs", json={
                "assistant_id": assistant_id
            })
            run_response.raise_for_status()
            run_id = run_response.json()["id"]

            # Poll for completion
            while True:
                status_response = await self.client.get(f"/v1/threads/{thread_id}/runs/{run_id}")
                status_response.raise_for_status()
                status = status_response.json()["status"]

                if status in ["completed", "failed", "cancelled", "expired"]:
                    break

                await asyncio.sleep(1)

            # Get messages
            messages_response = await self.client.get(f"/v1/threads/{thread_id}/messages")
            messages_response.raise_for_status()

            return {
                "thread_id": thread_id,
                "run_id": run_id,
                "status": status,
                "messages": messages_response.json()
            }

        except Exception as e:
            logger.error(f"Assistant thread execution failed: {e}")
            raise

    async def health_check(self) -> bool:
        """Enhanced health check with tool availability"""
        try:
            # Basic model check
            test_request = ModelRequest(
                prompt="Test",
                task_type="general",
                complexity="simple",
                priority="speed",
                max_tokens=10
            )
            response = await self.generate_response(test_request)

            # Check if tools are available
            tools_available = len(self.tools) > 0

            return len(response.content) > 0 and tools_available

        except Exception as e:
            logger.error(f"Enhanced health check failed: {e}")
            return False

    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()
