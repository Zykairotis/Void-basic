"""
Enhanced Anthropic Provider - Claude Advanced AI Capabilities
Supports computer use, text editor, web search, multi-modal processing, and autonomous sessions.
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from pathlib import Path
import uuid

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from PIL import Image, ImageDraw
import pyautogui
import psutil

from .model_manager import ModelProvider, ModelConfig, ModelRequest, ModelResponse

logger = logging.getLogger(__name__)


class ClaudeToolType(Enum):
    """Available Claude tool types"""
    COMPUTER_USE = "computer_20241022"
    TEXT_EDITOR = "str_replace_editor_tool"
    WEB_SEARCH = "web_search_tool"
    FUNCTION = "function"


class ComputerAction(Enum):
    """Computer use actions"""
    SCREENSHOT = "screenshot"
    CLICK = "click"
    TYPE = "type"
    KEY = "key"
    SCROLL = "scroll"
    CURSOR_POSITION = "cursor_position"


@dataclass
class ComputerCommand:
    """Computer use command structure"""
    action: str
    coordinate: Optional[Tuple[int, int]] = None
    text: Optional[str] = None
    key: Optional[str] = None
    scroll_direction: Optional[str] = None


@dataclass
class TextEditorCommand:
    """Text editor command structure"""
    command: str  # "str_replace", "create", "view", "list"
    path: str
    old_str: Optional[str] = None
    new_str: Optional[str] = None
    view_range: Optional[Tuple[int, int]] = None


@dataclass
class WebSearchQuery:
    """Web search query structure"""
    query: str
    max_results: int = 10
    recency_filter: Optional[str] = None  # "day", "week", "month", "year"


@dataclass
class ClaudeMessage:
    """Claude message with multi-modal support"""
    role: str
    content: Union[str, List[Dict[str, Any]]]


@dataclass
class AutonomousSession:
    """Autonomous coding session state"""
    session_id: str
    start_time: datetime
    max_duration: timedelta
    current_task: Optional[str]
    completed_tasks: List[str]
    active_files: List[str]
    memory: Dict[str, Any]


class EnhancedAnthropicProvider(ModelProvider):
    """
    Enhanced Anthropic Provider with Claude's advanced capabilities:
    - Computer use for direct system interaction
    - Text editor for file manipulation
    - Web search for real-time information
    - Multi-modal processing (text + images)
    - Autonomous coding sessions
    - Memory integration and persistence
    - Long context handling (200K tokens)
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tools: Dict[str, Callable] = {}
        self.autonomous_sessions: Dict[str, AutonomousSession] = {}
        self.memory_store: Dict[str, Any] = {}
        self.conversation_context: List[ClaudeMessage] = []
        self.max_context_tokens = 200000  # Claude 4's context window
        self.computer_use_enabled = False
        self.text_editor_enabled = False
        self.web_search_enabled = False

        # Computer use setup
        self.screen_size = None
        self.screenshot_dir = Path(tempfile.gettempdir()) / "claude_screenshots"
        self.screenshot_dir.mkdir(exist_ok=True)

    async def initialize(self):
        """Initialize Anthropic client with advanced capabilities"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = httpx.AsyncClient(
            base_url=self.config.api_base,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "computer-use-2024-10-22,text-editor-2024-10-22"
            },
            timeout=self.config.timeout_seconds
        )

        # Initialize capabilities
        await self._initialize_computer_use()
        await self._initialize_text_editor()
        await self._initialize_web_search()
        await self._load_persistent_memory()

        logger.info(f"Enhanced Anthropic provider initialized with advanced capabilities")

    async def _initialize_computer_use(self):
        """Initialize computer use capabilities"""
        try:
            self.screen_size = pyautogui.size()
            self.computer_use_enabled = True

            # Register computer use handler
            self.tools["computer_use"] = self._handle_computer_use

            logger.info(f"Computer use enabled - Screen size: {self.screen_size}")
        except Exception as e:
            logger.warning(f"Computer use initialization failed: {e}")
            self.computer_use_enabled = False

    async def _initialize_text_editor(self):
        """Initialize text editor capabilities"""
        try:
            self.text_editor_enabled = True
            self.tools["text_editor"] = self._handle_text_editor
            logger.info("Text editor capabilities enabled")
        except Exception as e:
            logger.warning(f"Text editor initialization failed: {e}")
            self.text_editor_enabled = False

    async def _initialize_web_search(self):
        """Initialize web search capabilities"""
        try:
            self.web_search_enabled = True
            self.tools["web_search"] = self._handle_web_search
            logger.info("Web search capabilities enabled")
        except Exception as e:
            logger.warning(f"Web search initialization failed: {e}")
            self.web_search_enabled = False

    async def _load_persistent_memory(self):
        """Load persistent memory from storage"""
        memory_file = Path.home() / ".claude_memory.json"
        try:
            if memory_file.exists():
                with open(memory_file, 'r') as f:
                    self.memory_store = json.load(f)
                logger.info(f"Loaded {len(self.memory_store)} memory entries")
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")
            self.memory_store = {}

    async def _save_persistent_memory(self):
        """Save memory to persistent storage"""
        memory_file = Path.home() / ".claude_memory.json"
        try:
            with open(memory_file, 'w') as f:
                json.dump(self.memory_store, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def encode_image(self, image_path: str) -> Dict[str, Any]:
        """Encode image for Claude's multi-modal input"""
        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"

                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": encoded
                    }
                }
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

    def create_multimodal_content(self, text: str, images: List[str] = None) -> List[Dict[str, Any]]:
        """Create multi-modal content for Claude"""
        content = []

        # Add images first (Claude's preferred order)
        if images:
            for image_path in images:
                content.append(self.encode_image(image_path))

        # Add text content
        if text:
            content.append({
                "type": "text",
                "text": text
            })

        return content

    async def take_screenshot(self) -> str:
        """Take screenshot for computer use"""
        try:
            timestamp = int(time.time())
            screenshot_path = self.screenshot_dir / f"screenshot_{timestamp}.png"

            screenshot = pyautogui.screenshot()
            screenshot.save(screenshot_path)

            return str(screenshot_path)
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate response with Claude's advanced capabilities"""
        start_time = time.time()
        request_id = hashlib.md5(f"{request.prompt[:100]}{start_time}".encode()).hexdigest()

        # Wait for rate limit
        estimated_tokens = len(str(request.prompt).split()) * 1.3
        while not await self.rate_limiter.acquire(int(estimated_tokens)):
            await asyncio.sleep(1)

        try:
            # Prepare messages with multi-modal support
            messages = self._prepare_messages(request)

            # Prepare tools
            tools = self._prepare_tools(request)

            # Handle autonomous session if requested
            if hasattr(request, 'autonomous_mode') and request.autonomous_mode:
                return await self._run_autonomous_session(request, messages, tools, request_id)

            # Standard interaction with tool support
            response = await self._chat_with_tools(messages, tools, request)

            latency_ms = int((time.time() - start_time) * 1000)
            return self._create_response(response, request_id, latency_ms)

        except Exception as e:
            logger.error(f"Enhanced Anthropic API error: {e}")
            raise

    def _prepare_messages(self, request: ModelRequest) -> List[Dict[str, Any]]:
        """Prepare messages for Claude API with multi-modal support"""
        messages = []

        # Add context from memory if relevant
        context = self._retrieve_relevant_context(request.prompt)
        if context:
            messages.append({
                "role": "user",
                "content": f"Relevant context from previous interactions:\n{context}\n\nCurrent request:"
            })

        # Handle multi-modal content
        if hasattr(request, 'images') and request.images:
            content = self.create_multimodal_content(request.prompt, request.images)
            messages.append({
                "role": "user",
                "content": content
            })
        else:
            messages.append({
                "role": "user",
                "content": request.prompt
            })

        return messages

    def _prepare_tools(self, request: ModelRequest) -> List[Dict[str, Any]]:
        """Prepare tools for Claude API"""
        tools = []

        # Computer use tool
        if (hasattr(request, 'enable_computer_use') and request.enable_computer_use
            and self.computer_use_enabled):
            tools.append({
                "type": "computer_20241022",
                "name": "computer",
                "display_width_px": self.screen_size.width if self.screen_size else 1920,
                "display_height_px": self.screen_size.height if self.screen_size else 1080
            })

        # Text editor tool
        if (hasattr(request, 'enable_text_editor') and request.enable_text_editor
            and self.text_editor_enabled):
            tools.append({
                "type": "str_replace_editor_tool",
                "name": "str_replace_editor"
            })

        # Web search tool
        if (hasattr(request, 'enable_web_search') and request.enable_web_search
            and self.web_search_enabled):
            tools.append({
                "type": "web_search_tool",
                "name": "web_search"
            })

        # Custom functions
        if hasattr(request, 'custom_tools') and request.custom_tools:
            for tool in request.custom_tools:
                tools.append(tool)

        return tools

    async def _chat_with_tools(self, messages: List[Dict[str, Any]],
                             tools: List[Dict[str, Any]],
                             request: ModelRequest) -> Dict[str, Any]:
        """Chat with Claude using tools"""
        payload = {
            "model": self.config.model_name,
            "max_tokens": min(request.max_tokens or 4000, 4000),  # Claude's max
            "temperature": request.temperature,
            "messages": messages
        }

        if tools:
            payload["tools"] = tools

        # Add system prompt for tool usage guidance
        if tools:
            system_prompt = self._create_system_prompt_for_tools(tools)
            payload["system"] = system_prompt

        response = await self.client.post("/v1/messages", json=payload)
        response.raise_for_status()
        data = response.json()

        # Handle tool use in response
        if self._has_tool_use(data):
            return await self._handle_tool_use_response(data, messages, tools, request)

        return data

    def _create_system_prompt_for_tools(self, tools: List[Dict[str, Any]]) -> str:
        """Create system prompt based on available tools"""
        prompt_parts = [
            "You are Claude, an AI assistant with advanced capabilities.",
            "You have access to the following tools:"
        ]

        for tool in tools:
            if tool["type"] == "computer_20241022":
                prompt_parts.append(
                    "- Computer use: Take screenshots, click, type, scroll, and interact with the desktop"
                )
            elif tool["type"] == "str_replace_editor_tool":
                prompt_parts.append(
                    "- Text editor: Create, view, and edit files with precise string replacement"
                )
            elif tool["type"] == "web_search_tool":
                prompt_parts.append(
                    "- Web search: Search for current information on the internet"
                )

        prompt_parts.extend([
            "",
            "Use these tools when helpful to complete the user's request.",
            "Always explain what you're doing when using tools.",
            "Be precise and methodical in your approach."
        ])

        return "\n".join(prompt_parts)

    def _has_tool_use(self, response_data: Dict[str, Any]) -> bool:
        """Check if response contains tool use"""
        content = response_data.get("content", [])
        return any(block.get("type") == "tool_use" for block in content)

    async def _handle_tool_use_response(self, response_data: Dict[str, Any],
                                      messages: List[Dict[str, Any]],
                                      tools: List[Dict[str, Any]],
                                      request: ModelRequest) -> Dict[str, Any]:
        """Handle tool use in Claude's response"""
        assistant_message = {
            "role": "assistant",
            "content": response_data["content"]
        }
        messages.append(assistant_message)

        # Execute tool calls
        tool_results = []
        for content_block in response_data["content"]:
            if content_block["type"] == "tool_use":
                result = await self._execute_tool_call(content_block)
                tool_results.append(result)

        # Add tool results to conversation
        if tool_results:
            messages.append({
                "role": "user",
                "content": tool_results
            })

            # Get follow-up response
            payload = {
                "model": self.config.model_name,
                "max_tokens": request.max_tokens or 4000,
                "temperature": request.temperature,
                "messages": messages,
                "tools": tools
            }

            response = await self.client.post("/v1/messages", json=payload)
            response.raise_for_status()
            return response.json()

        return response_data

    async def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return the result"""
        tool_name = tool_call["name"]
        tool_input = tool_call["input"]

        try:
            if tool_name == "computer":
                result = await self._handle_computer_use(**tool_input)
            elif tool_name == "str_replace_editor":
                result = await self._handle_text_editor(**tool_input)
            elif tool_name == "web_search":
                result = await self._handle_web_search(**tool_input)
            else:
                # Custom tool
                if tool_name in self.tools:
                    result = await self.tools[tool_name](**tool_input)
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

            return {
                "type": "tool_result",
                "tool_use_id": tool_call["id"],
                "content": json.dumps(result) if isinstance(result, dict) else str(result)
            }

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "type": "tool_result",
                "tool_use_id": tool_call["id"],
                "content": f"Tool execution error: {str(e)}",
                "is_error": True
            }

    async def _handle_computer_use(self, action: str, **kwargs) -> Dict[str, Any]:
        """Handle computer use commands"""
        try:
            if action == "screenshot":
                screenshot_path = await self.take_screenshot()
                with open(screenshot_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode()

                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": encoded
                    }
                }

            elif action == "click":
                coordinate = kwargs.get("coordinate")
                if coordinate:
                    pyautogui.click(coordinate[0], coordinate[1])
                    return {"status": "clicked", "coordinate": coordinate}

            elif action == "type":
                text = kwargs.get("text", "")
                pyautogui.write(text)
                return {"status": "typed", "text": text}

            elif action == "key":
                key = kwargs.get("key", "")
                pyautogui.press(key)
                return {"status": "pressed", "key": key}

            elif action == "scroll":
                direction = kwargs.get("direction", "down")
                clicks = kwargs.get("clicks", 3)
                pyautogui.scroll(-clicks if direction == "down" else clicks)
                return {"status": "scrolled", "direction": direction, "clicks": clicks}

            else:
                return {"error": f"Unknown computer action: {action}"}

        except Exception as e:
            return {"error": f"Computer use failed: {str(e)}"}

    async def _handle_text_editor(self, command: str, path: str, **kwargs) -> Dict[str, Any]:
        """Handle text editor commands"""
        try:
            file_path = Path(path)

            if command == "str_replace":
                old_str = kwargs.get("old_str", "")
                new_str = kwargs.get("new_str", "")

                if file_path.exists():
                    content = file_path.read_text()
                    if old_str in content:
                        new_content = content.replace(old_str, new_str, 1)
                        file_path.write_text(new_content)
                        return {"status": "replaced", "path": str(file_path)}
                    else:
                        return {"error": f"String not found: {old_str}"}
                else:
                    return {"error": f"File not found: {path}"}

            elif command == "create":
                file_text = kwargs.get("file_text", "")
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(file_text)
                return {"status": "created", "path": str(file_path)}

            elif command == "view":
                if file_path.exists():
                    content = file_path.read_text()
                    view_range = kwargs.get("view_range")
                    if view_range:
                        lines = content.split('\n')
                        start, end = view_range
                        content = '\n'.join(lines[start-1:end])
                    return {"content": content, "path": str(file_path)}
                else:
                    return {"error": f"File not found: {path}"}

            elif command == "list":
                if file_path.is_dir():
                    files = [str(p) for p in file_path.iterdir()]
                    return {"files": files, "path": str(file_path)}
                else:
                    return {"error": f"Not a directory: {path}"}

            else:
                return {"error": f"Unknown editor command: {command}"}

        except Exception as e:
            return {"error": f"Text editor failed: {str(e)}"}

    async def _handle_web_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Handle web search requests"""
        try:
            # This is a placeholder - in production, integrate with actual search API
            max_results = kwargs.get("max_results", 10)
            recency_filter = kwargs.get("recency_filter")

            # Store query in memory for context
            self.memory_store[f"search_{int(time.time())}"] = {
                "query": query,
                "timestamp": datetime.now().isoformat()
            }

            return {
                "query": query,
                "results": f"Web search for '{query}' would return {max_results} results",
                "recency_filter": recency_filter,
                "status": "completed"
            }

        except Exception as e:
            return {"error": f"Web search failed: {str(e)}"}

    async def _run_autonomous_session(self, request: ModelRequest,
                                    messages: List[Dict[str, Any]],
                                    tools: List[Dict[str, Any]],
                                    request_id: str) -> ModelResponse:
        """Run an autonomous coding session"""
        session_id = str(uuid.uuid4())
        session = AutonomousSession(
            session_id=session_id,
            start_time=datetime.now(),
            max_duration=timedelta(hours=getattr(request, 'max_hours', 7)),
            current_task=request.prompt,
            completed_tasks=[],
            active_files=[],
            memory={}
        )

        self.autonomous_sessions[session_id] = session

        try:
            # Enhanced system prompt for autonomous operation
            autonomous_system_prompt = """
            You are Claude in autonomous mode. You can work independently for extended periods.

            Key principles:
            1. Break down complex tasks into smaller, manageable steps
            2. Use tools methodically and explain your actions
            3. Maintain detailed memory of your progress
            4. Handle errors gracefully and adapt your approach
            5. Provide regular status updates
            6. Ask for clarification when needed

            Work systematically and persistently towards the goal.
            """

            # Add autonomous context to messages
            enhanced_messages = messages.copy()
            enhanced_messages.insert(0, {
                "role": "user",
                "content": f"AUTONOMOUS SESSION - {autonomous_system_prompt}\n\nTask: {request.prompt}"
            })

            # Execute autonomous workflow
            response = await self._chat_with_tools(enhanced_messages, tools, request)

            # Update session memory
            session.completed_tasks.append(request.prompt)
            await self._save_session_state(session)

            latency_ms = int((time.time() - time.time()) * 1000)  # Placeholder
            return self._create_response(response, request_id, latency_ms)

        except Exception as e:
            logger.error(f"Autonomous session failed: {e}")
            raise

    def _retrieve_relevant_context(self, prompt: str) -> str:
        """Retrieve relevant context from memory"""
        # Simple keyword-based retrieval
        relevant_entries = []
        prompt_lower = prompt.lower()

        for key, value in self.memory_store.items():
            if isinstance(value, dict) and "query" in value:
                if any(word in value["query"].lower() for word in prompt_lower.split()):
                    relevant_entries.append(value)

        if relevant_entries:
            return json.dumps(relevant_entries[-3:], indent=2)  # Last 3 relevant entries

        return ""

    async def _save_session_state(self, session: AutonomousSession):
        """Save autonomous session state"""
        session_data = {
            "session_id": session.session_id,
            "start_time": session.start_time.isoformat(),
            "current_task": session.current_task,
            "completed_tasks": session.completed_tasks,
            "active_files": session.active_files,
            "memory": session.memory
        }

        self.memory_store[f"session_{session.session_id}"] = session_data
        await self._save_persistent_memory()

    def _create_response(self, data: Dict[str, Any], request_id: str, latency_ms: int) -> ModelResponse:
        """Create ModelResponse from Claude API response"""
        content_blocks = data.get("content", [])

        # Extract text content
        text_content = []
        tool_calls = []

        for block in content_blocks:
            if block["type"] == "text":
                text_content.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append(block)

        content = "\n".join(text_content)
        usage = data.get("usage", {})

        # Calculate cost
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cost = (input_tokens * self.config.cost_per_1m_input / 1_000_000 +
               output_tokens * self.config.cost_per_1m_output / 1_000_000)

        metadata = {
            "stop_reason": data.get("stop_reason"),
            "stop_sequence": data.get("stop_sequence"),
            "tool_calls": tool_calls,
            "model_version": data.get("model")
        }

        return ModelResponse(
            content=content,
            model_used=self.config.model_name,
            provider=self.config.provider,
            tokens_used={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            cost_estimate=cost,
            latency_ms=latency_ms,
            request_id=request_id,
            metadata=metadata
        )

    async def start_autonomous_session(self, task: str, max_hours: int = 7) -> str:
        """Start a new autonomous coding session"""
        session_id = str(uuid.uuid4())
        session = AutonomousSession(
            session_id=session_id,
            start_time=datetime.now(),
            max_duration=timedelta(hours=max_hours),
            current_task=task,
            completed_tasks=[],
            active_files=[],
            memory={}
        )

        self.autonomous_sessions[session_id] = session
        await self._save_session_state(session)

        logger.info(f"Started autonomous session {session_id} for task: {task}")
        return session_id

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of an autonomous session"""
        session = self.autonomous_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        elapsed = datetime.now() - session.start_time
        remaining = session.max_duration - elapsed

        return {
            "session_id": session_id,
            "status": "active" if remaining.total_seconds() > 0 else "expired",
            "elapsed_hours": elapsed.total_seconds() / 3600,
            "remaining_hours": max(0, remaining.total_seconds() / 3600),
            "current_task": session.current_task,
            "completed_tasks": len(session.completed_tasks),
            "active_files": len(session.active_files)
        }

    async def health_check(self) -> bool:
        """Enhanced health check with capability verification"""
        try:
            # Basic API check
            test_request = ModelRequest(
                prompt="Test",
                task_type="general",
                complexity="simple",
                priority="speed",
                max_tokens=10
            )
            response = await self.generate_response(test_request)

            # Check capabilities
            capabilities_ok = (
                len(response.content) > 0 and
                len(self.tools) > 0 and
                self.computer_use_enabled and
                self.text_editor_enabled
            )

            return capabilities_ok

        except Exception as e:
            logger.error(f"Enhanced health check failed: {e}")
            return False

    async def close(self):
        """Close client and save state"""
        # Save memory before closing
        await self._save_persistent_memory()

        if self.client:
            await self.client.aclose()

        logger.info("Enhanced Anthropic provider closed")
