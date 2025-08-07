# Enhanced AI Providers - Advanced Capabilities Suite

A comprehensive collection of enhanced AI providers with advanced capabilities including function calling, multi-modal processing, live search, computer use, code execution, and multi-agent collaboration.

## 🚀 Overview

The Enhanced AI Providers suite extends the basic model providers with cutting-edge capabilities that leverage each AI platform's unique strengths:

- **OpenAI Enhanced**: Function calling, multi-modal processing, built-in tools, assistants API
- **Anthropic Enhanced**: Computer use, text editor, autonomous sessions, 200K context
- **xAI Enhanced**: Live search, multi-agent architecture, real-time data integration
- **Google Gemini Enhanced**: Code execution, 1M token context, multimodal analysis

## 📋 Features Matrix

| Feature | OpenAI | Anthropic | xAI | Gemini |
|---------|---------|-----------|-----|--------|
| Function Calling | ✅ | ✅ | ✅ | ✅ |
| Multi-modal (Images) | ✅ | ✅ | ✅ | ✅ |
| Multi-modal (Video) | ✅ | ❌ | ❌ | ✅ |
| Multi-modal (Audio) | ✅ | ❌ | ❌ | ✅ |
| Code Execution | ✅ | ❌ | ❌ | ✅ |
| Web Search | ✅ | ✅ | ✅ | ✅ |
| Computer Use | ❌ | ✅ | ❌ | ❌ |
| Live Search | ❌ | ❌ | ✅ | ❌ |
| Multi-Agent | ❌ | ❌ | ✅ | ❌ |
| Large Context (>100K) | ✅ | ✅ | ✅ | ✅ |
| Autonomous Sessions | ❌ | ✅ | ❌ | ❌ |
| File Upload | ✅ | ❌ | ❌ | ✅ |

## 🔧 Installation

### Prerequisites

```bash
pip install httpx tenacity pillow pyautogui psutil
```

### Environment Variables

Set up your API keys:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key" 

# xAI
export XAI_API_KEY="your-xai-api-key"

# Google Gemini
export GOOGLE_API_KEY="your-google-api-key"
# or
export GEMINI_API_KEY="your-gemini-api-key"
```

### Basic Setup

```python
from aider.models import (
    EnhancedOpenAIProvider,
    EnhancedAnthropicProvider, 
    EnhancedXAIProvider,
    EnhancedGeminiProvider,
    ModelConfig
)

# Initialize providers
providers = {}

# OpenAI Enhanced
openai_config = ModelConfig(
    provider="openai",
    model_name="gpt-4-1106-preview",
    api_base="https://api.openai.com",
    cost_per_1m_input=10.00,
    cost_per_1m_output=30.00,
    context_window=128000,
    rate_limit_rpm=60,
    rate_limit_tpm=60000,
    strengths=["function_calling", "multimodal", "tools"],
    supported_tasks=["code_generation", "analysis"]
)

providers["openai"] = EnhancedOpenAIProvider(openai_config)
await providers["openai"].initialize()
```

## 🎯 Usage Examples

### OpenAI Enhanced Provider

#### Function Calling with Structured Outputs

```python
from aider.models import EnhancedOpenAIProvider, ModelRequest, FunctionDefinition

# Define a custom function
def get_weather(location: str, unit: str = "celsius"):
    return {
        "location": location,
        "temperature": 22,
        "condition": "sunny",
        "unit": unit
    }

# Create function definition
weather_function = FunctionDefinition(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    },
    strict=True
)

# Register function
provider.register_function("get_weather", "Get weather", 
                         weather_function.parameters, get_weather)

# Use in request
request = ModelRequest(
    prompt="What's the weather in Paris?",
    task_type="general",
    complexity="simple",
    priority="quality"
)
request.functions = [weather_function]

response = await provider.generate_response(request)
```

#### Multi-modal Processing

```python
# Create multi-modal request
request = ModelRequest(
    prompt="Analyze this image and describe what you see",
    task_type="analysis", 
    complexity="medium",
    priority="quality"
)
request.images = ["path/to/image.jpg"]
request.audio_path = "path/to/audio.wav"

response = await provider.generate_response(request)
```

#### Built-in Tools

```python
request = ModelRequest(
    prompt="Search for AI news and write code to analyze trends",
    task_type="analysis",
    complexity="complex", 
    priority="quality"
)
request.enable_web_search = True
request.enable_code_interpreter = True
request.enable_file_search = True

response = await provider.generate_response(request)
```

### Anthropic Enhanced Provider

#### Computer Use

```python
request = ModelRequest(
    prompt="Take a screenshot and click on the search button",
    task_type="analysis",
    complexity="simple",
    priority="speed"
)
request.enable_computer_use = True

response = await provider.generate_response(request)
```

#### Text Editor Tool

```python
request = ModelRequest(
    prompt="Create a Python web scraper and save it to scraper.py",
    task_type="code_generation",
    complexity="medium", 
    priority="quality"
)
request.enable_text_editor = True

response = await provider.generate_response(request)
```

#### Autonomous Sessions

```python
# Start autonomous session
session_id = await provider.start_autonomous_session(
    "Create a complete REST API with authentication",
    max_hours=2
)

# Check status
status = await provider.get_session_status(session_id)
print(f"Session progress: {status}")
```

### xAI Enhanced Provider

#### Live Search

```python
# Configure search sources
request = ModelRequest(
    prompt="What are recent developments in quantum computing?",
    task_type="analysis",
    complexity="medium",
    priority="quality"
)
request.enable_live_search = True
request.search_mode = "on"
request.max_search_results = 20
request.enable_web_search = True
request.enable_x_search = True  
request.enable_news_search = True

response = await provider.generate_response(request)
print(f"Sources used: {response.metadata['sources_used']}")
```

#### Multi-Agent Collaboration

```python
request = ModelRequest(
    prompt="Analyze ML frameworks and recommend the best for startups",
    task_type="analysis",
    complexity="expert",
    priority="quality"
)
request.use_multi_agent = True
request.agents_count = 3
request.coordination_strategy = "collaborative"
request.agent_roles = ["researcher", "analyst", "business_advisor"]

response = await provider.generate_response(request)
```

#### Real-time Data

```python
# Get trending topics
trending = await provider._get_trending_topics()
print(f"Trending: {trending}")

# Get live events
events = await provider._get_live_events()
print(f"Live events: {events}")
```

### Google Gemini Enhanced Provider

#### Code Execution

```python
request = ModelRequest(
    prompt="Write Python code to analyze data and create visualizations",
    task_type="code_generation",
    complexity="complex",
    priority="quality"
)
request.enable_code_execution = True

response = await provider.generate_response(request)
```

#### Large Context Processing

```python
# Test with large document
large_document = "Very large text content..." * 10000

request = ModelRequest(
    prompt=f"Analyze this document: {large_document}",
    task_type="analysis",
    complexity="complex",
    priority="quality"
)

# Count tokens first
token_count = await provider.count_tokens(request.prompt)
print(f"Tokens: {token_count}")

response = await provider.generate_response(request)
```

#### File Upload and Processing

```python
# Upload file
file_uri = await provider.upload_file(
    "document.pdf", 
    display_name="Analysis Document"
)

# Use in request
request = ModelRequest(
    prompt="Analyze the uploaded document",
    task_type="analysis",
    complexity="medium",
    priority="quality"  
)
request.files = [file_uri]

response = await provider.generate_response(request)
```

#### Google Search Integration

```python
request = ModelRequest(
    prompt="Search for recent LLM research and summarize findings",
    task_type="analysis", 
    complexity="medium",
    priority="quality"
)
request.enable_google_search = True

response = await provider.generate_response(request)
```

## ⚙️ Configuration

### Model Selection

Each provider supports multiple model variants:

```python
# OpenAI models
"gpt-4-1106-preview"  # Function calling, 128K context
"gpt-4o"              # Multi-modal, latest
"o1-preview"          # Advanced reasoning

# Anthropic models  
"claude-3-5-sonnet-20241022"  # Latest, 200K context
"claude-3-opus-20240229"      # Most capable
"claude-3-haiku-20240307"     # Fastest

# xAI models
"grok-4"              # Standard
"grok-4-heavy"        # Multi-agent
"grok-4-code"         # Code-specialized

# Gemini models
"gemini-2.5-pro"      # 1M context, multimodal
"gemini-2.5-flash"    # Fast, cost-effective
"gemini-2.0-flash-exp" # Experimental
```

### Safety Settings (Gemini)

```python
from aider.models import GeminiSafetySettings, HarmBlockThreshold

safety_settings = [
    GeminiSafetySettings(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_MEDIUM_AND_ABOVE"
    ),
    GeminiSafetySettings(
        category="HARM_CATEGORY_HATE_SPEECH", 
        threshold="BLOCK_MEDIUM_AND_ABOVE"
    )
]

provider.set_safety_settings(safety_settings)
```

### Rate Limiting

```python
# Customize rate limits
config = ModelConfig(
    # ... other config
    rate_limit_rpm=100,    # Requests per minute
    rate_limit_tpm=50000,  # Tokens per minute
)
```

## 📊 Performance Monitoring

### Health Checks

```python
# Check individual provider
is_healthy = await provider.health_check()
print(f"Provider healthy: {is_healthy}")

# Get capabilities
capabilities = await provider.get_capabilities()
print(f"Capabilities: {capabilities}")
```

### Usage Metrics

```python
# Get response metadata
response = await provider.generate_response(request)

print(f"Tokens used: {response.tokens_used}")
print(f"Cost: ${response.cost_estimate:.4f}")
print(f"Latency: {response.latency_ms}ms")
print(f"Model: {response.model_used}")
print(f"Metadata: {response.metadata}")
```

### Cost Optimization

```python
# Token counting (Gemini)
token_count = await gemini_provider.count_tokens(text)
print(f"Estimated tokens: {token_count}")

# Search cost tracking (xAI)
sources_used = response.metadata.get("sources_used", 0)
search_cost = sources_used * 0.025  # $0.025 per source
print(f"Search cost: ${search_cost:.4f}")
```

## 🔍 Troubleshooting

### Common Issues

#### Authentication Errors
```python
# Verify API keys are set
import os
print(f"OpenAI key: {'✓' if os.getenv('OPENAI_API_KEY') else '✗'}")
print(f"Anthropic key: {'✓' if os.getenv('ANTHROPIC_API_KEY') else '✗'}")
print(f"xAI key: {'✓' if os.getenv('XAI_API_KEY') else '✗'}")
print(f"Google key: {'✓' if os.getenv('GOOGLE_API_KEY') else '✗'}")
```

#### Rate Limiting
```python
# Check rate limiter status
rate_limiter = provider.rate_limiter
can_proceed = await rate_limiter.acquire(1000)  # Check for 1000 tokens
print(f"Rate limit OK: {can_proceed}")
```

#### Computer Use Issues (Anthropic)
```python
# Check if computer use is available
if provider.computer_use_enabled:
    print("Computer use available")
    print(f"Screen size: {provider.screen_size}")
else:
    print("Computer use not available - check dependencies")
```

### Error Handling

```python
from tenacity import RetryError
from httpx import HTTPStatusError

try:
    response = await provider.generate_response(request)
except HTTPStatusError as e:
    print(f"HTTP error: {e.response.status_code}")
except RetryError as e:
    print(f"Retry attempts exhausted: {e}")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 🧪 Testing

### Run Demo Suite

```python
from aider.models.enhanced_providers_demo import EnhancedProvidersDemo

demo = EnhancedProvidersDemo()
report = await demo.run_full_demo()
```

### Unit Tests

```python
import pytest

@pytest.mark.asyncio
async def test_openai_function_calling():
    provider = EnhancedOpenAIProvider(config)
    await provider.initialize()
    
    # Test function calling
    response = await provider.generate_response(request)
    assert len(response.content) > 0
    assert response.metadata["function_calls"] is not None
```

## 🔗 API Reference

### Base Classes

#### ModelProvider
- `initialize()` - Initialize provider
- `generate_response(request)` - Generate AI response  
- `health_check()` - Check provider health
- `close()` - Close connections

#### ModelRequest
- `prompt: str` - Input prompt
- `task_type: TaskType` - Type of task
- `complexity: ComplexityLevel` - Task complexity
- `priority: Priority` - Response priority
- `max_tokens: int` - Maximum output tokens
- `temperature: float` - Generation temperature

#### ModelResponse  
- `content: str` - Generated content
- `model_used: str` - Model that generated response
- `tokens_used: Dict` - Token usage breakdown
- `cost_estimate: float` - Estimated cost
- `latency_ms: int` - Response latency
- `metadata: Dict` - Additional metadata

### Provider-Specific Methods

#### EnhancedOpenAIProvider
- `register_function(name, description, parameters, handler)`
- `create_multimodal_message(text, images, audio)`
- `create_assistant(name, instructions, tools)`
- `run_assistant_thread(assistant_id, message)`

#### EnhancedAnthropicProvider  
- `take_screenshot()` - Capture screen
- `start_autonomous_session(task, max_hours)` - Start autonomous work
- `get_session_status(session_id)` - Check session status

#### EnhancedXAIProvider
- `create_live_search_params(mode, max_results, sources)`
- `create_multi_agent_team(description, count, strategy)`
- `create_web_search_source(country, excluded_sites)`

#### EnhancedGeminiProvider
- `count_tokens(text)` - Count tokens in text
- `upload_file(path, mime_type)` - Upload file
- `embed_content(text, task_type)` - Generate embeddings
- `create_cached_content(contents)` - Cache content

## 📈 Roadmap

### Planned Features

- [ ] **Multi-provider orchestration** - Automatic provider selection and fallback
- [ ] **Cost optimization** - Smart routing based on cost/quality tradeoffs  
- [ ] **Persistent memory** - Cross-session context retention
- [ ] **Custom tool marketplace** - Community-contributed tools
- [ ] **Advanced monitoring** - Real-time performance dashboards
- [ ] **Batch processing** - Efficient bulk request handling
- [ ] **Edge deployment** - Local model integration

### Version History

- **v3.0.0** - Enhanced providers with advanced capabilities
- **v2.1.0** - Basic multi-provider support
- **v2.0.0** - ModelManager architecture
- **v1.0.0** - Initial provider implementations

## 🤝 Contributing

### Development Setup

```bash
git clone <repository>
cd aider/models
pip install -e .[dev]
```

### Adding New Capabilities

1. Extend the appropriate enhanced provider class
2. Add configuration options to ModelConfig
3. Update the routing rules in ModelManager
4. Add tests and documentation
5. Update the demo script

### Guidelines

- Follow existing patterns for consistency
- Add comprehensive error handling
- Include usage examples in docstrings
- Update capability matrices
- Test with multiple models

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check the inline docstrings and examples
- **Issues**: Report bugs and request features via GitHub issues
- **Community**: Join our Discord for discussions and help
- **Enterprise**: Contact us for enterprise support and custom integrations

---

**Made with ❤️ by the Aider AI Team**

*Enhancing AI capabilities for developers worldwide*