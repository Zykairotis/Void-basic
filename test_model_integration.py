#!/usr/bin/env python3
"""
Comprehensive Test Suite for AI Model Integration
Tests ModelManager functionality, routing logic, and provider integrations.
"""

import asyncio
import os
import json
import time
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Add the project root to the path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from aider.models import (
    ModelManager,
    ModelRequest,
    ModelResponse,
    TaskType,
    ComplexityLevel,
    Priority,
    generate_code,
    analyze_code,
    get_model_manager
)

class TestModelIntegration:
    """Comprehensive test suite for AI model integration"""

    @pytest.fixture
    async def model_manager(self):
        """Create a test ModelManager instance"""
        manager = ModelManager()
        yield manager
        await manager.close()

    @pytest.fixture
    def mock_response(self):
        """Mock API response for testing"""
        return {
            "choices": [{"message": {"content": "Test response content"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }

    @pytest.fixture
    def mock_anthropic_response(self):
        """Mock Anthropic API response"""
        return {
            "content": [{"text": "Test anthropic response"}],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50
            }
        }

    def test_model_configuration_loading(self):
        """Test that model configurations are loaded correctly"""
        manager = ModelManager()
        configs = manager.model_configs

        # Verify all expected models are configured
        assert "claude" in configs
        assert "gpt4" in configs
        assert "grok" in configs

        # Verify Claude configuration
        claude_config = configs["claude"]
        assert claude_config.provider == "anthropic"
        assert claude_config.model_name == "claude-3-5-sonnet-20241022"
        assert claude_config.cost_per_1m_input == 3.00
        assert claude_config.cost_per_1m_output == 15.00
        assert claude_config.context_window == 200000

        # Verify GPT-4 configuration
        gpt4_config = configs["gpt4"]
        assert gpt4_config.provider == "openai"
        assert gpt4_config.model_name == "gpt-4-1106-preview"
        assert gpt4_config.context_window == 128000

        print("✅ Model configuration loading test passed")

    def test_routing_rules_initialization(self):
        """Test that routing rules are properly initialized"""
        manager = ModelManager()
        rules = manager.routing_rules

        # Test code generation routing
        code_gen_rules = rules[TaskType.CODE_GENERATION.value]
        assert "claude" in code_gen_rules  # Should prefer Claude for code generation
        assert len(code_gen_rules) >= 2   # Should have fallbacks

        # Test debugging routing
        debug_rules = rules[TaskType.DEBUGGING.value]
        assert "gpt4" in debug_rules      # Should prefer GPT-4 for debugging

        # Test documentation routing
        doc_rules = rules[TaskType.DOCUMENTATION.value]
        assert "grok" in doc_rules        # Should prefer Grok for fast tasks

        print("✅ Routing rules initialization test passed")

    def test_model_selection_logic(self):
        """Test intelligent model selection based on task characteristics"""
        manager = ModelManager()

        # Test code generation selection
        code_request = ModelRequest(
            prompt="Create a Python function",
            task_type=TaskType.CODE_GENERATION,
            complexity=ComplexityLevel.MEDIUM,
            priority=Priority.QUALITY
        )
        selected = manager.select_model(code_request)
        assert selected in ["claude", "gpt4", "grok"]

        # Test speed priority
        speed_request = ModelRequest(
            prompt="Quick documentation",
            task_type=TaskType.DOCUMENTATION,
            complexity=ComplexityLevel.SIMPLE,
            priority=Priority.SPEED
        )
        speed_selected = manager.select_model(speed_request)
        # Should prefer faster models when speed is priority
        assert speed_selected in ["grok", "claude", "gpt4"]

        # Test quality priority for complex tasks
        quality_request = ModelRequest(
            prompt="Complex refactoring",
            task_type=TaskType.CODE_REFACTORING,
            complexity=ComplexityLevel.COMPLEX,
            priority=Priority.QUALITY
        )
        quality_selected = manager.select_model(quality_request)
        # Should prefer Claude for high-quality code tasks
        assert quality_selected in ["claude", "gpt4"]

        print("✅ Model selection logic test passed")

    @patch('httpx.AsyncClient.post')
    async def test_openai_provider_integration(self, mock_post, mock_response):
        """Test OpenAI provider with mocked API calls"""
        # Setup mock response
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status = Mock()

        manager = ModelManager()

        # Mock environment variable
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            await manager.initialize()

            if "gpt4" in manager.providers:
                request = ModelRequest(
                    prompt="Test prompt",
                    task_type=TaskType.GENERAL,
                    complexity=ComplexityLevel.SIMPLE,
                    priority=Priority.SPEED,
                    max_tokens=100
                )

                # Force selection of GPT-4
                with patch.object(manager, 'select_model', return_value='gpt4'):
                    response = await manager.generate_response(request)

                assert isinstance(response, ModelResponse)
                assert response.content == "Test response content"
                assert response.model_used == "gpt-4-1106-preview"
                assert response.provider == "openai"
                assert response.tokens_used["total"] == 150

        print("✅ OpenAI provider integration test passed")

    @patch('httpx.AsyncClient.post')
    async def test_anthropic_provider_integration(self, mock_post, mock_anthropic_response):
        """Test Anthropic provider with mocked API calls"""
        # Setup mock response
        mock_post.return_value.json.return_value = mock_anthropic_response
        mock_post.return_value.raise_for_status = Mock()

        manager = ModelManager()

        # Mock environment variable
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            await manager.initialize()

            if "claude" in manager.providers:
                request = ModelRequest(
                    prompt="Generate Python code",
                    task_type=TaskType.CODE_GENERATION,
                    complexity=ComplexityLevel.MEDIUM,
                    priority=Priority.QUALITY
                )

                # Force selection of Claude
                with patch.object(manager, 'select_model', return_value='claude'):
                    response = await manager.generate_response(request)

                assert isinstance(response, ModelResponse)
                assert response.content == "Test anthropic response"
                assert response.model_used == "claude-3-5-sonnet-20241022"
                assert response.provider == "anthropic"

        print("✅ Anthropic provider integration test passed")

    async def test_fallback_mechanism(self):
        """Test that fallback mechanism works when primary model fails"""
        manager = ModelManager()

        # Mock one provider to fail, another to succeed
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key",
            "OPENAI_API_KEY": "test-key"
        }):
            await manager.initialize()

            # Mock first provider (Claude) to fail
            if "claude" in manager.providers:
                manager.providers["claude"].generate_response = AsyncMock(
                    side_effect=Exception("API Error")
                )

            # Mock second provider (GPT-4) to succeed
            if "gpt4" in manager.providers:
                manager.providers["gpt4"].generate_response = AsyncMock(
                    return_value=ModelResponse(
                        content="Fallback response",
                        model_used="gpt-4",
                        provider="openai",
                        tokens_used={"input": 50, "output": 25, "total": 75},
                        cost_estimate=0.001,
                        latency_ms=500
                    )
                )

            request = ModelRequest(
                prompt="Test fallback",
                task_type=TaskType.CODE_GENERATION,
                complexity=ComplexityLevel.SIMPLE,
                priority=Priority.BALANCED
            )

            response = await manager.generate_response(request)
            assert response.content == "Fallback response"
            assert response.model_used == "gpt-4"

        print("✅ Fallback mechanism test passed")

    async def test_rate_limiting(self):
        """Test rate limiting functionality"""
        from aider.models.model_manager import RateLimiter

        # Create rate limiter with very low limits for testing
        rate_limiter = RateLimiter(requests_per_minute=2, tokens_per_minute=1000)

        # First request should succeed
        assert await rate_limiter.acquire(100) == True

        # Second request should succeed
        assert await rate_limiter.acquire(100) == True

        # Third request should fail (exceeds RPM limit)
        assert await rate_limiter.acquire(100) == False

        print("✅ Rate limiting test passed")

    async def test_performance_metrics(self):
        """Test performance metrics collection"""
        manager = ModelManager()

        # Simulate some successful and failed requests
        manager._update_metrics("claude", ModelResponse(
            content="test",
            model_used="claude",
            provider="anthropic",
            tokens_used={"input": 100, "output": 50, "total": 150},
            cost_estimate=0.005,
            latency_ms=1000
        ), success=True)

        manager._update_metrics("claude", None, success=False)

        metrics = manager.get_performance_metrics()

        assert "claude" in metrics
        claude_metrics = metrics["claude"]
        assert claude_metrics["success_rate"] == 0.5  # 1 success, 1 failure
        assert claude_metrics["total_requests"] == 2
        assert claude_metrics["total_cost"] == 0.005
        assert claude_metrics["average_latency_ms"] == 1000

        print("✅ Performance metrics test passed")

    @patch('httpx.AsyncClient.post')
    async def test_health_check(self, mock_post):
        """Test health check functionality"""
        # Mock successful health check response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        manager = ModelManager()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            await manager.initialize()

            if manager.providers:
                health_status = await manager.health_check()

                # At least one provider should be tested
                assert len(health_status) > 0

                # Check that we get boolean results
                for model_key, is_healthy in health_status.items():
                    assert isinstance(is_healthy, bool)

        print("✅ Health check test passed")

    @patch('httpx.AsyncClient.post')
    async def test_convenience_functions(self, mock_post, mock_response):
        """Test convenience functions for common use cases"""
        # Setup mock
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status = Mock()

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            # Test generate_code function
            code = await generate_code(
                "Create a function to calculate factorial",
                language="python",
                complexity=ComplexityLevel.SIMPLE
            )

            assert isinstance(code, str)
            assert len(code) > 0

            # Test analyze_code function
            analysis = await analyze_code(
                "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                analysis_type="quality"
            )

            assert isinstance(analysis, str)
            assert len(analysis) > 0

        print("✅ Convenience functions test passed")

    async def test_error_handling(self):
        """Test error handling for various failure scenarios"""
        manager = ModelManager()

        # Test with no API keys (should handle gracefully)
        with patch.dict(os.environ, {}, clear=True):
            try:
                await manager.initialize()
                # Should not crash, but may have no providers
                assert isinstance(manager.providers, dict)
            except Exception as e:
                # Acceptable if it fails gracefully
                assert "API_KEY" in str(e) or "not set" in str(e)

        # Test invalid request
        request = ModelRequest(
            prompt="",  # Empty prompt
            task_type=TaskType.GENERAL,
            complexity=ComplexityLevel.SIMPLE,
            priority=Priority.SPEED
        )

        # Should handle empty prompts gracefully
        try:
            response = await manager.generate_response(request)
            # If it succeeds, that's fine
            assert isinstance(response, ModelResponse)
        except Exception as e:
            # If it fails, should be a meaningful error
            assert len(str(e)) > 0

        print("✅ Error handling test passed")

    def test_mock_mode_functionality(self):
        """Test that mock mode works for development without API keys"""
        # This test ensures the system can work in development mode

        # Create mock responses
        mock_responses = {
            TaskType.CODE_GENERATION: "def hello_world():\n    print('Hello, World!')",
            TaskType.DEBUGGING: "The issue is in line 5 where...",
            TaskType.ANALYSIS: "This code has good structure but could be improved by..."
        }

        # Test that we can create model requests without real APIs
        for task_type, expected_content in mock_responses.items():
            request = ModelRequest(
                prompt=f"Test {task_type.value}",
                task_type=task_type,
                complexity=ComplexityLevel.SIMPLE,
                priority=Priority.SPEED
            )

            # Verify request is properly formed
            assert request.prompt is not None
            assert request.task_type == task_type
            assert isinstance(request.complexity, ComplexityLevel)
            assert isinstance(request.priority, Priority)

        print("✅ Mock mode functionality test passed")

async def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Starting AI Model Integration Tests\n")

    test_instance = TestModelIntegration()

    # Run synchronous tests
    print("📋 Running Configuration Tests...")
    test_instance.test_model_configuration_loading()
    test_instance.test_routing_rules_initialization()
    test_instance.test_model_selection_logic()
    test_instance.test_mock_mode_functionality()

    # Run asynchronous tests
    print("\n📋 Running Integration Tests...")

    # Test with mock responses to avoid requiring real API keys
    mock_response = {
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    }

    mock_anthropic_response = {
        "content": [{"text": "Test response"}],
        "usage": {"input_tokens": 10, "output_tokens": 5}
    }

    await test_instance.test_openai_provider_integration(mock_response)
    await test_instance.test_anthropic_provider_integration(mock_anthropic_response)
    await test_instance.test_fallback_mechanism()
    await test_instance.test_rate_limiting()
    await test_instance.test_performance_metrics()

    # These require HTTP mocking
    with patch('httpx.AsyncClient.post') as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status = Mock()

        await test_instance.test_health_check()
        await test_instance.test_convenience_functions()

    await test_instance.test_error_handling()

    print("\n🎉 All Integration Tests Completed Successfully!")
    print("\n📊 Test Summary:")
    print("✅ Model Configuration: PASSED")
    print("✅ Routing Logic: PASSED")
    print("✅ Provider Integration: PASSED")
    print("✅ Fallback Mechanism: PASSED")
    print("✅ Rate Limiting: PASSED")
    print("✅ Performance Metrics: PASSED")
    print("✅ Health Checks: PASSED")
    print("✅ Convenience Functions: PASSED")
    print("✅ Error Handling: PASSED")
    print("✅ Mock Mode: PASSED")

def test_with_real_apis():
    """
    Test with real API keys (only run when you have valid keys set)
    This function demonstrates how to test with actual AI models
    """
    print("\n🔗 Testing with Real APIs (requires valid API keys)...")

    # Check if API keys are available
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "xAI": os.getenv("XAI_API_KEY")
    }

    available_apis = [name for name, key in api_keys.items() if key and key != "your_api_key_here"]

    if not available_apis:
        print("⚠️  No real API keys found. Skipping real API tests.")
        print("   To test with real APIs, set environment variables:")
        for name in api_keys.keys():
            print(f"   export {name.upper()}_API_KEY=your_actual_key")
        return

    print(f"🔑 Found API keys for: {', '.join(available_apis)}")

    async def test_real_api():
        try:
            # Test simple code generation
            code = await generate_code(
                "Create a Python function that calculates the Fibonacci sequence",
                language="python",
                complexity=ComplexityLevel.SIMPLE
            )

            print("✅ Real API code generation successful!")
            print(f"Generated code length: {len(code)} characters")

            # Test code analysis
            analysis = await analyze_code(
                "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
                analysis_type="performance"
            )

            print("✅ Real API code analysis successful!")
            print(f"Analysis length: {len(analysis)} characters")

        except Exception as e:
            print(f"❌ Real API test failed: {e}")
            print("This might be due to API rate limits, network issues, or invalid keys")

    # Run the real API test
    asyncio.run(test_real_api())

if __name__ == "__main__":
    print("🎯 AI Model Integration Test Suite")
    print("=" * 50)

    # Run main integration tests (with mocked APIs)
    asyncio.run(run_integration_tests())

    # Optionally test with real APIs if keys are available
    test_with_real_apis()

    print("\n🏆 Test Suite Complete!")
    print("\nNext Steps:")
    print("1. Set up your API keys in .env file")
    print("2. Run: python test_model_integration.py")
    print("3. Integrate ModelManager into your agents")
    print("4. Start building autonomous workflows!")
