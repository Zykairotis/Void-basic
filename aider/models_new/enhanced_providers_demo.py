"""
Enhanced AI Providers Demo Script
Demonstrates advanced capabilities of each AI provider with real-world examples.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from .model_manager import ModelConfig, ModelRequest, TaskType, ComplexityLevel, Priority
from .openai_enhanced_provider import EnhancedOpenAIProvider, FunctionDefinition
from .anthropic_enhanced_provider import EnhancedAnthropicProvider, ComputerCommand
from .xai_enhanced_provider import EnhancedXAIProvider, LiveSearchParams
from .gemini_enhanced_provider import EnhancedGeminiProvider, GeminiFunctionDeclaration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedProvidersDemo:
    """Demonstration of enhanced AI provider capabilities"""

    def __init__(self):
        self.providers = {}
        self.demo_results = {}

    async def setup_providers(self):
        """Initialize all enhanced providers"""
        logger.info("Setting up enhanced AI providers...")

        # OpenAI Enhanced Provider
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
            supported_tasks=[TaskType.CODE_GENERATION, TaskType.ANALYSIS]
        )

        # Anthropic Enhanced Provider
        anthropic_config = ModelConfig(
            provider="anthropic",
            model_name="claude-3-5-sonnet-20241022",
            api_base="https://api.anthropic.com",
            cost_per_1m_input=3.00,
            cost_per_1m_output=15.00,
            context_window=200000,
            rate_limit_rpm=50,
            rate_limit_tpm=40000,
            strengths=["computer_use", "text_editor", "autonomous"],
            supported_tasks=[TaskType.CODE_GENERATION, TaskType.CODE_REVIEW]
        )

        # xAI Enhanced Provider
        xai_config = ModelConfig(
            provider="xai",
            model_name="grok-4",
            api_base="https://api.x.ai",
            cost_per_1m_input=3.00,
            cost_per_1m_output=15.00,
            context_window=256000,
            rate_limit_rpm=100,
            rate_limit_tpm=80000,
            strengths=["live_search", "multi_agent", "real_time"],
            supported_tasks=[TaskType.ANALYSIS, TaskType.GENERAL]
        )

        # Google Gemini Enhanced Provider
        gemini_config = ModelConfig(
            provider="google",
            model_name="gemini-2.5-pro",
            api_base="https://generativelanguage.googleapis.com/v1beta",
            cost_per_1m_input=1.25,
            cost_per_1m_output=5.00,
            context_window=1000000,
            rate_limit_rpm=60,
            rate_limit_tpm=100000,
            strengths=["multimodal", "code_execution", "large_context"],
            supported_tasks=[TaskType.CODE_GENERATION, TaskType.DOCUMENTATION]
        )

        try:
            # Initialize providers
            self.providers["openai"] = EnhancedOpenAIProvider(openai_config)
            await self.providers["openai"].initialize()

            self.providers["anthropic"] = EnhancedAnthropicProvider(anthropic_config)
            await self.providers["anthropic"].initialize()

            self.providers["xai"] = EnhancedXAIProvider(xai_config)
            await self.providers["xai"].initialize()

            self.providers["gemini"] = EnhancedGeminiProvider(gemini_config)
            await self.providers["gemini"].initialize()

            logger.info("All enhanced providers initialized successfully!")

        except Exception as e:
            logger.error(f"Provider initialization failed: {e}")
            raise

    async def demo_openai_enhanced(self):
        """Demonstrate OpenAI enhanced capabilities"""
        logger.info("=== OpenAI Enhanced Provider Demo ===")
        provider = self.providers["openai"]

        # 1. Function Calling Demo
        logger.info("1. Function Calling with Structured Outputs")

        # Register a custom function
        def get_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
            return {
                "location": location,
                "temperature": 22,
                "condition": "sunny",
                "unit": unit,
                "timestamp": datetime.now().isoformat()
            }

        weather_function = FunctionDefinition(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        )

        provider.register_function("get_weather", "Get weather data",
                                 weather_function.parameters, get_weather)

        # Create request with function calling
        request = ModelRequest(
            prompt="What's the weather like in Paris? Please use the weather function.",
            task_type=TaskType.GENERAL,
            complexity=ComplexityLevel.SIMPLE,
            priority=Priority.QUALITY
        )
        request.functions = [weather_function]
        request.enable_function_calling = True

        try:
            response = await provider.generate_response(request)
            logger.info(f"Function calling response: {response.content[:200]}...")
            self.demo_results["openai_function_calling"] = response.metadata
        except Exception as e:
            logger.error(f"Function calling demo failed: {e}")

        # 2. Multi-modal Demo
        logger.info("2. Multi-modal Processing")
        try:
            # Create a simple test image (would normally be an actual image file)
            multimodal_request = ModelRequest(
                prompt="Analyze this image and describe what you see.",
                task_type=TaskType.ANALYSIS,
                complexity=ComplexityLevel.MEDIUM,
                priority=Priority.QUALITY
            )
            # multimodal_request.images = ["path/to/test_image.jpg"]  # Uncomment with real image

            # For demo, we'll simulate without actual image
            response = await provider.generate_response(multimodal_request)
            logger.info(f"Multi-modal analysis: {response.content[:200]}...")
            self.demo_results["openai_multimodal"] = response.tokens_used
        except Exception as e:
            logger.error(f"Multi-modal demo failed: {e}")

        # 3. Built-in Tools Demo
        logger.info("3. Built-in Tools (Web Search, Code Interpreter)")
        try:
            tools_request = ModelRequest(
                prompt="Search for recent developments in AI and write Python code to analyze the trends.",
                task_type=TaskType.ANALYSIS,
                complexity=ComplexityLevel.COMPLEX,
                priority=Priority.QUALITY
            )
            tools_request.enable_web_search = True
            tools_request.enable_code_interpreter = True

            response = await provider.generate_response(tools_request)
            logger.info(f"Tools integration: {response.content[:200]}...")
            self.demo_results["openai_tools"] = response.cost_estimate
        except Exception as e:
            logger.error(f"Tools demo failed: {e}")

    async def demo_anthropic_enhanced(self):
        """Demonstrate Anthropic enhanced capabilities"""
        logger.info("=== Anthropic Enhanced Provider Demo ===")
        provider = self.providers["anthropic"]

        # 1. Text Editor Capabilities
        logger.info("1. Text Editor Tool")
        try:
            request = ModelRequest(
                prompt="Create a Python file with a simple web scraper and then modify it to add error handling.",
                task_type=TaskType.CODE_GENERATION,
                complexity=ComplexityLevel.MEDIUM,
                priority=Priority.QUALITY
            )
            request.enable_text_editor = True

            response = await provider.generate_response(request)
            logger.info(f"Text editor demo: {response.content[:200]}...")
            self.demo_results["anthropic_text_editor"] = response.latency_ms
        except Exception as e:
            logger.error(f"Text editor demo failed: {e}")

        # 2. Computer Use (if available)
        logger.info("2. Computer Use Capabilities")
        if provider.computer_use_enabled:
            try:
                computer_request = ModelRequest(
                    prompt="Take a screenshot and describe what's currently on screen.",
                    task_type=TaskType.ANALYSIS,
                    complexity=ComplexityLevel.SIMPLE,
                    priority=Priority.SPEED
                )
                computer_request.enable_computer_use = True

                response = await provider.generate_response(computer_request)
                logger.info(f"Computer use demo: {response.content[:200]}...")
                self.demo_results["anthropic_computer_use"] = True
            except Exception as e:
                logger.error(f"Computer use demo failed: {e}")
        else:
            logger.info("Computer use not available in current environment")

        # 3. Autonomous Session Demo
        logger.info("3. Autonomous Coding Session")
        try:
            session_id = await provider.start_autonomous_session(
                "Create a complete REST API for a todo application with authentication",
                max_hours=1
            )
            logger.info(f"Started autonomous session: {session_id}")

            # Check session status
            status = await provider.get_session_status(session_id)
            logger.info(f"Session status: {status}")
            self.demo_results["anthropic_autonomous"] = status
        except Exception as e:
            logger.error(f"Autonomous session demo failed: {e}")

    async def demo_xai_enhanced(self):
        """Demonstrate xAI enhanced capabilities"""
        logger.info("=== xAI Enhanced Provider Demo ===")
        provider = self.providers["xai"]

        # 1. Live Search Demo
        logger.info("1. Live Search with Multiple Sources")
        try:
            # Create search parameters
            search_sources = [
                provider.create_web_search_source(safe_search=True),
                provider.create_x_search_source(excluded_handles=["spam_account"]),
                {"type": "news", "safe_search": True}
            ]

            request = ModelRequest(
                prompt="What are the latest developments in quantum computing?",
                task_type=TaskType.ANALYSIS,
                complexity=ComplexityLevel.MEDIUM,
                priority=Priority.QUALITY
            )
            request.enable_live_search = True
            request.search_mode = "on"
            request.max_search_results = 15
            request.web_search_config = {}
            request.x_search_config = {}
            request.enable_news_search = True

            response = await provider.generate_response(request)
            logger.info(f"Live search response: {response.content[:200]}...")
            sources_used = response.metadata.get("sources_used", 0)
            logger.info(f"Sources used: {sources_used}")
            self.demo_results["xai_live_search"] = {"sources_used": sources_used}
        except Exception as e:
            logger.error(f"Live search demo failed: {e}")

        # 2. Multi-Agent Architecture Demo
        logger.info("2. Multi-Agent Collaboration (Grok 4 Heavy)")
        try:
            multi_agent_request = ModelRequest(
                prompt="Analyze the pros and cons of different machine learning frameworks and recommend the best one for a startup.",
                task_type=TaskType.ANALYSIS,
                complexity=ComplexityLevel.EXPERT,
                priority=Priority.QUALITY
            )
            multi_agent_request.use_multi_agent = True
            multi_agent_request.agents_count = 3
            multi_agent_request.coordination_strategy = "collaborative"
            multi_agent_request.agent_roles = ["researcher", "analyst", "business_advisor"]

            response = await provider.generate_response(multi_agent_request)
            logger.info(f"Multi-agent response: {response.content[:200]}...")
            multi_agent_info = response.metadata.get("multi_agent", {})
            logger.info(f"Multi-agent info: {multi_agent_info}")
            self.demo_results["xai_multi_agent"] = multi_agent_info
        except Exception as e:
            logger.error(f"Multi-agent demo failed: {e}")

        # 3. Real-time Data Integration
        logger.info("3. Real-time Data and Trending Topics")
        try:
            realtime_request = ModelRequest(
                prompt="What are the current trending topics in technology and how do they relate to recent market movements?",
                task_type=TaskType.ANALYSIS,
                complexity=ComplexityLevel.MEDIUM,
                priority=Priority.SPEED
            )
            realtime_request.enable_live_search = True
            realtime_request.enable_x_search = True

            response = await provider.generate_response(realtime_request)
            logger.info(f"Real-time data response: {response.content[:200]}...")
            citations = response.metadata.get("citations", [])
            logger.info(f"Citations count: {len(citations)}")
            self.demo_results["xai_realtime"] = {"citations_count": len(citations)}
        except Exception as e:
            logger.error(f"Real-time data demo failed: {e}")

    async def demo_gemini_enhanced(self):
        """Demonstrate Google Gemini enhanced capabilities"""
        logger.info("=== Google Gemini Enhanced Provider Demo ===")
        provider = self.providers["gemini"]

        # 1. Code Execution Demo
        logger.info("1. Code Execution Environment")
        try:
            code_request = ModelRequest(
                prompt="Write Python code to analyze a dataset and create visualizations. Execute the code to show results.",
                task_type=TaskType.CODE_GENERATION,
                complexity=ComplexityLevel.COMPLEX,
                priority=Priority.QUALITY
            )
            code_request.enable_code_execution = True

            response = await provider.generate_response(code_request)
            logger.info(f"Code execution response: {response.content[:200]}...")
            code_executions = response.metadata.get("code_executions", [])
            logger.info(f"Code executions: {len(code_executions)}")
            self.demo_results["gemini_code_execution"] = {"executions_count": len(code_executions)}
        except Exception as e:
            logger.error(f"Code execution demo failed: {e}")

        # 2. Large Context Processing
        logger.info("2. Large Context Window (1M tokens)")
        try:
            # Generate a large context example
            large_context = "This is a large document. " * 1000  # Simulated large document

            large_context_request = ModelRequest(
                prompt=f"Analyze this large document and provide a comprehensive summary:\n\n{large_context}",
                task_type=TaskType.ANALYSIS,
                complexity=ComplexityLevel.COMPLEX,
                priority=Priority.QUALITY
            )

            # Test token counting first
            token_count = await provider.count_tokens(large_context_request.prompt)
            logger.info(f"Token count for large context: {token_count}")

            response = await provider.generate_response(large_context_request)
            logger.info(f"Large context analysis: {response.content[:200]}...")
            self.demo_results["gemini_large_context"] = token_count
        except Exception as e:
            logger.error(f"Large context demo failed: {e}")

        # 3. Multimodal with Function Calling
        logger.info("3. Multimodal Processing with Function Calling")
        try:
            # Register a custom function for image analysis
            def analyze_chart_data(chart_type: str, data_points: int) -> Dict[str, Any]:
                return {
                    "chart_type": chart_type,
                    "data_points": data_points,
                    "analysis": f"Analyzed {chart_type} chart with {data_points} data points",
                    "insights": ["Trend is upward", "Data shows seasonal variation"]
                }

            chart_function = GeminiFunctionDeclaration(
                name="analyze_chart_data",
                description="Analyze chart data and provide insights",
                parameters={
                    "type": "object",
                    "properties": {
                        "chart_type": {"type": "string"},
                        "data_points": {"type": "integer"}
                    },
                    "required": ["chart_type", "data_points"]
                }
            )

            provider.register_custom_function(
                "analyze_chart_data",
                "Analyze chart data",
                chart_function.parameters,
                analyze_chart_data
            )

            multimodal_request = ModelRequest(
                prompt="Analyze the uploaded image and if it contains a chart, use the chart analysis function.",
                task_type=TaskType.ANALYSIS,
                complexity=ComplexityLevel.MEDIUM,
                priority=Priority.QUALITY
            )
            multimodal_request.functions = [chart_function]
            # multimodal_request.images = ["path/to/chart.jpg"]  # Uncomment with real image

            response = await provider.generate_response(multimodal_request)
            logger.info(f"Multimodal + functions: {response.content[:200]}...")
            function_calls = response.metadata.get("function_calls", [])
            logger.info(f"Function calls made: {len(function_calls)}")
            self.demo_results["gemini_multimodal_functions"] = {"function_calls": len(function_calls)}
        except Exception as e:
            logger.error(f"Multimodal + functions demo failed: {e}")

        # 4. Google Search Integration
        logger.info("4. Google Search Integration")
        try:
            search_request = ModelRequest(
                prompt="Search for the latest research papers on large language models and summarize the key findings.",
                task_type=TaskType.ANALYSIS,
                complexity=ComplexityLevel.MEDIUM,
                priority=Priority.QUALITY
            )
            search_request.enable_google_search = True

            response = await provider.generate_response(search_request)
            logger.info(f"Google search integration: {response.content[:200]}...")
            grounding = response.metadata.get("grounding_attributions", [])
            logger.info(f"Grounding attributions: {len(grounding)}")
            self.demo_results["gemini_google_search"] = {"grounding_count": len(grounding)}
        except Exception as e:
            logger.error(f"Google search demo failed: {e}")

    async def comparative_analysis(self):
        """Compare capabilities across all providers"""
        logger.info("=== Comparative Analysis ===")

        # Test the same task across all providers
        test_prompt = "Create a Python web scraper that extracts article titles from a news website and saves them to a CSV file. Include error handling and logging."

        comparison_results = {}

        for provider_name, provider in self.providers.items():
            logger.info(f"Testing {provider_name} provider...")

            try:
                request = ModelRequest(
                    prompt=test_prompt,
                    task_type=TaskType.CODE_GENERATION,
                    complexity=ComplexityLevel.MEDIUM,
                    priority=Priority.BALANCED
                )

                start_time = datetime.now()
                response = await provider.generate_response(request)
                end_time = datetime.now()

                comparison_results[provider_name] = {
                    "response_length": len(response.content),
                    "tokens_used": response.tokens_used,
                    "cost": response.cost_estimate,
                    "latency_ms": response.latency_ms,
                    "time_taken": (end_time - start_time).total_seconds(),
                    "model_used": response.model_used
                }

                logger.info(f"{provider_name}: {response.tokens_used['total']} tokens, ${response.cost_estimate:.4f}, {response.latency_ms}ms")

            except Exception as e:
                logger.error(f"Comparison test failed for {provider_name}: {e}")
                comparison_results[provider_name] = {"error": str(e)}

        self.demo_results["comparison"] = comparison_results
        return comparison_results

    async def health_check_all(self):
        """Perform health checks on all providers"""
        logger.info("=== Health Check All Providers ===")

        health_results = {}

        for provider_name, provider in self.providers.items():
            logger.info(f"Health checking {provider_name}...")

            try:
                is_healthy = await provider.health_check()
                capabilities = await provider.get_capabilities() if hasattr(provider, 'get_capabilities') else {}

                health_results[provider_name] = {
                    "healthy": is_healthy,
                    "capabilities": capabilities
                }

                logger.info(f"{provider_name}: {'✓ Healthy' if is_healthy else '✗ Unhealthy'}")

            except Exception as e:
                logger.error(f"Health check failed for {provider_name}: {e}")
                health_results[provider_name] = {
                    "healthy": False,
                    "error": str(e)
                }

        self.demo_results["health_check"] = health_results
        return health_results

    async def generate_report(self):
        """Generate a comprehensive report of all demonstrations"""
        logger.info("=== Generating Comprehensive Report ===")

        report = {
            "demo_timestamp": datetime.now().isoformat(),
            "providers_tested": list(self.providers.keys()),
            "results": self.demo_results,
            "summary": {
                "total_providers": len(self.providers),
                "successful_demos": len([k for k, v in self.demo_results.items() if "error" not in str(v)]),
                "capabilities_demonstrated": [
                    "Function calling and tool integration",
                    "Multi-modal processing (text, images, video)",
                    "Live search and real-time data",
                    "Code execution and analysis",
                    "Computer use and automation",
                    "Multi-agent collaboration",
                    "Large context processing",
                    "Autonomous task execution"
                ]
            }
        }

        # Save report to file
        report_path = Path("enhanced_providers_demo_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to: {report_path}")
        return report

    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up providers...")

        for provider_name, provider in self.providers.items():
            try:
                await provider.close()
                logger.info(f"✓ Closed {provider_name} provider")
            except Exception as e:
                logger.error(f"Error closing {provider_name}: {e}")

    async def run_full_demo(self):
        """Run the complete demonstration suite"""
        logger.info("Starting Enhanced AI Providers Demonstration...")

        try:
            # Setup
            await self.setup_providers()

            # Individual provider demos
            await self.demo_openai_enhanced()
            await self.demo_anthropic_enhanced()
            await self.demo_xai_enhanced()
            await self.demo_gemini_enhanced()

            # Comparative analysis
            await self.comparative_analysis()

            # Health checks
            await self.health_check_all()

            # Generate report
            report = await self.generate_report()

            logger.info("=== Demo Complete! ===")
            logger.info(f"Report generated with {len(self.demo_results)} test results")

            return report

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """Main demo execution function"""
    demo = EnhancedProvidersDemo()

    try:
        report = await demo.run_full_demo()
        print("\n" + "="*60)
        print("ENHANCED AI PROVIDERS DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Providers tested: {report['summary']['total_providers']}")
        print(f"Successful demos: {report['summary']['successful_demos']}")
        print("\nCapabilities demonstrated:")
        for capability in report['summary']['capabilities_demonstrated']:
            print(f"  • {capability}")
        print(f"\nDetailed report saved to: enhanced_providers_demo_report.json")
        print("="*60)

    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    # Run the demo
    exit_code = asyncio.run(main())
    exit(exit_code)
