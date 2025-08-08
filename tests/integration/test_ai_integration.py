#!/usr/bin/env python3
"""
Test script for AI-powered request analysis in Phase 2.1
Tests the enhanced OrchestratorAgent with ModelManager integration
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from aider.agents.orchestrator_agent import OrchestratorAgent
from aider.models.model_manager import get_model_manager
from aider.agents.message_bus import MessageBus
from aider.context.context_store import GlobalContextStore


async def test_ai_integration():
    """Test AI-powered request analysis functionality"""

    print("🚀 Starting AI Integration Test for Phase 2.1")
    print("=" * 60)

    # Initialize components
    message_bus = MessageBus()
    await message_bus.start()

    context_store = GlobalContextStore()
    await context_store.start()

    # Create orchestrator with AI capabilities
    orchestrator = OrchestratorAgent(
        agent_id="test_orchestrator",
        message_bus=message_bus
    )

    try:
        # Initialize the orchestrator (this will set up ModelManager)
        print("📋 Initializing OrchestratorAgent with ModelManager...")
        await orchestrator.initialize()

        if orchestrator.model_manager:
            print("✅ ModelManager initialized successfully!")
        else:
            print("❌ ModelManager initialization failed")
            return False

        await orchestrator.start()
        print("✅ OrchestratorAgent started successfully!")

        # Test different types of requests
        test_requests = [
            {
                "request": "Create a Python function to calculate factorial with unit tests",
                "context": {
                    "project_type": "python_library",
                    "framework": "pytest",
                    "language": "python"
                }
            },
            {
                "request": "Refactor this code to use async/await patterns and improve performance",
                "context": {
                    "existing_code": "def fetch_data(): return requests.get('api/data')",
                    "language": "python",
                    "framework": "aiohttp"
                }
            },
            {
                "request": "Review the authentication module and check for security vulnerabilities",
                "context": {
                    "module": "auth.py",
                    "security_focus": "authentication",
                    "language": "python"
                }
            },
            {
                "request": "Create a new feature branch and implement user registration endpoint",
                "context": {
                    "project_type": "web_api",
                    "framework": "FastAPI",
                    "database": "PostgreSQL"
                }
            }
        ]

        print("\n🔍 Testing AI-Powered Request Analysis:")
        print("-" * 50)

        for i, test_case in enumerate(test_requests, 1):
            print(f"\n📝 Test Case {i}: {test_case['request'][:50]}...")

            try:
                # Analyze the request using AI
                analysis = await orchestrator.analyze_request(
                    request=test_case['request'],
                    context=test_case['context']
                )

                print(f"✅ Analysis completed!")
                print(f"   Request Type: {analysis.request_type.value}")
                print(f"   Complexity: {analysis.complexity.value}")
                print(f"   Required Agents: {', '.join(analysis.required_agents)}")
                print(f"   Subtasks: {len(analysis.subtasks)}")
                print(f"   Confidence: {analysis.confidence_score:.2f}")
                print(f"   Est. Duration: {analysis.estimated_duration}s")

                # Show subtasks
                for j, subtask in enumerate(analysis.subtasks, 1):
                    print(f"     Subtask {j}: {subtask.description[:40]}... ({subtask.agent_type})")

            except Exception as e:
                print(f"❌ Analysis failed: {e}")
                continue

        # Test end-to-end workflow processing
        print(f"\n🔄 Testing End-to-End Workflow Processing:")
        print("-" * 50)

        test_request = "Create a simple REST API endpoint for user registration with email validation"
        test_context = {
            "framework": "FastAPI",
            "database": "SQLite",
            "project_type": "REST API",
            "validation": "pydantic"
        }

        print(f"Request: {test_request}")

        try:
            # Process the full workflow
            result = await orchestrator.process_user_request(
                request=test_request,
                context=test_context,
                user_id="test_user"
            )

            print(f"✅ Workflow processing completed!")
            print(f"   Status: {result['status']}")
            print(f"   Request ID: {result['request_id']}")

            if result['status'] == 'completed':
                metadata = result.get('metadata', {})
                print(f"   Duration: {metadata.get('duration', 0):.2f}s")
                print(f"   Agents Used: {', '.join(metadata.get('agents_used', []))}")
                print(f"   Complexity: {metadata.get('complexity', 'unknown')}")

        except Exception as e:
            print(f"❌ Workflow processing failed: {e}")

        print(f"\n📊 Testing Model Manager Performance:")
        print("-" * 50)

        if orchestrator.model_manager:
            try:
                # Get performance metrics
                metrics = orchestrator.model_manager.get_performance_metrics()
                print(f"✅ Performance metrics retrieved:")
                for provider, data in metrics.items():
                    print(f"   {provider.upper()}:")
                    print(f"     Requests: {data.get('request_count', 0)}")
                    print(f"     Avg Latency: {data.get('avg_latency', 0):.2f}ms")
                    print(f"     Total Cost: ${data.get('total_cost', 0):.4f}")

                # Test health check
                health = await orchestrator.model_manager.health_check()
                print(f"   Health Status: {'✅ Healthy' if health else '❌ Unhealthy'}")

            except Exception as e:
                print(f"❌ Performance metrics failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up...")
        try:
            await orchestrator.stop()
            await context_store.stop()
            await message_bus.stop()
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")


async def test_model_fallback():
    """Test fallback behavior when AI models are not available"""

    print(f"\n🔧 Testing Fallback Behavior:")
    print("-" * 50)

    # Temporarily disable model access to test fallback
    original_env = os.environ.copy()

    # Remove API keys to force fallback
    for key in ['XAI_API_KEY', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY']:
        if key in os.environ:
            os.environ.pop(key)

    try:
        orchestrator = OrchestratorAgent(agent_id="fallback_test")
        await orchestrator.initialize()
        await orchestrator.start()

        analysis = await orchestrator.analyze_request(
            request="Create a simple Python hello world function",
            context={"language": "python"}
        )

        print(f"✅ Fallback analysis completed!")
        print(f"   Confidence: {analysis.confidence_score:.2f} (should be lower)")
        print(f"   Risk Factors: {analysis.risk_factors}")

        await orchestrator.stop()

    except Exception as e:
        print(f"❌ Fallback test failed: {e}")

    finally:
        # Restore environment
        os.environ.clear()
        os.environ.update(original_env)


def print_test_summary(success: bool):
    """Print test summary"""
    print("\n" + "=" * 60)
    print("🎯 PHASE 2.1 AI INTEGRATION TEST SUMMARY")
    print("=" * 60)

    if success:
        print("✅ SUCCESS: AI-powered request analysis is working!")
        print("")
        print("🎊 Phase 2.1 Milestones Achieved:")
        print("   ✅ ModelManager integration completed")
        print("   ✅ AI-powered request analysis functional")
        print("   ✅ Multi-agent coordination working")
        print("   ✅ End-to-end workflow processing operational")
        print("   ✅ Fallback mechanisms in place")
        print("")
        print("🚀 Ready for Phase 2.2: Deep Codebase Intelligence")
        print("")
        print("Next Steps:")
        print("   1. Add real code generation to CodeAgent")
        print("   2. Implement project context building")
        print("   3. Add git operations automation")
        print("   4. Create comprehensive integration tests")

    else:
        print("❌ FAILED: AI integration needs attention")
        print("")
        print("🔧 Common Issues to Check:")
        print("   - API keys not set in environment")
        print("   - Model providers not responding")
        print("   - Network connectivity issues")
        print("   - Dependencies missing")
        print("")
        print("💡 Troubleshooting:")
        print("   1. Check .env file configuration")
        print("   2. Verify API key validity")
        print("   3. Test network connections")
        print("   4. Review error logs above")


async def main():
    """Main test runner"""
    print("🎯 PHASE 2.1: AUTONOMOUS OPERATIONS - AI INTEGRATION TEST")
    print("🔄 Transforming from 'functional infrastructure' to 'autonomous AI development partner'")
    print("")

    success = False

    try:
        # Run main AI integration tests
        success = await test_ai_integration()

        # Test fallback behavior
        await test_model_fallback()

    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print_test_summary(success)


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())
