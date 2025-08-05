#!/usr/bin/env python3
"""
Comprehensive Test Suite for AI-Enhanced CodeAgent

This test suite validates the Phase 2.2 AI integration enhancements:
- Real AI-powered code generation
- AI-powered code analysis
- AI-powered code validation
- Fallback mechanisms
- Multi-language support
- Performance and reliability

Run with: python test_code_agent_ai_enhancement.py
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from aider.agents.code_agent import CodeAgent, CodeLanguage, CodeGenerationRequest, CodeQuality
from aider.models.model_manager import ComplexityLevel


class AICodeAgentTester:
    """Comprehensive tester for AI-enhanced CodeAgent capabilities."""

    def __init__(self):
        self.agent = None
        self.test_results = []
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize the CodeAgent for testing."""
        print("🚀 Initializing AI-Enhanced CodeAgent...")

        self.agent = CodeAgent(
            agent_id="test_code_agent",
            config={
                'enable_testing': True,
                'enable_validation': True,
                'generation_timeout': 60.0
            }
        )

        await self.agent.initialize()
        print("✅ CodeAgent initialized successfully")

    async def test_ai_code_generation(self):
        """Test AI-powered code generation across different languages and complexities."""
        print("\n🧪 Testing AI-Powered Code Generation...")

        test_cases = [
            {
                'name': 'Python Simple Function',
                'language': CodeLanguage.PYTHON,
                'description': 'Create a function that calculates the factorial of a number using recursion',
                'expected_keywords': ['def', 'factorial', 'return', 'if']
            },
            {
                'name': 'Python Class with Methods',
                'language': CodeLanguage.PYTHON,
                'description': 'Create a BankAccount class with deposit, withdraw, and get_balance methods. Include proper error handling for insufficient funds.',
                'expected_keywords': ['class', 'BankAccount', 'def deposit', 'def withdraw', 'def get_balance']
            },
            {
                'name': 'JavaScript Async Function',
                'language': CodeLanguage.JAVASCRIPT,
                'description': 'Create an async function that fetches user data from an API endpoint and handles errors gracefully',
                'expected_keywords': ['async', 'await', 'fetch', 'try', 'catch']
            },
            {
                'name': 'Complex Algorithm',
                'language': CodeLanguage.PYTHON,
                'description': 'Implement a binary search tree with insert, search, and delete operations. Include tree traversal methods.',
                'expected_keywords': ['class', 'Node', 'insert', 'search', 'delete', 'traversal']
            }
        ]

        successful_generations = 0

        for test_case in test_cases:
            try:
                print(f"  📝 Testing: {test_case['name']}")

                request = CodeGenerationRequest(
                    description=test_case['description'],
                    language=test_case['language']
                )

                start_time = time.time()
                result = await self.agent.generate_code(request)
                generation_time = time.time() - start_time

                # Validate result structure
                assert result.generated_code, "Generated code should not be empty"
                assert result.analysis, "Code analysis should be provided"
                assert result.confidence_score > 0, "Confidence score should be positive"

                # Check for expected keywords
                code_lower = result.generated_code.lower()
                found_keywords = [kw for kw in test_case['expected_keywords'] if kw.lower() in code_lower]

                success = len(found_keywords) >= len(test_case['expected_keywords']) * 0.6  # 60% threshold

                print(f"    ✅ Generated {len(result.generated_code)} chars in {generation_time:.2f}s")
                print(f"    🎯 Confidence: {result.confidence_score:.1f}%")
                print(f"    📊 Quality Score: {result.analysis.quality_score}")

                if success:
                    successful_generations += 1
                    print(f"    ✅ Keywords found: {found_keywords}")
                else:
                    print(f"    ⚠️  Limited keywords found: {found_keywords}")

                # Store detailed results
                self.test_results.append({
                    'test': f"code_generation_{test_case['name'].lower().replace(' ', '_')}",
                    'success': success,
                    'generation_time': generation_time,
                    'confidence_score': result.confidence_score,
                    'quality_score': result.analysis.quality_score,
                    'code_length': len(result.generated_code),
                    'keywords_found': len(found_keywords),
                    'keywords_expected': len(test_case['expected_keywords'])
                })

            except Exception as e:
                print(f"    ❌ Failed: {str(e)}")
                self.test_results.append({
                    'test': f"code_generation_{test_case['name'].lower().replace(' ', '_')}",
                    'success': False,
                    'error': str(e)
                })

        success_rate = (successful_generations / len(test_cases)) * 100
        print(f"\n📈 Code Generation Success Rate: {success_rate:.1f}% ({successful_generations}/{len(test_cases)})")

        return success_rate >= 75  # 75% success threshold

    async def test_ai_code_analysis(self):
        """Test AI-powered code analysis capabilities."""
        print("\n🔍 Testing AI-Powered Code Analysis...")

        test_codes = [
            {
                'name': 'Clean Python Code',
                'language': CodeLanguage.PYTHON,
                'code': '''
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number using dynamic programming."""
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]
''',
                'expected_quality_min': 70
            },
            {
                'name': 'Problematic Code',
                'language': CodeLanguage.PYTHON,
                'code': '''
def bad_function(x,y,z):
    # No docstring, poor formatting
    if x>0:
        return x*y*z*x*y*z*x*y*z*x*y*z
    else:
        print("error")
        return None
''',
                'expected_quality_max': 50
            },
            {
                'name': 'JavaScript Function',
                'language': CodeLanguage.JAVASCRIPT,
                'code': '''
function processUserData(userData) {
    if (!userData || !userData.id) {
        throw new Error('Invalid user data');
    }

    return {
        id: userData.id,
        name: userData.name || 'Anonymous',
        email: userData.email?.toLowerCase(),
        createdAt: new Date().toISOString()
    };
}
''',
                'expected_quality_min': 60
            }
        ]

        successful_analyses = 0

        for test_code in test_codes:
            try:
                print(f"  🔬 Analyzing: {test_code['name']}")

                analysis = await self.agent._analyze_code(test_code['code'], test_code['language'])

                print(f"    📊 Quality Score: {analysis.quality_score}")
                print(f"    🔧 Complexity: {analysis.complexity_score}")
                print(f"    📝 Lines of Code: {analysis.lines_of_code}")
                print(f"    ⚠️  Issues Found: {len(analysis.issues)}")
                print(f"    💡 Suggestions: {len(analysis.suggestions)}")

                # Validate analysis quality
                quality_ok = True
                if 'expected_quality_min' in test_code:
                    quality_ok = analysis.quality_score >= test_code['expected_quality_min']
                elif 'expected_quality_max' in test_code:
                    quality_ok = analysis.quality_score <= test_code['expected_quality_max']

                if quality_ok:
                    successful_analyses += 1
                    print(f"    ✅ Analysis quality meets expectations")
                else:
                    print(f"    ⚠️  Analysis quality outside expected range")

                self.test_results.append({
                    'test': f"code_analysis_{test_code['name'].lower().replace(' ', '_')}",
                    'success': quality_ok,
                    'quality_score': analysis.quality_score,
                    'complexity_score': analysis.complexity_score,
                    'issues_count': len(analysis.issues),
                    'suggestions_count': len(analysis.suggestions)
                })

            except Exception as e:
                print(f"    ❌ Analysis failed: {str(e)}")
                self.test_results.append({
                    'test': f"code_analysis_{test_code['name'].lower().replace(' ', '_')}",
                    'success': False,
                    'error': str(e)
                })

        success_rate = (successful_analyses / len(test_codes)) * 100
        print(f"\n📈 Code Analysis Success Rate: {success_rate:.1f}% ({successful_analyses}/{len(test_codes)})")

        return success_rate >= 70  # 70% success threshold

    async def test_ai_code_validation(self):
        """Test AI-powered code validation capabilities."""
        print("\n✅ Testing AI-Powered Code Validation...")

        validation_tests = [
            {
                'name': 'Valid Python Code',
                'language': CodeLanguage.PYTHON,
                'code': '''
def greet(name: str) -> str:
    """Greet a person by name."""
    if not name:
        raise ValueError("Name cannot be empty")
    return f"Hello, {name}!"
''',
                'should_be_valid': True
            },
            {
                'name': 'Invalid Python Syntax',
                'language': CodeLanguage.PYTHON,
                'code': '''
def broken_function(
    if x > 0
        return x
    else
        return 0
''',
                'should_be_valid': False
            },
            {
                'name': 'Valid JavaScript Code',
                'language': CodeLanguage.JAVASCRIPT,
                'code': '''
function validateEmail(email) {
    const emailRegex = /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
    return emailRegex.test(email);
}
''',
                'should_be_valid': True
            }
        ]

        successful_validations = 0

        for test in validation_tests:
            try:
                print(f"  ✔️  Validating: {test['name']}")

                validation_result = await self.agent.validate_code(
                    test['code'],
                    test['language'],
                    strict_mode=False
                )

                is_valid = validation_result['overall_valid']
                expected_valid = test['should_be_valid']

                print(f"    📝 Syntax Valid: {validation_result['syntax_valid']}")
                print(f"    🏗️  Structure Valid: {validation_result['structure_valid']}")
                print(f"    🧠 Semantic Valid: {validation_result['semantic_valid']}")
                print(f"    🔢 Errors: {len(validation_result['errors'])}")
                print(f"    ⚠️  Warnings: {len(validation_result['warnings'])}")

                validation_correct = (is_valid == expected_valid)

                if validation_correct:
                    successful_validations += 1
                    print(f"    ✅ Validation result matches expectation")
                else:
                    print(f"    ❌ Expected {expected_valid}, got {is_valid}")

                self.test_results.append({
                    'test': f"code_validation_{test['name'].lower().replace(' ', '_')}",
                    'success': validation_correct,
                    'is_valid': is_valid,
                    'expected_valid': expected_valid,
                    'errors_count': len(validation_result['errors']),
                    'warnings_count': len(validation_result['warnings'])
                })

            except Exception as e:
                print(f"    ❌ Validation failed: {str(e)}")
                self.test_results.append({
                    'test': f"code_validation_{test['name'].lower().replace(' ', '_')}",
                    'success': False,
                    'error': str(e)
                })

        success_rate = (successful_validations / len(validation_tests)) * 100
        print(f"\n📈 Code Validation Success Rate: {success_rate:.1f}% ({successful_validations}/{len(validation_tests)})")

        return success_rate >= 75  # 75% success threshold

    async def test_fallback_mechanisms(self):
        """Test fallback mechanisms when AI services are unavailable."""
        print("\n🛡️  Testing Fallback Mechanisms...")

        # Temporarily disable AI by simulating API failure
        original_generate_code = self.agent._generate_code_content

        async def mock_failing_generate_code(*args, **kwargs):
            raise Exception("Simulated AI service failure")

        self.agent._generate_code_content = mock_failing_generate_code

        try:
            print("  🧪 Testing code generation fallback...")

            request = CodeGenerationRequest(
                description="Create a simple hello world function",
                language=CodeLanguage.PYTHON
            )

            result = await self.agent.generate_code(request)

            # Should still generate code using templates
            assert result.generated_code, "Fallback should still generate code"
            print("    ✅ Fallback code generation successful")

            self.test_results.append({
                'test': 'fallback_code_generation',
                'success': True,
                'fallback_used': True
            })

            return True

        except Exception as e:
            print(f"    ❌ Fallback mechanism failed: {str(e)}")
            self.test_results.append({
                'test': 'fallback_code_generation',
                'success': False,
                'error': str(e)
            })
            return False

        finally:
            # Restore original method
            self.agent._generate_code_content = original_generate_code

    async def test_performance_benchmarks(self):
        """Test performance benchmarks for AI-enhanced operations."""
        print("\n⚡ Testing Performance Benchmarks...")

        # Test code generation performance
        print("  🏃 Benchmarking code generation...")

        start_time = time.time()

        request = CodeGenerationRequest(
            description="Create a function to sort a list of numbers using quicksort algorithm",
            language=CodeLanguage.PYTHON
        )

        result = await self.agent.generate_code(request)
        generation_time = time.time() - start_time

        print(f"    ⏱️  Generation Time: {generation_time:.2f}s")
        print(f"    📏 Code Length: {len(result.generated_code)} characters")
        print(f"    🎯 Confidence: {result.confidence_score:.1f}%")

        # Performance thresholds
        performance_ok = generation_time < 30.0  # Should complete within 30 seconds

        self.test_results.append({
            'test': 'performance_benchmark',
            'success': performance_ok,
            'generation_time': generation_time,
            'code_length': len(result.generated_code),
            'confidence_score': result.confidence_score
        })

        if performance_ok:
            print("    ✅ Performance meets expectations")
        else:
            print("    ⚠️  Performance below expectations")

        return performance_ok

    async def run_comprehensive_test_suite(self):
        """Run the complete test suite and generate report."""
        print("🚀 Starting Comprehensive AI-Enhanced CodeAgent Test Suite")
        print("=" * 80)

        start_time = time.time()

        # Initialize agent
        await self.initialize()

        # Run all tests
        test_results = {
            'ai_code_generation': await self.test_ai_code_generation(),
            'ai_code_analysis': await self.test_ai_code_analysis(),
            'ai_code_validation': await self.test_ai_code_validation(),
            'fallback_mechanisms': await self.test_fallback_mechanisms(),
            'performance_benchmarks': await self.test_performance_benchmarks()
        }

        total_time = time.time() - start_time

        # Generate comprehensive report
        await self.generate_test_report(test_results, total_time)

        return test_results

    async def generate_test_report(self, test_results: Dict[str, bool], total_time: float):
        """Generate a comprehensive test report."""
        print("\n" + "=" * 80)
        print("🏆 AI-Enhanced CodeAgent Test Report")
        print("=" * 80)

        # Overall results
        passed_tests = sum(1 for result in test_results.values() if result)
        total_tests = len(test_results)
        success_rate = (passed_tests / total_tests) * 100

        print(f"\n📊 Overall Results:")
        print(f"   ✅ Tests Passed: {passed_tests}/{total_tests}")
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        print(f"   ⏱️  Total Time: {total_time:.2f}s")

        # Individual test results
        print(f"\n📋 Test Breakdown:")
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {status} {test_name.replace('_', ' ').title()}")

        # Detailed metrics
        print(f"\n📈 Detailed Metrics:")

        # Code generation metrics
        gen_results = [r for r in self.test_results if r['test'].startswith('code_generation')]
        if gen_results:
            avg_confidence = sum(r.get('confidence_score', 0) for r in gen_results) / len(gen_results)
            avg_quality = sum(r.get('quality_score', 0) for r in gen_results) / len(gen_results)
            avg_time = sum(r.get('generation_time', 0) for r in gen_results) / len(gen_results)

            print(f"   🎯 Average Confidence: {avg_confidence:.1f}%")
            print(f"   🏆 Average Quality: {avg_quality:.1f}")
            print(f"   ⚡ Average Generation Time: {avg_time:.2f}s")

        # Analysis metrics
        analysis_results = [r for r in self.test_results if r['test'].startswith('code_analysis')]
        if analysis_results:
            avg_issues = sum(r.get('issues_count', 0) for r in analysis_results) / len(analysis_results)
            avg_suggestions = sum(r.get('suggestions_count', 0) for r in analysis_results) / len(analysis_results)

            print(f"   🔍 Average Issues Found: {avg_issues:.1f}")
            print(f"   💡 Average Suggestions: {avg_suggestions:.1f}")

        # Success criteria
        print(f"\n🎯 Success Criteria:")
        criteria = [
            ("Code Generation Success Rate ≥ 75%", test_results['ai_code_generation']),
            ("Code Analysis Success Rate ≥ 70%", test_results['ai_code_analysis']),
            ("Code Validation Success Rate ≥ 75%", test_results['ai_code_validation']),
            ("Fallback Mechanisms Working", test_results['fallback_mechanisms']),
            ("Performance Within Limits", test_results['performance_benchmarks'])
        ]

        for criterion, met in criteria:
            status = "✅" if met else "❌"
            print(f"   {status} {criterion}")

        # Final verdict
        print(f"\n🏆 Final Verdict:")
        if success_rate >= 80:
            print("   🎉 EXCELLENT: AI-Enhanced CodeAgent is ready for production!")
        elif success_rate >= 60:
            print("   ✅ GOOD: AI-Enhanced CodeAgent is functional with minor issues")
        else:
            print("   ⚠️  NEEDS WORK: AI-Enhanced CodeAgent requires further development")

        # Save detailed results to file
        results_file = project_root / "ai_code_agent_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'overall_results': test_results,
                'success_rate': success_rate,
                'total_time': total_time,
                'detailed_results': self.test_results
            }, f, indent=2)

        print(f"\n💾 Detailed results saved to: {results_file}")


async def main():
    """Main test execution function."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    tester = AICodeAgentTester()

    try:
        test_results = await tester.run_comprehensive_test_suite()

        # Exit with appropriate code
        success_count = sum(1 for result in test_results.values() if result)
        exit_code = 0 if success_count >= len(test_results) * 0.8 else 1  # 80% success threshold

        return exit_code

    except Exception as e:
        print(f"\n❌ Test suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
