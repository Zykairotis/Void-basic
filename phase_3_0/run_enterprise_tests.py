#!/usr/bin/env python3
"""
🚀 **PHASE 3.0 ENTERPRISE TEST RUNNER**

Comprehensive test execution orchestrator for Phase 3.0 Enterprise Production Deployment.
This script coordinates and executes all categories of enterprise testing including:

- Unit Tests: Component validation
- Integration Tests: Cross-component functionality
- Security Tests: Vulnerability and penetration testing
- Compliance Tests: GDPR, HIPAA, SOX validation
- Performance Tests: Load and scalability testing
- Multi-Tenant Tests: Isolation and data segregation
- Production Readiness: Deployment validation

Usage:
    python run_enterprise_tests.py [options]

Options:
    --suite <name>      Run specific test suite (unit|integration|security|compliance|performance|all)
    --environment <env> Target environment (development|staging|production)
    --report-format     Output format (console|json|html|junit)
    --parallel          Run tests in parallel where possible
    --verbose           Detailed output and logging
    --fail-fast         Stop on first failure
    --coverage          Generate code coverage reports
    --benchmark         Include performance benchmarking
    --export-results    Export results to monitoring system
"""

import asyncio
import sys
import os
import time
import json
import argparse
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import aiohttp
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from phase_3_0.tests.test_enterprise_suite import EnterpriseTestSuite, run_enterprise_testing
    from phase_3_0.monitoring.grafana_dashboards import generate_phase3_dashboards
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("📁 Make sure you're running from the correct directory")


@dataclass
class TestConfiguration:
    """Test execution configuration."""
    suite: str = "all"
    environment: str = "development"
    report_format: str = "console"
    parallel: bool = True
    verbose: bool = False
    fail_fast: bool = False
    coverage: bool = False
    benchmark: bool = False
    export_results: bool = False
    timeout: int = 1800  # 30 minutes default timeout


@dataclass
class TestResult:
    """Individual test result data."""
    name: str
    category: str
    status: str  # passed, failed, skipped, error
    duration: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class TestSuiteResult:
    """Complete test suite execution result."""
    suite_name: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    success_rate: float
    results: List[TestResult]
    environment: str
    configuration: TestConfiguration


class Phase3TestRunner:
    """
    🏢 **PHASE 3.0 ENTERPRISE TEST ORCHESTRATOR**

    Coordinates execution of all enterprise testing suites with comprehensive
    reporting, parallel execution, and integration with monitoring systems.
    """

    def __init__(self, config: TestConfiguration):
        self.config = config
        self.start_time = datetime.now()
        self.logger = self._setup_logging()
        self.results = []
        self.metrics = {}

    async def run_all_tests(self) -> TestSuiteResult:
        """Execute comprehensive Phase 3.0 enterprise test suite."""
        print("🚀 PHASE 3.0 ENTERPRISE TEST EXECUTION")
        print("=" * 60)
        print(f"📅 Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"🎯 Test Suite: {self.config.suite}")
        print(f"🌍 Environment: {self.config.environment}")
        print(f"⚡ Parallel Execution: {'Enabled' if self.config.parallel else 'Disabled'}")
        print(f"📊 Coverage Analysis: {'Enabled' if self.config.coverage else 'Disabled'}")
        print("=" * 60)

        # Test suite orchestration
        test_suites = self._get_test_suites()

        if self.config.parallel and len(test_suites) > 1:
            results = await self._run_parallel_tests(test_suites)
        else:
            results = await self._run_sequential_tests(test_suites)

        # Generate comprehensive report
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()

        suite_result = TestSuiteResult(
            suite_name=self.config.suite,
            start_time=self.start_time,
            end_time=end_time,
            total_duration=total_duration,
            total_tests=sum(len(r.results) if hasattr(r, 'results') else 1 for r in results),
            passed=sum(r.get('passed', 0) if isinstance(r, dict) else 1 for r in results if self._is_success(r)),
            failed=sum(r.get('failed', 0) if isinstance(r, dict) else 1 for r in results if not self._is_success(r)),
            skipped=0,
            errors=0,
            success_rate=0.0,
            results=results,
            environment=self.config.environment,
            configuration=self.config
        )

        # Calculate success rate
        if suite_result.total_tests > 0:
            suite_result.success_rate = (suite_result.passed / suite_result.total_tests) * 100

        # Generate reports
        await self._generate_reports(suite_result)

        # Export to monitoring if enabled
        if self.config.export_results:
            await self._export_to_monitoring(suite_result)

        return suite_result

    def _get_test_suites(self) -> List[str]:
        """Get list of test suites to execute based on configuration."""
        all_suites = [
            "unit_tests",
            "integration_tests",
            "security_tests",
            "compliance_tests",
            "multi_tenant_tests",
            "performance_tests",
            "websocket_tests",
            "database_tests",
            "api_tests",
            "production_readiness_tests"
        ]

        if self.config.suite == "all":
            return all_suites
        elif self.config.suite in all_suites:
            return [self.config.suite]
        else:
            # Handle custom suite specifications
            suite_mapping = {
                "unit": ["unit_tests"],
                "integration": ["integration_tests", "api_tests", "database_tests"],
                "security": ["security_tests"],
                "compliance": ["compliance_tests"],
                "performance": ["performance_tests", "websocket_tests"],
                "tenant": ["multi_tenant_tests"],
                "readiness": ["production_readiness_tests"]
            }
            return suite_mapping.get(self.config.suite, all_suites)

    async def _run_parallel_tests(self, test_suites: List[str]) -> List[Dict]:
        """Execute test suites in parallel for improved performance."""
        print("⚡ Running test suites in parallel...")

        tasks = []
        for suite_name in test_suites:
            task = asyncio.create_task(
                self._execute_test_suite(suite_name),
                name=f"suite_{suite_name}"
            )
            tasks.append(task)

        # Execute with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.timeout
            )
            return results
        except asyncio.TimeoutError:
            print(f"⚠️ Test execution timed out after {self.config.timeout} seconds")
            return [{"error": "timeout", "suite": suite} for suite in test_suites]

    async def _run_sequential_tests(self, test_suites: List[str]) -> List[Dict]:
        """Execute test suites sequentially."""
        print("🔄 Running test suites sequentially...")

        results = []
        for suite_name in test_suites:
            print(f"\n🧪 Executing {suite_name}...")

            if self.config.fail_fast and any(not self._is_success(r) for r in results):
                print(f"⚠️ Skipping {suite_name} due to fail-fast configuration")
                break

            try:
                result = await self._execute_test_suite(suite_name)
                results.append(result)
            except Exception as e:
                error_result = {
                    "suite": suite_name,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(error_result)

                if self.config.fail_fast:
                    break

        return results

    async def _execute_test_suite(self, suite_name: str) -> Dict[str, Any]:
        """Execute individual test suite with error handling and metrics."""
        start_time = time.time()

        try:
            if suite_name == "enterprise_full_suite":
                # Run the comprehensive enterprise test suite
                enterprise_suite = EnterpriseTestSuite()
                result = await enterprise_suite.run_comprehensive_test_suite()
                return result
            else:
                # Run specific test category
                result = await self._run_specific_test_category(suite_name)
                return result

        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Test suite {suite_name} failed: {str(e)}")

            return {
                "suite": suite_name,
                "success": False,
                "duration": duration,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _run_specific_test_category(self, category: str) -> Dict[str, Any]:
        """Run specific test category using appropriate test framework."""
        start_time = time.time()

        # Category-specific test execution
        category_executors = {
            "unit_tests": self._run_unit_tests,
            "integration_tests": self._run_integration_tests,
            "security_tests": self._run_security_tests,
            "compliance_tests": self._run_compliance_tests,
            "multi_tenant_tests": self._run_multi_tenant_tests,
            "performance_tests": self._run_performance_tests,
            "websocket_tests": self._run_websocket_tests,
            "database_tests": self._run_database_tests,
            "api_tests": self._run_api_tests,
            "production_readiness_tests": self._run_production_readiness_tests
        }

        executor = category_executors.get(category)
        if not executor:
            raise ValueError(f"Unknown test category: {category}")

        result = await executor()
        result["duration"] = time.time() - start_time
        result["category"] = category
        return result

    async def _run_unit_tests(self) -> Dict[str, Any]:
        """Execute unit tests with pytest."""
        print("  🧪 Running unit tests...")

        try:
            cmd = ["python", "-m", "pytest", "tests/", "-v"]

            if self.config.coverage:
                cmd.extend(["--cov=phase_3_0", "--cov-report=xml", "--cov-report=html"])

            if self.config.verbose:
                cmd.append("-s")

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "return_code": result.returncode,
                "tests_run": self._parse_pytest_output(result.stdout)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tests_run": 0
            }

    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Execute integration tests."""
        print("  🔗 Running integration tests...")

        try:
            # Run enterprise test suite integration tests
            enterprise_suite = EnterpriseTestSuite()
            result = await enterprise_suite._run_integration_tests()
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "total_tests": 0,
                "passed": 0,
                "failed": 1
            }

    async def _run_security_tests(self) -> Dict[str, Any]:
        """Execute security tests including vulnerability scanning."""
        print("  🔒 Running security tests...")

        try:
            # Run security test suite
            enterprise_suite = EnterpriseTestSuite()
            result = await enterprise_suite._run_security_tests()

            # Add additional security scanning if in appropriate environment
            if self.config.environment in ['staging', 'production']:
                await self._run_security_scanning()

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "security_score": 0
            }

    async def _run_compliance_tests(self) -> Dict[str, Any]:
        """Execute compliance validation tests."""
        print("  📋 Running compliance tests...")

        try:
            enterprise_suite = EnterpriseTestSuite()
            result = await enterprise_suite._run_compliance_tests()
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "compliance_score": 0
            }

    async def _run_multi_tenant_tests(self) -> Dict[str, Any]:
        """Execute multi-tenant isolation and management tests."""
        print("  🏢 Running multi-tenant tests...")

        try:
            enterprise_suite = EnterpriseTestSuite()
            result = await enterprise_suite._run_multi_tenant_tests()
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "isolation_verified": False
            }

    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Execute performance and load tests."""
        print("  ⚡ Running performance tests...")

        try:
            enterprise_suite = EnterpriseTestSuite()
            result = await enterprise_suite._run_load_tests()

            # Add benchmarking if enabled
            if self.config.benchmark:
                benchmark_result = await self._run_performance_benchmarks()
                result["benchmarks"] = benchmark_result

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "performance_score": 0
            }

    async def _run_websocket_tests(self) -> Dict[str, Any]:
        """Execute WebSocket and real-time functionality tests."""
        print("  🌐 Running WebSocket tests...")

        try:
            enterprise_suite = EnterpriseTestSuite()
            result = await enterprise_suite._run_websocket_tests()
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "websocket_tests": 0
            }

    async def _run_database_tests(self) -> Dict[str, Any]:
        """Execute database functionality and integrity tests."""
        print("  🗄️ Running database tests...")

        try:
            enterprise_suite = EnterpriseTestSuite()
            result = await enterprise_suite._run_database_tests()
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "database_tests": 0
            }

    async def _run_api_tests(self) -> Dict[str, Any]:
        """Execute API endpoint functionality tests."""
        print("  🔌 Running API tests...")

        try:
            enterprise_suite = EnterpriseTestSuite()
            result = await enterprise_suite._run_api_tests()
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "api_tests": 0
            }

    async def _run_production_readiness_tests(self) -> Dict[str, Any]:
        """Execute production readiness validation."""
        print("  🚀 Running production readiness tests...")

        try:
            enterprise_suite = EnterpriseTestSuite()
            result = await enterprise_suite._run_production_readiness_tests()
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "production_ready": False
            }

    async def _run_security_scanning(self) -> Dict[str, Any]:
        """Run additional security scanning tools."""
        scanning_results = {}

        # Bandit security linting
        try:
            result = subprocess.run(
                ["bandit", "-r", "phase_3_0/", "-f", "json"],
                capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            scanning_results["bandit"] = {
                "success": result.returncode == 0,
                "output": result.stdout
            }
        except Exception:
            scanning_results["bandit"] = {"success": False, "error": "Tool not available"}

        # Safety dependency checking
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True, text=True, cwd=PROJECT_ROOT
            )
            scanning_results["safety"] = {
                "success": result.returncode == 0,
                "output": result.stdout
            }
        except Exception:
            scanning_results["safety"] = {"success": False, "error": "Tool not available"}

        return scanning_results

    async def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarking tests."""
        benchmarks = {}

        # CPU benchmark
        start_time = time.time()
        # Simulate CPU intensive task
        for _ in range(1000000):
            pass
        benchmarks["cpu_benchmark"] = time.time() - start_time

        # Memory benchmark
        start_time = time.time()
        test_data = [i for i in range(100000)]
        benchmarks["memory_benchmark"] = time.time() - start_time

        # I/O benchmark
        start_time = time.time()
        temp_file = "benchmark_temp.txt"
        with open(temp_file, 'w') as f:
            for i in range(10000):
                f.write(f"line {i}\n")
        os.remove(temp_file)
        benchmarks["io_benchmark"] = time.time() - start_time

        return benchmarks

    async def _generate_reports(self, suite_result: TestSuiteResult):
        """Generate comprehensive test reports in specified format."""
        print(f"\n📊 Generating test reports in {self.config.report_format} format...")

        if self.config.report_format in ['console', 'all']:
            self._generate_console_report(suite_result)

        if self.config.report_format in ['json', 'all']:
            await self._generate_json_report(suite_result)

        if self.config.report_format in ['html', 'all']:
            await self._generate_html_report(suite_result)

        if self.config.report_format in ['junit', 'all']:
            await self._generate_junit_report(suite_result)

    def _generate_console_report(self, suite_result: TestSuiteResult):
        """Generate detailed console report."""
        print("\n" + "=" * 80)
        print("🏆 PHASE 3.0 ENTERPRISE TEST EXECUTION COMPLETE")
        print("=" * 80)

        print(f"📅 Execution Time: {suite_result.start_time.strftime('%Y-%m-%d %H:%M:%S')} - {suite_result.end_time.strftime('%H:%M:%S')}")
        print(f"⏱️  Total Duration: {suite_result.total_duration:.2f} seconds")
        print(f"🎯 Test Suite: {suite_result.suite_name}")
        print(f"🌍 Environment: {suite_result.environment}")

        print(f"\n📊 TEST RESULTS SUMMARY:")
        print(f"   🧪 Total Tests: {suite_result.total_tests}")
        print(f"   ✅ Passed: {suite_result.passed}")
        print(f"   ❌ Failed: {suite_result.failed}")
        print(f"   ⏭️  Skipped: {suite_result.skipped}")
        print(f"   🚨 Errors: {suite_result.errors}")
        print(f"   📈 Success Rate: {suite_result.success_rate:.1f}%")

        # Detailed results by category
        print(f"\n📋 DETAILED RESULTS BY CATEGORY:")
        for result in suite_result.results:
            if isinstance(result, dict):
                status_icon = "✅" if result.get('success', False) else "❌"
                suite_name = result.get('suite', result.get('category', 'Unknown'))
                duration = result.get('duration', 0)
                print(f"   {status_icon} {suite_name}: {duration:.2f}s")

                if not result.get('success', False) and 'error' in result:
                    print(f"      🚨 Error: {result['error']}")

        # Overall status
        overall_success = suite_result.success_rate >= 95
        print(f"\n{'🎉 OVERALL STATUS: SUCCESS!' if overall_success else '⚠️ OVERALL STATUS: ISSUES DETECTED'}")

        if overall_success:
            print("🚀 System ready for production deployment!")
        else:
            print("🔧 Please address failing tests before production deployment.")

        print("=" * 80)

    async def _generate_json_report(self, suite_result: TestSuiteResult):
        """Generate JSON format test report."""
        report_data = asdict(suite_result)
        report_file = f"phase3_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"📄 JSON report saved: {report_file}")

    async def _generate_html_report(self, suite_result: TestSuiteResult):
        """Generate HTML format test report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Phase 3.0 Enterprise Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f0f8ff; padding: 20px; border-radius: 10px; }}
                .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
                .metric {{ background: #f9f9f9; padding: 15px; border-radius: 5px; text-align: center; }}
                .success {{ color: #28a745; }}
                .failure {{ color: #dc3545; }}
                .results {{ margin-top: 30px; }}
                .result-item {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ddd; }}
                .result-item.passed {{ border-left-color: #28a745; }}
                .result-item.failed {{ border-left-color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Phase 3.0 Enterprise Test Report</h1>
                <p>Execution Date: {suite_result.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Duration: {suite_result.total_duration:.2f} seconds</p>
                <p>Environment: {suite_result.environment}</p>
            </div>

            <div class="summary">
                <div class="metric">
                    <h3>{suite_result.total_tests}</h3>
                    <p>Total Tests</p>
                </div>
                <div class="metric success">
                    <h3>{suite_result.passed}</h3>
                    <p>Passed</p>
                </div>
                <div class="metric failure">
                    <h3>{suite_result.failed}</h3>
                    <p>Failed</p>
                </div>
                <div class="metric">
                    <h3>{suite_result.success_rate:.1f}%</h3>
                    <p>Success Rate</p>
                </div>
            </div>

            <div class="results">
                <h2>Test Results by Category</h2>
        """

        for result in suite_result.results:
            if isinstance(result, dict):
                status = "passed" if result.get('success', False) else "failed"
                suite_name = result.get('suite', result.get('category', 'Unknown'))
                duration = result.get('duration', 0)

                html_content += f"""
                <div class="result-item {status}">
                    <h4>{suite_name}</h4>
                    <p>Duration: {duration:.2f}s</p>
                    <p>Status: {'✅ Passed' if status == 'passed' else '❌ Failed'}</p>
                """

                if 'error' in result:
                    html_content += f"<p>Error: {result['error']}</p>"

                html_content += "</div>"

        html_content += """
            </div>
        </body>
        </html>
        """

        report_file = f"phase3_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_file, 'w') as f:
            f.write(html_content)

        print(f"📄 HTML report saved: {report_file}")

    async def _generate_junit_report(self, suite_result: TestSuiteResult):
        """Generate JUnit XML format test report."""
        junit_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="Phase3EnterpriseTests"
           tests="{suite_result.total_tests}"
           failures="{suite_result.failed}"
           errors="{suite_result.errors}"
           time="{suite_result.total_duration:.2f}">
"""

        for result in suite_result.results:
            if isinstance(result, dict):
                suite_name = result.get('suite', result.get('category', 'Unknown'))
                duration = result.get('duration', 0)
                success = result.get('success', False)

                junit_content += f"""
    <testsuite name="{suite_name}" tests="1" failures="{'0' if success else '1'}" errors="0" time="{duration:.2f}">
        <testcase name="{suite_name}" classname="Phase3Enterprise" time="{duration:.2f}">
"""

                if not success and 'error' in result:
                    junit_content += f"""
            <failure message="{result['error']}" type="TestFailure">
                {result['error']}
            </failure>
"""

                junit_content += """
        </testcase>
    </testsuite>
"""

        junit_content += "</testsuites>"

        report_file = f"phase3_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
        with open(report_file, 'w') as f:
            f.write(junit_content)

        print(f"📄 JUnit report saved: {report_file}")

    async def _export_to_monitoring(self, suite_result: TestSuiteResult):
        """Export test results to monitoring system."""
        print("📊 Exporting results to monitoring system...")

        # Simulate exporting to monitoring (would integrate with Prometheus/Grafana)
        monitoring_data = {
            "timestamp": suite_result.end_time.isoformat(),
            "environment": suite_result.environment,
            "suite": suite_result.suite_name,
            "total_tests": suite_result.total_tests,
            "passed": suite_result.passed,
            "failed": suite_result.failed,
            "success_rate": suite_result.success_rate,
            "duration": suite_result.total_duration
        }

        # In real implementation, this would send to monitoring endpoint
        monitoring_file = f"monitoring_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(monitoring_file, 'w') as f:
            json.dump(monitoring_data, f, indent=2)

        print(f"📄 Monitoring data exported: {monitoring_file}")

    def _parse_pytest_output(self, output: str) -> int:
        """Parse pytest output to count number of tests run."""
        lines = output.split('\n')
        for line in lines:
            if 'passed' in line or 'failed' in line:
                # Try to extract test count from pytest summary
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.isdigit():
                            return int(part)
                except:
                    pass
        return 0

    def _is_success(self, result: Any) -> bool:
        """Check if a test result represents success."""
        if isinstance(result, dict):
            return result.get('success', False)
        return False

    def _setup_logging(self) -> logging.Logger:
        """Configure logging for test execution."""
        logger = logging.getLogger("phase3_test_runner")

        if self.config.verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO

        logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler('phase3_tests.log')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger


def parse_arguments() -> TestConfiguration:
    """Parse command line arguments and return test configuration."""
    parser = argparse.ArgumentParser(
        description="Phase 3.0 Enterprise Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_enterprise_tests.py --suite all --environment production
  python run_enterprise_tests.py --suite security --verbose --fail-fast
  python run_enterprise_tests.py --suite compliance --environment staging --export-results
        """
    )

    parser.add_argument(
        '--suite', '-s',
        choices=['all', 'unit', 'integration', 'security', 'compliance', 'performance',
                'tenant', 'websocket', 'database', 'api', 'readiness'],
        default='all',
        help='Test suite to execute (default: all)'
    )

    parser.add_argument(
        '--environment', '-e',
        choices=['development', 'staging', 'production'],
        default='development',
        help='Target environment (default: development)'
    )

    parser.add_argument(
        '--report-format', '-f',
        choices=['console', 'json', 'html', 'junit', 'all'],
        default='console',
        help='Output report format (default: console)'
    )

    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Run tests in parallel where possible'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output and detailed logging'
    )

    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Stop execution on first test failure'
    )

    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        help='Generate code coverage reports'
    )

    parser.add_argument(
        '--benchmark', '-b',
        action='store_true',
        help='Include performance benchmarking'
    )

    parser.add_argument(
        '--export-results',
        action='store_true',
        help='Export results to monitoring system'
    )

    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=1800,
        help='Test execution timeout in seconds (default: 1800)'
    )

    args = parser.parse_args()

    return TestConfiguration(
        suite=args.suite,
        environment=args.environment,
        report_format=args.report_format,
        parallel=args.parallel,
        verbose=args.verbose,
        fail_fast=args.fail_fast,
        coverage=args.coverage,
        benchmark=args.benchmark,
        export_results=args.export_results,
        timeout=args.timeout
    )


async def main():
    """
    🚀 **MAIN ENTRY POINT FOR PHASE 3.0 ENTERPRISE TESTING**

    Orchestrates the complete Phase 3.0 enterprise test execution pipeline
    with comprehensive reporting and monitoring integration.
    """
    print("🎯 PHASE 3.0 ENTERPRISE TEST RUNNER")
    print("=" * 50)
    print("🏢 Enterprise Production Deployment Validation")
    print("🔒 Security, Compliance, and Multi-Tenancy Testing")
    print("⚡ Performance and Scalability Validation")
    print("🚀 Production Readiness Assessment")
    print("=" * 50)

    try:
        # Parse configuration
        config = parse_arguments()

        # Initialize test runner
        test_runner = Phase3TestRunner(config)

        # Execute test suite
        results = await test_runner.run_all_tests()

        # Determine exit code based on results
        if results.success_rate >= 95.0:
            print("\n🎉 PHASE 3.0 ENTERPRISE TESTING SUCCESSFUL!")
            print("🚀 System validated for production deployment")
            return 0
        else:
            print("\n⚠️ PHASE 3.0 ENTERPRISE TESTING COMPLETED WITH ISSUES")
            print(f"📊 Success Rate: {results.success_rate:.1f}% (Target: ≥95%)")
            print("🔧 Please address failing tests before production deployment")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Test execution failed: {str(e)}")
        if config and config.verbose:
            import traceback
            traceback.print_exc()
        return 1


def run_quick_health_check():
    """
    🏥 **QUICK HEALTH CHECK**

    Performs rapid system health validation for CI/CD pipelines.
    """
    print("🏥 Running Quick Health Check...")

    health_checks = [
        ("Python Environment", lambda: sys.version_info >= (3, 9)),
        ("Project Structure", lambda: Path("phase_3_0").exists()),
        ("Dependencies", lambda: Path("requirements.txt").exists()),
        ("Test Suite", lambda: Path("phase_3_0/tests").exists()),
        ("Infrastructure", lambda: Path("phase_3_0/infrastructure").exists()),
        ("Monitoring", lambda: Path("phase_3_0/monitoring").exists()),
    ]

    passed = 0
    total = len(health_checks)

    for check_name, check_func in health_checks:
        try:
            result = check_func()
            status = "✅" if result else "❌"
            print(f"  {status} {check_name}")
            if result:
                passed += 1
        except Exception as e:
            print(f"  ❌ {check_name} (Error: {e})")

    health_score = (passed / total) * 100
    print(f"\n🏥 Health Score: {health_score:.1f}% ({passed}/{total})")

    if health_score >= 80:
        print("✅ System health check passed")
        return 0
    else:
        print("❌ System health check failed")
        return 1


if __name__ == "__main__":
    # Check if quick health check requested
    if len(sys.argv) > 1 and sys.argv[1] == "--health-check":
        exit(run_quick_health_check())

    print("🚀 Starting Phase 3.0 Enterprise Test Runner...")

    try:
        # Run main test execution
        result = asyncio.run(main())
        exit(result)
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        exit(1)


# Export main components
__all__ = [
    'Phase3TestRunner',
    'TestConfiguration',
    'TestResult',
    'TestSuiteResult',
    'parse_arguments',
    'main',
    'run_quick_health_check'
]
