#!/usr/bin/env python3
"""
Test Runner for Void-basic Project
==================================

A comprehensive test runner for the reorganized test structure.
Supports running tests by category, individual files, or all tests.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --category agents  # Run agent tests only
    python run_tests.py --category integration  # Run integration tests
    python run_tests.py --file test_ai_integration.py  # Run specific test
    python run_tests.py --list             # List available test categories
    python run_tests.py --verbose          # Verbose output
    python run_tests.py --parallel         # Run tests in parallel
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
import importlib.util

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test categories and their descriptions
TEST_CATEGORIES = {
    'agents': {
        'path': 'tests/agents',
        'description': 'Agent-specific tests (CodeAgent, ContextAgent, GitAgent, etc.)',
        'files': ['test_code_agent_ai_enhancement.py', 'test_context_agent_enhancement.py', 'test_git_agent_enhancement.py']
    },
    'integration': {
        'path': 'tests/integration',
        'description': 'Integration tests for multi-component interactions',
        'files': ['test_ai_integration.py']
    },
    'models': {
        'path': 'tests/models',
        'description': 'AI model integration and management tests',
        'files': ['test_model_integration.py']
    },
    'workflows': {
        'path': 'tests/workflows',
        'description': 'Autonomous workflow and orchestration tests',
        'files': ['test_autonomous_workflow_system.py']
    },
    'basic': {
        'path': 'tests/basic',
        'description': 'Basic functionality and unit tests',
        'files': []  # Will be discovered dynamically
    },
    'browser': {
        'path': 'tests/browser',
        'description': 'Browser automation and web interface tests',
        'files': []
    },
    'help': {
        'path': 'tests/help',
        'description': 'Help system and documentation tests',
        'files': []
    },
    'scrape': {
        'path': 'tests/scrape',
        'description': 'Web scraping functionality tests',
        'files': []
    }
}

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class TestRunner:
    """Main test runner class."""

    def __init__(self, verbose: bool = False, parallel: bool = False):
        self.verbose = verbose
        self.parallel = parallel
        self.results = {}
        self.start_time = time.time()

    def print_header(self):
        """Print test runner header."""
        print(f"{Colors.BOLD}{Colors.CYAN}")
        print("=" * 70)
        print("🚀 VOID-BASIC TEST RUNNER")
        print("=" * 70)
        print(f"{Colors.END}")
        print(f"Project Root: {PROJECT_ROOT}")
        print(f"Python Version: {sys.version.split()[0]}")
        print()

    def list_categories(self):
        """List all available test categories."""
        print(f"{Colors.BOLD}{Colors.BLUE}Available Test Categories:{Colors.END}\n")

        for category, info in TEST_CATEGORIES.items():
            path = PROJECT_ROOT / info['path']
            test_count = len(self.discover_test_files(path))

            print(f"{Colors.GREEN}📁 {category}{Colors.END}")
            print(f"   Path: {info['path']}")
            print(f"   Description: {info['description']}")
            print(f"   Test Files: {test_count}")

            if self.verbose and test_count > 0:
                files = self.discover_test_files(path)
                for file in files[:5]:  # Show first 5 files
                    print(f"     • {file}")
                if len(files) > 5:
                    print(f"     ... and {len(files) - 5} more")
            print()

    def discover_test_files(self, test_dir: Path) -> List[str]:
        """Discover test files in a directory."""
        if not test_dir.exists():
            return []

        test_files = []
        for file in test_dir.rglob("*.py"):
            if file.name.startswith("test_") or file.name.endswith("_test.py"):
                test_files.append(file.name)

        return sorted(test_files)

    def run_pytest(self, test_path: str) -> Dict:
        """Run tests using pytest."""
        cmd = [sys.executable, "-m", "pytest", test_path, "-v"]

        if self.parallel:
            cmd.extend(["-n", "auto"])  # Requires pytest-xdist

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=300  # 5 minute timeout
            )

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': 'Test execution timed out after 5 minutes',
                'returncode': -1
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': f"Error running tests: {str(e)}",
                'returncode': -1
            }

    def run_python_file(self, file_path: Path) -> Dict:
        """Run a Python test file directly."""
        try:
            result = subprocess.run(
                [sys.executable, str(file_path)],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=180  # 3 minute timeout
            )

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': 'Test execution timed out after 3 minutes',
                'returncode': -1
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': f"Error running test: {str(e)}",
                'returncode': -1
            }

    def run_category_tests(self, category: str) -> bool:
        """Run all tests in a category."""
        if category not in TEST_CATEGORIES:
            print(f"{Colors.RED}❌ Unknown category: {category}{Colors.END}")
            return False

        info = TEST_CATEGORIES[category]
        test_dir = PROJECT_ROOT / info['path']

        if not test_dir.exists():
            print(f"{Colors.YELLOW}⚠️  Test directory not found: {test_dir}{Colors.END}")
            return False

        print(f"{Colors.BOLD}{Colors.BLUE}🧪 Running {category} tests...{Colors.END}")
        print(f"Directory: {info['path']}")
        print(f"Description: {info['description']}")
        print()

        # Try pytest first, fall back to direct execution
        result = self.run_pytest(str(test_dir))

        if result['success']:
            print(f"{Colors.GREEN}✅ {category} tests passed{Colors.END}")
            if self.verbose:
                print(result['stdout'])
        else:
            print(f"{Colors.RED}❌ {category} tests failed{Colors.END}")
            if result['stderr']:
                print(f"{Colors.RED}Error output:{Colors.END}")
                print(result['stderr'])
            if self.verbose and result['stdout']:
                print(f"{Colors.YELLOW}Standard output:{Colors.END}")
                print(result['stdout'])

        self.results[category] = result
        return result['success']

    def run_specific_file(self, filename: str) -> bool:
        """Run a specific test file."""
        # Search for the file in all test directories
        test_file = None

        for category, info in TEST_CATEGORIES.items():
            test_dir = PROJECT_ROOT / info['path']
            potential_file = test_dir / filename
            if potential_file.exists():
                test_file = potential_file
                break

        # Also check the main tests directory
        main_test_file = PROJECT_ROOT / "tests" / filename
        if main_test_file.exists():
            test_file = main_test_file

        if not test_file:
            print(f"{Colors.RED}❌ Test file not found: {filename}{Colors.END}")
            return False

        print(f"{Colors.BOLD}{Colors.BLUE}🧪 Running {filename}...{Colors.END}")
        print(f"File: {test_file}")
        print()

        # Try pytest first, then direct execution
        result = self.run_pytest(str(test_file))

        if not result['success']:
            # Fall back to direct Python execution
            result = self.run_python_file(test_file)

        if result['success']:
            print(f"{Colors.GREEN}✅ {filename} passed{Colors.END}")
            if self.verbose:
                print(result['stdout'])
        else:
            print(f"{Colors.RED}❌ {filename} failed{Colors.END}")
            if result['stderr']:
                print(f"{Colors.RED}Error output:{Colors.END}")
                print(result['stderr'])
            if self.verbose and result['stdout']:
                print(f"{Colors.YELLOW}Standard output:{Colors.END}")
                print(result['stdout'])

        self.results[filename] = result
        return result['success']

    def run_all_tests(self) -> bool:
        """Run all tests in all categories."""
        print(f"{Colors.BOLD}{Colors.MAGENTA}🚀 Running ALL tests...{Colors.END}\n")

        all_passed = True

        for category in TEST_CATEGORIES.keys():
            success = self.run_category_tests(category)
            all_passed = all_passed and success
            print()  # Add spacing between categories

        return all_passed

    def print_summary(self):
        """Print test execution summary."""
        end_time = time.time()
        duration = end_time - self.start_time

        print(f"{Colors.BOLD}{Colors.CYAN}")
        print("=" * 70)
        print("📊 TEST EXECUTION SUMMARY")
        print("=" * 70)
        print(f"{Colors.END}")

        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result['success'])
        failed_tests = total_tests - passed_tests

        print(f"⏱️  Total Duration: {duration:.2f} seconds")
        print(f"📁 Total Categories/Files: {total_tests}")
        print(f"{Colors.GREEN}✅ Passed: {passed_tests}{Colors.END}")
        print(f"{Colors.RED}❌ Failed: {failed_tests}{Colors.END}")

        if failed_tests > 0:
            print(f"\n{Colors.RED}Failed Tests:{Colors.END}")
            for name, result in self.results.items():
                if not result['success']:
                    print(f"  • {name}")

        print(f"\n{Colors.BOLD}Overall Result: ", end="")
        if failed_tests == 0:
            print(f"{Colors.GREEN}ALL TESTS PASSED! 🎉{Colors.END}")
        else:
            print(f"{Colors.RED}SOME TESTS FAILED 😞{Colors.END}")

        print()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Void-basic Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                     # Run all tests
  python run_tests.py --category agents   # Run agent tests
  python run_tests.py --file test_ai_integration.py  # Run specific file
  python run_tests.py --list              # List test categories
  python run_tests.py --verbose           # Verbose output
        """
    )

    parser.add_argument(
        '--category', '-c',
        choices=list(TEST_CATEGORIES.keys()),
        help='Run tests for a specific category'
    )

    parser.add_argument(
        '--file', '-f',
        help='Run a specific test file'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available test categories'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Run tests in parallel (requires pytest-xdist)'
    )

    args = parser.parse_args()

    runner = TestRunner(verbose=args.verbose, parallel=args.parallel)
    runner.print_header()

    if args.list:
        runner.list_categories()
        return

    success = True

    if args.category:
        success = runner.run_category_tests(args.category)
    elif args.file:
        success = runner.run_specific_file(args.file)
    else:
        success = runner.run_all_tests()

    runner.print_summary()

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
