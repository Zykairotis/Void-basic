"""
Enhanced CLI Interface for the Aider Multi-Agent Hive Architecture.

This module provides a comprehensive command-line interface for interacting with
the Aider Hive system, including:
- System startup and shutdown
- Request processing
- Status monitoring
- Agent management
- Interactive mode
- Configuration management
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

# Import hive components
from ..hive.hive_coordinator import HiveCoordinator, create_hive_coordinator, HiveState


class HiveCLI:
    """
    Enhanced command-line interface for the Aider Multi-Agent Hive Architecture.

    Provides comprehensive CLI commands for managing and interacting with
    the hive system.
    """

    def __init__(self):
        """Initialize the CLI."""
        self.coordinator: Optional[HiveCoordinator] = None
        self.logger = structlog.get_logger().bind(component="hive_cli")
        self.interactive_mode = False

    async def run(self, args: List[str] = None) -> int:
        """Run the CLI with the given arguments."""
        if args is None:
            args = sys.argv[1:]

        parser = self._create_parser()
        parsed_args = parser.parse_args(args)

        # Setup logging
        self._setup_logging(parsed_args)

        try:
            # Execute the command
            result = await self._execute_command(parsed_args)
            return result
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            return 1
        except Exception as e:
            self.logger.error(f"CLI error: {e}", exc_info=True)
            return 1
        finally:
            if self.coordinator:
                await self.coordinator.stop()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            prog="aider-hive",
            description="Aider Multi-Agent Hive Architecture CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  aider-hive start                     # Start the hive system
  aider-hive request "create a function" # Process a request
  aider-hive status                    # Show system status
  aider-hive interactive               # Enter interactive mode
  aider-hive agent scale code 3        # Scale code agents to 3 instances
            """
        )

        # Global options
        parser.add_argument(
            "--config", "-c",
            type=str,
            help="Path to configuration file"
        )
        parser.add_argument(
            "--project-root", "-p",
            type=str,
            default=".",
            help="Project root directory (default: current directory)"
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode"
        )
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose output"
        )
        parser.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="Suppress output except errors"
        )
        parser.add_argument(
            "--output-format",
            choices=["text", "json", "yaml"],
            default="text",
            help="Output format (default: text)"
        )

        # Create subparsers for commands
        subparsers = parser.add_subparsers(
            dest="command",
            help="Available commands",
            metavar="COMMAND"
        )

        # Start command
        start_parser = subparsers.add_parser(
            "start",
            help="Start the hive system"
        )
        start_parser.add_argument(
            "--daemon", "-d",
            action="store_true",
            help="Run as daemon"
        )
        start_parser.add_argument(
            "--port",
            type=int,
            default=8080,
            help="Port for web interface (default: 8080)"
        )

        # Stop command
        stop_parser = subparsers.add_parser(
            "stop",
            help="Stop the hive system"
        )
        stop_parser.add_argument(
            "--timeout",
            type=float,
            default=30.0,
            help="Shutdown timeout in seconds (default: 30)"
        )

        # Request command
        request_parser = subparsers.add_parser(
            "request",
            help="Process a user request"
        )
        request_parser.add_argument(
            "text",
            help="Request text"
        )
        request_parser.add_argument(
            "--context",
            type=str,
            help="Additional context (JSON string)"
        )
        request_parser.add_argument(
            "--user-id",
            type=str,
            help="User identifier"
        )
        request_parser.add_argument(
            "--wait",
            action="store_true",
            help="Wait for completion"
        )

        # Status command
        status_parser = subparsers.add_parser(
            "status",
            help="Show system status"
        )
        status_parser.add_argument(
            "--detailed",
            action="store_true",
            help="Show detailed status"
        )
        status_parser.add_argument(
            "--refresh",
            type=int,
            default=0,
            help="Auto-refresh interval in seconds"
        )

        # Health command
        health_parser = subparsers.add_parser(
            "health",
            help="Check system health"
        )
        health_parser.add_argument(
            "--component",
            type=str,
            help="Check specific component"
        )

        # Metrics command
        metrics_parser = subparsers.add_parser(
            "metrics",
            help="Show system metrics"
        )
        metrics_parser.add_argument(
            "--format",
            choices=["table", "json", "prometheus"],
            default="table",
            help="Metrics format"
        )

        # Agent management commands
        agent_parser = subparsers.add_parser(
            "agent",
            help="Agent management commands"
        )
        agent_subparsers = agent_parser.add_subparsers(
            dest="agent_command",
            help="Agent commands"
        )

        # Agent list
        agent_subparsers.add_parser(
            "list",
            help="List all agents"
        )

        # Agent scale
        scale_parser = agent_subparsers.add_parser(
            "scale",
            help="Scale agent instances"
        )
        scale_parser.add_argument(
            "agent_type",
            choices=["orchestrator", "code", "context", "git"],
            help="Agent type to scale"
        )
        scale_parser.add_argument(
            "instances",
            type=int,
            help="Target number of instances"
        )

        # Agent status
        agent_status_parser = agent_subparsers.add_parser(
            "status",
            help="Show agent status"
        )
        agent_status_parser.add_argument(
            "agent_id",
            nargs="?",
            help="Specific agent ID (optional)"
        )

        # Interactive command
        interactive_parser = subparsers.add_parser(
            "interactive",
            help="Enter interactive mode"
        )
        interactive_parser.add_argument(
            "--prompt",
            type=str,
            default="aider-hive> ",
            help="Custom prompt"
        )

        # Config command
        config_parser = subparsers.add_parser(
            "config",
            help="Configuration management"
        )
        config_subparsers = config_parser.add_subparsers(
            dest="config_command",
            help="Config commands"
        )

        config_subparsers.add_parser(
            "show",
            help="Show current configuration"
        )

        generate_parser = config_subparsers.add_parser(
            "generate",
            help="Generate default configuration"
        )
        generate_parser.add_argument(
            "--output",
            type=str,
            help="Output file path"
        )

        validate_parser = config_subparsers.add_parser(
            "validate",
            help="Validate configuration"
        )
        validate_parser.add_argument(
            "config_file",
            help="Configuration file to validate"
        )

        return parser

    def _setup_logging(self, args: argparse.Namespace) -> None:
        """Setup logging based on CLI arguments."""
        if args.quiet:
            level = logging.ERROR
        elif args.verbose:
            level = logging.DEBUG
        elif args.debug:
            level = logging.DEBUG
        else:
            level = logging.INFO

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if args.output_format == "json" else structlog.dev.ConsoleRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Set root logger level
        logging.basicConfig(level=level)

    async def _execute_command(self, args: argparse.Namespace) -> int:
        """Execute the parsed command."""
        if not args.command:
            print("No command specified. Use --help for usage information.")
            return 1

        # Initialize coordinator for most commands
        if args.command not in ["config"]:
            self.coordinator = HiveCoordinator(
                config_path=args.config,
                project_root=os.path.abspath(args.project_root),
                debug_mode=args.debug
            )

            if not await self.coordinator.initialize():
                self.logger.error("Failed to initialize hive coordinator")
                return 1

        # Route to command handlers
        if args.command == "start":
            return await self._handle_start(args)
        elif args.command == "stop":
            return await self._handle_stop(args)
        elif args.command == "request":
            return await self._handle_request(args)
        elif args.command == "status":
            return await self._handle_status(args)
        elif args.command == "health":
            return await self._handle_health(args)
        elif args.command == "metrics":
            return await self._handle_metrics(args)
        elif args.command == "agent":
            return await self._handle_agent(args)
        elif args.command == "interactive":
            return await self._handle_interactive(args)
        elif args.command == "config":
            return await self._handle_config(args)
        else:
            self.logger.error(f"Unknown command: {args.command}")
            return 1

    async def _handle_start(self, args: argparse.Namespace) -> int:
        """Handle start command."""
        try:
            self.logger.info("Starting Aider Hive System...")

            if not await self.coordinator.start():
                self.logger.error("Failed to start hive system")
                return 1

            self.logger.info("Hive system started successfully")

            if args.daemon:
                self.logger.info("Running in daemon mode - press Ctrl+C to stop")
                try:
                    await self.coordinator.shutdown_event.wait()
                except KeyboardInterrupt:
                    self.logger.info("Shutdown signal received")
            else:
                # Show status and exit
                await self._show_status(detailed=False)

            return 0

        except Exception as e:
            self.logger.error(f"Failed to start hive system: {e}")
            return 1

    async def _handle_stop(self, args: argparse.Namespace) -> int:
        """Handle stop command."""
        try:
            self.logger.info("Stopping Aider Hive System...")
            await self.coordinator.stop(timeout=args.timeout)
            self.logger.info("Hive system stopped successfully")
            return 0
        except Exception as e:
            self.logger.error(f"Failed to stop hive system: {e}")
            return 1

    async def _handle_request(self, args: argparse.Namespace) -> int:
        """Handle request command."""
        try:
            # Start the system if not already running
            if self.coordinator.state != HiveState.RUNNING:
                if not await self.coordinator.start():
                    self.logger.error("Failed to start hive system")
                    return 1

            # Parse context if provided
            context = None
            if args.context:
                try:
                    context = json.loads(args.context)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid context JSON: {e}")
                    return 1

            self.logger.info(f"Processing request: {args.text[:100]}...")

            # Process the request
            result = await self.coordinator.process_request(
                request=args.text,
                context=context,
                user_id=args.user_id
            )

            # Display result
            self._display_result(result, args.output_format)

            return 0 if result.get('success', False) else 1

        except Exception as e:
            self.logger.error(f"Failed to process request: {e}")
            return 1

    async def _handle_status(self, args: argparse.Namespace) -> int:
        """Handle status command."""
        try:
            if args.refresh > 0:
                # Auto-refresh mode
                while True:
                    os.system('clear' if os.name == 'posix' else 'cls')
                    await self._show_status(args.detailed)
                    await asyncio.sleep(args.refresh)
            else:
                await self._show_status(args.detailed)
            return 0
        except KeyboardInterrupt:
            return 0
        except Exception as e:
            self.logger.error(f"Failed to get status: {e}")
            return 1

    async def _handle_health(self, args: argparse.Namespace) -> int:
        """Handle health command."""
        try:
            if self.coordinator.state != HiveState.RUNNING:
                if not await self.coordinator.start():
                    self.logger.error("Failed to start hive system")
                    return 1

            status = await self.coordinator.get_system_status()
            health = status.get('health_status', {})

            if args.component:
                # Show specific component health
                component_health = health.get('component_health', {})
                if args.component in component_health:
                    is_healthy = component_health[args.component]
                    print(f"Component '{args.component}': {'HEALTHY' if is_healthy else 'UNHEALTHY'}")
                    return 0 if is_healthy else 1
                else:
                    print(f"Component '{args.component}' not found")
                    return 1
            else:
                # Show overall health
                is_healthy = health.get('is_healthy', False)
                print(f"System Health: {'HEALTHY' if is_healthy else 'UNHEALTHY'}")

                if health.get('issues'):
                    print("\nIssues:")
                    for issue in health['issues']:
                        print(f"  - {issue}")

                if health.get('warnings'):
                    print("\nWarnings:")
                    for warning in health['warnings']:
                        print(f"  - {warning}")

                return 0 if is_healthy else 1

        except Exception as e:
            self.logger.error(f"Failed to check health: {e}")
            return 1

    async def _handle_metrics(self, args: argparse.Namespace) -> int:
        """Handle metrics command."""
        try:
            if self.coordinator.state != HiveState.RUNNING:
                if not await self.coordinator.start():
                    self.logger.error("Failed to start hive system")
                    return 1

            status = await self.coordinator.get_system_status()
            metrics = status.get('metrics', {})

            if args.format == "json":
                print(json.dumps(metrics, indent=2))
            elif args.format == "table":
                self._display_metrics_table(metrics)
            elif args.format == "prometheus":
                self._display_metrics_prometheus(metrics)

            return 0

        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            return 1

    async def _handle_agent(self, args: argparse.Namespace) -> int:
        """Handle agent management commands."""
        try:
            if not args.agent_command:
                print("No agent command specified")
                return 1

            if self.coordinator.state != HiveState.RUNNING:
                if not await self.coordinator.start():
                    self.logger.error("Failed to start hive system")
                    return 1

            if args.agent_command == "list":
                return await self._handle_agent_list(args)
            elif args.agent_command == "scale":
                return await self._handle_agent_scale(args)
            elif args.agent_command == "status":
                return await self._handle_agent_status(args)
            else:
                print(f"Unknown agent command: {args.agent_command}")
                return 1

        except Exception as e:
            self.logger.error(f"Agent command failed: {e}")
            return 1

    async def _handle_agent_list(self, args: argparse.Namespace) -> int:
        """Handle agent list command."""
        status = await self.coordinator.get_system_status()
        agents = status.get('agents', {})

        print("Active Agents:")
        print("-" * 50)
        for agent_id, agent_info in agents.items():
            agent_type = agent_info.get('agent_type', 'unknown')
            agent_state = agent_info.get('state', 'unknown')
            print(f"  {agent_id}: {agent_type} ({agent_state})")

        return 0

    async def _handle_agent_scale(self, args: argparse.Namespace) -> int:
        """Handle agent scale command."""
        result = await self.coordinator.scale_agent(args.agent_type, args.instances)

        if result['success']:
            print(f"Successfully scaled {args.agent_type} to {args.instances} instances")
            return 0
        else:
            print(f"Failed to scale {args.agent_type}: {result.get('error', 'Unknown error')}")
            return 1

    async def _handle_agent_status(self, args: argparse.Namespace) -> int:
        """Handle agent status command."""
        status = await self.coordinator.get_system_status()
        agents = status.get('agents', {})

        if args.agent_id:
            if args.agent_id in agents:
                agent_info = agents[args.agent_id]
                print(f"Agent {args.agent_id}:")
                print(json.dumps(agent_info, indent=2))
            else:
                print(f"Agent {args.agent_id} not found")
                return 1
        else:
            print("Agent Status Summary:")
            print("-" * 50)
            for agent_id, agent_info in agents.items():
                state = agent_info.get('state', 'unknown')
                print(f"  {agent_id}: {state}")

        return 0

    async def _handle_interactive(self, args: argparse.Namespace) -> int:
        """Handle interactive mode."""
        try:
            if self.coordinator.state != HiveState.RUNNING:
                if not await self.coordinator.start():
                    self.logger.error("Failed to start hive system")
                    return 1

            self.interactive_mode = True
            print("Entering interactive mode. Type 'help' for commands, 'exit' to quit.")

            while True:
                try:
                    user_input = input(args.prompt).strip()

                    if not user_input:
                        continue
                    elif user_input.lower() in ['exit', 'quit', 'q']:
                        break
                    elif user_input.lower() == 'help':
                        self._show_interactive_help()
                    elif user_input.lower() == 'status':
                        await self._show_status(detailed=False)
                    elif user_input.lower() == 'health':
                        await self._handle_health(argparse.Namespace(component=None))
                    else:
                        # Process as a request
                        result = await self.coordinator.process_request(user_input)
                        self._display_result(result, "text")

                except EOFError:
                    break
                except KeyboardInterrupt:
                    print("\nUse 'exit' to quit.")
                    continue

            print("Exiting interactive mode.")
            return 0

        except Exception as e:
            self.logger.error(f"Interactive mode failed: {e}")
            return 1

    async def _handle_config(self, args: argparse.Namespace) -> int:
        """Handle configuration commands."""
        try:
            if not args.config_command:
                print("No config command specified")
                return 1

            if args.config_command == "show":
                return self._handle_config_show(args)
            elif args.config_command == "generate":
                return self._handle_config_generate(args)
            elif args.config_command == "validate":
                return self._handle_config_validate(args)
            else:
                print(f"Unknown config command: {args.config_command}")
                return 1

        except Exception as e:
            self.logger.error(f"Config command failed: {e}")
            return 1

    def _handle_config_show(self, args: argparse.Namespace) -> int:
        """Show current configuration."""
        # This would show the current config
        print("Configuration management not yet implemented")
        return 0

    def _handle_config_generate(self, args: argparse.Namespace) -> int:
        """Generate default configuration."""
        # This would generate default config
        print("Configuration generation not yet implemented")
        return 0

    def _handle_config_validate(self, args: argparse.Namespace) -> int:
        """Validate configuration file."""
        # This would validate config
        print("Configuration validation not yet implemented")
        return 0

    async def _show_status(self, detailed: bool = False) -> None:
        """Show system status."""
        try:
            status = await self.coordinator.get_system_status()

            print("Aider Hive System Status")
            print("=" * 50)
            print(f"State: {status['hive_state']}")
            print(f"Coordinator ID: {status['coordinator_id']}")

            if status.get('started_at'):
                started_at = datetime.fromisoformat(status['started_at'].replace('Z', '+00:00'))
                uptime = datetime.now(started_at.tzinfo) - started_at
                print(f"Uptime: {uptime}")

            # Health status
            health = status.get('health_status', {})
            health_status = "HEALTHY" if health.get('is_healthy') else "UNHEALTHY"
            print(f"Health: {health_status}")

            # Metrics
            metrics = status.get('metrics', {})
            print(f"Active Agents: {metrics.get('active_agents', 0)}/{metrics.get('total_agents', 0)}")
            print(f"Requests Processed: {metrics.get('total_requests_processed', 0)}")
            print(f"Average Response Time: {metrics.get('average_response_time', 0):.2f}s")
            print(f"Error Rate: {metrics.get('error_rate', 0):.2f}%")

            if detailed:
                print("\nDetailed Information:")
                print("-" * 30)

                # Show agent details
                agents = status.get('agents', {})
                if agents:
                    print("Agents:")
                    for agent_id, agent_info in agents.items():
                        print(f"  {agent_id}: {agent_info.get('state', 'unknown')}")

                # Show health issues
                if health.get('issues'):
                    print("Issues:")
                    for issue in health['issues']:
                        print(f"  - {issue}")

                if health.get('warnings'):
                    print("Warnings:")
                    for warning in health['warnings']:
                        print(f"  - {warning}")

        except Exception as e:
            self.logger.error(f"Failed to show status: {e}")

    def _display_result(self, result: Dict[str, Any], format_type: str) -> None:
        """Display command result."""
        if format_type == "json":
            print(json.dumps(result, indent=2, default=str))
        else:
            # Text format
            if result.get('success'):
                print("✓ Request completed successfully")
                if 'response' in result:
                    response = result['response']
                    if isinstance(response, dict):
                        if 'summary' in response:
                            print(f"Summary: {response['summary']}")
                        if 'successful_steps' in response:
                            print(f"Successful steps: {response['successful_steps']}")
                        if 'failed_steps' in response:
                            print(f"Failed steps: {response['failed_steps']}")
                    else:
                        print(f"Response: {response}")
            else:
                print("✗ Request failed")
                if 'error' in result:
                    print(f"Error: {result['error']}")

            if 'processing_time' in result:
                print(f"Processing time: {result['processing_time']:.2f}s")

    def _display_metrics_table(self, metrics: Dict[str, Any]) -> None:
        """Display metrics in table format."""
        print("System Metrics")
        print("-" * 30)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key.replace('_', ' ').title()}: {value}")

    def _display_metrics_prometheus(self, metrics: Dict[str, Any]) -> None:
        """Display metrics in Prometheus format."""
        print("# Aider Hive Metrics")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"aider_hive_{key} {value}")

    def _show_interactive_help(self) -> None:
        """Show help for interactive mode."""
        print("Interactive Mode Commands:")
        print("  help     - Show this help")
        print("  status   - Show system status")
        print("  health   - Check system health")
        print("  exit     - Exit interactive mode")
        print("  <text>   - Process request")


def main() -> int:
    """Main CLI entry point."""
    cli = HiveCLI()
    return asyncio.run(cli.run())


if __name__ == "__main__":
    sys.exit(main())
