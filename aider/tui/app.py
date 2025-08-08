from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widgets import Header, Footer, Input, Button, DataTable, TextLog, Static
from textual.containers import Container, Horizontal

from ..hive import create_hive_coordinator, HiveCoordinator


class HiveTUI(App):
    """Textual-based TUI for the Aider Hive system.

    Provides:
    - Live system status (state, health, key metrics)
    - Agent table view
    - Request input + async processing
    - Log output
    """

    CSS = """
    Screen { layout: vertical; }
    #top-bar { layout: horizontal; height: 3; padding: 0 1; }
    #status { height: auto; padding: 1; }
    #main { layout: horizontal; height: 1fr; }
    #left { width: 1fr; padding: 1; }
    #right { width: 1fr; padding: 1; }
    #agents { height: 1fr; }
    #log { height: 1fr; border: round $secondary; } 
    """

    coordinator: reactive[Optional[HiveCoordinator]] = reactive(None)
    status: reactive[Dict[str, Any]] = reactive(dict)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="top-bar"):
            yield Input(placeholder="Type a request and press Enter…", id="request")
            yield Button("Send", id="send", variant="primary")

        yield Static("Starting…", id="status")

        with Container(id="main"):
            with Container(id="left"):
                agents = DataTable(id="agents")
                agents.cursor_type = "row"
                yield agents
            with Container(id="right"):
                yield TextLog(id="log", highlight=True, wrap=True)

        yield Footer()

    def on_mount(self) -> None:
        self._log("Initializing Hive TUI…")
        # Start hive in background and set a periodic status refresh
        self.run_worker(self._start_hive(), exclusive=True)
        # Poll status every 2 seconds
        self.set_interval(2.0, self._refresh_status)

    async def _start_hive(self) -> None:
        try:
            self._log("Creating HiveCoordinator…")
            coordinator = await create_hive_coordinator()
            started = await coordinator.start()
            if not started:
                self._log("Failed to start Hive system", error=True)
                return
            self.coordinator = coordinator
            self._log("Hive system started")
            await self._update_status_once()
        except Exception as exc:  # noqa: BLE001
            self._log(f"Start error: {exc}", error=True)

    async def _stop_hive(self) -> None:
        try:
            if self.coordinator is not None:
                self._log("Stopping Hive system…")
                await self.coordinator.stop()
                self._log("Hive system stopped")
        except Exception as exc:  # noqa: BLE001
            self._log(f"Stop error: {exc}", error=True)

    def _refresh_status(self) -> None:
        if self.coordinator is not None:
            self.run_worker(self._update_status_once())

    async def _update_status_once(self) -> None:
        if self.coordinator is None:
            return
        try:
            status = await self.coordinator.get_system_status()
            self.status = status
            self._render_status(status)
            self._render_agents(status.get("agents", {}))
        except Exception as exc:  # noqa: BLE001
            self._log(f"Status error: {exc}", error=True)

    def _render_status(self, status: Dict[str, Any]) -> None:
        state = status.get("hive_state", "unknown").upper()
        metrics = status.get("metrics", {})
        health = status.get("health_status", {})
        text = (
            f"State: {state}\n"
            f"Health: {'HEALTHY' if health.get('is_healthy') else 'UNHEALTHY'}\n"
            f"Active Agents: {metrics.get('active_agents', 0)}/{metrics.get('total_agents', 0)}\n"
            f"Requests Processed: {metrics.get('total_requests_processed', 0)}\n"
            f"Avg Response Time: {metrics.get('average_response_time', 0):.2f}s\n"
            f"Error Rate: {metrics.get('error_rate', 0):.2f}%\n"
        )
        self.query_one("#status", Static).update(text)

    def _render_agents(self, agents: Dict[str, Any]) -> None:
        table = self.query_one(DataTable)
        table.clear(columns=True)
        table.add_columns("Agent ID", "State")
        for agent_id, info in sorted(agents.items()):
            state = info.get("state", "unknown")
            table.add_row(agent_id, state)

    async def _process_request_text(self, text: str) -> None:
        if not text.strip():
            return
        if self.coordinator is None:
            self._log("Hive not started yet", error=True)
            return
        try:
            self._log(f"→ {text}")
            result = await self.coordinator.process_request(text)
            if result.get("success"):
                response = result.get("response")
                self._log(f"✓ Done: {response}")
            else:
                self._log(f"✗ Failed: {result.get('error')}", error=True)
        except Exception as exc:  # noqa: BLE001
            self._log(f"Request error: {exc}", error=True)
        finally:
            await self._update_status_once()

    def _log(self, message: str, *, error: bool = False) -> None:
        log = self.query_one(TextLog)
        if error:
            log.write(f"[red]{message}[/red]")
        else:
            log.write(message)

    def on_input_submitted(self, event: Input.Submitted) -> None:  # noqa: D401
        """Handle Enter in the request input."""
        if event.input.id == "request":
            text = event.value
            event.input.value = ""
            self.run_worker(self._process_request_text(text))

    def on_button_pressed(self, event: Button.Pressed) -> None:  # noqa: D401
        """Handle Send button."""
        if event.button.id == "send":
            input_widget = self.query_one("#request", Input)
            text = input_widget.value
            input_widget.value = ""
            self.run_worker(self._process_request_text(text))

    async def on_unmount(self) -> None:
        # Ensure hive shuts down cleanly when app exits
        await self._stop_hive()


def main() -> None:
    app = HiveTUI()
    app.run()


if __name__ == "__main__":
    main()


