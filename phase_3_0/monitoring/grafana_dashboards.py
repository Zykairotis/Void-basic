"""
🚀 **PHASE 3.0 GRAFANA DASHBOARD CONFIGURATIONS**

Comprehensive Grafana dashboard definitions for enterprise production monitoring.
This module generates complete dashboard configurations for all Phase 3.0 components
including multi-tenancy, compliance, security, performance, and real-time monitoring.

Dashboard Categories:
- System Overview: High-level enterprise system health
- Multi-Tenant: Tenant-specific monitoring and isolation
- Compliance: GDPR, HIPAA, SOX automated compliance monitoring
- Performance: System performance, load, and scalability metrics
- Security: Security events, threats, and access monitoring
- Agent Operations: Multi-agent system monitoring
- Real-Time: WebSocket and live data monitoring
- Infrastructure: Kubernetes and system resource monitoring
"""

import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class GrafanaPanel:
    """Grafana dashboard panel configuration."""
    id: int
    title: str
    type: str
    targets: List[Dict]
    gridPos: Dict[str, int]
    options: Optional[Dict] = None
    fieldConfig: Optional[Dict] = None
    transformations: Optional[List[Dict]] = None


@dataclass
class GrafanaDashboard:
    """Complete Grafana dashboard configuration."""
    id: Optional[str]
    uid: str
    title: str
    description: str
    tags: List[str]
    panels: List[GrafanaPanel]
    time_range: Dict[str, str]
    refresh: str
    schemaVersion: int = 36
    version: int = 1


class Phase3MonitoringDashboards:
    """
    🏢 **PHASE 3.0 ENTERPRISE MONITORING DASHBOARDS**

    Generates comprehensive Grafana dashboards for enterprise production monitoring.
    Covers all critical aspects of Phase 3.0 including multi-tenancy, compliance,
    security, performance, and real-time operations monitoring.
    """

    def __init__(self):
        self.dashboards = {}
        self.panel_id_counter = 1

    def generate_all_dashboards(self) -> Dict[str, Dict]:
        """Generate all Phase 3.0 monitoring dashboards."""
        print("🚀 Generating Phase 3.0 Enterprise Grafana Dashboards...")

        self.dashboards = {
            "system_overview": self.create_system_overview_dashboard(),
            "multi_tenant": self.create_multi_tenant_dashboard(),
            "compliance": self.create_compliance_dashboard(),
            "performance": self.create_performance_dashboard(),
            "security": self.create_security_dashboard(),
            "agent_operations": self.create_agent_operations_dashboard(),
            "real_time": self.create_real_time_dashboard(),
            "infrastructure": self.create_infrastructure_dashboard()
        }

        print(f"✅ Generated {len(self.dashboards)} enterprise dashboards")
        return self.dashboards

    def create_system_overview_dashboard(self) -> Dict:
        """Create system overview dashboard for enterprise health monitoring."""
        panels = [
            # System Health Status
            self._create_stat_panel(
                title="System Health Score",
                query="system_health_score",
                unit="percent",
                color_mode="value",
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 4}
            ),

            # Active Tenants
            self._create_stat_panel(
                title="Active Tenants",
                query="count(active_tenants)",
                unit="short",
                color_mode="value",
                grid_pos={"x": 6, "y": 0, "w": 6, "h": 4}
            ),

            # Running Workflows
            self._create_stat_panel(
                title="Running Workflows",
                query="sum(active_workflows)",
                unit="short",
                color_mode="value",
                grid_pos={"x": 12, "y": 0, "w": 6, "h": 4}
            ),

            # Compliance Score
            self._create_stat_panel(
                title="Compliance Score",
                query="avg(compliance_score)",
                unit="percent",
                color_mode="value",
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 4}
            ),

            # System Performance Over Time
            self._create_timeseries_panel(
                title="System Performance Metrics",
                queries=[
                    "avg(cpu_usage_percent)",
                    "avg(memory_usage_percent)",
                    "avg(disk_usage_percent)"
                ],
                grid_pos={"x": 0, "y": 4, "w": 12, "h": 8}
            ),

            # Request Volume
            self._create_timeseries_panel(
                title="API Request Volume",
                queries=[
                    "rate(api_requests_total[5m])",
                    "rate(websocket_connections_total[5m])"
                ],
                grid_pos={"x": 12, "y": 4, "w": 12, "h": 8}
            ),

            # Error Rate Tracking
            self._create_timeseries_panel(
                title="System Error Rates",
                queries=[
                    "rate(api_errors_total[5m])",
                    "rate(application_errors_total[5m])",
                    "rate(database_errors_total[5m])"
                ],
                grid_pos={"x": 0, "y": 12, "w": 24, "h": 6}
            ),

            # Top Tenants by Activity
            self._create_table_panel(
                title="Top Active Tenants",
                query="topk(10, sum(tenant_activity) by (tenant_id))",
                grid_pos={"x": 0, "y": 18, "w": 12, "h": 6}
            ),

            # Recent System Events
            self._create_table_panel(
                title="Recent System Events",
                query="system_events",
                grid_pos={"x": 12, "y": 18, "w": 12, "h": 6}
            )
        ]

        return self._create_dashboard(
            uid="phase3-system-overview",
            title="Phase 3.0 - System Overview",
            description="Enterprise system health and performance overview",
            tags=["phase3", "overview", "enterprise"],
            panels=panels
        )

    def create_multi_tenant_dashboard(self) -> Dict:
        """Create multi-tenant monitoring dashboard."""
        panels = [
            # Tenant Count by Status
            self._create_stat_panel(
                title="Total Tenants",
                query="count(tenants)",
                unit="short",
                grid_pos={"x": 0, "y": 0, "w": 4, "h": 4}
            ),

            self._create_stat_panel(
                title="Active Tenants",
                query="count(tenants{status='active'})",
                unit="short",
                grid_pos={"x": 4, "y": 0, "w": 4, "h": 4}
            ),

            self._create_stat_panel(
                title="Suspended Tenants",
                query="count(tenants{status='suspended'})",
                unit="short",
                grid_pos={"x": 8, "y": 0, "w": 4, "h": 4}
            ),

            # Tenant Resource Usage
            self._create_timeseries_panel(
                title="Tenant CPU Usage",
                queries=["sum(tenant_cpu_usage) by (tenant_id)"],
                grid_pos={"x": 12, "y": 0, "w": 12, "h": 8}
            ),

            # Tenant Data Isolation Validation
            self._create_stat_panel(
                title="Data Isolation Score",
                query="avg(tenant_isolation_score)",
                unit="percent",
                color_mode="value",
                grid_pos={"x": 0, "y": 4, "w": 6, "h": 4}
            ),

            self._create_stat_panel(
                title="Cross-Tenant Violations",
                query="sum(cross_tenant_violations)",
                unit="short",
                color_mode="value",
                grid_pos={"x": 6, "y": 4, "w": 6, "h": 4}
            ),

            # Tenant Activity Heatmap
            self._create_heatmap_panel(
                title="Tenant Activity Heatmap",
                query="sum(tenant_requests_total) by (tenant_id, time)",
                grid_pos={"x": 0, "y": 8, "w": 24, "h": 8}
            ),

            # Tenant Resource Limits
            self._create_table_panel(
                title="Tenant Resource Usage vs Limits",
                query="tenant_resource_usage_with_limits",
                grid_pos={"x": 0, "y": 16, "w": 12, "h": 8}
            ),

            # Tenant Performance Metrics
            self._create_timeseries_panel(
                title="Tenant Response Times",
                queries=["avg(tenant_response_time) by (tenant_id)"],
                grid_pos={"x": 12, "y": 16, "w": 12, "h": 8}
            )
        ]

        return self._create_dashboard(
            uid="phase3-multi-tenant",
            title="Phase 3.0 - Multi-Tenant Monitoring",
            description="Multi-tenant system monitoring and isolation validation",
            tags=["phase3", "multi-tenant", "enterprise"],
            panels=panels
        )

    def create_compliance_dashboard(self) -> Dict:
        """Create compliance monitoring dashboard for GDPR, HIPAA, SOX."""
        panels = [
            # Overall Compliance Scores
            self._create_stat_panel(
                title="GDPR Compliance",
                query="gdpr_compliance_score",
                unit="percent",
                color_mode="value",
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 4}
            ),

            self._create_stat_panel(
                title="HIPAA Compliance",
                query="hipaa_compliance_score",
                unit="percent",
                color_mode="value",
                grid_pos={"x": 6, "y": 0, "w": 6, "h": 4}
            ),

            self._create_stat_panel(
                title="SOX Compliance",
                query="sox_compliance_score",
                unit="percent",
                color_mode="value",
                grid_pos={"x": 12, "y": 0, "w": 6, "h": 4}
            ),

            self._create_stat_panel(
                title="Overall Compliance",
                query="avg(compliance_scores)",
                unit="percent",
                color_mode="value",
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 4}
            ),

            # Compliance Violations Over Time
            self._create_timeseries_panel(
                title="Compliance Violations",
                queries=[
                    "sum(gdpr_violations_total)",
                    "sum(hipaa_violations_total)",
                    "sum(sox_violations_total)"
                ],
                grid_pos={"x": 0, "y": 4, "w": 12, "h": 8}
            ),

            # Data Subject Rights Requests
            self._create_timeseries_panel(
                title="Data Subject Rights Requests",
                queries=[
                    "sum(data_access_requests)",
                    "sum(data_erasure_requests)",
                    "sum(data_portability_requests)"
                ],
                grid_pos={"x": 12, "y": 4, "w": 12, "h": 8}
            ),

            # Audit Trail Completeness
            self._create_gauge_panel(
                title="Audit Trail Completeness",
                query="audit_trail_completeness_percent",
                grid_pos={"x": 0, "y": 12, "w": 8, "h": 8}
            ),

            # Compliance Automation Status
            self._create_table_panel(
                title="Compliance Automation Status",
                query="compliance_automation_status",
                grid_pos={"x": 8, "y": 12, "w": 16, "h": 8}
            ),

            # Recent Compliance Events
            self._create_table_panel(
                title="Recent Compliance Events",
                query="recent_compliance_events",
                grid_pos={"x": 0, "y": 20, "w": 24, "h": 6}
            )
        ]

        return self._create_dashboard(
            uid="phase3-compliance",
            title="Phase 3.0 - Compliance Monitoring",
            description="Automated compliance monitoring for GDPR, HIPAA, and SOX",
            tags=["phase3", "compliance", "gdpr", "hipaa", "sox"],
            panels=panels
        )

    def create_performance_dashboard(self) -> Dict:
        """Create performance monitoring dashboard."""
        panels = [
            # System Performance KPIs
            self._create_stat_panel(
                title="Avg Response Time",
                query="avg(response_time_seconds)",
                unit="s",
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 4}
            ),

            self._create_stat_panel(
                title="Requests/Second",
                query="rate(requests_total[1m])",
                unit="reqps",
                grid_pos={"x": 6, "y": 0, "w": 6, "h": 4}
            ),

            self._create_stat_panel(
                title="Error Rate",
                query="rate(errors_total[1m]) / rate(requests_total[1m]) * 100",
                unit="percent",
                grid_pos={"x": 12, "y": 0, "w": 6, "h": 4}
            ),

            self._create_stat_panel(
                title="Uptime",
                query="up",
                unit="percent",
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 4}
            ),

            # Response Time Distribution
            self._create_timeseries_panel(
                title="Response Time Percentiles",
                queries=[
                    "histogram_quantile(0.50, response_time_histogram)",
                    "histogram_quantile(0.95, response_time_histogram)",
                    "histogram_quantile(0.99, response_time_histogram)"
                ],
                grid_pos={"x": 0, "y": 4, "w": 12, "h": 8}
            ),

            # Throughput
            self._create_timeseries_panel(
                title="System Throughput",
                queries=[
                    "rate(api_requests_total[5m])",
                    "rate(workflow_executions_total[5m])",
                    "rate(database_operations_total[5m])"
                ],
                grid_pos={"x": 12, "y": 4, "w": 12, "h": 8}
            ),

            # Resource Utilization
            self._create_timeseries_panel(
                title="Resource Utilization",
                queries=[
                    "avg(cpu_usage_percent)",
                    "avg(memory_usage_percent)",
                    "avg(disk_io_utilization)"
                ],
                grid_pos={"x": 0, "y": 12, "w": 24, "h": 8}
            ),

            # Load Distribution
            self._create_table_panel(
                title="Load Distribution by Component",
                query="component_load_distribution",
                grid_pos={"x": 0, "y": 20, "w": 12, "h": 6}
            ),

            # Performance Alerts
            self._create_table_panel(
                title="Active Performance Alerts",
                query="performance_alerts",
                grid_pos={"x": 12, "y": 20, "w": 12, "h": 6}
            )
        ]

        return self._create_dashboard(
            uid="phase3-performance",
            title="Phase 3.0 - Performance Monitoring",
            description="System performance, throughput, and resource monitoring",
            tags=["phase3", "performance", "monitoring"],
            panels=panels
        )

    def create_security_dashboard(self) -> Dict:
        """Create security monitoring dashboard."""
        panels = [
            # Security Threat Level
            self._create_gauge_panel(
                title="Current Threat Level",
                query="security_threat_level",
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 8}
            ),

            # Failed Authentication Attempts
            self._create_stat_panel(
                title="Failed Auth (Last Hour)",
                query="sum(failed_auth_attempts[1h])",
                unit="short",
                grid_pos={"x": 6, "y": 0, "w": 6, "h": 4}
            ),

            # Active Security Incidents
            self._create_stat_panel(
                title="Active Security Incidents",
                query="count(active_security_incidents)",
                unit="short",
                color_mode="value",
                grid_pos={"x": 12, "y": 0, "w": 6, "h": 4}
            ),

            # Blocked Attacks
            self._create_stat_panel(
                title="Blocked Attacks (24h)",
                query="sum(blocked_attacks[24h])",
                unit="short",
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 4}
            ),

            # Security Events Timeline
            self._create_timeseries_panel(
                title="Security Events Over Time",
                queries=[
                    "rate(sql_injection_attempts[5m])",
                    "rate(xss_attempts[5m])",
                    "rate(brute_force_attempts[5m])",
                    "rate(unauthorized_access_attempts[5m])"
                ],
                grid_pos={"x": 6, "y": 4, "w": 18, "h": 8}
            ),

            # Geographic Attack Distribution
            self._create_worldmap_panel(
                title="Attack Origins (Geographic)",
                query="sum(attack_attempts) by (country)",
                grid_pos={"x": 0, "y": 12, "w": 12, "h": 8}
            ),

            # Top Attack Types
            self._create_pie_panel(
                title="Attack Types Distribution",
                query="sum(attacks) by (attack_type)",
                grid_pos={"x": 12, "y": 12, "w": 12, "h": 8}
            ),

            # Recent Security Events
            self._create_table_panel(
                title="Recent Security Events",
                query="recent_security_events",
                grid_pos={"x": 0, "y": 20, "w": 24, "h": 8}
            )
        ]

        return self._create_dashboard(
            uid="phase3-security",
            title="Phase 3.0 - Security Monitoring",
            description="Enterprise security monitoring and threat detection",
            tags=["phase3", "security", "monitoring"],
            panels=panels
        )

    def create_agent_operations_dashboard(self) -> Dict:
        """Create agent operations monitoring dashboard."""
        panels = [
            # Agent Status Overview
            self._create_stat_panel(
                title="Total Agents",
                query="count(agents)",
                unit="short",
                grid_pos={"x": 0, "y": 0, "w": 4, "h": 4}
            ),

            self._create_stat_panel(
                title="Active Agents",
                query="count(agents{status='active'})",
                unit="short",
                grid_pos={"x": 4, "y": 0, "w": 4, "h": 4}
            ),

            self._create_stat_panel(
                title="Failed Agents",
                query="count(agents{status='failed'})",
                unit="short",
                grid_pos={"x": 8, "y": 0, "w": 4, "h": 4}
            ),

            self._create_stat_panel(
                title="Agent Health Score",
                query="avg(agent_health_score)",
                unit="percent",
                grid_pos={"x": 12, "y": 0, "w": 4, "h": 4}
            ),

            # Agent Performance by Type
            self._create_timeseries_panel(
                title="Agent Performance by Type",
                queries=[
                    "avg(agent_response_time) by (agent_type)",
                    "rate(agent_requests_total) by (agent_type)"
                ],
                grid_pos={"x": 16, "y": 0, "w": 8, "h": 8}
            ),

            # Workflow Execution Status
            self._create_timeseries_panel(
                title="Workflow Execution Rate",
                queries=[
                    "rate(workflow_started_total[5m])",
                    "rate(workflow_completed_total[5m])",
                    "rate(workflow_failed_total[5m])"
                ],
                grid_pos={"x": 0, "y": 4, "w": 16, "h": 8}
            ),

            # Agent Resource Usage
            self._create_table_panel(
                title="Agent Resource Usage",
                query="agent_resource_usage",
                grid_pos={"x": 0, "y": 12, "w": 12, "h": 8}
            ),

            # Recent Agent Operations
            self._create_table_panel(
                title="Recent Agent Operations",
                query="recent_agent_operations",
                grid_pos={"x": 12, "y": 12, "w": 12, "h": 8}
            )
        ]

        return self._create_dashboard(
            uid="phase3-agent-operations",
            title="Phase 3.0 - Agent Operations",
            description="Multi-agent system operations and performance monitoring",
            tags=["phase3", "agents", "operations"],
            panels=panels
        )

    def create_real_time_dashboard(self) -> Dict:
        """Create real-time operations monitoring dashboard."""
        panels = [
            # WebSocket Connections
            self._create_stat_panel(
                title="Active WebSocket Connections",
                query="sum(websocket_connections_active)",
                unit="short",
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 4}
            ),

            # Real-time Update Rate
            self._create_stat_panel(
                title="Updates/Second",
                query="rate(realtime_updates_total[1m])",
                unit="ops",
                grid_pos={"x": 6, "y": 0, "w": 6, "h": 4}
            ),

            # Connection Quality
            self._create_stat_panel(
                title="Connection Quality",
                query="avg(websocket_connection_quality)",
                unit="percent",
                grid_pos={"x": 12, "y": 0, "w": 6, "h": 4}
            ),

            # Message Latency
            self._create_stat_panel(
                title="Avg Message Latency",
                query="avg(websocket_message_latency)",
                unit="ms",
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 4}
            ),

            # Real-time Data Flow
            self._create_timeseries_panel(
                title="Real-time Data Flow",
                queries=[
                    "rate(websocket_messages_sent[1m])",
                    "rate(websocket_messages_received[1m])",
                    "rate(websocket_messages_failed[1m])"
                ],
                grid_pos={"x": 0, "y": 4, "w": 24, "h": 8}
            ),

            # Connection Distribution
            self._create_pie_panel(
                title="Connections by Type",
                query="sum(websocket_connections) by (connection_type)",
                grid_pos={"x": 0, "y": 12, "w": 12, "h": 8}
            ),

            # Recent Real-time Events
            self._create_table_panel(
                title="Recent Real-time Events",
                query="recent_realtime_events",
                grid_pos={"x": 12, "y": 12, "w": 12, "h": 8}
            )
        ]

        return self._create_dashboard(
            uid="phase3-real-time",
            title="Phase 3.0 - Real-time Operations",
            description="WebSocket and real-time data monitoring",
            tags=["phase3", "realtime", "websocket"],
            panels=panels
        )

    def create_infrastructure_dashboard(self) -> Dict:
        """Create infrastructure monitoring dashboard for Kubernetes."""
        panels = [
            # Cluster Status
            self._create_stat_panel(
                title="Cluster Health",
                query="kube_cluster_health",
                unit="percent",
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 4}
            ),

            # Pod Status
            self._create_stat_panel(
                title="Running Pods",
                query="count(kube_pod_info{phase='Running'})",
                unit="short",
                grid_pos={"x": 6, "y": 0, "w": 6, "h": 4}
            ),

            # Node Status
            self._create_stat_panel(
                title="Ready Nodes",
                query="count(kube_node_status_condition{condition='Ready', status='true'})",
                unit="short",
                grid_pos={"x": 12, "y": 0, "w": 6, "h": 4}
            ),

            # Resource Usage
            self._create_stat_panel(
                title="Cluster CPU Usage",
                query="avg(cluster_cpu_usage_percent)",
                unit="percent",
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 4}
            ),

            # Pod Resource Usage
            self._create_timeseries_panel(
                title="Pod Resource Usage",
                queries=[
                    "sum(pod_cpu_usage) by (pod)",
                    "sum(pod_memory_usage) by (pod)"
                ],
                grid_pos={"x": 0, "y": 4, "w": 12, "h": 8}
            ),

            # Network Traffic
            self._create_timeseries_panel(
                title="Network Traffic",
                queries=[
                    "rate(container_network_receive_bytes_total[5m])",
                    "rate(container_network_transmit_bytes_total[5m])"
                ],
                grid_pos={"x": 12, "y": 4, "w": 12, "h": 8}
            ),

            # Storage Usage
            self._create_table_panel(
                title="Persistent Volume Usage",
                query="pv_usage_stats",
                grid_pos={"x": 0, "y": 12, "w": 12, "h": 8}
            ),

            # Recent Infrastructure Events
            self._create_table_panel(
                title="Kubernetes Events",
                query="kube_events",
                grid_pos={"x": 12, "y": 12, "w": 12, "h": 8}
            )
        ]

        return self._create_dashboard(
            uid="phase3-infrastructure",
            title="Phase 3.0 - Infrastructure Monitoring",
            description="Kubernetes cluster and infrastructure monitoring",
            tags=["phase3", "kubernetes", "infrastructure"],
            panels=panels
        )

    def export_dashboards(self, output_dir: str = "grafana_dashboards") -> Dict[str, str]:
        """Export all dashboards to JSON files."""
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        exported_files = {}

        for dashboard_name, dashboard_config in self.dashboards.items():
            filename = f"{dashboard_name}.json"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w') as f:
                json.dump(dashboard_config, f, indent=2)

            exported_files[dashboard_name] = filepath
            print(f"✅ Exported {dashboard_name} dashboard to {filepath}")

        return exported_files

    def _create_dashboard(self, uid: str, title: str, description: str,
                         tags: List[str], panels: List[Dict]) -> Dict:
        """Create base dashboard structure."""
        return {
            "id": None,
            "uid": uid,
            "title": title,
            "tags": tags,
            "style": "dark",
            "timezone": "browser",
            "panels": panels,
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "timepicker": {},
            "templating": {"list": []},
            "annotations": {"list": []},
            "refresh": "5s",
            "schemaVersion": 36,
            "version": 1,
            "links": []
        }

    def _create_stat_panel(self, title: str, query: str, unit: str = "short",
                          color_mode: str = "palette-classic",
                          grid_pos: Dict[str, int] = None) -> Dict:
        """Create a stat/single value panel."""
        panel_id = self._get_next_panel_id()

        return {
            "id": panel_id,
            "title": title,
            "type": "stat",
            "targets": [{
                "expr": query,
                "refId": "A"
            }],
            "gridPos": grid_pos or {"x": 0, "y": 0, "w": 6, "h": 4},
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "orientation": "auto",
                "textMode": "auto",
                "colorMode": color_mode
            },
            "fieldConfig": {
                "defaults": {
                    "unit": unit,
                    "custom": {
                        "displayMode": "list",
                        "placement": "bottom"
                    }
                }
            }
        }

    def _create_timeseries_panel(self, title: str, queries: List[str],
                                grid_pos: Dict[str, int] = None) -> Dict:
        """Create a time series panel."""
        panel_id = self._get_next_panel_id()

        targets = []
        for i, query in enumerate(queries):
            targets.append({
                "expr": query,
                "refId": chr(65 + i)  # A, B, C, etc.
            })

        return {
            "id": panel_id,
            "title": title,
            "type": "timeseries",
            "targets": targets,
            "gridPos": grid_pos or {"x": 0, "y": 0, "w": 12, "h": 8},
            "options": {
                "tooltip": {"mode": "single", "sort": "none"},
                "legend": {"displayMode": "list", "placement": "bottom"}
            },
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "drawStyle": "line",
                        "lineInterpolation": "linear",
                        "lineWidth": 1,
                        "fillOpacity": 0,
                        "gradientMode": "none",
                        "spanNulls": False,
                        "insertNulls": False,
                        "showPoints": "auto",
                        "pointSize": 5
                    }
                }
            }
        }

    def _create_table_panel(self, title: str, query: str,
                           grid_pos: Dict[str, int] = None) -> Dict:
        """Create a table panel."""
        panel_id = self._get_next_panel_id()

        return {
            "id": panel_id,
            "title": title,
            "type": "table",
            "targets": [{
                "expr": query,
                "refId": "A",
                "format": "table"
            }],
            "gridPos": grid_pos or {"x": 0, "y": 0, "w": 12, "h": 8},
            "options": {
                "showHeader": True,
                "sortBy": []
            },
            "fieldConfig": {
                "defaults": {
                    "custom": {
                        "align": "auto",
                        "displayMode": "auto"
                    }
                }
            }
        }

    def _create_heatmap_panel(self, title: str, query: str,
                             grid_pos: Dict[str, int] = None) -> Dict:
        """Create a heatmap panel."""
        panel_id = self._get_next_panel_id()

        return {
            "id": panel_id,
            "title": title,
            "type": "heatmap",
            "targets": [{
                "expr": query,
                "refId": "A"
            }],
            "gridPos": grid_pos or {"x": 0, "y": 0, "w": 24, "h": 8},
            "options": {
                "calculate": False,
                "calculation": {},
                "cellGap": 1,
                "cellValues": {},
                "color": {
                    "exponent": 0.5,
                    "fill": "dark-orange",
                    "mode": "spectrum",
                    "reverse": False,
                    "scale": "exponential",
                    "scheme": "Oranges",
                    "steps": 64
                },
                "exemplars": {"color": "rgba(255,0,255,0.7)"},
                "filterValues": {"le": 1e-9},
                "legend": {"show": False},
                "rowsFrame": {"layout": "auto"},
                "tooltip": {"show": True, "yHistogram": False},
                "yAxis": {"axisPlacement": "left", "reverse": False}
            }
        }

    def _create_gauge_panel(self, title: str, query: str,
                           grid_pos: Dict[str, int] = None) -> Dict:
        """Create a gauge panel."""
        panel_id = self._get_next_panel_id()

        return {
            "id": panel_id,
            "title": title,
            "type": "gauge",
            "targets": [{
                "expr": query,
                "refId": "A"
            }],
            "gridPos": grid_pos or {"x": 0, "y": 0, "w": 6, "h": 8},
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "orientation": "auto",
                "textMode": "auto",
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto"
            },
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "red", "value": 80}
                        ]
                    }
                }
            }
        }

    def _create_worldmap_panel(self, title: str, query: str,
                              grid_pos: Dict[str, int] = None) -> Dict:
        """Create a worldmap panel."""
        panel_id = self._get_next_panel_id()

        return {
            "id": panel_id,
            "title": title,
            "type": "grafana-worldmap-panel",
            "targets": [{
                "expr": query,
                "refId": "A"
            }],
            "gridPos": grid_pos or {"x": 0, "y": 0, "w": 12, "h": 8},
            "worldmapOptions": {
                "showTooltip": True,
                "showLegend": True,
                "defaultColor": "rgb(128, 128, 128)",
                "thresholds": "0,10,20,30",
                "colors": ["rgba(245, 54, 54, 0.9)", "rgba(237, 129, 40, 0.89)", "rgba(50, 172, 45, 0.97)"],
                "unitSingular": "",
                "unitPlural": "",
                "showZoomControl": True,
                "mouseWheelZoom": False
            }
        }

    def _create_pie_panel(self, title: str, query: str,
                         grid_pos: Dict[str, int] = None) -> Dict:
        """Create a pie chart panel."""
        panel_id = self._get_next_panel_id()

        return {
            "id": panel_id,
            "title": title,
            "type": "piechart",
            "targets": [{
                "expr": query,
                "refId": "A"
            }],
            "gridPos": grid_pos or {"x": 0, "y": 0, "w": 12, "h": 8},
            "options": {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "pieType": "pie",
                "tooltip": {"mode": "single", "sort": "none"},
                "legend": {"displayMode": "list", "placement": "bottom"},
                "displayLabels": ["name"]
            }
        }

    def _get_next_panel_id(self) -> int:
        """Get the next available panel ID."""
        panel_id = self.panel_id_counter
        self.panel_id_counter += 1
        return panel_id


# Main execution function for dashboard generation
def generate_phase3_dashboards():
    """
    🚀 **MAIN DASHBOARD GENERATION FUNCTION**

    Generate all Phase 3.0 Grafana dashboards and export them to JSON files.
    This function creates comprehensive monitoring dashboards for all enterprise
    components and saves them for Grafana import.
    """
    print("\n🎯 PHASE 3.0 GRAFANA DASHBOARD GENERATION")
    print("=" * 50)
    print("🏢 Generating enterprise monitoring dashboards")
    print("📊 Creating comprehensive visualization configs")
    print("🚀 Preparing production monitoring setup")
    print("=" * 50)

    # Initialize dashboard generator
    dashboard_generator = Phase3MonitoringDashboards()

    try:
        # Generate all dashboards
        dashboards = dashboard_generator.generate_all_dashboards()

        # Export to JSON files
        exported_files = dashboard_generator.export_dashboards()

        print("\n" + "=" * 60)
        print("🏆 PHASE 3.0 DASHBOARD GENERATION COMPLETE")
        print("=" * 60)

        print(f"✅ Generated {len(dashboards)} enterprise dashboards:")
        for dashboard_name in dashboards.keys():
            print(f"   📊 {dashboard_name.replace('_', ' ').title()}")

        print(f"\n📁 Dashboard files exported to:")
        for dashboard_name, filepath in exported_files.items():
            print(f"   📄 {filepath}")

        print("\n🚀 NEXT STEPS:")
        print("   📊 Import dashboard JSON files into Grafana")
        print("   🔧 Configure Prometheus data sources")
        print("   📈 Set up alerting rules for critical metrics")
        print("   🎯 Customize thresholds for your environment")

        print("\n" + "=" * 60)
        print("🎉 ENTERPRISE MONITORING READY FOR PRODUCTION!")

        return exported_files

    except Exception as e:
        print(f"\n❌ DASHBOARD GENERATION FAILED: {str(e)}")
        return {}


# Export main components
__all__ = [
    'Phase3MonitoringDashboards',
    'GrafanaPanel',
    'GrafanaDashboard',
    'generate_phase3_dashboards'
]


# Main execution
if __name__ == "__main__":
    print("🚀 Starting Phase 3.0 Grafana Dashboard Generation...")

    try:
        result = generate_phase3_dashboards()
        if result:
            print(f"✅ Successfully generated {len(result)} dashboard files")
            exit(0)
        else:
            print("❌ Dashboard generation failed")
            exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ Dashboard generation interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        exit(1)
