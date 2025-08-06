'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  BarElement
} from 'chart.js';
import { Line, Doughnut, Bar } from 'react-chartjs-2';
import {
  PlayIcon,
  PauseIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  CpuChipIcon,
  CloudIcon,
  ShieldCheckIcon,
  UserGroupIcon,
  ClockIcon,
  BoltIcon,
  ChartBarIcon,
  Cog6ToothIcon
} from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  BarElement
);

interface WorkflowExecution {
  id: string;
  type: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  priority: 'low' | 'medium' | 'high';
  started_at: string;
  completed_at?: string;
  tenant_id: string;
  user_id: string;
  compliance_requirements: string[];
  result?: any;
  error?: string;
}

interface AgentStatus {
  name: string;
  status: 'active' | 'idle' | 'busy' | 'error';
  utilization: number;
  tasks_completed: number;
  last_activity: string;
  version: string;
}

interface SystemMetrics {
  active_workflows: number;
  completed_today: number;
  success_rate: number;
  average_completion_time: string;
  agent_utilization: Record<string, number>;
  compliance_status: Record<string, boolean>;
  system_health: string;
  uptime: string;
  resource_usage: {
    cpu: string;
    memory: string;
    disk: string;
  };
}

interface DashboardProps {
  tenantId: string;
  userRole: string;
}

const Dashboard: React.FC<DashboardProps> = ({ tenantId, userRole }) => {
  // State management
  const [workflows, setWorkflows] = useState<WorkflowExecution[]>([]);
  const [agents, setAgents] = useState<AgentStatus[]>([
    {
      name: 'CodeAgent',
      status: 'active',
      utilization: 85,
      tasks_completed: 127,
      last_activity: '2 minutes ago',
      version: '3.0.0'
    },
    {
      name: 'ContextAgent',
      status: 'busy',
      utilization: 78,
      tasks_completed: 95,
      last_activity: '30 seconds ago',
      version: '3.0.0'
    },
    {
      name: 'GitAgent',
      status: 'active',
      utilization: 92,
      tasks_completed: 156,
      last_activity: '1 minute ago',
      version: '3.0.0'
    },
    {
      name: 'QualityAgent',
      status: 'idle',
      utilization: 88,
      tasks_completed: 89,
      last_activity: '5 minutes ago',
      version: '3.0.0'
    },
    {
      name: 'DeploymentAgent',
      status: 'active',
      utilization: 76,
      tasks_completed: 67,
      last_activity: '3 minutes ago',
      version: '3.0.0'
    },
    {
      name: 'WorkflowOrchestrator',
      status: 'busy',
      utilization: 94,
      tasks_completed: 203,
      last_activity: '10 seconds ago',
      version: '3.0.0'
    }
  ]);
  const [metrics, setMetrics] = useState<SystemMetrics>({
    active_workflows: 12,
    completed_today: 87,
    success_rate: 96.6,
    average_completion_time: '2.3 minutes',
    agent_utilization: {
      code_agent: 85,
      context_agent: 78,
      git_agent: 92,
      quality_agent: 88,
      deployment_agent: 76,
      workflow_orchestrator: 94
    },
    compliance_status: {
      sox_compliant: true,
      gdpr_compliant: true,
      hipaa_compliant: true
    },
    system_health: 'healthy',
    uptime: '99.9%',
    resource_usage: {
      cpu: '45%',
      memory: '62%',
      disk: '23%'
    }
  });

  const [isConnected, setIsConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [selectedTimeframe, setSelectedTimeframe] = useState('24h');

  // WebSocket connection for real-time updates
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/${tenantId}`);

    ws.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setLastUpdate(new Date());

      switch (data.type) {
        case 'workflow_started':
          setWorkflows(prev => [data.data, ...prev]);
          break;
        case 'workflow_completed':
        case 'workflow_failed':
          setWorkflows(prev =>
            prev.map(w => w.id === data.workflow_id ? { ...w, ...data.data } : w)
          );
          break;
        case 'metrics_update':
          setMetrics(prev => ({ ...prev, ...data.data }));
          break;
        case 'agent_status_update':
          setAgents(prev =>
            prev.map(agent =>
              agent.name === data.agent_name
                ? { ...agent, ...data.data }
                : agent
            )
          );
          break;
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [tenantId]);

  // Chart data preparation
  const workflowTrendData = {
    labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
    datasets: [
      {
        label: 'Completed Workflows',
        data: [12, 8, 15, 23, 18, 25, 19],
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.4,
      },
      {
        label: 'Failed Workflows',
        data: [1, 0, 2, 1, 0, 1, 0],
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        tension: 0.4,
      }
    ]
  };

  const agentUtilizationData = {
    labels: Object.keys(metrics.agent_utilization),
    datasets: [
      {
        data: Object.values(metrics.agent_utilization),
        backgroundColor: [
          '#3B82F6',
          '#10B981',
          '#8B5CF6',
          '#F59E0B',
          '#EF4444',
          '#6366F1'
        ],
        borderWidth: 2,
        borderColor: '#ffffff'
      }
    ]
  };

  const complianceData = {
    labels: ['SOX', 'GDPR', 'HIPAA'],
    datasets: [
      {
        label: 'Compliance Status',
        data: [
          metrics.compliance_status.sox_compliant ? 100 : 0,
          metrics.compliance_status.gdpr_compliant ? 100 : 0,
          metrics.compliance_status.hipaa_compliant ? 100 : 0
        ],
        backgroundColor: [
          metrics.compliance_status.sox_compliant ? '#10B981' : '#EF4444',
          metrics.compliance_status.gdpr_compliant ? '#10B981' : '#EF4444',
          metrics.compliance_status.hipaa_compliant ? '#10B981' : '#EF4444'
        ]
      }
    ]
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
      case 'completed':
      case 'healthy':
        return 'text-green-600 bg-green-100';
      case 'busy':
      case 'running':
        return 'text-blue-600 bg-blue-100';
      case 'idle':
      case 'pending':
        return 'text-yellow-600 bg-yellow-100';
      case 'error':
      case 'failed':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high':
        return 'text-red-600 bg-red-100';
      case 'medium':
        return 'text-yellow-600 bg-yellow-100';
      case 'low':
        return 'text-green-600 bg-green-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Autonomous Development Dashboard
            </h1>
            <p className="text-gray-600 mt-1">
              Real-time monitoring and management for enterprise workflows
            </p>
          </div>

          <div className="flex items-center space-x-4">
            {/* Connection Status */}
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
              isConnected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-500' : 'bg-red-500'
              }`} />
              <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>

            {/* Last Update */}
            <div className="text-sm text-gray-500">
              Last update: {lastUpdate.toLocaleTimeString()}
            </div>

            {/* Timeframe Selector */}
            <select
              value={selectedTimeframe}
              onChange={(e) => setSelectedTimeframe(e.target.value)}
              className="px-3 py-1 border border-gray-300 rounded-md text-sm"
            >
              <option value="1h">Last Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
          </div>
        </div>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <motion.div
          className="bg-white rounded-lg shadow-sm p-6 border border-gray-200"
          whileHover={{ scale: 1.02 }}
          transition={{ duration: 0.2 }}
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Active Workflows</p>
              <p className="text-3xl font-bold text-gray-900">{metrics.active_workflows}</p>
            </div>
            <div className="p-3 bg-blue-100 rounded-full">
              <PlayIcon className="w-6 h-6 text-blue-600" />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-green-600">↗ 12% from yesterday</span>
          </div>
        </motion.div>

        <motion.div
          className="bg-white rounded-lg shadow-sm p-6 border border-gray-200"
          whileHover={{ scale: 1.02 }}
          transition={{ duration: 0.2 }}
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Success Rate</p>
              <p className="text-3xl font-bold text-gray-900">{metrics.success_rate}%</p>
            </div>
            <div className="p-3 bg-green-100 rounded-full">
              <CheckCircleIcon className="w-6 h-6 text-green-600" />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-green-600">↗ 0.2% from yesterday</span>
          </div>
        </motion.div>

        <motion.div
          className="bg-white rounded-lg shadow-sm p-6 border border-gray-200"
          whileHover={{ scale: 1.02 }}
          transition={{ duration: 0.2 }}
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Avg Completion</p>
              <p className="text-3xl font-bold text-gray-900">{metrics.average_completion_time}</p>
            </div>
            <div className="p-3 bg-yellow-100 rounded-full">
              <ClockIcon className="w-6 h-6 text-yellow-600" />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-green-600">↘ 15s improvement</span>
          </div>
        </motion.div>

        <motion.div
          className="bg-white rounded-lg shadow-sm p-6 border border-gray-200"
          whileHover={{ scale: 1.02 }}
          transition={{ duration: 0.2 }}
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">System Health</p>
              <p className="text-3xl font-bold text-green-600 capitalize">{metrics.system_health}</p>
            </div>
            <div className="p-3 bg-green-100 rounded-full">
              <ShieldCheckIcon className="w-6 h-6 text-green-600" />
            </div>
          </div>
          <div className="mt-4">
            <span className="text-sm text-gray-600">Uptime: {metrics.uptime}</span>
          </div>
        </motion.div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column - Workflows and Charts */}
        <div className="lg:col-span-2 space-y-6">
          {/* Workflow Trend Chart */}
          <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-900">Workflow Trends</h3>
              <ChartBarIcon className="w-5 h-5 text-gray-400" />
            </div>
            <div className="h-64">
              <Line
                data={workflowTrendData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: { position: 'top' },
                    title: { display: false }
                  },
                  scales: {
                    y: { beginAtZero: true }
                  }
                }}
              />
            </div>
          </div>

          {/* Recent Workflows */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200">
            <div className="p-6 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">Recent Workflows</h3>
            </div>
            <div className="divide-y divide-gray-200">
              <AnimatePresence>
                {workflows.slice(0, 6).map((workflow) => (
                  <motion.div
                    key={workflow.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="p-6 hover:bg-gray-50 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(workflow.status)}`}>
                            {workflow.status}
                          </span>
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getPriorityColor(workflow.priority)}`}>
                            {workflow.priority}
                          </span>
                          <span className="text-sm text-gray-500">{workflow.type}</span>
                        </div>
                        <p className="text-sm font-medium text-gray-900 mt-1">
                          {workflow.description}
                        </p>
                        <p className="text-xs text-gray-500 mt-1">
                          Started: {new Date(workflow.started_at).toLocaleString()}
                        </p>
                        {workflow.compliance_requirements.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-2">
                            {workflow.compliance_requirements.map((req) => (
                              <span key={req} className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded">
                                {req}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                      <div className="flex items-center space-x-2">
                        {workflow.status === 'running' && (
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                        )}
                        {workflow.status === 'completed' && (
                          <CheckCircleIcon className="w-4 h-4 text-green-600" />
                        )}
                        {workflow.status === 'failed' && (
                          <ExclamationTriangleIcon className="w-4 h-4 text-red-600" />
                        )}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>
        </div>

        {/* Right Column - Agents and System Info */}
        <div className="space-y-6">
          {/* Agent Status */}
          <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-900">Agent Status</h3>
              <CpuChipIcon className="w-5 h-5 text-gray-400" />
            </div>
            <div className="space-y-4">
              {agents.map((agent) => (
                <div key={agent.name} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-900">{agent.name}</span>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(agent.status)}`}>
                        {agent.status}
                      </span>
                    </div>
                    <div className="mt-2">
                      <div className="flex justify-between text-xs text-gray-500 mb-1">
                        <span>Utilization</span>
                        <span>{agent.utilization}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${agent.utilization}%` }}
                        />
                      </div>
                    </div>
                    <div className="mt-2 text-xs text-gray-500">
                      <span>Tasks: {agent.tasks_completed} • </span>
                      <span>Last: {agent.last_activity}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Agent Utilization Chart */}
          <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-6">Agent Utilization</h3>
            <div className="h-48">
              <Doughnut
                data={agentUtilizationData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: { position: 'bottom' }
                  }
                }}
              />
            </div>
          </div>

          {/* Compliance Status */}
          <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-900">Compliance Status</h3>
              <ShieldCheckIcon className="w-5 h-5 text-gray-400" />
            </div>
            <div className="space-y-4">
              {Object.entries(metrics.compliance_status).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-900 uppercase">{key}</span>
                  <div className="flex items-center space-x-2">
                    {value ? (
                      <CheckCircleIcon className="w-5 h-5 text-green-600" />
                    ) : (
                      <ExclamationTriangleIcon className="w-5 h-5 text-red-600" />
                    )}
                    <span className={`text-sm ${value ? 'text-green-600' : 'text-red-600'}`}>
                      {value ? 'Compliant' : 'Non-compliant'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-6 h-32">
              <Bar
                data={complianceData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: { display: false }
                  },
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 100,
                      ticks: { callback: (value) => `${value}%` }
                    }
                  }
                }}
              />
            </div>
          </div>

          {/* Resource Usage */}
          <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-semibold text-gray-900">Resource Usage</h3>
              <CloudIcon className="w-5 h-5 text-gray-400" />
            </div>
            <div className="space-y-4">
              {Object.entries(metrics.resource_usage).map(([resource, usage]) => (
                <div key={resource} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium text-gray-900 capitalize">{resource}</span>
                    <span className="text-gray-600">{usage}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-300 ${
                        parseInt(usage) > 80 ? 'bg-red-500' :
                        parseInt(usage) > 60 ? 'bg-yellow-500' : 'bg-green-500'
                      }`}
                      style={{ width: usage }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
