# Aider Hive Enhanced - Multi-Agent AI Architecture

<div align="center">

![Aider Hive Enhanced](https://img.shields.io/badge/Aider%20Hive-Enhanced-blue?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-2.0.0-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

**Advanced Multi-Agent AI System for Code Generation, Analysis, and Project Management**

</div>

## 🚀 Overview

Aider Hive Enhanced is a sophisticated multi-agent AI architecture that revolutionizes code generation, analysis, and project management. Built with modern async Python, it features advanced session management, hierarchical context understanding, fault-tolerant recovery mechanisms, and a beautiful real-time web interface.

### 🌟 Key Enhancements

- **🔄 Advanced Session Management**: Persistent sessions with event sourcing and automatic recovery
- **🧠 Hierarchical Context**: Semantic context organization with vector search capabilities
- **🌐 Modern Web Interface**: Real-time dashboard with WebSocket communication
- **⚡ High Performance**: Optimized for scalability with intelligent caching and load balancing
- **🔒 Enterprise Security**: Authentication, authorization, and audit logging
- **🛡️ Fault Tolerance**: Multiple recovery strategies with consistency guarantees
- **📊 Rich Monitoring**: Comprehensive metrics and performance analytics
- **🔌 Extensible**: Plugin architecture with extensive integration options

## 🏗️ Architecture Overview

```mermaid
graph TB
    UI[Web Interface] --> HC[Hive Coordinator]
    HC --> SM[Session Manager]
    HC --> AB[Agent Bus]
    
    SM --> ES[Event Store]
    SM --> SS[Snapshot Store]
    SM --> HCM[Hierarchical Context Manager]
    
    HCM --> VS[Vector Search Engine]
    HCM --> CS[Context Store]
    
    AB --> OA[Orchestrator Agent]
    AB --> CA[Code Agent]
    AB --> CTA[Context Agent]
    AB --> GA[Git Agent]
    AB --> QA[Quality Agent]
    
    ES --> DB[(Database)]
    SS --> DB
    CS --> VDB[(Vector DB)]
    
    HC --> RM[Recovery Manager]
    RM --> ES
    RM --> SS
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ (for web interface development)
- Redis (recommended) or SQLite
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/void-basic.git
   cd Void-basic
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements-hive.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Initialize the system**
   ```bash
   python -m aider.hive.setup --initialize
   ```

6. **Run the Hive system**
   ```bash
   python -m aider.hive.cli start --config ./aider/hive/config/hive_config_enhanced.json
   ```

### Docker Installation

```bash
# Pull and run with Docker Compose
docker-compose up -d

# Or build locally
docker build -t aider-hive-enhanced .
docker run -p 8080:8080 -p 8081:8081 aider-hive-enhanced
```

## ⚙️ Configuration

The system uses a comprehensive JSON configuration file. Copy and customize the enhanced configuration:

```bash
cp aider/hive/config/hive_config.example.json config/my_hive_config.json
```

### Key Configuration Sections

#### System Configuration
```json
{
  "system": {
    "version": "2.0.0",
    "environment": "production",
    "debug_mode": false,
    "session_database_url": "postgresql://user:password@localhost/hive_sessions",
    "enable_compression": true,
    "max_concurrent_tasks": 200
  }
}
```

#### Agent Configuration
```json
{
  "agents": {
    "code": {
      "instances": 3,
      "auto_scaling": {
        "enabled": true,
        "min_instances": 2,
        "max_instances": 8,
        "target_cpu": 80.0
      },
      "configuration": {
        "supported_languages": ["python", "javascript", "typescript"],
        "quality_thresholds": {
          "complexity_max": 15,
          "test_coverage_min": 80.0
        }
      }
    }
  }
}
```

#### Web Interface
```json
{
  "web_interface": {
    "enabled": true,
    "host": "0.0.0.0",
    "port": 8080,
    "websocket": {
      "enabled": true,
      "max_connections": 1000
    },
    "ui_features": {
      "enable_dashboard": true,
      "enable_code_diff": true,
      "enable_chat_interface": true
    }
  }
}
```

## 🚀 Usage

### Command Line Interface

```bash
# Start the hive system
python -m aider.hive.cli start

# Create a new session
python -m aider.hive.cli session create --user-id "developer1" --project-path "/path/to/project"

# Send a message to a session
python -m aider.hive.cli session message <session-id> "Please implement a REST API for user management"

# Monitor system health
python -m aider.hive.cli status

# Scale agents
python -m aider.hive.cli scale --agent-type code --instances 5
```

### Web Interface

Open your browser to `http://localhost:8080` to access the comprehensive web dashboard:

#### Dashboard Features
- **Real-time System Monitoring**: Live metrics, agent status, and performance graphs
- **Session Management**: Create, view, and manage user sessions
- **Interactive Chat**: Direct communication with the AI agents
- **Code Diff Viewer**: Visual comparison of code changes
- **Context Explorer**: Navigate and search through hierarchical context
- **Recovery Console**: Monitor and manage session recovery operations

### Python API

```python
import asyncio
from aider.hive import create_hive_coordinator

async def main():
    # Create and start hive coordinator
    async with create_hive_coordinator(config_path="config.json") as hive:
        # Process a request
        result = await hive.process_request(
            request="Create a FastAPI endpoint for user authentication",
            context={"project_type": "web_api", "framework": "fastapi"},
            user_id="developer1"
        )
        
        print(f"Result: {result}")

asyncio.run(main())
```

### Session Management API

```python
from aider.hive.session import SessionManager, SessionMetadata

async def session_example():
    session_manager = SessionManager(database_url="sqlite:///sessions.db")
    await session_manager.initialize()
    
    # Create session
    session_id = await session_manager.create_session(
        user_id="developer1",
        project_path="/path/to/project",
        session_type="coding",
        initial_context={"language": "python"}
    )
    
    # Add message
    await session_manager.add_conversation_message(
        session_id=session_id,
        message={
            "content": "Implement user authentication",
            "type": "user_request"
        }
    )
    
    # Get session history
    history = await session_manager.get_session_history(session_id)
    return history
```

## 📚 API Documentation

### REST API Endpoints

#### System Management
- `GET /api/system/status` - Get system health and status
- `GET /api/metrics` - Retrieve performance metrics
- `POST /api/system/restart` - Restart the hive system

#### Session Management
- `POST /api/sessions` - Create a new session
- `GET /api/sessions/{session_id}` - Get session details
- `GET /api/sessions` - List sessions (with filtering)
- `PUT /api/sessions/{session_id}/suspend` - Suspend a session
- `PUT /api/sessions/{session_id}/resume` - Resume a session
- `DELETE /api/sessions/{session_id}` - Delete a session

#### Messaging
- `POST /api/sessions/{session_id}/messages` - Send a message
- `GET /api/sessions/{session_id}/messages` - Get message history
- `GET /api/sessions/{session_id}/stream` - Server-sent events stream

#### Agent Management
- `GET /api/agents` - List all agents and their status
- `GET /api/agents/{agent_id}` - Get agent details
- `POST /api/agents/{agent_id}/scale` - Scale agent instances

### WebSocket API

Connect to `/ws/{session_id}` for real-time communication:

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/session-123');

ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'chat_message',
        content: 'Hello, please help me with my code',
        context: { language: 'python' }
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

## 🎯 Core Features

### Advanced Session Management

- **Persistent Sessions**: Sessions survive system restarts with full state recovery
- **Event Sourcing**: Complete audit trail with event replay capabilities
- **Smart Snapshots**: Periodic state snapshots for fast recovery
- **Context Preservation**: Maintains conversation and project context across sessions

### Hierarchical Context System

- **Semantic Chunking**: Intelligent code and text segmentation
- **Vector Search**: Fast similarity search using embeddings
- **Context Tiers**: Core, Relevant, Ambient, and Archived context levels
- **Dynamic Relevance**: Context importance adjusts based on usage patterns

### Fault Tolerance & Recovery

- **Multiple Recovery Strategies**: Snapshot-based, event replay, hybrid, and best-effort
- **Consistency Checks**: Validates data integrity during recovery
- **Graceful Degradation**: System continues operating with reduced functionality
- **Automatic Failover**: Seamless switching between agent instances

### Real-time Web Interface

- **Live Dashboard**: System metrics, agent status, and performance graphs
- **Interactive Chat**: Direct communication with AI agents
- **Code Visualization**: Syntax highlighting, diff viewing, and editing
- **Session Explorer**: Browse and search through conversation history

### Security & Compliance

- **Authentication & Authorization**: JWT-based auth with role-based access control
- **Rate Limiting**: Configurable request throttling
- **Audit Logging**: Comprehensive logging of all system activities
- **Data Encryption**: Optional at-rest and in-transit encryption

## 🔌 Integrations

### LLM Providers
- **OpenAI**: GPT-4, GPT-3.5, with function calling support
- **Anthropic**: Claude-3 models with enhanced reasoning
- **Azure OpenAI**: Enterprise-grade OpenAI integration
- **Local Models**: Support for self-hosted models

### Vector Databases
- **FAISS**: High-performance similarity search
- **Pinecone**: Managed vector database service
- **Weaviate**: Open-source vector database
- **Chroma**: Lightweight embedded vector store

### Development Tools
- **Git Integration**: Automatic commit management and branch handling
- **Code Quality**: Linting, testing, and security scanning
- **CI/CD**: Integration with popular deployment pipelines
- **Monitoring**: Prometheus, Grafana, and DataDog support

## 📊 Performance & Monitoring

### Metrics Collection
- **System Metrics**: CPU, memory, disk, and network usage
- **Application Metrics**: Request rates, response times, error rates
- **Agent Metrics**: Task completion rates, processing times
- **Session Metrics**: Creation rates, duration, recovery success

### Health Monitoring
- **Health Checks**: Automated system health verification
- **Alerting**: Configurable alerts for system issues
- **Performance Targets**: SLA monitoring and reporting
- **Capacity Planning**: Resource usage trends and projections

## 🛠️ Development

### Setting up Development Environment

```bash
# Clone repository
git clone https://github.com/your-username/void-basic.git
cd Void-basic

# Install development dependencies
pip install -r requirements-hive.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=aider

# Start development server
python -m aider.hive.cli start --debug
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=aider --cov-report=html
```

### Code Quality

```bash
# Format code
black aider/
isort aider/

# Lint code
flake8 aider/
pylint aider/

# Type checking
mypy aider/
```

## 🐳 Docker Deployment

### Docker Compose (Recommended)

```yaml
version: '3.8'
services:
  hive-coordinator:
    build: .
    ports:
      - "8080:8080"
      - "8081:8081"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/hive
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    
  db:
    image: postgres:14
    environment:
      POSTGRES_DB: hive
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Scale deployment
kubectl scale deployment hive-coordinator --replicas=3

# Check status
kubectl get pods -l app=hive-coordinator
```

## 🔍 Troubleshooting

### Common Issues

#### Session Recovery Fails
```bash
# Check session integrity
python -m aider.hive.cli session validate <session-id>

# Force recovery with specific strategy
python -m aider.hive.cli session recover <session-id> --strategy best_effort
```

#### Agent Not Responding
```bash
# Check agent health
python -m aider.hive.cli agent health <agent-id>

# Restart agent
python -m aider.hive.cli agent restart <agent-id>
```

#### Performance Issues
```bash
# Check system metrics
python -m aider.hive.cli metrics

# Enable profiling
python -m aider.hive.cli start --profile

# Analyze memory usage
python -m aider.hive.cli debug memory
```

### Log Analysis

```bash
# View recent logs
tail -f logs/hive_enhanced.log

# Search for errors
grep -i error logs/hive_enhanced.log

# Filter by component
grep "session_manager" logs/hive_enhanced.log
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Include docstrings for public APIs
- Maintain test coverage above 80%
- Update documentation for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE.txt) file for details.

## 🙏 Acknowledgments

- Built on the foundation of the original Aider project
- Inspired by modern multi-agent AI architectures
- Uses state-of-the-art AI models and vector databases
- Community feedback and contributions

## 📞 Support

- **Documentation**: [https://docs.aider-hive.com](https://docs.aider-hive.com)
- **Issues**: [GitHub Issues](https://github.com/your-username/void-basic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/void-basic/discussions)
- **Discord**: [Community Discord](https://discord.gg/aider-hive)

## 🗺️ Roadmap

### Version 2.1 (Q2 2024)
- [ ] Multi-modal support (images, voice)
- [ ] Advanced reasoning capabilities
- [ ] Federated learning for agent improvement
- [ ] Mobile companion app

### Version 2.2 (Q3 2024)
- [ ] Voice interface integration
- [ ] Advanced code refactoring agents
- [ ] Project template marketplace
- [ ] Enterprise SSO integration

### Version 3.0 (Q4 2024)
- [ ] Distributed agent orchestration
- [ ] AI-powered system optimization
- [ ] Advanced security features
- [ ] Industry-specific agent specializations

---

<div align="center">

**Built with ❤️ by the Aider Hive Team**

[⭐ Star us on GitHub](https://github.com/your-username/void-basic) | [📖 Documentation](https://docs.aider-hive.com) | [💬 Join Discord](https://discord.gg/aider-hive)

</div>