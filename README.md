# RAGagument - Enterprise RAG System

A powerful, enterprise-grade Retrieval-Augmented Generation system with full containerization support, allowing you to chat with your documents using local or cloud-based AI models.

## 🚀 Features

### Core Features
- **Document Processing**: Support for PDF, DOCX, TXT, and MD files
- **Flexible Embeddings**: Switch between HuggingFace and Ollama embedding models
- **Vector Search**: FAISS-based semantic search with configurable parameters
- **LLM Integration**: DeepSeek API for intelligent response generation
- **Chat History**: Persistent conversation storage with SQLite
- **Streamlit UI**: Beautiful web interface with real-time updates
- **Evaluation Metrics**: Comprehensive performance tracking and analytics

### 🐳 Containerization Features
- **Multi-stage Docker builds** with security hardening
- **Development & Production** environments
- **Kubernetes deployment** with auto-scaling
- **CI/CD pipeline** with security scanning
- **Health monitoring** and performance metrics
- **Enterprise security** with vulnerability scanning

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:
### Core Components
- **Frontend**: Streamlit web interface with real-time chat and document upload
- **Application**: Python orchestration layer managing the RAG pipeline
- **Processing**: Intelligent document chunking and multi-provider embedding generation
- **Retrieval**: FAISS vector search with configurable parameters and RAG fusion
- **Generation**: DeepSeek LLM integration with streaming responses
- **Data**: SQLite chat history + local file storage for documents and vectors

### Key Features
- **Flexible Embeddings**: Switch between HuggingFace and Ollama providers via config
- **Advanced Retrieval**: RAG fusion, query expansion, and hybrid search capabilities
- **Real-time UI**: Streamlit-based interface with live updates and progress indicators
- **Session Management**: Persistent chat history with automatic cleanup
- **Evaluation System**: Comprehensive performance metrics and quality assessment

## 📦 Installation

### 🚀 Quick Start with Docker (Recommended)

1. **Clone and setup**
   ```bash
   git clone https://github.com/Thugney/rag.git
   cd rag
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your DeepSeek API key
   ```

3. **Start development environment**
   ```bash
   make dev
   ```

4. **Access the application**
   - Open http://localhost:8501 in your browser

### 🐍 Traditional Installation (Alternative)

1. **Clone the repository**
   ```bash
   git clone https://github.com/Thugney/rag.git
   cd rag
   ```

2. **Create virtual environment**
   ```bash
   python -m venv rag_venv
   source rag_venv/bin/activate  # On Windows: rag_venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

### 🏭 Production Deployment

For production deployment with Kubernetes:

```bash
# Build production image
make build-prod

# Deploy to Kubernetes
make k8s-deploy

# Check deployment status
kubectl get pods -n ragagument-production
```

## ⚙️ Configuration

Edit `config.yaml` to customize:
- Embedding provider (HuggingFace or Ollama)
- Chunk size and overlap
- Vector store settings
- LLM parameters
- Advanced features like RAG fusion

## 🎯 Usage

1. **Upload Documents**: Use the upload section to add PDF, DOCX, TXT, or MD files
2. **Process Documents**: Click "Process & Index" to make documents searchable
3. **Ask Questions**: Use the chat interface to ask questions about your documents
4. **Manage Sessions**: Switch between chat sessions or start new ones

## 🔧 Advanced Features

### Core Features
- **RAG Fusion**: Generate multiple query variants for comprehensive results
- **Tool Integration**: Built-in tools for enhanced functionality
- **Performance Metrics**: Real-time evaluation and analytics
- **Session Management**: Persistent chat history with cleanup options

### 🐳 Containerization Features
- **Multi-stage Builds**: Optimized for development and production
- **Security Hardening**: Non-root containers with vulnerability scanning
- **Auto-scaling**: Kubernetes HPA with custom metrics
- **Health Monitoring**: Comprehensive health checks and alerting
- **CI/CD Integration**: Automated testing and deployment pipelines

## ⚡ Performance & Docker Behavior

### Expected Build Times
- **First Build**: 5-15 minutes (downloading dependencies, compiling packages)
- **Subsequent Builds**: 30-60 seconds (using Docker layer caching)
- **Dependencies**: FAISS, sentence-transformers, and ML libraries take time to install

### Docker Output Format
The verbose build output you see is **normal and expected**:
```
#5 [base 1/3] FROM docker.io/library/python:3.9-slim-bullseye
#5 sha256:abc123... 1.75kB / 1.75kB done
#5 extracting sha256:abc123... 1.5s done
```
This detailed output helps with debugging and shows build progress. It's standard for Docker multi-stage builds.

### Optimization Tips
```bash
# Use build cache for faster rebuilds
make build-dev  # Subsequent builds will be faster

# Clean build (if needed)
make clean
make dev

# Monitor build progress
docker build --progress=plain -t ragagument:dev .
```

## 📊 Evaluation

The system includes comprehensive evaluation metrics:
- Answer relevance
- Faithfulness to source material
- Context relevance
- Response latency
- Cost tracking

## 🛠️ Development & Operations

### 📁 Complete Project Structure
```
RAGagument/
├── 📄 app.py                    # Main Streamlit application
├── 📄 config.yaml              # Legacy configuration file
├── 📄 requirements.txt         # Python dependencies
├── 📄 requirements-test.txt    # Testing dependencies
├── 🐳 Dockerfile               # Multi-stage container build
├── 🐳 docker-compose.dev.yml   # Development environment
├── 🐳 docker-compose.production.yml  # Production environment
├── 📋 Makefile                 # Build automation & commands
├── 🔒 .env                     # Environment variables (git ignored)
├── 🚫 .gitignore              # Git exclusion rules
├── 🚫 .dockerignore           # Docker build exclusions
├── 🔍 .trivyignore            # Security scan exclusions
├── 🔍 .gitleaks.toml          # Secret scanning rules
├── 📁 config/                  # Environment configurations
│   ├── base.yaml              # Base configuration
│   ├── development.yaml       # Development overrides
│   ├── production.yaml        # Production overrides
│   └── staging.yaml           # Staging overrides
├── 📁 k8s/                    # Kubernetes manifests
│   ├── namespace.yaml         # Namespace definitions
│   ├── configmap.yaml         # Configuration maps
│   ├── secret.yaml            # Secrets (git ignored)
│   ├── deployment.yaml        # Application deployment
│   ├── service.yaml           # Service definitions
│   ├── hpa.yaml              # Auto-scaling rules
│   └── ingress.yaml           # Ingress configuration
├── 📁 .github/                # CI/CD workflows
│   └── workflows/
│       ├── ci-cd.yml         # Main CI/CD pipeline
│       └── security-scan.yml # Security scanning
├── 📁 docs/                   # Documentation
│   ├── containerization-plan.md
│   ├── docker-compose-design.md
│   ├── health-monitoring-config.md
│   ├── environment-config-management.md
│   ├── kubernetes-deployment-manifests.md
│   └── cicd-pipeline-integration.md
├── 📁 scripts/                # Utility scripts
│   ├── validate-setup.sh     # Setup validation
│   └── security-audit.sh     # Security auditing
├── 📁 uploaded_docs/         # User documents (git ignored)
├── 📁 vector_db/             # Vector database (git ignored)
└── 📁 rag_venv/              # Virtual environment (git ignored)
```

### 🔑 Key Files & Components

#### Core Application
- `app.py` - Main Streamlit application with RAG pipeline
- `config_loader.py` - Environment-aware configuration management
- `embedding_factory.py` - Multi-provider embedding system (HuggingFace/Ollama)
- `vector_store.py` - FAISS vector database integration
- `retriever.py` - Advanced retrieval with RAG fusion
- `generator.py` - DeepSeek LLM integration with streaming
- `chat_history_db.py` - SQLite chat history management
- `chunker.py` - Intelligent document processing and chunking
- `evaluator.py` - Performance evaluation and metrics

#### 🐳 Containerization
- `Dockerfile` - Multi-stage build with security hardening
- `docker-compose.*.yml` - Environment-specific container orchestration
- `Makefile` - 20+ commands for development and deployment
- `k8s/` - Complete Kubernetes deployment manifests

#### 🔒 Security & Compliance
- `.gitignore` & `.dockerignore` - Comprehensive exclusion rules
- `.trivyignore` & `.gitleaks.toml` - Security scanning configuration
- `scripts/security-audit.sh` - Automated security validation

#### 📊 CI/CD & Monitoring
- `.github/workflows/` - GitHub Actions for automated testing and deployment
- Health checks and monitoring integration
- Performance metrics and alerting

### 🚀 Quick Commands

```bash
# Development
make dev              # Start development environment
make build-dev        # Build development image
make test            # Run tests

# Production
make build-prod      # Build production image
make k8s-deploy      # Deploy to Kubernetes
make security-scan   # Run security scanning

# Utilities
make help            # Show all available commands
./scripts/security-audit.sh  # Security audit
```

## 🐛 Troubleshooting

### Common Docker Issues

#### Build Takes Too Long
```bash
# Check Docker resources
docker system info

# Clear cache if needed
docker system prune -a

# Use build cache
make build-dev  # Subsequent builds are faster
```

#### Permission Issues
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### Port Conflicts
```bash
# Check what's using port 8501
lsof -i :8501

# Change port in .env
STREAMLIT_SERVER_PORT=8502
```

#### Memory Issues
```bash
# Check available memory
free -h

# Increase Docker memory limit in Docker Desktop settings
```

### Application Issues

#### API Key Problems
```bash
# Check .env file
cat .env | grep DEEPSEEK_API_KEY

# Test API connectivity
curl -H "Authorization: Bearer YOUR_KEY" https://api.deepseek.com/v1/models
```

#### Database Issues
```bash
# Reset database
rm -f chat_history.db
make dev  # Will recreate database
```

### Getting Help

```bash
# Run diagnostics
./scripts/validate-setup.sh

# Security audit
./scripts/security-audit.sh

# View logs
make logs

# Debug container
make shell
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions, please check:
- [Issue Tracker](https://github.com/Thugney/rag/issues)
- [Discussion Forum](https://github.com/Thugney/rag/discussions)

## 🏢 Enterprise Features

### Security & Compliance
- ✅ **Vulnerability Scanning**: Trivy integration for container security
- ✅ **Secret Detection**: Gitleaks for preventing credential leaks
- ✅ **Non-root Containers**: Security hardening with proper user isolation
- ✅ **RBAC**: Kubernetes role-based access control
- ✅ **Audit Logging**: Comprehensive security event logging

### Scalability & Performance
- ✅ **Auto-scaling**: Kubernetes HPA with custom metrics
- ✅ **Load Balancing**: Ingress with SSL termination
- ✅ **Health Monitoring**: Comprehensive health checks and alerting
- ✅ **Resource Management**: CPU/memory limits and requests
- ✅ **Performance Metrics**: Real-time monitoring and analytics

### DevOps & CI/CD
- ✅ **Automated Testing**: Unit, integration, and security tests
- ✅ **Multi-environment**: Development, staging, production pipelines
- ✅ **GitOps Ready**: ArgoCD integration for deployment automation
- ✅ **Monitoring**: Prometheus/Grafana dashboards
- ✅ **Backup & Recovery**: Automated data persistence and recovery

## 📚 Documentation

- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Complete testing instructions
- **[docs/containerization-plan.md](docs/containerization-plan.md)** - Architecture overview
- **[docs/kubernetes-deployment-manifests.md](docs/kubernetes-deployment-manifests.md)** - K8s deployment guide
- **[scripts/security-audit.sh](scripts/security-audit.sh)** - Security validation script

## 🔗 Links

- [DeepSeek API](https://platform.deepseek.com/)
- [HuggingFace Models](https://huggingface.co/models)
- [Ollama](https://ollama.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
