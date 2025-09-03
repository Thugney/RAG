# 🐳 RAGagument Technology Stack & Containerization Guide

## Overview
This document explains all the **new technologies** introduced during the containerization process, excluding Python, Streamlit, and the original project components. It covers what each technology does, how they work together, and what happens when you run the commands.

## 📋 Table of Contents
- [Make (Build Automation)](#make-build-automation)
- [Docker (Containerization)](#docker-containerization)
- [Docker Compose (Multi-Container Orchestration)](#docker-compose-multi-container-orchestration)
- [Kubernetes (Container Orchestration)](#kubernetes-container-orchestration)
- [Helm (Kubernetes Package Manager)](#helm-kubernetes-package-manager)
- [GitHub Actions (CI/CD)](#github-actions-cicd)
- [Trivy (Security Scanning)](#trivy-security-scanning)
- [Gitleaks (Secret Detection)](#gitleaks-secret-detection)
- [FAISS (Vector Database)](#faiss-vector-database)
- [Ollama (Local AI Models)](#ollama-local-ai-models)
- [YAML (Configuration Format)](#yaml-configuration-format)

---

## Make (Build Automation)

### What is Make?
**Make** is a build automation tool that reads `Makefile` and executes commands based on rules and dependencies.

### What it does in our setup:
- **Automates repetitive tasks** - Single commands instead of long Docker commands
- **Manages dependencies** - Ensures proper build order
- **Provides shortcuts** - `make dev` instead of `docker compose -f docker-compose.dev.yml up --build`

### What happens when you run `make dev`:
1. **Reads Makefile** - Finds the `dev` target
2. **Executes command** - Runs `docker compose -f docker-compose.dev.yml up --build`
3. **Shows progress** - Displays build output and container logs
4. **Handles errors** - Stops if any step fails

### Key Make commands in our setup:
```bash
make dev           # Start development environment
make build-prod    # Build production image
make k8s-deploy    # Deploy to Kubernetes
make security-scan # Run security checks
make clean         # Clean up containers
```

### 📚 Resources:
- **Official Docs**: https://www.gnu.org/software/make/
- **Tutorial**: https://makefiletutorial.com/
- **GNU Make Manual**: https://www.gnu.org/software/make/manual/

---

## Docker (Containerization)

### What is Docker?
**Docker** is a platform for developing, shipping, and running applications in **containers** - lightweight, portable environments.

### What it does in our setup:
- **Isolates applications** - Each container has its own environment
- **Ensures consistency** - Same behavior across development, staging, production
- **Simplifies deployment** - No "works on my machine" issues
- **Multi-stage builds** - Optimized images for different environments

### What happens when you run `make dev`:
1. **Builds image** - Creates container from Dockerfile
2. **Starts container** - Runs isolated environment
3. **Mounts volumes** - Shares code and data with host
4. **Exposes ports** - Makes app accessible at localhost:8501
5. **Runs health checks** - Monitors container health

### Key Docker concepts in our setup:
- **Images**: Blueprints for containers (like class definitions)
- **Containers**: Running instances of images (like objects)
- **Volumes**: Persistent data storage
- **Networks**: Container communication
- **Multi-stage builds**: Different images for dev/prod

### 📚 Resources:
- **Official Docs**: https://docs.docker.com/
- **Get Started**: https://docs.docker.com/get-started/
- **Best Practices**: https://docs.docker.com/develop/dev-best-practices/

---

## Docker Compose (Multi-Container Orchestration)

### What is Docker Compose?
**Docker Compose** is a tool for defining and running **multi-container Docker applications** using YAML files.

### What it does in our setup:
- **Orchestrates multiple services** - App + optional Ollama + Traefik
- **Manages networks** - Automatic service discovery
- **Handles volumes** - Persistent data across restarts
- **Environment management** - Different configs for dev/prod

### What happens when you run `make dev`:
1. **Reads docker-compose.dev.yml** - Loads service definitions
2. **Creates network** - For container communication
3. **Builds/pulls images** - Prepares containers
4. **Starts services** - In dependency order
5. **Sets up volumes** - For data persistence
6. **Configures ports** - For external access

### Key Docker Compose features in our setup:
```yaml
services:        # Define containers
  ragagument:    # Main application
  ollama:        # Optional AI service

volumes:         # Persistent storage
networks:        # Container networking
profiles:        # Optional services
```

### 📚 Resources:
- **Official Docs**: https://docs.docker.com/compose/
- **Compose File Reference**: https://docs.docker.com/compose/compose-file/
- **Getting Started**: https://docs.docker.com/compose/gettingstarted/

---

## Kubernetes (Container Orchestration)

### What is Kubernetes?
**Kubernetes (K8s)** is a container orchestration platform that automates deployment, scaling, and management of containerized applications.

### What it does in our setup:
- **Auto-scaling** - Automatically adjusts container count based on load
- **Load balancing** - Distributes traffic across containers
- **Self-healing** - Restarts failed containers automatically
- **Rolling updates** - Zero-downtime deployments
- **Resource management** - CPU/memory limits and requests

### What happens when you run `make k8s-deploy`:
1. **Applies manifests** - Creates K8s resources (deployments, services, etc.)
2. **Schedules pods** - Places containers on cluster nodes
3. **Creates services** - Enables network access
4. **Sets up ingress** - External traffic routing
5. **Configures HPA** - Auto-scaling based on metrics

### Key Kubernetes resources in our setup:
- **Deployments**: Manage container replicas
- **Services**: Network access to pods
- **ConfigMaps**: Configuration data
- **Secrets**: Sensitive data (API keys)
- **Ingress**: External access routing
- **HorizontalPodAutoscaler**: Auto-scaling

### 📚 Resources:
- **Official Docs**: https://kubernetes.io/docs/
- **Concepts**: https://kubernetes.io/docs/concepts/
- **Tutorials**: https://kubernetes.io/docs/tutorials/
- **kubectl Cheat Sheet**: https://kubernetes.io/docs/reference/kubectl/cheatsheet/

---

## Helm (Kubernetes Package Manager)

### What is Helm?
**Helm** is a package manager for Kubernetes that simplifies deployment and management of complex applications.

### What it does in our setup:
- **Templates manifests** - Generates K8s YAML from templates
- **Manages releases** - Version control for deployments
- **Handles dependencies** - Manages related charts
- **Environment customization** - Different values for dev/staging/prod

### What happens when you use Helm:
1. **Reads Chart.yaml** - Chart metadata
2. **Processes templates** - Generates K8s manifests
3. **Applies values** - Customizes for environment
4. **Manages releases** - Tracks deployment versions

### 📚 Resources:
- **Official Docs**: https://helm.sh/docs/
- **Chart Guide**: https://helm.sh/docs/topics/charts/
- **Helm Hub**: https://artifacthub.io/

---

## GitHub Actions (CI/CD)

### What is GitHub Actions?
**GitHub Actions** is a CI/CD platform that automates software workflows directly in GitHub repositories.

### What it does in our setup:
- **Automated testing** - Runs tests on every push
- **Security scanning** - Checks for vulnerabilities
- **Multi-environment deployment** - Dev → Staging → Production
- **Artifact management** - Stores build artifacts
- **Notification system** - Alerts on failures

### What happens when you push code:
1. **Triggers workflow** - Based on push/PR events
2. **Runs jobs in parallel** - Code quality, tests, security
3. **Builds containers** - Creates optimized images
4. **Runs security scans** - Trivy and Gitleaks
5. **Deploys to environments** - Progressive rollout

### Key GitHub Actions features in our setup:
```yaml
on:              # When to run
  push:
  pull_request:

jobs:            # What to do
  code-quality:
  unit-tests:
  build-and-scan:
  deploy-staging:
  deploy-production:
```

### 📚 Resources:
- **Official Docs**: https://docs.github.com/en/actions
- **Workflow Syntax**: https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions
- **Marketplace**: https://github.com/marketplace?type=actions

---

## Trivy (Security Scanning)

### What is Trivy?
**Trivy** is a comprehensive security scanner for containers, code, and infrastructure.

### What it does in our setup:
- **Container scanning** - Detects OS package vulnerabilities
- **Dependency scanning** - Checks Python packages for issues
- **SBOM generation** - Creates Software Bill of Materials
- **CI/CD integration** - Automated security gates

### What happens when you run `make security-scan`:
1. **Scans container image** - Analyzes OS packages and dependencies
2. **Checks vulnerabilities** - Compares against known CVEs
3. **Generates report** - SARIF format for GitHub integration
4. **Fails on critical issues** - Blocks deployment if needed

### Key Trivy features in our setup:
- **Vulnerability scanning** - OS and library packages
- **Secret detection** - Finds exposed credentials
- **License checking** - Compliance verification
- **SBOM generation** - Supply chain transparency

### 📚 Resources:
- **Official Docs**: https://aquasecurity.github.io/trivy/
- **Getting Started**: https://aquasecurity.github.io/trivy/getting-started/
- **GitHub Action**: https://github.com/aquasecurity/trivy-action

---

## Gitleaks (Secret Detection)

### What is Gitleaks?
**Gitleaks** is a tool for detecting secrets and sensitive information in Git repositories.

### What it does in our setup:
- **Scans commits** - Checks for exposed API keys, passwords
- **Pattern matching** - Recognizes common secret formats
- **Prevents leaks** - Blocks commits with sensitive data
- **Audit trail** - Tracks secret exposure history

### What happens during CI/CD:
1. **Scans repository** - Checks all files for secrets
2. **Pattern matching** - Uses regex for common formats
3. **Reports findings** - Lists potential security issues
4. **Fails pipeline** - Prevents secret exposure

### Key Gitleaks features in our setup:
```toml
[[rules]]
id = "ragagument-api-key"
description = "RAGagument API Key"
regex = '''(?i)(deepseek|ollama)[_\-]?api[_\-]?key["']?\s*[:=]\s*["']([a-zA-Z0-9_\-]{32,})["']'''
```

### 📚 Resources:
- **Official Docs**: https://github.com/gitleaks/gitleaks
- **Configuration**: https://github.com/gitleaks/gitleaks#configuration
- **GitHub Action**: https://github.com/gitleaks/gitleaks-action

---

## FAISS (Vector Database)

### What is FAISS?
**FAISS (Facebook AI Similarity Search)** is a library for efficient similarity search and clustering of dense vectors.

### What it does in our setup:
- **Vector storage** - Stores document embeddings
- **Similarity search** - Finds relevant documents for queries
- **High performance** - Optimized for large-scale search
- **GPU support** - Accelerated search operations

### What happens during RAG processing:
1. **Document ingestion** - Converts documents to vectors
2. **Index building** - Creates searchable FAISS index
3. **Query processing** - Finds similar vectors
4. **Result ranking** - Returns most relevant documents

### Key FAISS features in our setup:
- **IVF indexing** - Inverted File System for scalability
- **PQ quantization** - Reduces memory usage
- **GPU acceleration** - Faster search operations
- **Persistence** - Saves/loads indexes to disk

### 📚 Resources:
- **Official Docs**: https://github.com/facebookresearch/faiss
- **Wiki**: https://github.com/facebookresearch/faiss/wiki
- **Python Tutorial**: https://github.com/facebookresearch/faiss/blob/main/tutorial/python/1-Flat.py

---

## Ollama (Local AI Models)

### What is Ollama?
**Ollama** is a tool for running large language models locally on your machine.

### What it does in our setup:
- **Local AI inference** - Runs models without cloud APIs
- **Privacy preservation** - Keeps data on-premises
- **Cost reduction** - No API usage charges
- **Offline capability** - Works without internet

### What happens when you use Ollama:
1. **Model download** - Fetches models from registry
2. **Local inference** - Processes prompts locally
3. **API compatibility** - OpenAI-compatible endpoints
4. **Resource management** - GPU/CPU optimization

### Key Ollama features in our setup:
```bash
# Pull models
ollama pull llama2

# Run inference
ollama run llama2

# API access
curl http://localhost:11434/api/generate
```

### 📚 Resources:
- **Official Docs**: https://github.com/jmorganca/ollama
- **Model Library**: https://ollama.com/library
- **API Reference**: https://github.com/jmorganca/ollama/blob/main/docs/api.md

---

## YAML (Configuration Format)

### What is YAML?
**YAML (YAML Ain't Markup Language)** is a human-readable data serialization format.

### What it does in our setup:
- **Configuration files** - Docker Compose, Kubernetes manifests
- **Environment settings** - Config maps and secrets
- **CI/CD workflows** - GitHub Actions definitions
- **Infrastructure as Code** - Declarative resource definitions

### Key YAML files in our setup:
- **docker-compose.yml** - Multi-container orchestration
- **k8s/deployment.yaml** - Kubernetes resource definitions
- **config/base.yaml** - Application configuration
- **.github/workflows/** - CI/CD pipeline definitions

### YAML syntax examples:
```yaml
# Simple key-value
environment: production

# Lists
services:
  - web
  - api
  - db

# Nested objects
app:
  port: 8501
  debug: false
```

### 📚 Resources:
- **Official Spec**: https://yaml.org/spec/
- **YAML Tutorial**: https://yaml.org/learn/
- **Online Validator**: https://www.yamllint.com/

---

## 🔄 Technology Workflow Summary

### When you run `make dev`:

1. **Make** → Reads Makefile and executes `docker compose` command
2. **Docker Compose** → Orchestrates container startup
3. **Docker** → Builds image from Dockerfile, starts containers
4. **Application** → Uses FAISS for vector search, optionally Ollama for local AI
5. **Health checks** → Monitor container status

### When you run `make k8s-deploy`:

1. **Make** → Executes `kubectl apply` commands
2. **Kubernetes** → Schedules containers on cluster
3. **Helm** → Manages complex deployments (if used)
4. **Ingress** → Routes external traffic
5. **HPA** → Auto-scales based on metrics

### When you push code:

1. **GitHub Actions** → Triggers CI/CD pipeline
2. **Trivy** → Scans for security vulnerabilities
3. **Gitleaks** → Checks for exposed secrets
4. **Docker** → Builds optimized production images
5. **Kubernetes** → Deploys to staging/production

---

## 🎯 Quick Reference

| Technology | Purpose | When Used | Key Command |
|------------|---------|-----------|-------------|
| **Make** | Build automation | `make dev` | `make help` |
| **Docker** | Containerization | Always | `docker --version` |
| **Docker Compose** | Multi-container | Development | `docker compose ps` |
| **Kubernetes** | Orchestration | Production | `kubectl get pods` |
| **GitHub Actions** | CI/CD | On push | Automatic |
| **Trivy** | Security scanning | CI/CD | `make security-scan` |
| **Gitleaks** | Secret detection | CI/CD | Automatic |
| **FAISS** | Vector search | Runtime | Automatic |
| **Ollama** | Local AI | Optional | `ollama pull model` |
| **YAML** | Configuration | Everywhere | File editing |

---

## 📚 Learning Path

### Beginner → Intermediate → Advanced:
1. **Start with Docker** - Learn container basics
2. **Docker Compose** - Multi-service applications
3. **Kubernetes** - Production orchestration
4. **CI/CD** - Automated workflows
5. **Security** - Scanning and compliance

### Recommended Order:
1. [Docker Getting Started](https://docs.docker.com/get-started/)
2. [Docker Compose Overview](https://docs.docker.com/compose/)
3. [Kubernetes Basics](https://kubernetes.io/docs/tutorials/kubernetes-basics/)
4. [GitHub Actions Guide](https://docs.github.com/en/actions/learn-github-actions)

---

## 🚨 Troubleshooting

### Common Issues:
- **Port conflicts**: Check `lsof -i :8501`
- **Permission denied**: Run with `sudo` or add to docker group
- **Build failures**: Check logs with `docker compose logs`
- **Memory issues**: Increase Docker memory limits

### Debug Commands:
```bash
# Check container status
docker compose ps

# View logs
docker compose logs -f

# Enter container
docker compose exec ragagument bash

# Check resources
docker stats
```

---

## 🎉 Summary

This containerization setup uses **enterprise-grade technologies** that work together to provide:

- ✅ **Reliable deployment** - Docker containers
- ✅ **Scalable orchestration** - Kubernetes
- ✅ **Automated pipelines** - GitHub Actions
- ✅ **Security scanning** - Trivy & Gitleaks
- ✅ **High performance** - FAISS vector search
- ✅ **Local AI options** - Ollama integration

Each technology serves a specific purpose in the **DevOps lifecycle**, from development to production deployment! 🚀