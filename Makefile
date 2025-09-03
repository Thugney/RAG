.PHONY: help dev up down logs clean build test security-scan k8s-deploy k8s-clean

# Default target
help: ## Show this help message
	@echo "RAGagument Container Management"
	@echo ""
	@echo "Development:"
	@echo "  dev         Start development environment with hot reload"
	@echo "  up          Start all services"
	@echo "  down        Stop all services"
	@echo "  logs        Show logs from all services"
	@echo "  clean       Clean up containers and volumes"
	@echo ""
	@echo "Building:"
	@echo "  build       Build Docker images"
	@echo "  build-dev   Build development image"
	@echo "  build-prod  Build production image"
	@echo ""
	@echo "Testing:"
	@echo "  test        Run tests in container"
	@echo "  test-local  Run tests locally"
	@echo ""
	@echo "Security:"
	@echo "  security-scan    Run security scanning"
	@echo "  vuln-check       Check for vulnerabilities"
	@echo ""
	@echo "Kubernetes:"
	@echo "  k8s-deploy      Deploy to Kubernetes"
	@echo "  k8s-clean       Clean Kubernetes resources"
	@echo ""
	@echo "Environment variables:"
	@echo "  REGISTRY    Docker registry (default: local)"
	@echo "  TAG         Image tag (default: latest)"
	@echo "  ENVIRONMENT Environment (default: development)"

# Development environment
dev: ## Start development environment with hot reload
	docker compose -f docker-compose.dev.yml up --build

dev-ollama: ## Start development with Ollama
	docker-compose -f docker-compose.dev.yml --profile with-ollama up --build

up: ## Start all services
	docker compose -f docker-compose.dev.yml up -d

down: ## Stop all services
	docker compose -f docker-compose.dev.yml down

logs: ## Show logs from all services
	docker compose -f docker-compose.dev.yml logs -f

logs-app: ## Show logs from app service only
	docker compose -f docker-compose.dev.yml logs -f ragagument

clean: ## Clean up containers, volumes, and images
	docker compose -f docker-compose.dev.yml down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

# Building
build: ## Build all Docker images
	docker build -t ragagument:latest .

build-dev: ## Build development image
	docker build --target development -t ragagument:dev .

build-prod: ## Build production image
	docker build --target production -t ragagument:prod .

build-multi: ## Build for multiple architectures
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--target production \
		-t $(REGISTRY)ragagument:$(TAG) \
		--push .

# Testing
test: ## Run tests in container
	docker run --rm -v $(PWD):/app ragagument:test pytest tests/ -v

test-local: ## Run tests locally
	pytest tests/ -v --cov=./ --cov-report=html

test-integration: ## Run integration tests
	docker compose -f docker-compose.test.yml up --abort-on-container-exit

# Security
security-scan: ## Run comprehensive security scanning
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasecurity/trivy image ragagument:latest \
		--format table \
		--severity HIGH,CRITICAL

vuln-check: ## Check for vulnerabilities
	docker run --rm -v $(PWD):/app \
		aquasecurity/trivy fs /app \
		--format table \
		--severity HIGH,CRITICAL

sbom: ## Generate Software Bill of Materials
	docker run --rm -v $(PWD):/app \
		aquasecurity/trivy fs /app \
		--format spdx-json \
		--output /app/sbom.json

# Production
prod-up: ## Start production environment
	docker compose -f docker-compose.production.yml up -d

prod-down: ## Stop production environment
	docker compose -f docker-compose.production.yml down

prod-logs: ## Show production logs
	docker compose -f docker-compose.production.yml logs -f

prod-clean: ## Clean production environment
	docker-compose -f docker-compose.production.yml down -v

# Kubernetes
k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/secret.yaml
	kubectl apply -f k8s/persistent-volume-claims.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl apply -f k8s/ingress.yaml
	kubectl apply -f k8s/hpa.yaml
	kubectl apply -f k8s/network-policy.yaml

k8s-status: ## Check Kubernetes deployment status
	kubectl get all -n ragagument-production
	kubectl get pvc -n ragagument-production
	kubectl get ingress -n ragagument-production

k8s-logs: ## Show Kubernetes pod logs
	kubectl logs -f deployment/ragagument -n ragagument-production

k8s-clean: ## Clean Kubernetes resources
	kubectl delete namespace ragagument-production --ignore-not-found=true

k8s-rollback: ## Rollback Kubernetes deployment
	kubectl rollout undo deployment/ragagument -n ragagument-production

# Utility
shell: ## Open shell in running container
	docker compose -f docker-compose.dev.yml exec ragagument /bin/bash

shell-prod: ## Open shell in production container
	docker compose -f docker-compose.production.yml exec ragagument /bin/bash

health: ## Check application health
	curl -f http://localhost:8501/_stcore/health || echo "Health check failed"

health-prod: ## Check production health
	curl -f http://localhost:8501/_stcore/health || echo "Health check failed"

# Environment setup
setup: ## Initial setup
	@echo "Setting up RAGagument development environment..."
	@cp .env.example .env 2>/dev/null || echo ".env.example not found"
	@echo "Please edit .env file with your configuration"
	@mkdir -p uploaded_docs vector_db

setup-prod: ## Production setup
	@echo "Setting up production environment..."
	@echo "Please ensure the following environment variables are set:"
	@echo "  - DEEPSEEK_API_KEY"
	@echo "  - OLLAMA_HOST (optional)"
	@echo "  - REGISTRY (optional)"
	@echo "  - TAG (optional)"

# CI/CD
ci-build: ## CI build process
	docker build --target test -t ragagument:test .
	docker run --rm ragagument:test pytest tests/ -v

ci-security: ## CI security scanning
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasecurity/trivy image ragagument:test \
		--exit-code 1 \
		--severity HIGH,CRITICAL

# Help
help-all: ## Show all available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'