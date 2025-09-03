#!/bin/bash

# RAGagument Containerization Setup Validation Script
# This script validates the containerization setup without requiring Docker

set -e

echo "🔍 RAGagument Containerization Setup Validation"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Validation functions
validate_file() {
    local file=$1
    local description=$2

    if [[ -f "$file" ]]; then
        echo -e "${GREEN}✓${NC} $description: $file"
        return 0
    else
        echo -e "${RED}✗${NC} $description: $file (MISSING)"
        return 1
    fi
}

validate_directory() {
    local dir=$1
    local description=$2

    if [[ -d "$dir" ]]; then
        echo -e "${GREEN}✓${NC} $description: $dir"
        return 0
    else
        echo -e "${RED}✗${NC} $description: $dir (MISSING)"
        return 1
    fi
}

validate_yaml() {
    local file=$1
    local description=$2

    if [[ -f "$file" ]]; then
        if command -v python3 &> /dev/null; then
            if python3 -c "import yaml; list(yaml.safe_load_all(open('$file')))" 2>/dev/null; then
                echo -e "${GREEN}✓${NC} $description: $file (valid YAML)"
                return 0
            else
                echo -e "${RED}✗${NC} $description: $file (invalid YAML)"
                return 1
            fi
        else
            echo -e "${YELLOW}⚠${NC} $description: $file (cannot validate YAML - python3 not found)"
            return 0
        fi
    else
        echo -e "${RED}✗${NC} $description: $file (MISSING)"
        return 1
    fi
}

echo ""
echo "📁 Checking Directory Structure..."
echo "----------------------------------"

# Core files
validate_file "Dockerfile" "Multi-stage Dockerfile"
validate_file ".dockerignore" "Docker ignore file"
validate_file "Makefile" "Build automation script"
validate_file "requirements.txt" "Python dependencies"
validate_file "requirements-test.txt" "Test dependencies"

# Configuration
validate_directory "config" "Configuration directory"
validate_file "config/base.yaml" "Base configuration"
validate_file "config/development.yaml" "Development configuration"
validate_file "config/production.yaml" "Production configuration"
validate_file "config/staging.yaml" "Staging configuration"

# Docker Compose
validate_file "docker-compose.dev.yml" "Development Docker Compose"
validate_file "docker-compose.production.yml" "Production Docker Compose"

# Kubernetes
validate_directory "k8s" "Kubernetes manifests directory"
validate_file "k8s/namespace.yaml" "Kubernetes namespaces"
validate_file "k8s/configmap.yaml" "Kubernetes ConfigMaps"
validate_file "k8s/secret.yaml" "Kubernetes Secrets"
validate_file "k8s/persistent-volume-claims.yaml" "Persistent Volume Claims"
validate_file "k8s/deployment.yaml" "Kubernetes deployment"
validate_file "k8s/service.yaml" "Kubernetes service"
validate_file "k8s/hpa.yaml" "Horizontal Pod Autoscaler"
validate_file "k8s/ingress.yaml" "Ingress configuration"

# CI/CD
validate_directory ".github" "GitHub Actions directory"
validate_directory ".github/workflows" "GitHub Actions workflows"
validate_file ".github/workflows/ci-cd.yml" "CI/CD pipeline"
validate_file ".github/workflows/security-scan.yml" "Security scanning workflow"

# Security
validate_file ".trivyignore" "Trivy ignore file"
validate_file ".trivy.yaml" "Trivy configuration"
validate_file ".gitleaks.toml" "Gitleaks configuration"

echo ""
echo "🔧 Checking Configuration Files..."
echo "-----------------------------------"

# Validate YAML files
validate_yaml "config/base.yaml" "Base configuration"
validate_yaml "config/development.yaml" "Development configuration"
validate_yaml "config/production.yaml" "Production configuration"
validate_yaml "config/staging.yaml" "Staging configuration"

validate_yaml "docker-compose.dev.yml" "Development Docker Compose"
validate_yaml "docker-compose.production.yml" "Production Docker Compose"

validate_yaml "k8s/namespace.yaml" "Kubernetes namespaces"
validate_yaml "k8s/configmap.yaml" "Kubernetes ConfigMaps"
validate_yaml "k8s/secret.yaml" "Kubernetes Secrets"
validate_yaml "k8s/persistent-volume-claims.yaml" "Persistent Volume Claims"
validate_yaml "k8s/deployment.yaml" "Kubernetes deployment"
validate_yaml "k8s/service.yaml" "Kubernetes service"
validate_yaml "k8s/hpa.yaml" "Horizontal Pod Autoscaler"
validate_yaml "k8s/ingress.yaml" "Ingress configuration"

validate_yaml ".github/workflows/ci-cd.yml" "CI/CD pipeline"
validate_yaml ".github/workflows/security-scan.yml" "Security scanning workflow"

echo ""
echo "📊 Setup Summary"
echo "================="

# Count files
total_files=$(find . -name "*.yml" -o -name "*.yaml" -o -name "Dockerfile" -o -name "Makefile" -o -name "*.txt" -o -name "*.toml" -o -name "*.sh" | grep -E "(config/|k8s/|\.github/|Dockerfile|Makefile|requirements|\.trivy|\.gitleaks)" | wc -l)
echo "Total configuration files: $total_files"

# Check if all required files exist
required_files=(
    "Dockerfile"
    ".dockerignore"
    "Makefile"
    "requirements.txt"
    "requirements-test.txt"
    "config/base.yaml"
    "config/development.yaml"
    "config/production.yaml"
    "config/staging.yaml"
    "docker-compose.dev.yml"
    "docker-compose.production.yml"
    "k8s/namespace.yaml"
    "k8s/configmap.yaml"
    "k8s/secret.yaml"
    "k8s/persistent-volume-claims.yaml"
    "k8s/deployment.yaml"
    "k8s/service.yaml"
    "k8s/hpa.yaml"
    "k8s/ingress.yaml"
    ".github/workflows/ci-cd.yml"
    ".github/workflows/security-scan.yml"
    ".trivyignore"
    ".trivy.yaml"
    ".gitleaks.toml"
)

missing_files=0
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        ((missing_files++))
    fi
done

if [[ $missing_files -eq 0 ]]; then
    echo -e "${GREEN}✅ All required files are present!${NC}"
    echo ""
    echo "🚀 Next Steps:"
    echo "1. Ensure Docker is installed and running"
    echo "2. Run 'make build-dev' to build the development image"
    echo "3. Run 'make dev' to start the development environment"
    echo "4. Run 'make test' to execute tests"
    echo "5. For production deployment, configure your Kubernetes cluster"
    echo "6. Set up CI/CD secrets in your GitHub repository"
    echo ""
    echo "📚 Useful Commands:"
    echo "  make help          - Show all available commands"
    echo "  make dev           - Start development environment"
    echo "  make build-prod    - Build production image"
    echo "  make security-scan - Run security scanning"
    echo "  make k8s-deploy    - Deploy to Kubernetes"
else
    echo -e "${RED}❌ $missing_files required files are missing!${NC}"
    exit 1
fi