#!/bin/bash

# RAGagument Security Audit Script
# Checks for sensitive files that might be exposed to Git or Docker

set -e

echo "🔒 RAGagument Security Audit"
echo "============================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if file is ignored by git
is_git_ignored() {
    local file="$1"
    if git check-ignore "$file" 2>/dev/null; then
        echo -e "${GREEN}✓ IGNORED${NC}"
    else
        echo -e "${RED}✗ TRACKED${NC}"
    fi
}

# Function to check if file is ignored by docker
is_docker_ignored() {
    local file="$1"
    if [[ -f ".dockerignore" ]]; then
        # Simple check - in production you'd want more sophisticated parsing
        if grep -q "^$(basename "$file")$" .dockerignore 2>/dev/null || grep -q "^$(dirname "$file")/" .dockerignore 2>/dev/null; then
            echo -e "${GREEN}✓ IGNORED${NC}"
        else
            echo -e "${RED}✗ INCLUDED${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ NO .dockerignore${NC}"
    fi
}

echo ""
echo "🔍 Checking Sensitive Files..."
echo "=============================="

# Critical security files to check
SENSITIVE_FILES=(
    ".env"
    ".env.local"
    ".env.development.local"
    ".env.test.local"
    ".env.production.local"
    "config/production.yaml"
    "k8s/secret.yaml"
    ".trivyignore"
    ".gitleaks.toml"
)

for file in "${SENSITIVE_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        printf "%-30s Git: " "$file"
        is_git_ignored "$file"
        printf "%-30s Docker: " ""
        is_docker_ignored "$file"
        echo ""
    fi
done

echo ""
echo "📁 Checking Data Directories..."
echo "==============================="

# Data directories that should never be committed
DATA_DIRS=(
    "uploaded_docs/"
    "vector_db/"
    "rag_venv/"
    "logs/"
    "__pycache__/"
)

for dir in "${DATA_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        printf "%-30s Git: " "$dir"
        is_git_ignored "$dir"
        printf "%-30s Docker: " ""
        is_docker_ignored "$dir"
        echo ""
    fi
done

echo ""
echo "🔐 Checking API Keys in Files..."
echo "==============================="

# Check for potential API keys in tracked files
echo "Scanning for potential API keys in tracked files..."
if command -v grep &> /dev/null; then
    # Look for common API key patterns in tracked files
    TRACKED_FILES=$(git ls-files 2>/dev/null || find . -type f -not -path './.*' -not -path './rag_venv/*' -not -path './uploaded_docs/*' -not -path './vector_db/*' | head -20)

    API_KEY_FOUND=false
    for file in $TRACKED_FILES; do
        if [[ -f "$file" ]] && [[ "$file" != *.pyc ]] && [[ "$file" != *.log ]]; then
            # Check for common API key patterns
            if grep -q -E "(sk-|pk_|api_key|secret|token|password).*[=:]" "$file" 2>/dev/null; then
                echo -e "${YELLOW}⚠ Potential API key pattern found in: $file${NC}"
                API_KEY_FOUND=true
            fi
        fi
    done

    if [[ "$API_KEY_FOUND" = false ]]; then
        echo -e "${GREEN}✓ No obvious API key patterns found in tracked files${NC}"
    fi
else
    echo -e "${YELLOW}⚠ grep not available for API key scanning${NC}"
fi

echo ""
echo "📊 Git Status Summary"
echo "====================="

# Show git status
echo "Files currently tracked by Git:"
git status --porcelain | head -10

if [[ $(git status --porcelain | wc -l) -gt 10 ]]; then
    echo "... and $(($(git status --porcelain | wc -l) - 10)) more files"
fi

echo ""
echo "🎯 Security Recommendations"
echo "==========================="

# Check if .env exists and is properly ignored
if [[ -f ".env" ]] && ! git check-ignore .env 2>/dev/null; then
    echo -e "${RED}❌ CRITICAL: .env file is being tracked by Git!${NC}"
    echo "   Run: git rm --cached .env"
    echo "   Make sure .env is in .gitignore"
fi

# Check for common security issues
if [[ ! -f ".gitignore" ]]; then
    echo -e "${RED}❌ CRITICAL: No .gitignore file found!${NC}"
fi

if [[ ! -f ".dockerignore" ]]; then
    echo -e "${RED}❌ CRITICAL: No .dockerignore file found!${NC}"
fi

# Check if sensitive directories exist and are ignored
for dir in "${DATA_DIRS[@]}"; do
    if [[ -d "$dir" ]] && ! git check-ignore "$dir" 2>/dev/null; then
        echo -e "${YELLOW}⚠ WARNING: $dir exists and may be tracked by Git${NC}"
    fi
done

echo ""
echo "✅ Audit Complete"
echo "================="
echo "Review the output above and address any security issues."
echo ""
echo "Next steps:"
echo "1. Ensure all sensitive files are properly ignored"
echo "2. Never commit API keys, passwords, or secrets"
echo "3. Use environment variables for sensitive configuration"
echo "4. Regularly audit your repository for exposed secrets"