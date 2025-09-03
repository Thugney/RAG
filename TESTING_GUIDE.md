# 🧪 RAGagument Containerization Testing Guide

## Prerequisites

Before testing, ensure you have:
- ✅ Docker installed and running
- ✅ Docker Compose (modern version that uses `docker compose` command)
- ✅ At least 4GB RAM available
- ✅ Updated `.env` file with your API keys
- ✅ Existing `rag_venv` virtual environment (optional, for local development)

## 🐳 Virtual Environment vs Docker

### Understanding Your Setup
You have both a Python virtual environment (`rag_venv`) and Docker containers:

- **Virtual Environment (`rag_venv`)**: For local development, fast iteration
- **Docker Containers**: For deployment consistency, production environments

### When to Use Each:
```bash
# Use virtual environment for development
source rag_venv/bin/activate
streamlit run app.py

# Use Docker for deployment testing
make dev
```

## 🚀 Quick Start Testing

### 1. Validate Setup
```bash
# Run the validation script
./scripts/validate-setup.sh
```

### 2. Build Development Image
```bash
# Build the development image
make build-dev

# Or manually:
docker build --target development -t ragagument:dev .
```

### 3. Start Development Environment
```bash
# Start with hot reload
make dev

# Or manually:
docker-compose -f docker-compose.dev.yml up --build
```

### 4. Verify Application is Running
```bash
# Check if containers are running
docker-compose ps

# Check application health
curl http://localhost:8501/_stcore/health

# View application logs
docker-compose logs -f ragagument
```

### 5. Access the Application
- **Streamlit UI**: http://localhost:8501
- **Debug Port**: localhost:5678 (for VS Code debugging)
- **Health Check**: http://localhost:8501/_stcore/health

## 🧪 Comprehensive Testing

### Development Environment Testing
```bash
# 1. Start development environment
make dev

# 2. Test hot reload (in another terminal)
echo "# Add this line to app.py" >> app.py
# Watch the container restart automatically

# 3. Test volume mounts
echo "test file" > uploaded_docs/test.txt
# File should appear in container

# 4. Test debugging
# Connect VS Code debugger to localhost:5678
```

### Production Environment Testing
```bash
# 1. Build production image
make build-prod

# 2. Start production environment
make prod-up

# 3. Test production features
curl -f http://localhost:8501/_stcore/health
docker-compose logs ragagument

# 4. Test resource limits
docker stats

# 5. Clean up
make prod-clean
```

### Security Testing
```bash
# 1. Run security scan
make security-scan

# 2. Check for vulnerabilities
make vuln-check

# 3. Generate SBOM
make sbom
```

## 🔍 Expected Results

### ✅ Successful Development Environment
```
$ make dev
[+] Building 45.2s
[+] Running 3/3
 ⠿ Container ragagument-ragagument-1  Started
 ⠿ Container ragagument-ragagument-1  Healthy
 ⠿ Container ragagument-ragagument-1  Logs: "Streamlit app is running on http://0.0.0.0:8501"
```

### ✅ Health Check Response
```json
{
  "status": "healthy",
  "timestamp": "2025-01-03T16:40:00.000Z",
  "version": "1.0.0"
}
```

### ✅ Container Logs
```
INFO:streamlit:Streamlit app is running on http://0.0.0.0:8501
INFO:config_loader:Loaded configuration for environment: development
INFO:embedder_factory:Initializing HuggingFace embedding model
INFO:vector_store:FAISS vector store initialized
INFO:chat_history_db:SQLite database initialized
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Docker Permission Denied
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again
```

#### 2. Port Already in Use
```bash
# Find process using port
lsof -i :8501
# Kill the process or change port in .env
```

#### 3. Build Failures
```bash
# Clear Docker cache
docker system prune -a
# Rebuild without cache
docker compose build --no-cache
```

#### 3.1. Logging Package Error
```bash
# If you see "logging==0.4.9.6" error:
# This package has been removed from requirements.txt
# The logging module is part of Python's standard library
# No action needed - this was already fixed
```

#### 4. Memory Issues
```bash
# Check available memory
docker system info | grep "Total Memory"
# Increase Docker memory limit in Docker Desktop
```

#### 5. API Key Issues
```bash
# Check .env file
cat .env | grep DEEPSEEK_API_KEY
# Ensure API key is valid and has proper format
```

### Debug Commands
```bash
# View detailed logs
docker compose logs -f --tail=100

# Enter container shell
docker compose exec ragagument /bin/bash

# Check container resource usage
docker stats

# View container environment
docker compose exec ragagument env

# Test network connectivity
docker compose exec ragagument curl -I http://localhost:8501
```

## 📊 Performance Benchmarks

### Expected Performance
- **Build Time**: 2-5 minutes (first build), 30-60 seconds (cached)
- **Startup Time**: 10-30 seconds
- **Memory Usage**: 300-800 MB (development), 150-300 MB (production)
- **Health Check**: < 2 seconds response time

### Resource Requirements
- **Development**: 2GB RAM, 2 CPU cores
- **Production**: 1GB RAM, 1 CPU core
- **Storage**: 5GB free space for images and volumes

## 🔄 CI/CD Testing

### Local CI/CD Simulation
```bash
# Simulate CI pipeline locally
make test
make build-prod
make security-scan

# Test production deployment locally
make prod-up
curl -f http://localhost:8501/_stcore/health
make prod-clean
```

### GitHub Actions Testing
1. Push changes to a feature branch
2. Check GitHub Actions tab for pipeline status
3. Review security scan results
4. Verify deployment to staging environment

## 📈 Monitoring & Observability

### Application Metrics
```bash
# View application metrics (if enabled)
curl http://localhost:8000/metrics

# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### Log Analysis
```bash
# Follow application logs
docker compose logs -f ragagument

# Search for specific log patterns
docker compose logs ragagument | grep ERROR

# Export logs for analysis
docker compose logs ragagument > app_logs.txt
```

## 🎯 Success Criteria

### ✅ Development Environment
- [ ] Application starts successfully
- [ ] Health check returns 200 OK
- [ ] UI is accessible at http://localhost:8501
- [ ] Hot reload works when code changes
- [ ] Volume mounts are functional
- [ ] Debug port is accessible

### ✅ Production Environment
- [ ] Container uses non-root user
- [ ] Resource limits are enforced
- [ ] Health checks pass
- [ ] Security scan shows no critical vulnerabilities
- [ ] Application responds within SLA (< 2 seconds)

### ✅ CI/CD Pipeline
- [ ] All tests pass
- [ ] Security scans complete without critical findings
- [ ] Images build successfully
- [ ] Deployment to staging works
- [ ] Rollback procedures work

## 🚨 Emergency Procedures

### Container Won't Start
```bash
# Check container logs
docker compose logs ragagument

# Remove and restart
docker compose down
docker compose up --build --force-recreate
```

### Application Crashes
```bash
# Check application logs
docker compose logs -f ragagument

# Restart application
docker compose restart ragagument

# Full rebuild
make clean
make dev
```

### Database Issues
```bash
# Reset database volume
docker compose down -v
docker volume rm ragagument_rag_prod_database
docker compose up -d
```

## 📞 Support

If you encounter issues:
1. Check this troubleshooting guide
2. Review the logs: `docker-compose logs`
3. Verify your `.env` configuration
4. Check Docker resource allocation
5. Consult the documentation in `docs/` directory

## 🎉 Next Steps

Once testing is successful:
1. **Configure production secrets** in your deployment environment
2. **Set up monitoring** (Prometheus/Grafana)
3. **Configure CI/CD secrets** in GitHub repository
4. **Deploy to staging** environment
5. **Set up production** deployment pipeline
6. **Configure backup** and disaster recovery procedures