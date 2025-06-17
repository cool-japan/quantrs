# QuantRS2 Docker Deployment Guide

This guide provides comprehensive instructions for deploying QuantRS2 using Docker in various environments.

## Quick Start

### Prerequisites

- Docker 20.0+ installed
- Docker Compose 2.0+ installed
- 8GB+ RAM recommended
- 10GB+ free disk space

### 1. Basic Deployment

```bash
# Clone the repository
git clone https://github.com/cool-japan/quantrs.git
cd quantrs/py

# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f quantrs2-jupyter
```

### 2. Access Services

- **Jupyter Lab**: http://localhost:8888
- **Documentation**: http://localhost:8000
- **Grafana Dashboard**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Traefik Dashboard**: http://localhost:8081

## Production Deployment

### Security Configuration

1. **Change Default Passwords**:
```bash
# Edit docker-compose.yml
vim docker-compose.yml

# Update these values:
- POSTGRES_PASSWORD=your_secure_password
- GF_SECURITY_ADMIN_PASSWORD=your_grafana_password
```

2. **Enable Authentication**:
```bash
# Create Jupyter configuration
mkdir -p config
cat > config/jupyter_config.py << 'EOF'
c.ServerApp.token = 'your-secure-token'
c.ServerApp.password = 'sha256:your-hashed-password'
EOF
```

3. **Configure SSL/TLS**:
```bash
# Create SSL certificates directory
mkdir -p ssl
# Add your certificates to ssl/ directory
# Update docker-compose.yml to mount SSL certificates
```

### Resource Limits

Update docker-compose.yml with appropriate resource limits:

```yaml
services:
  quantrs2-base:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### Persistence Configuration

Ensure data persistence with named volumes:

```yaml
volumes:
  quantrs2-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /path/to/persistent/storage
```

## Environment-Specific Deployments

### Development Environment

```bash
# Start development services only
docker-compose up quantrs2-dev quantrs2-jupyter quantrs2-db

# Enter development container
docker-compose exec quantrs2-dev bash

# Run tests
docker-compose exec quantrs2-dev pytest tests/

# Format code
docker-compose exec quantrs2-dev black .
```

### Production Environment

```bash
# Use production override
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale services
docker-compose up --scale quantrs2-base=3 -d

# Update services with zero downtime
docker-compose up -d --no-deps quantrs2-base
```

### GPU-Accelerated Environment

```bash
# Prerequisites: NVIDIA Docker runtime
# Install nvidia-docker2

# Start GPU service
docker-compose up quantrs2-gpu -d

# Verify GPU access
docker-compose exec quantrs2-gpu nvidia-smi
```

## Cloud Deployments

### AWS Deployment

1. **EC2 Instance**:
```bash
# Launch EC2 instance with Docker pre-installed
# Recommended: t3.large or larger for production

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.0.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

2. **ECS Deployment**:
```bash
# Convert to ECS task definition
docker-compose config > ecs-task-definition.json

# Deploy to ECS
aws ecs create-service --service-name quantrs2 --task-definition quantrs2
```

### Google Cloud Platform

```bash
# Deploy to Cloud Run
gcloud run deploy quantrs2 --image gcr.io/project/quantrs2:latest

# Deploy to GKE
kubectl apply -f k8s/
```

### Azure Container Instances

```bash
# Deploy to ACI
az container create --resource-group quantrs2 --name quantrs2 --image quantrs2:latest
```

## Kubernetes Deployment

### 1. Convert Docker Compose to Kubernetes

```bash
# Using Kompose
kompose convert -f docker-compose.yml

# Or use provided Kubernetes manifests
kubectl apply -f k8s/
```

### 2. Helm Deployment

```bash
# Add QuantRS2 Helm repository
helm repo add quantrs2 https://charts.quantrs2.io

# Install QuantRS2
helm install quantrs2 quantrs2/quantrs2 \
  --set persistence.enabled=true \
  --set ingress.enabled=true \
  --set-string ingress.hosts[0]=quantrs2.yourdomain.com
```

## Monitoring and Observability

### Prometheus Metrics

QuantRS2 exposes metrics at `/metrics` endpoint:

- `quantrs2_circuits_executed_total`: Total circuits executed
- `quantrs2_circuit_execution_duration_seconds`: Circuit execution time
- `quantrs2_memory_usage_bytes`: Memory usage
- `quantrs2_active_sessions`: Active user sessions

### Custom Dashboards

Import provided Grafana dashboards:

```bash
# Import dashboard
curl -X POST http://admin:admin@localhost:3001/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @docker/grafana-dashboards/quantrs2-overview.json
```

### Log Aggregation

Configure centralized logging:

```yaml
# Add to docker-compose.yml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

## Backup and Recovery

### Database Backup

```bash
# Backup PostgreSQL database
docker-compose exec quantrs2-db pg_dump -U quantrs2 quantrs2 > backup.sql

# Restore database
docker-compose exec -T quantrs2-db psql -U quantrs2 quantrs2 < backup.sql
```

### Data Volumes Backup

```bash
# Backup named volumes
docker run --rm -v quantrs2_quantrs2-data:/data -v $(pwd):/backup alpine tar czf /backup/quantrs2-data-$(date +%Y%m%d).tar.gz -C /data .

# Restore volumes
docker run --rm -v quantrs2_quantrs2-data:/data -v $(pwd):/backup alpine tar xzf /backup/quantrs2-data-20231201.tar.gz -C /data
```

## Performance Tuning

### Resource Optimization

1. **Memory Settings**:
```bash
# Set JVM heap size for services
export JAVA_OPTS="-Xmx2g -Xms1g"

# Configure Python memory limits
export PYTHONMALLOC=malloc
```

2. **CPU Optimization**:
```bash
# Set CPU affinity
docker run --cpuset-cpus="0-3" quantrs2:latest

# Use all available cores
docker run --cpus="0.000" quantrs2:latest
```

3. **I/O Optimization**:
```bash
# Use faster filesystem
docker run --tmpfs /tmp quantrs2:latest

# Mount SSD volumes
-v /fast/ssd/path:/app/data
```

## Troubleshooting

### Common Issues

1. **Container Won't Start**:
```bash
# Check logs
docker-compose logs quantrs2-base

# Check container status
docker-compose ps

# Restart services
docker-compose restart quantrs2-base
```

2. **Out of Memory**:
```bash
# Check memory usage
docker stats

# Increase memory limits
# Edit docker-compose.yml memory constraints
```

3. **Permission Issues**:
```bash
# Fix volume permissions
sudo chown -R 1000:1000 ./data

# Run with user mapping
docker run --user $(id -u):$(id -g) quantrs2:latest
```

4. **Network Issues**:
```bash
# Check network connectivity
docker network ls
docker network inspect quantrs2_quantrs2-network

# Reset networks
docker-compose down
docker network prune
docker-compose up
```

### Health Checks

```bash
# Run manual health check
docker-compose exec quantrs2-base python docker/healthcheck.py --verbose

# Check service health
docker-compose exec quantrs2-base python -c "import quantrs2; print('OK')"
```

### Performance Debugging

```bash
# Profile container performance
docker stats --no-stream

# Monitor resource usage
docker-compose exec quantrs2-base htop

# Check disk usage
docker system df
docker system prune
```

## Scaling

### Horizontal Scaling

```bash
# Scale base service
docker-compose up --scale quantrs2-base=5 -d

# Load balance with nginx
# Add nginx reverse proxy configuration
```

### Vertical Scaling

```bash
# Increase resource limits
docker-compose up --scale quantrs2-base=1 -d --force-recreate
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Deploy QuantRS2
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to production
        run: |
          docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### GitLab CI

```yaml
deploy:
  stage: deploy
  script:
    - docker-compose up -d
  only:
    - main
```

## Security Best Practices

1. **Network Security**:
   - Use internal networks for service communication
   - Expose only necessary ports
   - Configure firewall rules

2. **Container Security**:
   - Run containers as non-root users
   - Use read-only filesystems where possible
   - Scan images for vulnerabilities

3. **Data Security**:
   - Encrypt data at rest
   - Use secrets management
   - Regular security updates

## Maintenance

### Regular Tasks

```bash
# Update images
docker-compose pull
docker-compose up -d

# Clean up unused resources
docker system prune -a

# Rotate logs
docker-compose exec quantrs2-base logrotate /etc/logrotate.conf
```

### Monitoring Checklist

- [ ] All services running
- [ ] Disk space > 20%
- [ ] Memory usage < 80%
- [ ] CPU usage reasonable
- [ ] Database connections healthy
- [ ] Backup completion
- [ ] Security updates applied

## Support

For deployment issues:

1. Check logs: `docker-compose logs`
2. Verify configuration: `docker-compose config`
3. Test connectivity: `docker-compose exec service_name ping other_service`
4. Review resource usage: `docker stats`

For additional help, consult:
- [QuantRS2 Documentation](https://quantrs2.readthedocs.io)
- [Docker Documentation](https://docs.docker.com)
- [GitHub Issues](https://github.com/cool-japan/quantrs/issues)