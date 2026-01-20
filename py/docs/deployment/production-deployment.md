# QuantRS2 Production Deployment Guide

This comprehensive guide covers deploying QuantRS2 in production environments with all enterprise-grade features enabled including security hardening, monitoring, alerting, and structured logging.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Security Setup](#security-setup)  
3. [Environment Configuration](#environment-configuration)
4. [Production Deployment](#production-deployment)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Backup and Recovery](#backup-and-recovery)
7. [Scaling and Performance](#scaling-and-performance)
8. [Maintenance and Operations](#maintenance-and-operations)

## Prerequisites

### System Requirements

**Minimum Production Requirements:**
- **CPU**: 8 cores (16 recommended)
- **RAM**: 16GB (32GB recommended)
- **Storage**: 100GB SSD (500GB recommended)
- **Network**: 1Gbps bandwidth
- **OS**: Ubuntu 22.04 LTS, RHEL 8+, or CentOS 8+

**Container Runtime:**
- Docker 24.0+ with security features enabled
- Docker Compose 2.20+
- Container security scanner (Trivy, Clair, or similar)

**External Dependencies:**
- PostgreSQL 15+ (managed or self-hosted)
- Redis 7+ with authentication
- SSL/TLS certificates (Let's Encrypt or CA-issued)
- Load balancer (optional but recommended)

### Network Requirements

**Required Ports:**
- `443` - HTTPS (public)
- `80` - HTTP redirect to HTTPS (public)
- `8888` - Jupyter Lab (internal/VPN)
- `9090` - Prometheus metrics (internal)
- `3000` - Grafana dashboard (internal/VPN)
- `5432` - PostgreSQL (internal)
- `6379` - Redis (internal)

**Firewall Configuration:**
```bash
# Enable required ports
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp  # SSH

# Internal network access only
sudo ufw allow from 10.0.0.0/8 to any port 8888
sudo ufw allow from 10.0.0.0/8 to any port 9090
sudo ufw allow from 10.0.0.0/8 to any port 3000

# Database access (internal only)
sudo ufw allow from 172.20.0.0/16 to any port 5432
sudo ufw allow from 172.20.0.0/16 to any port 6379

sudo ufw enable
```

## Security Setup

### 1. Secrets Management

Create secure environment configuration:

```bash
# Create secure directory
sudo mkdir -p /opt/quantrs2/secrets
sudo chmod 700 /opt/quantrs2/secrets

# Generate strong passwords and keys
openssl rand -base64 32 > /opt/quantrs2/secrets/postgres_password
openssl rand -base64 32 > /opt/quantrs2/secrets/redis_password
openssl rand -base64 64 > /opt/quantrs2/secrets/master_key
openssl rand -base64 64 > /opt/quantrs2/secrets/jwt_secret

# Generate Jupyter password hash
python3 -c "from jupyter_server.auth import passwd; print(passwd('your-secure-password'))" > /opt/quantrs2/secrets/jupyter_password_hash

# Secure file permissions
sudo chmod 600 /opt/quantrs2/secrets/*
sudo chown -R quantrs2:quantrs2 /opt/quantrs2/secrets
```

### 2. SSL/TLS Configuration

**Option A: Let's Encrypt (Recommended)**
```bash
# Install Certbot
sudo apt-get update
sudo apt-get install certbot

# Generate certificates
sudo certbot certonly --standalone \
  -d yourdomain.com \
  -d api.yourdomain.com \
  -d jupyter.yourdomain.com \
  -d grafana.yourdomain.com

# Setup auto-renewal
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

**Option B: Corporate CA Certificates**
```bash
# Copy certificates to secure location
sudo mkdir -p /opt/quantrs2/certs
sudo cp your-cert.pem /opt/quantrs2/certs/cert.pem
sudo cp your-key.pem /opt/quantrs2/certs/key.pem
sudo cp ca-cert.pem /opt/quantrs2/certs/ca.pem

# Secure permissions
sudo chmod 644 /opt/quantrs2/certs/cert.pem
sudo chmod 600 /opt/quantrs2/certs/key.pem
sudo chown -R quantrs2:quantrs2 /opt/quantrs2/certs
```

### 3. Container Security

Enable advanced container security features:

```bash
# Install and configure AppArmor
sudo apt-get install apparmor apparmor-utils
sudo systemctl enable apparmor
sudo systemctl start apparmor

# Enable Docker security features
sudo mkdir -p /etc/docker
cat << EOF | sudo tee /etc/docker/daemon.json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  },
  "live-restore": true,
  "userland-proxy": false,
  "no-new-privileges": true,
  "seccomp-profile": "/etc/docker/seccomp.json",
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  }
}
EOF

sudo systemctl restart docker
```

### 4. Environment Variables

Create production environment file:

```bash
cat << 'EOF' > /opt/quantrs2/.env.production
# Core Configuration
QUANTRS2_ENV=production
QUANTRS2_DEBUG=false
QUANTRS2_LOG_LEVEL=INFO
TRAEFIK_DOMAIN=yourdomain.com

# Security
QUANTRS2_MASTER_KEY=$(cat /opt/quantrs2/secrets/master_key)
JWT_SECRET_KEY=$(cat /opt/quantrs2/secrets/jwt_secret)
JUPYTER_TOKEN=$(openssl rand -hex 32)
JUPYTER_PASSWORD_HASH=$(cat /opt/quantrs2/secrets/jupyter_password_hash)

# Database
POSTGRES_HOST=quantrs2-db
POSTGRES_PORT=5432
POSTGRES_DB=quantrs2_prod
POSTGRES_USER=quantrs2_prod
POSTGRES_PASSWORD=$(cat /opt/quantrs2/secrets/postgres_password)

# Cache
REDIS_HOST=quantrs2-redis
REDIS_PORT=6379
REDIS_PASSWORD=$(cat /opt/quantrs2/secrets/redis_password)

# Performance
MAX_QUBITS=1024
MAX_CIRCUIT_DEPTH=10000
MAX_CONCURRENT_JOBS=20
PYTHON_MEMORY_LIMIT=8g
GPU_MEMORY_LIMIT=16g

# Resource Limits
QUANTRS2_UID=1000
QUANTRS2_GID=1000
DATA_DIR=/opt/quantrs2/data

# Monitoring
PROMETHEUS_RETENTION_TIME=90d
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=$(cat /opt/quantrs2/secrets/postgres_password)

# Ports (internal binding)
JUPYTER_PORT=127.0.0.1:8888
PROMETHEUS_PORT=127.0.0.1:9090
GRAFANA_PORT=127.0.0.1:3000
POSTGRES_EXTERNAL_PORT=127.0.0.1:5432
REDIS_EXTERNAL_PORT=127.0.0.1:6379

# Network Security
DOCKER_SUBNET=172.20.0.0/16
DOCKER_GATEWAY=172.20.0.1

# Features
FEATURE_AUTHENTICATION=true
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
TRAEFIK_DASHBOARD_ENABLED=false
TRAEFIK_INSECURE=false
EOF

# Secure environment file
sudo chmod 600 /opt/quantrs2/.env.production
sudo chown quantrs2:quantrs2 /opt/quantrs2/.env.production
```

## Environment Configuration

### 1. Production Configuration Files

QuantRS2 uses environment-specific YAML configuration. Copy and customize the production configuration:

```bash
# Copy configuration template
sudo cp /opt/quantrs2/config/production.yaml /opt/quantrs2/config/production.local.yaml

# Edit production configuration
sudo nano /opt/quantrs2/config/production.local.yaml
```

Key production configuration changes:

```yaml
# /opt/quantrs2/config/production.local.yaml
environment: production
debug: false
secret_key: null  # Loaded from environment

database:
  host: quantrs2-db
  port: 5432
  database: quantrs2_prod
  username: quantrs2_prod
  max_connections: 100  # Increased for production
  ssl_mode: require
  connection_timeout: 30
  pool_size: 20
  max_overflow: 30

security:
  session_timeout: 1800  # 30 minutes
  max_login_attempts: 3
  enable_2fa: true
  rate_limit_requests: 100
  rate_limit_window: 60
  cors_origins:
    - https://yourdomain.com
    - https://api.yourdomain.com

performance:
  max_circuit_qubits: 100
  simulation_memory_limit: 65536  # 64GB
  max_concurrent_jobs: 50
  circuit_cache_size: 50000
  result_cache_ttl: 14400  # 4 hours
  enable_gpu: true
  gpu_memory_fraction: 0.9

logging:
  level: INFO
  file_path: /app/logs/quantrs2.log
  max_file_size: 104857600  # 100MB
  backup_count: 30
  enable_structured_logging: true
  enable_json_logging: true
  log_to_console: false

monitoring:
  enable_metrics: true
  metrics_port: 9090
  enable_health_checks: true
  health_check_port: 8080
  enable_tracing: true
  alert_webhook_url: https://your-alert-webhook.com/alerts
  prometheus_metrics: true
  grafana_integration: true

custom:
  production_mode: true
  data_retention_days: 365
  enable_analytics: true
  backup_frequency: daily
  enable_disaster_recovery: true
  compliance_mode: true
```

### 2. Monitoring Configuration

Configure comprehensive monitoring:

```yaml
# /opt/quantrs2/config/monitoring.yaml
monitoring:
  systems:
    - name: "quantrs2-monitoring"
      enabled: true
      storage_path: "/app/data/monitoring"
      
  metrics_collection:
    interval_seconds: 30
    retention_days: 90
    
  alert_rules:
    - name: "High CPU Usage"
      metric: "system.cpu_percent"
      threshold: 85.0
      comparison: ">"
      severity: "high"
      evaluation_window: 300
      
    - name: "High Memory Usage"
      metric: "system.memory_percent"
      threshold: 90.0
      comparison: ">"
      severity: "critical"
      evaluation_window: 180
      
    - name: "Circuit Execution Failures"
      metric: "app.circuit.failures"
      threshold: 10.0
      comparison: ">"
      severity: "high"
      evaluation_window: 600
      
  notification_channels:
    - name: "slack-alerts"
      type: "slack"
      webhook_url: "${SLACK_WEBHOOK_URL}"
      severity_filter: ["high", "critical"]
      
    - name: "email-alerts"
      type: "email"
      smtp_host: "smtp.yourdomain.com"
      smtp_port: 587
      from_address: "alerts@yourdomain.com"
      to_addresses: ["ops@yourdomain.com"]
      severity_filter: ["critical"]

logging:
  structured_logging:
    enabled: true
    format: "json"
    include_trace_context: true
    
  aggregation:
    destinations:
      - name: "elasticsearch-logs"
        type: "elasticsearch"
        endpoint: "https://elasticsearch.yourdomain.com"
        index_pattern: "quantrs2-logs-{date}"
        
      - name: "file-logs"
        type: "file"
        path: "/app/logs/aggregated.log"
        format: "json"
        
  error_analysis:
    enabled: true
    pattern_detection: true
    incident_management: true
    auto_incident_creation: true
```

### 3. Security Configuration

Configure advanced security features:

```yaml
# /opt/quantrs2/config/security.yaml
security:
  authentication:
    enabled: true
    provider: "jwt"
    token_expiry: 3600  # 1 hour
    refresh_token_expiry: 86400  # 24 hours
    
  authorization:
    enabled: true
    default_role: "user"
    admin_users: ["admin@yourdomain.com"]
    
  input_validation:
    strict_mode: true
    max_input_size: 1048576  # 1MB
    allowed_file_types: [".py", ".ipynb", ".qasm"]
    sanitize_inputs: true
    
  rate_limiting:
    enabled: true
    requests_per_minute: 100
    burst_limit: 200
    
  encryption:
    algorithm: "AES-256-GCM"
    key_rotation_days: 90
    
  audit_logging:
    enabled: true
    log_all_requests: true
    log_failed_attempts: true
    retention_days: 365
```

## Production Deployment

### 1. Infrastructure Setup

Create required directories and users:

```bash
# Create quantrs2 user
sudo useradd -r -m -s /bin/bash quantrs2
sudo usermod -aG docker quantrs2

# Create directory structure
sudo mkdir -p /opt/quantrs2/{data,logs,config,certs,secrets}
sudo mkdir -p /opt/quantrs2/data/{postgres,redis,grafana,prometheus,quantrs2}

# Set permissions
sudo chown -R quantrs2:quantrs2 /opt/quantrs2
sudo chmod -R 755 /opt/quantrs2
sudo chmod 700 /opt/quantrs2/secrets
```

### 2. Deploy Production Stack

```bash
# Switch to quantrs2 user
sudo su - quantrs2

# Clone repository
cd /opt/quantrs2
git clone https://github.com/cool-japan/quantrs.git
cd quantrs/py

# Copy production environment
cp /opt/quantrs2/.env.production .env

# Validate configuration
docker-compose -f docker-compose.secure.yml config

# Deploy production stack
docker-compose -f docker-compose.secure.yml up -d

# Verify deployment
docker-compose -f docker-compose.secure.yml ps
docker-compose -f docker-compose.secure.yml logs -f
```

### 3. Health Checks and Validation

Verify all components are healthy:

```bash
# Check service health
docker-compose -f docker-compose.secure.yml exec quantrs2-base python docker/healthcheck.py --verbose

# Test database connectivity
docker-compose -f docker-compose.secure.yml exec quantrs2-db pg_isready -U quantrs2_prod

# Test Redis connectivity
docker-compose -f docker-compose.secure.yml exec quantrs2-redis redis-cli -a "${REDIS_PASSWORD}" ping

# Test QuantRS2 API
curl -k https://yourdomain.com/health

# Test Jupyter access
curl -k https://jupyter.yourdomain.com

# Test monitoring endpoints
curl -k https://grafana.yourdomain.com/api/health
curl http://localhost:9090/metrics
```

### 4. Load Balancer Configuration

**Nginx Configuration (if using external load balancer):**

```nginx
# /etc/nginx/sites-available/quantrs2
upstream quantrs2_backend {
    least_conn;
    server 127.0.0.1:8080 max_fails=3 fail_timeout=30s;
    # Add more backend servers for scaling
    # server 127.0.0.1:8081 max_fails=3 fail_timeout=30s;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com api.yourdomain.com;
    
    ssl_certificate /opt/quantrs2/certs/cert.pem;
    ssl_certificate_key /opt/quantrs2/certs/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    location / {
        proxy_pass http://quantrs2_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
        
        # Health checks
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
    }
    
    location /health {
        access_log off;
        proxy_pass http://quantrs2_backend/health;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name yourdomain.com api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

## Monitoring and Observability

### 1. Prometheus Configuration

Configure Prometheus for comprehensive metrics collection:

```yaml
# /opt/quantrs2/prometheus.yml
global:
  scrape_interval: 30s
  evaluation_interval: 30s
  
rule_files:
  - "/etc/prometheus/rules/*.yml"
  
scrape_configs:
  - job_name: 'quantrs2-app'
    static_configs:
      - targets: ['quantrs2-base:9090']
    scrape_interval: 30s
    metrics_path: /metrics
    
  - job_name: 'quantrs2-system'
    static_configs:
      - targets: ['quantrs2-base:9091']
    scrape_interval: 60s
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['quantrs2-db:9187']
    scrape_interval: 60s
    
  - job_name: 'redis'
    static_configs:
      - targets: ['quantrs2-redis:9121']
    scrape_interval: 60s
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
    scrape_interval: 60s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']
```

### 2. Grafana Dashboard Setup

Import QuantRS2 monitoring dashboards:

```bash
# Copy dashboards
sudo cp docker/grafana-dashboards/* /opt/quantrs2/data/grafana/dashboards/

# Import via API
curl -X POST \
  "http://admin:${GRAFANA_ADMIN_PASSWORD}@localhost:3000/api/dashboards/db" \
  -H "Content-Type: application/json" \
  -d @docker/grafana-dashboards/quantrs2-overview.json
```

### 3. Log Aggregation Setup

Configure structured logging with external aggregation:

```bash
# Configure Fluent Bit for log forwarding
cat << 'EOF' > /opt/quantrs2/fluent-bit.conf
[SERVICE]
    Flush         5
    Daemon        off
    Log_Level     info

[INPUT]
    Name              tail
    Path              /var/log/quantrs2/*.log
    Tag               quantrs2.*
    Parser            json
    DB                /tmp/flb_quantrs2.db

[FILTER]
    Name              grep
    Match             quantrs2.*
    Regex             level (ERROR|WARNING|CRITICAL)

[OUTPUT]
    Name              es
    Match             *
    Host              elasticsearch.yourdomain.com
    Port              443
    tls               on
    Index             quantrs2-logs
    Type              _doc
EOF
```

### 4. Alerting Rules

Configure production alerting rules:

```yaml
# /opt/quantrs2/prometheus-rules.yml
groups:
  - name: quantrs2.rules
    rules:
      - alert: QuantRS2HighErrorRate
        expr: rate(quantrs2_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"
          
      - alert: QuantRS2ServiceDown
        expr: up{job="quantrs2-app"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "QuantRS2 service is down"
          description: "QuantRS2 application is not responding"
          
      - alert: QuantRS2HighMemoryUsage
        expr: quantrs2_memory_usage_bytes / quantrs2_memory_limit_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }}"
          
      - alert: QuantRS2DatabaseConnectionsHigh
        expr: quantrs2_db_connections_active / quantrs2_db_connections_max > 0.8
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "High database connection usage"
          description: "Database connections at {{ $value | humanizePercentage }}"
```

## Backup and Recovery

### 1. Database Backup Strategy

Implement automated database backups:

```bash
#!/bin/bash
# /opt/quantrs2/scripts/backup-database.sh

set -euo pipefail

BACKUP_DIR="/opt/quantrs2/backups/database"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Database backup
docker-compose exec -T quantrs2-db pg_dump \
  -U quantrs2_prod \
  -h localhost \
  -d quantrs2_prod \
  --no-password \
  --format=custom \
  --compress=9 \
  > "$BACKUP_DIR/quantrs2_${TIMESTAMP}.dump"

# Compress and encrypt backup
gpg --cipher-algo AES256 \
    --digest-algo SHA512 \
    --cert-digest-algo SHA512 \
    --compress-algo 1 \
    --symmetric \
    --output "$BACKUP_DIR/quantrs2_${TIMESTAMP}.dump.gpg" \
    "$BACKUP_DIR/quantrs2_${TIMESTAMP}.dump"

# Remove unencrypted backup
rm "$BACKUP_DIR/quantrs2_${TIMESTAMP}.dump"

# Cleanup old backups
find "$BACKUP_DIR" -name "*.dump.gpg" -mtime +$RETENTION_DAYS -delete

# Upload to S3 (optional)
# aws s3 cp "$BACKUP_DIR/quantrs2_${TIMESTAMP}.dump.gpg" s3://your-backup-bucket/quantrs2/

echo "Database backup completed: quantrs2_${TIMESTAMP}.dump.gpg"
```

### 2. Data Volume Backup

Backup persistent data volumes:

```bash
#!/bin/bash
# /opt/quantrs2/scripts/backup-volumes.sh

set -euo pipefail

BACKUP_DIR="/opt/quantrs2/backups/volumes"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATA_DIR="/opt/quantrs2/data"

mkdir -p "$BACKUP_DIR"

# Backup each volume
for volume in quantrs2 prometheus grafana; do
    echo "Backing up $volume volume..."
    
    tar czf "$BACKUP_DIR/${volume}_${TIMESTAMP}.tar.gz" \
        -C "$DATA_DIR" \
        "$volume"
        
    echo "Backup completed: ${volume}_${TIMESTAMP}.tar.gz"
done

# Cleanup old backups
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete
```

### 3. Automated Backup Schedule

Setup automated backups with cron:

```bash
# Add to crontab for quantrs2 user
crontab -e

# Database backup every 6 hours
0 */6 * * * /opt/quantrs2/scripts/backup-database.sh >> /opt/quantrs2/logs/backup.log 2>&1

# Volume backup daily at 2 AM
0 2 * * * /opt/quantrs2/scripts/backup-volumes.sh >> /opt/quantrs2/logs/backup.log 2>&1

# Log rotation weekly
0 0 * * 0 /opt/quantrs2/scripts/rotate-logs.sh >> /opt/quantrs2/logs/maintenance.log 2>&1
```

### 4. Disaster Recovery Procedures

**Recovery from Database Backup:**

```bash
#!/bin/bash
# /opt/quantrs2/scripts/restore-database.sh

BACKUP_FILE="$1"
TEMP_DIR="/tmp/quantrs2-restore"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup-file.dump.gpg>"
    exit 1
fi

# Decrypt backup
mkdir -p "$TEMP_DIR"
gpg --decrypt "$BACKUP_FILE" > "$TEMP_DIR/restore.dump"

# Stop application
docker-compose -f docker-compose.secure.yml stop quantrs2-base

# Restore database
docker-compose -f docker-compose.secure.yml exec -T quantrs2-db pg_restore \
    -U quantrs2_prod \
    -h localhost \
    -d quantrs2_prod \
    --clean \
    --if-exists \
    < "$TEMP_DIR/restore.dump"

# Restart application
docker-compose -f docker-compose.secure.yml start quantrs2-base

# Cleanup
rm -rf "$TEMP_DIR"

echo "Database restore completed"
```

## Scaling and Performance

### 1. Horizontal Scaling

Scale QuantRS2 for high availability:

```bash
# Scale base service
docker-compose -f docker-compose.secure.yml up --scale quantrs2-base=3 -d

# Update load balancer configuration
# Add backend servers to nginx upstream block
```

### 2. Performance Tuning

Optimize for production workloads:

```yaml
# docker-compose.override.yml for performance tuning
version: '3.8'

services:
  quantrs2-base:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 16g
        reservations:
          cpus: '2.0'
          memory: 8g
    environment:
      - PYTHONOPTIMIZE=2
      - MALLOC_TRIM_THRESHOLD_=100000
      - MALLOC_MMAP_THRESHOLD_=100000
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
      nproc:
        soft: 32768
        hard: 32768
```

### 3. Database Performance

Optimize PostgreSQL for production:

```sql
-- /opt/quantrs2/scripts/optimize-postgres.sql

-- Performance tuning
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Connection tuning
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET max_worker_processes = 8;
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET max_parallel_workers = 8;

-- Apply changes
SELECT pg_reload_conf();
```

## Maintenance and Operations

### 1. Routine Maintenance Tasks

Create maintenance scripts:

```bash
#!/bin/bash
# /opt/quantrs2/scripts/daily-maintenance.sh

set -euo pipefail

echo "Starting daily maintenance - $(date)"

# Update Docker images
docker-compose -f docker-compose.secure.yml pull

# Cleanup Docker system
docker system prune -f --volumes

# Rotate logs
docker-compose -f docker-compose.secure.yml exec quantrs2-base logrotate /etc/logrotate.conf

# Update security signatures
docker-compose -f docker-compose.secure.yml exec quantrs2-base apt-get update -q

# Check disk space
df -h /opt/quantrs2

# Verify service health
docker-compose -f docker-compose.secure.yml exec quantrs2-base python docker/healthcheck.py

echo "Daily maintenance completed - $(date)"
```

### 2. Security Updates

Automated security update process:

```bash
#!/bin/bash
# /opt/quantrs2/scripts/security-updates.sh

set -euo pipefail

echo "Starting security updates - $(date)"

# Update base system
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

# Update Docker images with security patches
docker-compose -f docker-compose.secure.yml pull

# Rebuild custom images if needed
docker-compose -f docker-compose.secure.yml build --no-cache

# Rolling restart
for service in quantrs2-db quantrs2-redis quantrs2-base; do
    echo "Restarting $service..."
    docker-compose -f docker-compose.secure.yml restart "$service"
    sleep 30
    
    # Verify health
    docker-compose -f docker-compose.secure.yml exec "$service" python /app/docker/healthcheck.py
done

echo "Security updates completed - $(date)"
```

### 3. Monitoring and Alerting

Monitor system health continuously:

```bash
#!/bin/bash
# /opt/quantrs2/scripts/health-check.sh

check_service() {
    local service=$1
    local endpoint=$2
    
    if curl -sf "$endpoint" > /dev/null; then
        echo "✅ $service is healthy"
        return 0
    else
        echo "❌ $service is unhealthy"
        return 1
    fi
}

# Check all services
check_service "QuantRS2 API" "http://localhost:8080/health"
check_service "Jupyter" "http://localhost:8888/api"
check_service "Grafana" "http://localhost:3000/api/health"
check_service "Prometheus" "http://localhost:9090/-/healthy"

# Check database connectivity
docker-compose -f docker-compose.secure.yml exec quantrs2-db pg_isready -U quantrs2_prod || exit 1

# Check Redis connectivity
docker-compose -f docker-compose.secure.yml exec quantrs2-redis redis-cli ping || exit 1

echo "All health checks passed"
```

### 4. Log Management

Configure log rotation and cleanup:

```bash
# /etc/logrotate.d/quantrs2
/opt/quantrs2/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 quantrs2 quantrs2
    postrotate
        docker-compose -f /opt/quantrs2/quantrs/py/docker-compose.secure.yml restart quantrs2-base
    endscript
}
```

This production deployment guide provides comprehensive coverage of deploying QuantRS2 with enterprise-grade features. For specific deployment scenarios or troubleshooting, refer to the [Operations Runbook](operations-runbook.md) and [Troubleshooting Guide](troubleshooting.md).