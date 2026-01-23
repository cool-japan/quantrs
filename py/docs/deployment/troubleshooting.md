# QuantRS2 Troubleshooting Guide

This guide provides solutions to common issues encountered in QuantRS2 production deployments, including diagnostic steps, resolution procedures, and prevention strategies.

## Table of Contents

1. [Service Startup Issues](#service-startup-issues)
2. [Performance Problems](#performance-problems)
3. [Database Issues](#database-issues)
4. [Authentication and Security](#authentication-and-security)
5. [Circuit Execution Problems](#circuit-execution-problems)
6. [Monitoring and Alerting](#monitoring-and-alerting)
7. [Container and Docker Issues](#container-and-docker-issues)
8. [Network and Connectivity](#network-and-connectivity)
9. [Resource Exhaustion](#resource-exhaustion)
10. [Backup and Recovery Issues](#backup-and-recovery-issues)

## Service Startup Issues

### Problem: QuantRS2 Container Won't Start

**Symptoms:**
- Container exits immediately after starting
- Error messages in logs
- Health checks failing

**Diagnostic Steps:**

```bash
# Check container status
docker-compose -f docker-compose.secure.yml ps

# Check logs for errors
docker-compose -f docker-compose.secure.yml logs quantrs2-base

# Check resource availability
df -h /opt/quantrs2
free -h

# Check configuration
docker-compose -f docker-compose.secure.yml config
```

**Common Causes and Solutions:**

1. **Configuration Error:**
   ```bash
   # Validate environment variables
   env | grep QUANTRS2
   
   # Check secrets are readable
   ls -la /opt/quantrs2/secrets/
   cat /opt/quantrs2/secrets/master_key | wc -c  # Should be > 40
   ```

2. **Port Conflicts:**
   ```bash
   # Check for port conflicts
   sudo netstat -tlnp | grep -E ":(8080|8888|5432|6379|9090|3000)"
   
   # Kill conflicting processes
   sudo fuser -k 8080/tcp
   ```

3. **Permission Issues:**
   ```bash
   # Fix ownership
   sudo chown -R quantrs2:quantrs2 /opt/quantrs2/data
   
   # Fix permissions
   sudo chmod -R 755 /opt/quantrs2/data
   sudo chmod 600 /opt/quantrs2/secrets/*
   ```

4. **Missing Dependencies:**
   ```bash
   # Check dependencies
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "import quantrs2; print('OK')"
   
   # Rebuild if needed
   docker-compose -f docker-compose.secure.yml build --no-cache quantrs2-base
   ```

### Problem: Database Connection Failed

**Symptoms:**
- "Connection refused" errors
- Timeout connecting to database
- Authentication failures

**Diagnostic Steps:**

```bash
# Check database container status
docker-compose -f docker-compose.secure.yml ps quantrs2-db

# Test database connectivity
docker-compose -f docker-compose.secure.yml exec quantrs2-db pg_isready -U quantrs2_prod

# Check database logs
docker-compose -f docker-compose.secure.yml logs quantrs2-db

# Test connection from application
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
import psycopg2
conn = psycopg2.connect(
    host='quantrs2-db',
    database='quantrs2_prod',
    user='quantrs2_prod',
    password='$(cat /opt/quantrs2/secrets/postgres_password)'
)
print('Database connection successful')
"
```

**Solutions:**

1. **Database not running:**
   ```bash
   # Start database
   docker-compose -f docker-compose.secure.yml up -d quantrs2-db
   
   # Wait for startup
   sleep 30
   ```

2. **Wrong credentials:**
   ```bash
   # Verify password
   echo $POSTGRES_PASSWORD
   cat /opt/quantrs2/secrets/postgres_password
   
   # Reset password if needed
   docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U postgres -c "ALTER USER quantrs2_prod PASSWORD 'new_password';"
   ```

3. **Network issues:**
   ```bash
   # Check network connectivity
   docker network ls
   docker network inspect quantrs2_quantrs2-network
   
   # Recreate network if needed
   docker-compose -f docker-compose.secure.yml down
   docker network prune
   docker-compose -f docker-compose.secure.yml up -d
   ```

## Performance Problems

### Problem: Slow Response Times

**Symptoms:**
- API responses taking > 5 seconds
- Timeouts in Jupyter notebooks
- High CPU or memory usage

**Diagnostic Steps:**

```bash
# Check response times
curl -w "Total time: %{time_total}s\n" -o /dev/null -s http://localhost:8080/health

# Monitor resource usage
docker stats --no-stream

# Check for bottlenecks
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.performance_manager import PerformanceManager
manager = PerformanceManager()
metrics = manager.get_current_metrics()
for metric, value in metrics.items():
    print(f'{metric}: {value}')
"
```

**Solutions:**

1. **High CPU Usage:**
   ```bash
   # Check running processes
   docker-compose -f docker-compose.secure.yml exec quantrs2-base top
   
   # Scale horizontally
   docker-compose -f docker-compose.secure.yml up --scale quantrs2-base=3 -d
   
   # Optimize CPU limits
   # Edit docker-compose.secure.yml to increase CPU limits
   ```

2. **Memory Issues:**
   ```bash
   # Check memory usage
   docker-compose -f docker-compose.secure.yml exec quantrs2-base free -h
   
   # Clear caches
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
   from quantrs2.circuit_optimization_cache import CircuitOptimizationCache
   cache = CircuitOptimizationCache()
   cache.clear_expired_entries()
   "
   
   # Increase memory limits
   # Edit docker-compose.secure.yml memory limits
   ```

3. **Database Performance:**
   ```bash
   # Check slow queries
   docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC 
   LIMIT 10;"
   
   # Optimize database
   docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
   ANALYZE;
   VACUUM;
   "
   ```

### Problem: Circuit Execution Timeouts

**Symptoms:**
- Circuit executions timing out
- "Execution timeout" errors
- Jobs stuck in pending state

**Diagnostic Steps:**

```bash
# Check active jobs
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.resource_management import ResourceManager
manager = ResourceManager()
stats = manager.get_resource_usage()
print(f'Active jobs: {stats[\"active_jobs\"]}/{stats[\"max_concurrent_jobs\"]}')
"

# Check resource limits
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.config_management import ConfigManager
config = ConfigManager()
limits = config.get_performance_config()
print(f'Max qubits: {limits[\"max_circuit_qubits\"]}')
print(f'Memory limit: {limits[\"simulation_memory_limit\"]}')
"
```

**Solutions:**

1. **Increase Timeouts:**
   ```bash
   # Update configuration
   # Edit /opt/quantrs2/config/production.local.yaml
   # quantum_backends:
   #   simulation:
   #     timeout_seconds: 600  # Increase from 300
   
   # Restart service
   docker-compose -f docker-compose.secure.yml restart quantrs2-base
   ```

2. **Resource Constraints:**
   ```bash
   # Check resource usage
   docker stats quantrs2-base
   
   # Increase memory limits
   # Edit docker-compose.secure.yml:
   # deploy:
   #   resources:
   #     limits:
   #       memory: 16g  # Increase from 8g
   ```

3. **Queue Management:**
   ```bash
   # Clear stuck jobs
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
   from quantrs2.resource_management import ResourceManager
   manager = ResourceManager()
   manager.clear_stuck_jobs()
   "
   ```

## Database Issues

### Problem: Database Connection Pool Exhausted

**Symptoms:**
- "Connection pool exhausted" errors
- Long waits for database connections
- High number of idle connections

**Diagnostic Steps:**

```bash
# Check connection pool status
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.connection_pooling import ConnectionPoolManager
manager = ConnectionPoolManager()
stats = manager.get_pool_statistics()
for pool, stats in stats.items():
    print(f'{pool}: {stats[\"active_connections\"]}/{stats[\"pool_size\"]} active')
"

# Check database connections
docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
SELECT state, count(*) 
FROM pg_stat_activity 
GROUP BY state;"
```

**Solutions:**

1. **Increase Pool Size:**
   ```bash
   # Edit configuration
   # /opt/quantrs2/config/production.local.yaml:
   # database:
   #   pool_size: 30  # Increase from 20
   #   max_overflow: 50  # Increase from 30
   
   # Restart application
   docker-compose -f docker-compose.secure.yml restart quantrs2-base
   ```

2. **Close Idle Connections:**
   ```bash
   # Kill idle connections
   docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
   SELECT pg_terminate_backend(pid) 
   FROM pg_stat_activity 
   WHERE state = 'idle' 
   AND state_change < now() - interval '1 hour';"
   ```

3. **Connection Leak Detection:**
   ```bash
   # Enable connection tracking
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
   from quantrs2.connection_pooling import ConnectionPoolManager
   manager = ConnectionPoolManager()
   manager.enable_connection_tracking()
   leaks = manager.detect_connection_leaks()
   for leak in leaks:
       print(f'Leak detected: {leak}')
   "
   ```

### Problem: Database Deadlocks

**Symptoms:**
- "Deadlock detected" errors
- Transactions hanging
- Inconsistent data states

**Diagnostic Steps:**

```bash
# Check for deadlocks
docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
SELECT * FROM pg_stat_database_conflicts WHERE datname = 'quantrs2_prod';"

# Check long-running transactions
docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';"
```

**Solutions:**

1. **Kill Long Transactions:**
   ```bash
   # Terminate long-running queries
   docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
   SELECT pg_terminate_backend(pid) 
   FROM pg_stat_activity 
   WHERE (now() - pg_stat_activity.query_start) > interval '10 minutes';"
   ```

2. **Configure Deadlock Detection:**
   ```bash
   # Reduce deadlock timeout
   docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
   ALTER SYSTEM SET deadlock_timeout = '1s';
   SELECT pg_reload_conf();"
   ```

## Authentication and Security

### Problem: Authentication Failures

**Symptoms:**
- Users cannot log in
- "Invalid credentials" errors
- JWT token validation failures

**Diagnostic Steps:**

```bash
# Check authentication service
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.security.auth_manager import AuthManager
auth = AuthManager()
print(f'Auth service status: {auth.is_healthy()}')
"

# Check JWT configuration
echo $JWT_SECRET_KEY | wc -c  # Should be > 40

# Check failed login attempts
grep "authentication.*failed" /opt/quantrs2/logs/quantrs2.log | tail -10
```

**Solutions:**

1. **Reset JWT Secret:**
   ```bash
   # Generate new JWT secret
   openssl rand -base64 64 > /opt/quantrs2/secrets/jwt_secret
   
   # Update environment
   export JWT_SECRET_KEY=$(cat /opt/quantrs2/secrets/jwt_secret)
   
   # Restart service
   docker-compose -f docker-compose.secure.yml restart quantrs2-base
   ```

2. **Clear Failed Attempts:**
   ```bash
   # Reset failed login counters
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
   from quantrs2.security.auth_manager import AuthManager
   auth = AuthManager()
   auth.reset_failed_attempts()
   "
   ```

3. **Check User Status:**
   ```bash
   # List user status
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
   from quantrs2.security.auth_manager import AuthManager
   auth = AuthManager()
   users = auth.list_users()
   for user in users:
       print(f'{user.username}: active={user.active}, locked={user.locked}')
   "
   ```

### Problem: SSL/TLS Certificate Issues

**Symptoms:**
- "Certificate expired" warnings
- SSL handshake failures
- Browser security warnings

**Diagnostic Steps:**

```bash
# Check certificate expiration
openssl x509 -in /opt/quantrs2/certs/cert.pem -noout -dates

# Test SSL connection
openssl s_client -connect yourdomain.com:443 -servername yourdomain.com

# Check certificate chain
openssl verify -CAfile /opt/quantrs2/certs/ca.pem /opt/quantrs2/certs/cert.pem
```

**Solutions:**

1. **Renew Let's Encrypt Certificate:**
   ```bash
   # Manual renewal
   sudo certbot renew --dry-run
   sudo certbot renew
   
   # Copy new certificates
   sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem /opt/quantrs2/certs/cert.pem
   sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem /opt/quantrs2/certs/key.pem
   
   # Restart services
   docker-compose -f docker-compose.secure.yml restart reverse-proxy
   ```

2. **Update Certificate Configuration:**
   ```bash
   # Verify certificate format
   openssl x509 -in /opt/quantrs2/certs/cert.pem -text -noout
   
   # Fix permissions
   sudo chmod 644 /opt/quantrs2/certs/cert.pem
   sudo chmod 600 /opt/quantrs2/certs/key.pem
   ```

## Circuit Execution Problems

### Problem: Invalid Circuit Errors

**Symptoms:**
- "Invalid circuit definition" errors
- Quantum gate validation failures
- QASM parsing errors

**Diagnostic Steps:**

```bash
# Test basic circuit execution
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2 import QuantumCircuit
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
try:
    result = circuit.execute()
    print('Basic circuit execution successful')
except Exception as e:
    print(f'Circuit execution failed: {e}')
"

# Check circuit validation
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.security.quantum_input_validator import QuantumInputValidator
validator = QuantumInputValidator()
print(f'Validator status: {validator.is_enabled()}')
"
```

**Solutions:**

1. **Circuit Validation Issues:**
   ```bash
   # Disable strict validation temporarily
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
   from quantrs2.security.quantum_input_validator import QuantumInputValidator
   validator = QuantumInputValidator()
   validator.set_strict_mode(False)
   "
   ```

2. **Gate Library Issues:**
   ```bash
   # Check available gates
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
   from quantrs2.gates import get_available_gates
   gates = get_available_gates()
   print(f'Available gates: {list(gates.keys())}')
   "
   
   # Rebuild gate library
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
   from quantrs2.gates import rebuild_gate_library
   rebuild_gate_library()
   "
   ```

### Problem: Backend Connection Failures

**Symptoms:**
- Cannot connect to quantum backends
- API authentication failures with cloud providers
- Backend timeout errors

**Diagnostic Steps:**

```bash
# Test backend connectivity
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.hardware_backends import get_available_backends
backends = get_available_backends()
for backend in backends:
    print(f'{backend.name}: {backend.status}')
"

# Check API credentials
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
import os
print('IBM Token:', 'SET' if os.getenv('IBM_QUANTUM_TOKEN') else 'NOT SET')
print('Google Token:', 'SET' if os.getenv('GOOGLE_QUANTUM_TOKEN') else 'NOT SET')
"
```

**Solutions:**

1. **Update API Credentials:**
   ```bash
   # Update secrets
   echo "your_new_token" > /opt/quantrs2/secrets/ibm_quantum_token
   
   # Restart service
   docker-compose -f docker-compose.secure.yml restart quantrs2-base
   ```

2. **Backend Configuration:**
   ```bash
   # Reset backend configuration
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
   from quantrs2.hardware_backends import BackendManager
   manager = BackendManager()
   manager.reset_backend_connections()
   manager.refresh_backend_status()
   "
   ```

## Monitoring and Alerting

### Problem: Alerts Not Firing

**Symptoms:**
- Expected alerts not triggering
- Notification channels not receiving alerts
- Alert rules not evaluating

**Diagnostic Steps:**

```bash
# Check alert manager status
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.monitoring_alerting import MonitoringSystem
system = MonitoringSystem()
stats = system.alert_manager.get_alert_statistics()
print(f'Alert rules: {stats[\"total_rules\"]}')
print(f'Active alerts: {stats[\"active_alerts\"]}')
"

# Check notification channels
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.monitoring_alerting import MonitoringSystem
system = MonitoringSystem()
stats = system.notification_manager.get_notification_stats()
print(f'Notification channels: {stats[\"total_channels\"]}')
print(f'Enabled channels: {stats[\"enabled_channels\"]}')
"
```

**Solutions:**

1. **Fix Alert Rule Configuration:**
   ```bash
   # Check alert rule syntax
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
   from quantrs2.monitoring_alerting import MonitoringSystem
   system = MonitoringSystem()
   for rule_id, rule in system.alert_manager.alert_rules.items():
       if not rule.enabled:
           print(f'Rule {rule.name} is disabled')
   "
   
   # Enable disabled rules
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
   from quantrs2.monitoring_alerting import MonitoringSystem
   system = MonitoringSystem()
   for rule in system.alert_manager.alert_rules.values():
       rule.enabled = True
   "
   ```

2. **Test Notification Channels:**
   ```bash
   # Test Slack webhook
   curl -X POST "$SLACK_WEBHOOK_URL" -d '{"text":"Test alert from QuantRS2"}'
   
   # Test email configuration
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
   import smtplib
   from email.mime.text import MimeText
   
   msg = MimeText('Test email')
   msg['Subject'] = 'QuantRS2 Test'
   msg['From'] = 'alerts@yourdomain.com'
   msg['To'] = 'admin@yourdomain.com'
   
   with smtplib.SMTP('smtp.yourdomain.com', 587) as server:
       server.starttls()
       server.login('username', 'password')
       server.send_message(msg)
   print('Test email sent successfully')
   "
   ```

### Problem: Missing Metrics

**Symptoms:**
- Grafana dashboards showing no data
- Prometheus targets down
- Metrics collection errors

**Diagnostic Steps:**

```bash
# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# Check metrics collection
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.monitoring_alerting import MonitoringSystem
system = MonitoringSystem()
recent_metrics = system.metrics_collector.get_recent_metrics('system.cpu_percent', 300)
print(f'Recent CPU metrics: {len(recent_metrics)} points')
"

# Test metrics endpoint
curl http://localhost:9090/metrics | head -20
```

**Solutions:**

1. **Restart Metrics Collection:**
   ```bash
   # Restart monitoring services
   docker-compose -f docker-compose.secure.yml restart quantrs2-prometheus quantrs2-grafana
   
   # Clear metrics cache
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
   from quantrs2.monitoring_alerting import MonitoringSystem
   system = MonitoringSystem()
   system.metrics_collector._persist_buffered_metrics()
   "
   ```

2. **Fix Grafana Data Sources:**
   ```bash
   # Test Grafana API
   curl -u admin:$GRAFANA_ADMIN_PASSWORD http://localhost:3000/api/datasources
   
   # Recreate Prometheus data source
   curl -X POST -u admin:$GRAFANA_ADMIN_PASSWORD \
     -H "Content-Type: application/json" \
     http://localhost:3000/api/datasources \
     -d '{
       "name": "Prometheus",
       "type": "prometheus",
       "url": "http://quantrs2-prometheus:9090",
       "access": "proxy",
       "isDefault": true
     }'
   ```

## Container and Docker Issues

### Problem: Out of Disk Space

**Symptoms:**
- "No space left on device" errors
- Container creation failures
- Application crashes

**Diagnostic Steps:**

```bash
# Check disk usage
df -h
docker system df

# Check large files
du -h /opt/quantrs2 | sort -rh | head -10

# Check Docker volumes
docker volume ls
docker system df -v
```

**Solutions:**

1. **Clean Up Docker:**
   ```bash
   # Remove unused containers, networks, images
   docker system prune -a -f
   
   # Remove unused volumes
   docker volume prune -f
   
   # Remove old logs
   docker-compose -f docker-compose.secure.yml exec quantrs2-base logrotate /etc/logrotate.conf
   ```

2. **Clean Application Data:**
   ```bash
   # Clear old backups
   find /opt/quantrs2/backups -name "*.dump.gpg" -mtime +30 -delete
   
   # Clear temporary files
   rm -rf /tmp/quantrs2-*
   
   # Clear application caches
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
   from quantrs2.circuit_optimization_cache import CircuitOptimizationCache
   cache = CircuitOptimizationCache()
   cache.clear_expired_entries()
   "
   ```

### Problem: Container Memory Issues

**Symptoms:**
- Container killed by OOM killer
- "Memory allocation failed" errors
- Slow performance

**Diagnostic Steps:**

```bash
# Check memory usage
free -h
docker stats --no-stream

# Check container limits
docker inspect quantrs2-base | grep -A 10 -B 10 Memory

# Check for memory leaks
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
import psutil
process = psutil.Process()
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
print(f'Memory percent: {process.memory_percent():.2f}%')
"
```

**Solutions:**

1. **Increase Memory Limits:**
   ```bash
   # Edit docker-compose.secure.yml
   # deploy:
   #   resources:
   #     limits:
   #       memory: 16g  # Increase from current limit
   
   # Apply changes
   docker-compose -f docker-compose.secure.yml up -d --force-recreate quantrs2-base
   ```

2. **Optimize Memory Usage:**
   ```bash
   # Clear Python caches
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
   import gc
   gc.collect()
   "
   
   # Restart service to clear memory
   docker-compose -f docker-compose.secure.yml restart quantrs2-base
   ```

## Network and Connectivity

### Problem: Service Discovery Issues

**Symptoms:**
- Services cannot reach each other
- DNS resolution failures
- Connection timeouts between containers

**Diagnostic Steps:**

```bash
# Check network configuration
docker network ls
docker network inspect quantrs2_quantrs2-network

# Test connectivity between containers
docker-compose -f docker-compose.secure.yml exec quantrs2-base ping quantrs2-db
docker-compose -f docker-compose.secure.yml exec quantrs2-base nslookup quantrs2-redis

# Check port accessibility
docker-compose -f docker-compose.secure.yml exec quantrs2-base telnet quantrs2-db 5432
```

**Solutions:**

1. **Recreate Network:**
   ```bash
   # Stop all services
   docker-compose -f docker-compose.secure.yml down
   
   # Remove network
   docker network rm quantrs2_quantrs2-network
   
   # Restart services (will recreate network)
   docker-compose -f docker-compose.secure.yml up -d
   ```

2. **Fix DNS Issues:**
   ```bash
   # Update /etc/hosts if needed
   docker-compose -f docker-compose.secure.yml exec quantrs2-base bash -c "
   echo '172.20.0.2 quantrs2-db' >> /etc/hosts
   echo '172.20.0.3 quantrs2-redis' >> /etc/hosts
   "
   ```

### Problem: External API Connectivity

**Symptoms:**
- Cannot reach external quantum providers
- HTTP timeout errors
- SSL/TLS handshake failures

**Diagnostic Steps:**

```bash
# Test external connectivity
docker-compose -f docker-compose.secure.yml exec quantrs2-base curl -I https://quantum-computing.ibm.com
docker-compose -f docker-compose.secure.yml exec quantrs2-base curl -I https://quantumai.google

# Check DNS resolution
docker-compose -f docker-compose.secure.yml exec quantrs2-base nslookup quantum-computing.ibm.com

# Check firewall rules
sudo iptables -L OUTPUT
```

**Solutions:**

1. **Configure Proxy (if needed):**
   ```bash
   # Add proxy configuration
   # Edit docker-compose.secure.yml:
   # environment:
   #   - HTTP_PROXY=http://proxy.company.com:8080
   #   - HTTPS_PROXY=http://proxy.company.com:8080
   #   - NO_PROXY=localhost,127.0.0.1,quantrs2-db,quantrs2-redis
   ```

2. **Update CA Certificates:**
   ```bash
   # Update CA certificates in container
   docker-compose -f docker-compose.secure.yml exec quantrs2-base update-ca-certificates
   
   # Add custom CA if needed
   docker cp your-ca.crt quantrs2-base:/usr/local/share/ca-certificates/
   docker-compose -f docker-compose.secure.yml exec quantrs2-base update-ca-certificates
   ```

## Resource Exhaustion

### Problem: CPU Throttling

**Symptoms:**
- High CPU wait times
- Slow response times during peak usage
- CPU throttling messages in logs

**Diagnostic Steps:**

```bash
# Check CPU usage
top -p $(docker inspect -f '{{.State.Pid}}' quantrs2-base)
docker stats quantrs2-base

# Check for throttling
docker-compose -f docker-compose.secure.yml exec quantrs2-base cat /sys/fs/cgroup/cpu/cpu.stat

# Check CPU limits
docker inspect quantrs2-base | grep -A 5 -B 5 CpuQuota
```

**Solutions:**

1. **Increase CPU Limits:**
   ```bash
   # Edit docker-compose.secure.yml
   # deploy:
   #   resources:
   #     limits:
   #       cpus: '4.0'  # Increase from current limit
   
   # Apply changes
   docker-compose -f docker-compose.secure.yml up -d --force-recreate quantrs2-base
   ```

2. **Scale Horizontally:**
   ```bash
   # Scale to multiple instances
   docker-compose -f docker-compose.secure.yml up --scale quantrs2-base=3 -d
   
   # Configure load balancer to distribute traffic
   ```

### Problem: File Descriptor Exhaustion

**Symptoms:**
- "Too many open files" errors
- Connection failures
- Socket creation errors

**Diagnostic Steps:**

```bash
# Check file descriptor usage
docker-compose -f docker-compose.secure.yml exec quantrs2-base lsof | wc -l
docker-compose -f docker-compose.secure.yml exec quantrs2-base ulimit -n

# Check for file descriptor leaks
docker-compose -f docker-compose.secure.yml exec quantrs2-base lsof | grep quantrs2 | head -20
```

**Solutions:**

1. **Increase Limits:**
   ```bash
   # Edit docker-compose.secure.yml
   # ulimits:
   #   nofile:
   #     soft: 65536
   #     hard: 65536
   
   # Apply changes
   docker-compose -f docker-compose.secure.yml up -d --force-recreate quantrs2-base
   ```

2. **Fix File Descriptor Leaks:**
   ```bash
   # Restart service to close leaked descriptors
   docker-compose -f docker-compose.secure.yml restart quantrs2-base
   
   # Check for application-level leaks
   docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
   import resource
   soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
   print(f'File descriptor limit: {soft}/{hard}')
   "
   ```

## Backup and Recovery Issues

### Problem: Backup Failures

**Symptoms:**
- Backup scripts failing
- Incomplete backup files
- Corruption in backup data

**Diagnostic Steps:**

```bash
# Check backup status
ls -la /opt/quantrs2/backups/database/
ls -la /opt/quantrs2/backups/volumes/

# Check backup logs
tail -n 50 /opt/quantrs2/logs/backup.log

# Test backup integrity
latest_backup=$(ls -t /opt/quantrs2/backups/database/*.dump.gpg | head -1)
gpg --decrypt "$latest_backup" > /dev/null && echo "Backup file is valid" || echo "Backup file is corrupted"
```

**Solutions:**

1. **Fix Backup Permissions:**
   ```bash
   # Fix ownership
   sudo chown -R quantrs2:quantrs2 /opt/quantrs2/backups
   
   # Fix permissions
   sudo chmod -R 755 /opt/quantrs2/backups
   ```

2. **Recreate Backup Directory:**
   ```bash
   # Recreate backup structure
   sudo mkdir -p /opt/quantrs2/backups/{database,volumes}
   sudo chown -R quantrs2:quantrs2 /opt/quantrs2/backups
   ```

3. **Test Backup Process:**
   ```bash
   # Run backup manually
   /opt/quantrs2/scripts/backup-database.sh
   /opt/quantrs2/scripts/backup-volumes.sh
   
   # Check results
   ls -la /opt/quantrs2/backups/database/
   ```

### Problem: Recovery Failures

**Symptoms:**
- Cannot restore from backup
- Database restore errors
- Data corruption after restore

**Diagnostic Steps:**

```bash
# Test backup file
latest_backup=$(ls -t /opt/quantrs2/backups/database/*.dump.gpg | head -1)
gpg --decrypt "$latest_backup" | head -10

# Check database status
docker-compose -f docker-compose.secure.yml exec quantrs2-db pg_isready -U quantrs2_prod

# Check available space
df -h /opt/quantrs2
```

**Solutions:**

1. **Manual Recovery Process:**
   ```bash
   # Stop application
   docker-compose -f docker-compose.secure.yml stop quantrs2-base
   
   # Create recovery database
   docker-compose -f docker-compose.secure.yml exec quantrs2-db createdb -U postgres quantrs2_recovery
   
   # Restore to recovery database
   latest_backup=$(ls -t /opt/quantrs2/backups/database/*.dump.gpg | head -1)
   gpg --decrypt "$latest_backup" | \
   docker-compose -f docker-compose.secure.yml exec -T quantrs2-db pg_restore \
     -U postgres -d quantrs2_recovery --clean --if-exists
   
   # Verify recovery
   docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U postgres -d quantrs2_recovery -c "\dt"
   ```

2. **Point-in-Time Recovery:**
   ```bash
   # Find backup from specific time
   backup_date="2026-01-15"
   backup_file=$(ls /opt/quantrs2/backups/database/quantrs2_${backup_date}*.dump.gpg)
   
   # Restore specific backup
   gpg --decrypt "$backup_file" | \
   docker-compose -f docker-compose.secure.yml exec -T quantrs2-db pg_restore \
     -U postgres -d quantrs2_prod --clean --if-exists
   ```

This troubleshooting guide covers the most common issues encountered in production QuantRS2 deployments. For issues not covered here, check the system logs, enable debug logging, and consider consulting the [Operations Runbook](operations-runbook.md) for additional diagnostic procedures.