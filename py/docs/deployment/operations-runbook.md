# QuantRS2 Operations Runbook

This runbook provides day-to-day operational procedures for managing QuantRS2 in production environments. It covers routine tasks, emergency procedures, and troubleshooting workflows.

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Monitoring and Alerting](#monitoring-and-alerting)
3. [Incident Response](#incident-response)
4. [Performance Management](#performance-management)
5. [Security Operations](#security-operations)
6. [Backup and Recovery](#backup-and-recovery)
7. [Capacity Planning](#capacity-planning)
8. [Change Management](#change-management)

## Daily Operations

### Morning Health Check

Perform these checks at the start of each business day:

```bash
#!/bin/bash
# /opt/quantrs2/scripts/morning-check.sh

echo "=== QuantRS2 Morning Health Check - $(date) ==="

# 1. Check all services are running
echo "📋 Checking service status..."
docker-compose -f docker-compose.secure.yml ps

# 2. Verify service health
echo "🏥 Checking service health..."
docker-compose -f docker-compose.secure.yml exec quantrs2-base python docker/healthcheck.py --verbose

# 3. Check disk space
echo "💾 Checking disk space..."
df -h /opt/quantrs2

# 4. Check memory usage
echo "🧠 Checking memory usage..."
free -h

# 5. Check database connections
echo "🗄️ Checking database..."
docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "SELECT count(*) as active_connections FROM pg_stat_activity WHERE state = 'active';"

# 6. Check Redis status
echo "🔴 Checking Redis..."
docker-compose -f docker-compose.secure.yml exec quantrs2-redis redis-cli -a "${REDIS_PASSWORD}" info replication

# 7. Check recent errors
echo "🚨 Checking recent errors..."
docker-compose -f docker-compose.secure.yml logs --since="24h" quantrs2-base | grep -i error | tail -10

# 8. Check monitoring alerts
echo "📊 Checking active alerts..."
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.state=="firing") | {alertname: .labels.alertname, severity: .labels.severity}'

# 9. Check backup status
echo "💽 Checking last backup..."
ls -la /opt/quantrs2/backups/database/ | tail -5

echo "✅ Morning health check completed"
```

### Resource Monitoring

Monitor key system resources throughout the day:

```bash
#!/bin/bash
# /opt/quantrs2/scripts/resource-monitor.sh

# CPU and Memory usage
echo "=== Resource Usage - $(date) ==="
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1

echo "Memory Usage:"
free | grep Mem | awk '{printf "%.2f%%\n", $3/$2 * 100.0}'

echo "Docker Container Resources:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# Database performance
echo "Database Performance:"
docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE schemaname = 'public' 
ORDER BY n_distinct DESC 
LIMIT 10;"

# Active connections
echo "Active Database Connections:"
docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
SELECT state, count(*) 
FROM pg_stat_activity 
WHERE state IS NOT NULL 
GROUP BY state;"
```

### Log Analysis

Analyze logs for patterns and issues:

```bash
#!/bin/bash
# /opt/quantrs2/scripts/log-analysis.sh

LOG_DIR="/opt/quantrs2/logs"
ANALYSIS_DATE=$(date +%Y-%m-%d)

echo "=== Log Analysis for $ANALYSIS_DATE ==="

# Error summary
echo "Error Summary (last 24 hours):"
find "$LOG_DIR" -name "*.log" -mtime -1 -exec grep -h "ERROR\|CRITICAL" {} \; | \
sort | uniq -c | sort -rn | head -10

# Most active users/IPs
echo "Most Active Sources:"
find "$LOG_DIR" -name "*.log" -mtime -1 -exec grep -h "INFO" {} \; | \
grep -o '\b[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\b' | \
sort | uniq -c | sort -rn | head -10

# Circuit execution statistics
echo "Circuit Execution Statistics:"
find "$LOG_DIR" -name "*.log" -mtime -1 -exec grep -h "circuit.execution" {} \; | \
grep -o '"qubits":[0-9]*' | cut -d':' -f2 | \
awk '{sum+=$1; count++} END {if(count>0) printf "Avg qubits: %.2f, Total executions: %d\n", sum/count, count}'

# Performance metrics
echo "Performance Metrics:"
find "$LOG_DIR" -name "*.log" -mtime -1 -exec grep -h "duration_ms" {} \; | \
grep -o '"duration_ms":[0-9.]*' | cut -d':' -f2 | \
awk '{sum+=$1; count++; if($1>max) max=$1} END {if(count>0) printf "Avg duration: %.2fms, Max: %.2fms, Total: %d\n", sum/count, max, count}'
```

## Monitoring and Alerting

### Alert Management

**View Active Alerts:**
```bash
# Check Prometheus alerts
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.state=="firing")'

# Check QuantRS2 monitoring system alerts
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.monitoring_alerting import MonitoringSystem
system = MonitoringSystem()
alerts = system.alert_manager.get_active_alerts()
for alert in alerts:
    print(f'{alert.severity.value.upper()}: {alert.rule_name} - {alert.message}')
"
```

**Acknowledge Alerts:**
```bash
# Acknowledge specific alert
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.monitoring_alerting import MonitoringSystem
system = MonitoringSystem()
system.alert_manager.acknowledge_alert('ALERT_ID', 'operator_name')
"
```

**Resolve Alerts:**
```bash
# Resolve alert when issue is fixed
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.monitoring_alerting import MonitoringSystem
system = MonitoringSystem()
system.alert_manager.resolve_alert('ALERT_ID', 'operator_name')
"
```

### Custom Alert Rules

Add custom alert rules for specific scenarios:

```bash
#!/bin/bash
# /opt/quantrs2/scripts/add-alert-rule.sh

cat << 'EOF' | docker-compose -f docker-compose.secure.yml exec -T quantrs2-base python -
from quantrs2.monitoring_alerting import MonitoringSystem, AlertRule, AlertSeverity, MetricType

system = MonitoringSystem()

# High quantum job failure rate
rule = AlertRule(
    id="high_quantum_job_failures",
    name="High Quantum Job Failure Rate",
    description="Quantum job failure rate is above 10%",
    metric_name="app.quantum.job_failures",
    metric_type=MetricType.APPLICATION,
    severity=AlertSeverity.HIGH,
    threshold_value=0.1,
    comparison=">",
    evaluation_window=600,
    data_points_required=5
)

system.alert_manager.add_rule(rule)
print(f"Added alert rule: {rule.name}")
EOF
```

### Dashboard Management

**Access Grafana Dashboard:**
```bash
# Get Grafana admin password
echo "Grafana URL: http://localhost:3000"
echo "Username: admin"
echo "Password: $(cat /opt/quantrs2/secrets/postgres_password)"
```

**Export Dashboard Configuration:**
```bash
# Export current dashboard
curl -H "Authorization: Bearer YOUR_API_KEY" \
  "http://localhost:3000/api/dashboards/uid/DASHBOARD_UID" \
  > dashboard-backup.json
```

## Incident Response

### Severity Classification

**Critical (P0):**
- Complete service outage
- Data corruption or loss
- Security breach
- Response time: Immediate

**High (P1):**
- Partial service degradation affecting multiple users
- Performance issues (>50% slower than baseline)
- Failed backups
- Response time: Within 30 minutes

**Medium (P2):**
- Single user or limited functionality issues
- Minor performance degradation
- Non-critical alerts
- Response time: Within 2 hours

**Low (P3):**
- Cosmetic issues
- Enhancement requests
- Documentation updates
- Response time: Within 24 hours

### Incident Response Procedures

**1. Service Outage Response:**

```bash
#!/bin/bash
# /opt/quantrs2/scripts/outage-response.sh

echo "=== INCIDENT RESPONSE: Service Outage ==="
echo "Time: $(date)"
echo "Operator: $(whoami)"

# Step 1: Assess situation
echo "1. Checking service status..."
docker-compose -f docker-compose.secure.yml ps

# Step 2: Check logs for errors
echo "2. Checking recent errors..."
docker-compose -f docker-compose.secure.yml logs --tail=50 quantrs2-base | grep -E "(ERROR|CRITICAL|FATAL)"

# Step 3: Check resource availability
echo "3. Checking system resources..."
df -h /opt/quantrs2
free -m

# Step 4: Check database connectivity
echo "4. Checking database..."
if docker-compose -f docker-compose.secure.yml exec quantrs2-db pg_isready -U quantrs2_prod; then
    echo "✅ Database is accessible"
else
    echo "❌ Database is not accessible"
fi

# Step 5: Attempt service restart
echo "5. Attempting service restart..."
docker-compose -f docker-compose.secure.yml restart quantrs2-base

# Wait and verify
sleep 30
if curl -sf http://localhost:8080/health > /dev/null; then
    echo "✅ Service restored"
else
    echo "❌ Service still down - escalating"
    # Send alert
    curl -X POST "$ALERT_WEBHOOK_URL" -d '{"text":"CRITICAL: QuantRS2 service restart failed"}'
fi
```

**2. Performance Degradation Response:**

```bash
#!/bin/bash
# /opt/quantrs2/scripts/performance-response.sh

echo "=== PERFORMANCE DEGRADATION RESPONSE ==="

# Check current performance metrics
echo "1. Current response times:"
curl -w "@/opt/quantrs2/scripts/curl-format.txt" -s -o /dev/null http://localhost:8080/health

# Check resource usage
echo "2. Resource utilization:"
docker stats --no-stream

# Check database performance
echo "3. Database performance:"
docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;"

# Check for long-running queries
echo "4. Long-running queries:"
docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';"

# Check Redis performance
echo "5. Redis performance:"
docker-compose -f docker-compose.secure.yml exec quantrs2-redis redis-cli -a "${REDIS_PASSWORD}" info stats
```

**3. Security Incident Response:**

```bash
#!/bin/bash
# /opt/quantrs2/scripts/security-response.sh

echo "=== SECURITY INCIDENT RESPONSE ==="
echo "⚠️  SECURITY ALERT DETECTED ⚠️"

# Step 1: Immediate containment
echo "1. Checking for suspicious activity..."
grep -E "(SECURITY|authentication|authorization)" /opt/quantrs2/logs/quantrs2.log | tail -20

# Step 2: Check failed login attempts
echo "2. Failed login attempts:"
grep "authentication.*failed" /opt/quantrs2/logs/quantrs2.log | tail -10

# Step 3: Check active sessions
echo "3. Active sessions:"
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.security.auth_manager import AuthManager
auth = AuthManager()
sessions = auth.get_active_sessions()
for session in sessions:
    print(f'User: {session.user_id}, IP: {session.ip_address}, Started: {session.created_at}')
"

# Step 4: If breach confirmed, lock down
if [ "$1" = "--lockdown" ]; then
    echo "🔒 INITIATING LOCKDOWN PROCEDURES"
    
    # Disable new logins
    docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
    from quantrs2.security.auth_manager import AuthManager
    auth = AuthManager()
    auth.disable_new_logins()
    "
    
    # Terminate all sessions
    docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
    from quantrs2.security.auth_manager import AuthManager
    auth = AuthManager()
    auth.terminate_all_sessions()
    "
    
    # Block suspicious IPs
    # Implementation depends on firewall setup
    
    echo "🚨 Lockdown complete - manual review required"
fi
```

### Communication Templates

**Incident Notification:**
```
INCIDENT ALERT - QuantRS2 Production

Severity: [CRITICAL/HIGH/MEDIUM/LOW]
Service: QuantRS2 Production Environment
Impact: [Description of user impact]
Started: [Timestamp]
Status: Investigating/Identified/Monitoring/Resolved

Current Actions:
- [Action 1]
- [Action 2]

Next Update: [Time]
Incident Commander: [Name]
```

**Resolution Notification:**
```
INCIDENT RESOLVED - QuantRS2 Production

Incident: [Brief description]
Duration: [Start time] - [End time] ([Total duration])
Root Cause: [Brief explanation]

Resolution:
- [Action taken 1]
- [Action taken 2]

Lessons Learned:
- [Lesson 1]
- [Lesson 2]

Follow-up Actions:
- [Action 1] - [Owner] - [Due date]
- [Action 2] - [Owner] - [Due date]
```

## Performance Management

### Baseline Performance Metrics

Establish and monitor baseline performance:

```bash
#!/bin/bash
# /opt/quantrs2/scripts/performance-baseline.sh

echo "=== Performance Baseline - $(date) ==="

# API Response Times
echo "API Response Times:"
for endpoint in /health /api/circuits /api/execute; do
    response_time=$(curl -w "%{time_total}" -s -o /dev/null "http://localhost:8080$endpoint")
    echo "$endpoint: ${response_time}s"
done

# Circuit Execution Performance
echo "Circuit Execution Performance:"
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
import time
from quantrs2 import QuantumCircuit

# Small circuit test
start = time.time()
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
result = circuit.execute()
small_time = time.time() - start
print(f'Small circuit (2 qubits): {small_time:.3f}s')

# Medium circuit test
start = time.time()
circuit = QuantumCircuit(10)
for i in range(10):
    circuit.h(i)
for i in range(9):
    circuit.cx(i, i+1)
result = circuit.execute()
medium_time = time.time() - start
print(f'Medium circuit (10 qubits): {medium_time:.3f}s')
"

# Memory Usage
echo "Memory Usage:"
docker stats --no-stream --format "{{.Container}}: {{.MemUsage}}"

# Database Performance
echo "Database Performance:"
docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
SELECT 
    'Query Performance' as metric,
    round(avg(mean_time)::numeric, 2) as avg_time_ms,
    round(max(mean_time)::numeric, 2) as max_time_ms
FROM pg_stat_statements;
"
```

### Performance Optimization

Monitor and optimize key performance indicators:

```bash
#!/bin/bash
# /opt/quantrs2/scripts/optimize-performance.sh

echo "=== Performance Optimization ==="

# 1. Clear caches if memory usage is high
memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
if [ "$memory_usage" -gt 85 ]; then
    echo "High memory usage detected ($memory_usage%), clearing caches..."
    
    # Clear filesystem cache
    echo 3 > /proc/sys/vm/drop_caches
    
    # Clear Redis cache if safe
    docker-compose -f docker-compose.secure.yml exec quantrs2-redis redis-cli -a "${REDIS_PASSWORD}" FLUSHDB
    
    # Clear application cache
    docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
    from quantrs2.circuit_optimization_cache import CircuitOptimizationCache
    cache = CircuitOptimizationCache()
    cache.clear_expired_entries()
    "
fi

# 2. Optimize database if needed
docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
-- Update table statistics
ANALYZE;

-- Reindex if fragmentation is high
REINDEX DATABASE quantrs2_prod;
"

# 3. Check connection pool utilization
echo "Connection pool status:"
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.connection_pooling import ConnectionPoolManager
manager = ConnectionPoolManager()
stats = manager.get_pool_statistics()
for pool_name, stats in stats.items():
    print(f'{pool_name}: {stats[\"active_connections\"]}/{stats[\"pool_size\"]} active')
"

# 4. Restart services if performance is degraded
avg_response_time=$(curl -w "%{time_total}" -s -o /dev/null http://localhost:8080/health)
if (( $(echo "$avg_response_time > 5.0" | bc -l) )); then
    echo "High response time detected (${avg_response_time}s), considering restart..."
    # Log the decision
    echo "$(date): High response time $avg_response_time - restart recommended" >> /opt/quantrs2/logs/performance.log
fi
```

## Security Operations

### Daily Security Checks

```bash
#!/bin/bash
# /opt/quantrs2/scripts/security-check.sh

echo "=== Daily Security Check - $(date) ==="

# 1. Check for failed authentication attempts
echo "1. Failed authentication attempts (last 24h):"
grep "authentication.*failed" /opt/quantrs2/logs/quantrs2.log | \
grep "$(date -d '1 day ago' '+%Y-%m-%d')\|$(date '+%Y-%m-%d')" | \
wc -l

# 2. Check for unusual access patterns
echo "2. Unusual access patterns:"
awk '{print $1}' /opt/quantrs2/logs/access.log | \
sort | uniq -c | sort -rn | head -10

# 3. Check SSL certificate expiration
echo "3. SSL certificate status:"
for domain in yourdomain.com api.yourdomain.com; do
    expiry=$(echo | openssl s_client -servername "$domain" -connect "$domain:443" 2>/dev/null | \
            openssl x509 -noout -dates | grep notAfter | cut -d= -f2)
    echo "$domain: expires $expiry"
done

# 4. Check for security updates
echo "4. Security updates:"
sudo apt list --upgradable 2>/dev/null | grep -i security | wc -l

# 5. Verify container security
echo "5. Container security scan:"
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
    aquasec/trivy image quantrs2:latest --exit-code 1 --severity HIGH,CRITICAL

# 6. Check for suspicious database activity
echo "6. Database security check:"
docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
SELECT datname, usename, client_addr, state 
FROM pg_stat_activity 
WHERE client_addr IS NOT NULL 
AND client_addr NOT LIKE '172.20.%'
AND client_addr != '127.0.0.1';"
```

### Access Control Management

```bash
#!/bin/bash
# /opt/quantrs2/scripts/manage-access.sh

case "$1" in
    "list-users")
        echo "Active users:"
        docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
        from quantrs2.security.auth_manager import AuthManager
        auth = AuthManager()
        users = auth.list_active_users()
        for user in users:
            print(f'{user.username} - Last login: {user.last_login} - Role: {user.role}')
        "
        ;;
    
    "revoke-user")
        if [ -z "$2" ]; then
            echo "Usage: $0 revoke-user <username>"
            exit 1
        fi
        
        docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
        from quantrs2.security.auth_manager import AuthManager
        auth = AuthManager()
        auth.revoke_user_access('$2')
        print('User $2 access revoked')
        "
        ;;
    
    "rotate-keys")
        echo "Rotating encryption keys..."
        # Generate new keys
        openssl rand -base64 64 > /opt/quantrs2/secrets/master_key.new
        openssl rand -base64 64 > /opt/quantrs2/secrets/jwt_secret.new
        
        # Update configuration (requires restart)
        echo "New keys generated. Manual restart required."
        ;;
    
    *)
        echo "Usage: $0 {list-users|revoke-user|rotate-keys}"
        exit 1
        ;;
esac
```

## Backup and Recovery

### Automated Backup Verification

```bash
#!/bin/bash
# /opt/quantrs2/scripts/verify-backups.sh

echo "=== Backup Verification - $(date) ==="

BACKUP_DIR="/opt/quantrs2/backups"
TEMP_DIR="/tmp/backup-test"

# 1. Check backup file integrity
echo "1. Checking backup file integrity..."
latest_backup=$(ls -t "$BACKUP_DIR/database/"*.dump.gpg | head -1)

if [ -f "$latest_backup" ]; then
    # Test GPG decryption
    if gpg --decrypt "$latest_backup" > /dev/null 2>&1; then
        echo "✅ Latest backup file is intact: $(basename "$latest_backup")"
    else
        echo "❌ Latest backup file is corrupted: $(basename "$latest_backup")"
        exit 1
    fi
else
    echo "❌ No backup files found"
    exit 1
fi

# 2. Test restore procedure (non-destructive)
echo "2. Testing restore procedure..."
mkdir -p "$TEMP_DIR"

# Decrypt backup
gpg --decrypt "$latest_backup" > "$TEMP_DIR/test.dump"

# Test restore to temporary database
docker run --rm \
    -v "$TEMP_DIR:/backup" \
    -e POSTGRES_PASSWORD=testpass \
    postgres:15-alpine \
    sh -c "
    postgres &
    sleep 10
    createdb -U postgres test_restore
    pg_restore -U postgres -d test_restore /backup/test.dump
    echo 'Restore test completed successfully'
    "

# Cleanup
rm -rf "$TEMP_DIR"

# 3. Check backup schedule
echo "3. Checking backup schedule..."
last_backup_time=$(stat -c %Y "$latest_backup")
current_time=$(date +%s)
hours_since_backup=$(( (current_time - last_backup_time) / 3600 ))

if [ "$hours_since_backup" -lt 8 ]; then
    echo "✅ Backup is recent ($hours_since_backup hours old)"
else
    echo "⚠️  Backup is old ($hours_since_backup hours old)"
fi

# 4. Check backup retention
echo "4. Checking backup retention..."
backup_count=$(ls -1 "$BACKUP_DIR/database/"*.dump.gpg | wc -l)
echo "Total backups: $backup_count"

if [ "$backup_count" -lt 7 ]; then
    echo "⚠️  Low backup count ($backup_count < 7)"
fi
```

### Disaster Recovery Testing

```bash
#!/bin/bash
# /opt/quantrs2/scripts/dr-test.sh

echo "=== DISASTER RECOVERY TEST ==="
echo "⚠️  This will perform a non-destructive DR test"

# 1. Create test environment
echo "1. Setting up test environment..."
DR_DIR="/tmp/quantrs2-dr-test"
mkdir -p "$DR_DIR"

# 2. Copy configuration
echo "2. Copying configuration..."
cp -r /opt/quantrs2/config "$DR_DIR/"
cp /opt/quantrs2/.env.production "$DR_DIR/.env.test"

# Modify for test environment
sed -i 's/quantrs2_prod/quantrs2_test/g' "$DR_DIR/.env.test"
sed -i 's/POSTGRES_PORT=5432/POSTGRES_PORT=5433/g' "$DR_DIR/.env.test"

# 3. Start test database
echo "3. Starting test database..."
docker run -d \
    --name quantrs2-dr-test-db \
    -e POSTGRES_DB=quantrs2_test \
    -e POSTGRES_USER=quantrs2_test \
    -e POSTGRES_PASSWORD=testpass \
    -p 5433:5432 \
    postgres:15-alpine

sleep 10

# 4. Restore latest backup
echo "4. Restoring latest backup..."
latest_backup=$(ls -t /opt/quantrs2/backups/database/*.dump.gpg | head -1)
gpg --decrypt "$latest_backup" | \
docker exec -i quantrs2-dr-test-db pg_restore \
    -U quantrs2_test \
    -d quantrs2_test \
    --clean --if-exists

# 5. Test data integrity
echo "5. Testing data integrity..."
table_count=$(docker exec quantrs2-dr-test-db psql -U quantrs2_test -d quantrs2_test -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';")
echo "Tables restored: $table_count"

# 6. Cleanup
echo "6. Cleaning up test environment..."
docker stop quantrs2-dr-test-db
docker rm quantrs2-dr-test-db
rm -rf "$DR_DIR"

echo "✅ DR test completed successfully"
```

## Capacity Planning

### Resource Usage Analysis

```bash
#!/bin/bash
# /opt/quantrs2/scripts/capacity-analysis.sh

echo "=== Capacity Analysis - $(date) ==="

# 1. Historical resource usage
echo "1. Resource usage trends (last 30 days):"

# CPU usage trend
echo "CPU Usage Trend:"
sar -u 1 1 | tail -1 | awk '{print "Current CPU: " $3 "%"}'

# Memory usage trend
echo "Memory Usage Trend:"
free -m | grep Mem | awk '{printf "Current Memory: %.1f%% (%d/%d MB)\n", $3/$2*100, $3, $2}'

# Disk usage trend
echo "Disk Usage Trend:"
df -h /opt/quantrs2 | tail -1 | awk '{print "Current Disk: " $5 " (" $3 "/" $2 ")"}'

# 2. Database growth analysis
echo "2. Database growth analysis:"
docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;"

# 3. Connection pool analysis
echo "3. Connection pool utilization:"
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.connection_pooling import ConnectionPoolManager
manager = ConnectionPoolManager()
stats = manager.get_pool_statistics()
for pool_name, pool_stats in stats.items():
    utilization = (pool_stats['active_connections'] / pool_stats['pool_size']) * 100
    print(f'{pool_name}: {utilization:.1f}% utilization ({pool_stats[\"active_connections\"]}/{pool_stats[\"pool_size\"]})')
"

# 4. Circuit execution volume
echo "4. Circuit execution volume (last 7 days):"
grep "circuit.execution" /opt/quantrs2/logs/quantrs2.log | \
grep "$(date -d '7 days ago' '+%Y-%m-%d')" | wc -l

# 5. Capacity projections
echo "5. Capacity projections:"
current_disk_usage=$(df /opt/quantrs2 | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$current_disk_usage" -gt 70 ]; then
    echo "⚠️  Disk usage is high ($current_disk_usage%) - expansion needed within 30 days"
elif [ "$current_disk_usage" -gt 50 ]; then
    echo "📊 Disk usage is moderate ($current_disk_usage%) - monitor growth"
else
    echo "✅ Disk usage is healthy ($current_disk_usage%)"
fi
```

### Scaling Recommendations

```bash
#!/bin/bash
# /opt/quantrs2/scripts/scaling-recommendations.sh

echo "=== Scaling Recommendations - $(date) ==="

# Analyze current performance metrics
cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d',' -f1)
memory_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
disk_usage=$(df /opt/quantrs2 | tail -1 | awk '{print $5}' | sed 's/%//')

echo "Current utilization:"
echo "- CPU: ${cpu_usage}%"
echo "- Memory: ${memory_usage}%"
echo "- Disk: ${disk_usage}%"

# Generate recommendations
echo ""
echo "Scaling recommendations:"

if (( $(echo "$cpu_usage > 70" | bc -l) )); then
    echo "🔴 HIGH CPU: Scale horizontally (add more containers) or vertically (more CPU cores)"
fi

if [ "$memory_usage" -gt 80 ]; then
    echo "🔴 HIGH MEMORY: Add more RAM or implement memory optimization"
fi

if [ "$disk_usage" -gt 80 ]; then
    echo "🔴 HIGH DISK: Expand storage or implement data retention policies"
fi

# Database scaling analysis
connection_count=$(docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -t -c "SELECT count(*) FROM pg_stat_activity;")
max_connections=$(docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -t -c "SHOW max_connections;")

connection_usage=$(( connection_count * 100 / max_connections ))
if [ "$connection_usage" -gt 70 ]; then
    echo "🔴 HIGH DB CONNECTIONS: Consider connection pooling optimization or database scaling"
fi

# Provide specific recommendations
echo ""
echo "Specific recommendations:"
echo "1. Current setup can handle approximately $(( 100 - cpu_usage ))% more CPU load"
echo "2. Memory headroom: $(( 100 - memory_usage ))%"
echo "3. Database connection utilization: ${connection_usage}%"

if [ "$cpu_usage" -gt 60 ] || [ "$memory_usage" -gt 60 ]; then
    echo "4. Consider scaling within 2 weeks"
fi
```

## Change Management

### Deployment Procedures

```bash
#!/bin/bash
# /opt/quantrs2/scripts/deploy-update.sh

VERSION="$1"
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

echo "=== Deploying QuantRS2 Version $VERSION ==="

# 1. Pre-deployment checks
echo "1. Pre-deployment validation..."
./scripts/morning-check.sh

# 2. Backup current state
echo "2. Creating pre-deployment backup..."
./scripts/backup-database.sh
./scripts/backup-volumes.sh

# 3. Pull new images
echo "3. Pulling new images..."
docker pull quantrs2:$VERSION
docker pull quantrs2:$VERSION-jupyter

# 4. Update docker-compose
echo "4. Updating configuration..."
sed -i "s/quantrs2:latest/quantrs2:$VERSION/g" docker-compose.secure.yml

# 5. Rolling update
echo "5. Performing rolling update..."

# Update application containers one by one
for service in quantrs2-base quantrs2-jupyter; do
    echo "Updating $service..."
    
    # Scale up new version
    docker-compose -f docker-compose.secure.yml up -d --scale $service=2 $service
    
    # Wait for health check
    sleep 30
    
    # Scale down old version
    docker-compose -f docker-compose.secure.yml up -d --scale $service=1 $service
    
    # Verify health
    if ! curl -sf http://localhost:8080/health > /dev/null; then
        echo "❌ Health check failed for $service"
        echo "Rolling back..."
        
        # Rollback
        sed -i "s/quantrs2:$VERSION/quantrs2:latest/g" docker-compose.secure.yml
        docker-compose -f docker-compose.secure.yml up -d $service
        exit 1
    fi
    
    echo "✅ $service updated successfully"
done

# 6. Post-deployment verification
echo "6. Post-deployment verification..."
./scripts/health-check.sh

# 7. Update monitoring
echo "7. Updating monitoring dashboards..."
# Import any new dashboards
# Update alert rules if needed

echo "✅ Deployment completed successfully"
```

### Rollback Procedures

```bash
#!/bin/bash
# /opt/quantrs2/scripts/rollback.sh

PREVIOUS_VERSION="$1"
if [ -z "$PREVIOUS_VERSION" ]; then
    echo "Usage: $0 <previous_version>"
    exit 1
fi

echo "=== EMERGENCY ROLLBACK to $PREVIOUS_VERSION ==="
echo "⚠️  This will rollback to previous version"

# 1. Stop current services
echo "1. Stopping current services..."
docker-compose -f docker-compose.secure.yml stop quantrs2-base quantrs2-jupyter

# 2. Revert configuration
echo "2. Reverting configuration..."
sed -i "s/quantrs2:[^[:space:]]*/quantrs2:$PREVIOUS_VERSION/g" docker-compose.secure.yml

# 3. Start previous version
echo "3. Starting previous version..."
docker-compose -f docker-compose.secure.yml up -d quantrs2-base quantrs2-jupyter

# 4. Verify rollback
echo "4. Verifying rollback..."
sleep 30

if curl -sf http://localhost:8080/health > /dev/null; then
    echo "✅ Rollback successful"
else
    echo "❌ Rollback failed - manual intervention required"
    exit 1
fi

# 5. Restore database if needed
if [ "$2" = "--restore-db" ]; then
    echo "5. Restoring database..."
    latest_backup=$(ls -t /opt/quantrs2/backups/database/*.dump.gpg | head -1)
    ./scripts/restore-database.sh "$latest_backup"
fi

echo "✅ Rollback completed"
```

This operations runbook provides comprehensive day-to-day operational procedures for managing QuantRS2 in production. For specific troubleshooting scenarios, refer to the [Troubleshooting Guide](troubleshooting.md).