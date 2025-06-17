# QuantRS2 Security Configuration Guide

This guide provides comprehensive instructions for configuring and maintaining security features in QuantRS2 production deployments, including authentication, authorization, encryption, audit logging, and security monitoring.

## Table of Contents

1. [Security Architecture Overview](#security-architecture-overview)
2. [Authentication and Authorization](#authentication-and-authorization)
3. [Secrets Management](#secrets-management)
4. [Input Validation and Sanitization](#input-validation-and-sanitization)
5. [Encryption and Data Protection](#encryption-and-data-protection)
6. [Network Security](#network-security)
7. [Container Security](#container-security)
8. [Audit Logging and Monitoring](#audit-logging-and-monitoring)
9. [Security Hardening](#security-hardening)
10. [Compliance and Standards](#compliance-and-standards)

## Security Architecture Overview

QuantRS2 implements a defense-in-depth security architecture with multiple layers of protection:

```
┌─────────────────────────────────────────────────────────────┐
│                    External Layer                           │
│  - Load Balancer with SSL/TLS                              │
│  - WAF and DDoS Protection                                 │
│  - Rate Limiting                                           │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                  Application Layer                          │
│  - Authentication & Authorization                          │
│  - Input Validation & Sanitization                         │
│  - Quantum Circuit Security                                │
│  - API Security                                            │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  - Database Encryption                                      │
│  - Connection Security                                      │
│  - Data Classification                                      │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                Infrastructure Layer                         │
│  - Container Security                                       │
│  - Network Segmentation                                     │
│  - Host Security                                            │
│  - Secrets Management                                       │
└─────────────────────────────────────────────────────────────┘
```

### Security Components

**Core Security Modules:**
- `quantrs2.security.auth_manager` - Authentication and session management
- `quantrs2.security.input_validator` - Input validation and sanitization
- `quantrs2.security.quantum_input_validator` - Quantum circuit security
- `quantrs2.security.secrets_manager` - Secrets and key management
- `quantrs2.security.security_utils` - Security utilities and helpers

## Authentication and Authorization

### 1. Authentication Configuration

QuantRS2 supports multiple authentication methods:

**JWT-Based Authentication (Recommended):**

```yaml
# /opt/quantrs2/config/security.yaml
authentication:
  provider: "jwt"
  enabled: true
  
  jwt_config:
    algorithm: "RS256"  # Use RSA signatures
    token_expiry: 3600  # 1 hour
    refresh_token_expiry: 86400  # 24 hours
    issuer: "quantrs2.yourdomain.com"
    audience: ["quantrs2-api", "quantrs2-jupyter"]
    
  session_config:
    timeout: 1800  # 30 minutes
    max_concurrent_sessions: 5
    secure_cookies: true
    httponly_cookies: true
    samesite: "strict"
    
  password_policy:
    min_length: 12
    require_uppercase: true
    require_lowercase: true
    require_numbers: true
    require_special_chars: true
    max_age_days: 90
    history_count: 12  # Prevent reuse of last 12 passwords
    
  security_settings:
    max_login_attempts: 3
    lockout_duration: 900  # 15 minutes
    enable_2fa: true
    enable_captcha: true
    require_password_change_on_first_login: true
```

**Setup JWT Authentication:**

```bash
#!/bin/bash
# /opt/quantrs2/scripts/setup-auth.sh

# Generate RSA key pair for JWT signing
openssl genrsa -out /opt/quantrs2/secrets/jwt_private_key.pem 4096
openssl rsa -in /opt/quantrs2/secrets/jwt_private_key.pem -pubout -out /opt/quantrs2/secrets/jwt_public_key.pem

# Secure key files
chmod 600 /opt/quantrs2/secrets/jwt_private_key.pem
chmod 644 /opt/quantrs2/secrets/jwt_public_key.pem
chown quantrs2:quantrs2 /opt/quantrs2/secrets/jwt_*.pem

# Initialize authentication system
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.security.auth_manager import AuthManager
auth = AuthManager()
auth.initialize_auth_system()
print('Authentication system initialized')
"
```

### 2. User Management

**Create Administrative User:**

```bash
# Create admin user
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.security.auth_manager import AuthManager
auth = AuthManager()

# Create admin user
admin_user = auth.create_user(
    username='admin',
    email='admin@yourdomain.com',
    password='secure_admin_password',
    role='admin',
    require_password_change=True
)

print(f'Admin user created: {admin_user.username}')
"
```

**User Role Configuration:**

```python
# Configure user roles and permissions
from quantrs2.security.auth_manager import AuthManager, Role, Permission

auth = AuthManager()

# Define roles
admin_role = Role(
    name="admin",
    permissions=[
        Permission.ADMIN,
        Permission.EXECUTE_CIRCUITS,
        Permission.MANAGE_USERS,
        Permission.VIEW_SYSTEM_METRICS,
        Permission.MODIFY_SYSTEM_CONFIG
    ]
)

user_role = Role(
    name="user",
    permissions=[
        Permission.EXECUTE_CIRCUITS,
        Permission.VIEW_OWN_RESULTS,
        Permission.CREATE_CIRCUITS
    ]
)

researcher_role = Role(
    name="researcher",
    permissions=[
        Permission.EXECUTE_CIRCUITS,
        Permission.VIEW_OWN_RESULTS,
        Permission.CREATE_CIRCUITS,
        Permission.ACCESS_ADVANCED_FEATURES,
        Permission.VIEW_SYSTEM_METRICS
    ]
)

# Register roles
auth.register_role(admin_role)
auth.register_role(user_role)
auth.register_role(researcher_role)
```

### 3. Two-Factor Authentication (2FA)

**Enable 2FA:**

```bash
# Setup 2FA for user
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.security.auth_manager import AuthManager
auth = AuthManager()

# Enable 2FA for user
secret = auth.enable_2fa('username')
print(f'2FA Secret (for QR code): {secret}')
print('User should scan this with their authenticator app')
"
```

**2FA Configuration:**

```yaml
# 2FA settings in security.yaml
two_factor_auth:
  enabled: true
  issuer_name: "QuantRS2"
  token_validity: 30  # seconds
  backup_codes_count: 10
  
  # Supported methods
  methods:
    - "totp"  # Time-based OTP (Google Authenticator, Authy)
    - "sms"   # SMS-based (if configured)
    - "email" # Email-based backup
    
  # Grace period for new 2FA setup
  grace_period_hours: 24
  
  # Require 2FA for admin users
  require_for_admins: true
  require_for_api_access: true
```

## Secrets Management

### 1. Secrets Manager Setup

QuantRS2 includes a comprehensive secrets management system:

```bash
# Initialize secrets manager
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.security.secrets_manager import SecretsManager
secrets = SecretsManager()

# Initialize secrets storage
secrets.initialize_storage()
print('Secrets manager initialized')
"
```

### 2. Managing Secrets

**Store Secrets Securely:**

```bash
# Store database credentials
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.security.secrets_manager import SecretsManager
secrets = SecretsManager()

# Store database password
secrets.store_secret('database.password', '$(cat /opt/quantrs2/secrets/postgres_password)')

# Store API keys
secrets.store_secret('ibm_quantum.api_key', 'your_ibm_api_key')
secrets.store_secret('google_quantum.api_key', 'your_google_api_key')

# Store encryption keys
secrets.store_secret('encryption.master_key', '$(cat /opt/quantrs2/secrets/master_key)')

print('Secrets stored successfully')
"
```

**Access Secrets in Application:**

```python
# Access secrets securely in application code
from quantrs2.security.secrets_manager import SecretsManager

secrets = SecretsManager()

# Retrieve secrets
db_password = secrets.get_secret('database.password')
ibm_api_key = secrets.get_secret('ibm_quantum.api_key')
master_key = secrets.get_secret('encryption.master_key')

# Secrets are automatically decrypted and returned as strings
```

### 3. Key Rotation

**Automated Key Rotation:**

```bash
#!/bin/bash
# /opt/quantrs2/scripts/rotate-keys.sh

echo "Starting key rotation process..."

# Rotate JWT signing keys
echo "Rotating JWT keys..."
mv /opt/quantrs2/secrets/jwt_private_key.pem /opt/quantrs2/secrets/jwt_private_key.pem.old
mv /opt/quantrs2/secrets/jwt_public_key.pem /opt/quantrs2/secrets/jwt_public_key.pem.old

# Generate new keys
openssl genrsa -out /opt/quantrs2/secrets/jwt_private_key.pem 4096
openssl rsa -in /opt/quantrs2/secrets/jwt_private_key.pem -pubout -out /opt/quantrs2/secrets/jwt_public_key.pem

# Secure new keys
chmod 600 /opt/quantrs2/secrets/jwt_private_key.pem
chmod 644 /opt/quantrs2/secrets/jwt_public_key.pem
chown quantrs2:quantrs2 /opt/quantrs2/secrets/jwt_*.pem

# Rotate master encryption key
echo "Rotating master key..."
openssl rand -base64 64 > /opt/quantrs2/secrets/master_key.new

# Update secrets manager
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.security.secrets_manager import SecretsManager
secrets = SecretsManager()

# Rotate master key
old_key = secrets.get_secret('encryption.master_key')
new_key = open('/opt/quantrs2/secrets/master_key.new').read().strip()

# Re-encrypt all secrets with new key
secrets.rotate_master_key(old_key, new_key)
secrets.store_secret('encryption.master_key', new_key)

print('Master key rotated successfully')
"

# Update environment variables
mv /opt/quantrs2/secrets/master_key.new /opt/quantrs2/secrets/master_key

# Restart services with new keys
docker-compose -f docker-compose.secure.yml restart quantrs2-base

echo "Key rotation completed"
```

## Input Validation and Sanitization

### 1. General Input Validation

Configure strict input validation:

```yaml
# Input validation configuration
input_validation:
  enabled: true
  strict_mode: true
  
  # File upload restrictions
  file_upload:
    max_size: 10485760  # 10MB
    allowed_extensions: [".py", ".ipynb", ".qasm", ".json"]
    scan_for_malware: true
    quarantine_suspicious: true
    
  # API input limits
  api_limits:
    max_request_size: 1048576  # 1MB
    max_string_length: 10000
    max_array_size: 1000
    max_object_depth: 10
    
  # Content filtering
  content_filter:
    block_html: true
    block_javascript: true
    block_sql_injection: true
    block_xss: true
    block_path_traversal: true
    
  # Circuit-specific validation
  quantum_validation:
    max_qubits: 100
    max_circuit_depth: 10000
    max_gates_per_circuit: 100000
    allowed_gate_types: ["standard", "parametric"]
    block_custom_gates: false
```

**Setup Input Validation:**

```bash
# Initialize input validation
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.security.input_validator import InputValidator
from quantrs2.security.quantum_input_validator import QuantumInputValidator

# Initialize validators
input_validator = InputValidator()
quantum_validator = QuantumInputValidator()

# Configure validation rules
input_validator.configure_strict_mode()
quantum_validator.configure_circuit_limits(max_qubits=100, max_depth=10000)

print('Input validation configured')
"
```

### 2. Quantum Circuit Security

**Circuit Validation Rules:**

```python
# Configure quantum circuit security
from quantrs2.security.quantum_input_validator import QuantumInputValidator

validator = QuantumInputValidator()

# Configure circuit limits
validator.set_max_qubits(100)
validator.set_max_circuit_depth(10000)
validator.set_max_gates_per_circuit(100000)

# Configure allowed operations
validator.set_allowed_gate_types(["H", "X", "Y", "Z", "CNOT", "CZ", "RX", "RY", "RZ"])
validator.set_blocked_operations(["CUSTOM_UNSAFE_GATE"])

# Enable circuit analysis
validator.enable_circuit_analysis()
validator.enable_resource_estimation()

# Configure security policies
validator.set_execution_timeout(300)  # 5 minutes max
validator.set_memory_limit(8 * 1024 * 1024 * 1024)  # 8GB max
validator.enable_circuit_fingerprinting()  # Detect malicious patterns
```

**Circuit Security Scanning:**

```bash
# Scan circuit for security issues
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.security.quantum_input_validator import QuantumInputValidator
from quantrs2 import QuantumCircuit

validator = QuantumInputValidator()

# Create test circuit
circuit = QuantumCircuit(5)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)

# Validate circuit
validation_result = validator.validate_circuit(circuit)

if validation_result.is_valid:
    print('Circuit passed security validation')
    print(f'Resource estimate: {validation_result.resource_estimate}')
else:
    print('Circuit failed security validation')
    for issue in validation_result.issues:
        print(f'Security issue: {issue}')
"
```

## Encryption and Data Protection

### 1. Data Encryption Configuration

**Encryption Settings:**

```yaml
# Encryption configuration
encryption:
  # Data at rest
  data_at_rest:
    algorithm: "AES-256-GCM"
    key_derivation: "PBKDF2"
    key_iterations: 100000
    encrypt_database: true
    encrypt_file_storage: true
    encrypt_backups: true
    
  # Data in transit
  data_in_transit:
    tls_version: "1.3"
    cipher_suites:
      - "TLS_AES_256_GCM_SHA384"
      - "TLS_CHACHA20_POLY1305_SHA256"
      - "TLS_AES_128_GCM_SHA256"
    certificate_validation: true
    hsts_enabled: true
    
  # Application-level encryption
  application:
    encrypt_quantum_circuits: true
    encrypt_user_data: true
    encrypt_api_keys: true
    encrypt_session_data: true
    
  # Key management
  key_management:
    key_rotation_days: 90
    backup_encryption_keys: true
    use_hardware_security_module: false  # Set to true if HSM available
    key_escrow: false  # For regulatory compliance if needed
```

**Setup Database Encryption:**

```bash
# Configure PostgreSQL encryption
docker-compose -f docker-compose.secure.yml exec quantrs2-db psql -U quantrs2_prod -d quantrs2_prod -c "
-- Enable transparent data encryption
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Create encryption functions
CREATE OR REPLACE FUNCTION encrypt_sensitive_data(data text, key text)
RETURNS bytea AS \$\$
BEGIN
    RETURN pgp_sym_encrypt(data, key);
END;
\$\$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION decrypt_sensitive_data(encrypted_data bytea, key text)
RETURNS text AS \$\$
BEGIN
    RETURN pgp_sym_decrypt(encrypted_data, key);
END;
\$\$ LANGUAGE plpgsql;
"

# Apply encryption to sensitive tables
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.security.encryption import DatabaseEncryption
encryption = DatabaseEncryption()

# Encrypt existing sensitive data
encryption.encrypt_table_columns('users', ['email', 'phone'])
encryption.encrypt_table_columns('quantum_circuits', ['circuit_data'])
encryption.encrypt_table_columns('api_keys', ['key_value'])

print('Database encryption applied')
"
```

### 2. File and Backup Encryption

**Setup File Encryption:**

```bash
#!/bin/bash
# /opt/quantrs2/scripts/setup-file-encryption.sh

# Install and configure encryption tools
apt-get update
apt-get install -y cryptsetup ecryptfs-utils

# Create encrypted directory for sensitive data
mkdir -p /opt/quantrs2/encrypted-data
mount -t tmpfs -o size=1G tmpfs /opt/quantrs2/encrypted-data

# Setup encrypted backup storage
dd if=/dev/zero of=/opt/quantrs2/encrypted-backup.img bs=1M count=1024
cryptsetup luksFormat /opt/quantrs2/encrypted-backup.img
cryptsetup luksOpen /opt/quantrs2/encrypted-backup.img quantrs2-backup
mkfs.ext4 /dev/mapper/quantrs2-backup
mkdir -p /opt/quantrs2/backup-mount
mount /dev/mapper/quantrs2-backup /opt/quantrs2/backup-mount

echo "File encryption setup completed"
```

## Network Security

### 1. Network Segmentation

**Configure Network Security:**

```yaml
# Network security configuration
network_security:
  # Network segmentation
  segmentation:
    enable_internal_networks: true
    isolate_database: true
    isolate_cache: true
    dmz_enabled: true
    
  # Firewall rules
  firewall:
    default_policy: "deny"
    allowed_inbound:
      - port: 443
        protocol: "tcp"
        source: "0.0.0.0/0"
      - port: 80
        protocol: "tcp" 
        source: "0.0.0.0/0"
    allowed_outbound:
      - port: 443
        protocol: "tcp"
        destination: "0.0.0.0/0"
      - port: 53
        protocol: "udp"
        destination: "0.0.0.0/0"
        
  # Intrusion detection
  ids:
    enabled: true
    monitor_network_traffic: true
    detect_port_scans: true
    detect_brute_force: true
    alert_on_suspicious_activity: true
    
  # Rate limiting
  rate_limiting:
    enabled: true
    requests_per_minute: 100
    burst_limit: 200
    block_duration: 300  # 5 minutes
```

**Setup Network Security:**

```bash
#!/bin/bash
# /opt/quantrs2/scripts/setup-network-security.sh

# Configure iptables rules
echo "Configuring firewall rules..."

# Flush existing rules
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (modify port as needed)
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP and HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow internal Docker network
iptables -A INPUT -s 172.20.0.0/16 -j ACCEPT

# Rate limiting for HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT

# Log dropped packets
iptables -A INPUT -j LOG --log-prefix "DROPPED: "

# Save rules
iptables-save > /etc/iptables/rules.v4

echo "Firewall configured"

# Configure fail2ban for intrusion prevention
cat > /etc/fail2ban/jail.local << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true

[quantrs2-auth]
enabled = true
port = 80,443
logpath = /opt/quantrs2/logs/quantrs2.log
filter = quantrs2-auth
banaction = iptables-multiport
maxretry = 3

[quantrs2-api]
enabled = true
port = 80,443
logpath = /opt/quantrs2/logs/quantrs2.log
filter = quantrs2-api
banaction = iptables-multiport
maxretry = 10
EOF

# Create fail2ban filters
cat > /etc/fail2ban/filter.d/quantrs2-auth.conf << 'EOF'
[Definition]
failregex = ^.*authentication.*failed.*<HOST>.*$
ignoreregex =
EOF

cat > /etc/fail2ban/filter.d/quantrs2-api.conf << 'EOF'
[Definition]
failregex = ^.*"(GET|POST|PUT|DELETE).*" 4[0-9][0-9] .*<HOST>.*$
ignoreregex =
EOF

# Start fail2ban
systemctl enable fail2ban
systemctl start fail2ban

echo "Network security setup completed"
```

### 2. SSL/TLS Configuration

**Advanced SSL/TLS Setup:**

```bash
# Generate strong SSL configuration
cat > /opt/quantrs2/ssl/ssl-params.conf << 'EOF'
# SSL Configuration for QuantRS2
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;

ssl_session_timeout 1d;
ssl_session_cache shared:SSL:50m;
ssl_stapling on;
ssl_stapling_verify on;

# Security headers
add_header Strict-Transport-Security "max-age=63072000" always;
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Referrer-Policy "strict-origin-when-cross-origin";
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'";
EOF
```

## Container Security

### 1. Container Hardening

**Security-Hardened Dockerfile:**

```dockerfile
# Enhanced security Dockerfile for QuantRS2
FROM python:3.11-slim-bullseye

# Create non-root user
RUN groupadd -r quantrs2 && useradd -r -g quantrs2 -u 1000 quantrs2

# Install security updates
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Remove unnecessary packages
RUN apt-get autoremove -y && \
    apt-get autoclean

# Set security attributes
RUN chmod -R go-w /etc /usr/local

# Create secure directories
RUN mkdir -p /app /app/data /app/logs && \
    chown -R quantrs2:quantrs2 /app && \
    chmod 755 /app && \
    chmod 750 /app/data && \
    chmod 750 /app/logs

# Copy application with proper ownership
COPY --chown=quantrs2:quantrs2 . /app/

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Switch to non-root user
USER quantrs2

# Set security-focused environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PYTHONOPTIMIZE=2

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/docker/healthcheck.py

# Run application
CMD ["python", "-m", "quantrs2.main"]
```

### 2. Runtime Security

**Container Security Configuration:**

```yaml
# Container security in docker-compose.secure.yml
security_opt:
  - no-new-privileges:true
  - apparmor:quantrs2-profile
  - seccomp:/opt/quantrs2/security/seccomp.json

cap_drop:
  - ALL
cap_add:
  - SETGID
  - SETUID
  - NET_BIND_SERVICE

read_only: true
tmpfs:
  - /tmp:noexec,nosuid,size=100m
  - /var/tmp:noexec,nosuid,size=50m

ulimits:
  nproc: 1024
  nofile:
    soft: 1024
    hard: 2048
```

**AppArmor Profile:**

```bash
# Create AppArmor profile
cat > /etc/apparmor.d/quantrs2-profile << 'EOF'
#include <tunables/global>

profile quantrs2-profile flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>
  #include <abstractions/python>
  #include <abstractions/nameservice>
  
  # Application directory
  /app/** r,
  /app/data/** rw,
  /app/logs/** rw,
  
  # Python specific
  /usr/bin/python3* ix,
  /usr/lib/python3*/** r,
  
  # Temporary files
  /tmp/** rw,
  
  # Network access
  network tcp,
  network udp,
  
  # Deny dangerous capabilities
  deny capability sys_admin,
  deny capability sys_module,
  deny capability sys_rawio,
  deny capability dac_override,
  
  # Deny file system modifications
  deny /etc/** w,
  deny /usr/** w,
  deny /bin/** w,
  deny /sbin/** w,
}
EOF

# Load profile
apparmor_parser -r -W /etc/apparmor.d/quantrs2-profile
```

## Audit Logging and Monitoring

### 1. Security Event Logging

**Configure Security Logging:**

```yaml
# Security logging configuration
security_logging:
  enabled: true
  log_level: "INFO"
  
  # Events to log
  events:
    authentication:
      login_attempts: true
      login_failures: true
      logout_events: true
      session_timeouts: true
      password_changes: true
      
    authorization:
      permission_denied: true
      role_changes: true
      privilege_escalation: true
      
    data_access:
      sensitive_data_access: true
      circuit_executions: true
      api_key_usage: true
      
    system_events:
      configuration_changes: true
      service_starts_stops: true
      security_policy_changes: true
      
    security_incidents:
      intrusion_attempts: true
      malware_detection: true
      suspicious_activity: true
      
  # Log destinations
  destinations:
    - type: "file"
      path: "/app/logs/security.log"
      format: "json"
      
    - type: "syslog"
      facility: "auth"
      severity: "info"
      
    - type: "external"
      endpoint: "https://siem.yourdomain.com/api/events"
      authentication: "bearer_token"
```

**Setup Security Monitoring:**

```bash
# Initialize security monitoring
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.security.security_monitor import SecurityMonitor
monitor = SecurityMonitor()

# Configure security rules
monitor.add_rule('failed_login_threshold', max_attempts=5, window=300)
monitor.add_rule('suspicious_circuit_patterns', enable=True)
monitor.add_rule('api_abuse_detection', rate_limit=100, window=60)

# Start monitoring
monitor.start_monitoring()
print('Security monitoring started')
"
```

### 2. Compliance Logging

**Configure Compliance Logging:**

```python
# Compliance logging setup
from quantrs2.security.compliance_logger import ComplianceLogger

logger = ComplianceLogger()

# Configure for specific standards
logger.configure_for_standard("SOC2_TYPE2")
logger.configure_for_standard("ISO27001")
logger.configure_for_standard("GDPR")

# Log compliance events
logger.log_data_access("user123", "quantum_circuit_data", "read")
logger.log_data_modification("admin", "user_profile", "update")
logger.log_privileged_action("admin", "system_config", "modify")
```

## Security Hardening

### 1. System Hardening Checklist

**Automated Hardening Script:**

```bash
#!/bin/bash
# /opt/quantrs2/scripts/security-hardening.sh

echo "Starting QuantRS2 security hardening..."

# 1. Update system packages
echo "Updating system packages..."
apt-get update && apt-get upgrade -y

# 2. Configure kernel security
echo "Configuring kernel security..."
cat >> /etc/sysctl.conf << 'EOF'
# Network security
net.ipv4.ip_forward = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.tcp_syncookies = 1

# Memory protection
kernel.randomize_va_space = 2
kernel.exec-shield = 1
kernel.dmesg_restrict = 1
kernel.kptr_restrict = 2

# Process restrictions
fs.suid_dumpable = 0
kernel.core_uses_pid = 1
EOF

sysctl -p

# 3. Configure file permissions
echo "Securing file permissions..."
chmod 600 /opt/quantrs2/secrets/*
chmod 700 /opt/quantrs2/secrets
chmod 755 /opt/quantrs2
chown -R quantrs2:quantrs2 /opt/quantrs2

# 4. Configure login security
echo "Configuring login security..."
cat >> /etc/security/limits.conf << 'EOF'
* hard maxlogins 3
* hard core 0
EOF

# 5. Configure SSH security
echo "Hardening SSH configuration..."
sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
echo "AllowUsers quantrs2" >> /etc/ssh/sshd_config

systemctl restart ssh

# 6. Install and configure auditd
echo "Installing audit daemon..."
apt-get install -y auditd audispd-plugins

cat > /etc/audit/rules.d/quantrs2.rules << 'EOF'
# QuantRS2 audit rules
-w /opt/quantrs2/config/ -p wa -k quantrs2_config
-w /opt/quantrs2/secrets/ -p wa -k quantrs2_secrets
-w /opt/quantrs2/data/ -p wa -k quantrs2_data
-w /opt/quantrs2/logs/ -p wa -k quantrs2_logs

# Monitor privileged commands
-a always,exit -F arch=b64 -S execve -F euid=0 -k root_commands
-a always,exit -F arch=b32 -S execve -F euid=0 -k root_commands

# Monitor file access
-a always,exit -F arch=b64 -S open -F dir=/opt/quantrs2/secrets/ -F success=1 -k secret_access
-a always,exit -F arch=b32 -S open -F dir=/opt/quantrs2/secrets/ -F success=1 -k secret_access
EOF

service auditd restart

# 7. Configure log rotation
echo "Configuring log rotation..."
cat > /etc/logrotate.d/quantrs2-security << 'EOF'
/opt/quantrs2/logs/security.log {
    daily
    missingok
    rotate 365
    compress
    delaycompress
    notifempty
    create 644 quantrs2 quantrs2
    postrotate
        systemctl reload rsyslog
    endscript
}
EOF

echo "Security hardening completed"
```

### 2. Security Validation

**Security Validation Script:**

```bash
#!/bin/bash
# /opt/quantrs2/scripts/validate-security.sh

echo "=== QuantRS2 Security Validation ==="

# Check file permissions
echo "Checking file permissions..."
ls -la /opt/quantrs2/secrets/ | grep -E "^-rw-------"
if [ $? -eq 0 ]; then
    echo "✅ Secret file permissions are secure"
else
    echo "❌ Secret file permissions are insecure"
fi

# Check service security
echo "Checking service security..."
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.security.security_validator import SecurityValidator
validator = SecurityValidator()

# Run security checks
results = validator.run_security_audit()

for check, result in results.items():
    status = '✅' if result['passed'] else '❌'
    print(f'{status} {check}: {result[\"message\"]}')
"

# Check network security
echo "Checking network security..."
if iptables -L | grep -q "DROP"; then
    echo "✅ Firewall is configured"
else
    echo "❌ Firewall is not properly configured"
fi

# Check SSL configuration
echo "Checking SSL configuration..."
if [ -f "/opt/quantrs2/certs/cert.pem" ]; then
    cert_expiry=$(openssl x509 -in /opt/quantrs2/certs/cert.pem -noout -enddate | cut -d= -f2)
    echo "✅ SSL certificate expires: $cert_expiry"
else
    echo "❌ SSL certificate not found"
fi

echo "Security validation completed"
```

## Compliance and Standards

### 1. SOC 2 Type II Compliance

**SOC 2 Configuration:**

```yaml
# SOC 2 compliance configuration
compliance:
  soc2:
    enabled: true
    
    # Security controls
    security:
      access_controls: true
      authentication_required: true
      authorization_required: true
      encryption_in_transit: true
      encryption_at_rest: true
      
    # Availability controls
    availability:
      monitoring_enabled: true
      backup_procedures: true
      disaster_recovery: true
      capacity_planning: true
      
    # Processing integrity
    processing_integrity:
      input_validation: true
      error_handling: true
      data_validation: true
      audit_logging: true
      
    # Confidentiality
    confidentiality:
      data_classification: true
      access_restrictions: true
      secure_disposal: true
      
    # Privacy (if applicable)
    privacy:
      data_inventory: true
      consent_management: false  # Not applicable for quantum computing
      data_subject_rights: false
```

### 2. ISO 27001 Compliance

**ISO 27001 Security Controls:**

```bash
# Implement ISO 27001 controls
docker-compose -f docker-compose.secure.yml exec quantrs2-base python -c "
from quantrs2.security.compliance_manager import ComplianceManager

compliance = ComplianceManager()

# Implement required controls
compliance.implement_control('A.9.1.1', 'Access control policy')
compliance.implement_control('A.10.1.1', 'Cryptographic controls')
compliance.implement_control('A.12.1.2', 'Malware protection')
compliance.implement_control('A.12.6.1', 'Management of technical vulnerabilities')
compliance.implement_control('A.13.1.1', 'Network controls')

print('ISO 27001 controls implemented')
"
```

### 3. Regulatory Compliance

**Data Protection Compliance:**

```python
# GDPR/Data protection compliance
from quantrs2.security.data_protection import DataProtectionManager

dp_manager = DataProtectionManager()

# Configure data protection policies
dp_manager.configure_data_retention(retention_days=365)
dp_manager.configure_data_anonymization(enabled=True)
dp_manager.configure_audit_trail(enabled=True)

# Implement right to erasure
dp_manager.enable_data_deletion()

# Configure data breach notifications
dp_manager.configure_breach_notification(
    notification_email="dpo@yourdomain.com",
    notification_within_hours=72
)
```

This security configuration guide provides comprehensive coverage of QuantRS2's security features and best practices for production deployments. Regular security audits and updates should be performed to maintain the security posture of your QuantRS2 installation.