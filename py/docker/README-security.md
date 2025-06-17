# QuantRS2 Security-Hardened Container Deployment

This directory contains security-hardened Docker configurations for production deployment of QuantRS2. The configurations implement comprehensive container security measures following industry best practices.

## 🔒 Security Features

### Container Security
- **Non-root execution**: All containers run as unprivileged users
- **Read-only filesystems**: Containers use read-only root filesystems with specific writable areas
- **Resource limits**: CPU, memory, and process limits enforced
- **Security options**: 
  - `no-new-privileges` prevents privilege escalation
  - AppArmor profiles for mandatory access control
  - Custom Seccomp profiles restrict system calls
- **Minimal attack surface**: Multi-stage builds with minimal runtime dependencies

### Network Security
- **Network isolation**: Internal networks for service communication
- **Minimal exposure**: Only necessary ports exposed to host
- **Encrypted communication**: TLS/SSL for all external communications

### Secrets Management
- **Docker secrets**: Sensitive data stored securely
- **No hardcoded secrets**: All secrets loaded from secure sources
- **Encryption at rest**: Secrets encrypted when stored

### Monitoring & Auditing
- **Falco integration**: Runtime security monitoring
- **Resource monitoring**: Continuous container resource tracking
- **Security logging**: Centralized security event logging
- **Vulnerability scanning**: Regular image security scans

## 📋 Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- OpenSSL (for secret generation)
- Sufficient system resources (4GB RAM minimum)

### 1. Setup Security Environment

```bash
# Run the security setup script
./container-security.sh

# This will:
# - Generate secure secrets
# - Create security configurations
# - Set up monitoring
# - Configure logging
```

### 2. Deploy Secure Containers

```bash
# Deploy with security-hardened configuration
docker-compose -f docker-compose.secure.yml up -d

# Verify deployment
docker-compose -f docker-compose.secure.yml ps
```

### 3. Validate Security

```bash
# Run comprehensive security validation
./security-validation.sh

# Check the security report
cat security-validation-report.txt
```

## 🛠️ Configuration Files

| File | Purpose |
|------|---------|
| `Dockerfile.secure` | Security-hardened production image |
| `docker-compose.secure.yml` | Secure multi-service deployment |
| `container-security.sh` | Security setup automation |
| `security-validation.sh` | Security testing and validation |
| `.dockerignore.secure` | Prevents sensitive files in builds |

### Security Configurations

| File | Description |
|------|-------------|
| `security/quantrs2-apparmor-profile` | AppArmor mandatory access control |
| `security/quantrs2-seccomp-profile.json` | System call restrictions |
| `security/falco.yaml` | Runtime security monitoring rules |

## 🔧 Customization

### Environment Variables

Key security-related environment variables:

```bash
# Deployment environment
QUANTRS2_ENVIRONMENT=production

# Security settings
QUANTRS2_JWT_SECRET_FILE=/run/secrets/jwt_secret
QUANTRS2_ENCRYPTION_KEY_FILE=/run/secrets/encryption_key

# Database security
POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password

# Resource limits
QUANTRS2_MAX_MEMORY=4g
QUANTRS2_MAX_CPUS=2.0
```

### Secrets Management

Secrets are stored in the `secrets/` directory with restricted permissions:

```bash
secrets/
├── postgres_password.txt     (600)
├── jwt_secret.txt           (600)
├── encryption_key.txt       (600)
└── redis_password.txt       (600)
```

### Network Configuration

The deployment uses two networks:

- `quantrs_internal`: Internal communication only
- `quantrs_external`: External access (limited)

## 🔍 Security Monitoring

### Falco Integration

Falco monitors for suspicious activities:

- Privilege escalation attempts
- Unexpected network connections
- File access violations
- Container escape attempts

### Container Monitoring

Continuous monitoring includes:

- Resource usage tracking
- Process monitoring
- Network connection analysis
- File system integrity checks

### Log Analysis

Security logs are centralized and include:

- Authentication events
- Authorization failures
- Security violations
- Resource limit breaches

## 🚨 Security Checklist

### Pre-deployment

- [ ] All secrets generated and secured
- [ ] Security configurations reviewed
- [ ] Vulnerability scans completed
- [ ] Network policies configured
- [ ] Monitoring systems enabled

### Post-deployment

- [ ] Security validation tests passed
- [ ] Monitoring alerts configured
- [ ] Log aggregation working
- [ ] Backup procedures tested
- [ ] Incident response plan activated

### Ongoing Security

- [ ] Regular vulnerability scans
- [ ] Security log reviews
- [ ] Secret rotation procedures
- [ ] Security policy updates
- [ ] Staff security training

## 📊 Security Compliance

This configuration helps meet requirements for:

- **SOC 2 Type II**: Security, availability, processing integrity
- **NIST Cybersecurity Framework**: Identify, protect, detect, respond, recover
- **CIS Docker Benchmarks**: Container security best practices
- **OWASP Container Security**: Application security standards

## 🆘 Incident Response

### Security Incident Procedures

1. **Detection**: Monitor Falco alerts and security logs
2. **Isolation**: Use `docker stop` to isolate affected containers
3. **Investigation**: Analyze logs and container state
4. **Recovery**: Restore from known-good configurations
5. **Post-incident**: Update security measures

### Emergency Contacts

- **Security Team**: security@quantrs2.com
- **Incident Response**: incident-response@quantrs2.com
- **Emergency Hotline**: Available 24/7

## 🔄 Maintenance

### Regular Tasks

- **Weekly**: Security scan results review
- **Monthly**: Secret rotation (if required)
- **Quarterly**: Security configuration audit
- **Annually**: Penetration testing

### Updates

- **Container images**: Updated monthly or on security advisories
- **Security configurations**: Updated as threats evolve
- **Monitoring rules**: Updated based on incident learnings

## 📚 Additional Resources

- [QuantRS2 Security Documentation](../docs/security.md)
- [Container Security Best Practices](https://docs.docker.com/engine/security/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [OWASP Container Security](https://owasp.org/www-project-container-security/)

## 🐛 Troubleshooting

### Common Issues

**Container fails to start as non-root**
```bash
# Check file permissions
docker exec -it container_name ls -la /app
# Fix ownership if needed
docker exec -it container_name chown -R quantrs:quantrs /app
```

**Health checks failing**
```bash
# Check container logs
docker logs container_name
# Verify health check command
docker exec -it container_name python -c "import quantrs2; print('healthy')"
```

**Network connectivity issues**
```bash
# Check network configuration
docker network ls
docker network inspect quantrs_internal
```

### Debug Mode

For debugging (development only):

```bash
# Enable debug mode
export QUANTRS2_DEBUG=true
export QUANTRS2_LOG_LEVEL=DEBUG

# Use development configuration
docker-compose -f docker-compose.yml up -d
```

## 📞 Support

For security-related questions or issues:

- **Documentation**: [Security Guide](../docs/security.md)
- **Community**: [QuantRS2 Discussions](https://github.com/cool-japan/quantrs/discussions)
- **Security Issues**: security@quantrs2.com (GPG key available)
- **Bug Reports**: [GitHub Issues](https://github.com/cool-japan/quantrs/issues)

---

**⚠️ Security Warning**: This configuration is designed for production use. Never disable security features or run containers as root in production environments. Regularly update and monitor your deployment for security threats.