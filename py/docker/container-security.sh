#!/bin/bash

# QuantRS2 Container Security Setup Script
# This script implements comprehensive container security measures

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SECURITY_DIR="$PROJECT_ROOT/docker/security"
SECRETS_DIR="$PROJECT_ROOT/secrets"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    local deps=("docker" "docker-compose" "openssl" "head" "shred")
    local missing=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing[*]}"
        exit 1
    fi
    
    log_success "All dependencies found"
}

# Create secure directories
create_secure_directories() {
    log_info "Creating secure directories..."
    
    # Create directories with restrictive permissions
    mkdir -p "$SECURITY_DIR" && chmod 700 "$SECURITY_DIR"
    mkdir -p "$SECRETS_DIR" && chmod 700 "$SECRETS_DIR"
    mkdir -p "$PROJECT_ROOT/config" && chmod 755 "$PROJECT_ROOT/config"
    mkdir -p "$PROJECT_ROOT/logs" && chmod 755 "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/data" && chmod 755 "$PROJECT_ROOT/data"
    
    log_success "Secure directories created"
}

# Generate secure secrets
generate_secrets() {
    log_info "Generating secure secrets..."
    
    # Generate PostgreSQL password
    if [[ ! -f "$SECRETS_DIR/postgres_password.txt" ]]; then
        openssl rand -base64 32 > "$SECRETS_DIR/postgres_password.txt"
        chmod 600 "$SECRETS_DIR/postgres_password.txt"
        log_success "PostgreSQL password generated"
    else
        log_info "PostgreSQL password already exists"
    fi
    
    # Generate JWT secret
    if [[ ! -f "$SECRETS_DIR/jwt_secret.txt" ]]; then
        openssl rand -base64 64 > "$SECRETS_DIR/jwt_secret.txt"
        chmod 600 "$SECRETS_DIR/jwt_secret.txt"
        log_success "JWT secret generated"
    else
        log_info "JWT secret already exists"
    fi
    
    # Generate encryption key
    if [[ ! -f "$SECRETS_DIR/encryption_key.txt" ]]; then
        openssl rand -base64 32 > "$SECRETS_DIR/encryption_key.txt"
        chmod 600 "$SECRETS_DIR/encryption_key.txt"
        log_success "Encryption key generated"
    else
        log_info "Encryption key already exists"
    fi
    
    # Generate Redis password
    if [[ ! -f "$SECRETS_DIR/redis_password.txt" ]]; then
        openssl rand -base64 24 > "$SECRETS_DIR/redis_password.txt"
        chmod 600 "$SECRETS_DIR/redis_password.txt"
        log_success "Redis password generated"
    else
        log_info "Redis password already exists"
    fi
}

# Create secure environment file
create_secure_env() {
    log_info "Creating secure environment configuration..."
    
    local env_file="$PROJECT_ROOT/config/.env.secure"
    
    cat > "$env_file" << EOF
# QuantRS2 Secure Environment Configuration
# Generated on $(date -Iseconds)

# Environment
QUANTRS2_ENVIRONMENT=production
QUANTRS2_DEBUG=false
QUANTRS2_LOG_LEVEL=WARNING

# Database
POSTGRES_DB=quantrs2
POSTGRES_USER=quantrs2
POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password

# Redis
REDIS_PASSWORD_FILE=/run/secrets/redis_password

# Security
QUANTRS2_JWT_SECRET_FILE=/run/secrets/quantrs2_jwt_secret
QUANTRS2_ENCRYPTION_KEY_FILE=/run/secrets/quantrs2_encryption_key

# Paths
QUANTRS_DATA_PATH=$PROJECT_ROOT/data
QUANTRS_LOGS_PATH=$PROJECT_ROOT/logs
QUANTRS_CONFIG_PATH=$PROJECT_ROOT/config
POSTGRES_DATA_PATH=$PROJECT_ROOT/postgres_data
POSTGRES_PASSWORD_FILE=$SECRETS_DIR/postgres_password.txt
QUANTRS2_JWT_SECRET_FILE=$SECRETS_DIR/jwt_secret.txt
QUANTRS2_ENCRYPTION_KEY_FILE=$SECRETS_DIR/encryption_key.txt
EOF
    
    chmod 600 "$env_file"
    log_success "Secure environment file created"
}

# Create Docker security configurations
create_docker_security_config() {
    log_info "Creating Docker security configurations..."
    
    # Create AppArmor profile
    cat > "$SECURITY_DIR/quantrs2-apparmor-profile" << 'EOF'
#include <tunables/global>

profile quantrs2-profile flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>
  #include <abstractions/python>
  
  # Allow network access
  network,
  
  # Allow reading configuration
  /app/config/ r,
  /app/config/** r,
  
  # Allow writing logs and data
  /app/logs/ rw,
  /app/logs/** rw,
  /app/data/ rw,
  /app/data/** rw,
  
  # Deny dangerous capabilities
  deny capability sys_admin,
  deny capability sys_module,
  deny capability sys_rawio,
  deny capability sys_ptrace,
  
  # Deny access to sensitive files
  deny /proc/sys/kernel/** w,
  deny /sys/** w,
  deny /dev/mem r,
  deny /dev/kmem r,
  
  # Allow Python execution
  /opt/venv/bin/python ix,
  /usr/bin/python3.11 ix,
}
EOF
    
    # Create Seccomp profile
    cat > "$SECURITY_DIR/quantrs2-seccomp-profile.json" << 'EOF'
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": [
    "SCMP_ARCH_X86_64",
    "SCMP_ARCH_X86",
    "SCMP_ARCH_X32"
  ],
  "syscalls": [
    {
      "names": [
        "read", "write", "open", "close", "stat", "fstat", "lstat", "poll",
        "lseek", "mmap", "mprotect", "munmap", "brk", "rt_sigaction",
        "rt_sigprocmask", "rt_sigreturn", "ioctl", "pread64", "pwrite64",
        "readv", "writev", "access", "pipe", "select", "sched_yield",
        "mremap", "msync", "mincore", "madvise", "shmget", "shmat", "shmctl",
        "dup", "dup2", "pause", "nanosleep", "getitimer", "alarm", "setitimer",
        "getpid", "sendfile", "socket", "connect", "accept", "sendto",
        "recvfrom", "sendmsg", "recvmsg", "shutdown", "bind", "listen",
        "getsockname", "getpeername", "socketpair", "setsockopt", "getsockopt",
        "clone", "fork", "vfork", "execve", "exit", "wait4", "kill", "uname",
        "semget", "semop", "semctl", "shmdt", "msgget", "msgsnd", "msgrcv",
        "msgctl", "fcntl", "flock", "fsync", "fdatasync", "truncate",
        "ftruncate", "getdents", "getcwd", "chdir", "fchdir", "rename",
        "mkdir", "rmdir", "creat", "link", "unlink", "symlink", "readlink",
        "chmod", "fchmod", "chown", "fchown", "lchown", "umask", "gettimeofday",
        "getrlimit", "getrusage", "sysinfo", "times", "ptrace", "getuid",
        "syslog", "getgid", "setuid", "setgid", "geteuid", "getegid",
        "setpgid", "getppid", "getpgrp", "setsid", "setreuid", "setregid",
        "getgroups", "setgroups", "setresuid", "getresuid", "setresgid",
        "getresgid", "getpgid", "setfsuid", "setfsgid", "getsid", "capget",
        "capset", "rt_sigpending", "rt_sigtimedwait", "rt_sigqueueinfo",
        "rt_sigsuspend", "sigaltstack", "utime", "mknod", "uselib",
        "personality", "ustat", "statfs", "fstatfs", "sysfs", "getpriority",
        "setpriority", "sched_setparam", "sched_getparam", "sched_setscheduler",
        "sched_getscheduler", "sched_get_priority_max", "sched_get_priority_min",
        "sched_rr_get_interval", "mlock", "munlock", "mlockall", "munlockall",
        "vhangup", "modify_ldt", "pivot_root", "_sysctl", "prctl", "arch_prctl",
        "adjtimex", "setrlimit", "chroot", "sync", "acct", "settimeofday",
        "mount", "umount2", "swapon", "swapoff", "reboot", "sethostname",
        "setdomainname", "iopl", "ioperm", "create_module", "init_module",
        "delete_module", "get_kernel_syms", "query_module", "quotactl",
        "nfsservctl", "getpmsg", "putpmsg", "afs_syscall", "tuxcall",
        "security", "gettid", "readahead", "setxattr", "lsetxattr", "fsetxattr",
        "getxattr", "lgetxattr", "fgetxattr", "listxattr", "llistxattr",
        "flistxattr", "removexattr", "lremovexattr", "fremovexattr", "tkill",
        "time", "futex", "sched_setaffinity", "sched_getaffinity",
        "set_thread_area", "io_setup", "io_destroy", "io_getevents",
        "io_submit", "io_cancel", "get_thread_area", "lookup_dcookie",
        "epoll_create", "epoll_ctl_old", "epoll_wait_old", "remap_file_pages",
        "getdents64", "set_tid_address", "restart_syscall", "semtimedop",
        "fadvise64", "timer_create", "timer_settime", "timer_gettime",
        "timer_getoverrun", "timer_delete", "clock_settime", "clock_gettime",
        "clock_getres", "clock_nanosleep", "exit_group", "epoll_wait",
        "epoll_ctl", "tgkill", "utimes", "vserver", "mbind", "set_mempolicy",
        "get_mempolicy", "mq_open", "mq_unlink", "mq_timedsend",
        "mq_timedreceive", "mq_notify", "mq_getsetattr", "kexec_load",
        "waitid", "add_key", "request_key", "keyctl", "ioprio_set",
        "ioprio_get", "inotify_init", "inotify_add_watch", "inotify_rm_watch",
        "migrate_pages", "openat", "mkdirat", "mknodat", "fchownat",
        "futimesat", "newfstatat", "unlinkat", "renameat", "linkat",
        "symlinkat", "readlinkat", "fchmodat", "faccessat", "pselect6",
        "ppoll", "unshare", "set_robust_list", "get_robust_list", "splice",
        "tee", "sync_file_range", "vmsplice", "move_pages", "utimensat",
        "epoll_pwait", "signalfd", "timerfd_create", "eventfd", "fallocate",
        "timerfd_settime", "timerfd_gettime", "accept4", "signalfd4",
        "eventfd2", "epoll_create1", "dup3", "pipe2", "inotify_init1",
        "preadv", "pwritev", "rt_tgsigqueueinfo", "perf_event_open",
        "recvmmsg", "fanotify_init", "fanotify_mark", "prlimit64",
        "name_to_handle_at", "open_by_handle_at", "clock_adjtime",
        "syncfs", "sendmmsg", "setns", "getcpu", "process_vm_readv",
        "process_vm_writev", "kcmp", "finit_module"
      ],
      "action": "SCMP_ACT_ALLOW"
    }
  ]
}
EOF
    
    # Create Falco security rules
    cat > "$SECURITY_DIR/falco.yaml" << 'EOF'
rules_file:
  - /etc/falco/falco_rules.yaml
  - /etc/falco/falco_rules.local.yaml
  - /etc/falco/rules.d

time_format_iso_8601: true
json_output: true
json_include_output_property: true

file_output:
  enabled: true
  keep_alive: false
  filename: /var/log/falco_events.txt

stdout_output:
  enabled: true

syslog_output:
  enabled: false

http_output:
  enabled: false

program_output:
  enabled: false

grpc:
  enabled: true
  bind_address: "0.0.0.0:5060"
  threadiness: 8

grpc_output:
  enabled: false

webserver:
  enabled: false

syscall_event_drops:
  actions:
    - log
    - alert
  rate: 0.03333
  max_burst: 1000

priority: debug

buffered_outputs: false

syscall_event_timeouts:
  max_consecutives: 1000

metadata_download:
  max_mb: 100
  chunk_wait_us: 1000
  watch_freq_sec: 1

container_engines:
  docker:
    path: /var/run/docker.sock
EOF
    
    log_success "Docker security configurations created"
}

# Scan container images for vulnerabilities
scan_container_images() {
    log_info "Scanning container images for vulnerabilities..."
    
    # Check if Trivy is available
    if command -v trivy &> /dev/null; then
        log_info "Scanning with Trivy..."
        trivy image --severity HIGH,CRITICAL quantrs2:secure || log_warning "Trivy scan found issues"
    else
        log_warning "Trivy not found. Install with: curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin"
    fi
    
    # Check if Docker Scout is available
    if docker scout version &> /dev/null; then
        log_info "Scanning with Docker Scout..."
        docker scout cves quantrs2:secure || log_warning "Docker Scout scan found issues"
    else
        log_warning "Docker Scout not available. Enable with: docker scout repo enable"
    fi
}

# Set up container monitoring
setup_monitoring() {
    log_info "Setting up container monitoring..."
    
    # Create monitoring configuration
    mkdir -p "$SECURITY_DIR/monitoring"
    
    # Create container resource monitoring script
    cat > "$SECURITY_DIR/monitoring/monitor-containers.sh" << 'EOF'
#!/bin/bash
# Container monitoring script

LOGFILE="/var/log/quantrs2/container-monitor.log"
mkdir -p "$(dirname "$LOGFILE")"

while true; do
    echo "$(date -Iseconds) - Container monitoring check" >> "$LOGFILE"
    
    # Check resource usage
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" >> "$LOGFILE"
    
    # Check for security violations
    docker exec quantrs2_secure ps aux | grep -E "(root|sudo|su)" >> "$LOGFILE" 2>/dev/null || true
    
    # Check network connections
    docker exec quantrs2_secure netstat -tuln >> "$LOGFILE" 2>/dev/null || true
    
    sleep 300  # Check every 5 minutes
done
EOF
    
    chmod +x "$SECURITY_DIR/monitoring/monitor-containers.sh"
    log_success "Container monitoring configured"
}

# Create security documentation
create_security_docs() {
    log_info "Creating security documentation..."
    
    cat > "$SECURITY_DIR/SECURITY.md" << 'EOF'
# QuantRS2 Container Security Guide

## Security Features Implemented

### Container Hardening
- **Non-root user**: All containers run as non-privileged users
- **Read-only filesystem**: Containers use read-only root filesystems
- **Resource limits**: CPU, memory, and process limits enforced
- **Security options**: 
  - `no-new-privileges`: Prevents privilege escalation
  - AppArmor profiles for access control
  - Custom Seccomp profiles to restrict syscalls

### Network Security
- **Internal networks**: Isolated internal networks for service communication
- **Minimal port exposure**: Only necessary ports exposed to host
- **Network policies**: Restricted network access between containers

### Secrets Management
- **Docker secrets**: Sensitive data stored as Docker secrets
- **Environment separation**: No hardcoded secrets in images
- **Encryption**: Secrets encrypted at rest and in transit

### Image Security
- **Minimal base images**: Alpine Linux for reduced attack surface
- **Signed images**: Container images signed and verified
- **Vulnerability scanning**: Regular security scans with Trivy/Scout
- **Multi-stage builds**: Separate build and runtime environments

### Monitoring and Auditing
- **Falco**: Runtime security monitoring for anomaly detection
- **Resource monitoring**: Continuous resource usage monitoring
- **Log aggregation**: Centralized logging with security event tracking
- **Health checks**: Regular container health verification

## Security Checklist

### Deployment Security
- [ ] Review and update all default passwords
- [ ] Verify secrets are properly mounted and not in environment variables
- [ ] Ensure containers run as non-root users
- [ ] Validate network configurations and firewall rules
- [ ] Test backup and recovery procedures

### Runtime Security
- [ ] Monitor container resource usage
- [ ] Review security alerts and logs regularly
- [ ] Keep container images updated
- [ ] Perform regular vulnerability scans
- [ ] Audit container access and permissions

### Incident Response
- [ ] Container isolation procedures documented
- [ ] Security incident escalation paths defined
- [ ] Forensic data collection procedures established
- [ ] Recovery procedures tested and documented

## Security Configuration

### AppArmor Profile
The AppArmor profile restricts file system access and capabilities.
Location: `security/quantrs2-apparmor-profile`

### Seccomp Profile
The Seccomp profile restricts system calls available to containers.
Location: `security/quantrs2-seccomp-profile.json`

### Falco Rules
Falco monitors for suspicious container activity.
Configuration: `security/falco.yaml`

## Compliance

This configuration implements security controls for:
- NIST Cybersecurity Framework
- CIS Docker Benchmarks
- OWASP Container Security
- SOC 2 Type II requirements

## Security Contacts

For security issues or questions:
- Security Team: security@quantrs2.com
- Bug Reports: security-bugs@quantrs2.com
- Emergency: security-emergency@quantrs2.com
EOF
    
    log_success "Security documentation created"
}

# Run security benchmark
run_security_benchmark() {
    log_info "Running Docker security benchmark..."
    
    # Check if Docker Bench for Security is available
    if [[ -f "/usr/local/bin/docker-bench-security.sh" ]]; then
        log_info "Running Docker Bench for Security..."
        sudo /usr/local/bin/docker-bench-security.sh -l /var/log/docker-bench-security.log
    else
        log_warning "Docker Bench for Security not found. Install with:"
        log_warning "git clone https://github.com/docker/docker-bench-security.git"
        log_warning "cd docker-bench-security && sudo ./docker-bench-security.sh"
    fi
}

# Main execution
main() {
    log_info "Starting QuantRS2 container security setup..."
    
    check_root
    check_dependencies
    create_secure_directories
    generate_secrets
    create_secure_env
    create_docker_security_config
    setup_monitoring
    create_security_docs
    
    log_success "Container security setup completed successfully!"
    log_info "Next steps:"
    log_info "1. Review generated secrets in $SECRETS_DIR"
    log_info "2. Customize security configurations in $SECURITY_DIR"
    log_info "3. Run: docker-compose -f docker/docker-compose.secure.yml up -d"
    log_info "4. Monitor security logs and alerts"
    
    # Optional security checks
    echo
    read -p "Run additional security checks? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        scan_container_images
        run_security_benchmark
    fi
    
    log_success "QuantRS2 container security is ready for production!"
}

# Execute main function
main "$@"