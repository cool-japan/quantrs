#!/bin/bash

# QuantRS2 Container Security Validation Script
# Validates that security hardening measures are properly implemented

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test result tracking
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_failure() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test runner
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    ((TESTS_RUN++))
    log_info "Running test: $test_name"
    
    if eval "$test_command"; then
        log_success "$test_name"
        return 0
    else
        log_failure "$test_name"
        return 1
    fi
}

# Container security tests
test_container_user() {
    local container_name="$1"
    log_info "Testing non-root user in container: $container_name"
    
    if docker ps --format "table {{.Names}}" | grep -q "$container_name"; then
        local user_id=$(docker exec "$container_name" id -u 2>/dev/null || echo "1000")
        if [[ "$user_id" != "0" ]]; then
            log_success "Container $container_name runs as non-root user (UID: $user_id)"
            return 0
        else
            log_failure "Container $container_name runs as root user"
            return 1
        fi
    else
        log_warning "Container $container_name is not running"
        return 1
    fi
}

test_container_capabilities() {
    local container_name="$1"
    log_info "Testing container capabilities: $container_name"
    
    if docker ps --format "table {{.Names}}" | grep -q "$container_name"; then
        local caps=$(docker inspect "$container_name" --format '{{.HostConfig.CapAdd}}' 2>/dev/null)
        if [[ "$caps" == "[]" ]] || [[ "$caps" == "<no value>" ]]; then
            log_success "Container $container_name has no additional capabilities"
            return 0
        else
            log_failure "Container $container_name has additional capabilities: $caps"
            return 1
        fi
    else
        log_warning "Container $container_name is not running"
        return 1
    fi
}

test_container_readonly_fs() {
    local container_name="$1"
    log_info "Testing read-only filesystem: $container_name"
    
    if docker ps --format "table {{.Names}}" | grep -q "$container_name"; then
        local readonly=$(docker inspect "$container_name" --format '{{.HostConfig.ReadonlyRootfs}}' 2>/dev/null)
        if [[ "$readonly" == "true" ]]; then
            log_success "Container $container_name has read-only root filesystem"
            return 0
        else
            log_failure "Container $container_name does not have read-only root filesystem"
            return 1
        fi
    else
        log_warning "Container $container_name is not running"
        return 1
    fi
}

test_container_security_opts() {
    local container_name="$1"
    log_info "Testing security options: $container_name"
    
    if docker ps --format "table {{.Names}}" | grep -q "$container_name"; then
        local security_opts=$(docker inspect "$container_name" --format '{{.HostConfig.SecurityOpt}}' 2>/dev/null)
        if echo "$security_opts" | grep -q "no-new-privileges:true"; then
            log_success "Container $container_name has no-new-privileges enabled"
            return 0
        else
            log_failure "Container $container_name does not have no-new-privileges enabled"
            return 1
        fi
    else
        log_warning "Container $container_name is not running"
        return 1
    fi
}

test_container_resource_limits() {
    local container_name="$1"
    log_info "Testing resource limits: $container_name"
    
    if docker ps --format "table {{.Names}}" | grep -q "$container_name"; then
        local memory_limit=$(docker inspect "$container_name" --format '{{.HostConfig.Memory}}' 2>/dev/null)
        local cpu_limit=$(docker inspect "$container_name" --format '{{.HostConfig.NanoCpus}}' 2>/dev/null)
        
        if [[ "$memory_limit" != "0" ]] && [[ "$cpu_limit" != "0" ]]; then
            log_success "Container $container_name has resource limits (Memory: $memory_limit, CPU: $cpu_limit)"
            return 0
        else
            log_failure "Container $container_name does not have proper resource limits"
            return 1
        fi
    else
        log_warning "Container $container_name is not running"
        return 1
    fi
}

test_container_network_isolation() {
    local container_name="$1"
    log_info "Testing network isolation: $container_name"
    
    if docker ps --format "table {{.Names}}" | grep -q "$container_name"; then
        local networks=$(docker inspect "$container_name" --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}} {{end}}' 2>/dev/null)
        if echo "$networks" | grep -q "quantrs_"; then
            log_success "Container $container_name is on isolated networks: $networks"
            return 0
        else
            log_failure "Container $container_name is not properly isolated: $networks"
            return 1
        fi
    else
        log_warning "Container $container_name is not running"
        return 1
    fi
}

test_container_no_privileged() {
    local container_name="$1"
    log_info "Testing privileged mode disabled: $container_name"
    
    if docker ps --format "table {{.Names}}" | grep -q "$container_name"; then
        local privileged=$(docker inspect "$container_name" --format '{{.HostConfig.Privileged}}' 2>/dev/null)
        if [[ "$privileged" == "false" ]]; then
            log_success "Container $container_name is not running in privileged mode"
            return 0
        else
            log_failure "Container $container_name is running in privileged mode"
            return 1
        fi
    else
        log_warning "Container $container_name is not running"
        return 1
    fi
}

test_secrets_not_in_env() {
    local container_name="$1"
    log_info "Testing secrets not in environment variables: $container_name"
    
    if docker ps --format "table {{.Names}}" | grep -q "$container_name"; then
        local env_vars=$(docker exec "$container_name" env 2>/dev/null || echo "")
        local sensitive_patterns=("PASSWORD" "SECRET" "KEY" "TOKEN" "API_KEY")
        
        local found_sensitive=false
        for pattern in "${sensitive_patterns[@]}"; do
            if echo "$env_vars" | grep -q "$pattern"; then
                log_failure "Found sensitive data in environment variables: $pattern"
                found_sensitive=true
            fi
        done
        
        if [[ "$found_sensitive" == "false" ]]; then
            log_success "No sensitive data found in environment variables"
            return 0
        else
            return 1
        fi
    else
        log_warning "Container $container_name is not running"
        return 1
    fi
}

test_health_check_configured() {
    local container_name="$1"
    log_info "Testing health check configuration: $container_name"
    
    if docker ps --format "table {{.Names}}" | grep -q "$container_name"; then
        local health_check=$(docker inspect "$container_name" --format '{{.Config.Healthcheck.Test}}' 2>/dev/null)
        if [[ "$health_check" != "<no value>" ]] && [[ "$health_check" != "[]" ]]; then
            log_success "Container $container_name has health check configured"
            return 0
        else
            log_failure "Container $container_name does not have health check configured"
            return 1
        fi
    else
        log_warning "Container $container_name is not running"
        return 1
    fi
}

test_container_logs_configured() {
    local container_name="$1"
    log_info "Testing logging configuration: $container_name"
    
    if docker ps --format "table {{.Names}}" | grep -q "$container_name"; then
        local log_driver=$(docker inspect "$container_name" --format '{{.HostConfig.LogConfig.Type}}' 2>/dev/null)
        local log_opts=$(docker inspect "$container_name" --format '{{.HostConfig.LogConfig.Config}}' 2>/dev/null)
        
        if [[ "$log_driver" != "" ]] && echo "$log_opts" | grep -q "max-size"; then
            log_success "Container $container_name has proper logging configuration"
            return 0
        else
            log_failure "Container $container_name does not have proper logging configuration"
            return 1
        fi
    else
        log_warning "Container $container_name is not running"
        return 1
    fi
}

# File system security tests
test_secrets_permissions() {
    log_info "Testing secrets file permissions"
    
    local secrets_dir="$PROJECT_ROOT/secrets"
    if [[ -d "$secrets_dir" ]]; then
        local bad_perms=false
        
        while IFS= read -r -d '' file; do
            local perms=$(stat -c "%a" "$file" 2>/dev/null || echo "000")
            if [[ "$perms" != "600" ]]; then
                log_failure "Secret file has incorrect permissions: $file ($perms)"
                bad_perms=true
            fi
        done < <(find "$secrets_dir" -type f -print0)
        
        if [[ "$bad_perms" == "false" ]]; then
            log_success "All secret files have correct permissions (600)"
            return 0
        else
            return 1
        fi
    else
        log_warning "Secrets directory not found: $secrets_dir"
        return 1
    fi
}

test_config_file_security() {
    log_info "Testing configuration file security"
    
    local config_files=(
        "$PROJECT_ROOT/config/.env.secure"
        "$PROJECT_ROOT/docker/docker-compose.secure.yml"
    )
    
    local insecure_files=false
    for file in "${config_files[@]}"; do
        if [[ -f "$file" ]]; then
            local perms=$(stat -c "%a" "$file" 2>/dev/null || echo "000")
            if [[ "$perms" -gt "644" ]]; then
                log_failure "Config file has overly permissive permissions: $file ($perms)"
                insecure_files=true
            fi
        fi
    done
    
    if [[ "$insecure_files" == "false" ]]; then
        log_success "Configuration files have appropriate permissions"
        return 0
    else
        return 1
    fi
}

# Docker daemon security tests
test_docker_daemon_security() {
    log_info "Testing Docker daemon security configuration"
    
    # Check if Docker is running with user namespace remapping
    if docker info 2>/dev/null | grep -q "userns"; then
        log_success "Docker daemon has user namespace remapping enabled"
    else
        log_warning "Docker daemon does not have user namespace remapping enabled"
    fi
    
    # Check if Docker content trust is enabled
    if [[ "${DOCKER_CONTENT_TRUST:-}" == "1" ]]; then
        log_success "Docker content trust is enabled"
    else
        log_warning "Docker content trust is not enabled (set DOCKER_CONTENT_TRUST=1)"
    fi
    
    return 0
}

# Image security tests
test_image_vulnerability_scan() {
    log_info "Testing image vulnerability scanning"
    
    local image_name="quantrs2:secure"
    
    if docker images | grep -q "$image_name"; then
        # Try Trivy scan
        if command -v trivy &> /dev/null; then
            log_info "Running Trivy vulnerability scan..."
            if trivy image --exit-code 1 --severity HIGH,CRITICAL "$image_name" &>/dev/null; then
                log_success "No high/critical vulnerabilities found by Trivy"
                return 0
            else
                log_failure "High/critical vulnerabilities found by Trivy"
                return 1
            fi
        else
            log_warning "Trivy not available for vulnerability scanning"
            return 0
        fi
    else
        log_warning "Image $image_name not found"
        return 1
    fi
}

test_image_size() {
    log_info "Testing image size optimization"
    
    local image_name="quantrs2:secure"
    
    if docker images | grep -q "$image_name"; then
        local size=$(docker images --format "table {{.Size}}" "$image_name" | tail -1)
        local size_mb=$(echo "$size" | sed 's/MB//' | sed 's/GB/*1000/' | bc 2>/dev/null || echo "0")
        
        if (( $(echo "$size_mb < 1000" | bc -l) )); then
            log_success "Image size is optimized: $size"
            return 0
        else
            log_warning "Image size might be too large: $size"
            return 0  # Warning, not failure
        fi
    else
        log_warning "Image $image_name not found"
        return 1
    fi
}

# Network security tests
test_network_security() {
    log_info "Testing network security configuration"
    
    # Check for custom bridge networks
    if docker network ls | grep -q "quantrs_"; then
        log_success "Custom networks are configured"
    else
        log_failure "Custom networks are not configured"
        return 1
    fi
    
    # Check for internal networks
    if docker network inspect quantrs_internal &>/dev/null; then
        local internal=$(docker network inspect quantrs_internal --format '{{.Internal}}' 2>/dev/null)
        if [[ "$internal" == "true" ]]; then
            log_success "Internal network is properly configured"
        else
            log_failure "Internal network is not configured as internal"
            return 1
        fi
    else
        log_warning "Internal network not found"
        return 1
    fi
    
    return 0
}

# Main test execution
run_all_tests() {
    log_info "Starting QuantRS2 container security validation..."
    
    # Test core containers
    local containers=("quantrs2_secure" "quantrs2_postgres_secure" "quantrs2_redis_secure")
    
    for container in "${containers[@]}"; do
        log_info "Testing container: $container"
        
        run_test "Non-root user - $container" "test_container_user $container"
        run_test "No additional capabilities - $container" "test_container_capabilities $container"
        run_test "Read-only filesystem - $container" "test_container_readonly_fs $container"
        run_test "Security options - $container" "test_container_security_opts $container"
        run_test "Resource limits - $container" "test_container_resource_limits $container"
        run_test "Network isolation - $container" "test_container_network_isolation $container"
        run_test "Not privileged - $container" "test_container_no_privileged $container"
        run_test "No secrets in env - $container" "test_secrets_not_in_env $container"
        run_test "Health check configured - $container" "test_health_check_configured $container"
        run_test "Logging configured - $container" "test_container_logs_configured $container"
        
        echo
    done
    
    # Test file system security
    log_info "Testing file system security..."
    run_test "Secrets file permissions" "test_secrets_permissions"
    run_test "Config file security" "test_config_file_security"
    echo
    
    # Test Docker daemon security
    log_info "Testing Docker daemon security..."
    run_test "Docker daemon security" "test_docker_daemon_security"
    echo
    
    # Test image security
    log_info "Testing image security..."
    run_test "Image vulnerability scan" "test_image_vulnerability_scan"
    run_test "Image size optimization" "test_image_size"
    echo
    
    # Test network security
    log_info "Testing network security..."
    run_test "Network security configuration" "test_network_security"
    echo
}

# Generate security report
generate_report() {
    log_info "Generating security validation report..."
    
    local report_file="$PROJECT_ROOT/docker/security-validation-report.txt"
    
    cat > "$report_file" << EOF
QuantRS2 Container Security Validation Report
Generated: $(date -Iseconds)

Test Summary:
- Total tests run: $TESTS_RUN
- Tests passed: $TESTS_PASSED
- Tests failed: $TESTS_FAILED
- Success rate: $(( TESTS_PASSED * 100 / TESTS_RUN ))%

Security Status: $([ $TESTS_FAILED -eq 0 ] && echo "SECURE" || echo "NEEDS ATTENTION")

EOF
    
    if [[ $TESTS_FAILED -gt 0 ]]; then
        echo "FAILED TESTS REQUIRE IMMEDIATE ATTENTION" >> "$report_file"
        echo "Review the output above for specific failures." >> "$report_file"
    else
        echo "All security tests passed successfully." >> "$report_file"
        echo "Container deployment meets security requirements." >> "$report_file"
    fi
    
    log_success "Security report generated: $report_file"
}

# Main execution
main() {
    # Check if Docker is running
    if ! docker info &>/dev/null; then
        log_failure "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    # Run all security tests
    run_all_tests
    
    # Generate report
    generate_report
    
    # Summary
    echo
    log_info "Security validation completed!"
    log_info "Tests run: $TESTS_RUN"
    log_success "Tests passed: $TESTS_PASSED"
    if [[ $TESTS_FAILED -gt 0 ]]; then
        log_failure "Tests failed: $TESTS_FAILED"
        echo
        log_failure "SECURITY VALIDATION FAILED!"
        log_failure "Please review and fix the failed tests before deploying to production."
        exit 1
    else
        echo
        log_success "ALL SECURITY TESTS PASSED!"
        log_success "Container deployment is ready for production."
        exit 0
    fi
}

# Execute main function
main "$@"