#!/usr/bin/env python3
"""
QuantRS2 Docker Health Check Script

This script performs health checks for QuantRS2 Docker containers.
It can be used as a Docker HEALTHCHECK command or run manually.
"""

import sys
import time
import json
import signal
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path

class HealthChecker:
    """Comprehensive health checker for QuantRS2 services."""
    
    def __init__(self, service_type: str = "base", timeout: int = 30):
        self.service_type = service_type
        self.timeout = timeout
        self.checks_passed = 0
        self.checks_failed = 0
        
    def log(self, message: str, level: str = "INFO"):
        """Log health check messages."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}", flush=True)
        
    def timeout_handler(self, signum, frame):
        """Handle timeout for health checks."""
        self.log(f"Health check timed out after {self.timeout} seconds", "ERROR")
        sys.exit(2)
        
    def check_basic_import(self) -> bool:
        """Check if QuantRS2 can be imported."""
        try:
            import quantrs2
            self.log("✓ QuantRS2 import successful")
            return True
        except ImportError as e:
            self.log(f"✗ QuantRS2 import failed: {e}", "ERROR")
            return False
            
    def check_circuit_creation(self) -> bool:
        """Check if basic circuit creation works."""
        try:
            import quantrs2
            circuit = quantrs2.Circuit(2)
            circuit.h(0)
            circuit.cnot(0, 1)
            self.log("✓ Circuit creation successful")
            return True
        except Exception as e:
            self.log(f"✗ Circuit creation failed: {e}", "ERROR")
            return False
            
    def check_simulation(self) -> bool:
        """Check if circuit simulation works."""
        try:
            import quantrs2
            circuit = quantrs2.Circuit(2)
            circuit.h(0)
            circuit.cnot(0, 1)
            result = circuit.run()
            
            if result is None:
                self.log("✗ Simulation returned None", "WARNING")
                return False
                
            probs = result.state_probabilities()
            if not probs:
                self.log("✗ No state probabilities returned", "WARNING") 
                return False
                
            self.log("✓ Circuit simulation successful")
            return True
        except Exception as e:
            self.log(f"✗ Circuit simulation failed: {e}", "ERROR")
            return False
            
    def check_gpu_availability(self) -> bool:
        """Check GPU availability if this is a GPU container."""
        if self.service_type != "gpu":
            return True
            
        try:
            import cupy
            cupy.cuda.Device(0).compute_capability
            self.log("✓ GPU acceleration available")
            return True
        except Exception as e:
            self.log(f"✗ GPU check failed: {e}", "WARNING")
            return False
            
    def check_jupyter_service(self) -> bool:
        """Check Jupyter service if this is a Jupyter container."""
        if self.service_type != "jupyter":
            return True
            
        try:
            import jupyter
            import jupyterlab
            self.log("✓ Jupyter components available")
            return True
        except ImportError as e:
            self.log(f"✗ Jupyter check failed: {e}", "ERROR")
            return False
            
    def check_development_tools(self) -> bool:
        """Check development tools if this is a dev container."""
        if self.service_type != "dev":
            return True
            
        try:
            import pytest
            import black
            import mypy
            self.log("✓ Development tools available")
            return True
        except ImportError as e:
            self.log(f"✗ Development tools check failed: {e}", "ERROR")
            return False
            
    def check_memory_usage(self) -> bool:
        """Check memory usage is reasonable."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            if usage_percent > 95:
                self.log(f"✗ High memory usage: {usage_percent}%", "WARNING")
                return False
            elif usage_percent > 80:
                self.log(f"⚠ Moderate memory usage: {usage_percent}%", "WARNING")
                
            self.log(f"✓ Memory usage OK: {usage_percent}%")
            return True
        except Exception as e:
            self.log(f"✗ Memory check failed: {e}", "WARNING")
            return True  # Don't fail health check for memory issues
            
    def check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_percent = (free / total) * 100
            
            if free_percent < 5:
                self.log(f"✗ Low disk space: {free_percent:.1f}% free", "ERROR")
                return False
            elif free_percent < 15:
                self.log(f"⚠ Moderate disk space: {free_percent:.1f}% free", "WARNING")
                
            self.log(f"✓ Disk space OK: {free_percent:.1f}% free")
            return True
        except Exception as e:
            self.log(f"✗ Disk space check failed: {e}", "WARNING")
            return True  # Don't fail health check for disk issues
            
    def check_network_connectivity(self) -> bool:
        """Check basic network connectivity."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            self.log("✓ Network connectivity OK")
            return True
        except Exception as e:
            self.log(f"✗ Network connectivity failed: {e}", "WARNING")
            return True  # Don't fail health check for network issues
            
    def check_file_permissions(self) -> bool:
        """Check file system permissions."""
        try:
            test_file = Path("/tmp/quantrs2_health_check")
            test_file.write_text("test")
            test_file.unlink()
            self.log("✓ File permissions OK")
            return True
        except Exception as e:
            self.log(f"✗ File permissions check failed: {e}", "ERROR")
            return False
            
    def run_all_checks(self) -> bool:
        """Run all appropriate health checks."""
        self.log(f"Starting health check for {self.service_type} service")
        
        # Set up timeout handler
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.timeout)
        
        try:
            checks = [
                ("Basic Import", self.check_basic_import),
                ("Circuit Creation", self.check_circuit_creation),
                ("Circuit Simulation", self.check_simulation),
                ("GPU Availability", self.check_gpu_availability),
                ("Jupyter Service", self.check_jupyter_service),
                ("Development Tools", self.check_development_tools),
                ("Memory Usage", self.check_memory_usage),
                ("Disk Space", self.check_disk_space),
                ("Network Connectivity", self.check_network_connectivity),
                ("File Permissions", self.check_file_permissions),
            ]
            
            for check_name, check_func in checks:
                try:
                    if check_func():
                        self.checks_passed += 1
                    else:
                        self.checks_failed += 1
                except Exception as e:
                    self.log(f"✗ {check_name} threw exception: {e}", "ERROR")
                    self.checks_failed += 1
                    
            # Cancel the alarm
            signal.alarm(0)
            
            # Determine overall health
            total_checks = self.checks_passed + self.checks_failed
            success_rate = (self.checks_passed / total_checks) * 100 if total_checks > 0 else 0
            
            self.log(f"Health check complete: {self.checks_passed}/{total_checks} checks passed ({success_rate:.1f}%)")
            
            # Consider healthy if >= 80% of checks pass
            is_healthy = success_rate >= 80
            
            if is_healthy:
                self.log("🟢 Service is HEALTHY")
            else:
                self.log("🔴 Service is UNHEALTHY")
                
            return is_healthy
            
        except Exception as e:
            signal.alarm(0)
            self.log(f"Health check failed with exception: {e}", "ERROR")
            return False
            
    def get_health_report(self) -> Dict[str, Any]:
        """Get detailed health report as JSON."""
        return {
            "service_type": self.service_type,
            "timestamp": time.time(),
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "success_rate": (self.checks_passed / (self.checks_passed + self.checks_failed)) * 100 if (self.checks_passed + self.checks_failed) > 0 else 0,
            "healthy": self.checks_passed >= self.checks_failed
        }

def main():
    """Main health check entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantRS2 Docker Health Check")
    parser.add_argument(
        "--service-type", 
        choices=["base", "dev", "jupyter", "gpu"], 
        default="base",
        help="Type of service to check"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=30,
        help="Timeout in seconds for health check"
    )
    parser.add_argument(
        "--json", 
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create health checker
    checker = HealthChecker(args.service_type, args.timeout)
    
    # Run health checks
    is_healthy = checker.run_all_checks()
    
    # Output results
    if args.json:
        print(json.dumps(checker.get_health_report(), indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if is_healthy else 1)

if __name__ == "__main__":
    main()