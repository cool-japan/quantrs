#!/usr/bin/env python3
"""
Health Check and System Diagnostics for QuantRS2

This module provides health checking and system diagnostics capabilities
for production deployments of QuantRS2.

Features:
    - System health checks
    - Dependency verification
    - Hardware capability detection
    - Resource availability monitoring
    - Configuration validation
    - Integration testing
    - Status reporting (JSON, text, HTML)

Usage:
    from quantrs2.health_check import run_health_check, HealthStatus

    # Run health check
    status = run_health_check()

    if status.is_healthy():
        print("System is healthy!")
    else:
        print(f"Health issues: {status.get_issues()}")
"""

import sys
import platform
import subprocess
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import warnings

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class HealthLevel(Enum):
    """Health check severity levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    level: HealthLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def is_healthy(self) -> bool:
        """Check if result is healthy."""
        return self.level == HealthLevel.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'level': self.level.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
        }


@dataclass
class HealthStatus:
    """Overall health status."""
    overall_health: HealthLevel
    checks: List[HealthCheckResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    system_info: Dict[str, Any] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """Check if overall status is healthy."""
        return self.overall_health == HealthLevel.HEALTHY

    def get_issues(self) -> List[HealthCheckResult]:
        """Get all non-healthy checks."""
        return [check for check in self.checks if not check.is_healthy()]

    def get_by_level(self, level: HealthLevel) -> List[HealthCheckResult]:
        """Get checks by severity level."""
        return [check for check in self.checks if check.level == level]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'overall_health': self.overall_health.value,
            'timestamp': self.timestamp,
            'system_info': self.system_info,
            'checks': [check.to_dict() for check in self.checks],
            'summary': {
                'total_checks': len(self.checks),
                'healthy': len(self.get_by_level(HealthLevel.HEALTHY)),
                'warnings': len(self.get_by_level(HealthLevel.WARNING)),
                'errors': len(self.get_by_level(HealthLevel.ERROR)),
                'critical': len(self.get_by_level(HealthLevel.CRITICAL)),
            }
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class HealthChecker:
    """Main health checker class."""

    def __init__(self):
        """Initialize health checker."""
        self.checks: List[HealthCheckResult] = []

    def check_python_version(self) -> HealthCheckResult:
        """Check Python version compatibility."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.major < 3 or (version.major == 3 and version.minor < 8):
            return HealthCheckResult(
                name="python_version",
                level=HealthLevel.ERROR,
                message=f"Python {version_str} is not supported (requires >= 3.8)",
                details={'version': version_str, 'required': '>=3.8'}
            )
        elif version.minor < 10:
            return HealthCheckResult(
                name="python_version",
                level=HealthLevel.WARNING,
                message=f"Python {version_str} is supported but outdated (recommend >= 3.10)",
                details={'version': version_str, 'recommended': '>=3.10'}
            )
        else:
            return HealthCheckResult(
                name="python_version",
                level=HealthLevel.HEALTHY,
                message=f"Python {version_str} is compatible",
                details={'version': version_str}
            )

    def check_quantrs2_import(self) -> HealthCheckResult:
        """Check if QuantRS2 can be imported."""
        try:
            import quantrs2
            version = getattr(quantrs2, '__version__', 'unknown')
            return HealthCheckResult(
                name="quantrs2_import",
                level=HealthLevel.HEALTHY,
                message=f"QuantRS2 {version} imported successfully",
                details={'version': version}
            )
        except ImportError as e:
            return HealthCheckResult(
                name="quantrs2_import",
                level=HealthLevel.CRITICAL,
                message=f"Failed to import QuantRS2: {e}",
                details={'error': str(e)}
            )

    def check_dependencies(self) -> List[HealthCheckResult]:
        """Check optional dependencies."""
        results = []

        # Check numpy
        try:
            import numpy as np
            results.append(HealthCheckResult(
                name="dependency_numpy",
                level=HealthLevel.HEALTHY,
                message=f"NumPy {np.__version__} available",
                details={'version': np.__version__}
            ))
        except ImportError:
            results.append(HealthCheckResult(
                name="dependency_numpy",
                level=HealthLevel.WARNING,
                message="NumPy not available (optional)",
                details={}
            ))

        # Check psutil
        if PSUTIL_AVAILABLE:
            import psutil
            results.append(HealthCheckResult(
                name="dependency_psutil",
                level=HealthLevel.HEALTHY,
                message=f"psutil {psutil.__version__} available",
                details={'version': psutil.__version__}
            ))
        else:
            results.append(HealthCheckResult(
                name="dependency_psutil",
                level=HealthLevel.WARNING,
                message="psutil not available (optional, needed for resource monitoring)",
                details={}
            ))

        # Check matplotlib
        try:
            import matplotlib
            results.append(HealthCheckResult(
                name="dependency_matplotlib",
                level=HealthLevel.HEALTHY,
                message=f"matplotlib {matplotlib.__version__} available",
                details={'version': matplotlib.__version__}
            ))
        except ImportError:
            results.append(HealthCheckResult(
                name="dependency_matplotlib",
                level=HealthLevel.WARNING,
                message="matplotlib not available (optional, needed for visualization)",
                details={}
            ))

        return results

    def check_system_resources(self) -> List[HealthCheckResult]:
        """Check system resources."""
        results = []

        if not PSUTIL_AVAILABLE:
            results.append(HealthCheckResult(
                name="system_resources",
                level=HealthLevel.WARNING,
                message="Cannot check system resources (psutil not available)",
                details={}
            ))
            return results

        # Check available memory
        mem = psutil.virtual_memory()
        mem_available_gb = mem.available / (1024 ** 3)
        mem_total_gb = mem.total / (1024 ** 3)
        mem_percent = mem.percent

        if mem_percent > 90:
            level = HealthLevel.ERROR
            message = f"Low memory: {mem_available_gb:.1f} GB available ({100-mem_percent:.1f}% free)"
        elif mem_percent > 80:
            level = HealthLevel.WARNING
            message = f"Memory usage high: {mem_available_gb:.1f} GB available ({100-mem_percent:.1f}% free)"
        else:
            level = HealthLevel.HEALTHY
            message = f"Memory available: {mem_available_gb:.1f} GB ({100-mem_percent:.1f}% free)"

        results.append(HealthCheckResult(
            name="memory_availability",
            level=level,
            message=message,
            details={
                'total_gb': round(mem_total_gb, 2),
                'available_gb': round(mem_available_gb, 2),
                'percent_used': mem_percent
            }
        ))

        # Check CPU
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=0.1)

        if cpu_percent > 90:
            level = HealthLevel.WARNING
            message = f"High CPU usage: {cpu_percent}%"
        else:
            level = HealthLevel.HEALTHY
            message = f"CPU usage: {cpu_percent}% ({cpu_count} cores)"

        results.append(HealthCheckResult(
            name="cpu_usage",
            level=level,
            message=message,
            details={
                'cpu_count': cpu_count,
                'cpu_percent': cpu_percent
            }
        ))

        # Check disk space
        disk = psutil.disk_usage('/')
        disk_free_gb = disk.free / (1024 ** 3)
        disk_percent = disk.percent

        if disk_percent > 95:
            level = HealthLevel.ERROR
            message = f"Low disk space: {disk_free_gb:.1f} GB free ({100-disk_percent:.1f}% available)"
        elif disk_percent > 85:
            level = HealthLevel.WARNING
            message = f"Disk space getting low: {disk_free_gb:.1f} GB free ({100-disk_percent:.1f}% available)"
        else:
            level = HealthLevel.HEALTHY
            message = f"Disk space: {disk_free_gb:.1f} GB free ({100-disk_percent:.1f}% available)"

        results.append(HealthCheckResult(
            name="disk_space",
            level=level,
            message=message,
            details={
                'total_gb': round(disk.total / (1024 ** 3), 2),
                'free_gb': round(disk_free_gb, 2),
                'percent_used': disk_percent
            }
        ))

        return results

    def check_hardware_capabilities(self) -> List[HealthCheckResult]:
        """Check hardware capabilities for quantum simulation."""
        results = []

        # Check if running on known architecture
        machine = platform.machine()
        supported_archs = ['x86_64', 'AMD64', 'arm64', 'aarch64']

        if machine in supported_archs:
            results.append(HealthCheckResult(
                name="architecture",
                level=HealthLevel.HEALTHY,
                message=f"Architecture {machine} is supported",
                details={'architecture': machine}
            ))
        else:
            results.append(HealthCheckResult(
                name="architecture",
                level=HealthLevel.WARNING,
                message=f"Architecture {machine} is untested",
                details={'architecture': machine}
            ))

        # Check for GPU availability (basic check)
        gpu_available = False
        try:
            # Try CUDA
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                timeout=2
            )
            if result.returncode == 0:
                gpu_available = True
                results.append(HealthCheckResult(
                    name="gpu_nvidia",
                    level=HealthLevel.HEALTHY,
                    message="NVIDIA GPU detected",
                    details={}
                ))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        if not gpu_available:
            results.append(HealthCheckResult(
                name="gpu",
                level=HealthLevel.WARNING,
                message="No GPU detected (CPU-only mode)",
                details={}
            ))

        return results

    def check_basic_functionality(self) -> List[HealthCheckResult]:
        """Check basic QuantRS2 functionality."""
        results = []

        # Try to create a simple circuit
        try:
            from quantrs2 import create_bell_state
            result = create_bell_state()
            if result is not None:
                results.append(HealthCheckResult(
                    name="basic_circuit_creation",
                    level=HealthLevel.HEALTHY,
                    message="Basic circuit creation works",
                    details={}
                ))
            else:
                results.append(HealthCheckResult(
                    name="basic_circuit_creation",
                    level=HealthLevel.WARNING,
                    message="Circuit creation returned None",
                    details={}
                ))
        except Exception as e:
            results.append(HealthCheckResult(
                name="basic_circuit_creation",
                level=HealthLevel.ERROR,
                message=f"Circuit creation failed: {e}",
                details={'error': str(e)}
            ))

        return results

    def run_all_checks(self) -> HealthStatus:
        """Run all health checks."""
        checks = []

        # Core checks
        checks.append(self.check_python_version())
        checks.append(self.check_quantrs2_import())

        # Dependency checks
        checks.extend(self.check_dependencies())

        # Resource checks
        checks.extend(self.check_system_resources())

        # Hardware checks
        checks.extend(self.check_hardware_capabilities())

        # Functionality checks
        checks.extend(self.check_basic_functionality())

        # Determine overall health
        levels = [check.level for check in checks]
        if HealthLevel.CRITICAL in levels:
            overall = HealthLevel.CRITICAL
        elif HealthLevel.ERROR in levels:
            overall = HealthLevel.ERROR
        elif HealthLevel.WARNING in levels:
            overall = HealthLevel.WARNING
        else:
            overall = HealthLevel.HEALTHY

        # System info
        system_info = {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        }

        return HealthStatus(
            overall_health=overall,
            checks=checks,
            system_info=system_info
        )


def run_health_check() -> HealthStatus:
    """
    Run a comprehensive health check.

    Returns:
        HealthStatus object with results
    """
    checker = HealthChecker()
    return checker.run_all_checks()


def print_health_status(status: HealthStatus, verbose: bool = False) -> None:
    """
    Print health status in a formatted way.

    Args:
        status: HealthStatus to print
        verbose: Include detailed information
    """
    # Header
    print("\n" + "="*80)
    print("QUANTRS2 HEALTH CHECK")
    print("="*80)

    # Overall status
    status_symbols = {
        HealthLevel.HEALTHY: "✅",
        HealthLevel.WARNING: "⚠️",
        HealthLevel.ERROR: "❌",
        HealthLevel.CRITICAL: "🚨",
    }

    symbol = status_symbols.get(status.overall_health, "❓")
    print(f"\nOverall Status: {symbol} {status.overall_health.value.upper()}")
    print(f"Timestamp: {status.timestamp}")

    # Summary
    summary = status.to_dict()['summary']
    print(f"\nChecks Summary:")
    print(f"  Total: {summary['total_checks']}")
    print(f"  ✅ Healthy: {summary['healthy']}")
    if summary['warnings'] > 0:
        print(f"  ⚠️  Warnings: {summary['warnings']}")
    if summary['errors'] > 0:
        print(f"  ❌ Errors: {summary['errors']}")
    if summary['critical'] > 0:
        print(f"  🚨 Critical: {summary['critical']}")

    # Issues
    issues = status.get_issues()
    if issues:
        print("\n" + "-"*80)
        print("ISSUES DETECTED:")
        print("-"*80)
        for issue in issues:
            symbol = status_symbols.get(issue.level, "❓")
            print(f"\n{symbol} {issue.name}")
            print(f"   {issue.message}")
            if verbose and issue.details:
                print(f"   Details: {issue.details}")

    # All checks (if verbose)
    if verbose:
        print("\n" + "-"*80)
        print("ALL CHECKS:")
        print("-"*80)
        for check in status.checks:
            symbol = status_symbols.get(check.level, "❓")
            print(f"\n{symbol} {check.name}")
            print(f"   {check.message}")
            if check.details:
                print(f"   Details: {check.details}")

    # System info
    if verbose:
        print("\n" + "-"*80)
        print("SYSTEM INFORMATION:")
        print("-"*80)
        for key, value in status.system_info.items():
            print(f"  {key}: {value}")

    print("\n" + "="*80 + "\n")


def export_health_check(output_file: str, format: str = 'json') -> None:
    """
    Run health check and export results to file.

    Args:
        output_file: Path to output file
        format: Output format ('json', 'text', 'html')
    """
    status = run_health_check()

    if format == 'json':
        with open(output_file, 'w') as f:
            f.write(status.to_json())
        print(f"Health check exported to {output_file} (JSON)")

    elif format == 'text':
        import sys
        from io import StringIO

        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()

        print_health_status(status, verbose=True)

        sys.stdout = old_stdout
        output = buffer.getvalue()

        with open(output_file, 'w') as f:
            f.write(output)
        print(f"Health check exported to {output_file} (text)")

    elif format == 'html':
        html = generate_html_report(status)
        with open(output_file, 'w') as f:
            f.write(html)
        print(f"Health check exported to {output_file} (HTML)")

    else:
        raise ValueError(f"Unknown format: {format}")


def generate_html_report(status: HealthStatus) -> str:
    """Generate HTML health check report."""
    status_colors = {
        HealthLevel.HEALTHY: "#28a745",
        HealthLevel.WARNING: "#ffc107",
        HealthLevel.ERROR: "#dc3545",
        HealthLevel.CRITICAL: "#721c24",
    }

    color = status_colors.get(status.overall_health, "#6c757d")

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>QuantRS2 Health Check</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 3px solid {color}; padding-bottom: 10px; }}
        .status {{ font-size: 24px; font-weight: bold; color: {color}; margin: 20px 0; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .summary-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }}
        .check {{ margin: 10px 0; padding: 10px; border-left: 4px solid #28a745; background: #f8f9fa; border-radius: 4px; }}
        .check.warning {{ border-left-color: #ffc107; }}
        .check.error {{ border-left-color: #dc3545; }}
        .check.critical {{ border-left-color: #721c24; }}
        .check-name {{ font-weight: bold; margin-bottom: 5px; }}
        .details {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
        pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>QuantRS2 Health Check Report</h1>
        <div class="status">Status: {status.overall_health.value.upper()}</div>
        <p><strong>Timestamp:</strong> {status.timestamp}</p>

        <h2>Summary</h2>
        <div class="summary">
"""

    summary = status.to_dict()['summary']
    html += f"""
            <div class="summary-card">
                <div style="font-size: 32px; font-weight: bold;">{summary['total_checks']}</div>
                <div>Total Checks</div>
            </div>
            <div class="summary-card">
                <div style="font-size: 32px; font-weight: bold; color: #28a745;">{summary['healthy']}</div>
                <div>Healthy</div>
            </div>
            <div class="summary-card">
                <div style="font-size: 32px; font-weight: bold; color: #ffc107;">{summary['warnings']}</div>
                <div>Warnings</div>
            </div>
            <div class="summary-card">
                <div style="font-size: 32px; font-weight: bold; color: #dc3545;">{summary['errors']}</div>
                <div>Errors</div>
            </div>
"""

    html += """
        </div>

        <h2>Check Results</h2>
"""

    for check in status.checks:
        css_class = check.level.value if check.level != HealthLevel.HEALTHY else "healthy"
        html += f"""
        <div class="check {css_class}">
            <div class="check-name">{check.name}</div>
            <div>{check.message}</div>
"""
        if check.details:
            html += f"""
            <div class="details">
                <pre>{json.dumps(check.details, indent=2)}</pre>
            </div>
"""
        html += """
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    return html


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QuantRS2 Health Check")
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--export', type=str, help='Export to file')
    parser.add_argument('--format', type=str, choices=['json', 'text', 'html'],
                       default='json', help='Export format')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    if args.export:
        export_health_check(args.export, args.format)
    else:
        status = run_health_check()

        if args.json:
            print(status.to_json())
        else:
            print_health_status(status, verbose=args.verbose)

        # Exit with appropriate code
        if status.overall_health == HealthLevel.CRITICAL:
            sys.exit(2)
        elif status.overall_health == HealthLevel.ERROR:
            sys.exit(1)
        else:
            sys.exit(0)
