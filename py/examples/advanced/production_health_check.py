#!/usr/bin/env python3
"""
Production Health Check Example

This example demonstrates how to use the health check module for
production monitoring and system diagnostics.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from quantrs2.health_check import (
    run_health_check,
    print_health_status,
    export_health_check,
    HealthLevel
)


def main():
    """Run health check demonstrations."""

    print("\n" + "="*80)
    print("QUANTRS2 PRODUCTION HEALTH CHECK EXAMPLES")
    print("="*80 + "\n")

    # Example 1: Basic health check
    print("Example 1: Basic Health Check")
    print("-" * 80)
    status = run_health_check()
    print_health_status(status)

    # Example 2: Verbose health check
    print("\nExample 2: Verbose Health Check (with details)")
    print("-" * 80)
    status = run_health_check()
    print_health_status(status, verbose=True)

    # Example 3: JSON export
    print("\nExample 3: Export to JSON")
    print("-" * 80)
    import tempfile
    import os

    json_file = os.path.join(tempfile.gettempdir(), 'quantrs2_health.json')
    export_health_check(json_file, format='json')
    print(f"✅ Health check exported to: {json_file}")

    # Show JSON content
    with open(json_file, 'r') as f:
        print("\nJSON content (first 500 chars):")
        content = f.read()
        print(content[:500] + "..." if len(content) > 500 else content)

    # Example 4: HTML report
    print("\nExample 4: Export to HTML Report")
    print("-" * 80)
    html_file = os.path.join(tempfile.gettempdir(), 'quantrs2_health.html')
    export_health_check(html_file, format='html')
    print(f"✅ HTML report exported to: {html_file}")
    print(f"   Open in browser: file://{html_file}")

    # Example 5: Check specific issues
    print("\nExample 5: Check for Issues")
    print("-" * 80)
    issues = status.get_issues()

    if issues:
        print(f"⚠️  Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue.name}: {issue.message}")
    else:
        print("✅ No issues found!")

    # Example 6: Get checks by level
    print("\nExample 6: Checks by Severity Level")
    print("-" * 80)

    for level in [HealthLevel.CRITICAL, HealthLevel.ERROR, HealthLevel.WARNING, HealthLevel.HEALTHY]:
        checks = status.get_by_level(level)
        if checks:
            print(f"\n{level.value.upper()} ({len(checks)}):")
            for check in checks:
                print(f"  - {check.name}: {check.message}")

    # Example 7: Programmatic health check
    print("\nExample 7: Programmatic Health Check")
    print("-" * 80)

    status = run_health_check()

    # Check overall health
    if status.is_healthy():
        print("✅ System is healthy!")
        exit_code = 0
    else:
        print(f"⚠️  System health: {status.overall_health.value}")

        # Get issues
        critical = status.get_by_level(HealthLevel.CRITICAL)
        errors = status.get_by_level(HealthLevel.ERROR)
        warnings = status.get_by_level(HealthLevel.WARNING)

        if critical:
            print(f"🚨 {len(critical)} critical issue(s)")
            exit_code = 2
        elif errors:
            print(f"❌ {len(errors)} error(s)")
            exit_code = 1
        else:
            print(f"⚠️  {len(warnings)} warning(s)")
            exit_code = 0

    # Example 8: Integration with monitoring systems
    print("\nExample 8: Integration with Monitoring (Simulated)")
    print("-" * 80)

    # Simulate sending to monitoring system
    def send_to_monitoring(status):
        """Simulate sending health check to monitoring system."""
        metrics = {
            'quantrs2_health_total_checks': len(status.checks),
            'quantrs2_health_healthy': len(status.get_by_level(HealthLevel.HEALTHY)),
            'quantrs2_health_warnings': len(status.get_by_level(HealthLevel.WARNING)),
            'quantrs2_health_errors': len(status.get_by_level(HealthLevel.ERROR)),
            'quantrs2_health_critical': len(status.get_by_level(HealthLevel.CRITICAL)),
            'quantrs2_health_status': 1 if status.is_healthy() else 0,
        }

        print("Metrics to send to monitoring system:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # In production, you would send these to:
        # - Prometheus
        # - Datadog
        # - CloudWatch
        # - Custom monitoring solution

        return metrics

    metrics = send_to_monitoring(status)

    # Summary
    print("\n" + "="*80)
    print("HEALTH CHECK EXAMPLES COMPLETED")
    print("="*80)
    print(f"\nOverall Status: {status.overall_health.value.upper()}")
    print(f"Exit Code: {exit_code}")
    print("\nUse these examples to:")
    print("  - Monitor production deployments")
    print("  - Integrate with CI/CD pipelines")
    print("  - Diagnose system issues")
    print("  - Track system health over time")
    print("\n")


if __name__ == "__main__":
    main()
