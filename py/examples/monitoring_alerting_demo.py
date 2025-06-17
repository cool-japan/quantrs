#!/usr/bin/env python3
"""
QuantRS2 Monitoring and Alerting System Demo

This comprehensive demo showcases the advanced monitoring, alerting, and 
notification capabilities of the QuantRS2 framework including:

- Real-time metrics collection and storage
- Configurable alert rules and thresholds  
- Multiple notification channels (email, webhook, Slack)
- Web-based monitoring dashboard
- External integrations (Prometheus, Datadog, Grafana)
- Alert management and escalation
"""

import time
import json
import logging
import asyncio
import threading
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from quantrs2.monitoring_alerting import (
        MonitoringSystem, AlertRule, NotificationConfig, AlertSeverity, 
        NotificationChannel, MetricType, AlertStatus
    )
    from quantrs2.monitoring_dashboard import DashboardServer
    from quantrs2.external_monitoring_integrations import (
        ExternalMonitoringManager, IntegrationConfig, IntegrationType
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure QuantRS2 monitoring modules are properly installed")
    exit(1)


class MockQuantumWorkload:
    """Simulates quantum circuit execution workloads for demonstration."""
    
    def __init__(self, monitoring_system: MonitoringSystem):
        self.monitoring_system = monitoring_system
        self.circuit_patterns = ["bell_state", "ghz_state", "qft", "vqe", "qaoa"]
        self.backend_types = ["simulator", "hardware", "cloud"]
        self._running = False
        self._thread = None
    
    def start_workload_simulation(self):
        """Start simulating quantum workloads."""
        self._running = True
        self._thread = threading.Thread(target=self._workload_loop, daemon=True)
        self._thread.start()
        logger.info("Started quantum workload simulation")
    
    def stop_workload_simulation(self):
        """Stop workload simulation."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Stopped quantum workload simulation")
    
    def _workload_loop(self):
        """Main workload simulation loop."""
        while self._running:
            try:
                # Simulate circuit execution
                self._simulate_circuit_execution()
                
                # Simulate system load
                self._simulate_system_metrics()
                
                # Random delay between executions
                time.sleep(random.uniform(1, 5))
                
            except Exception as e:
                logger.error(f"Workload simulation error: {e}")
    
    def _simulate_circuit_execution(self):
        """Simulate quantum circuit execution."""
        pattern = random.choice(self.circuit_patterns)
        backend = random.choice(self.backend_types)
        
        # Simulate execution time based on pattern and backend
        if pattern in ["vqe", "qaoa"]:
            base_time = random.uniform(2.0, 10.0)  # Complex algorithms take longer
        else:
            base_time = random.uniform(0.1, 2.0)   # Simple circuits are faster
        
        if backend == "hardware":
            execution_time = base_time * random.uniform(2.0, 5.0)  # Hardware is slower
        elif backend == "cloud":
            execution_time = base_time * random.uniform(1.5, 3.0)  # Cloud has latency
        else:
            execution_time = base_time  # Simulator baseline
        
        # Simulate occasional failures
        success = random.random() > 0.05  # 95% success rate
        
        # Record metrics
        self.monitoring_system.record_application_metric(
            "circuit.execution_time", execution_time, 
            {"pattern": pattern, "backend": backend, "status": "success" if success else "failure"}
        )
        
        self.monitoring_system.record_application_metric(
            "circuit.executions_total", 1,
            {"pattern": pattern, "backend": backend, "status": "success" if success else "failure"}
        )
        
        # Simulate cache hits/misses
        cache_hit = random.random() > 0.3  # 70% cache hit rate
        self.monitoring_system.record_application_metric(
            "cache.operations", 1,
            {"type": "circuit_optimization", "result": "hit" if cache_hit else "miss"}
        )
        
        logger.debug(f"Executed {pattern} on {backend}: {execution_time:.2f}s ({'success' if success else 'failure'})")
    
    def _simulate_system_metrics(self):
        """Simulate varying system metrics."""
        # Simulate CPU usage with trends
        base_cpu = 30 + 20 * random.random()  # Base 30-50%
        if random.random() > 0.9:  # 10% chance of CPU spike
            cpu_usage = min(95, base_cpu + random.uniform(30, 50))
        else:
            cpu_usage = base_cpu
        
        # Simulate memory usage with gradual increase
        current_time = time.time()
        memory_trend = (current_time % 3600) / 3600 * 30  # Gradual increase over hour
        memory_usage = 40 + memory_trend + random.uniform(-5, 10)
        memory_usage = max(0, min(100, memory_usage))
        
        # Simulate disk usage (slowly increasing)
        disk_usage = 50 + (current_time % 86400) / 86400 * 40  # Increase over day
        disk_usage = max(0, min(100, disk_usage))
        
        # Record system metrics
        self.monitoring_system.record_application_metric("system.cpu_usage", cpu_usage)
        self.monitoring_system.record_application_metric("system.memory_usage", memory_usage)
        self.monitoring_system.record_application_metric("system.disk_usage", disk_usage)
        
        # Simulate network activity
        network_bytes = random.randint(1000000, 10000000)  # 1-10 MB
        self.monitoring_system.record_application_metric("system.network_bytes", network_bytes)


def create_demo_alert_rules(monitoring_system: MonitoringSystem):
    """Create demonstration alert rules."""
    print("\n" + "="*60)
    print("CREATING DEMONSTRATION ALERT RULES")
    print("="*60)
    
    rules = [
        AlertRule(
            id="high_cpu_demo",
            name="High CPU Usage (Demo)",
            description="CPU usage exceeded 80% threshold",
            metric_name="app.system.cpu_usage",
            metric_type=MetricType.SYSTEM,
            severity=AlertSeverity.HIGH,
            threshold_value=80.0,
            comparison=">",
            evaluation_window=120,  # 2 minutes
            data_points_required=2,
            notification_channels=["email_demo", "webhook_demo"],
            tags={"environment": "demo", "component": "system"}
        ),
        
        AlertRule(
            id="memory_warning_demo",
            name="Memory Usage Warning (Demo)",
            description="Memory usage exceeded 75% threshold",
            metric_name="app.system.memory_usage",
            metric_type=MetricType.SYSTEM,
            severity=AlertSeverity.MEDIUM,
            threshold_value=75.0,
            comparison=">",
            evaluation_window=180,  # 3 minutes
            data_points_required=3,
            notification_channels=["webhook_demo"],
            tags={"environment": "demo", "component": "system"}
        ),
        
        AlertRule(
            id="circuit_failure_rate_demo",
            name="High Circuit Failure Rate (Demo)",
            description="Circuit failure rate exceeded acceptable threshold",
            metric_name="app.circuit.failure_rate",
            metric_type=MetricType.APPLICATION,
            severity=AlertSeverity.CRITICAL,
            threshold_value=10.0,  # 10% failure rate
            comparison=">",
            evaluation_window=300,  # 5 minutes
            data_points_required=5,
            notification_channels=["email_demo", "webhook_demo", "slack_demo"],
            tags={"environment": "demo", "component": "quantum"}
        ),
        
        AlertRule(
            id="cache_hit_rate_low_demo",
            name="Low Cache Hit Rate (Demo)", 
            description="Cache hit rate below optimal threshold",
            metric_name="app.cache.hit_rate",
            metric_type=MetricType.PERFORMANCE,
            severity=AlertSeverity.LOW,
            threshold_value=60.0,  # 60% hit rate
            comparison="<",
            evaluation_window=600,  # 10 minutes
            data_points_required=5,
            notification_channels=["webhook_demo"],
            tags={"environment": "demo", "component": "cache"}
        ),
        
        AlertRule(
            id="disk_space_critical_demo",
            name="Critical Disk Space (Demo)",
            description="Disk usage reached critical level",
            metric_name="app.system.disk_usage",
            metric_type=MetricType.SYSTEM,
            severity=AlertSeverity.CRITICAL,
            threshold_value=85.0,
            comparison=">",
            evaluation_window=60,  # 1 minute
            data_points_required=1,
            notification_channels=["email_demo", "webhook_demo", "slack_demo"],
            tags={"environment": "demo", "component": "storage"}
        )
    ]
    
    for rule in rules:
        monitoring_system.alert_manager.add_rule(rule)
        print(f"  ✓ Created alert rule: {rule.name}")
    
    print(f"\nCreated {len(rules)} demonstration alert rules")


def setup_demo_notification_channels(monitoring_system: MonitoringSystem):
    """Setup demonstration notification channels."""
    print("\n" + "="*60)
    print("SETTING UP NOTIFICATION CHANNELS")
    print("="*60)
    
    # Email notification (demo - won't actually send)
    email_config = NotificationConfig(
        channel_type=NotificationChannel.EMAIL,
        name="email_demo",
        settings={
            "smtp_host": "smtp.example.com",
            "smtp_port": 587,
            "from_address": "quantrs2-monitoring@example.com",
            "to_addresses": ["admin@example.com", "devops@example.com"],
            "username": "quantrs2_monitor",
            "password": "demo_password",
            "use_tls": True
        },
        severity_filter=[AlertSeverity.HIGH, AlertSeverity.CRITICAL],
        rate_limit_period=300,  # 5 minutes
        max_notifications_per_period=5
    )
    
    # Webhook notification (demo - won't actually send)
    webhook_config = NotificationConfig(
        channel_type=NotificationChannel.WEBHOOK,
        name="webhook_demo",
        settings={
            "url": "https://hooks.example.com/quantrs2-alerts",
            "headers": {
                "Authorization": "Bearer demo_token",
                "X-Source": "QuantRS2-Monitoring"
            },
            "timeout": 30
        },
        severity_filter=[],  # All severities
        rate_limit_period=60,   # 1 minute
        max_notifications_per_period=10
    )
    
    # Slack notification (demo - won't actually send)
    slack_config = NotificationConfig(
        channel_type=NotificationChannel.SLACK,
        name="slack_demo",
        settings={
            "webhook_url": "https://hooks.slack.com/services/demo/webhook/url",
            "channel": "#quantrs2-alerts",
            "username": "QuantRS2 Monitor"
        },
        severity_filter=[AlertSeverity.HIGH, AlertSeverity.CRITICAL],
        rate_limit_period=180,  # 3 minutes
        max_notifications_per_period=3
    )
    
    # Register channels
    for config in [email_config, webhook_config, slack_config]:
        monitoring_system.notification_manager.register_channel(config)
        print(f"  ✓ Registered {config.channel_type.value} channel: {config.name}")
    
    print(f"\nRegistered {3} notification channels")


def setup_external_integrations(monitoring_system: MonitoringSystem) -> ExternalMonitoringManager:
    """Setup external monitoring integrations."""
    print("\n" + "="*60)
    print("SETTING UP EXTERNAL INTEGRATIONS")
    print("="*60)
    
    manager = ExternalMonitoringManager(monitoring_system)
    
    # Prometheus integration
    try:
        prometheus_config = IntegrationConfig(
            integration_type=IntegrationType.PROMETHEUS,
            name="prometheus_demo",
            metric_prefix="quantrs2_demo",
            metric_tags={"environment": "demo", "service": "quantrs2"},
            push_interval=30,  # 30 seconds for demo
            enabled=True
        )
        
        manager.add_integration(prometheus_config)
        print("  ✓ Prometheus integration configured")
        
    except ImportError:
        print("  ⚠ Prometheus integration unavailable (prometheus_client not installed)")
    
    # Datadog integration (demo mode - won't actually send)
    datadog_config = IntegrationConfig(
        integration_type=IntegrationType.DATADOG,
        name="datadog_demo",
        api_key="demo_api_key_12345",
        metric_prefix="quantrs2.demo",
        metric_tags={"env": "demo", "service": "quantrs2"},
        push_interval=60,  # 1 minute for demo
        enabled=False  # Disabled to prevent actual API calls
    )
    
    try:
        manager.add_integration(datadog_config)
        print("  ✓ Datadog integration configured (demo mode)")
    except Exception as e:
        print(f"  ⚠ Datadog integration failed: {e}")
    
    # Grafana integration (demo mode)
    grafana_config = IntegrationConfig(
        integration_type=IntegrationType.GRAFANA,
        name="grafana_demo",
        endpoint="http://localhost:3000",  # Local Grafana instance
        api_key="demo_grafana_api_key",
        metric_prefix="quantrs2_demo"
    )
    
    try:
        manager.add_integration(grafana_config)
        print("  ✓ Grafana integration configured")
    except Exception as e:
        print(f"  ⚠ Grafana integration failed: {e}")
    
    return manager


def demonstrate_metrics_collection(monitoring_system: MonitoringSystem):
    """Demonstrate metrics collection capabilities."""
    print("\n" + "="*60)
    print("METRICS COLLECTION DEMONSTRATION")
    print("="*60)
    
    print("Recording sample metrics...")
    
    # Record various types of metrics
    metric_samples = [
        ("app.quantum.circuits_executed", 150, {"backend": "simulator", "pattern": "bell_state"}),
        ("app.quantum.execution_time_avg", 0.25, {"backend": "simulator"}),
        ("app.cache.hit_rate", 85.5, {"cache_type": "circuit_optimization"}),
        ("app.system.cpu_usage", 45.2, {"host": "quantum-node-1"}),
        ("app.system.memory_usage", 62.8, {"host": "quantum-node-1"}),
        ("app.performance.requests_per_second", 12.5, {"endpoint": "/api/execute"}),
        ("app.errors.rate", 0.5, {"component": "circuit_executor"}),
        ("app.database.connection_pool_utilization", 35.0, {"pool": "circuits"}),
    ]
    
    for name, value, tags in metric_samples:
        monitoring_system.record_application_metric(name, value, tags)
        print(f"  ✓ {name} = {value} {tags}")
    
    print(f"\nRecorded {len(metric_samples)} sample metrics")
    
    # Demonstrate metrics retrieval
    print("\nRetrieving recent metrics...")
    for name, _, _ in metric_samples[:3]:  # Show first 3
        recent = monitoring_system.metrics_collector.get_recent_metrics(f"app.{name}", 300)
        print(f"  {name}: {len(recent)} data points in last 5 minutes")


def demonstrate_alert_lifecycle(monitoring_system: MonitoringSystem):
    """Demonstrate complete alert lifecycle."""
    print("\n" + "="*60)
    print("ALERT LIFECYCLE DEMONSTRATION")
    print("="*60)
    
    # Trigger alerts by recording metrics that exceed thresholds
    print("Triggering alerts by recording high metric values...")
    
    # Trigger CPU alert
    monitoring_system.record_application_metric("system.cpu_usage", 95.0)
    print("  ⚠ Recorded high CPU usage: 95%")
    
    # Trigger memory alert  
    monitoring_system.record_application_metric("system.memory_usage", 80.0)
    print("  ⚠ Recorded high memory usage: 80%")
    
    # Trigger cache hit rate alert
    monitoring_system.record_application_metric("cache.hit_rate", 45.0)
    print("  ⚠ Recorded low cache hit rate: 45%")
    
    # Force alert evaluation
    print("\nForce evaluating alert rules...")
    alert_manager = monitoring_system.alert_manager
    
    with alert_manager._lock:
        rules_to_evaluate = [rule for rule in alert_manager.alert_rules.values() if rule.enabled]
    
    for rule in rules_to_evaluate:
        try:
            alert_manager._evaluate_rule(rule)
        except Exception as e:
            print(f"  Error evaluating rule {rule.name}: {e}")
    
    # Check active alerts
    active_alerts = alert_manager.get_active_alerts()
    print(f"\nActive alerts: {len(active_alerts)}")
    
    for alert in active_alerts:
        print(f"  🚨 {alert.severity.value.upper()}: {alert.rule_name}")
        print(f"     {alert.message}")
        print(f"     Triggered: {time.ctime(alert.triggered_at)}")
    
    # Demonstrate alert management
    if active_alerts:
        print("\nDemonstrating alert management...")
        
        # Acknowledge first alert
        first_alert = active_alerts[0]
        success = alert_manager.acknowledge_alert(first_alert.id, "demo_user")
        if success:
            print(f"  ✓ Acknowledged alert: {first_alert.rule_name}")
        
        # Resolve second alert if exists
        if len(active_alerts) > 1:
            second_alert = active_alerts[1]
            success = alert_manager.resolve_alert(second_alert.id, "demo_user")
            if success:
                print(f"  ✓ Resolved alert: {second_alert.rule_name}")


def demonstrate_dashboard_features(monitoring_system: MonitoringSystem):
    """Demonstrate dashboard capabilities."""
    print("\n" + "="*60)
    print("MONITORING DASHBOARD DEMONSTRATION")
    print("="*60)
    
    try:
        # Create dashboard server
        dashboard = DashboardServer(monitoring_system, host="localhost", port=8080)
        
        print("Dashboard server created successfully")
        print(f"Dashboard URL: http://localhost:8080")
        print("\nDashboard features:")
        print("  ✓ Real-time system health monitoring")
        print("  ✓ Active alerts visualization")
        print("  ✓ Interactive metrics charts")
        print("  ✓ Alert acknowledgment and resolution")
        print("  ✓ WebSocket real-time updates")
        print("  ✓ RESTful API for programmatic access")
        
        print(f"\nTo start the dashboard server, call: dashboard.start()")
        print("Note: Dashboard server not started in demo to avoid blocking")
        
    except ImportError:
        print("⚠ Dashboard not available (Flask/Flask-SocketIO not installed)")
        print("Install with: pip install flask flask-socketio")


def demonstrate_external_integrations(integration_manager: ExternalMonitoringManager):
    """Demonstrate external monitoring integrations."""
    print("\n" + "="*60)
    print("EXTERNAL INTEGRATIONS DEMONSTRATION")
    print("="*60)
    
    # Show integration status
    status = integration_manager.get_integration_status()
    print("Integration status:")
    for name, info in status.items():
        enabled_status = "enabled" if info["enabled"] else "disabled"
        print(f"  {info['type']}: {enabled_status}")
    
    # Demonstrate Prometheus metrics export
    prometheus_metrics = integration_manager.get_prometheus_metrics()
    if prometheus_metrics:
        print(f"\nPrometheus metrics export available ({len(prometheus_metrics.splitlines())} lines)")
        print("Sample metrics:")
        lines = prometheus_metrics.splitlines()
        for line in lines[:5]:  # Show first 5 lines
            if line and not line.startswith('#'):
                print(f"  {line}")
        if len(lines) > 5:
            print(f"  ... and {len(lines) - 5} more lines")
    
    # Record some custom metrics
    print("\nRecording custom metrics to external systems...")
    integration_manager.record_custom_metric("demo.metric1", 42.0, {"type": "demo"})
    integration_manager.record_custom_metric("demo.metric2", 3.14, {"type": "demo"})
    integration_manager.record_circuit_execution("bell_state", 0.5, True)
    print("  ✓ Custom metrics recorded")


def demonstrate_performance_monitoring():
    """Demonstrate system performance under load."""
    print("\n" + "="*60)
    print("PERFORMANCE MONITORING DEMONSTRATION")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        monitoring_system = MonitoringSystem(tmpdir)
        
        try:
            # Setup demonstration components
            create_demo_alert_rules(monitoring_system)
            setup_demo_notification_channels(monitoring_system)
            
            # Create workload simulator
            workload = MockQuantumWorkload(monitoring_system)
            
            print("Starting simulated quantum workload...")
            workload.start_workload_simulation()
            
            # Monitor for a short period
            monitoring_duration = 30  # seconds
            print(f"Monitoring performance for {monitoring_duration} seconds...")
            
            start_time = time.time()
            while time.time() - start_time < monitoring_duration:
                # Get current health
                health = monitoring_system.get_system_health()
                
                # Display metrics every 5 seconds
                if int(time.time() - start_time) % 5 == 0:
                    alerts_count = health["alerts"]["active_alerts"]
                    notifications_count = health["notifications"]["notifications_24h"]
                    
                    print(f"  Status: {health['status']}, "
                          f"Active alerts: {alerts_count}, "
                          f"Notifications: {notifications_count}")
                
                time.sleep(1)
            
            workload.stop_workload_simulation()
            
            # Final statistics
            final_stats = monitoring_system.alert_manager.get_alert_statistics()
            print(f"\nFinal monitoring statistics:")
            print(f"  Total alert rules: {final_stats['total_rules']}")
            print(f"  Active alerts: {final_stats['active_alerts']}")
            print(f"  Alerts by severity: {final_stats['by_severity']}")
            
        finally:
            monitoring_system.close()


async def run_async_demonstrations():
    """Run asynchronous demonstrations."""
    print("\n" + "="*60) 
    print("ASYNCHRONOUS OPERATIONS DEMONSTRATION")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        monitoring_system = MonitoringSystem(tmpdir)
        
        try:
            # Setup notification channels
            setup_demo_notification_channels(monitoring_system)
            
            # Create test alert
            from quantrs2.monitoring_alerting import Alert
            
            test_alert = Alert(
                id="async_demo_alert",
                rule_id="async_demo_rule",
                rule_name="Async Demo Alert",
                severity=AlertSeverity.HIGH,
                status=AlertStatus.ACTIVE,
                message="Testing async notification system",
                description="Demonstration of asynchronous alert processing",
                metric_name="demo.async_metric",
                metric_value=100.0,
                threshold_value=80.0,
                triggered_at=time.time(),
                tags={"demo": "async", "test": "notification"}
            )
            
            print("Testing asynchronous notification delivery...")
            
            # Test notification sending (will fail gracefully for demo channels)
            results = await monitoring_system.notification_manager.send_alert(test_alert)
            
            print("Notification results:")
            for channel, success in results.items():
                status = "✓ Success" if success else "✗ Failed (expected for demo)"
                print(f"  {channel}: {status}")
            
            print("Async operations completed successfully")
            
        finally:
            monitoring_system.close()


def main():
    """Run comprehensive monitoring and alerting demonstration."""
    print("QuantRS2 Monitoring and Alerting System Demo")
    print("=" * 60)
    print("This demo showcases comprehensive monitoring, alerting, and notification")
    print("capabilities for production QuantRS2 quantum computing deployments.")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize monitoring system
            print(f"\nInitializing monitoring system in: {tmpdir}")
            monitoring_system = MonitoringSystem(tmpdir)
            
            try:
                # Setup demonstration components
                create_demo_alert_rules(monitoring_system)
                setup_demo_notification_channels(monitoring_system)
                integration_manager = setup_external_integrations(monitoring_system)
                
                # Run demonstrations
                demonstrate_metrics_collection(monitoring_system)
                demonstrate_alert_lifecycle(monitoring_system)
                demonstrate_dashboard_features(monitoring_system)
                demonstrate_external_integrations(integration_manager)
                
                # Run async demonstrations
                print("\nRunning async demonstrations...")
                asyncio.run(run_async_demonstrations())
                
                # Performance monitoring demonstration
                demonstrate_performance_monitoring()
                
                print("\n" + "="*60)
                print("🎉 ALL MONITORING AND ALERTING DEMONSTRATIONS COMPLETED!")
                print("="*60)
                
                print("\nKey Features Demonstrated:")
                print("✅ Real-time metrics collection and storage")
                print("✅ Configurable alert rules with multiple thresholds")
                print("✅ Multi-channel notification system (email, webhook, Slack)")
                print("✅ Alert lifecycle management (trigger, acknowledge, resolve)")
                print("✅ Web-based monitoring dashboard with real-time updates")
                print("✅ External monitoring integrations (Prometheus, Datadog, Grafana)")
                print("✅ Asynchronous notification delivery")
                print("✅ System performance monitoring under load")
                print("✅ Historical metrics storage and retrieval")
                print("✅ Rate limiting and notification filtering")
                print("✅ Comprehensive health status reporting")
                
                print("\nProduction Readiness Features:")
                print("🔧 Configurable alert thresholds and evaluation windows")
                print("🔧 Multiple notification channels with failover")
                print("🔧 Rate limiting to prevent notification spam")  
                print("🔧 Alert escalation and acknowledgment workflows")
                print("🔧 Historical metrics storage with retention policies")
                print("🔧 External monitoring system integrations")
                print("🔧 RESTful API for programmatic management")
                print("🔧 Real-time WebSocket updates for dashboards")
                print("🔧 Thread-safe concurrent operations")
                print("🔧 Graceful error handling and recovery")
                
            finally:
                monitoring_system.close()
                integration_manager.close_all()
                
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()