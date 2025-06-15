#!/usr/bin/env python3
"""
Basic test for monitoring and alerting functionality

This tests core monitoring components that don't require external dependencies.
"""

import sys
import time
import tempfile
from pathlib import Path

# Add the Python path
sys.path.insert(0, str(Path(__file__).parent / 'python'))

from quantrs2.monitoring_alerting import (
    MetricsCollector, AlertManager, NotificationManager,
    AlertRule, AlertSeverity, MetricType, NotificationChannel,
    NotificationConfig, MonitoringSystem
)

def test_metrics_collector():
    """Test basic metrics collection."""
    print("Testing MetricsCollector...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        collector = MetricsCollector(tmp.name)
        
        try:
            # Record a metric
            collector.record_metric("test.metric", 42.5, tags={"env": "test"})
            
            # Get recent metrics
            recent = collector.get_recent_metrics("test.metric", 60)
            
            assert len(recent) == 1
            assert recent[0].value == 42.5
            assert recent[0].tags["env"] == "test"
            
            print("  ✓ Metrics collection working")
            
        finally:
            collector.close()

def test_alert_manager():
    """Test basic alert management."""
    print("Testing AlertManager...")
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        metrics_collector = MetricsCollector(tmp.name)
        notification_manager = NotificationManager()
        alert_manager = AlertManager(metrics_collector, notification_manager)
        
        try:
            # Create alert rule
            rule = AlertRule(
                id="test_rule",
                name="Test Rule",
                description="Test alert rule",
                metric_name="test.metric",
                metric_type=MetricType.SYSTEM,
                severity=AlertSeverity.HIGH,
                threshold_value=90.0,
                comparison=">"
            )
            
            # Add rule
            alert_manager.add_rule(rule)
            
            assert "test_rule" in alert_manager.alert_rules
            assert alert_manager.alert_rules["test_rule"].name == "Test Rule"
            
            print("  ✓ Alert rule creation working")
            
            # Test threshold evaluation
            assert alert_manager._evaluate_threshold(95, 90, ">")
            assert not alert_manager._evaluate_threshold(85, 90, ">")
            
            print("  ✓ Threshold evaluation working")
            
        finally:
            alert_manager.close()
            metrics_collector.close()

def test_notification_manager():
    """Test basic notification management."""
    print("Testing NotificationManager...")
    
    notification_manager = NotificationManager()
    
    # Register channel (webhook - doesn't actually send)
    config = NotificationConfig(
        channel_type=NotificationChannel.WEBHOOK,
        name="test_webhook",
        settings={"url": "http://test.com/webhook"}
    )
    
    notification_manager.register_channel(config)
    
    assert "test_webhook" in notification_manager.channels
    assert notification_manager.channels["test_webhook"].channel_type == NotificationChannel.WEBHOOK
    
    print("  ✓ Notification channel registration working")

def test_monitoring_system():
    """Test complete monitoring system."""
    print("Testing MonitoringSystem...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        system = MonitoringSystem(tmpdir)
        
        try:
            # Record application metric
            system.record_application_metric("circuit.executions", 5, {"pattern": "bell"})
            
            # Get health status
            health = system.get_system_health()
            
            assert "status" in health
            assert "alerts" in health
            assert "notifications" in health
            
            print("  ✓ System health reporting working")
            
            # Get recent metrics
            recent = system.metrics_collector.get_recent_metrics("app.circuit.executions", 60)
            assert len(recent) == 1
            assert recent[0].value == 5
            assert recent[0].tags["pattern"] == "bell"
            
            print("  ✓ Application metrics recording working")
            
        finally:
            system.close()

def main():
    """Run basic monitoring tests."""
    print("QuantRS2 Basic Monitoring and Alerting Tests")
    print("=" * 50)
    
    try:
        test_metrics_collector()
        test_alert_manager()
        test_notification_manager() 
        test_monitoring_system()
        
        print("\n" + "=" * 50)
        print("🎉 ALL BASIC TESTS PASSED!")
        print("=" * 50)
        
        print("\nCore functionality validated:")
        print("✅ Metrics collection and storage")
        print("✅ Alert rule creation and evaluation")
        print("✅ Notification channel management")
        print("✅ Complete monitoring system integration")
        print("✅ System health reporting")
        print("✅ Application metrics recording")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)