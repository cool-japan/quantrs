"""
Tests for Monitoring and Alerting System

This module tests the comprehensive monitoring, alerting, and notification
capabilities including metrics collection, alert rules, and external integrations.
"""

import pytest
import time
import json
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from quantrs2.monitoring_alerting import (
    MonitoringSystem, AlertManager, NotificationManager, MetricsCollector,
    AlertRule, Alert, NotificationConfig, MetricDataPoint,
    AlertSeverity, AlertStatus, NotificationChannel, MetricType
)
from quantrs2.external_monitoring_integrations import (
    ExternalMonitoringManager, IntegrationConfig, IntegrationType
)


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    def test_metrics_collector_creation(self):
        """Test creating metrics collector."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            collector = MetricsCollector(tmp.name)
            
            try:
                assert collector.storage_path == tmp.name
                assert len(collector._metrics_buffer) == 0
                
            finally:
                collector.close()
    
    def test_record_metric(self):
        """Test recording metrics."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            collector = MetricsCollector(tmp.name)
            
            try:
                # Record a metric
                collector.record_metric("test.metric", 42.5, tags={"env": "test"})
                
                # Check it's in buffer
                recent = collector.get_recent_metrics("test.metric", 60)
                assert len(recent) == 1
                assert recent[0].value == 42.5
                assert recent[0].tags["env"] == "test"
                
            finally:
                collector.close()
    
    def test_recent_metrics_window(self):
        """Test recent metrics time window filtering."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            collector = MetricsCollector(tmp.name)
            
            try:
                # Record metrics at different times
                now = time.time()
                
                # Recent metric
                collector._metrics_buffer["test.metric"].append(
                    MetricDataPoint(timestamp=now, value=1.0)
                )
                
                # Old metric
                collector._metrics_buffer["test.metric"].append(
                    MetricDataPoint(timestamp=now - 600, value=2.0)  # 10 minutes ago
                )
                
                # Get recent metrics (last 5 minutes)
                recent = collector.get_recent_metrics("test.metric", 300)
                
                # Should only have the recent one
                assert len(recent) == 1
                assert recent[0].value == 1.0
                
            finally:
                collector.close()
    
    def test_metrics_persistence(self):
        """Test metrics persistence to database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            collector = MetricsCollector(tmp.name)
            
            try:
                # Record metric
                collector.record_metric("persist.test", 123.45)
                
                # Force persistence
                collector._persist_buffered_metrics()
                
                # Get historical data
                start_time = time.time() - 60
                end_time = time.time() + 60
                historical = collector.get_historical_metrics("persist.test", start_time, end_time)
                
                assert len(historical) == 1
                assert historical[0].value == 123.45
                
            finally:
                collector.close()


class TestNotificationManager:
    """Test notification management."""
    
    @pytest.fixture
    def notification_manager(self):
        """Create notification manager for testing."""
        return NotificationManager()
    
    def test_register_channel(self, notification_manager):
        """Test registering notification channels."""
        config = NotificationConfig(
            channel_type=NotificationChannel.EMAIL,
            name="test_email",
            settings={
                "smtp_host": "smtp.test.com",
                "from_address": "test@test.com",
                "to_addresses": ["admin@test.com"]
            }
        )
        
        notification_manager.register_channel(config)
        
        assert "test_email" in notification_manager.channels
        assert notification_manager.channels["test_email"].channel_type == NotificationChannel.EMAIL
    
    def test_channel_filtering(self, notification_manager):
        """Test alert filtering for channels."""
        # Register channel with severity filter
        config = NotificationConfig(
            channel_type=NotificationChannel.WEBHOOK,
            name="critical_only",
            severity_filter=[AlertSeverity.CRITICAL],
            settings={"url": "http://test.com/webhook"}
        )
        notification_manager.register_channel(config)
        
        # Create alerts with different severities
        critical_alert = Alert(
            id="alert1",
            rule_id="rule1",
            rule_name="Critical Alert",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.ACTIVE,
            message="Critical issue",
            description="Test critical alert",
            metric_name="test.metric",
            metric_value=100,
            threshold_value=90,
            triggered_at=time.time()
        )
        
        low_alert = Alert(
            id="alert2", 
            rule_id="rule2",
            rule_name="Low Alert",
            severity=AlertSeverity.LOW,
            status=AlertStatus.ACTIVE,
            message="Low issue",
            description="Test low alert",
            metric_name="test.metric",
            metric_value=10,
            threshold_value=20,
            triggered_at=time.time()
        )
        
        # Test filtering
        critical_channels = notification_manager._get_eligible_channels(critical_alert)
        low_channels = notification_manager._get_eligible_channels(low_alert)
        
        assert "critical_only" in critical_channels
        assert "critical_only" not in low_channels
    
    def test_rate_limiting(self, notification_manager):
        """Test notification rate limiting."""
        config = NotificationConfig(
            channel_type=NotificationChannel.WEBHOOK,
            name="rate_limited",
            rate_limit_period=60,
            max_notifications_per_period=2,
            settings={"url": "http://test.com/webhook"}
        )
        notification_manager.register_channel(config)
        
        # First notification should pass
        assert notification_manager._check_rate_limit("rate_limited", config)
        
        # Second notification should pass
        assert notification_manager._check_rate_limit("rate_limited", config)
        
        # Third notification should be rate limited
        assert not notification_manager._check_rate_limit("rate_limited", config)


class TestAlertManager:
    """Test alert management."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create alert manager for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            metrics_collector = MetricsCollector(tmp.name)
            notification_manager = NotificationManager()
            
            manager = AlertManager(metrics_collector, notification_manager)
            
            yield manager
            
            manager.close()
            metrics_collector.close()
    
    def test_add_alert_rule(self, alert_manager):
        """Test adding alert rules."""
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
        
        alert_manager.add_rule(rule)
        
        assert "test_rule" in alert_manager.alert_rules
        assert alert_manager.alert_rules["test_rule"].name == "Test Rule"
    
    def test_threshold_evaluation(self, alert_manager):
        """Test threshold evaluation logic."""
        # Test different comparison operators
        assert alert_manager._evaluate_threshold(95, 90, ">")
        assert not alert_manager._evaluate_threshold(85, 90, ">")
        
        assert alert_manager._evaluate_threshold(85, 90, "<")
        assert not alert_manager._evaluate_threshold(95, 90, "<")
        
        assert alert_manager._evaluate_threshold(90, 90, ">=")
        assert alert_manager._evaluate_threshold(95, 90, ">=")
        
        assert alert_manager._evaluate_threshold(90, 90, "<=")
        assert alert_manager._evaluate_threshold(85, 90, "<=")
        
        assert alert_manager._evaluate_threshold(90, 90, "==")
        assert not alert_manager._evaluate_threshold(95, 90, "==")
    
    def test_alert_acknowledgment(self, alert_manager):
        """Test alert acknowledgment."""
        # Create a test alert
        alert = Alert(
            id="test_alert",
            rule_id="test_rule",
            rule_name="Test Alert",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            message="Test message",
            description="Test alert",
            metric_name="test.metric",
            metric_value=95,
            threshold_value=90,
            triggered_at=time.time()
        )
        
        alert_manager.active_alerts["test_alert"] = alert
        
        # Acknowledge the alert
        success = alert_manager.acknowledge_alert("test_alert", "test_user")
        
        assert success
        assert alert_manager.active_alerts["test_alert"].status == AlertStatus.ACKNOWLEDGED
        assert alert_manager.active_alerts["test_alert"].acknowledged_by == "test_user"
        assert alert_manager.active_alerts["test_alert"].acknowledged_at is not None
    
    def test_alert_resolution(self, alert_manager):
        """Test alert resolution."""
        # Create a test alert
        alert = Alert(
            id="test_alert",
            rule_id="test_rule", 
            rule_name="Test Alert",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            message="Test message",
            description="Test alert",
            metric_name="test.metric",
            metric_value=95,
            threshold_value=90,
            triggered_at=time.time()
        )
        
        alert_manager.active_alerts["test_alert"] = alert
        
        # Resolve the alert
        success = alert_manager.resolve_alert("test_alert", "test_user")
        
        assert success
        assert "test_alert" not in alert_manager.active_alerts
        assert len(alert_manager.alert_history) == 1
        assert alert_manager.alert_history[0].status == AlertStatus.RESOLVED


class TestMonitoringSystem:
    """Test complete monitoring system."""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create monitoring system for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = MonitoringSystem(tmpdir)
            
            yield system
            
            system.close()
    
    def test_monitoring_system_creation(self, monitoring_system):
        """Test monitoring system initialization."""
        assert monitoring_system.metrics_collector is not None
        assert monitoring_system.notification_manager is not None
        assert monitoring_system.alert_manager is not None
        assert monitoring_system.monitoring_enabled
    
    def test_record_application_metric(self, monitoring_system):
        """Test recording application metrics."""
        monitoring_system.record_application_metric("circuit.executions", 5, {"pattern": "bell"})
        
        recent = monitoring_system.metrics_collector.get_recent_metrics("app.circuit.executions", 60)
        assert len(recent) == 1
        assert recent[0].value == 5
        assert recent[0].tags["pattern"] == "bell"
        assert recent[0].tags["type"] == "application"
    
    def test_create_default_alerts(self, monitoring_system):
        """Test creating default alert rules."""
        monitoring_system.create_default_alerts()
        
        rules = monitoring_system.alert_manager.alert_rules
        assert len(rules) > 0
        
        # Check for expected default rules
        rule_names = [rule.name for rule in rules.values()]
        assert "High CPU Usage" in rule_names
        assert "High Memory Usage" in rule_names
        assert "Low Disk Space" in rule_names
    
    def test_system_health_status(self, monitoring_system):
        """Test system health assessment."""
        health = monitoring_system.get_system_health()
        
        assert "status" in health
        assert "alerts" in health
        assert "notifications" in health
        assert "monitoring_enabled" in health
        
        # Should be healthy with no alerts
        assert health["status"] == "healthy"
    
    @patch('quantrs2.monitoring_alerting.psutil')
    def test_system_metrics_collection(self, mock_psutil, monitoring_system):
        """Test system metrics collection."""
        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 75.5
        mock_psutil.virtual_memory.return_value = Mock(percent=60.0, available=1024*1024*1024)
        mock_psutil.disk_usage.return_value = Mock(used=500*1024*1024*1024, total=1000*1024*1024*1024)
        mock_psutil.net_io_counters.return_value = Mock(bytes_sent=1000000, bytes_recv=2000000)
        
        # Trigger metrics collection
        monitoring_system._collect_system_metrics()
        
        # Check that metrics were recorded
        cpu_metrics = monitoring_system.metrics_collector.get_recent_metrics("system.cpu_percent", 60)
        memory_metrics = monitoring_system.metrics_collector.get_recent_metrics("system.memory_percent", 60)
        
        assert len(cpu_metrics) >= 1
        assert len(memory_metrics) >= 1
        assert cpu_metrics[-1].value == 75.5
        assert memory_metrics[-1].value == 60.0


class TestExternalIntegrations:
    """Test external monitoring integrations."""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create monitoring system for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            system = MonitoringSystem(tmpdir)
            yield system
            system.close()
    
    def test_integration_manager_creation(self, monitoring_system):
        """Test external monitoring manager creation."""
        manager = ExternalMonitoringManager(monitoring_system)
        
        assert manager.monitoring_system is monitoring_system
        assert len(manager.integrations) == 0
    
    @patch('quantrs2.external_monitoring_integrations.PROMETHEUS_AVAILABLE', True)
    @patch('quantrs2.external_monitoring_integrations.CollectorRegistry')
    @patch('quantrs2.external_monitoring_integrations.Gauge')
    def test_prometheus_integration(self, mock_gauge, mock_registry, monitoring_system):
        """Test Prometheus integration."""
        manager = ExternalMonitoringManager(monitoring_system)
        
        config = IntegrationConfig(
            integration_type=IntegrationType.PROMETHEUS,
            name="test_prometheus",
            metric_prefix="quantrs2_test",
            metric_tags={"env": "test"},
            enabled=False  # Disable to prevent background thread
        )
        
        manager.add_integration(config)
        
        assert "test_prometheus" in manager.integrations
        prometheus_integration = manager.get_integration("test_prometheus")
        assert prometheus_integration is not None
    
    def test_datadog_integration_config(self, monitoring_system):
        """Test Datadog integration configuration."""
        manager = ExternalMonitoringManager(monitoring_system)
        
        config = IntegrationConfig(
            integration_type=IntegrationType.DATADOG,
            name="test_datadog",
            api_key="test_api_key",
            metric_prefix="quantrs2_test",
            enabled=False  # Disable to prevent API calls
        )
        
        manager.add_integration(config)
        
        assert "test_datadog" in manager.integrations
        datadog_integration = manager.get_integration("test_datadog")
        assert datadog_integration is not None
        assert datadog_integration.config.api_key == "test_api_key"
    
    def test_grafana_integration_config(self, monitoring_system):
        """Test Grafana integration configuration."""
        manager = ExternalMonitoringManager(monitoring_system)
        
        config = IntegrationConfig(
            integration_type=IntegrationType.GRAFANA,
            name="test_grafana",
            endpoint="http://grafana.test.com",
            api_key="test_api_key"
        )
        
        manager.add_integration(config)
        
        assert "test_grafana" in manager.integrations
        grafana_integration = manager.get_integration("test_grafana")
        assert grafana_integration is not None
        assert grafana_integration.api_url == "http://grafana.test.com"
    
    def test_integration_status(self, monitoring_system):
        """Test integration status reporting."""
        manager = ExternalMonitoringManager(monitoring_system)
        
        # Add test integration
        config = IntegrationConfig(
            integration_type=IntegrationType.GRAFANA,
            name="test_integration",
            endpoint="http://test.com"
        )
        manager.add_integration(config)
        
        status = manager.get_integration_status()
        
        assert "test_integration" in status
        assert status["test_integration"]["type"] == "GrafanaIntegration"
        assert status["test_integration"]["enabled"] is True


class TestIntegrationScenarios:
    """Test integrated monitoring scenarios."""
    
    def test_end_to_end_alerting_flow(self):
        """Test complete alerting workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create monitoring system
            monitoring_system = MonitoringSystem(tmpdir)
            
            try:
                # Create alert rule
                rule = AlertRule(
                    id="cpu_test",
                    name="CPU Test Alert",
                    description="Test CPU alert",
                    metric_name="system.cpu_percent",
                    metric_type=MetricType.SYSTEM,
                    severity=AlertSeverity.HIGH,
                    threshold_value=80.0,
                    comparison=">",
                    evaluation_window=60,
                    data_points_required=1
                )
                
                monitoring_system.alert_manager.add_rule(rule)
                
                # Record metric that should trigger alert
                monitoring_system.metrics_collector.record_metric(
                    "system.cpu_percent", 
                    90.0,
                    tags={"type": "system"}
                )
                
                # Force evaluation
                monitoring_system.alert_manager._evaluate_rule(rule)
                
                # Check if alert was created
                active_alerts = monitoring_system.alert_manager.get_active_alerts()
                assert len(active_alerts) == 1
                assert active_alerts[0].rule_name == "CPU Test Alert"
                assert active_alerts[0].severity == AlertSeverity.HIGH
                
            finally:
                monitoring_system.close()
    
    def test_notification_workflow(self):
        """Test notification workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitoring_system = MonitoringSystem(tmpdir)
            
            try:
                # Register webhook notification channel
                webhook_config = NotificationConfig(
                    channel_type=NotificationChannel.WEBHOOK,
                    name="test_webhook",
                    settings={"url": "http://test.com/webhook"},
                    severity_filter=[AlertSeverity.HIGH, AlertSeverity.CRITICAL]
                )
                
                monitoring_system.notification_manager.register_channel(webhook_config)
                
                # Create test alert
                alert = Alert(
                    id="test_notification",
                    rule_id="test_rule",
                    rule_name="Test Notification Alert",
                    severity=AlertSeverity.HIGH,
                    status=AlertStatus.ACTIVE,
                    message="Test notification",
                    description="Testing notification workflow",
                    metric_name="test.metric",
                    metric_value=100,
                    threshold_value=90,
                    triggered_at=time.time()
                )
                
                # Test channel eligibility
                eligible = monitoring_system.notification_manager._get_eligible_channels(alert)
                assert "test_webhook" in eligible
                
                # Test rate limiting
                assert monitoring_system.notification_manager._check_rate_limit("test_webhook", webhook_config)
                
            finally:
                monitoring_system.close()
    
    def test_metrics_persistence_and_retrieval(self):
        """Test metrics persistence and historical retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = str(Path(tmpdir) / "test_metrics.db")
            collector = MetricsCollector(storage_path)
            
            try:
                # Record multiple metrics over time
                base_time = time.time()
                
                for i in range(10):
                    timestamp = base_time + i * 60  # Every minute
                    collector._metrics_buffer["test.series"].append(
                        MetricDataPoint(
                            timestamp=timestamp,
                            value=float(i * 10),
                            tags={"iteration": str(i)}
                        )
                    )
                
                # Force persistence
                collector._persist_buffered_metrics()
                
                # Retrieve historical data
                historical = collector.get_historical_metrics(
                    "test.series",
                    base_time - 60,
                    base_time + 600
                )
                
                assert len(historical) == 10
                assert historical[0].value == 0.0
                assert historical[-1].value == 90.0
                
                # Test time window filtering
                recent = collector.get_historical_metrics(
                    "test.series",
                    base_time + 300,  # From 5 minutes in
                    base_time + 600   # To end
                )
                
                assert len(recent) == 5  # Last 5 data points
                
            finally:
                collector.close()


if __name__ == "__main__":
    pytest.main([__file__])