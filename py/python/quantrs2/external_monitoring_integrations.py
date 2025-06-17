"""
External Monitoring System Integrations for QuantRS2

This module provides integrations with external monitoring and observability
platforms like Prometheus, Grafana, Datadog, New Relic, and others.
"""

import time
import json
import logging
import threading
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import weakref

try:
    from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from .monitoring_alerting import MonitoringSystem, MetricDataPoint, Alert, AlertSeverity

logger = logging.getLogger(__name__)


class IntegrationType(Enum):
    """Types of external monitoring integrations."""
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    ELASTIC = "elastic"
    SPLUNK = "splunk"
    STATSD = "statsd"


@dataclass
class IntegrationConfig:
    """Configuration for external monitoring integration."""
    integration_type: IntegrationType
    name: str
    enabled: bool = True
    
    # Connection settings
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    
    # Metric settings
    metric_prefix: str = "quantrs2"
    metric_tags: Dict[str, str] = field(default_factory=dict)
    push_interval: int = 60  # seconds
    
    # Filtering
    metric_filters: List[str] = field(default_factory=list)
    alert_filters: List[AlertSeverity] = field(default_factory=list)
    
    # Integration-specific settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class PrometheusIntegration:
    """Prometheus metrics integration."""
    
    def __init__(self, config: IntegrationConfig, monitoring_system: MonitoringSystem):
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client required for Prometheus integration")
        
        self.config = config
        self.monitoring_system = monitoring_system
        
        # Create custom registry
        self.registry = CollectorRegistry()
        
        # Create metric instances
        self.metrics = {}
        self._create_prometheus_metrics()
        
        # Background update thread
        self._update_thread = None
        self._shutdown = threading.Event()
        
        if config.enabled:
            self._start_update_thread()
        
        logger.info(f"Prometheus integration initialized: {config.name}")
    
    def _create_prometheus_metrics(self):
        """Create Prometheus metric instances."""
        prefix = self.config.metric_prefix
        labels = list(self.config.metric_tags.keys())
        
        # System metrics
        self.metrics['cpu_usage'] = Gauge(
            f'{prefix}_cpu_usage_percent',
            'CPU usage percentage',
            labels,
            registry=self.registry
        )
        
        self.metrics['memory_usage'] = Gauge(
            f'{prefix}_memory_usage_percent',
            'Memory usage percentage',
            labels,
            registry=self.registry
        )
        
        self.metrics['disk_usage'] = Gauge(
            f'{prefix}_disk_usage_percent',
            'Disk usage percentage',
            labels,
            registry=self.registry
        )
        
        # Application metrics
        self.metrics['active_alerts'] = Gauge(
            f'{prefix}_active_alerts_total',
            'Number of active alerts',
            labels + ['severity'],
            registry=self.registry
        )
        
        self.metrics['circuit_executions'] = Counter(
            f'{prefix}_circuit_executions_total',
            'Total circuit executions',
            labels + ['status', 'pattern'],
            registry=self.registry
        )
        
        self.metrics['execution_duration'] = Histogram(
            f'{prefix}_execution_duration_seconds',
            'Circuit execution duration',
            labels + ['pattern'],
            registry=self.registry
        )
        
        # Cache metrics
        self.metrics['cache_hits'] = Counter(
            f'{prefix}_cache_hits_total',
            'Cache hits',
            labels + ['cache_type'],
            registry=self.registry
        )
        
        self.metrics['cache_misses'] = Counter(
            f'{prefix}_cache_misses_total',
            'Cache misses',
            labels + ['cache_type'],
            registry=self.registry
        )
        
        # Connection pool metrics
        self.metrics['db_connections_active'] = Gauge(
            f'{prefix}_db_connections_active',
            'Active database connections',
            labels + ['pool'],
            registry=self.registry
        )
    
    def _start_update_thread(self):
        """Start background metrics update thread."""
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
    
    def _update_loop(self):
        """Background loop for updating Prometheus metrics."""
        while not self._shutdown.wait(self.config.push_interval):
            try:
                self._update_metrics()
            except Exception as e:
                logger.error(f"Prometheus metrics update failed: {e}")
    
    def _update_metrics(self):
        """Update Prometheus metrics with current values."""
        tags = self.config.metric_tags
        
        # Get system health
        health = self.monitoring_system.get_system_health()
        
        # Update system metrics
        if 'current_metrics' in health and health['current_metrics']:
            system_metrics = health['current_metrics'].get('system', {})
            
            self.metrics['cpu_usage'].labels(**tags).set(
                system_metrics.get('cpu_percent', 0)
            )
            
            self.metrics['memory_usage'].labels(**tags).set(
                system_metrics.get('memory_percent', 0)
            )
            
            self.metrics['disk_usage'].labels(**tags).set(
                system_metrics.get('disk_usage_percent', 0)
            )
        
        # Update alert metrics
        alert_stats = health.get('alerts', {})
        by_severity = alert_stats.get('by_severity', {})
        
        for severity, count in by_severity.items():
            self.metrics['active_alerts'].labels(**tags, severity=severity).set(count)
        
        # Update connection pool metrics
        try:
            conn_stats = self.monitoring_system.connection_manager.get_statistics()
            for pool_name, stats in conn_stats.items():
                self.metrics['db_connections_active'].labels(
                    **tags, pool=pool_name
                ).set(stats.get('connections_borrowed', 0))
        except Exception as e:
            logger.debug(f"Connection stats update failed: {e}")
        
        # Update cache metrics
        try:
            cache_stats = self.monitoring_system.cache_manager.get_statistics()
            
            for cache_type, stats in cache_stats.items():
                if isinstance(stats, dict):
                    hits = stats.get('hits', 0)
                    misses = stats.get('misses', 0)
                    
                    # Update counters (Prometheus counters are cumulative)
                    # We need to track the difference
                    cache_key = f"{cache_type}_hits"
                    if not hasattr(self, '_last_cache_values'):
                        self._last_cache_values = {}
                    
                    last_hits = self._last_cache_values.get(cache_key, 0)
                    if hits > last_hits:
                        self.metrics['cache_hits'].labels(
                            **tags, cache_type=cache_type
                        )._value._value += (hits - last_hits)
                    
                    self._last_cache_values[cache_key] = hits
                    
                    cache_key = f"{cache_type}_misses"
                    last_misses = self._last_cache_values.get(cache_key, 0)
                    if misses > last_misses:
                        self.metrics['cache_misses'].labels(
                            **tags, cache_type=cache_type
                        )._value._value += (misses - last_misses)
                    
                    self._last_cache_values[cache_key] = misses
                    
        except Exception as e:
            logger.debug(f"Cache stats update failed: {e}")
    
    def get_metrics_output(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry).decode('utf-8')
    
    def record_circuit_execution(self, pattern: str, duration: float, success: bool):
        """Record circuit execution metrics."""
        tags = self.config.metric_tags
        
        # Update execution counter
        status = "success" if success else "failure"
        self.metrics['circuit_executions'].labels(
            **tags, status=status, pattern=pattern
        ).inc()
        
        # Update duration histogram
        self.metrics['execution_duration'].labels(
            **tags, pattern=pattern
        ).observe(duration)
    
    def close(self):
        """Close Prometheus integration."""
        self._shutdown.set()
        if self._update_thread:
            self._update_thread.join(timeout=5)


class DatadogIntegration:
    """Datadog metrics integration."""
    
    def __init__(self, config: IntegrationConfig, monitoring_system: MonitoringSystem):
        self.config = config
        self.monitoring_system = monitoring_system
        
        # Validate configuration
        if not config.api_key:
            raise ValueError("Datadog API key required")
        
        self.api_url = config.endpoint or "https://api.datadoghq.com/api/v1/series"
        self.headers = {
            'Content-Type': 'application/json',
            'DD-API-KEY': config.api_key
        }
        
        # Metrics buffer
        self.metrics_buffer = []
        self._buffer_lock = threading.RLock()
        
        # Background push thread
        self._push_thread = None
        self._shutdown = threading.Event()
        
        if config.enabled:
            self._start_push_thread()
        
        logger.info(f"Datadog integration initialized: {config.name}")
    
    def _start_push_thread(self):
        """Start background metrics push thread."""
        self._push_thread = threading.Thread(target=self._push_loop, daemon=True)
        self._push_thread.start()
    
    def _push_loop(self):
        """Background loop for pushing metrics to Datadog."""
        while not self._shutdown.wait(self.config.push_interval):
            try:
                self._push_metrics()
            except Exception as e:
                logger.error(f"Datadog metrics push failed: {e}")
    
    def _push_metrics(self):
        """Push metrics to Datadog."""
        # Collect current metrics
        health = self.monitoring_system.get_system_health()
        timestamp = int(time.time())
        
        metrics_data = []
        base_tags = [f"{k}:{v}" for k, v in self.config.metric_tags.items()]
        
        # System metrics
        if 'current_metrics' in health and health['current_metrics']:
            system_metrics = health['current_metrics'].get('system', {})
            
            for metric_name, value in system_metrics.items():
                metrics_data.append({
                    'metric': f"{self.config.metric_prefix}.system.{metric_name}",
                    'points': [[timestamp, value]],
                    'tags': base_tags + ['source:quantrs2']
                })
        
        # Alert metrics
        alert_stats = health.get('alerts', {})
        metrics_data.append({
            'metric': f"{self.config.metric_prefix}.alerts.active",
            'points': [[timestamp, alert_stats.get('active_alerts', 0)]],
            'tags': base_tags + ['source:quantrs2']
        })
        
        for severity, count in alert_stats.get('by_severity', {}).items():
            metrics_data.append({
                'metric': f"{self.config.metric_prefix}.alerts.by_severity",
                'points': [[timestamp, count]],
                'tags': base_tags + [f'severity:{severity}', 'source:quantrs2']
            })
        
        # Send to Datadog
        if metrics_data:
            self._send_to_datadog({'series': metrics_data})
    
    def _send_to_datadog(self, payload: Dict[str, Any]):
        """Send metrics payload to Datadog."""
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            logger.debug(f"Sent {len(payload['series'])} metrics to Datadog")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send metrics to Datadog: {e}")
    
    def record_custom_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record custom metric to Datadog."""
        timestamp = int(time.time())
        metric_tags = [f"{k}:{v}" for k, v in self.config.metric_tags.items()]
        
        if tags:
            metric_tags.extend([f"{k}:{v}" for k, v in tags.items()])
        
        metric_data = {
            'metric': f"{self.config.metric_prefix}.{name}",
            'points': [[timestamp, value]],
            'tags': metric_tags + ['source:quantrs2']
        }
        
        # Buffer for batch sending
        with self._buffer_lock:
            self.metrics_buffer.append(metric_data)
            
            # Send if buffer is full
            if len(self.metrics_buffer) >= 100:
                self._send_to_datadog({'series': self.metrics_buffer.copy()})
                self.metrics_buffer.clear()
    
    def close(self):
        """Close Datadog integration."""
        self._shutdown.set()
        if self._push_thread:
            self._push_thread.join(timeout=5)
        
        # Send remaining buffered metrics
        with self._buffer_lock:
            if self.metrics_buffer:
                self._send_to_datadog({'series': self.metrics_buffer})


class GrafanaIntegration:
    """Grafana dashboard integration."""
    
    def __init__(self, config: IntegrationConfig, monitoring_system: MonitoringSystem):
        self.config = config
        self.monitoring_system = monitoring_system
        
        if not config.endpoint:
            raise ValueError("Grafana endpoint required")
        
        self.api_url = config.endpoint.rstrip('/')
        self.headers = {
            'Content-Type': 'application/json'
        }
        
        if config.api_key:
            self.headers['Authorization'] = f'Bearer {config.api_key}'
        
        logger.info(f"Grafana integration initialized: {config.name}")
    
    def create_quantrs2_dashboard(self) -> Dict[str, Any]:
        """Create a comprehensive Grafana dashboard for QuantRS2."""
        dashboard = {
            'dashboard': {
                'id': None,
                'uid': 'quantrs2-monitoring',
                'title': 'QuantRS2 Monitoring Dashboard',
                'description': 'Comprehensive monitoring for QuantRS2 quantum computing framework',
                'tags': ['quantrs2', 'quantum', 'monitoring'],
                'timezone': 'browser',
                'refresh': '30s',
                'time': {
                    'from': 'now-1h',
                    'to': 'now'
                },
                'panels': self._create_dashboard_panels()
            },
            'overwrite': True
        }
        
        return dashboard
    
    def _create_dashboard_panels(self) -> List[Dict[str, Any]]:
        """Create dashboard panels."""
        panels = []
        panel_id = 1
        
        # System overview row
        panels.append({
            'id': panel_id,
            'title': 'System Overview',
            'type': 'row',
            'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': 0}
        })
        panel_id += 1
        
        # CPU usage panel
        panels.append({
            'id': panel_id,
            'title': 'CPU Usage',
            'type': 'stat',
            'targets': [{
                'expr': f'{self.config.metric_prefix}_cpu_usage_percent',
                'refId': 'A'
            }],
            'fieldConfig': {
                'defaults': {
                    'unit': 'percent',
                    'min': 0,
                    'max': 100,
                    'thresholds': {
                        'steps': [
                            {'color': 'green', 'value': 0},
                            {'color': 'yellow', 'value': 70},
                            {'color': 'red', 'value': 90}
                        ]
                    }
                }
            },
            'gridPos': {'h': 8, 'w': 6, 'x': 0, 'y': 1}
        })
        panel_id += 1
        
        # Memory usage panel
        panels.append({
            'id': panel_id,
            'title': 'Memory Usage',
            'type': 'stat',
            'targets': [{
                'expr': f'{self.config.metric_prefix}_memory_usage_percent',
                'refId': 'A'
            }],
            'fieldConfig': {
                'defaults': {
                    'unit': 'percent',
                    'min': 0,
                    'max': 100,
                    'thresholds': {
                        'steps': [
                            {'color': 'green', 'value': 0},
                            {'color': 'yellow', 'value': 70},
                            {'color': 'red', 'value': 85}
                        ]
                    }
                }
            },
            'gridPos': {'h': 8, 'w': 6, 'x': 6, 'y': 1}
        })
        panel_id += 1
        
        # Active alerts panel
        panels.append({
            'id': panel_id,
            'title': 'Active Alerts',
            'type': 'stat',
            'targets': [{
                'expr': f'sum({self.config.metric_prefix}_active_alerts_total)',
                'refId': 'A'
            }],
            'fieldConfig': {
                'defaults': {
                    'color': {'mode': 'thresholds'},
                    'thresholds': {
                        'steps': [
                            {'color': 'green', 'value': 0},
                            {'color': 'yellow', 'value': 1},
                            {'color': 'red', 'value': 5}
                        ]
                    }
                }
            },
            'gridPos': {'h': 8, 'w': 6, 'x': 12, 'y': 1}
        })
        panel_id += 1
        
        # Circuit executions panel
        panels.append({
            'id': panel_id,
            'title': 'Circuit Executions',
            'type': 'stat',
            'targets': [{
                'expr': f'rate({self.config.metric_prefix}_circuit_executions_total[5m])',
                'refId': 'A'
            }],
            'fieldConfig': {
                'defaults': {
                    'unit': 'ops',
                    'decimals': 2
                }
            },
            'gridPos': {'h': 8, 'w': 6, 'x': 18, 'y': 1}
        })
        panel_id += 1
        
        # Performance metrics row
        panels.append({
            'id': panel_id,
            'title': 'Performance Metrics',
            'type': 'row',
            'gridPos': {'h': 1, 'w': 24, 'x': 0, 'y': 9}
        })
        panel_id += 1
        
        # Execution duration histogram
        panels.append({
            'id': panel_id,
            'title': 'Circuit Execution Duration',
            'type': 'graph',
            'targets': [{
                'expr': f'histogram_quantile(0.95, {self.config.metric_prefix}_execution_duration_seconds_bucket)',
                'legendFormat': '95th percentile',
                'refId': 'A'
            }, {
                'expr': f'histogram_quantile(0.50, {self.config.metric_prefix}_execution_duration_seconds_bucket)',
                'legendFormat': '50th percentile',
                'refId': 'B'
            }],
            'yAxes': [{
                'unit': 's',
                'min': 0
            }],
            'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 10}
        })
        panel_id += 1
        
        # Cache hit rates
        panels.append({
            'id': panel_id,
            'title': 'Cache Hit Rates',
            'type': 'graph',
            'targets': [{
                'expr': f'rate({self.config.metric_prefix}_cache_hits_total[5m]) / (rate({self.config.metric_prefix}_cache_hits_total[5m]) + rate({self.config.metric_prefix}_cache_misses_total[5m]))',
                'legendFormat': 'Hit Rate - {{cache_type}}',
                'refId': 'A'
            }],
            'yAxes': [{
                'unit': 'percentunit',
                'min': 0,
                'max': 1
            }],
            'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 10}
        })
        
        return panels
    
    def deploy_dashboard(self) -> bool:
        """Deploy dashboard to Grafana."""
        try:
            dashboard_config = self.create_quantrs2_dashboard()
            
            response = requests.post(
                f"{self.api_url}/api/dashboards/db",
                headers=self.headers,
                json=dashboard_config,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Grafana dashboard deployed: {result.get('url', 'Unknown URL')}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to deploy Grafana dashboard: {e}")
            return False
    
    def get_dashboard_url(self) -> Optional[str]:
        """Get URL for the QuantRS2 dashboard."""
        return f"{self.api_url}/d/quantrs2-monitoring/quantrs2-monitoring-dashboard"


class ExternalMonitoringManager:
    """Manager for external monitoring integrations."""
    
    def __init__(self, monitoring_system: MonitoringSystem):
        self.monitoring_system = monitoring_system
        self.integrations: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
        logger.info("External monitoring manager initialized")
    
    def add_integration(self, config: IntegrationConfig):
        """Add external monitoring integration."""
        with self._lock:
            try:
                if config.integration_type == IntegrationType.PROMETHEUS:
                    integration = PrometheusIntegration(config, self.monitoring_system)
                elif config.integration_type == IntegrationType.DATADOG:
                    integration = DatadogIntegration(config, self.monitoring_system)
                elif config.integration_type == IntegrationType.GRAFANA:
                    integration = GrafanaIntegration(config, self.monitoring_system)
                else:
                    raise ValueError(f"Unsupported integration type: {config.integration_type}")
                
                self.integrations[config.name] = integration
                logger.info(f"Added integration: {config.name} ({config.integration_type.value})")
                
            except Exception as e:
                logger.error(f"Failed to add integration {config.name}: {e}")
                raise
    
    def remove_integration(self, name: str):
        """Remove external monitoring integration."""
        with self._lock:
            if name in self.integrations:
                integration = self.integrations[name]
                if hasattr(integration, 'close'):
                    integration.close()
                
                del self.integrations[name]
                logger.info(f"Removed integration: {name}")
    
    def get_integration(self, name: str) -> Optional[Any]:
        """Get integration by name."""
        with self._lock:
            return self.integrations.get(name)
    
    def get_prometheus_metrics(self, integration_name: str = None) -> Optional[str]:
        """Get Prometheus metrics output."""
        with self._lock:
            if integration_name:
                integration = self.integrations.get(integration_name)
                if isinstance(integration, PrometheusIntegration):
                    return integration.get_metrics_output()
            else:
                # Return from first Prometheus integration found
                for integration in self.integrations.values():
                    if isinstance(integration, PrometheusIntegration):
                        return integration.get_metrics_output()
        
        return None
    
    def record_circuit_execution(self, pattern: str, duration: float, success: bool):
        """Record circuit execution across all integrations."""
        with self._lock:
            for integration in self.integrations.values():
                try:
                    if hasattr(integration, 'record_circuit_execution'):
                        integration.record_circuit_execution(pattern, duration, success)
                except Exception as e:
                    logger.error(f"Failed to record metric in {type(integration).__name__}: {e}")
    
    def record_custom_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record custom metric across all integrations."""
        with self._lock:
            for integration in self.integrations.values():
                try:
                    if hasattr(integration, 'record_custom_metric'):
                        integration.record_custom_metric(name, value, tags)
                except Exception as e:
                    logger.error(f"Failed to record custom metric in {type(integration).__name__}: {e}")
    
    def get_integration_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all integrations."""
        status = {}
        
        with self._lock:
            for name, integration in self.integrations.items():
                status[name] = {
                    'type': type(integration).__name__,
                    'enabled': getattr(integration.config, 'enabled', True),
                    'healthy': True  # Could add health checks here
                }
        
        return status
    
    def close_all(self):
        """Close all external monitoring integrations."""
        with self._lock:
            for name, integration in self.integrations.items():
                try:
                    if hasattr(integration, 'close'):
                        integration.close()
                except Exception as e:
                    logger.error(f"Failed to close integration {name}: {e}")
            
            self.integrations.clear()
            logger.info("All external monitoring integrations closed")


# Export main classes
__all__ = [
    'IntegrationType',
    'IntegrationConfig',
    'PrometheusIntegration',
    'DatadogIntegration', 
    'GrafanaIntegration',
    'ExternalMonitoringManager'
]