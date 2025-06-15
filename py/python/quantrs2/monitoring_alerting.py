"""
Comprehensive Monitoring and Alerting System for QuantRS2

This module provides advanced monitoring, alerting, and notification capabilities
for production QuantRS2 deployments with configurable thresholds, multiple
notification channels, and alert management.
"""

import time
import json
import logging
import asyncio
import threading
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import weakref

try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    # Create mock email classes
    class MimeText:
        def __init__(self, *args, **kwargs):
            pass
    class MimeMultipart:
        def __init__(self, *args, **kwargs):
            pass
        def __setitem__(self, key, value):
            pass
        def attach(self, *args):
            pass

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    # Create a mock requests module for graceful degradation
    class MockRequests:
        @staticmethod
        def post(*args, **kwargs):
            raise RuntimeError("requests module not available")
    requests = MockRequests()

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """Notification channel types."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    SMS = "sms"
    PAGERDUTY = "pagerduty"
    DISCORD = "discord"
    TEAMS = "teams"


class MetricType(Enum):
    """Types of metrics to monitor."""
    SYSTEM = "system"
    APPLICATION = "application"
    QUANTUM = "quantum"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS = "business"


@dataclass
class AlertRule:
    """Definition of an alert rule."""
    id: str
    name: str
    description: str
    metric_name: str
    metric_type: MetricType
    severity: AlertSeverity
    
    # Threshold configuration
    threshold_value: float
    comparison: str = ">"  # >, <, >=, <=, ==, !=
    evaluation_window: int = 300  # seconds
    data_points_required: int = 3
    
    # Notification settings
    notification_channels: List[str] = field(default_factory=list)
    cooldown_period: int = 900  # seconds
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass 
class Alert:
    """Active alert instance."""
    id: str
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    
    # Alert details
    message: str
    description: str
    metric_name: str
    metric_value: float
    threshold_value: float
    
    # Timing
    triggered_at: float
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    last_notified_at: Optional[float] = None
    
    # Context
    tags: Dict[str, str] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Management
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    escalation_level: int = 0
    notification_count: int = 0


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""
    channel_type: NotificationChannel
    name: str
    
    # Channel-specific settings
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # Filtering
    severity_filter: List[AlertSeverity] = field(default_factory=list)
    tag_filters: Dict[str, List[str]] = field(default_factory=dict)
    
    # Rate limiting
    rate_limit_period: int = 300  # seconds
    max_notifications_per_period: int = 10
    
    # Retry settings
    retry_attempts: int = 3
    retry_delay: int = 60  # seconds
    
    enabled: bool = True


@dataclass
class MetricDataPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores metrics for monitoring."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or str(Path.home() / '.quantrs2' / 'metrics.db')
        self._metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._buffer_lock = threading.RLock()
        
        # Initialize database
        self._init_metrics_db()
        
        # Background storage thread
        self._storage_thread = None
        self._shutdown = threading.Event()
        self._start_storage_thread()
    
    def _init_metrics_db(self):
        """Initialize metrics database."""
        try:
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_name TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        value REAL NOT NULL,
                        tags TEXT,
                        metadata TEXT,
                        created_at REAL DEFAULT (julianday('now'))
                    )
                """)
                
                # Create indices for performance
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp 
                    ON metrics(metric_name, timestamp DESC)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                    ON metrics(timestamp DESC)
                """)
                
            logger.info(f"Metrics database initialized: {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics database: {e}")
    
    def _start_storage_thread(self):
        """Start background storage thread."""
        self._storage_thread = threading.Thread(target=self._storage_loop, daemon=True)
        self._storage_thread.start()
    
    def _storage_loop(self):
        """Background loop for persisting metrics."""
        while not self._shutdown.wait(30):  # Store every 30 seconds
            try:
                self._persist_buffered_metrics()
            except Exception as e:
                logger.error(f"Metrics storage error: {e}")
    
    def record_metric(self, name: str, value: float, 
                     tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a metric data point."""
        try:
            timestamp = time.time()
            data_point = MetricDataPoint(
                timestamp=timestamp,
                value=value,
                tags=tags or {},
                metadata=metadata or {}
            )
            
            with self._buffer_lock:
                self._metrics_buffer[name].append(data_point)
            
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    def get_recent_metrics(self, name: str, window_seconds: int = 300) -> List[MetricDataPoint]:
        """Get recent metrics from buffer."""
        cutoff_time = time.time() - window_seconds
        
        with self._buffer_lock:
            buffer = self._metrics_buffer.get(name, deque())
            return [
                point for point in buffer 
                if point.timestamp >= cutoff_time
            ]
    
    def get_historical_metrics(self, name: str, start_time: float, 
                             end_time: float) -> List[MetricDataPoint]:
        """Get historical metrics from database."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, value, tags, metadata 
                    FROM metrics 
                    WHERE metric_name = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp ASC
                """, (name, start_time, end_time))
                
                results = []
                for row in cursor:
                    timestamp, value, tags_json, metadata_json = row
                    tags = json.loads(tags_json) if tags_json else {}
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    results.append(MetricDataPoint(
                        timestamp=timestamp,
                        value=value,
                        tags=tags,
                        metadata=metadata
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get historical metrics for {name}: {e}")
            return []
    
    def _persist_buffered_metrics(self):
        """Persist buffered metrics to database."""
        try:
            # Copy and clear buffers
            with self._buffer_lock:
                buffers_copy = {
                    name: list(buffer) for name, buffer in self._metrics_buffer.items()
                }
                # Clear buffers after copying
                for buffer in self._metrics_buffer.values():
                    buffer.clear()
            
            if not buffers_copy:
                return
            
            # Batch insert to database
            with sqlite3.connect(self.storage_path) as conn:
                for metric_name, data_points in buffers_copy.items():
                    if not data_points:
                        continue
                    
                    rows = []
                    for point in data_points:
                        rows.append((
                            metric_name,
                            point.timestamp,
                            point.value,
                            json.dumps(point.tags) if point.tags else None,
                            json.dumps(point.metadata) if point.metadata else None
                        ))
                    
                    conn.executemany("""
                        INSERT INTO metrics (metric_name, timestamp, value, tags, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, rows)
                
                logger.debug(f"Persisted {sum(len(points) for points in buffers_copy.values())} metric points")
                
        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")
    
    def cleanup_old_metrics(self, retention_days: int = 30):
        """Clean up old metrics beyond retention period."""
        try:
            cutoff_time = time.time() - (retention_days * 86400)
            
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_time,))
                deleted_count = cursor.rowcount
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old metric records")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
    
    def close(self):
        """Close metrics collector."""
        self._shutdown.set()
        if self._storage_thread:
            self._storage_thread.join(timeout=5)
        
        # Final persistence
        self._persist_buffered_metrics()


class NotificationManager:
    """Manages alert notifications across multiple channels."""
    
    def __init__(self):
        self.channels: Dict[str, NotificationConfig] = {}
        self.notification_history: deque = deque(maxlen=10000)
        self._rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._lock = threading.RLock()
    
    def register_channel(self, config: NotificationConfig):
        """Register a notification channel."""
        with self._lock:
            self.channels[config.name] = config
            logger.info(f"Registered notification channel: {config.name} ({config.channel_type.value})")
    
    def remove_channel(self, name: str):
        """Remove a notification channel."""
        with self._lock:
            if name in self.channels:
                del self.channels[name]
                logger.info(f"Removed notification channel: {name}")
    
    async def send_alert(self, alert: Alert) -> Dict[str, bool]:
        """Send alert through configured channels."""
        results = {}
        
        with self._lock:
            eligible_channels = self._get_eligible_channels(alert)
        
        for channel_name in eligible_channels:
            try:
                config = self.channels[channel_name]
                
                # Check rate limits
                if not self._check_rate_limit(channel_name, config):
                    logger.warning(f"Rate limit exceeded for channel {channel_name}")
                    results[channel_name] = False
                    continue
                
                # Send notification
                success = await self._send_notification(config, alert)
                results[channel_name] = success
                
                if success:
                    self._record_notification(channel_name, alert)
                
            except Exception as e:
                logger.error(f"Failed to send notification via {channel_name}: {e}")
                results[channel_name] = False
        
        return results
    
    def _get_eligible_channels(self, alert: Alert) -> List[str]:
        """Get channels eligible for this alert."""
        eligible = []
        
        for channel_name, config in self.channels.items():
            if not config.enabled:
                continue
            
            # Check severity filter
            if config.severity_filter and alert.severity not in config.severity_filter:
                continue
            
            # Check tag filters
            if config.tag_filters:
                matches_tags = True
                for tag_key, allowed_values in config.tag_filters.items():
                    alert_value = alert.tags.get(tag_key)
                    if alert_value not in allowed_values:
                        matches_tags = False
                        break
                
                if not matches_tags:
                    continue
            
            eligible.append(channel_name)
        
        return eligible
    
    def _check_rate_limit(self, channel_name: str, config: NotificationConfig) -> bool:
        """Check if channel is within rate limits."""
        now = time.time()
        window_start = now - config.rate_limit_period
        
        # Clean old entries
        rate_limit_queue = self._rate_limits[channel_name]
        while rate_limit_queue and rate_limit_queue[0] < window_start:
            rate_limit_queue.popleft()
        
        # Check limit
        if len(rate_limit_queue) >= config.max_notifications_per_period:
            return False
        
        # Record this notification
        rate_limit_queue.append(now)
        return True
    
    async def _send_notification(self, config: NotificationConfig, alert: Alert) -> bool:
        """Send notification via specific channel."""
        try:
            if config.channel_type == NotificationChannel.EMAIL:
                return await self._send_email(config, alert)
            elif config.channel_type == NotificationChannel.WEBHOOK:
                return await self._send_webhook(config, alert)
            elif config.channel_type == NotificationChannel.SLACK:
                return await self._send_slack(config, alert)
            else:
                logger.warning(f"Unsupported notification channel: {config.channel_type}")
                return False
                
        except Exception as e:
            logger.error(f"Notification send failed: {e}")
            return False
    
    async def _send_email(self, config: NotificationConfig, alert: Alert) -> bool:
        """Send email notification."""
        if not EMAIL_AVAILABLE:
            logger.error("Email functionality not available (email modules not imported)")
            return False
            
        try:
            settings = config.settings
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = settings['from_address']
            msg['To'] = ', '.join(settings['to_addresses'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.rule_name}"
            
            # Create email body
            body = f"""
Alert: {alert.rule_name}
Severity: {alert.severity.value.upper()}
Status: {alert.status.value}

Description: {alert.description}
Metric: {alert.metric_name} = {alert.metric_value}
Threshold: {alert.threshold_value}

Triggered at: {datetime.fromtimestamp(alert.triggered_at)}

Alert ID: {alert.id}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(settings['smtp_host'], settings.get('smtp_port', 587)) as server:
                if settings.get('use_tls', True):
                    server.starttls()
                
                if 'username' in settings and 'password' in settings:
                    server.login(settings['username'], settings['password'])
                
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False
    
    async def _send_webhook(self, config: NotificationConfig, alert: Alert) -> bool:
        """Send webhook notification."""
        if not REQUESTS_AVAILABLE:
            logger.error("Webhook functionality not available (requests module not available)")
            return False
            
        try:
            settings = config.settings
            url = settings['url']
            
            payload = {
                'alert_id': alert.id,
                'rule_name': alert.rule_name,
                'severity': alert.severity.value,
                'status': alert.status.value,
                'message': alert.message,
                'description': alert.description,
                'metric_name': alert.metric_name,
                'metric_value': alert.metric_value,
                'threshold_value': alert.threshold_value,
                'triggered_at': alert.triggered_at,
                'tags': alert.tags,
                'context': alert.context
            }
            
            headers = {
                'Content-Type': 'application/json',
                **settings.get('headers', {})
            }
            
            timeout = settings.get('timeout', 30)
            
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False
    
    async def _send_slack(self, config: NotificationConfig, alert: Alert) -> bool:
        """Send Slack notification."""
        if not REQUESTS_AVAILABLE:
            logger.error("Slack functionality not available (requests module not available)")
            return False
            
        try:
            settings = config.settings
            webhook_url = settings['webhook_url']
            
            # Severity color mapping
            color_map = {
                AlertSeverity.CRITICAL: '#FF0000',
                AlertSeverity.HIGH: '#FF8C00',
                AlertSeverity.MEDIUM: '#FFD700',
                AlertSeverity.LOW: '#32CD32',
                AlertSeverity.INFO: '#87CEEB'
            }
            
            payload = {
                'attachments': [{
                    'color': color_map.get(alert.severity, '#808080'),
                    'title': f"{alert.severity.value.upper()}: {alert.rule_name}",
                    'text': alert.description,
                    'fields': [
                        {
                            'title': 'Metric',
                            'value': f"{alert.metric_name} = {alert.metric_value}",
                            'short': True
                        },
                        {
                            'title': 'Threshold',
                            'value': str(alert.threshold_value),
                            'short': True
                        },
                        {
                            'title': 'Status',
                            'value': alert.status.value,
                            'short': True
                        },
                        {
                            'title': 'Alert ID',
                            'value': alert.id,
                            'short': True
                        }
                    ],
                    'footer': 'QuantRS2 Monitoring',
                    'ts': int(alert.triggered_at)
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False
    
    def _record_notification(self, channel_name: str, alert: Alert):
        """Record notification in history."""
        self.notification_history.append({
            'timestamp': time.time(),
            'channel': channel_name,
            'alert_id': alert.id,
            'rule_name': alert.rule_name,
            'severity': alert.severity.value
        })
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        now = time.time()
        last_24h = now - 86400
        
        recent_notifications = [
            n for n in self.notification_history 
            if n['timestamp'] >= last_24h
        ]
        
        stats = {
            'total_channels': len(self.channels),
            'enabled_channels': sum(1 for c in self.channels.values() if c.enabled),
            'notifications_24h': len(recent_notifications),
            'by_channel': defaultdict(int),
            'by_severity': defaultdict(int)
        }
        
        for notification in recent_notifications:
            stats['by_channel'][notification['channel']] += 1
            stats['by_severity'][notification['severity']] += 1
        
        return dict(stats)


class AlertManager:
    """Manages alert rules, evaluation, and lifecycle."""
    
    def __init__(self, metrics_collector: MetricsCollector, 
                 notification_manager: NotificationManager):
        self.metrics_collector = metrics_collector
        self.notification_manager = notification_manager
        
        # Alert storage
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Evaluation state
        self._evaluation_thread = None
        self._shutdown = threading.Event()
        self._lock = threading.RLock()
        
        # Start evaluation loop
        self._start_evaluation_loop()
        
        logger.info("Alert manager initialized")
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        with self._lock:
            self.alert_rules[rule.id] = rule
            logger.info(f"Added alert rule: {rule.name} ({rule.id})")
    
    def remove_rule(self, rule_id: str):
        """Remove an alert rule."""
        with self._lock:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                logger.info(f"Removed alert rule: {rule_id}")
    
    def update_rule(self, rule: AlertRule):
        """Update an alert rule."""
        with self._lock:
            if rule.id in self.alert_rules:
                rule.updated_at = time.time()
                self.alert_rules[rule.id] = rule
                logger.info(f"Updated alert rule: {rule.name} ({rule.id})")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system"):
        """Acknowledge an active alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = time.time()
                alert.acknowledged_by = acknowledged_by
                
                logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system"):
        """Resolve an active alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = time.time()
                alert.resolved_by = resolved_by
                
                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
                return True
        
        return False
    
    def _start_evaluation_loop(self):
        """Start alert evaluation loop."""
        self._evaluation_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
        self._evaluation_thread.start()
    
    def _evaluation_loop(self):
        """Main alert evaluation loop."""
        while not self._shutdown.wait(60):  # Evaluate every minute
            try:
                self._evaluate_all_rules()
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")
    
    def _evaluate_all_rules(self):
        """Evaluate all alert rules."""
        with self._lock:
            rules_to_evaluate = [rule for rule in self.alert_rules.values() if rule.enabled]
        
        for rule in rules_to_evaluate:
            try:
                self._evaluate_rule(rule)
            except Exception as e:
                logger.error(f"Rule evaluation failed for {rule.id}: {e}")
    
    def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule."""
        # Get recent metrics
        recent_metrics = self.metrics_collector.get_recent_metrics(
            rule.metric_name, rule.evaluation_window
        )
        
        if len(recent_metrics) < rule.data_points_required:
            return  # Not enough data points
        
        # Extract values
        values = [point.value for point in recent_metrics]
        latest_value = values[-1] if values else 0
        
        # Evaluate threshold
        threshold_breached = self._evaluate_threshold(
            latest_value, rule.threshold_value, rule.comparison
        )
        
        alert_id = f"{rule.id}:{rule.metric_name}"
        
        if threshold_breached:
            # Check if alert already exists
            if alert_id not in self.active_alerts:
                # Create new alert
                alert = Alert(
                    id=alert_id,
                    rule_id=rule.id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    status=AlertStatus.ACTIVE,
                    message=f"{rule.name}: {rule.metric_name} = {latest_value}",
                    description=rule.description,
                    metric_name=rule.metric_name,
                    metric_value=latest_value,
                    threshold_value=rule.threshold_value,
                    triggered_at=time.time(),
                    tags=rule.tags.copy(),
                    context={
                        'evaluation_window': rule.evaluation_window,
                        'comparison': rule.comparison,
                        'data_points': len(recent_metrics)
                    }
                )
                
                with self._lock:
                    self.active_alerts[alert_id] = alert
                
                # Send notifications
                asyncio.create_task(self._handle_alert_notification(alert))
                
                logger.warning(f"Alert triggered: {rule.name} ({alert_id})")
        
        else:
            # Check if alert should be auto-resolved
            if alert_id in self.active_alerts:
                self.resolve_alert(alert_id, "auto-resolved")
    
    def _evaluate_threshold(self, value: float, threshold: float, comparison: str) -> bool:
        """Evaluate threshold condition."""
        if comparison == ">":
            return value > threshold
        elif comparison == "<":
            return value < threshold
        elif comparison == ">=":
            return value >= threshold
        elif comparison == "<=":
            return value <= threshold
        elif comparison == "==":
            return abs(value - threshold) < 1e-9
        elif comparison == "!=":
            return abs(value - threshold) >= 1e-9
        else:
            logger.error(f"Unknown comparison operator: {comparison}")
            return False
    
    async def _handle_alert_notification(self, alert: Alert):
        """Handle alert notification with retries."""
        try:
            # Send initial notification
            results = await self.notification_manager.send_alert(alert)
            
            alert.last_notified_at = time.time()
            alert.notification_count += 1
            
            successful_channels = [k for k, v in results.items() if v]
            failed_channels = [k for k, v in results.items() if not v]
            
            if successful_channels:
                logger.info(f"Alert {alert.id} sent via: {', '.join(successful_channels)}")
            
            if failed_channels:
                logger.error(f"Alert {alert.id} failed via: {', '.join(failed_channels)}")
            
        except Exception as e:
            logger.error(f"Alert notification handling failed: {e}")
    
    def get_active_alerts(self, severity_filter: Optional[List[AlertSeverity]] = None) -> List[Alert]:
        """Get active alerts with optional severity filter."""
        with self._lock:
            alerts = list(self.active_alerts.values())
        
        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity in severity_filter]
        
        return sorted(alerts, key=lambda a: a.triggered_at, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            active_count = len(self.active_alerts)
            total_rules = len(self.alert_rules)
            enabled_rules = sum(1 for rule in self.alert_rules.values() if rule.enabled)
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1
        
        # Count by status
        status_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            status_counts[alert.status.value] += 1
        
        return {
            'active_alerts': active_count,
            'total_rules': total_rules,
            'enabled_rules': enabled_rules,
            'by_severity': dict(severity_counts),
            'by_status': dict(status_counts),
            'history_size': len(self.alert_history)
        }
    
    def close(self):
        """Close alert manager."""
        self._shutdown.set()
        if self._evaluation_thread:
            self._evaluation_thread.join(timeout=5)


class MonitoringSystem:
    """Comprehensive monitoring and alerting system."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or str(Path.home() / '.quantrs2' / 'monitoring')
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metrics_collector = MetricsCollector(
            str(Path(self.storage_path) / 'metrics.db')
        )
        self.notification_manager = NotificationManager()
        self.alert_manager = AlertManager(self.metrics_collector, self.notification_manager)
        
        # Monitoring configuration
        self.monitoring_enabled = True
        self._monitoring_thread = None
        self._shutdown = threading.Event()
        
        # Start system monitoring
        self._start_system_monitoring()
        
        logger.info("Monitoring system initialized")
    
    def _start_system_monitoring(self):
        """Start system metrics monitoring."""
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
    
    def _monitoring_loop(self):
        """Main monitoring loop for system metrics."""
        while not self._shutdown.wait(60):  # Collect every minute
            if not self.monitoring_enabled:
                continue
                
            try:
                self._collect_system_metrics()
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            import psutil
            
            # System metrics
            self.metrics_collector.record_metric(
                "system.cpu_percent", 
                psutil.cpu_percent(),
                tags={"type": "system"}
            )
            
            memory = psutil.virtual_memory()
            self.metrics_collector.record_metric(
                "system.memory_percent", 
                memory.percent,
                tags={"type": "system"}
            )
            
            self.metrics_collector.record_metric(
                "system.memory_available_mb", 
                memory.available / 1024 / 1024,
                tags={"type": "system"}
            )
            
            disk = psutil.disk_usage('/')
            self.metrics_collector.record_metric(
                "system.disk_percent", 
                (disk.used / disk.total) * 100,
                tags={"type": "system"}
            )
            
            # Network metrics
            network = psutil.net_io_counters()
            self.metrics_collector.record_metric(
                "system.network_bytes_sent", 
                network.bytes_sent,
                tags={"type": "system"}
            )
            
            self.metrics_collector.record_metric(
                "system.network_bytes_recv", 
                network.bytes_recv,
                tags={"type": "system"}
            )
            
        except ImportError:
            logger.warning("psutil not available for system monitoring")
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    def record_application_metric(self, name: str, value: float, 
                                tags: Optional[Dict[str, str]] = None):
        """Record application-specific metric."""
        app_tags = {"type": "application", **(tags or {})}
        self.metrics_collector.record_metric(f"app.{name}", value, tags=app_tags)
    
    def create_default_alerts(self):
        """Create default alert rules for common scenarios."""
        default_rules = [
            AlertRule(
                id="high_cpu_usage",
                name="High CPU Usage",
                description="CPU usage is above 90%",
                metric_name="system.cpu_percent",
                metric_type=MetricType.SYSTEM,
                severity=AlertSeverity.HIGH,
                threshold_value=90.0,
                comparison=">",
                evaluation_window=300,
                data_points_required=3
            ),
            AlertRule(
                id="high_memory_usage",
                name="High Memory Usage", 
                description="Memory usage is above 85%",
                metric_name="system.memory_percent",
                metric_type=MetricType.SYSTEM,
                severity=AlertSeverity.MEDIUM,
                threshold_value=85.0,
                comparison=">",
                evaluation_window=300,
                data_points_required=3
            ),
            AlertRule(
                id="disk_space_low",
                name="Low Disk Space",
                description="Disk usage is above 90%",
                metric_name="system.disk_percent",
                metric_type=MetricType.SYSTEM,
                severity=AlertSeverity.HIGH,
                threshold_value=90.0,
                comparison=">",
                evaluation_window=600,
                data_points_required=2
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.add_rule(rule)
        
        logger.info(f"Created {len(default_rules)} default alert rules")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        alert_stats = self.alert_manager.get_alert_statistics()
        notification_stats = self.notification_manager.get_notification_stats()
        
        # Determine health status
        critical_alerts = alert_stats['by_severity'].get('critical', 0)
        high_alerts = alert_stats['by_severity'].get('high', 0)
        
        if critical_alerts > 0:
            health_status = "critical"
        elif high_alerts > 0:
            health_status = "degraded"
        elif alert_stats['active_alerts'] > 0:
            health_status = "warning"
        else:
            health_status = "healthy"
        
        return {
            'status': health_status,
            'alerts': alert_stats,
            'notifications': notification_stats,
            'monitoring_enabled': self.monitoring_enabled,
            'uptime': time.time() - self.metrics_collector._start_time if hasattr(self.metrics_collector, '_start_time') else 0
        }
    
    def close(self):
        """Close monitoring system."""
        logger.info("Closing monitoring system")
        
        self._shutdown.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        
        self.alert_manager.close()
        self.metrics_collector.close()


# Export main classes
__all__ = [
    'AlertSeverity',
    'AlertStatus', 
    'NotificationChannel',
    'MetricType',
    'AlertRule',
    'Alert',
    'NotificationConfig',
    'MetricDataPoint',
    'MetricsCollector',
    'NotificationManager',
    'AlertManager',
    'MonitoringSystem'
]