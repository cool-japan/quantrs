"""
Real-time Monitoring Dashboard for QuantRS2

This module provides a web-based dashboard for monitoring QuantRS2 performance,
viewing alerts, managing notification channels, and analyzing metrics trends.
"""

import json
import time
import logging
import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import weakref

try:
    from flask import Flask, render_template_string, jsonify, request, Response
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from .monitoring_alerting import (
    MonitoringSystem, AlertSeverity, AlertStatus, NotificationChannel,
    AlertRule, Alert, NotificationConfig, MetricType
)

logger = logging.getLogger(__name__)


class DashboardServer:
    """Web-based monitoring dashboard server."""
    
    def __init__(self, monitoring_system: MonitoringSystem, 
                 host: str = "0.0.0.0", port: int = 8080):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask and Flask-SocketIO required for dashboard")
        
        self.monitoring_system = monitoring_system
        self.host = host
        self.port = port
        
        # Create Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'quantrs2-monitoring-dashboard'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Dashboard state
        self.connected_clients = set()
        self._update_thread = None
        self._shutdown = threading.Event()
        
        # Setup routes
        self._setup_routes()
        self._setup_websocket_handlers()
        
        logger.info(f"Dashboard server initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page."""
            return render_template_string(DASHBOARD_HTML_TEMPLATE)
        
        @self.app.route('/api/health')
        def get_health():
            """Get system health status."""
            return jsonify(self.monitoring_system.get_system_health())
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get active alerts."""
            severity_filter = request.args.getlist('severity')
            if severity_filter:
                severity_filter = [AlertSeverity(s) for s in severity_filter]
            
            alerts = self.monitoring_system.alert_manager.get_active_alerts(severity_filter)
            
            return jsonify([{
                'id': alert.id,
                'rule_name': alert.rule_name,
                'severity': alert.severity.value,
                'status': alert.status.value,
                'message': alert.message,
                'metric_name': alert.metric_name,
                'metric_value': alert.metric_value,
                'threshold_value': alert.threshold_value,
                'triggered_at': alert.triggered_at,
                'acknowledged_at': alert.acknowledged_at,
                'tags': alert.tags
            } for alert in alerts])
        
        @self.app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
        def acknowledge_alert(alert_id):
            """Acknowledge an alert."""
            data = request.get_json() or {}
            acknowledged_by = data.get('acknowledged_by', 'dashboard-user')
            
            success = self.monitoring_system.alert_manager.acknowledge_alert(
                alert_id, acknowledged_by
            )
            
            return jsonify({'success': success})
        
        @self.app.route('/api/alerts/<alert_id>/resolve', methods=['POST'])
        def resolve_alert(alert_id):
            """Resolve an alert."""
            data = request.get_json() or {}
            resolved_by = data.get('resolved_by', 'dashboard-user')
            
            success = self.monitoring_system.alert_manager.resolve_alert(
                alert_id, resolved_by
            )
            
            return jsonify({'success': success})
        
        @self.app.route('/api/metrics/<metric_name>')
        def get_metric_data(metric_name):
            """Get metric data for charting."""
            hours = int(request.args.get('hours', 1))
            end_time = time.time()
            start_time = end_time - (hours * 3600)
            
            # Try recent data first
            if hours <= 1:
                data_points = self.monitoring_system.metrics_collector.get_recent_metrics(
                    metric_name, int(hours * 3600)
                )
            else:
                data_points = self.monitoring_system.metrics_collector.get_historical_metrics(
                    metric_name, start_time, end_time
                )
            
            return jsonify([{
                'timestamp': point.timestamp,
                'value': point.value,
                'tags': point.tags
            } for point in data_points])
        
        @self.app.route('/api/rules')
        def get_rules():
            """Get alert rules."""
            rules = []
            for rule in self.monitoring_system.alert_manager.alert_rules.values():
                rules.append({
                    'id': rule.id,
                    'name': rule.name,
                    'description': rule.description,
                    'metric_name': rule.metric_name,
                    'metric_type': rule.metric_type.value,
                    'severity': rule.severity.value,
                    'threshold_value': rule.threshold_value,
                    'comparison': rule.comparison,
                    'enabled': rule.enabled,
                    'tags': rule.tags
                })
            
            return jsonify(rules)
        
        @self.app.route('/api/rules', methods=['POST'])
        def create_rule():
            """Create new alert rule."""
            data = request.get_json()
            
            try:
                rule = AlertRule(
                    id=data['id'],
                    name=data['name'],
                    description=data['description'],
                    metric_name=data['metric_name'],
                    metric_type=MetricType(data['metric_type']),
                    severity=AlertSeverity(data['severity']),
                    threshold_value=float(data['threshold_value']),
                    comparison=data.get('comparison', '>'),
                    evaluation_window=int(data.get('evaluation_window', 300)),
                    data_points_required=int(data.get('data_points_required', 3)),
                    notification_channels=data.get('notification_channels', []),
                    tags=data.get('tags', {})
                )
                
                self.monitoring_system.alert_manager.add_rule(rule)
                return jsonify({'success': True, 'rule_id': rule.id})
                
            except Exception as e:
                logger.error(f"Failed to create rule: {e}")
                return jsonify({'success': False, 'error': str(e)}), 400
        
        @self.app.route('/api/channels')
        def get_channels():
            """Get notification channels."""
            channels = []
            for name, config in self.monitoring_system.notification_manager.channels.items():
                channels.append({
                    'name': name,
                    'channel_type': config.channel_type.value,
                    'enabled': config.enabled,
                    'severity_filter': [s.value for s in config.severity_filter],
                    'rate_limit_period': config.rate_limit_period,
                    'max_notifications_per_period': config.max_notifications_per_period
                })
            
            return jsonify(channels)
        
        @self.app.route('/api/channels', methods=['POST'])
        def create_channel():
            """Create notification channel."""
            data = request.get_json()
            
            try:
                config = NotificationConfig(
                    channel_type=NotificationChannel(data['channel_type']),
                    name=data['name'],
                    settings=data.get('settings', {}),
                    severity_filter=[AlertSeverity(s) for s in data.get('severity_filter', [])],
                    enabled=data.get('enabled', True),
                    rate_limit_period=int(data.get('rate_limit_period', 300)),
                    max_notifications_per_period=int(data.get('max_notifications_per_period', 10))
                )
                
                self.monitoring_system.notification_manager.register_channel(config)
                return jsonify({'success': True, 'channel_name': config.name})
                
            except Exception as e:
                logger.error(f"Failed to create channel: {e}")
                return jsonify({'success': False, 'error': str(e)}), 400
        
        @self.app.route('/api/stats')
        def get_stats():
            """Get comprehensive monitoring statistics."""
            return jsonify({
                'alerts': self.monitoring_system.alert_manager.get_alert_statistics(),
                'notifications': self.monitoring_system.notification_manager.get_notification_stats()
            })
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket handlers for real-time updates."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            self.connected_clients.add(request.sid)
            logger.debug(f"Dashboard client connected: {request.sid}")
            
            # Send initial data
            emit('system_health', self.monitoring_system.get_system_health())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            self.connected_clients.discard(request.sid)
            logger.debug(f"Dashboard client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe_metric')
        def handle_metric_subscription(data):
            """Handle metric subscription."""
            metric_name = data.get('metric_name')
            if metric_name:
                # Send recent data for the metric
                recent_data = self.monitoring_system.metrics_collector.get_recent_metrics(
                    metric_name, 300  # Last 5 minutes
                )
                
                emit('metric_data', {
                    'metric_name': metric_name,
                    'data': [{
                        'timestamp': point.timestamp,
                        'value': point.value
                    } for point in recent_data]
                })
    
    def start(self, debug: bool = False):
        """Start the dashboard server."""
        # Start update thread for real-time data
        self._start_update_thread()
        
        try:
            logger.info(f"Starting dashboard server on {self.host}:{self.port}")
            self.socketio.run(
                self.app, 
                host=self.host, 
                port=self.port, 
                debug=debug,
                allow_unsafe_werkzeug=True
            )
        except Exception as e:
            logger.error(f"Dashboard server failed to start: {e}")
            raise
    
    def _start_update_thread(self):
        """Start background thread for real-time updates."""
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
    
    def _update_loop(self):
        """Background loop for sending real-time updates."""
        while not self._shutdown.wait(10):  # Update every 10 seconds
            if not self.connected_clients:
                continue
            
            try:
                # Send system health updates
                health_data = self.monitoring_system.get_system_health()
                self.socketio.emit('system_health', health_data)
                
                # Send alert updates
                alerts = self.monitoring_system.alert_manager.get_active_alerts()
                alert_data = [{
                    'id': alert.id,
                    'rule_name': alert.rule_name,
                    'severity': alert.severity.value,
                    'status': alert.status.value,
                    'triggered_at': alert.triggered_at
                } for alert in alerts]
                
                self.socketio.emit('alerts_update', alert_data)
                
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
    
    def stop(self):
        """Stop the dashboard server."""
        self._shutdown.set()
        if self._update_thread:
            self._update_thread.join(timeout=5)


# HTML template for the dashboard
DASHBOARD_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuantRS2 Monitoring Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 1rem;
            text-align: center;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin-bottom: 1rem;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-healthy { background: #27ae60; }
        .status-warning { background: #f39c12; }
        .status-degraded { background: #e74c3c; }
        .status-critical { background: #c0392b; }
        .alert-item {
            background: #f8f9fa;
            border-left: 4px solid #e74c3c;
            padding: 1rem;
            margin-bottom: 0.5rem;
            border-radius: 4px;
        }
        .alert-critical { border-left-color: #c0392b; }
        .alert-high { border-left-color: #e74c3c; }
        .alert-medium { border-left-color: #f39c12; }
        .alert-low { border-left-color: #27ae60; }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #3498db;
        }
        .metric-chart {
            height: 200px;
            margin-top: 1rem;
        }
        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            margin-right: 0.5rem;
        }
        .btn:hover { background: #2980b9; }
        .btn-success { background: #27ae60; }
        .btn-success:hover { background: #219a52; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        .stat-item {
            text-align: center;
            padding: 1rem;
            background: #ecf0f1;
            border-radius: 4px;
        }
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
        }
        .stat-label {
            font-size: 0.9rem;
            color: #7f8c8d;
            margin-top: 0.25rem;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>QuantRS2 Monitoring Dashboard</h1>
        <p>Real-time monitoring and alerting for quantum computing infrastructure</p>
    </div>
    
    <div class="container">
        <!-- System Health Overview -->
        <div class="card">
            <h3>System Health</h3>
            <div id="health-status">
                <span class="status-indicator status-healthy"></span>
                <span id="health-text">Loading...</span>
            </div>
            <div class="stats-grid" id="health-stats">
                <!-- Stats will be populated by JavaScript -->
            </div>
        </div>
        
        <!-- Active Alerts -->
        <div class="card">
            <h3>Active Alerts</h3>
            <div id="alerts-list">
                <p>Loading alerts...</p>
            </div>
        </div>
        
        <!-- Metrics Charts -->
        <div class="grid">
            <div class="card">
                <h3>CPU Usage</h3>
                <canvas id="cpu-chart" class="metric-chart"></canvas>
            </div>
            <div class="card">
                <h3>Memory Usage</h3>
                <canvas id="memory-chart" class="metric-chart"></canvas>
            </div>
        </div>
        
        <!-- Alert Rules Management -->
        <div class="card">
            <h3>Alert Rules</h3>
            <button class="btn" onclick="loadRules()">Refresh Rules</button>
            <div id="rules-list" style="margin-top: 1rem;">
                <!-- Rules will be populated by JavaScript -->
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO connection
        const socket = io();
        
        // Chart instances
        let cpuChart, memoryChart;
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
            loadInitialData();
        });
        
        // Socket event handlers
        socket.on('connect', function() {
            console.log('Connected to monitoring server');
        });
        
        socket.on('system_health', function(data) {
            updateHealthStatus(data);
        });
        
        socket.on('alerts_update', function(data) {
            updateAlertsList(data);
        });
        
        socket.on('metric_data', function(data) {
            updateMetricChart(data.metric_name, data.data);
        });
        
        // Initialize charts
        function initializeCharts() {
            const cpuCtx = document.getElementById('cpu-chart').getContext('2d');
            cpuChart = new Chart(cpuCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU %',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { beginAtZero: true, max: 100 }
                    }
                }
            });
            
            const memoryCtx = document.getElementById('memory-chart').getContext('2d');
            memoryChart = new Chart(memoryCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Memory %',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: { beginAtZero: true, max: 100 }
                    }
                }
            });
            
            // Subscribe to metrics
            socket.emit('subscribe_metric', { metric_name: 'system.cpu_percent' });
            socket.emit('subscribe_metric', { metric_name: 'system.memory_percent' });
        }
        
        // Load initial data
        function loadInitialData() {
            fetch('/api/health')
                .then(response => response.json())
                .then(data => updateHealthStatus(data));
                
            fetch('/api/alerts')
                .then(response => response.json())
                .then(data => updateAlertsList(data));
        }
        
        // Update health status
        function updateHealthStatus(data) {
            const indicator = document.querySelector('.status-indicator');
            const text = document.getElementById('health-text');
            const statsContainer = document.getElementById('health-stats');
            
            // Update status indicator
            indicator.className = 'status-indicator status-' + data.status;
            text.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
            
            // Update stats
            const stats = [
                { label: 'Active Alerts', value: data.alerts.active_alerts },
                { label: 'Total Rules', value: data.alerts.total_rules },
                { label: 'Notifications (24h)', value: data.notifications.notifications_24h },
                { label: 'Enabled Channels', value: data.notifications.enabled_channels }
            ];
            
            statsContainer.innerHTML = stats.map(stat => `
                <div class="stat-item">
                    <div class="stat-value">${stat.value}</div>
                    <div class="stat-label">${stat.label}</div>
                </div>
            `).join('');
        }
        
        // Update alerts list
        function updateAlertsList(alerts) {
            const container = document.getElementById('alerts-list');
            
            if (alerts.length === 0) {
                container.innerHTML = '<p style="color: #27ae60;">No active alerts</p>';
                return;
            }
            
            container.innerHTML = alerts.map(alert => `
                <div class="alert-item alert-${alert.severity}">
                    <div style="display: flex; justify-content: between; align-items: center;">
                        <div>
                            <strong>${alert.rule_name}</strong>
                            <span style="color: #7f8c8d; margin-left: 10px;">
                                ${alert.severity.toUpperCase()}
                            </span>
                        </div>
                        <div>
                            ${alert.status === 'active' ? `
                                <button class="btn btn-success" onclick="acknowledgeAlert('${alert.id}')">
                                    Acknowledge
                                </button>
                                <button class="btn" onclick="resolveAlert('${alert.id}')">
                                    Resolve
                                </button>
                            ` : `<span style="color: #27ae60;">✓ ${alert.status}</span>`}
                        </div>
                    </div>
                    <div style="margin-top: 0.5rem; color: #666;">
                        ${alert.message}
                    </div>
                    <div style="margin-top: 0.25rem; font-size: 0.9rem; color: #999;">
                        Triggered: ${new Date(alert.triggered_at * 1000).toLocaleString()}
                    </div>
                </div>
            `).join('');
        }
        
        // Update metric chart
        function updateMetricChart(metricName, data) {
            let chart;
            if (metricName === 'system.cpu_percent') {
                chart = cpuChart;
            } else if (metricName === 'system.memory_percent') {
                chart = memoryChart;
            } else {
                return;
            }
            
            const labels = data.map(point => 
                new Date(point.timestamp * 1000).toLocaleTimeString()
            );
            const values = data.map(point => point.value);
            
            chart.data.labels = labels.slice(-20); // Keep last 20 points
            chart.data.datasets[0].data = values.slice(-20);
            chart.update('none');
        }
        
        // Alert actions
        function acknowledgeAlert(alertId) {
            fetch(`/api/alerts/${alertId}/acknowledge`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ acknowledged_by: 'dashboard-user' })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadInitialData(); // Refresh alerts
                }
            });
        }
        
        function resolveAlert(alertId) {
            fetch(`/api/alerts/${alertId}/resolve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ resolved_by: 'dashboard-user' })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadInitialData(); // Refresh alerts
                }
            });
        }
        
        // Load alert rules
        function loadRules() {
            fetch('/api/rules')
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('rules-list');
                    container.innerHTML = data.map(rule => `
                        <div style="background: #f8f9fa; padding: 1rem; margin-bottom: 0.5rem; border-radius: 4px;">
                            <strong>${rule.name}</strong>
                            <span style="float: right; color: ${rule.enabled ? '#27ae60' : '#e74c3c'};">
                                ${rule.enabled ? 'Enabled' : 'Disabled'}
                            </span>
                            <div style="margin-top: 0.5rem; color: #666;">
                                ${rule.description}
                            </div>
                            <div style="margin-top: 0.25rem; font-size: 0.9rem; color: #999;">
                                Metric: ${rule.metric_name} ${rule.comparison} ${rule.threshold_value}
                            </div>
                        </div>
                    `).join('');
                });
        }
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            fetch('/api/metrics/system.cpu_percent?hours=0.1')
                .then(response => response.json())
                .then(data => updateMetricChart('system.cpu_percent', data));
                
            fetch('/api/metrics/system.memory_percent?hours=0.1')
                .then(response => response.json())
                .then(data => updateMetricChart('system.memory_percent', data));
        }, 30000);
    </script>
</body>
</html>
"""


# Export main classes
__all__ = [
    'DashboardServer'
]