"""
Log Aggregation and External Integration for QuantRS2

This module provides log aggregation, forwarding to external log management
systems, log analysis, and integration with monitoring platforms.
"""

import time
import json
import threading
import gzip
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import uuid
import re

from .structured_logging import LogRecord, EventType, LogLevel, get_global_logging_system
from .error_analysis import ErrorAnalysisSystem

logger = get_global_logging_system().get_logger(__name__)


class LogDestination(Enum):
    """Types of log destinations."""
    ELASTICSEARCH = "elasticsearch"
    SPLUNK = "splunk"
    FLUENTD = "fluentd"
    LOGSTASH = "logstash"
    SYSLOG = "syslog"
    KAFKA = "kafka"
    CLOUDWATCH = "cloudwatch"
    STACKDRIVER = "stackdriver"
    FILE = "file"
    HTTP_ENDPOINT = "http_endpoint"


class LogFormat(Enum):
    """Log output formats."""
    JSON = "json"
    LOGFMT = "logfmt"
    CEF = "cef"  # Common Event Format
    GELF = "gelf"  # Graylog Extended Log Format
    RFC5424 = "rfc5424"  # Syslog RFC 5424
    CUSTOM = "custom"


@dataclass
class LogDestinationConfig:
    """Configuration for log destination."""
    destination_type: LogDestination
    name: str
    enabled: bool = True
    
    # Connection settings
    endpoint: Optional[str] = None
    port: Optional[int] = None
    
    # Authentication
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Formatting
    log_format: LogFormat = LogFormat.JSON
    custom_format: Optional[str] = None
    
    # Filtering
    level_filter: Optional[List[str]] = None
    event_type_filter: Optional[List[str]] = None
    tag_filters: Dict[str, List[str]] = field(default_factory=dict)
    
    # Buffering and performance
    buffer_size: int = 1000
    flush_interval: int = 30  # seconds
    compression: bool = False
    
    # Reliability
    retry_attempts: int = 3
    retry_delay: int = 5  # seconds
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogMetrics:
    """Metrics for log processing."""
    logs_processed: int = 0
    logs_dropped: int = 0
    logs_forwarded: int = 0
    errors_encountered: int = 0
    
    # Performance metrics
    processing_time_total: float = 0.0
    forwarding_time_total: float = 0.0
    
    # By destination
    destination_metrics: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def add_processing_time(self, duration: float):
        """Add processing time."""
        self.processing_time_total += duration
    
    def add_forwarding_time(self, duration: float):
        """Add forwarding time."""
        self.forwarding_time_total += duration
    
    def get_average_processing_time(self) -> float:
        """Get average processing time."""
        if self.logs_processed == 0:
            return 0.0
        return self.processing_time_total / self.logs_processed
    
    def get_average_forwarding_time(self) -> float:
        """Get average forwarding time."""
        if self.logs_forwarded == 0:
            return 0.0
        return self.forwarding_time_total / self.logs_forwarded


class LogFormatter:
    """Formats log records for different destinations."""
    
    def __init__(self, log_format: LogFormat, custom_format: Optional[str] = None):
        self.log_format = log_format
        self.custom_format = custom_format
    
    def format(self, record: LogRecord) -> str:
        """Format log record according to specified format."""
        if self.log_format == LogFormat.JSON:
            return self._format_json(record)
        elif self.log_format == LogFormat.LOGFMT:
            return self._format_logfmt(record)
        elif self.log_format == LogFormat.CEF:
            return self._format_cef(record)
        elif self.log_format == LogFormat.GELF:
            return self._format_gelf(record)
        elif self.log_format == LogFormat.RFC5424:
            return self._format_rfc5424(record)
        elif self.log_format == LogFormat.CUSTOM and self.custom_format:
            return self._format_custom(record)
        else:
            # Fallback to JSON
            return self._format_json(record)
    
    def _format_json(self, record: LogRecord) -> str:
        """Format as JSON."""
        return json.dumps(record.to_dict(), separators=(',', ':'))
    
    def _format_logfmt(self, record: LogRecord) -> str:
        """Format as logfmt (key=value pairs)."""
        data = record.to_dict()
        
        # Flatten nested dictionaries
        flat_data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_data[f"{key}_{sub_key}"] = sub_value
            else:
                flat_data[key] = value
        
        # Format as key=value pairs
        pairs = []
        for key, value in flat_data.items():
            if value is not None:
                # Escape special characters
                if isinstance(value, str) and (' ' in value or '"' in value):
                    value = f'"{value.replace('"', '\\"')}"'
                pairs.append(f"{key}={value}")
        
        return ' '.join(pairs)
    
    def _format_cef(self, record: LogRecord) -> str:
        """Format as Common Event Format (CEF)."""
        # CEF format: CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension
        
        severity_map = {
            'TRACE': 1, 'DEBUG': 2, 'INFO': 4, 'QUANTUM': 4,
            'WARNING': 6, 'ERROR': 8, 'CRITICAL': 10, 'SECURITY': 10
        }
        
        severity = severity_map.get(record.level, 4)
        
        # Build extension fields
        extensions = []
        if record.structured_data:
            for key, value in record.structured_data.items():
                extensions.append(f"{key}={value}")
        
        if record.trace_context:
            extensions.append(f"traceId={record.trace_context.trace_id}")
        
        extension_str = ' '.join(extensions)
        
        return (f"CEF:0|QuantRS2|Quantum Computing Framework|1.0|"
                f"{record.event_type.value}|{record.message}|{severity}|{extension_str}")
    
    def _format_gelf(self, record: LogRecord) -> str:
        """Format as Graylog Extended Log Format (GELF)."""
        gelf_record = {
            "version": "1.1",
            "host": socket.gethostname(),
            "short_message": record.message,
            "timestamp": record.timestamp,
            "level": self._gelf_level(record.level),
            "_logger": record.logger_name,
            "_event_type": record.event_type.value,
            "_filename": record.filename,
            "_line_number": record.line_number,
            "_function_name": record.function_name
        }
        
        # Add structured data as custom fields
        if record.structured_data:
            for key, value in record.structured_data.items():
                gelf_record[f"_{key}"] = value
        
        # Add trace context
        if record.trace_context:
            gelf_record["_trace_id"] = record.trace_context.trace_id
            gelf_record["_span_id"] = record.trace_context.span_id
        
        # Add error information
        if record.error_info:
            gelf_record["_error_id"] = record.error_info.error_id
            gelf_record["_error_type"] = record.error_info.error_type
            gelf_record["_error_category"] = record.error_info.error_category.value
        
        return json.dumps(gelf_record, separators=(',', ':'))
    
    def _format_rfc5424(self, record: LogRecord) -> str:
        """Format as RFC 5424 Syslog."""
        # RFC 5424 format: <priority>VERSION TIMESTAMP HOSTNAME APP-NAME PROCID MSGID STRUCTURED-DATA MSG
        
        priority = self._syslog_priority(record.level)
        timestamp = datetime.fromtimestamp(record.timestamp).isoformat() + 'Z'
        hostname = socket.gethostname()
        app_name = "quantrs2"
        proc_id = "-"
        msg_id = record.event_type.value
        
        # Build structured data
        structured_data = "-"
        if record.structured_data or record.trace_context:
            sd_parts = []
            
            # Add custom structured data
            if record.structured_data:
                sd_elements = []
                for key, value in record.structured_data.items():
                    sd_elements.append(f'{key}="{value}"')
                if sd_elements:
                    sd_parts.append(f"[quantrs2@0 {' '.join(sd_elements)}]")
            
            # Add trace context
            if record.trace_context:
                trace_elements = [
                    f'traceId="{record.trace_context.trace_id}"',
                    f'spanId="{record.trace_context.span_id}"'
                ]
                sd_parts.append(f"[trace@0 {' '.join(trace_elements)}]")
            
            if sd_parts:
                structured_data = ''.join(sd_parts)
        
        return f"<{priority}>1 {timestamp} {hostname} {app_name} {proc_id} {msg_id} {structured_data} {record.message}"
    
    def _format_custom(self, record: LogRecord) -> str:
        """Format using custom format string."""
        if not self.custom_format:
            return self._format_json(record)
        
        # Simple template substitution
        data = record.to_dict()
        
        try:
            # Flatten data for template access
            flat_data = self._flatten_dict(data)
            return self.custom_format.format(**flat_data)
        except (KeyError, ValueError):
            # Fallback to JSON if custom format fails
            return self._format_json(record)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _gelf_level(self, level: str) -> int:
        """Convert log level to GELF level."""
        level_map = {
            'TRACE': 7, 'DEBUG': 7, 'INFO': 6, 'QUANTUM': 6,
            'WARNING': 4, 'ERROR': 3, 'CRITICAL': 2, 'SECURITY': 1
        }
        return level_map.get(level, 6)
    
    def _syslog_priority(self, level: str) -> int:
        """Calculate syslog priority (facility * 8 + severity)."""
        # Using facility 16 (local use 0)
        severity_map = {
            'TRACE': 7, 'DEBUG': 7, 'INFO': 6, 'QUANTUM': 6,
            'WARNING': 4, 'ERROR': 3, 'CRITICAL': 2, 'SECURITY': 0
        }
        facility = 16
        severity = severity_map.get(level, 6)
        return facility * 8 + severity


class LogForwarder:
    """Forwards logs to external destinations."""
    
    def __init__(self, config: LogDestinationConfig):
        self.config = config
        self.formatter = LogFormatter(config.log_format, config.custom_format)
        self.metrics = LogMetrics()
        
        # Buffering
        self._buffer: deque = deque(maxlen=config.buffer_size)
        self._buffer_lock = threading.RLock()
        
        # Flush thread
        self._flush_thread = None
        self._shutdown = threading.Event()
        
        if config.enabled:
            self._start_flush_thread()
    
    def _start_flush_thread(self):
        """Start background flush thread."""
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
    
    def _flush_loop(self):
        """Background flush loop."""
        while not self._shutdown.wait(self.config.flush_interval):
            try:
                self._flush_buffer()
            except Exception as e:
                logger.error(f"Log flush error for {self.config.name}: {e}")
                self.metrics.errors_encountered += 1
    
    def forward_log(self, record: LogRecord) -> bool:
        """Forward a log record."""
        if not self._should_forward(record):
            return False
        
        start_time = time.time()
        
        try:
            # Format the record
            formatted_log = self.formatter.format(record)
            
            # Add to buffer
            with self._buffer_lock:
                self._buffer.append(formatted_log)
                
                # Flush if buffer is full
                if len(self._buffer) >= self.config.buffer_size:
                    self._flush_buffer()
            
            self.metrics.logs_processed += 1
            self.metrics.add_processing_time(time.time() - start_time)
            
            return True
            
        except Exception as e:
            logger.error(f"Log forwarding error for {self.config.name}: {e}")
            self.metrics.errors_encountered += 1
            self.metrics.logs_dropped += 1
            return False
    
    def _should_forward(self, record: LogRecord) -> bool:
        """Check if record should be forwarded based on filters."""
        # Level filter
        if self.config.level_filter and record.level not in self.config.level_filter:
            return False
        
        # Event type filter
        if self.config.event_type_filter and record.event_type.value not in self.config.event_type_filter:
            return False
        
        # Tag filters
        if self.config.tag_filters:
            for tag_key, allowed_values in self.config.tag_filters.items():
                record_value = record.tags.get(tag_key)
                if record_value not in allowed_values:
                    return False
        
        return True
    
    def _flush_buffer(self):
        """Flush buffered logs."""
        with self._buffer_lock:
            if not self._buffer:
                return
            
            # Copy buffer and clear
            logs_to_send = list(self._buffer)
            self._buffer.clear()
        
        # Send logs
        self._send_logs(logs_to_send)
    
    def _send_logs(self, logs: List[str]):
        """Send logs to destination."""
        if not logs:
            return
        
        start_time = time.time()
        
        try:
            if self.config.destination_type == LogDestination.HTTP_ENDPOINT:
                self._send_http(logs)
            elif self.config.destination_type == LogDestination.SYSLOG:
                self._send_syslog(logs)
            elif self.config.destination_type == LogDestination.FILE:
                self._send_file(logs)
            elif self.config.destination_type == LogDestination.ELASTICSEARCH:
                self._send_elasticsearch(logs)
            else:
                logger.warning(f"Unsupported destination type: {self.config.destination_type}")
                return
            
            self.metrics.logs_forwarded += len(logs)
            self.metrics.add_forwarding_time(time.time() - start_time)
            
        except Exception as e:
            logger.error(f"Failed to send logs to {self.config.name}: {e}")
            self.metrics.errors_encountered += 1
            self.metrics.logs_dropped += len(logs)
    
    def _send_http(self, logs: List[str]):
        """Send logs via HTTP."""
        try:
            import requests
            
            # Prepare payload
            if self.config.log_format == LogFormat.JSON:
                # Send as JSON array
                payload = '[' + ','.join(logs) + ']'
                headers = {'Content-Type': 'application/json'}
            else:
                # Send as plain text
                payload = '\n'.join(logs)
                headers = {'Content-Type': 'text/plain'}
            
            # Add authentication
            if self.config.api_key:
                headers['Authorization'] = f'Bearer {self.config.api_key}'
            elif self.config.username and self.config.password:
                headers['Authorization'] = f'Basic {self.config.username}:{self.config.password}'
            
            # Compress if enabled
            if self.config.compression:
                payload = gzip.compress(payload.encode())
                headers['Content-Encoding'] = 'gzip'
            
            # Send request
            response = requests.post(
                self.config.endpoint,
                data=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
        except ImportError:
            logger.error("requests module not available for HTTP log forwarding")
            raise
        except Exception as e:
            logger.error(f"HTTP log forwarding failed: {e}")
            raise
    
    def _send_syslog(self, logs: List[str]):
        """Send logs via syslog."""
        try:
            # Use UDP syslog
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            host = self.config.endpoint or 'localhost'
            port = self.config.port or 514
            
            for log in logs:
                sock.sendto(log.encode('utf-8'), (host, port))
            
            sock.close()
            
        except Exception as e:
            logger.error(f"Syslog forwarding failed: {e}")
            raise
    
    def _send_file(self, logs: List[str]):
        """Send logs to file."""
        try:
            file_path = self.config.endpoint or f'/tmp/quantrs2_{self.config.name}.log'
            
            with open(file_path, 'a', encoding='utf-8') as f:
                for log in logs:
                    f.write(log + '\n')
                f.flush()
            
        except Exception as e:
            logger.error(f"File log forwarding failed: {e}")
            raise
    
    def _send_elasticsearch(self, logs: List[str]):
        """Send logs to Elasticsearch."""
        try:
            import requests
            
            # Prepare bulk request
            bulk_data = []
            for log in logs:
                # Add index action
                index_action = {
                    "index": {
                        "_index": self.config.custom_settings.get('index', 'quantrs2-logs'),
                        "_type": "_doc"
                    }
                }
                bulk_data.append(json.dumps(index_action))
                bulk_data.append(log)
            
            payload = '\n'.join(bulk_data) + '\n'
            
            # Send to Elasticsearch bulk API
            url = f"{self.config.endpoint}/_bulk"
            headers = {'Content-Type': 'application/x-ndjson'}
            
            if self.config.username and self.config.password:
                auth = (self.config.username, self.config.password)
            else:
                auth = None
            
            response = requests.post(
                url,
                data=payload,
                headers=headers,
                auth=auth,
                timeout=30
            )
            response.raise_for_status()
            
        except ImportError:
            logger.error("requests module not available for Elasticsearch forwarding")
            raise
        except Exception as e:
            logger.error(f"Elasticsearch forwarding failed: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get forwarder metrics."""
        return {
            'logs_processed': self.metrics.logs_processed,
            'logs_dropped': self.metrics.logs_dropped,
            'logs_forwarded': self.metrics.logs_forwarded,
            'errors_encountered': self.metrics.errors_encountered,
            'average_processing_time_ms': self.metrics.get_average_processing_time() * 1000,
            'average_forwarding_time_ms': self.metrics.get_average_forwarding_time() * 1000,
            'buffer_size': len(self._buffer),
            'enabled': self.config.enabled
        }
    
    def close(self):
        """Close forwarder and flush remaining logs."""
        self._shutdown.set()
        
        if self._flush_thread:
            self._flush_thread.join(timeout=5)
        
        # Final flush
        self._flush_buffer()


class LogAnalyzer:
    """Analyzes log patterns and generates insights."""
    
    def __init__(self, analysis_window: int = 3600):  # 1 hour
        self.analysis_window = analysis_window
        self._log_history: List[Tuple[float, LogRecord]] = []
        self._lock = threading.RLock()
    
    def add_log(self, record: LogRecord):
        """Add log record for analysis."""
        with self._lock:
            timestamp = record.timestamp
            self._log_history.append((timestamp, record))
            
            # Clean up old logs
            cutoff_time = time.time() - self.analysis_window
            self._log_history = [
                (t, r) for t, r in self._log_history if t >= cutoff_time
            ]
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze log patterns and return insights."""
        with self._lock:
            if not self._log_history:
                return {}
            
            analysis = {
                'total_logs': len(self._log_history),
                'time_window_hours': self.analysis_window / 3600,
                'log_rate_per_hour': len(self._log_history) / (self.analysis_window / 3600),
                'by_level': self._analyze_by_level(),
                'by_event_type': self._analyze_by_event_type(),
                'by_logger': self._analyze_by_logger(),
                'error_rate': self._calculate_error_rate(),
                'performance_insights': self._analyze_performance(),
                'anomalies': self._detect_anomalies()
            }
            
            return analysis
    
    def _analyze_by_level(self) -> Dict[str, int]:
        """Analyze logs by level."""
        level_counts = defaultdict(int)
        for _, record in self._log_history:
            level_counts[record.level] += 1
        return dict(level_counts)
    
    def _analyze_by_event_type(self) -> Dict[str, int]:
        """Analyze logs by event type."""
        event_counts = defaultdict(int)
        for _, record in self._log_history:
            event_counts[record.event_type.value] += 1
        return dict(event_counts)
    
    def _analyze_by_logger(self) -> Dict[str, int]:
        """Analyze logs by logger name."""
        logger_counts = defaultdict(int)
        for _, record in self._log_history:
            logger_counts[record.logger_name] += 1
        return dict(logger_counts)
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate as percentage."""
        total_logs = len(self._log_history)
        if total_logs == 0:
            return 0.0
        
        error_logs = sum(
            1 for _, record in self._log_history
            if record.level in ['ERROR', 'CRITICAL']
        )
        
        return (error_logs / total_logs) * 100
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance-related logs."""
        performance_logs = [
            record for _, record in self._log_history
            if record.event_type == EventType.PERFORMANCE_EVENT
        ]
        
        if not performance_logs:
            return {}
        
        durations = []
        for record in performance_logs:
            if record.duration_ms:
                durations.append(record.duration_ms)
            elif 'duration_ms' in record.structured_data:
                durations.append(record.structured_data['duration_ms'])
        
        if durations:
            return {
                'performance_logs_count': len(performance_logs),
                'average_duration_ms': sum(durations) / len(durations),
                'max_duration_ms': max(durations),
                'min_duration_ms': min(durations)
            }
        
        return {'performance_logs_count': len(performance_logs)}
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in log patterns."""
        anomalies = []
        
        # Check for error bursts
        error_logs = [
            (timestamp, record) for timestamp, record in self._log_history
            if record.level in ['ERROR', 'CRITICAL']
        ]
        
        if len(error_logs) > 10:  # Threshold for error burst
            anomalies.append({
                'type': 'error_burst',
                'description': f'High error rate detected: {len(error_logs)} errors',
                'severity': 'high' if len(error_logs) > 50 else 'medium'
            })
        
        # Check for unusual logger activity
        logger_counts = self._analyze_by_logger()
        if logger_counts:
            max_count = max(logger_counts.values())
            avg_count = sum(logger_counts.values()) / len(logger_counts)
            
            if max_count > avg_count * 5:  # One logger dominates
                dominant_logger = max(logger_counts.keys(), key=lambda k: logger_counts[k])
                anomalies.append({
                    'type': 'dominant_logger',
                    'description': f'Logger {dominant_logger} produced {max_count} logs',
                    'severity': 'medium'
                })
        
        return anomalies


class LogAggregationSystem:
    """Main log aggregation and forwarding system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Components
        self.forwarders: Dict[str, LogForwarder] = {}
        self.analyzer = LogAnalyzer(
            analysis_window=self.config.get('analysis_window', 3600)
        )
        
        # Error analysis integration
        self.error_analysis = None
        if self.config.get('enable_error_analysis', True):
            try:
                self.error_analysis = ErrorAnalysisSystem(
                    self.config.get('error_analysis_config', {})
                )
            except Exception as e:
                logger.error(f"Failed to initialize error analysis: {e}")
        
        # Processing metrics
        self.total_logs_processed = 0
        self.start_time = time.time()
        
        # Setup structured logging integration
        self._setup_logging_integration()
        
        logger.info("Log aggregation system initialized")
    
    def _setup_logging_integration(self):
        """Setup integration with structured logging system."""
        # Add ourselves as a handler to the global logging system
        logging_system = get_global_logging_system()
        
        # Create a handler that forwards to our system
        def log_handler(record: LogRecord):
            self.process_log(record)
        
        # Get a logger and add our handler
        system_logger = logging_system.get_logger('quantrs2.log_aggregation')
        system_logger.add_handler(log_handler)
    
    def add_destination(self, config: LogDestinationConfig):
        """Add log destination."""
        try:
            forwarder = LogForwarder(config)
            self.forwarders[config.name] = forwarder
            logger.info(f"Added log destination: {config.name} ({config.destination_type.value})")
            
        except Exception as e:
            logger.error(f"Failed to add log destination {config.name}: {e}")
            raise
    
    def remove_destination(self, name: str):
        """Remove log destination."""
        if name in self.forwarders:
            forwarder = self.forwarders[name]
            forwarder.close()
            del self.forwarders[name]
            logger.info(f"Removed log destination: {name}")
    
    def process_log(self, record: LogRecord):
        """Process a log record."""
        self.total_logs_processed += 1
        
        # Add to analyzer
        self.analyzer.add_log(record)
        
        # Forward to all destinations
        for forwarder in self.forwarders.values():
            try:
                forwarder.forward_log(record)
            except Exception as e:
                logger.error(f"Forwarder error: {e}")
        
        # Error analysis integration
        if self.error_analysis and record.error_info:
            try:
                self.error_analysis.analyze_error(record.error_info)
            except Exception as e:
                logger.error(f"Error analysis failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        uptime = time.time() - self.start_time
        
        status = {
            'uptime_seconds': uptime,
            'uptime_hours': uptime / 3600,
            'total_logs_processed': self.total_logs_processed,
            'logs_per_second': self.total_logs_processed / uptime if uptime > 0 else 0,
            'destinations': {},
            'log_analysis': self.analyzer.analyze_patterns()
        }
        
        # Destination metrics
        for name, forwarder in self.forwarders.items():
            status['destinations'][name] = forwarder.get_metrics()
        
        # Error analysis status
        if self.error_analysis:
            try:
                status['error_analysis'] = self.error_analysis.get_analysis_report()
            except Exception as e:
                logger.error(f"Error getting analysis report: {e}")
                status['error_analysis'] = {'error': str(e)}
        
        return status
    
    def create_default_destinations(self):
        """Create default log destinations."""
        # Console JSON logs
        console_config = LogDestinationConfig(
            destination_type=LogDestination.FILE,
            name="console_json",
            endpoint="/dev/stdout",
            log_format=LogFormat.JSON,
            level_filter=["INFO", "WARNING", "ERROR", "CRITICAL", "QUANTUM"]
        )
        
        # Error logs file
        error_config = LogDestinationConfig(
            destination_type=LogDestination.FILE,
            name="error_logs",
            endpoint=str(Path.home() / '.quantrs2' / 'logs' / 'errors.log'),
            log_format=LogFormat.JSON,
            level_filter=["ERROR", "CRITICAL"],
            buffer_size=100,
            flush_interval=10
        )
        
        # Performance logs file
        performance_config = LogDestinationConfig(
            destination_type=LogDestination.FILE,
            name="performance_logs",
            endpoint=str(Path.home() / '.quantrs2' / 'logs' / 'performance.log'),
            log_format=LogFormat.JSON,
            event_type_filter=["performance_event", "quantum_execution"],
            buffer_size=500,
            flush_interval=30
        )
        
        # Security logs file
        security_config = LogDestinationConfig(
            destination_type=LogDestination.FILE,
            name="security_logs", 
            endpoint=str(Path.home() / '.quantrs2' / 'logs' / 'security.log'),
            log_format=LogFormat.JSON,
            level_filter=["SECURITY"],
            event_type_filter=["security_event", "audit_event"],
            buffer_size=50,
            flush_interval=5
        )
        
        # Add destinations
        for config in [console_config, error_config, performance_config, security_config]:
            try:
                self.add_destination(config)
            except Exception as e:
                logger.error(f"Failed to create default destination {config.name}: {e}")
    
    def close(self):
        """Close log aggregation system."""
        logger.info("Closing log aggregation system")
        
        # Close all forwarders
        for forwarder in self.forwarders.values():
            forwarder.close()
        
        # Close error analysis
        if self.error_analysis:
            self.error_analysis.close()


# Global log aggregation system
_global_aggregation_system: Optional[LogAggregationSystem] = None
_global_lock = threading.RLock()


def get_global_aggregation_system() -> LogAggregationSystem:
    """Get global log aggregation system."""
    global _global_aggregation_system
    
    with _global_lock:
        if _global_aggregation_system is None:
            _global_aggregation_system = LogAggregationSystem()
        
        return _global_aggregation_system


def configure_log_aggregation(config: Dict[str, Any]):
    """Configure global log aggregation system."""
    global _global_aggregation_system
    
    with _global_lock:
        if _global_aggregation_system:
            _global_aggregation_system.close()
        
        _global_aggregation_system = LogAggregationSystem(config)


# Export main classes
__all__ = [
    'LogDestination',
    'LogFormat',
    'LogDestinationConfig',
    'LogMetrics',
    'LogFormatter',
    'LogForwarder',
    'LogAnalyzer',
    'LogAggregationSystem',
    'get_global_aggregation_system',
    'configure_log_aggregation'
]