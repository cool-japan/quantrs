"""
Tests for Structured Logging and Error Tracking System

This module tests the comprehensive structured logging, error tracking,
and log aggregation capabilities.
"""

import pytest
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Safe import pattern for structured logging
HAS_STRUCTURED_LOGGING = True
try:
    from quantrs2.structured_logging import (
        LoggingSystem, StructuredLogger, TraceManager, ErrorTracker,
        LogLevel, EventType, ErrorCategory, TraceContext, ErrorInfo,
        LogRecord, JSONLogHandler, ConsoleLogHandler, PerformanceLogger,
        log_function_calls, log_quantum_operation, get_logger
    )
except ImportError as e:
    HAS_STRUCTURED_LOGGING = False
    
    # Create stub implementations
    from enum import Enum
    from uuid import uuid4
    
    class LogLevel(Enum):
        DEBUG = "DEBUG"
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"
        QUANTUM = "QUANTUM"
    
    class EventType(Enum):
        APPLICATION = "application"
        QUANTUM_EXECUTION = "quantum_execution"
        PERFORMANCE_EVENT = "performance_event"
        AUDIT_EVENT = "audit_event"
    
    class ErrorCategory(Enum):
        VALIDATION = "validation"
        NETWORK = "network"
        SYSTEM = "system"
    
    class TraceContext:
        def __init__(self, trace_id=None, span_id=None, parent_span_id=None, operation_name="", tags=None):
            self.trace_id = trace_id or str(uuid4())
            self.span_id = span_id or str(uuid4())
            self.parent_span_id = parent_span_id
            self.operation_name = operation_name
            self.tags = tags or {}
            self.start_time = time.time()
    
    class ErrorInfo:
        def __init__(self, error_id=None, error_type="", error_category=None, message="", context=None, occurred_at=None, resolved=False, resolution_notes=None):
            self.error_id = error_id or str(uuid4())
            self.error_type = error_type
            self.error_category = error_category
            self.message = message
            self.context = context or {}
            self.occurred_at = occurred_at or time.time()
            self.resolved = resolved
            self.resolution_notes = resolution_notes
    
    class LogRecord:
        def __init__(self, timestamp=None, level="INFO", message="", event_type=EventType.APPLICATION, logger_name="", structured_data=None, tags=None, trace_context=None, error_info=None, filename=None, line_number=None, function_name=None):
            self.timestamp = timestamp or time.time()
            self.level = level
            self.message = message
            self.event_type = event_type
            self.logger_name = logger_name
            self.structured_data = structured_data or {}
            self.tags = tags or {}
            self.trace_context = trace_context
            self.error_info = error_info
            self.filename = filename
            self.line_number = line_number
            self.function_name = function_name
    
    class TraceManager:
        def __init__(self):
            self._current_context = None
        def start_trace(self, operation_name, tags=None):
            context = TraceContext(operation_name=operation_name, tags=tags)
            self._current_context = context
            return context
        def start_span(self, operation_name, tags=None):
            parent_context = self._current_context
            context = TraceContext(
                trace_id=parent_context.trace_id if parent_context else None,
                parent_span_id=parent_context.span_id if parent_context else None,
                operation_name=operation_name, tags=tags
            )
            return context
        def trace_span(self, operation_name, tags=None):
            return TraceSpanContext(self, operation_name, tags)
    
    class TraceSpanContext:
        def __init__(self, trace_manager, operation_name, tags):
            self.trace_manager = trace_manager
            self.operation_name = operation_name
            self.tags = tags or {}
        def __enter__(self):
            self.context = self.trace_manager.start_span(self.operation_name, self.tags)
            return self.context
        def __exit__(self, *args):
            duration = (time.time() - self.context.start_time) * 1000
            self.context.tags["duration_ms"] = duration
    
    class ErrorTracker:
        def __init__(self, max_errors=1000):
            self.max_errors = max_errors
            self._errors = {}
        def track_error(self, error, context=None, category=None):
            error_info = ErrorInfo(
                error_type=type(error).__name__,
                error_category=category,
                message=str(error),
                context=context or {}
            )
            self._errors[error_info.error_id] = error_info
            if len(self._errors) > self.max_errors:
                oldest_key = min(self._errors.keys(), key=lambda k: self._errors[k].occurred_at)
                del self._errors[oldest_key]
            return error_info
        def resolve_error(self, error_id, resolution_notes):
            if error_id in self._errors:
                self._errors[error_id].resolved = True
                self._errors[error_id].resolution_notes = resolution_notes
        def get_error_statistics(self):
            total = len(self._errors)
            unresolved = sum(1 for e in self._errors.values() if not e.resolved)
            by_category = {}
            for error in self._errors.values():
                if error.error_category:
                    key = error.error_category.value if hasattr(error.error_category, 'value') else str(error.error_category)
                    by_category[key] = by_category.get(key, 0) + 1
            return {'total_errors': total, 'unresolved_errors': unresolved, 'by_category': by_category}
    
    class StructuredLogger:
        def __init__(self, name, trace_manager=None, error_tracker=None):
            self.name = name
            self.trace_manager = trace_manager or TraceManager()
            self.error_tracker = error_tracker or ErrorTracker()
            self._handlers = []
            self._filters = []
        def add_handler(self, handler): self._handlers.append(handler)
        def add_filter(self, filter_func): self._filters.append(filter_func)
        def _log(self, level, message, **kwargs):
            record = LogRecord(level=level, message=message, logger_name=self.name, **kwargs)
            for filter_func in self._filters:
                if not filter_func(record):
                    return
            for handler in self._handlers:
                handler(record)
        def info(self, message, **kwargs): self._log("INFO", message, **kwargs)
        def error(self, message, error=None, error_category=None, **kwargs):
            if error:
                error_info = self.error_tracker.track_error(error, kwargs.get('context'), error_category)
                kwargs['error_info'] = error_info
            self._log("ERROR", message, **kwargs)
        def quantum(self, message, **kwargs): self._log("QUANTUM", message, event_type=EventType.QUANTUM_EXECUTION, **kwargs)
        def performance(self, message, **kwargs): self._log("INFO", message, event_type=EventType.PERFORMANCE_EVENT, **kwargs)
        def audit(self, message, user_id=None, action=None, resource=None, **kwargs):
            data = kwargs.get('structured_data', {})
            if user_id: data['user_id'] = user_id
            if action: data['action'] = action
            if resource: data['resource'] = resource
            kwargs['structured_data'] = data
            self._log("INFO", message, event_type=EventType.AUDIT_EVENT, **kwargs)
        def trace(self, message, **kwargs): self._log("DEBUG", message, **kwargs)
    
    class JSONLogHandler:
        def __init__(self, filename):
            self.filename = filename
            self._file = None
        def __call__(self, record):
            if not self._file:
                self._file = open(self.filename, 'a')
            import json
            data = {
                'timestamp': record.timestamp,
                'level': record.level,
                'message': record.message,
                'logger_name': record.logger_name,
                'structured_data': record.structured_data
            }
            self._file.write(json.dumps(data) + '\n')
            self._file.flush()
        def close(self):
            if self._file:
                self._file.close()
    
    class ConsoleLogHandler:
        def __init__(self, use_colors=True):
            self.use_colors = use_colors
        def __call__(self, record):
            pass  # In tests, we don't actually print to console
    
    class PerformanceLogger:
        def __init__(self, structured_logger):
            self.logger = structured_logger
        def measure(self, operation_name, tags=None):
            return PerformanceMeasureContext(self, operation_name, tags)
        def log_quantum_execution(self, circuit_info, execution_time_seconds, success, backend):
            self.logger.quantum(
                "Circuit execution completed",
                structured_data={
                    'circuit_info': circuit_info,
                    'execution_time_ms': execution_time_seconds * 1000,
                    'success': success
                },
                tags={'backend': backend}
            )
    
    class PerformanceMeasureContext:
        def __init__(self, perf_logger, operation_name, tags):
            self.perf_logger = perf_logger
            self.operation_name = operation_name
            self.tags = tags or {}
            self.start_time = None
        def __enter__(self):
            self.start_time = time.time()
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = (time.time() - self.start_time) * 1000
            if exc_type:
                self.perf_logger.logger.error(
                    f"Operation failed: {self.operation_name}",
                    error=exc_val,
                    structured_data={'duration_ms': duration},
                    tags=self.tags
                )
            else:
                self.perf_logger.logger.performance(
                    f"Operation completed: {self.operation_name}",
                    structured_data={'duration_ms': duration},
                    tags=self.tags
                )
    
    class LoggingSystem:
        def __init__(self, config):
            self.config = config
            self.trace_manager = TraceManager()
            self.error_tracker = ErrorTracker()
        def get_logger(self, name):
            logger = StructuredLogger(name, self.trace_manager, self.error_tracker)
            if 'json_log_file' in self.config:
                logger.add_handler(JSONLogHandler(self.config['json_log_file']))
            return logger
        def get_performance_logger(self, name):
            return PerformanceLogger(self.get_logger(name))
        def get_error_statistics(self):
            return self.error_tracker.get_error_statistics()
        def close(self): pass
    
    def log_function_calls(logger_name="", include_args=False):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def log_quantum_operation(circuit_type=""):
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_logger(name):
        return StructuredLogger(name)

HAS_ERROR_ANALYSIS = True
try:
    from quantrs2.error_analysis import (
        ErrorAnalysisSystem, ErrorPatternDetector, IncidentManager,
        ErrorPattern, IncidentSeverity, IncidentStatus
    )
except ImportError as e:
    HAS_ERROR_ANALYSIS = False
    
    from enum import Enum
    
    class ErrorPattern(Enum):
        FREQUENT_ERRORS = "frequent_errors"
        ERROR_BURST = "error_burst"
        TIMEOUT_CLUSTER = "timeout_cluster"
    
    class IncidentSeverity(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    class IncidentStatus(Enum):
        OPEN = "open"
        RESOLVED = "resolved"
        CLOSED = "closed"
    
    class ErrorPatternMatch:
        def __init__(self, pattern_type, confidence, errors, time_window, description, severity):
            self.pattern_type = pattern_type
            self.confidence = confidence
            self.errors = errors
            self.time_window = time_window
            self.description = description
            self.severity = severity
    
    class ErrorPatternDetector:
        def __init__(self, detection_window=3600):
            self.detection_window = detection_window
            self._errors = []
        def add_error(self, error_info):
            self._errors.append(error_info)
        def detect_patterns(self):
            patterns = []
            # Frequent errors detection
            if len(self._errors) >= 5:
                patterns.append(ErrorPatternMatch(
                    ErrorPattern.FREQUENT_ERRORS, 0.8, self._errors,
                    (time.time() - 300, time.time()),
                    "Frequent errors detected", IncidentSeverity.HIGH
                ))
            # Error burst detection
            recent_errors = [e for e in self._errors if time.time() - e.occurred_at < 300]
            if len(recent_errors) >= 10:
                patterns.append(ErrorPatternMatch(
                    ErrorPattern.ERROR_BURST, 0.9, recent_errors,
                    (time.time() - 300, time.time()),
                    "Error burst detected", IncidentSeverity.CRITICAL
                ))
            return patterns
    
    class Incident:
        def __init__(self, incident_id, title, description, severity, error_patterns, assignee=None):
            self.incident_id = incident_id
            self.title = title
            self.description = description
            self.severity = severity
            self.status = IncidentStatus.OPEN
            self.error_patterns = error_patterns
            self.assignee = assignee
            self.created_at = time.time()
            self.resolved_at = None
            self.resolution = None
            self.root_cause = None
    
    class IncidentManager:
        def __init__(self, db_path):
            self.db_path = db_path
            self._incidents = {}
        def create_incident(self, title, description, severity, error_patterns, assignee=None):
            incident_id = str(uuid4())
            incident = Incident(incident_id, title, description, severity, error_patterns, assignee)
            self._incidents[incident_id] = incident
            return incident
        def resolve_incident(self, incident_id, resolution, root_cause=None):
            if incident_id in self._incidents:
                incident = self._incidents[incident_id]
                incident.status = IncidentStatus.RESOLVED
                incident.resolution = resolution
                incident.root_cause = root_cause
                incident.resolved_at = time.time()
                return True
            return False
    
    class ErrorAnalysisSystem:
        def __init__(self, config):
            self.config = config
            self.detector = ErrorPatternDetector()
            self.incident_manager = IncidentManager(config.get('incidents_db_path', 'incidents.db'))
        def analyze_error(self, error_info):
            self.detector.add_error(error_info)
            return {
                'error_id': error_info.error_id,
                'category': error_info.error_category.value if error_info.error_category else 'unknown'
            }
        def get_analysis_report(self):
            return {
                'pattern_detection': {'total_errors_tracked': len(self.detector._errors)},
                'incident_management': {'total_incidents': len(self.incident_manager._incidents)}
            }
        def close(self): pass

HAS_LOG_AGGREGATION = True
try:
    from quantrs2.log_aggregation import (
        LogAggregationSystem, LogDestinationConfig, LogDestination,
        LogFormat, LogForwarder, LogAnalyzer
    )
except ImportError as e:
    HAS_LOG_AGGREGATION = False
    
    from enum import Enum
    
    class LogDestination(Enum):
        FILE = "file"
        HTTP = "http"
        SYSLOG = "syslog"
    
    class LogFormat(Enum):
        JSON = "json"
        TEXT = "text"
    
    class LogDestinationConfig:
        def __init__(self, destination_type, name, endpoint, log_format=LogFormat.JSON, enabled=True, level_filter=None, event_type_filter=None):
            self.destination_type = destination_type
            self.name = name
            self.endpoint = endpoint
            self.log_format = log_format
            self.enabled = enabled
            self.level_filter = level_filter or []
            self.event_type_filter = event_type_filter or []
    
    class LogFormatter:
        def __init__(self, log_format):
            self.log_format = log_format
    
    class LogForwarder:
        def __init__(self, config):
            self.config = config
            self.formatter = LogFormatter(config.log_format)
        def _should_forward(self, record):
            if self.config.level_filter and record.level not in self.config.level_filter:
                return False
            if self.config.event_type_filter and record.event_type.value not in self.config.event_type_filter:
                return False
            return True
    
    class LogAnalyzer:
        def __init__(self, analysis_window=3600):
            self.analysis_window = analysis_window
            self._logs = []
        def add_log(self, record):
            self._logs.append(record)
        def analyze_patterns(self):
            total = len(self._logs)
            error_count = sum(1 for log in self._logs if log.level == "ERROR")
            by_level = {}
            by_logger = {}
            for log in self._logs:
                by_level[log.level] = by_level.get(log.level, 0) + 1
                by_logger[log.logger_name] = by_logger.get(log.logger_name, 0) + 1
            return {
                'total_logs': total,
                'error_rate': (error_count / total * 100) if total > 0 else 0,
                'by_level': by_level,
                'by_logger': by_logger
            }


@pytest.mark.skipif(not HAS_STRUCTURED_LOGGING, reason="quantrs2.structured_logging not available")
class TestTraceManager:
    """Test distributed tracing functionality."""
    
    def test_trace_creation(self):
        """Test creating traces and spans."""
        trace_manager = TraceManager()
        
        # Start a trace
        context = trace_manager.start_trace("test_operation", {"component": "test"})
        
        assert context.trace_id is not None
        assert context.span_id is not None
        assert context.operation_name == "test_operation"
        assert context.tags["component"] == "test"
        assert context.parent_span_id is None
    
    def test_span_hierarchy(self):
        """Test span parent-child relationships."""
        trace_manager = TraceManager()
        
        # Start parent trace
        parent_context = trace_manager.start_trace("parent_operation")
        parent_trace_id = parent_context.trace_id
        parent_span_id = parent_context.span_id
        
        # Start child span
        child_context = trace_manager.start_span("child_operation")
        
        assert child_context.trace_id == parent_trace_id
        assert child_context.parent_span_id == parent_span_id
        assert child_context.span_id != parent_span_id
    
    def test_trace_context_manager(self):
        """Test trace span context manager."""
        trace_manager = TraceManager()
        
        with trace_manager.trace_span("test_span", {"test": "value"}) as context:
            assert context.operation_name == "test_span"
            assert context.tags["test"] == "value"
            
            start_time = context.start_time
            time.sleep(0.01)  # Small delay
            
        # After context, duration should be calculated
        assert "duration_ms" in context.tags
        assert context.tags["duration_ms"] > 0


@pytest.mark.skipif(not HAS_STRUCTURED_LOGGING, reason="quantrs2.structured_logging not available")
class TestErrorTracker:
    """Test error tracking functionality."""
    
    def test_error_tracking(self):
        """Test basic error tracking."""
        error_tracker = ErrorTracker(max_errors=100)
        
        # Create test error
        test_error = ValueError("Test error message")
        context = {"function": "test_function", "line": 42}
        
        # Track error
        error_info = error_tracker.track_error(test_error, context, ErrorCategory.VALIDATION)
        
        assert error_info.error_id is not None
        assert error_info.error_type == "ValueError"
        assert error_info.error_category == ErrorCategory.VALIDATION
        assert error_info.message == "Test error message"
        assert error_info.context == context
        assert not error_info.resolved
    
    def test_error_resolution(self):
        """Test error resolution."""
        error_tracker = ErrorTracker()
        
        test_error = RuntimeError("Runtime error")
        error_info = error_tracker.track_error(test_error)
        
        # Resolve error
        error_tracker.resolve_error(error_info.error_id, "Fixed by restart")
        
        assert error_info.resolved
        assert error_info.resolution_notes == "Fixed by restart"
    
    def test_error_statistics(self):
        """Test error statistics."""
        error_tracker = ErrorTracker()
        
        # Track multiple errors
        for i in range(5):
            error = ValueError(f"Error {i}")
            error_tracker.track_error(error, category=ErrorCategory.VALIDATION)
        
        for i in range(3):
            error = ConnectionError(f"Connection error {i}")
            error_tracker.track_error(error, category=ErrorCategory.NETWORK)
        
        stats = error_tracker.get_error_statistics()
        
        assert stats['total_errors'] == 8
        assert stats['unresolved_errors'] == 8
        assert stats['by_category'][ErrorCategory.VALIDATION.value] == 5
        assert stats['by_category'][ErrorCategory.NETWORK.value] == 3
    
    def test_error_cleanup(self):
        """Test error cleanup when max limit is reached."""
        error_tracker = ErrorTracker(max_errors=3)
        
        # Add more errors than the limit
        for i in range(5):
            error = ValueError(f"Error {i}")
            error_tracker.track_error(error)
        
        stats = error_tracker.get_error_statistics()
        assert stats['total_errors'] <= 3  # Should be cleaned up


@pytest.mark.skipif(not HAS_STRUCTURED_LOGGING, reason="quantrs2.structured_logging not available")
class TestStructuredLogger:
    """Test structured logging functionality."""
    
    @pytest.fixture
    def logger_setup(self):
        """Setup logger for testing."""
        trace_manager = TraceManager()
        error_tracker = ErrorTracker()
        logger = StructuredLogger("test_logger", trace_manager, error_tracker)
        
        # Mock handler to capture logs
        captured_logs = []
        
        def mock_handler(record: LogRecord):
            captured_logs.append(record)
        
        logger.add_handler(mock_handler)
        
        return logger, captured_logs
    
    def test_basic_logging(self, logger_setup):
        """Test basic logging functionality."""
        logger, captured_logs = logger_setup
        
        logger.info("Test message", structured_data={"key": "value"}, tags={"env": "test"})
        
        assert len(captured_logs) == 1
        record = captured_logs[0]
        
        assert record.message == "Test message"
        assert record.level == "INFO"
        assert record.structured_data["key"] == "value"
        assert record.tags["env"] == "test"
        assert record.logger_name == "test_logger"
    
    def test_error_logging(self, logger_setup):
        """Test error logging with automatic error tracking."""
        logger, captured_logs = logger_setup
        
        test_error = RuntimeError("Test runtime error")
        logger.error("Error occurred", error=test_error, error_category=ErrorCategory.SYSTEM)
        
        assert len(captured_logs) == 1
        record = captured_logs[0]
        
        assert record.level == "ERROR"
        assert record.error_info is not None
        assert record.error_info.error_type == "RuntimeError"
        assert record.error_info.message == "Test runtime error"
        assert record.error_info.error_category == ErrorCategory.SYSTEM
    
    def test_quantum_logging(self, logger_setup):
        """Test quantum-specific logging."""
        logger, captured_logs = logger_setup
        
        logger.quantum("Circuit executed", structured_data={"qubits": 5, "depth": 10})
        
        assert len(captured_logs) == 1
        record = captured_logs[0]
        
        assert record.level == "QUANTUM"
        assert record.event_type == EventType.QUANTUM_EXECUTION
        assert record.structured_data["qubits"] == 5
    
    def test_performance_logging(self, logger_setup):
        """Test performance logging."""
        logger, captured_logs = logger_setup
        
        logger.performance("Operation completed", duration_ms=150.5, tags={"operation": "test"})
        
        assert len(captured_logs) == 1
        record = captured_logs[0]
        
        assert record.event_type == EventType.PERFORMANCE_EVENT
        assert record.structured_data["duration_ms"] == 150.5
        assert record.tags["operation"] == "test"
    
    def test_audit_logging(self, logger_setup):
        """Test audit logging."""
        logger, captured_logs = logger_setup
        
        logger.audit("User action", user_id="user123", action="circuit_execution", resource="quantum_circuit")
        
        assert len(captured_logs) == 1
        record = captured_logs[0]
        
        assert record.event_type == EventType.AUDIT_EVENT
        assert record.structured_data["user_id"] == "user123"
        assert record.structured_data["action"] == "circuit_execution"
        assert record.structured_data["resource"] == "quantum_circuit"
    
    def test_log_filtering(self, logger_setup):
        """Test log filtering."""
        logger, captured_logs = logger_setup
        
        # Add filter that only allows ERROR level
        def error_only_filter(record: LogRecord) -> bool:
            return record.level == "ERROR"
        
        logger.add_filter(error_only_filter)
        
        logger.info("Info message")
        logger.error("Error message")
        
        assert len(captured_logs) == 1
        assert captured_logs[0].level == "ERROR"


@pytest.mark.skipif(not HAS_STRUCTURED_LOGGING, reason="quantrs2.structured_logging not available")
class TestLogHandlers:
    """Test log handlers."""
    
    def test_json_log_handler(self):
        """Test JSON log handler."""
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
            handler = JSONLogHandler(tmp.name)
            
            # Create test record
            record = LogRecord(
                timestamp=time.time(),
                level="INFO",
                message="Test message",
                event_type=EventType.APPLICATION,
                logger_name="test",
                structured_data={"key": "value"}
            )
            
            handler(record)
            handler.close()
            
            # Read and verify JSON output
            with open(tmp.name, 'r') as f:
                content = f.read().strip()
                
            import json
            parsed = json.loads(content)
            assert parsed["message"] == "Test message"
            assert parsed["level"] == "INFO"
            assert parsed["structured_data"]["key"] == "value"
    
    def test_console_log_handler(self):
        """Test console log handler."""
        handler = ConsoleLogHandler(use_colors=False)
        
        record = LogRecord(
            timestamp=time.time(),
            level="WARNING",
            message="Warning message",
            event_type=EventType.APPLICATION,
            logger_name="test",
            filename="test.py",
            line_number=42,
            function_name="test_function"
        )
        
        # This would normally print to console
        # In test, we just ensure it doesn't crash
        handler(record)


@pytest.mark.skipif(not HAS_STRUCTURED_LOGGING, reason="quantrs2.structured_logging not available")
class TestPerformanceLogger:
    """Test performance logging functionality."""
    
    @pytest.fixture
    def performance_logger(self):
        """Setup performance logger."""
        trace_manager = TraceManager()
        error_tracker = ErrorTracker()
        structured_logger = StructuredLogger("perf_test", trace_manager, error_tracker)
        
        captured_logs = []
        
        def mock_handler(record: LogRecord):
            captured_logs.append(record)
        
        structured_logger.add_handler(mock_handler)
        
        perf_logger = PerformanceLogger(structured_logger)
        return perf_logger, captured_logs
    
    def test_performance_measurement(self, performance_logger):
        """Test performance measurement context manager."""
        perf_logger, captured_logs = performance_logger
        
        with perf_logger.measure("test_operation", tags={"component": "test"}):
            time.sleep(0.01)  # Small delay
        
        assert len(captured_logs) == 1
        record = captured_logs[0]
        
        assert record.event_type == EventType.PERFORMANCE_EVENT
        assert "Operation completed: test_operation" in record.message
        assert record.structured_data["duration_ms"] > 0
        assert record.tags["component"] == "test"
    
    def test_performance_measurement_with_error(self, performance_logger):
        """Test performance measurement when error occurs."""
        perf_logger, captured_logs = performance_logger
        
        with pytest.raises(ValueError):
            with perf_logger.measure("failing_operation"):
                time.sleep(0.01)
                raise ValueError("Test error")
        
        assert len(captured_logs) == 1
        record = captured_logs[0]
        
        assert record.level == "ERROR"
        assert "Operation failed: failing_operation" in record.message
        assert record.error_info is not None
        assert record.error_info.error_type == "ValueError"
    
    def test_quantum_execution_logging(self, performance_logger):
        """Test quantum execution performance logging."""
        perf_logger, captured_logs = performance_logger
        
        circuit_info = {"qubits": 3, "depth": 5, "gates": 10}
        perf_logger.log_quantum_execution(circuit_info, 0.15, True, "simulator")
        
        assert len(captured_logs) == 1
        record = captured_logs[0]
        
        assert record.level == "QUANTUM"
        assert record.event_type == EventType.QUANTUM_EXECUTION
        assert "Circuit execution completed" in record.message
        assert record.structured_data["circuit_info"] == circuit_info
        assert record.structured_data["execution_time_ms"] == 150.0
        assert record.structured_data["success"] is True
        assert record.tags["backend"] == "simulator"


@pytest.mark.skipif(not HAS_ERROR_ANALYSIS, reason="quantrs2.error_analysis not available")
class TestErrorPatternDetector:
    """Test error pattern detection."""
    
    def test_frequent_errors_detection(self):
        """Test detection of frequent error patterns."""
        detector = ErrorPatternDetector(detection_window=300)
        
        # Add multiple similar errors
        for i in range(10):
            error_info = ErrorInfo(
                error_id=f"error_{i}",
                error_type="ValueError",
                error_category=ErrorCategory.VALIDATION,
                message="Invalid input parameter",
                occurred_at=time.time() - i
            )
            detector.add_error(error_info)
        
        patterns = detector.detect_patterns()
        
        # Should detect frequent errors pattern
        frequent_patterns = [p for p in patterns if p.pattern_type == ErrorPattern.FREQUENT_ERRORS]
        assert len(frequent_patterns) > 0
        
        pattern = frequent_patterns[0]
        assert len(pattern.errors) == 10
        assert pattern.confidence > 0.0
    
    def test_error_burst_detection(self):
        """Test detection of error bursts."""
        detector = ErrorPatternDetector(detection_window=1800)  # 30 minutes
        
        current_time = time.time()
        
        # Add normal rate of errors
        for i in range(5):
            error_info = ErrorInfo(
                error_id=f"normal_{i}",
                error_type="RuntimeError",
                error_category=ErrorCategory.SYSTEM,
                message=f"Normal error {i}",
                occurred_at=current_time - 1200 + i * 60  # Spread over 20 minutes
            )
            detector.add_error(error_info)
        
        # Add burst of errors in recent time
        for i in range(15):
            error_info = ErrorInfo(
                error_id=f"burst_{i}",
                error_type="ConnectionError",
                error_category=ErrorCategory.NETWORK,
                message=f"Connection failed {i}",
                occurred_at=current_time - i * 10  # 15 errors in last 2.5 minutes
            )
            detector.add_error(error_info)
        
        patterns = detector.detect_patterns()
        
        # Should detect error burst
        burst_patterns = [p for p in patterns if p.pattern_type == ErrorPattern.ERROR_BURST]
        assert len(burst_patterns) > 0


@pytest.mark.skipif(not HAS_ERROR_ANALYSIS, reason="quantrs2.error_analysis not available")
class TestIncidentManager:
    """Test incident management."""
    
    def test_incident_creation(self):
        """Test creating incidents."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            incident_manager = IncidentManager(tmp.name)
            
            # Create test error pattern
            pattern = ErrorPatternMatch(
                pattern_type=ErrorPattern.FREQUENT_ERRORS,
                confidence=0.8,
                errors=["error1", "error2", "error3"],
                time_window=(time.time() - 300, time.time()),
                description="Frequent validation errors",
                severity=IncidentSeverity.HIGH
            )
            
            # Create incident
            incident = incident_manager.create_incident(
                title="High Error Rate",
                description="Multiple validation errors detected",
                severity=IncidentSeverity.HIGH,
                error_patterns=[pattern],
                assignee="test_user"
            )
            
            assert incident.incident_id is not None
            assert incident.title == "High Error Rate"
            assert incident.severity == IncidentSeverity.HIGH
            assert incident.status == IncidentStatus.OPEN
            assert incident.assignee == "test_user"
            assert len(incident.error_patterns) == 1
    
    def test_incident_resolution(self):
        """Test incident resolution."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            incident_manager = IncidentManager(tmp.name)
            
            pattern = ErrorPatternMatch(
                pattern_type=ErrorPattern.TIMEOUT_CLUSTER,
                confidence=0.7,
                errors=["timeout1", "timeout2"],
                time_window=(time.time() - 600, time.time()),
                description="Multiple timeouts",
                severity=IncidentSeverity.MEDIUM
            )
            
            incident = incident_manager.create_incident(
                title="Timeout Issues",
                description="Multiple timeout errors",
                severity=IncidentSeverity.MEDIUM,
                error_patterns=[pattern]
            )
            
            # Resolve incident
            success = incident_manager.resolve_incident(
                incident.incident_id,
                resolution="Increased timeout values",
                root_cause="Network latency"
            )
            
            assert success
            assert incident.status == IncidentStatus.RESOLVED
            assert incident.resolution == "Increased timeout values"
            assert incident.root_cause == "Network latency"
            assert incident.resolved_at is not None


@pytest.mark.skipif(not HAS_LOG_AGGREGATION, reason="quantrs2.log_aggregation not available")
class TestLogAggregation:
    """Test log aggregation and forwarding."""
    
    def test_log_forwarder_creation(self):
        """Test creating log forwarders."""
        config = LogDestinationConfig(
            destination_type=LogDestination.FILE,
            name="test_forwarder",
            endpoint="/tmp/test.log",
            log_format=LogFormat.JSON,
            enabled=False  # Disable to prevent actual file operations
        )
        
        forwarder = LogForwarder(config)
        
        assert forwarder.config.name == "test_forwarder"
        assert forwarder.config.destination_type == LogDestination.FILE
        assert forwarder.formatter.log_format == LogFormat.JSON
    
    def test_log_filtering(self):
        """Test log filtering in forwarder."""
        config = LogDestinationConfig(
            destination_type=LogDestination.FILE,
            name="test_filter",
            endpoint="/tmp/test.log",
            level_filter=["ERROR", "CRITICAL"],
            event_type_filter=["application"],
            enabled=False
        )
        
        forwarder = LogForwarder(config)
        
        # Test record that should be forwarded
        error_record = LogRecord(
            timestamp=time.time(),
            level="ERROR",
            message="Error message",
            event_type=EventType.APPLICATION,
            logger_name="test"
        )
        
        assert forwarder._should_forward(error_record)
        
        # Test record that should be filtered out
        info_record = LogRecord(
            timestamp=time.time(),
            level="INFO",
            message="Info message",
            event_type=EventType.APPLICATION,
            logger_name="test"
        )
        
        assert not forwarder._should_forward(info_record)
    
    def test_log_analyzer(self):
        """Test log analysis."""
        analyzer = LogAnalyzer(analysis_window=3600)
        
        # Add test logs
        for i in range(100):
            record = LogRecord(
                timestamp=time.time() - i,
                level="INFO" if i % 10 != 0 else "ERROR",
                message=f"Test message {i}",
                event_type=EventType.APPLICATION,
                logger_name=f"logger_{i % 5}"
            )
            analyzer.add_log(record)
        
        analysis = analyzer.analyze_patterns()
        
        assert analysis['total_logs'] == 100
        assert analysis['error_rate'] == 10.0  # 10% error rate
        assert len(analysis['by_logger']) == 5  # 5 different loggers
        assert 'INFO' in analysis['by_level']
        assert 'ERROR' in analysis['by_level']


@pytest.mark.skipif(not HAS_STRUCTURED_LOGGING, reason="quantrs2.structured_logging not available")
class TestDecorators:
    """Test logging decorators."""
    
    def test_function_call_logging(self):
        """Test function call logging decorator."""
        
        @log_function_calls(logger_name="test_decorator", include_args=True)
        def test_function(x, y, z="default"):
            return x + y
        
        # Mock the logging system to capture logs
        with patch('quantrs2.structured_logging.get_global_logging_system') as mock_logging:
            mock_logger = Mock()
            mock_logging.return_value.get_logger.return_value = mock_logger
            mock_logger.trace_manager.trace_span.return_value.__enter__ = Mock()
            mock_logger.trace_manager.trace_span.return_value.__exit__ = Mock(return_value=None)
            
            result = test_function(1, 2, z="test")
            
            assert result == 3
            assert mock_logger.trace.call_count == 2  # Entry and exit
    
    def test_quantum_operation_logging(self):
        """Test quantum operation logging decorator."""
        
        @log_quantum_operation(circuit_type="test_circuit")
        def quantum_test_function():
            time.sleep(0.01)
            return "quantum_result"
        
        # Mock the logging system
        with patch('quantrs2.structured_logging.get_global_logging_system') as mock_logging:
            mock_logger = Mock()
            mock_performance_logger = Mock()
            mock_logging.return_value.get_logger.return_value = mock_logger
            mock_logging.return_value.get_performance_logger.return_value = mock_performance_logger
            
            # Mock the context manager
            mock_performance_logger.measure.return_value.__enter__ = Mock()
            mock_performance_logger.measure.return_value.__exit__ = Mock(return_value=None)
            
            result = quantum_test_function()
            
            assert result == "quantum_result"
            mock_performance_logger.measure.assert_called_once()


@pytest.mark.skipif(not (HAS_STRUCTURED_LOGGING and HAS_ERROR_ANALYSIS), reason="quantrs2.structured_logging or quantrs2.error_analysis not available")
class TestIntegrationScenarios:
    """Test integrated logging scenarios."""
    
    def test_end_to_end_logging_flow(self):
        """Test complete logging workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure logging system
            config = {
                'json_log_file': str(Path(tmpdir) / 'test.log'),
                'max_tracked_errors': 1000
            }
            
            logging_system = LoggingSystem(config)
            
            try:
                # Get logger
                logger = logging_system.get_logger("integration_test")
                
                # Log various types of messages
                logger.info("Application started", structured_data={"version": "1.0.0"})
                logger.quantum("Circuit executed", structured_data={"qubits": 5, "gates": 20})
                
                # Log error
                test_error = ValueError("Test validation error")
                logger.error("Validation failed", error=test_error, 
                           error_category=ErrorCategory.VALIDATION)
                
                # Performance logging
                perf_logger = logging_system.get_performance_logger("integration_test")
                with perf_logger.measure("test_operation"):
                    time.sleep(0.01)
                
                # Check error statistics
                error_stats = logging_system.get_error_statistics()
                assert error_stats['total_errors'] == 1
                assert error_stats['unresolved_errors'] == 1
                
                # Verify log file exists and has content
                log_file = Path(tmpdir) / 'test.log'
                assert log_file.exists()
                
                with open(log_file, 'r') as f:
                    content = f.read()
                    assert len(content) > 0
                    # Should have multiple JSON log entries
                    assert content.count('\n') >= 3
                
            finally:
                logging_system.close()
    
    def test_error_analysis_integration(self):
        """Test integration between logging and error analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup error analysis system
            error_analysis = ErrorAnalysisSystem({
                'incidents_db_path': str(Path(tmpdir) / 'incidents.db'),
                'auto_incident_creation': True,
                'enable_background_analysis': False  # Disable for testing
            })
            
            try:
                # Simulate multiple related errors
                for i in range(10):
                    error = ConnectionError(f"Connection timeout {i}")
                    error_info = ErrorInfo(
                        error_id=f"conn_error_{i}",
                        error_type="ConnectionError",
                        error_category=ErrorCategory.NETWORK,
                        message=f"Connection timeout {i}",
                        occurred_at=time.time() - i
                    )
                    
                    # Analyze error
                    analysis = error_analysis.analyze_error(error_info)
                    assert analysis['error_id'] == f"conn_error_{i}"
                    assert analysis['category'] == 'network'
                
                # Get analysis report
                report = error_analysis.get_analysis_report()
                assert 'pattern_detection' in report
                assert 'incident_management' in report
                
                # Should have detected some patterns
                assert report['pattern_detection']['total_errors_tracked'] == 10
                
            finally:
                error_analysis.close()


if __name__ == "__main__":
    pytest.main([__file__])