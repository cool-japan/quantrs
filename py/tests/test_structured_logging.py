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

from quantrs2.structured_logging import (
    LoggingSystem, StructuredLogger, TraceManager, ErrorTracker,
    LogLevel, EventType, ErrorCategory, TraceContext, ErrorInfo,
    LogRecord, JSONLogHandler, ConsoleLogHandler, PerformanceLogger,
    log_function_calls, log_quantum_operation, get_logger
)
from quantrs2.error_analysis import (
    ErrorAnalysisSystem, ErrorPatternDetector, IncidentManager,
    ErrorPattern, IncidentSeverity, IncidentStatus
)
from quantrs2.log_aggregation import (
    LogAggregationSystem, LogDestinationConfig, LogDestination,
    LogFormat, LogForwarder, LogAnalyzer
)


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