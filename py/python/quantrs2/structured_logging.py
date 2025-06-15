"""
Structured Logging and Error Tracking System for QuantRS2

This module provides comprehensive structured logging, error tracking, and 
observability features for production QuantRS2 deployments including:
- JSON-formatted structured logging
- Distributed tracing and correlation
- Error classification and reporting
- Performance logging and profiling
- Security event tracking
- Integration with external log management systems
"""

import os
import sys
import time
import json
import uuid
import logging
import traceback
import threading
import contextlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, ContextManager
from dataclasses import dataclass, field, asdict
from enum import Enum
import inspect
import functools
import weakref

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Extended log levels for quantum computing contexts."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60
    QUANTUM = 25  # Special level for quantum-specific events


class ErrorCategory(Enum):
    """Categories for error classification."""
    SYSTEM = "system"
    QUANTUM = "quantum"
    NETWORK = "network"
    SECURITY = "security"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    PERFORMANCE = "performance"
    USER = "user"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class EventType(Enum):
    """Types of events for structured logging."""
    APPLICATION = "application"
    QUANTUM_EXECUTION = "quantum_execution"
    CIRCUIT_COMPILATION = "circuit_compilation"
    CACHE_OPERATION = "cache_operation"
    DATABASE_OPERATION = "database_operation"
    SECURITY_EVENT = "security_event"
    PERFORMANCE_EVENT = "performance_event"
    ERROR_EVENT = "error_event"
    AUDIT_EVENT = "audit_event"
    SYSTEM_EVENT = "system_event"


@dataclass
class TraceContext:
    """Distributed tracing context."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: float = field(default_factory=time.time)
    tags: Dict[str, Any] = field(default_factory=dict)
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace context to dictionary."""
        return {
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'operation_name': self.operation_name,
            'start_time': self.start_time,
            'tags': self.tags,
            'baggage': self.baggage
        }


@dataclass
class ErrorInfo:
    """Structured error information."""
    error_id: str
    error_type: str
    error_category: ErrorCategory
    message: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    occurred_at: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error info to dictionary."""
        return {
            'error_id': self.error_id,
            'error_type': self.error_type,
            'error_category': self.error_category.value,
            'message': self.message,
            'stack_trace': self.stack_trace,
            'context': self.context,
            'occurred_at': self.occurred_at,
            'resolved': self.resolved,
            'resolution_notes': self.resolution_notes
        }


@dataclass
class LogRecord:
    """Structured log record."""
    timestamp: float
    level: str
    message: str
    event_type: EventType
    logger_name: str
    
    # Context information
    trace_context: Optional[TraceContext] = None
    error_info: Optional[ErrorInfo] = None
    
    # Structured data
    structured_data: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Source information
    filename: Optional[str] = None
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    
    # Performance data
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log record to dictionary for JSON serialization."""
        record = {
            'timestamp': datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            'level': self.level,
            'message': self.message,
            'event_type': self.event_type.value,
            'logger_name': self.logger_name,
            'structured_data': self.structured_data,
            'tags': self.tags,
            'filename': self.filename,
            'line_number': self.line_number,
            'function_name': self.function_name,
            'duration_ms': self.duration_ms
        }
        
        if self.trace_context:
            record['trace_context'] = self.trace_context.to_dict()
        
        if self.error_info:
            record['error_info'] = self.error_info.to_dict()
        
        return record


class TraceManager:
    """Manages distributed tracing contexts."""
    
    def __init__(self):
        self._local = threading.local()
        self._lock = threading.RLock()
    
    def _get_current_context(self) -> Optional[TraceContext]:
        """Get current trace context for this thread."""
        return getattr(self._local, 'trace_context', None)
    
    def _set_current_context(self, context: Optional[TraceContext]):
        """Set current trace context for this thread."""
        self._local.trace_context = context
    
    def start_trace(self, operation_name: str, tags: Optional[Dict[str, Any]] = None) -> TraceContext:
        """Start a new trace."""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            operation_name=operation_name,
            tags=tags or {}
        )
        
        self._set_current_context(context)
        return context
    
    def start_span(self, operation_name: str, tags: Optional[Dict[str, Any]] = None) -> TraceContext:
        """Start a new span within current trace."""
        current = self._get_current_context()
        
        if current:
            trace_id = current.trace_id
            parent_span_id = current.span_id
        else:
            trace_id = str(uuid.uuid4())
            parent_span_id = None
        
        span_id = str(uuid.uuid4())
        
        context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            tags=tags or {}
        )
        
        self._set_current_context(context)
        return context
    
    def finish_span(self, context: TraceContext, tags: Optional[Dict[str, Any]] = None):
        """Finish a span."""
        if tags:
            context.tags.update(tags)
        
        # Calculate duration
        duration = time.time() - context.start_time
        context.tags['duration_ms'] = duration * 1000
        
        # Restore parent context if available
        if context.parent_span_id:
            # In a real implementation, we'd maintain a stack of contexts
            pass
        else:
            self._set_current_context(None)
    
    def get_current_context(self) -> Optional[TraceContext]:
        """Get current trace context."""
        return self._get_current_context()
    
    @contextlib.contextmanager
    def trace_span(self, operation_name: str, tags: Optional[Dict[str, Any]] = None):
        """Context manager for tracing spans."""
        context = self.start_span(operation_name, tags)
        try:
            yield context
        finally:
            self.finish_span(context)


class ErrorTracker:
    """Tracks and manages errors across the system."""
    
    def __init__(self, max_errors: int = 10000):
        self.max_errors = max_errors
        self._errors: Dict[str, ErrorInfo] = {}
        self._error_stats: Dict[ErrorCategory, int] = {}
        self._lock = threading.RLock()
    
    def track_error(self, error: Exception, context: Optional[Dict[str, Any]] = None,
                   category: ErrorCategory = ErrorCategory.UNKNOWN) -> ErrorInfo:
        """Track an error occurrence."""
        error_id = str(uuid.uuid4())
        
        # Extract stack trace
        stack_trace = None
        if hasattr(error, '__traceback__') and error.__traceback__:
            stack_trace = ''.join(traceback.format_exception(
                type(error), error, error.__traceback__
            ))
        
        error_info = ErrorInfo(
            error_id=error_id,
            error_type=type(error).__name__,
            error_category=category,
            message=str(error),
            stack_trace=stack_trace,
            context=context or {}
        )
        
        with self._lock:
            # Store error
            self._errors[error_id] = error_info
            
            # Update statistics
            self._error_stats[category] = self._error_stats.get(category, 0) + 1
            
            # Cleanup old errors if necessary
            if len(self._errors) > self.max_errors:
                # Remove oldest errors
                sorted_errors = sorted(
                    self._errors.items(),
                    key=lambda x: x[1].occurred_at
                )
                
                for old_error_id, _ in sorted_errors[:len(self._errors) - self.max_errors]:
                    del self._errors[old_error_id]
        
        return error_info
    
    def resolve_error(self, error_id: str, resolution_notes: str = ""):
        """Mark an error as resolved."""
        with self._lock:
            if error_id in self._errors:
                self._errors[error_id].resolved = True
                self._errors[error_id].resolution_notes = resolution_notes
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            total_errors = len(self._errors)
            resolved_errors = sum(1 for e in self._errors.values() if e.resolved)
            
            return {
                'total_errors': total_errors,
                'resolved_errors': resolved_errors,
                'unresolved_errors': total_errors - resolved_errors,
                'by_category': dict(self._error_stats),
                'recent_errors': [
                    e.to_dict() for e in sorted(
                        self._errors.values(),
                        key=lambda x: x.occurred_at,
                        reverse=True
                    )[:10]
                ]
            }
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorInfo]:
        """Get errors by category."""
        with self._lock:
            return [e for e in self._errors.values() if e.error_category == category]


class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(self, name: str, trace_manager: TraceManager, error_tracker: ErrorTracker):
        self.name = name
        self.trace_manager = trace_manager
        self.error_tracker = error_tracker
        self._handlers: List[Callable[[LogRecord], None]] = []
        self._filters: List[Callable[[LogRecord], bool]] = []
    
    def add_handler(self, handler: Callable[[LogRecord], None]):
        """Add a log handler."""
        self._handlers.append(handler)
    
    def add_filter(self, filter_func: Callable[[LogRecord], bool]):
        """Add a log filter."""
        self._filters.append(filter_func)
    
    def _should_log(self, record: LogRecord) -> bool:
        """Check if record should be logged based on filters."""
        return all(f(record) for f in self._filters)
    
    def _emit_record(self, record: LogRecord):
        """Emit log record to handlers."""
        if self._should_log(record):
            for handler in self._handlers:
                try:
                    handler(record)
                except Exception as e:
                    # Fallback logging to avoid infinite loops
                    print(f"Log handler error: {e}", file=sys.stderr)
    
    def log(self, level: Union[LogLevel, int], message: str, 
           event_type: EventType = EventType.APPLICATION,
           structured_data: Optional[Dict[str, Any]] = None,
           tags: Optional[Dict[str, str]] = None,
           error: Optional[Exception] = None,
           error_category: ErrorCategory = ErrorCategory.UNKNOWN):
        """Log a structured message."""
        
        # Get caller information
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename
        line_number = frame.f_lineno
        function_name = frame.f_code.co_name
        
        # Handle error tracking
        error_info = None
        if error:
            error_info = self.error_tracker.track_error(error, structured_data, error_category)
        
        # Create log record
        record = LogRecord(
            timestamp=time.time(),
            level=level.name if isinstance(level, LogLevel) else logging.getLevelName(level),
            message=message,
            event_type=event_type,
            logger_name=self.name,
            trace_context=self.trace_manager.get_current_context(),
            error_info=error_info,
            structured_data=structured_data or {},
            tags=tags or {},
            filename=Path(filename).name,
            line_number=line_number,
            function_name=function_name
        )
        
        self._emit_record(record)
    
    def trace(self, message: str, **kwargs):
        """Log trace level message."""
        self.log(LogLevel.TRACE, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info level message."""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error level message."""
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical level message."""
        self.log(LogLevel.CRITICAL, message, **kwargs)
    
    def security(self, message: str, **kwargs):
        """Log security event."""
        kwargs['event_type'] = EventType.SECURITY_EVENT
        self.log(LogLevel.SECURITY, message, **kwargs)
    
    def quantum(self, message: str, **kwargs):
        """Log quantum-specific event."""
        kwargs['event_type'] = EventType.QUANTUM_EXECUTION
        self.log(LogLevel.QUANTUM, message, **kwargs)
    
    def performance(self, message: str, duration_ms: float, **kwargs):
        """Log performance event."""
        kwargs['event_type'] = EventType.PERFORMANCE_EVENT
        kwargs['structured_data'] = kwargs.get('structured_data', {})
        kwargs['structured_data']['duration_ms'] = duration_ms
        self.log(LogLevel.INFO, message, **kwargs)
    
    def audit(self, message: str, user_id: Optional[str] = None, 
             action: Optional[str] = None, resource: Optional[str] = None, **kwargs):
        """Log audit event."""
        kwargs['event_type'] = EventType.AUDIT_EVENT
        kwargs['structured_data'] = kwargs.get('structured_data', {})
        
        if user_id:
            kwargs['structured_data']['user_id'] = user_id
        if action:
            kwargs['structured_data']['action'] = action
        if resource:
            kwargs['structured_data']['resource'] = resource
        
        self.log(LogLevel.INFO, message, **kwargs)


class JSONLogHandler:
    """Handler that outputs JSON-formatted logs."""
    
    def __init__(self, output_file: Optional[str] = None):
        self.output_file = output_file
        self._file_handle = None
        self._lock = threading.RLock()
        
        if output_file:
            self._ensure_output_file()
    
    def _ensure_output_file(self):
        """Ensure output file exists and is writable."""
        if self.output_file:
            Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self.output_file, 'a', encoding='utf-8')
    
    def __call__(self, record: LogRecord):
        """Handle log record."""
        with self._lock:
            json_data = json.dumps(record.to_dict(), separators=(',', ':'))
            
            if self._file_handle:
                self._file_handle.write(json_data + '\n')
                self._file_handle.flush()
            else:
                print(json_data)
    
    def close(self):
        """Close file handle."""
        with self._lock:
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None


class ConsoleLogHandler:
    """Handler that outputs human-readable logs to console."""
    
    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        
        # ANSI color codes
        self.colors = {
            'TRACE': '\033[90m',     # Dark gray
            'DEBUG': '\033[36m',     # Cyan
            'INFO': '\033[32m',      # Green
            'QUANTUM': '\033[35m',   # Magenta
            'WARNING': '\033[33m',   # Yellow
            'ERROR': '\033[31m',     # Red
            'CRITICAL': '\033[91m',  # Bright red
            'SECURITY': '\033[95m',  # Bright magenta
            'RESET': '\033[0m'       # Reset
        }
    
    def _colorize(self, text: str, level: str) -> str:
        """Add color to text based on log level."""
        if not self.use_colors:
            return text
        
        color = self.colors.get(level, '')
        reset = self.colors['RESET']
        return f"{color}{text}{reset}"
    
    def __call__(self, record: LogRecord):
        """Handle log record."""
        timestamp = datetime.fromtimestamp(record.timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        # Build log line
        parts = [
            timestamp,
            self._colorize(f"[{record.level}]", record.level),
            f"{record.logger_name}:{record.function_name}:{record.line_number}",
            record.message
        ]
        
        log_line = " ".join(parts)
        
        # Add structured data if present
        if record.structured_data:
            structured_str = json.dumps(record.structured_data, separators=(',', ':'))
            log_line += f" | data={structured_str}"
        
        # Add trace context if present
        if record.trace_context:
            log_line += f" | trace_id={record.trace_context.trace_id[:8]}"
        
        # Add error info if present
        if record.error_info:
            log_line += f" | error_id={record.error_info.error_id[:8]}"
        
        print(log_line)


class PerformanceLogger:
    """Specialized logger for performance monitoring."""
    
    def __init__(self, structured_logger: StructuredLogger):
        self.logger = structured_logger
    
    @contextlib.contextmanager
    def measure(self, operation: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for measuring operation performance."""
        start_time = time.time()
        
        with self.logger.trace_manager.trace_span(operation, tags) as span:
            try:
                yield span
                
                # Log successful completion
                duration_ms = (time.time() - start_time) * 1000
                self.logger.performance(
                    f"Operation completed: {operation}",
                    duration_ms=duration_ms,
                    tags=tags
                )
                
            except Exception as e:
                # Log error with performance data
                duration_ms = (time.time() - start_time) * 1000
                self.logger.error(
                    f"Operation failed: {operation}",
                    error=e,
                    structured_data={'duration_ms': duration_ms},
                    tags=tags
                )
                raise
    
    def log_quantum_execution(self, circuit_info: Dict[str, Any], 
                            execution_time: float, success: bool,
                            backend: str = "unknown"):
        """Log quantum circuit execution performance."""
        self.logger.quantum(
            f"Circuit execution {'completed' if success else 'failed'}",
            structured_data={
                'circuit_info': circuit_info,
                'execution_time_ms': execution_time * 1000,
                'success': success,
                'backend': backend
            },
            tags={
                'component': 'quantum_executor',
                'backend': backend,
                'status': 'success' if success else 'failure'
            }
        )


class LoggingSystem:
    """Main logging system that coordinates all components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.trace_manager = TraceManager()
        self.error_tracker = ErrorTracker(
            max_errors=self.config.get('max_tracked_errors', 10000)
        )
        
        # Logger registry
        self._loggers: Dict[str, StructuredLogger] = {}
        self._lock = threading.RLock()
        
        # Global handlers
        self._setup_default_handlers()
        
        logger.info("Structured logging system initialized")
    
    def _setup_default_handlers(self):
        """Setup default log handlers."""
        # Console handler
        console_handler = ConsoleLogHandler(
            use_colors=self.config.get('console_colors', True)
        )
        
        # JSON file handler if configured
        json_file = self.config.get('json_log_file')
        json_handler = None
        if json_file:
            json_handler = JSONLogHandler(json_file)
        
        # Store handlers for later use
        self._default_handlers = [console_handler]
        if json_handler:
            self._default_handlers.append(json_handler)
    
    def get_logger(self, name: str) -> StructuredLogger:
        """Get or create a structured logger."""
        with self._lock:
            if name not in self._loggers:
                logger_instance = StructuredLogger(name, self.trace_manager, self.error_tracker)
                
                # Add default handlers
                for handler in self._default_handlers:
                    logger_instance.add_handler(handler)
                
                self._loggers[name] = logger_instance
            
            return self._loggers[name]
    
    def get_performance_logger(self, name: str) -> PerformanceLogger:
        """Get performance logger for a component."""
        structured_logger = self.get_logger(name)
        return PerformanceLogger(structured_logger)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get system-wide error statistics."""
        return self.error_tracker.get_error_statistics()
    
    def register_error_handler(self, handler: Callable[[ErrorInfo], None]):
        """Register a handler for error events."""
        # This would integrate with the monitoring system
        pass
    
    def close(self):
        """Close logging system and cleanup resources."""
        with self._lock:
            for handler in self._default_handlers:
                if hasattr(handler, 'close'):
                    handler.close()


# Decorators for automatic logging
def log_function_calls(logger_name: Optional[str] = None, 
                      include_args: bool = False,
                      include_result: bool = False):
    """Decorator to automatically log function calls."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            name = logger_name or func.__module__
            logger_instance = get_global_logging_system().get_logger(name)
            
            # Prepare structured data
            structured_data = {
                'function': func.__name__,
                'module': func.__module__
            }
            
            if include_args:
                structured_data['args'] = str(args)[:1000]  # Limit size
                structured_data['kwargs'] = str(kwargs)[:1000]
            
            # Log function entry
            with logger_instance.trace_manager.trace_span(f"function:{func.__name__}"):
                logger_instance.trace(
                    f"Function called: {func.__name__}",
                    structured_data=structured_data,
                    tags={'component': 'function_tracer'}
                )
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Log successful completion
                    if include_result:
                        structured_data['result'] = str(result)[:1000]
                    
                    logger_instance.trace(
                        f"Function completed: {func.__name__}",
                        structured_data=structured_data,
                        tags={'component': 'function_tracer'}
                    )
                    
                    return result
                    
                except Exception as e:
                    # Log error
                    logger_instance.error(
                        f"Function failed: {func.__name__}",
                        error=e,
                        structured_data=structured_data,
                        tags={'component': 'function_tracer'}
                    )
                    raise
        
        return wrapper
    return decorator


def log_quantum_operation(circuit_type: Optional[str] = None):
    """Decorator for logging quantum operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger_instance = get_global_logging_system().get_logger(func.__module__)
            performance_logger = PerformanceLogger(logger_instance)
            
            operation_name = circuit_type or func.__name__
            
            with performance_logger.measure(f"quantum_operation:{operation_name}",
                                          tags={'operation': operation_name}):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global logging system instance
_global_logging_system: Optional[LoggingSystem] = None
_global_lock = threading.RLock()


def get_global_logging_system() -> LoggingSystem:
    """Get global logging system instance."""
    global _global_logging_system
    
    with _global_lock:
        if _global_logging_system is None:
            _global_logging_system = LoggingSystem()
        
        return _global_logging_system


def configure_logging(config: Dict[str, Any]):
    """Configure global logging system."""
    global _global_logging_system
    
    with _global_lock:
        if _global_logging_system:
            _global_logging_system.close()
        
        _global_logging_system = LoggingSystem(config)


def get_logger(name: str) -> StructuredLogger:
    """Get structured logger instance."""
    return get_global_logging_system().get_logger(name)


def get_performance_logger(name: str) -> PerformanceLogger:
    """Get performance logger instance."""
    return get_global_logging_system().get_performance_logger(name)


# Export main classes
__all__ = [
    'LogLevel',
    'ErrorCategory', 
    'EventType',
    'TraceContext',
    'ErrorInfo',
    'LogRecord',
    'TraceManager',
    'ErrorTracker',
    'StructuredLogger',
    'JSONLogHandler',
    'ConsoleLogHandler',
    'PerformanceLogger',
    'LoggingSystem',
    'log_function_calls',
    'log_quantum_operation',
    'get_global_logging_system',
    'configure_logging',
    'get_logger',
    'get_performance_logger'
]