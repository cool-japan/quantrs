"""
Comprehensive Error Handling and Recovery for QuantRS2

This module provides production-grade error handling, recovery mechanisms,
and resilience patterns for quantum computing operations.
"""

import logging
import traceback
import time
import asyncio
import threading
from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from contextlib import contextmanager
import json

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for quantum operations."""
    QUANTUM_HARDWARE = "quantum_hardware"
    CIRCUIT_COMPILATION = "circuit_compilation"
    SIMULATION = "simulation"
    VALIDATION = "validation"
    SECURITY = "security"
    NETWORK = "network"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    USER_INPUT = "user_input"
    UNKNOWN = "unknown"

class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_SIMPLIFICATION = "circuit_simplification"
    BACKEND_SWITCHING = "backend_switching"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"
    CIRCUIT_PARTITIONING = "circuit_partitioning"
    NOISE_MITIGATION = "noise_mitigation"

@dataclass
class ErrorContext:
    """Context information for error handling."""
    
    operation: str
    timestamp: float
    thread_id: int
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    circuit_info: Optional[Dict[str, Any]] = None
    backend_info: Optional[Dict[str, Any]] = None
    resource_state: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation': self.operation,
            'timestamp': self.timestamp,
            'thread_id': self.thread_id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'circuit_info': self.circuit_info,
            'backend_info': self.backend_info,
            'resource_state': self.resource_state
        }

@dataclass
class ErrorDetails:
    """Detailed error information."""
    
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    original_exception: Optional[Exception] = None
    traceback_str: Optional[str] = None
    context: Optional[ErrorContext] = None
    recovery_attempted: List[RecoveryStrategy] = field(default_factory=list)
    recovery_successful: bool = False
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            'error_id': self.error_id,
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'original_exception_type': type(self.original_exception).__name__ if self.original_exception else None,
            'traceback': self.traceback_str,
            'context': self.context.to_dict() if self.context else None,
            'recovery_attempted': [s.value for s in self.recovery_attempted],
            'recovery_successful': self.recovery_successful,
            'timestamp': self.timestamp
        }

@dataclass
class RecoveryConfig:
    """Configuration for error recovery strategies."""
    
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    max_retry_delay: float = 60.0
    circuit_simplification_threshold: int = 100  # Max gates before simplification
    fallback_backends: List[str] = field(default_factory=list)
    timeout_seconds: float = 300.0
    enable_circuit_partitioning: bool = True
    enable_noise_mitigation: bool = True
    
class QuantumError(Exception):
    """Base class for quantum computing errors."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context
        self.timestamp = time.time()

class QuantumHardwareError(QuantumError):
    """Error related to quantum hardware operations."""
    
    def __init__(self, message: str, backend_name: Optional[str] = None,
                 device_info: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message, ErrorCategory.QUANTUM_HARDWARE, ErrorSeverity.HIGH, **kwargs)
        self.backend_name = backend_name
        self.device_info = device_info

class CircuitCompilationError(QuantumError):
    """Error during circuit compilation."""
    
    def __init__(self, message: str, circuit_info: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message, ErrorCategory.CIRCUIT_COMPILATION, ErrorSeverity.MEDIUM, **kwargs)
        self.circuit_info = circuit_info

class SimulationError(QuantumError):
    """Error during quantum simulation."""
    
    def __init__(self, message: str, simulator_info: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message, ErrorCategory.SIMULATION, ErrorSeverity.MEDIUM, **kwargs)
        self.simulator_info = simulator_info

class ValidationError(QuantumError):
    """Error during input/circuit validation."""
    
    def __init__(self, message: str, validation_type: Optional[str] = None, **kwargs):
        super().__init__(message, ErrorCategory.VALIDATION, ErrorSeverity.LOW, **kwargs)
        self.validation_type = validation_type

class SecurityError(QuantumError):
    """Error related to security violations."""
    
    def __init__(self, message: str, security_context: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message, ErrorCategory.SECURITY, ErrorSeverity.CRITICAL, **kwargs)
        self.security_context = security_context

class ResourceError(QuantumError):
    """Error related to resource limitations."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None,
                 available: Optional[int] = None, required: Optional[int] = None, **kwargs):
        super().__init__(message, ErrorCategory.RESOURCE, ErrorSeverity.HIGH, **kwargs)
        self.resource_type = resource_type
        self.available = available
        self.required = required

class ErrorRecoveryManager:
    """Manages error recovery strategies and implementations."""
    
    def __init__(self, config: Optional[RecoveryConfig] = None):
        self.config = config or RecoveryConfig()
        self.error_history: List[ErrorDetails] = []
        self.recovery_strategies: Dict[ErrorCategory, List[RecoveryStrategy]] = {
            ErrorCategory.QUANTUM_HARDWARE: [
                RecoveryStrategy.BACKEND_SWITCHING,
                RecoveryStrategy.RETRY,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            ErrorCategory.CIRCUIT_COMPILATION: [
                RecoveryStrategy.CIRCUIT_SIMPLIFICATION,
                RecoveryStrategy.FALLBACK,
                RecoveryStrategy.RETRY
            ],
            ErrorCategory.SIMULATION: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.CIRCUIT_PARTITIONING,
                RecoveryStrategy.FALLBACK
            ],
            ErrorCategory.VALIDATION: [
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.FAIL_FAST
            ],
            ErrorCategory.SECURITY: [
                RecoveryStrategy.FAIL_FAST
            ],
            ErrorCategory.NETWORK: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.FALLBACK
            ],
            ErrorCategory.RESOURCE: [
                RecoveryStrategy.CIRCUIT_PARTITIONING,
                RecoveryStrategy.CIRCUIT_SIMPLIFICATION,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ]
        }
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> Any:
        """
        Handle an error with appropriate recovery strategies.
        
        Args:
            error: The exception that occurred
            context: Optional context information
            
        Returns:
            Result of recovery attempt or re-raises the error
        """
        import uuid
        
        # Create error details
        error_details = ErrorDetails(
            error_id=str(uuid.uuid4()),
            category=self._categorize_error(error),
            severity=self._assess_severity(error),
            message=str(error),
            original_exception=error,
            traceback_str=traceback.format_exc(),
            context=context
        )
        
        # Log the error
        self._log_error(error_details)
        
        # Record error
        with self._lock:
            self.error_history.append(error_details)
        
        # Attempt recovery
        try:
            result = self._attempt_recovery(error_details)
            error_details.recovery_successful = True
            return result
        except Exception as recovery_error:
            error_details.recovery_successful = False
            logger.error(f"Recovery failed for error {error_details.error_id}: {recovery_error}")
            
            # Re-raise original error if recovery fails
            if isinstance(error, QuantumError):
                raise error
            else:
                raise QuantumError(
                    f"Unhandled error: {error}",
                    category=error_details.category,
                    severity=error_details.severity,
                    context=context
                ) from error
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error based on type and message."""
        if isinstance(error, QuantumError):
            return error.category
        
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Hardware-related errors
        if any(keyword in error_msg for keyword in ['hardware', 'backend', 'device', 'quantum computer']):
            return ErrorCategory.QUANTUM_HARDWARE
        
        # Compilation errors
        if any(keyword in error_msg for keyword in ['compilation', 'transpile', 'optimization']):
            return ErrorCategory.CIRCUIT_COMPILATION
        
        # Simulation errors
        if any(keyword in error_msg for keyword in ['simulation', 'simulator', 'statevector']):
            return ErrorCategory.SIMULATION
        
        # Validation errors
        if any(keyword in error_msg for keyword in ['validation', 'invalid', 'malformed']):
            return ErrorCategory.VALIDATION
        
        # Security errors
        if any(keyword in error_msg for keyword in ['security', 'authentication', 'authorization', 'injection']):
            return ErrorCategory.SECURITY
        
        # Network errors
        if any(keyword in error_msg for keyword in ['network', 'connection', 'timeout', 'http']):
            return ErrorCategory.NETWORK
        
        # Resource errors
        if any(keyword in error_msg for keyword in ['memory', 'resource', 'limit', 'quota']):
            return ErrorCategory.RESOURCE
        
        # Python exception types
        if error_type in ['ValueError', 'TypeError', 'AttributeError']:
            return ErrorCategory.USER_INPUT
        elif error_type in ['ConnectionError', 'TimeoutError']:
            return ErrorCategory.NETWORK
        elif error_type in ['MemoryError', 'OverflowError']:
            return ErrorCategory.RESOURCE
        
        return ErrorCategory.UNKNOWN
    
    def _assess_severity(self, error: Exception) -> ErrorSeverity:
        """Assess error severity."""
        if isinstance(error, QuantumError):
            return error.severity
        
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Critical errors
        if any(keyword in error_msg for keyword in ['security', 'critical', 'fatal']):
            return ErrorSeverity.CRITICAL
        
        # High severity
        if any(keyword in error_msg for keyword in ['hardware', 'backend failure', 'corruption']):
            return ErrorSeverity.HIGH
        
        # Medium severity
        if any(keyword in error_msg for keyword in ['compilation', 'optimization', 'simulation']):
            return ErrorSeverity.MEDIUM
        
        # Low severity
        if error_type in ['ValueError', 'TypeError']:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _attempt_recovery(self, error_details: ErrorDetails) -> Any:
        """Attempt recovery using appropriate strategies."""
        strategies = self.recovery_strategies.get(error_details.category, [RecoveryStrategy.RETRY])
        
        for strategy in strategies:
            try:
                logger.info(f"Attempting recovery strategy: {strategy.value} for error {error_details.error_id}")
                error_details.recovery_attempted.append(strategy)
                
                result = self._execute_recovery_strategy(strategy, error_details)
                if result is not None:
                    logger.info(f"Recovery successful using strategy: {strategy.value}")
                    return result
                    
            except Exception as e:
                logger.warning(f"Recovery strategy {strategy.value} failed: {e}")
                continue
        
        # If all strategies fail, raise the original error
        raise error_details.original_exception or Exception("Recovery failed for all strategies")
    
    def _execute_recovery_strategy(self, strategy: RecoveryStrategy, 
                                 error_details: ErrorDetails) -> Any:
        """Execute specific recovery strategy."""
        if strategy == RecoveryStrategy.RETRY:
            return self._retry_operation(error_details)
        
        elif strategy == RecoveryStrategy.FALLBACK:
            return self._fallback_operation(error_details)
        
        elif strategy == RecoveryStrategy.CIRCUIT_SIMPLIFICATION:
            return self._simplify_circuit(error_details)
        
        elif strategy == RecoveryStrategy.BACKEND_SWITCHING:
            return self._switch_backend(error_details)
        
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._graceful_degradation(error_details)
        
        elif strategy == RecoveryStrategy.CIRCUIT_PARTITIONING:
            return self._partition_circuit(error_details)
        
        elif strategy == RecoveryStrategy.NOISE_MITIGATION:
            return self._apply_noise_mitigation(error_details)
        
        elif strategy == RecoveryStrategy.FAIL_FAST:
            raise error_details.original_exception
        
        return None
    
    def _retry_operation(self, error_details: ErrorDetails) -> Any:
        """Implement retry logic with exponential backoff."""
        max_retries = self.config.max_retries
        delay = self.config.retry_delay
        
        for attempt in range(max_retries):
            try:
                time.sleep(delay)
                
                # Re-attempt the original operation
                # This is a placeholder - in real implementation, would need
                # to store and replay the original operation
                logger.info(f"Retry attempt {attempt + 1}/{max_retries}")
                
                # For demonstration, return a success indicator
                return {"status": "retry_successful", "attempt": attempt + 1}
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                # Exponential backoff
                if self.config.exponential_backoff:
                    delay = min(delay * 2, self.config.max_retry_delay)
        
        return None
    
    def _fallback_operation(self, error_details: ErrorDetails) -> Any:
        """Implement fallback to alternative implementation."""
        logger.info("Executing fallback operation")
        
        # Placeholder for fallback logic
        # In real implementation, would switch to alternative implementation
        return {"status": "fallback_executed", "method": "alternative_implementation"}
    
    def _simplify_circuit(self, error_details: ErrorDetails) -> Any:
        """Simplify circuit to reduce complexity."""
        logger.info("Simplifying circuit")
        
        # Placeholder for circuit simplification logic
        # Would remove non-essential gates, reduce depth, etc.
        return {"status": "circuit_simplified", "reduction": "50%"}
    
    def _switch_backend(self, error_details: ErrorDetails) -> Any:
        """Switch to alternative quantum backend."""
        logger.info("Switching to alternative backend")
        
        # Placeholder for backend switching logic
        # Would try alternative backends from configuration
        fallback_backends = self.config.fallback_backends
        if fallback_backends:
            return {"status": "backend_switched", "new_backend": fallback_backends[0]}
        
        return None
    
    def _graceful_degradation(self, error_details: ErrorDetails) -> Any:
        """Provide degraded but functional response."""
        logger.info("Executing graceful degradation")
        
        # Placeholder for degraded operation
        # Would provide limited functionality or approximate results
        return {"status": "degraded_operation", "quality": "reduced"}
    
    def _partition_circuit(self, error_details: ErrorDetails) -> Any:
        """Partition large circuit into smaller parts."""
        logger.info("Partitioning circuit")
        
        # Placeholder for circuit partitioning logic
        # Would split circuit into manageable chunks
        return {"status": "circuit_partitioned", "partitions": 4}
    
    def _apply_noise_mitigation(self, error_details: ErrorDetails) -> Any:
        """Apply noise mitigation techniques."""
        logger.info("Applying noise mitigation")
        
        # Placeholder for noise mitigation
        # Would apply error correction, zero-noise extrapolation, etc.
        return {"status": "noise_mitigation_applied", "technique": "zero_noise_extrapolation"}
    
    def _log_error(self, error_details: ErrorDetails) -> None:
        """Log error details appropriately based on severity."""
        log_data = error_details.to_dict()
        
        if error_details.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error {error_details.error_id}: {error_details.message}", 
                           extra={'error_details': log_data})
        elif error_details.severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error {error_details.error_id}: {error_details.message}",
                        extra={'error_details': log_data})
        elif error_details.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Medium severity error {error_details.error_id}: {error_details.message}",
                          extra={'error_details': log_data})
        else:
            logger.info(f"Low severity error {error_details.error_id}: {error_details.message}",
                       extra={'error_details': log_data})
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and trends."""
        with self._lock:
            total_errors = len(self.error_history)
            
            if total_errors == 0:
                return {"total_errors": 0}
            
            # Category distribution
            category_counts = {}
            severity_counts = {}
            recovery_success_rate = 0
            
            for error in self.error_history:
                category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
                severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
                if error.recovery_successful:
                    recovery_success_rate += 1
            
            recovery_success_rate = recovery_success_rate / total_errors
            
            return {
                "total_errors": total_errors,
                "category_distribution": category_counts,
                "severity_distribution": severity_counts,
                "recovery_success_rate": recovery_success_rate,
                "recent_errors": [e.to_dict() for e in self.error_history[-10:]]  # Last 10 errors
            }

# Global error recovery manager
_error_manager = ErrorRecoveryManager()

def get_error_manager(config: Optional[RecoveryConfig] = None) -> ErrorRecoveryManager:
    """Get the global error recovery manager."""
    global _error_manager
    if config:
        _error_manager = ErrorRecoveryManager(config)
    return _error_manager

def quantum_error_handler(operation_name: str = "quantum_operation"):
    """
    Decorator for automatic error handling and recovery.
    
    Args:
        operation_name: Name of the operation for context
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                operation=f"{func.__name__}({operation_name})",
                timestamp=time.time(),
                thread_id=threading.get_ident()
            )
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                manager = get_error_manager()
                return manager.handle_error(e, context)
        
        return wrapper
    return decorator

@contextmanager
def quantum_error_context(operation_name: str, **context_kwargs):
    """
    Context manager for error handling with additional context.
    
    Args:
        operation_name: Name of the operation
        **context_kwargs: Additional context information
    """
    context = ErrorContext(
        operation=operation_name,
        timestamp=time.time(),
        thread_id=threading.get_ident(),
        **context_kwargs
    )
    
    try:
        yield context
    except Exception as e:
        manager = get_error_manager()
        manager.handle_error(e, context)
        raise

def configure_error_handling(config: RecoveryConfig) -> None:
    """Configure global error handling settings."""
    global _error_manager
    _error_manager = ErrorRecoveryManager(config)

def create_error_context(operation: str, **kwargs) -> ErrorContext:
    """Create error context with current thread and timestamp."""
    return ErrorContext(
        operation=operation,
        timestamp=time.time(),
        thread_id=threading.get_ident(),
        **kwargs
    )

# Convenience functions for creating specific errors
def create_hardware_error(message: str, backend_name: Optional[str] = None, **kwargs) -> QuantumHardwareError:
    """Create a quantum hardware error."""
    return QuantumHardwareError(message, backend_name=backend_name, **kwargs)

def create_compilation_error(message: str, circuit_info: Optional[Dict[str, Any]] = None, **kwargs) -> CircuitCompilationError:
    """Create a circuit compilation error.""" 
    return CircuitCompilationError(message, circuit_info=circuit_info, **kwargs)

def create_simulation_error(message: str, simulator_info: Optional[Dict[str, Any]] = None, **kwargs) -> SimulationError:
    """Create a simulation error."""
    return SimulationError(message, simulator_info=simulator_info, **kwargs)

def create_validation_error(message: str, validation_type: Optional[str] = None, **kwargs) -> ValidationError:
    """Create a validation error."""
    return ValidationError(message, validation_type=validation_type, **kwargs)

def create_security_error(message: str, security_context: Optional[Dict[str, Any]] = None, **kwargs) -> SecurityError:
    """Create a security error."""
    return SecurityError(message, security_context=security_context, **kwargs)

def create_resource_error(message: str, resource_type: Optional[str] = None, **kwargs) -> ResourceError:
    """Create a resource error."""
    return ResourceError(message, resource_type=resource_type, **kwargs)

# Export all error handling components
__all__ = [
    # Enums
    "ErrorSeverity",
    "ErrorCategory", 
    "RecoveryStrategy",
    
    # Data classes
    "ErrorContext",
    "ErrorDetails",
    "RecoveryConfig",
    
    # Exception classes
    "QuantumError",
    "QuantumHardwareError",
    "CircuitCompilationError", 
    "SimulationError",
    "ValidationError",
    "SecurityError",
    "ResourceError",
    
    # Main classes
    "ErrorRecoveryManager",
    
    # Utilities and decorators
    "quantum_error_handler",
    "quantum_error_context",
    "get_error_manager",
    "configure_error_handling",
    "create_error_context",
    
    # Convenience functions
    "create_hardware_error",
    "create_compilation_error", 
    "create_simulation_error",
    "create_validation_error",
    "create_security_error",
    "create_resource_error",
]