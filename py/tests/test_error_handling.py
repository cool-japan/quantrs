"""
Tests for QuantRS2 Error Handling and Recovery System
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch

try:
    from quantrs2.error_handling import (
        ErrorSeverity,
        ErrorCategory,
        RecoveryStrategy,
        ErrorContext,
        ErrorDetails,
        RecoveryConfig,
        QuantumError,
        QuantumHardwareError,
        CircuitCompilationError,
        SimulationError,
        ValidationError,
        SecurityError,
        ResourceError,
        ErrorRecoveryManager,
        quantum_error_handler,
        quantum_error_context,
        get_error_manager,
        configure_error_handling,
        create_error_context,
        create_hardware_error,
        create_compilation_error,
        create_simulation_error,
        create_validation_error,
        create_security_error,
        create_resource_error,
    )
    HAS_ERROR_HANDLING = True
except ImportError:
    HAS_ERROR_HANDLING = False
    
    # Stub implementations
    class ErrorSeverity: pass
    class ErrorCategory: pass
    class RecoveryStrategy: pass
    class ErrorContext: pass
    class ErrorDetails: pass
    class RecoveryConfig: pass
    class QuantumError(Exception): pass
    class QuantumHardwareError(Exception): pass
    class CircuitCompilationError(Exception): pass
    class SimulationError(Exception): pass
    class ValidationError(Exception): pass
    class SecurityError(Exception): pass
    class ResourceError(Exception): pass
    class ErrorRecoveryManager: pass
    def quantum_error_handler(): pass
    def quantum_error_context(): pass
    def get_error_manager(): pass
    def configure_error_handling(): pass
    def create_error_context(): pass
    def create_hardware_error(): pass
    def create_compilation_error(): pass
    def create_simulation_error(): pass
    def create_validation_error(): pass
    def create_security_error(): pass
    def create_resource_error(): pass

@pytest.mark.skipif(not HAS_ERROR_HANDLING, reason="quantrs2.error_handling module not available")
class TestErrorContext:
    """Test error context functionality."""
    
    def test_error_context_creation(self):
        """Test creating error context."""
        context = ErrorContext(
            operation="test_operation",
            timestamp=time.time(),
            thread_id=threading.get_ident(),
            session_id="session_123",
            user_id="user_456"
        )
        
        assert context.operation == "test_operation"
        assert context.session_id == "session_123"
        assert context.user_id == "user_456"
        assert isinstance(context.timestamp, float)
        assert isinstance(context.thread_id, int)
    
    def test_error_context_to_dict(self):
        """Test error context serialization."""
        context = ErrorContext(
            operation="test_operation",
            timestamp=123456.789,
            thread_id=12345,
            session_id="session_123"
        )
        
        context_dict = context.to_dict()
        
        assert context_dict["operation"] == "test_operation"
        assert context_dict["timestamp"] == 123456.789
        assert context_dict["thread_id"] == 12345
        assert context_dict["session_id"] == "session_123"

@pytest.mark.skipif(not HAS_ERROR_HANDLING, reason="quantrs2.error_handling module not available")
class TestErrorDetails:
    """Test error details functionality."""
    
    def test_error_details_creation(self):
        """Test creating error details."""
        context = ErrorContext("test_op", time.time(), threading.get_ident())
        details = ErrorDetails(
            error_id="error_123",
            category=ErrorCategory.SIMULATION,
            severity=ErrorSeverity.HIGH,
            message="Test error",
            context=context
        )
        
        assert details.error_id == "error_123"
        assert details.category == ErrorCategory.SIMULATION
        assert details.severity == ErrorSeverity.HIGH
        assert details.message == "Test error"
        assert details.context == context
        assert not details.recovery_successful
        assert len(details.recovery_attempted) == 0
    
    def test_error_details_to_dict(self):
        """Test error details serialization."""
        details = ErrorDetails(
            error_id="error_123",
            category=ErrorCategory.SIMULATION,
            severity=ErrorSeverity.HIGH,
            message="Test error"
        )
        
        details_dict = details.to_dict()
        
        assert details_dict["error_id"] == "error_123"
        assert details_dict["category"] == "simulation"
        assert details_dict["severity"] == "high"
        assert details_dict["message"] == "Test error"
        assert details_dict["recovery_successful"] is False

@pytest.mark.skipif(not HAS_ERROR_HANDLING, reason="quantrs2.error_handling module not available")
class TestQuantumErrors:
    """Test quantum-specific error classes."""
    
    def test_quantum_error_base(self):
        """Test base quantum error."""
        context = ErrorContext("test_op", time.time(), threading.get_ident())
        error = QuantumError(
            "Test quantum error",
            category=ErrorCategory.SIMULATION,
            severity=ErrorSeverity.MEDIUM,
            context=context
        )
        
        assert str(error) == "Test quantum error"
        assert error.category == ErrorCategory.SIMULATION
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context == context
        assert isinstance(error.timestamp, float)
    
    def test_hardware_error(self):
        """Test quantum hardware error."""
        error = QuantumHardwareError(
            "Hardware failure",
            backend_name="test_backend",
            device_info={"qubits": 5, "status": "offline"}
        )
        
        assert str(error) == "Hardware failure"
        assert error.category == ErrorCategory.QUANTUM_HARDWARE
        assert error.severity == ErrorSeverity.HIGH
        assert error.backend_name == "test_backend"
        assert error.device_info["qubits"] == 5
    
    def test_compilation_error(self):
        """Test circuit compilation error."""
        circuit_info = {"gates": 100, "depth": 50}
        error = CircuitCompilationError(
            "Compilation failed",
            circuit_info=circuit_info
        )
        
        assert str(error) == "Compilation failed"
        assert error.category == ErrorCategory.CIRCUIT_COMPILATION
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.circuit_info == circuit_info
    
    def test_simulation_error(self):
        """Test simulation error."""
        error = SimulationError(
            "Simulation timeout",
            simulator_info={"type": "statevector", "qubits": 20}
        )
        
        assert str(error) == "Simulation timeout"
        assert error.category == ErrorCategory.SIMULATION
        assert error.severity == ErrorSeverity.MEDIUM
    
    def test_validation_error(self):
        """Test validation error."""
        error = ValidationError(
            "Invalid input",
            validation_type="qubit_index"
        )
        
        assert str(error) == "Invalid input"
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.LOW
        assert error.validation_type == "qubit_index"
    
    def test_security_error(self):
        """Test security error."""
        error = SecurityError(
            "Authentication failed",
            security_context={"user": "test_user", "action": "circuit_execution"}
        )
        
        assert str(error) == "Authentication failed"
        assert error.category == ErrorCategory.SECURITY
        assert error.severity == ErrorSeverity.CRITICAL
    
    def test_resource_error(self):
        """Test resource error."""
        error = ResourceError(
            "Insufficient memory",
            resource_type="memory",
            available=1024,
            required=2048
        )
        
        assert str(error) == "Insufficient memory"
        assert error.category == ErrorCategory.RESOURCE
        assert error.severity == ErrorSeverity.HIGH
        assert error.resource_type == "memory"
        assert error.available == 1024
        assert error.required == 2048

@pytest.mark.skipif(not HAS_ERROR_HANDLING, reason="quantrs2.error_handling module not available")
class TestErrorRecoveryManager:
    """Test error recovery manager functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = RecoveryConfig(
            max_retries=2,
            retry_delay=0.1,
            fallback_backends=["simulator", "local"]
        )
        self.manager = ErrorRecoveryManager(self.config)
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        assert self.manager.config.max_retries == 2
        assert self.manager.config.retry_delay == 0.1
        assert len(self.manager.error_history) == 0
        assert ErrorCategory.QUANTUM_HARDWARE in self.manager.recovery_strategies
    
    def test_error_categorization(self):
        """Test error categorization."""
        # Test QuantumError categorization
        quantum_error = QuantumHardwareError("Hardware issue")
        category = self.manager._categorize_error(quantum_error)
        assert category == ErrorCategory.QUANTUM_HARDWARE
        
        # Test standard exception categorization
        value_error = ValueError("Invalid qubit index")
        category = self.manager._categorize_error(value_error)
        assert category == ErrorCategory.USER_INPUT
        
        # Test keyword-based categorization
        network_error = Exception("Connection timeout occurred")
        category = self.manager._categorize_error(network_error)
        assert category == ErrorCategory.NETWORK
    
    def test_severity_assessment(self):
        """Test error severity assessment."""
        # Test QuantumError severity
        security_error = SecurityError("Access denied")
        severity = self.manager._assess_severity(security_error)
        assert severity == ErrorSeverity.CRITICAL
        
        # Test keyword-based severity
        critical_error = Exception("Critical system failure")
        severity = self.manager._assess_severity(critical_error)
        assert severity == ErrorSeverity.CRITICAL
        
        # Test default severity
        generic_error = Exception("Something went wrong")
        severity = self.manager._assess_severity(generic_error)
        assert severity == ErrorSeverity.MEDIUM
    
    def test_handle_error_with_recovery(self):
        """Test error handling with successful recovery."""
        # Mock a successful retry
        with patch.object(self.manager, '_retry_operation', return_value={"status": "success"}):
            error = Exception("Temporary failure")
            context = ErrorContext("test_operation", time.time(), threading.get_ident())
            
            result = self.manager.handle_error(error, context)
            
            assert result == {"status": "success"}
            assert len(self.manager.error_history) == 1
            assert self.manager.error_history[0].recovery_successful
    
    def test_handle_error_with_failed_recovery(self):
        """Test error handling with failed recovery."""
        # Mock failed recovery strategies
        with patch.object(self.manager, '_retry_operation', side_effect=Exception("Retry failed")), \
             patch.object(self.manager, '_fallback_operation', side_effect=Exception("Fallback failed")):
            
            error = Exception("Persistent failure")
            context = ErrorContext("test_operation", time.time(), threading.get_ident())
            
            with pytest.raises(QuantumError):
                self.manager.handle_error(error, context)
            
            assert len(self.manager.error_history) == 1
            assert not self.manager.error_history[0].recovery_successful
    
    def test_retry_strategy(self):
        """Test retry recovery strategy."""
        error_details = ErrorDetails(
            error_id="test_error",
            category=ErrorCategory.SIMULATION,
            severity=ErrorSeverity.MEDIUM,
            message="Test error"
        )
        
        result = self.manager._retry_operation(error_details)
        
        assert result["status"] == "retry_successful"
        assert "attempt" in result
    
    def test_backend_switching_strategy(self):
        """Test backend switching strategy."""
        error_details = ErrorDetails(
            error_id="test_error",
            category=ErrorCategory.QUANTUM_HARDWARE,
            severity=ErrorSeverity.HIGH,
            message="Backend failure"
        )
        
        result = self.manager._switch_backend(error_details)
        
        assert result["status"] == "backend_switched"
        assert result["new_backend"] == "simulator"
    
    def test_error_statistics(self):
        """Test error statistics generation."""
        # Add some test errors
        errors = [
            QuantumHardwareError("Hardware error 1"),
            SimulationError("Simulation error 1"),
            QuantumHardwareError("Hardware error 2"),
            ValidationError("Validation error 1")
        ]
        
        for error in errors:
            try:
                self.manager.handle_error(error)
            except:
                pass  # We don't care about the recovery result here
        
        stats = self.manager.get_error_statistics()
        
        assert stats["total_errors"] == 4
        assert stats["category_distribution"]["quantum_hardware"] == 2
        assert stats["category_distribution"]["simulation"] == 1
        assert stats["category_distribution"]["validation"] == 1
        assert "severity_distribution" in stats
        assert "recovery_success_rate" in stats

@pytest.mark.skipif(not HAS_ERROR_HANDLING, reason="quantrs2.error_handling module not available")
class TestErrorHandlingDecorators:
    """Test error handling decorators and context managers."""
    
    def test_quantum_error_handler_decorator(self):
        """Test error handling decorator."""
        @quantum_error_handler("test_circuit_operation")
        def failing_function():
            raise ValueError("Test error")
        
        # Mock successful recovery
        with patch.object(get_error_manager(), 'handle_error', return_value={"recovered": True}) as mock_handle:
            result = failing_function()
            
            assert result == {"recovered": True}
            mock_handle.assert_called_once()
    
    def test_quantum_error_context_manager(self):
        """Test error handling context manager."""
        mock_manager = Mock()
        
        with patch('quantrs2.error_handling.get_error_manager', return_value=mock_manager):
            try:
                with quantum_error_context("test_operation", session_id="session_123"):
                    raise ValueError("Test context error")
            except ValueError:
                pass  # Expected to re-raise after handling
            
            mock_manager.handle_error.assert_called_once()
    
    def test_create_error_context(self):
        """Test error context creation utility."""
        context = create_error_context(
            "test_operation",
            session_id="session_123",
            user_id="user_456"
        )
        
        assert context.operation == "test_operation"
        assert context.session_id == "session_123"
        assert context.user_id == "user_456"
        assert isinstance(context.timestamp, float)
        assert isinstance(context.thread_id, int)

@pytest.mark.skipif(not HAS_ERROR_HANDLING, reason="quantrs2.error_handling module not available")
class TestErrorCreationFunctions:
    """Test convenience functions for creating errors."""
    
    def test_create_hardware_error(self):
        """Test hardware error creation."""
        error = create_hardware_error(
            "Backend offline",
            backend_name="test_backend"
        )
        
        assert isinstance(error, QuantumHardwareError)
        assert str(error) == "Backend offline"
        assert error.backend_name == "test_backend"
    
    def test_create_compilation_error(self):
        """Test compilation error creation."""
        circuit_info = {"gates": 50, "depth": 25}
        error = create_compilation_error(
            "Optimization failed",
            circuit_info=circuit_info
        )
        
        assert isinstance(error, CircuitCompilationError)
        assert str(error) == "Optimization failed"
        assert error.circuit_info == circuit_info
    
    def test_create_simulation_error(self):
        """Test simulation error creation."""
        error = create_simulation_error("Memory limit exceeded")
        
        assert isinstance(error, SimulationError)
        assert str(error) == "Memory limit exceeded"
    
    def test_create_validation_error(self):
        """Test validation error creation."""
        error = create_validation_error(
            "Invalid parameter",
            validation_type="gate_parameter"
        )
        
        assert isinstance(error, ValidationError)
        assert str(error) == "Invalid parameter"
        assert error.validation_type == "gate_parameter"
    
    def test_create_security_error(self):
        """Test security error creation."""
        error = create_security_error("Unauthorized access")
        
        assert isinstance(error, SecurityError)
        assert str(error) == "Unauthorized access"
    
    def test_create_resource_error(self):
        """Test resource error creation."""
        error = create_resource_error(
            "Quota exceeded",
            resource_type="qubits"
        )
        
        assert isinstance(error, ResourceError)
        assert str(error) == "Quota exceeded"
        assert error.resource_type == "qubits"

@pytest.mark.skipif(not HAS_ERROR_HANDLING, reason="quantrs2.error_handling module not available")
class TestConfigurationAndIntegration:
    """Test configuration and integration functionality."""
    
    def test_configure_error_handling(self):
        """Test global error handling configuration."""
        config = RecoveryConfig(
            max_retries=5,
            retry_delay=2.0,
            fallback_backends=["new_backend"]
        )
        
        configure_error_handling(config)
        manager = get_error_manager()
        
        assert manager.config.max_retries == 5
        assert manager.config.retry_delay == 2.0
        assert manager.config.fallback_backends == ["new_backend"]
    
    def test_get_error_manager_with_config(self):
        """Test getting error manager with custom config."""
        config = RecoveryConfig(max_retries=10)
        manager = get_error_manager(config)
        
        assert manager.config.max_retries == 10

@pytest.mark.skipif(not HAS_ERROR_HANDLING, reason="quantrs2.error_handling module not available")
class TestThreadSafety:
    """Test thread safety of error handling."""
    
    def test_concurrent_error_handling(self):
        """Test concurrent error handling."""
        manager = ErrorRecoveryManager()
        results = []
        
        def worker():
            try:
                context = ErrorContext("worker_operation", time.time(), threading.get_ident())
                error = Exception(f"Error from thread {threading.get_ident()}")
                result = manager.handle_error(error, context)
                results.append(result)
            except:
                pass  # Ignore recovery failures for this test
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all errors were recorded
        assert len(manager.error_history) == 5
        
        # Check that all thread IDs are different
        thread_ids = [error.context.thread_id for error in manager.error_history if error.context]
        assert len(set(thread_ids)) == 5  # All unique thread IDs

# Integration tests
@pytest.mark.skipif(not HAS_ERROR_HANDLING, reason="quantrs2.error_handling module not available")
class TestErrorHandlingIntegration:
    """Test integration with quantum operations."""
    
    def test_end_to_end_error_recovery(self):
        """Test complete error handling and recovery flow."""
        @quantum_error_handler("integration_test")
        def quantum_operation_with_error():
            # Simulate a quantum operation that fails initially
            if not hasattr(quantum_operation_with_error, 'call_count'):
                quantum_operation_with_error.call_count = 0
            
            quantum_operation_with_error.call_count += 1
            
            if quantum_operation_with_error.call_count <= 2:
                raise QuantumHardwareError("Backend temporarily unavailable")
            
            return {"result": "success", "calls": quantum_operation_with_error.call_count}
        
        # Mock the retry operation to actually retry the function
        original_retry = get_error_manager()._retry_operation
        
        def mock_retry(error_details):
            # Retry the operation by calling it again
            return quantum_operation_with_error()
        
        with patch.object(get_error_manager(), '_retry_operation', side_effect=mock_retry):
            result = quantum_operation_with_error()
            
            assert result["result"] == "success"
            assert result["calls"] == 3  # Failed twice, succeeded on third try

if __name__ == "__main__":
    pytest.main([__file__])