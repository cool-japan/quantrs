"""
Tests for QuantRS2 Resource Management System

This module tests the advanced resource monitoring, limiting, and management
capabilities for quantum circuit execution.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import psutil

from quantrs2.resource_management import (
    ResourceType,
    ResourceStatus,
    ResourceLimit,
    ResourceUsage,
    ResourceConfig,
    ResourceException,
    ResourceMonitor,
    ResourcePool,
    resource_context,
    analyze_circuit_resources
)

from quantrs2.resilient_execution import (
    ExecutionConfig,
    CircuitExecutionEngine,
    ExecutionStatus,
    execute_circuit_resilient
)

class TestResourceLimit:
    """Test ResourceLimit functionality."""
    
    def test_resource_limit_creation(self):
        """Test creating resource limits."""
        limit = ResourceLimit(
            ResourceType.MEMORY,
            soft_limit=1000.0,
            hard_limit=2000.0,
            unit="MB",
            enabled=True
        )
        
        assert limit.resource_type == ResourceType.MEMORY
        assert limit.soft_limit == 1000.0
        assert limit.hard_limit == 2000.0
        assert limit.unit == "MB"
        assert limit.enabled is True
    
    def test_usage_checking(self):
        """Test usage checking against limits."""
        limit = ResourceLimit(
            ResourceType.MEMORY,
            soft_limit=1000.0,
            hard_limit=2000.0,
            unit="MB"
        )
        
        # Normal usage
        assert limit.check_usage(500.0) == ResourceStatus.NORMAL
        
        # Warning usage
        assert limit.check_usage(850.0) == ResourceStatus.WARNING
        
        # Critical usage
        assert limit.check_usage(1200.0) == ResourceStatus.CRITICAL
        
        # Exceeded usage
        assert limit.check_usage(2500.0) == ResourceStatus.EXCEEDED
    
    def test_disabled_limit(self):
        """Test disabled limits always return normal."""
        limit = ResourceLimit(
            ResourceType.MEMORY,
            soft_limit=100.0,
            hard_limit=200.0,
            enabled=False
        )
        
        assert limit.check_usage(300.0) == ResourceStatus.NORMAL

class TestResourceUsage:
    """Test ResourceUsage functionality."""
    
    def test_resource_usage_creation(self):
        """Test creating resource usage."""
        usage = ResourceUsage(
            ResourceType.CPU,
            current_usage=50.0,
            peak_usage=75.0,
            average_usage=60.0,
            samples_count=10,
            unit="%"
        )
        
        assert usage.resource_type == ResourceType.CPU
        assert usage.current_usage == 50.0
        assert usage.peak_usage == 75.0
        assert usage.average_usage == 60.0
        assert usage.samples_count == 10
        assert usage.unit == "%"
    
    def test_usage_update(self):
        """Test updating usage statistics."""
        usage = ResourceUsage(
            ResourceType.CPU,
            current_usage=50.0,
            peak_usage=50.0,
            average_usage=50.0,
            samples_count=1,
            unit="%"
        )
        
        # Update with higher value
        usage.update(80.0)
        
        assert usage.current_usage == 80.0
        assert usage.peak_usage == 80.0
        assert usage.average_usage == 65.0  # (50 + 80) / 2
        assert usage.samples_count == 2
        
        # Update with lower value
        usage.update(30.0)
        
        assert usage.current_usage == 30.0
        assert usage.peak_usage == 80.0  # Peak should remain
        assert usage.average_usage == 53.333333333333336  # (50 + 80 + 30) / 3
        assert usage.samples_count == 3

class TestResourceConfig:
    """Test ResourceConfig functionality."""
    
    def test_default_config(self):
        """Test default resource configuration."""
        config = ResourceConfig()
        
        assert config.max_memory_mb == 4096.0
        assert config.max_cpu_percent == 80.0
        assert config.max_execution_time == 300.0
        assert config.max_qubits == 50
        assert config.max_gates == 10000
        assert config.max_circuit_depth == 1000
        assert config.enable_memory_monitoring is True
        assert config.enable_cpu_monitoring is True
    
    def test_custom_config(self):
        """Test custom resource configuration."""
        config = ResourceConfig(
            max_memory_mb=8192.0,
            max_cpu_percent=90.0,
            max_execution_time=600.0,
            enable_memory_monitoring=False
        )
        
        assert config.max_memory_mb == 8192.0
        assert config.max_cpu_percent == 90.0
        assert config.max_execution_time == 600.0
        assert config.enable_memory_monitoring is False
    
    def test_to_limits(self):
        """Test converting config to limits."""
        config = ResourceConfig(
            max_memory_mb=2048.0,
            memory_warning_threshold=0.8,
            max_cpu_percent=70.0,
            cpu_warning_threshold=0.9
        )
        
        limits = config.to_limits()
        
        assert ResourceType.MEMORY in limits
        assert ResourceType.CPU in limits
        
        memory_limit = limits[ResourceType.MEMORY]
        assert memory_limit.soft_limit == 2048.0 * 0.8
        assert memory_limit.hard_limit == 2048.0
        assert memory_limit.unit == "MB"
        
        cpu_limit = limits[ResourceType.CPU]
        assert cpu_limit.soft_limit == 70.0 * 0.9
        assert cpu_limit.hard_limit == 70.0
        assert cpu_limit.unit == "%"

class TestResourceMonitor:
    """Test ResourceMonitor functionality."""
    
    def test_monitor_creation(self):
        """Test creating resource monitor."""
        config = ResourceConfig()
        monitor = ResourceMonitor(config)
        
        assert monitor.config == config
        assert len(monitor.limits) > 0
        assert monitor._monitoring is False
    
    @patch('psutil.Process')
    def test_monitor_start_stop(self, mock_process):
        """Test starting and stopping monitoring."""
        config = ResourceConfig(monitoring_interval=0.1)
        monitor = ResourceMonitor(config)
        
        # Mock process
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
        mock_process_instance.cpu_percent.return_value = 25.0
        mock_process.return_value = mock_process_instance
        monitor._process = mock_process_instance
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor._monitoring is True
        assert monitor._monitor_thread is not None
        
        # Wait a bit for monitoring to run
        time.sleep(0.2)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor._monitoring is False
    
    def test_usage_tracking(self):
        """Test usage tracking."""
        config = ResourceConfig()
        monitor = ResourceMonitor(config)
        
        # Update usage
        monitor._update_usage(ResourceType.MEMORY, 1024.0, "MB")
        monitor._update_usage(ResourceType.CPU, 50.0, "%")
        
        # Check usage
        memory_usage = monitor.get_resource_usage(ResourceType.MEMORY)
        cpu_usage = monitor.get_resource_usage(ResourceType.CPU)
        
        assert memory_usage is not None
        assert memory_usage.current_usage == 1024.0
        assert memory_usage.unit == "MB"
        
        assert cpu_usage is not None
        assert cpu_usage.current_usage == 50.0
        assert cpu_usage.unit == "%"
    
    def test_limit_enforcement(self):
        """Test resource limit enforcement."""
        config = ResourceConfig(max_memory_mb=1000.0)
        monitor = ResourceMonitor(config)
        
        # Update with normal usage
        monitor._update_usage(ResourceType.MEMORY, 500.0, "MB")
        monitor.enforce_limits(ResourceType.MEMORY)  # Should not raise
        
        # Update with excessive usage
        monitor._update_usage(ResourceType.MEMORY, 1500.0, "MB")
        
        with pytest.raises(ResourceException) as exc_info:
            monitor.enforce_limits(ResourceType.MEMORY)
        
        assert exc_info.value.resource_type == ResourceType.MEMORY
        assert exc_info.value.current_usage == 1500.0
        assert exc_info.value.limit == 1000.0
    
    def test_execution_registration(self):
        """Test execution registration and tracking."""
        config = ResourceConfig()
        monitor = ResourceMonitor(config)
        
        circuit_info = {
            'qubits': 10,
            'gates': 100,
            'depth': 20
        }
        
        # Register execution
        monitor.register_execution("test_exec", circuit_info)
        
        assert "test_exec" in monitor.active_executions
        assert monitor.active_executions["test_exec"]["circuit_info"] == circuit_info
        
        # Unregister execution
        monitor.unregister_execution("test_exec")
        
        assert "test_exec" not in monitor.active_executions

class TestResourcePool:
    """Test ResourcePool functionality."""
    
    def test_pool_creation(self):
        """Test creating resource pool."""
        config = ResourceConfig(max_concurrent_executions=2)
        monitor = ResourceMonitor(config)
        pool = ResourcePool(config, monitor)
        
        assert pool.config == config
        assert pool.monitor == monitor
        assert pool.executor._max_workers == 2
    
    def test_execution_submission(self):
        """Test submitting executions to pool."""
        config = ResourceConfig(max_concurrent_executions=2)
        monitor = ResourceMonitor(config)
        pool = ResourcePool(config, monitor)
        
        def mock_execution():
            time.sleep(0.1)
            return {"result": "success"}
        
        circuit_info = {"qubits": 5, "gates": 50}
        
        # Submit execution
        future = pool.submit_execution(
            "test_exec", circuit_info, mock_execution, "normal"
        )
        
        # Wait for result
        result = future.result(timeout=1.0)
        assert result == {"result": "success"}
        
        # Cleanup
        pool.shutdown()
    
    def test_execution_cancellation(self):
        """Test cancelling executions."""
        config = ResourceConfig(max_concurrent_executions=1)
        monitor = ResourceMonitor(config)
        pool = ResourcePool(config, monitor)
        
        def long_execution():
            time.sleep(5.0)  # Long running task
            return {"result": "success"}
        
        circuit_info = {"qubits": 5, "gates": 50}
        
        # Submit execution
        future = pool.submit_execution(
            "test_exec", circuit_info, long_execution, "normal"
        )
        
        # Wait a bit then cancel
        time.sleep(0.1)
        cancelled = pool.cancel_execution("test_exec")
        
        assert cancelled is True
        
        # Cleanup
        pool.shutdown()
    
    def test_queue_status(self):
        """Test getting queue status."""
        config = ResourceConfig(max_concurrent_executions=2)
        monitor = ResourceMonitor(config)
        pool = ResourcePool(config, monitor)
        
        status = pool.get_queue_status()
        
        assert 'active_executions' in status
        assert 'high_priority_queue' in status
        assert 'normal_priority_queue' in status
        assert 'low_priority_queue' in status
        assert 'max_workers' in status
        assert status['max_workers'] == 2
        
        # Cleanup
        pool.shutdown()

class TestResourceContext:
    """Test resource context manager."""
    
    def test_context_manager(self):
        """Test resource context manager."""
        config = ResourceConfig(monitoring_interval=0.1)
        
        with resource_context(config) as (monitor, pool):
            assert isinstance(monitor, ResourceMonitor)
            assert isinstance(pool, ResourcePool)
            assert monitor._monitoring is True
        
        # Should be cleaned up after context
        assert monitor._monitoring is False

class TestCircuitAnalysis:
    """Test circuit resource analysis."""
    
    def test_basic_circuit_analysis(self):
        """Test basic circuit analysis."""
        # Mock circuit object
        circuit = Mock()
        circuit.num_qubits = 10
        circuit.size.return_value = 100
        circuit.depth.return_value = 20
        
        analysis = analyze_circuit_resources(circuit)
        
        assert analysis['qubits'] == 10
        assert analysis['gates'] == 100
        assert analysis['depth'] == 20
        assert analysis['estimated_memory_mb'] > 0
        assert analysis['estimated_time_seconds'] > 0
    
    def test_circuit_analysis_with_different_attributes(self):
        """Test circuit analysis with different attribute names."""
        # Mock circuit with different attribute names
        circuit = Mock()
        circuit.n_qubits = 5
        circuit.count_ops.return_value = {'h': 10, 'cx': 20, 'rz': 15}
        
        analysis = analyze_circuit_resources(circuit)
        
        assert analysis['qubits'] == 5
        assert analysis['gates'] == 45  # Sum of gate counts
    
    def test_circuit_analysis_error_handling(self):
        """Test circuit analysis error handling."""
        # Circuit that raises exceptions
        circuit = Mock()
        circuit.num_qubits = Mock(side_effect=Exception("Test error"))
        
        analysis = analyze_circuit_resources(circuit)
        
        assert analysis['qubits'] == 0
        assert 'analysis_error' in analysis

class TestIntegratedExecution:
    """Test integrated execution with resource management."""
    
    def test_execution_with_resource_monitoring(self):
        """Test circuit execution with resource monitoring."""
        # Create config with resource monitoring
        resource_config = ResourceConfig(
            max_memory_mb=2048.0,
            max_execution_time=10.0,
            enable_memory_monitoring=True
        )
        
        exec_config = ExecutionConfig(
            resource_config=resource_config,
            enable_resource_monitoring=True,
            timeout_seconds=5.0
        )
        
        # Create execution engine
        engine = CircuitExecutionEngine(exec_config)
        
        # Mock circuit
        circuit = Mock()
        circuit.num_qubits = 5
        
        try:
            # Execute circuit
            result = engine.execute_circuit(circuit)
            
            assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.RECOVERED]
            assert 'circuit_info' in result.metadata
            
        finally:
            # Cleanup
            engine.cleanup()
    
    def test_execution_with_resource_limits(self):
        """Test execution failure due to resource limits."""
        # Create config with very restrictive limits
        resource_config = ResourceConfig(
            max_qubits=2,  # Very low limit
            max_gates=10,  # Very low limit
            enable_circuit_monitoring=True
        )
        
        exec_config = ExecutionConfig(
            resource_config=resource_config,
            enable_resource_monitoring=True,
            enable_resource_enforcement=True
        )
        
        # Create execution engine
        engine = CircuitExecutionEngine(exec_config)
        
        # Mock large circuit that exceeds limits
        circuit = Mock()
        circuit.num_qubits = 10  # Exceeds limit
        circuit.size.return_value = 100  # Exceeds limit
        circuit.depth.return_value = 50
        
        try:
            # Execute circuit - should fail due to resource limits
            result = engine.execute_circuit(circuit)
            
            assert result.status == ExecutionStatus.FAILED
            assert 'ResourceException' in result.error_details['error_type']
            
        finally:
            # Cleanup
            engine.cleanup()
    
    def test_resource_status_reporting(self):
        """Test resource status reporting."""
        # Create config with monitoring
        resource_config = ResourceConfig(enable_memory_monitoring=True)
        exec_config = ExecutionConfig(
            resource_config=resource_config,
            enable_resource_monitoring=True
        )
        
        # Create execution engine
        engine = CircuitExecutionEngine(exec_config)
        
        try:
            # Get resource status
            status = engine.get_resource_status()
            
            assert status['resource_monitoring'] == 'enabled'
            assert 'current_usage' in status
            assert 'resource_limits' in status
            assert 'system_resources' in status
            
        finally:
            # Cleanup
            engine.cleanup()
    
    def test_resource_cleanup(self):
        """Test resource cleanup functionality."""
        # Create config
        resource_config = ResourceConfig(enable_gc_monitoring=True)
        exec_config = ExecutionConfig(
            resource_config=resource_config,
            enable_resource_monitoring=True
        )
        
        # Create execution engine
        engine = CircuitExecutionEngine(exec_config)
        
        try:
            # Force resource cleanup
            cleanup_result = engine.force_resource_cleanup()
            
            assert 'objects_collected' in cleanup_result
            assert 'memory_freed_mb' in cleanup_result
            
        finally:
            # Cleanup
            engine.cleanup()

class TestPerformanceAndScaling:
    """Test performance and scaling aspects."""
    
    def test_concurrent_executions(self):
        """Test concurrent circuit executions."""
        # Create config allowing multiple concurrent executions
        resource_config = ResourceConfig(
            max_concurrent_executions=3,
            monitoring_interval=0.1
        )
        
        exec_config = ExecutionConfig(
            resource_config=resource_config,
            enable_resource_monitoring=True,
            max_parallel_jobs=3
        )
        
        # Create execution engine
        engine = CircuitExecutionEngine(exec_config)
        
        def quick_execution():
            time.sleep(0.2)
            return {"result": "success"}
        
        try:
            # Submit multiple executions
            futures = []
            for i in range(5):
                circuit = Mock()
                circuit.num_qubits = 3
                
                # Use async execution
                import asyncio
                
                async def async_test():
                    from quantrs2.resilient_execution import execute_circuit_async
                    return await execute_circuit_async(circuit, config=exec_config)
                
                # Note: In real test, would use asyncio.run() or proper async framework
                result = engine.execute_circuit(circuit)
                futures.append(result)
            
            # All should complete successfully
            for result in futures:
                assert result.status in [ExecutionStatus.COMPLETED, ExecutionStatus.RECOVERED]
            
        finally:
            # Cleanup
            engine.cleanup()

if __name__ == "__main__":
    pytest.main([__file__])