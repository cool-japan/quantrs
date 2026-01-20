#!/usr/bin/env python3
"""
QuantRS2 Resource Management Demo

This demo showcases the advanced resource monitoring, limiting, and management
capabilities of the QuantRS2 quantum computing framework.
"""

import time
import logging
import asyncio
from typing import Dict, Any
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from quantrs2.resource_management import (
        ResourceType,
        ResourceStatus,
        ResourceConfig,
        ResourceMonitor,
        ResourcePool,
        ResourceException,
        resource_context,
        analyze_circuit_resources
    )
    
    from quantrs2.resilient_execution import (
        ExecutionConfig,
        CircuitExecutionEngine,
        ExecutionStatus,
        execute_circuit_resilient,
        execute_circuit_async
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure QuantRS2 is properly installed")
    exit(1)

class MockQuantumCircuit:
    """Mock quantum circuit for demonstration."""
    
    def __init__(self, num_qubits: int, num_gates: int, depth: int = None):
        self.num_qubits = num_qubits
        self.num_gates = num_gates
        self.circuit_depth = depth or (num_gates // num_qubits)
    
    def size(self) -> int:
        return self.num_gates
    
    def depth(self) -> int:
        return self.circuit_depth
    
    def count_ops(self) -> Dict[str, int]:
        # Distribute gates among different types
        return {
            'h': self.num_gates // 4,
            'cx': self.num_gates // 2,
            'rz': self.num_gates // 4
        }

def demonstrate_resource_limits():
    """Demonstrate resource limit checking and enforcement."""
    print("\n" + "="*60)
    print("RESOURCE LIMITS DEMONSTRATION")
    print("="*60)
    
    # Create restrictive resource configuration
    config = ResourceConfig(
        max_memory_mb=512.0,  # 512 MB limit
        max_cpu_percent=50.0,  # 50% CPU limit
        max_execution_time=10.0,  # 10 seconds
        max_qubits=20,  # 20 qubit limit
        max_gates=1000,  # 1000 gate limit
        max_circuit_depth=100,  # 100 depth limit
        enable_memory_monitoring=True,
        enable_circuit_monitoring=True
    )
    
    print(f"Resource limits configured:")
    print(f"  Memory: {config.max_memory_mb} MB")
    print(f"  CPU: {config.max_cpu_percent}%")
    print(f"  Execution time: {config.max_execution_time} seconds")
    print(f"  Qubits: {config.max_qubits}")
    print(f"  Gates: {config.max_gates}")
    print(f"  Circuit depth: {config.max_circuit_depth}")
    
    # Test circuits with different resource requirements
    test_circuits = [
        ("Small circuit", MockQuantumCircuit(5, 50, 10)),
        ("Medium circuit", MockQuantumCircuit(15, 500, 50)),
        ("Large circuit (exceeds limits)", MockQuantumCircuit(30, 2000, 150))
    ]
    
    monitor = ResourceMonitor(config)
    monitor.start_monitoring()
    
    try:
        for name, circuit in test_circuits:
            print(f"\nTesting {name}:")
            
            # Analyze circuit resources
            analysis = analyze_circuit_resources(circuit)
            print(f"  Circuit analysis:")
            print(f"    Qubits: {analysis['qubits']}")
            print(f"    Gates: {analysis['gates']}")
            print(f"    Depth: {analysis['depth']}")
            print(f"    Estimated memory: {analysis['estimated_memory_mb']:.2f} MB")
            print(f"    Estimated time: {analysis['estimated_time_seconds']:.2f} seconds")
            
            # Register circuit for monitoring
            monitor.register_execution(f"test_{name}", analysis)
            
            # Check if circuit exceeds limits
            try:
                # Simulate checking circuit complexity limits
                limits = monitor.limits
                
                if analysis['qubits'] > limits[ResourceType.QUBITS].hard_limit:
                    raise ResourceException(
                        f"Circuit requires {analysis['qubits']} qubits, exceeds limit",
                        ResourceType.QUBITS, analysis['qubits'], 
                        limits[ResourceType.QUBITS].hard_limit
                    )
                
                if analysis['gates'] > limits[ResourceType.GATES].hard_limit:
                    raise ResourceException(
                        f"Circuit has {analysis['gates']} gates, exceeds limit",
                        ResourceType.GATES, analysis['gates'],
                        limits[ResourceType.GATES].hard_limit
                    )
                
                print(f"  ✅ Circuit passes resource checks")
                
            except ResourceException as e:
                print(f"  ❌ Circuit exceeds resource limits: {e}")
            
            finally:
                monitor.unregister_execution(f"test_{name}")
    
    finally:
        monitor.stop_monitoring()

def demonstrate_execution_with_monitoring():
    """Demonstrate quantum circuit execution with resource monitoring."""
    print("\n" + "="*60)
    print("EXECUTION WITH MONITORING DEMONSTRATION")
    print("="*60)
    
    # Create execution configuration with monitoring
    resource_config = ResourceConfig(
        max_memory_mb=2048.0,
        max_cpu_percent=80.0,
        max_execution_time=30.0,
        max_concurrent_executions=3,
        monitoring_interval=0.5
    )
    
    exec_config = ExecutionConfig(
        resource_config=resource_config,
        enable_resource_monitoring=True,
        enable_resource_enforcement=True,
        timeout_seconds=15.0
    )
    
    print(f"Execution configuration:")
    print(f"  Resource monitoring: {exec_config.enable_resource_monitoring}")
    print(f"  Resource enforcement: {exec_config.enable_resource_enforcement}")
    print(f"  Timeout: {exec_config.timeout_seconds} seconds")
    
    # Create execution engine
    engine = CircuitExecutionEngine(exec_config)
    
    try:
        # Test circuits
        test_circuits = [
            ("Fast circuit", MockQuantumCircuit(3, 20, 5)),
            ("Medium circuit", MockQuantumCircuit(8, 100, 25)),
            ("Complex circuit", MockQuantumCircuit(12, 300, 50))
        ]
        
        results = []
        
        for name, circuit in test_circuits:
            print(f"\nExecuting {name}...")
            
            start_time = time.time()
            result = engine.execute_circuit(circuit)
            execution_time = time.time() - start_time
            
            print(f"  Status: {result.status.value}")
            print(f"  Execution time: {execution_time:.2f} seconds")
            print(f"  Execution ID: {result.execution_id}")
            
            if result.status == ExecutionStatus.COMPLETED:
                print(f"  ✅ Execution successful")
                if 'resource_usage' in result.metadata:
                    print(f"  Resource usage captured in metadata")
            elif result.status == ExecutionStatus.FAILED:
                print(f"  ❌ Execution failed: {result.error_details}")
            elif result.status == ExecutionStatus.RECOVERED:
                print(f"  🔄 Execution recovered with: {result.recovery_applied}")
            
            results.append(result)
        
        # Get execution statistics
        stats = engine.get_execution_statistics()
        print(f"\nExecution Statistics:")
        print(f"  Total executions: {stats['total_executions']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Average execution time: {stats['average_execution_time']:.2f} seconds")
        print(f"  Recovery rate: {stats['recovery_rate']:.2%}")
        
        # Get resource status
        resource_status = engine.get_resource_status()
        if resource_status['resource_monitoring'] == 'enabled':
            print(f"\nCurrent Resource Status:")
            for resource_type, usage in resource_status['current_usage'].items():
                print(f"  {resource_type}: {usage['current_usage']:.2f} {usage['unit']} "
                      f"(peak: {usage['peak_usage']:.2f})")
    
    finally:
        engine.cleanup()

def demonstrate_concurrent_execution():
    """Demonstrate concurrent circuit execution with resource pooling."""
    print("\n" + "="*60)
    print("CONCURRENT EXECUTION DEMONSTRATION")
    print("="*60)
    
    # Create configuration for concurrent execution
    resource_config = ResourceConfig(
        max_concurrent_executions=3,
        monitoring_interval=0.2,
        enable_priority_scheduling=True
    )
    
    exec_config = ExecutionConfig(
        resource_config=resource_config,
        enable_resource_monitoring=True,
        enable_parallel_execution=True,
        max_parallel_jobs=3
    )
    
    print(f"Concurrent execution configuration:")
    print(f"  Max concurrent executions: {resource_config.max_concurrent_executions}")
    print(f"  Priority scheduling: {resource_config.enable_priority_scheduling}")
    
    # Create execution engine
    engine = CircuitExecutionEngine(exec_config)
    
    try:
        # Create circuits with different priorities
        circuits = [
            ("High priority circuit", MockQuantumCircuit(4, 30), "high"),
            ("Normal circuit 1", MockQuantumCircuit(6, 50), "normal"), 
            ("Normal circuit 2", MockQuantumCircuit(5, 40), "normal"),
            ("Low priority circuit", MockQuantumCircuit(8, 80), "low"),
            ("Normal circuit 3", MockQuantumCircuit(7, 60), "normal")
        ]
        
        print(f"\nSubmitting {len(circuits)} circuits for execution...")
        
        # Submit all circuits
        start_time = time.time()
        results = []
        
        for name, circuit, priority in circuits:
            # Update execution config for priority
            priority_config = ExecutionConfig(
                resource_config=resource_config,
                enable_resource_monitoring=True,
                execution_priority=priority
            )
            
            print(f"  Submitting {name} (priority: {priority})")
            result = engine.execute_circuit(circuit, config=priority_config)
            results.append((name, result, priority))
        
        total_time = time.time() - start_time
        
        # Report results
        print(f"\nAll executions completed in {total_time:.2f} seconds")
        print(f"Results:")
        
        for name, result, priority in results:
            status_emoji = {
                ExecutionStatus.COMPLETED: "✅",
                ExecutionStatus.RECOVERED: "🔄", 
                ExecutionStatus.FAILED: "❌"
            }.get(result.status, "❓")
            
            print(f"  {status_emoji} {name} ({priority}): {result.status.value} "
                  f"({result.execution_time:.2f}s)")
        
        # Get queue status
        if hasattr(engine, 'resource_pool') and engine.resource_pool:
            queue_status = engine.resource_pool.get_queue_status()
            print(f"\nFinal queue status:")
            print(f"  Completed executions: {queue_status['completed_executions']}")
    
    finally:
        engine.cleanup()

async def demonstrate_async_execution():
    """Demonstrate asynchronous circuit execution."""
    print("\n" + "="*60)
    print("ASYNCHRONOUS EXECUTION DEMONSTRATION")
    print("="*60)
    
    # Create circuits for async execution
    circuits = [
        ("Async circuit 1", MockQuantumCircuit(4, 25)),
        ("Async circuit 2", MockQuantumCircuit(6, 45)),
        ("Async circuit 3", MockQuantumCircuit(5, 35))
    ]
    
    exec_config = ExecutionConfig(
        enable_resource_monitoring=True,
        timeout_seconds=10.0
    )
    
    print(f"Executing {len(circuits)} circuits asynchronously...")
    
    # Execute circuits asynchronously
    start_time = time.time()
    
    tasks = []
    for name, circuit in circuits:
        print(f"  Scheduling {name}")
        task = execute_circuit_async(circuit, config=exec_config)
        tasks.append((name, task))
    
    # Wait for all to complete
    results = []
    for name, task in tasks:
        try:
            result = await task
            results.append((name, result))
            print(f"  ✅ {name} completed: {result.status.value}")
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
    
    total_time = time.time() - start_time
    print(f"\nAll async executions completed in {total_time:.2f} seconds")

def demonstrate_resource_context():
    """Demonstrate resource context manager."""
    print("\n" + "="*60)
    print("RESOURCE CONTEXT DEMONSTRATION")
    print("="*60)
    
    config = ResourceConfig(
        max_memory_mb=1024.0,
        monitoring_interval=0.1,
        max_concurrent_executions=2
    )
    
    print("Using resource context manager for automatic cleanup...")
    
    with resource_context(config) as (monitor, pool):
        print(f"  Resource monitor active: {monitor._monitoring}")
        print(f"  Resource pool max workers: {pool.executor._max_workers}")
        
        # Do some work
        def test_execution():
            time.sleep(0.2)
            return {"status": "success"}
        
        circuit_info = {"qubits": 3, "gates": 20}
        future = pool.submit_execution("test", circuit_info, test_execution)
        result = future.result()
        
        print(f"  ✅ Test execution completed: {result}")
        
        # Get status
        status = pool.get_queue_status()
        print(f"  Active executions: {status['active_executions']}")
        print(f"  Completed executions: {status['completed_executions']}")
    
    print("  Resource context automatically cleaned up")

def main():
    """Run all demonstrations."""
    print("QuantRS2 Advanced Resource Management Demo")
    print("==========================================")
    
    try:
        # Basic resource limit demonstrations
        demonstrate_resource_limits()
        
        # Execution with monitoring
        demonstrate_execution_with_monitoring()
        
        # Concurrent execution
        demonstrate_concurrent_execution()
        
        # Resource context manager
        demonstrate_resource_context()
        
        # Async execution
        print("\nRunning async demonstration...")
        asyncio.run(demonstrate_async_execution())
        
        print("\n" + "="*60)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nKey Features Demonstrated:")
        print("✅ Resource monitoring and limits")
        print("✅ Circuit complexity analysis")
        print("✅ Execution timeouts and cancellation")
        print("✅ Concurrent execution with priority scheduling")
        print("✅ Resource pool management")
        print("✅ Asynchronous execution")
        print("✅ Automatic resource cleanup")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()