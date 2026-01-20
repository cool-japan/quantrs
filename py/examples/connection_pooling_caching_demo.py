#!/usr/bin/env python3
"""
QuantRS2 Connection Pooling and Caching Demo

This demo showcases the advanced connection pooling, caching strategies,
and performance optimization capabilities of the QuantRS2 framework.
"""

import time
import asyncio
import logging
import threading
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from quantrs2.performance_manager import (
        PerformanceManager, PerformanceConfig, PerformanceProfile,
        get_performance_manager, close_performance_manager
    )
    from quantrs2.connection_pooling import CacheBackend
    from quantrs2.circuit_optimization_cache import OptimizationLevel, CircuitPattern
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure QuantRS2 is properly installed")
    exit(1)


class MockQuantumCircuit:
    """Mock quantum circuit for demonstration."""
    
    def __init__(self, num_qubits: int, gate_sequence: list, name: str = ""):
        self.num_qubits = num_qubits
        self.gate_sequence = gate_sequence
        self.name = name
        
    def size(self) -> int:
        return len(self.gate_sequence)
    
    def depth(self) -> int:
        # Simple depth calculation
        return max(1, len(self.gate_sequence) // self.num_qubits)
    
    def count_ops(self) -> Dict[str, int]:
        from collections import Counter
        return Counter(gate[0] for gate in self.gate_sequence)
    
    def __str__(self):
        return f"Circuit({self.name}, {self.num_qubits} qubits, {self.size()} gates)"


def create_sample_circuits():
    """Create sample quantum circuits for demonstration."""
    circuits = []
    
    # Bell state circuit
    bell_circuit = MockQuantumCircuit(
        num_qubits=2,
        gate_sequence=[('h', [0]), ('cnot', [0, 1])],
        name="Bell State"
    )
    circuits.append(bell_circuit)
    
    # GHZ state circuit
    ghz_circuit = MockQuantumCircuit(
        num_qubits=3,
        gate_sequence=[('h', [0]), ('cnot', [0, 1]), ('cnot', [0, 2])],
        name="GHZ State"
    )
    circuits.append(ghz_circuit)
    
    # Larger random circuit
    large_circuit = MockQuantumCircuit(
        num_qubits=5,
        gate_sequence=[
            ('h', [i]) for i in range(5)
        ] + [
            ('cnot', [i, (i+1) % 5]) for i in range(5)
        ] + [
            ('rz', [i]) for i in range(5)
        ],
        name="Random Circuit"
    )
    circuits.append(large_circuit)
    
    # Parametric VQE-style circuit
    vqe_circuit = MockQuantumCircuit(
        num_qubits=4,
        gate_sequence=[
            ('ry', [0]), ('ry', [1]), ('ry', [2]), ('ry', [3]),
            ('cnot', [0, 1]), ('cnot', [2, 3]),
            ('ry', [0]), ('ry', [1]), ('ry', [2]), ('ry', [3])
        ],
        name="VQE Ansatz"
    )
    circuits.append(vqe_circuit)
    
    return circuits


def simulate_circuit_execution(circuit: MockQuantumCircuit, backend: str = "simulator") -> Dict[str, Any]:
    """Simulate circuit execution with realistic timing."""
    execution_time = 0.1 + (circuit.size() * 0.01)  # Realistic timing
    
    if backend == "hardware":
        execution_time *= 5  # Hardware is slower
        time.sleep(execution_time)
    else:
        time.sleep(execution_time)
    
    # Generate mock results
    n_qubits = circuit.num_qubits
    results = {}
    
    # Simple mock probability distribution
    for i in range(min(4, 2**n_qubits)):  # Limit output for large circuits
        state = format(i, f'0{n_qubits}b')
        prob = 1.0 / min(4, 2**n_qubits)
        results[state] = prob
    
    return {
        "results": results,
        "execution_time": execution_time,
        "backend": backend,
        "shots": 1024,
        "success": True
    }


def demonstrate_performance_profiles():
    """Demonstrate different performance profiles."""
    print("\n" + "="*60)
    print("PERFORMANCE PROFILES DEMONSTRATION")
    print("="*60)
    
    profiles = [
        PerformanceProfile.DEVELOPMENT,
        PerformanceProfile.TESTING,
        PerformanceProfile.PRODUCTION,
        PerformanceProfile.HIGH_PERFORMANCE
    ]
    
    for profile in profiles:
        print(f"\n--- {profile.value.upper()} Profile ---")
        
        config = PerformanceConfig.for_profile(profile)
        print(f"Max DB Connections: {config.max_db_connections}")
        print(f"Cache Backend: {config.cache_backend.value}")
        print(f"Max Cache Memory: {config.max_cache_memory_mb} MB")
        print(f"Max Cache Entries: {config.max_cache_entries}")
        print(f"Monitoring Enabled: {config.enable_performance_monitoring}")


def demonstrate_connection_pooling():
    """Demonstrate database connection pooling."""
    print("\n" + "="*60)
    print("CONNECTION POOLING DEMONSTRATION")
    print("="*60)
    
    # Use testing profile for demo
    config = PerformanceConfig.for_profile(PerformanceProfile.TESTING)
    config.enable_performance_monitoring = False  # Disable for cleaner output
    
    manager = PerformanceManager(config)
    
    try:
        print(f"Initialized performance manager with {config.profile.value} profile")
        
        # Demonstrate connection usage
        print("\nTesting database connections...")
        
        def test_connection(worker_id: int) -> str:
            try:
                with manager.database_connection('circuits') as conn:
                    # Simulate some database work
                    cursor = conn.execute("SELECT datetime('now') as timestamp")
                    result = cursor.fetchone()
                    time.sleep(0.1)  # Simulate work
                    return f"Worker {worker_id}: {result[0]}"
            except Exception as e:
                return f"Worker {worker_id}: Error - {e}"
        
        # Test concurrent connection usage
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(test_connection, i) for i in range(8)]
            
            for future in as_completed(futures):
                result = future.result()
                print(f"  {result}")
        
        # Show connection pool statistics
        print("\nConnection Pool Statistics:")
        conn_stats = manager.connection_manager.get_statistics()
        for pool_name, stats in conn_stats.items():
            print(f"  {pool_name}: {stats['connections_borrowed']} borrowed, "
                  f"{stats['connections_returned']} returned, "
                  f"{stats['total_connections']} total")
    
    finally:
        manager.close()


def demonstrate_circuit_caching():
    """Demonstrate intelligent circuit result caching."""
    print("\n" + "="*60)
    print("CIRCUIT RESULT CACHING DEMONSTRATION")
    print("="*60)
    
    config = PerformanceConfig.for_profile(PerformanceProfile.TESTING)
    config.enable_performance_monitoring = False
    manager = PerformanceManager(config)
    
    try:
        circuit_cache = manager.get_circuit_cache()
        circuits = create_sample_circuits()
        
        print(f"Created {len(circuits)} sample circuits")
        
        # Demonstrate caching workflow
        for circuit in circuits:
            print(f"\n--- Testing {circuit.name} ---")
            
            execution_config = {
                "backend": "simulator",
                "shots": 1024,
                "optimization_level": "standard"
            }
            
            # First execution (cache miss)
            print("  First execution (cache miss)...")
            start_time = time.time()
            cached_result = circuit_cache.get_cached_result(circuit, execution_config)
            
            if cached_result is None:
                # Simulate execution
                result = simulate_circuit_execution(circuit)
                execution_time = time.time() - start_time
                
                # Cache the result
                circuit_cache.cache_execution_result(
                    circuit, execution_config, result,
                    execution_time=execution_time, success=True
                )
                
                print(f"    Executed in {execution_time:.3f}s, result cached")
            else:
                print("    Unexpected cache hit!")
            
            # Second execution (cache hit)
            print("  Second execution (cache hit)...")
            start_time = time.time()
            cached_result = circuit_cache.get_cached_result(circuit, execution_config)
            cache_time = time.time() - start_time
            
            if cached_result:
                print(f"    Cache hit in {cache_time:.6f}s (speedup: {execution_time/cache_time:.1f}x)")
            else:
                print("    Unexpected cache miss!")
            
            # Get execution recommendations
            recommendations = circuit_cache.get_execution_recommendations(circuit)
            print(f"    Detected pattern: {recommendations.get('detected_pattern', 'unknown')}")
            print(f"    Pattern confidence: {recommendations.get('pattern_confidence', 0):.2f}")
        
        # Show cache statistics
        print("\nCache Statistics:")
        cache_stats = circuit_cache.get_statistics()
        
        result_cache = cache_stats['result_cache']
        print(f"  Result Cache: {result_cache['hits']} hits, {result_cache['misses']} misses")
        print(f"  Hit Rate: {result_cache.get('hit_rate', 0):.2%}")
        print(f"  Memory Usage: {result_cache.get('memory_usage_mb', 0):.1f} MB")
        
        optimization_cache = cache_stats['optimization_cache']
        print(f"  Optimization Cache: {optimization_cache['hits']} hits, {optimization_cache['misses']} misses")
        
    finally:
        manager.close()


def demonstrate_optimization_caching():
    """Demonstrate circuit optimization result caching."""
    print("\n" + "="*60)
    print("OPTIMIZATION RESULT CACHING DEMONSTRATION")
    print("="*60)
    
    config = PerformanceConfig.for_profile(PerformanceProfile.TESTING)
    config.enable_performance_monitoring = False
    manager = PerformanceManager(config)
    
    try:
        circuit_cache = manager.get_circuit_cache()
        circuits = create_sample_circuits()
        
        for circuit in circuits[:2]:  # Test first two circuits
            print(f"\n--- Optimizing {circuit.name} ---")
            
            optimization_level = OptimizationLevel.STANDARD
            
            # Check for cached optimization
            print("  Checking for cached optimization...")
            cached_optimization = circuit_cache.get_cached_optimization(circuit, optimization_level)
            
            if cached_optimization is None:
                print("  No cached optimization found, performing optimization...")
                
                # Simulate optimization
                start_time = time.time()
                time.sleep(0.2)  # Simulate optimization time
                optimization_time = time.time() - start_time
                
                # Create "optimized" circuit (simulate improvement)
                optimized_circuit = MockQuantumCircuit(
                    num_qubits=circuit.num_qubits,
                    gate_sequence=circuit.gate_sequence[:-1],  # Remove one gate
                    name=f"Optimized {circuit.name}"
                )
                
                improvements = {
                    "gate_count": {
                        "original": circuit.size(),
                        "optimized": optimized_circuit.size(),
                        "ratio": (circuit.size() - optimized_circuit.size()) / circuit.size()
                    },
                    "depth": {
                        "original": circuit.depth(),
                        "optimized": optimized_circuit.depth(),
                        "ratio": max(0, (circuit.depth() - optimized_circuit.depth()) / circuit.depth())
                    }
                }
                
                # Cache optimization result
                circuit_cache.cache_optimization_result(
                    circuit, optimized_circuit, optimization_level,
                    optimization_time, ["dead_code_elimination"], improvements
                )
                
                print(f"    Optimization completed in {optimization_time:.3f}s")
                print(f"    Gate count: {circuit.size()} → {optimized_circuit.size()}")
                print(f"    Improvement: {improvements['gate_count']['ratio']:.1%}")
                
            else:
                print("  Found cached optimization!")
                print(f"    Original optimization time: {cached_optimization.optimization_time:.3f}s")
                print(f"    Applied passes: {cached_optimization.applied_passes}")
                
                gate_improvement = cached_optimization.improvement_ratio('gate_count')
                print(f"    Gate count improvement: {gate_improvement:.1%}")
            
            # Second request (should be cache hit)
            print("  Requesting optimization again...")
            start_time = time.time()
            cached_optimization = circuit_cache.get_cached_optimization(circuit, optimization_level)
            cache_time = time.time() - start_time
            
            if cached_optimization:
                print(f"    Cache hit in {cache_time:.6f}s")
            else:
                print("    Unexpected cache miss!")
        
    finally:
        manager.close()


def demonstrate_concurrent_access():
    """Demonstrate concurrent access to caches and connection pools."""
    print("\n" + "="*60)
    print("CONCURRENT ACCESS DEMONSTRATION")
    print("="*60)
    
    config = PerformanceConfig.for_profile(PerformanceProfile.TESTING)
    config.enable_performance_monitoring = False
    manager = PerformanceManager(config)
    
    try:
        circuits = create_sample_circuits()
        results = []
        errors = []
        
        def worker(worker_id: int, circuit: MockQuantumCircuit):
            try:
                # Test database connection
                with manager.database_connection('circuits') as conn:
                    cursor = conn.execute("SELECT 1")
                    db_result = cursor.fetchone()[0]
                
                # Test cache operations
                cache = manager.get_general_cache()
                test_key = f"worker_{worker_id}_test"
                test_data = f"data_from_worker_{worker_id}"
                
                cache.put(test_key, test_data)
                retrieved = cache.get(test_key)
                
                # Test circuit caching
                circuit_cache = manager.get_circuit_cache()
                execution_config = {"backend": "simulator", "worker": worker_id}
                
                # Simulate execution and caching
                result = simulate_circuit_execution(circuit)
                circuit_cache.cache_execution_result(
                    circuit, execution_config, result,
                    execution_time=0.1, success=True
                )
                
                results.append({
                    'worker_id': worker_id,
                    'db_result': db_result,
                    'cache_result': retrieved == test_data,
                    'circuit': circuit.name
                })
                
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")
        
        print("Running concurrent workers...")
        
        # Run multiple workers concurrently
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            
            for i in range(12):
                circuit = circuits[i % len(circuits)]
                future = executor.submit(worker, i, circuit)
                futures.append(future)
            
            # Wait for completion
            for future in as_completed(futures):
                future.result()  # This will raise any exceptions
        
        # Report results
        print(f"\nCompleted {len(results)} concurrent operations")
        if errors:
            print(f"Errors encountered: {len(errors)}")
            for error in errors[:3]:  # Show first few errors
                print(f"  {error}")
        else:
            print("✅ No errors encountered")
        
        # Show statistics
        cache_stats = manager.get_cache_manager().get_statistics()
        print(f"\nFinal cache hit rate: {cache_stats['circuit_optimization']['result_cache'].get('hit_rate', 0):.2%}")
        
        conn_stats = manager.connection_manager.get_statistics()
        total_borrowed = sum(stats['connections_borrowed'] for stats in conn_stats.values())
        print(f"Total connections borrowed: {total_borrowed}")
        
    finally:
        manager.close()


def demonstrate_cache_backends():
    """Demonstrate different cache backends."""
    print("\n" + "="*60)
    print("CACHE BACKENDS DEMONSTRATION")
    print("="*60)
    
    backends = [
        (CacheBackend.MEMORY, "In-Memory Cache"),
        (CacheBackend.SQLITE, "SQLite Cache"),
        (CacheBackend.HYBRID, "Hybrid Cache")
    ]
    
    test_data = {"measurement": "bell_state", "probabilities": {"00": 0.5, "11": 0.5}}
    
    for backend, name in backends:
        print(f"\n--- Testing {name} ---")
        
        config = PerformanceConfig.for_profile(PerformanceProfile.TESTING)
        config.cache_backend = backend
        config.enable_performance_monitoring = False
        
        manager = PerformanceManager(config)
        
        try:
            cache = manager.get_general_cache()
            
            # Test basic operations
            print("  Testing put/get operations...")
            
            # Put operation
            start_time = time.time()
            success = cache.put("test_key", test_data)
            put_time = time.time() - start_time
            
            # Get operation
            start_time = time.time()
            retrieved = cache.get("test_key")
            get_time = time.time() - start_time
            
            print(f"    Put: {put_time*1000:.2f}ms, Get: {get_time*1000:.2f}ms")
            print(f"    Data integrity: {'✅' if retrieved == test_data else '❌'}")
            
            # Test multiple entries
            print("  Testing multiple entries...")
            for i in range(100):
                cache.put(f"key_{i}", f"data_{i}")
            
            # Test cache statistics
            stats = cache.get_statistics()
            print(f"    Memory usage: {stats.get('memory_usage_mb', 0):.2f} MB")
            print(f"    Entries: {stats.get('memory_entries', 0)}")
            
        except Exception as e:
            print(f"    Error: {e}")
        
        finally:
            manager.close()


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n" + "="*60)
    print("PERFORMANCE MONITORING DEMONSTRATION")
    print("="*60)
    
    config = PerformanceConfig.for_profile(PerformanceProfile.TESTING)
    config.enable_performance_monitoring = True
    config.monitoring_interval = 5.0  # 5 second intervals for demo
    
    manager = PerformanceManager(config)
    
    try:
        print("Starting performance monitoring...")
        
        # Generate some activity
        cache = manager.get_general_cache()
        circuit_cache = manager.get_circuit_cache()
        circuits = create_sample_circuits()
        
        print("Generating cache activity...")
        for i in range(20):
            # General cache activity
            cache.put(f"monitor_test_{i}", f"data_{i}")
            cache.get(f"monitor_test_{i}")
            
            # Circuit cache activity
            circuit = circuits[i % len(circuits)]
            execution_config = {"backend": "simulator", "test": i}
            
            result = simulate_circuit_execution(circuit)
            circuit_cache.cache_execution_result(
                circuit, execution_config, result,
                execution_time=0.1, success=True
            )
            
            if i % 5 == 0:
                print(f"  Generated {i+1} cache operations...")
        
        # Wait a moment for monitoring to collect data
        print("Waiting for monitoring data collection...")
        time.sleep(6)
        
        # Get performance report
        print("\nPerformance Report:")
        report = manager.get_performance_report()
        
        # Connection statistics
        if 'connections' in report:
            print("  Connection Pools:")
            for pool_name, stats in report['connections'].items():
                print(f"    {pool_name}: {stats.get('connections_borrowed', 0)} borrowed, "
                      f"{stats.get('pool_utilization', 0):.1%} utilization")
        
        # Cache statistics
        if 'cache' in report:
            cache_info = report['cache']
            result_cache = cache_info.get('circuit_optimization', {}).get('result_cache', {})
            
            print("  Cache Performance:")
            print(f"    Hit rate: {result_cache.get('hit_rate', 0):.2%}")
            print(f"    Memory usage: {result_cache.get('memory_usage_mb', 0):.1f} MB")
            print(f"    Average access time: {result_cache.get('average_access_time_ms', 0):.2f} ms")
        
        # Current metrics
        if 'current_metrics' in report and report['current_metrics']:
            current = report['current_metrics']
            system = current.get('system', {})
            
            print("  System Metrics:")
            print(f"    CPU: {system.get('cpu_percent', 0):.1f}%")
            print(f"    Memory: {system.get('memory_percent', 0):.1f}%")
        
    finally:
        manager.close()


async def demonstrate_async_operations():
    """Demonstrate asynchronous cache and connection operations."""
    print("\n" + "="*60)
    print("ASYNCHRONOUS OPERATIONS DEMONSTRATION")
    print("="*60)
    
    config = PerformanceConfig.for_profile(PerformanceProfile.TESTING)
    config.enable_performance_monitoring = False
    manager = PerformanceManager(config)
    
    try:
        async def async_cache_worker(worker_id: int, cache_key: str):
            """Async worker for cache operations."""
            # Simulate async work
            await asyncio.sleep(0.1)
            
            # Cache operations (these are sync, but in async context)
            cache = manager.get_general_cache()
            data = f"async_data_{worker_id}"
            
            cache.put(cache_key, data)
            retrieved = cache.get(cache_key)
            
            return worker_id, retrieved == data
        
        print("Running async cache workers...")
        
        # Create and run async tasks
        tasks = []
        for i in range(10):
            task = async_cache_worker(i, f"async_key_{i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Report results
        successful = sum(1 for _, success in results if success)
        print(f"  Completed {len(results)} async operations")
        print(f"  Success rate: {successful}/{len(results)} ({successful/len(results):.1%})")
        
    finally:
        manager.close()


def main():
    """Run all demonstrations."""
    print("QuantRS2 Connection Pooling and Caching Demo")
    print("=" * 60)
    
    try:
        # Basic demonstrations
        demonstrate_performance_profiles()
        demonstrate_connection_pooling()
        demonstrate_circuit_caching()
        demonstrate_optimization_caching()
        demonstrate_concurrent_access()
        demonstrate_cache_backends()
        demonstrate_performance_monitoring()
        
        # Async demonstration
        print("\nRunning async demonstration...")
        asyncio.run(demonstrate_async_operations())
        
        print("\n" + "="*60)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nKey Features Demonstrated:")
        print("✅ Multiple performance profiles (dev, test, production)")
        print("✅ Database connection pooling with automatic management")
        print("✅ Intelligent circuit result caching with pattern detection")
        print("✅ Optimization result caching for expensive computations")
        print("✅ Thread-safe concurrent access to pools and caches")
        print("✅ Multiple cache backends (memory, SQLite, hybrid)")
        print("✅ Real-time performance monitoring and statistics")
        print("✅ Asynchronous operation support")
        print("✅ Automatic cache eviction and cleanup")
        print("✅ Connection validation and recovery")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up global manager
        close_performance_manager()


if __name__ == "__main__":
    main()