#!/usr/bin/env python3
"""
Advanced Dynamic Qubit Allocation Demo for QuantRS2

This example demonstrates the advanced dynamic qubit allocation capabilities,
including runtime expansion, intelligent resource management, and optimization.
"""

import time
import numpy as np
from typing import List, Dict, Any

try:
    from quantrs2.dynamic_allocation import (
        QubitAllocator,
        DynamicCircuit,
        AllocationStrategy,
        create_dynamic_circuit,
        configure_allocation_strategy,
        get_global_allocation_stats
    )
except ImportError:
    print("Dynamic allocation module not available. Please check installation.")
    exit(1)


def demo_basic_dynamic_allocation():
    """Demonstrate basic dynamic allocation features."""
    print("=== Basic Dynamic Allocation Demo ===")
    
    # Create a custom allocator
    allocator = QubitAllocator(
        max_qubits=128,
        initial_pool_size=16,
        strategy=AllocationStrategy.COMPACT,
        enable_gc=True
    )
    
    print(f"Initial stats: {allocator.get_allocation_stats()}")
    
    # Create a dynamic circuit
    circuit = DynamicCircuit(initial_qubits=4, allocator=allocator, auto_expand=True)
    print(f"Created circuit with {circuit.get_qubit_count()} qubits")
    
    # Build a growing circuit
    circuit.h(0)
    circuit.cnot(0, 1)
    
    # Auto-expand by accessing higher qubit indices
    circuit.h(5)  # This will auto-allocate qubits 2, 3, 4, 5
    circuit.cnot(2, 5)
    
    print(f"After auto-expansion: {circuit.get_qubit_count()} qubits")
    print(f"Operations: {len(circuit.operations)}")
    
    # Manual expansion
    new_qubits = circuit.allocate_qubits(3, contiguous=True)
    print(f"Manually allocated qubits: {new_qubits}")
    print(f"Final qubit count: {circuit.get_qubit_count()}")
    
    # Show allocation information
    info = circuit.get_allocation_info()
    print(f"Allocation info: {info}")
    
    print(f"Final allocator stats: {allocator.get_allocation_stats()}")
    print()


def demo_allocation_strategies():
    """Demonstrate different allocation strategies."""
    print("=== Allocation Strategies Demo ===")
    
    strategies = [
        AllocationStrategy.COMPACT,
        AllocationStrategy.BALANCED,
        AllocationStrategy.OPTIMAL
    ]
    
    for strategy in strategies:
        print(f"\nTesting {strategy.value} strategy:")
        
        allocator = QubitAllocator(
            max_qubits=32,
            initial_pool_size=16,
            strategy=strategy
        )
        
        # Allocate qubits and observe pattern
        qubits1 = allocator.allocate_qubits(4)
        qubits2 = allocator.allocate_qubits(4)
        
        print(f"  First allocation: {qubits1}")
        print(f"  Second allocation: {qubits2}")
        
        # Deallocate some and allocate again
        allocator.deallocate_qubits(qubits1[::2])  # Deallocate every other
        qubits3 = allocator.allocate_qubits(3)
        print(f"  After fragmentation: {qubits3}")


def demo_resource_management():
    """Demonstrate advanced resource management."""
    print("=== Resource Management Demo ===")
    
    allocator = QubitAllocator(
        max_qubits=64,
        initial_pool_size=16,
        enable_gc=True,
        gc_threshold=0.7
    )
    
    circuits = []
    
    # Create multiple circuits
    for i in range(5):
        circuit = DynamicCircuit(
            initial_qubits=4 + i * 2,
            allocator=allocator,
            auto_deallocate=True
        )
        circuits.append(circuit)
        
        # Build some operations
        for j in range(circuit.get_qubit_count() - 1):
            circuit.h(j)
            circuit.cnot(j, j + 1)
        
        print(f"Circuit {i}: {circuit.get_qubit_count()} qubits, "
              f"{len(circuit.operations)} operations")
    
    print(f"Total allocation: {allocator.get_allocation_stats()}")
    
    # Demonstrate reservation system
    reserved = allocator.reserve_qubits(6, circuit_id=999)
    print(f"Reserved qubits: {reserved}")
    
    # Promote some reserved qubits
    promoted = allocator.promote_reserved_qubits(reserved[:3])
    print(f"Promoted qubits: {promoted}")
    
    # Clean up some circuits
    del circuits[1]
    del circuits[2]  # Now index 2 is the old index 3
    
    # Force garbage collection
    allocator._garbage_collect()
    print(f"After cleanup: {allocator.get_allocation_stats()}")


def demo_performance_optimization():
    """Demonstrate performance optimization features."""
    print("=== Performance Optimization Demo ===")
    
    # Create a larger allocator for performance testing
    allocator = QubitAllocator(
        max_qubits=256,
        initial_pool_size=32,
        strategy=AllocationStrategy.COMPACT
    )
    
    # Timing test: rapid allocation/deallocation
    start_time = time.time()
    allocated_sets = []
    
    for i in range(20):
        qubits = allocator.allocate_qubits(8)
        allocated_sets.append(qubits)
        
        # Occasionally deallocate some
        if i % 4 == 3:
            to_deallocate = allocated_sets[::2]  # Every other set
            for qubit_set in to_deallocate:
                allocator.deallocate_qubits(qubit_set)
            allocated_sets = allocated_sets[1::2]  # Keep remaining
    
    allocation_time = time.time() - start_time
    print(f"Rapid allocation/deallocation took: {allocation_time:.4f}s")
    
    # Test optimization
    start_time = time.time()
    optimization_result = allocator.optimize_allocation()
    optimization_time = time.time() - start_time
    
    print(f"Optimization took: {optimization_time:.4f}s")
    print(f"Optimization result: {optimization_result}")
    
    # Final stats
    final_stats = allocator.get_allocation_stats()
    print(f"Final stats: {final_stats}")
    print(f"Utilization: {final_stats['utilization']:.2%}")


def demo_advanced_circuit_patterns():
    """Demonstrate advanced circuit construction patterns."""
    print("=== Advanced Circuit Patterns Demo ===")
    
    allocator = QubitAllocator(max_qubits=64, initial_pool_size=16)
    
    # Pattern 1: Gradually expanding quantum neural network
    print("\n1. Expanding Quantum Neural Network:")
    qnn_circuit = DynamicCircuit(initial_qubits=4, allocator=allocator)
    
    # Layer 1: Small network
    for i in range(4):
        qnn_circuit.ry(i, np.pi/4)
    for i in range(3):
        qnn_circuit.cnot(i, i + 1)
    
    # Expand for Layer 2
    layer2_qubits = qnn_circuit.allocate_qubits(4)
    print(f"  Layer 2 qubits: {layer2_qubits}")
    
    for i in layer2_qubits:
        qnn_circuit.ry(i, np.pi/6)
    
    # Cross-layer connections
    for i in range(4):
        qnn_circuit.cnot(i, layer2_qubits[i])
    
    print(f"  QNN final size: {qnn_circuit.get_qubit_count()} qubits")
    
    # Pattern 2: Dynamic quantum error correction
    print("\n2. Dynamic Quantum Error Correction:")
    qec_circuit = DynamicCircuit(initial_qubits=1, allocator=allocator)  # Start with 1 logical qubit
    
    # Add syndrome qubits as needed
    syndrome_qubits = qec_circuit.allocate_qubits(3)  # For 3-qubit repetition code
    
    # Encoding
    qec_circuit.cnot(0, syndrome_qubits[0])
    qec_circuit.cnot(0, syndrome_qubits[1])
    
    # Add more ancilla qubits for error detection
    ancilla_qubits = qec_circuit.allocate_qubits(2)
    qec_circuit.cnot(0, ancilla_qubits[0])
    qec_circuit.cnot(syndrome_qubits[0], ancilla_qubits[0])
    
    print(f"  QEC circuit size: {qec_circuit.get_qubit_count()} qubits")
    
    # Pattern 3: Adaptive quantum algorithm
    print("\n3. Adaptive Quantum Algorithm:")
    adaptive_circuit = DynamicCircuit(initial_qubits=3, allocator=allocator)
    
    # Initial state preparation
    adaptive_circuit.h(0)
    adaptive_circuit.cnot(0, 1)
    
    # Simulate adaptive behavior based on "measurement results"
    measurement_result = np.random.choice([0, 1])
    
    if measurement_result == 1:
        # Expand circuit for additional processing
        extra_qubits = adaptive_circuit.allocate_qubits(2)
        print(f"  Adaptive expansion: added qubits {extra_qubits}")
        
        adaptive_circuit.h(extra_qubits[0])
        adaptive_circuit.cnot(1, extra_qubits[0])
        adaptive_circuit.cnot(extra_qubits[0], extra_qubits[1])
    else:
        # Deallocate unused qubit
        adaptive_circuit.deallocate_qubits([2])
        print(f"  Adaptive reduction: deallocated qubit 2")
    
    print(f"  Adaptive circuit final size: {adaptive_circuit.get_qubit_count()} qubits")


def demo_memory_efficiency():
    """Demonstrate memory efficiency features."""
    print("=== Memory Efficiency Demo ===")
    
    allocator = QubitAllocator(
        max_qubits=128,
        initial_pool_size=16,
        enable_gc=True,
        gc_threshold=0.6
    )
    
    print("Creating many short-lived circuits...")
    
    start_stats = allocator.get_allocation_stats()
    
    # Create many circuits with automatic cleanup
    for iteration in range(10):
        temp_circuits = []
        
        # Create several circuits
        for i in range(5):
            circuit = DynamicCircuit(
                initial_qubits=4 + i,
                allocator=allocator,
                auto_deallocate=True
            )
            
            # Add some operations
            for j in range(circuit.get_qubit_count()):
                circuit.h(j)
                if j > 0:
                    circuit.cnot(j-1, j)
            
            temp_circuits.append(circuit)
        
        # Let circuits go out of scope (automatic cleanup)
        del temp_circuits
        
        # Periodic garbage collection
        if iteration % 3 == 2:
            allocator._garbage_collect()
            
        current_stats = allocator.get_allocation_stats()
        print(f"  Iteration {iteration}: "
              f"Allocated: {current_stats['allocated_qubits']}, "
              f"Utilization: {current_stats['utilization']:.2%}")
    
    end_stats = allocator.get_allocation_stats()
    
    print(f"\nMemory efficiency results:")
    print(f"  Start: {start_stats['allocated_qubits']} allocated")
    print(f"  End: {end_stats['allocated_qubits']} allocated")
    print(f"  GC runs: {end_stats['gc_count']}")
    print(f"  Total allocations: {end_stats['allocation_count']}")


def demo_thread_safety():
    """Demonstrate thread-safe allocation."""
    print("=== Thread Safety Demo ===")
    
    import threading
    
    allocator = QubitAllocator(max_qubits=64, initial_pool_size=32)
    results = {"allocations": [], "errors": []}
    lock = threading.Lock()
    
    def worker_thread(worker_id: int):
        """Worker thread that allocates and deallocates qubits."""
        try:
            for i in range(5):
                # Allocate some qubits
                qubits = allocator.allocate_qubits(3)
                
                time.sleep(0.001)  # Simulate work
                
                # Deallocate qubits
                deallocated = allocator.deallocate_qubits(qubits)
                
                with lock:
                    results["allocations"].append((worker_id, i, qubits, deallocated))
                    
        except Exception as e:
            with lock:
                results["errors"].append((worker_id, str(e)))
    
    # Start multiple worker threads
    threads = []
    for i in range(8):
        thread = threading.Thread(target=worker_thread, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print(f"Thread safety test completed:")
    print(f"  Total allocations: {len(results['allocations'])}")
    print(f"  Errors: {len(results['errors'])}")
    
    if results["errors"]:
        print("  Errors encountered:")
        for worker_id, error in results["errors"]:
            print(f"    Worker {worker_id}: {error}")
    else:
        print("  ✓ No thread safety issues detected")
    
    final_stats = allocator.get_allocation_stats()
    print(f"  Final allocation state: {final_stats['allocated_qubits']} qubits allocated")


def main():
    """Run all demonstrations."""
    print("QuantRS2 Advanced Dynamic Qubit Allocation Demo")
    print("=" * 50)
    
    try:
        demo_basic_dynamic_allocation()
        demo_allocation_strategies()
        demo_resource_management()
        demo_performance_optimization()
        demo_advanced_circuit_patterns()
        demo_memory_efficiency()
        demo_thread_safety()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        
        # Show global stats
        global_stats = get_global_allocation_stats()
        print(f"Global allocation stats: {global_stats}")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()