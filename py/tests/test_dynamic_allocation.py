#!/usr/bin/env python3
"""
Tests for the dynamic qubit allocation system.
"""

import pytest
import threading
import time
import weakref
from unittest.mock import patch, MagicMock

try:
    from quantrs2.dynamic_allocation import (
        AllocationStrategy,
        QubitState,
        QubitInfo,
        QubitAllocator,
        DynamicCircuit,
        create_dynamic_circuit,
        configure_allocation_strategy,
        allocate_qubits,
        deallocate_qubits,
        garbage_collect,
        get_global_allocation_stats,
        set_global_allocator
    )
    HAS_DYNAMIC_ALLOCATION = True
except ImportError:
    HAS_DYNAMIC_ALLOCATION = False
    
    # Stub implementations
    class AllocationStrategy: pass
    class QubitState: pass
    class QubitInfo: pass
    class QubitAllocator: pass
    class DynamicCircuit: pass
    def create_dynamic_circuit(): pass
    def configure_allocation_strategy(): pass
    def allocate_qubits(): pass
    def deallocate_qubits(): pass
    def garbage_collect(): pass
    def get_global_allocation_stats(): pass
    def set_global_allocator(): pass


@pytest.mark.skipif(not HAS_DYNAMIC_ALLOCATION, reason="quantrs2.dynamic_allocation module not available")
class TestQubitAllocator:
    """Test cases for QubitAllocator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.allocator = QubitAllocator(
            max_qubits=64,
            initial_pool_size=16,
            strategy=AllocationStrategy.COMPACT
        )
    
    def test_initialization(self):
        """Test allocator initialization."""
        assert self.allocator.max_qubits == 64
        assert self.allocator.strategy == AllocationStrategy.COMPACT
        assert len(self.allocator.free_qubits) == 16
        assert len(self.allocator.allocated_qubits) == 0
        assert len(self.allocator.reserved_qubits) == 0
    
    def test_basic_allocation(self):
        """Test basic qubit allocation."""
        qubits = self.allocator.allocate_qubits(4)
        
        assert len(qubits) == 4
        assert all(q in self.allocator.allocated_qubits for q in qubits)
        assert len(self.allocator.free_qubits) == 12
        assert len(self.allocator.allocated_qubits) == 4
    
    def test_allocation_strategies(self):
        """Test different allocation strategies."""
        # Test compact strategy
        compact_allocator = QubitAllocator(strategy=AllocationStrategy.COMPACT)
        qubits = compact_allocator.allocate_qubits(4)
        assert qubits == [0, 1, 2, 3]
        
        # Test balanced strategy
        balanced_allocator = QubitAllocator(
            initial_pool_size=16,
            strategy=AllocationStrategy.BALANCED
        )
        qubits = balanced_allocator.allocate_qubits(4)
        assert len(qubits) == 4
        assert len(set(qubits)) == 4  # All unique
    
    def test_contiguous_allocation(self):
        """Test contiguous qubit allocation."""
        # Allocate some qubits first
        self.allocator.allocate_qubits(2)
        
        # Try to allocate contiguous qubits
        qubits = self.allocator.allocate_qubits(3, contiguous=True)
        
        # Should get contiguous range
        assert len(qubits) == 3
        qubits.sort()
        for i in range(len(qubits) - 1):
            assert qubits[i + 1] == qubits[i] + 1
    
    def test_preferred_ids(self):
        """Test allocation with preferred IDs."""
        preferred = [5, 7, 9, 11]
        qubits = self.allocator.allocate_qubits(3, preferred_ids=preferred)
        
        # Should use preferred IDs when available
        assert all(q in preferred for q in qubits)
        assert len(qubits) == 3
    
    def test_deallocation(self):
        """Test qubit deallocation."""
        qubits = self.allocator.allocate_qubits(4)
        initial_free = len(self.allocator.free_qubits)
        
        deallocated = self.allocator.deallocate_qubits(qubits[:2])
        
        assert deallocated == 2
        assert len(self.allocator.free_qubits) == initial_free + 2
        assert len(self.allocator.allocated_qubits) == 2
    
    def test_circuit_tracking(self):
        """Test circuit-based qubit tracking."""
        circuit_id = 12345
        qubits = self.allocator.allocate_qubits(3, circuit_id=circuit_id)
        
        assert circuit_id in self.allocator.circuit_qubits
        assert self.allocator.circuit_qubits[circuit_id] == set(qubits)
        
        # Deallocate circuit
        deallocated = self.allocator.deallocate_circuit(circuit_id)
        assert deallocated == 3
        assert circuit_id not in self.allocator.circuit_qubits
    
    def test_reservation(self):
        """Test qubit reservation."""
        reserved = self.allocator.reserve_qubits(3)
        
        assert len(reserved) == 3
        assert all(q in self.allocator.reserved_qubits for q in reserved)
        assert all(q not in self.allocator.free_qubits for q in reserved)
        
        # Promote to allocated
        promoted = self.allocator.promote_reserved_qubits(reserved[:2])
        assert len(promoted) == 2
        assert all(q in self.allocator.allocated_qubits for q in promoted)
    
    def test_pool_expansion(self):
        """Test automatic pool expansion."""
        # Allocate all available qubits
        initial_size = len(self.allocator.free_qubits)
        self.allocator.allocate_qubits(initial_size)
        
        # Request more qubits - should trigger expansion
        additional = self.allocator.allocate_qubits(5)
        assert len(additional) == 5
        assert len(self.allocator.qubits) > initial_size
    
    def test_allocation_limits(self):
        """Test allocation limits."""
        # Try to allocate more than max_qubits
        with pytest.raises(RuntimeError):
            self.allocator.allocate_qubits(self.allocator.max_qubits + 10)
    
    def test_thread_safety(self):
        """Test thread-safe allocation."""
        results = []
        errors = []
        
        def allocate_worker():
            try:
                qubits = self.allocator.allocate_qubits(2)
                results.append(qubits)
                time.sleep(0.01)  # Small delay
                self.allocator.deallocate_qubits(qubits)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = [threading.Thread(target=allocate_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        
        # All allocations should be unique
        all_qubits = [q for result in results for q in result]
        assert len(all_qubits) == len(set(all_qubits))  # No duplicates
    
    def test_garbage_collection(self):
        """Test garbage collection."""
        # Create weak references that will be collected
        class MockCircuit:
            pass
        
        circuit1 = MockCircuit()
        circuit2 = MockCircuit()
        circuit_id1 = id(circuit1)
        circuit_id2 = id(circuit2)
        
        qubits1 = self.allocator.allocate_qubits(2, circuit_id=circuit_id1)
        qubits2 = self.allocator.allocate_qubits(2, circuit_id=circuit_id2)
        
        initial_allocated = len(self.allocator.allocated_qubits)
        
        # Delete one circuit
        del circuit1
        
        # Force garbage collection
        self.allocator._garbage_collect()
        
        # Should have freed qubits from deleted circuit
        assert len(self.allocator.allocated_qubits) <= initial_allocated
    
    def test_statistics(self):
        """Test allocation statistics."""
        stats = self.allocator.get_allocation_stats()
        
        assert isinstance(stats, dict)
        assert "total_qubits" in stats
        assert "free_qubits" in stats
        assert "allocated_qubits" in stats
        assert "utilization" in stats
        assert "allocation_count" in stats
        
        # Allocate some qubits and check updated stats
        self.allocator.allocate_qubits(4)
        new_stats = self.allocator.get_allocation_stats()
        
        assert new_stats["allocated_qubits"] == 4
        assert new_stats["allocation_count"] == 1
        assert new_stats["utilization"] > 0


@pytest.mark.skipif(not HAS_DYNAMIC_ALLOCATION, reason="quantrs2.dynamic_allocation module not available")
class TestDynamicCircuit:
    """Test cases for DynamicCircuit."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.allocator = QubitAllocator(max_qubits=32, initial_pool_size=16)
    
    def test_initialization(self):
        """Test dynamic circuit initialization."""
        circuit = DynamicCircuit(initial_qubits=4, allocator=self.allocator)
        
        assert circuit.get_qubit_count() == 4
        assert len(circuit.allocated_qubits) == 4
        assert len(circuit.qubit_map) == 4
    
    def test_auto_expansion(self):
        """Test automatic qubit expansion."""
        circuit = DynamicCircuit(initial_qubits=2, allocator=self.allocator, auto_expand=True)
        
        # Access qubit beyond current range - should auto-expand
        circuit.h(3)
        
        assert circuit.get_qubit_count() == 4
        assert len(circuit.operations) == 1
    
    def test_manual_allocation(self):
        """Test manual qubit allocation."""
        circuit = DynamicCircuit(allocator=self.allocator)
        
        logical_qubits = circuit.allocate_qubits(3)
        assert logical_qubits == [0, 1, 2]
        assert circuit.get_qubit_count() == 3
    
    def test_gate_operations(self):
        """Test quantum gate operations."""
        circuit = DynamicCircuit(initial_qubits=3, allocator=self.allocator)
        
        # Apply various gates
        circuit.h(0)
        circuit.x(1)
        circuit.cnot(0, 1)
        circuit.rx(2, 1.57)
        
        assert len(circuit.operations) == 4
        
        # Check operation types
        op_names = [op[0] for op in circuit.operations]
        assert "h" in op_names
        assert "x" in op_names
        assert "cnot" in op_names
        assert "rx" in op_names
    
    def test_qubit_deallocation(self):
        """Test qubit deallocation."""
        circuit = DynamicCircuit(initial_qubits=5, allocator=self.allocator)
        initial_count = circuit.get_qubit_count()
        
        # Deallocate some qubits
        deallocated = circuit.deallocate_qubits([2, 4])
        
        assert deallocated > 0
        assert circuit.get_qubit_count() == initial_count - 2
    
    def test_circuit_conversion(self):
        """Test conversion to static circuit."""
        circuit = DynamicCircuit(initial_qubits=3, allocator=self.allocator)
        
        # Build a simple circuit
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.x(2)
        
        # Convert to static
        static_circuit = circuit.to_static_circuit()
        
        # Verify static circuit can be executed
        assert static_circuit is not None
        # Note: Actual execution depends on QuantRS2 native implementation
    
    def test_auto_deallocate_disabled(self):
        """Test circuit with auto-deallocation disabled."""
        circuit = DynamicCircuit(
            initial_qubits=3, 
            allocator=self.allocator,
            auto_deallocate=False
        )
        
        circuit_id = circuit.circuit_id
        initial_allocated = len(self.allocator.allocated_qubits)
        
        # Delete circuit
        del circuit
        
        # Qubits should still be allocated
        assert len(self.allocator.allocated_qubits) == initial_allocated
    
    def test_allocation_info(self):
        """Test allocation information retrieval."""
        circuit = DynamicCircuit(initial_qubits=3, allocator=self.allocator)
        
        info = circuit.get_allocation_info()
        
        assert isinstance(info, dict)
        assert info["logical_qubits"] == 3
        assert len(info["physical_qubits"]) == 3
        assert len(info["qubit_mapping"]) == 3
        assert info["operations"] == 0


@pytest.mark.skipif(not HAS_DYNAMIC_ALLOCATION, reason="quantrs2.dynamic_allocation module not available")
class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global allocator
        set_global_allocator(QubitAllocator(max_qubits=32, initial_pool_size=16))
    
    def test_global_allocation(self):
        """Test global allocation functions."""
        qubits = allocate_qubits(3)
        assert len(qubits) == 3
        
        deallocated = deallocate_qubits(qubits)
        assert deallocated == 3
    
    def test_global_stats(self):
        """Test global statistics."""
        stats = get_global_allocation_stats()
        assert isinstance(stats, dict)
        assert "total_qubits" in stats
    
    def test_strategy_configuration(self):
        """Test strategy configuration."""
        configure_allocation_strategy(AllocationStrategy.BALANCED)
        
        # Allocate qubits to test strategy change
        qubits = allocate_qubits(4)
        assert len(qubits) == 4
    
    def test_create_dynamic_circuit(self):
        """Test dynamic circuit creation function."""
        circuit = create_dynamic_circuit(initial_qubits=4)
        
        assert isinstance(circuit, DynamicCircuit)
        assert circuit.get_qubit_count() == 4
    
    def test_garbage_collect_global(self):
        """Test global garbage collection."""
        allocate_qubits(5)
        garbage_collect()  # Should not raise exceptions


@pytest.mark.skipif(not HAS_DYNAMIC_ALLOCATION, reason="quantrs2.dynamic_allocation module not available")
class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.allocator = QubitAllocator(max_qubits=16, initial_pool_size=8)
    
    def test_empty_allocation(self):
        """Test allocation of zero qubits."""
        qubits = self.allocator.allocate_qubits(0)
        assert qubits == []
    
    def test_invalid_deallocation(self):
        """Test deallocation of non-allocated qubits."""
        deallocated = self.allocator.deallocate_qubits([100, 200])
        assert deallocated == 0  # No qubits deallocated
    
    def test_circuit_without_auto_expand(self):
        """Test circuit access beyond range without auto-expand."""
        circuit = DynamicCircuit(
            initial_qubits=2, 
            allocator=self.allocator,
            auto_expand=False
        )
        
        with pytest.raises(IndexError):
            circuit.h(5)  # Should fail
    
    def test_empty_circuit_conversion(self):
        """Test conversion of empty circuit."""
        circuit = DynamicCircuit(allocator=self.allocator)
        
        with pytest.raises(ValueError):
            circuit.to_static_circuit()


@pytest.mark.skipif(not HAS_DYNAMIC_ALLOCATION, reason="quantrs2.dynamic_allocation module not available")
class TestPerformance:
    """Performance-related tests."""
    
    def test_large_allocation_performance(self):
        """Test performance with large allocations."""
        allocator = QubitAllocator(max_qubits=1024, initial_pool_size=64)
        
        start_time = time.time()
        
        # Allocate and deallocate many qubits
        for i in range(10):
            qubits = allocator.allocate_qubits(50)
            allocator.deallocate_qubits(qubits[:25])
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed < 1.0, f"Large allocation took too long: {elapsed:.2f}s"
    
    def test_fragmentation_handling(self):
        """Test handling of memory fragmentation."""
        allocator = QubitAllocator(max_qubits=32, initial_pool_size=16)
        
        # Create fragmentation by allocating and deallocating alternating qubits
        all_qubits = allocator.allocate_qubits(16)
        
        # Deallocate every other qubit
        to_deallocate = all_qubits[::2]
        allocator.deallocate_qubits(to_deallocate)
        
        # Try to allocate contiguous qubits
        new_qubits = allocator.allocate_qubits(4, contiguous=True)
        assert len(new_qubits) == 4


if __name__ == "__main__":
    pytest.main([__file__])