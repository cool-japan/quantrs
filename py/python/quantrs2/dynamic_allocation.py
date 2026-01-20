#!/usr/bin/env python3
"""
Dynamic Qubit Allocation System for QuantRS2

This module provides advanced dynamic qubit allocation capabilities that allow
circuits to grow and shrink at runtime, manage qubit registers, and optimize
resource usage.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import weakref
import warnings


class AllocationStrategy(Enum):
    """Strategy for allocating qubits."""
    COMPACT = "compact"          # Allocate qubits in compact, sequential order
    BALANCED = "balanced"        # Balance allocation across available ranges
    OPTIMAL = "optimal"          # Optimize for circuit topology and operations
    LAZY = "lazy"               # Allocate only when needed
    EAGER = "eager"             # Pre-allocate likely-needed qubits


class QubitState(Enum):
    """State of a qubit in the allocation system."""
    FREE = "free"               # Available for allocation
    ALLOCATED = "allocated"     # Currently in use
    RESERVED = "reserved"       # Reserved for future use
    ENTANGLED = "entangled"     # Entangled with other qubits
    MEASURED = "measured"       # Has been measured (may need reset)


@dataclass
class QubitInfo:
    """Information about a qubit in the system."""
    id: int
    state: QubitState = QubitState.FREE
    circuit_ref: Optional[weakref.ref] = None
    allocated_at: Optional[float] = None
    last_used: Optional[float] = None
    entangled_with: Set[int] = field(default_factory=set)
    operation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class QubitAllocator:
    """
    Advanced dynamic qubit allocator with intelligent resource management.
    
    Features:
    - Dynamic allocation and deallocation
    - Multiple allocation strategies
    - Automatic garbage collection
    - Thread-safe operations
    - Resource optimization
    - Circuit topology awareness
    """
    
    def __init__(self, 
                 max_qubits: int = 1024,
                 initial_pool_size: int = 32,
                 strategy: AllocationStrategy = AllocationStrategy.COMPACT,
                 enable_gc: bool = True,
                 gc_threshold: float = 0.8):
        """
        Initialize the qubit allocator.
        
        Args:
            max_qubits: Maximum number of qubits available
            initial_pool_size: Initial size of the qubit pool
            strategy: Allocation strategy to use
            enable_gc: Whether to enable automatic garbage collection
            gc_threshold: Threshold for triggering garbage collection (0.0-1.0)
        """
        self.max_qubits = max_qubits
        self.strategy = strategy
        self.enable_gc = enable_gc
        self.gc_threshold = gc_threshold
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Qubit pool management
        self.qubits: Dict[int, QubitInfo] = {}
        self.free_qubits: Set[int] = set()
        self.allocated_qubits: Set[int] = set()
        self.reserved_qubits: Set[int] = set()
        
        # Circuit tracking
        self.circuits: Dict[int, weakref.ref] = {}
        self.circuit_qubits: Dict[int, Set[int]] = {}
        
        # Statistics
        self.allocation_count = 0
        self.deallocation_count = 0
        self.gc_count = 0
        
        # Initialize the pool
        self._initialize_pool(initial_pool_size)
    
    def _initialize_pool(self, size: int):
        """Initialize the initial qubit pool."""
        with self._lock:
            for i in range(min(size, self.max_qubits)):
                qubit_info = QubitInfo(id=i)
                self.qubits[i] = qubit_info
                self.free_qubits.add(i)
    
    def allocate_qubits(self, 
                       count: int,
                       circuit_id: Optional[int] = None,
                       preferred_ids: Optional[List[int]] = None,
                       contiguous: bool = False) -> List[int]:
        """
        Allocate the specified number of qubits.
        
        Args:
            count: Number of qubits to allocate
            circuit_id: ID of the circuit requesting qubits
            preferred_ids: Preferred qubit IDs (if available)
            contiguous: Whether qubits should be contiguous
            
        Returns:
            List of allocated qubit IDs
            
        Raises:
            RuntimeError: If unable to allocate the requested qubits
        """
        with self._lock:
            if count <= 0:
                return []
            
            # Check if we have enough free qubits
            available = len(self.free_qubits)
            if available < count:
                # Try to expand the pool
                self._expand_pool(count - available)
                available = len(self.free_qubits)
                
                if available < count:
                    # Try garbage collection
                    if self.enable_gc:
                        self._garbage_collect()
                        available = len(self.free_qubits)
                    
                    if available < count:
                        raise RuntimeError(f"Cannot allocate {count} qubits. Only {available} available.")
            
            # Select qubits based on strategy
            selected_qubits = self._select_qubits(count, preferred_ids, contiguous)
            
            # Mark qubits as allocated
            current_time = self._current_time()
            for qubit_id in selected_qubits:
                qubit_info = self.qubits[qubit_id]
                qubit_info.state = QubitState.ALLOCATED
                qubit_info.allocated_at = current_time
                qubit_info.last_used = current_time
                
                if circuit_id is not None:
                    qubit_info.circuit_ref = weakref.ref(self._get_circuit_placeholder(circuit_id))
                    if circuit_id not in self.circuit_qubits:
                        self.circuit_qubits[circuit_id] = set()
                    self.circuit_qubits[circuit_id].add(qubit_id)
                
                self.free_qubits.discard(qubit_id)
                self.allocated_qubits.add(qubit_id)
            
            self.allocation_count += 1
            return selected_qubits
    
    def deallocate_qubits(self, qubit_ids: Union[int, List[int]]) -> int:
        """
        Deallocate the specified qubits.
        
        Args:
            qubit_ids: Qubit ID or list of qubit IDs to deallocate
            
        Returns:
            Number of qubits actually deallocated
        """
        if isinstance(qubit_ids, int):
            qubit_ids = [qubit_ids]
        
        with self._lock:
            deallocated = 0
            
            for qubit_id in qubit_ids:
                if qubit_id in self.allocated_qubits:
                    qubit_info = self.qubits[qubit_id]
                    
                    # Clear entanglement information
                    for entangled_id in qubit_info.entangled_with:
                        if entangled_id in self.qubits:
                            self.qubits[entangled_id].entangled_with.discard(qubit_id)
                    qubit_info.entangled_with.clear()
                    
                    # Reset qubit info
                    qubit_info.state = QubitState.FREE
                    qubit_info.circuit_ref = None
                    qubit_info.allocated_at = None
                    qubit_info.metadata.clear()
                    
                    # Update tracking
                    self.allocated_qubits.discard(qubit_id)
                    self.free_qubits.add(qubit_id)
                    
                    # Remove from circuit tracking
                    for circuit_id, qubits in self.circuit_qubits.items():
                        qubits.discard(qubit_id)
                    
                    deallocated += 1
            
            if deallocated > 0:
                self.deallocation_count += 1
            
            return deallocated
    
    def deallocate_circuit(self, circuit_id: int) -> int:
        """
        Deallocate all qubits associated with a circuit.
        
        Args:
            circuit_id: ID of the circuit
            
        Returns:
            Number of qubits deallocated
        """
        with self._lock:
            if circuit_id in self.circuit_qubits:
                qubit_ids = list(self.circuit_qubits[circuit_id])
                deallocated = self.deallocate_qubits(qubit_ids)
                del self.circuit_qubits[circuit_id]
                return deallocated
            return 0
    
    def reserve_qubits(self, count: int, circuit_id: Optional[int] = None) -> List[int]:
        """
        Reserve qubits for future allocation.
        
        Args:
            count: Number of qubits to reserve
            circuit_id: ID of the circuit requesting reservation
            
        Returns:
            List of reserved qubit IDs
        """
        with self._lock:
            if count <= 0:
                return []
            
            available = len(self.free_qubits)
            if available < count:
                self._expand_pool(count - available)
            
            selected_qubits = list(self.free_qubits)[:count]
            
            for qubit_id in selected_qubits:
                qubit_info = self.qubits[qubit_id]
                qubit_info.state = QubitState.RESERVED
                if circuit_id is not None:
                    qubit_info.circuit_ref = weakref.ref(self._get_circuit_placeholder(circuit_id))
                
                self.free_qubits.discard(qubit_id)
                self.reserved_qubits.add(qubit_id)
            
            return selected_qubits
    
    def promote_reserved_qubits(self, qubit_ids: List[int]) -> List[int]:
        """
        Promote reserved qubits to allocated status.
        
        Args:
            qubit_ids: List of reserved qubit IDs to promote
            
        Returns:
            List of successfully promoted qubit IDs
        """
        with self._lock:
            promoted = []
            current_time = self._current_time()
            
            for qubit_id in qubit_ids:
                if qubit_id in self.reserved_qubits:
                    qubit_info = self.qubits[qubit_id]
                    qubit_info.state = QubitState.ALLOCATED
                    qubit_info.allocated_at = current_time
                    qubit_info.last_used = current_time
                    
                    self.reserved_qubits.discard(qubit_id)
                    self.allocated_qubits.add(qubit_id)
                    promoted.append(qubit_id)
            
            return promoted
    
    def expand_allocation(self, additional_count: int, circuit_id: Optional[int] = None) -> List[int]:
        """
        Expand allocation for an existing circuit.
        
        Args:
            additional_count: Number of additional qubits needed
            circuit_id: ID of the circuit to expand
            
        Returns:
            List of newly allocated qubit IDs
        """
        return self.allocate_qubits(additional_count, circuit_id)
    
    def _select_qubits(self, 
                      count: int, 
                      preferred_ids: Optional[List[int]] = None,
                      contiguous: bool = False) -> List[int]:
        """Select qubits based on allocation strategy."""
        available = list(self.free_qubits)
        
        if preferred_ids:
            # Try to use preferred IDs first
            preferred_available = [qid for qid in preferred_ids if qid in available]
            if len(preferred_available) >= count:
                return preferred_available[:count]
        
        if contiguous:
            # Find contiguous range
            available.sort()
            for start_idx in range(len(available) - count + 1):
                candidate_range = available[start_idx:start_idx + count]
                if candidate_range == list(range(candidate_range[0], candidate_range[0] + count)):
                    return candidate_range
            
            # Fallback to non-contiguous if contiguous not available
            pass
        
        if self.strategy == AllocationStrategy.COMPACT:
            available.sort()
            return available[:count]
        elif self.strategy == AllocationStrategy.BALANCED:
            # Distribute across available range
            step = max(1, len(available) // count)
            return [available[i * step] for i in range(count)]
        else:  # OPTIMAL, LAZY, EAGER - use compact for now
            available.sort()
            return available[:count]
    
    def _expand_pool(self, additional_count: int):
        """Expand the qubit pool."""
        current_size = len(self.qubits)
        new_size = min(current_size + additional_count, self.max_qubits)
        
        for i in range(current_size, new_size):
            if i not in self.qubits:
                qubit_info = QubitInfo(id=i)
                self.qubits[i] = qubit_info
                self.free_qubits.add(i)
    
    def _garbage_collect(self):
        """Perform garbage collection on unused qubits."""
        with self._lock:
            # Find circuits that have been garbage collected
            dead_circuits = []
            for circuit_id, circuit_ref in list(self.circuits.items()):
                if circuit_ref() is None:
                    dead_circuits.append(circuit_id)
            
            # Deallocate qubits from dead circuits
            for circuit_id in dead_circuits:
                self.deallocate_circuit(circuit_id)
                del self.circuits[circuit_id]
            
            self.gc_count += 1
    
    def _current_time(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    def _get_circuit_placeholder(self, circuit_id: int) -> object:
        """Get or create a placeholder object for circuit tracking."""
        if circuit_id not in self.circuits:
            placeholder = object()
            self.circuits[circuit_id] = weakref.ref(placeholder)
            return placeholder
        return self.circuits[circuit_id]()
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get allocation statistics."""
        with self._lock:
            return {
                "total_qubits": len(self.qubits),
                "free_qubits": len(self.free_qubits),
                "allocated_qubits": len(self.allocated_qubits),
                "reserved_qubits": len(self.reserved_qubits),
                "utilization": len(self.allocated_qubits) / len(self.qubits) if self.qubits else 0.0,
                "allocation_count": self.allocation_count,
                "deallocation_count": self.deallocation_count,
                "gc_count": self.gc_count,
                "active_circuits": len([ref for ref in self.circuits.values() if ref() is not None])
            }
    
    def optimize_allocation(self) -> Dict[str, int]:
        """Optimize current allocations."""
        with self._lock:
            # Simple optimization: compact allocated qubits
            allocated_list = sorted(self.allocated_qubits)
            optimizations = 0
            
            # This is a simplified optimization - in practice, this would
            # involve more sophisticated algorithms considering circuit topology
            
            return {
                "moves_performed": optimizations,
                "memory_saved": 0,
                "fragmentation_reduced": 0
            }


class DynamicCircuit:
    """
    A quantum circuit with dynamic qubit allocation capabilities.
    
    This class automatically manages qubit allocation and deallocation,
    allowing circuits to grow and shrink as needed.
    """
    
    def __init__(self, 
                 initial_qubits: int = 0,
                 allocator: Optional[QubitAllocator] = None,
                 auto_expand: bool = True,
                 auto_deallocate: bool = True):
        """
        Initialize a dynamic circuit.
        
        Args:
            initial_qubits: Initial number of qubits to allocate
            allocator: Custom allocator to use (creates default if None)
            auto_expand: Whether to automatically expand when needed
            auto_deallocate: Whether to automatically deallocate on destruction
        """
        self.allocator = allocator or _get_global_allocator()
        self.auto_expand = auto_expand
        self.auto_deallocate = auto_deallocate
        
        # Circuit identification
        self.circuit_id = id(self)
        
        # Qubit management
        self.allocated_qubits: List[int] = []
        self.qubit_map: Dict[int, int] = {}  # logical -> physical mapping
        
        # Operation tracking
        self.operations: List[Tuple[str, List[int], Dict[str, Any]]] = []
        
        # Initialize with requested qubits
        if initial_qubits > 0:
            self.allocate_qubits(initial_qubits)
    
    def __del__(self):
        """Cleanup qubits when circuit is destroyed."""
        if self.auto_deallocate and hasattr(self, 'allocator'):
            try:
                self.deallocate_all_qubits()
            except:
                pass  # Ignore errors during cleanup
    
    def allocate_qubits(self, count: int, contiguous: bool = False) -> List[int]:
        """
        Allocate additional qubits to this circuit.
        
        Args:
            count: Number of qubits to allocate
            contiguous: Whether qubits should be contiguous
            
        Returns:
            List of logical qubit indices in this circuit
        """
        physical_qubits = self.allocator.allocate_qubits(
            count, 
            circuit_id=self.circuit_id,
            contiguous=contiguous
        )
        
        logical_indices = []
        for physical_id in physical_qubits:
            logical_id = len(self.allocated_qubits)
            self.allocated_qubits.append(physical_id)
            self.qubit_map[logical_id] = physical_id
            logical_indices.append(logical_id)
        
        return logical_indices
    
    def deallocate_qubits(self, logical_indices: List[int]) -> int:
        """
        Deallocate specific qubits from this circuit.
        
        Args:
            logical_indices: List of logical qubit indices to deallocate
            
        Returns:
            Number of qubits deallocated
        """
        physical_ids = []
        for logical_id in logical_indices:
            if logical_id in self.qubit_map:
                physical_ids.append(self.qubit_map[logical_id])
                del self.qubit_map[logical_id]
        
        # Remove from allocated list (keeping order)
        for logical_id in sorted(logical_indices, reverse=True):
            if logical_id < len(self.allocated_qubits):
                del self.allocated_qubits[logical_id]
        
        # Update mapping for remaining qubits
        new_map = {}
        for i, physical_id in enumerate(self.allocated_qubits):
            new_map[i] = physical_id
        self.qubit_map = new_map
        
        return self.allocator.deallocate_qubits(physical_ids)
    
    def deallocate_all_qubits(self) -> int:
        """Deallocate all qubits from this circuit."""
        return self.allocator.deallocate_circuit(self.circuit_id)
    
    def get_qubit_count(self) -> int:
        """Get the current number of qubits in the circuit."""
        return len(self.allocated_qubits)
    
    def _ensure_qubit_exists(self, logical_id: int):
        """Ensure a qubit exists, allocating if necessary and auto_expand is enabled."""
        if logical_id >= len(self.allocated_qubits):
            if self.auto_expand:
                needed = logical_id - len(self.allocated_qubits) + 1
                self.allocate_qubits(needed)
            else:
                raise IndexError(f"Qubit {logical_id} does not exist and auto_expand is disabled")
    
    def _get_physical_id(self, logical_id: int) -> int:
        """Get the physical qubit ID for a logical ID."""
        self._ensure_qubit_exists(logical_id)
        return self.qubit_map[logical_id]
    
    # Quantum gate operations
    def h(self, qubit: int):
        """Apply Hadamard gate."""
        physical_id = self._get_physical_id(qubit)
        self.operations.append(("h", [physical_id], {}))
        return self
    
    def x(self, qubit: int):
        """Apply Pauli-X gate."""
        physical_id = self._get_physical_id(qubit)
        self.operations.append(("x", [physical_id], {}))
        return self
    
    def y(self, qubit: int):
        """Apply Pauli-Y gate."""
        physical_id = self._get_physical_id(qubit)
        self.operations.append(("y", [physical_id], {}))
        return self
    
    def z(self, qubit: int):
        """Apply Pauli-Z gate."""
        physical_id = self._get_physical_id(qubit)
        self.operations.append(("z", [physical_id], {}))
        return self
    
    def cnot(self, control: int, target: int):
        """Apply CNOT gate."""
        control_physical = self._get_physical_id(control)
        target_physical = self._get_physical_id(target)
        self.operations.append(("cnot", [control_physical, target_physical], {}))
        return self
    
    def rx(self, qubit: int, theta: float):
        """Apply Rx rotation gate."""
        physical_id = self._get_physical_id(qubit)
        self.operations.append(("rx", [physical_id], {"theta": theta}))
        return self
    
    def ry(self, qubit: int, theta: float):
        """Apply Ry rotation gate."""
        physical_id = self._get_physical_id(qubit)
        self.operations.append(("ry", [physical_id], {"theta": theta}))
        return self
    
    def rz(self, qubit: int, theta: float):
        """Apply Rz rotation gate."""
        physical_id = self._get_physical_id(qubit)
        self.operations.append(("rz", [physical_id], {"theta": theta}))
        return self
    
    def to_static_circuit(self):
        """Convert to a static QuantRS2 circuit for execution."""
        from quantrs2 import Circuit
        
        if not self.allocated_qubits:
            raise ValueError("Cannot convert empty dynamic circuit")
        
        # Create static circuit with current qubit count
        max_physical_id = max(self.allocated_qubits)
        static_circuit = Circuit(max_physical_id + 1)
        
        # Apply all operations
        for op_name, physical_qubits, params in self.operations:
            if op_name == "h":
                static_circuit.h(physical_qubits[0])
            elif op_name == "x":
                static_circuit.x(physical_qubits[0])
            elif op_name == "y":
                static_circuit.y(physical_qubits[0])
            elif op_name == "z":
                static_circuit.z(physical_qubits[0])
            elif op_name == "cnot":
                static_circuit.cnot(physical_qubits[0], physical_qubits[1])
            elif op_name == "rx":
                static_circuit.rx(physical_qubits[0], params["theta"])
            elif op_name == "ry":
                static_circuit.ry(physical_qubits[0], params["theta"])
            elif op_name == "rz":
                static_circuit.rz(physical_qubits[0], params["theta"])
        
        return static_circuit
    
    def run(self, **kwargs):
        """Execute the circuit."""
        static_circuit = self.to_static_circuit()
        return static_circuit.run(**kwargs)
    
    def get_allocation_info(self) -> Dict[str, Any]:
        """Get information about current qubit allocation."""
        return {
            "logical_qubits": len(self.allocated_qubits),
            "physical_qubits": self.allocated_qubits.copy(),
            "qubit_mapping": self.qubit_map.copy(),
            "operations": len(self.operations),
            "auto_expand": self.auto_expand,
            "auto_deallocate": self.auto_deallocate
        }


# Global allocator instance
_global_allocator = None
_allocator_lock = threading.Lock()


def _get_global_allocator() -> QubitAllocator:
    """Get the global qubit allocator instance."""
    global _global_allocator
    with _allocator_lock:
        if _global_allocator is None:
            _global_allocator = QubitAllocator()
        return _global_allocator


def set_global_allocator(allocator: QubitAllocator):
    """Set the global qubit allocator."""
    global _global_allocator
    with _allocator_lock:
        _global_allocator = allocator


def get_global_allocation_stats() -> Dict[str, Any]:
    """Get global allocation statistics."""
    return _get_global_allocator().get_allocation_stats()


def create_dynamic_circuit(initial_qubits: int = 0, **kwargs) -> DynamicCircuit:
    """
    Create a new dynamic circuit.
    
    Args:
        initial_qubits: Initial number of qubits
        **kwargs: Additional arguments for DynamicCircuit
        
    Returns:
        New DynamicCircuit instance
    """
    return DynamicCircuit(initial_qubits, **kwargs)


def configure_allocation_strategy(strategy: AllocationStrategy):
    """Configure the global allocation strategy."""
    allocator = _get_global_allocator()
    allocator.strategy = strategy


# Convenience functions
def allocate_qubits(count: int, **kwargs) -> List[int]:
    """Allocate qubits using the global allocator."""
    return _get_global_allocator().allocate_qubits(count, **kwargs)


def deallocate_qubits(qubit_ids: Union[int, List[int]]) -> int:
    """Deallocate qubits using the global allocator."""
    return _get_global_allocator().deallocate_qubits(qubit_ids)


def garbage_collect():
    """Trigger garbage collection on the global allocator."""
    _get_global_allocator()._garbage_collect()


__all__ = [
    'AllocationStrategy',
    'QubitState', 
    'QubitInfo',
    'QubitAllocator',
    'DynamicCircuit',
    'create_dynamic_circuit',
    'configure_allocation_strategy',
    'allocate_qubits',
    'deallocate_qubits', 
    'garbage_collect',
    'get_global_allocation_stats',
    'set_global_allocator'
]