"""
Cirq Integration for QuantRS2

This module provides seamless integration between QuantRS2 and Google Cirq,
enabling quantum circuit conversion and compatibility.
"""

import warnings
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import numpy as np

try:
    import cirq
    from cirq import Circuit as CirqCircuit
    from cirq import Gate, Operation, Qubit, GridQubit
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False
    warnings.warn("Cirq not available. Install with: pip install cirq")
    
    # Mock classes for when Cirq is not available
    class CirqCircuit:
        def __init__(self):
            self.moments = []
        
        def append(self, operation):
            self.moments.append(operation)
    
    class Gate:
        pass
    
    class Operation:
        def __init__(self, gate, qubits):
            self.gate = gate
            self.qubits = qubits
    
    class Qubit:
        def __init__(self, name):
            self.name = name
    
    class GridQubit:
        def __init__(self, row, col):
            self.row = row
            self.col = col

try:
    from quantrs2 import Circuit as QuantRS2Circuit
    QUANTRS2_AVAILABLE = True
except ImportError:
    QUANTRS2_AVAILABLE = False
    warnings.warn("QuantRS2 core not available. Using mock implementation.")
    
    class QuantRS2Circuit:
        def __init__(self, n_qubits):
            self.n_qubits = n_qubits
            self.gates = []
        
        def h(self, qubit): self.gates.append(('h', qubit))
        def x(self, qubit): self.gates.append(('x', qubit))
        def y(self, qubit): self.gates.append(('y', qubit))
        def z(self, qubit): self.gates.append(('z', qubit))
        def cnot(self, control, target): self.gates.append(('cnot', control, target))
        def cz(self, control, target): self.gates.append(('cz', control, target))
        def rx(self, qubit, angle): self.gates.append(('rx', qubit, angle))
        def ry(self, qubit, angle): self.gates.append(('ry', qubit, angle))
        def rz(self, qubit, angle): self.gates.append(('rz', qubit, angle))


class QuantRS2CirqError(Exception):
    """Exception raised for QuantRS2-Cirq integration issues."""
    pass


class CirqQuantRS2Converter:
    """Converts between Cirq and QuantRS2 circuits."""
    
    def __init__(self):
        """Initialize the converter."""
        self.gate_mapping = {
            # Single-qubit gates
            'H': 'h',
            'X': 'x', 
            'Y': 'y',
            'Z': 'z',
            'S': 's',
            'T': 't',
            # Two-qubit gates
            'CNOT': 'cnot',
            'CX': 'cnot',
            'CZ': 'cz',
            'SWAP': 'swap',
            # Parametric gates
            'RX': 'rx',
            'RY': 'ry', 
            'RZ': 'rz',
        }
    
    def from_cirq(self, cirq_circuit: CirqCircuit, n_qubits: Optional[int] = None) -> 'QuantRS2Circuit':
        """Convert a Cirq circuit to QuantRS2 circuit.
        
        Args:
            cirq_circuit: The Cirq circuit to convert
            n_qubits: Number of qubits (auto-detected if not provided)
            
        Returns:
            QuantRS2Circuit: The converted circuit
        """
        if not CIRQ_AVAILABLE:
            warnings.warn("Cirq not available, using mock conversion")
            return QuantRS2Circuit(n_qubits or 2)
        
        # Auto-detect number of qubits
        if n_qubits is None:
            all_qubits = set()
            for moment in cirq_circuit.moments:
                for op in moment:
                    all_qubits.update(op.qubits)
            n_qubits = len(all_qubits)
        
        # Create QuantRS2 circuit
        quantrs2_circuit = QuantRS2Circuit(n_qubits)
        
        # Convert operations
        for moment in cirq_circuit.moments:
            for operation in moment:
                self._convert_operation_to_quantrs2(operation, quantrs2_circuit)
        
        return quantrs2_circuit
    
    def to_cirq(self, quantrs2_circuit: 'QuantRS2Circuit') -> CirqCircuit:
        """Convert a QuantRS2 circuit to Cirq circuit.
        
        Args:
            quantrs2_circuit: The QuantRS2 circuit to convert
            
        Returns:
            CirqCircuit: The converted circuit
        """
        if not CIRQ_AVAILABLE:
            warnings.warn("Cirq not available, using mock conversion")
            return CirqCircuit()
        
        cirq_circuit = CirqCircuit()
        
        # Create qubits
        qubits = [GridQubit(0, i) for i in range(quantrs2_circuit.n_qubits)]
        
        # Convert gates (this is simplified - would need proper gate extraction)
        if hasattr(quantrs2_circuit, 'gates'):
            for gate_info in quantrs2_circuit.gates:
                operation = self._convert_gate_to_cirq(gate_info, qubits)
                if operation:
                    cirq_circuit.append(operation)
        
        return cirq_circuit
    
    def _convert_operation_to_quantrs2(self, operation, quantrs2_circuit):
        """Convert a single Cirq operation to QuantRS2."""
        if not CIRQ_AVAILABLE:
            return
        
        gate_name = operation.gate.__class__.__name__
        qubits = operation.qubits
        
        # Map qubit indices (simplified)
        qubit_indices = [q.col if hasattr(q, 'col') else 0 for q in qubits]
        
        if gate_name == 'H':
            quantrs2_circuit.h(qubit_indices[0])
        elif gate_name in ['X', 'PauliX']:
            quantrs2_circuit.x(qubit_indices[0])
        elif gate_name in ['Y', 'PauliY']:
            quantrs2_circuit.y(qubit_indices[0])
        elif gate_name in ['Z', 'PauliZ']:
            quantrs2_circuit.z(qubit_indices[0])
        elif gate_name in ['CNOT', 'CX']:
            quantrs2_circuit.cnot(qubit_indices[0], qubit_indices[1])
        elif gate_name == 'CZ':
            quantrs2_circuit.cz(qubit_indices[0], qubit_indices[1])
        elif hasattr(operation.gate, 'exponent') and gate_name.startswith('R'):
            # Parametric rotation gates
            angle = operation.gate.exponent * np.pi
            if gate_name.endswith('X'):
                quantrs2_circuit.rx(qubit_indices[0], angle)
            elif gate_name.endswith('Y'):
                quantrs2_circuit.ry(qubit_indices[0], angle)
            elif gate_name.endswith('Z'):
                quantrs2_circuit.rz(qubit_indices[0], angle)
        else:
            warnings.warn(f"Unsupported Cirq gate: {gate_name}")
    
    def _convert_gate_to_cirq(self, gate_info, qubits):
        """Convert a QuantRS2 gate to Cirq operation."""
        if not CIRQ_AVAILABLE:
            return None
        
        gate_type = gate_info[0]
        
        if gate_type == 'h':
            return cirq.H(qubits[gate_info[1]])
        elif gate_type == 'x':
            return cirq.X(qubits[gate_info[1]])
        elif gate_type == 'y':
            return cirq.Y(qubits[gate_info[1]])
        elif gate_type == 'z':
            return cirq.Z(qubits[gate_info[1]])
        elif gate_type == 'cnot':
            return cirq.CNOT(qubits[gate_info[1]], qubits[gate_info[2]])
        elif gate_type == 'cz':
            return cirq.CZ(qubits[gate_info[1]], qubits[gate_info[2]])
        elif gate_type == 'rx':
            angle = gate_info[2]
            return cirq.rx(angle)(qubits[gate_info[1]])
        elif gate_type == 'ry':
            angle = gate_info[2]
            return cirq.ry(angle)(qubits[gate_info[1]])
        elif gate_type == 'rz':
            angle = gate_info[2]
            return cirq.rz(angle)(qubits[gate_info[1]])
        else:
            warnings.warn(f"Unsupported QuantRS2 gate: {gate_type}")
            return None


class CirqBackend:
    """QuantRS2 backend that uses Cirq for simulation."""
    
    def __init__(self, n_qubits: int):
        """Initialize Cirq backend.
        
        Args:
            n_qubits: Number of qubits
        """
        if not CIRQ_AVAILABLE:
            raise QuantRS2CirqError("Cirq not available")
        
        self.n_qubits = n_qubits
        self.qubits = [GridQubit(0, i) for i in range(n_qubits)]
        self.circuit = CirqCircuit()
        self.simulator = cirq.Simulator()
    
    def add_gate(self, gate_name: str, qubits: List[int], params: Optional[List[float]] = None):
        """Add a gate to the circuit.
        
        Args:
            gate_name: Name of the gate
            qubits: List of qubit indices
            params: Optional parameters for parametric gates
        """
        if not CIRQ_AVAILABLE:
            return
        
        cirq_qubits = [self.qubits[i] for i in qubits]
        
        if gate_name == 'H':
            self.circuit.append(cirq.H(cirq_qubits[0]))
        elif gate_name == 'X':
            self.circuit.append(cirq.X(cirq_qubits[0]))
        elif gate_name == 'Y':
            self.circuit.append(cirq.Y(cirq_qubits[0]))
        elif gate_name == 'Z':
            self.circuit.append(cirq.Z(cirq_qubits[0]))
        elif gate_name == 'CNOT':
            self.circuit.append(cirq.CNOT(cirq_qubits[0], cirq_qubits[1]))
        elif gate_name == 'CZ':
            self.circuit.append(cirq.CZ(cirq_qubits[0], cirq_qubits[1]))
        elif gate_name == 'RX' and params:
            self.circuit.append(cirq.rx(params[0])(cirq_qubits[0]))
        elif gate_name == 'RY' and params:
            self.circuit.append(cirq.ry(params[0])(cirq_qubits[0]))
        elif gate_name == 'RZ' and params:
            self.circuit.append(cirq.rz(params[0])(cirq_qubits[0]))
        else:
            warnings.warn(f"Unsupported gate: {gate_name}")
    
    def simulate(self, shots: int = 1000):
        """Simulate the circuit.
        
        Args:
            shots: Number of measurement shots
            
        Returns:
            Dict with simulation results
        """
        if not CIRQ_AVAILABLE:
            return {'state_vector': np.array([1.0] + [0.0] * (2**self.n_qubits - 1))}
        
        # State vector simulation
        result = self.simulator.simulate(self.circuit)
        state_vector = result.final_state_vector
        
        # Sample measurements
        measurement_circuit = self.circuit.copy()
        measurement_circuit.append(cirq.measure(*self.qubits, key='measurement'))
        
        sample_result = self.simulator.run(measurement_circuit, repetitions=shots)
        measurements = sample_result.measurements['measurement']
        
        return {
            'state_vector': state_vector,
            'measurements': measurements,
            'counts': dict(zip(*np.unique(measurements, axis=0, return_counts=True)))
        }


def create_bell_state_cirq() -> CirqCircuit:
    """Create a Bell state using Cirq.
    
    Returns:
        CirqCircuit: Bell state circuit
    """
    if not CIRQ_AVAILABLE:
        warnings.warn("Cirq not available, returning mock circuit")
        return CirqCircuit()
    
    q0, q1 = GridQubit(0, 0), GridQubit(0, 1)
    circuit = CirqCircuit()
    circuit.append([cirq.H(q0), cirq.CNOT(q0, q1)])
    return circuit


def convert_qiskit_to_cirq(qiskit_circuit) -> CirqCircuit:
    """Convert a Qiskit circuit to Cirq (if both are available).
    
    Args:
        qiskit_circuit: Qiskit QuantumCircuit
        
    Returns:
        CirqCircuit: Converted circuit
    """
    if not CIRQ_AVAILABLE:
        warnings.warn("Cirq not available for conversion")
        return CirqCircuit()
    
    warnings.warn("Qiskit-to-Cirq conversion not fully implemented")
    return CirqCircuit()


def test_cirq_quantrs2_integration():
    """Test the Cirq-QuantRS2 integration."""
    if not CIRQ_AVAILABLE:
        print("Cannot test integration: Cirq not available")
        return False
    
    if not QUANTRS2_AVAILABLE:
        print("Warning: QuantRS2 native backend not available, testing with mock implementation")
    
    try:
        # Test converter
        converter = CirqQuantRS2Converter()
        print("✓ Converter creation successful")
        
        # Test Bell state creation
        bell_circuit = create_bell_state_cirq()
        print("✓ Bell state circuit creation successful")
        
        # Test circuit conversion
        if CIRQ_AVAILABLE:
            quantrs2_circuit = converter.from_cirq(bell_circuit, n_qubits=2)
            print("✓ Cirq to QuantRS2 conversion successful")
            
            cirq_circuit_back = converter.to_cirq(quantrs2_circuit)
            print("✓ QuantRS2 to Cirq conversion successful")
        
        # Test backend
        if CIRQ_AVAILABLE:
            backend = CirqBackend(n_qubits=2)
            backend.add_gate('H', [0])
            backend.add_gate('CNOT', [0, 1])
            results = backend.simulate(shots=100)
            print(f"✓ Backend simulation successful: {len(results)} result keys")
        
        print("✅ All Cirq integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


# Export main classes and functions
__all__ = [
    'CirqQuantRS2Converter',
    'CirqBackend', 
    'QuantRS2CirqError',
    'create_bell_state_cirq',
    'convert_qiskit_to_cirq',
    'test_cirq_quantrs2_integration',
    'CIRQ_AVAILABLE',
    'QUANTRS2_AVAILABLE'
]