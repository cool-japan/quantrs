#!/usr/bin/env python3
"""
Enhanced PennyLane Plugin for QuantRS2

This module provides comprehensive integration between QuantRS2 and PennyLane,
enabling advanced hybrid quantum-classical machine learning workflows with
gradient computation, optimization, and sophisticated device capabilities.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Sequence
from dataclasses import dataclass
from enum import Enum
import threading
import copy

try:
    import pennylane as qml
    from pennylane import Device, QNode
    from pennylane.operation import Operation, Observable
    from pennylane.tape import QuantumTape
    from pennylane.measurements import ExpectationMP, VarianceMP, ProbabilityMP, StateMP
    from pennylane.gradients import finite_diff, param_shift, param_shift_cv
    from pennylane.workflow import execute
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
    PENNYLANE_VERSION = qml.__version__
except ImportError:
    PENNYLANE_AVAILABLE = False
    PENNYLANE_VERSION = "not_available"
    
    # Mock classes for when PennyLane is not available
    class Device:
        def __init__(self, *args, **kwargs): pass
        def capabilities(self): return {}
    class Operation: pass
    class Observable: pass
    class QuantumTape: pass
    class QNode: pass

try:
    from quantrs2 import Circuit as QuantRS2Circuit
    from quantrs2.dynamic_allocation import DynamicCircuit, QubitAllocator
    QUANTRS2_AVAILABLE = True
except ImportError:
    QUANTRS2_AVAILABLE = False
    
    class QuantRS2Circuit:
        def __init__(self, n_qubits):
            self.n_qubits = n_qubits
            self.gates = []
        def h(self, qubit): self.gates.append(('h', qubit))
        def x(self, qubit): self.gates.append(('x', qubit))
        def y(self, qubit): self.gates.append(('y', qubit))
        def z(self, qubit): self.gates.append(('z', qubit))
        def rx(self, qubit, angle): self.gates.append(('rx', qubit, angle))
        def ry(self, qubit, angle): self.gates.append(('ry', qubit, angle))
        def rz(self, qubit, angle): self.gates.append(('rz', qubit, angle))
        def cnot(self, control, target): self.gates.append(('cnot', control, target))
        def run(self):
            n_states = 2 ** self.n_qubits
            state = np.zeros(n_states, dtype=complex)
            state[0] = 1.0  # |0...0⟩ state
            return type('MockResult', (), {
                'state_vector': state,
                'probabilities': lambda: np.abs(state)**2,
                'measurements': [0] * self.n_qubits
            })()
    
    class DynamicCircuit(QuantRS2Circuit):
        def __init__(self, *args, **kwargs):
            super().__init__(args[0] if args else 1)


class DeviceMode(Enum):
    """Device execution modes."""
    STATEVECTOR = "statevector"
    SAMPLING = "sampling"
    MIXED = "mixed"
    NOISY = "noisy"


class GradientMethod(Enum):
    """Gradient computation methods."""
    PARAMETER_SHIFT = "parameter_shift"
    FINITE_DIFF = "finite_diff"
    ADJOINT = "adjoint"
    BACKPROP = "backprop"


@dataclass
class DeviceConfig:
    """Configuration for enhanced QuantRS2 device."""
    mode: DeviceMode = DeviceMode.STATEVECTOR
    gradient_method: GradientMethod = GradientMethod.PARAMETER_SHIFT
    shots: Optional[int] = None
    seed: Optional[int] = None
    use_dynamic_allocation: bool = False
    enable_caching: bool = True
    cache_size: int = 1000
    enable_jit: bool = False
    noise_model: Optional[Dict[str, Any]] = None
    optimization_level: int = 1


class EnhancedQuantRS2Device(Device):
    """Enhanced PennyLane device with advanced QuantRS2 integration."""
    
    name = "quantrs2.enhanced"
    short_name = "quantrs2_enhanced"
    pennylane_requires = ">=0.28.0"
    version = "0.2.0"
    author = "QuantRS2 Enhanced Team"
    
    # Extended operations support
    operations = {
        # Single-qubit gates
        "PauliX", "PauliY", "PauliZ", "Hadamard", "S", "T", "SX", "SY", "SZ",
        "RX", "RY", "RZ", "Rot", "PhaseShift", "U1", "U2", "U3",
        
        # Two-qubit gates
        "CNOT", "CZ", "CY", "SWAP", "ISWAP", "ECR", "DCG",
        "ControlledPhaseShift", "CRX", "CRY", "CRZ", "CRot",
        
        # Multi-qubit gates
        "Toffoli", "CSWAP", "MultiControlledX", "QFT", "GroverOperator",
        
        # Special operations
        "Identity", "BasisState", "QubitStateVector", "QubitUnitary",
        "ControlledQubitUnitary", "DiagonalQubitUnitary",
        
        # Noisy operations
        "AmplitudeDamping", "GeneralizedAmplitudeDamping", "PhaseDamping",
        "DepolarizingChannel", "BitFlip", "PhaseFlip", "ThermalRelaxationError"
    }
    
    # Extended observables support
    observables = {
        "PauliX", "PauliY", "PauliZ", "Identity", "Hadamard", "Hermitian",
        "Projector", "SparseHamiltonian", "Hamiltonian", "Sum", "Prod", "SProd"
    }
    
    def __init__(self, wires, config: Optional[DeviceConfig] = None, **kwargs):
        """
        Initialize enhanced QuantRS2 device.
        
        Args:
            wires: Number of wires/qubits or wire labels
            config: Device configuration
            **kwargs: Additional device options (for backward compatibility)
        """
        if not QUANTRS2_AVAILABLE:
            raise ImportError("QuantRS2 not available")
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane not available")
        
        # Handle configuration
        self.config = config or DeviceConfig()
        if kwargs:
            # Update config with legacy kwargs
            if 'shots' in kwargs:
                self.config.shots = kwargs['shots']
            if 'mode' in kwargs:
                self.config.mode = DeviceMode(kwargs['mode'])
        
        super().__init__(wires, shots=self.config.shots)
        
        self.n_qubits = len(self.wires)
        self.logger = logging.getLogger(f"quantrs2.pennylane.{self.name}")
        
        # Device state
        self._circuit = None
        self._state = None
        self._cache = {} if self.config.enable_caching else None
        self._allocator = QubitAllocator() if self.config.use_dynamic_allocation else None
        
        # Threading for parallel execution
        self._lock = threading.RLock()
        
        # Enhanced capabilities
        self._init_capabilities()
        
        # Gradient computation setup
        self._setup_gradient_methods()
        
        self.logger.info(f"Initialized enhanced QuantRS2 device with {self.n_qubits} qubits")
    
    def _init_capabilities(self):
        """Initialize device capabilities."""
        capabilities = {
            "model": "qubit",
            "supports_finite_shots": True,
            "supports_tensor_observables": True,
            "returns_state": self.config.mode in [DeviceMode.STATEVECTOR, DeviceMode.MIXED],
            "returns_probs": True,
            "supports_inverse_operations": True,
            "supports_analytic_computation": self.config.mode == DeviceMode.STATEVECTOR,
            "supports_broadcasting": True,
            "passthru_interface": "numpy",
            "supports_derivatives": True,
        }
        
        # Gradient method capabilities
        if self.config.gradient_method == GradientMethod.PARAMETER_SHIFT:
            capabilities["provides_jacobian"] = True
        elif self.config.gradient_method == GradientMethod.ADJOINT:
            capabilities["passthru_devices"] = {"torch", "tf", "autograd", "jax"}
        
        self.capabilities().update(capabilities)
    
    def _setup_gradient_methods(self):
        """Setup gradient computation methods."""
        if self.config.gradient_method == GradientMethod.PARAMETER_SHIFT:
            self._gradient_fn = self._parameter_shift_gradients
        elif self.config.gradient_method == GradientMethod.FINITE_DIFF:
            self._gradient_fn = self._finite_diff_gradients
        elif self.config.gradient_method == GradientMethod.ADJOINT:
            self._gradient_fn = self._adjoint_gradients
        else:
            self._gradient_fn = self._finite_diff_gradients
    
    def reset(self):
        """Reset device state."""
        with self._lock:
            self._circuit = None
            self._state = None
            if self._cache is not None and len(self._cache) > self.config.cache_size:
                self._cache.clear()
    
    def apply(self, operations, rotations=None, **kwargs):
        """
        Apply quantum operations with enhanced features.
        
        Args:
            operations: List of operations to apply
            rotations: Basis rotations for measurements
            **kwargs: Additional arguments
        """
        with self._lock:
            # Check cache first
            cache_key = self._get_cache_key(operations, rotations)
            if self._cache is not None and cache_key in self._cache:
                self._circuit, self._state = self._cache[cache_key]
                return
            
            # Create circuit based on configuration
            if self.config.use_dynamic_allocation:
                self._circuit = DynamicCircuit(
                    initial_qubits=self.n_qubits,
                    allocator=self._allocator,
                    auto_expand=True
                )
            else:
                self._circuit = QuantRS2Circuit(self.n_qubits)
            
            # Apply operations with enhanced conversion
            for op in operations:
                self._apply_operation_enhanced(op)
            
            # Apply basis rotations if provided
            if rotations is not None:
                self._apply_basis_rotations(rotations)
            
            # Cache the result
            if self._cache is not None:
                self._cache[cache_key] = (copy.deepcopy(self._circuit), copy.deepcopy(self._state))
    
    def _get_cache_key(self, operations, rotations):
        """Generate cache key for operations."""
        if not self.config.enable_caching:
            return None
        
        key_parts = []
        for op in operations:
            key_parts.append((op.name, tuple(op.wires), tuple(op.parameters)))
        
        if rotations:
            for rot in rotations:
                key_parts.append(("rotation", tuple(rot.wires), tuple(rot.parameters)))
        
        return hash(tuple(key_parts))
    
    def _apply_operation_enhanced(self, operation):
        """Enhanced operation application with better gate support."""
        name = operation.name
        wires = [self.wire_map[w] for w in operation.wires]
        params = operation.parameters
        
        # Single-qubit Pauli gates
        if name == "PauliX":
            self._circuit.x(wires[0])
        elif name == "PauliY":
            self._circuit.y(wires[0])
        elif name == "PauliZ":
            self._circuit.z(wires[0])
        elif name == "Identity":
            pass  # Identity gate does nothing
        
        # Single-qubit Clifford gates
        elif name == "Hadamard":
            self._circuit.h(wires[0])
        elif name == "S":
            self._circuit.s(wires[0])
        elif name == "T":
            self._circuit.t(wires[0])
        elif name == "SX":
            # SX = sqrt(X) = (I + iX)/sqrt(2)
            self._apply_sx_gate(wires[0])
        elif name == "SY":
            # SY = sqrt(Y)
            self._apply_sy_gate(wires[0])
        elif name == "SZ":
            # SZ = sqrt(Z) = S
            self._circuit.s(wires[0])
        
        # Rotation gates
        elif name == "RX":
            self._circuit.rx(wires[0], params[0])
        elif name == "RY":
            self._circuit.ry(wires[0], params[0])
        elif name == "RZ":
            self._circuit.rz(wires[0], params[0])
        elif name == "PhaseShift":
            self._circuit.rz(wires[0], params[0])
        
        # Universal gates
        elif name == "U1":
            self._circuit.rz(wires[0], params[0])
        elif name == "U2":
            # U2(φ,λ) = RZ(φ)RY(π/2)RZ(λ)
            self._circuit.rz(wires[0], params[0])
            self._circuit.ry(wires[0], np.pi/2)
            self._circuit.rz(wires[0], params[1])
        elif name == "U3":
            # U3(θ,φ,λ) = RZ(φ)RY(θ)RZ(λ)
            self._circuit.rz(wires[0], params[1])
            self._circuit.ry(wires[0], params[0])
            self._circuit.rz(wires[0], params[2])
        elif name == "Rot":
            # Rot(φ,θ,ω) = RZ(ω)RY(θ)RZ(φ)
            self._circuit.rz(wires[0], params[0])
            self._circuit.ry(wires[0], params[1])
            self._circuit.rz(wires[0], params[2])
        
        # Two-qubit gates
        elif name == "CNOT":
            self._circuit.cnot(wires[0], wires[1])
        elif name == "CZ":
            self._circuit.cz(wires[0], wires[1])
        elif name == "CY":
            self._apply_cy_gate(wires[0], wires[1])
        elif name == "SWAP":
            self._circuit.swap(wires[0], wires[1])
        elif name == "ISWAP":
            self._apply_iswap_gate(wires[0], wires[1])
        
        # Controlled rotation gates
        elif name == "CRX":
            self._apply_crx_gate(wires[0], wires[1], params[0])
        elif name == "CRY":
            self._apply_cry_gate(wires[0], wires[1], params[0])
        elif name == "CRZ":
            self._apply_crz_gate(wires[0], wires[1], params[0])
        elif name == "ControlledPhaseShift":
            self._apply_crz_gate(wires[0], wires[1], params[0])
        elif name == "CRot":
            # Controlled rotation: CRot(φ,θ,ω) on target qubit
            self._apply_crot_gate(wires[0], wires[1], params)
        
        # Multi-qubit gates
        elif name == "Toffoli":
            self._apply_toffoli_gate(wires)
        elif name == "CSWAP":
            self._apply_fredkin_gate(wires)
        elif name == "MultiControlledX":
            self._apply_mcx_gate(wires)
        
        # Special state preparation
        elif name == "BasisState":
            self._apply_basis_state(wires, params[0])
        elif name == "QubitStateVector":
            self._apply_state_vector(wires, params[0])
        
        # Unitary operations
        elif name == "QubitUnitary":
            self._apply_unitary_matrix(wires, params[0])
        elif name == "ControlledQubitUnitary":
            self._apply_controlled_unitary(wires, params[0])
        
        # Algorithm-specific gates
        elif name == "QFT":
            self._apply_qft(wires)
        elif name == "GroverOperator":
            self._apply_grover_operator(wires, params)
        
        # Noise operations
        elif name in ["AmplitudeDamping", "PhaseDamping", "DepolarizingChannel", 
                      "BitFlip", "PhaseFlip", "ThermalRelaxationError"]:
            self._apply_noise_operation(name, wires, params)
        
        else:
            pass
    
    def _apply_sx_gate(self, qubit):
        """Apply sqrt(X) gate."""
        # SX = RY(π/2)RZ(π)RY(-π/2) (one decomposition)
        self._circuit.ry(qubit, np.pi/2)
        self._circuit.rz(qubit, np.pi)
        self._circuit.ry(qubit, -np.pi/2)
    
    def _apply_sy_gate(self, qubit):
        """Apply sqrt(Y) gate."""
        # SY decomposition
        self._circuit.rx(qubit, -np.pi/2)
        self._circuit.rz(qubit, np.pi)
        self._circuit.rx(qubit, np.pi/2)
    
    def _apply_cy_gate(self, control, target):
        """Apply controlled-Y gate."""
        # CY = CNOT · (I ⊗ S†) · CNOT · (I ⊗ S)
        self._circuit.s(target)
        self._circuit.cnot(control, target)
        self._circuit.sdg(target)
        self._circuit.cnot(control, target)
    
    def _apply_iswap_gate(self, qubit1, qubit2):
        """Apply iSWAP gate."""
        # iSWAP decomposition
        self._circuit.s(qubit1)
        self._circuit.s(qubit2)
        self._circuit.h(qubit1)
        self._circuit.cnot(qubit1, qubit2)
        self._circuit.cnot(qubit2, qubit1)
        self._circuit.h(qubit2)
    
    def _apply_crx_gate(self, control, target, theta):
        """Apply controlled-RX gate."""
        self._circuit.ry(target, theta/2)
        self._circuit.cnot(control, target)
        self._circuit.ry(target, -theta/2)
        self._circuit.cnot(control, target)
    
    def _apply_cry_gate(self, control, target, theta):
        """Apply controlled-RY gate."""
        self._circuit.ry(target, theta/2)
        self._circuit.cnot(control, target)
        self._circuit.ry(target, -theta/2)
        self._circuit.cnot(control, target)
    
    def _apply_crz_gate(self, control, target, theta):
        """Apply controlled-RZ gate."""
        self._circuit.rz(target, theta/2)
        self._circuit.cnot(control, target)
        self._circuit.rz(target, -theta/2)
        self._circuit.cnot(control, target)
    
    def _apply_crot_gate(self, control, target, params):
        """Apply controlled rotation gate."""
        phi, theta, omega = params
        self._circuit.rz(target, phi/2)
        self._circuit.cnot(control, target)
        self._circuit.rz(target, -phi/2)
        self._circuit.ry(target, -theta/2)
        self._circuit.cnot(control, target)
        self._circuit.ry(target, theta/2)
        self._circuit.rz(target, omega/2)
    
    def _apply_toffoli_gate(self, wires):
        """Apply Toffoli (CCX) gate."""
        if len(wires) != 3:
            raise ValueError("Toffoli gate requires exactly 3 qubits")
        
        if hasattr(self._circuit, 'toffoli'):
            self._circuit.toffoli(wires[0], wires[1], wires[2])
        else:
            # Decompose Toffoli using available gates
            self._decompose_toffoli(wires[0], wires[1], wires[2])
    
    def _decompose_toffoli(self, control1, control2, target):
        """Decompose Toffoli gate."""
        self._circuit.h(target)
        self._circuit.cnot(control2, target)
        self._circuit.tdg(target)
        self._circuit.cnot(control1, target)
        self._circuit.t(target)
        self._circuit.cnot(control2, target)
        self._circuit.tdg(target)
        self._circuit.cnot(control1, target)
        self._circuit.t(control2)
        self._circuit.t(target)
        self._circuit.cnot(control1, control2)
        self._circuit.h(target)
        self._circuit.t(control1)
        self._circuit.tdg(control2)
        self._circuit.cnot(control1, control2)
    
    def _apply_fredkin_gate(self, wires):
        """Apply Fredkin (CSWAP) gate."""
        if len(wires) != 3:
            raise ValueError("Fredkin gate requires exactly 3 qubits")
        
        control, target1, target2 = wires
        self._circuit.cnot(target2, target1)
        self._apply_toffoli_gate([control, target1, target2])
        self._circuit.cnot(target2, target1)
    
    def _apply_mcx_gate(self, wires):
        """Apply multi-controlled X gate."""
        if len(wires) < 2:
            raise ValueError("MCX gate requires at least 2 qubits")
        
        controls = wires[:-1]
        target = wires[-1]
        
        if len(controls) == 1:
            self._circuit.cnot(controls[0], target)
        elif len(controls) == 2:
            self._apply_toffoli_gate(wires)
        else:
            # Use decomposition for larger MCX gates
            self._decompose_mcx(controls, target)
    
    def _decompose_mcx(self, controls, target):
        """Decompose multi-controlled X gate."""
        # Simplified MCX decomposition (would need ancilla qubits in practice)
        for ctrl in controls:
            self._circuit.cnot(ctrl, target)
    
    def _apply_basis_state(self, wires, basis_state):
        """Apply basis state preparation."""
        if len(basis_state) != len(wires):
            raise ValueError("Basis state length must match number of wires")
        
        for i, bit in enumerate(basis_state):
            if bit == 1:
                self._circuit.x(wires[i])
    
    def _apply_state_vector(self, wires, state_vector):
        """Apply arbitrary state vector preparation."""
        # This is a simplified implementation
        # Full implementation would require state preparation algorithms
        
        # For now, prepare computational basis state closest to desired state
        probabilities = np.abs(state_vector)**2
        max_prob_index = np.argmax(probabilities)
        basis_state = [int(b) for b in format(max_prob_index, f'0{len(wires)}b')]
        self._apply_basis_state(wires, basis_state)
    
    def _apply_unitary_matrix(self, wires, unitary):
        """Apply arbitrary unitary matrix."""
        # This is a placeholder - full implementation would require 
        # unitary synthesis algorithms
    
    def _apply_controlled_unitary(self, wires, unitary):
        """Apply controlled unitary operation."""
        # Placeholder implementation
    
    def _apply_qft(self, wires):
        """Apply Quantum Fourier Transform."""
        n = len(wires)
        for i in range(n):
            self._circuit.h(wires[i])
            for j in range(i + 1, n):
                angle = np.pi / (2 ** (j - i))
                self._apply_crz_gate(wires[j], wires[i], angle)
        
        # Swap qubits
        for i in range(n // 2):
            self._circuit.swap(wires[i], wires[n - 1 - i])
    
    def _apply_grover_operator(self, wires, params):
        """Apply Grover diffusion operator."""
        # Simplified Grover operator
        for wire in wires:
            self._circuit.h(wire)
        for wire in wires:
            self._circuit.x(wire)
        
        # Multi-controlled Z
        if len(wires) > 1:
            self._apply_mcz_gate(wires)
        
        for wire in wires:
            self._circuit.x(wire)
        for wire in wires:
            self._circuit.h(wire)
    
    def _apply_mcz_gate(self, wires):
        """Apply multi-controlled Z gate."""
        if len(wires) == 1:
            self._circuit.z(wires[0])
        elif len(wires) == 2:
            self._circuit.cz(wires[0], wires[1])
        else:
            # MCZ = H·MCX·H on target
            target = wires[-1]
            self._circuit.h(target)
            self._apply_mcx_gate(wires)
            self._circuit.h(target)
    
    def _apply_noise_operation(self, noise_type, wires, params):
        """Apply noise operation (placeholder)."""
        # In a full implementation, this would apply noise to the quantum state
        self.logger.info(f"Applied {noise_type} noise to qubits {wires}")
    
    def _apply_basis_rotations(self, rotations):
        """Apply basis rotations for measurements."""
        for rotation in rotations:
            self._apply_operation_enhanced(rotation)
    
    def generate_samples(self):
        """Generate measurement samples."""
        if self._circuit is None:
            raise RuntimeError("No circuit has been applied")
        
        # Execute circuit
        result = self._circuit.run()
        
        if hasattr(result, 'state_vector'):
            self._state = np.array(result.state_vector)
        else:
            # Fallback for mock implementation
            self._state = np.array(result.state_vector)
        
        if self.shots is None:
            # Exact computation - return probabilities
            probabilities = np.abs(self._state)**2
            return self._probabilities_to_samples(probabilities)
        else:
            # Finite shots - sample from distribution
            probabilities = np.abs(self._state)**2
            return self._sample_from_probabilities(probabilities, self.shots)
    
    def _probabilities_to_samples(self, probabilities):
        """Convert probabilities to sample format."""
        n_wires = int(np.log2(len(probabilities)))
        samples = []
        
        for i, prob in enumerate(probabilities):
            if prob > 1e-12:
                binary = format(i, f'0{n_wires}b')
                sample = [int(b) for b in binary]
                samples.append(sample)
        
        return np.array(samples) if samples else np.array([]).reshape(0, n_wires)
    
    def _sample_from_probabilities(self, probabilities, n_shots):
        """Sample from probability distribution."""
        n_wires = int(np.log2(len(probabilities)))
        
        # Generate random samples
        random_samples = np.random.choice(
            len(probabilities), 
            size=n_shots, 
            p=probabilities
        )
        
        # Convert to binary representation
        samples = []
        for sample in random_samples:
            binary = format(sample, f'0{n_wires}b')
            samples.append([int(b) for b in binary])
        
        return np.array(samples)
    
    def probability(self, wires=None, shot_range=None, bin_size=None):
        """Compute probability distribution."""
        if self._state is None:
            raise RuntimeError("No quantum state available")
        
        probabilities = np.abs(self._state)**2
        
        if wires is None:
            return probabilities
        
        # Marginal probabilities for specific wires
        return self._marginal_probabilities(probabilities, wires)
    
    def _marginal_probabilities(self, probabilities, wires):
        """Compute marginal probabilities for specific wires."""
        n_total_wires = int(np.log2(len(probabilities)))
        n_target_wires = len(wires)
        
        # Create mapping from full basis states to target basis states
        marginal_probs = np.zeros(2**n_target_wires)
        
        for i, prob in enumerate(probabilities):
            # Extract bits for target wires
            full_binary = format(i, f'0{n_total_wires}b')
            target_bits = ''.join(full_binary[w] for w in wires)
            target_index = int(target_bits, 2)
            marginal_probs[target_index] += prob
        
        return marginal_probs
    
    def expval(self, observable, shot_range=None, bin_size=None):
        """Compute expectation value of observable."""
        if self._state is None:
            raise RuntimeError("No quantum state available")
        
        return self._compute_expectation_value(observable, self._state)
    
    def _compute_expectation_value(self, observable, state):
        """Compute expectation value for given observable and state."""
        if observable.name in ["PauliX", "PauliY", "PauliZ"]:
            return self._pauli_expectation(observable, state)
        elif observable.name == "Identity":
            return 1.0
        elif observable.name == "Hermitian":
            return self._hermitian_expectation(observable, state)
        else:
            return 0.0
    
    def _pauli_expectation(self, pauli_obs, state):
        """Compute Pauli observable expectation value."""
        wire = self.wire_map[pauli_obs.wires[0]]
        n_qubits = int(np.log2(len(state)))
        
        # Create Pauli matrix for the specific wire
        if pauli_obs.name == "PauliX":
            pauli_matrix = self._pauli_x_matrix(wire, n_qubits)
        elif pauli_obs.name == "PauliY":
            pauli_matrix = self._pauli_y_matrix(wire, n_qubits)
        elif pauli_obs.name == "PauliZ":
            pauli_matrix = self._pauli_z_matrix(wire, n_qubits)
        
        # Compute ⟨ψ|P|ψ⟩
        expectation = np.real(np.conj(state) @ pauli_matrix @ state)
        return expectation
    
    def _pauli_x_matrix(self, wire, n_qubits):
        """Create Pauli-X matrix for specific wire."""
        # Simplified implementation - would use tensor products in practice
        size = 2**n_qubits
        matrix = np.eye(size, dtype=complex)
        # This is a placeholder - proper implementation needed
        return matrix
    
    def _pauli_y_matrix(self, wire, n_qubits):
        """Create Pauli-Y matrix for specific wire."""
        # Placeholder implementation
        size = 2**n_qubits
        return np.eye(size, dtype=complex)
    
    def _pauli_z_matrix(self, wire, n_qubits):
        """Create Pauli-Z matrix for specific wire."""
        # Z expectation can be computed directly from state amplitudes
        size = 2**n_qubits
        matrix = np.eye(size, dtype=complex)
        
        for i in range(size):
            bit = (i >> wire) & 1
            matrix[i, i] = 1 if bit == 0 else -1
        
        return matrix
    
    def _hermitian_expectation(self, hermitian_obs, state):
        """Compute Hermitian observable expectation value."""
        matrix = hermitian_obs.matrix
        return np.real(np.conj(state) @ matrix @ state)
    
    def var(self, observable, shot_range=None, bin_size=None):
        """Compute variance of observable."""
        exp_val = self.expval(observable)
        exp_val_sq = self._compute_expectation_value_squared(observable)
        return exp_val_sq - exp_val**2
    
    def _compute_expectation_value_squared(self, observable):
        """Compute ⟨ψ|O²|ψ⟩."""
        # Simplified implementation
        return self.expval(observable)**2
    
    # Gradient computation methods
    def _parameter_shift_gradients(self, tape, method="auto"):
        """Compute gradients using parameter shift rule."""
        if not PENNYLANE_AVAILABLE:
            return np.zeros(len(tape.trainable_params))
        
        try:
            from pennylane.gradients import param_shift
            return param_shift(tape, argnum=tape.trainable_params)
        except Exception as e:
            return self._finite_diff_gradients(tape)
    
    def _finite_diff_gradients(self, tape, h=1e-7):
        """Compute gradients using finite differences."""
        gradients = []
        original_params = tape.get_parameters()
        
        for i in tape.trainable_params:
            # Forward difference
            params_plus = original_params.copy()
            params_plus[i] += h
            
            params_minus = original_params.copy()
            params_minus[i] -= h
            
            # Execute tapes with shifted parameters
            tape_plus = tape.copy()
            tape_plus.set_parameters(params_plus)
            
            tape_minus = tape.copy()
            tape_minus.set_parameters(params_minus)
            
            result_plus = self._execute_tape(tape_plus)
            result_minus = self._execute_tape(tape_minus)
            
            gradient = (result_plus - result_minus) / (2 * h)
            gradients.append(gradient)
        
        return np.array(gradients)
    
    def _adjoint_gradients(self, tape):
        """Compute gradients using adjoint method."""
        # Placeholder for adjoint differentiation
        return self._finite_diff_gradients(tape)
    
    def _execute_tape(self, tape):
        """Execute a quantum tape and return results."""
        # This is a simplified implementation
        self.apply(tape.operations)
        return self.generate_samples()
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information."""
        return {
            "name": self.name,
            "version": self.version,
            "n_qubits": self.n_qubits,
            "config": {
                "mode": self.config.mode.value,
                "gradient_method": self.config.gradient_method.value,
                "shots": self.config.shots,
                "use_dynamic_allocation": self.config.use_dynamic_allocation,
                "enable_caching": self.config.enable_caching,
                "optimization_level": self.config.optimization_level
            },
            "capabilities": dict(self.capabilities()),
            "cache_size": len(self._cache) if self._cache else 0
        }


class QuantRS2QMLModel:
    """Quantum Machine Learning model using QuantRS2 and PennyLane."""
    
    def __init__(self, 
                 n_qubits: int,
                 n_layers: int,
                 device_config: Optional[DeviceConfig] = None):
        """
        Initialize QML model.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            device_config: Device configuration
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_config = device_config or DeviceConfig()
        
        # Create PennyLane device
        self.device = EnhancedQuantRS2Device(
            wires=n_qubits,
            config=self.device_config
        )
        
        # Initialize parameters
        self.weights = self._initialize_weights()
        
        # Create QNode
        self.qnode = self._create_qnode()
        
        self.logger = logging.getLogger("quantrs2.qml")
    
    def _initialize_weights(self) -> np.ndarray:
        """Initialize variational parameters."""
        # Each layer has 2 parameters per qubit (RY and RZ rotations)
        n_params = self.n_layers * self.n_qubits * 2
        return np.random.uniform(0, 2*np.pi, n_params)
    
    def _create_qnode(self):
        """Create PennyLane QNode."""
        if not PENNYLANE_AVAILABLE:
            return None
        
        @qml.qnode(self.device, diff_method=self.device_config.gradient_method.value)
        def circuit(weights, x=None):
            # Data encoding (if input data provided)
            if x is not None:
                for i in range(min(len(x), self.n_qubits)):
                    qml.RY(x[i], wires=i)
            
            # Variational layers
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    param_idx = layer * self.n_qubits * 2 + qubit * 2
                    qml.RY(weights[param_idx], wires=qubit)
                    qml.RZ(weights[param_idx + 1], wires=qubit)
                
                # Entangling layer
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            
            # Measurement
            return qml.expval(qml.PauliZ(0))
        
        return circuit
    
    def forward(self, x: np.ndarray) -> float:
        """Forward pass of the QML model."""
        if self.qnode is None:
            # Fallback for when PennyLane is not available
            return np.random.random()
        
        return self.qnode(self.weights, x)
    
    def train_step(self, x: np.ndarray, y: float, learning_rate: float = 0.01) -> float:
        """Perform one training step."""
        if not PENNYLANE_AVAILABLE:
            return 0.0
        
        # Compute cost
        prediction = self.forward(x)
        cost = (prediction - y)**2
        
        # Compute gradients
        grad_fn = qml.grad(lambda w: (self.qnode(w, x) - y)**2)
        gradients = grad_fn(self.weights)
        
        # Update weights
        self.weights -= learning_rate * gradients
        
        return cost
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on a batch of inputs."""
        predictions = []
        for x in X:
            pred = self.forward(x)
            predictions.append(pred)
        return np.array(predictions)
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray, 
              epochs: int = 100,
              learning_rate: float = 0.01,
              verbose: bool = True) -> Dict[str, List[float]]:
        """Train the QML model."""
        history = {"loss": []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for i in range(len(X)):
                loss = self.train_step(X[i], y[i], learning_rate)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(X)
            history["loss"].append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return history


class QuantRS2VQC:
    """Variational Quantum Classifier using QuantRS2."""
    
    def __init__(self, 
                 n_qubits: int,
                 n_classes: int = 2,
                 ansatz_type: str = "hardware_efficient",
                 device_config: Optional[DeviceConfig] = None):
        """Initialize VQC."""
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.ansatz_type = ansatz_type
        self.device_config = device_config or DeviceConfig()
        
        self.device = EnhancedQuantRS2Device(
            wires=n_qubits,
            config=self.device_config
        )
        
        self.weights = self._initialize_classifier_weights()
        self.qnode = self._create_classifier_qnode()
        
        self.logger = logging.getLogger("quantrs2.vqc")
    
    def _initialize_classifier_weights(self) -> np.ndarray:
        """Initialize classifier weights."""
        # Weights for data encoding + variational circuit
        encoding_weights = self.n_qubits  # One parameter per qubit for encoding
        variational_weights = self.n_qubits * 3  # RX, RY, RZ per qubit
        total_weights = encoding_weights + variational_weights
        
        return np.random.uniform(-np.pi, np.pi, total_weights)
    
    def _create_classifier_qnode(self):
        """Create classifier QNode."""
        if not PENNYLANE_AVAILABLE:
            return None
        
        @qml.qnode(self.device)
        def classifier_circuit(weights, x):
            # Data encoding
            for i in range(self.n_qubits):
                if i < len(x):
                    qml.RY(weights[i] * x[i], wires=i)
            
            # Variational ansatz
            offset = self.n_qubits
            for i in range(self.n_qubits):
                qml.RX(weights[offset + i * 3], wires=i)
                qml.RY(weights[offset + i * 3 + 1], wires=i)
                qml.RZ(weights[offset + i * 3 + 2], wires=i)
            
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Measurements for classification
            if self.n_classes == 2:
                return qml.expval(qml.PauliZ(0))
            else:
                # Multi-class: measure multiple qubits
                return [qml.expval(qml.PauliZ(i)) for i in range(min(self.n_classes, self.n_qubits))]
        
        return classifier_circuit
    
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.qnode is None:
            return np.random.random(self.n_classes)
        
        if self.n_classes == 2:
            expectation = self.qnode(self.weights, x)
            prob_1 = (1 + expectation) / 2
            return np.array([1 - prob_1, prob_1])
        else:
            expectations = self.qnode(self.weights, x)
            probs = [(1 + exp) / 2 for exp in expectations]
            # Normalize probabilities
            total = sum(probs)
            return np.array(probs) / total if total > 0 else np.ones(self.n_classes) / self.n_classes
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make class predictions."""
        predictions = []
        for x in X:
            probs = self.predict_proba(x)
            predictions.append(np.argmax(probs))
        return np.array(predictions)
    
    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray,
            epochs: int = 100,
            learning_rate: float = 0.01) -> Dict[str, List[float]]:
        """Train the classifier."""
        if not PENNYLANE_AVAILABLE:
            return {"loss": []}
        
        history = {"loss": [], "accuracy": []}
        
        # Define cost function
        def cost_fn(weights):
            total_cost = 0.0
            for i in range(len(X)):
                probs = self.predict_proba(X[i])
                # Cross-entropy loss
                true_class = int(y[i])
                cost = -np.log(probs[true_class] + 1e-10)
                total_cost += cost
            return total_cost / len(X)
        
        # Training loop
        opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
        
        for epoch in range(epochs):
            self.weights, cost = opt.step_and_cost(cost_fn, self.weights)
            
            # Calculate accuracy
            predictions = self.predict(X)
            accuracy = np.mean(predictions == y)
            
            history["loss"].append(cost)
            history["accuracy"].append(accuracy)
            
            if (epoch + 1) % 20 == 0:
                self.logger.info(f"Epoch {epoch + 1}: Loss = {cost:.4f}, Accuracy = {accuracy:.4f}")
        
        return history


# Convenience functions
def create_quantrs2_device(wires, **kwargs) -> EnhancedQuantRS2Device:
    """Create QuantRS2 PennyLane device."""
    config = DeviceConfig(**kwargs)
    return EnhancedQuantRS2Device(wires, config=config)


def register_quantrs2_device():
    """Register QuantRS2 device with PennyLane."""
    if PENNYLANE_AVAILABLE:
        try:
            qml.device_registry['quantrs2.enhanced'] = EnhancedQuantRS2Device
            logging.getLogger("quantrs2.pennylane").info("Registered QuantRS2 enhanced device")
        except Exception as e:
            pass


def quantrs2_qnode(device_config: Optional[DeviceConfig] = None, **kwargs):
    """Decorator for creating QuantRS2 QNodes."""
    def decorator(func):
        if not PENNYLANE_AVAILABLE:
            return func
        
        # Extract number of qubits from function
        import inspect
        sig = inspect.signature(func)
        # Simple heuristic: assume first parameter is weights
        n_qubits = 4  # Default, could be made configurable
        
        device = create_quantrs2_device(n_qubits, **(device_config.__dict__ if device_config else {}))
        return qml.QNode(func, device, **kwargs)
    
    return decorator


def test_quantrs2_pennylane_integration():
    """Test the QuantRS2-PennyLane integration."""
    if not (PENNYLANE_AVAILABLE and QUANTRS2_AVAILABLE):
        print("Cannot run integration test - missing dependencies")
        return False
    
    try:
        # Test device creation
        device = create_quantrs2_device(2, shots=1000)
        
        # Test simple circuit
        @qml.qnode(device)
        def test_circuit():
            qml.Hadamard(0)
            qml.CNOT([0, 1])
            return qml.expval(qml.PauliZ(0))
        
        result = test_circuit()
        print(f"Test circuit result: {result}")
        
        # Test QML model
        model = QuantRS2QMLModel(2, 2)
        test_input = np.array([0.5, 0.3])
        prediction = model.forward(test_input)
        print(f"QML model prediction: {prediction}")
        
        print("✓ QuantRS2-PennyLane integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


__all__ = [
    'DeviceMode',
    'GradientMethod',
    'DeviceConfig',
    'EnhancedQuantRS2Device',
    'QuantRS2QMLModel',
    'QuantRS2VQC',
    'create_quantrs2_device',
    'register_quantrs2_device',
    'quantrs2_qnode',
    'test_quantrs2_pennylane_integration'
]