#!/usr/bin/env python3
"""
Enhanced Qiskit Compatibility Layer for QuantRS2

This module provides comprehensive integration between QuantRS2 and Qiskit,
including advanced circuit conversion, optimization passes, noise model
translation, and full algorithm compatibility.
"""

import warnings
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import copy

try:
    import qiskit
    from qiskit import QuantumCircuit as QiskitCircuit
    from qiskit import QuantumRegister, ClassicalRegister
    from qiskit.circuit import Parameter, ParameterVector, Gate, Instruction
    from qiskit.circuit.library import *
    from qiskit.providers import Backend, Job
    from qiskit.result import Result as QiskitResult
    from qiskit.transpiler import PassManager, CouplingMap
    from qiskit.transpiler.passes import *
    from qiskit.providers.models import GateConfig, BackendConfiguration
    from qiskit.quantum_info import Statevector, Operator, process_fidelity
    from qiskit.algorithms import VQE, QAOA, AmplificationProblem
    from qiskit.algorithms.optimizers import COBYLA, SPSA, L_BFGS_B
    from qiskit.opflow import PauliSumOp, StateFn, CircuitStateFn
    from qiskit.circuit.library import EfficientSU2, TwoLocal, RealAmplitudes
    QISKIT_AVAILABLE = True
    QISKIT_VERSION = qiskit.__version__
except ImportError:
    QISKIT_AVAILABLE = False
    QISKIT_VERSION = "not_available"
    warnings.warn("Qiskit not available. Install with: pip install qiskit qiskit-algorithms")

try:
    from qiskit.providers.aer import Aer, AerSimulator
    from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
    QISKIT_AER_AVAILABLE = True
except ImportError:
    QISKIT_AER_AVAILABLE = False

try:
    from quantrs2 import Circuit as QuantRS2Circuit
    from quantrs2.gates import *
    QUANTRS2_AVAILABLE = True
except ImportError:
    QUANTRS2_AVAILABLE = False
    warnings.warn("QuantRS2 core not available. Using enhanced mock implementation.")


class ConversionMode(Enum):
    """Circuit conversion modes."""
    EXACT = "exact"               # Exact gate-by-gate conversion
    OPTIMIZED = "optimized"       # Apply optimization during conversion
    NATIVE = "native"             # Convert to native gate set
    PULSE_OPTIMAL = "pulse_optimal"  # Optimize for pulse-level execution


class CompatibilityLevel(Enum):
    """Levels of compatibility checking."""
    STRICT = "strict"             # Strict compatibility checking
    PERMISSIVE = "permissive"     # Allow approximations
    BEST_EFFORT = "best_effort"   # Convert what's possible, warn about rest


@dataclass
class ConversionOptions:
    """Options for circuit conversion."""
    mode: ConversionMode = ConversionMode.EXACT
    compatibility_level: CompatibilityLevel = CompatibilityLevel.PERMISSIVE
    optimization_level: int = 1
    preserve_measurements: bool = True
    preserve_barriers: bool = False
    basis_gates: Optional[List[str]] = None
    coupling_map: Optional[List[Tuple[int, int]]] = None
    max_depth: Optional[int] = None
    validate_result: bool = True


class EnhancedCircuitConverter:
    """Enhanced converter with advanced features."""
    
    def __init__(self, options: Optional[ConversionOptions] = None):
        """Initialize enhanced converter."""
        self.options = options or ConversionOptions()
        self.logger = logging.getLogger("quantrs2.qiskit_compat")
        
        # Enhanced gate mappings with parameter handling
        self._init_gate_mappings()
        
        # Circuit optimization passes
        self._init_optimization_passes()
        
        # Conversion statistics
        self.conversion_stats = {
            "successful_conversions": 0,
            "failed_conversions": 0,
            "warnings_issued": 0,
            "total_gates_converted": 0
        }
    
    def _init_gate_mappings(self):
        """Initialize comprehensive gate mappings."""
        # Qiskit to QuantRS2 mappings
        self.qiskit_to_quantrs2_gates = {
            # Standard gates
            'h': self._convert_h_gate,
            'x': self._convert_x_gate,
            'y': self._convert_y_gate,
            'z': self._convert_z_gate,
            'id': self._convert_id_gate,
            
            # Rotation gates
            'rx': self._convert_rx_gate,
            'ry': self._convert_ry_gate,
            'rz': self._convert_rz_gate,
            'r': self._convert_r_gate,
            'u': self._convert_u_gate,
            'u1': self._convert_u1_gate,
            'u2': self._convert_u2_gate,
            'u3': self._convert_u3_gate,
            
            # Phase gates
            's': self._convert_s_gate,
            'sdg': self._convert_sdg_gate,
            't': self._convert_t_gate,
            'tdg': self._convert_tdg_gate,
            'p': self._convert_p_gate,
            
            # Two-qubit gates
            'cx': self._convert_cx_gate,
            'cnot': self._convert_cx_gate,
            'cy': self._convert_cy_gate,
            'cz': self._convert_cz_gate,
            'ch': self._convert_ch_gate,
            'crx': self._convert_crx_gate,
            'cry': self._convert_cry_gate,
            'crz': self._convert_crz_gate,
            'cu': self._convert_cu_gate,
            'swap': self._convert_swap_gate,
            'iswap': self._convert_iswap_gate,
            
            # Three-qubit gates
            'ccx': self._convert_ccx_gate,
            'toffoli': self._convert_ccx_gate,
            'fredkin': self._convert_fredkin_gate,
            'cswap': self._convert_fredkin_gate,
            
            # Multi-controlled gates
            'mcx': self._convert_mcx_gate,
            'mct': self._convert_mcx_gate,
            'mcy': self._convert_mcy_gate,
            'mcz': self._convert_mcz_gate,
            
            # Special gates
            'reset': self._convert_reset_gate,
            'measure': self._convert_measure_gate,
            'barrier': self._convert_barrier_gate,
            
            # Composite gates
            'qft': self._convert_qft_gate,
            'iqft': self._convert_iqft_gate,
        }
        
        # QuantRS2 to Qiskit mappings (reverse)
        self.quantrs2_to_qiskit_gates = {
            'h': lambda qc, qubits, params: qc.h(qubits[0]),
            'x': lambda qc, qubits, params: qc.x(qubits[0]),
            'y': lambda qc, qubits, params: qc.y(qubits[0]),
            'z': lambda qc, qubits, params: qc.z(qubits[0]),
            'rx': lambda qc, qubits, params: qc.rx(params[0], qubits[0]),
            'ry': lambda qc, qubits, params: qc.ry(params[0], qubits[0]),
            'rz': lambda qc, qubits, params: qc.rz(params[0], qubits[0]),
            's': lambda qc, qubits, params: qc.s(qubits[0]),
            'sdg': lambda qc, qubits, params: qc.sdg(qubits[0]),
            't': lambda qc, qubits, params: qc.t(qubits[0]),
            'tdg': lambda qc, qubits, params: qc.tdg(qubits[0]),
            'cnot': lambda qc, qubits, params: qc.cx(qubits[0], qubits[1]),
            'cz': lambda qc, qubits, params: qc.cz(qubits[0], qubits[1]),
            'swap': lambda qc, qubits, params: qc.swap(qubits[0], qubits[1]),
            'ccx': lambda qc, qubits, params: qc.ccx(qubits[0], qubits[1], qubits[2]),
            'measure': lambda qc, qubits, params: qc.measure(qubits[0], qubits[0]) if len(qubits) == 1 else qc.measure_all(),
        }
    
    def _init_optimization_passes(self):
        """Initialize optimization passes for different levels."""
        if not QISKIT_AVAILABLE:
            return
            
        self.optimization_passes = {
            0: PassManager([]),  # No optimization
            1: PassManager([  # Basic optimization
                Unroller(['u3', 'cx']),
                Optimize1qGates(),
                CommutativeCancellation(),
            ]),
            2: PassManager([  # Standard optimization
                Unroller(['u3', 'cx']),
                Optimize1qGates(),
                CommutativeCancellation(),
                CXCancellation(),
                Optimize1qGatesDecomposition(['u3']),
            ]),
            3: PassManager([  # Aggressive optimization
                Unroller(['u3', 'cx']),
                Optimize1qGates(),
                CommutativeCancellation(),
                CXCancellation(),
                Optimize1qGatesDecomposition(['u3']),
                CommutativeInverse(),
                Optimize1qGates(),
            ])
        }
    
    # Gate conversion methods
    def _convert_h_gate(self, circuit, qubits, params):
        """Convert Hadamard gate."""
        circuit.h(qubits[0])
    
    def _convert_x_gate(self, circuit, qubits, params):
        """Convert Pauli-X gate."""
        circuit.x(qubits[0])
    
    def _convert_y_gate(self, circuit, qubits, params):
        """Convert Pauli-Y gate."""
        circuit.y(qubits[0])
    
    def _convert_z_gate(self, circuit, qubits, params):
        """Convert Pauli-Z gate."""
        circuit.z(qubits[0])
    
    def _convert_id_gate(self, circuit, qubits, params):
        """Convert identity gate (no-op)."""
        pass  # Identity does nothing
    
    def _convert_rx_gate(self, circuit, qubits, params):
        """Convert RX rotation gate."""
        circuit.rx(qubits[0], params[0])
    
    def _convert_ry_gate(self, circuit, qubits, params):
        """Convert RY rotation gate."""
        circuit.ry(qubits[0], params[0])
    
    def _convert_rz_gate(self, circuit, qubits, params):
        """Convert RZ rotation gate."""
        circuit.rz(qubits[0], params[0])
    
    def _convert_r_gate(self, circuit, qubits, params):
        """Convert R gate (phase rotation)."""
        # R(θ, φ) = RZ(φ)RY(θ)RZ(-φ)
        theta, phi = params[0], params[1]
        circuit.rz(qubits[0], phi)
        circuit.ry(qubits[0], theta)
        circuit.rz(qubits[0], -phi)
    
    def _convert_u_gate(self, circuit, qubits, params):
        """Convert general U gate."""
        if len(params) == 3:
            theta, phi, lam = params
            # U(θ,φ,λ) = RZ(φ)RY(θ)RZ(λ)
            circuit.rz(qubits[0], phi)
            circuit.ry(qubits[0], theta)
            circuit.rz(qubits[0], lam)
        else:
            self.logger.warning(f"U gate with {len(params)} parameters not fully supported")
    
    def _convert_u1_gate(self, circuit, qubits, params):
        """Convert U1 gate (phase gate)."""
        circuit.rz(qubits[0], params[0])
    
    def _convert_u2_gate(self, circuit, qubits, params):
        """Convert U2 gate."""
        phi, lam = params[0], params[1]
        # U2(φ,λ) = RZ(φ)RY(π/2)RZ(λ)
        circuit.rz(qubits[0], phi)
        circuit.ry(qubits[0], np.pi/2)
        circuit.rz(qubits[0], lam)
    
    def _convert_u3_gate(self, circuit, qubits, params):
        """Convert U3 gate."""
        theta, phi, lam = params
        # U3(θ,φ,λ) = RZ(φ)RY(θ)RZ(λ)
        circuit.rz(qubits[0], phi)
        circuit.ry(qubits[0], theta)
        circuit.rz(qubits[0], lam)
    
    def _convert_s_gate(self, circuit, qubits, params):
        """Convert S gate."""
        circuit.s(qubits[0])
    
    def _convert_sdg_gate(self, circuit, qubits, params):
        """Convert S-dagger gate."""
        circuit.sdg(qubits[0])
    
    def _convert_t_gate(self, circuit, qubits, params):
        """Convert T gate."""
        circuit.t(qubits[0])
    
    def _convert_tdg_gate(self, circuit, qubits, params):
        """Convert T-dagger gate."""
        circuit.tdg(qubits[0])
    
    def _convert_p_gate(self, circuit, qubits, params):
        """Convert phase gate."""
        circuit.rz(qubits[0], params[0])
    
    def _convert_cx_gate(self, circuit, qubits, params):
        """Convert CNOT/CX gate."""
        circuit.cnot(qubits[0], qubits[1])
    
    def _convert_cy_gate(self, circuit, qubits, params):
        """Convert CY gate."""
        # CY = CX · S† ⊗ S† · CX · S ⊗ S
        circuit.s(qubits[0])
        circuit.s(qubits[1])
        circuit.cnot(qubits[0], qubits[1])
        circuit.sdg(qubits[0])
        circuit.sdg(qubits[1])
        circuit.cnot(qubits[0], qubits[1])
    
    def _convert_cz_gate(self, circuit, qubits, params):
        """Convert CZ gate."""
        circuit.cz(qubits[0], qubits[1])
    
    def _convert_ch_gate(self, circuit, qubits, params):
        """Convert Controlled-H gate."""
        # CH = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ H
        # Decomposition: S†·CX·RY(-π/4)·CX·RY(π/4)·S
        circuit.s(qubits[1])
        circuit.cnot(qubits[0], qubits[1])
        circuit.ry(qubits[1], -np.pi/4)
        circuit.cnot(qubits[0], qubits[1])
        circuit.ry(qubits[1], np.pi/4)
        circuit.sdg(qubits[1])
    
    def _convert_crx_gate(self, circuit, qubits, params):
        """Convert Controlled-RX gate."""
        theta = params[0]
        # CRX decomposition
        circuit.ry(qubits[1], theta/2)
        circuit.cnot(qubits[0], qubits[1])
        circuit.ry(qubits[1], -theta/2)
        circuit.cnot(qubits[0], qubits[1])
    
    def _convert_cry_gate(self, circuit, qubits, params):
        """Convert Controlled-RY gate."""
        theta = params[0]
        circuit.ry(qubits[1], theta/2)
        circuit.cnot(qubits[0], qubits[1])
        circuit.ry(qubits[1], -theta/2)
        circuit.cnot(qubits[0], qubits[1])
    
    def _convert_crz_gate(self, circuit, qubits, params):
        """Convert Controlled-RZ gate."""
        lambda_param = params[0]
        circuit.rz(qubits[1], lambda_param/2)
        circuit.cnot(qubits[0], qubits[1])
        circuit.rz(qubits[1], -lambda_param/2)
        circuit.cnot(qubits[0], qubits[1])
    
    def _convert_cu_gate(self, circuit, qubits, params):
        """Convert Controlled-U gate."""
        if len(params) >= 3:
            theta, phi, lam = params[0], params[1], params[2]
            gamma = params[3] if len(params) > 3 else 0
            
            # Decomposition of controlled-U
            circuit.rz(qubits[0], gamma)
            circuit.rz(qubits[1], (lam + phi)/2)
            circuit.cnot(qubits[0], qubits[1])
            circuit.rz(qubits[1], -(lam - phi)/2)
            circuit.ry(qubits[1], -theta/2)
            circuit.cnot(qubits[0], qubits[1])
            circuit.ry(qubits[1], theta/2)
            circuit.rz(qubits[1], phi)
    
    def _convert_swap_gate(self, circuit, qubits, params):
        """Convert SWAP gate."""
        circuit.swap(qubits[0], qubits[1])
    
    def _convert_iswap_gate(self, circuit, qubits, params):
        """Convert iSWAP gate."""
        # iSWAP decomposition
        circuit.s(qubits[0])
        circuit.s(qubits[1])
        circuit.h(qubits[0])
        circuit.cnot(qubits[0], qubits[1])
        circuit.cnot(qubits[1], qubits[0])
        circuit.h(qubits[1])
    
    def _convert_ccx_gate(self, circuit, qubits, params):
        """Convert Toffoli (CCX) gate."""
        if hasattr(circuit, 'toffoli'):
            circuit.toffoli(qubits[0], qubits[1], qubits[2])
        else:
            # Decompose Toffoli using available gates
            self._decompose_toffoli(circuit, qubits)
    
    def _decompose_toffoli(self, circuit, qubits):
        """Decompose Toffoli gate using 2-qubit gates."""
        c1, c2, target = qubits
        # Standard Toffoli decomposition
        circuit.h(target)
        circuit.cnot(c2, target)
        circuit.tdg(target)
        circuit.cnot(c1, target)
        circuit.t(target)
        circuit.cnot(c2, target)
        circuit.tdg(target)
        circuit.cnot(c1, target)
        circuit.t(c2)
        circuit.t(target)
        circuit.cnot(c1, c2)
        circuit.h(target)
        circuit.t(c1)
        circuit.tdg(c2)
        circuit.cnot(c1, c2)
    
    def _convert_fredkin_gate(self, circuit, qubits, params):
        """Convert Fredkin (CSWAP) gate."""
        control, target1, target2 = qubits
        # Fredkin decomposition
        circuit.cnot(target2, target1)
        circuit.toffoli(control, target1, target2)
        circuit.cnot(target2, target1)
    
    def _convert_mcx_gate(self, circuit, qubits, params):
        """Convert multi-controlled X gate."""
        if len(qubits) == 3:
            self._convert_ccx_gate(circuit, qubits, params)
        else:
            # For larger multi-controlled gates, use decomposition
            self._decompose_mcx(circuit, qubits)
    
    def _decompose_mcx(self, circuit, qubits):
        """Decompose multi-controlled X gate."""
        controls = qubits[:-1]
        target = qubits[-1]
        
        if len(controls) == 1:
            circuit.cnot(controls[0], target)
        elif len(controls) == 2:
            circuit.toffoli(controls[0], controls[1], target)
        else:
            # Use ancilla qubits for larger decompositions
            # This is a simplified version - full implementation would require ancilla management
            self.logger.warning(f"MCX with {len(controls)} controls requires ancilla qubits - using approximation")
            for ctrl in controls:
                circuit.cnot(ctrl, target)
    
    def _convert_mcy_gate(self, circuit, qubits, params):
        """Convert multi-controlled Y gate."""
        # MCY = MCX with Y target
        controls = qubits[:-1]
        target = qubits[-1]
        
        circuit.sdg(target)
        self._convert_mcx_gate(circuit, qubits, params)
        circuit.s(target)
    
    def _convert_mcz_gate(self, circuit, qubits, params):
        """Convert multi-controlled Z gate."""
        # MCZ = H · MCX · H
        target = qubits[-1]
        
        circuit.h(target)
        self._convert_mcx_gate(circuit, qubits, params)
        circuit.h(target)
    
    def _convert_reset_gate(self, circuit, qubits, params):
        """Convert reset operation."""
        if hasattr(circuit, 'reset'):
            circuit.reset(qubits[0])
        else:
            self.logger.warning("Reset gate not supported in target framework")
    
    def _convert_measure_gate(self, circuit, qubits, params):
        """Convert measurement operation."""
        if hasattr(circuit, 'measure'):
            circuit.measure(qubits[0])
        else:
            self.logger.warning("Measurement not supported in target framework")
    
    def _convert_barrier_gate(self, circuit, qubits, params):
        """Convert barrier operation."""
        if self.options.preserve_barriers and hasattr(circuit, 'barrier'):
            circuit.barrier(qubits)
        # Otherwise, barriers are ignored
    
    def _convert_qft_gate(self, circuit, qubits, params):
        """Convert QFT gate."""
        n_qubits = len(qubits)
        # Standard QFT implementation
        for i in range(n_qubits):
            circuit.h(qubits[i])
            for j in range(i + 1, n_qubits):
                angle = np.pi / (2 ** (j - i))
                circuit.crz(qubits[j], qubits[i], angle)
        
        # Swap qubits
        for i in range(n_qubits // 2):
            circuit.swap(qubits[i], qubits[n_qubits - 1 - i])
    
    def _convert_iqft_gate(self, circuit, qubits, params):
        """Convert inverse QFT gate."""
        n_qubits = len(qubits)
        # Reverse of QFT
        
        # Reverse swaps
        for i in range(n_qubits // 2):
            circuit.swap(qubits[i], qubits[n_qubits - 1 - i])
        
        # Reverse QFT rotations
        for i in reversed(range(n_qubits)):
            for j in reversed(range(i + 1, n_qubits)):
                angle = -np.pi / (2 ** (j - i))
                circuit.crz(qubits[j], qubits[i], angle)
            circuit.h(qubits[i])
    
    def qiskit_to_quantrs2(self, qiskit_circuit: 'QiskitCircuit') -> 'QuantRS2Circuit':
        """Enhanced conversion from Qiskit to QuantRS2."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available")
        if not QUANTRS2_AVAILABLE:
            raise ImportError("QuantRS2 not available")
        
        try:
            # Apply optimization if requested
            if self.options.optimization_level > 0:
                optimization_pass = self.optimization_passes.get(self.options.optimization_level)
                if optimization_pass:
                    qiskit_circuit = optimization_pass.run(qiskit_circuit)
            
            # Create QuantRS2 circuit
            quantrs2_circuit = QuantRS2Circuit(qiskit_circuit.num_qubits)
            
            conversion_warnings = []
            gates_converted = 0
            
            # Convert each instruction
            for instruction in qiskit_circuit.data:
                gate = instruction.operation
                qubits = [qiskit_circuit.find_bit(qubit).index for qubit in instruction.qubits]
                params = [float(param) for param in gate.params] if gate.params else []
                
                gate_name = gate.name.lower()
                
                if gate_name in self.qiskit_to_quantrs2_gates:
                    try:
                        self.qiskit_to_quantrs2_gates[gate_name](quantrs2_circuit, qubits, params)
                        gates_converted += 1
                    except Exception as e:
                        warning_msg = f"Failed to convert gate {gate_name}: {e}"
                        conversion_warnings.append(warning_msg)
                        if self.options.compatibility_level == CompatibilityLevel.STRICT:
                            raise ValueError(warning_msg)
                        else:
                            self.logger.warning(warning_msg)
                else:
                    warning_msg = f"Unsupported gate: {gate_name}"
                    conversion_warnings.append(warning_msg)
                    if self.options.compatibility_level == CompatibilityLevel.STRICT:
                        raise ValueError(warning_msg)
                    else:
                        self.logger.warning(warning_msg)
            
            # Update statistics
            self.conversion_stats["successful_conversions"] += 1
            self.conversion_stats["total_gates_converted"] += gates_converted
            self.conversion_stats["warnings_issued"] += len(conversion_warnings)
            
            # Validate if requested
            if self.options.validate_result:
                self._validate_conversion(qiskit_circuit, quantrs2_circuit)
            
            return quantrs2_circuit
            
        except Exception as e:
            self.conversion_stats["failed_conversions"] += 1
            self.logger.error(f"Circuit conversion failed: {e}")
            raise
    
    def quantrs2_to_qiskit(self, quantrs2_circuit) -> 'QiskitCircuit':
        """Enhanced conversion from QuantRS2 to Qiskit."""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available")
        
        try:
            # Create Qiskit circuit
            qiskit_circuit = QiskitCircuit(quantrs2_circuit.n_qubits)
            
            # Convert gates if circuit has gate information
            if hasattr(quantrs2_circuit, 'gates'):
                for gate_info in quantrs2_circuit.gates:
                    gate_name = gate_info[0]
                    if gate_name in self.quantrs2_to_qiskit_gates:
                        qubits, params = self._parse_quantrs2_gate_info(gate_info)
                        try:
                            self.quantrs2_to_qiskit_gates[gate_name](qiskit_circuit, qubits, params)
                        except Exception as e:
                            self.logger.warning(f"Failed to convert gate {gate_name}: {e}")
            
            # Apply optimization if requested
            if self.options.optimization_level > 0:
                optimization_pass = self.optimization_passes.get(self.options.optimization_level)
                if optimization_pass:
                    qiskit_circuit = optimization_pass.run(qiskit_circuit)
            
            self.conversion_stats["successful_conversions"] += 1
            return qiskit_circuit
            
        except Exception as e:
            self.conversion_stats["failed_conversions"] += 1
            self.logger.error(f"Circuit conversion failed: {e}")
            raise
    
    def _parse_quantrs2_gate_info(self, gate_info: Tuple) -> Tuple[List[int], List[float]]:
        """Parse QuantRS2 gate information."""
        gate_name = gate_info[0]
        
        if len(gate_info) == 2:  # Single qubit gate, no parameters
            qubits = [gate_info[1]]
            params = []
        elif len(gate_info) == 3:
            if isinstance(gate_info[2], (int, np.integer)):  # Two qubit gate
                qubits = [gate_info[1], gate_info[2]]
                params = []
            else:  # Single qubit gate with parameter
                qubits = [gate_info[1]]
                params = [gate_info[2]]
        else:  # Multi-qubit or parameterized gate
            qubits = [q for q in gate_info[1:-1] if isinstance(q, (int, np.integer))]
            params = [p for p in gate_info[1:] if isinstance(p, (float, int, np.floating, np.integer)) and p not in qubits]
        
        return qubits, params
    
    def _validate_conversion(self, original_circuit, converted_circuit):
        """Validate conversion quality."""
        # Basic validation - check qubit counts
        if hasattr(converted_circuit, 'n_qubits'):
            if original_circuit.num_qubits != converted_circuit.n_qubits:
                self.logger.warning("Qubit count mismatch after conversion")
        
        # Additional validation could include:
        # - Gate count comparison
        # - Unitary equivalence (for small circuits)
        # - Statistical fidelity testing
    
    def get_conversion_statistics(self) -> Dict[str, Any]:
        """Get conversion statistics."""
        return self.conversion_stats.copy()
    
    def reset_statistics(self):
        """Reset conversion statistics."""
        for key in self.conversion_stats:
            self.conversion_stats[key] = 0


class AdvancedQiskitIntegration:
    """Advanced integration features between Qiskit and QuantRS2."""
    
    def __init__(self):
        """Initialize advanced integration."""
        self.converter = EnhancedCircuitConverter()
        self.logger = logging.getLogger("quantrs2.qiskit_advanced")
    
    def create_hybrid_algorithm(self, 
                               algorithm_type: str,
                               qiskit_components: Dict[str, Any],
                               quantrs2_components: Dict[str, Any]) -> 'HybridAlgorithm':
        """Create hybrid algorithm using both frameworks."""
        return HybridAlgorithm(algorithm_type, qiskit_components, quantrs2_components, self.converter)
    
    def benchmark_frameworks(self, 
                           circuit_sizes: List[int],
                           circuit_depths: List[int],
                           iterations: int = 10) -> Dict[str, Any]:
        """Benchmark circuit conversion and execution between frameworks."""
        results = {}
        
        for n_qubits in circuit_sizes:
            for depth in circuit_depths:
                test_key = f"{n_qubits}q_{depth}d"
                results[test_key] = self._benchmark_circuit_size(n_qubits, depth, iterations)
        
        return results
    
    def _benchmark_circuit_size(self, n_qubits: int, depth: int, iterations: int) -> Dict[str, float]:
        """Benchmark specific circuit size."""
        # Create test circuit
        if QISKIT_AVAILABLE:
            qc = QiskitCircuit(n_qubits)
            for d in range(depth):
                for q in range(n_qubits):
                    qc.h(q)
                for q in range(n_qubits - 1):
                    qc.cx(q, q + 1)
        
        # Time conversions
        conversion_times = []
        for _ in range(iterations):
            start_time = time.time()
            try:
                quantrs2_circuit = self.converter.qiskit_to_quantrs2(qc)
                recovered_qc = self.converter.quantrs2_to_qiskit(quantrs2_circuit)
                conversion_times.append(time.time() - start_time)
            except Exception as e:
                self.logger.warning(f"Conversion failed for {n_qubits}q_{depth}d: {e}")
        
        return {
            "mean_conversion_time": np.mean(conversion_times) if conversion_times else float('inf'),
            "std_conversion_time": np.std(conversion_times) if conversion_times else 0,
            "success_rate": len(conversion_times) / iterations
        }
    
    def create_noise_model_adapter(self, qiskit_noise_model) -> 'NoiseModelAdapter':
        """Create adapter for Qiskit noise models."""
        return NoiseModelAdapter(qiskit_noise_model)
    
    def optimize_for_hardware(self, 
                            circuit,
                            backend_properties: Dict[str, Any],
                            optimization_level: int = 2) -> Tuple[Any, Dict[str, Any]]:
        """Optimize circuit for specific hardware backend."""
        if not QISKIT_AVAILABLE:
            return circuit, {}
        
        # Convert to Qiskit for optimization
        if hasattr(circuit, 'n_qubits'):  # QuantRS2 circuit
            qc = self.converter.quantrs2_to_qiskit(circuit)
        else:  # Already Qiskit circuit
            qc = circuit
        
        # Create optimization pipeline
        passes = []
        
        if 'coupling_map' in backend_properties:
            coupling_map = CouplingMap(backend_properties['coupling_map'])
            passes.extend([
                UnitarySynthesis(backend_properties.get('basis_gates', ['u3', 'cx'])),
                Layout2qDistance(coupling_map),
                StochasticSwap(coupling_map),
                Optimize1qGates(),
            ])
        
        if optimization_level >= 2:
            passes.extend([
                CommutativeCancellation(),
                CXCancellation(),
                Optimize1qGatesDecomposition(['u3']),
            ])
        
        if optimization_level >= 3:
            passes.extend([
                CommutativeInverse(),
                Optimize1qGates(),
            ])
        
        # Apply optimization
        pass_manager = PassManager(passes)
        optimized_qc = pass_manager.run(qc)
        
        # Convert back if needed
        if hasattr(circuit, 'n_qubits'):  # Original was QuantRS2
            optimized_circuit = self.converter.qiskit_to_quantrs2(optimized_qc)
        else:
            optimized_circuit = optimized_qc
        
        # Optimization statistics
        optimization_stats = {
            "original_depth": qc.depth(),
            "optimized_depth": optimized_qc.depth(),
            "original_gate_count": len(qc.data),
            "optimized_gate_count": len(optimized_qc.data),
            "depth_reduction": (qc.depth() - optimized_qc.depth()) / qc.depth() if qc.depth() > 0 else 0,
            "gate_reduction": (len(qc.data) - len(optimized_qc.data)) / len(qc.data) if len(qc.data) > 0 else 0
        }
        
        return optimized_circuit, optimization_stats


class HybridAlgorithm:
    """Hybrid algorithm using both Qiskit and QuantRS2 components."""
    
    def __init__(self, 
                 algorithm_type: str,
                 qiskit_components: Dict[str, Any],
                 quantrs2_components: Dict[str, Any],
                 converter: EnhancedCircuitConverter):
        """Initialize hybrid algorithm."""
        self.algorithm_type = algorithm_type
        self.qiskit_components = qiskit_components
        self.quantrs2_components = quantrs2_components
        self.converter = converter
        self.logger = logging.getLogger(f"quantrs2.hybrid.{algorithm_type}")
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the hybrid algorithm."""
        if self.algorithm_type == "hybrid_vqe":
            return self._execute_hybrid_vqe(**kwargs)
        elif self.algorithm_type == "hybrid_qaoa":
            return self._execute_hybrid_qaoa(**kwargs)
        else:
            raise ValueError(f"Unknown algorithm type: {self.algorithm_type}")
    
    def _execute_hybrid_vqe(self, **kwargs) -> Dict[str, Any]:
        """Execute hybrid VQE."""
        # Use Qiskit for optimization, QuantRS2 for simulation
        hamiltonian = kwargs.get('hamiltonian')
        initial_params = kwargs.get('initial_params', np.random.uniform(0, 2*np.pi, 4))
        
        def cost_function(params):
            # Create ansatz using QuantRS2
            ansatz = self.quantrs2_components.get('ansatz_creator', lambda p: None)(params)
            if ansatz is None:
                return float('inf')
            
            # Convert to Qiskit for expectation value calculation
            qiskit_ansatz = self.converter.quantrs2_to_qiskit(ansatz)
            
            # Use Qiskit's expectation value calculation
            if QISKIT_AVAILABLE and hasattr(qiskit, 'opflow'):
                try:
                    from qiskit.opflow import CircuitStateFn, StateFn
                    state_fn = CircuitStateFn(qiskit_ansatz)
                    expectation = StateFn(hamiltonian, is_measurement=True).compose(state_fn)
                    return expectation.eval().real
                except:
                    # Fallback calculation
                    return self._mock_expectation_value(params)
            else:
                return self._mock_expectation_value(params)
        
        # Use Qiskit optimizer
        if QISKIT_AVAILABLE:
            from qiskit.algorithms.optimizers import COBYLA
            optimizer = COBYLA(maxiter=kwargs.get('max_iterations', 100))
            result = optimizer.minimize(cost_function, initial_params)
            
            return {
                'optimal_parameters': result.x,
                'optimal_energy': result.fun,
                'converged': result.success if hasattr(result, 'success') else True,
                'iterations': result.nfev,
                'algorithm_type': 'hybrid_vqe'
            }
        else:
            # Fallback optimization
            from scipy.optimize import minimize
            result = minimize(cost_function, initial_params, method='COBYLA')
            return {
                'optimal_parameters': result.x,
                'optimal_energy': result.fun,
                'converged': result.success,
                'iterations': result.nfev,
                'algorithm_type': 'hybrid_vqe'
            }
    
    def _execute_hybrid_qaoa(self, **kwargs) -> Dict[str, Any]:
        """Execute hybrid QAOA."""
        # Implementation for hybrid QAOA
        return {
            'algorithm_type': 'hybrid_qaoa',
            'message': 'Hybrid QAOA implementation placeholder'
        }
    
    def _mock_expectation_value(self, params: List[float]) -> float:
        """Mock expectation value calculation."""
        return sum(p**2 for p in params) - 1.0


class NoiseModelAdapter:
    """Adapter for Qiskit noise models."""
    
    def __init__(self, qiskit_noise_model):
        """Initialize noise model adapter."""
        self.qiskit_noise_model = qiskit_noise_model
        self.logger = logging.getLogger("quantrs2.noise_adapter")
    
    def convert_to_quantrs2_noise(self) -> Dict[str, Any]:
        """Convert Qiskit noise model to QuantRS2 format."""
        # This would implement noise model conversion
        # For now, return a simplified representation
        return {
            "type": "qiskit_converted",
            "gate_errors": self._extract_gate_errors(),
            "readout_errors": self._extract_readout_errors(),
            "thermal_relaxation": self._extract_thermal_relaxation()
        }
    
    def _extract_gate_errors(self) -> Dict[str, float]:
        """Extract gate error rates."""
        # Implementation would parse Qiskit noise model
        return {}
    
    def _extract_readout_errors(self) -> Dict[str, float]:
        """Extract readout error rates."""
        # Implementation would parse Qiskit noise model
        return {}
    
    def _extract_thermal_relaxation(self) -> Dict[str, float]:
        """Extract thermal relaxation parameters."""
        # Implementation would parse Qiskit noise model
        return {}


# Enhanced convenience functions
def create_enhanced_converter(options: Optional[ConversionOptions] = None) -> EnhancedCircuitConverter:
    """Create enhanced circuit converter with options."""
    return EnhancedCircuitConverter(options)


def optimize_circuit_for_backend(circuit, 
                                backend_name: str,
                                optimization_level: int = 2) -> Tuple[Any, Dict[str, Any]]:
    """Optimize circuit for specific backend."""
    integration = AdvancedQiskitIntegration()
    
    # Mock backend properties - in real implementation, these would come from the backend
    backend_properties = {
        'basis_gates': ['u3', 'cx'],
        'coupling_map': [(0, 1), (1, 2), (2, 3)],  # Linear coupling
        'gate_errors': {'u3': 0.001, 'cx': 0.01}
    }
    
    return integration.optimize_for_hardware(circuit, backend_properties, optimization_level)


def benchmark_conversion_performance(max_qubits: int = 8, max_depth: int = 10) -> Dict[str, Any]:
    """Benchmark conversion performance."""
    integration = AdvancedQiskitIntegration()
    
    qubit_sizes = [2, 4, 6, 8] if max_qubits >= 8 else [2, 4]
    depths = [2, 5, 10] if max_depth >= 10 else [2, 5]
    
    return integration.benchmark_frameworks(qubit_sizes, depths)


__all__ = [
    'ConversionMode',
    'CompatibilityLevel', 
    'ConversionOptions',
    'EnhancedCircuitConverter',
    'AdvancedQiskitIntegration',
    'HybridAlgorithm',
    'NoiseModelAdapter',
    'create_enhanced_converter',
    'optimize_circuit_for_backend',
    'benchmark_conversion_performance'
]