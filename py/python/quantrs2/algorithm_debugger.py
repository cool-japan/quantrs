"""
Quantum Algorithm Debugger

This module provides comprehensive debugging capabilities for quantum algorithms
with step-by-step execution, state inspection, and interactive debugging tools.
"""

import json
import time
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import threading

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from . import _quantrs2
    _NATIVE_AVAILABLE = True
except ImportError:
    _NATIVE_AVAILABLE = False


class DebugMode(Enum):
    """Debugging modes for quantum algorithms."""
    STEP_BY_STEP = "step_by_step"
    BREAKPOINT = "breakpoint"
    TRACE = "trace"
    PROFILE = "profile"


class ExecutionState(Enum):
    """Execution states for debugger."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    STEP = "step"
    BREAKPOINT_HIT = "breakpoint_hit"
    COMPLETED = "completed"
    ERROR = "error"


class BreakpointType(Enum):
    """Types of breakpoints."""
    GATE = "gate"
    QUBIT = "qubit"
    CONDITION = "condition"
    ITERATION = "iteration"


@dataclass
class QuantumState:
    """Represents a quantum state at a debugging step."""
    step_number: int
    amplitudes: List[complex]
    n_qubits: int
    probabilities: Dict[str, float] = field(default_factory=dict)
    entanglement_entropy: float = 0.0
    fidelity_to_target: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    gate_applied: Optional[str] = None
    qubits_affected: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived properties."""
        if self.amplitudes and not self.probabilities:
            self.probabilities = self._calculate_probabilities()
        
        if self.n_qubits > 1 and self.amplitudes:
            self.entanglement_entropy = self._calculate_entanglement_entropy()
    
    def _calculate_probabilities(self) -> Dict[str, float]:
        """Calculate state probabilities."""
        probs = {}
        for i, amp in enumerate(self.amplitudes):
            if i >= 2**self.n_qubits:
                break
            
            basis_state = format(i, f'0{self.n_qubits}b')
            prob = abs(amp)**2 if isinstance(amp, complex) else abs(amp)**2
            
            if prob > 1e-10:  # Only include significant probabilities
                probs[basis_state] = prob
        
        return probs
    
    def _calculate_entanglement_entropy(self) -> float:
        """Calculate von Neumann entanglement entropy."""
        if not NUMPY_AVAILABLE or self.n_qubits < 2:
            return 0.0
        
        try:
            # Convert amplitudes to numpy array
            state_vector = np.array(self.amplitudes, dtype=complex)
            
            # Reshape for bipartition (simple 1-qubit vs rest)
            dim_a = 2  # First qubit
            dim_b = 2**(self.n_qubits - 1)  # Rest
            
            state_matrix = state_vector.reshape(dim_a, dim_b)
            
            # Calculate reduced density matrix for subsystem A
            rho_a = np.dot(state_matrix, state_matrix.conj().T)
            
            # Calculate eigenvalues
            eigenvals = np.linalg.eigvals(rho_a)
            eigenvals = eigenvals[eigenvals > 1e-12]  # Remove zero eigenvalues
            
            # Calculate von Neumann entropy
            entropy = -np.sum(eigenvals * np.log2(eigenvals))
            return float(entropy)
            
        except Exception:
            return 0.0
    
    def get_dominant_states(self, threshold: float = 0.01) -> Dict[str, float]:
        """Get states with probability above threshold."""
        return {state: prob for state, prob in self.probabilities.items() 
                if prob >= threshold}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'step_number': self.step_number,
            'n_qubits': self.n_qubits,
            'probabilities': self.probabilities,
            'entanglement_entropy': self.entanglement_entropy,
            'fidelity_to_target': self.fidelity_to_target,
            'timestamp': self.timestamp,
            'gate_applied': self.gate_applied,
            'qubits_affected': self.qubits_affected,
            'dominant_states': self.get_dominant_states()
        }


@dataclass
class Breakpoint:
    """Represents a debugging breakpoint."""
    breakpoint_id: str
    breakpoint_type: BreakpointType
    condition: Any  # Gate name, qubit index, or condition function
    enabled: bool = True
    hit_count: int = 0
    description: str = ""
    
    def should_trigger(self, gate_info: Dict[str, Any]) -> bool:
        """Check if breakpoint should trigger."""
        if not self.enabled:
            return False
        
        if self.breakpoint_type == BreakpointType.GATE:
            return gate_info.get('gate') == self.condition
        elif self.breakpoint_type == BreakpointType.QUBIT:
            return self.condition in gate_info.get('qubits', [])
        elif self.breakpoint_type == BreakpointType.CONDITION:
            if callable(self.condition):
                try:
                    return self.condition(gate_info)
                except Exception:
                    return False
        elif self.breakpoint_type == BreakpointType.ITERATION:
            return gate_info.get('step_number', 0) == self.condition
        
        return False
    
    def trigger(self) -> None:
        """Trigger the breakpoint."""
        self.hit_count += 1


@dataclass
class DebugSession:
    """Represents a debugging session."""
    session_id: str
    circuit_data: Dict[str, Any]
    debug_mode: DebugMode
    created_at: float = field(default_factory=time.time)
    current_step: int = 0
    execution_state: ExecutionState = ExecutionState.STOPPED
    states_history: List[QuantumState] = field(default_factory=list)
    breakpoints: Dict[str, Breakpoint] = field(default_factory=dict)
    watchpoints: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'session_id': self.session_id,
            'circuit_data': self.circuit_data,
            'debug_mode': self.debug_mode.value,
            'created_at': self.created_at,
            'current_step': self.current_step,
            'execution_state': self.execution_state.value,
            'num_states': len(self.states_history),
            'num_breakpoints': len(self.breakpoints),
            'performance_metrics': self.performance_metrics
        }


class QuantumStateSimulator:
    """Simulates quantum states for debugging purposes."""
    
    def __init__(self):
        self.use_native = _NATIVE_AVAILABLE
    
    def create_initial_state(self, n_qubits: int) -> List[complex]:
        """Create initial |00...0⟩ state."""
        state_size = 2**n_qubits
        state = [complex(0.0, 0.0)] * state_size
        state[0] = complex(1.0, 0.0)  # |00...0⟩
        return state
    
    def apply_gate(self, state: List[complex], gate_name: str, 
                   qubits: List[int], params: List[float], n_qubits: int) -> List[complex]:
        """Apply a quantum gate to the state."""
        if self.use_native:
            return self._apply_gate_native(state, gate_name, qubits, params, n_qubits)
        else:
            return self._apply_gate_matrix(state, gate_name, qubits, params, n_qubits)
    
    def _apply_gate_native(self, state: List[complex], gate_name: str,
                          qubits: List[int], params: List[float], n_qubits: int) -> List[complex]:
        """Apply gate using native implementation."""
        try:
            # Create circuit and set state
            circuit = _quantrs2.PyCircuit(n_qubits)
            
            # Apply the gate
            gate_name = gate_name.lower()
            if gate_name == 'h' and len(qubits) >= 1:
                circuit.h(qubits[0])
            elif gate_name == 'x' and len(qubits) >= 1:
                circuit.x(qubits[0])
            elif gate_name == 'y' and len(qubits) >= 1:
                circuit.y(qubits[0])
            elif gate_name == 'z' and len(qubits) >= 1:
                circuit.z(qubits[0])
            elif gate_name == 's' and len(qubits) >= 1:
                circuit.s(qubits[0])
            elif gate_name == 't' and len(qubits) >= 1:
                circuit.t(qubits[0])
            elif gate_name == 'cnot' and len(qubits) >= 2:
                circuit.cnot(qubits[0], qubits[1])
            elif gate_name == 'cz' and len(qubits) >= 2:
                circuit.cz(qubits[0], qubits[1])
            elif gate_name == 'rx' and len(qubits) >= 1 and len(params) >= 1:
                circuit.rx(qubits[0], params[0])
            elif gate_name == 'ry' and len(qubits) >= 1 and len(params) >= 1:
                circuit.ry(qubits[0], params[0])
            elif gate_name == 'rz' and len(qubits) >= 1 and len(params) >= 1:
                circuit.rz(qubits[0], params[0])
            else:
                # Unsupported gate, return original state
                return state
            
            # Run simulation and get state
            result = circuit.run()
            if hasattr(result, 'amplitudes'):
                return result.amplitudes
            else:
                # Fallback to matrix simulation
                return self._apply_gate_matrix(state, gate_name, qubits, params, n_qubits)
            
        except Exception:
            # Fallback to matrix simulation
            return self._apply_gate_matrix(state, gate_name, qubits, params, n_qubits)
    
    def _apply_gate_matrix(self, state: List[complex], gate_name: str,
                          qubits: List[int], params: List[float], n_qubits: int) -> List[complex]:
        """Apply gate using matrix multiplication."""
        if not NUMPY_AVAILABLE:
            # Very basic simulation without numpy
            return self._apply_gate_basic(state, gate_name, qubits, params, n_qubits)
        
        try:
            state_vector = np.array(state, dtype=complex)
            
            # Get gate matrix
            gate_matrix = self._get_gate_matrix(gate_name, params)
            if gate_matrix is None:
                return state  # Unknown gate
            
            # Apply gate to specific qubits
            if len(qubits) == 1:
                new_state = self._apply_single_qubit_gate(state_vector, gate_matrix, qubits[0], n_qubits)
            elif len(qubits) == 2:
                new_state = self._apply_two_qubit_gate(state_vector, gate_matrix, qubits[0], qubits[1], n_qubits)
            else:
                return state  # Unsupported
            
            return new_state.tolist()
            
        except Exception:
            return self._apply_gate_basic(state, gate_name, qubits, params, n_qubits)
    
    def _apply_gate_basic(self, state: List[complex], gate_name: str,
                         qubits: List[int], params: List[float], n_qubits: int) -> List[complex]:
        """Basic gate application without numpy."""
        # Simple simulation for common gates
        new_state = state.copy()
        
        if gate_name.lower() == 'x' and len(qubits) == 1:
            # Pauli-X gate (bit flip)
            qubit = qubits[0]
            for i in range(len(state)):
                # Flip bit at position qubit
                flipped_i = i ^ (1 << (n_qubits - 1 - qubit))
                if i < len(new_state) and flipped_i < len(new_state):
                    new_state[flipped_i] = state[i]
        elif gate_name.lower() == 'h' and len(qubits) == 1:
            # Hadamard gate (simplified)
            qubit = qubits[0]
            sqrt2_inv = 1.0 / (2**0.5)
            for i in range(len(state)):
                flipped_i = i ^ (1 << (n_qubits - 1 - qubit))
                if i <= flipped_i:  # Process each pair only once
                    val_0 = state[i]
                    val_1 = state[flipped_i]
                    new_state[i] = complex(sqrt2_inv) * (val_0 + val_1)
                    new_state[flipped_i] = complex(sqrt2_inv) * (val_0 - val_1)
        
        return new_state
    
    def _get_gate_matrix(self, gate_name: str, params: List[float]) -> Optional[np.ndarray]:
        """Get matrix representation of a gate."""
        gate_name = gate_name.lower()
        
        if gate_name == 'i':
            return np.eye(2, dtype=complex)
        elif gate_name == 'x':
            return np.array([[0, 1], [1, 0]], dtype=complex)
        elif gate_name == 'y':
            return np.array([[0, -1j], [1j, 0]], dtype=complex)
        elif gate_name == 'z':
            return np.array([[1, 0], [0, -1]], dtype=complex)
        elif gate_name == 'h':
            return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        elif gate_name == 's':
            return np.array([[1, 0], [0, 1j]], dtype=complex)
        elif gate_name == 't':
            return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
        elif gate_name == 'rx' and len(params) >= 1:
            theta = params[0]
            return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                           [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
        elif gate_name == 'ry' and len(params) >= 1:
            theta = params[0]
            return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                           [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
        elif gate_name == 'rz' and len(params) >= 1:
            theta = params[0]
            return np.array([[np.exp(-1j*theta/2), 0],
                           [0, np.exp(1j*theta/2)]], dtype=complex)
        elif gate_name == 'cnot':
            return np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 1, 0]], dtype=complex)
        elif gate_name == 'cz':
            return np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, -1]], dtype=complex)
        else:
            return None
    
    def _apply_single_qubit_gate(self, state: np.ndarray, gate: np.ndarray, 
                                qubit: int, n_qubits: int) -> np.ndarray:
        """Apply single-qubit gate."""
        new_state = state.copy()
        
        for i in range(len(state)):
            # Extract bit value for target qubit
            bit_val = (i >> (n_qubits - 1 - qubit)) & 1
            
            # Find the paired state (with flipped bit)
            paired_i = i ^ (1 << (n_qubits - 1 - qubit))
            
            if i <= paired_i:  # Process each pair only once
                # Apply gate matrix
                old_0 = state[i if bit_val == 0 else paired_i]
                old_1 = state[paired_i if bit_val == 0 else i]
                
                new_0 = gate[0, 0] * old_0 + gate[0, 1] * old_1
                new_1 = gate[1, 0] * old_0 + gate[1, 1] * old_1
                
                new_state[i if bit_val == 0 else paired_i] = new_0
                new_state[paired_i if bit_val == 0 else i] = new_1
        
        return new_state
    
    def _apply_two_qubit_gate(self, state: np.ndarray, gate: np.ndarray,
                             control: int, target: int, n_qubits: int) -> np.ndarray:
        """Apply two-qubit gate."""
        new_state = state.copy()
        
        for i in range(len(state)):
            # Extract bit values for control and target qubits
            control_bit = (i >> (n_qubits - 1 - control)) & 1
            target_bit = (i >> (n_qubits - 1 - target)) & 1
            
            # Create basis state index
            basis_state = control_bit * 2 + target_bit  # 00, 01, 10, 11
            
            # Find all related states for this 2-qubit subspace
            base_i = i & ~((1 << (n_qubits - 1 - control)) | (1 << (n_qubits - 1 - target)))
            
            states = [
                base_i,  # 00
                base_i | (1 << (n_qubits - 1 - target)),  # 01
                base_i | (1 << (n_qubits - 1 - control)),  # 10
                base_i | (1 << (n_qubits - 1 - control)) | (1 << (n_qubits - 1 - target))  # 11
            ]
            
            # Only process if this is the lowest index in the group
            if i == min(states):
                # Get old amplitudes
                old_amps = [state[s] for s in states]
                
                # Apply gate matrix
                new_amps = gate @ np.array(old_amps)
                
                # Update state
                for j, s in enumerate(states):
                    new_state[s] = new_amps[j]
        
        return new_state
    
    def calculate_fidelity(self, state1: List[complex], state2: List[complex]) -> float:
        """Calculate fidelity between two quantum states."""
        if not NUMPY_AVAILABLE:
            # Simple overlap calculation
            overlap = sum(a.conjugate() * b for a, b in zip(state1, state2))
            return abs(overlap)**2
        
        try:
            s1 = np.array(state1, dtype=complex)
            s2 = np.array(state2, dtype=complex)
            
            # Normalize states
            s1 = s1 / np.linalg.norm(s1)
            s2 = s2 / np.linalg.norm(s2)
            
            # Calculate fidelity
            overlap = np.vdot(s1, s2)
            return abs(overlap)**2
            
        except Exception:
            return 0.0


class QuantumAlgorithmDebugger:
    """Main quantum algorithm debugger."""
    
    def __init__(self):
        self.sessions: Dict[str, DebugSession] = {}
        self.simulator = QuantumStateSimulator()
        self.visualizer = None
        self.logger = logging.getLogger(__name__)
        
        if MATPLOTLIB_AVAILABLE:
            self.visualizer = QuantumStateVisualizer()
    
    def create_debug_session(self, circuit_data: Dict[str, Any], 
                           debug_mode: DebugMode = DebugMode.STEP_BY_STEP,
                           session_id: Optional[str] = None) -> str:
        """Create a new debugging session."""
        if session_id is None:
            session_id = f"debug_{int(time.time() * 1000)}"
        
        session = DebugSession(
            session_id=session_id,
            circuit_data=circuit_data,
            debug_mode=debug_mode
        )
        
        self.sessions[session_id] = session
        
        # Initialize with starting state
        n_qubits = circuit_data.get('n_qubits', 1)
        initial_state = self.simulator.create_initial_state(n_qubits)
        
        initial_quantum_state = QuantumState(
            step_number=0,
            amplitudes=initial_state,
            n_qubits=n_qubits,
            gate_applied="initialization"
        )
        
        session.states_history.append(initial_quantum_state)
        
        return session_id
    
    def set_breakpoint(self, session_id: str, breakpoint_type: BreakpointType,
                      condition: Any, description: str = "") -> str:
        """Set a breakpoint in the debugging session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        breakpoint_id = f"bp_{len(self.sessions[session_id].breakpoints)}"
        
        breakpoint = Breakpoint(
            breakpoint_id=breakpoint_id,
            breakpoint_type=breakpoint_type,
            condition=condition,
            description=description
        )
        
        self.sessions[session_id].breakpoints[breakpoint_id] = breakpoint
        return breakpoint_id
    
    def remove_breakpoint(self, session_id: str, breakpoint_id: str) -> bool:
        """Remove a breakpoint."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        if breakpoint_id in session.breakpoints:
            del session.breakpoints[breakpoint_id]
            return True
        
        return False
    
    def run_debug_session(self, session_id: str, max_steps: Optional[int] = None) -> bool:
        """Run debugging session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.execution_state = ExecutionState.RUNNING
        
        try:
            gates = session.circuit_data.get('gates', [])
            n_qubits = session.circuit_data.get('n_qubits', 1)
            
            # Get current state
            current_state = session.states_history[-1].amplitudes
            
            # Process remaining gates
            start_step = session.current_step
            for step, gate_data in enumerate(gates[start_step:], start_step):
                if max_steps and step >= start_step + max_steps:
                    break
                
                session.current_step = step + 1
                
                # Check for breakpoints
                gate_info = {
                    'gate': gate_data.get('gate', ''),
                    'qubits': gate_data.get('qubits', []),
                    'params': gate_data.get('params', []),
                    'step_number': step + 1
                }
                
                breakpoint_hit = self._check_breakpoints(session, gate_info)
                
                if breakpoint_hit:
                    session.execution_state = ExecutionState.BREAKPOINT_HIT
                    return True
                
                # Apply gate
                current_state = self.simulator.apply_gate(
                    current_state,
                    gate_data.get('gate', ''),
                    gate_data.get('qubits', []),
                    gate_data.get('params', []),
                    n_qubits
                )
                
                # Record state
                quantum_state = QuantumState(
                    step_number=step + 1,
                    amplitudes=current_state,
                    n_qubits=n_qubits,
                    gate_applied=gate_data.get('gate', ''),
                    qubits_affected=gate_data.get('qubits', [])
                )
                
                session.states_history.append(quantum_state)
                
                # Handle step-by-step mode
                if session.debug_mode == DebugMode.STEP_BY_STEP:
                    session.execution_state = ExecutionState.PAUSED
                    return True
            
            # Completed execution
            session.execution_state = ExecutionState.COMPLETED
            return True
            
        except Exception as e:
            self.logger.error(f"Debug session error: {e}")
            session.execution_state = ExecutionState.ERROR
            return False
    
    def step_forward(self, session_id: str) -> bool:
        """Execute one step forward."""
        return self.run_debug_session(session_id, max_steps=1)
    
    def step_backward(self, session_id: str) -> bool:
        """Step backward in execution."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        if session.current_step > 0:
            session.current_step -= 1
            # Remove last state from history
            if len(session.states_history) > 1:
                session.states_history.pop()
            
            session.execution_state = ExecutionState.PAUSED
            return True
        
        return False
    
    def continue_execution(self, session_id: str) -> bool:
        """Continue execution until next breakpoint or completion."""
        return self.run_debug_session(session_id)
    
    def get_current_state(self, session_id: str) -> Optional[QuantumState]:
        """Get current quantum state."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        if session.states_history:
            return session.states_history[-1]
        
        return None
    
    def get_state_at_step(self, session_id: str, step: int) -> Optional[QuantumState]:
        """Get quantum state at specific step."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        for state in session.states_history:
            if state.step_number == step:
                return state
        
        return None
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get debugging session information."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        return session.to_dict()
    
    def analyze_algorithm_properties(self, session_id: str) -> Dict[str, Any]:
        """Analyze quantum algorithm properties."""
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        states = session.states_history
        
        if len(states) < 2:
            return {}
        
        analysis = {
            'total_steps': len(states) - 1,
            'max_entanglement': max(state.entanglement_entropy for state in states),
            'final_state_probabilities': states[-1].probabilities,
            'dominant_final_states': states[-1].get_dominant_states(),
            'entanglement_evolution': [state.entanglement_entropy for state in states],
            'gate_sequence': [state.gate_applied for state in states[1:]],
            'qubit_usage': self._analyze_qubit_usage(session),
            'circuit_depth': len(states) - 1,
            'two_qubit_gate_count': self._count_two_qubit_gates(session)
        }
        
        # Calculate state trajectory metrics
        if NUMPY_AVAILABLE and len(states) > 1:
            analysis['state_distances'] = self._calculate_state_distances(states)
            analysis['convergence_analysis'] = self._analyze_convergence(states)
        
        return analysis
    
    def compare_with_target(self, session_id: str, target_state: List[complex]) -> Dict[str, Any]:
        """Compare current state with target state."""
        current_state = self.get_current_state(session_id)
        
        if not current_state:
            return {}
        
        fidelity = self.simulator.calculate_fidelity(current_state.amplitudes, target_state)
        
        comparison = {
            'fidelity': fidelity,
            'distance': 1.0 - fidelity,
            'current_probabilities': current_state.probabilities,
            'target_probabilities': self._calculate_target_probabilities(target_state, current_state.n_qubits)
        }
        
        # Update current state with fidelity info
        current_state.fidelity_to_target = fidelity
        
        return comparison
    
    def export_debug_trace(self, session_id: str, format: str = "json") -> Optional[str]:
        """Export debugging trace."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        if format.lower() == "json":
            trace_data = {
                'session_info': session.to_dict(),
                'states_history': [state.to_dict() for state in session.states_history],
                'breakpoints': {bp_id: {
                    'type': bp.breakpoint_type.value,
                    'condition': str(bp.condition),
                    'hit_count': bp.hit_count,
                    'description': bp.description
                } for bp_id, bp in session.breakpoints.items()},
                'analysis': self.analyze_algorithm_properties(session_id)
            }
            
            return json.dumps(trace_data, indent=2)
        
        return None
    
    def visualize_state_evolution(self, session_id: str) -> bool:
        """Visualize quantum state evolution."""
        if not self.visualizer:
            return False
        
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        return self.visualizer.plot_state_evolution(session.states_history)
    
    def _check_breakpoints(self, session: DebugSession, gate_info: Dict[str, Any]) -> bool:
        """Check if any breakpoints should trigger."""
        for breakpoint in session.breakpoints.values():
            if breakpoint.should_trigger(gate_info):
                breakpoint.trigger()
                return True
        
        return False
    
    def _analyze_qubit_usage(self, session: DebugSession) -> Dict[str, Any]:
        """Analyze qubit usage patterns."""
        qubit_operations = {}
        gates = session.circuit_data.get('gates', [])
        
        for gate_data in gates:
            qubits = gate_data.get('qubits', [])
            gate_name = gate_data.get('gate', '')
            
            for qubit in qubits:
                if qubit not in qubit_operations:
                    qubit_operations[qubit] = []
                qubit_operations[qubit].append(gate_name)
        
        return {
            'operations_per_qubit': {str(q): len(ops) for q, ops in qubit_operations.items()},
            'gate_types_per_qubit': {str(q): list(set(ops)) for q, ops in qubit_operations.items()},
            'most_used_qubit': max(qubit_operations.keys(), key=lambda q: len(qubit_operations[q])) if qubit_operations else None
        }
    
    def _count_two_qubit_gates(self, session: DebugSession) -> int:
        """Count two-qubit gates in circuit."""
        count = 0
        gates = session.circuit_data.get('gates', [])
        
        for gate_data in gates:
            qubits = gate_data.get('qubits', [])
            if len(qubits) >= 2:
                count += 1
        
        return count
    
    def _calculate_state_distances(self, states: List[QuantumState]) -> List[float]:
        """Calculate distances between consecutive states."""
        distances = []
        
        for i in range(1, len(states)):
            prev_state = states[i-1].amplitudes
            curr_state = states[i].amplitudes
            
            # Calculate 2-norm distance
            if NUMPY_AVAILABLE:
                prev_vec = np.array(prev_state)
                curr_vec = np.array(curr_state)
                distance = np.linalg.norm(curr_vec - prev_vec)
                distances.append(float(distance))
            else:
                # Simple distance calculation
                distance = sum(abs(a - b)**2 for a, b in zip(prev_state, curr_state))**0.5
                distances.append(distance)
        
        return distances
    
    def _analyze_convergence(self, states: List[QuantumState]) -> Dict[str, Any]:
        """Analyze convergence properties."""
        if len(states) < 3:
            return {}
        
        # Look at probability distribution changes
        prob_changes = []
        for i in range(1, len(states)):
            prev_probs = states[i-1].probabilities
            curr_probs = states[i].probabilities
            
            # Calculate total variation distance
            all_states = set(prev_probs.keys()) | set(curr_probs.keys())
            change = sum(abs(curr_probs.get(s, 0) - prev_probs.get(s, 0)) for s in all_states) / 2
            prob_changes.append(change)
        
        return {
            'probability_changes': prob_changes,
            'converging': len(prob_changes) > 2 and all(
                prob_changes[i] >= prob_changes[i+1] for i in range(len(prob_changes)-1)
            ),
            'final_change': prob_changes[-1] if prob_changes else 0.0
        }
    
    def _calculate_target_probabilities(self, target_state: List[complex], n_qubits: int) -> Dict[str, float]:
        """Calculate probabilities for target state."""
        probs = {}
        for i, amp in enumerate(target_state):
            if i >= 2**n_qubits:
                break
            
            basis_state = format(i, f'0{n_qubits}b')
            prob = abs(amp)**2 if isinstance(amp, complex) else abs(amp)**2
            
            if prob > 1e-10:
                probs[basis_state] = prob
        
        return probs
    
    def cleanup_session(self, session_id: str) -> bool:
        """Clean up debugging session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False


class QuantumStateVisualizer:
    """Visualizes quantum states and debugging information."""
    
    def __init__(self):
        if not MATPLOTLIB_AVAILABLE:
            self.logger = logging.getLogger(__name__)
            self.logger.warning("Matplotlib not available, visualization disabled")
    
    def plot_state_evolution(self, states_history: List[QuantumState]) -> bool:
        """Plot evolution of quantum state probabilities."""
        if not MATPLOTLIB_AVAILABLE:
            return False
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Probability evolution
            self._plot_probability_evolution(axes[0, 0], states_history)
            
            # Plot 2: Entanglement entropy
            self._plot_entanglement_evolution(axes[0, 1], states_history)
            
            # Plot 3: Final state probabilities
            self._plot_final_probabilities(axes[1, 0], states_history[-1])
            
            # Plot 4: State distance evolution
            self._plot_state_distances(axes[1, 1], states_history)
            
            plt.tight_layout()
            plt.show()
            
            return True
            
        except Exception as e:
            logging.error(f"Visualization error: {e}")
            return False
    
    def _plot_probability_evolution(self, ax, states_history: List[QuantumState]) -> None:
        """Plot evolution of basis state probabilities."""
        if not states_history:
            return
        
        # Get all basis states that appear
        all_states = set()
        for state in states_history:
            all_states.update(state.probabilities.keys())
        
        # Plot top few states
        top_states = sorted(all_states, key=lambda s: states_history[-1].probabilities.get(s, 0), reverse=True)[:4]
        
        steps = [state.step_number for state in states_history]
        
        for basis_state in top_states:
            probs = [state.probabilities.get(basis_state, 0) for state in states_history]
            ax.plot(steps, probs, label=f'|{basis_state}⟩', marker='o', markersize=3)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Probability')
        ax.set_title('Basis State Probability Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_entanglement_evolution(self, ax, states_history: List[QuantumState]) -> None:
        """Plot evolution of entanglement entropy."""
        steps = [state.step_number for state in states_history]
        entropies = [state.entanglement_entropy for state in states_history]
        
        ax.plot(steps, entropies, 'b-', marker='o', markersize=3)
        ax.set_xlabel('Step')
        ax.set_ylabel('Entanglement Entropy')
        ax.set_title('Entanglement Evolution')
        ax.grid(True, alpha=0.3)
    
    def _plot_final_probabilities(self, ax, final_state: QuantumState) -> None:
        """Plot final state probabilities as bar chart."""
        probs = final_state.get_dominant_states(threshold=0.01)
        
        if probs:
            states = list(probs.keys())
            values = list(probs.values())
            
            bars = ax.bar(states, values)
            ax.set_xlabel('Basis State')
            ax.set_ylabel('Probability')
            ax.set_title('Final State Probabilities')
            
            # Color bars by probability
            max_prob = max(values)
            for bar, val in zip(bars, values):
                bar.set_color(plt.cm.viridis(val / max_prob))
            
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No significant probabilities', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_state_distances(self, ax, states_history: List[QuantumState]) -> None:
        """Plot distances between consecutive states."""
        if len(states_history) < 2:
            return
        
        if not NUMPY_AVAILABLE:
            ax.text(0.5, 0.5, 'NumPy required for distance calculation', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        distances = []
        steps = []
        
        for i in range(1, len(states_history)):
            prev_state = np.array(states_history[i-1].amplitudes)
            curr_state = np.array(states_history[i].amplitudes)
            
            distance = np.linalg.norm(curr_state - prev_state)
            distances.append(distance)
            steps.append(states_history[i].step_number)
        
        ax.plot(steps, distances, 'r-', marker='o', markersize=3)
        ax.set_xlabel('Step')
        ax.set_ylabel('State Distance')
        ax.set_title('State Change per Step')
        ax.grid(True, alpha=0.3)
    
    def create_circuit_visualization(self, circuit_data: Dict[str, Any], 
                                   current_step: int = -1) -> bool:
        """Create a visual representation of the quantum circuit."""
        if not MATPLOTLIB_AVAILABLE:
            return False
        
        try:
            n_qubits = circuit_data.get('n_qubits', 1)
            gates = circuit_data.get('gates', [])
            
            fig, ax = plt.subplots(figsize=(max(8, len(gates)), max(4, n_qubits)))
            
            # Draw qubit lines
            for i in range(n_qubits):
                ax.plot([0, len(gates) + 1], [i, i], 'k-', linewidth=2)
                ax.text(-0.2, i, f'q{i}', ha='right', va='center', fontsize=12)
            
            # Draw gates
            for step, gate_data in enumerate(gates):
                gate_name = gate_data.get('gate', '')
                qubits = gate_data.get('qubits', [])
                
                x_pos = step + 1
                
                if len(qubits) == 1:
                    # Single-qubit gate
                    qubit = qubits[0]
                    color = 'lightblue' if step < current_step else 'lightgray'
                    if step == current_step:
                        color = 'yellow'
                    
                    rect = patches.Rectangle((x_pos - 0.2, qubit - 0.2), 0.4, 0.4, 
                                           linewidth=1, edgecolor='black', facecolor=color)
                    ax.add_patch(rect)
                    ax.text(x_pos, qubit, gate_name.upper(), ha='center', va='center', fontsize=10)
                
                elif len(qubits) == 2:
                    # Two-qubit gate
                    q1, q2 = sorted(qubits)
                    color = 'lightcoral' if step < current_step else 'lightgray'
                    if step == current_step:
                        color = 'yellow'
                    
                    # Draw connection line
                    ax.plot([x_pos, x_pos], [q1, q2], 'k-', linewidth=3)
                    
                    # Draw control and target
                    ax.plot(x_pos, qubits[0], 'ko', markersize=8)  # Control
                    
                    if gate_name.lower() == 'cnot':
                        circle = plt.Circle((x_pos, qubits[1]), 0.15, color=color, ec='black')
                        ax.add_patch(circle)
                        ax.plot([x_pos - 0.1, x_pos + 0.1], [qubits[1], qubits[1]], 'k-', linewidth=2)
                        ax.plot([x_pos, x_pos], [qubits[1] - 0.1, qubits[1] + 0.1], 'k-', linewidth=2)
                    else:
                        rect = patches.Rectangle((x_pos - 0.15, qubits[1] - 0.15), 0.3, 0.3,
                                               linewidth=1, edgecolor='black', facecolor=color)
                        ax.add_patch(rect)
                        ax.text(x_pos, qubits[1], gate_name.upper(), ha='center', va='center', fontsize=8)
            
            ax.set_xlim(-0.5, len(gates) + 1.5)
            ax.set_ylim(-0.5, n_qubits - 0.5)
            ax.set_aspect('equal')
            ax.set_title(f'Quantum Circuit (Step {current_step}/{len(gates)})')
            ax.set_xlabel('Gate Sequence')
            ax.set_ylabel('Qubits')
            
            # Remove ticks
            ax.set_xticks(range(1, len(gates) + 1))
            ax.set_xticklabels([f'G{i}' for i in range(1, len(gates) + 1)])
            ax.set_yticks(range(n_qubits))
            ax.set_yticklabels([f'q{i}' for i in range(n_qubits)])
            
            plt.tight_layout()
            plt.show()
            
            return True
            
        except Exception as e:
            logging.error(f"Circuit visualization error: {e}")
            return False


# Global debugger instance
_algorithm_debugger: Optional[QuantumAlgorithmDebugger] = None


def get_algorithm_debugger() -> QuantumAlgorithmDebugger:
    """Get global algorithm debugger instance."""
    global _algorithm_debugger
    if _algorithm_debugger is None:
        _algorithm_debugger = QuantumAlgorithmDebugger()
    return _algorithm_debugger


def debug_quantum_algorithm(circuit_data: Dict[str, Any], 
                          debug_mode: DebugMode = DebugMode.STEP_BY_STEP) -> str:
    """Convenience function to start debugging a quantum algorithm."""
    debugger = get_algorithm_debugger()
    return debugger.create_debug_session(circuit_data, debug_mode)


def set_gate_breakpoint(session_id: str, gate_name: str) -> str:
    """Convenience function to set a gate breakpoint."""
    debugger = get_algorithm_debugger()
    return debugger.set_breakpoint(session_id, BreakpointType.GATE, gate_name, 
                                 f"Break on {gate_name} gate")


def set_qubit_breakpoint(session_id: str, qubit: int) -> str:
    """Convenience function to set a qubit breakpoint."""
    debugger = get_algorithm_debugger()
    return debugger.set_breakpoint(session_id, BreakpointType.QUBIT, qubit,
                                 f"Break on qubit {qubit}")


# CLI interface
def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantRS2 Algorithm Debugger")
    parser.add_argument("--circuit-file", help="Circuit file to debug (JSON)")
    parser.add_argument("--mode", choices=["step", "breakpoint", "trace"], 
                       default="step", help="Debug mode")
    parser.add_argument("--visualize", action="store_true", 
                       help="Enable visualization")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive debugging mode")
    
    args = parser.parse_args()
    
    if not args.circuit_file:
        parser.print_help()
        return 1
    
    try:
        with open(args.circuit_file, 'r') as f:
            circuit_data = json.load(f)
        
        # Map mode
        mode_map = {
            'step': DebugMode.STEP_BY_STEP,
            'breakpoint': DebugMode.BREAKPOINT,
            'trace': DebugMode.TRACE
        }
        debug_mode = mode_map.get(args.mode, DebugMode.STEP_BY_STEP)
        
        # Start debugging session
        session_id = debug_quantum_algorithm(circuit_data, debug_mode)
        debugger = get_algorithm_debugger()
        
        print(f"Started debugging session: {session_id}")
        print(f"Circuit: {circuit_data.get('n_qubits', 0)} qubits, {len(circuit_data.get('gates', []))} gates")
        
        if args.interactive:
            # Interactive mode
            while True:
                print("\nDebugger Commands:")
                print("  s - Step forward")
                print("  c - Continue execution")
                print("  b - Step backward")
                print("  i - Show current state info")
                print("  a - Show analysis")
                print("  v - Visualize (if available)")
                print("  q - Quit")
                
                cmd = input("Debug> ").strip().lower()
                
                if cmd == 'q':
                    break
                elif cmd == 's':
                    success = debugger.step_forward(session_id)
                    if success:
                        state = debugger.get_current_state(session_id)
                        print(f"Step {state.step_number}: Applied {state.gate_applied}")
                    else:
                        print("Cannot step forward")
                elif cmd == 'c':
                    success = debugger.continue_execution(session_id)
                    if success:
                        state = debugger.get_current_state(session_id)
                        session_info = debugger.get_session_info(session_id)
                        print(f"Execution status: {session_info['execution_state']}")
                    else:
                        print("Execution failed")
                elif cmd == 'b':
                    success = debugger.step_backward(session_id)
                    if success:
                        state = debugger.get_current_state(session_id)
                        print(f"Stepped back to step {state.step_number}")
                    else:
                        print("Cannot step backward")
                elif cmd == 'i':
                    state = debugger.get_current_state(session_id)
                    if state:
                        print(f"Step: {state.step_number}")
                        print(f"Entanglement entropy: {state.entanglement_entropy:.4f}")
                        print("Dominant states:")
                        for basis_state, prob in state.get_dominant_states().items():
                            print(f"  |{basis_state}⟩: {prob:.4f}")
                elif cmd == 'a':
                    analysis = debugger.analyze_algorithm_properties(session_id)
                    print("Algorithm Analysis:")
                    print(f"  Total steps: {analysis.get('total_steps', 0)}")
                    print(f"  Max entanglement: {analysis.get('max_entanglement', 0):.4f}")
                    print(f"  Two-qubit gates: {analysis.get('two_qubit_gate_count', 0)}")
                elif cmd == 'v':
                    if args.visualize:
                        success = debugger.visualize_state_evolution(session_id)
                        if not success:
                            print("Visualization not available")
                    else:
                        print("Visualization not enabled")
                else:
                    print("Unknown command")
        else:
            # Non-interactive mode - run to completion
            success = debugger.continue_execution(session_id)
            
            if success:
                analysis = debugger.analyze_algorithm_properties(session_id)
                print("\nExecution completed successfully!")
                print(f"Total steps: {analysis.get('total_steps', 0)}")
                print(f"Max entanglement: {analysis.get('max_entanglement', 0):.4f}")
                
                final_state = debugger.get_current_state(session_id)
                if final_state:
                    print("\nFinal state probabilities:")
                    for basis_state, prob in final_state.get_dominant_states().items():
                        print(f"  |{basis_state}⟩: {prob:.4f}")
                
                if args.visualize:
                    debugger.visualize_state_evolution(session_id)
            else:
                print("Execution failed")
                return 1
        
        # Cleanup
        debugger.cleanup_session(session_id)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())