#!/usr/bin/env python3
"""
Test suite for quantum algorithm debugger functionality.
"""

import pytest
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

try:
    from quantrs2.algorithm_debugger import (
        DebugMode, ExecutionState, BreakpointType, QuantumState, Breakpoint,
        DebugSession, QuantumStateSimulator, QuantumAlgorithmDebugger,
        QuantumStateVisualizer, get_algorithm_debugger, debug_quantum_algorithm,
        set_gate_breakpoint, set_qubit_breakpoint
    )
    HAS_ALGORITHM_DEBUGGER = True
except ImportError:
    HAS_ALGORITHM_DEBUGGER = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@pytest.mark.skipif(not HAS_ALGORITHM_DEBUGGER, reason="algorithm_debugger module not available")
class TestQuantumState:
    """Test QuantumState functionality."""
    
    def test_quantum_state_creation(self):
        """Test creating quantum state."""
        amplitudes = [complex(1.0, 0.0), complex(0.0, 0.0)]
        
        state = QuantumState(
            step_number=1,
            amplitudes=amplitudes,
            n_qubits=1,
            gate_applied="h",
            qubits_affected=[0]
        )
        
        assert state.step_number == 1
        assert state.amplitudes == amplitudes
        assert state.n_qubits == 1
        assert state.gate_applied == "h"
        assert state.qubits_affected == [0]
        assert state.entanglement_entropy >= 0
    
    def test_probability_calculation(self):
        """Test probability calculation."""
        # |+⟩ state (equal superposition)
        sqrt2_inv = 1.0 / (2**0.5)
        amplitudes = [complex(sqrt2_inv, 0.0), complex(sqrt2_inv, 0.0)]
        
        state = QuantumState(
            step_number=1,
            amplitudes=amplitudes,
            n_qubits=1
        )
        
        assert '0' in state.probabilities
        assert '1' in state.probabilities
        assert abs(state.probabilities['0'] - 0.5) < 1e-10
        assert abs(state.probabilities['1'] - 0.5) < 1e-10
    
    def test_entanglement_entropy_calculation(self):
        """Test entanglement entropy calculation."""
        if not NUMPY_AVAILABLE:
            pytest.skip("NumPy not available")
        
        # Bell state |00⟩ + |11⟩
        sqrt2_inv = 1.0 / (2**0.5)
        amplitudes = [complex(sqrt2_inv, 0.0), complex(0.0, 0.0), 
                     complex(0.0, 0.0), complex(sqrt2_inv, 0.0)]
        
        state = QuantumState(
            step_number=1,
            amplitudes=amplitudes,
            n_qubits=2
        )
        
        # Bell state should have maximum entanglement (entropy = 1)
        assert abs(state.entanglement_entropy - 1.0) < 0.1
    
    def test_dominant_states(self):
        """Test getting dominant states."""
        # State with one dominant component
        amplitudes = [complex(0.9, 0.0), complex(0.436, 0.0)]  # ~81% and ~19%
        
        state = QuantumState(
            step_number=1,
            amplitudes=amplitudes,
            n_qubits=1
        )
        
        dominant = state.get_dominant_states(threshold=0.1)
        assert len(dominant) == 2  # Both states above 10%
        
        dominant_high = state.get_dominant_states(threshold=0.5)
        assert len(dominant_high) == 1  # Only one state above 50%
    
    def test_state_to_dict(self):
        """Test converting state to dictionary."""
        amplitudes = [complex(1.0, 0.0), complex(0.0, 0.0)]
        
        state = QuantumState(
            step_number=2,
            amplitudes=amplitudes,
            n_qubits=1,
            gate_applied="x"
        )
        
        state_dict = state.to_dict()
        
        assert state_dict['step_number'] == 2
        assert state_dict['n_qubits'] == 1
        assert state_dict['gate_applied'] == "x"
        assert 'probabilities' in state_dict
        assert 'dominant_states' in state_dict


@pytest.mark.skipif(not HAS_ALGORITHM_DEBUGGER, reason="algorithm_debugger module not available")
class TestBreakpoint:
    """Test Breakpoint functionality."""
    
    def test_breakpoint_creation(self):
        """Test creating breakpoint."""
        bp = Breakpoint(
            breakpoint_id="bp1",
            breakpoint_type=BreakpointType.GATE,
            condition="h",
            description="Break on Hadamard gate"
        )
        
        assert bp.breakpoint_id == "bp1"
        assert bp.breakpoint_type == BreakpointType.GATE
        assert bp.condition == "h"
        assert bp.enabled is True
        assert bp.hit_count == 0
    
    def test_gate_breakpoint_trigger(self):
        """Test gate breakpoint triggering."""
        bp = Breakpoint("bp1", BreakpointType.GATE, "h")
        
        # Should trigger on H gate
        gate_info = {'gate': 'h', 'qubits': [0]}
        assert bp.should_trigger(gate_info) is True
        
        # Should not trigger on X gate
        gate_info = {'gate': 'x', 'qubits': [0]}
        assert bp.should_trigger(gate_info) is False
    
    def test_qubit_breakpoint_trigger(self):
        """Test qubit breakpoint triggering."""
        bp = Breakpoint("bp1", BreakpointType.QUBIT, 1)
        
        # Should trigger when qubit 1 is involved
        gate_info = {'gate': 'h', 'qubits': [1]}
        assert bp.should_trigger(gate_info) is True
        
        gate_info = {'gate': 'cnot', 'qubits': [0, 1]}
        assert bp.should_trigger(gate_info) is True
        
        # Should not trigger when qubit 1 is not involved
        gate_info = {'gate': 'h', 'qubits': [0]}
        assert bp.should_trigger(gate_info) is False
    
    def test_condition_breakpoint_trigger(self):
        """Test condition breakpoint triggering."""
        def condition_func(gate_info):
            return gate_info.get('gate') == 'cnot' and 0 in gate_info.get('qubits', [])
        
        bp = Breakpoint("bp1", BreakpointType.CONDITION, condition_func)
        
        # Should trigger on CNOT with qubit 0
        gate_info = {'gate': 'cnot', 'qubits': [0, 1]}
        assert bp.should_trigger(gate_info) is True
        
        # Should not trigger on other gates
        gate_info = {'gate': 'h', 'qubits': [0]}
        assert bp.should_trigger(gate_info) is False
    
    def test_disabled_breakpoint(self):
        """Test disabled breakpoint behavior."""
        bp = Breakpoint("bp1", BreakpointType.GATE, "h", enabled=False)
        
        gate_info = {'gate': 'h', 'qubits': [0]}
        assert bp.should_trigger(gate_info) is False
    
    def test_breakpoint_hit_count(self):
        """Test breakpoint hit counting."""
        bp = Breakpoint("bp1", BreakpointType.GATE, "h")
        
        assert bp.hit_count == 0
        
        bp.trigger()
        assert bp.hit_count == 1
        
        bp.trigger()
        assert bp.hit_count == 2


@pytest.mark.skipif(not HAS_ALGORITHM_DEBUGGER, reason="algorithm_debugger module not available")
class TestQuantumStateSimulator:
    """Test QuantumStateSimulator functionality."""
    
    def setup_method(self):
        """Set up test simulator."""
        self.simulator = QuantumStateSimulator()
    
    def test_initial_state_creation(self):
        """Test creating initial state."""
        state = self.simulator.create_initial_state(2)
        
        assert len(state) == 4  # 2^2 states
        assert state[0] == complex(1.0, 0.0)  # |00⟩
        assert all(abs(amp) < 1e-10 for amp in state[1:])  # All others zero
    
    def test_pauli_x_gate(self):
        """Test Pauli-X gate application."""
        initial_state = [complex(1.0, 0.0), complex(0.0, 0.0)]  # |0⟩
        
        final_state = self.simulator.apply_gate(initial_state, "x", [0], [], 1)
        
        # Should flip to |1⟩
        assert abs(final_state[0]) < 1e-10
        assert abs(final_state[1] - 1.0) < 1e-10
    
    def test_hadamard_gate(self):
        """Test Hadamard gate application."""
        initial_state = [complex(1.0, 0.0), complex(0.0, 0.0)]  # |0⟩
        
        final_state = self.simulator.apply_gate(initial_state, "h", [0], [], 1)
        
        # Should create equal superposition |+⟩
        expected_amp = 1.0 / (2**0.5)
        assert abs(abs(final_state[0]) - expected_amp) < 1e-10
        assert abs(abs(final_state[1]) - expected_amp) < 1e-10
    
    def test_rotation_gate(self):
        """Test rotation gate application."""
        initial_state = [complex(1.0, 0.0), complex(0.0, 0.0)]  # |0⟩
        
        # π rotation should give |1⟩
        final_state = self.simulator.apply_gate(initial_state, "rx", [0], [3.14159], 1)
        
        # Should be close to |1⟩ (allowing for numerical precision)
        assert abs(final_state[0]) < 0.1
        assert abs(abs(final_state[1]) - 1.0) < 0.1
    
    def test_cnot_gate(self):
        """Test CNOT gate application."""
        # |10⟩ state
        initial_state = [complex(0.0, 0.0), complex(0.0, 0.0), 
                        complex(1.0, 0.0), complex(0.0, 0.0)]
        
        # CNOT(0,1) should flip target when control is 1
        final_state = self.simulator.apply_gate(initial_state, "cnot", [0, 1], [], 2)
        
        # Should become |11⟩
        assert abs(final_state[0]) < 1e-10
        assert abs(final_state[1]) < 1e-10
        assert abs(final_state[2]) < 1e-10
        assert abs(final_state[3] - 1.0) < 1e-10
    
    def test_fidelity_calculation(self):
        """Test fidelity calculation between states."""
        state1 = [complex(1.0, 0.0), complex(0.0, 0.0)]  # |0⟩
        state2 = [complex(0.0, 0.0), complex(1.0, 0.0)]  # |1⟩
        state3 = [complex(1.0, 0.0), complex(0.0, 0.0)]  # |0⟩ again
        
        # Orthogonal states should have zero fidelity
        fidelity12 = self.simulator.calculate_fidelity(state1, state2)
        assert abs(fidelity12) < 1e-10
        
        # Identical states should have fidelity 1
        fidelity13 = self.simulator.calculate_fidelity(state1, state3)
        assert abs(fidelity13 - 1.0) < 1e-10
    
    def test_gate_sequence(self):
        """Test applying sequence of gates."""
        # Start with |0⟩
        state = [complex(1.0, 0.0), complex(0.0, 0.0)]
        
        # Apply H then X
        state = self.simulator.apply_gate(state, "h", [0], [], 1)
        state = self.simulator.apply_gate(state, "x", [0], [], 1)
        
        # Should be |-⟩ state (or equivalent)
        # |−⟩ = (|0⟩ - |1⟩)/√2
        expected_amp = 1.0 / (2**0.5)
        assert abs(abs(state[0]) - expected_amp) < 1e-10
        assert abs(abs(state[1]) - expected_amp) < 1e-10


@pytest.mark.skipif(not HAS_ALGORITHM_DEBUGGER, reason="algorithm_debugger module not available")
class TestQuantumAlgorithmDebugger:
    """Test QuantumAlgorithmDebugger functionality."""
    
    def setup_method(self):
        """Set up test debugger."""
        self.debugger = QuantumAlgorithmDebugger()
    
    def test_debug_session_creation(self):
        """Test creating debug session."""
        circuit_data = {
            'n_qubits': 2,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]}
            ]
        }
        
        session_id = self.debugger.create_debug_session(circuit_data, DebugMode.STEP_BY_STEP)
        
        assert session_id is not None
        assert session_id in self.debugger.sessions
        
        session = self.debugger.sessions[session_id]
        assert session.circuit_data == circuit_data
        assert session.debug_mode == DebugMode.STEP_BY_STEP
        assert len(session.states_history) == 1  # Initial state
        assert session.execution_state == ExecutionState.STOPPED
    
    def test_step_by_step_execution(self):
        """Test step-by-step execution."""
        circuit_data = {
            'n_qubits': 1,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'x', 'qubits': [0]}
            ]
        }
        
        session_id = self.debugger.create_debug_session(circuit_data, DebugMode.STEP_BY_STEP)
        
        # Step 1: Apply H gate
        success = self.debugger.step_forward(session_id)
        assert success is True
        
        session = self.debugger.sessions[session_id]
        assert session.current_step == 1
        assert len(session.states_history) == 2
        assert session.states_history[-1].gate_applied == 'h'
        
        # Step 2: Apply X gate
        success = self.debugger.step_forward(session_id)
        assert success is True
        
        assert session.current_step == 2
        assert len(session.states_history) == 3
        assert session.states_history[-1].gate_applied == 'x'
    
    def test_backward_stepping(self):
        """Test stepping backward."""
        circuit_data = {
            'n_qubits': 1,
            'gates': [{'gate': 'h', 'qubits': [0]}]
        }
        
        session_id = self.debugger.create_debug_session(circuit_data)
        
        # Step forward
        self.debugger.step_forward(session_id)
        session = self.debugger.sessions[session_id]
        assert session.current_step == 1
        
        # Step backward
        success = self.debugger.step_backward(session_id)
        assert success is True
        assert session.current_step == 0
        assert len(session.states_history) == 1  # Back to initial state
    
    def test_breakpoint_functionality(self):
        """Test breakpoint setting and triggering."""
        circuit_data = {
            'n_qubits': 1,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'x', 'qubits': [0]},
                {'gate': 'h', 'qubits': [0]}
            ]
        }
        
        session_id = self.debugger.create_debug_session(circuit_data, DebugMode.BREAKPOINT)
        
        # Set breakpoint on X gate
        bp_id = self.debugger.set_breakpoint(session_id, BreakpointType.GATE, 'x')
        assert bp_id is not None
        
        # Run until breakpoint
        success = self.debugger.run_debug_session(session_id)
        assert success is True
        
        session = self.debugger.sessions[session_id]
        assert session.execution_state == ExecutionState.BREAKPOINT_HIT
        assert session.current_step == 2  # Should stop at X gate
        
        # Check breakpoint was triggered
        breakpoint = session.breakpoints[bp_id]
        assert breakpoint.hit_count == 1
    
    def test_continue_execution(self):
        """Test continuing execution after breakpoint."""
        circuit_data = {
            'n_qubits': 1,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'x', 'qubits': [0]}
            ]
        }
        
        session_id = self.debugger.create_debug_session(circuit_data)
        
        # Run to completion
        success = self.debugger.continue_execution(session_id)
        assert success is True
        
        session = self.debugger.sessions[session_id]
        assert session.execution_state == ExecutionState.COMPLETED
        assert session.current_step == 2
        assert len(session.states_history) == 3  # Initial + 2 gates
    
    def test_get_current_state(self):
        """Test getting current quantum state."""
        circuit_data = {
            'n_qubits': 1,
            'gates': [{'gate': 'h', 'qubits': [0]}]
        }
        
        session_id = self.debugger.create_debug_session(circuit_data)
        
        # Initial state
        initial_state = self.debugger.get_current_state(session_id)
        assert initial_state is not None
        assert initial_state.step_number == 0
        
        # After one step
        self.debugger.step_forward(session_id)
        current_state = self.debugger.get_current_state(session_id)
        assert current_state.step_number == 1
        assert current_state.gate_applied == 'h'
    
    def test_algorithm_analysis(self):
        """Test algorithm property analysis."""
        circuit_data = {
            'n_qubits': 2,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]},
                {'gate': 'h', 'qubits': [1]}
            ]
        }
        
        session_id = self.debugger.create_debug_session(circuit_data)
        self.debugger.continue_execution(session_id)
        
        analysis = self.debugger.analyze_algorithm_properties(session_id)
        
        assert analysis['total_steps'] == 3
        assert analysis['circuit_depth'] == 3
        assert analysis['two_qubit_gate_count'] == 1
        assert 'max_entanglement' in analysis
        assert 'final_state_probabilities' in analysis
        assert 'gate_sequence' in analysis
    
    def test_target_comparison(self):
        """Test comparing with target state."""
        circuit_data = {
            'n_qubits': 1,
            'gates': [{'gate': 'x', 'qubits': [0]}]
        }
        
        session_id = self.debugger.create_debug_session(circuit_data)
        self.debugger.continue_execution(session_id)
        
        # Target state |1⟩
        target_state = [complex(0.0, 0.0), complex(1.0, 0.0)]
        
        comparison = self.debugger.compare_with_target(session_id, target_state)
        
        assert 'fidelity' in comparison
        assert comparison['fidelity'] > 0.9  # Should be very close
        assert 'current_probabilities' in comparison
        assert 'target_probabilities' in comparison
    
    def test_trace_export(self):
        """Test exporting debug trace."""
        circuit_data = {
            'n_qubits': 1,
            'gates': [{'gate': 'h', 'qubits': [0]}]
        }
        
        session_id = self.debugger.create_debug_session(circuit_data)
        self.debugger.continue_execution(session_id)
        
        trace_json = self.debugger.export_debug_trace(session_id, "json")
        
        assert trace_json is not None
        
        # Should be valid JSON
        trace_data = json.loads(trace_json)
        assert 'session_info' in trace_data
        assert 'states_history' in trace_data
        assert 'analysis' in trace_data
    
    def test_session_cleanup(self):
        """Test session cleanup."""
        circuit_data = {'n_qubits': 1, 'gates': []}
        
        session_id = self.debugger.create_debug_session(circuit_data)
        assert session_id in self.debugger.sessions
        
        success = self.debugger.cleanup_session(session_id)
        assert success is True
        assert session_id not in self.debugger.sessions
    
    def test_breakpoint_removal(self):
        """Test removing breakpoints."""
        circuit_data = {'n_qubits': 1, 'gates': []}
        session_id = self.debugger.create_debug_session(circuit_data)
        
        # Add breakpoint
        bp_id = self.debugger.set_breakpoint(session_id, BreakpointType.GATE, 'h')
        session = self.debugger.sessions[session_id]
        assert bp_id in session.breakpoints
        
        # Remove breakpoint
        success = self.debugger.remove_breakpoint(session_id, bp_id)
        assert success is True
        assert bp_id not in session.breakpoints
    
    def test_multiple_sessions(self):
        """Test managing multiple debug sessions."""
        circuit1 = {'n_qubits': 1, 'gates': [{'gate': 'h', 'qubits': [0]}]}
        circuit2 = {'n_qubits': 2, 'gates': [{'gate': 'cnot', 'qubits': [0, 1]}]}
        
        session1 = self.debugger.create_debug_session(circuit1)
        session2 = self.debugger.create_debug_session(circuit2)
        
        assert session1 != session2
        assert session1 in self.debugger.sessions
        assert session2 in self.debugger.sessions
        
        # Step each session independently
        self.debugger.step_forward(session1)
        self.debugger.step_forward(session2)
        
        state1 = self.debugger.get_current_state(session1)
        state2 = self.debugger.get_current_state(session2)
        
        assert state1.n_qubits == 1
        assert state2.n_qubits == 2
        assert state1.gate_applied == 'h'
        assert state2.gate_applied == 'cnot'


@pytest.mark.skipif(not HAS_ALGORITHM_DEBUGGER or not MATPLOTLIB_AVAILABLE, 
                   reason="algorithm_debugger or matplotlib not available")
class TestQuantumStateVisualizer:
    """Test QuantumStateVisualizer functionality."""
    
    def setup_method(self):
        """Set up test visualizer."""
        self.visualizer = QuantumStateVisualizer()
    
    @patch('matplotlib.pyplot.show')
    def test_state_evolution_visualization(self, mock_show):
        """Test state evolution visualization."""
        # Create mock states
        states = []
        for i in range(3):
            amplitudes = [complex(1.0 - i*0.3, 0.0), complex(i*0.3, 0.0)]
            state = QuantumState(
                step_number=i,
                amplitudes=amplitudes,
                n_qubits=1,
                gate_applied=f"gate_{i}"
            )
            states.append(state)
        
        success = self.visualizer.plot_state_evolution(states)
        assert success is True
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    def test_circuit_visualization(self, mock_show):
        """Test circuit visualization."""
        circuit_data = {
            'n_qubits': 2,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]},
                {'gate': 'x', 'qubits': [1]}
            ]
        }
        
        success = self.visualizer.create_circuit_visualization(circuit_data, current_step=1)
        assert success is True
        mock_show.assert_called_once()


@pytest.mark.skipif(not HAS_ALGORITHM_DEBUGGER, reason="algorithm_debugger module not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_algorithm_debugger(self):
        """Test getting global debugger instance."""
        debugger1 = get_algorithm_debugger()
        debugger2 = get_algorithm_debugger()
        
        # Should be singleton
        assert debugger1 is debugger2
        assert isinstance(debugger1, QuantumAlgorithmDebugger)
    
    def test_debug_quantum_algorithm(self):
        """Test convenience function for debugging."""
        circuit_data = {
            'n_qubits': 1,
            'gates': [{'gate': 'h', 'qubits': [0]}]
        }
        
        session_id = debug_quantum_algorithm(circuit_data, DebugMode.STEP_BY_STEP)
        
        assert session_id is not None
        
        # Should be in global debugger
        debugger = get_algorithm_debugger()
        assert session_id in debugger.sessions
    
    def test_set_gate_breakpoint(self):
        """Test convenience function for gate breakpoint."""
        circuit_data = {'n_qubits': 1, 'gates': []}
        session_id = debug_quantum_algorithm(circuit_data)
        
        bp_id = set_gate_breakpoint(session_id, "h")
        
        assert bp_id is not None
        
        debugger = get_algorithm_debugger()
        session = debugger.sessions[session_id]
        assert bp_id in session.breakpoints
        assert session.breakpoints[bp_id].breakpoint_type == BreakpointType.GATE
    
    def test_set_qubit_breakpoint(self):
        """Test convenience function for qubit breakpoint."""
        circuit_data = {'n_qubits': 2, 'gates': []}
        session_id = debug_quantum_algorithm(circuit_data)
        
        bp_id = set_qubit_breakpoint(session_id, 1)
        
        assert bp_id is not None
        
        debugger = get_algorithm_debugger()
        session = debugger.sessions[session_id]
        assert bp_id in session.breakpoints
        assert session.breakpoints[bp_id].breakpoint_type == BreakpointType.QUBIT


@pytest.mark.skipif(not HAS_ALGORITHM_DEBUGGER, reason="algorithm_debugger module not available")
class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_invalid_session_id(self):
        """Test handling invalid session IDs."""
        debugger = QuantumAlgorithmDebugger()
        
        # Non-existent session
        current_state = debugger.get_current_state("invalid_session")
        assert current_state is None
        
        session_info = debugger.get_session_info("invalid_session")
        assert session_info is None
        
        success = debugger.step_forward("invalid_session")
        assert success is False
    
    def test_malformed_circuit_data(self):
        """Test handling malformed circuit data."""
        debugger = QuantumAlgorithmDebugger()
        
        # Empty circuit data
        empty_circuit = {}
        session_id = debugger.create_debug_session(empty_circuit)
        
        # Should handle gracefully
        assert session_id is not None
        success = debugger.continue_execution(session_id)
        # Should complete (nothing to execute)
        assert success is True
    
    def test_invalid_gate_application(self):
        """Test handling invalid gates."""
        debugger = QuantumAlgorithmDebugger()
        
        circuit_data = {
            'n_qubits': 1,
            'gates': [{'gate': 'invalid_gate', 'qubits': [0]}]
        }
        
        session_id = debugger.create_debug_session(circuit_data)
        
        # Should handle invalid gate gracefully
        success = debugger.continue_execution(session_id)
        assert success is True  # Should complete without crashing
    
    def test_qubit_index_out_of_range(self):
        """Test handling out-of-range qubit indices."""
        debugger = QuantumAlgorithmDebugger()
        
        circuit_data = {
            'n_qubits': 1,
            'gates': [{'gate': 'h', 'qubits': [5]}]  # Qubit 5 doesn't exist
        }
        
        session_id = debugger.create_debug_session(circuit_data)
        
        # Should handle gracefully
        success = debugger.continue_execution(session_id)
        assert success is True
    
    def test_breakpoint_errors(self):
        """Test breakpoint error handling."""
        debugger = QuantumAlgorithmDebugger()
        circuit_data = {'n_qubits': 1, 'gates': []}
        session_id = debugger.create_debug_session(circuit_data)
        
        # Try to remove non-existent breakpoint
        success = debugger.remove_breakpoint(session_id, "invalid_bp")
        assert success is False
        
        # Try to set breakpoint on invalid session
        try:
            debugger.set_breakpoint("invalid_session", BreakpointType.GATE, "h")
            assert False, "Should raise exception"
        except ValueError:
            pass  # Expected
    
    def test_visualizer_without_matplotlib(self):
        """Test visualizer behavior without matplotlib."""
        if MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib is available")
        
        debugger = QuantumAlgorithmDebugger()
        
        # Should handle missing matplotlib gracefully
        assert debugger.visualizer is None
        
        circuit_data = {'n_qubits': 1, 'gates': []}
        session_id = debugger.create_debug_session(circuit_data)
        
        success = debugger.visualize_state_evolution(session_id)
        assert success is False


@pytest.mark.skipif(not HAS_ALGORITHM_DEBUGGER, reason="algorithm_debugger module not available")
class TestAlgorithmDebuggingIntegration:
    """Test integration scenarios for algorithm debugging."""
    
    def test_bell_state_debugging(self):
        """Test debugging Bell state preparation."""
        # Bell state circuit: H on qubit 0, then CNOT(0,1)
        circuit_data = {
            'n_qubits': 2,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]}
            ]
        }
        
        debugger = QuantumAlgorithmDebugger()
        session_id = debugger.create_debug_session(circuit_data, DebugMode.STEP_BY_STEP)
        
        # Initial state should be |00⟩
        initial_state = debugger.get_current_state(session_id)
        assert initial_state.probabilities.get('00', 0) > 0.99
        
        # After H gate: should be |+0⟩ = (|00⟩ + |10⟩)/√2
        debugger.step_forward(session_id)
        after_h = debugger.get_current_state(session_id)
        assert abs(after_h.probabilities.get('00', 0) - 0.5) < 0.1
        assert abs(after_h.probabilities.get('10', 0) - 0.5) < 0.1
        
        # After CNOT: should be Bell state (|00⟩ + |11⟩)/√2
        debugger.step_forward(session_id)
        final_state = debugger.get_current_state(session_id)
        assert abs(final_state.probabilities.get('00', 0) - 0.5) < 0.1
        assert abs(final_state.probabilities.get('11', 0) - 0.5) < 0.1
        assert final_state.probabilities.get('01', 0) < 0.1
        assert final_state.probabilities.get('10', 0) < 0.1
        
        # Should have maximum entanglement
        if NUMPY_AVAILABLE:
            assert final_state.entanglement_entropy > 0.9  # Close to 1
    
    def test_quantum_fourier_transform_debugging(self):
        """Test debugging simple QFT-like circuit."""
        # Simple 2-qubit QFT-like circuit
        circuit_data = {
            'n_qubits': 2,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'rz', 'qubits': [0], 'params': [1.57]},  # π/2 rotation
                {'gate': 'cnot', 'qubits': [0, 1]},
                {'gate': 'h', 'qubits': [1]}
            ]
        }
        
        debugger = QuantumAlgorithmDebugger()
        session_id = debugger.create_debug_session(circuit_data)
        
        # Set breakpoint on last gate
        bp_id = debugger.set_breakpoint(session_id, BreakpointType.GATE, 'h')
        
        # Run until breakpoint
        success = debugger.run_debug_session(session_id)
        assert success is True
        
        session = debugger.sessions[session_id]
        assert session.execution_state == ExecutionState.BREAKPOINT_HIT
        
        # Continue to completion
        success = debugger.continue_execution(session_id)
        assert success is True
        
        # Analyze the algorithm
        analysis = debugger.analyze_algorithm_properties(session_id)
        assert analysis['total_steps'] == 4
        assert analysis['two_qubit_gate_count'] == 1
        
        # Should have some entanglement
        assert analysis['max_entanglement'] > 0
    
    def test_error_correction_code_debugging(self):
        """Test debugging a simple error correction encoding."""
        # Simple 3-qubit repetition code encoding
        circuit_data = {
            'n_qubits': 3,
            'gates': [
                {'gate': 'h', 'qubits': [0]},  # Prepare |+⟩ state
                {'gate': 'cnot', 'qubits': [0, 1]},  # Copy to qubit 1
                {'gate': 'cnot', 'qubits': [0, 2]}   # Copy to qubit 2
            ]
        }
        
        debugger = QuantumAlgorithmDebugger()
        session_id = debugger.create_debug_session(circuit_data)
        
        # Set breakpoint after each CNOT
        bp1 = debugger.set_breakpoint(session_id, BreakpointType.ITERATION, 2)  # After first CNOT
        bp2 = debugger.set_breakpoint(session_id, BreakpointType.ITERATION, 3)  # After second CNOT
        
        # Run to first breakpoint
        success = debugger.run_debug_session(session_id)
        assert success is True
        
        # Should be at step 2
        session = debugger.sessions[session_id]
        assert session.current_step == 2
        
        # Continue to second breakpoint
        success = debugger.continue_execution(session_id)
        assert success is True
        assert session.current_step == 3
        
        # Final state should be |+++⟩ or |---⟩ superposition
        final_state = debugger.get_current_state(session_id)
        
        # Should have equal probabilities for |000⟩ and |111⟩
        prob_000 = final_state.probabilities.get('000', 0)
        prob_111 = final_state.probabilities.get('111', 0)
        assert abs(prob_000 - 0.5) < 0.1
        assert abs(prob_111 - 0.5) < 0.1
    
    def test_variational_algorithm_debugging(self):
        """Test debugging variational quantum algorithm."""
        # Simple variational ansatz
        circuit_data = {
            'n_qubits': 2,
            'gates': [
                {'gate': 'ry', 'qubits': [0], 'params': [0.5]},
                {'gate': 'ry', 'qubits': [1], 'params': [1.0]},
                {'gate': 'cnot', 'qubits': [0, 1]},
                {'gate': 'rz', 'qubits': [0], 'params': [0.3]},
                {'gate': 'rz', 'qubits': [1], 'params': [0.7]}
            ]
        }
        
        debugger = QuantumAlgorithmDebugger()
        session_id = debugger.create_debug_session(circuit_data)
        
        # Run to completion
        success = debugger.continue_execution(session_id)
        assert success is True
        
        # Analyze properties
        analysis = debugger.analyze_algorithm_properties(session_id)
        
        # Should use both qubits
        qubit_usage = analysis['qubit_usage']
        assert '0' in qubit_usage['operations_per_qubit']
        assert '1' in qubit_usage['operations_per_qubit']
        
        # Should have parametric gates
        gate_sequence = analysis['gate_sequence']
        assert 'ry' in gate_sequence
        assert 'rz' in gate_sequence
        
        # Define target state for comparison
        target_state = [complex(0.0, 0.0), complex(0.0, 0.0), 
                       complex(0.0, 0.0), complex(1.0, 0.0)]  # |11⟩
        
        comparison = debugger.compare_with_target(session_id, target_state)
        assert 'fidelity' in comparison
        assert comparison['fidelity'] >= 0.0  # Valid fidelity
    
    def test_debug_session_persistence(self):
        """Test debug session state persistence."""
        circuit_data = {
            'n_qubits': 2,
            'gates': [
                {'gate': 'h', 'qubits': [0]},
                {'gate': 'cnot', 'qubits': [0, 1]},
                {'gate': 'x', 'qubits': [1]}
            ]
        }
        
        debugger = QuantumAlgorithmDebugger()
        session_id = debugger.create_debug_session(circuit_data)
        
        # Execute partially
        debugger.step_forward(session_id)
        debugger.step_forward(session_id)
        
        # Export trace
        trace_json = debugger.export_debug_trace(session_id)
        trace_data = json.loads(trace_json)
        
        # Should contain all states up to current point
        assert len(trace_data['states_history']) == 3  # Initial + 2 steps
        assert trace_data['session_info']['current_step'] == 2
        
        # Should be able to continue from current state
        debugger.step_forward(session_id)
        
        final_analysis = debugger.analyze_algorithm_properties(session_id)
        assert final_analysis['total_steps'] == 3


@pytest.mark.skipif(not HAS_ALGORITHM_DEBUGGER, reason="algorithm_debugger module not available")
class TestAlgorithmDebuggingPerformance:
    """Test performance characteristics of debugging."""
    
    def test_large_circuit_debugging(self):
        """Test debugging performance with larger circuits."""
        # Create circuit with many gates
        gates = []
        for i in range(50):  # 50 single-qubit gates
            gates.append({'gate': 'h' if i % 2 == 0 else 'x', 'qubits': [i % 3]})
        
        circuit_data = {
            'n_qubits': 3,
            'gates': gates
        }
        
        debugger = QuantumAlgorithmDebugger()
        
        start_time = time.time()
        session_id = debugger.create_debug_session(circuit_data)
        creation_time = time.time() - start_time
        
        # Session creation should be fast
        assert creation_time < 1.0
        
        start_time = time.time()
        success = debugger.continue_execution(session_id)
        execution_time = time.time() - start_time
        
        # Execution should complete
        assert success is True
        
        # Should complete within reasonable time
        assert execution_time < 5.0  # 5 seconds for 50 gates
        
        # Should have all states recorded
        session = debugger.sessions[session_id]
        assert len(session.states_history) == 51  # Initial + 50 gates
    
    def test_memory_usage_monitoring(self):
        """Test memory usage doesn't grow excessively."""
        debugger = QuantumAlgorithmDebugger()
        
        initial_session_count = len(debugger.sessions)
        
        # Create and clean up many sessions
        for i in range(20):
            circuit_data = {
                'n_qubits': 2,
                'gates': [{'gate': 'h', 'qubits': [0]}]
            }
            
            session_id = debugger.create_debug_session(circuit_data)
            debugger.continue_execution(session_id)
            debugger.cleanup_session(session_id)
        
        final_session_count = len(debugger.sessions)
        
        # Should not accumulate sessions
        assert final_session_count == initial_session_count
    
    def test_state_history_efficiency(self):
        """Test efficiency of state history storage."""
        circuit_data = {
            'n_qubits': 4,  # 16-dimensional state space
            'gates': [{'gate': 'h', 'qubits': [i]} for i in range(4)]  # 4 gates
        }
        
        debugger = QuantumAlgorithmDebugger()
        session_id = debugger.create_debug_session(circuit_data)
        
        start_time = time.time()
        success = debugger.continue_execution(session_id)
        total_time = time.time() - start_time
        
        assert success is True
        assert total_time < 2.0  # Should be fast even for 4-qubit system
        
        # Check memory efficiency
        session = debugger.sessions[session_id]
        assert len(session.states_history) == 5  # Initial + 4 gates
        
        # Each state should have reasonable size
        for state in session.states_history:
            assert len(state.amplitudes) == 16  # 2^4
            assert len(state.probabilities) <= 16


if __name__ == "__main__":
    pytest.main([__file__])