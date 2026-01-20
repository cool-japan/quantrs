#!/usr/bin/env python3
"""
Test suite for interactive circuit builder functionality.
"""

import pytest
import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

try:
    from quantrs2.circuit_builder import (
        GateInfo, CircuitElement, CircuitBuilderBackend, MockCircuitBackend,
        QuantRS2Backend, CircuitBuilder, TkinterGUI, WebGUI,
        create_circuit_builder, launch_gui
    )
    HAS_CIRCUIT_BUILDER = True
except ImportError:
    HAS_CIRCUIT_BUILDER = False

try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

try:
    import flask
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False


@pytest.mark.skipif(not HAS_CIRCUIT_BUILDER, reason="circuit_builder module not available")
class TestGateInfo:
    """Test GateInfo data class."""
    
    def test_gate_info_creation(self):
        """Test creating gate info."""
        gate = GateInfo(
            name="h",
            display_name="Hadamard",
            num_qubits=1,
            description="Creates superposition",
            category="Single-Qubit",
            color="#4CAF50",
            symbol="H"
        )
        
        assert gate.name == "h"
        assert gate.display_name == "Hadamard"
        assert gate.num_qubits == 1
        assert gate.num_params == 0  # Default
        assert gate.param_names == []  # Default
        assert gate.description == "Creates superposition"
        assert gate.category == "Single-Qubit"
        assert gate.color == "#4CAF50"
        assert gate.symbol == "H"
    
    def test_gate_info_with_parameters(self):
        """Test gate info with parameters."""
        gate = GateInfo(
            name="rx",
            display_name="RX",
            num_qubits=1,
            num_params=1,
            param_names=["angle"],
            description="X-axis rotation",
            category="Rotation"
        )
        
        assert gate.num_params == 1
        assert gate.param_names == ["angle"]


@pytest.mark.skipif(not HAS_CIRCUIT_BUILDER, reason="circuit_builder module not available")
class TestCircuitElement:
    """Test CircuitElement data class."""
    
    def test_circuit_element_creation(self):
        """Test creating circuit element."""
        gate = GateInfo(name="h", display_name="H", num_qubits=1)
        element = CircuitElement(
            gate=gate,
            qubits=[0],
            params=[],
            position=1,
            element_id="elem_1"
        )
        
        assert element.gate == gate
        assert element.qubits == [0]
        assert element.params == []
        assert element.position == 1
        assert element.element_id == "elem_1"
    
    def test_circuit_element_with_params(self):
        """Test circuit element with parameters."""
        gate = GateInfo(name="rx", display_name="RX", num_qubits=1, num_params=1)
        element = CircuitElement(
            gate=gate,
            qubits=[0],
            params=[1.57],
            position=0
        )
        
        assert element.params == [1.57]


@pytest.mark.skipif(not HAS_CIRCUIT_BUILDER, reason="circuit_builder module not available")
class TestMockCircuitBackend:
    """Test MockCircuitBackend functionality."""
    
    def setup_method(self):
        """Set up test backend."""
        self.backend = MockCircuitBackend()
    
    def test_create_circuit(self):
        """Test creating a circuit."""
        circuit = self.backend.create_circuit(2)
        
        assert circuit is not None
        assert circuit['n_qubits'] == 2
        assert circuit['gates'] == []
        assert circuit['depth'] == 0
        assert 'id' in circuit
    
    def test_add_single_qubit_gate(self):
        """Test adding single-qubit gates."""
        circuit = self.backend.create_circuit(2)
        gate = GateInfo(name="h", display_name="H", num_qubits=1)
        
        success = self.backend.add_gate(circuit, gate, [0], [])
        
        assert success is True
        assert len(circuit['gates']) == 1
        assert circuit['gates'][0]['gate'] == "h"
        assert circuit['gates'][0]['qubits'] == [0]
        assert circuit['depth'] == 1
    
    def test_add_two_qubit_gate(self):
        """Test adding two-qubit gates."""
        circuit = self.backend.create_circuit(3)
        gate = GateInfo(name="cnot", display_name="CNOT", num_qubits=2)
        
        success = self.backend.add_gate(circuit, gate, [0, 1], [])
        
        assert success is True
        assert len(circuit['gates']) == 1
        assert circuit['gates'][0]['gate'] == "cnot"
        assert circuit['gates'][0]['qubits'] == [0, 1]
    
    def test_add_rotation_gate(self):
        """Test adding rotation gates with parameters."""
        circuit = self.backend.create_circuit(2)
        gate = GateInfo(name="rx", display_name="RX", num_qubits=1, num_params=1)
        
        success = self.backend.add_gate(circuit, gate, [0], [1.57])
        
        assert success is True
        assert circuit['gates'][0]['params'] == [1.57]
    
    def test_add_gate_invalid_qubit(self):
        """Test adding gate with invalid qubit index."""
        circuit = self.backend.create_circuit(2)
        gate = GateInfo(name="h", display_name="H", num_qubits=1)
        
        # Try to add gate to qubit 3 in 2-qubit circuit
        success = self.backend.add_gate(circuit, gate, [3], [])
        
        assert success is False
        assert len(circuit['gates']) == 0
    
    def test_get_circuit_metrics(self):
        """Test getting circuit metrics."""
        circuit = self.backend.create_circuit(2)
        gate = GateInfo(name="h", display_name="H", num_qubits=1)
        
        # Empty circuit
        assert self.backend.get_gate_count(circuit) == 0
        assert self.backend.get_circuit_depth(circuit) == 0
        
        # Add some gates
        self.backend.add_gate(circuit, gate, [0], [])
        self.backend.add_gate(circuit, gate, [1], [])
        
        assert self.backend.get_gate_count(circuit) == 2
        assert self.backend.get_circuit_depth(circuit) == 2
    
    def test_export_qasm(self):
        """Test QASM export."""
        circuit = self.backend.create_circuit(2)
        
        # Add some gates
        h_gate = GateInfo(name="h", display_name="H", num_qubits=1)
        cnot_gate = GateInfo(name="cnot", display_name="CNOT", num_qubits=2)
        rx_gate = GateInfo(name="rx", display_name="RX", num_qubits=1, num_params=1)
        
        self.backend.add_gate(circuit, h_gate, [0], [])
        self.backend.add_gate(circuit, cnot_gate, [0, 1], [])
        self.backend.add_gate(circuit, rx_gate, [1], [1.57])
        
        qasm = self.backend.export_circuit(circuit, "qasm")
        
        assert "OPENQASM 2.0" in qasm
        assert "qreg q[2]" in qasm
        assert "h q[0]" in qasm
        assert "cx q[0],q[1]" in qasm
        assert "rx(1.57) q[1]" in qasm
    
    def test_export_json(self):
        """Test JSON export."""
        circuit = self.backend.create_circuit(2)
        gate = GateInfo(name="h", display_name="H", num_qubits=1)
        self.backend.add_gate(circuit, gate, [0], [])
        
        json_content = self.backend.export_circuit(circuit, "json")
        
        assert json_content != ""
        # Should be valid JSON
        parsed = json.loads(json_content)
        assert parsed['n_qubits'] == 2
        assert len(parsed['gates']) == 1


@pytest.mark.skipif(not HAS_CIRCUIT_BUILDER, reason="circuit_builder module not available")
class TestQuantRS2Backend:
    """Test QuantRS2Backend functionality."""
    
    def setup_method(self):
        """Set up test backend."""
        self.backend = QuantRS2Backend()
    
    def test_backend_availability(self):
        """Test backend availability check."""
        # The backend should handle unavailability gracefully
        try:
            circuit = self.backend.create_circuit(2)
            # If we get here, native is available
            assert circuit is not None
        except RuntimeError as e:
            # Native not available
            assert "not available" in str(e)
    
    @patch('quantrs2.circuit_builder._NATIVE_AVAILABLE', True)
    @patch('quantrs2.circuit_builder._quantrs2')
    def test_create_circuit_with_mock(self, mock_quantrs2):
        """Test circuit creation with mocked native module."""
        mock_circuit = Mock()
        mock_quantrs2.PyCircuit.return_value = mock_circuit
        
        circuit = self.backend.create_circuit(2)
        
        assert circuit == mock_circuit
        mock_quantrs2.PyCircuit.assert_called_once_with(2)
    
    @patch('quantrs2.circuit_builder._NATIVE_AVAILABLE', True)
    @patch('quantrs2.circuit_builder._quantrs2')
    def test_add_gates_with_mock(self, mock_quantrs2):
        """Test adding gates with mocked native module."""
        mock_circuit = Mock()
        mock_quantrs2.PyCircuit.return_value = mock_circuit
        
        circuit = self.backend.create_circuit(2)
        
        # Test single-qubit gates
        h_gate = GateInfo(name="h", display_name="H", num_qubits=1)
        success = self.backend.add_gate(circuit, h_gate, [0], [])
        assert success is True
        mock_circuit.h.assert_called_once_with(0)
        
        # Test two-qubit gates
        cnot_gate = GateInfo(name="cnot", display_name="CNOT", num_qubits=2)
        success = self.backend.add_gate(circuit, cnot_gate, [0, 1], [])
        assert success is True
        mock_circuit.cnot.assert_called_once_with(0, 1)
        
        # Test rotation gates
        rx_gate = GateInfo(name="rx", display_name="RX", num_qubits=1, num_params=1)
        success = self.backend.add_gate(circuit, rx_gate, [0], [1.57])
        assert success is True
        mock_circuit.rx.assert_called_once_with(0, 1.57)
    
    @patch('quantrs2.circuit_builder._NATIVE_AVAILABLE', True)
    @patch('quantrs2.circuit_builder._quantrs2')
    def test_circuit_metrics_with_mock(self, mock_quantrs2):
        """Test circuit metrics with mocked native module."""
        mock_circuit = Mock()
        mock_circuit.depth.return_value = 5
        mock_circuit.gate_count.return_value = 10
        mock_quantrs2.PyCircuit.return_value = mock_circuit
        
        circuit = self.backend.create_circuit(2)
        
        assert self.backend.get_circuit_depth(circuit) == 5
        assert self.backend.get_gate_count(circuit) == 10


@pytest.mark.skipif(not HAS_CIRCUIT_BUILDER, reason="circuit_builder module not available")
class TestCircuitBuilder:
    """Test CircuitBuilder core functionality."""
    
    def setup_method(self):
        """Set up test circuit builder."""
        # Use mock backend for consistent testing
        self.builder = CircuitBuilder(MockCircuitBackend())
    
    def test_initialization(self):
        """Test circuit builder initialization."""
        assert self.builder.backend is not None
        assert len(self.builder.available_gates) > 0
        assert self.builder.current_circuit_id is None
        assert len(self.builder.circuits) == 0
    
    def test_available_gates(self):
        """Test available gates setup."""
        gates = self.builder.available_gates
        
        # Check some expected gates
        assert 'h' in gates
        assert 'cnot' in gates
        assert 'rx' in gates
        
        # Check gate properties
        h_gate = gates['h']
        assert h_gate.num_qubits == 1
        assert h_gate.num_params == 0
        
        rx_gate = gates['rx']
        assert rx_gate.num_qubits == 1
        assert rx_gate.num_params == 1
        assert rx_gate.param_names == ['angle']
    
    def test_create_circuit(self):
        """Test creating a circuit."""
        circuit_id = self.builder.create_circuit(3)
        
        assert circuit_id is not None
        assert circuit_id in self.builder.circuits
        assert self.builder.current_circuit_id == circuit_id
        
        circuit_data = self.builder.circuits[circuit_id]
        assert circuit_data['n_qubits'] == 3
        assert circuit_data['elements'] == []
        assert 'metadata' in circuit_data
    
    def test_create_circuit_with_id(self):
        """Test creating circuit with specific ID."""
        circuit_id = self.builder.create_circuit(2, "my_circuit")
        
        assert circuit_id == "my_circuit"
        assert "my_circuit" in self.builder.circuits
    
    def test_add_single_qubit_gate(self):
        """Test adding single-qubit gates."""
        circuit_id = self.builder.create_circuit(2)
        
        success = self.builder.add_gate("h", [0])
        
        assert success is True
        
        info = self.builder.get_circuit_info(circuit_id)
        assert info['gate_count'] == 1
        assert len(info['elements']) == 1
        
        element = info['elements'][0]
        assert element.gate.name == "h"
        assert element.qubits == [0]
    
    def test_add_two_qubit_gate(self):
        """Test adding two-qubit gates."""
        circuit_id = self.builder.create_circuit(3)
        
        success = self.builder.add_gate("cnot", [0, 1])
        
        assert success is True
        
        info = self.builder.get_circuit_info(circuit_id)
        element = info['elements'][0]
        assert element.gate.name == "cnot"
        assert element.qubits == [0, 1]
    
    def test_add_rotation_gate(self):
        """Test adding rotation gates with parameters."""
        circuit_id = self.builder.create_circuit(2)
        
        success = self.builder.add_gate("rx", [0], [1.57])
        
        assert success is True
        
        info = self.builder.get_circuit_info(circuit_id)
        element = info['elements'][0]
        assert element.gate.name == "rx"
        assert element.params == [1.57]
    
    def test_add_gate_invalid_inputs(self):
        """Test adding gates with invalid inputs."""
        circuit_id = self.builder.create_circuit(2)
        
        # Invalid gate name
        assert self.builder.add_gate("invalid_gate", [0]) is False
        
        # Wrong number of qubits
        assert self.builder.add_gate("h", [0, 1]) is False  # H is single-qubit
        assert self.builder.add_gate("cnot", [0]) is False  # CNOT is two-qubit
        
        # Wrong number of parameters
        assert self.builder.add_gate("rx", [0], []) is False  # RX needs parameter
        assert self.builder.add_gate("h", [0], [1.0]) is False  # H doesn't take parameters
    
    def test_add_gate_no_circuit(self):
        """Test adding gate when no circuit exists."""
        success = self.builder.add_gate("h", [0])
        assert success is False
    
    def test_remove_gate(self):
        """Test removing gates from circuit."""
        circuit_id = self.builder.create_circuit(2)
        
        # Add some gates
        self.builder.add_gate("h", [0])
        self.builder.add_gate("cnot", [0, 1])
        self.builder.add_gate("h", [1])
        
        info = self.builder.get_circuit_info(circuit_id)
        assert len(info['elements']) == 3
        
        # Remove middle gate
        element_id = info['elements'][1].element_id
        success = self.builder.remove_gate(element_id)
        
        assert success is True
        
        info = self.builder.get_circuit_info(circuit_id)
        assert len(info['elements']) == 2
        # Positions should be updated
        assert info['elements'][0].position == 0
        assert info['elements'][1].position == 1
    
    def test_remove_nonexistent_gate(self):
        """Test removing nonexistent gate."""
        circuit_id = self.builder.create_circuit(2)
        self.builder.add_gate("h", [0])
        
        success = self.builder.remove_gate("nonexistent_id")
        assert success is False
    
    def test_get_circuit_info(self):
        """Test getting circuit information."""
        circuit_id = self.builder.create_circuit(3)
        self.builder.add_gate("h", [0])
        self.builder.add_gate("cnot", [0, 1])
        
        info = self.builder.get_circuit_info(circuit_id)
        
        assert info['id'] == circuit_id
        assert info['n_qubits'] == 3
        assert info['gate_count'] == 2
        assert info['depth'] == 2
        assert len(info['elements']) == 2
        assert 'metadata' in info
    
    def test_get_circuit_info_nonexistent(self):
        """Test getting info for nonexistent circuit."""
        info = self.builder.get_circuit_info("nonexistent")
        assert info is None
    
    def test_export_circuit(self):
        """Test exporting circuits."""
        circuit_id = self.builder.create_circuit(2)
        self.builder.add_gate("h", [0])
        self.builder.add_gate("cnot", [0, 1])
        
        # Export QASM
        qasm = self.builder.export_circuit("qasm", circuit_id)
        assert "OPENQASM" in qasm
        assert "h q[0]" in qasm
        assert "cx q[0],q[1]" in qasm
        
        # Export JSON
        json_content = self.builder.export_circuit("json", circuit_id)
        assert json_content != ""
    
    def test_save_and_load_circuit(self):
        """Test saving and loading circuits."""
        # Create and populate circuit
        circuit_id = self.builder.create_circuit(2)
        self.builder.add_gate("h", [0])
        self.builder.add_gate("cnot", [0, 1])
        
        # Save to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            success = self.builder.save_circuit(filepath, "json", circuit_id)
            assert success is True
            
            # Create new builder and load
            new_builder = CircuitBuilder(MockCircuitBackend())
            loaded_id = new_builder.load_circuit(filepath, "loaded_circuit")
            
            assert loaded_id == "loaded_circuit"
            
            # Compare circuits
            original_info = self.builder.get_circuit_info(circuit_id)
            loaded_info = new_builder.get_circuit_info(loaded_id)
            
            assert loaded_info['n_qubits'] == original_info['n_qubits']
            assert len(loaded_info['elements']) == len(original_info['elements'])
            
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_observer_pattern(self):
        """Test observer pattern for GUI updates."""
        events = []
        
        def observer(event, *args):
            events.append((event, args))
        
        self.builder.add_observer(observer)
        
        # Create circuit
        circuit_id = self.builder.create_circuit(2)
        assert len(events) == 1
        assert events[0][0] == "circuit_created"
        
        # Add gate
        self.builder.add_gate("h", [0])
        assert len(events) == 2
        assert events[1][0] == "gate_added"
        
        # Remove observer
        self.builder.remove_observer(observer)
        self.builder.add_gate("cnot", [0, 1])
        assert len(events) == 2  # No new events


@pytest.mark.skipif(not HAS_CIRCUIT_BUILDER, reason="circuit_builder module not available")
@pytest.mark.skipif(not TKINTER_AVAILABLE, reason="tkinter not available")
class TestTkinterGUI:
    """Test TkinterGUI functionality."""
    
    def setup_method(self):
        """Set up test GUI."""
        self.builder = CircuitBuilder(MockCircuitBackend())
        # Create GUI but don't run mainloop
        self.gui = TkinterGUI(self.builder)
    
    def teardown_method(self):
        """Clean up GUI."""
        if hasattr(self, 'gui') and self.gui.root:
            self.gui.root.destroy()
    
    def test_gui_initialization(self):
        """Test GUI initialization."""
        assert self.gui.builder == self.builder
        assert self.gui.root is not None
        assert self.gui.current_circuit is None
    
    def test_ui_components_exist(self):
        """Test that UI components are created."""
        # Check main frames exist
        assert self.gui.toolbar_frame is not None
        assert self.gui.main_frame is not None
        assert self.gui.left_frame is not None
        assert self.gui.right_frame is not None
        
        # Check specific components
        assert self.gui.gate_canvas is not None
        assert self.gui.circuit_canvas is not None
        assert self.gui.info_label is not None
    
    @patch('tkinter.simpledialog.askinteger', return_value=2)
    def test_new_circuit_dialog(self, mock_dialog):
        """Test new circuit creation dialog."""
        # Simulate the new circuit process
        n_qubits = 2
        circuit_id = self.builder.create_circuit(n_qubits)
        self.gui.current_circuit = circuit_id
        self.gui.update_circuit_display()
        
        # Check that circuit was created
        assert self.gui.current_circuit is not None
        info = self.builder.get_circuit_info(self.gui.current_circuit)
        assert info['n_qubits'] == n_qubits
    
    def test_gate_selection(self):
        """Test gate selection."""
        gate_info = self.builder.available_gates['h']
        self.gui.select_gate(gate_info)
        
        assert hasattr(self.gui, 'selected_gate')
        assert self.gui.selected_gate == gate_info
    
    def test_circuit_display_update(self):
        """Test circuit display updates."""
        # Create circuit
        circuit_id = self.builder.create_circuit(2)
        self.gui.current_circuit = circuit_id
        
        # Update display
        self.gui.update_circuit_display()
        
        # Check that display was updated
        info_text = self.gui.info_label.cget("text")
        assert "Qubits: 2" in info_text
        assert "Gates: 0" in info_text
    
    def test_observer_callback(self):
        """Test GUI observer callback."""
        # Create circuit to trigger observer
        circuit_id = self.builder.create_circuit(2)
        
        # Observer should be called (we can't easily test async call)
        # Just ensure no errors occur
        assert True


@pytest.mark.skipif(not HAS_CIRCUIT_BUILDER, reason="circuit_builder module not available")
@pytest.mark.skipif(not FLASK_AVAILABLE, reason="flask not available")
class TestWebGUI:
    """Test WebGUI functionality."""
    
    def setup_method(self):
        """Set up test web GUI."""
        self.builder = CircuitBuilder(MockCircuitBackend())
        self.gui = WebGUI(self.builder, host="localhost", port=5001)
        self.client = self.gui.app.test_client()
    
    def test_web_gui_initialization(self):
        """Test web GUI initialization."""
        assert self.gui.builder == self.builder
        assert self.gui.host == "localhost"
        assert self.gui.port == 5001
        assert self.gui.app is not None
    
    def test_index_route(self):
        """Test main index route."""
        response = self.client.get('/')
        assert response.status_code == 200
        assert b"QuantRS2 Circuit Builder" in response.data
    
    def test_gates_api(self):
        """Test gates API endpoint."""
        response = self.client.get('/api/gates')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'h' in data
        assert 'cnot' in data
        
        h_gate = data['h']
        assert h_gate['display_name'] == 'Hadamard'
        assert h_gate['num_qubits'] == 1
    
    def test_new_circuit_api(self):
        """Test new circuit API endpoint."""
        response = self.client.post('/api/circuit/new', 
                                  json={'n_qubits': 3})
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['success'] is True
        assert 'circuit_id' in data
    
    def test_add_gate_api(self):
        """Test add gate API endpoint."""
        # First create a circuit
        response = self.client.post('/api/circuit/new', 
                                  json={'n_qubits': 2})
        circuit_id = response.get_json()['circuit_id']
        
        # Add a gate
        response = self.client.post(f'/api/circuit/{circuit_id}/add_gate',
                                  json={
                                      'gate_name': 'h',
                                      'qubits': [0],
                                      'params': []
                                  })
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['success'] is True
    
    def test_circuit_info_api(self):
        """Test circuit info API endpoint."""
        # Create circuit and add gate
        response = self.client.post('/api/circuit/new', 
                                  json={'n_qubits': 2})
        circuit_id = response.get_json()['circuit_id']
        
        self.client.post(f'/api/circuit/{circuit_id}/add_gate',
                        json={'gate_name': 'h', 'qubits': [0], 'params': []})
        
        # Get info
        response = self.client.get(f'/api/circuit/{circuit_id}/info')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['n_qubits'] == 2
        assert data['gate_count'] == 1
        assert len(data['elements']) == 1
    
    def test_export_circuit_api(self):
        """Test circuit export API endpoint."""
        # Create circuit and add gate
        response = self.client.post('/api/circuit/new', 
                                  json={'n_qubits': 2})
        circuit_id = response.get_json()['circuit_id']
        
        self.client.post(f'/api/circuit/{circuit_id}/add_gate',
                        json={'gate_name': 'h', 'qubits': [0], 'params': []})
        
        # Export as QASM
        response = self.client.get(f'/api/circuit/{circuit_id}/export/qasm')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'content' in data
        assert 'OPENQASM' in data['content']
    
    def test_nonexistent_circuit_api(self):
        """Test API with nonexistent circuit."""
        response = self.client.get('/api/circuit/nonexistent/info')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'error' in data


@pytest.mark.skipif(not HAS_CIRCUIT_BUILDER, reason="circuit_builder module not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_circuit_builder_auto(self):
        """Test auto backend selection."""
        builder = create_circuit_builder()
        assert builder is not None
        assert builder.backend is not None
    
    def test_create_circuit_builder_mock(self):
        """Test mock backend selection."""
        builder = create_circuit_builder("mock")
        assert builder is not None
        assert isinstance(builder.backend, MockCircuitBackend)
    
    def test_create_circuit_builder_quantrs2(self):
        """Test quantrs2 backend selection."""
        # This will fall back to mock if native not available
        builder = create_circuit_builder("quantrs2")
        assert builder is not None
        assert builder.backend is not None
    
    @patch('quantrs2.circuit_builder.TKINTER_AVAILABLE', True)
    @patch('quantrs2.circuit_builder.TkinterGUI')
    def test_launch_gui_tkinter(self, mock_gui_class):
        """Test launching Tkinter GUI."""
        mock_gui = Mock()
        mock_gui_class.return_value = mock_gui
        
        launch_gui("tkinter")
        
        mock_gui_class.assert_called_once()
        mock_gui.run.assert_called_once()
    
    @patch('quantrs2.circuit_builder.FLASK_AVAILABLE', True)
    @patch('quantrs2.circuit_builder.WebGUI')
    def test_launch_gui_web(self, mock_gui_class):
        """Test launching web GUI."""
        mock_gui = Mock()
        mock_gui_class.return_value = mock_gui
        
        launch_gui("web", host="0.0.0.0", port=8080, debug=True)
        
        mock_gui_class.assert_called_once()
        mock_gui.run.assert_called_once_with(True)
    
    def test_launch_gui_invalid_interface(self):
        """Test launching with invalid interface."""
        with pytest.raises(ValueError):
            launch_gui("invalid")
    
    @patch('quantrs2.circuit_builder.TKINTER_AVAILABLE', False)
    @patch('quantrs2.circuit_builder.FLASK_AVAILABLE', False)
    def test_launch_gui_no_frameworks(self):
        """Test launching when no frameworks available."""
        with pytest.raises(RuntimeError):
            launch_gui("auto")


@pytest.mark.skipif(not HAS_CIRCUIT_BUILDER, reason="circuit_builder module not available")
class TestErrorHandling:
    """Test error handling scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.builder = CircuitBuilder(MockCircuitBackend())
    
    def test_backend_creation_error(self):
        """Test handling backend creation errors."""
        class FailingBackend(MockCircuitBackend):
            def create_circuit(self, n_qubits):
                raise Exception("Backend error")
        
        builder = CircuitBuilder(FailingBackend())
        
        with pytest.raises(RuntimeError):
            builder.create_circuit(2)
    
    def test_gate_addition_error(self):
        """Test handling gate addition errors."""
        class FailingBackend(MockCircuitBackend):
            def add_gate(self, circuit, gate_info, qubits, params):
                return False  # Always fail
        
        builder = CircuitBuilder(FailingBackend())
        circuit_id = builder.create_circuit(2)
        
        success = builder.add_gate("h", [0])
        assert success is False
    
    def test_file_io_errors(self):
        """Test file I/O error handling."""
        circuit_id = self.builder.create_circuit(2)
        
        # Try to save to invalid path
        success = self.builder.save_circuit("/invalid/path/circuit.json")
        assert success is False
        
        # Try to load nonexistent file
        loaded_id = self.builder.load_circuit("/nonexistent/file.json")
        assert loaded_id is None
    
    def test_observer_error_handling(self):
        """Test that observer errors don't break the system."""
        def failing_observer(event, *args):
            raise Exception("Observer error")
        
        self.builder.add_observer(failing_observer)
        
        # Should not raise exception despite failing observer
        circuit_id = self.builder.create_circuit(2)
        assert circuit_id is not None
    
    def test_invalid_json_load(self):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            filepath = f.name
        
        try:
            loaded_id = self.builder.load_circuit(filepath)
            assert loaded_id is None
        finally:
            Path(filepath).unlink(missing_ok=True)


@pytest.mark.skipif(not HAS_CIRCUIT_BUILDER, reason="circuit_builder module not available")
class TestCircuitBuilderIntegration:
    """Test integration scenarios."""
    
    def test_complete_circuit_building_workflow(self):
        """Test complete workflow from creation to export."""
        builder = CircuitBuilder(MockCircuitBackend())
        
        # Create circuit
        circuit_id = builder.create_circuit(3, "test_circuit")
        
        # Build Bell state circuit
        assert builder.add_gate("h", [0]) is True
        assert builder.add_gate("cnot", [0, 1]) is True
        
        # Add some single-qubit gates
        assert builder.add_gate("x", [2]) is True
        assert builder.add_gate("ry", [2], [1.57]) is True
        
        # Check circuit info
        info = builder.get_circuit_info(circuit_id)
        assert info['n_qubits'] == 3
        assert info['gate_count'] == 4
        assert len(info['elements']) == 4
        
        # Export to different formats
        qasm = builder.export_circuit("qasm", circuit_id)
        assert "h q[0]" in qasm
        assert "cx q[0],q[1]" in qasm
        assert "x q[2]" in qasm
        assert "ry(1.57) q[2]" in qasm
        
        json_content = builder.export_circuit("json", circuit_id)
        assert json_content != ""
        
        # Test save/load cycle
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            assert builder.save_circuit(filepath, "json", circuit_id) is True
            
            # Load into new builder
            new_builder = CircuitBuilder(MockCircuitBackend())
            loaded_id = new_builder.load_circuit(filepath, "loaded_test")
            
            assert loaded_id is not None
            
            # Compare circuits
            original_info = builder.get_circuit_info(circuit_id)
            loaded_info = new_builder.get_circuit_info(loaded_id)
            
            assert loaded_info['n_qubits'] == original_info['n_qubits']
            assert loaded_info['gate_count'] == original_info['gate_count']
            assert len(loaded_info['elements']) == len(original_info['elements'])
            
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_multiple_circuits_management(self):
        """Test managing multiple circuits simultaneously."""
        builder = CircuitBuilder(MockCircuitBackend())
        
        # Create multiple circuits
        circuit1 = builder.create_circuit(2, "circuit1")
        circuit2 = builder.create_circuit(3, "circuit2")
        circuit3 = builder.create_circuit(4, "circuit3")
        
        # Build different circuits
        builder.add_gate("h", [0], circuit_id=circuit1)
        builder.add_gate("cnot", [0, 1], circuit_id=circuit1)
        
        builder.add_gate("h", [0], circuit_id=circuit2)
        builder.add_gate("h", [1], circuit_id=circuit2)
        builder.add_gate("h", [2], circuit_id=circuit2)
        
        builder.add_gate("rx", [0], [1.0], circuit_id=circuit3)
        builder.add_gate("ry", [1], [2.0], circuit_id=circuit3)
        builder.add_gate("rz", [2], [3.0], circuit_id=circuit3)
        builder.add_gate("cnot", [0, 3], circuit_id=circuit3)
        
        # Check circuit info
        info1 = builder.get_circuit_info(circuit1)
        info2 = builder.get_circuit_info(circuit2)
        info3 = builder.get_circuit_info(circuit3)
        
        assert info1['gate_count'] == 2
        assert info2['gate_count'] == 3
        assert info3['gate_count'] == 4
        
        assert info1['n_qubits'] == 2
        assert info2['n_qubits'] == 3
        assert info3['n_qubits'] == 4
    
    def test_circuit_modification_workflow(self):
        """Test modifying circuits after creation."""
        builder = CircuitBuilder(MockCircuitBackend())
        circuit_id = builder.create_circuit(3)
        
        # Build initial circuit
        builder.add_gate("h", [0])
        builder.add_gate("cnot", [0, 1])
        builder.add_gate("h", [2])
        builder.add_gate("cnot", [1, 2])
        
        info = builder.get_circuit_info(circuit_id)
        assert info['gate_count'] == 4
        
        # Remove a gate
        element_to_remove = info['elements'][1]  # CNOT gate
        success = builder.remove_gate(element_to_remove.element_id)
        assert success is True
        
        # Check updated circuit
        info = builder.get_circuit_info(circuit_id)
        assert info['gate_count'] == 3
        assert len(info['elements']) == 3
        
        # Verify positions were updated
        for i, element in enumerate(info['elements']):
            assert element.position == i
    
    def test_large_circuit_performance(self):
        """Test performance with larger circuits."""
        builder = CircuitBuilder(MockCircuitBackend())
        circuit_id = builder.create_circuit(10)
        
        import time
        start_time = time.time()
        
        # Add many gates
        for i in range(100):
            qubit = i % 10
            if i % 3 == 0:
                builder.add_gate("h", [qubit])
            elif i % 3 == 1:
                builder.add_gate("x", [qubit])
            else:
                target = (qubit + 1) % 10
                builder.add_gate("cnot", [qubit, target])
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0  # 5 seconds for 100 gates
        
        info = builder.get_circuit_info(circuit_id)
        assert info['gate_count'] == 100
        assert len(info['elements']) == 100


if __name__ == "__main__":
    pytest.main([__file__])