"""
Interactive Circuit Builder for QuantRS2

This module provides an interactive GUI for building quantum circuits visually,
with real-time visualization and export capabilities.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
import time

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

try:
    import flask
    from flask import Flask, render_template_string, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    from . import _quantrs2
    _NATIVE_AVAILABLE = True
except ImportError:
    _NATIVE_AVAILABLE = False


@dataclass
class GateInfo:
    """Information about a quantum gate for the circuit builder."""
    name: str
    display_name: str
    num_qubits: int
    num_params: int = 0
    param_names: List[str] = field(default_factory=list)
    description: str = ""
    category: str = "Basic"
    color: str = "#4CAF50"
    symbol: str = ""


@dataclass
class CircuitElement:
    """Represents an element in the circuit builder."""
    gate: GateInfo
    qubits: List[int]
    params: List[float] = field(default_factory=list)
    position: int = 0  # Position in the circuit
    element_id: str = ""


class CircuitBuilderBackend(ABC):
    """Abstract backend for circuit building."""
    
    @abstractmethod
    def create_circuit(self, n_qubits: int) -> Any:
        """Create a new circuit with specified number of qubits."""
        pass
    
    @abstractmethod
    def add_gate(self, circuit: Any, gate_info: GateInfo, qubits: List[int], params: List[float]) -> bool:
        """Add a gate to the circuit."""
        pass
    
    @abstractmethod
    def get_circuit_depth(self, circuit: Any) -> int:
        """Get the depth of the circuit."""
        pass
    
    @abstractmethod
    def get_gate_count(self, circuit: Any) -> int:
        """Get the total number of gates in the circuit."""
        pass
    
    @abstractmethod
    def export_circuit(self, circuit: Any, format: str = "qasm") -> str:
        """Export circuit to specified format."""
        pass


class MockCircuitBackend(CircuitBuilderBackend):
    """Mock backend for testing and fallback."""
    
    def __init__(self):
        self._circuits = {}
        self._gate_counter = 0
    
    def create_circuit(self, n_qubits: int) -> Dict[str, Any]:
        """Create a mock circuit."""
        circuit_id = f"circuit_{int(time.time() * 1000)}"
        circuit = {
            'id': circuit_id,
            'n_qubits': n_qubits,
            'gates': [],
            'depth': 0
        }
        self._circuits[circuit_id] = circuit
        return circuit
    
    def add_gate(self, circuit: Dict[str, Any], gate_info: GateInfo, qubits: List[int], params: List[float]) -> bool:
        """Add a gate to the mock circuit."""
        try:
            if not circuit or 'gates' not in circuit:
                return False
            
            # Validate qubits
            for qubit in qubits:
                if qubit < 0 or qubit >= circuit['n_qubits']:
                    return False
            
            gate_entry = {
                'gate': gate_info.name,
                'qubits': qubits,
                'params': params,
                'position': len(circuit['gates'])
            }
            
            circuit['gates'].append(gate_entry)
            circuit['depth'] = len(circuit['gates'])  # Simplified depth calculation
            self._gate_counter += 1
            
            return True
        except Exception:
            return False
    
    def get_circuit_depth(self, circuit: Dict[str, Any]) -> int:
        """Get circuit depth."""
        return circuit.get('depth', 0) if circuit else 0
    
    def get_gate_count(self, circuit: Dict[str, Any]) -> int:
        """Get gate count."""
        return len(circuit.get('gates', [])) if circuit else 0
    
    def export_circuit(self, circuit: Dict[str, Any], format: str = "qasm") -> str:
        """Export circuit to specified format."""
        if not circuit:
            return ""
        
        if format.lower() == "qasm":
            return self._export_qasm(circuit)
        elif format.lower() == "json":
            return json.dumps(circuit, indent=2)
        else:
            return str(circuit)
    
    def _export_qasm(self, circuit: Dict[str, Any]) -> str:
        """Export to OpenQASM format."""
        qasm_lines = [
            "OPENQASM 2.0;",
            "include \"qelib1.inc\";",
            f"qreg q[{circuit['n_qubits']}];",
            f"creg c[{circuit['n_qubits']}];",
            ""
        ]
        
        for gate_entry in circuit['gates']:
            gate_name = gate_entry['gate'].lower()
            qubits = gate_entry['qubits']
            params = gate_entry.get('params', [])
            
            if gate_name == 'h':
                qasm_lines.append(f"h q[{qubits[0]}];")
            elif gate_name == 'x':
                qasm_lines.append(f"x q[{qubits[0]}];")
            elif gate_name == 'y':
                qasm_lines.append(f"y q[{qubits[0]}];")
            elif gate_name == 'z':
                qasm_lines.append(f"z q[{qubits[0]}];")
            elif gate_name == 'cnot':
                qasm_lines.append(f"cx q[{qubits[0]}],q[{qubits[1]}];")
            elif gate_name in ['rx', 'ry', 'rz'] and params:
                qasm_lines.append(f"{gate_name}({params[0]}) q[{qubits[0]}];")
            else:
                qasm_lines.append(f"// Unknown gate: {gate_name}")
        
        return "\n".join(qasm_lines)


class QuantRS2Backend(CircuitBuilderBackend):
    """Native QuantRS2 backend."""
    
    def create_circuit(self, n_qubits: int) -> Any:
        """Create a QuantRS2 circuit."""
        if not _NATIVE_AVAILABLE:
            raise RuntimeError("Native QuantRS2 not available")
        return _quantrs2.PyCircuit(n_qubits)
    
    def add_gate(self, circuit: Any, gate_info: GateInfo, qubits: List[int], params: List[float]) -> bool:
        """Add gate to QuantRS2 circuit."""
        try:
            gate_name = gate_info.name.lower()
            
            if gate_name == 'h':
                circuit.h(qubits[0])
            elif gate_name == 'x':
                circuit.x(qubits[0])
            elif gate_name == 'y':
                circuit.y(qubits[0])
            elif gate_name == 'z':
                circuit.z(qubits[0])
            elif gate_name == 's':
                circuit.s(qubits[0])
            elif gate_name == 't':
                circuit.t(qubits[0])
            elif gate_name == 'cnot':
                circuit.cnot(qubits[0], qubits[1])
            elif gate_name == 'cz':
                circuit.cz(qubits[0], qubits[1])
            elif gate_name == 'rx' and params:
                circuit.rx(qubits[0], params[0])
            elif gate_name == 'ry' and params:
                circuit.ry(qubits[0], params[0])
            elif gate_name == 'rz' and params:
                circuit.rz(qubits[0], params[0])
            else:
                return False
            
            return True
        except Exception:
            return False
    
    def get_circuit_depth(self, circuit: Any) -> int:
        """Get circuit depth."""
        try:
            return circuit.depth()
        except Exception:
            return 0
    
    def get_gate_count(self, circuit: Any) -> int:
        """Get circuit gate count."""
        try:
            return circuit.gate_count()
        except Exception:
            return 0
    
    def export_circuit(self, circuit: Any, format: str = "qasm") -> str:
        """Export circuit."""
        try:
            if format.lower() == "qasm":
                # Use existing QASM export if available
                try:
                    from .qasm import export_qasm
                    return export_qasm(circuit)
                except ImportError:
                    return "// QASM export not available"
            else:
                return str(circuit)
        except Exception:
            return ""


class CircuitBuilder:
    """Main circuit builder class."""
    
    def __init__(self, backend: Optional[CircuitBuilderBackend] = None):
        """Initialize circuit builder."""
        self.backend = backend or self._get_default_backend()
        self.available_gates = self._initialize_gates()
        self.circuits = {}  # Active circuits
        self.current_circuit_id = None
        self.observers = []  # For GUI updates
        
    def _get_default_backend(self) -> CircuitBuilderBackend:
        """Get the default backend."""
        if _NATIVE_AVAILABLE:
            try:
                return QuantRS2Backend()
            except Exception:
                pass
        return MockCircuitBackend()
    
    def _initialize_gates(self) -> Dict[str, GateInfo]:
        """Initialize available gates."""
        gates = {
            'h': GateInfo(
                name='h', display_name='Hadamard', num_qubits=1,
                description='Creates superposition', category='Single-Qubit',
                color='#4CAF50', symbol='H'
            ),
            'x': GateInfo(
                name='x', display_name='Pauli-X', num_qubits=1,
                description='Bit flip gate', category='Single-Qubit',
                color='#F44336', symbol='X'
            ),
            'y': GateInfo(
                name='y', display_name='Pauli-Y', num_qubits=1,
                description='Bit and phase flip', category='Single-Qubit',
                color='#FF9800', symbol='Y'
            ),
            'z': GateInfo(
                name='z', display_name='Pauli-Z', num_qubits=1,
                description='Phase flip gate', category='Single-Qubit',
                color='#9C27B0', symbol='Z'
            ),
            's': GateInfo(
                name='s', display_name='S Gate', num_qubits=1,
                description='Phase gate (π/2)', category='Single-Qubit',
                color='#3F51B5', symbol='S'
            ),
            't': GateInfo(
                name='t', display_name='T Gate', num_qubits=1,
                description='Phase gate (π/4)', category='Single-Qubit',
                color='#2196F3', symbol='T'
            ),
            'cnot': GateInfo(
                name='cnot', display_name='CNOT', num_qubits=2,
                description='Controlled-X gate', category='Two-Qubit',
                color='#607D8B', symbol='⊕'
            ),
            'cz': GateInfo(
                name='cz', display_name='CZ', num_qubits=2,
                description='Controlled-Z gate', category='Two-Qubit',
                color='#795548', symbol='CZ'
            ),
            'rx': GateInfo(
                name='rx', display_name='RX', num_qubits=1, num_params=1,
                param_names=['angle'], description='X-axis rotation',
                category='Rotation', color='#E91E63', symbol='RX'
            ),
            'ry': GateInfo(
                name='ry', display_name='RY', num_qubits=1, num_params=1,
                param_names=['angle'], description='Y-axis rotation',
                category='Rotation', color='#CDDC39', symbol='RY'
            ),
            'rz': GateInfo(
                name='rz', display_name='RZ', num_qubits=1, num_params=1,
                param_names=['angle'], description='Z-axis rotation',
                category='Rotation', color='#00BCD4', symbol='RZ'
            ),
        }
        return gates
    
    def create_circuit(self, n_qubits: int, circuit_id: Optional[str] = None) -> str:
        """Create a new circuit."""
        if circuit_id is None:
            circuit_id = f"circuit_{len(self.circuits)}"
        
        try:
            circuit = self.backend.create_circuit(n_qubits)
            self.circuits[circuit_id] = {
                'circuit': circuit,
                'n_qubits': n_qubits,
                'elements': [],
                'metadata': {
                    'created_at': time.time(),
                    'name': circuit_id
                }
            }
            self.current_circuit_id = circuit_id
            self._notify_observers('circuit_created', circuit_id)
            return circuit_id
        except Exception as e:
            raise RuntimeError(f"Failed to create circuit: {e}")
    
    def add_gate(self, gate_name: str, qubits: List[int], params: Optional[List[float]] = None,
                circuit_id: Optional[str] = None) -> bool:
        """Add a gate to the circuit."""
        circuit_id = circuit_id or self.current_circuit_id
        if not circuit_id or circuit_id not in self.circuits:
            return False
        
        if gate_name not in self.available_gates:
            return False
        
        gate_info = self.available_gates[gate_name]
        params = params or []
        
        # Validate inputs
        if len(qubits) != gate_info.num_qubits:
            return False
        
        if len(params) != gate_info.num_params:
            if gate_info.num_params > 0:
                return False
        
        circuit_data = self.circuits[circuit_id]
        circuit = circuit_data['circuit']
        
        # Add to backend
        if not self.backend.add_gate(circuit, gate_info, qubits, params):
            return False
        
        # Track in builder
        element = CircuitElement(
            gate=gate_info,
            qubits=qubits,
            params=params,
            position=len(circuit_data['elements']),
            element_id=f"element_{len(circuit_data['elements'])}"
        )
        
        circuit_data['elements'].append(element)
        self._notify_observers('gate_added', circuit_id, element)
        return True
    
    def remove_gate(self, element_id: str, circuit_id: Optional[str] = None) -> bool:
        """Remove a gate from the circuit."""
        circuit_id = circuit_id or self.current_circuit_id
        if not circuit_id or circuit_id not in self.circuits:
            return False
        
        # For simplicity, rebuild circuit without the specified element
        circuit_data = self.circuits[circuit_id]
        elements = circuit_data['elements']
        
        # Find and remove element
        new_elements = [e for e in elements if e.element_id != element_id]
        if len(new_elements) == len(elements):
            return False  # Element not found
        
        # Rebuild circuit
        n_qubits = circuit_data['n_qubits']
        new_circuit = self.backend.create_circuit(n_qubits)
        
        for element in new_elements:
            if not self.backend.add_gate(new_circuit, element.gate, element.qubits, element.params):
                return False
        
        # Update circuit data
        circuit_data['circuit'] = new_circuit
        circuit_data['elements'] = new_elements
        
        # Update positions
        for i, element in enumerate(new_elements):
            element.position = i
        
        self._notify_observers('gate_removed', circuit_id, element_id)
        return True
    
    def get_circuit_info(self, circuit_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get information about a circuit."""
        circuit_id = circuit_id or self.current_circuit_id
        if not circuit_id or circuit_id not in self.circuits:
            return None
        
        circuit_data = self.circuits[circuit_id]
        circuit = circuit_data['circuit']
        
        return {
            'id': circuit_id,
            'n_qubits': circuit_data['n_qubits'],
            'gate_count': self.backend.get_gate_count(circuit),
            'depth': self.backend.get_circuit_depth(circuit),
            'elements': circuit_data['elements'],
            'metadata': circuit_data['metadata']
        }
    
    def export_circuit(self, format: str = "qasm", circuit_id: Optional[str] = None) -> str:
        """Export circuit to specified format."""
        circuit_id = circuit_id or self.current_circuit_id
        if not circuit_id or circuit_id not in self.circuits:
            return ""
        
        circuit = self.circuits[circuit_id]['circuit']
        return self.backend.export_circuit(circuit, format)
    
    def save_circuit(self, filepath: Union[str, Path], format: str = "json",
                    circuit_id: Optional[str] = None) -> bool:
        """Save circuit to file."""
        try:
            content = ""
            if format.lower() == "json":
                circuit_info = self.get_circuit_info(circuit_id)
                if not circuit_info:
                    return False
                # Convert elements to serializable format
                serializable_info = dict(circuit_info)
                serializable_info['elements'] = [
                    {
                        'gate_name': e.gate.name,
                        'qubits': e.qubits,
                        'params': e.params,
                        'position': e.position
                    } for e in circuit_info['elements']
                ]
                content = json.dumps(serializable_info, indent=2)
            else:
                content = self.export_circuit(format, circuit_id)
            
            if not content:
                return False
            
            Path(filepath).write_text(content)
            return True
        except Exception:
            return False
    
    def load_circuit(self, filepath: Union[str, Path], circuit_id: Optional[str] = None) -> Optional[str]:
        """Load circuit from file."""
        try:
            content = Path(filepath).read_text()
            
            if filepath.suffix.lower() == '.json':
                data = json.loads(content)
                circuit_id = circuit_id or data.get('id', f"loaded_{int(time.time())}")
                
                # Create circuit
                self.create_circuit(data['n_qubits'], circuit_id)
                
                # Add gates
                for element_data in data.get('elements', []):
                    self.add_gate(
                        element_data['gate_name'],
                        element_data['qubits'],
                        element_data.get('params', []),
                        circuit_id
                    )
                
                return circuit_id
            else:
                # TODO: Implement QASM import
                return None
        except Exception:
            return None
    
    def add_observer(self, callback: Callable) -> None:
        """Add observer for circuit changes."""
        if callback not in self.observers:
            self.observers.append(callback)
    
    def remove_observer(self, callback: Callable) -> None:
        """Remove observer."""
        if callback in self.observers:
            self.observers.remove(callback)
    
    def _notify_observers(self, event: str, *args) -> None:
        """Notify observers of changes."""
        for callback in self.observers:
            try:
                callback(event, *args)
            except Exception:
                pass  # Ignore observer errors


# GUI Implementations

class TkinterGUI:
    """Tkinter-based GUI for circuit builder."""
    
    def __init__(self, builder: CircuitBuilder):
        if not TKINTER_AVAILABLE:
            raise RuntimeError("Tkinter not available")
        
        self.builder = builder
        self.root = tk.Tk()
        self.root.title("QuantRS2 Circuit Builder")
        self.root.geometry("1200x800")
        
        self.current_circuit = None
        self.setup_ui()
        
        # Register as observer
        self.builder.add_observer(self.on_circuit_change)
    
    def setup_ui(self):
        """Setup the user interface."""
        # Create main frames
        self.toolbar_frame = ttk.Frame(self.root)
        self.toolbar_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Setup toolbar
        self.setup_toolbar()
        
        # Setup main area
        self.setup_main_area()
    
    def setup_toolbar(self):
        """Setup toolbar with common actions."""
        # New circuit
        ttk.Button(self.toolbar_frame, text="New Circuit", 
                  command=self.new_circuit).pack(side=tk.LEFT, padx=2)
        
        # Load/Save
        ttk.Button(self.toolbar_frame, text="Load", 
                  command=self.load_circuit).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar_frame, text="Save", 
                  command=self.save_circuit).pack(side=tk.LEFT, padx=2)
        
        # Export
        ttk.Button(self.toolbar_frame, text="Export QASM", 
                  command=self.export_qasm).pack(side=tk.LEFT, padx=2)
        
        # Circuit info
        self.info_label = ttk.Label(self.toolbar_frame, text="No circuit loaded")
        self.info_label.pack(side=tk.RIGHT, padx=10)
    
    def setup_main_area(self):
        """Setup main working area."""
        # Left panel - Gate palette
        self.left_frame = ttk.Frame(self.main_frame, width=200)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.left_frame.pack_propagate(False)
        
        # Right panel - Circuit display
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.setup_gate_palette()
        self.setup_circuit_display()
    
    def setup_gate_palette(self):
        """Setup gate palette."""
        ttk.Label(self.left_frame, text="Gates", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Scrollable frame for gates
        self.gate_canvas = tk.Canvas(self.left_frame)
        self.gate_scrollbar = ttk.Scrollbar(self.left_frame, orient="vertical", 
                                          command=self.gate_canvas.yview)
        self.gate_scrollable = ttk.Frame(self.gate_canvas)
        
        self.gate_canvas.configure(yscrollcommand=self.gate_scrollbar.set)
        self.gate_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.gate_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.gate_canvas.create_window((0, 0), window=self.gate_scrollable, anchor="nw")
        
        # Add gates by category
        categories = {}
        for gate_info in self.builder.available_gates.values():
            if gate_info.category not in categories:
                categories[gate_info.category] = []
            categories[gate_info.category].append(gate_info)
        
        for category, gates in categories.items():
            # Category header
            ttk.Label(self.gate_scrollable, text=category, 
                     font=("Arial", 10, "bold")).pack(pady=(10, 2))
            
            # Gates in category
            for gate in gates:
                btn = ttk.Button(self.gate_scrollable, text=gate.display_name,
                               command=lambda g=gate: self.select_gate(g))
                btn.pack(fill=tk.X, padx=5, pady=1)
        
        self.gate_scrollable.update_idletasks()
        self.gate_canvas.configure(scrollregion=self.gate_canvas.bbox("all"))
    
    def setup_circuit_display(self):
        """Setup circuit display area."""
        ttk.Label(self.right_frame, text="Circuit", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Circuit canvas
        self.circuit_canvas = tk.Canvas(self.right_frame, bg="white", height=400)
        self.circuit_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Circuit controls
        self.control_frame = ttk.Frame(self.right_frame)
        self.control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.control_frame, text="Target Qubit:").pack(side=tk.LEFT)
        self.qubit_var = tk.StringVar()
        self.qubit_spin = ttk.Spinbox(self.control_frame, from_=0, to=0, 
                                     textvariable=self.qubit_var, width=5)
        self.qubit_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.control_frame, text="Add Gate", 
                  command=self.add_selected_gate).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(self.control_frame, text="Clear Circuit", 
                  command=self.clear_circuit).pack(side=tk.LEFT, padx=5)
    
    def new_circuit(self):
        """Create new circuit dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("New Circuit")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Number of qubits:").pack(pady=10)
        
        n_qubits_var = tk.StringVar(value="2")
        ttk.Spinbox(dialog, from_=1, to=20, textvariable=n_qubits_var, width=10).pack(pady=5)
        
        def create():
            try:
                n_qubits = int(n_qubits_var.get())
                circuit_id = self.builder.create_circuit(n_qubits)
                self.current_circuit = circuit_id
                self.update_circuit_display()
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Error", "Invalid number of qubits")
        
        ttk.Button(dialog, text="Create", command=create).pack(pady=10)
    
    def load_circuit(self):
        """Load circuit from file."""
        filepath = filedialog.askopenfilename(
            title="Load Circuit",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            circuit_id = self.builder.load_circuit(filepath)
            if circuit_id:
                self.current_circuit = circuit_id
                self.update_circuit_display()
                messagebox.showinfo("Success", "Circuit loaded successfully")
            else:
                messagebox.showerror("Error", "Failed to load circuit")
    
    def save_circuit(self):
        """Save current circuit to file."""
        if not self.current_circuit:
            messagebox.showwarning("Warning", "No circuit to save")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Circuit",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            if self.builder.save_circuit(filepath, "json", self.current_circuit):
                messagebox.showinfo("Success", "Circuit saved successfully")
            else:
                messagebox.showerror("Error", "Failed to save circuit")
    
    def export_qasm(self):
        """Export circuit to QASM."""
        if not self.current_circuit:
            messagebox.showwarning("Warning", "No circuit to export")
            return
        
        qasm_content = self.builder.export_circuit("qasm", self.current_circuit)
        
        # Show in dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("QASM Export")
        dialog.geometry("600x400")
        
        text_widget = tk.Text(dialog, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, qasm_content)
        text_widget.config(state=tk.DISABLED)
        
        # Copy button
        def copy_to_clipboard():
            dialog.clipboard_clear()
            dialog.clipboard_append(qasm_content)
            messagebox.showinfo("Copied", "QASM copied to clipboard")
        
        ttk.Button(dialog, text="Copy to Clipboard", 
                  command=copy_to_clipboard).pack(pady=5)
    
    def select_gate(self, gate_info: GateInfo):
        """Select a gate for placement."""
        self.selected_gate = gate_info
        # Update UI to show selection
    
    def add_selected_gate(self):
        """Add the selected gate to the circuit."""
        if not hasattr(self, 'selected_gate') or not self.current_circuit:
            messagebox.showwarning("Warning", "No gate selected or no circuit loaded")
            return
        
        try:
            qubit = int(self.qubit_var.get())
            gate = self.selected_gate
            
            if gate.num_qubits == 1:
                qubits = [qubit]
            elif gate.num_qubits == 2:
                # For two-qubit gates, prompt for second qubit
                target_qubit = tk.simpledialog.askinteger("Target Qubit", 
                                                        f"Enter target qubit for {gate.display_name}:")
                if target_qubit is None:
                    return
                qubits = [qubit, target_qubit]
            else:
                messagebox.showerror("Error", "Multi-qubit gates not supported in this interface")
                return
            
            # Handle parameters
            params = []
            if gate.num_params > 0:
                param_value = tk.simpledialog.askfloat("Parameter", 
                                                     f"Enter {gate.param_names[0]}:")
                if param_value is None:
                    return
                params = [param_value]
            
            if self.builder.add_gate(gate.name, qubits, params, self.current_circuit):
                self.update_circuit_display()
            else:
                messagebox.showerror("Error", "Failed to add gate")
                
        except ValueError:
            messagebox.showerror("Error", "Invalid qubit number")
    
    def clear_circuit(self):
        """Clear the current circuit."""
        if self.current_circuit:
            # Recreate circuit with same number of qubits
            info = self.builder.get_circuit_info(self.current_circuit)
            if info:
                self.builder.create_circuit(info['n_qubits'], self.current_circuit)
                self.update_circuit_display()
    
    def update_circuit_display(self):
        """Update the circuit visualization."""
        if not self.current_circuit:
            self.info_label.config(text="No circuit loaded")
            self.circuit_canvas.delete("all")
            return
        
        info = self.builder.get_circuit_info(self.current_circuit)
        if not info:
            return
        
        # Update info
        self.info_label.config(text=f"Qubits: {info['n_qubits']}, Gates: {info['gate_count']}, Depth: {info['depth']}")
        
        # Update qubit spinner
        self.qubit_spin.config(to=info['n_qubits']-1)
        
        # Simple circuit visualization
        self.circuit_canvas.delete("all")
        self.draw_circuit(info)
    
    def draw_circuit(self, info: Dict[str, Any]):
        """Draw the circuit on canvas."""
        canvas = self.circuit_canvas
        n_qubits = info['n_qubits']
        elements = info['elements']
        
        # Calculate dimensions
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            # Canvas not yet sized
            self.root.after(100, lambda: self.update_circuit_display())
            return
        
        margin = 50
        qubit_spacing = (height - 2 * margin) / max(1, n_qubits - 1) if n_qubits > 1 else 0
        gate_spacing = 60
        
        # Draw qubit lines
        for i in range(n_qubits):
            y = margin + i * qubit_spacing
            canvas.create_line(margin, y, width - margin, y, fill="black", width=2)
            canvas.create_text(20, y, text=f"q{i}", anchor="w")
        
        # Draw gates
        gate_positions = {}  # Track x position for each gate
        for element in elements:
            gate = element.gate
            qubits = element.qubits
            x = margin + (element.position + 1) * gate_spacing
            
            if gate.num_qubits == 1:
                y = margin + qubits[0] * qubit_spacing
                # Draw gate box
                canvas.create_rectangle(x-15, y-10, x+15, y+10, fill=gate.color, outline="black")
                canvas.create_text(x, y, text=gate.symbol or gate.name.upper(), fill="white")
            
            elif gate.num_qubits == 2:
                y1 = margin + qubits[0] * qubit_spacing
                y2 = margin + qubits[1] * qubit_spacing
                # Draw connection line
                canvas.create_line(x, y1, x, y2, fill="black", width=3)
                # Draw control and target
                canvas.create_oval(x-5, y1-5, x+5, y1+5, fill="black")
                if gate.name == 'cnot':
                    canvas.create_oval(x-8, y2-8, x+8, y2+8, fill="white", outline="black", width=2)
                    canvas.create_line(x-5, y2, x+5, y2, fill="black")
                    canvas.create_line(x, y2-5, x, y2+5, fill="black")
                else:
                    canvas.create_rectangle(x-8, y2-8, x+8, y2+8, fill=gate.color, outline="black")
                    canvas.create_text(x, y2, text=gate.symbol, fill="white")
    
    def on_circuit_change(self, event: str, *args):
        """Handle circuit change events."""
        self.root.after_idle(self.update_circuit_display)
    
    def run(self):
        """Run the GUI."""
        # Auto-create initial circuit
        self.new_circuit()
        self.root.mainloop()


class WebGUI:
    """Web-based GUI using Flask."""
    
    def __init__(self, builder: CircuitBuilder, host: str = "localhost", port: int = 5000):
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask not available")
        
        self.builder = builder
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template_string(self._get_html_template())
        
        @self.app.route('/api/gates')
        def get_gates():
            gates = {}
            for name, gate in self.builder.available_gates.items():
                gates[name] = {
                    'name': gate.name,
                    'display_name': gate.display_name,
                    'num_qubits': gate.num_qubits,
                    'num_params': gate.num_params,
                    'param_names': gate.param_names,
                    'description': gate.description,
                    'category': gate.category,
                    'color': gate.color,
                    'symbol': gate.symbol
                }
            return jsonify(gates)
        
        @self.app.route('/api/circuit/new', methods=['POST'])
        def new_circuit():
            data = request.get_json()
            n_qubits = data.get('n_qubits', 2)
            try:
                circuit_id = self.builder.create_circuit(n_qubits)
                return jsonify({'success': True, 'circuit_id': circuit_id})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/circuit/<circuit_id>/add_gate', methods=['POST'])
        def add_gate(circuit_id):
            data = request.get_json()
            gate_name = data.get('gate_name')
            qubits = data.get('qubits', [])
            params = data.get('params', [])
            
            success = self.builder.add_gate(gate_name, qubits, params, circuit_id)
            return jsonify({'success': success})
        
        @self.app.route('/api/circuit/<circuit_id>/info')
        def get_circuit_info(circuit_id):
            info = self.builder.get_circuit_info(circuit_id)
            if info:
                # Convert elements to serializable format
                serializable_elements = []
                for element in info['elements']:
                    serializable_elements.append({
                        'gate_name': element.gate.name,
                        'gate_display_name': element.gate.display_name,
                        'qubits': element.qubits,
                        'params': element.params,
                        'position': element.position,
                        'element_id': element.element_id,
                        'color': element.gate.color,
                        'symbol': element.gate.symbol
                    })
                
                info['elements'] = serializable_elements
                return jsonify(info)
            else:
                return jsonify({'error': 'Circuit not found'})
        
        @self.app.route('/api/circuit/<circuit_id>/export/<format>')
        def export_circuit(circuit_id, format):
            content = self.builder.export_circuit(format, circuit_id)
            return jsonify({'content': content})
    
    def _get_html_template(self) -> str:
        """Get the HTML template for the web interface."""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>QuantRS2 Circuit Builder</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
        .header { background: #2196F3; color: white; padding: 10px; }
        .container { display: flex; height: calc(100vh - 60px); }
        .sidebar { width: 250px; background: #f5f5f5; padding: 10px; overflow-y: auto; }
        .main { flex: 1; padding: 10px; }
        .gate-category { margin-bottom: 15px; }
        .gate-category h3 { margin: 0 0 5px 0; }
        .gate-btn { 
            display: block; width: 100%; margin: 2px 0; padding: 8px; 
            border: none; background: #4CAF50; color: white; cursor: pointer; 
        }
        .gate-btn:hover { background: #45a049; }
        .circuit-canvas { 
            border: 1px solid #ccc; background: white; 
            width: 100%; height: 400px; margin: 10px 0; 
        }
        .controls { margin: 10px 0; }
        .controls input, .controls select, .controls button { 
            margin: 0 5px; padding: 5px; 
        }
        .info { background: #e3f2fd; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>QuantRS2 Circuit Builder</h1>
        <button onclick="newCircuit()">New Circuit</button>
        <button onclick="exportQASM()">Export QASM</button>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <h2>Gates</h2>
            <div id="gate-palette"></div>
        </div>
        
        <div class="main">
            <div class="info">
                <div id="circuit-info">No circuit loaded</div>
            </div>
            
            <div class="controls">
                <label>Gate:</label>
                <select id="selected-gate"></select>
                <label>Qubit:</label>
                <input type="number" id="target-qubit" min="0" value="0">
                <label>Param:</label>
                <input type="number" id="gate-param" step="0.1" value="0">
                <button onclick="addGate()">Add Gate</button>
                <button onclick="clearCircuit()">Clear</button>
            </div>
            
            <canvas id="circuit-canvas" class="circuit-canvas"></canvas>
            
            <div id="qasm-output" style="display:none;">
                <h3>QASM Export</h3>
                <textarea id="qasm-content" rows="10" cols="80" readonly></textarea>
            </div>
        </div>
    </div>

    <script>
        let currentCircuit = null;
        let availableGates = {};
        
        // Load gates on page load
        window.onload = function() {
            loadGates();
        };
        
        function loadGates() {
            fetch('/api/gates')
                .then(response => response.json())
                .then(gates => {
                    availableGates = gates;
                    populateGatePalette();
                    populateGateSelector();
                });
        }
        
        function populateGatePalette() {
            const palette = document.getElementById('gate-palette');
            const categories = {};
            
            // Group by category
            for (let gateName in availableGates) {
                const gate = availableGates[gateName];
                if (!categories[gate.category]) {
                    categories[gate.category] = [];
                }
                categories[gate.category].push(gate);
            }
            
            // Create UI
            for (let category in categories) {
                const div = document.createElement('div');
                div.className = 'gate-category';
                div.innerHTML = '<h3>' + category + '</h3>';
                
                categories[category].forEach(gate => {
                    const btn = document.createElement('button');
                    btn.className = 'gate-btn';
                    btn.textContent = gate.display_name;
                    btn.style.backgroundColor = gate.color;
                    btn.onclick = () => selectGate(gate.name);
                    div.appendChild(btn);
                });
                
                palette.appendChild(div);
            }
        }
        
        function populateGateSelector() {
            const selector = document.getElementById('selected-gate');
            selector.innerHTML = '<option value="">Select gate...</option>';
            
            for (let gateName in availableGates) {
                const option = document.createElement('option');
                option.value = gateName;
                option.textContent = availableGates[gateName].display_name;
                selector.appendChild(option);
            }
        }
        
        function selectGate(gateName) {
            document.getElementById('selected-gate').value = gateName;
        }
        
        function newCircuit() {
            const nQubits = prompt('Number of qubits:', '2');
            if (nQubits) {
                fetch('/api/circuit/new', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ n_qubits: parseInt(nQubits) })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        currentCircuit = data.circuit_id;
                        updateCircuitDisplay();
                    } else {
                        alert('Failed to create circuit: ' + data.error);
                    }
                });
            }
        }
        
        function addGate() {
            if (!currentCircuit) {
                alert('No circuit loaded');
                return;
            }
            
            const gateName = document.getElementById('selected-gate').value;
            const qubit = parseInt(document.getElementById('target-qubit').value);
            const param = parseFloat(document.getElementById('gate-param').value);
            
            if (!gateName) {
                alert('Please select a gate');
                return;
            }
            
            const gate = availableGates[gateName];
            let qubits = [qubit];
            let params = [];
            
            if (gate.num_qubits === 2) {
                const targetQubit = prompt('Enter target qubit:');
                if (targetQubit === null) return;
                qubits.push(parseInt(targetQubit));
            }
            
            if (gate.num_params > 0) {
                params.push(param);
            }
            
            fetch('/api/circuit/' + currentCircuit + '/add_gate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    gate_name: gateName, 
                    qubits: qubits, 
                    params: params 
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateCircuitDisplay();
                } else {
                    alert('Failed to add gate');
                }
            });
        }
        
        function clearCircuit() {
            if (currentCircuit) {
                // Get current circuit info to preserve qubit count
                fetch('/api/circuit/' + currentCircuit + '/info')
                    .then(response => response.json())
                    .then(info => {
                        newCircuitWithQubits(info.n_qubits);
                    });
            }
        }
        
        function newCircuitWithQubits(nQubits) {
            fetch('/api/circuit/new', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ n_qubits: nQubits })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    currentCircuit = data.circuit_id;
                    updateCircuitDisplay();
                }
            });
        }
        
        function updateCircuitDisplay() {
            if (!currentCircuit) return;
            
            fetch('/api/circuit/' + currentCircuit + '/info')
                .then(response => response.json())
                .then(info => {
                    // Update info display
                    document.getElementById('circuit-info').textContent = 
                        'Qubits: ' + info.n_qubits + ', Gates: ' + info.gate_count + ', Depth: ' + info.depth;
                    
                    // Update qubit input max
                    document.getElementById('target-qubit').max = info.n_qubits - 1;
                    
                    // Draw circuit
                    drawCircuit(info);
                });
        }
        
        function drawCircuit(info) {
            const canvas = document.getElementById('circuit-canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const margin = 50;
            const qubitSpacing = (canvas.height - 2 * margin) / Math.max(1, info.n_qubits - 1);
            const gateSpacing = 60;
            
            // Draw qubit lines
            for (let i = 0; i < info.n_qubits; i++) {
                const y = margin + i * (info.n_qubits > 1 ? qubitSpacing : 0);
                ctx.beginPath();
                ctx.moveTo(margin, y);
                ctx.lineTo(canvas.width - margin, y);
                ctx.stroke();
                
                // Qubit label
                ctx.fillText('q' + i, 10, y + 5);
            }
            
            // Draw gates
            info.elements.forEach((element, index) => {
                const x = margin + (index + 1) * gateSpacing;
                const qubits = element.qubits;
                
                if (element.gate_name === 'cnot') {
                    // Draw CNOT
                    const y1 = margin + qubits[0] * (info.n_qubits > 1 ? qubitSpacing : 0);
                    const y2 = margin + qubits[1] * (info.n_qubits > 1 ? qubitSpacing : 0);
                    
                    // Connection line
                    ctx.beginPath();
                    ctx.moveTo(x, y1);
                    ctx.lineTo(x, y2);
                    ctx.stroke();
                    
                    // Control dot
                    ctx.beginPath();
                    ctx.arc(x, y1, 4, 0, 2 * Math.PI);
                    ctx.fill();
                    
                    // Target circle
                    ctx.beginPath();
                    ctx.arc(x, y2, 8, 0, 2 * Math.PI);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(x - 5, y2);
                    ctx.lineTo(x + 5, y2);
                    ctx.moveTo(x, y2 - 5);
                    ctx.lineTo(x, y2 + 5);
                    ctx.stroke();
                } else {
                    // Single-qubit gate
                    const y = margin + qubits[0] * (info.n_qubits > 1 ? qubitSpacing : 0);
                    
                    // Gate box
                    ctx.fillStyle = element.color || '#4CAF50';
                    ctx.fillRect(x - 15, y - 10, 30, 20);
                    ctx.strokeRect(x - 15, y - 10, 30, 20);
                    
                    // Gate label
                    ctx.fillStyle = 'white';
                    ctx.textAlign = 'center';
                    ctx.fillText(element.symbol || element.gate_name.toUpperCase(), x, y + 4);
                    ctx.textAlign = 'left';
                    ctx.fillStyle = 'black';
                }
            });
        }
        
        function exportQASM() {
            if (!currentCircuit) {
                alert('No circuit loaded');
                return;
            }
            
            fetch('/api/circuit/' + currentCircuit + '/export/qasm')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('qasm-content').value = data.content;
                    document.getElementById('qasm-output').style.display = 'block';
                });
        }
    </script>
</body>
</html>
        '''
    
    def run(self, debug: bool = False):
        """Run the web server."""
        print(f"Starting web interface at http://{self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)


def create_circuit_builder(backend: Optional[str] = None) -> CircuitBuilder:
    """Create a circuit builder with specified backend."""
    if backend == "mock":
        return CircuitBuilder(MockCircuitBackend())
    elif backend == "quantrs2":
        if _NATIVE_AVAILABLE:
            return CircuitBuilder(QuantRS2Backend())
        else:
            print("Warning: Native QuantRS2 not available, using mock backend")
            return CircuitBuilder(MockCircuitBackend())
    else:
        return CircuitBuilder()  # Auto-select backend


def launch_gui(interface: str = "auto", **kwargs) -> None:
    """Launch the circuit builder GUI."""
    builder = create_circuit_builder()
    
    if interface == "auto":
        if TKINTER_AVAILABLE:
            interface = "tkinter"
        elif FLASK_AVAILABLE:
            interface = "web"
        else:
            raise RuntimeError("No GUI framework available")
    
    if interface == "tkinter":
        if not TKINTER_AVAILABLE:
            raise RuntimeError("Tkinter not available")
        gui = TkinterGUI(builder)
        gui.run()
    
    elif interface == "web":
        if not FLASK_AVAILABLE:
            raise RuntimeError("Flask not available")
        host = kwargs.get('host', 'localhost')
        port = kwargs.get('port', 5000)
        debug = kwargs.get('debug', False)
        gui = WebGUI(builder, host, port)
        gui.run(debug)
    
    else:
        raise ValueError(f"Unknown interface: {interface}")


# CLI interface for terminal usage
def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantRS2 Circuit Builder")
    parser.add_argument("--interface", choices=["tkinter", "web", "auto"], 
                       default="auto", help="GUI interface to use")
    parser.add_argument("--host", default="localhost", 
                       help="Host for web interface")
    parser.add_argument("--port", type=int, default=5000, 
                       help="Port for web interface")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode for web interface")
    
    args = parser.parse_args()
    
    try:
        launch_gui(
            interface=args.interface,
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    except Exception as e:
        print(f"Error launching GUI: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())