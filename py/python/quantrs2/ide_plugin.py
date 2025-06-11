"""
Quantum Development IDE Plugin

This module provides comprehensive IDE integration for quantum development,
including circuit visualization, debugging integration, code analysis, and more.
"""

import json
import time
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import socket
import http.server
import socketserver
import webbrowser

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    import io
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from . import algorithm_debugger
    from . import profiler
    from . import compilation_service
    from . import circuit_builder
    from . import qasm
    QUANTRS_MODULES_AVAILABLE = True
except ImportError:
    QUANTRS_MODULES_AVAILABLE = False


class IDEType(Enum):
    """Supported IDE types."""
    VSCODE = "vscode"
    PYCHARM = "pycharm"
    ATOM = "atom"
    SUBLIME = "sublime"
    VIM = "vim"
    EMACS = "emacs"
    JUPYTER = "jupyter"
    GENERIC = "generic"


class PluginState(Enum):
    """Plugin states."""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


class AnalysisType(Enum):
    """Code analysis types."""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    PERFORMANCE = "performance"
    OPTIMIZATION = "optimization"
    DEBUGGING = "debugging"


@dataclass
class CodeCompletionItem:
    """Represents a code completion suggestion."""
    label: str
    detail: str
    documentation: str
    insert_text: str
    kind: str = "function"  # function, class, variable, etc.
    priority: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'label': self.label,
            'detail': self.detail,
            'documentation': self.documentation,
            'insertText': self.insert_text,
            'kind': self.kind,
            'priority': self.priority
        }


@dataclass
class DiagnosticMessage:
    """Represents a diagnostic message (error, warning, info)."""
    line: int
    column: int
    message: str
    severity: str  # error, warning, info, hint
    source: str = "quantrs2"
    code: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'line': self.line,
            'column': self.column,
            'message': self.message,
            'severity': self.severity,
            'source': self.source,
            'code': self.code
        }


@dataclass
class HoverInfo:
    """Information to display on hover."""
    content: str
    content_type: str = "markdown"  # markdown, plaintext
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'contents': {'kind': self.content_type, 'value': self.content}
        }


class QuantumCodeAnalyzer:
    """Analyzes quantum code for various issues and optimizations."""
    
    def __init__(self):
        self.gate_patterns = {
            'single_qubit': ['h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz'],
            'two_qubit': ['cnot', 'cz', 'cx', 'cy'],
            'multi_qubit': ['ccx', 'ccz']
        }
        
        self.optimization_rules = [
            self._check_redundant_gates,
            self._check_gate_decomposition,
            self._check_circuit_depth,
            self._check_qubit_utilization
        ]
    
    def analyze_code(self, code: str, analysis_types: List[AnalysisType] = None) -> List[DiagnosticMessage]:
        """Analyze quantum code and return diagnostics."""
        if analysis_types is None:
            analysis_types = [AnalysisType.SYNTAX, AnalysisType.SEMANTIC]
        
        diagnostics = []
        lines = code.split('\n')
        
        try:
            # Parse circuit from code
            circuit_data = self._parse_circuit_from_code(code, lines)
            
            for analysis_type in analysis_types:
                if analysis_type == AnalysisType.SYNTAX:
                    diagnostics.extend(self._analyze_syntax(lines))
                elif analysis_type == AnalysisType.SEMANTIC:
                    diagnostics.extend(self._analyze_semantics(circuit_data, lines))
                elif analysis_type == AnalysisType.PERFORMANCE:
                    diagnostics.extend(self._analyze_performance(circuit_data, lines))
                elif analysis_type == AnalysisType.OPTIMIZATION:
                    diagnostics.extend(self._analyze_optimizations(circuit_data, lines))
        
        except Exception as e:
            diagnostics.append(DiagnosticMessage(
                line=0, column=0,
                message=f"Analysis error: {str(e)}",
                severity="error"
            ))
        
        return diagnostics
    
    def _parse_circuit_from_code(self, code: str, lines: List[str]) -> Dict[str, Any]:
        """Extract circuit information from Python code."""
        circuit_data = {
            'n_qubits': 0,
            'gates': [],
            'measurements': []
        }
        
        # Simple pattern matching for quantum operations
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for circuit creation
            if 'PyCircuit(' in line or 'Circuit(' in line:
                # Extract number of qubits
                try:
                    parts = line.split('(')[1].split(')')[0]
                    if parts.strip().isdigit():
                        circuit_data['n_qubits'] = int(parts.strip())
                except:
                    pass
            
            # Look for gate operations
            for gate in self.gate_patterns['single_qubit'] + self.gate_patterns['two_qubit']:
                if f'.{gate}(' in line:
                    # Extract qubit indices
                    try:
                        gate_part = line.split(f'.{gate}(')[1].split(')')[0]
                        qubits = [int(x.strip()) for x in gate_part.split(',') if x.strip().isdigit()]
                        circuit_data['gates'].append({
                            'gate': gate,
                            'qubits': qubits,
                            'line': i + 1
                        })
                    except:
                        pass
        
        return circuit_data
    
    def _analyze_syntax(self, lines: List[str]) -> List[DiagnosticMessage]:
        """Analyze syntax issues."""
        diagnostics = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for common syntax issues
            if line_stripped and not line_stripped.startswith('#'):
                # Unclosed parentheses
                if line_stripped.count('(') != line_stripped.count(')'):
                    diagnostics.append(DiagnosticMessage(
                        line=i + 1, column=0,
                        message="Unclosed parentheses detected",
                        severity="error"
                    ))
                
                # Missing import
                if any(gate in line_stripped for gate in self.gate_patterns['single_qubit']):
                    if not any('import' in prev_line and 'quantrs2' in prev_line 
                             for prev_line in lines[:i]):
                        diagnostics.append(DiagnosticMessage(
                            line=i + 1, column=0,
                            message="Quantum gates used without importing quantrs2",
                            severity="warning"
                        ))
        
        return diagnostics
    
    def _analyze_semantics(self, circuit_data: Dict[str, Any], lines: List[str]) -> List[DiagnosticMessage]:
        """Analyze semantic issues."""
        diagnostics = []
        n_qubits = circuit_data.get('n_qubits', 0)
        
        # Check qubit bounds
        for gate_info in circuit_data.get('gates', []):
            qubits = gate_info.get('qubits', [])
            for qubit in qubits:
                if qubit >= n_qubits:
                    diagnostics.append(DiagnosticMessage(
                        line=gate_info.get('line', 1), column=0,
                        message=f"Qubit index {qubit} out of bounds (circuit has {n_qubits} qubits)",
                        severity="error"
                    ))
        
        # Check gate arity
        for gate_info in circuit_data.get('gates', []):
            gate = gate_info.get('gate', '')
            qubits = gate_info.get('qubits', [])
            
            if gate in self.gate_patterns['single_qubit'] and len(qubits) != 1:
                diagnostics.append(DiagnosticMessage(
                    line=gate_info.get('line', 1), column=0,
                    message=f"Gate {gate} expects 1 qubit, got {len(qubits)}",
                    severity="error"
                ))
            elif gate in self.gate_patterns['two_qubit'] and len(qubits) != 2:
                diagnostics.append(DiagnosticMessage(
                    line=gate_info.get('line', 1), column=0,
                    message=f"Gate {gate} expects 2 qubits, got {len(qubits)}",
                    severity="error"
                ))
        
        return diagnostics
    
    def _analyze_performance(self, circuit_data: Dict[str, Any], lines: List[str]) -> List[DiagnosticMessage]:
        """Analyze performance issues."""
        diagnostics = []
        gates = circuit_data.get('gates', [])
        
        # Check circuit depth
        if len(gates) > 100:
            diagnostics.append(DiagnosticMessage(
                line=1, column=0,
                message=f"Circuit depth ({len(gates)}) may impact performance. Consider optimization.",
                severity="info"
            ))
        
        # Check for excessive two-qubit gates
        two_qubit_gates = [g for g in gates if g.get('gate', '') in self.gate_patterns['two_qubit']]
        if len(two_qubit_gates) > 20:
            diagnostics.append(DiagnosticMessage(
                line=1, column=0,
                message=f"High number of two-qubit gates ({len(two_qubit_gates)}) may reduce fidelity",
                severity="warning"
            ))
        
        return diagnostics
    
    def _analyze_optimizations(self, circuit_data: Dict[str, Any], lines: List[str]) -> List[DiagnosticMessage]:
        """Analyze optimization opportunities."""
        diagnostics = []
        
        for rule in self.optimization_rules:
            diagnostics.extend(rule(circuit_data, lines))
        
        return diagnostics
    
    def _check_redundant_gates(self, circuit_data: Dict[str, Any], lines: List[str]) -> List[DiagnosticMessage]:
        """Check for redundant gate sequences."""
        diagnostics = []
        gates = circuit_data.get('gates', [])
        
        # Check for X-X cancellation
        for i in range(len(gates) - 1):
            gate1 = gates[i]
            gate2 = gates[i + 1]
            
            if (gate1.get('gate') == 'x' and gate2.get('gate') == 'x' and
                gate1.get('qubits') == gate2.get('qubits')):
                diagnostics.append(DiagnosticMessage(
                    line=gate1.get('line', 1), column=0,
                    message="Redundant X-X gate sequence can be removed",
                    severity="info"
                ))
        
        return diagnostics
    
    def _check_gate_decomposition(self, circuit_data: Dict[str, Any], lines: List[str]) -> List[DiagnosticMessage]:
        """Check for suboptimal gate decompositions."""
        diagnostics = []
        gates = circuit_data.get('gates', [])
        
        # Check for H-Z-H pattern (can be X)
        for i in range(len(gates) - 2):
            if (gates[i].get('gate') == 'h' and 
                gates[i+1].get('gate') == 'z' and 
                gates[i+2].get('gate') == 'h' and
                gates[i].get('qubits') == gates[i+1].get('qubits') == gates[i+2].get('qubits')):
                diagnostics.append(DiagnosticMessage(
                    line=gates[i].get('line', 1), column=0,
                    message="H-Z-H sequence can be replaced with single X gate",
                    severity="info"
                ))
        
        return diagnostics
    
    def _check_circuit_depth(self, circuit_data: Dict[str, Any], lines: List[str]) -> List[DiagnosticMessage]:
        """Check circuit depth optimization opportunities."""
        diagnostics = []
        
        # Simple depth analysis
        gates = circuit_data.get('gates', [])
        n_qubits = circuit_data.get('n_qubits', 1)
        
        if len(gates) > n_qubits * 10:
            diagnostics.append(DiagnosticMessage(
                line=1, column=0,
                message="Consider using compilation optimization to reduce circuit depth",
                severity="info"
            ))
        
        return diagnostics
    
    def _check_qubit_utilization(self, circuit_data: Dict[str, Any], lines: List[str]) -> List[DiagnosticMessage]:
        """Check qubit utilization patterns."""
        diagnostics = []
        gates = circuit_data.get('gates', [])
        n_qubits = circuit_data.get('n_qubits', 0)
        
        if n_qubits > 0:
            used_qubits = set()
            for gate in gates:
                used_qubits.update(gate.get('qubits', []))
            
            unused_qubits = set(range(n_qubits)) - used_qubits
            if unused_qubits:
                diagnostics.append(DiagnosticMessage(
                    line=1, column=0,
                    message=f"Unused qubits: {sorted(unused_qubits)}. Consider reducing circuit size.",
                    severity="info"
                ))
        
        return diagnostics


class QuantumCodeCompletion:
    """Provides code completion for quantum programming."""
    
    def __init__(self):
        self.completions = self._build_completion_database()
    
    def _build_completion_database(self) -> List[CodeCompletionItem]:
        """Build database of completion items."""
        completions = []
        
        # Basic quantum gates
        gate_completions = [
            CodeCompletionItem(
                label="h",
                detail="Hadamard gate",
                documentation="Apply Hadamard gate to put qubit in superposition",
                insert_text="h(${1:qubit})"
            ),
            CodeCompletionItem(
                label="x",
                detail="Pauli-X gate (NOT)",
                documentation="Apply Pauli-X gate (bit flip)",
                insert_text="x(${1:qubit})"
            ),
            CodeCompletionItem(
                label="y",
                detail="Pauli-Y gate",
                documentation="Apply Pauli-Y gate",
                insert_text="y(${1:qubit})"
            ),
            CodeCompletionItem(
                label="z",
                detail="Pauli-Z gate",
                documentation="Apply Pauli-Z gate (phase flip)",
                insert_text="z(${1:qubit})"
            ),
            CodeCompletionItem(
                label="cnot",
                detail="CNOT gate",
                documentation="Apply controlled-NOT gate",
                insert_text="cnot(${1:control}, ${2:target})"
            ),
            CodeCompletionItem(
                label="rx",
                detail="X-rotation gate",
                documentation="Apply rotation around X-axis",
                insert_text="rx(${1:qubit}, ${2:angle})"
            ),
            CodeCompletionItem(
                label="ry",
                detail="Y-rotation gate",
                documentation="Apply rotation around Y-axis",
                insert_text="ry(${1:qubit}, ${2:angle})"
            ),
            CodeCompletionItem(
                label="rz",
                detail="Z-rotation gate",
                documentation="Apply rotation around Z-axis",
                insert_text="rz(${1:qubit}, ${2:angle})"
            )
        ]
        
        # Circuit operations
        circuit_completions = [
            CodeCompletionItem(
                label="PyCircuit",
                detail="Create quantum circuit",
                documentation="Create a new quantum circuit with specified number of qubits",
                insert_text="PyCircuit(${1:n_qubits})",
                kind="class"
            ),
            CodeCompletionItem(
                label="run",
                detail="Run circuit simulation",
                documentation="Execute the quantum circuit and return results",
                insert_text="run(use_gpu=${1:False})"
            ),
            CodeCompletionItem(
                label="measure",
                detail="Measure qubit",
                documentation="Add measurement operation to circuit",
                insert_text="measure(${1:qubit})"
            )
        ]
        
        # Advanced features
        advanced_completions = [
            CodeCompletionItem(
                label="bell_state",
                detail="Create Bell state",
                documentation="Create entangled Bell state between two qubits",
                insert_text="bell_state()",
                kind="function"
            ),
            CodeCompletionItem(
                label="QNN",
                detail="Quantum Neural Network",
                documentation="Create quantum neural network",
                insert_text="QNN(${1:n_qubits}, ${2:n_layers})",
                kind="class"
            ),
            CodeCompletionItem(
                label="VQE",
                detail="Variational Quantum Eigensolver",
                documentation="Create VQE algorithm instance",
                insert_text="VQE(${1:hamiltonian})",
                kind="class"
            )
        ]
        
        # Debugging and analysis
        debug_completions = [
            CodeCompletionItem(
                label="debug_quantum_algorithm",
                detail="Start debugging session",
                documentation="Begin step-by-step debugging of quantum algorithm",
                insert_text="debug_quantum_algorithm(${1:circuit_data})",
                kind="function"
            ),
            CodeCompletionItem(
                label="profile_circuit",
                detail="Profile circuit performance",
                documentation="Analyze circuit performance and resource usage",
                insert_text="profile_circuit(${1:circuit})",
                kind="function"
            )
        ]
        
        completions.extend(gate_completions)
        completions.extend(circuit_completions)
        completions.extend(advanced_completions)
        completions.extend(debug_completions)
        
        return completions
    
    def get_completions(self, context: str, position: Tuple[int, int]) -> List[CodeCompletionItem]:
        """Get completion suggestions for given context."""
        # Simple context-aware filtering
        line_text = context.split('\n')[position[0]] if position[0] < len(context.split('\n')) else ""
        
        # If we're after a dot, suggest methods
        if '.' in line_text:
            parts = line_text.split('.')
            if len(parts) >= 2:
                obj_part = parts[-2].strip()
                if 'circuit' in obj_part.lower() or 'PyCircuit' in obj_part:
                    # Return gate completions
                    return [c for c in self.completions if c.kind == "function" and c.label in 
                           ['h', 'x', 'y', 'z', 'cnot', 'rx', 'ry', 'rz', 'run', 'measure']]
        
        # General completions
        return self.completions


class QuantumHoverProvider:
    """Provides hover information for quantum code elements."""
    
    def __init__(self):
        self.hover_info = self._build_hover_database()
    
    def _build_hover_database(self) -> Dict[str, HoverInfo]:
        """Build database of hover information."""
        hover_db = {}
        
        # Quantum gates
        hover_db['h'] = HoverInfo(
            content="""## Hadamard Gate
            
The Hadamard gate creates superposition:
- |0⟩ → (|0⟩ + |1⟩)/√2
- |1⟩ → (|0⟩ - |1⟩)/√2

**Matrix representation:**
```
1/√2 * [[1,  1],
         [1, -1]]
```""")
        
        hover_db['x'] = HoverInfo(
            content="""## Pauli-X Gate (NOT Gate)
            
The X gate flips qubit states:
- |0⟩ → |1⟩
- |1⟩ → |0⟩

**Matrix representation:**
```
[[0, 1],
 [1, 0]]
```""")
        
        hover_db['cnot'] = HoverInfo(
            content="""## CNOT Gate (Controlled-NOT)
            
Two-qubit gate that flips target if control is |1⟩:
- |00⟩ → |00⟩
- |01⟩ → |01⟩ 
- |10⟩ → |11⟩
- |11⟩ → |10⟩

**Matrix representation:**
```
[[1, 0, 0, 0],
 [0, 1, 0, 0],
 [0, 0, 0, 1],
 [0, 0, 1, 0]]
```""")
        
        # Circuit elements
        hover_db['PyCircuit'] = HoverInfo(
            content="""## Quantum Circuit
            
Main class for building quantum circuits.

**Usage:**
```python
circuit = PyCircuit(n_qubits)
circuit.h(0)  # Apply Hadamard to qubit 0
result = circuit.run()
```""")
        
        # Advanced concepts
        hover_db['bell_state'] = HoverInfo(
            content="""## Bell State
            
Maximally entangled two-qubit state:
|Φ+⟩ = (|00⟩ + |11⟩)/√2

Bell states are fundamental for quantum communication protocols.""")
        
        return hover_db
    
    def get_hover_info(self, word: str, context: str) -> Optional[HoverInfo]:
        """Get hover information for a word."""
        return self.hover_info.get(word.lower())


class IDEPluginServer:
    """HTTP server for IDE plugin communication."""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.analyzer = QuantumCodeAnalyzer()
        self.completion_provider = QuantumCodeCompletion()
        self.hover_provider = QuantumHoverProvider()
        self.server = None
        self.server_thread = None
        
        # Integration with other QuantRS2 modules
        self.debugger = None
        self.profiler = None
        
        if QUANTRS_MODULES_AVAILABLE:
            try:
                self.debugger = algorithm_debugger.get_algorithm_debugger()
                self.profiler = profiler.CircuitProfiler()
            except:
                pass
    
    def start_server(self) -> bool:
        """Start the plugin server."""
        try:
            handler = self._create_request_handler()
            self.server = socketserver.TCPServer(("localhost", self.port), handler)
            
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to start plugin server: {e}")
            return False
    
    def stop_server(self) -> None:
        """Stop the plugin server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        
        if self.server_thread:
            self.server_thread.join(timeout=1.0)
            self.server_thread = None
    
    def _create_request_handler(self):
        """Create HTTP request handler class."""
        analyzer = self.analyzer
        completion_provider = self.completion_provider
        hover_provider = self.hover_provider
        debugger = self.debugger
        profiler = self.profiler
        
        class PluginRequestHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                """Handle POST requests."""
                try:
                    content_length = int(self.headers['Content-Length'])
                    request_data = json.loads(self.rfile.read(content_length).decode('utf-8'))
                    
                    method = request_data.get('method')
                    params = request_data.get('params', {})
                    
                    if method == 'analyze':
                        response = self._handle_analyze(params)
                    elif method == 'complete':
                        response = self._handle_completion(params)
                    elif method == 'hover':
                        response = self._handle_hover(params)
                    elif method == 'debug':
                        response = self._handle_debug(params)
                    elif method == 'profile':
                        response = self._handle_profile(params)
                    elif method == 'visualize':
                        response = self._handle_visualize(params)
                    else:
                        response = {'error': f'Unknown method: {method}'}
                    
                    self._send_json_response(response)
                    
                except Exception as e:
                    self._send_json_response({'error': str(e)})
            
            def _handle_analyze(self, params):
                """Handle code analysis request."""
                code = params.get('code', '')
                analysis_types = [AnalysisType(t) for t in params.get('types', ['syntax', 'semantic'])]
                
                diagnostics = analyzer.analyze_code(code, analysis_types)
                return {
                    'diagnostics': [d.to_dict() for d in diagnostics]
                }
            
            def _handle_completion(self, params):
                """Handle code completion request."""
                code = params.get('code', '')
                position = params.get('position', [0, 0])
                
                completions = completion_provider.get_completions(code, tuple(position))
                return {
                    'completions': [c.to_dict() for c in completions]
                }
            
            def _handle_hover(self, params):
                """Handle hover information request."""
                word = params.get('word', '')
                context = params.get('context', '')
                
                hover_info = hover_provider.get_hover_info(word, context)
                if hover_info:
                    return {'hover': hover_info.to_dict()}
                else:
                    return {'hover': None}
            
            def _handle_debug(self, params):
                """Handle debugging request."""
                if not debugger:
                    return {'error': 'Debugger not available'}
                
                action = params.get('action')
                session_id = params.get('session_id')
                
                if action == 'start':
                    circuit_data = params.get('circuit_data', {})
                    session_id = debugger.create_debug_session(circuit_data)
                    return {'session_id': session_id}
                
                elif action == 'step' and session_id:
                    success = debugger.step_forward(session_id)
                    state = debugger.get_current_state(session_id)
                    return {
                        'success': success,
                        'state': state.to_dict() if state else None
                    }
                
                elif action == 'continue' and session_id:
                    success = debugger.continue_execution(session_id)
                    state = debugger.get_current_state(session_id)
                    return {
                        'success': success,
                        'state': state.to_dict() if state else None
                    }
                
                else:
                    return {'error': 'Invalid debug action'}
            
            def _handle_profile(self, params):
                """Handle profiling request."""
                if not profiler:
                    return {'error': 'Profiler not available'}
                
                circuit_data = params.get('circuit_data', {})
                # Simplified profiling
                return {
                    'profile': {
                        'gate_count': len(circuit_data.get('gates', [])),
                        'qubit_count': circuit_data.get('n_qubits', 0),
                        'depth': len(circuit_data.get('gates', [])),
                        'two_qubit_gates': len([g for g in circuit_data.get('gates', []) 
                                              if len(g.get('qubits', [])) >= 2])
                    }
                }
            
            def _handle_visualize(self, params):
                """Handle visualization request."""
                circuit_data = params.get('circuit_data', {})
                
                if MATPLOTLIB_AVAILABLE:
                    # Generate SVG circuit diagram
                    svg_data = self._generate_circuit_svg(circuit_data)
                    return {'svg': svg_data}
                else:
                    return {'error': 'Matplotlib not available for visualization'}
            
            def _generate_circuit_svg(self, circuit_data):
                """Generate SVG representation of circuit."""
                try:
                    n_qubits = circuit_data.get('n_qubits', 1)
                    gates = circuit_data.get('gates', [])
                    
                    fig, ax = plt.subplots(figsize=(max(8, len(gates)), max(4, n_qubits)))
                    
                    # Draw qubit lines
                    for i in range(n_qubits):
                        ax.plot([0, len(gates) + 1], [i, i], 'k-', linewidth=2)
                    
                    # Draw gates
                    for step, gate_data in enumerate(gates):
                        gate_name = gate_data.get('gate', '')
                        qubits = gate_data.get('qubits', [])
                        x_pos = step + 1
                        
                        if len(qubits) == 1:
                            qubit = qubits[0]
                            rect = patches.Rectangle((x_pos - 0.2, qubit - 0.2), 0.4, 0.4,
                                                   linewidth=1, edgecolor='black', facecolor='lightblue')
                            ax.add_patch(rect)
                            ax.text(x_pos, qubit, gate_name.upper(), ha='center', va='center')
                    
                    ax.set_xlim(-0.5, len(gates) + 1.5)
                    ax.set_ylim(-0.5, n_qubits - 0.5)
                    ax.set_aspect('equal')
                    
                    # Convert to SVG
                    buffer = io.StringIO()
                    canvas = FigureCanvasSVG(fig)
                    canvas.print_svg(buffer)
                    plt.close(fig)
                    
                    return buffer.getvalue()
                    
                except Exception as e:
                    return f"<!-- SVG generation error: {e} -->"
            
            def _send_json_response(self, data):
                """Send JSON response."""
                response_json = json.dumps(data)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response_json)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                self.wfile.write(response_json.encode('utf-8'))
            
            def log_message(self, format, *args):
                """Suppress default logging."""
                pass
        
        return PluginRequestHandler


class VSCodePlugin:
    """VS Code specific plugin implementation."""
    
    def __init__(self):
        self.extension_path = None
        self.server = IDEPluginServer()
    
    def install(self) -> bool:
        """Install VS Code extension."""
        try:
            # Generate extension files
            extension_dir = self._create_extension_directory()
            if not extension_dir:
                return False
            
            self._generate_package_json(extension_dir)
            self._generate_main_js(extension_dir)
            self._generate_language_config(extension_dir)
            
            # Start server
            if not self.server.start_server():
                return False
            
            print(f"VS Code extension generated at: {extension_dir}")
            print("To install:")
            print(f"1. Open VS Code")
            print(f"2. Open Command Palette (Ctrl+Shift+P)")
            print(f"3. Run 'Developer: Install Extension from Location'")
            print(f"4. Select the directory: {extension_dir}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to install VS Code plugin: {e}")
            return False
    
    def _create_extension_directory(self) -> Optional[Path]:
        """Create extension directory structure."""
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="quantrs2_vscode_"))
            extension_dir = temp_dir / "quantrs2-extension"
            extension_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            (extension_dir / "src").mkdir(exist_ok=True)
            (extension_dir / "syntaxes").mkdir(exist_ok=True)
            
            self.extension_path = extension_dir
            return extension_dir
            
        except Exception as e:
            logging.error(f"Failed to create extension directory: {e}")
            return None
    
    def _generate_package_json(self, extension_dir: Path) -> None:
        """Generate package.json for VS Code extension."""
        package_json = {
            "name": "quantrs2-quantum-development",
            "displayName": "QuantRS2 Quantum Development",
            "description": "Comprehensive quantum programming support for QuantRS2",
            "version": "0.1.0",
            "publisher": "quantrs2",
            "engines": {"vscode": "^1.60.0"},
            "categories": ["Programming Languages", "Debuggers", "Other"],
            "main": "./src/extension.js",
            "contributes": {
                "languages": [{
                    "id": "quantum-python",
                    "aliases": ["Quantum Python"],
                    "extensions": [".qpy"],
                    "configuration": "./language-configuration.json"
                }],
                "grammars": [{
                    "language": "quantum-python",
                    "scopeName": "source.quantum.python",
                    "path": "./syntaxes/quantum-python.tmGrammar.json"
                }],
                "commands": [
                    {
                        "command": "quantrs2.startDebugger",
                        "title": "Start Quantum Debugger",
                        "category": "QuantRS2"
                    },
                    {
                        "command": "quantrs2.profileCircuit",
                        "title": "Profile Quantum Circuit",
                        "category": "QuantRS2"
                    },
                    {
                        "command": "quantrs2.visualizeCircuit",
                        "title": "Visualize Circuit",
                        "category": "QuantRS2"
                    }
                ],
                "menus": {
                    "editor/context": [
                        {
                            "command": "quantrs2.startDebugger",
                            "when": "editorTextFocus && resourceExtname == .py",
                            "group": "quantrs2"
                        }
                    ]
                }
            },
            "scripts": {
                "vscode:prepublish": "npm run compile",
                "compile": "node ./src/extension.js"
            }
        }
        
        with open(extension_dir / "package.json", 'w') as f:
            json.dump(package_json, f, indent=2)
    
    def _generate_main_js(self, extension_dir: Path) -> None:
        """Generate main extension JavaScript file."""
        main_js = '''
const vscode = require('vscode');
const http = require('http');

let quantumServer = null;

function activate(context) {
    console.log('QuantRS2 extension is now active!');
    
    // Start quantum analysis server
    startQuantumServer();
    
    // Register commands
    let debugCommand = vscode.commands.registerCommand('quantrs2.startDebugger', () => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            startQuantumDebugger(editor.document.getText());
        }
    });
    
    let profileCommand = vscode.commands.registerCommand('quantrs2.profileCircuit', () => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            profileQuantumCircuit(editor.document.getText());
        }
    });
    
    let visualizeCommand = vscode.commands.registerCommand('quantrs2.visualizeCircuit', () => {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            visualizeQuantumCircuit(editor.document.getText());
        }
    });
    
    context.subscriptions.push(debugCommand, profileCommand, visualizeCommand);
    
    // Register diagnostic provider
    const diagnosticCollection = vscode.languages.createDiagnosticCollection('quantrs2');
    context.subscriptions.push(diagnosticCollection);
    
    // Analyze on document change
    vscode.workspace.onDidChangeTextDocument(event => {
        if (event.document.languageId === 'python') {
            analyzeQuantumCode(event.document, diagnosticCollection);
        }
    });
    
    // Register completion provider
    const completionProvider = vscode.languages.registerCompletionItemProvider(
        'python',
        {
            provideCompletionItems(document, position) {
                return getQuantumCompletions(document, position);
            }
        },
        '.'
    );
    context.subscriptions.push(completionProvider);
    
    // Register hover provider
    const hoverProvider = vscode.languages.registerHoverProvider('python', {
        provideHover(document, position) {
            return getQuantumHover(document, position);
        }
    });
    context.subscriptions.push(hoverProvider);
}

function startQuantumServer() {
    // Server is already started by Python plugin
    quantumServer = 'http://localhost:8765';
}

async function analyzeQuantumCode(document, diagnosticCollection) {
    try {
        const response = await makeRequest('analyze', {
            code: document.getText(),
            types: ['syntax', 'semantic', 'optimization']
        });
        
        const diagnostics = response.diagnostics.map(d => {
            const range = new vscode.Range(d.line - 1, d.column, d.line - 1, d.column + 10);
            const severity = d.severity === 'error' ? vscode.DiagnosticSeverity.Error :
                           d.severity === 'warning' ? vscode.DiagnosticSeverity.Warning :
                           vscode.DiagnosticSeverity.Information;
            
            return new vscode.Diagnostic(range, d.message, severity);
        });
        
        diagnosticCollection.set(document.uri, diagnostics);
        
    } catch (error) {
        console.error('Quantum analysis error:', error);
    }
}

async function getQuantumCompletions(document, position) {
    try {
        const response = await makeRequest('complete', {
            code: document.getText(),
            position: [position.line, position.character]
        });
        
        return response.completions.map(c => {
            const item = new vscode.CompletionItem(c.label, vscode.CompletionItemKind.Function);
            item.detail = c.detail;
            item.documentation = c.documentation;
            item.insertText = new vscode.SnippetString(c.insertText);
            return item;
        });
        
    } catch (error) {
        console.error('Completion error:', error);
        return [];
    }
}

async function getQuantumHover(document, position) {
    try {
        const wordRange = document.getWordRangeAtPosition(position);
        if (!wordRange) return null;
        
        const word = document.getText(wordRange);
        const response = await makeRequest('hover', {
            word: word,
            context: document.getText()
        });
        
        if (response.hover) {
            return new vscode.Hover(
                new vscode.MarkdownString(response.hover.contents.value)
            );
        }
        
    } catch (error) {
        console.error('Hover error:', error);
    }
    
    return null;
}

async function startQuantumDebugger(code) {
    try {
        // Parse circuit data from code (simplified)
        const circuitData = parseCircuitData(code);
        
        const response = await makeRequest('debug', {
            action: 'start',
            circuit_data: circuitData
        });
        
        if (response.session_id) {
            vscode.window.showInformationMessage(
                `Quantum debugger started: ${response.session_id}`,
                'Step Forward', 'Continue'
            ).then(selection => {
                if (selection === 'Step Forward') {
                    stepQuantumDebugger(response.session_id);
                } else if (selection === 'Continue') {
                    continueQuantumDebugger(response.session_id);
                }
            });
        }
        
    } catch (error) {
        vscode.window.showErrorMessage(`Debug error: ${error.message}`);
    }
}

async function stepQuantumDebugger(sessionId) {
    try {
        const response = await makeRequest('debug', {
            action: 'step',
            session_id: sessionId
        });
        
        if (response.success && response.state) {
            const state = response.state;
            vscode.window.showInformationMessage(
                `Step ${state.step_number}: Entropy = ${state.entanglement_entropy.toFixed(4)}`
            );
        }
        
    } catch (error) {
        vscode.window.showErrorMessage(`Step error: ${error.message}`);
    }
}

async function profileQuantumCircuit(code) {
    try {
        const circuitData = parseCircuitData(code);
        
        const response = await makeRequest('profile', {
            circuit_data: circuitData
        });
        
        const profile = response.profile;
        vscode.window.showInformationMessage(
            `Circuit Profile - Gates: ${profile.gate_count}, Depth: ${profile.depth}, 2Q Gates: ${profile.two_qubit_gates}`
        );
        
    } catch (error) {
        vscode.window.showErrorMessage(`Profile error: ${error.message}`);
    }
}

async function visualizeQuantumCircuit(code) {
    try {
        const circuitData = parseCircuitData(code);
        
        const response = await makeRequest('visualize', {
            circuit_data: circuitData
        });
        
        if (response.svg) {
            // Create webview to show SVG
            const panel = vscode.window.createWebviewPanel(
                'quantumCircuit',
                'Quantum Circuit',
                vscode.ViewColumn.Two,
                {}
            );
            
            panel.webview.html = `
                <html>
                <body>
                    <h2>Quantum Circuit Visualization</h2>
                    ${response.svg}
                </body>
                </html>
            `;
        }
        
    } catch (error) {
        vscode.window.showErrorMessage(`Visualization error: ${error.message}`);
    }
}

function parseCircuitData(code) {
    // Simplified circuit parsing from Python code
    const lines = code.split('\\n');
    const circuitData = { n_qubits: 1, gates: [] };
    
    lines.forEach((line, index) => {
        if (line.includes('PyCircuit(') || line.includes('Circuit(')) {
            const match = line.match(/\\((\\d+)\\)/);
            if (match) {
                circuitData.n_qubits = parseInt(match[1]);
            }
        }
        
        // Simple gate detection
        const gatePattern = /\\.(h|x|y|z|cnot|rx|ry|rz)\\s*\\(/;
        const match = line.match(gatePattern);
        if (match) {
            circuitData.gates.push({
                gate: match[1],
                qubits: [0], // Simplified
                line: index + 1
            });
        }
    });
    
    return circuitData;
}

async function makeRequest(method, params) {
    return new Promise((resolve, reject) => {
        const data = JSON.stringify({ method, params });
        
        const options = {
            hostname: 'localhost',
            port: 8765,
            path: '/',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(data)
            }
        };
        
        const req = http.request(options, (res) => {
            let body = '';
            res.on('data', chunk => body += chunk);
            res.on('end', () => {
                try {
                    resolve(JSON.parse(body));
                } catch (e) {
                    reject(e);
                }
            });
        });
        
        req.on('error', reject);
        req.write(data);
        req.end();
    });
}

function deactivate() {
    // Cleanup
}

module.exports = { activate, deactivate };
'''
        
        with open(extension_dir / "src" / "extension.js", 'w') as f:
            f.write(main_js)
    
    def _generate_language_config(self, extension_dir: Path) -> None:
        """Generate language configuration files."""
        # Language configuration
        lang_config = {
            "comments": {
                "lineComment": "#",
                "blockComment": ["'''", "'''"]
            },
            "brackets": [
                ["[", "]"],
                ["(", ")"],
                ["{", "}"]
            ],
            "autoClosingPairs": [
                ["[", "]"],
                ["(", ")"],
                ["{", "}"],
                ["\"", "\""],
                ["'", "'"]
            ],
            "surroundingPairs": [
                ["[", "]"],
                ["(", ")"],
                ["{", "}"],
                ["\"", "\""],
                ["'", "'"]
            ]
        }
        
        with open(extension_dir / "language-configuration.json", 'w') as f:
            json.dump(lang_config, f, indent=2)


class QuantumIDEPlugin:
    """Main IDE plugin interface."""
    
    def __init__(self):
        self.state = PluginState.INACTIVE
        self.supported_ides = {
            IDEType.VSCODE: VSCodePlugin,
            IDEType.JUPYTER: self._create_jupyter_plugin,
            IDEType.GENERIC: self._create_generic_plugin
        }
        self.active_plugins = {}
        self.server = IDEPluginServer()
    
    def install_for_ide(self, ide_type: IDEType) -> bool:
        """Install plugin for specific IDE."""
        try:
            self.state = PluginState.INITIALIZING
            
            if ide_type not in self.supported_ides:
                logging.error(f"IDE {ide_type} not supported")
                self.state = PluginState.ERROR
                return False
            
            # Start server first
            if not self.server.start_server():
                self.state = PluginState.ERROR
                return False
            
            # Create and install plugin
            if ide_type == IDEType.VSCODE:
                plugin = VSCodePlugin()
                success = plugin.install()
            elif ide_type == IDEType.JUPYTER:
                success = self._install_jupyter_plugin()
            else:
                success = self._install_generic_plugin()
            
            if success:
                self.active_plugins[ide_type] = plugin if ide_type == IDEType.VSCODE else True
                self.state = PluginState.ACTIVE
                return True
            else:
                self.state = PluginState.ERROR
                return False
                
        except Exception as e:
            logging.error(f"Plugin installation failed: {e}")
            self.state = PluginState.ERROR
            return False
    
    def _install_jupyter_plugin(self) -> bool:
        """Install Jupyter plugin."""
        try:
            # Create Jupyter magic commands
            jupyter_magic = '''
from IPython.core.magic import Magics, line_magic, cell_magic, magics_class
from IPython.core.display import display, HTML, SVG
import requests
import json

@magics_class
class QuantRS2Magic(Magics):
    
    @line_magic
    def quantrs2_debug(self, line):
        """Start quantum debugging session."""
        # Parse circuit from current cell or previous cells
        circuit_data = self._extract_circuit_data()
        
        response = requests.post('http://localhost:8765', 
                               json={'method': 'debug', 'params': {
                                   'action': 'start',
                                   'circuit_data': circuit_data
                               }})
        
        if response.ok:
            result = response.json()
            print(f"Debug session started: {result.get('session_id')}")
        else:
            print("Failed to start debugging session")
    
    @line_magic 
    def quantrs2_profile(self, line):
        """Profile quantum circuit."""
        circuit_data = self._extract_circuit_data()
        
        response = requests.post('http://localhost:8765',
                               json={'method': 'profile', 'params': {
                                   'circuit_data': circuit_data
                               }})
        
        if response.ok:
            result = response.json()
            profile = result.get('profile', {})
            print(f"Circuit Profile:")
            print(f"  Gates: {profile.get('gate_count', 0)}")
            print(f"  Depth: {profile.get('depth', 0)}")
            print(f"  Two-qubit gates: {profile.get('two_qubit_gates', 0)}")
    
    @cell_magic
    def quantrs2_visualize(self, line, cell):
        """Visualize quantum circuit."""
        # Execute cell to get circuit
        exec(cell, self.shell.user_ns)
        
        circuit_data = self._extract_circuit_data()
        
        response = requests.post('http://localhost:8765',
                               json={'method': 'visualize', 'params': {
                                   'circuit_data': circuit_data
                               }})
        
        if response.ok:
            result = response.json()
            svg_data = result.get('svg', '')
            if svg_data:
                display(SVG(svg_data))
            else:
                print("No visualization available")
    
    def _extract_circuit_data(self):
        """Extract circuit data from notebook context."""
        # Simplified extraction - look for circuit variables
        user_ns = self.shell.user_ns
        
        circuit_data = {'n_qubits': 1, 'gates': []}
        
        # Look for PyCircuit instances
        for name, obj in user_ns.items():
            if hasattr(obj, '__class__') and 'Circuit' in str(obj.__class__):
                # Try to extract circuit info
                if hasattr(obj, 'n_qubits'):
                    circuit_data['n_qubits'] = obj.n_qubits
                break
        
        return circuit_data

# Register magic commands
get_ipython().register_magic_function(QuantRS2Magic(get_ipython()).quantrs2_debug, 'line', 'quantrs2_debug')
get_ipython().register_magic_function(QuantRS2Magic(get_ipython()).quantrs2_profile, 'line', 'quantrs2_profile')
get_ipython().register_magic_function(QuantRS2Magic(get_ipython()).quantrs2_visualize, 'cell', 'quantrs2_visualize')

print("QuantRS2 Jupyter magic commands loaded!")
print("Available commands:")
print("  %quantrs2_debug - Start debugging session")
print("  %quantrs2_profile - Profile circuit performance") 
print("  %%quantrs2_visualize - Visualize circuit in cell")
'''
            
            # Save to user's IPython startup directory
            try:
                from IPython.paths import get_ipython_dir
                startup_dir = Path(get_ipython_dir()) / "profile_default" / "startup"
                startup_dir.mkdir(parents=True, exist_ok=True)
                
                magic_file = startup_dir / "quantrs2_magic.py"
                with open(magic_file, 'w') as f:
                    f.write(jupyter_magic)
                
                print(f"Jupyter magic commands installed to: {magic_file}")
                print("Restart Jupyter to activate QuantRS2 integration")
                return True
                
            except ImportError:
                print("IPython not available - creating magic file in current directory")
                with open("quantrs2_magic.py", 'w') as f:
                    f.write(jupyter_magic)
                print("Load with: %run quantrs2_magic.py")
                return True
                
        except Exception as e:
            logging.error(f"Jupyter plugin installation failed: {e}")
            return False
    
    def _install_generic_plugin(self) -> bool:
        """Install generic plugin (command-line tools)."""
        try:
            # Create command-line tools
            cli_script = '''#!/usr/bin/env python3
"""
QuantRS2 CLI Tools for IDE Integration
"""

import requests
import json
import sys
import argparse

def analyze_file(filename):
    """Analyze quantum code file."""
    try:
        with open(filename, 'r') as f:
            code = f.read()
        
        response = requests.post('http://localhost:8765',
                               json={'method': 'analyze', 'params': {
                                   'code': code,
                                   'types': ['syntax', 'semantic', 'optimization']
                               }})
        
        if response.ok:
            result = response.json()
            diagnostics = result.get('diagnostics', [])
            
            for d in diagnostics:
                print(f"{filename}:{d['line']}:{d['column']}: {d['severity']}: {d['message']}")
        else:
            print(f"Analysis failed: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

def debug_file(filename):
    """Start debugging quantum code file."""
    try:
        with open(filename, 'r') as f:
            code = f.read()
        
        # Extract circuit data (simplified)
        circuit_data = {'n_qubits': 1, 'gates': []}
        
        response = requests.post('http://localhost:8765',
                               json={'method': 'debug', 'params': {
                                   'action': 'start',
                                   'circuit_data': circuit_data
                               }})
        
        if response.ok:
            result = response.json()
            session_id = result.get('session_id')
            print(f"Debug session started: {session_id}")
            
            # Interactive debugging
            while True:
                cmd = input("Debug> ").strip().lower()
                if cmd == 'q':
                    break
                elif cmd == 's':
                    step_response = requests.post('http://localhost:8765',
                                                json={'method': 'debug', 'params': {
                                                    'action': 'step',
                                                    'session_id': session_id
                                                }})
                    if step_response.ok:
                        step_result = step_response.json()
                        if step_result.get('success'):
                            state = step_result.get('state', {})
                            print(f"Step {state.get('step_number', 0)}: Entropy = {state.get('entanglement_entropy', 0):.4f}")
                        else:
                            print("Step failed")
                else:
                    print("Commands: s (step), q (quit)")
        else:
            print(f"Debug start failed: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="QuantRS2 CLI Tools")
    parser.add_argument("command", choices=["analyze", "debug"], help="Command to run")
    parser.add_argument("file", help="Python file to process")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        analyze_file(args.file)
    elif args.command == "debug":
        debug_file(args.file)

if __name__ == "__main__":
    main()
'''
            
            # Save CLI script
            cli_path = Path("quantrs2_cli.py")
            with open(cli_path, 'w') as f:
                f.write(cli_script)
            
            # Make executable
            cli_path.chmod(0o755)
            
            print(f"CLI tools installed: {cli_path.absolute()}")
            print("Usage:")
            print(f"  python {cli_path} analyze file.py")
            print(f"  python {cli_path} debug file.py")
            
            return True
            
        except Exception as e:
            logging.error(f"Generic plugin installation failed: {e}")
            return False
    
    def _create_jupyter_plugin(self):
        """Create Jupyter plugin instance."""
        return None  # Handled by _install_jupyter_plugin
    
    def _create_generic_plugin(self):
        """Create generic plugin instance.""" 
        return None  # Handled by _install_generic_plugin
    
    def get_status(self) -> Dict[str, Any]:
        """Get plugin status."""
        return {
            'state': self.state.value,
            'active_plugins': list(self.active_plugins.keys()),
            'server_running': self.server.server is not None,
            'server_port': self.server.port
        }
    
    def stop(self) -> None:
        """Stop all plugins."""
        self.server.stop_server()
        self.active_plugins.clear()
        self.state = PluginState.INACTIVE


# Global plugin instance
_ide_plugin: Optional[QuantumIDEPlugin] = None


def get_ide_plugin() -> QuantumIDEPlugin:
    """Get global IDE plugin instance."""
    global _ide_plugin
    if _ide_plugin is None:
        _ide_plugin = QuantumIDEPlugin()
    return _ide_plugin


def install_vscode_plugin() -> bool:
    """Convenience function to install VS Code plugin."""
    plugin = get_ide_plugin()
    return plugin.install_for_ide(IDEType.VSCODE)


def install_jupyter_plugin() -> bool:
    """Convenience function to install Jupyter plugin."""
    plugin = get_ide_plugin()
    return plugin.install_for_ide(IDEType.JUPYTER)


def install_generic_tools() -> bool:
    """Convenience function to install generic CLI tools."""
    plugin = get_ide_plugin()
    return plugin.install_for_ide(IDEType.GENERIC)


def analyze_quantum_code(code: str) -> List[DiagnosticMessage]:
    """Convenience function to analyze quantum code."""
    analyzer = QuantumCodeAnalyzer()
    return analyzer.analyze_code(code)


# CLI interface
def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantRS2 IDE Plugin")
    parser.add_argument("--install", choices=["vscode", "jupyter", "cli"], 
                       help="Install plugin for specific IDE")
    parser.add_argument("--analyze", help="Analyze quantum code file")
    parser.add_argument("--start-server", action="store_true", 
                       help="Start plugin server")
    parser.add_argument("--port", type=int, default=8765, 
                       help="Server port")
    
    args = parser.parse_args()
    
    if args.install:
        plugin = get_ide_plugin()
        
        if args.install == "vscode":
            success = plugin.install_for_ide(IDEType.VSCODE)
        elif args.install == "jupyter":
            success = plugin.install_for_ide(IDEType.JUPYTER)
        elif args.install == "cli":
            success = plugin.install_for_ide(IDEType.GENERIC)
        else:
            print(f"Unknown IDE: {args.install}")
            return 1
        
        if success:
            print(f"Successfully installed {args.install} plugin")
            return 0
        else:
            print(f"Failed to install {args.install} plugin")
            return 1
    
    elif args.analyze:
        try:
            with open(args.analyze, 'r') as f:
                code = f.read()
            
            diagnostics = analyze_quantum_code(code)
            
            for d in diagnostics:
                print(f"{args.analyze}:{d.line}:{d.column}: {d.severity}: {d.message}")
            
            return 0
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return 1
    
    elif args.start_server:
        server = IDEPluginServer(args.port)
        
        if server.start_server():
            print(f"QuantRS2 IDE plugin server started on port {args.port}")
            print("Press Ctrl+C to stop")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping server...")
                server.stop_server()
                return 0
        else:
            print("Failed to start server")
            return 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())