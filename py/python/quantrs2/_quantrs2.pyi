"""
Type stubs for _quantrs2 native module.

This file provides type hints for the native PyO3 bindings to improve
IDE support and static type checking.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray

# Type aliases
Complex = Union[complex, float, int]
Matrix = NDArray[np.complex128]
StateVector = NDArray[np.complex128]
Probabilities = Dict[str, float]

@runtime_checkable
class PyCircuit(Protocol):
    """Python circuit interface."""
    
    def __init__(self, n_qubits: int) -> None: ...
    
    @property
    def n_qubits(self) -> int: ...
    
    def copy(self) -> 'PyCircuit': ...
    def clear(self) -> None: ...
    def depth(self) -> int: ...
    def gate_count(self) -> int: ...
    
    # Single-qubit gates
    def h(self, qubit: int) -> None: ...
    def x(self, qubit: int) -> None: ...
    def y(self, qubit: int) -> None: ...
    def z(self, qubit: int) -> None: ...
    def s(self, qubit: int) -> None: ...
    def t(self, qubit: int) -> None: ...
    def sx(self, qubit: int) -> None: ...
    def sdg(self, qubit: int) -> None: ...
    def tdg(self, qubit: int) -> None: ...
    def sxdg(self, qubit: int) -> None: ...
    
    # Rotation gates
    def rx(self, qubit: int, theta: float) -> None: ...
    def ry(self, qubit: int, theta: float) -> None: ...
    def rz(self, qubit: int, theta: float) -> None: ...
    def p(self, qubit: int, lambda_: float) -> None: ...
    def u(self, qubit: int, theta: float, phi: float, lambda_: float) -> None: ...
    
    # Two-qubit gates
    def cnot(self, control: int, target: int) -> None: ...
    def cx(self, control: int, target: int) -> None: ...
    def cy(self, control: int, target: int) -> None: ...
    def cz(self, control: int, target: int) -> None: ...
    def ch(self, control: int, target: int) -> None: ...
    def swap(self, qubit1: int, qubit2: int) -> None: ...
    def iswap(self, qubit1: int, qubit2: int) -> None: ...
    
    # Controlled rotation gates
    def crx(self, control: int, target: int, theta: float) -> None: ...
    def cry(self, control: int, target: int, theta: float) -> None: ...
    def crz(self, control: int, target: int, theta: float) -> None: ...
    def cp(self, control: int, target: int, lambda_: float) -> None: ...
    def cu(self, control: int, target: int, theta: float, phi: float, lambda_: float, gamma: float) -> None: ...
    
    # Three-qubit gates
    def ccx(self, control1: int, control2: int, target: int) -> None: ...
    def toffoli(self, control1: int, control2: int, target: int) -> None: ...
    def cswap(self, control: int, target1: int, target2: int) -> None: ...
    def fredkin(self, control: int, target1: int, target2: int) -> None: ...
    
    # Measurements
    def measure(self, qubit: int, clbit: Optional[int] = None) -> None: ...
    def measure_all(self, clbits: Optional[List[int]] = None) -> None: ...
    
    # Custom gates
    def custom_gate(self, name: str, qubits: List[int], matrix: Matrix) -> None: ...
    def add_gate(self, gate: 'PyGate') -> None: ...
    
    # Parametric operations
    def add_parameter(self, name: str, value: float) -> None: ...
    def set_parameter(self, name: str, value: float) -> None: ...
    def get_parameter(self, name: str) -> float: ...
    def get_parameters(self) -> Dict[str, float]: ...
    def bind_parameters(self, parameters: Dict[str, float]) -> 'PyCircuit': ...
    
    # Circuit composition
    def append(self, other: 'PyCircuit', qubits: Optional[List[int]] = None) -> None: ...
    def compose(self, other: 'PyCircuit', qubits: Optional[List[int]] = None, clbits: Optional[List[int]] = None) -> 'PyCircuit': ...
    def inverse(self) -> 'PyCircuit': ...
    def power(self, power: int) -> 'PyCircuit': ...
    
    # Execution
    def run(self, shots: int = 1024, backend: Optional[str] = None) -> 'PySimulationResult': ...
    def simulate(self, shots: int = 1024) -> 'PySimulationResult': ...
    
    # Visualization
    def draw(self, output: str = 'text') -> str: ...
    def visualize(self) -> 'PyCircuitVisualizer': ...
    
    # Circuit analysis
    def count_ops(self) -> Dict[str, int]: ...
    def size(self) -> int: ...
    def width(self) -> int: ...
    def num_qubits(self) -> int: ...
    def num_clbits(self) -> int: ...
    
    # Optimization
    def optimize(self, level: int = 1) -> 'PyCircuit': ...
    def decompose(self, gates_to_decompose: Optional[List[str]] = None) -> 'PyCircuit': ...


@runtime_checkable  
class PySimulationResult(Protocol):
    """Simulation result interface."""
    
    def __init__(self) -> None: ...
    
    @property
    def n_qubits(self) -> int: ...
    @property
    def shots(self) -> int: ...
    
    def get_counts(self) -> Dict[str, int]: ...
    def get_probabilities(self) -> Probabilities: ...
    def state_probabilities(self) -> Probabilities: ...
    def get_statevector(self) -> StateVector: ...
    def get_memory(self) -> List[str]: ...
    
    # Expectation values
    def expectation_value(self, observable: Matrix) -> complex: ...
    def expectation_values(self, observables: List[Matrix]) -> List[complex]: ...
    
    # State analysis
    def purity(self) -> float: ...
    def entropy(self) -> float: ...
    def fidelity(self, other: Union['PySimulationResult', StateVector]) -> float: ...
    
    # Measurement statistics
    def marginal_counts(self, indices: List[int]) -> Dict[str, int]: ...
    def marginal_probabilities(self, indices: List[int]) -> Probabilities: ...
    
    # Visualization
    def plot_histogram(self, title: Optional[str] = None) -> Any: ...
    def plot_state(self, output: str = 'text') -> Any: ...


@runtime_checkable
class PyGate(Protocol):
    """Gate interface."""
    
    def __init__(self, name: str, qubits: List[int], params: Optional[List[float]] = None) -> None: ...
    
    @property 
    def name(self) -> str: ...
    @property
    def qubits(self) -> List[int]: ...
    @property
    def params(self) -> List[float]: ...
    @property
    def num_qubits(self) -> int: ...
    
    def matrix(self) -> Matrix: ...
    def inverse(self) -> 'PyGate': ...
    def power(self, power: float) -> 'PyGate': ...
    def controlled(self, num_ctrl_qubits: int = 1) -> 'PyGate': ...
    
    def to_instruction(self) -> 'PyInstruction': ...
    def copy(self) -> 'PyGate': ...


@runtime_checkable
class PyInstruction(Protocol):
    """Instruction interface."""
    
    def __init__(self, operation: PyGate, qubits: List[int], clbits: Optional[List[int]] = None) -> None: ...
    
    @property
    def operation(self) -> PyGate: ...
    @property
    def qubits(self) -> List[int]: ...
    @property
    def clbits(self) -> List[int]: ...
    
    def copy(self) -> 'PyInstruction': ...


@runtime_checkable
class PyQuantumRegister(Protocol):
    """Quantum register interface."""
    
    def __init__(self, size: int, name: Optional[str] = None) -> None: ...
    
    @property
    def size(self) -> int: ...
    @property
    def name(self) -> str: ...
    
    def __getitem__(self, index: int) -> 'PyQubit': ...
    def __len__(self) -> int: ...


@runtime_checkable
class PyClassicalRegister(Protocol):
    """Classical register interface."""
    
    def __init__(self, size: int, name: Optional[str] = None) -> None: ...
    
    @property
    def size(self) -> int: ...
    @property
    def name(self) -> str: ...
    
    def __getitem__(self, index: int) -> 'PyClbit': ...
    def __len__(self) -> int: ...


@runtime_checkable
class PyQubit(Protocol):
    """Qubit interface."""
    
    def __init__(self, register: Optional[PyQuantumRegister] = None, index: int = 0) -> None: ...
    
    @property
    def register(self) -> Optional[PyQuantumRegister]: ...
    @property
    def index(self) -> int: ...


@runtime_checkable
class PyClbit(Protocol):
    """Classical bit interface."""
    
    def __init__(self, register: Optional[PyClassicalRegister] = None, index: int = 0) -> None: ...
    
    @property
    def register(self) -> Optional[PyClassicalRegister]: ...
    @property
    def index(self) -> int: ...


@runtime_checkable
class PyCircuitVisualizer(Protocol):
    """Circuit visualizer interface."""
    
    def __init__(self, circuit: PyCircuit) -> None: ...
    
    def to_text(self) -> str: ...
    def to_html(self) -> str: ...
    def to_svg(self) -> str: ...
    def to_image(self, filename: str, format: str = 'png') -> None: ...


# Noise modeling
@runtime_checkable
class PyNoiseModel(Protocol):
    """Noise model interface."""
    
    def __init__(self) -> None: ...
    
    def add_quantum_error(self, error: 'PyQuantumError', instructions: List[str], qubits: Optional[List[List[int]]] = None) -> None: ...
    def add_readout_error(self, error: 'PyReadoutError', qubits: List[int]) -> None: ...
    def add_thermal_relaxation(self, t1: float, t2: float, gate_time: float, qubits: List[int]) -> None: ...


@runtime_checkable
class PyQuantumError(Protocol):
    """Quantum error interface."""
    
    def __init__(self, noise_ops: List[Tuple[Matrix, float]]) -> None: ...
    
    @property
    def size(self) -> int: ...
    @property
    def num_qubits(self) -> int: ...
    
    def compose(self, other: 'PyQuantumError') -> 'PyQuantumError': ...
    def tensor(self, other: 'PyQuantumError') -> 'PyQuantumError': ...


@runtime_checkable
class PyReadoutError(Protocol):
    """Readout error interface."""
    
    def __init__(self, probabilities: Matrix) -> None: ...
    
    @property
    def probabilities(self) -> Matrix: ...
    @property
    def num_qubits(self) -> int: ...


# Pulse-level control
@runtime_checkable
class PyPulse(Protocol):
    """Pulse interface."""
    
    def __init__(self, envelope: NDArray[np.complex128], duration: float, name: Optional[str] = None) -> None: ...
    
    @property
    def envelope(self) -> NDArray[np.complex128]: ...
    @property
    def duration(self) -> float: ...
    @property
    def name(self) -> str: ...


@runtime_checkable
class PyChannel(Protocol):
    """Channel interface."""
    
    def __init__(self, index: int, name: Optional[str] = None) -> None: ...
    
    @property
    def index(self) -> int: ...
    @property
    def name(self) -> str: ...


@runtime_checkable
class PySchedule(Protocol):
    """Schedule interface."""
    
    def __init__(self, name: Optional[str] = None) -> None: ...
    
    @property
    def name(self) -> str: ...
    @property
    def duration(self) -> float: ...
    
    def play(self, pulse: PyPulse, channel: PyChannel, time: float = 0) -> None: ...
    def delay(self, duration: float, channel: PyChannel, time: float = 0) -> None: ...
    def barrier(self, channels: List[PyChannel], time: float = 0) -> None: ...


# Machine learning interfaces
@runtime_checkable
class PyVQE(Protocol):
    """Variational Quantum Eigensolver interface."""
    
    def __init__(self, n_qubits: int, hamiltonian: Optional[Matrix] = None, ansatz: str = "hardware_efficient") -> None: ...
    
    @property
    def n_qubits(self) -> int: ...
    @property
    def hamiltonian(self) -> Matrix: ...
    @property
    def parameters(self) -> NDArray[np.float64]: ...
    
    def expectation(self, parameters: NDArray[np.float64]) -> float: ...
    def optimize(self, max_iterations: int = 100) -> Tuple[float, NDArray[np.float64]]: ...


@runtime_checkable
class PyQAOA(Protocol):
    """Quantum Approximate Optimization Algorithm interface."""
    
    def __init__(self, n_qubits: int, cost_hamiltonian: Matrix, mixer_hamiltonian: Optional[Matrix] = None, p: int = 1) -> None: ...
    
    @property
    def n_qubits(self) -> int: ...
    @property
    def p(self) -> int: ...
    @property
    def parameters(self) -> NDArray[np.float64]: ...
    
    def expectation(self, parameters: NDArray[np.float64]) -> float: ...
    def optimize(self, max_iterations: int = 100) -> Tuple[float, NDArray[np.float64]]: ...


# Backend interfaces
@runtime_checkable
class PyBackend(Protocol):
    """Backend interface."""
    
    def __init__(self, name: str) -> None: ...
    
    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...
    @property
    def max_shots(self) -> int: ...
    @property
    def max_experiments(self) -> int: ...
    
    def run(self, circuits: Union[PyCircuit, List[PyCircuit]], shots: int = 1024) -> Union[PySimulationResult, List[PySimulationResult]]: ...
    def configuration(self) -> Dict[str, Any]: ...
    def properties(self) -> Dict[str, Any]: ...
    def status(self) -> Dict[str, Any]: ...


# Utility functions  
def create_circuit(n_qubits: int) -> PyCircuit: ...
def load_circuit(filename: str) -> PyCircuit: ...
def save_circuit(circuit: PyCircuit, filename: str) -> None: ...

def random_circuit(n_qubits: int, depth: int, gate_set: Optional[List[str]] = None, seed: Optional[int] = None) -> PyCircuit: ...
def bell_circuit(n_qubits: int = 2) -> PyCircuit: ...
def ghz_circuit(n_qubits: int) -> PyCircuit: ...

def pauli_x(qubit: int = 0) -> PyGate: ...
def pauli_y(qubit: int = 0) -> PyGate: ...
def pauli_z(qubit: int = 0) -> PyGate: ...
def hadamard(qubit: int = 0) -> PyGate: ...
def cnot_gate(control: int, target: int) -> PyGate: ...

def state_fidelity(state1: StateVector, state2: StateVector) -> float: ...
def process_fidelity(channel1: Matrix, channel2: Matrix) -> float: ...
def purity(state: StateVector) -> float: ...
def entropy(state: StateVector) -> float: ...

# Constants
PI: float
E: float
SQRT_2: float

# Gate sets
CLIFFORD_GATES: List[str]
UNIVERSAL_GATES: List[str]
PAULI_GATES: List[str]