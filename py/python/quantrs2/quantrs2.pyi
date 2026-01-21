"""
Type stubs for the QuantRS2 native module.

This file provides type hints for the Rust extension module built with PyO3.
"""

from typing import Any, Dict, List, Optional, Tuple, Union


class PyCircuit:
    """
    Quantum circuit representation.

    Args:
        n_qubits: Number of qubits in the circuit (must be at least 2)
    """

    def __init__(self, n_qubits: int) -> None: ...

    @property
    def n_qubits(self) -> int:
        """Get the number of qubits in the circuit."""
        ...

    @property
    def num_gates(self) -> int:
        """Get the number of gates in the circuit."""
        ...

    def depth(self) -> int:
        """Get the depth of the circuit."""
        ...

    # Single-qubit gates
    def h(self, qubit: int) -> None:
        """Apply a Hadamard gate to the specified qubit."""
        ...

    def x(self, qubit: int) -> None:
        """Apply a Pauli-X (NOT) gate to the specified qubit."""
        ...

    def y(self, qubit: int) -> None:
        """Apply a Pauli-Y gate to the specified qubit."""
        ...

    def z(self, qubit: int) -> None:
        """Apply a Pauli-Z gate to the specified qubit."""
        ...

    def s(self, qubit: int) -> None:
        """Apply an S gate (phase gate) to the specified qubit."""
        ...

    def sdg(self, qubit: int) -> None:
        """Apply an S-dagger gate to the specified qubit."""
        ...

    def t(self, qubit: int) -> None:
        """Apply a T gate (π/8 gate) to the specified qubit."""
        ...

    def tdg(self, qubit: int) -> None:
        """Apply a T-dagger gate to the specified qubit."""
        ...

    def sx(self, qubit: int) -> None:
        """Apply a SX gate (square root of X) to the specified qubit."""
        ...

    def sxdg(self, qubit: int) -> None:
        """Apply a SX-dagger gate to the specified qubit."""
        ...

    def rx(self, qubit: int, theta: float) -> None:
        """Apply an Rx gate (rotation around X-axis) to the specified qubit."""
        ...

    def ry(self, qubit: int, theta: float) -> None:
        """Apply an Ry gate (rotation around Y-axis) to the specified qubit."""
        ...

    def rz(self, qubit: int, theta: float) -> None:
        """Apply an Rz gate (rotation around Z-axis) to the specified qubit."""
        ...

    def p(self, qubit: int, lambda_: float) -> None:
        """Apply a phase gate (P gate) with an arbitrary angle to the specified qubit."""
        ...

    def id(self, qubit: int) -> None:
        """Apply an identity gate to the specified qubit."""
        ...

    def u(self, qubit: int, theta: float, phi: float, lambda_: float) -> None:
        """Apply a U gate (general single-qubit rotation) to the specified qubit."""
        ...

    # Two-qubit gates
    def cnot(self, control: int, target: int) -> None:
        """Apply a CNOT gate with the specified control and target qubits."""
        ...

    def swap(self, qubit1: int, qubit2: int) -> None:
        """Apply a SWAP gate between the specified qubits."""
        ...

    def cy(self, control: int, target: int) -> None:
        """Apply a CY gate (controlled-Y) to the specified qubits."""
        ...

    def cz(self, control: int, target: int) -> None:
        """Apply a CZ gate (controlled-Z) to the specified qubits."""
        ...

    def ch(self, control: int, target: int) -> None:
        """Apply a CH gate (controlled-H) to the specified qubits."""
        ...

    def cs(self, control: int, target: int) -> None:
        """Apply a CS gate (controlled-S) to the specified qubits."""
        ...

    def crx(self, control: int, target: int, theta: float) -> None:
        """Apply a CRX gate (controlled-RX) to the specified qubits."""
        ...

    def cry(self, control: int, target: int, theta: float) -> None:
        """Apply a CRY gate (controlled-RY) to the specified qubits."""
        ...

    def crz(self, control: int, target: int, theta: float) -> None:
        """Apply a CRZ gate (controlled-RZ) to the specified qubits."""
        ...

    def iswap(self, qubit1: int, qubit2: int) -> None:
        """Apply an iSWAP gate to the specified qubits."""
        ...

    def ecr(self, control: int, target: int) -> None:
        """Apply an ECR gate (IBM native echoed cross-resonance) to the specified qubits."""
        ...

    def rxx(self, qubit1: int, qubit2: int, theta: float) -> None:
        """Apply an RXX gate (two-qubit XX rotation) to the specified qubits."""
        ...

    def ryy(self, qubit1: int, qubit2: int, theta: float) -> None:
        """Apply an RYY gate (two-qubit YY rotation) to the specified qubits."""
        ...

    def rzz(self, qubit1: int, qubit2: int, theta: float) -> None:
        """Apply an RZZ gate (two-qubit ZZ rotation) to the specified qubits."""
        ...

    def rzx(self, control: int, target: int, theta: float) -> None:
        """Apply an RZX gate (two-qubit ZX rotation / cross-resonance) to the specified qubits."""
        ...

    def dcx(self, qubit1: int, qubit2: int) -> None:
        """Apply a DCX gate (double CNOT) to the specified qubits."""
        ...

    # Three-qubit gates
    def toffoli(self, control1: int, control2: int, target: int) -> None:
        """Apply a Toffoli gate (CCNOT) to the specified qubits."""
        ...

    def cswap(self, control: int, target1: int, target2: int) -> None:
        """Apply a Fredkin gate (CSWAP) to the specified qubits."""
        ...

    # Simulation methods
    def run(self, use_gpu: bool = False) -> "PySimulationResult":
        """
        Run the circuit on a state vector simulator.

        Args:
            use_gpu: Whether to use the GPU for simulation if available.

        Returns:
            PySimulationResult: The result of the simulation.
        """
        ...

    def simulate_with_noise(
        self,
        noise_model: "PyRealisticNoiseModel",
        use_gpu: bool = False
    ) -> "PySimulationResult":
        """
        Run the circuit with a noise model.

        Args:
            noise_model: The noise model to use for simulation.
            use_gpu: Whether to use the GPU for simulation if available.

        Returns:
            PySimulationResult: The result of the simulation with noise applied.
        """
        ...

    def run_auto(self) -> "PySimulationResult":
        """
        Run the circuit on the best available simulator.

        Returns:
            PySimulationResult: The result of the simulation.
        """
        ...

    @staticmethod
    def is_gpu_available() -> bool:
        """Check if GPU acceleration is available."""
        ...

    # Visualization methods
    def draw(self) -> str:
        """Get a text-based visualization of the circuit."""
        ...

    def draw_html(self) -> str:
        """Get an HTML representation of the circuit for Jupyter notebooks."""
        ...

    def visualize(self) -> "PyCircuitVisualizer":
        """Get a visualization object for the circuit."""
        ...

    def _repr_html_(self) -> str:
        """Jupyter notebook HTML representation."""
        ...

    # Circuit manipulation
    def decompose(self) -> "PyCircuit":
        """Decompose complex gates into simpler gates."""
        ...

    def copy(self) -> "PyCircuit":
        """Copy the circuit (returns an identical circuit)."""
        ...

    def compose(self, other: "PyCircuit") -> None:
        """Compose this circuit with another circuit."""
        ...


class PySimulationResult:
    """Result from a quantum simulation."""

    @property
    def n_qubits(self) -> int:
        """Get the number of qubits."""
        ...

    def amplitudes(self) -> List[complex]:
        """Get the state vector amplitudes."""
        ...

    def probabilities(self) -> List[float]:
        """Get the probabilities for each basis state."""
        ...

    def state_probabilities(self) -> Dict[str, float]:
        """Get a dictionary mapping basis states to probabilities."""
        ...

    def expectation_value(self, operator: str) -> float:
        """
        Get the expectation value of a Pauli operator.

        Args:
            operator: A string of Pauli operators (I, X, Y, Z), one per qubit.

        Returns:
            The expectation value ⟨ψ|P|ψ⟩.
        """
        ...


class PyRealisticNoiseModel:
    """Realistic noise model for quantum simulations."""

    @property
    def num_channels(self) -> int:
        """Get the number of noise channels in this model."""
        ...

    @staticmethod
    def ibm_device(device_name: str) -> "PyRealisticNoiseModel":
        """
        Create a new realistic noise model for IBM quantum devices.

        Args:
            device_name: The name of the IBM quantum device (e.g., "ibmq_lima", "ibm_cairo")

        Returns:
            A noise model configured with the specified device parameters.
        """
        ...

    @staticmethod
    def rigetti_device(device_name: str) -> "PyRealisticNoiseModel":
        """
        Create a new realistic noise model for Rigetti quantum devices.

        Args:
            device_name: The name of the Rigetti quantum device (e.g., "Aspen-M-2")

        Returns:
            A noise model configured with the specified device parameters.
        """
        ...

    @staticmethod
    def custom(
        t1_us: float = 100.0,
        t2_us: float = 50.0,
        gate_time_ns: float = 40.0,
        gate_error_1q: float = 0.001,
        gate_error_2q: float = 0.01,
        readout_error: float = 0.02
    ) -> "PyRealisticNoiseModel":
        """
        Create a new realistic noise model with custom parameters.

        Args:
            t1_us: T1 relaxation time in microseconds.
            t2_us: T2 dephasing time in microseconds.
            gate_time_ns: Gate time in nanoseconds.
            gate_error_1q: Single-qubit gate error rate (0.0 to 1.0).
            gate_error_2q: Two-qubit gate error rate (0.0 to 1.0).
            readout_error: Readout error rate (0.0 to 1.0).

        Returns:
            A custom noise model with the specified parameters.
        """
        ...


class PyCircuitVisualizer:
    """Circuit visualization helper."""

    def __init__(self, n_qubits: int) -> None: ...

    @property
    def n_qubits(self) -> int:
        """Get the number of qubits."""
        ...

    def add_gate(
        self,
        name: str,
        qubits: List[int],
        param: Optional[str] = None
    ) -> None:
        """Add a gate to the visualization."""
        ...

    def to_ascii(self) -> str:
        """Generate ASCII art representation of the circuit."""
        ...

    def to_html(self) -> str:
        """Generate HTML representation of the circuit."""
        ...

    def to_svg(self) -> str:
        """Generate SVG representation of the circuit."""
        ...

    def _repr_html_(self) -> str:
        """Jupyter notebook HTML representation."""
        ...


class PyDynamicCircuit:
    """Dynamic qubit count circuit (alias to PyCircuit for backward compatibility)."""

    def __init__(self, n_qubits: int) -> None: ...

    @property
    def n_qubits(self) -> int:
        """Get the number of qubits in the circuit."""
        ...

    # Same gate methods as PyCircuit
    def h(self, qubit: int) -> None: ...
    def x(self, qubit: int) -> None: ...
    def y(self, qubit: int) -> None: ...
    def z(self, qubit: int) -> None: ...
    def s(self, qubit: int) -> None: ...
    def sdg(self, qubit: int) -> None: ...
    def t(self, qubit: int) -> None: ...
    def tdg(self, qubit: int) -> None: ...
    def rx(self, qubit: int, theta: float) -> None: ...
    def ry(self, qubit: int, theta: float) -> None: ...
    def rz(self, qubit: int, theta: float) -> None: ...
    def cnot(self, control: int, target: int) -> None: ...
    def swap(self, qubit1: int, qubit2: int) -> None: ...
    def cz(self, control: int, target: int) -> None: ...

    def run(self, use_gpu: bool = False) -> PySimulationResult: ...
    def simulate_with_noise(
        self,
        noise_model: PyRealisticNoiseModel,
        use_gpu: bool = False
    ) -> PySimulationResult: ...
    def run_auto(self) -> PySimulationResult: ...


# Module-level constants
__version__: str
MAX_QUBITS: int
SUPPORTED_QUBITS: List[int]
