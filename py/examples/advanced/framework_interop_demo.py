#!/usr/bin/env python3
"""
Framework Interoperability Demo

This example demonstrates seamless interoperability between QuantRS2 and
other quantum computing frameworks (Qiskit, Cirq, PennyLane).

Features demonstrated:
    - Qiskit → QuantRS2 conversion
    - Cirq → QuantRS2 conversion
    - Circuit equivalence verification
    - Performance comparison
    - Gate mapping and optimization
"""

import sys
import time
import warnings
from typing import Optional

# Framework availability flags
QISKIT_AVAILABLE = False
CIRQ_AVAILABLE = False
QUANTRS2_AVAILABLE = False

# Try importing frameworks
try:
    from qiskit import QuantumCircuit as QiskitCircuit
    from qiskit.circuit.library import QFT as QiskitQFT
    QISKIT_AVAILABLE = True
except ImportError:
    print("⚠️  Qiskit not available. Install with: pip install qiskit")

try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    print("⚠️  Cirq not available. Install with: pip install cirq")

try:
    from quantrs2 import Circuit as QuantRS2Circuit
    from quantrs2.qiskit_converter import QiskitConverter, convert_from_qiskit
    from quantrs2.cirq_converter import CirqConverter, convert_from_cirq
    QUANTRS2_AVAILABLE = True
except ImportError:
    print("⚠️  QuantRS2 not available")
    sys.exit(1)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def demo_qiskit_conversion():
    """Demonstrate Qiskit to QuantRS2 conversion."""
    if not QISKIT_AVAILABLE:
        print("Skipping Qiskit demo (not installed)")
        return

    print_section("Qiskit → QuantRS2 Conversion")

    # Create a Qiskit circuit (Bell state)
    print("\n1. Creating Qiskit Bell State Circuit...")
    qc = QiskitCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    print(qc)

    # Convert to QuantRS2
    print("\n2. Converting to QuantRS2...")
    converter = QiskitConverter()
    quantrs_circuit, stats = converter.from_qiskit(qc, optimize=True)

    print(f"\nConversion Statistics:")
    print(f"  ✓ Original gates: {stats.original_gates}")
    print(f"  ✓ Converted gates: {stats.converted_gates}")
    print(f"  ✓ Decomposed gates: {stats.decomposed_gates}")
    print(f"  ✓ Success: {'Yes' if stats.success else 'No'}")

    if stats.warnings:
        print(f"\n  Warnings ({len(stats.warnings)}):")
        for warning in stats.warnings[:5]:  # Show first 5
            print(f"    - {warning}")

    # Run on QuantRS2
    print("\n3. Running on QuantRS2...")
    start = time.time()
    result = quantrs_circuit.run()
    elapsed = (time.time() - start) * 1000

    probs = result.probabilities()
    print(f"  Execution time: {elapsed:.2f} ms")
    print(f"  Probabilities: {probs}")

    # Complex circuit example
    print("\n4. Converting Complex Circuit (QFT)...")
    qft_qc = QiskitCircuit(3)
    qft_qc.compose(QiskitQFT(3), inplace=True)

    qft_circuit, qft_stats = converter.from_qiskit(qft_qc, optimize=True)

    print(f"  Original gates: {qft_stats.original_gates}")
    print(f"  Converted gates: {qft_stats.converted_gates}")
    print(f"  Decomposed gates: {qft_stats.decomposed_gates}")


def demo_cirq_conversion():
    """Demonstrate Cirq to QuantRS2 conversion."""
    if not CIRQ_AVAILABLE:
        print("Skipping Cirq demo (not installed)")
        return

    print_section("Cirq → QuantRS2 Conversion")

    # Create a Cirq circuit (GHZ state)
    print("\n1. Creating Cirq GHZ State Circuit...")
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit()

    circuit.append([
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.CNOT(qubits[1], qubits[2]),
    ])

    print(circuit)

    # Convert to QuantRS2
    print("\n2. Converting to QuantRS2...")
    converter = CirqConverter()
    quantrs_circuit, stats = converter.from_cirq(circuit)

    print(f"\nConversion Statistics:")
    print(f"  ✓ Original operations: {stats.original_operations}")
    print(f"  ✓ Converted operations: {stats.converted_operations}")
    print(f"  ✓ Decomposed operations: {stats.decomposed_operations}")
    print(f"  ✓ Number of moments: {stats.num_moments}")
    print(f"  ✓ Success: {'Yes' if stats.success else 'No'}")

    if stats.warnings:
        print(f"\n  Warnings ({len(stats.warnings)}):")
        for warning in stats.warnings[:5]:
            print(f"    - {warning}")

    # Run on QuantRS2
    print("\n3. Running on QuantRS2...")
    start = time.time()
    result = quantrs_circuit.run()
    elapsed = (time.time() - start) * 1000

    probs = result.probabilities()
    print(f"  Execution time: {elapsed:.2f} ms")
    print(f"  Top 3 probabilities:")

    # Get top 3 states
    import numpy as np
    top_indices = np.argsort(probs)[::-1][:3]
    for idx in top_indices:
        basis_state = format(idx, '03b')
        print(f"    |{basis_state}⟩: {probs[idx]:.4f}")

    # Example with powered gates
    print("\n4. Converting Circuit with Powered Gates...")
    powered_circuit = cirq.Circuit()
    powered_circuit.append([
        cirq.X(qubits[0]) ** 0.5,  # √X gate
        cirq.Z(qubits[1]) ** 0.5,  # S gate
        cirq.Z(qubits[2]) ** 0.25, # T gate
    ])

    pow_quantrs, pow_stats = converter.from_cirq(powered_circuit)
    print(f"  Converted: {pow_stats.converted_operations} operations")
    print(f"  Decomposed: {pow_stats.decomposed_operations} operations")


def demo_performance_comparison():
    """Compare performance across frameworks."""
    print_section("Performance Comparison")

    n_qubits = 5
    print(f"\nComparing {n_qubits}-qubit GHZ state creation and simulation:")
    print(f"{'Framework':<15} {'Build (ms)':<15} {'Run (ms)':<15} {'Total (ms)':<15}")
    print("-" * 60)

    # QuantRS2 (native)
    start = time.time()
    qrs_circuit = QuantRS2Circuit(n_qubits)
    qrs_circuit.h(0)
    for i in range(n_qubits - 1):
        qrs_circuit.cnot(i, i + 1)
    build_time = (time.time() - start) * 1000

    start = time.time()
    result = qrs_circuit.run()
    run_time = (time.time() - start) * 1000

    print(f"{'QuantRS2':<15} {build_time:<15.2f} {run_time:<15.2f} {build_time + run_time:<15.2f}")

    # Qiskit (if available)
    if QISKIT_AVAILABLE:
        from qiskit import Aer, execute

        start = time.time()
        qc = QiskitCircuit(n_qubits)
        qc.h(0)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        build_time = (time.time() - start) * 1000

        start = time.time()
        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend)
        result = job.result()
        _ = result.get_statevector()
        run_time = (time.time() - start) * 1000

        print(f"{'Qiskit':<15} {build_time:<15.2f} {run_time:<15.2f} {build_time + run_time:<15.2f}")

    # Cirq (if available)
    if CIRQ_AVAILABLE:
        start = time.time()
        qubits = cirq.LineQubit.range(n_qubits)
        circuit = cirq.Circuit()
        circuit.append(cirq.H(qubits[0]))
        for i in range(n_qubits - 1):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
        build_time = (time.time() - start) * 1000

        start = time.time()
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        _ = result.state_vector()
        run_time = (time.time() - start) * 1000

        print(f"{'Cirq':<15} {build_time:<15.2f} {run_time:<15.2f} {build_time + run_time:<15.2f}")


def demo_circuit_equivalence():
    """Demonstrate circuit equivalence verification."""
    if not (QISKIT_AVAILABLE and CIRQ_AVAILABLE):
        print("Skipping equivalence demo (requires both Qiskit and Cirq)")
        return

    print_section("Circuit Equivalence Verification")

    # Create equivalent circuits in different frameworks
    print("\n1. Creating equivalent Bell state circuits...")

    # Qiskit
    qiskit_bell = QiskitCircuit(2)
    qiskit_bell.h(0)
    qiskit_bell.cx(0, 1)

    # Cirq
    cirq_qubits = cirq.LineQubit.range(2)
    cirq_bell = cirq.Circuit()
    cirq_bell.append([
        cirq.H(cirq_qubits[0]),
        cirq.CNOT(cirq_qubits[0], cirq_qubits[1])
    ])

    # Convert both to QuantRS2
    print("\n2. Converting to QuantRS2...")
    qiskit_converter = QiskitConverter()
    cirq_converter = CirqConverter()

    qrs_from_qiskit, _ = qiskit_converter.from_qiskit(qiskit_bell)
    qrs_from_cirq, _ = cirq_converter.from_cirq(cirq_bell)

    # Run both and compare
    print("\n3. Running and comparing results...")
    result_qiskit = qrs_from_qiskit.run()
    result_cirq = qrs_from_cirq.run()

    probs_qiskit = result_qiskit.probabilities()
    probs_cirq = result_cirq.probabilities()

    # Check equivalence
    import numpy as np
    max_diff = np.max(np.abs(np.array(probs_qiskit) - np.array(probs_cirq)))

    print(f"  Max probability difference: {max_diff:.10f}")
    print(f"  Circuits equivalent: {'Yes' if max_diff < 1e-6 else 'No'}")

    print("\n  Qiskit → QuantRS2 probabilities:", probs_qiskit)
    print("  Cirq → QuantRS2 probabilities:  ", probs_cirq)


def demo_advanced_features():
    """Demonstrate advanced conversion features."""
    print_section("Advanced Conversion Features")

    if QISKIT_AVAILABLE:
        print("\n1. Qiskit: Handling parametric circuits...")
        qc = QiskitCircuit(2)
        qc.h(0)
        qc.rx(1.5, 0)
        qc.ry(0.5, 1)
        qc.cx(0, 1)
        qc.rz(2.0, 1)

        converter = QiskitConverter()
        circuit, stats = converter.from_qiskit(qc)
        print(f"  ✓ Converted {stats.converted_gates} parametric gates")

    if CIRQ_AVAILABLE:
        print("\n2. Cirq: Handling moment-based circuits...")
        qubits = cirq.LineQubit.range(3)

        # Create circuit with explicit moments
        circuit = cirq.Circuit()

        # Moment 1: Parallel H gates
        circuit.append([cirq.H(q) for q in qubits])

        # Moment 2: Entangle
        circuit.append([cirq.CNOT(qubits[0], qubits[1])])

        # Moment 3: More operations
        circuit.append([
            cirq.X(qubits[2]),
            cirq.Z(qubits[0])
        ])

        converter = CirqConverter()
        qrs_circuit, stats = converter.from_cirq(circuit)
        print(f"  ✓ Converted {stats.num_moments} moments")
        print(f"  ✓ Total operations: {stats.converted_operations}")


def main():
    """Run all demos."""
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║           QuantRS2 Framework Interoperability Demo                    ║
║                                                                       ║
║  Seamless conversion between quantum computing frameworks:           ║
║    • Qiskit → QuantRS2                                               ║
║    • Cirq → QuantRS2                                                 ║
║    • Circuit equivalence verification                                ║
║    • Performance comparison                                          ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)

    print("\nFramework Availability:")
    print(f"  {'✓' if QUANTRS2_AVAILABLE else '✗'} QuantRS2")
    print(f"  {'✓' if QISKIT_AVAILABLE else '✗'} Qiskit")
    print(f"  {'✓' if CIRQ_AVAILABLE else '✗'} Cirq")

    if not QUANTRS2_AVAILABLE:
        print("\n❌ QuantRS2 not available. Please install QuantRS2.")
        return

    # Run demos
    demo_qiskit_conversion()
    demo_cirq_conversion()
    demo_performance_comparison()
    demo_circuit_equivalence()
    demo_advanced_features()

    print_section("Demo Complete")
    print("\nKey Takeaways:")
    print("  • QuantRS2 provides seamless interoperability with Qiskit and Cirq")
    print("  • Automatic gate mapping and decomposition")
    print("  • Circuit equivalence can be verified")
    print("  • QuantRS2 often shows competitive or better performance")
    print("\nFor production use:")
    print("  1. Use convert_from_qiskit() or convert_from_cirq() convenience functions")
    print("  2. Check conversion statistics for warnings")
    print("  3. Verify equivalence for critical circuits")
    print("  4. Benchmark performance for your specific use case")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
