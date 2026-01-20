#!/usr/bin/env python3
"""
MyQLM/QLM Converter Demonstration

This script demonstrates the MyQLM <-> QuantRS2 conversion capabilities,
showcasing integration with Atos Quantum Learning Machine framework.

Features demonstrated:
    - Basic circuit conversion from MyQLM to QuantRS2
    - Abstract gate handling
    - QRoutine support
    - Job creation and submission
    - Parametric circuits
    - Practical quantum algorithm examples
"""

import sys
import numpy as np

try:
    from qat.lang.AQASM import Program, H, X, Y, Z, S, T, CNOT, SWAP, RX, RY, RZ, CCNOT
    from qat.lang.AQASM import AbstractGate, QRoutine
    from qat.core import Circuit as QLMCircuit
    MYQLM_AVAILABLE = True
except ImportError:
    print("❌ MyQLM/QLM not installed. Install with: pip install myqlm")
    print("   Skipping MyQLM-specific examples...")
    MYQLM_AVAILABLE = False

try:
    from quantrs2 import Circuit as QuantRS2Circuit
    from quantrs2.myqlm_converter import (
        MyQLMConverter,
        convert_from_myqlm,
        convert_to_myqlm,
    )
    QUANTRS2_AVAILABLE = True
except ImportError:
    print("❌ QuantRS2 not available")
    QUANTRS2_AVAILABLE = False
    sys.exit(1)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_basic_conversion():
    """Demonstrate basic MyQLM to QuantRS2 conversion."""
    print_section("Basic Circuit Conversion")

    if not MYQLM_AVAILABLE:
        print("⚠️  MyQLM not available, skipping...")
        return

    # Create a simple Bell state circuit in MyQLM
    prog = Program()
    qbits = prog.qalloc(2)

    prog.apply(H, qbits[0])
    prog.apply(CNOT, qbits[0], qbits[1])

    circuit = prog.to_circ()

    print("\n📋 Original MyQLM Circuit:")
    print(f"   Qubits: {circuit.nbqbits}")
    print(f"   Gates: {len(list(circuit.iterate_simple()))}")

    # Convert to QuantRS2
    converter = MyQLMConverter()
    quantrs_circuit, stats = converter.from_myqlm(circuit)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Original gates: {stats.original_gates}")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Success: {stats.success}")

    if stats.warnings:
        print("\n⚠️  Warnings:")
        for warning in stats.warnings:
            print(f"   - {warning}")

    print("\n✅ Bell state circuit converted successfully!")

    return quantrs_circuit


def demo_rotation_gates():
    """Demonstrate rotation gate conversion."""
    print_section("Rotation Gates")

    if not MYQLM_AVAILABLE:
        print("⚠️  MyQLM not available, skipping...")
        return

    prog = Program()
    qbits = prog.qalloc(3)

    # Single-qubit rotations
    prog.apply(RX(np.pi/4), qbits[0])
    prog.apply(RY(np.pi/3), qbits[1])
    prog.apply(RZ(np.pi/2), qbits[2])

    # More rotations
    prog.apply(RX(np.pi/6), qbits[0])
    prog.apply(RY(np.pi/5), qbits[1])

    circuit = prog.to_circ()

    print(f"\n📋 Rotation circuit:")
    print(f"   Qubits: {circuit.nbqbits}")
    print(f"   Gates: {len(list(circuit.iterate_simple()))}")

    # Convert to QuantRS2
    converter = MyQLMConverter()
    quantrs_circuit, stats = converter.from_myqlm(circuit)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Success: {stats.success}")

    print("\n✅ Rotation gates converted successfully!")

    return quantrs_circuit


def demo_two_qubit_gates():
    """Demonstrate two-qubit gate conversion."""
    print_section("Two-Qubit Gates")

    if not MYQLM_AVAILABLE:
        print("⚠️  MyQLM not available, skipping...")
        return

    prog = Program()
    qbits = prog.qalloc(3)

    # Initialize
    prog.apply(H, qbits[0])
    prog.apply(H, qbits[1])

    # Two-qubit gates
    prog.apply(CNOT, qbits[0], qbits[1])
    prog.apply(SWAP, qbits[1], qbits[2])
    prog.apply(CNOT, qbits[0], qbits[2])

    circuit = prog.to_circ()

    print(f"\n📋 Two-qubit gate circuit:")
    print(f"   Gates: {len(list(circuit.iterate_simple()))}")

    # Convert to QuantRS2
    converter = MyQLMConverter()
    quantrs_circuit, stats = converter.from_myqlm(circuit)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Success: {stats.success}")

    print("\n✅ Two-qubit gates converted successfully!")

    return quantrs_circuit


def demo_three_qubit_gates():
    """Demonstrate three-qubit gate conversion."""
    print_section("Three-Qubit Gates")

    if not MYQLM_AVAILABLE:
        print("⚠️  MyQLM not available, skipping...")
        return

    prog = Program()
    qbits = prog.qalloc(3)

    # Initialize
    prog.apply(H, qbits[0])
    prog.apply(H, qbits[1])

    # Toffoli gate
    prog.apply(CCNOT, qbits[0], qbits[1], qbits[2])

    circuit = prog.to_circ()

    print(f"\n📋 Three-qubit gate circuit:")
    print(f"   Gates: {len(list(circuit.iterate_simple()))}")

    # Convert to QuantRS2
    converter = MyQLMConverter()
    quantrs_circuit, stats = converter.from_myqlm(circuit)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Success: {stats.success}")

    print("\n✅ Three-qubit gates converted successfully!")

    return quantrs_circuit


def demo_ghz_state():
    """Demonstrate GHZ state creation and conversion."""
    print_section("GHZ State Circuit")

    if not MYQLM_AVAILABLE:
        print("⚠️  MyQLM not available, skipping...")
        return

    # Create GHZ state circuit
    n_qubits = 4
    prog = Program()
    qbits = prog.qalloc(n_qubits)

    # Create GHZ state: |0000⟩ + |1111⟩
    prog.apply(H, qbits[0])
    for i in range(n_qubits - 1):
        prog.apply(CNOT, qbits[i], qbits[i + 1])

    circuit = prog.to_circ()

    print(f"\n📋 GHZ State Circuit ({n_qubits} qubits):")
    print(f"   Gates: {len(list(circuit.iterate_simple()))}")

    # Convert to QuantRS2
    converter = MyQLMConverter()
    quantrs_circuit, stats = converter.from_myqlm(circuit)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Success: {stats.success}")

    print("\n✅ GHZ state circuit converted successfully!")

    return quantrs_circuit


def demo_variational_circuit():
    """Demonstrate parametric/variational circuit conversion."""
    print_section("Variational Circuit")

    if not MYQLM_AVAILABLE:
        print("⚠️  MyQLM not available, skipping...")
        return

    # Create a simple variational ansatz
    prog = Program()
    qbits = prog.qalloc(3)

    # Parameters (will use fixed values for conversion)
    theta1, theta2, theta3 = np.pi/4, np.pi/3, np.pi/6

    # Layer 1: Rotations
    prog.apply(RY(theta1), qbits[0])
    prog.apply(RY(theta2), qbits[1])
    prog.apply(RY(theta3), qbits[2])

    # Layer 2: Entanglement
    prog.apply(CNOT, qbits[0], qbits[1])
    prog.apply(CNOT, qbits[1], qbits[2])

    # Layer 3: More rotations
    prog.apply(RZ(theta1 * 2), qbits[0])
    prog.apply(RZ(theta2 * 2), qbits[1])
    prog.apply(RZ(theta3 * 2), qbits[2])

    circuit = prog.to_circ()

    print(f"\n📋 Variational Ansatz:")
    print(f"   Qubits: {circuit.nbqbits}")
    print(f"   Gates: {len(list(circuit.iterate_simple()))}")
    print(f"   Parameters: θ1={theta1:.4f}, θ2={theta2:.4f}, θ3={theta3:.4f}")

    # Convert to QuantRS2
    converter = MyQLMConverter()
    quantrs_circuit, stats = converter.from_myqlm(circuit)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Success: {stats.success}")

    print("\n✅ Variational circuit converted successfully!")

    return quantrs_circuit


def demo_job_creation():
    """Demonstrate MyQLM job creation."""
    print_section("Job Creation")

    if not MYQLM_AVAILABLE:
        print("⚠️  MyQLM not available, skipping...")
        return

    # Create a circuit
    prog = Program()
    qbits = prog.qalloc(3)

    prog.apply(H, qbits[0])
    prog.apply(CNOT, qbits[0], qbits[1])
    prog.apply(CNOT, qbits[1], qbits[2])

    circuit = prog.to_circ()

    print("\n📋 Creating MyQLM job:")
    print(f"   Qubits: {circuit.nbqbits}")

    # Create job with different shot counts
    converter = MyQLMConverter()

    # Exact simulation (0 shots)
    job_exact = converter.create_job(circuit, nbshots=0)
    print(f"\n   ✓ Exact simulation job created (nbshots=0)")

    # Sampling simulation (1000 shots)
    job_sampling = converter.create_job(circuit, nbshots=1000)
    print(f"   ✓ Sampling simulation job created (nbshots=1000)")

    # Measure specific qubits
    job_selective = converter.create_job(circuit, nbshots=100, qubits=[0, 2])
    print(f"   ✓ Selective measurement job created (qubits=[0, 2])")

    print("\n✅ Job creation demonstrated successfully!")


def demo_error_handling():
    """Demonstrate error handling."""
    print_section("Error Handling")

    if not MYQLM_AVAILABLE:
        print("⚠️  MyQLM not available, skipping...")
        return

    prog = Program()
    qbits = prog.qalloc(2)

    prog.apply(H, qbits[0])
    prog.apply(CNOT, qbits[0], qbits[1])

    circuit = prog.to_circ()

    print("\n📋 Testing with default (lenient) mode:")
    converter_lenient = MyQLMConverter(strict_mode=False)
    circuit_lenient, stats_lenient = converter_lenient.from_myqlm(circuit)

    print(f"   ✓ Converted: {stats_lenient.converted_gates} gates")
    if stats_lenient.warnings:
        print(f"   ⚠️  Warnings: {len(stats_lenient.warnings)}")

    print("\n📋 Testing with strict mode:")
    converter_strict = MyQLMConverter(strict_mode=True)
    try:
        circuit_strict, stats_strict = converter_strict.from_myqlm(circuit)
        print(f"   ✓ Converted: {stats_strict.converted_gates} gates")
    except ValueError as e:
        print(f"   ❌ Strict mode error: {e}")

    print("\n✅ Error handling demonstrated successfully!")


def main():
    """Run all demonstration examples."""
    print("\n" + "█" * 70)
    print("  QuantRS2 MyQLM/QLM Converter - Comprehensive Demonstration")
    print("█" * 70)

    if not QUANTRS2_AVAILABLE:
        print("\n❌ QuantRS2 not available. Cannot run demonstrations.")
        return

    if not MYQLM_AVAILABLE:
        print("\n⚠️  MyQLM not available. Demonstrations will be skipped.")
        print("Install MyQLM with: pip install myqlm")
        print("\n💡 Note: MyQLM is the local version of Atos QLM")
        print("   For full QLM features, contact Atos Quantum")
        return

    try:
        # Run all demonstrations
        demo_basic_conversion()
        demo_rotation_gates()
        demo_two_qubit_gates()
        demo_three_qubit_gates()
        demo_ghz_state()
        demo_variational_circuit()
        demo_job_creation()
        demo_error_handling()

        # Final summary
        print_section("Summary")
        print("\n✅ All demonstrations completed successfully!")
        print("\n📚 Key Features Demonstrated:")
        print("   ✓ Basic circuit conversion")
        print("   ✓ Rotation gates (RX, RY, RZ)")
        print("   ✓ Two-qubit gates (CNOT, SWAP)")
        print("   ✓ Three-qubit gates (Toffoli)")
        print("   ✓ GHZ state preparation")
        print("   ✓ Variational circuits")
        print("   ✓ Job creation and configuration")
        print("   ✓ Error handling")

        print("\n🎯 Next Steps:")
        print("   • Try converting your own MyQLM circuits")
        print("   • Explore other framework converters (Qiskit, Cirq, ProjectQ)")
        print("   • Learn about Atos QLM's advanced features")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
