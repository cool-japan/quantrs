#!/usr/bin/env python3
"""
ProjectQ Converter Demonstration

This script demonstrates the ProjectQ <-> QuantRS2 conversion capabilities,
showcasing command extraction, backend integration, and practical quantum algorithms.

Features demonstrated:
    - Basic circuit conversion from ProjectQ to QuantRS2
    - Command extraction from MainEngine
    - Controlled gate support
    - Backend adapter usage
    - Qubit allocation and deallocation
    - Practical quantum algorithm examples
"""

import sys
import numpy as np

try:
    import projectq
    from projectq import MainEngine
    from projectq.ops import (
        H, X, Y, Z, S, T, Rx, Ry, Rz, CNOT, CX, CZ, Swap,
        Toffoli, Measure, All, Barrier, SqrtX
    )
    from projectq.backends import CommandPrinter, Simulator
    from projectq.meta import Control
    PROJECTQ_AVAILABLE = True
except ImportError:
    print("❌ ProjectQ not installed. Install with: pip install projectq")
    print("   Skipping ProjectQ-specific examples...")
    PROJECTQ_AVAILABLE = False

try:
    from quantrs2 import Circuit as QuantRS2Circuit
    from quantrs2.projectq_converter import (
        ProjectQConverter,
        convert_from_projectq,
        convert_to_projectq,
        ProjectQBackend,
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
    """Demonstrate basic ProjectQ to QuantRS2 conversion."""
    print_section("Basic Circuit Conversion")

    if not PROJECTQ_AVAILABLE:
        print("⚠️  ProjectQ not available, skipping...")
        return

    # Create a simple Bell state circuit in ProjectQ
    eng = MainEngine()
    qubits = eng.allocate_qureg(2)

    H | qubits[0]
    CNOT | (qubits[0], qubits[1])

    eng.flush()

    print("\n📋 Original ProjectQ Circuit:")
    print(f"   Qubits: {len(qubits)}")
    print(f"   Commands: {len(eng.backend.received_commands) if hasattr(eng.backend, 'received_commands') else 'N/A'}")

    # Convert to QuantRS2
    converter = ProjectQConverter()
    quantrs_circuit, stats = converter.from_projectq(eng)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Original commands: {stats.original_commands}")
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

    if not PROJECTQ_AVAILABLE:
        print("⚠️  ProjectQ not available, skipping...")
        return

    eng = MainEngine()
    qubits = eng.allocate_qureg(3)

    # Single-qubit rotations
    Rx(np.pi/4) | qubits[0]
    Ry(np.pi/3) | qubits[1]
    Rz(np.pi/2) | qubits[2]

    # More rotations
    Rx(np.pi/6) | qubits[0]
    Ry(np.pi/5) | qubits[1]

    eng.flush()

    print(f"\n📋 Rotation circuit:")
    print(f"   Qubits: {len(qubits)}")

    # Convert to QuantRS2
    converter = ProjectQConverter()
    quantrs_circuit, stats = converter.from_projectq(eng)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Success: {stats.success}")

    print("\n✅ Rotation gates converted successfully!")

    return quantrs_circuit


def demo_controlled_gates():
    """Demonstrate controlled gate conversion."""
    print_section("Controlled Gates")

    if not PROJECTQ_AVAILABLE:
        print("⚠️  ProjectQ not available, skipping...")
        return

    eng = MainEngine()
    qubits = eng.allocate_qureg(3)

    # Initialize
    H | qubits[0]

    # Controlled gates
    with Control(eng, qubits[0]):
        X | qubits[1]  # Controlled-X (CNOT)

    with Control(eng, qubits[0]):
        Z | qubits[2]  # Controlled-Z

    # Standard CNOT
    CNOT | (qubits[1], qubits[2])

    eng.flush()

    print(f"\n📋 Controlled gate circuit:")
    print(f"   Qubits: {len(qubits)}")

    # Convert to QuantRS2
    converter = ProjectQConverter()
    quantrs_circuit, stats = converter.from_projectq(eng)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Success: {stats.success}")

    print("\n✅ Controlled gates converted successfully!")

    return quantrs_circuit


def demo_three_qubit_gates():
    """Demonstrate three-qubit gate conversion."""
    print_section("Three-Qubit Gates")

    if not PROJECTQ_AVAILABLE:
        print("⚠️  ProjectQ not available, skipping...")
        return

    eng = MainEngine()
    qubits = eng.allocate_qureg(3)

    # Initialize
    H | qubits[0]
    H | qubits[1]

    # Toffoli gate
    Toffoli | (qubits[0], qubits[1], qubits[2])

    eng.flush()

    print(f"\n📋 Three-qubit gate circuit:")
    print(f"   Qubits: {len(qubits)}")

    # Convert to QuantRS2
    converter = ProjectQConverter()
    quantrs_circuit, stats = converter.from_projectq(eng)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Success: {stats.success}")

    print("\n✅ Three-qubit gates converted successfully!")

    return quantrs_circuit


def demo_ghz_state():
    """Demonstrate GHZ state creation and conversion."""
    print_section("GHZ State Circuit")

    if not PROJECTQ_AVAILABLE:
        print("⚠️  ProjectQ not available, skipping...")
        return

    # Create GHZ state circuit
    n_qubits = 4
    eng = MainEngine()
    qubits = eng.allocate_qureg(n_qubits)

    # Create GHZ state: |0000⟩ + |1111⟩
    H | qubits[0]
    for i in range(n_qubits - 1):
        CNOT | (qubits[i], qubits[i + 1])

    eng.flush()

    print(f"\n📋 GHZ State Circuit ({n_qubits} qubits):")
    print(f"   Qubits: {len(qubits)}")

    # Convert to QuantRS2
    converter = ProjectQConverter()
    quantrs_circuit, stats = converter.from_projectq(eng)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Success: {stats.success}")

    print("\n✅ GHZ state circuit converted successfully!")

    return quantrs_circuit


def demo_quantum_fourier_transform():
    """Demonstrate QFT circuit conversion."""
    print_section("Quantum Fourier Transform")

    if not PROJECTQ_AVAILABLE:
        print("⚠️  ProjectQ not available, skipping...")
        return

    # Create simple 3-qubit QFT
    n_qubits = 3
    eng = MainEngine()
    qubits = eng.allocate_qureg(n_qubits)

    # Simplified QFT (just the structure, not full implementation)
    for i in range(n_qubits):
        H | qubits[i]
        for j in range(i + 1, n_qubits):
            angle = np.pi / (2 ** (j - i))
            # Controlled phase rotation (approximated with Rz)
            with Control(eng, qubits[j]):
                Rz(angle) | qubits[i]

    # Swap qubits
    for i in range(n_qubits // 2):
        Swap | (qubits[i], qubits[n_qubits - 1 - i])

    eng.flush()

    print(f"\n📋 QFT Circuit ({n_qubits} qubits):")
    print(f"   Qubits: {len(qubits)}")

    # Convert to QuantRS2
    converter = ProjectQConverter()
    quantrs_circuit, stats = converter.from_projectq(eng)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Decomposed gates: {stats.decomposed_gates}")
    print(f"   ✓ Success: {stats.success}")

    print("\n✅ QFT circuit converted successfully!")

    return quantrs_circuit


def demo_backend_adapter():
    """Demonstrate ProjectQBackend adapter usage."""
    print_section("QuantRS2 Backend Adapter")

    if not PROJECTQ_AVAILABLE:
        print("⚠️  ProjectQ not available, skipping...")
        return

    print("\n📋 Using QuantRS2 as ProjectQ backend:")

    # Create engine with QuantRS2 backend
    backend = ProjectQBackend()
    eng = MainEngine(backend=backend)

    qubits = eng.allocate_qureg(2)

    # Build circuit
    H | qubits[0]
    CNOT | (qubits[0], qubits[1])

    eng.flush()

    print(f"   ✓ Circuit executed on QuantRS2 backend")
    print(f"   ✓ Commands received: {len(backend.received_commands)}")

    # The backend automatically converts to QuantRS2
    if backend._circuit:
        print(f"   ✓ QuantRS2 circuit created with {backend._circuit.n_qubits} qubits")

    print("\n✅ Backend adapter demonstrated successfully!")


def demo_error_handling():
    """Demonstrate error handling."""
    print_section("Error Handling")

    if not PROJECTQ_AVAILABLE:
        print("⚠️  ProjectQ not available, skipping...")
        return

    eng = MainEngine()
    qubits = eng.allocate_qureg(2)

    H | qubits[0]
    CNOT | (qubits[0], qubits[1])

    eng.flush()

    print("\n📋 Testing with default (lenient) mode:")
    converter_lenient = ProjectQConverter(strict_mode=False)
    circuit_lenient, stats_lenient = converter_lenient.from_projectq(eng)

    print(f"   ✓ Converted: {stats_lenient.converted_gates} gates")
    if stats_lenient.warnings:
        print(f"   ⚠️  Warnings: {len(stats_lenient.warnings)}")

    print("\n📋 Testing with strict mode:")
    converter_strict = ProjectQConverter(strict_mode=True)
    try:
        circuit_strict, stats_strict = converter_strict.from_projectq(eng)
        print(f"   ✓ Converted: {stats_strict.converted_gates} gates")
    except ValueError as e:
        print(f"   ❌ Strict mode error: {e}")

    print("\n✅ Error handling demonstrated successfully!")


def demo_variational_circuit():
    """Demonstrate variational circuit conversion."""
    print_section("Variational Circuit")

    if not PROJECTQ_AVAILABLE:
        print("⚠️  ProjectQ not available, skipping...")
        return

    # Create a simple variational ansatz
    eng = MainEngine()
    qubits = eng.allocate_qureg(3)

    # Parameters
    theta1, theta2, theta3 = np.pi/4, np.pi/3, np.pi/6

    # Layer 1: Rotations
    Ry(theta1) | qubits[0]
    Ry(theta2) | qubits[1]
    Ry(theta3) | qubits[2]

    # Layer 2: Entanglement
    CNOT | (qubits[0], qubits[1])
    CNOT | (qubits[1], qubits[2])

    # Layer 3: More rotations
    Rz(theta1 * 2) | qubits[0]
    Rz(theta2 * 2) | qubits[1]
    Rz(theta3 * 2) | qubits[2]

    eng.flush()

    print(f"\n📋 Variational Ansatz:")
    print(f"   Qubits: {len(qubits)}")
    print(f"   Parameters: θ1={theta1:.4f}, θ2={theta2:.4f}, θ3={theta3:.4f}")

    # Convert to QuantRS2
    converter = ProjectQConverter()
    quantrs_circuit, stats = converter.from_projectq(eng)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Success: {stats.success}")

    print("\n✅ Variational circuit converted successfully!")

    return quantrs_circuit


def main():
    """Run all demonstration examples."""
    print("\n" + "█" * 70)
    print("  QuantRS2 ProjectQ Converter - Comprehensive Demonstration")
    print("█" * 70)

    if not QUANTRS2_AVAILABLE:
        print("\n❌ QuantRS2 not available. Cannot run demonstrations.")
        return

    if not PROJECTQ_AVAILABLE:
        print("\n⚠️  ProjectQ not available. Demonstrations will be skipped.")
        print("Install ProjectQ with: pip install projectq")
        return

    try:
        # Run all demonstrations
        demo_basic_conversion()
        demo_rotation_gates()
        demo_controlled_gates()
        demo_three_qubit_gates()
        demo_ghz_state()
        demo_quantum_fourier_transform()
        demo_variational_circuit()
        demo_backend_adapter()
        demo_error_handling()

        # Final summary
        print_section("Summary")
        print("\n✅ All demonstrations completed successfully!")
        print("\n📚 Key Features Demonstrated:")
        print("   ✓ Basic circuit conversion")
        print("   ✓ Rotation gates (Rx, Ry, Rz)")
        print("   ✓ Controlled gates (CNOT, CZ, etc.)")
        print("   ✓ Three-qubit gates (Toffoli)")
        print("   ✓ GHZ state preparation")
        print("   ✓ Quantum Fourier Transform")
        print("   ✓ Variational circuits")
        print("   ✓ QuantRS2 backend adapter")
        print("   ✓ Error handling")

        print("\n🎯 Next Steps:")
        print("   • Try converting your own ProjectQ circuits")
        print("   • Use QuantRS2 as a ProjectQ backend")
        print("   • Explore other framework converters (Qiskit, Cirq, MyQLM)")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
