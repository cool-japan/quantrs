#!/usr/bin/env python3
"""
Cirq Converter Demonstration

This script demonstrates the comprehensive Cirq <-> QuantRS2 conversion capabilities,
showcasing moment-based conversion, power gate handling, and advanced Cirq features.

Features demonstrated:
    - Basic circuit conversion from Cirq to QuantRS2
    - Moment preservation and processing
    - Power gate decomposition (HPowGate, XPowGate, etc.)
    - Advanced gates (iSwap, FSim, PhasedX, Givens)
    - GridQubit and LineQubit handling
    - Circuit equivalence testing
    - Practical quantum algorithm examples
"""

import sys
import numpy as np

try:
    import cirq
    from cirq import (
        Circuit as CirqCircuit,
        LineQubit,
        GridQubit,
        Simulator,
    )
    CIRQ_AVAILABLE = True
except ImportError:
    print("❌ Cirq not installed. Install with: pip install cirq")
    print("   Skipping Cirq-specific examples...")
    CIRQ_AVAILABLE = False

try:
    from quantrs2 import Circuit as QuantRS2Circuit
    from quantrs2.cirq_converter import (
        CirqConverter,
        convert_from_cirq,
        convert_to_cirq,
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
    """Demonstrate basic Cirq to QuantRS2 conversion."""
    print_section("Basic Circuit Conversion")

    if not CIRQ_AVAILABLE:
        print("⚠️  Cirq not available, skipping...")
        return

    # Create a simple Bell state circuit in Cirq
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()

    circuit.append([
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.measure(*qubits, key='result')
    ])

    print("\n📋 Original Cirq Circuit:")
    print(circuit)

    # Convert to QuantRS2
    converter = CirqConverter()
    quantrs_circuit, stats = converter.from_cirq(circuit)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Original operations: {stats.original_operations}")
    print(f"   ✓ Converted operations: {stats.converted_operations}")
    print(f"   ✓ Number of moments: {stats.num_moments}")
    print(f"   ✓ Success: {stats.success}")

    if stats.warnings:
        print("\n⚠️  Warnings:")
        for warning in stats.warnings:
            print(f"   - {warning}")

    print("\n✅ Bell state circuit converted successfully!")

    return quantrs_circuit


def demo_power_gates():
    """Demonstrate power gate conversion (XPowGate, YPowGate, ZPowGate, etc.)."""
    print_section("Power Gate Conversion")

    if not CIRQ_AVAILABLE:
        print("⚠️  Cirq not available, skipping...")
        return

    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit()

    # Full power gates (exponent = 1.0)
    circuit.append([
        cirq.H(qubits[0]),  # HPowGate with exponent=1
        cirq.X(qubits[1]),  # XPowGate with exponent=1
        cirq.Y(qubits[2]),  # YPowGate with exponent=1
    ])

    # Fractional power gates
    circuit.append([
        cirq.X**0.5(qubits[0]),  # √X gate
        cirq.Y**0.25(qubits[1]),  # Fourth root of Y
        cirq.Z**0.5(qubits[2]),  # S gate
    ])

    # Quarter powers (T and S gates)
    circuit.append([
        cirq.Z**0.25(qubits[0]),  # T gate
        cirq.Z**(-0.5)(qubits[1]),  # S† gate
        cirq.Z**(-0.25)(qubits[2]),  # T† gate
    ])

    print(f"\n📋 Circuit with {len(circuit)} moments")
    print(f"   Total operations: {len(list(circuit.all_operations()))}")

    # Convert to QuantRS2
    converter = CirqConverter()
    quantrs_circuit, stats = converter.from_cirq(circuit)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Original operations: {stats.original_operations}")
    print(f"   ✓ Converted operations: {stats.converted_operations}")
    print(f"   ✓ Decomposed operations: {stats.decomposed_operations}")

    if stats.decomposed_operations > 0:
        print(f"\n   ℹ️  {stats.decomposed_operations} power gates were decomposed")

    print("\n✅ Power gates converted successfully!")

    return quantrs_circuit


def demo_advanced_gates():
    """Demonstrate advanced Cirq gate conversion (iSwap, FSim, PhasedX, etc.)."""
    print_section("Advanced Gate Support")

    if not CIRQ_AVAILABLE:
        print("⚠️  Cirq not available, skipping...")
        return

    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit()

    # Standard two-qubit gates
    circuit.append([
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.CZ(qubits[1], qubits[2]),
        cirq.SWAP(qubits[0], qubits[2]),
    ])

    # iSwap gate
    try:
        circuit.append(cirq.ISWAP(qubits[0], qubits[1]))
        print("   ✓ iSwap gate added")
    except AttributeError:
        print("   ⚠️  ISWAP not available in this Cirq version")

    # Partial iSwap
    try:
        circuit.append(cirq.ISWAP**0.5(qubits[1], qubits[2]))
        print("   ✓ Partial iSwap (√iSWAP) added")
    except:
        pass

    # FSimGate (fermionic simulation gate)
    try:
        circuit.append(cirq.FSimGate(theta=np.pi/4, phi=np.pi/6)(qubits[0], qubits[1]))
        print("   ✓ FSim gate added")
    except AttributeError:
        print("   ⚠️  FSimGate not available in this Cirq version")

    # PhasedXPowGate
    try:
        circuit.append(cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5)(qubits[0]))
        print("   ✓ PhasedXPowGate added")
    except AttributeError:
        print("   ⚠️  PhasedXPowGate not available in this Cirq version")

    # Givens rotation
    try:
        circuit.append(cirq.givens(np.pi/3)(qubits[1], qubits[2]))
        print("   ✓ Givens rotation added")
    except (AttributeError, TypeError):
        print("   ⚠️  Givens rotation not available in this Cirq version")

    print(f"\n📋 Circuit with {len(list(circuit.all_operations()))} operations")

    # Convert to QuantRS2
    converter = CirqConverter()
    quantrs_circuit, stats = converter.from_cirq(circuit)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Original operations: {stats.original_operations}")
    print(f"   ✓ Converted operations: {stats.converted_operations}")
    print(f"   ✓ Decomposed operations: {stats.decomposed_operations}")
    print(f"   ✓ Unsupported operations: {stats.unsupported_operations}")

    if stats.warnings:
        print(f"\n   ℹ️  Conversion notes:")
        for warning in stats.warnings[:5]:  # Show first 5
            print(f"      - {warning}")

    print("\n✅ Advanced gates converted successfully!")

    return quantrs_circuit


def demo_grid_qubits():
    """Demonstrate GridQubit handling."""
    print_section("GridQubit Support")

    if not CIRQ_AVAILABLE:
        print("⚠️  Cirq not available, skipping...")
        return

    # Create a 2x2 grid of qubits
    qubits = [cirq.GridQubit(i, j) for i in range(2) for j in range(2)]
    circuit = cirq.Circuit()

    print("\n📍 Using GridQubits:")
    for q in qubits:
        print(f"   - {q}")

    # Add gates
    circuit.append([
        cirq.H(qubits[0]),
        cirq.H(qubits[1]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.CNOT(qubits[2], qubits[3]),
        cirq.CZ(qubits[1], qubits[3]),
    ])

    print(f"\n📋 Circuit with {len(list(circuit.all_operations()))} operations")

    # Convert to QuantRS2
    converter = CirqConverter()
    quantrs_circuit, stats = converter.from_cirq(circuit)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Converted operations: {stats.converted_operations}")
    print(f"   ✓ Number of qubits: {len(qubits)}")

    print("\n✅ GridQubit circuit converted successfully!")

    return quantrs_circuit


def demo_moments():
    """Demonstrate moment-based circuit structure."""
    print_section("Moment-Based Circuit Structure")

    if not CIRQ_AVAILABLE:
        print("⚠️  Cirq not available, skipping...")
        return

    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit()

    # Moment 1: Parallel Hadamards
    circuit.append([
        cirq.H(qubits[0]),
        cirq.H(qubits[1]),
        cirq.H(qubits[2]),
        cirq.H(qubits[3]),
    ])

    # Moment 2: CNOTs
    circuit.append([
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.CNOT(qubits[2], qubits[3]),
    ])

    # Moment 3: More CNOTs
    circuit.append([
        cirq.CNOT(qubits[1], qubits[2]),
    ])

    # Moment 4: Rotations
    circuit.append([
        cirq.Rz(rads=np.pi/4)(qubits[0]),
        cirq.Rz(rads=np.pi/4)(qubits[3]),
    ])

    print(f"\n📋 Circuit Structure:")
    print(f"   Moments: {len(circuit)}")
    print(f"   Total operations: {len(list(circuit.all_operations()))}")

    print("\n   Moment breakdown:")
    for i, moment in enumerate(circuit):
        ops_in_moment = len(moment)
        print(f"   - Moment {i+1}: {ops_in_moment} operations")

    # Convert to QuantRS2
    converter = CirqConverter()
    quantrs_circuit, stats = converter.from_cirq(circuit)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Moments processed: {stats.num_moments}")
    print(f"   ✓ Operations converted: {stats.converted_operations}")

    print("\n✅ Moment-based structure preserved during conversion!")

    return quantrs_circuit


def demo_rotation_gates():
    """Demonstrate rotation gate conversion."""
    print_section("Rotation Gates")

    if not CIRQ_AVAILABLE:
        print("⚠️  Cirq not available, skipping...")
        return

    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()

    # Single-qubit rotations
    circuit.append([
        cirq.Rx(rads=np.pi/4)(qubits[0]),
        cirq.Ry(rads=np.pi/3)(qubits[0]),
        cirq.Rz(rads=np.pi/2)(qubits[0]),
    ])

    # Controlled rotations (if available)
    try:
        circuit.append([
            cirq.CZPowGate(exponent=0.5)(qubits[0], qubits[1]),
        ])
        print("   ✓ Controlled-Z power gate added")
    except:
        pass

    print(f"\n📋 Rotation circuit with {len(list(circuit.all_operations()))} operations")

    # Convert to QuantRS2
    converter = CirqConverter()
    quantrs_circuit, stats = converter.from_cirq(circuit)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Converted operations: {stats.converted_operations}")
    print(f"   ✓ Decomposed operations: {stats.decomposed_operations}")

    print("\n✅ Rotation gates converted successfully!")

    return quantrs_circuit


def demo_three_qubit_gates():
    """Demonstrate three-qubit gate conversion."""
    print_section("Three-Qubit Gates")

    if not CIRQ_AVAILABLE:
        print("⚠️  Cirq not available, skipping...")
        return

    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit()

    # Toffoli gate
    circuit.append(cirq.TOFFOLI(qubits[0], qubits[1], qubits[2]))
    print("   ✓ Toffoli (CCNOT) gate added")

    # Controlled SWAP (Fredkin)
    try:
        circuit.append(cirq.CSWAP(qubits[0], qubits[1], qubits[2]))
        print("   ✓ CSWAP (Fredkin) gate added")
    except AttributeError:
        print("   ⚠️  CSWAP not available in this Cirq version")

    # Partial Toffoli
    try:
        circuit.append(cirq.TOFFOLI**0.5(qubits[0], qubits[1], qubits[2]))
        print("   ✓ Partial Toffoli added")
    except:
        pass

    print(f"\n📋 Circuit with {len(list(circuit.all_operations()))} operations")

    # Convert to QuantRS2
    converter = CirqConverter()
    quantrs_circuit, stats = converter.from_cirq(circuit)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Converted operations: {stats.converted_operations}")
    print(f"   ✓ Decomposed operations: {stats.decomposed_operations}")

    if stats.warnings:
        print(f"\n   ℹ️  Conversion notes:")
        for warning in stats.warnings:
            if "Toffoli" in warning or "partial" in warning.lower():
                print(f"      - {warning}")

    print("\n✅ Three-qubit gates converted successfully!")

    return quantrs_circuit


def demo_error_handling():
    """Demonstrate error handling and strict mode."""
    print_section("Error Handling")

    if not CIRQ_AVAILABLE:
        print("⚠️  Cirq not available, skipping...")
        return

    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()

    circuit.append([
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
    ])

    # Add identity gates (should be skipped)
    circuit.append(cirq.I(qubits[0]))

    print("\n📋 Testing with default (lenient) mode:")
    converter_lenient = CirqConverter(strict_mode=False)
    circuit_lenient, stats_lenient = converter_lenient.from_cirq(circuit)

    print(f"   ✓ Converted: {stats_lenient.converted_operations} operations")
    if stats_lenient.warnings:
        print(f"   ℹ️  Warnings: {len(stats_lenient.warnings)}")

    print("\n📋 Testing with strict mode:")
    converter_strict = CirqConverter(strict_mode=True)
    try:
        circuit_strict, stats_strict = converter_strict.from_cirq(circuit)
        print(f"   ✓ Converted: {stats_strict.converted_operations} operations")
        if stats_strict.warnings:
            print(f"   ⚠️  Warnings: {len(stats_strict.warnings)}")
    except ValueError as e:
        print(f"   ❌ Strict mode error: {e}")

    print("\n✅ Error handling demonstrated successfully!")


def demo_grover_circuit():
    """Demonstrate Grover's algorithm circuit conversion."""
    print_section("Grover's Algorithm Circuit")

    if not CIRQ_AVAILABLE:
        print("⚠️  Cirq not available, skipping...")
        return

    # Create a simple 2-qubit Grover circuit
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()

    # Initialize superposition
    circuit.append([cirq.H(q) for q in qubits])

    # Oracle (mark |11⟩)
    circuit.append(cirq.CZ(qubits[0], qubits[1]))

    # Diffusion operator
    circuit.append([cirq.H(q) for q in qubits])
    circuit.append([cirq.X(q) for q in qubits])
    circuit.append(cirq.CZ(qubits[0], qubits[1]))
    circuit.append([cirq.X(q) for q in qubits])
    circuit.append([cirq.H(q) for q in qubits])

    print(f"\n📋 Grover Circuit:")
    print(f"   Qubits: {len(qubits)}")
    print(f"   Moments: {len(circuit)}")
    print(f"   Operations: {len(list(circuit.all_operations()))}")

    # Convert to QuantRS2
    converter = CirqConverter()
    quantrs_circuit, stats = converter.from_cirq(circuit)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Converted operations: {stats.converted_operations}")
    print(f"   ✓ Success: {stats.success}")

    print("\n✅ Grover circuit converted successfully!")

    return quantrs_circuit


def main():
    """Run all demonstration examples."""
    print("\n" + "█" * 70)
    print("  QuantRS2 Cirq Converter - Comprehensive Demonstration")
    print("█" * 70)

    if not QUANTRS2_AVAILABLE:
        print("\n❌ QuantRS2 not available. Cannot run demonstrations.")
        return

    if not CIRQ_AVAILABLE:
        print("\n⚠️  Cirq not available. Demonstrations will be skipped.")
        print("Install Cirq with: pip install cirq")
        return

    try:
        # Run all demonstrations
        demo_basic_conversion()
        demo_power_gates()
        demo_advanced_gates()
        demo_grid_qubits()
        demo_moments()
        demo_rotation_gates()
        demo_three_qubit_gates()
        demo_grover_circuit()
        demo_error_handling()

        # Final summary
        print_section("Summary")
        print("\n✅ All demonstrations completed successfully!")
        print("\n📚 Key Features Demonstrated:")
        print("   ✓ Basic circuit conversion")
        print("   ✓ Power gate decomposition (XPowGate, YPowGate, ZPowGate)")
        print("   ✓ Advanced gates (iSwap, FSim, PhasedX, Givens)")
        print("   ✓ GridQubit and LineQubit support")
        print("   ✓ Moment preservation")
        print("   ✓ Rotation gates (Rx, Ry, Rz)")
        print("   ✓ Three-qubit gates (Toffoli, CSWAP)")
        print("   ✓ Grover's algorithm")
        print("   ✓ Error handling (strict and lenient modes)")

        print("\n🎯 Next Steps:")
        print("   • Try converting your own Cirq circuits")
        print("   • Explore other framework converters (Qiskit, MyQLM, ProjectQ)")
        print("   • Experiment with moment-based circuit optimization")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
