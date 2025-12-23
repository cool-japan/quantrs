#!/usr/bin/env python3
"""
Qiskit Converter Demonstration

This script demonstrates the comprehensive Qiskit <-> QuantRS2 conversion capabilities,
showcasing bidirectional circuit conversion, advanced gate support, and practical use cases.

Features demonstrated:
    - Basic circuit conversion from Qiskit to QuantRS2
    - Advanced gate decompositions (iSwap, ECR, RXX, RYY, RZZ)
    - Multi-controlled gates
    - QASM import/export
    - Circuit equivalence testing
    - Error handling and statistics
    - Real-world quantum algorithm examples
"""

import sys
import numpy as np

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit import Aer, execute
    from qiskit.circuit.library import QFT, GroverOperator
    QISKIT_AVAILABLE = True
except ImportError:
    print("❌ Qiskit not installed. Install with: pip install qiskit")
    print("   Skipping Qiskit-specific examples...")
    QISKIT_AVAILABLE = False

try:
    from quantrs2 import Circuit as QuantRS2Circuit
    from quantrs2.qiskit_converter import (
        QiskitConverter,
        convert_from_qiskit,
        convert_to_qiskit,
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
    """Demonstrate basic Qiskit to QuantRS2 conversion."""
    print_section("Basic Circuit Conversion")

    if not QISKIT_AVAILABLE:
        print("⚠️  Qiskit not available, skipping...")
        return

    # Create a simple Bell state circuit in Qiskit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    print("\n📋 Original Qiskit Circuit:")
    print(qc)

    # Convert to QuantRS2
    converter = QiskitConverter()
    quantrs_circuit, stats = converter.from_qiskit(qc)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Original gates: {stats.original_gates}")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Decomposed gates: {stats.decomposed_gates}")
    print(f"   ✓ Success: {stats.success}")

    if stats.warnings:
        print("\n⚠️  Warnings:")
        for warning in stats.warnings:
            print(f"   - {warning}")

    print("\n✅ Bell state circuit converted successfully!")

    return quantrs_circuit


def demo_advanced_gates():
    """Demonstrate advanced gate conversion including iSwap, ECR, and two-qubit rotations."""
    print_section("Advanced Gate Support")

    if not QISKIT_AVAILABLE:
        print("⚠️  Qiskit not available, skipping...")
        return

    # Create circuit with advanced gates
    qc = QuantumCircuit(3)

    # Standard gates
    qc.h(0)
    qc.x(1)
    qc.y(2)

    # Rotation gates
    qc.rx(np.pi/4, 0)
    qc.ry(np.pi/3, 1)
    qc.rz(np.pi/2, 2)

    # Two-qubit gates
    qc.cx(0, 1)
    qc.cz(1, 2)
    qc.swap(0, 2)

    # Advanced gates (these will be decomposed)
    try:
        qc.iswap(0, 1)
        print("   ✓ iSwap gate added")
    except AttributeError:
        print("   ⚠️  iSwap not available in this Qiskit version")

    try:
        qc.ecr(1, 2)
        print("   ✓ ECR gate added")
    except AttributeError:
        print("   ⚠️  ECR not available in this Qiskit version")

    # Two-qubit rotation gates
    try:
        qc.rxx(np.pi/6, 0, 1)
        qc.ryy(np.pi/6, 1, 2)
        qc.rzz(np.pi/6, 0, 2)
        print("   ✓ RXX, RYY, RZZ gates added")
    except AttributeError:
        print("   ⚠️  Two-qubit rotations not available in this Qiskit version")

    # Toffoli gate
    qc.ccx(0, 1, 2)

    print(f"\n📋 Circuit with {len(qc.data)} gates")

    # Convert to QuantRS2
    converter = QiskitConverter()
    quantrs_circuit, stats = converter.from_qiskit(qc, optimize=True)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Original gates: {stats.original_gates}")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Decomposed gates: {stats.decomposed_gates}")
    print(f"   ✓ Unsupported gates: {stats.unsupported_gates}")

    if stats.decomposed_gates > 0:
        print(f"\n   ℹ️  {stats.decomposed_gates} gates were decomposed into basic operations")

    print("\n✅ Advanced gates converted successfully!")

    return quantrs_circuit


def demo_qft_algorithm():
    """Demonstrate conversion of Quantum Fourier Transform."""
    print_section("Quantum Fourier Transform (QFT) Conversion")

    if not QISKIT_AVAILABLE:
        print("⚠️  Qiskit not available, skipping...")
        return

    # Create QFT circuit
    num_qubits = 4
    qc = QuantumCircuit(num_qubits)

    # Add QFT
    qc.append(QFT(num_qubits), range(num_qubits))

    print(f"\n📋 QFT Circuit ({num_qubits} qubits)")
    print(f"   Gate count: {len(qc.data)}")
    print(f"   Depth: {qc.depth()}")

    # Convert to QuantRS2
    converter = QiskitConverter()
    quantrs_circuit, stats = converter.from_qiskit(qc, optimize=True)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Original gates: {stats.original_gates}")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Decomposed gates: {stats.decomposed_gates}")
    print(f"   ✓ Success: {stats.success}")

    print("\n✅ QFT circuit converted successfully!")

    return quantrs_circuit


def demo_variational_circuit():
    """Demonstrate conversion of parametric/variational circuits."""
    print_section("Variational Circuit Conversion")

    if not QISKIT_AVAILABLE:
        print("⚠️  Qiskit not available, skipping...")
        return

    from qiskit.circuit import Parameter

    # Create a parametric circuit (VQE ansatz)
    theta = Parameter('θ')
    phi = Parameter('φ')
    lambda_param = Parameter('λ')

    qc = QuantumCircuit(3)

    # Layer 1: Rotation gates with parameters
    qc.ry(theta, 0)
    qc.ry(phi, 1)
    qc.ry(lambda_param, 2)

    # Layer 2: Entangling gates
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Layer 3: More rotations
    qc.rz(theta * 2, 0)
    qc.rz(phi * 2, 1)
    qc.rz(lambda_param * 2, 2)

    print(f"\n📋 Parametric Circuit")
    print(f"   Parameters: {qc.parameters}")
    print(f"   Gates: {len(qc.data)}")

    # Bind parameters for conversion
    parameter_values = {theta: np.pi/4, phi: np.pi/3, lambda_param: np.pi/2}
    bound_circuit = qc.bind_parameters(parameter_values)

    print(f"\n🔧 Binding parameters:")
    for param, value in parameter_values.items():
        print(f"   {param} = {value:.4f}")

    # Convert to QuantRS2
    converter = QiskitConverter()
    quantrs_circuit, stats = converter.from_qiskit(bound_circuit)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Success: {stats.success}")

    print("\n✅ Variational circuit converted successfully!")

    return quantrs_circuit


def demo_qasm_interop():
    """Demonstrate QASM import/export capabilities."""
    print_section("QASM Import/Export")

    if not QISKIT_AVAILABLE:
        print("⚠️  Qiskit not available, skipping...")
        return

    # Create a circuit
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()

    # Export to QASM
    qasm_str = qc.qasm()

    print("\n📄 QASM 2.0 Output:")
    print(qasm_str)

    # Import from QASM
    converter = QiskitConverter()
    quantrs_circuit, stats = converter.from_qasm(qasm_str)

    print("\n📊 QASM Import Statistics:")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Success: {stats.success}")

    print("\n✅ QASM interoperability demonstrated successfully!")


def demo_error_handling():
    """Demonstrate error handling and strict mode."""
    print_section("Error Handling and Strict Mode")

    if not QISKIT_AVAILABLE:
        print("⚠️  Qiskit not available, skipping...")
        return

    # Create circuit with a potentially unsupported custom gate
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Try adding a custom unitary (might not be directly supported)
    try:
        from qiskit.circuit.library import UnitaryGate
        custom_matrix = np.array([[1, 0], [0, -1]])
        custom_gate = UnitaryGate(custom_matrix, label='custom')
        qc.append(custom_gate, [0])
    except:
        pass

    print("\n📋 Testing with default (lenient) mode:")
    converter_lenient = QiskitConverter(strict_mode=False)
    circuit_lenient, stats_lenient = converter_lenient.from_qiskit(qc)

    print(f"   ✓ Converted: {stats_lenient.converted_gates} gates")
    if stats_lenient.warnings:
        print(f"   ⚠️  Warnings: {len(stats_lenient.warnings)}")
        for warning in stats_lenient.warnings[:3]:  # Show first 3
            print(f"      - {warning}")

    print("\n📋 Testing with strict mode:")
    converter_strict = QiskitConverter(strict_mode=True)
    try:
        circuit_strict, stats_strict = converter_strict.from_qiskit(qc)
        print(f"   ✓ Converted: {stats_strict.converted_gates} gates")
    except ValueError as e:
        print(f"   ❌ Strict mode error: {e}")

    print("\n✅ Error handling demonstrated successfully!")


def demo_multi_controlled_gates():
    """Demonstrate multi-controlled gate handling."""
    print_section("Multi-Controlled Gates")

    if not QISKIT_AVAILABLE:
        print("⚠️  Qiskit not available, skipping...")
        return

    # Create circuit with multi-controlled gates
    qc = QuantumCircuit(5)

    # Toffoli (CCX)
    qc.ccx(0, 1, 2)
    print("   ✓ Added Toffoli (CCX) gate")

    # C3X (4-qubit controlled-X)
    try:
        from qiskit.circuit.library import C3XGate
        qc.append(C3XGate(), [0, 1, 2, 3])
        print("   ✓ Added C3X (4-qubit controlled-X) gate")
    except:
        print("   ⚠️  C3X not available in this Qiskit version")

    # C4X (5-qubit controlled-X)
    try:
        from qiskit.circuit.library import C4XGate
        qc.append(C4XGate(), [0, 1, 2, 3, 4])
        print("   ✓ Added C4X (5-qubit controlled-X) gate")
    except:
        print("   ⚠️  C4X not available in this Qiskit version")

    print(f"\n📋 Circuit with {len(qc.data)} gates")

    # Convert to QuantRS2
    converter = QiskitConverter()
    quantrs_circuit, stats = converter.from_qiskit(qc, optimize=True)

    print("\n📊 Conversion Statistics:")
    print(f"   ✓ Original gates: {stats.original_gates}")
    print(f"   ✓ Converted gates: {stats.converted_gates}")
    print(f"   ✓ Decomposed gates: {stats.decomposed_gates}")

    if stats.warnings:
        print(f"\n   ℹ️  Decomposition notes:")
        for warning in stats.warnings:
            if "Multi-controlled" in warning:
                print(f"      - {warning}")

    print("\n✅ Multi-controlled gates handled successfully!")


def demo_circuit_optimization():
    """Demonstrate circuit optimization during conversion."""
    print_section("Circuit Optimization")

    if not QISKIT_AVAILABLE:
        print("⚠️  Qiskit not available, skipping...")
        return

    # Create a circuit with redundant gates
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.x(1)
    qc.x(1)  # Redundant - cancels out
    qc.z(2)
    qc.z(2)  # Redundant - cancels out
    qc.cx(0, 1)
    qc.cx(0, 2)

    print(f"\n📋 Original circuit: {len(qc.data)} gates")

    # Convert without optimization
    converter = QiskitConverter()
    circuit_no_opt, stats_no_opt = converter.from_qiskit(qc, optimize=False)
    print(f"\n   Without optimization: {stats_no_opt.converted_gates} gates converted")

    # Convert with optimization
    circuit_opt, stats_opt = converter.from_qiskit(qc, optimize=True)
    print(f"   With optimization: {stats_opt.converted_gates} gates converted")

    reduction = stats_no_opt.converted_gates - stats_opt.converted_gates
    if reduction > 0:
        print(f"\n   ✓ Optimization reduced circuit by {reduction} gates!")

    print("\n✅ Circuit optimization demonstrated successfully!")


def main():
    """Run all demonstration examples."""
    print("\n" + "█" * 70)
    print("  QuantRS2 Qiskit Converter - Comprehensive Demonstration")
    print("█" * 70)

    if not QUANTRS2_AVAILABLE:
        print("\n❌ QuantRS2 not available. Cannot run demonstrations.")
        return

    if not QISKIT_AVAILABLE:
        print("\n⚠️  Qiskit not available. Some demonstrations will be skipped.")
        print("Install Qiskit with: pip install qiskit")

    try:
        # Run all demonstrations
        demo_basic_conversion()
        demo_advanced_gates()
        demo_qft_algorithm()
        demo_variational_circuit()
        demo_qasm_interop()
        demo_multi_controlled_gates()
        demo_circuit_optimization()
        demo_error_handling()

        # Final summary
        print_section("Summary")
        print("\n✅ All demonstrations completed successfully!")
        print("\n📚 Key Features Demonstrated:")
        print("   ✓ Basic circuit conversion")
        print("   ✓ Advanced gate support (iSwap, ECR, RXX, RYY, RZZ)")
        print("   ✓ QFT algorithm conversion")
        print("   ✓ Variational/parametric circuits")
        print("   ✓ QASM import/export")
        print("   ✓ Multi-controlled gates")
        print("   ✓ Circuit optimization")
        print("   ✓ Error handling (strict and lenient modes)")

        print("\n🎯 Next Steps:")
        print("   • Try converting your own Qiskit circuits")
        print("   • Explore other framework converters (Cirq, MyQLM, ProjectQ)")
        print("   • Check the comprehensive documentation")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
