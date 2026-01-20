#!/usr/bin/env python3
"""
OpenQASM 3.0 Import/Export Demo

This example demonstrates how to use QuantRS2's OpenQASM 3.0 support
to import and export quantum circuits for interoperability with other
quantum computing frameworks.
"""

import numpy as np
import tempfile
import os
from pathlib import Path

try:
    import quantrs2
    from quantrs2 import Circuit
    from quantrs2.qasm import (
        parse_qasm, export_qasm, validate_qasm,
        QasmExportOptions, QasmParser, QasmExporter, QasmValidator
    )
    HAS_QUANTRS2 = True
except ImportError:
    print("QuantRS2 not available. Please install QuantRS2 first.")
    HAS_QUANTRS2 = False
    exit(1)


def demo_basic_export():
    """Demonstrate basic circuit export to QASM."""
    print("="*60)
    print("Basic Circuit Export to OpenQASM 3.0")
    print("="*60)
    
    # Create a simple Bell state circuit
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    
    print("Created Bell state circuit:")
    print(f"  Circuit has {circuit.n_qubits if hasattr(circuit, 'n_qubits') else 'unknown'} qubits")
    
    # Export to QASM with default options
    qasm_code = export_qasm(circuit)
    
    print("\nExported QASM code:")
    print("-" * 40)
    print(qasm_code)
    print("-" * 40)


def demo_custom_export_options():
    """Demonstrate export with custom options."""
    print("\n" + "="*60)
    print("Custom Export Options Demo")
    print("="*60)
    
    # Create a more complex circuit
    circuit = Circuit(3)
    circuit.h(0)
    circuit.rx(0, np.pi/4)
    circuit.ry(1, np.pi/3)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.rz(2, np.pi/6)
    
    # Create custom export options
    options = QasmExportOptions(
        include_stdgates=True,
        decompose_custom=False,
        include_gate_comments=True,
        optimize=True,
        pretty_print=True
    )
    
    print("Circuit with rotation gates:")
    print(f"  RX(π/4) on qubit 0")
    print(f"  RY(π/3) on qubit 1") 
    print(f"  RZ(π/6) on qubit 2")
    print(f"  Plus Hadamard and CNOT gates")
    
    # Export with custom options
    qasm_code = export_qasm(circuit, options)
    
    print("\nExported QASM with custom options:")
    print("-" * 40)
    print(qasm_code)
    print("-" * 40)


def demo_parse_qasm():
    """Demonstrate parsing QASM code into circuits."""
    print("\n" + "="*60)
    print("Parsing OpenQASM 3.0 Code")
    print("="*60)
    
    # Example QASM code for a GHZ state
    ghz_qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    
    qubit[3] q;
    bit[3] c;
    
    // Create GHZ state
    h q[0];
    cx q[0], q[1];
    cx q[1], q[2];
    
    // Measure all qubits
    c = measure q;
    """
    
    print("Parsing GHZ state QASM:")
    print("-" * 40)
    print(ghz_qasm.strip())
    print("-" * 40)
    
    # Parse the QASM code
    try:
        circuit = parse_qasm(ghz_qasm)
        print(f"\nSuccessfully parsed circuit!")
        print(f"  Circuit type: {type(circuit).__name__}")
        
        # Try to get circuit info
        if hasattr(circuit, 'n_qubits'):
            print(f"  Number of qubits: {circuit.n_qubits}")
        
    except Exception as e:
        print(f"\nParsing failed (using fallback): {e}")


def demo_validation():
    """Demonstrate QASM code validation."""
    print("\n" + "="*60)
    print("QASM Code Validation")
    print("="*60)
    
    # Valid QASM code
    valid_qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    
    qubit[2] q;
    bit[2] c;
    
    h q[0];
    cx q[0], q[1];
    c = measure q;
    """
    
    # Invalid QASM code
    invalid_qasm = """
    OPENQASM 2.0;
    
    qreg q[2];
    creg c[2];
    
    h q[0];
    invalid_gate q[1];
    measure q -> c;
    """
    
    print("Validating valid QASM code...")
    result = validate_qasm(valid_qasm)
    print(f"  Valid: {result['is_valid']}")
    print(f"  Errors: {len(result['errors'])}")
    print(f"  Warnings: {len(result['warnings'])}")
    if result['info']:
        print(f"  Circuit info: {result['info']}")
    
    print("\nValidating invalid QASM code...")
    result = validate_qasm(invalid_qasm)
    print(f"  Valid: {result['is_valid']}")
    print(f"  Errors: {len(result['errors'])}")
    if result['errors']:
        print(f"  First error: {result['errors'][0]}")


def demo_file_operations():
    """Demonstrate reading/writing QASM files."""
    print("\n" + "="*60)
    print("File Import/Export Operations")
    print("="*60)
    
    # Create a quantum Fourier transform circuit
    circuit = Circuit(3)
    
    # Simple QFT approximation
    circuit.h(0)
    circuit.h(1)
    circuit.h(2)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    
    # Create temporary file for demonstration
    with tempfile.NamedTemporaryFile(mode='w', suffix='.qasm', delete=False) as f:
        temp_file = f.name
    
    try:
        # Export circuit to file
        print("Exporting circuit to file...")
        exporter = QasmExporter()
        exporter.export_to_file(circuit, temp_file)
        print(f"  Exported to: {temp_file}")
        
        # Read the file content
        with open(temp_file, 'r') as f:
            content = f.read()
        
        print("\nFile content:")
        print("-" * 40)
        print(content)
        print("-" * 40)
        
        # Parse the file back
        print("\nParsing file back to circuit...")
        parser = QasmParser()
        reconstructed = parser.parse_file(temp_file)
        print(f"  Successfully reconstructed circuit: {type(reconstructed).__name__}")
        
    except Exception as e:
        print(f"File operations failed: {e}")
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def demo_round_trip():
    """Demonstrate round-trip conversion."""
    print("\n" + "="*60)
    print("Round-Trip Conversion Demo")
    print("="*60)
    
    # Create original circuit
    print("Creating original circuit...")
    original = Circuit(2)
    original.h(0)
    original.ry(1, np.pi/4)
    original.cx(0, 1)
    original.rz(0, np.pi/8)
    
    print("  H gate on qubit 0")
    print("  RY(π/4) gate on qubit 1") 
    print("  CNOT from qubit 0 to 1")
    print("  RZ(π/8) gate on qubit 0")
    
    # Convert to QASM
    print("\nConverting to QASM...")
    qasm_code = export_qasm(original)
    
    # Convert back to circuit
    print("Converting back to circuit...")
    reconstructed = parse_qasm(qasm_code)
    
    print("Round-trip conversion completed!")
    print(f"  Original type: {type(original).__name__}")
    print(f"  Reconstructed type: {type(reconstructed).__name__}")
    
    # Show the QASM code
    print("\nIntermediate QASM representation:")
    print("-" * 40)
    print(qasm_code)
    print("-" * 40)


def demo_advanced_features():
    """Demonstrate advanced QASM features."""
    print("\n" + "="*60)
    print("Advanced QASM Features")
    print("="*60)
    
    # Example with different export options
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    
    print("Testing different export options:")
    
    # Compact format
    compact_options = QasmExportOptions(
        pretty_print=False,
        include_gate_comments=False
    )
    compact_qasm = export_qasm(circuit, compact_options)
    print("\n1. Compact format:")
    print(compact_qasm[:200] + "..." if len(compact_qasm) > 200 else compact_qasm)
    
    # Verbose format with comments
    verbose_options = QasmExportOptions(
        pretty_print=True,
        include_gate_comments=True,
        include_stdgates=True
    )
    verbose_qasm = export_qasm(circuit, verbose_options)
    print("\n2. Verbose format with comments:")
    print(verbose_qasm[:300] + "..." if len(verbose_qasm) > 300 else verbose_qasm)


def main():
    """Run all QASM demos."""
    print("QuantRS2 OpenQASM 3.0 Support Demo")
    print("This demo shows import/export capabilities for interoperability")
    
    try:
        # Run all demos
        demo_basic_export()
        demo_custom_export_options()
        demo_parse_qasm()
        demo_validation()
        demo_file_operations()
        demo_round_trip()
        demo_advanced_features()
        
        print("\n" + "="*60)
        print("Demo Complete!")
        print("="*60)
        print("\nKey takeaways:")
        print("• QuantRS2 supports OpenQASM 3.0 import and export")
        print("• Customizable export options for different use cases")
        print("• Validation ensures QASM code correctness")
        print("• Round-trip conversion preserves circuit structure")
        print("• File I/O enables easy integration with other tools")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("This may be due to missing native QASM support.")
        print("The fallback implementations provide basic functionality.")


if __name__ == "__main__":
    if HAS_QUANTRS2:
        main()
    else:
        print("Please install QuantRS2 to run this demo.")