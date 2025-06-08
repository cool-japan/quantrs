#!/usr/bin/env python3
"""
Demonstration of quantum circuit optimization passes.

This example shows how to:
1. Apply various optimization passes to quantum circuits
2. Analyze circuit depth and gate counts
3. Route circuits to device topology
4. Transpile circuits to different basis gate sets
"""

from quantrs2 import CircuitTranspiler, DeviceRouter

def basic_optimization_example():
    """Basic circuit optimization example."""
    print("=== Basic Optimization Example ===")
    
    # Create a circuit with redundant gates
    circuit = [
        ("h", [0], None),
        ("h", [0], None),  # H^2 = I, should be cancelled
        ("rx", [1], [0.5]),
        ("rx", [1], [0.7]),  # Should be merged into single RX(1.2)
        ("cnot", [0, 1], None),
        ("cnot", [0, 1], None),  # CNOT^2 = I, should be cancelled
        ("s", [2], None),
        ("sdg", [2], None),  # S * S† = I, should be cancelled
    ]
    
    # Create transpiler with optimization level 1
    transpiler = CircuitTranspiler(optimization_level=1)
    
    print("Original circuit:")
    for gate in circuit:
        print(f"  {gate}")
    
    # Optimize the circuit
    optimized_circuit, stats = transpiler.optimize(circuit)
    
    print("\nOptimized circuit:")
    for gate in optimized_circuit:
        print(f"  {gate}")
    
    print("\nOptimization statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def commutation_example():
    """Example of gate commutation optimization."""
    print("\n=== Commutation Example ===")
    
    # Circuit where single-qubit gates can be moved before two-qubit gates
    circuit = [
        ("cnot", [0, 1], None),
        ("h", [2], None),  # Acts on different qubit, can be moved earlier
        ("cnot", [1, 2], None),
        ("z", [0], None),  # Can be moved earlier
        ("cnot", [0, 2], None),
    ]
    
    transpiler = CircuitTranspiler(optimization_level=2)
    
    print("Original circuit:")
    for i, gate in enumerate(circuit):
        print(f"  {i}: {gate}")
    
    optimized_circuit, stats = transpiler.optimize(circuit)
    
    print("\nOptimized circuit (after commutation):")
    for i, gate in enumerate(optimized_circuit):
        print(f"  {i}: {gate}")
    
    print(f"\nGates commuted: {stats.get('gates_commuted', 0)}")


def decomposition_example():
    """Example of decomposing complex gates."""
    print("\n=== Decomposition Example ===")
    
    # Circuit with Toffoli and SWAP gates
    circuit = [
        ("h", [0], None),
        ("toffoli", [0, 1, 2], None),
        ("swap", [1, 2], None),
    ]
    
    # Create transpiler that decomposes to basic gates
    transpiler = CircuitTranspiler(
        optimization_level=3,
        basis_gates=["h", "t", "tdg", "cnot", "rx", "ry", "rz"]
    )
    
    print("Original circuit:")
    for gate in circuit:
        print(f"  {gate}")
    
    # Analyze depth before optimization
    original_depth = transpiler.analyze_depth(circuit)
    print(f"\nOriginal depth: {original_depth}")
    
    # Optimize and decompose
    optimized_circuit, stats = transpiler.optimize(circuit)
    
    print("\nDecomposed circuit:")
    for gate in optimized_circuit:
        print(f"  {gate}")
    
    # Analyze depth after optimization
    optimized_depth = transpiler.analyze_depth(optimized_circuit)
    print(f"\nOptimized depth: {optimized_depth}")
    print(f"Gates decomposed: {stats.get('gates_decomposed', 0)}")


def gate_count_analysis():
    """Analyze gate counts in a circuit."""
    print("\n=== Gate Count Analysis ===")
    
    circuit = [
        ("h", [i], None) for i in range(4)
    ] + [
        ("cnot", [i, i+1], None) for i in range(3)
    ] + [
        ("rz", [i], [0.5]) for i in range(4)
    ] + [
        ("cnot", [i, i+1], None) for i in range(3)
    ] + [
        ("h", [i], None) for i in range(4)
    ]
    
    transpiler = CircuitTranspiler()
    
    # Count gates before optimization
    gate_counts_before = transpiler.gate_counts(circuit)
    print("Gate counts before optimization:")
    for gate, count in gate_counts_before.items():
        print(f"  {gate}: {count}")
    
    # Optimize
    optimized_circuit, _ = transpiler.optimize(circuit)
    
    # Count gates after optimization
    gate_counts_after = transpiler.gate_counts(optimized_circuit)
    print("\nGate counts after optimization:")
    for gate, count in gate_counts_after.items():
        print(f"  {gate}: {count}")


def device_routing_example():
    """Example of routing to device topology."""
    print("\n=== Device Routing Example ===")
    
    # Define a simple linear coupling map (0-1-2-3)
    coupling_map = [(0, 1), (1, 2), (2, 3)]
    router = DeviceRouter(coupling_map)
    
    # Circuit that requires routing
    circuit = [
        ("cnot", [0, 2], None),  # Not directly connected
        ("cnot", [1, 3], None),  # Not directly connected
        ("cnot", [0, 3], None),  # Not directly connected
    ]
    
    print("Coupling map (device topology):")
    print("  0 -- 1 -- 2 -- 3")
    print("\nLogical circuit (requires routing):")
    for gate in circuit:
        print(f"  {gate}")
    
    # Try to route (this simple example will fail without SWAP insertion)
    try:
        routed_circuit, layout = router.route(circuit)
        print("\nRouted circuit:")
        for gate in routed_circuit:
            print(f"  {gate}")
        print(f"\nQubit layout: {layout}")
    except ValueError as e:
        print(f"\nRouting failed: {e}")
        print("(A more sophisticated router would insert SWAP gates)")


def full_transpilation_example():
    """Complete transpilation pipeline example."""
    print("\n=== Full Transpilation Example ===")
    
    # Complex circuit
    circuit = [
        ("h", [0], None),
        ("h", [0], None),
        ("ry", [1], [0.5]),
        ("ry", [1], [0.5]),
        ("toffoli", [0, 1, 2], None),
        ("swap", [2, 3], None),
        ("cnot", [1, 2], None),
        ("cnot", [1, 2], None),
        ("s", [0], None),
        ("sdg", [0], None),
    ]
    
    # Create transpiler with all optimizations
    transpiler = CircuitTranspiler(
        optimization_level=3,
        basis_gates=["u3", "cnot"]  # IBM-style basis
    )
    
    print("Original circuit:")
    for gate in circuit:
        print(f"  {gate}")
    
    # Full analysis
    print("\nCircuit analysis:")
    print(f"  Depth: {transpiler.analyze_depth(circuit)}")
    print(f"  Gate counts: {transpiler.gate_counts(circuit)}")
    
    # Optimize
    optimized_circuit, stats = transpiler.optimize(circuit)
    
    print("\nOptimized circuit:")
    for gate in optimized_circuit:
        print(f"  {gate}")
    
    print("\nOptimization summary:")
    print(f"  Original gates: {len(circuit)}")
    print(f"  Optimized gates: {len(optimized_circuit)}")
    print(f"  Optimization stats: {stats}")


if __name__ == "__main__":
    basic_optimization_example()
    commutation_example()
    decomposition_example()
    gate_count_analysis()
    device_routing_example()
    full_transpilation_example()
    
    print("\n=== Circuit Optimization Demo Complete ===")