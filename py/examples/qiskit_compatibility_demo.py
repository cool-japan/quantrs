"""
Qiskit Compatibility Layer Demonstration

This example shows how to use the QuantRS2-Qiskit compatibility layer to:
1. Convert circuits between frameworks
2. Run QuantRS2 circuits on Qiskit backends
3. Use Qiskit algorithms in QuantRS2
4. Create hybrid workflows
"""

import numpy as np
import warnings
from typing import List, Dict, Any

# Import QuantRS2 components
try:
    from quantrs2 import Circuit as QuantRS2Circuit
    from quantrs2.gates import H, X, CNOT
except ImportError:
    print("QuantRS2 not available. Using mock implementation.")
    from quantrs2.qiskit_compatibility import QuantRS2Circuit

# Import compatibility layer
from quantrs2.qiskit_compatibility import (
    CircuitConverter,
    QiskitBackendAdapter,
    QiskitAlgorithmLibrary,
    from_qiskit,
    to_qiskit,
    run_on_qiskit_backend,
    create_qiskit_compatible_vqe,
    test_conversion_fidelity,
    benchmark_conversion_performance,
    QISKIT_AVAILABLE
)


def demo_basic_conversion():
    """Demonstrate basic circuit conversion between frameworks."""
    print("=" * 60)
    print("BASIC CIRCUIT CONVERSION DEMO")
    print("=" * 60)
    
    # Create a QuantRS2 circuit
    print("1. Creating QuantRS2 Bell state circuit...")
    quantrs2_circuit = QuantRS2Circuit(2)
    quantrs2_circuit.h(0)
    quantrs2_circuit.cnot(0, 1)
    print(f"   QuantRS2 circuit: {quantrs2_circuit.n_qubits} qubits")
    
    # Convert to Qiskit
    print("\n2. Converting to Qiskit...")
    try:
        qiskit_circuit = to_qiskit(quantrs2_circuit)
        print(f"   Qiskit circuit created successfully")
        if hasattr(qiskit_circuit, 'num_qubits'):
            print(f"   Qiskit circuit: {qiskit_circuit.num_qubits} qubits")
    except Exception as e:
        print(f"   Conversion failed: {e}")
        return
    
    # Convert back to QuantRS2
    print("\n3. Converting back to QuantRS2...")
    try:
        recovered_circuit = from_qiskit(qiskit_circuit)
        print(f"   Recovered circuit: {recovered_circuit.n_qubits} qubits")
        print("   Round-trip conversion successful!")
    except Exception as e:
        print(f"   Recovery failed: {e}")
    
    # Test conversion fidelity
    print("\n4. Testing conversion fidelity...")
    fidelity_passed = test_conversion_fidelity(quantrs2_circuit)
    print(f"   Fidelity test: {'PASSED' if fidelity_passed else 'FAILED'}")


def demo_qiskit_backend_execution():
    """Demonstrate running QuantRS2 circuits on Qiskit backends."""
    print("\n" + "=" * 60)
    print("QISKIT BACKEND EXECUTION DEMO")
    print("=" * 60)
    
    # Create a quantum circuit
    print("1. Creating quantum circuit for backend execution...")
    circuit = QuantRS2Circuit(3)
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.cnot(1, 2)  # GHZ state
    print("   Created 3-qubit GHZ state circuit")
    
    # Execute on default backend
    print("\n2. Executing on default backend...")
    try:
        result = run_on_qiskit_backend(circuit, shots=1024)
        print(f"   Execution successful: {result['success']}")
        print(f"   Backend: {result['backend']}")
        print(f"   Shots: {result['shots']}")
        print(f"   Results: {result['counts']}")
    except Exception as e:
        print(f"   Execution failed: {e}")
    
    # Create backend adapter for more control
    print("\n3. Using backend adapter...")
    try:
        adapter = QiskitBackendAdapter()
        result = adapter.execute(circuit, shots=2048)
        print(f"   Adapter execution: {result['success']}")
        print(f"   Measurement counts: {len(result['counts'])} outcomes")
        
        # Analyze results
        total_shots = sum(result['counts'].values())
        print(f"   Total measurements: {total_shots}")
        for bitstring, count in sorted(result['counts'].items()):
            probability = count / total_shots
            print(f"   |{bitstring}⟩: {count} ({probability:.3f})")
            
    except Exception as e:
        print(f"   Adapter execution failed: {e}")


def demo_algorithm_library():
    """Demonstrate using the Qiskit algorithm library."""
    print("\n" + "=" * 60)
    print("QISKIT ALGORITHM LIBRARY DEMO")
    print("=" * 60)
    
    # Initialize algorithm library
    library = QiskitAlgorithmLibrary()
    print("1. Initialized Qiskit algorithm library")
    
    # Create Bell state
    print("\n2. Creating Bell state...")
    bell_state = library.create_bell_state()
    print(f"   Bell state circuit: {bell_state.n_qubits} qubits")
    
    # Execute Bell state
    try:
        bell_result = run_on_qiskit_backend(bell_state, shots=1000)
        print(f"   Bell state execution: {bell_result['success']}")
        print(f"   Bell state counts: {bell_result['counts']}")
    except Exception as e:
        print(f"   Bell state execution failed: {e}")
    
    # Create Grover oracle
    print("\n3. Creating Grover oracle...")
    grover_oracle = library.create_grover_oracle(n_qubits=3, marked_items=[5])
    print(f"   Grover oracle: {grover_oracle.n_qubits} qubits, marking item 5")
    
    # Create QFT circuit
    print("\n4. Creating Quantum Fourier Transform...")
    qft_circuit = library.create_qft(n_qubits=3)
    print(f"   QFT circuit: {qft_circuit.n_qubits} qubits")
    
    # Test all algorithms
    algorithms = {
        "Bell State": bell_state,
        "Grover Oracle": grover_oracle,
        "QFT": qft_circuit
    }
    
    print("\n5. Testing all algorithm circuits...")
    for name, circuit in algorithms.items():
        try:
            result = run_on_qiskit_backend(circuit, shots=500)
            print(f"   {name}: {'SUCCESS' if result['success'] else 'FAILED'}")
        except Exception as e:
            print(f"   {name}: FAILED - {e}")


def demo_vqe_hybrid_algorithm():
    """Demonstrate VQE hybrid quantum-classical algorithm."""
    print("\n" + "=" * 60)
    print("VQE HYBRID ALGORITHM DEMO")
    print("=" * 60)
    
    # Create VQE instance
    print("1. Creating Qiskit-compatible VQE...")
    vqe = create_qiskit_compatible_vqe(hamiltonian="H2", ansatz_depth=2)
    print(f"   VQE ansatz depth: {vqe.ansatz_depth}")
    
    # Test ansatz creation
    print("\n2. Testing parameterized ansatz...")
    test_parameters = [0.5, 1.0, 1.5, 2.0]
    ansatz_circuit = vqe.create_ansatz(test_parameters)
    print(f"   Ansatz circuit: {ansatz_circuit.n_qubits} qubits")
    print(f"   Parameters used: {test_parameters}")
    
    # Run VQE optimization
    print("\n3. Running VQE optimization...")
    try:
        optimization_result = vqe.optimize()
        print(f"   Optimization converged: {optimization_result['converged']}")
        print(f"   Optimal energy: {optimization_result['optimal_energy']:.6f}")
        print(f"   Iterations: {optimization_result['iterations']}")
        print(f"   Optimal parameters: {[f'{p:.3f}' for p in optimization_result['optimal_parameters']]}")
    except Exception as e:
        print(f"   VQE optimization failed: {e}")
    
    # Test with different ansatz depths
    print("\n4. Comparing different ansatz depths...")
    for depth in [1, 2, 3]:
        try:
            vqe_test = create_qiskit_compatible_vqe("H2", ansatz_depth=depth)
            result = vqe_test.optimize()
            print(f"   Depth {depth}: Energy = {result['optimal_energy']:.6f}, "
                  f"Converged = {result['converged']}")
        except Exception as e:
            print(f"   Depth {depth}: Failed - {e}")


def demo_performance_analysis():
    """Demonstrate performance analysis and benchmarking."""
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS DEMO")
    print("=" * 60)
    
    # Benchmark conversion performance
    print("1. Benchmarking conversion performance...")
    try:
        benchmark_results = benchmark_conversion_performance()
        print("   Conversion performance results:")
        for n_qubits, metrics in benchmark_results.items():
            print(f"   {n_qubits} qubits: {metrics['conversion_time']:.6f} seconds")
    except Exception as e:
        print(f"   Benchmarking failed: {e}")
    
    # Test scaling behavior
    print("\n2. Testing scaling behavior...")
    scaling_results = {}
    for n_qubits in [2, 3, 4, 5]:
        try:
            # Create test circuit
            circuit = QuantRS2Circuit(n_qubits)
            for i in range(n_qubits):
                circuit.h(i)
            for i in range(n_qubits - 1):
                circuit.cnot(i, i + 1)
            
            # Time execution
            import time
            start_time = time.time()
            result = run_on_qiskit_backend(circuit, shots=100)
            execution_time = time.time() - start_time
            
            scaling_results[n_qubits] = {
                'execution_time': execution_time,
                'success': result['success']
            }
            
        except Exception as e:
            scaling_results[n_qubits] = {'error': str(e)}
    
    print("   Scaling test results:")
    for n_qubits, result in scaling_results.items():
        if 'error' in result:
            print(f"   {n_qubits} qubits: ERROR - {result['error']}")
        else:
            print(f"   {n_qubits} qubits: {result['execution_time']:.6f}s "
                  f"({'SUCCESS' if result['success'] else 'FAILED'})")


def demo_advanced_features():
    """Demonstrate advanced compatibility features."""
    print("\n" + "=" * 60)
    print("ADVANCED FEATURES DEMO")
    print("=" * 60)
    
    # Test parametric gates
    print("1. Testing parametric gate conversion...")
    try:
        circuit = QuantRS2Circuit(2)
        circuit.ry(0, np.pi/4)
        circuit.rz(1, np.pi/2)
        circuit.cnot(0, 1)
        
        qiskit_circuit = to_qiskit(circuit)
        print("   Parametric gates converted successfully")
        
        # Execute parametric circuit
        result = run_on_qiskit_backend(circuit, shots=1000)
        print(f"   Parametric circuit execution: {result['success']}")
        
    except Exception as e:
        print(f"   Parametric gate test failed: {e}")
    
    # Test multi-controlled gates
    print("\n2. Testing multi-controlled gates...")
    try:
        library = QiskitAlgorithmLibrary()
        grover_oracle = library.create_grover_oracle(4, [7, 11])
        print(f"   Multi-controlled oracle: {grover_oracle.n_qubits} qubits")
        
        result = run_on_qiskit_backend(grover_oracle, shots=500)
        print(f"   Multi-controlled execution: {result['success']}")
        
    except Exception as e:
        print(f"   Multi-controlled test failed: {e}")
    
    # Test error handling
    print("\n3. Testing error handling...")
    try:
        # Test with invalid circuit
        invalid_circuit = QuantRS2Circuit(0)  # Zero qubits
        result = to_qiskit(invalid_circuit)
        print("   Zero-qubit circuit handled gracefully")
    except Exception as e:
        print(f"   Zero-qubit circuit error: {e}")
    
    # Test framework availability detection
    print("\n4. Framework availability status...")
    print(f"   Qiskit available: {QISKIT_AVAILABLE}")
    print(f"   QuantRS2 core available: {'quantrs2' in globals()}")


def demo_comparison_workflows():
    """Demonstrate comparison between native and converted circuits."""
    print("\n" + "=" * 60)
    print("COMPARISON WORKFLOWS DEMO")
    print("=" * 60)
    
    # Create same algorithm in both frameworks
    print("1. Creating identical algorithms in both frameworks...")
    
    # QuantRS2 native implementation
    quantrs2_bell = QuantRS2Circuit(2)
    quantrs2_bell.h(0)
    quantrs2_bell.cnot(0, 1)
    
    # Qiskit library implementation
    library = QiskitAlgorithmLibrary()
    qiskit_bell = library.create_bell_state()
    
    print("   Created Bell states in both frameworks")
    
    # Compare execution results
    print("\n2. Comparing execution results...")
    try:
        quantrs2_result = run_on_qiskit_backend(quantrs2_bell, shots=1000)
        qiskit_result = run_on_qiskit_backend(qiskit_bell, shots=1000)
        
        print("   QuantRS2 native Bell state:")
        for state, count in quantrs2_result['counts'].items():
            prob = count / quantrs2_result['shots']
            print(f"     |{state}⟩: {count} ({prob:.3f})")
        
        print("   Qiskit library Bell state:")
        for state, count in qiskit_result['counts'].items():
            prob = count / qiskit_result['shots']
            print(f"     |{state}⟩: {count} ({prob:.3f})")
            
    except Exception as e:
        print(f"   Comparison failed: {e}")
    
    # Performance comparison
    print("\n3. Performance comparison...")
    import time
    
    circuits_to_test = [
        ("QuantRS2 Native", quantrs2_bell),
        ("Qiskit Library", qiskit_bell)
    ]
    
    for name, circuit in circuits_to_test:
        try:
            start_time = time.time()
            for _ in range(10):  # Run multiple times
                result = run_on_qiskit_backend(circuit, shots=100)
            avg_time = (time.time() - start_time) / 10
            
            print(f"   {name}: {avg_time:.6f}s average")
            
        except Exception as e:
            print(f"   {name}: Failed - {e}")


def main():
    """Run all compatibility layer demonstrations."""
    print("QuantRS2-Qiskit Compatibility Layer Demonstration")
    print("=" * 60)
    
    # Check framework availability
    print(f"Qiskit available: {QISKIT_AVAILABLE}")
    
    try:
        from quantrs2 import Circuit
        print("QuantRS2 core available: True")
    except ImportError:
        print("QuantRS2 core available: False (using mock)")
    
    print("\nRunning compatibility demonstrations...")
    
    # Run demonstrations
    demos = [
        ("Basic Conversion", demo_basic_conversion),
        ("Backend Execution", demo_qiskit_backend_execution),
        ("Algorithm Library", demo_algorithm_library),
        ("VQE Hybrid Algorithm", demo_vqe_hybrid_algorithm),
        ("Performance Analysis", demo_performance_analysis),
        ("Advanced Features", demo_advanced_features),
        ("Comparison Workflows", demo_comparison_workflows)
    ]
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*20} {demo_name} {'='*20}")
            demo_func()
        except Exception as e:
            print(f"\n{demo_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey takeaways:")
    print("1. QuantRS2 and Qiskit circuits can be converted seamlessly")
    print("2. QuantRS2 circuits can run on any Qiskit backend") 
    print("3. Qiskit algorithms can be used in QuantRS2 workflows")
    print("4. Hybrid algorithms work with both frameworks")
    print("5. Performance is comparable between native and converted circuits")
    
    print("\nNext steps:")
    print("- Try running your own circuits through the conversion")
    print("- Experiment with different Qiskit backends")
    print("- Use Qiskit's optimization passes on QuantRS2 circuits")
    print("- Combine QuantRS2's performance with Qiskit's ecosystem")


if __name__ == "__main__":
    main()