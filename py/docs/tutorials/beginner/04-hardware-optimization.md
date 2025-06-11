# Tutorial 4: Quantum Hardware Optimization

**Estimated time:** 40 minutes  
**Prerequisites:** [Tutorial 3: Quantum Algorithms](03-quantum-algorithms.md)  
**Goal:** Learn to optimize quantum circuits for real hardware constraints and NISQ devices

Real quantum computers aren't perfect! In this tutorial, you'll learn how to adapt your quantum algorithms for the constraints and noise of actual quantum hardware.

## Understanding Real Quantum Hardware

### The Reality Gap

**Ideal quantum computer:**
- Perfect gates with infinite precision
- No noise or decoherence 
- All-to-all connectivity
- Unlimited coherence time

**Real quantum computer:**
- Noisy gates with finite fidelity
- Decoherence and environmental interference
- Limited qubit connectivity
- Short coherence times (~100 microseconds)

Let's bridge this gap!

### NISQ Device Characteristics

**NISQ** = Noisy Intermediate-Scale Quantum

```python
import quantrs2
import numpy as np

def nisq_device_simulation():
    """Simulate characteristics of real NISQ devices."""
    
    print("🔧 NISQ Device Characteristics")
    print("=" * 35)
    
    # Typical NISQ device parameters
    device_specs = {
        "num_qubits": 65,
        "gate_fidelity": 0.999,        # 99.9% single-qubit gate fidelity
        "two_qubit_fidelity": 0.995,   # 99.5% two-qubit gate fidelity
        "readout_fidelity": 0.98,      # 98% measurement fidelity
        "t1_time": 100e-6,             # 100 μs relaxation time
        "t2_time": 80e-6,              # 80 μs dephasing time
        "gate_time": 50e-9,            # 50 ns gate time
        "connectivity": "limited"       # Not all-to-all connected
    }
    
    print("Typical NISQ Device Specifications:")
    for param, value in device_specs.items():
        if isinstance(value, float):
            if param.endswith('_time'):
                print(f"  {param:20}: {value*1e6:.1f} μs")
            elif param.endswith('_fidelity'):
                print(f"  {param:20}: {value:.3f}")
            else:
                print(f"  {param:20}: {value}")
        else:
            print(f"  {param:20}: {value}")
    
    print(f"\nCoherence Budget:")
    max_gates = int(device_specs["t2_time"] / device_specs["gate_time"])
    print(f"  Maximum gates before decoherence: ~{max_gates:,}")
    print(f"  Practical limit (with margins): ~{max_gates//10:,}")
    
    return device_specs

device_specs = nisq_device_simulation()
print()
```

## Hardware Constraint 1: Limited Connectivity

### Understanding Qubit Topology

Real quantum devices have limited connectivity - not every qubit can directly interact with every other qubit.

```python
def connectivity_constraints():
    """Demonstrate how limited connectivity affects circuits."""
    
    print("🔗 Qubit Connectivity Constraints")
    print("=" * 40)
    
    # Simulate different topologies
    topologies = {
        "Linear": [(0,1), (1,2), (2,3), (3,4)],
        "Grid": [(0,1), (1,2), (0,3), (1,4), (3,4)],
        "Star": [(0,1), (0,2), (0,3), (0,4)],
        "All-to-all": [(i,j) for i in range(5) for j in range(i+1,5)]
    }
    
    for name, edges in topologies.items():
        print(f"\n{name} Topology:")
        print(f"  Connections: {edges}")
        print(f"  Direct connections: {len(edges)}")
        
        # Check if we can do CNOT(0,4) directly
        if (0,4) in edges or (4,0) in edges:
            print(f"  CNOT(0,4): ✅ Direct")
        else:
            print(f"  CNOT(0,4): ❌ Needs SWAP gates")
    
    return topologies

def routing_example():
    """Show how to route gates on limited connectivity."""
    
    print("\n📍 Gate Routing Example")
    print("=" * 30)
    
    print("Problem: Want CNOT(0,4) on linear topology")
    print("Topology: 0—1—2—3—4")
    
    # Original circuit (doesn't work on linear topology)
    print("\nOriginal circuit:")
    circuit_ideal = quantrs2.Circuit(5)
    circuit_ideal.h(0)
    circuit_ideal.cx(0, 4)  # This needs routing!
    print("  H(0)")
    print("  CNOT(0,4)  ← Not directly connected!")
    
    # Routed circuit for linear topology
    print("\nRouted circuit:")
    circuit_routed = quantrs2.Circuit(5)
    circuit_routed.h(0)
    
    # Route: Move qubit 0's state to qubit 4
    print("  H(0)")
    print("  SWAP(0,1)")
    circuit_routed.cx(0, 1)
    circuit_routed.cx(1, 0) 
    circuit_routed.cx(0, 1)
    
    print("  SWAP(1,2)")
    circuit_routed.cx(1, 2)
    circuit_routed.cx(2, 1)
    circuit_routed.cx(1, 2)
    
    print("  SWAP(2,3)")
    circuit_routed.cx(2, 3)
    circuit_routed.cx(3, 2)
    circuit_routed.cx(2, 3)
    
    print("  CNOT(3,4)")
    circuit_routed.cx(3, 4)
    
    print(f"\nGate count comparison:")
    print(f"  Ideal circuit: {circuit_ideal.gate_count} gates")
    print(f"  Routed circuit: {circuit_routed.gate_count} gates")
    print(f"  Overhead: {circuit_routed.gate_count - circuit_ideal.gate_count} extra gates")

connectivity_constraints()
routing_example()
```

### Connectivity-Aware Circuit Design

```python
def connectivity_aware_design():
    """Design circuits that match hardware topology."""
    
    print("\n🏗️  Connectivity-Aware Design")
    print("=" * 35)
    
    print("Strategy 1: Use linear entanglement chains")
    
    def create_linear_entanglement(num_qubits):
        """Create entanglement using only nearest-neighbor gates."""
        circuit = quantrs2.Circuit(num_qubits)
        
        # Initialize first qubit in superposition
        circuit.h(0)
        
        # Propagate entanglement along chain
        for i in range(num_qubits - 1):
            circuit.cx(i, i + 1)
        
        return circuit
    
    linear_circuit = create_linear_entanglement(5)
    print(f"  Linear entanglement: {linear_circuit.gate_count} gates")
    print("  Works on any linear topology!")
    
    print("\nStrategy 2: Use circuit blocks")
    
    def modular_circuit_design():
        """Design using small, local circuit blocks."""
        circuit = quantrs2.Circuit(4)
        
        # Block 1: Local 2-qubit operations
        circuit.h(0)
        circuit.h(1)
        circuit.cx(0, 1)
        
        # Block 2: Another local operation
        circuit.h(2)
        circuit.h(3)
        circuit.cx(2, 3)
        
        # Block 3: Connect blocks (if topology allows)
        circuit.cx(1, 2)
        
        return circuit
    
    modular_circuit = modular_circuit_design()
    print(f"  Modular design: {modular_circuit.gate_count} gates")
    print("  Uses small blocks that fit hardware topology")

connectivity_aware_design()
```

## Hardware Constraint 2: Gate Errors and Noise

### Modeling Quantum Noise

```python
def noise_modeling():
    """Understand different types of quantum noise."""
    
    print("\n🔊 Quantum Noise Types")
    print("=" * 25)
    
    noise_types = [
        {
            "name": "Bit flip (X error)",
            "description": "|0⟩ ↔ |1⟩ flip",
            "probability": 0.001,
            "effect": "Computational errors"
        },
        {
            "name": "Phase flip (Z error)", 
            "description": "|+⟩ → |-⟩",
            "probability": 0.001,
            "effect": "Interference errors"
        },
        {
            "name": "Depolarizing noise",
            "description": "Random Pauli errors",
            "probability": 0.002,
            "effect": "General decoherence"
        },
        {
            "name": "Amplitude damping",
            "description": "|1⟩ → |0⟩ decay",
            "probability": 0.01,
            "effect": "Energy relaxation"
        }
    ]
    
    for noise in noise_types:
        print(f"\n{noise['name']}:")
        print(f"  Description: {noise['description']}")
        print(f"  Typical rate: {noise['probability']:.3f} per gate")
        print(f"  Effect: {noise['effect']}")
    
    print(f"\nNoise accumulation in deep circuits:")
    circuit_depths = [10, 50, 100, 500]
    error_rate = 0.001
    
    for depth in circuit_depths:
        total_error = 1 - (1 - error_rate) ** depth
        print(f"  Depth {depth:3d}: {total_error:.3f} error probability")

def noise_resilient_circuits():
    """Design circuits that are more resilient to noise."""
    
    print("\n🛡️  Noise-Resilient Circuit Design")
    print("=" * 40)
    
    print("Strategy 1: Minimize circuit depth")
    
    # Bad: Sequential operations
    def deep_circuit():
        circuit = quantrs2.Circuit(4)
        
        for i in range(4):
            circuit.h(i)
        
        for i in range(3):
            circuit.cx(i, i+1)
        
        for i in range(4):
            circuit.rz(i, np.pi/4)
        
        return circuit
    
    # Good: Parallel operations
    def shallow_circuit():
        circuit = quantrs2.Circuit(4)
        
        # Parallelize single-qubit gates
        for i in range(4):
            circuit.h(i)
            circuit.rz(i, np.pi/4)
        
        # Efficient entanglement
        circuit.cx(0, 1)
        circuit.cx(2, 3)
        circuit.cx(1, 2)
        
        return circuit
    
    deep = deep_circuit()
    shallow = shallow_circuit()
    
    print(f"  Deep circuit depth: {deep.depth}")
    print(f"  Shallow circuit depth: {shallow.depth}")
    print(f"  Noise reduction: ~{deep.depth/shallow.depth:.1f}x")
    
    print("\nStrategy 2: Use native gates")
    print("  Match gates to hardware-native operations")
    print("  Avoid gate decompositions when possible")
    
    print("\nStrategy 3: Add error correction")
    print("  Use quantum error correction codes")
    print("  Implement error mitigation techniques")

noise_modeling()
noise_resilient_circuits()
```

### Error Mitigation Techniques

```python
def error_mitigation_techniques():
    """Implement basic error mitigation methods."""
    
    print("\n🔧 Error Mitigation Techniques")
    print("=" * 35)
    
    print("Technique 1: Zero-Noise Extrapolation")
    
    def zero_noise_extrapolation_demo():
        """Demonstrate zero-noise extrapolation concept."""
        
        # Simulate noisy measurements at different noise levels
        noise_levels = [0.0, 0.01, 0.02, 0.03]
        measured_values = []
        
        # True value (noise-free)
        true_value = 0.8
        
        for noise in noise_levels:
            # Simulate how noise affects measurement
            noisy_value = true_value * (1 - noise) + 0.5 * noise
            measured_values.append(noisy_value)
        
        print("  Noise Level → Measured Value")
        for noise, value in zip(noise_levels, measured_values):
            print(f"  {noise:.3f}      → {value:.3f}")
        
        # Linear extrapolation to zero noise
        # In practice, would use more sophisticated fitting
        slope = (measured_values[1] - measured_values[0]) / (noise_levels[1] - noise_levels[0])
        extrapolated = measured_values[0] - slope * noise_levels[0]
        
        print(f"  Extrapolated → {extrapolated:.3f}")
        print(f"  True value   → {true_value:.3f}")
        print(f"  Error improvement: {abs(measured_values[1] - true_value):.3f} → {abs(extrapolated - true_value):.3f}")
    
    zero_noise_extrapolation_demo()
    
    print("\nTechnique 2: Readout Error Mitigation")
    
    def readout_error_mitigation():
        """Demonstrate readout error correction."""
        
        print("  Calibration: Measure known states")
        
        # Simulate readout confusion matrix
        # Perfect measurement would be identity matrix
        confusion_matrix = np.array([
            [0.97, 0.03],  # |0⟩ measured as 0: 97%, as 1: 3%
            [0.05, 0.95]   # |1⟩ measured as 0: 5%, as 1: 95%
        ])
        
        print("  Confusion Matrix:")
        print("    Prepared → Measured")
        print(f"    |0⟩      → |0⟩: {confusion_matrix[0,0]:.2f}, |1⟩: {confusion_matrix[0,1]:.2f}")
        print(f"    |1⟩      → |0⟩: {confusion_matrix[1,0]:.2f}, |1⟩: {confusion_matrix[1,1]:.2f}")
        
        # Simulate noisy measurement
        true_probs = np.array([0.6, 0.4])  # 60% |0⟩, 40% |1⟩
        measured_probs = confusion_matrix.T @ true_probs
        
        # Correct using inverse matrix
        corrected_probs = np.linalg.inv(confusion_matrix.T) @ measured_probs
        
        print(f"  True probabilities:      {true_probs}")
        print(f"  Measured probabilities:  {measured_probs}")
        print(f"  Corrected probabilities: {corrected_probs}")
    
    readout_error_mitigation()
    
    print("\nTechnique 3: Symmetry Verification")
    
    def symmetry_verification():
        """Use known symmetries to detect errors."""
        
        print("  Example: Bell state should have |01⟩ = |10⟩ = 0")
        
        # Simulate measurements with errors
        measured_probs = {
            '00': 0.48,
            '01': 0.03,  # Should be 0 for perfect Bell state
            '10': 0.02,  # Should be 0 for perfect Bell state  
            '11': 0.47
        }
        
        symmetry_violation = measured_probs['01'] + measured_probs['10']
        
        print(f"  Measured probabilities: {measured_probs}")
        print(f"  Symmetry violation: {symmetry_violation:.3f}")
        
        if symmetry_violation > 0.1:
            print("  ⚠️  High error detected - consider repeating")
        else:
            print("  ✅ Symmetry preserved - good measurement")
    
    symmetry_verification()

error_mitigation_techniques()
```

## Hardware Constraint 3: Limited Coherence Time

### Optimizing for Speed

```python
def coherence_time_optimization():
    """Optimize circuits for limited coherence time."""
    
    print("\n⏱️  Coherence Time Optimization")
    print("=" * 35)
    
    # Typical coherence times
    coherence_times = {
        "T1 (relaxation)": 100e-6,    # 100 μs
        "T2 (dephasing)": 50e-6,      # 50 μs  
        "Gate time": 50e-9,           # 50 ns
        "Readout time": 1e-6          # 1 μs
    }
    
    print("Typical coherence budget:")
    for process, time in coherence_times.items():
        if process == "Gate time":
            max_gates = coherence_times["T2"] / time
            print(f"  {process:20}: {time*1e9:.0f} ns")
            print(f"  {'Max gates in T2':<20}: ~{max_gates:.0f}")
        elif process == "Readout time":
            max_readouts = coherence_times["T2"] / time
            print(f"  {process:20}: {time*1e6:.0f} μs")
            print(f"  {'Max readouts in T2':<20}: ~{max_readouts:.0f}")
        else:
            print(f"  {process:20}: {time*1e6:.0f} μs")
    
    print(f"\nOptimization strategies:")
    
    def gate_count_optimization():
        """Minimize total gate count."""
        
        print("\n1. Gate Count Minimization")
        
        # Inefficient: Many small rotations
        def inefficient_rotation():
            circuit = quantrs2.Circuit(1)
            circuit.rx(0, np.pi/8)
            circuit.ry(0, np.pi/8) 
            circuit.rz(0, np.pi/8)
            circuit.rx(0, np.pi/8)
            return circuit
        
        # Efficient: Single combined rotation
        def efficient_rotation():
            circuit = quantrs2.Circuit(1)
            # Combine all rotations into single U3 gate
            circuit.u3(0, theta=np.pi/4, phi=np.pi/8, lam=np.pi/8)
            return circuit
        
        inefficient = inefficient_rotation()
        efficient = efficient_rotation()
        
        print(f"  Inefficient: {inefficient.gate_count} gates")
        print(f"  Efficient: {efficient.gate_count} gates")
        print(f"  Speedup: {inefficient.gate_count}x faster")
    
    def circuit_depth_optimization():
        """Minimize circuit depth through parallelization."""
        
        print("\n2. Circuit Depth Minimization")
        
        # Sequential (high depth)
        def sequential_circuit():
            circuit = quantrs2.Circuit(4)
            
            # Gates applied one after another
            for i in range(4):
                circuit.h(i)
            for i in range(3):
                circuit.cx(i, i+1)
            for i in range(4):
                circuit.rz(i, np.pi/4)
            
            return circuit
        
        # Parallel (low depth)
        def parallel_circuit():
            circuit = quantrs2.Circuit(4)
            
            # Combine gates that can run simultaneously
            circuit.h(0)
            circuit.h(1) 
            circuit.h(2)
            circuit.h(3)
            
            # Parallel two-qubit gates
            circuit.cx(0, 1)
            circuit.cx(2, 3)
            
            # Final rotations in parallel
            circuit.rz(0, np.pi/4)
            circuit.rz(1, np.pi/4)
            circuit.rz(2, np.pi/4) 
            circuit.rz(3, np.pi/4)
            
            circuit.cx(1, 2)
            
            return circuit
        
        sequential = sequential_circuit()
        parallel = parallel_circuit()
        
        print(f"  Sequential depth: {sequential.depth}")
        print(f"  Parallel depth: {parallel.depth}")
        print(f"  Speedup: {sequential.depth/parallel.depth:.1f}x faster")
    
    gate_count_optimization()
    circuit_depth_optimization()

coherence_time_optimization()
```

### Adaptive Circuit Compilation

```python
def adaptive_compilation():
    """Compile circuits adaptively for specific hardware."""
    
    print("\n🎯 Adaptive Circuit Compilation")
    print("=" * 35)
    
    class QuantumDevice:
        """Simplified quantum device model."""
        
        def __init__(self, name, topology, gate_times, error_rates):
            self.name = name
            self.topology = topology
            self.gate_times = gate_times
            self.error_rates = error_rates
    
    # Define different device types
    devices = {
        "superconducting": QuantumDevice(
            name="Superconducting Transmon",
            topology="grid",
            gate_times={"single": 50e-9, "two_qubit": 200e-9},
            error_rates={"single": 0.001, "two_qubit": 0.01}
        ),
        "trapped_ion": QuantumDevice(
            name="Trapped Ion",
            topology="all_to_all",
            gate_times={"single": 10e-6, "two_qubit": 100e-6},
            error_rates={"single": 0.0001, "two_qubit": 0.001}
        ),
        "photonic": QuantumDevice(
            name="Photonic",
            topology="limited",
            gate_times={"single": 1e-9, "two_qubit": 10e-9},
            error_rates={"single": 0.01, "two_qubit": 0.1}
        )
    }
    
    def compile_for_device(algorithm, device):
        """Compile algorithm for specific device."""
        
        print(f"\nCompiling for {device.name}:")
        
        if device.name == "Superconducting Transmon":
            print("  - Use native CZ gates")
            print("  - Minimize two-qubit gate count")
            print("  - Respect grid connectivity")
            strategy = "minimize_two_qubit"
            
        elif device.name == "Trapped Ion":
            print("  - Take advantage of all-to-all connectivity")
            print("  - Minimize total gate time")
            print("  - Use longer, more accurate gates")
            strategy = "minimize_gate_time"
            
        elif device.name == "Photonic":
            print("  - Avoid two-qubit gates when possible")
            print("  - Use measurement-based computation")
            print("  - Exploit fast single-qubit gates")
            strategy = "measurement_based"
        
        return strategy
    
    # Example: Compile Bell state for different devices
    print("Example: Compiling Bell state preparation")
    
    for device_type, device in devices.items():
        strategy = compile_for_device("bell_state", device)
        
        # Estimate execution time
        estimated_time = (2 * device.gate_times["single"] + 
                         1 * device.gate_times["two_qubit"])
        
        print(f"  Estimated time: {estimated_time*1e6:.1f} μs")

adaptive_compilation()
```

## NISQ Algorithm Design Patterns

### Variational Algorithms

```python
def nisq_algorithm_patterns():
    """Design patterns for NISQ-era algorithms."""
    
    print("\n🏗️  NISQ Algorithm Design Patterns")
    print("=" * 40)
    
    print("Pattern 1: Variational Quantum Algorithms")
    
    def variational_pattern():
        """Template for variational algorithms."""
        
        print("  Structure:")
        print("    1. Parameterized quantum circuit (ansatz)")
        print("    2. Classical parameter optimization")
        print("    3. Measurement-based cost function")
        print("    4. Iterative improvement")
        
        print("  Advantages:")
        print("    - Short circuit depth")
        print("    - Hardware-agnostic")
        print("    - Error-resilient")
        
        print("  Examples: VQE, QAOA, VQC")
    
    print("\nPattern 2: Hybrid Quantum-Classical")
    
    def hybrid_pattern():
        """Template for hybrid algorithms."""
        
        print("  Structure:")
        print("    1. Quantum subroutine (short)")
        print("    2. Classical processing")
        print("    3. Feedback loop")
        print("    4. Iterative refinement")
        
        print("  Advantages:")
        print("    - Leverages classical computation")
        print("    - Adaptive to hardware")
        print("    - Error mitigation built-in")
        
        print("  Examples: Quantum ML, optimization")
    
    print("\nPattern 3: Error-Aware Design")
    
    def error_aware_pattern():
        """Design with error tolerance."""
        
        print("  Principles:")
        print("    - Use redundancy and averaging")
        print("    - Minimize circuit depth")
        print("    - Include error mitigation")
        print("    - Validate results")
        
        print("  Techniques:")
        print("    - Multiple circuit instances")
        print("    - Statistical error analysis")
        print("    - Symmetry checks")
        
    variational_pattern()
    hybrid_pattern()
    error_aware_pattern()

nisq_algorithm_patterns()
```

## Real Device Integration

### Working with Quantum Hardware APIs

```python
def hardware_integration_guide():
    """Guide to integrating with real quantum hardware."""
    
    print("\n🌐 Real Hardware Integration")
    print("=" * 30)
    
    # Simulate hardware provider interfaces
    hardware_providers = {
        "IBM Quantum": {
            "backend_types": ["simulator", "real_device"],
            "devices": ["ibmq_qasm_simulator", "ibm_perth", "ibm_lagos"],
            "queue_time": "minutes to hours",
            "cost": "free tier available"
        },
        "Google Quantum AI": {
            "backend_types": ["simulator", "sycamore"],
            "devices": ["cirq_simulator", "weber", "rainbow"],
            "queue_time": "research access",
            "cost": "research partnerships"
        },
        "AWS Braket": {
            "backend_types": ["simulator", "rigetti", "ionq", "dwave"],
            "devices": ["sv1", "aspen-m", "harmony", "advantage"],
            "queue_time": "seconds to minutes",
            "cost": "pay per shot"
        }
    }
    
    print("Major quantum cloud providers:")
    for provider, specs in hardware_providers.items():
        print(f"\n{provider}:")
        for spec, value in specs.items():
            if isinstance(value, list):
                print(f"  {spec}: {', '.join(value[:2])}...")
            else:
                print(f"  {spec}: {value}")
    
    print(f"\nBest practices for hardware execution:")
    
    best_practices = [
        "Start with simulators for debugging",
        "Optimize circuits before submitting to hardware",
        "Use error mitigation techniques", 
        "Submit multiple shots for statistics",
        "Monitor queue times and costs",
        "Validate results against theory",
        "Use hardware-native gate sets",
        "Consider device calibration schedules"
    ]
    
    for i, practice in enumerate(best_practices, 1):
        print(f"  {i}. {practice}")

def quantum_cloud_workflow():
    """Typical workflow for cloud quantum computing."""
    
    print(f"\n☁️  Quantum Cloud Workflow")
    print("=" * 30)
    
    workflow_steps = [
        ("1. Algorithm Design", "Design quantum algorithm", "Local development"),
        ("2. Circuit Optimization", "Minimize gates and depth", "Classical simulation"),
        ("3. Hardware Selection", "Choose appropriate device", "Provider comparison"),
        ("4. Job Submission", "Submit to quantum cloud", "Queue monitoring"),
        ("5. Result Analysis", "Process measurement data", "Error analysis"),
        ("6. Iteration", "Refine based on results", "Continuous improvement")
    ]
    
    for step, description, tools in workflow_steps:
        print(f"\n{step}: {description}")
        print(f"   Tools: {tools}")
    
    print(f"\nExample job submission pseudocode:")
    print("""
    # Pseudocode for quantum cloud execution
    device = quantum_provider.get_backend('real_device')
    
    # Circuit optimization
    optimized_circuit = device.compile(original_circuit)
    
    # Job submission
    job = device.run(optimized_circuit, shots=1000)
    
    # Result processing
    result = job.result()
    counts = result.get_counts()
    
    # Error analysis
    error_mitigated_counts = apply_error_mitigation(counts)
    """)

hardware_integration_guide()
quantum_cloud_workflow()
```

## Performance Benchmarking

### Measuring Algorithm Performance

```python
def performance_benchmarking():
    """Benchmark quantum algorithms on different metrics."""
    
    print("\n📊 Performance Benchmarking")
    print("=" * 30)
    
    def benchmark_metrics():
        """Key metrics for quantum algorithm performance."""
        
        metrics = {
            "Fidelity": "How close to ideal result",
            "Gate count": "Total number of gates",
            "Circuit depth": "Critical path length", 
            "Two-qubit gate count": "Most error-prone gates",
            "Execution time": "Total runtime",
            "Success probability": "Probability of correct answer",
            "Quantum volume": "Overall device capability"
        }
        
        print("Key performance metrics:")
        for metric, description in metrics.items():
            print(f"  {metric:20}: {description}")
    
    def create_benchmark_suite():
        """Create suite of benchmark algorithms."""
        
        benchmarks = [
            ("Bell state preparation", "Basic entanglement", 2, 2),
            ("GHZ state preparation", "Multi-qubit entanglement", 3, 3), 
            ("Quantum Fourier Transform", "Algorithmic primitive", 3, 15),
            ("Grover search (4 items)", "Search algorithm", 2, 8),
            ("Variational classifier", "Machine learning", 4, 20)
        ]
        
        print(f"\nBenchmark algorithm suite:")
        print(f"{'Algorithm':<25} {'Purpose':<20} {'Qubits'} {'Gates'}")
        print("-" * 60)
        
        for name, purpose, qubits, gates in benchmarks:
            print(f"{name:<25} {purpose:<20} {qubits:6} {gates:5}")
    
    def performance_targets():
        """Set performance targets for NISQ devices."""
        
        targets = {
            "Single-qubit gate fidelity": "> 99.9%",
            "Two-qubit gate fidelity": "> 99%", 
            "Readout fidelity": "> 98%",
            "Circuit depth limit": "< 100 gates",
            "Coherence time": "> 100 μs",
            "Quantum volume": "> 64"
        }
        
        print(f"\nNISQ performance targets:")
        for target, value in targets.items():
            print(f"  {target:25}: {value}")
    
    benchmark_metrics()
    create_benchmark_suite()
    performance_targets()

performance_benchmarking()
```

## Key Takeaways

🎯 **What you learned:**

1. **Hardware Constraints**: Real quantum devices have limited connectivity, noise, and coherence
2. **Circuit Optimization**: Minimize gates, depth, and respect hardware topology
3. **Error Mitigation**: Techniques to improve results despite noise
4. **NISQ Algorithms**: Design patterns for near-term devices
5. **Cloud Integration**: Working with quantum cloud providers

🚀 **Optimization strategies:**

- **Connectivity-aware design**: Match circuits to hardware topology
- **Noise-resilient circuits**: Minimize errors through smart design
- **Coherence optimization**: Reduce execution time within coherence limits
- **Adaptive compilation**: Customize circuits for specific devices
- **Error mitigation**: Improve results post-execution

⚡ **Best practices:**

- Start with simulation, then move to hardware
- Always include error analysis
- Use native gate sets when possible
- Monitor and optimize circuit metrics
- Validate results against theoretical expectations

## NISQ Algorithm Checklist

Before deploying on quantum hardware:

- [ ] **Circuit optimized** for target device topology
- [ ] **Gate count minimized** using circuit optimization
- [ ] **Depth reduced** through parallelization
- [ ] **Error mitigation** strategies included
- [ ] **Multiple shots** planned for statistical accuracy
- [ ] **Result validation** method defined
- [ ] **Cost and queue time** estimated
- [ ] **Fallback simulation** available for comparison

## Common Optimization Mistakes

❌ **Avoid these mistakes:**
- Ignoring hardware connectivity constraints
- Using decomposed gates instead of native gates
- Not accounting for noise in circuit design
- Submitting unoptimized circuits to expensive hardware
- Forgetting to validate results

✅ **Best practices:**
- Profile circuits on simulators first
- Use hardware-specific compilation tools
- Include noise modeling in development
- Implement error mitigation from the start
- Always validate against classical simulation

## What's Next?

You now understand how to optimize quantum algorithms for real hardware! In the next tutorial, we'll explore advanced topics like quantum machine learning and optimization.

**Next:** [Tutorial 5: Quantum Machine Learning](05-quantum-machine-learning.md)

## Practice Exercises

1. **Circuit Routing**: Design a 5-qubit circuit optimized for linear topology
2. **Noise Analysis**: Estimate error rates for different circuit depths
3. **Compilation Comparison**: Compare circuit optimizations for different device types
4. **Error Mitigation**: Implement zero-noise extrapolation for a simple circuit

## Additional Resources

### Hardware Platforms
- [IBM Quantum Experience](https://quantum-computing.ibm.com/)
- [Google Quantum AI](https://quantumai.google/)
- [AWS Braket](https://aws.amazon.com/braket/)
- [Microsoft Azure Quantum](https://azure.microsoft.com/en-us/services/quantum/)

### Optimization Tools
- [Qiskit Circuit Optimization](https://qiskit.org/documentation/apidoc/transpiler.html)
- [Cirq Device Compilation](https://quantumai.google/cirq/tutorials/google/start)
- [TKET Compiler](https://cqcl.github.io/tket/)

---

**Ready for quantum machine learning?** [Continue to Tutorial 5: Quantum Machine Learning →](05-quantum-machine-learning.md)

*"In theory there is no difference between theory and practice. In practice there is." - Yogi Berra*

This definitely applies to quantum computing - but now you're ready for practice! 🚀