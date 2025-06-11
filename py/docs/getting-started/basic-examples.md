# Basic Examples

This collection of examples demonstrates fundamental quantum computing concepts using QuantRS2. Each example includes detailed explanations and practical code you can run immediately.

## Example 1: Bell States (Quantum Entanglement)

Bell states are maximally entangled two-qubit states that demonstrate quantum correlation.

```python
import quantrs2
import numpy as np

def create_bell_states():
    """Create all four Bell states."""
    
    bell_states = {
        'Φ⁺': '|00⟩ + |11⟩',  # Bell state |Φ⁺⟩
        'Φ⁻': '|00⟩ - |11⟩',  # Bell state |Φ⁻⟩
        'Ψ⁺': '|01⟩ + |10⟩',  # Bell state |Ψ⁺⟩
        'Ψ⁻': '|01⟩ - |10⟩'   # Bell state |Ψ⁻⟩
    }
    
    results = {}
    
    for name, description in bell_states.items():
        circuit = quantrs2.Circuit(2)
        
        if name == 'Φ⁺':
            # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
            circuit.h(0)
            circuit.cx(0, 1)
        
        elif name == 'Φ⁻':
            # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
            circuit.h(0)
            circuit.z(0)
            circuit.cx(0, 1)
        
        elif name == 'Ψ⁺':
            # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
            circuit.h(0)
            circuit.x(1)
            circuit.cx(0, 1)
        
        elif name == 'Ψ⁻':
            # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
            circuit.h(0)
            circuit.x(1)
            circuit.z(0)
            circuit.cx(0, 1)
        
        circuit.measure_all()
        result = circuit.run()
        results[name] = result.state_probabilities()
        
        print(f"{name} ({description}):")
        print(f"  Probabilities: {results[name]}")
        print()
    
    return results

# Create and demonstrate Bell states
bell_results = create_bell_states()

# Visualize one of the Bell state circuits
phi_plus = quantrs2.Circuit(2)
phi_plus.h(0)
phi_plus.cx(0, 1)
quantrs2.visualize_circuit(phi_plus, title="Bell State |Φ⁺⟩")
```

**Key Concepts:**
- **Entanglement**: Bell states show perfect correlation between qubits
- **Superposition**: Each qubit is in a quantum superposition until measured
- **No-cloning**: You cannot copy an unknown quantum state

## Example 2: Quantum Superposition

Explore quantum superposition with single and multiple qubits.

```python
def demonstrate_superposition():
    """Demonstrate quantum superposition principles."""
    
    print("=== Single Qubit Superposition ===")
    
    # |+⟩ state: equal superposition
    plus_state = quantrs2.Circuit(1)
    plus_state.h(0)  # Creates (|0⟩ + |1⟩)/√2
    plus_state.measure_all()
    
    result = plus_state.run()
    print("H|0⟩ = |+⟩ state:")
    print(f"Probabilities: {result.state_probabilities()}")
    
    # |−⟩ state: minus superposition
    minus_state = quantrs2.Circuit(1)
    minus_state.h(0)
    minus_state.z(0)  # Creates (|0⟩ - |1⟩)/√2
    minus_state.measure_all()
    
    result = minus_state.run()
    print("ZH|0⟩ = |−⟩ state:")
    print(f"Probabilities: {result.state_probabilities()}")
    
    print("\n=== Multi-Qubit Superposition ===")
    
    # W state: symmetric superposition
    w_state = quantrs2.Circuit(3)
    
    # Create |W⟩ = (|001⟩ + |010⟩ + |100⟩)/√3
    # This requires a more complex construction
    w_state.ry(0, 2 * np.arccos(np.sqrt(2/3)))
    w_state.cy(0, 1)
    w_state.cx(0, 2)
    w_state.cx(1, 2)
    
    w_state.measure_all()
    result = w_state.run()
    
    print("W state |W⟩:")
    print(f"Probabilities: {result.state_probabilities()}")

demonstrate_superposition()

# Visualize superposition evolution
def visualize_superposition_evolution():
    """Show how superposition evolves with rotations."""
    angles = np.linspace(0, 2*np.pi, 8)
    
    print("\n=== Superposition Evolution ===")
    for i, angle in enumerate(angles):
        circuit = quantrs2.Circuit(1)
        circuit.ry(0, angle)
        circuit.measure_all()
        
        result = circuit.run()
        probs = result.state_probabilities()
        prob_0 = probs.get('0', 0)
        prob_1 = probs.get('1', 0)
        
        print(f"RY({angle:.2f}): P(0)={prob_0:.3f}, P(1)={prob_1:.3f}")

visualize_superposition_evolution()
```

## Example 3: Quantum Interference

Demonstrate quantum interference using the Mach-Zehnder interferometer analogy.

```python
def quantum_interference_demo():
    """Demonstrate quantum interference patterns."""
    
    print("=== Quantum Interference Demo ===")
    
    # Demonstrate constructive interference
    constructive = quantrs2.Circuit(1)
    constructive.h(0)     # Split: |0⟩ → (|0⟩ + |1⟩)/√2
    constructive.h(0)     # Recombine: interfere constructively
    constructive.measure_all()
    
    result = constructive.run()
    print("Constructive interference (HH|0⟩):")
    print(f"Probabilities: {result.state_probabilities()}")
    print("Result: Returns to |0⟩ with probability 1")
    
    # Demonstrate destructive interference
    destructive = quantrs2.Circuit(1)
    destructive.h(0)      # Split: |0⟩ → (|0⟩ + |1⟩)/√2
    destructive.z(0)      # Phase shift: (|0⟩ - |1⟩)/√2
    destructive.h(0)      # Recombine: interfere destructively
    destructive.measure_all()
    
    result = destructive.run()
    print("\nDestructive interference (HZH|0⟩):")
    print(f"Probabilities: {result.state_probabilities()}")
    print("Result: Results in |1⟩ with probability 1")
    
    # Phase-dependent interference
    print("\n=== Phase-Dependent Interference ===")
    phases = np.linspace(0, 2*np.pi, 8)
    
    for phase in phases:
        circuit = quantrs2.Circuit(1)
        circuit.h(0)              # Create superposition
        circuit.rz(0, phase)      # Apply phase
        circuit.h(0)              # Interfere
        circuit.measure_all()
        
        result = circuit.run()
        prob_0 = result.state_probabilities().get('0', 0)
        
        print(f"Phase {phase:.2f}: P(|0⟩) = {prob_0:.3f}")

quantum_interference_demo()
```

## Example 4: Quantum Phase Estimation

Implement a simple quantum phase estimation algorithm.

```python
def quantum_phase_estimation():
    """Implement quantum phase estimation for a simple unitary."""
    
    print("=== Quantum Phase Estimation ===")
    
    # We'll estimate the phase of the T gate: T|1⟩ = e^(iπ/4)|1⟩
    # So the phase φ = π/4, and we should get 2^n * φ/2π = 2^n/8
    
    n_counting_qubits = 3  # Precision qubits
    n_total = n_counting_qubits + 1  # +1 for eigenstate qubit
    
    circuit = quantrs2.Circuit(n_total)
    
    # Prepare eigenstate |1⟩ on the last qubit
    circuit.x(n_counting_qubits)
    
    # Initialize counting qubits in superposition
    for i in range(n_counting_qubits):
        circuit.h(i)
    
    # Apply controlled unitary operations
    # Controlled-T gate with different powers
    for i in range(n_counting_qubits):
        # Apply T gate 2^i times (controlled by qubit i)
        for _ in range(2**i):
            # Controlled-T gate (simplified)
            circuit.crz(i, n_counting_qubits, np.pi/4)
    
    # Apply inverse QFT on counting qubits
    # Simplified inverse QFT
    for i in range(n_counting_qubits):
        for j in range(i):
            circuit.crz(j, i, -np.pi/2**(i-j))
        circuit.h(i)
    
    # Reverse the order of counting qubits
    for i in range(n_counting_qubits // 2):
        circuit.swap(i, n_counting_qubits - 1 - i)
    
    circuit.measure_all()
    result = circuit.run()
    
    print("Phase estimation results:")
    probs = result.state_probabilities()
    
    # Extract most likely phase estimate
    max_prob = 0
    best_estimate = None
    
    for state, prob in probs.items():
        if prob > max_prob:
            max_prob = prob
            best_estimate = state
    
    # Convert binary result to phase estimate
    if best_estimate:
        # Take first n_counting_qubits bits
        phase_bits = best_estimate[:n_counting_qubits]
        phase_estimate = int(phase_bits, 2) / (2**n_counting_qubits)
        theoretical_phase = 1/8  # π/4 divided by 2π
        
        print(f"Measured phase bits: {phase_bits}")
        print(f"Estimated phase: {phase_estimate:.3f}")
        print(f"Theoretical phase: {theoretical_phase:.3f}")
        print(f"Error: {abs(phase_estimate - theoretical_phase):.3f}")

quantum_phase_estimation()
```

## Example 5: Quantum Random Walk

Implement a quantum random walk on a line.

```python
def quantum_random_walk(steps=4):
    """Implement quantum random walk."""
    
    print(f"=== Quantum Random Walk ({steps} steps) ===")
    
    # We need qubits for position and a coin qubit
    # For simplicity, we'll use a finite line with wrap-around
    n_position_qubits = 3  # Can represent positions 0-7
    n_total = n_position_qubits + 1  # +1 for coin qubit
    
    circuit = quantrs2.Circuit(n_total)
    coin_qubit = n_total - 1
    
    # Start at position 0 (all position qubits in |0⟩)
    # Initialize coin in superposition
    circuit.h(coin_qubit)
    
    for step in range(steps):
        print(f"Step {step + 1}:")
        
        # Coin flip (already in superposition)
        circuit.h(coin_qubit)
        
        # Conditional move based on coin
        # If coin is |0⟩, move left (subtract 1)
        # If coin is |1⟩, move right (add 1)
        
        # Simplified movement (increment/decrement position)
        # This is a complex operation that requires quantum arithmetic
        # For demonstration, we'll use a simplified version
        
        # Controlled increment (when coin is |1⟩)
        for i in range(n_position_qubits):
            circuit.cx(coin_qubit, i)
        
        # Measure intermediate state for visualization
        temp_circuit = circuit.copy()
        temp_circuit.measure_all()
        temp_result = temp_circuit.run()
        
        print(f"  Position distribution: {temp_result.state_probabilities()}")
    
    # Final measurement
    circuit.measure_all()
    result = circuit.run()
    
    print("Final quantum random walk distribution:")
    print(result.state_probabilities())
    
    return result

# Run quantum random walk
qrw_result = quantum_random_walk(3)
```

## Example 6: Quantum Error Correction (3-Qubit Bit-Flip Code)

Demonstrate basic quantum error correction.

```python
def three_qubit_bit_flip_code():
    """Implement 3-qubit bit-flip error correction."""
    
    print("=== 3-Qubit Bit-Flip Error Correction ===")
    
    # Prepare initial state (we'll use |1⟩ for demonstration)
    circuit = quantrs2.Circuit(5)  # 3 data + 2 syndrome qubits
    
    # Prepare logical |1⟩ state
    circuit.x(0)  # Set first qubit to |1⟩
    
    # Encoding: Create 3-qubit repetition code
    circuit.cx(0, 1)  # Copy to second qubit
    circuit.cx(0, 2)  # Copy to third qubit
    
    print("Encoded state (no error):")
    temp_circuit = circuit.copy()
    temp_circuit.measure_all()
    temp_result = temp_circuit.run()
    print(f"  {temp_result.state_probabilities()}")
    
    # Introduce bit-flip error on qubit 1
    print("\nIntroducing error on qubit 1...")
    circuit.x(1)  # Bit-flip error
    
    print("State after error:")
    temp_circuit = circuit.copy()
    temp_circuit.measure_all()
    temp_result = temp_circuit.run()
    print(f"  {temp_result.state_probabilities()}")
    
    # Error detection using syndrome qubits
    # Syndrome qubit 3: checks qubits 0 and 1
    circuit.cx(0, 3)
    circuit.cx(1, 3)
    
    # Syndrome qubit 4: checks qubits 1 and 2
    circuit.cx(1, 4)
    circuit.cx(2, 4)
    
    # Measure syndrome qubits
    circuit.measure(3)
    circuit.measure(4)
    
    # Based on syndrome, correct the error
    # Syndrome 11: error on qubit 1
    # Syndrome 10: error on qubit 0
    # Syndrome 01: error on qubit 2
    # Syndrome 00: no error
    
    # For demonstration, we'll correct qubit 1 (syndrome should be 11)
    circuit.cx(3, 1)  # Correct if syndrome qubit 3 is |1⟩
    circuit.cx(4, 1)  # Correct if syndrome qubit 4 is |1⟩
    
    # Final measurement of data qubits
    circuit.measure(0)
    circuit.measure(1)
    circuit.measure(2)
    
    result = circuit.run()
    
    print("After error correction:")
    print(f"  {result.state_probabilities()}")
    
    return result

# Demonstrate error correction
error_correction_result = three_qubit_bit_flip_code()
```

## Example 7: Quantum Fourier Transform

Implement the Quantum Fourier Transform.

```python
def quantum_fourier_transform(n_qubits=3):
    """Implement Quantum Fourier Transform."""
    
    print(f"=== Quantum Fourier Transform ({n_qubits} qubits) ===")
    
    def qft_circuit(circuit, n):
        """Add QFT gates to circuit."""
        for i in range(n):
            # Apply Hadamard to current qubit
            circuit.h(i)
            
            # Apply controlled phase rotations
            for j in range(i + 1, n):
                # Controlled phase rotation by 2π/2^(j-i+1)
                phase = 2 * np.pi / (2**(j-i+1))
                circuit.crz(j, i, phase)
        
        # Reverse qubit order
        for i in range(n // 2):
            circuit.swap(i, n - 1 - i)
    
    # Test QFT on a specific input state
    circuit = quantrs2.Circuit(n_qubits)
    
    # Prepare input state |001⟩ (for demonstration)
    circuit.x(n_qubits - 1)
    
    print("Input state:")
    temp_circuit = circuit.copy()
    temp_circuit.measure_all()
    temp_result = temp_circuit.run()
    print(f"  {temp_result.state_probabilities()}")
    
    # Apply QFT
    qft_circuit(circuit, n_qubits)
    
    print("After QFT:")
    circuit.measure_all()
    result = circuit.run()
    print(f"  {result.state_probabilities()}")
    
    # The QFT should create a uniform superposition with phases
    # For input |001⟩, we expect specific phase relationships
    
    return result

# Demonstrate QFT
qft_result = quantum_fourier_transform(3)

# Visualize QFT circuit
def visualize_qft():
    """Visualize a QFT circuit."""
    circuit = quantrs2.Circuit(3)
    circuit.x(2)  # Input |001⟩
    
    # QFT implementation
    circuit.h(0)
    circuit.crz(1, 0, np.pi/2)
    circuit.crz(2, 0, np.pi/4)
    
    circuit.h(1)
    circuit.crz(2, 1, np.pi/2)
    
    circuit.h(2)
    
    # Reverse order
    circuit.swap(0, 2)
    
    quantrs2.visualize_circuit(circuit, title="3-Qubit QFT")

visualize_qft()
```

## Example 8: Variational Quantum Eigensolver (VQE) Basics

Simple VQE implementation for finding ground state energy.

```python
def simple_vqe():
    """Implement a basic VQE algorithm."""
    
    print("=== Simple VQE Implementation ===")
    
    # Define a simple Hamiltonian: H = Z₀ + Z₁ (sum of Pauli-Z on each qubit)
    # Ground state should be |11⟩ with energy -2
    
    def create_ansatz(parameters):
        """Create ansatz circuit."""
        circuit = quantrs2.Circuit(2)
        
        # RY rotations
        circuit.ry(0, parameters[0])
        circuit.ry(1, parameters[1])
        
        # Entangling gate
        circuit.cx(0, 1)
        
        # More RY rotations
        circuit.ry(0, parameters[2])
        circuit.ry(1, parameters[3])
        
        return circuit
    
    def measure_hamiltonian(circuit):
        """Measure expectation value of Hamiltonian Z₀ + Z₁."""
        
        # For this simple case, we can compute directly from probabilities
        circuit_copy = circuit.copy()
        circuit_copy.measure_all()
        result = circuit_copy.run()
        probs = result.state_probabilities()
        
        # Calculate ⟨Z₀ + Z₁⟩
        expectation = 0
        for state, prob in probs.items():
            if len(state) >= 2:
                z0_val = 1 if state[0] == '0' else -1
                z1_val = 1 if state[1] == '0' else -1
                expectation += prob * (z0_val + z1_val)
        
        return expectation
    
    # VQE optimization (simplified)
    best_energy = float('inf')
    best_params = None
    
    print("VQE Optimization Progress:")
    
    # Grid search over parameters (in practice, use gradient-based methods)
    num_steps = 8
    for i in range(num_steps):
        # Random parameters
        params = np.random.uniform(0, 2*np.pi, 4)
        
        # Create ansatz and measure energy
        ansatz = create_ansatz(params)
        energy = measure_hamiltonian(ansatz)
        
        if energy < best_energy:
            best_energy = energy
            best_params = params
        
        print(f"  Step {i+1}: Energy = {energy:.4f}")
    
    print(f"\nBest energy found: {best_energy:.4f}")
    print(f"Theoretical minimum: -2.0000")
    print(f"Error: {abs(best_energy + 2.0):.4f}")
    
    # Show final state
    final_ansatz = create_ansatz(best_params)
    final_ansatz.measure_all()
    final_result = final_ansatz.run()
    
    print(f"Final state probabilities: {final_result.state_probabilities()}")
    
    return best_energy, best_params

# Run VQE
vqe_energy, vqe_params = simple_vqe()
```

## Visualization and Analysis

Visualize all the examples we've created:

```python
def analyze_examples():
    """Analyze and visualize all examples."""
    
    print("\n=== Example Analysis Summary ===")
    
    examples = [
        ("Bell State", lambda: create_bell_states()),
        ("Superposition", lambda: demonstrate_superposition()),
        ("QFT", lambda: quantum_fourier_transform(3)),
        ("VQE", lambda: simple_vqe())
    ]
    
    for name, example_func in examples:
        print(f"\n{name} Example:")
        try:
            result = example_func()
            print(f"  ✓ Successfully executed")
        except Exception as e:
            print(f"  ✗ Error: {e}")

# Run analysis
analyze_examples()

# Create summary visualization
def create_summary_visualization():
    """Create a summary visualization of key concepts."""
    
    # Bell state circuit
    bell = quantrs2.Circuit(2)
    bell.h(0)
    bell.cx(0, 1)
    
    print("\nQuantum Circuit Gallery:")
    print("1. Bell State Preparation")
    quantrs2.visualize_circuit(bell, title="Bell State |Φ⁺⟩")

create_summary_visualization()
```

## Next Steps

These examples demonstrate fundamental quantum computing concepts. To continue learning:

1. **[User Guide](../user-guide/core-concepts.md)**: Dive deeper into quantum concepts
2. **[Quantum Algorithms](../user-guide/quantum-algorithms.md)**: Learn more complex algorithms
3. **[Machine Learning](../advanced/machine-learning.md)**: Explore quantum ML applications
4. **[Tutorials](../tutorials/beginner/)**: Follow structured learning paths

## Key Takeaways

From these examples, you should understand:

- **Entanglement**: How qubits become correlated through two-qubit gates
- **Superposition**: How qubits exist in multiple states simultaneously
- **Interference**: How quantum phases affect measurement outcomes
- **Measurement**: How observation collapses quantum states
- **Error Correction**: How to protect quantum information
- **Algorithms**: How to implement quantum algorithms like QFT and VQE

**Ready for more advanced topics?** Explore the [User Guide](../user-guide/core-concepts.md)!