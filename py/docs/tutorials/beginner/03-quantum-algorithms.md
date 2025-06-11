# Tutorial 3: Essential Quantum Algorithms

**Estimated time:** 45 minutes  
**Prerequisites:** [Tutorial 2: Your First Circuit](02-first-circuit.md)  
**Goal:** Implement and understand fundamental quantum algorithms that demonstrate quantum advantage

Now that you can build quantum circuits, let's explore the algorithms that make quantum computing revolutionary! In this tutorial, you'll implement famous quantum algorithms and understand why they're faster than classical approaches.

## What Makes a Quantum Algorithm?

### Classical vs Quantum Algorithms

**Classical algorithms** process one possibility at a time:
```
Input → Process step by step → Output
```

**Quantum algorithms** leverage quantum phenomena:
```
Input → Superposition → Interference → Measurement → Output
     ↳ Many possibilities ↳ Amplify correct answers
```

### Key Quantum Algorithm Patterns

1. **Initialization**: Prepare qubits in superposition
2. **Oracle**: Mark target solutions  
3. **Amplification**: Increase probability of correct answers
4. **Measurement**: Extract classical result

Let's see these patterns in action!

## Algorithm 1: Deutsch's Algorithm

### The Problem

Given a black-box function f(x) that takes a single bit input:
- **Constant function**: f(0) = f(1) (always 0 or always 1)
- **Balanced function**: f(0) ≠ f(1) (outputs differ)

**Classical approach**: Test both inputs → 2 function calls  
**Quantum approach**: Determine with just 1 quantum query!

### The Algorithm

```python
import quantrs2
import numpy as np

def deutsch_algorithm(oracle_type="constant_0"):
    """
    Implement Deutsch's algorithm.
    
    Args:
        oracle_type: "constant_0", "constant_1", "balanced_identity", "balanced_not"
    """
    
    print(f"🔍 Deutsch's Algorithm - Oracle: {oracle_type}")
    print("=" * 50)
    
    # Create circuit with 2 qubits
    circuit = quantrs2.Circuit(2)
    
    print("Step 1: Initialize qubits")
    print("  |0⟩ ⊗ |1⟩")
    circuit.x(1)  # Put ancilla in |1⟩
    
    print("\nStep 2: Create superposition")
    circuit.h(0)  # Control qubit in superposition
    circuit.h(1)  # Ancilla in |−⟩ = (|0⟩ - |1⟩)/√2
    print("  (|0⟩ + |1⟩)/√2 ⊗ (|0⟩ - |1⟩)/√2")
    
    print(f"\nStep 3: Apply oracle ({oracle_type})")
    # Oracle implementation depends on function type
    if oracle_type == "constant_0":
        # f(x) = 0 for all x → do nothing
        pass
    elif oracle_type == "constant_1":
        # f(x) = 1 for all x → flip ancilla
        circuit.x(1)
    elif oracle_type == "balanced_identity":
        # f(x) = x → CNOT
        circuit.cx(0, 1)
    elif oracle_type == "balanced_not":
        # f(x) = NOT x → X then CNOT
        circuit.x(0)
        circuit.cx(0, 1)
        circuit.x(0)
    
    print("\nStep 4: Final Hadamard on control")
    circuit.h(0)
    
    print("\nStep 5: Measure control qubit")
    circuit.measure_all()
    result = circuit.run()
    
    # Interpret result
    probs = result.state_probabilities()
    prob_0 = sum(prob for state, prob in probs.items() if state[0] == '0')
    
    if prob_0 > 0.9:
        function_type = "CONSTANT"
    else:
        function_type = "BALANCED"
    
    print(f"\nResult: P(|0⟩) = {prob_0:.3f}")
    print(f"Conclusion: Function is {function_type}")
    print(f"✅ Correct!" if (
        (oracle_type.startswith("constant") and function_type == "CONSTANT") or
        (oracle_type.startswith("balanced") and function_type == "BALANCED")
    ) else "❌ Error!")
    
    return function_type

# Test all oracle types
oracle_types = ["constant_0", "constant_1", "balanced_identity", "balanced_not"]

for oracle in oracle_types:
    deutsch_algorithm(oracle)
    print()
```

**🎯 Key Insight:** Quantum parallelism lets us test both inputs simultaneously!

## Algorithm 2: Grover's Search

### The Problem

Search an unsorted database of N items for a specific target:
- **Classical approach**: Check items one by one → O(N) queries
- **Quantum approach**: Grover's algorithm → O(√N) queries

For a million items: Classical needs ~500,000 queries, Quantum needs ~1,000!

### The Algorithm

```python
def grovers_algorithm(target_item=3, num_items=4):
    """
    Implement Grover's search algorithm.
    
    Args:
        target_item: Which item to search for (0 to num_items-1)
        num_items: Total number of items (must be power of 2)
    """
    
    print(f"🔍 Grover's Search Algorithm")
    print(f"Target: Item {target_item} out of {num_items} items")
    print("=" * 45)
    
    # Calculate number of qubits needed
    num_qubits = int(np.log2(num_items))
    circuit = quantrs2.Circuit(num_qubits)
    
    print("Step 1: Initialize uniform superposition")
    for qubit in range(num_qubits):
        circuit.h(qubit)
    print(f"  Equal probability for all {num_items} items")
    
    # Number of Grover iterations needed
    num_iterations = int(np.pi * np.sqrt(num_items) / 4)
    print(f"\nStep 2: Apply Grover operator {num_iterations} times")
    
    for iteration in range(num_iterations):
        print(f"  Iteration {iteration + 1}:")
        
        # Oracle: Mark the target item
        print(f"    Oracle: Mark item {target_item}")
        oracle_grover(circuit, target_item, num_qubits)
        
        # Diffusion operator: Amplitude amplification
        print("    Diffusion: Amplify marked item")
        diffusion_operator(circuit, num_qubits)
    
    print("\nStep 3: Measure result")
    circuit.measure_all()
    result = circuit.run()
    
    # Find most likely outcome
    probs = result.state_probabilities()
    found_state = max(probs, key=probs.get)
    found_item = int(found_state, 2)
    success_prob = probs[found_state]
    
    print(f"\nResult: Found item {found_item}")
    print(f"Success probability: {success_prob:.3f}")
    print(f"✅ Correct!" if found_item == target_item else f"❌ Wrong! Expected {target_item}")
    
    return found_item

def oracle_grover(circuit, target, num_qubits):
    """Oracle that marks the target item."""
    # Convert target to binary
    target_binary = format(target, f'0{num_qubits}b')
    
    # Flip qubits to prepare for Z-rotation
    for i, bit in enumerate(target_binary):
        if bit == '0':
            circuit.x(i)
    
    # Multi-controlled Z gate (marks target with phase flip)
    if num_qubits == 1:
        circuit.z(0)
    elif num_qubits == 2:
        circuit.cz(0, 1) if hasattr(circuit, 'cz') else controlled_z_2qubit(circuit)
    else:
        # Multi-controlled Z for more qubits (simplified)
        multi_controlled_z(circuit, list(range(num_qubits)))
    
    # Flip qubits back
    for i, bit in enumerate(target_binary):
        if bit == '0':
            circuit.x(i)

def controlled_z_2qubit(circuit):
    """Implement controlled-Z for 2 qubits using available gates."""
    circuit.h(1)
    circuit.cx(0, 1)
    circuit.h(1)

def multi_controlled_z(circuit, qubits):
    """Simplified multi-controlled Z gate."""
    # This is a simplified implementation
    # In practice, would use more sophisticated decomposition
    for i in range(len(qubits) - 1):
        circuit.cx(qubits[i], qubits[i + 1])
    circuit.z(qubits[-1])
    for i in range(len(qubits) - 2, -1, -1):
        circuit.cx(qubits[i], qubits[i + 1])

def diffusion_operator(circuit, num_qubits):
    """Grover diffusion operator (inversion about average)."""
    # Step 1: Transform to |+⟩^⊗n basis
    for qubit in range(num_qubits):
        circuit.h(qubit)
    
    # Step 2: Flip sign of |0⟩^⊗n state
    for qubit in range(num_qubits):
        circuit.x(qubit)
    
    # Multi-controlled Z gate
    if num_qubits == 1:
        circuit.z(0)
    elif num_qubits == 2:
        controlled_z_2qubit(circuit)
    else:
        multi_controlled_z(circuit, list(range(num_qubits)))
    
    for qubit in range(num_qubits):
        circuit.x(qubit)
    
    # Step 3: Transform back
    for qubit in range(num_qubits):
        circuit.h(qubit)

# Test Grover's algorithm
print("Testing Grover's algorithm on 4-item database:")
for target in range(4):
    found = grovers_algorithm(target_item=target, num_items=4)
    print()
```

**🎯 Key Insight:** Grover's algorithm quadratically speeds up unstructured search!

## Algorithm 3: Quantum Fourier Transform

### The Problem

The Quantum Fourier Transform (QFT) is the quantum version of the classical Discrete Fourier Transform, and it's the heart of many quantum algorithms like Shor's factoring algorithm.

### The Algorithm

```python
def quantum_fourier_transform(num_qubits=3):
    """
    Implement the Quantum Fourier Transform.
    
    The QFT transforms the computational basis to the Fourier basis:
    |j⟩ → (1/√N) Σₖ e^(2πijk/N) |k⟩
    """
    
    print(f"🌊 Quantum Fourier Transform ({num_qubits} qubits)")
    print("=" * 45)
    
    circuit = quantrs2.Circuit(num_qubits)
    
    print("Step 1: Prepare input state |011⟩")
    # Prepare a specific input state as example
    circuit.x(0)
    circuit.x(1)
    # qubit 2 stays |0⟩
    print("  Input: |011⟩ (binary) = |3⟩ (decimal)")
    
    print("\nStep 2: Apply QFT")
    qft_circuit(circuit, num_qubits)
    
    print("\nStep 3: Measure output")
    circuit.measure_all()
    result = circuit.run()
    
    print(f"\nQFT Result:")
    probs = result.state_probabilities()
    
    # Show top 3 most probable outcomes
    sorted_outcomes = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    print("Most probable outcomes:")
    for state, prob in sorted_outcomes[:3]:
        decimal = int(state, 2)
        print(f"  |{state}⟩ (|{decimal}⟩): {prob:.3f}")
    
    return result

def qft_circuit(circuit, num_qubits):
    """Build QFT circuit."""
    
    for target_qubit in range(num_qubits):
        print(f"  Processing qubit {target_qubit}")
        
        # Apply Hadamard
        circuit.h(target_qubit)
        
        # Apply controlled rotations
        for control_qubit in range(target_qubit + 1, num_qubits):
            rotation_angle = np.pi / (2 ** (control_qubit - target_qubit))
            print(f"    Controlled rotation: qubit {control_qubit} → {target_qubit}")
            
            # Implement controlled rotation using available gates
            controlled_rotation(circuit, control_qubit, target_qubit, rotation_angle)
    
    # Swap qubits to reverse order (QFT convention)
    print("  Swapping qubits to correct order")
    for i in range(num_qubits // 2):
        swap_qubits(circuit, i, num_qubits - 1 - i)

def controlled_rotation(circuit, control, target, angle):
    """Implement controlled rotation gate."""
    # Simplified implementation using RZ gates
    circuit.rz(target, angle/2)
    circuit.cx(control, target)
    circuit.rz(target, -angle/2)
    circuit.cx(control, target)

def swap_qubits(circuit, qubit1, qubit2):
    """Swap two qubits using CNOT gates."""
    circuit.cx(qubit1, qubit2)
    circuit.cx(qubit2, qubit1)
    circuit.cx(qubit1, qubit2)

# Demonstrate QFT
qft_result = quantum_fourier_transform(3)
print()
```

## Algorithm 4: Quantum Phase Estimation

### The Problem

Given a unitary operator U and an eigenstate |ψ⟩ such that U|ψ⟩ = e^(2πiφ)|ψ⟩, estimate the phase φ.

This is crucial for many quantum algorithms including Shor's algorithm.

### The Algorithm

```python
def quantum_phase_estimation():
    """
    Implement Quantum Phase Estimation algorithm.
    
    Estimates the phase of an eigenvalue of a unitary operator.
    """
    
    print("📐 Quantum Phase Estimation")
    print("=" * 35)
    
    # Use 3 qubits for phase estimation, 1 for eigenstate
    phase_qubits = 3
    total_qubits = phase_qubits + 1
    circuit = quantrs2.Circuit(total_qubits)
    
    # Known phase we're trying to estimate (φ = 1/4)
    true_phase = 1/4
    print(f"True phase: φ = {true_phase} = π/2")
    
    print("\nStep 1: Prepare eigenstate")
    # For this example, |+⟩ is an eigenstate of Z-rotation
    circuit.h(phase_qubits)  # Last qubit is eigenstate |+⟩
    print("  Eigenstate: |+⟩")
    
    print("\nStep 2: Create superposition in phase qubits")
    for i in range(phase_qubits):
        circuit.h(i)
    print(f"  Phase qubits in equal superposition")
    
    print("\nStep 3: Apply controlled unitaries")
    # Apply controlled U^(2^j) operations
    for j in range(phase_qubits):
        power = 2 ** j
        angle = 2 * np.pi * true_phase * power
        print(f"  Controlled U^{power}: angle = {angle:.3f}")
        
        # Controlled rotation (U = e^(iφZ))
        controlled_rotation(circuit, j, phase_qubits, angle)
    
    print("\nStep 4: Apply inverse QFT to phase qubits")
    inverse_qft_circuit(circuit, phase_qubits)
    
    print("\nStep 5: Measure phase qubits")
    circuit.measure_all()
    result = circuit.run()
    
    # Extract phase estimation
    probs = result.state_probabilities()
    
    # Get most likely measurement of phase qubits
    phase_outcomes = {}
    for state, prob in probs.items():
        phase_bits = state[:phase_qubits]  # First 3 bits are phase
        phase_outcomes[phase_bits] = phase_outcomes.get(phase_bits, 0) + prob
    
    most_likely_phase = max(phase_outcomes, key=phase_outcomes.get)
    estimated_phase_decimal = int(most_likely_phase, 2)
    estimated_phase = estimated_phase_decimal / (2 ** phase_qubits)
    
    print(f"\nResults:")
    print(f"  Measured phase bits: |{most_likely_phase}⟩")
    print(f"  Estimated phase: {estimated_phase:.3f}")
    print(f"  True phase: {true_phase:.3f}")
    print(f"  Error: {abs(estimated_phase - true_phase):.3f}")
    
    accuracy = "✅ Accurate!" if abs(estimated_phase - true_phase) < 0.1 else "❌ Needs more precision qubits"
    print(f"  {accuracy}")
    
    return estimated_phase

def inverse_qft_circuit(circuit, num_qubits):
    """Build inverse QFT circuit."""
    
    # Reverse the order of swaps first
    for i in range(num_qubits // 2):
        swap_qubits(circuit, i, num_qubits - 1 - i)
    
    # Apply QFT operations in reverse order
    for target_qubit in range(num_qubits - 1, -1, -1):
        # Reverse controlled rotations
        for control_qubit in range(num_qubits - 1, target_qubit, -1):
            rotation_angle = -np.pi / (2 ** (control_qubit - target_qubit))
            controlled_rotation(circuit, control_qubit, target_qubit, rotation_angle)
        
        # Apply Hadamard
        circuit.h(target_qubit)

# Demonstrate Quantum Phase Estimation
estimated_phase = quantum_phase_estimation()
print()
```

## Algorithm 5: Variational Quantum Eigensolver (VQE)

### The Problem

Find the ground state energy of a quantum system - crucial for quantum chemistry and materials science.

VQE is a hybrid classical-quantum algorithm perfect for near-term quantum devices.

### The Algorithm

```python
def variational_quantum_eigensolver():
    """
    Implement a simple Variational Quantum Eigensolver.
    
    Finds the minimum eigenvalue of a Hamiltonian using a parameterized circuit.
    """
    
    print("⚡ Variational Quantum Eigensolver (VQE)")
    print("=" * 45)
    
    # Simple Hamiltonian: H = Z₀ + Z₁ (ground state energy = -2)
    print("Hamiltonian: H = Z₀ + Z₁")
    print("Ground state: |11⟩ with energy = -2")
    
    def vqe_circuit(theta):
        """Parameterized quantum circuit (ansatz)."""
        circuit = quantrs2.Circuit(2)
        
        # Parameterized ansatz
        circuit.ry(0, theta[0])
        circuit.ry(1, theta[1])
        circuit.cx(0, 1)
        circuit.ry(0, theta[2])
        circuit.ry(1, theta[3])
        
        return circuit
    
    def compute_energy(theta):
        """Compute energy expectation value ⟨ψ(θ)|H|ψ(θ)⟩."""
        
        # Measure Z₀ expectation value
        circuit1 = vqe_circuit(theta)
        circuit1.measure_all()
        result1 = circuit1.run()
        
        probs1 = result1.state_probabilities()
        z0_expectation = sum(
            prob * (1 if state[0] == '0' else -1) 
            for state, prob in probs1.items()
        )
        
        # Measure Z₁ expectation value  
        circuit2 = vqe_circuit(theta)
        circuit2.measure_all()
        result2 = circuit2.run()
        
        probs2 = result2.state_probabilities()
        z1_expectation = sum(
            prob * (1 if state[1] == '0' else -1)
            for state, prob in probs2.items()
        )
        
        energy = z0_expectation + z1_expectation
        return energy
    
    print("\nStep 1: Initialize parameters randomly")
    np.random.seed(42)  # For reproducibility
    theta = np.random.uniform(0, 2*np.pi, 4)
    print(f"  Initial parameters: {theta}")
    
    print("\nStep 2: Optimize parameters classically")
    learning_rate = 0.1
    num_iterations = 20
    
    energies = []
    
    for iteration in range(num_iterations):
        # Compute energy and gradients
        current_energy = compute_energy(theta)
        energies.append(current_energy)
        
        if iteration % 5 == 0:
            print(f"  Iteration {iteration:2d}: Energy = {current_energy:.3f}")
        
        # Simple gradient descent (finite differences)
        grad = np.zeros_like(theta)
        epsilon = 0.01
        
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += epsilon
            theta_minus = theta.copy()
            theta_minus[i] -= epsilon
            
            grad[i] = (compute_energy(theta_plus) - compute_energy(theta_minus)) / (2 * epsilon)
        
        # Update parameters
        theta -= learning_rate * grad
    
    final_energy = compute_energy(theta)
    
    print(f"\nStep 3: Final results")
    print(f"  Optimized parameters: {theta}")
    print(f"  Final energy: {final_energy:.3f}")
    print(f"  Ground state energy: -2.000")
    print(f"  Error: {abs(final_energy - (-2)):.3f}")
    
    # Check final state
    final_circuit = vqe_circuit(theta)
    final_circuit.measure_all()
    final_result = final_circuit.run()
    
    print(f"\nFinal state probabilities:")
    probs = final_result.state_probabilities()
    for state in ['00', '01', '10', '11']:
        prob = probs.get(state, 0)
        print(f"  |{state}⟩: {prob:.3f}")
    
    accuracy = "✅ Converged!" if abs(final_energy - (-2)) < 0.1 else "❌ Needs more iterations"
    print(f"  {accuracy}")
    
    return final_energy

# Demonstrate VQE
final_energy = variational_quantum_eigensolver()
print()
```

## Algorithm Performance Comparison

Let's compare how these algorithms scale:

```python
def algorithm_scaling_analysis():
    """Analyze the scaling advantages of quantum algorithms."""
    
    print("📊 Quantum Algorithm Scaling Analysis")
    print("=" * 45)
    
    algorithms = [
        ("Search (Classical)", "O(N)", "Linear"),
        ("Grover's Search", "O(√N)", "Square root"),
        ("Factoring (Classical)", "O(exp(n))", "Exponential"),
        ("Shor's Algorithm", "O(n³)", "Polynomial"),
        ("Simulation (Classical)", "O(exp(n))", "Exponential"),
        ("Quantum Simulation", "O(poly(n))", "Polynomial")
    ]
    
    print("Algorithm Complexity Comparison:")
    print(f"{'Algorithm':<20} {'Complexity':<15} {'Scaling'}")
    print("-" * 50)
    
    for name, complexity, scaling in algorithms:
        print(f"{name:<20} {complexity:<15} {scaling}")
    
    print(f"\nDatabase Search Example (1 million items):")
    print(f"  Classical: ~500,000 queries")
    print(f"  Grover's:  ~1,000 queries")
    print(f"  Speedup:   ~500x faster!")
    
    print(f"\nFactoring Example (2048-bit number):")
    print(f"  Classical: >10¹⁵ years")
    print(f"  Shor's:    ~hours")
    print(f"  Speedup:   Exponential!")

algorithm_scaling_analysis()
```

## Real-World Applications

### Current Applications (NISQ Era)

```python
def nisq_applications():
    """Examples of near-term quantum applications."""
    
    print("🌍 Real-World Quantum Applications")
    print("=" * 40)
    
    applications = [
        {
            "domain": "Optimization",
            "problem": "Portfolio optimization",
            "algorithm": "QAOA (Quantum Approximate Optimization)",
            "advantage": "Better solutions for complex constraints"
        },
        {
            "domain": "Chemistry", 
            "problem": "Molecular simulation",
            "algorithm": "VQE (Variational Quantum Eigensolver)",
            "advantage": "Natural quantum system simulation"
        },
        {
            "domain": "Machine Learning",
            "problem": "Feature mapping",
            "algorithm": "Quantum kernel methods",
            "advantage": "Exponentially large feature spaces"
        },
        {
            "domain": "Cryptography",
            "problem": "Random number generation",
            "algorithm": "Quantum random sampling",
            "advantage": "True randomness from quantum mechanics"
        }
    ]
    
    for i, app in enumerate(applications, 1):
        print(f"{i}. {app['domain']}: {app['problem']}")
        print(f"   Algorithm: {app['algorithm']}")
        print(f"   Advantage: {app['advantage']}")
        print()

nisq_applications()
```

## Hands-On Challenge: Build Your Own Algorithm

```python
def quantum_random_walk():
    """
    Challenge: Implement a quantum random walk.
    
    A quantum particle can walk in superposition of directions!
    """
    
    print("🎯 Challenge: Quantum Random Walk")
    print("=" * 35)
    
    def quantum_walk_step(circuit, position_qubits, coin_qubit):
        """Single step of quantum walk."""
        
        # Quantum coin flip
        circuit.h(coin_qubit)
        
        # Conditional movement based on coin
        # If coin is |0⟩, move left (subtract 1)
        # If coin is |1⟩, move right (add 1)
        
        # This is simplified - real implementation needs
        # proper position encoding and increment/decrement
        for i in range(position_qubits):
            circuit.cx(coin_qubit, i)
    
    print("Quantum walks can spread faster than classical walks!")
    print("Try implementing position encoding and movement operators.")
    print("Challenge: Can you make the quantum walker spread quadratically?")

quantum_random_walk()
```

## Key Takeaways

🎯 **What you learned:**

1. **Deutsch's Algorithm**: Quantum parallelism in function evaluation
2. **Grover's Search**: Quadratic speedup for unstructured search  
3. **QFT**: Foundation for many quantum algorithms
4. **Phase Estimation**: Key subroutine for eigenvalue problems
5. **VQE**: Hybrid quantum-classical optimization

🚀 **Core concepts:**

- **Superposition**: Enables quantum parallelism
- **Interference**: Amplifies correct answers
- **Entanglement**: Creates quantum correlations
- **Oracle calls**: Quantum subroutines that mark solutions
- **Hybrid algorithms**: Combine quantum and classical processing

⚡ **Algorithm patterns:**

1. **Initialize**: Create superposition
2. **Oracle**: Mark target states  
3. **Amplify**: Increase success probability
4. **Measure**: Extract classical result

## Algorithm Selection Guide

Choose the right quantum algorithm:

| Problem Type | Quantum Algorithm | Classical Complexity | Quantum Advantage |
|--------------|------------------|---------------------|-------------------|
| Search unsorted database | Grover's | O(N) | O(√N) |
| Factor integers | Shor's | O(exp(n)) | O(n³) |
| Simulate quantum systems | Native simulation | O(exp(n)) | O(poly(n)) |
| Optimization | QAOA/VQE | Problem-dependent | Potential advantage |
| Linear systems | HHL | O(N) | O(log N) |

## Common Pitfalls and Best Practices

❌ **Common mistakes:**
- Not accounting for noise in NISQ devices
- Using too few qubits for phase estimation
- Forgetting to reverse bit order after QFT
- Not optimizing circuit depth for hardware

✅ **Best practices:**
- Start with small examples and scale up
- Validate against classical simulation
- Consider noise and error mitigation
- Optimize for your target hardware

## What's Next?

You've mastered the fundamentals! In the next tutorial, we'll explore how to optimize these algorithms for real quantum hardware.

**Next:** [Tutorial 4: Hardware Optimization](04-hardware-optimization.md)

## Practice Exercises

1. **Modify Grover's**: Change the oracle to search for multiple items
2. **Extend QFT**: Implement QFT for different input states
3. **Custom VQE**: Try different ansatz circuits for VQE
4. **Algorithm combination**: Use QFT in a phase estimation variant

## Additional Resources

### Research Papers
- Grover, "A fast quantum mechanical algorithm for database search" (1996)
- Shor, "Polynomial-time algorithms for prime factorization" (1994)  
- Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor" (2014)

### Implementation References
- [Qiskit Textbook](https://qiskit.org/textbook/)
- [Quantum Algorithm Zoo](https://quantumalgorithmzoo.org/)
- [Microsoft Quantum Katas](https://github.com/Microsoft/QuantumKatas)

---

**Ready for hardware optimization?** [Continue to Tutorial 4: Hardware Optimization →](04-hardware-optimization.md)

*"Quantum mechanics is certainly imposing. But an inner voice tells me that it is not yet the real thing." - Albert Einstein*

But these algorithms show it's real enough to revolutionize computing! 🚀