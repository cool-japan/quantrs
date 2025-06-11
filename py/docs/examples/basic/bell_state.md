# Bell State Preparation

**Level:** 🟢 Beginner  
**Runtime:** < 1 second  
**Topics:** Superposition, Entanglement, Bell states  

Learn to create and analyze quantum entanglement using the famous Bell states - the foundation of quantum communication and quantum computing.

## What are Bell States?

Bell states are maximally entangled two-qubit quantum states, discovered by physicist John Stewart Bell. They demonstrate "spooky action at a distance" - when you measure one qubit, you instantly know the state of the other, regardless of distance.

The four Bell states are:
- |Φ⁺⟩ = (|00⟩ + |11⟩)/√2  ← **We'll create this one**
- |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
- |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2  
- |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2

## Implementation

### Basic Bell State (|Φ⁺⟩)

```python
import quantrs2
import numpy as np

def create_bell_state():
    """Create the |Φ⁺⟩ Bell state: (|00⟩ + |11⟩)/√2"""
    
    print("🔬 Creating Bell State |Φ⁺⟩")
    print("=" * 40)
    
    # Create 2-qubit circuit
    circuit = quantrs2.Circuit(2)
    
    print("Step 1: Initial state |00⟩")
    
    # Step 1: Create superposition on first qubit
    circuit.h(0)
    print("Step 2: Apply Hadamard → (|00⟩ + |10⟩)/√2")
    
    # Step 2: Apply CNOT to create entanglement
    circuit.cx(0, 1)
    print("Step 3: Apply CNOT → (|00⟩ + |11⟩)/√2")
    
    # Measure the state
    circuit.measure_all()
    result = circuit.run()
    
    print("\n📊 Results:")
    probabilities = result.state_probabilities()
    
    for state, prob in probabilities.items():
        if prob > 1e-10:  # Only show non-zero probabilities
            print(f"  |{state}⟩: {prob:.3f} ({prob*100:.1f}%)")
    
    # Verify it's a proper Bell state
    print("\n✅ Verification:")
    expected_states = ['00', '11']
    measured_states = [state for state, prob in probabilities.items() if prob > 0.1]
    
    if set(measured_states) == set(expected_states):
        print("  ✓ Only |00⟩ and |11⟩ states observed")
        
        # Check if probabilities are approximately equal
        prob_00 = probabilities.get('00', 0)
        prob_11 = probabilities.get('11', 0)
        if abs(prob_00 - prob_11) < 0.1:
            print("  ✓ Equal probabilities for |00⟩ and |11⟩")
            print("  ✓ Perfect Bell state created! 🎉")
        else:
            print("  ⚠ Unequal probabilities - check for errors")
    else:
        print("  ❌ Unexpected states measured")
    
    return circuit, result

# Create the Bell state
bell_circuit, bell_result = create_bell_state()
```

### All Four Bell States

```python
def create_all_bell_states():
    """Create all four Bell states and compare them."""
    
    print("\n🌟 All Four Bell States")
    print("=" * 30)
    
    bell_states = [
        ("Φ⁺", "H₀, CNOT₀₁", "(|00⟩ + |11⟩)/√2"),
        ("Φ⁻", "H₀, Z₀, CNOT₀₁", "(|00⟩ - |11⟩)/√2"),
        ("Ψ⁺", "H₀, CNOT₀₁, X₁", "(|01⟩ + |10⟩)/√2"),
        ("Ψ⁻", "H₀, Z₀, CNOT₀₁, X₁", "(|01⟩ - |10⟩)/√2")
    ]
    
    results = {}
    
    for name, gates, description in bell_states:
        print(f"\n{name} State: {description}")
        print(f"Circuit: {gates}")
        
        circuit = quantrs2.Circuit(2)
        
        # Create specific Bell state
        if name == "Φ⁺":
            circuit.h(0)
            circuit.cx(0, 1)
        elif name == "Φ⁻":
            circuit.h(0)
            circuit.z(0)
            circuit.cx(0, 1)
        elif name == "Ψ⁺":
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.x(1)
        elif name == "Ψ⁻":
            circuit.h(0)
            circuit.z(0)
            circuit.cx(0, 1)
            circuit.x(1)
        
        circuit.measure_all()
        result = circuit.run()
        results[name] = result
        
        # Show probabilities
        probs = result.state_probabilities()
        print("Probabilities:")
        for state in ['00', '01', '10', '11']:
            prob = probs.get(state, 0)
            if prob > 1e-10:
                print(f"  |{state}⟩: {prob:.3f}")
    
    return results

# Create all Bell states
all_bell_results = create_all_bell_states()
```

### Demonstrating Entanglement

```python
def demonstrate_entanglement():
    """Demonstrate the 'spooky' correlation in Bell states."""
    
    print("\n👻 Demonstrating Quantum Entanglement")
    print("=" * 45)
    
    # Run many measurements to see correlation
    num_experiments = 1000
    correlations = {'00': 0, '01': 0, '10': 0, '11': 0}
    
    print(f"Running {num_experiments} Bell state measurements...")
    
    for experiment in range(num_experiments):
        # Create fresh Bell state for each measurement
        circuit = quantrs2.Circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        result = circuit.run()
        probs = result.state_probabilities()
        
        # Find the measured outcome (highest probability)
        measured_state = max(probs, key=probs.get)
        correlations[measured_state] += 1
    
    print(f"\nResults after {num_experiments} experiments:")
    for state, count in correlations.items():
        percentage = (count / num_experiments) * 100
        print(f"  |{state}⟩: {count:4d} times ({percentage:5.1f}%)")
    
    # Analyze correlations
    same_bits = correlations['00'] + correlations['11']
    different_bits = correlations['01'] + correlations['10']
    
    print(f"\nCorrelation Analysis:")
    print(f"  Same bits (|00⟩ + |11⟩): {same_bits:4d} ({same_bits/num_experiments*100:.1f}%)")
    print(f"  Different bits (|01⟩ + |10⟩): {different_bits:4d} ({different_bits/num_experiments*100:.1f}%)")
    
    if same_bits > 0.8 * num_experiments:
        print("  ✅ Strong quantum correlation observed!")
        print("  🎭 The qubits are perfectly entangled")
    else:
        print("  ⚠ Weak correlation - check circuit or increase measurements")
    
    return correlations

# Demonstrate entanglement
entanglement_results = demonstrate_entanglement()
```

### Bell State Fidelity

```python
def measure_bell_state_fidelity():
    """Measure how close our Bell state is to the ideal."""
    
    print("\n📏 Bell State Fidelity Measurement")
    print("=" * 40)
    
    # Create Bell state
    circuit = quantrs2.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    
    result = circuit.run()
    measured_probs = result.state_probabilities()
    
    # Ideal Bell state probabilities
    ideal_probs = {'00': 0.5, '01': 0.0, '10': 0.0, '11': 0.5}
    
    print("Probability comparison:")
    print(f"{'State':<6} {'Ideal':<8} {'Measured':<10} {'Difference'}")
    print("-" * 35)
    
    total_fidelity = 0
    for state in ['00', '01', '10', '11']:
        ideal = ideal_probs[state]
        measured = measured_probs.get(state, 0)
        difference = abs(ideal - measured)
        
        print(f"|{state}⟩    {ideal:<8.3f} {measured:<10.3f} {difference:<10.3f}")
        
        # Classical fidelity calculation
        total_fidelity += np.sqrt(ideal * measured)
    
    print(f"\nFidelity: {total_fidelity:.4f}")
    
    if total_fidelity > 0.99:
        print("🌟 Excellent! Near-perfect Bell state")
    elif total_fidelity > 0.95:
        print("✅ Good Bell state quality")
    elif total_fidelity > 0.90:
        print("⚠ Acceptable, but room for improvement")
    else:
        print("❌ Poor fidelity - check implementation")
    
    return total_fidelity

# Measure fidelity
fidelity = measure_bell_state_fidelity()
```

## Understanding the Circuit

### Step-by-Step Analysis

```python
def analyze_bell_state_step_by_step():
    """Analyze what happens at each step of Bell state creation."""
    
    print("\n🔍 Step-by-Step Bell State Analysis")
    print("=" * 45)
    
    # Step 1: Initial state
    print("Step 1: Initial state |00⟩")
    circuit1 = quantrs2.Circuit(2)
    circuit1.measure_all()
    result1 = circuit1.run()
    print(f"  State: {result1.state_probabilities()}")
    
    # Step 2: After Hadamard
    print("\nStep 2: After Hadamard on qubit 0")
    circuit2 = quantrs2.Circuit(2)
    circuit2.h(0)
    circuit2.measure_all()
    result2 = circuit2.run()
    print(f"  State: {result2.state_probabilities()}")
    print("  Qubit 0 is in superposition, qubit 1 is still |0⟩")
    
    # Step 3: After CNOT (Bell state)
    print("\nStep 3: After CNOT (final Bell state)")
    circuit3 = quantrs2.Circuit(2)
    circuit3.h(0)
    circuit3.cx(0, 1)
    circuit3.measure_all()
    result3 = circuit3.run()
    print(f"  State: {result3.state_probabilities()}")
    print("  Qubits are now entangled!")
    
    # Explain what happened
    print("\n🧠 What Happened:")
    print("  1. Started with separable state |00⟩")
    print("  2. Hadamard created superposition: (|00⟩ + |10⟩)/√2")
    print("  3. CNOT entangled the qubits: (|00⟩ + |11⟩)/√2")
    print("  4. Now measuring one qubit determines the other!")

# Run step-by-step analysis
analyze_bell_state_step_by_step()
```

### Circuit Properties

```python
def analyze_circuit_properties():
    """Analyze properties of the Bell state circuit."""
    
    print("\n📊 Bell State Circuit Properties")
    print("=" * 35)
    
    # Create Bell state circuit
    circuit = quantrs2.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    
    print(f"Circuit Properties:")
    print(f"  Number of qubits: {circuit.num_qubits}")
    print(f"  Number of gates: {circuit.gate_count}")
    print(f"  Circuit depth: {circuit.depth}")
    
    # Gate breakdown
    gate_types = circuit.gate_type_counts()
    print(f"\nGate breakdown:")
    for gate_type, count in gate_types.items():
        print(f"  {gate_type}: {count}")
    
    print(f"\nCircuit characteristics:")
    print(f"  ✓ Minimal depth (only 2 layers)")
    print(f"  ✓ Uses only 2 gates")
    print(f"  ✓ Creates maximum entanglement")
    print(f"  ✓ Hardware-efficient implementation")
    
    return circuit

# Analyze circuit properties
bell_properties = analyze_circuit_properties()
```

## Exercises and Extensions

### Exercise 1: Different Bell States
Try creating the other three Bell states and verify their properties:

```python
def exercise_bell_variants():
    """Exercise: Create and test different Bell state variants."""
    
    print("\n🎯 Exercise: Bell State Variants")
    print("=" * 35)
    
    # TODO: Implement Φ⁻ state
    # Hint: Add a Z gate before the CNOT
    
    # TODO: Implement Ψ⁺ state  
    # Hint: Add an X gate after the CNOT
    
    # TODO: Implement Ψ⁻ state
    # Hint: Combine Z and X modifications
    
    print("Your turn! Try implementing the other Bell states.")

exercise_bell_variants()
```

### Exercise 2: Bell State Measurement
Implement different measurement strategies:

```python
def exercise_measurement_bases():
    """Exercise: Measure Bell states in different bases."""
    
    print("\n🎯 Exercise: Measurement Bases")
    print("=" * 32)
    
    # TODO: Measure in X basis (add H before measurement)
    # TODO: Measure in Y basis (add S†H before measurement)
    # TODO: Compare results with Z basis measurement
    
    print("Try measuring Bell states in X and Y bases!")

exercise_measurement_bases()
```

### Exercise 3: Bell Inequality
Test Bell's inequality to prove quantum non-locality:

```python
def exercise_bell_inequality():
    """Exercise: Test Bell's inequality."""
    
    print("\n🎯 Exercise: Bell's Inequality")
    print("=" * 30)
    
    # TODO: Implement CHSH inequality test
    # TODO: Measure correlations at different angles
    # TODO: Show violation of classical bound (2.0)
    
    print("Implement CHSH inequality test!")
    print("Quantum mechanics can violate the classical bound of 2!")

exercise_bell_inequality()
```

## Real-World Applications

Bell states are the foundation for many quantum technologies:

### Quantum Communication
- **Quantum Key Distribution**: Secure communication using entangled photons
- **Quantum Internet**: Distributed quantum networks
- **Quantum Teleportation**: Transferring quantum information

### Quantum Computing
- **Quantum Error Correction**: Entangled ancilla qubits for error detection
- **Quantum Algorithms**: Many algorithms use Bell states as subroutines
- **Quantum Supremacy**: Demonstrating quantum advantage

### Quantum Sensing
- **Quantum Interferometry**: Enhanced precision measurements
- **Quantum Metrology**: Ultra-precise atomic clocks
- **Quantum Radar**: Enhanced detection capabilities

## Performance Notes

```python
def performance_analysis():
    """Analyze performance characteristics of Bell state preparation."""
    
    print("\n⚡ Performance Analysis")
    print("=" * 25)
    
    import time
    
    # Time circuit creation
    start_time = time.time()
    circuit = quantrs2.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    creation_time = time.time() - start_time
    
    # Time circuit execution
    start_time = time.time()
    result = circuit.run()
    execution_time = time.time() - start_time
    
    print(f"Performance metrics:")
    print(f"  Circuit creation: {creation_time*1000:.2f} ms")
    print(f"  Circuit execution: {execution_time*1000:.2f} ms")
    print(f"  Total runtime: {(creation_time + execution_time)*1000:.2f} ms")
    
    print(f"\nScaling notes:")
    print(f"  ✓ O(1) gates - constant complexity")
    print(f"  ✓ Works on any quantum device")
    print(f"  ✓ Minimal noise exposure")

performance_analysis()
```

## Common Mistakes and Troubleshooting

### Mistake 1: Forgetting the Hadamard
```python
# ❌ Wrong: Missing Hadamard gate
circuit = quantrs2.Circuit(2)
circuit.cx(0, 1)  # This just copies |0⟩ to both qubits

# ✅ Correct: Hadamard first, then CNOT
circuit = quantrs2.Circuit(2)
circuit.h(0)      # Create superposition first
circuit.cx(0, 1)  # Then entangle
```

### Mistake 2: Wrong CNOT Direction
```python
# ❌ Wrong: CNOT in wrong direction
circuit.h(1)      # Hadamard on qubit 1
circuit.cx(0, 1)  # But control is qubit 0

# ✅ Correct: Match Hadamard and control
circuit.h(0)      # Hadamard on qubit 0
circuit.cx(0, 1)  # Control is also qubit 0
```

### Mistake 3: Measuring Too Early
```python
# ❌ Wrong: Measuring destroys superposition
circuit.h(0)
circuit.measure(0)  # Collapses the state!
circuit.cx(0, 1)    # No longer creates entanglement

# ✅ Correct: Measure after all quantum operations
circuit.h(0)
circuit.cx(0, 1)
circuit.measure_all()  # Measure at the end
```

## References and Further Reading

### Foundational Papers
- Bell, J.S. (1964). "On the Einstein Podolsky Rosen Paradox"
- Aspect, A. et al. (1982). "Experimental Test of Bell's Inequalities"

### Modern Applications
- Quantum Key Distribution protocols
- Quantum teleportation experiments
- Quantum network implementations

### Educational Resources
- Nielsen & Chuang: "Quantum Computation and Quantum Information"
- Preskill's Quantum Computing Course Notes
- IBM Qiskit Textbook: Entanglement chapter

## Summary

🎉 **Congratulations!** You've learned to:
- Create Bell states using Hadamard and CNOT gates
- Understand quantum entanglement and its properties
- Measure and verify Bell state correlations
- Analyze circuit properties and performance
- Recognize common implementation mistakes

Bell states are your gateway to quantum computing. Master them, and you're ready for quantum teleportation, quantum cryptography, and advanced quantum algorithms!

**Next Steps:**
- Try the [Quantum Teleportation](../basic/teleportation.md) example
- Explore [Grover's Algorithm](../algorithms/grover.md)
- Learn about [Quantum Error Correction](../research/error_correction.md)

---

*"God does not play dice with the universe, but something strange is going on with the dice." - Stephen Hawking*

🚀 **Ready for more quantum adventures?** Explore the [Example Gallery](../index.md)!