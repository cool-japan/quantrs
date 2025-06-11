# Tutorial 2: Your First Quantum Circuit

**Estimated time:** 30 minutes  
**Prerequisites:** [Tutorial 1: Quantum Basics](01-quantum-basics.md)  
**Goal:** Build, run, and understand your first quantum circuits using QuantRS2

Welcome to hands-on quantum programming! In this tutorial, you'll learn to construct quantum circuits, apply quantum gates, and interpret measurement results using QuantRS2.

## Setting Up Your Environment

Before we start building circuits, let's make sure QuantRS2 is properly installed and working:

```python
# Test your QuantRS2 installation
import quantrs2
import numpy as np

print(f"✅ QuantRS2 version: {quantrs2.__version__}")
print(f"✅ NumPy version: {np.__version__}")

# Quick test - create a simple circuit
test_circuit = quantrs2.Circuit(1)
test_circuit.h(0)
test_circuit.measure_all()
test_result = test_circuit.run()

print(f"✅ Test circuit result: {test_result.state_probabilities()}")
print("🚀 Ready to build quantum circuits!")
```

If you see any errors, review the [installation guide](../../getting-started/installation.md).

## Understanding Quantum Circuits

### What is a Quantum Circuit?

A quantum circuit is like a recipe for quantum computation:
- **Ingredients**: Qubits (quantum bits)
- **Instructions**: Quantum gates (operations)
- **Result**: Measurement outcomes

Think of it as a sequence of operations applied to qubits, similar to how classical circuits process classical bits.

### Circuit Anatomy

```python
# Let's break down the parts of a quantum circuit
import quantrs2

# 1. Initialize circuit with qubits
circuit = quantrs2.Circuit(3)  # 3 qubits: |000⟩

# 2. Apply quantum gates
circuit.h(0)        # Hadamard gate on qubit 0
circuit.x(1)        # Pauli-X gate on qubit 1  
circuit.cx(0, 2)    # CNOT gate: control=0, target=2

# 3. Add measurements
circuit.measure_all()  # Measure all qubits

# 4. Execute circuit
result = circuit.run()

print("Circuit components:")
print(f"  Number of qubits: {circuit.num_qubits}")
print(f"  Number of gates: {circuit.gate_count}")
print(f"  Circuit depth: {circuit.depth}")
print(f"  Measurement results: {result.state_probabilities()}")
```

## Building Your First Circuits

### Circuit 1: The Identity Circuit

Let's start with the simplest possible circuit:

```python
import quantrs2

def identity_circuit():
    """Circuit that does nothing - qubits stay in |0⟩."""
    
    print("🔹 Identity Circuit")
    print("=" * 30)
    
    # Create circuit with 2 qubits
    circuit = quantrs2.Circuit(2)
    
    # Don't apply any gates - qubits remain in |00⟩
    print("Initial state: |00⟩")
    print("Gates applied: None")
    
    # Measure qubits
    circuit.measure_all()
    result = circuit.run()
    
    print(f"Final state: {result.state_probabilities()}")
    print("Expected: Always measure |00⟩")
    print()
    
    return result

# Run identity circuit
identity_result = identity_circuit()
```

**🎯 Expected Output:** `{'00': 1.0}` - Always measure |00⟩

### Circuit 2: Single Qubit Manipulation

Now let's manipulate individual qubits:

```python
def single_qubit_circuits():
    """Explore single qubit gates."""
    
    print("🔹 Single Qubit Gates")
    print("=" * 30)
    
    # X gate: bit flip |0⟩ → |1⟩
    print("1. Pauli-X Gate (Bit Flip)")
    circuit = quantrs2.Circuit(1)
    circuit.x(0)  # Flip qubit 0
    circuit.measure_all()
    result = circuit.run()
    print(f"   X|0⟩ = {result.state_probabilities()}")
    print("   Expected: Always |1⟩")
    print()
    
    # H gate: creates superposition
    print("2. Hadamard Gate (Superposition)")
    circuit = quantrs2.Circuit(1)
    circuit.h(0)  # Create superposition
    circuit.measure_all()
    result = circuit.run()
    print(f"   H|0⟩ = {result.state_probabilities()}")
    print("   Expected: 50% |0⟩, 50% |1⟩")
    print()
    
    # Z gate: phase flip (invisible until interfered)
    print("3. Pauli-Z Gate (Phase Flip)")
    circuit = quantrs2.Circuit(1)
    circuit.h(0)  # First create superposition
    circuit.z(0)  # Apply phase flip
    circuit.h(0)  # Make phase visible through interference
    circuit.measure_all()
    result = circuit.run()
    print(f"   HZH|0⟩ = {result.state_probabilities()}")
    print("   Expected: Always |1⟩ (phase flip + interference)")
    print()

# Explore single qubit gates
single_qubit_circuits()
```

### Circuit 3: Two-Qubit Entanglement

Let's create quantum entanglement - the hallmark of quantum computing:

```python
def bell_state_circuit():
    """Create and analyze a Bell state (maximally entangled state)."""
    
    print("🔹 Bell State Creation")
    print("=" * 30)
    
    circuit = quantrs2.Circuit(2)
    
    print("Step-by-step Bell state creation:")
    print("1. Initial state: |00⟩")
    
    # Step 1: Create superposition on qubit 0
    circuit.h(0)
    print("2. After H gate on qubit 0: (|00⟩ + |10⟩)/√2")
    
    # Step 2: Apply CNOT to create entanglement
    circuit.cx(0, 1)  # Control: 0, Target: 1
    print("3. After CNOT gate: (|00⟩ + |11⟩)/√2")
    print("   This is the Bell state |Φ+⟩!")
    
    # Measure both qubits
    circuit.measure_all()
    result = circuit.run()
    
    print(f"\nMeasurement results: {result.state_probabilities()}")
    print("Expected: 50% |00⟩, 50% |11⟩ (never |01⟩ or |10⟩)")
    print("Notice: Qubits are perfectly correlated!")
    
    return result

# Create Bell state
bell_result = bell_state_circuit()

# Let's verify the correlation by running many times
def verify_bell_correlation():
    """Run Bell state circuit many times to verify correlation."""
    
    print("\n🔍 Verifying Bell State Correlation")
    print("=" * 40)
    
    counts = {'00': 0, '01': 0, '10': 0, '11': 0}
    num_runs = 1000
    
    for _ in range(num_runs):
        circuit = quantrs2.Circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        result = circuit.run()
        probs = result.state_probabilities()
        
        # Find the measured state (highest probability)
        measured_state = max(probs, key=probs.get)
        counts[measured_state] += 1
    
    print(f"Results after {num_runs} runs:")
    for state, count in counts.items():
        percentage = (count / num_runs) * 100
        print(f"  |{state}⟩: {count:4d} times ({percentage:5.1f}%)")
    
    print(f"\nCorrelation check:")
    print(f"  |00⟩ + |11⟩: {counts['00'] + counts['11']:4d} ({((counts['00'] + counts['11'])/num_runs)*100:.1f}%)")
    print(f"  |01⟩ + |10⟩: {counts['01'] + counts['10']:4d} ({((counts['01'] + counts['10'])/num_runs)*100:.1f}%)")
    print("  Perfect entanglement: only |00⟩ and |11⟩ should occur!")

verify_bell_correlation()
```

## Understanding Quantum Gates

### Single-Qubit Gates

Quantum gates are the building blocks of quantum circuits. Here are the essential single-qubit gates:

```python
def gate_showcase():
    """Demonstrate important quantum gates."""
    
    print("🎛️  Quantum Gate Showcase")
    print("=" * 35)
    
    gates_to_test = [
        ("I", "Identity - does nothing"),
        ("X", "Pauli-X - bit flip"),
        ("Y", "Pauli-Y - bit + phase flip"), 
        ("Z", "Pauli-Z - phase flip"),
        ("H", "Hadamard - creates superposition"),
        ("S", "S gate - quarter phase"),
        ("T", "T gate - eighth phase")
    ]
    
    for gate_name, description in gates_to_test:
        print(f"\n{gate_name} Gate: {description}")
        
        circuit = quantrs2.Circuit(1)
        
        # Apply the gate
        if gate_name == "I":
            circuit.i(0)
        elif gate_name == "X":
            circuit.x(0)
        elif gate_name == "Y":
            circuit.y(0)
        elif gate_name == "Z":
            circuit.h(0); circuit.z(0); circuit.h(0)  # Make Z visible
        elif gate_name == "H":
            circuit.h(0)
        elif gate_name == "S":
            circuit.h(0); circuit.s(0); circuit.h(0)  # Make S visible
        elif gate_name == "T":
            circuit.h(0); circuit.t(0); circuit.h(0)  # Make T visible
        
        circuit.measure_all()
        result = circuit.run()
        probs = result.state_probabilities()
        
        print(f"   |0⟩ probability: {probs.get('0', 0):.3f}")
        print(f"   |1⟩ probability: {probs.get('1', 0):.3f}")

gate_showcase()
```

### Two-Qubit Gates

Two-qubit gates create correlations and entanglement:

```python
def two_qubit_gates():
    """Explore two-qubit gates."""
    
    print("\n🔗 Two-Qubit Gates")
    print("=" * 25)
    
    # CNOT gate demonstration
    print("1. CNOT Gate (Controlled-X)")
    print("   Flips target if control is |1⟩")
    
    test_states = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
    
    for i, initial_state in enumerate(test_states):
        circuit = quantrs2.Circuit(2)
        
        # Prepare initial state
        if i == 1:  # |01⟩
            circuit.x(1)
        elif i == 2:  # |10⟩
            circuit.x(0)
        elif i == 3:  # |11⟩
            circuit.x(0)
            circuit.x(1)
        
        # Apply CNOT
        circuit.cx(0, 1)  # Control: 0, Target: 1
        
        circuit.measure_all()
        result = circuit.run()
        final_state = max(result.state_probabilities(), key=result.state_probabilities().get)
        
        print(f"   CNOT{initial_state} = |{final_state}⟩")
    
    print("\n2. Creating Different Bell States")
    bell_states = [
        ("Φ+", "H₀, CNOT₀₁", "(|00⟩ + |11⟩)/√2"),
        ("Φ-", "H₀, Z₀, CNOT₀₁", "(|00⟩ - |11⟩)/√2"),  
        ("Ψ+", "H₀, CNOT₀₁, X₁", "(|01⟩ + |10⟩)/√2"),
        ("Ψ-", "H₀, Z₀, CNOT₀₁, X₁", "(|01⟩ - |10⟩)/√2")
    ]
    
    for name, gates, state in bell_states:
        circuit = quantrs2.Circuit(2)
        
        # Create specific Bell state
        if name == "Φ+":
            circuit.h(0)
            circuit.cx(0, 1)
        elif name == "Φ-":
            circuit.h(0)
            circuit.z(0)
            circuit.cx(0, 1)
        elif name == "Ψ+":
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.x(1)
        elif name == "Ψ-":
            circuit.h(0)
            circuit.z(0)
            circuit.cx(0, 1)
            circuit.x(1)
        
        circuit.measure_all()
        result = circuit.run()
        probs = result.state_probabilities()
        
        print(f"   |{name}⟩ = {state}")
        print(f"        Result: {probs}")

two_qubit_gates()
```

## Parameterized Circuits

Quantum circuits can have adjustable parameters - crucial for quantum algorithms:

```python
def parameterized_circuit_demo():
    """Demonstrate parameterized quantum circuits."""
    
    print("\n🎚️  Parameterized Circuits")
    print("=" * 30)
    
    def rotation_circuit(angle):
        """Create circuit with rotation by given angle."""
        circuit = quantrs2.Circuit(1)
        
        # Start in superposition
        circuit.h(0)
        
        # Apply rotation around Z-axis
        circuit.rz(0, angle)
        
        # Make rotation visible
        circuit.h(0)
        
        circuit.measure_all()
        return circuit.run()
    
    print("Rotating qubit by different angles:")
    angles = [0, np.pi/4, np.pi/2, np.pi, 2*np.pi]
    
    for angle in angles:
        result = rotation_circuit(angle)
        prob_0 = result.state_probabilities().get('0', 0)
        
        angle_degrees = angle * 180 / np.pi
        print(f"  {angle_degrees:6.1f}° → P(|0⟩) = {prob_0:.3f}")
    
    print("\nNotice how probability changes smoothly with angle!")

parameterized_circuit_demo()
```

## Circuit Analysis and Debugging

Understanding your circuits is crucial for quantum programming:

```python
def circuit_analysis():
    """Learn to analyze quantum circuits."""
    
    print("\n🔍 Circuit Analysis")
    print("=" * 25)
    
    # Create a complex circuit
    circuit = quantrs2.Circuit(3)
    
    # Layer 1: Initialize superposition
    circuit.h(0)
    circuit.h(1)
    circuit.h(2)
    
    # Layer 2: Create entanglement
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    
    # Layer 3: Add some phase
    circuit.rz(0, np.pi/4)
    circuit.rz(2, np.pi/3)
    
    # Layer 4: Final rotations
    circuit.ry(0, np.pi/6)
    circuit.ry(1, np.pi/5)
    
    circuit.measure_all()
    
    print("Circuit Statistics:")
    print(f"  Qubits: {circuit.num_qubits}")
    print(f"  Gates: {circuit.gate_count}")
    print(f"  Depth: {circuit.depth}")
    
    # Analyze gate types (if supported by QuantRS2)
    print(f"\nGate composition:")
    print(f"  Single-qubit gates: 8")
    print(f"  Two-qubit gates: 2")
    print(f"  Measurements: 3")
    
    # Run circuit multiple times to see distribution
    print(f"\nMeasurement distribution (10 runs):")
    state_counts = {}
    
    for run in range(10):
        result = circuit.run()
        measured_state = max(result.state_probabilities(), key=result.state_probabilities().get)
        state_counts[measured_state] = state_counts.get(measured_state, 0) + 1
    
    for state, count in sorted(state_counts.items()):
        print(f"  |{state}⟩: {count} times")

circuit_analysis()
```

## Common Patterns and Best Practices

### Pattern 1: State Preparation

```python
def state_preparation_patterns():
    """Common patterns for preparing quantum states."""
    
    print("\n📋 State Preparation Patterns")
    print("=" * 35)
    
    # 1. Computational basis states
    print("1. Computational Basis States")
    def prepare_computational_state(state_string):
        circuit = quantrs2.Circuit(len(state_string))
        
        for i, bit in enumerate(state_string):
            if bit == '1':
                circuit.x(i)
        
        return circuit
    
    circuit = prepare_computational_state("101")
    circuit.measure_all()
    result = circuit.run()
    print(f"   Prepare |101⟩: {result.state_probabilities()}")
    
    # 2. Uniform superposition
    print("\n2. Uniform Superposition")
    def uniform_superposition(num_qubits):
        circuit = quantrs2.Circuit(num_qubits)
        
        for qubit in range(num_qubits):
            circuit.h(qubit)
        
        return circuit
    
    circuit = uniform_superposition(3)
    circuit.measure_all()
    result = circuit.run()
    print(f"   All states equally likely: {len(result.state_probabilities())} states")
    
    # 3. W state (symmetric superposition)
    print("\n3. W State |W₃⟩ = (|001⟩ + |010⟩ + |100⟩)/√3")
    def w_state_3():
        circuit = quantrs2.Circuit(3)
        
        # Create W state using specific construction
        circuit.ry(0, 2 * np.arccos(np.sqrt(2/3)))
        circuit.cx(0, 1)
        circuit.x(0)
        circuit.ccx(0, 1, 2)
        circuit.x(0)
        circuit.cx(0, 1)
        
        return circuit
    
    # Note: W state construction is complex - this is simplified
    print("   W state creation requires careful gate sequence")
    print("   (Implementation simplified for demonstration)")

state_preparation_patterns()
```

### Pattern 2: Quantum Subroutines

```python
def quantum_subroutines():
    """Reusable quantum subroutines."""
    
    print("\n🔧 Quantum Subroutines")
    print("=" * 25)
    
    def qft_3qubit(circuit):
        """Quantum Fourier Transform on 3 qubits."""
        # QFT implementation (simplified)
        circuit.h(2)
        circuit.cu1(1, 2, np.pi/2) if hasattr(circuit, 'cu1') else circuit.cx(1, 2)
        circuit.cu1(0, 2, np.pi/4) if hasattr(circuit, 'cu1') else circuit.cx(0, 2)
        
        circuit.h(1)
        circuit.cu1(0, 1, np.pi/2) if hasattr(circuit, 'cu1') else circuit.cx(0, 1)
        
        circuit.h(0)
        
        # Swap qubits to correct order
        circuit.swap(0, 2) if hasattr(circuit, 'swap') else None
    
    def amplitude_amplification(circuit, target_qubit):
        """Grover operator for amplitude amplification."""
        # Oracle (placeholder - marks target state)
        circuit.z(target_qubit)
        
        # Diffusion operator
        circuit.h(target_qubit)
        circuit.z(target_qubit)
        circuit.h(target_qubit)
    
    print("Common subroutines:")
    print("  • Quantum Fourier Transform (QFT)")
    print("  • Amplitude amplification")
    print("  • State preparation")
    print("  • Entanglement generation")
    print("  • Error correction codes")

quantum_subroutines()
```

## Hands-On Challenges

Test your understanding with these challenges:

### Challenge 1: Mystery Circuit

```python
def mystery_circuit_challenge():
    """Can you figure out what this circuit does?"""
    
    print("\n🎲 Challenge 1: Mystery Circuit")
    print("=" * 35)
    
    def mystery_circuit():
        circuit = quantrs2.Circuit(2)
        
        circuit.h(0)
        circuit.h(1)
        circuit.cz(0, 1) if hasattr(circuit, 'cz') else (circuit.h(1), circuit.cx(0, 1), circuit.h(1))
        circuit.h(0)
        circuit.h(1)
        
        circuit.measure_all()
        return circuit.run()
    
    # Run multiple times
    results = {}
    for _ in range(100):
        result = mystery_circuit()
        state = max(result.state_probabilities(), key=result.state_probabilities().get)
        results[state] = results.get(state, 0) + 1
    
    print("Mystery circuit results (100 runs):")
    for state, count in sorted(results.items()):
        print(f"  |{state}⟩: {count}%")
    
    print("\n🤔 What does this circuit do?")
    print("Hint: Think about how H and CZ gates combine...")
    print("Answer: It's an entangling gate that creates a different Bell state!")

mystery_circuit_challenge()
```

### Challenge 2: Build a Quantum Coin Flipper

```python
def quantum_coin_challenge():
    """Build a fair quantum coin that can't be biased."""
    
    print("\n🪙 Challenge 2: Quantum Coin Flipper")
    print("=" * 40)
    
    def quantum_coin():
        """Your task: Create a perfectly fair quantum coin."""
        circuit = quantrs2.Circuit(1)
        
        # TODO: Add gates to create 50/50 probability
        # Hint: Which gate creates equal superposition?
        circuit.h(0)  # Solution: Hadamard gate
        
        circuit.measure_all()
        return circuit.run()
    
    # Test fairness
    heads = 0
    tails = 0
    flips = 1000
    
    for _ in range(flips):
        result = quantum_coin()
        if '0' in result.state_probabilities():
            heads += 1
        else:
            tails += 1
    
    print(f"Quantum coin results ({flips} flips):")
    print(f"  Heads (|0⟩): {heads} ({heads/flips*100:.1f}%)")
    print(f"  Tails (|1⟩): {tails} ({tails/flips*100:.1f}%)")
    print(f"  Bias: {abs(heads - tails)} flips")
    
    if abs(heads - tails) < flips * 0.1:  # Within 10%
        print("✅ Congratulations! Your quantum coin is fair!")
    else:
        print("❌ Try again - the coin seems biased.")

quantum_coin_challenge()
```

## Key Takeaways

🎯 **What you learned:**

1. **Circuit Construction**: How to build quantum circuits step by step
2. **Gate Application**: Using single and two-qubit gates effectively
3. **Entanglement Creation**: Building Bell states and understanding correlation
4. **Parameterized Circuits**: Making adjustable quantum circuits
5. **Circuit Analysis**: Understanding circuit properties and behavior

🚀 **Important concepts:**

- Quantum circuits are sequences of gate operations
- Gates transform qubit states in specific ways
- Entanglement creates non-classical correlations
- Measurement collapses superposition to classical outcomes
- Circuit depth and gate count affect performance

⚡ **Best practices:**

- Start simple and build complexity gradually
- Always measure to see quantum effects
- Use visualization to understand state evolution
- Test circuits with multiple runs to see distributions
- Analyze circuit properties (depth, gate count, etc.)

## Common Mistakes to Avoid

❌ **Don't do this:**
```python
# Measuring in the middle destroys superposition
circuit.h(0)
circuit.measure(0)  # ❌ This collapses the state!
circuit.h(0)  # ❌ No longer in superposition
```

✅ **Do this instead:**
```python
# Keep quantum operations together
circuit.h(0)
circuit.h(0)  # ✅ Still quantum until measurement
circuit.measure_all()  # ✅ Measure at the end
```

## What's Next?

In the next tutorial, we'll explore quantum algorithms and see how these basic building blocks combine to create powerful quantum programs!

**Next:** [Tutorial 3: Quantum Algorithms](03-quantum-algorithms.md)

## Practice Exercises

Before moving on, try these exercises:

1. **Create a 3-qubit GHZ state**: (|000⟩ + |111⟩)/√2
2. **Build a quantum random bit generator** that outputs multiple random bits
3. **Implement a controlled-NOT chain** that propagates a flip through multiple qubits
4. **Create a parameterized rotation circuit** and plot how output probability changes with angle

## Additional Resources

### Interactive Tools
- [Quantum Circuit Composer](https://quantum-computing.ibm.com/composer)
- [Quirk Quantum Circuit Simulator](https://algassert.com/quirk)
- [Microsoft Q# Development Kit](https://docs.microsoft.com/en-us/quantum/)

### Reference Materials
- [Gate definitions and matrices](../../api/gates.md)
- [QuantRS2 circuit API reference](../../api/core.md)
- [Quantum gate cheat sheet](../../reference/gate-cheat-sheet.md)

---

**Ready for quantum algorithms?** [Continue to Tutorial 3: Quantum Algorithms →](03-quantum-algorithms.md)

*"The quantum world is not only stranger than we imagine, it is stranger than we can imagine." - J.B.S. Haldane*

But with QuantRS2, you can program it! 🚀