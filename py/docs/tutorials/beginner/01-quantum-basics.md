# Tutorial 1: Introduction to Quantum Computing

**Estimated time:** 20 minutes  
**Prerequisites:** None  
**Goal:** Understand the fundamental concepts that make quantum computing special

Welcome to your first quantum computing tutorial! By the end of this session, you'll understand what makes quantum computers fundamentally different from classical computers and why they might revolutionize computing.

## What is Quantum Computing?

### Classical vs Quantum: A Simple Analogy

Imagine you're in a maze trying to find the exit:

**Classical Computer Approach:**
- Try one path at a time
- Remember which paths don't work
- Eventually find the exit through systematic exploration

**Quantum Computer Approach:**
- Explore ALL paths simultaneously (superposition)
- Paths can influence each other in mysterious ways (entanglement)
- "Collapse" to the correct path when you find the exit (measurement)

This is a simplified analogy, but it captures the essence: quantum computers can explore multiple possibilities simultaneously.

### The Power of Quantum

Quantum computers derive their power from three key phenomena:

1. **Superposition**: Being in multiple states at once
2. **Entanglement**: Spooky correlations between particles  
3. **Interference**: Amplifying right answers, canceling wrong ones

Let's explore each of these concepts.

## Key Concept 1: Qubits and Superposition

### Classical Bits

In classical computing, information is stored in bits:
- **Bit value**: Either 0 or 1
- **State**: Definite and measurable
- **Example**: A light switch is either ON (1) or OFF (0)

### Quantum Bits (Qubits)

Qubits are the quantum version of bits:
- **Qubit value**: Can be 0, 1, or **both simultaneously**
- **State**: Probabilistic until measured
- **Example**: A spinning coin is both heads and tails until it lands

### Mathematical Representation

A qubit state is written as:
```
|ψ⟩ = α|0⟩ + β|1⟩
```

Where:
- `α` and `β` are complex numbers (amplitudes)
- `|α|²` is the probability of measuring 0
- `|β|²` is the probability of measuring 1
- `|α|² + |β|² = 1` (probabilities sum to 1)

### Visualizing Qubits: The Bloch Sphere

Imagine a sphere where:
- **North pole**: |0⟩ state
- **South pole**: |1⟩ state  
- **Equator**: Maximum superposition states
- **Any point**: A valid qubit state

```python
# Let's see this in QuantRS2
import quantrs2
import numpy as np

# Create a single qubit
circuit = quantrs2.Circuit(1)

# Start in |0⟩ state (north pole)
print("Initial state: |0⟩")
circuit.measure_all()
result = circuit.run()
print(f"Measurement: {result.state_probabilities()}")
# Output: {'0': 1.0}

# Now let's create superposition
circuit = quantrs2.Circuit(1)
circuit.h(0)  # Hadamard gate creates superposition
print("\nAfter Hadamard gate: (|0⟩ + |1⟩)/√2")
circuit.measure_all()
result = circuit.run()
print(f"Measurement: {result.state_probabilities()}")
# Output: {'0': 0.5, '1': 0.5}
```

**🤔 Think About It:** The qubit is in superposition until we measure it. Each time we run the circuit, we get a random result, but over many runs, we see the probability pattern.

## Key Concept 2: Multiple Qubits

### Exponential State Space

With n qubits, we can represent 2ⁿ states simultaneously:
- **1 qubit**: 2 states (|0⟩, |1⟩)
- **2 qubits**: 4 states (|00⟩, |01⟩, |10⟩, |11⟩)
- **3 qubits**: 8 states
- **300 qubits**: More states than atoms in the universe!

```python
# Exploring multi-qubit states
import quantrs2

# Two qubits in superposition
circuit = quantrs2.Circuit(2)
circuit.h(0)  # First qubit in superposition
circuit.h(1)  # Second qubit in superposition

print("Two qubits in superposition:")
circuit.measure_all()
result = circuit.run()
print(f"All four states equally likely: {result.state_probabilities()}")
# Output: {'00': 0.25, '01': 0.25, '10': 0.25, '11': 0.25}
```

### Independent vs Entangled Qubits

**Independent qubits**: Each qubit has its own separate state

**Entangled qubits**: Qubits become correlated - measuring one instantly affects the other

## Key Concept 3: Quantum Entanglement

Entanglement is perhaps the most mysterious quantum phenomenon. When qubits are entangled, they become correlated in ways that seem to defy classical physics.

### Creating Entanglement: The Bell State

```python
# Creating a Bell state (maximally entangled state)
import quantrs2

circuit = quantrs2.Circuit(2)

# Step 1: Create superposition on first qubit
circuit.h(0)
print("After H gate on qubit 0: (|00⟩ + |10⟩)/√2")

# Step 2: Apply CNOT to create entanglement
circuit.cx(0, 1)  # CNOT gate: flips qubit 1 if qubit 0 is |1⟩
print("After CNOT gate: (|00⟩ + |11⟩)/√2")

circuit.measure_all()
result = circuit.run()
print(f"Bell state measurement: {result.state_probabilities()}")
# Output: {'00': 0.5, '11': 0.5}
```

**🤯 Mind-Blowing Fact:** In a Bell state, measuring qubit 0 as |0⟩ guarantees qubit 1 is also |0⟩, even if they're separated by galaxies! Einstein called this "spooky action at a distance."

### Understanding Entanglement

Key properties of entangled states:
1. **Correlation**: Measurements are correlated
2. **Non-locality**: Correlation persists regardless of distance
3. **Fragility**: Entanglement is easily destroyed by noise

```python
# Demonstrating correlation in Bell states
import quantrs2

def test_bell_state_correlation():
    """Run multiple measurements to see correlation."""
    
    results = {'00': 0, '01': 0, '10': 0, '11': 0}
    
    # Run 1000 experiments
    for _ in range(1000):
        circuit = quantrs2.Circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        result = circuit.run()
        probs = result.state_probabilities()
        
        # Find the measured state (highest probability)
        measured_state = max(probs, key=probs.get)
        results[measured_state] += 1
    
    print("Bell state correlation test (1000 runs):")
    for state, count in results.items():
        percentage = (count / 1000) * 100
        print(f"  {state}: {count} times ({percentage:.1f}%)")
    
    print("\nNotice: Only |00⟩ and |11⟩ occur - perfect correlation!")

test_bell_state_correlation()
```

## Key Concept 4: Quantum Interference

Quantum interference allows quantum computers to amplify correct answers and cancel out wrong answers.

### Constructive vs Destructive Interference

```python
# Demonstrating quantum interference
import quantrs2

print("=== Quantum Interference Demo ===\n")

# Constructive interference: HH|0⟩ = |0⟩
circuit1 = quantrs2.Circuit(1)
circuit1.h(0)  # Create superposition
circuit1.h(0)  # Second Hadamard
circuit1.measure_all()

result1 = circuit1.run()
print("Constructive interference (HH|0⟩):")
print(f"Result: {result1.state_probabilities()}")
print("Amplitudes interfere constructively → returns to |0⟩\n")

# Destructive interference: HZH|0⟩ = |1⟩
circuit2 = quantrs2.Circuit(1)
circuit2.h(0)  # Create superposition  
circuit2.z(0)  # Add phase (flip sign of |1⟩ component)
circuit2.h(0)  # Second Hadamard
circuit2.measure_all()

result2 = circuit2.run()
print("Destructive interference (HZH|0⟩):")
print(f"Result: {result2.state_probabilities()}")
print("Amplitudes interfere destructively → flips to |1⟩")
```

This interference is what allows quantum algorithms to find correct answers efficiently!

## Why Quantum Computing Matters

### Problems Quantum Computers Excel At

1. **Factoring large numbers** (Shor's algorithm)
   - Breaks current encryption
   - Threatens cybersecurity

2. **Searching unsorted databases** (Grover's algorithm)
   - Quadratic speedup over classical search
   - Applications in optimization

3. **Simulating quantum systems**
   - Drug discovery
   - Materials science
   - Chemical reactions

4. **Machine learning**
   - Quantum neural networks
   - Pattern recognition
   - Optimization problems

### Current Limitations

Quantum computers today face challenges:
- **Noise**: Quantum states are fragile
- **Limited qubits**: Current systems have 50-1000 qubits
- **Error rates**: Much higher than classical computers
- **Specialized**: Only good for specific problems

## Real-World Applications Today

### Near-term Applications (NISQ Era)

**NISQ** = Noisy Intermediate-Scale Quantum

1. **Optimization**: 
   - Portfolio optimization
   - Route planning
   - Supply chain management

2. **Chemistry simulation**:
   - Molecular modeling
   - Catalyst design
   - Drug discovery

3. **Machine learning**:
   - Feature mapping
   - Kernel methods
   - Quantum-enhanced neural networks

### Future Applications

1. **Cryptography**: Quantum-safe encryption
2. **Artificial Intelligence**: Quantum machine learning
3. **Climate modeling**: Complex system simulation
4. **Financial modeling**: Risk analysis and derivatives pricing

## Hands-On: Your First Quantum Experiment

Let's implement a simple quantum random number generator:

```python
import quantrs2
import numpy as np

def quantum_random_number_generator(num_bits=4):
    """Generate truly random numbers using quantum mechanics."""
    
    print(f"🎲 Quantum Random Number Generator ({num_bits} bits)")
    print("=" * 50)
    
    # Create circuit with num_bits qubits
    circuit = quantrs2.Circuit(num_bits)
    
    # Put all qubits in superposition
    for qubit in range(num_bits):
        circuit.h(qubit)
    
    print(f"Created {num_bits} qubits in superposition")
    print("Each qubit has 50% chance of being 0 or 1")
    
    # Measure all qubits
    circuit.measure_all()
    result = circuit.run()
    
    # Get the measured state
    probs = result.state_probabilities()
    measured_state = max(probs, key=probs.get)
    
    # Convert binary string to decimal
    random_number = int(measured_state, 2)
    
    print(f"\n📊 Measurement result: |{measured_state}⟩")
    print(f"🔢 Random number: {random_number}")
    print(f"📈 Range: 0 to {2**num_bits - 1}")
    
    return random_number

# Generate quantum random numbers
print("Generating 5 quantum random numbers:\n")
for i in range(5):
    number = quantum_random_number_generator(4)
    print()
```

**🎯 Try This:** Run the code multiple times. Notice how you get different random numbers each time? That's quantum randomness - truly unpredictable!

## Conceptual Checkpoints

Test your understanding:

### Checkpoint 1: Superposition
**Question**: A qubit in superposition is:
- [ ] Always measuring 0
- [ ] Always measuring 1  
- [x] Has probabilities of measuring both 0 and 1
- [ ] Broken

### Checkpoint 2: Entanglement
**Question**: In a Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2, if I measure the first qubit as 1, the second qubit will be:
- [ ] Definitely 0
- [x] Definitely 1
- [ ] 50% chance of 0 or 1
- [ ] Unmeasurable

### Checkpoint 3: Quantum Advantage  
**Question**: Quantum computers are faster than classical computers for:
- [ ] All problems
- [ ] No problems (they're just hype)
- [x] Specific problems like factoring and search
- [ ] Only addition and subtraction

## Key Takeaways

🎯 **What you learned:**

1. **Qubits** can exist in superposition of 0 and 1 simultaneously
2. **Multiple qubits** create exponentially large state spaces
3. **Entanglement** creates mysterious correlations between qubits
4. **Interference** allows amplification of correct answers
5. **Quantum algorithms** provide speedups for specific problems

🚀 **Why it matters:**

- Quantum computing isn't just faster - it's fundamentally different
- Some problems become exponentially easier to solve
- We're in the early days of a computing revolution
- Understanding quantum concepts opens new possibilities

## What's Next?

In the next tutorial, we'll get hands-on with QuantRS2 and build your first quantum circuits from scratch!

**Next:** [Tutorial 2: Your First Quantum Circuit →](02-first-circuit.md)

## Additional Resources

### Videos
- "Quantum Computing Explained" by MinutePhysics
- IBM Qiskit Textbook videos
- Microsoft Quantum Development Kit tutorials

### Interactive Demos
- IBM Quantum Experience
- Quirk (quantum circuit simulator)
- Quantum Game with Photons

### Books for Deeper Understanding
- "Quantum Computing: An Applied Approach" by Hidary
- "Programming Quantum Computers" by Johnston, Harrigan, and Gimeno-Segovia
- "Quantum Computation and Quantum Information" by Nielsen and Chuang (advanced)

---

**Ready for hands-on coding?** [Continue to Tutorial 2: Your First Quantum Circuit →](02-first-circuit.md)

*"If you think you understand quantum mechanics, you don't understand quantum mechanics." - Richard Feynman*

But don't worry - you understand enough to start building quantum programs! 🚀