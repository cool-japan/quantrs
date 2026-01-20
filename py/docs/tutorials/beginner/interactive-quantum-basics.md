# Interactive Quantum Computing Basics

Welcome to the interactive quantum computing tutorial! This hands-on guide will teach you the fundamentals of quantum computing using QuantRS2.

## Tutorial Overview

- **Duration**: 45-60 minutes
- **Prerequisites**: Basic Python knowledge
- **Learning Goals**: Understand qubits, gates, and basic quantum circuits

## Interactive Setup

```python
# Install required packages for this tutorial
!pip install quantrs2 matplotlib numpy jupyter-widgets

# Import necessary modules
import quantrs2
from quantrs2 import Circuit, visualize_circuit
from quantrs2.gates import H, X, Y, Z, CNOT
import numpy as np
import matplotlib.pyplot as plt

# Enable interactive widgets
from ipywidgets import interact, widgets
from IPython.display import display, HTML
```

## Chapter 1: Understanding Qubits

### Classical vs Quantum Bits

**Classical Bit**: Can be either 0 or 1
**Quantum Bit (Qubit)**: Can be in a superposition of 0 and 1

### Interactive Demo: Qubit Visualization

```python
# Interactive qubit state visualizer
@interact(
    alpha=widgets.FloatSlider(min=0, max=1, step=0.1, value=0.7, description='α:'),
    phase=widgets.FloatSlider(min=0, max=2*np.pi, step=0.1, value=0, description='Phase:')
)
def visualize_qubit_state(alpha, phase):
    """Visualize a single qubit state on the Bloch sphere."""
    
    # Calculate beta from alpha (normalization constraint)
    beta = np.sqrt(1 - alpha**2)
    
    # Create quantum state
    state = np.array([alpha, beta * np.exp(1j * phase)])
    
    # Display state information
    print(f"Quantum State: {alpha:.2f}|0⟩ + {beta:.2f}e^(i{phase:.2f})|1⟩")
    print(f"Probability of measuring |0⟩: {alpha**2:.2f}")
    print(f"Probability of measuring |1⟩: {beta**2:.2f}")
    
    # Create Bloch sphere visualization
    from quantrs2.visualization import create_bloch_sphere_visualization
    fig = create_bloch_sphere_visualization(state)
    plt.show()
    
    return state
```

### Try It Yourself! 🎯

**Exercise 1**: Experiment with different values of α and phase:
1. Set α = 1. What happens to the qubit?
2. Set α = 0. What state is this?
3. Set α = 0.707 (≈ 1/√2). This creates an equal superposition!

```python
# Your experimentation space
alpha = 0.707  # Try different values here
phase = 0      # Try different phases here

# Your code here to create and visualize states
```

## Chapter 2: Quantum Gates

### Single-Qubit Gates

Quantum gates are operations that manipulate qubit states. Let's explore the most common ones:

```python
# Interactive gate explorer
@interact(
    gate_type=widgets.Dropdown(
        options=['I', 'X', 'Y', 'Z', 'H', 'S', 'T'],
        value='H',
        description='Gate:'
    ),
    input_state=widgets.Dropdown(
        options=[
            ('|0⟩', [1, 0]),
            ('|1⟩', [0, 1]),
            ('|+⟩ = (|0⟩+|1⟩)/√2', [1/np.sqrt(2), 1/np.sqrt(2)]),
            ('|-⟩ = (|0⟩-|1⟩)/√2', [1/np.sqrt(2), -1/np.sqrt(2)])
        ],
        value=[1, 0],
        description='Input:'
    )
)
def explore_quantum_gates(gate_type, input_state):
    """Explore how different gates transform quantum states."""
    
    # Define gate matrices
    gates = {
        'I': np.array([[1, 0], [0, 1]]),      # Identity
        'X': np.array([[0, 1], [1, 0]]),      # Pauli-X (NOT)
        'Y': np.array([[0, -1j], [1j, 0]]),   # Pauli-Y
        'Z': np.array([[1, 0], [0, -1]]),     # Pauli-Z
        'H': np.array([[1, 1], [1, -1]]) / np.sqrt(2),  # Hadamard
        'S': np.array([[1, 0], [0, 1j]]),     # Phase gate
        'T': np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])  # π/8 gate
    }
    
    # Apply gate to input state
    input_vec = np.array(input_state)
    output_vec = gates[gate_type] @ input_vec
    
    # Display transformation
    print(f"Gate: {gate_type}")
    print(f"Input:  {input_vec}")
    print(f"Output: {output_vec}")
    
    # Create side-by-side Bloch sphere visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Visualize input and output states
    from quantrs2.visualization import plot_bloch_sphere
    plot_bloch_sphere(input_vec, ax=ax1, title="Input State")
    plot_bloch_sphere(output_vec, ax=ax2, title=f"After {gate_type} Gate")
    
    plt.tight_layout()
    plt.show()
    
    return output_vec
```

### Try It Yourself! 🎯

**Exercise 2**: Gate Effects
1. Apply X gate to |0⟩. What happens?
2. Apply H gate to |0⟩. What state do you get?
3. Apply H gate twice to |0⟩. What's the result?
4. What does Z gate do to |+⟩ state?

```python
# Your experimentation space
# Try different gate and input combinations here
```

## Chapter 3: Building Quantum Circuits

### Your First Circuit

Let's build quantum circuits step by step:

```python
# Interactive circuit builder
class InteractiveCircuitBuilder:
    def __init__(self):
        self.circuit = Circuit(2)  # Start with 2 qubits
        self.step = 0
        
    def add_gate(self, gate_type, qubit=0, control=None, target=None):
        """Add a gate to the circuit."""
        self.step += 1
        
        if gate_type == 'H':
            self.circuit.h(qubit)
        elif gate_type == 'X':
            self.circuit.x(qubit)
        elif gate_type == 'Y':
            self.circuit.y(qubit)
        elif gate_type == 'Z':
            self.circuit.z(qubit)
        elif gate_type == 'CNOT':
            self.circuit.cnot(control, target)
        
        print(f"Step {self.step}: Added {gate_type} gate")
        self.visualize_current_circuit()
        
    def visualize_current_circuit(self):
        """Visualize the current circuit."""
        print("\nCurrent Circuit:")
        print(self.circuit)
        
        # Simulate and show state
        result = self.circuit.run()
        state = result.state_vector
        
        print(f"Current state: {state}")
        print("State probabilities:")
        for i, amplitude in enumerate(state):
            prob = abs(amplitude)**2
            if prob > 1e-10:  # Only show non-zero probabilities
                print(f"  |{format(i, f'0{self.circuit.n_qubits}b')}⟩: {prob:.3f}")

# Create interactive circuit builder
builder = InteractiveCircuitBuilder()

# Interactive controls
@interact(
    gate=widgets.Dropdown(
        options=['H', 'X', 'Y', 'Z', 'CNOT'],
        description='Gate:'
    ),
    qubit=widgets.Dropdown(
        options=[0, 1],
        description='Qubit:'
    ),
    control_qubit=widgets.Dropdown(
        options=[0, 1],
        description='Control:'
    ),
    target_qubit=widgets.Dropdown(
        options=[0, 1],
        description='Target:'
    )
)
def add_gate_interactive(gate, qubit, control_qubit, target_qubit):
    """Interactive gate addition."""
    if gate == 'CNOT':
        builder.add_gate(gate, control=control_qubit, target=target_qubit)
    else:
        builder.add_gate(gate, qubit=qubit)
```

### Famous Quantum Circuits

Let's implement some famous quantum circuits:

#### Bell State Circuit

```python
def create_bell_state_tutorial():
    """Interactive Bell state creation tutorial."""
    
    print("🔔 Bell State Tutorial")
    print("===================")
    print("We'll create a Bell state: (|00⟩ + |11⟩)/√2")
    print("This is a maximally entangled two-qubit state.\n")
    
    # Step 1: Initialize circuit
    circuit = Circuit(2)
    print("Step 1: Start with |00⟩ state")
    print("Initial state: |00⟩")
    
    # Step 2: Apply Hadamard to first qubit
    circuit.h(0)
    print("\nStep 2: Apply Hadamard gate to qubit 0")
    print("State becomes: (|00⟩ + |10⟩)/√2")
    
    # Visualize intermediate state
    result = circuit.run()
    print(f"Actual state: {result.state_vector}")
    
    # Step 3: Apply CNOT
    circuit.cnot(0, 1)
    print("\nStep 3: Apply CNOT gate (control: 0, target: 1)")
    print("Final state: (|00⟩ + |11⟩)/√2")
    
    # Final result
    final_result = circuit.run()
    print(f"Final state: {final_result.state_vector}")
    
    # Measure entanglement
    print("\n🎯 This is a Bell state - perfectly entangled!")
    print("If you measure qubit 0:")
    print("  - 50% chance of |0⟩ → qubit 1 is definitely |0⟩")
    print("  - 50% chance of |1⟩ → qubit 1 is definitely |1⟩")
    
    return circuit

# Run the Bell state tutorial
bell_circuit = create_bell_state_tutorial()
```

### Try It Yourself! 🎯

**Exercise 3**: Create Different Bell States
```python
# There are 4 Bell states. Can you create them all?

# Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 (we just made this one)
def bell_phi_plus():
    circuit = Circuit(2)
    # Your code here
    return circuit

# Bell state |Φ-⟩ = (|00⟩ - |11⟩)/√2
def bell_phi_minus():
    circuit = Circuit(2)
    # Your code here
    return circuit

# Bell state |Ψ+⟩ = (|01⟩ + |10⟩)/√2
def bell_psi_plus():
    circuit = Circuit(2)
    # Your code here
    return circuit

# Bell state |Ψ-⟩ = (|01⟩ - |10⟩)/√2
def bell_psi_minus():
    circuit = Circuit(2)
    # Your code here
    return circuit

# Test your implementations
```

## Chapter 4: Quantum Measurements

### Understanding Measurement

Quantum measurement collapses the superposition and gives classical outcomes:

```python
# Interactive measurement demonstration
@interact(
    n_shots=widgets.IntSlider(min=100, max=5000, step=100, value=1000, description='Shots:'),
    circuit_type=widgets.Dropdown(
        options=[
            ('Single qubit in |+⟩', 'superposition'),
            ('Bell state |Φ+⟩', 'bell'),
            ('Random 3-qubit state', 'random3')
        ],
        description='Circuit:'
    )
)
def measurement_demo(n_shots, circuit_type):
    """Demonstrate quantum measurement statistics."""
    
    # Create different circuits
    if circuit_type == 'superposition':
        circuit = Circuit(1)
        circuit.h(0)
        circuit.measure_all()
    elif circuit_type == 'bell':
        circuit = Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.measure_all()
    else:  # random3
        circuit = Circuit(3)
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.h(2)
        circuit.cnot(1, 2)
        circuit.measure_all()
    
    # Run multiple measurements
    results = []
    for _ in range(n_shots):
        result = circuit.run()
        results.append(result.measurements)
    
    # Count outcomes
    from collections import Counter
    counts = Counter(tuple(result) for result in results)
    
    # Visualize results
    outcomes = list(counts.keys())
    frequencies = list(counts.values())
    
    plt.figure(figsize=(10, 6))
    outcome_labels = [''.join(map(str, outcome)) for outcome in outcomes]
    plt.bar(outcome_labels, frequencies)
    plt.xlabel('Measurement Outcome')
    plt.ylabel('Frequency')
    plt.title(f'Measurement Results ({n_shots} shots)')
    
    # Add probability annotations
    for i, freq in enumerate(frequencies):
        probability = freq / n_shots
        plt.text(i, freq + n_shots*0.01, f'{probability:.3f}', 
                ha='center', va='bottom')
    
    plt.show()
    
    # Display statistics
    print("Measurement Statistics:")
    for outcome, count in counts.items():
        outcome_str = ''.join(map(str, outcome))
        probability = count / n_shots
        print(f"  |{outcome_str}⟩: {count}/{n_shots} = {probability:.3f}")
```

### Try It Yourself! 🎯

**Exercise 4**: Measurement Exploration
1. Run the superposition measurement many times. Do you get ~50% for each outcome?
2. For the Bell state, do you see correlations? (00 and 11 should be frequent, 01 and 10 rare)
3. How do the probabilities change with different numbers of shots?

## Chapter 5: Quantum Algorithms

### Deutsch Algorithm

Let's implement our first quantum algorithm:

```python
def deutsch_algorithm_tutorial():
    """Interactive Deutsch algorithm tutorial."""
    
    print("🧠 Deutsch Algorithm Tutorial")
    print("============================")
    print("Goal: Determine if a function f(x) is constant or balanced")
    print("Classical: Need to check f(0) AND f(1)")
    print("Quantum: Need only ONE quantum evaluation!\n")
    
    # Define oracle functions
    oracles = {
        'constant_0': lambda circuit: None,  # f(x) = 0 always
        'constant_1': lambda circuit: circuit.x(1),  # f(x) = 1 always
        'balanced_identity': lambda circuit: circuit.cnot(0, 1),  # f(x) = x
        'balanced_not': lambda circuit: [circuit.x(1), circuit.cnot(0, 1)]  # f(x) = NOT x
    }
    
    @interact(
        oracle_type=widgets.Dropdown(
            options=list(oracles.keys()),
            description='Oracle:'
        )
    )
    def run_deutsch_algorithm(oracle_type):
        """Run Deutsch algorithm with chosen oracle."""
        
        print(f"Selected oracle: {oracle_type}")
        
        # Create Deutsch algorithm circuit
        circuit = Circuit(2)
        
        # Step 1: Initialize ancilla qubit to |1⟩
        circuit.x(1)
        
        # Step 2: Create superposition
        circuit.h(0)
        circuit.h(1)
        
        print("After initialization and superposition:")
        result = circuit.run()
        print(f"State: {result.state_vector}")
        
        # Step 3: Apply oracle
        oracle_func = oracles[oracle_type]
        if oracle_func:
            if isinstance(oracle_func(circuit), list):
                pass  # Multiple operations already applied
            else:
                oracle_func(circuit)
        
        print("\nAfter oracle application:")
        result = circuit.run()
        print(f"State: {result.state_vector}")
        
        # Step 4: Apply Hadamard to first qubit
        circuit.h(0)
        
        # Step 5: Measure first qubit
        circuit.measure(0)
        
        print("\nFinal measurement of qubit 0:")
        final_result = circuit.run()
        measurement = final_result.measurements[0]
        
        # Interpret result
        if measurement == 0:
            conclusion = "CONSTANT"
        else:
            conclusion = "BALANCED"
            
        print(f"Measurement result: {measurement}")
        print(f"Algorithm conclusion: Function is {conclusion}")
        
        # Verify correctness
        expected = "CONSTANT" if "constant" in oracle_type else "BALANCED"
        print(f"Correct answer: {expected}")
        print(f"✅ Correct!" if conclusion == expected else "❌ Error!")
        
        return circuit, conclusion
    
    return run_deutsch_algorithm

# Run Deutsch algorithm tutorial
deutsch_demo = deutsch_algorithm_tutorial()
```

### Try It Yourself! 🎯

**Exercise 5**: Quantum Algorithm Challenge
1. Test all four oracle types in the Deutsch algorithm
2. Verify that the algorithm always gives the correct answer
3. How many function evaluations did we save compared to classical?

## Chapter 6: Quantum Interference

### Creating Interference Patterns

```python
# Interactive interference demonstration
@interact(
    phase_shift=widgets.FloatSlider(
        min=0, max=2*np.pi, step=0.1, value=0,
        description='Phase shift:'
    ),
    amplitude_ratio=widgets.FloatSlider(
        min=0, max=1, step=0.05, value=0.5,
        description='Amplitude ratio:'
    )
)
def quantum_interference_demo(phase_shift, amplitude_ratio):
    """Demonstrate quantum interference effects."""
    
    # Create Mach-Zehnder interferometer analog
    circuit = Circuit(1)
    
    # First beam splitter (Hadamard)
    circuit.h(0)
    print("After first beam splitter: (|0⟩ + |1⟩)/√2")
    
    # Phase shift in one path
    if phase_shift != 0:
        circuit.rz(0, phase_shift)
        print(f"After phase shift of {phase_shift:.2f} radians")
    
    # Amplitude adjustment (simulated)
    # Note: This is for educational purposes - pure quantum circuits preserve amplitudes
    
    # Second beam splitter
    circuit.h(0)
    print("After second beam splitter")
    
    # Measure the result
    result = circuit.run()
    state = result.state_vector
    
    # Calculate probabilities
    prob_0 = abs(state[0])**2
    prob_1 = abs(state[1])**2
    
    print(f"\nFinal state: {state}")
    print(f"Probability of |0⟩: {prob_0:.3f}")
    print(f"Probability of |1⟩: {prob_1:.3f}")
    
    # Visualize interference
    phases = np.linspace(0, 2*np.pi, 100)
    probs_0 = []
    probs_1 = []
    
    for phase in phases:
        test_circuit = Circuit(1)
        test_circuit.h(0)
        test_circuit.rz(0, phase)
        test_circuit.h(0)
        test_result = test_circuit.run()
        test_state = test_result.state_vector
        probs_0.append(abs(test_state[0])**2)
        probs_1.append(abs(test_state[1])**2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(phases, probs_0, label='P(|0⟩)', linewidth=2)
    plt.plot(phases, probs_1, label='P(|1⟩)', linewidth=2)
    plt.axvline(phase_shift, color='red', linestyle='--', 
                label=f'Current phase: {phase_shift:.2f}')
    plt.xlabel('Phase shift (radians)')
    plt.ylabel('Probability')
    plt.title('Quantum Interference Pattern')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return circuit
```

## Final Challenge: Build Your Own Algorithm! 🏆

Now it's time to put everything together:

```python
def build_your_algorithm_challenge():
    """Challenge: Build a quantum algorithm of your choice."""
    
    print("🏆 Final Challenge: Design Your Quantum Algorithm!")
    print("=" * 50)
    print("Choose one of these challenges or create your own:")
    print("1. Quantum random number generator")
    print("2. Quantum coin flip with adjustable bias")
    print("3. Three-qubit GHZ state preparation")
    print("4. Quantum teleportation protocol")
    print("5. Your own creative quantum circuit!")
    
    # Provide template and guidance
    template_circuit = Circuit(3)  # 3 qubits to work with
    
    print("\nTemplate circuit with 3 qubits is ready.")
    print("Available gates: H, X, Y, Z, CNOT, RZ, measure")
    print("Your circuit should demonstrate a quantum effect!")
    
    return template_circuit

# Start the challenge
challenge_circuit = build_your_algorithm_challenge()

# Your solution space:
# Add your gates here
# challenge_circuit.h(0)
# challenge_circuit.cnot(0, 1)
# ... etc

# When ready, run your circuit:
# result = challenge_circuit.run()
# print(f"Your quantum state: {result.state_vector}")
```

## Summary and Next Steps

Congratulations! 🎉 You've completed the interactive quantum computing basics tutorial.

### What You've Learned:
- ✅ Qubit states and superposition
- ✅ Quantum gates and their effects
- ✅ Building quantum circuits
- ✅ Quantum measurement and statistics
- ✅ The Deutsch algorithm
- ✅ Quantum interference effects

### Next Steps:
1. **Intermediate Tutorials**: Explore more complex algorithms like Grover's search
2. **Advanced Topics**: Learn about quantum error correction and noise models
3. **Real Hardware**: Try running your circuits on actual quantum computers
4. **Specialization**: Dive into quantum machine learning, cryptography, or optimization

### Additional Resources:
- [QuantRS2 User Guide](../user-guide/core-concepts.md)
- [Advanced Examples](../../examples/)
- [API Reference](../../api/)
- [Community Forum](https://github.com/cool-japan/quantrs/discussions)

### Share Your Results! 📤
Did you create something interesting in the final challenge? Share it with the QuantRS2 community!

---

*This tutorial was created with ❤️ for the quantum computing community. Happy quantum coding!*