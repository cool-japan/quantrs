# Tutorial 5: Quantum Machine Learning Fundamentals

**Estimated time:** 50 minutes  
**Prerequisites:** [Tutorial 4: Hardware Optimization](04-hardware-optimization.md)  
**Goal:** Learn the principles and implementation of quantum machine learning algorithms

Quantum Machine Learning (QML) combines the power of quantum computing with machine learning to potentially achieve exponential speedups for certain learning tasks. In this tutorial, you'll implement fundamental QML algorithms and understand their advantages.

## Introduction to Quantum Machine Learning

### What is Quantum Machine Learning?

**Classical Machine Learning:**
```
Data → Feature extraction → Classical algorithm → Prediction
```

**Quantum Machine Learning:**
```
Data → Quantum encoding → Quantum algorithm → Measurement → Prediction
```

### Key Advantages

1. **Exponential state space**: n qubits can represent 2ⁿ states
2. **Quantum parallelism**: Process multiple data points simultaneously
3. **Quantum interference**: Amplify useful patterns, suppress noise
4. **Entanglement**: Capture complex correlations in data

```python
import quantrs2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

def qml_advantages_demo():
    """Demonstrate quantum advantages for machine learning."""
    
    print("🧠 Quantum Machine Learning Advantages")
    print("=" * 45)
    
    # State space comparison
    print("1. Exponential State Space")
    classical_bits = 10
    quantum_qubits = 10
    
    classical_states = 2 ** classical_bits
    quantum_states = 2 ** quantum_qubits
    
    print(f"  {classical_bits} classical bits: {classical_states:,} states")
    print(f"  {quantum_qubits} qubits: {quantum_states:,} states in superposition")
    print(f"  Same hardware, exponentially larger feature space!")
    
    # Feature space expansion
    print(f"\n2. Quantum Feature Maps")
    data_dimensions = [1, 2, 4, 8]
    
    for d in data_dimensions:
        classical_features = d
        quantum_features = 2 ** d  # Exponential expansion
        
        print(f"  {d}D data → {classical_features} classical features")
        print(f"          → {quantum_features} quantum features")
    
    print(f"\n3. Parallel Processing")
    print("  Classical: Process one data point at a time")  
    print("  Quantum: Process multiple data points in superposition")

qml_advantages_demo()
print()
```

## Core Concept 1: Quantum Data Encoding

### Amplitude Encoding

Encode classical data into quantum amplitudes:

```python
def amplitude_encoding_demo():
    """Demonstrate amplitude encoding of classical data."""
    
    print("📊 Amplitude Encoding")
    print("=" * 25)
    
    # Classical data vector
    classical_data = np.array([0.5, 0.3, 0.2, 0.1])
    
    print(f"Classical data: {classical_data}")
    
    # Normalize for quantum amplitudes
    normalized_data = classical_data / np.linalg.norm(classical_data)
    
    print(f"Normalized: {normalized_data}")
    print(f"Probability interpretation:")
    
    for i, amplitude in enumerate(normalized_data):
        probability = amplitude ** 2
        print(f"  |{i:02b}⟩: amplitude = {amplitude:.3f}, probability = {probability:.3f}")
    
    # Create quantum state with these amplitudes
    def create_amplitude_encoded_circuit(amplitudes):
        """Create circuit that encodes amplitudes."""
        
        # For 4 amplitudes, we need 2 qubits
        circuit = quantrs2.Circuit(2)
        
        # This is a simplified example - real amplitude encoding
        # requires more sophisticated state preparation
        
        # Prepare a state with desired amplitudes (approximation)
        angle1 = 2 * np.arccos(np.sqrt(amplitudes[0]**2 + amplitudes[1]**2))
        angle2 = 2 * np.arccos(amplitudes[0] / np.sqrt(amplitudes[0]**2 + amplitudes[1]**2))
        
        circuit.ry(0, angle1)
        circuit.cry(1, 0, angle2)
        
        return circuit
    
    circuit = create_amplitude_encoded_circuit(normalized_data)
    circuit.measure_all()
    result = circuit.run()
    
    print(f"\nQuantum encoding result:")
    probs = result.state_probabilities()
    for state, prob in probs.items():
        print(f"  |{state}⟩: {prob:.3f}")

amplitude_encoding_demo()
print()
```

### Angle Encoding

Encode classical data into rotation angles:

```python
def angle_encoding_demo():
    """Demonstrate angle encoding of classical data."""
    
    print("🔄 Angle Encoding")
    print("=" * 20)
    
    # Classical data points
    data_points = [0.2, 0.5, 0.8, 1.0]
    
    print("Encoding data into qubit rotation angles:")
    
    for i, data_point in enumerate(data_points):
        print(f"\nData point {i+1}: {data_point}")
        
        # Encode as rotation angle
        angle = data_point * np.pi  # Scale to [0, π]
        
        circuit = quantrs2.Circuit(1)
        circuit.ry(0, angle)
        circuit.measure_all()
        
        result = circuit.run()
        prob_0 = result.state_probabilities().get('0', 0)
        prob_1 = result.state_probabilities().get('1', 0)
        
        print(f"  Angle: {angle:.3f} radians")
        print(f"  P(|0⟩): {prob_0:.3f}, P(|1⟩): {prob_1:.3f}")
        print(f"  Encoded as: cos²({angle:.3f}/2) and sin²({angle:.3f}/2)")

def multi_dimensional_encoding():
    """Encode multi-dimensional data."""
    
    print(f"\n🎯 Multi-Dimensional Encoding")
    print("=" * 35)
    
    # 2D data point
    data_2d = np.array([0.3, 0.7])
    
    print(f"2D data point: {data_2d}")
    
    # Encode into 2-qubit state
    circuit = quantrs2.Circuit(2)
    
    # Encode each dimension into a qubit
    circuit.ry(0, data_2d[0] * np.pi)
    circuit.ry(1, data_2d[1] * np.pi)
    
    # Add entanglement to capture correlations
    circuit.cx(0, 1)
    
    circuit.measure_all()
    result = circuit.run()
    
    print(f"Quantum encoded state:")
    probs = result.state_probabilities()
    for state, prob in probs.items():
        print(f"  |{state}⟩: {prob:.3f}")

angle_encoding_demo()
multi_dimensional_encoding()
print()
```

## Core Concept 2: Quantum Feature Maps

### Pauli Feature Maps

Transform data into quantum feature space using Pauli operators:

```python
def pauli_feature_map():
    """Implement Pauli feature maps for data transformation."""
    
    print("🗺️  Pauli Feature Maps")
    print("=" * 25)
    
    def create_pauli_feature_map(data_point, num_qubits=2):
        """Create Pauli feature map circuit."""
        
        circuit = quantrs2.Circuit(num_qubits)
        
        # First layer: Hadamard gates for superposition
        print(f"Layer 1: Initialize superposition")
        for i in range(num_qubits):
            circuit.h(i)
        
        # Second layer: Data-dependent rotations
        print(f"Layer 2: Data-dependent rotations")
        for i in range(num_qubits):
            # Encode data into rotation
            angle = data_point[i % len(data_point)] * 2 * np.pi
            circuit.rz(i, angle)
            print(f"  RZ({i}, {angle:.3f})")
        
        # Third layer: Entangling gates
        print(f"Layer 3: Entangling gates")
        for i in range(num_qubits - 1):
            circuit.cx(i, i + 1)
        
        # Repeat for depth
        print(f"Layer 4: Repeat data encoding")
        for i in range(num_qubits):
            angle = data_point[i % len(data_point)] * np.pi
            circuit.rz(i, angle)
        
        return circuit
    
    # Example data point
    data = np.array([0.5, 0.3])
    
    print(f"Input data: {data}")
    print(f"Creating Pauli feature map:")
    
    circuit = create_pauli_feature_map(data)
    circuit.measure_all()
    result = circuit.run()
    
    print(f"\nFeature map output:")
    probs = result.state_probabilities()
    for state, prob in probs.items():
        print(f"  |{state}⟩: {prob:.3f}")
    
    return circuit

pauli_circuit = pauli_feature_map()
print()
```

### ZZ Feature Maps

Create interactions between data features:

```python
def zz_feature_map():
    """Implement ZZ feature maps with feature interactions."""
    
    print("⚡ ZZ Feature Maps")
    print("=" * 20)
    
    def create_zz_feature_map(data_point, num_qubits=3):
        """Create ZZ feature map with pairwise interactions."""
        
        circuit = quantrs2.Circuit(num_qubits)
        
        # Layer 1: Hadamard initialization
        for i in range(num_qubits):
            circuit.h(i)
        
        # Layer 2: Single-qubit data encoding
        for i in range(num_qubits):
            angle = data_point[i % len(data_point)] * 2 * np.pi
            circuit.rz(i, angle)
        
        # Layer 3: ZZ interactions (pairwise feature interactions)
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                # ZZ interaction: exp(i * φ * Z_i ⊗ Z_j)
                interaction_angle = (data_point[i % len(data_point)] * 
                                   data_point[j % len(data_point)] * np.pi)
                
                # Implement ZZ gate using CNOT + RZ + CNOT
                circuit.cx(i, j)
                circuit.rz(j, interaction_angle)
                circuit.cx(i, j)
                
                print(f"  ZZ interaction: qubits {i}-{j}, angle {interaction_angle:.3f}")
        
        return circuit
    
    # Multi-dimensional data
    data = np.array([0.4, 0.6, 0.2])
    
    print(f"Input data: {data}")
    print(f"Creating ZZ feature map with pairwise interactions:")
    
    circuit = create_zz_feature_map(data)
    circuit.measure_all()
    result = circuit.run()
    
    print(f"\nZZ feature map output:")
    probs = result.state_probabilities()
    
    # Show top 4 most probable states
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for state, prob in sorted_probs[:4]:
        print(f"  |{state}⟩: {prob:.3f}")

zz_feature_map()
print()
```

## Algorithm 1: Quantum Support Vector Machine

### Classical vs Quantum SVM

```python
def quantum_svm_demo():
    """Demonstrate quantum support vector machine."""
    
    print("🎯 Quantum Support Vector Machine")
    print("=" * 40)
    
    # Generate simple classification dataset
    def generate_classification_data(n_samples=8):
        """Generate simple 2D classification data."""
        
        np.random.seed(42)
        
        # Class 0: Points around (0.3, 0.3)
        class_0 = np.random.normal([0.3, 0.3], 0.1, (n_samples//2, 2))
        labels_0 = np.zeros(n_samples//2)
        
        # Class 1: Points around (0.7, 0.7)  
        class_1 = np.random.normal([0.7, 0.7], 0.1, (n_samples//2, 2))
        labels_1 = np.ones(n_samples//2)
        
        # Combine data
        X = np.vstack([class_0, class_1])
        y = np.hstack([labels_0, labels_1])
        
        # Normalize to [0, 1]
        X = np.clip(X, 0, 1)
        
        return X, y
    
    def quantum_kernel(x1, x2):
        """Compute quantum kernel between two data points."""
        
        # Create quantum feature maps for both points
        circuit1 = quantrs2.Circuit(2)
        circuit2 = quantrs2.Circuit(2)
        
        # Encode first data point
        circuit1.h(0)
        circuit1.h(1)
        circuit1.rz(0, x1[0] * 2 * np.pi)
        circuit1.rz(1, x1[1] * 2 * np.pi)
        circuit1.cx(0, 1)
        
        # Encode second data point
        circuit2.h(0)
        circuit2.h(1)
        circuit2.rz(0, x2[0] * 2 * np.pi)
        circuit2.rz(1, x2[1] * 2 * np.pi)
        circuit2.cx(0, 1)
        
        # In a real implementation, we would compute the overlap
        # between the two quantum states |φ(x1)⟩ and |φ(x2)⟩
        # For this demo, we'll simulate the kernel value
        
        # Classical approximation of quantum kernel
        # (In practice, this would use quantum interference)
        similarity = np.exp(-np.linalg.norm(x1 - x2) ** 2 / 0.1)
        return similarity
    
    def quantum_kernel_matrix(X):
        """Compute full quantum kernel matrix."""
        
        n_samples = len(X)
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = quantum_kernel(X[i], X[j])
        
        return K
    
    # Generate data
    X, y = generate_classification_data()
    
    print(f"Generated {len(X)} training samples")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Compute quantum kernel matrix
    print(f"\nComputing quantum kernel matrix...")
    K_quantum = quantum_kernel_matrix(X)
    
    print(f"Quantum kernel matrix shape: {K_quantum.shape}")
    print(f"Kernel matrix (sample):")
    print(K_quantum[:4, :4])
    
    # In practice, you would use this kernel with a classical SVM
    # For demonstration, we'll show the concept
    
    print(f"\nQuantum SVM advantages:")
    print(f"  1. Exponentially large feature space")
    print(f"  2. Quantum interference for pattern detection")
    print(f"  3. Potential exponential speedup for certain kernels")

quantum_svm_demo()
print()
```

## Algorithm 2: Variational Quantum Classifier

### Parameterized Quantum Circuits for Classification

```python
def variational_quantum_classifier():
    """Implement a variational quantum classifier."""
    
    print("🔄 Variational Quantum Classifier")
    print("=" * 40)
    
    class VQC:
        """Variational Quantum Classifier."""
        
        def __init__(self, num_qubits=2, num_layers=2):
            self.num_qubits = num_qubits
            self.num_layers = num_layers
            self.num_params = num_qubits * num_layers * 3  # 3 rotations per qubit per layer
            
        def create_ansatz(self, data_point, parameters):
            """Create parameterized quantum circuit."""
            
            circuit = quantrs2.Circuit(self.num_qubits)
            
            # Data encoding layer
            for i in range(self.num_qubits):
                angle = data_point[i % len(data_point)] * np.pi
                circuit.ry(i, angle)
            
            # Parameterized layers
            param_idx = 0
            for layer in range(self.num_layers):
                # Rotation layer
                for qubit in range(self.num_qubits):
                    circuit.rx(qubit, parameters[param_idx])
                    circuit.ry(qubit, parameters[param_idx + 1])
                    circuit.rz(qubit, parameters[param_idx + 2])
                    param_idx += 3
                
                # Entangling layer
                for qubit in range(self.num_qubits - 1):
                    circuit.cx(qubit, qubit + 1)
                
                # Add circular entanglement
                if self.num_qubits > 2:
                    circuit.cx(self.num_qubits - 1, 0)
            
            return circuit
        
        def predict_proba(self, data_point, parameters):
            """Predict class probability for a data point."""
            
            circuit = self.create_ansatz(data_point, parameters)
            circuit.measure_all()
            result = circuit.run()
            
            # Use first qubit measurement as classification
            probs = result.state_probabilities()
            
            # Probability of class 0 (first qubit = 0)
            prob_class_0 = sum(prob for state, prob in probs.items() 
                              if state[0] == '0')
            
            return prob_class_0
        
        def cost_function(self, X, y, parameters):
            """Compute cost function for training."""
            
            total_cost = 0
            
            for data_point, label in zip(X, y):
                predicted_prob = self.predict_proba(data_point, parameters)
                
                # Cross-entropy loss
                if label == 0:
                    cost = -np.log(predicted_prob + 1e-10)
                else:
                    cost = -np.log(1 - predicted_prob + 1e-10)
                
                total_cost += cost
            
            return total_cost / len(X)
        
        def train(self, X, y, num_iterations=50):
            """Train the quantum classifier."""
            
            # Initialize parameters randomly
            np.random.seed(42)
            parameters = np.random.uniform(0, 2*np.pi, self.num_params)
            
            print(f"Training VQC with {self.num_params} parameters...")
            
            learning_rate = 0.1
            costs = []
            
            for iteration in range(num_iterations):
                # Compute cost
                cost = self.cost_function(X, y, parameters)
                costs.append(cost)
                
                if iteration % 10 == 0:
                    print(f"  Iteration {iteration:2d}: Cost = {cost:.4f}")
                
                # Simple gradient descent (finite differences)
                grad = np.zeros_like(parameters)
                epsilon = 0.01
                
                for i in range(len(parameters)):
                    # Forward difference
                    params_plus = parameters.copy()
                    params_plus[i] += epsilon
                    cost_plus = self.cost_function(X, y, params_plus)
                    
                    params_minus = parameters.copy()
                    params_minus[i] -= epsilon
                    cost_minus = self.cost_function(X, y, params_minus)
                    
                    grad[i] = (cost_plus - cost_minus) / (2 * epsilon)
                
                # Update parameters
                parameters -= learning_rate * grad
            
            print(f"  Final cost: {costs[-1]:.4f}")
            return parameters, costs
    
    # Generate training data
    np.random.seed(42)
    
    # Simple 2D dataset
    X_train = np.array([
        [0.2, 0.3],  # Class 0
        [0.1, 0.4],  # Class 0
        [0.3, 0.2],  # Class 0
        [0.8, 0.7],  # Class 1
        [0.7, 0.8],  # Class 1
        [0.9, 0.6],  # Class 1
    ])
    
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    print(f"Training data: {len(X_train)} samples")
    print(f"Features shape: {X_train.shape}")
    print(f"Classes: {np.unique(y_train)}")
    
    # Create and train VQC
    vqc = VQC(num_qubits=2, num_layers=2)
    trained_params, training_costs = vqc.train(X_train, y_train)
    
    # Test predictions
    print(f"\nTesting trained classifier:")
    
    test_points = [
        ([0.15, 0.25], 0),  # Should be class 0
        ([0.85, 0.75], 1),  # Should be class 1
    ]
    
    for test_data, true_label in test_points:
        predicted_prob = vqc.predict_proba(test_data, trained_params)
        predicted_class = 0 if predicted_prob > 0.5 else 1
        
        print(f"  Data: {test_data}")
        print(f"  True class: {true_label}, Predicted: {predicted_class}")
        print(f"  Probability: {predicted_prob:.3f}")
        print(f"  Correct: {'✅' if predicted_class == true_label else '❌'}")

variational_quantum_classifier()
print()
```

## Algorithm 3: Quantum Neural Networks

### Quantum Perceptron

```python
def quantum_perceptron():
    """Implement a quantum perceptron."""
    
    print("🧠 Quantum Perceptron")
    print("=" * 25)
    
    class QuantumPerceptron:
        """Single quantum perceptron unit."""
        
        def __init__(self, num_inputs=2):
            self.num_inputs = num_inputs
            self.num_qubits = num_inputs + 1  # +1 for output qubit
            
        def forward_pass(self, inputs, weights, bias):
            """Forward pass through quantum perceptron."""
            
            circuit = quantrs2.Circuit(self.num_qubits)
            
            # Initialize output qubit in |+⟩ state
            circuit.h(self.num_inputs)
            
            # Encode inputs and apply weighted rotations
            for i, (inp, weight) in enumerate(zip(inputs, weights)):
                # Encode input
                circuit.ry(i, inp * np.pi)
                
                # Apply weighted interaction with output qubit
                rotation_angle = weight * inp * np.pi
                circuit.cry(self.num_inputs, i, rotation_angle)
            
            # Apply bias
            circuit.ry(self.num_inputs, bias)
            
            # Measure output qubit
            circuit.measure_all()
            result = circuit.run()
            
            # Extract output probability
            probs = result.state_probabilities()
            output_prob = sum(prob for state, prob in probs.items() 
                            if state[self.num_inputs] == '1')
            
            return output_prob
        
        def activation_function(self, x):
            """Quantum activation function."""
            # Probability output naturally acts as activation
            return x
    
    # Demonstrate quantum perceptron
    perceptron = QuantumPerceptron(num_inputs=2)
    
    # Example: Learn AND gate
    print("Training quantum perceptron for AND gate:")
    
    # Training data for AND gate
    training_data = [
        ([0, 0], 0),
        ([0, 1], 0), 
        ([1, 0], 0),
        ([1, 1], 1)
    ]
    
    # Initialize parameters
    weights = np.array([0.5, 0.5])
    bias = 0.0
    
    print(f"Initial weights: {weights}")
    print(f"Initial bias: {bias}")
    
    # Test with initial parameters
    print(f"\nTesting with initial parameters:")
    
    for inputs, target in training_data:
        output = perceptron.forward_pass(inputs, weights, bias)
        prediction = 1 if output > 0.5 else 0
        
        print(f"  Input: {inputs} → Output: {output:.3f} → Prediction: {prediction} (Target: {target})")
    
    # Simple training loop (conceptual)
    print(f"\nTraining concept:")
    print("  1. Measure quantum output probability")
    print("  2. Compute loss vs target")
    print("  3. Update weights using gradient descent")
    print("  4. Repeat until convergence")
    
    # Optimized parameters (simulated training result)
    optimized_weights = np.array([1.2, 1.2])
    optimized_bias = -1.5
    
    print(f"\nAfter training:")
    print(f"Optimized weights: {optimized_weights}")
    print(f"Optimized bias: {optimized_bias}")
    
    print(f"\nTesting with optimized parameters:")
    
    for inputs, target in training_data:
        output = perceptron.forward_pass(inputs, optimized_weights, optimized_bias)
        prediction = 1 if output > 0.5 else 0
        accuracy = "✅" if prediction == target else "❌"
        
        print(f"  Input: {inputs} → Output: {output:.3f} → Prediction: {prediction} (Target: {target}) {accuracy}")

quantum_perceptron()
print()
```

## Algorithm 4: Quantum Reinforcement Learning

### Quantum Policy Gradient

```python
def quantum_reinforcement_learning():
    """Demonstrate quantum reinforcement learning concepts."""
    
    print("🎮 Quantum Reinforcement Learning")
    print("=" * 40)
    
    class QuantumAgent:
        """Quantum agent for reinforcement learning."""
        
        def __init__(self, num_states=4, num_actions=2):
            self.num_states = num_states
            self.num_actions = num_actions
            self.num_qubits = int(np.ceil(np.log2(max(num_states, num_actions))))
            
        def encode_state(self, state):
            """Encode classical state into quantum circuit."""
            
            circuit = quantrs2.Circuit(self.num_qubits)
            
            # Encode state as binary
            state_binary = format(state, f'0{self.num_qubits}b')
            
            for i, bit in enumerate(state_binary):
                if bit == '1':
                    circuit.x(i)
            
            return circuit
        
        def quantum_policy(self, state, policy_params):
            """Quantum policy function."""
            
            circuit = quantrs2.Circuit(self.num_qubits + 1)  # +1 for action qubit
            
            # Encode state
            state_binary = format(state, f'0{self.num_qubits}b')
            for i, bit in enumerate(state_binary):
                if bit == '1':
                    circuit.x(i)
            
            # Parameterized policy circuit
            for i in range(self.num_qubits):
                circuit.ry(i, policy_params[i])
            
            # Action selection based on state
            for i in range(self.num_qubits):
                circuit.cx(i, self.num_qubits)  # Control action qubit
            
            # Final policy rotation
            circuit.ry(self.num_qubits, policy_params[-1])
            
            # Measure action qubit
            circuit.measure_all()
            result = circuit.run()
            
            # Extract action probability
            probs = result.state_probabilities()
            action_prob = sum(prob for state_result, prob in probs.items() 
                            if state_result[self.num_qubits] == '1')
            
            return action_prob
        
        def select_action(self, state, policy_params):
            """Select action based on quantum policy."""
            
            action_prob = self.quantum_policy(state, policy_params)
            
            # Stochastic action selection
            action = 1 if np.random.random() < action_prob else 0
            
            return action, action_prob
    
    # Simple environment: 4 states, 2 actions
    class SimpleEnvironment:
        """Simple grid world environment."""
        
        def __init__(self):
            self.state = 0
            self.goal_state = 3
            
        def step(self, action):
            """Take action and return reward."""
            
            if action == 1 and self.state < 3:  # Move right
                self.state += 1
                reward = 1 if self.state == self.goal_state else 0
            elif action == 0 and self.state > 0:  # Move left
                self.state -= 1
                reward = -0.1  # Small penalty for moving away
            else:
                reward = -0.1  # Penalty for invalid moves
            
            done = (self.state == self.goal_state)
            
            return self.state, reward, done
        
        def reset(self):
            """Reset environment."""
            self.state = 0
            return self.state
    
    # Initialize agent and environment
    agent = QuantumAgent(num_states=4, num_actions=2)
    env = SimpleEnvironment()
    
    print(f"Quantum RL setup:")
    print(f"  States: {agent.num_states}")
    print(f"  Actions: {agent.num_actions}")
    print(f"  Qubits: {agent.num_qubits + 1}")
    
    # Initialize policy parameters
    np.random.seed(42)
    policy_params = np.random.uniform(0, 2*np.pi, agent.num_qubits + 1)
    
    print(f"\nInitial policy parameters: {policy_params}")
    
    # Demonstrate episode
    print(f"\nSample episode:")
    
    state = env.reset()
    total_reward = 0
    step_count = 0
    
    while step_count < 10:  # Max 10 steps
        print(f"  Step {step_count}: State = {state}")
        
        # Select action using quantum policy
        action, action_prob = agent.select_action(state, policy_params)
        
        print(f"    Action probability: {action_prob:.3f}")
        print(f"    Selected action: {action}")
        
        # Take action
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        print(f"    Reward: {reward}")
        print(f"    Next state: {next_state}")
        
        if done:
            print(f"    Goal reached!")
            break
        
        state = next_state
        step_count += 1
    
    print(f"\nEpisode summary:")
    print(f"  Total reward: {total_reward}")
    print(f"  Steps taken: {step_count + 1}")
    
    print(f"\nQuantum RL advantages:")
    print("  1. Exponential policy space")
    print("  2. Quantum exploration strategies")
    print("  3. Superposition of policies")
    print("  4. Quantum speedup for value function approximation")

quantum_reinforcement_learning()
print()
```

## Performance Analysis and Comparison

### Classical vs Quantum ML Performance

```python
def qml_performance_analysis():
    """Analyze performance characteristics of QML algorithms."""
    
    print("📈 QML Performance Analysis")
    print("=" * 30)
    
    # Theoretical complexity comparison
    algorithms = [
        {
            "name": "Classical SVM",
            "training": "O(n²) to O(n³)",
            "prediction": "O(sv × d)",
            "feature_space": "d dimensions"
        },
        {
            "name": "Quantum SVM", 
            "training": "O(n²) with quantum kernel",
            "prediction": "O(poly(log d))",
            "feature_space": "Exponential: 2^d dimensions"
        },
        {
            "name": "Classical Neural Network",
            "training": "O(layers × neurons × data)",
            "prediction": "O(layers × neurons)",
            "feature_space": "Linear in parameters"
        },
        {
            "name": "Quantum Neural Network",
            "training": "O(params × data)",
            "prediction": "O(circuit_depth)",
            "feature_space": "Exponential in qubits"
        }
    ]
    
    print("Algorithm complexity comparison:")
    print(f"{'Algorithm':<25} {'Training':<20} {'Prediction':<20} {'Feature Space'}")
    print("-" * 85)
    
    for alg in algorithms:
        print(f"{alg['name']:<25} {alg['training']:<20} {alg['prediction']:<20} {alg['feature_space']}")
    
    # Data size scaling
    print(f"\nData scaling analysis:")
    
    data_sizes = [100, 1000, 10000, 100000]
    
    print(f"{'Data Size':<10} {'Classical':<15} {'Quantum (ideal)':<20} {'Potential Speedup'}")
    print("-" * 60)
    
    for n in data_sizes:
        classical_time = n ** 2  # Typical for SVM
        quantum_time = n * np.log(n)  # Potential quantum speedup
        speedup = classical_time / quantum_time
        
        print(f"{n:<10} {classical_time:<15} {quantum_time:<20.1f} {speedup:<15.1f}x")
    
    # Feature dimension scaling
    print(f"\nFeature space scaling:")
    
    feature_dims = [4, 8, 12, 16]
    
    print(f"{'Dimensions':<12} {'Classical Features':<18} {'Quantum Features':<18} {'Ratio'}")
    print("-" * 65)
    
    for d in feature_dims:
        classical_features = d
        quantum_features = 2 ** d
        ratio = quantum_features / classical_features
        
        print(f"{d:<12} {classical_features:<18} {quantum_features:<18} {ratio:<10.0f}x")

def nisq_limitations():
    """Discuss current limitations of NISQ-era QML."""
    
    print(f"\n⚠️  NISQ-Era QML Limitations")
    print("=" * 35)
    
    limitations = [
        {
            "limitation": "Noise and errors",
            "impact": "Reduces algorithm accuracy",
            "mitigation": "Error mitigation, shorter circuits"
        },
        {
            "limitation": "Limited qubit counts",
            "impact": "Small feature spaces",
            "mitigation": "Efficient encodings, hybrid methods"
        },
        {
            "limitation": "Short coherence times",
            "impact": "Shallow circuits only",
            "mitigation": "Circuit optimization, parallelization"
        },
        {
            "limitation": "Classical simulation bottleneck",
            "impact": "Hard to train quantum models",
            "mitigation": "Gradient-free optimization"
        }
    ]
    
    for i, lim in enumerate(limitations, 1):
        print(f"{i}. {lim['limitation']}")
        print(f"   Impact: {lim['impact']}")
        print(f"   Mitigation: {lim['mitigation']}")
        print()

qml_performance_analysis()
nisq_limitations()
```

## Real-World QML Applications

### Current and Future Applications

```python
def qml_applications():
    """Survey of real-world QML applications."""
    
    print("🌍 Real-World QML Applications")
    print("=" * 35)
    
    # Current NISQ applications
    current_apps = [
        {
            "domain": "Finance",
            "application": "Portfolio optimization",
            "algorithm": "QAOA, VQE",
            "advantage": "Quadratic constraints handling"
        },
        {
            "domain": "Drug Discovery", 
            "application": "Molecular property prediction",
            "algorithm": "Quantum neural networks",
            "advantage": "Natural quantum simulation"
        },
        {
            "domain": "Materials Science",
            "application": "Catalyst design",
            "algorithm": "VQE, quantum simulation",
            "advantage": "Quantum chemistry modeling"
        },
        {
            "domain": "Optimization",
            "application": "Traffic flow optimization",
            "algorithm": "QAOA",
            "advantage": "Combinatorial problem solving"
        }
    ]
    
    print("Current NISQ-era applications:")
    
    for app in current_apps:
        print(f"\n{app['domain']}: {app['application']}")
        print(f"  Algorithm: {app['algorithm']}")
        print(f"  Advantage: {app['advantage']}")
    
    # Future fault-tolerant applications
    print(f"\nFuture fault-tolerant QML:")
    
    future_apps = [
        "Exponential speedup for certain learning problems",
        "Quantum advantage in feature spaces",
        "True quantum neural networks",
        "Quantum-enhanced deep learning",
        "Quantum generative models",
        "Quantum reinforcement learning with exploration"
    ]
    
    for i, app in enumerate(future_apps, 1):
        print(f"  {i}. {app}")

def qml_best_practices():
    """Best practices for implementing QML algorithms."""
    
    print(f"\n💡 QML Best Practices")
    print("=" * 25)
    
    practices = [
        {
            "category": "Data Preparation",
            "practices": [
                "Normalize data to [0, 1] for angle encoding",
                "Use dimensionality reduction for high-D data",
                "Consider data structure for encoding choice"
            ]
        },
        {
            "category": "Circuit Design",
            "practices": [
                "Start with shallow circuits (< 10 layers)",
                "Use hardware-efficient ansatzes",
                "Include data re-uploading for expressivity"
            ]
        },
        {
            "category": "Training",
            "practices": [
                "Use gradient-free optimizers (COBYLA, SPSA)",
                "Include multiple random initializations",
                "Monitor for barren plateaus"
            ]
        },
        {
            "category": "Validation",
            "practices": [
                "Compare with classical baselines",
                "Use cross-validation",
                "Analyze quantum advantage sources"
            ]
        }
    ]
    
    for category in practices:
        print(f"\n{category['category']}:")
        for practice in category['practices']:
            print(f"  • {practice}")

qml_applications()
qml_best_practices()
```

## Hands-On Challenge: Build Your Own QML Model

```python
def qml_challenge():
    """Challenge: Build a complete QML pipeline."""
    
    print("🎯 QML Challenge: Complete Pipeline")
    print("=" * 40)
    
    print("Your mission: Build a quantum classifier for the Iris dataset!")
    
    print(f"\nSteps to complete:")
    
    steps = [
        "1. Load and preprocess Iris dataset",
        "2. Design quantum feature map",
        "3. Create variational quantum classifier",
        "4. Implement training loop",
        "5. Evaluate performance",
        "6. Compare with classical baseline"
    ]
    
    for step in steps:
        print(f"  {step}")
    
    print(f"\nTemplate structure:")
    
    template = """
    class IrisQuantumClassifier:
        def __init__(self):
            # Initialize quantum classifier
            pass
            
        def preprocess_data(self, X, y):
            # Normalize features and encode labels
            pass
            
        def feature_map(self, data_point):
            # Quantum feature encoding
            pass
            
        def variational_circuit(self, data_point, params):
            # Parameterized quantum circuit
            pass
            
        def cost_function(self, X, y, params):
            # Training objective
            pass
            
        def train(self, X_train, y_train):
            # Optimization loop
            pass
            
        def predict(self, X_test):
            # Make predictions
            pass
    """
    
    print(template)
    
    print(f"\nBonus challenges:")
    bonus = [
        "• Try different feature map designs",
        "• Implement data re-uploading",
        "• Add regularization to prevent overfitting", 
        "• Analyze quantum advantage vs classical methods",
        "• Optimize for NISQ device constraints"
    ]
    
    for challenge in bonus:
        print(f"  {challenge}")

qml_challenge()
```

## Key Takeaways

🎯 **What you learned:**

1. **Quantum Data Encoding**: Amplitude and angle encoding techniques
2. **Quantum Feature Maps**: Transforming data into quantum feature spaces
3. **QML Algorithms**: Quantum SVM, VQC, quantum neural networks
4. **Training Methods**: Variational optimization for quantum models
5. **Performance Analysis**: Understanding quantum advantages and limitations

🚀 **Core QML concepts:**

- **Exponential feature spaces**: Quantum computers can access exponentially large feature spaces
- **Quantum interference**: Use constructive/destructive interference for pattern recognition
- **Hybrid algorithms**: Combine quantum and classical processing
- **Variational training**: Optimize parameterized quantum circuits
- **NISQ constraints**: Work within current hardware limitations

⚡ **Quantum advantages:**

1. **Feature space**: Exponential expansion of feature dimensions
2. **Parallelism**: Process multiple data points in superposition
3. **Interference**: Amplify useful patterns, suppress noise
4. **Kernels**: Access to quantum kernels impossible classically

## QML Development Checklist

Before deploying quantum ML models:

- [ ] **Data preprocessing** appropriate for quantum encoding
- [ ] **Circuit design** optimized for NISQ constraints
- [ ] **Classical baseline** implemented for comparison
- [ ] **Training stability** verified (no barren plateaus)
- [ ] **Performance validation** on hold-out test set
- [ ] **Quantum advantage** analysis conducted
- [ ] **Noise resilience** tested
- [ ] **Scalability** considerations addressed

## Common QML Pitfalls

❌ **Avoid these mistakes:**
- Using too many qubits without justification
- Ignoring classical ML best practices
- Not comparing with classical baselines
- Assuming quantum = better without analysis
- Neglecting noise effects in NISQ devices

✅ **Best practices:**
- Start simple and scale up gradually
- Always include classical comparisons
- Focus on problems with quantum advantage
- Design for current hardware limitations
- Validate theoretical advantages empirically

## What's Next?

Congratulations! You've completed the beginner tutorial series and learned:
- Quantum computing fundamentals
- Circuit construction and algorithms
- Hardware optimization techniques
- Quantum machine learning applications

**Continue your quantum journey:**
- [Intermediate Tutorials](../intermediate/) - Advanced algorithms and applications
- [Example Gallery](../../examples/) - Real-world quantum programs
- [API Reference](../../api/) - Detailed QuantRS2 documentation
- [Community](../../community/) - Join the quantum computing community

## Practice Exercises

1. **Quantum Feature Engineering**: Implement different encoding strategies for the same dataset
2. **Hybrid Models**: Combine quantum and classical layers in a neural network
3. **Quantum Kernels**: Design custom quantum kernels for specific data types
4. **NISQ Optimization**: Adapt QML algorithms for hardware constraints

## Additional Resources

### Research Papers
- Schuld & Petruccione, "Supervised Learning with Quantum Computers" (2018)
- Biamonte et al., "Quantum machine learning" Nature (2017)
- Cerezo et al., "Variational quantum algorithms" Nature Reviews Physics (2021)

### Implementations
- [PennyLane](https://pennylane.ai/) - Quantum ML framework
- [Qiskit Machine Learning](https://qiskit.org/ecosystem/machine-learning/)
- [TensorFlow Quantum](https://www.tensorflow.org/quantum)

### Courses
- IBM Qiskit Textbook: Machine Learning
- Xanadu Quantum Codebook: QML modules
- Microsoft Quantum Katas: Machine Learning

---

**🎉 Congratulations!** You've completed the QuantRS2 beginner tutorial series and are now ready to build quantum machine learning applications!

*"The future belongs to those who understand both quantum physics and machine learning."* - Unknown

Start building the future today! 🚀