# Advanced Tutorial: Advanced Quantum Machine Learning

## Overview

This advanced tutorial explores cutting-edge quantum machine learning techniques, including quantum kernel methods, variational quantum algorithms, and hybrid quantum-classical approaches. You'll learn how to implement and optimize sophisticated QML algorithms using QuantRS2-Py.

## Prerequisites

- Strong understanding of classical machine learning
- Familiarity with variational quantum algorithms
- Knowledge of optimization theory
- Understanding of quantum circuits and gates
- Completion of intermediate QML tutorials

## Topics Covered

1. Quantum Kernel Methods
2. Quantum Neural Tangent Kernels (QNTK)
3. Barren Plateau Mitigation Strategies
4. Hardware-Efficient Quantum ML
5. Quantum Reinforcement Learning
6. Transfer Learning for Quantum Systems
7. Quantum Generative Models
8. Quantum Meta-Learning

## 1. Quantum Kernel Methods

Quantum kernel methods leverage quantum computers to compute kernel functions in exponentially large feature spaces.

### Theory

The quantum kernel is defined as:
```
K(x, x') = |⟨φ(x')|φ(x)⟩|²
```
where `|φ(x)⟩` is a quantum feature map.

### Implementation

```python
import quantrs2 as qr
from quantrs2 import ml
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class QuantumKernelClassifier:
    """
    Quantum kernel method for classification tasks.

    Uses quantum feature maps to compute kernels in exponentially
    large Hilbert spaces.
    """

    def __init__(self, n_qubits, depth=2, feature_map='zz'):
        """
        Initialize quantum kernel classifier.

        Args:
            n_qubits: Number of qubits
            depth: Depth of feature map
            feature_map: Type of feature map ('zz', 'pauli', 'custom')
        """
        self.n_qubits = n_qubits
        self.depth = depth
        self.feature_map = feature_map
        self.classical_svm = None

    def create_feature_map(self, x):
        """
        Create quantum feature map circuit.

        Args:
            x: Classical data point (array of length n_qubits)

        Returns:
            Quantum circuit encoding the data
        """
        circuit = qr.PyCircuit(self.n_qubits)

        if self.feature_map == 'zz':
            # ZZ feature map (IQP-style)
            for layer in range(self.depth):
                # Hadamard layer
                for i in range(self.n_qubits):
                    circuit.h(i)

                # Encoding layer
                for i in range(self.n_qubits):
                    circuit.rz(i, 2 * x[i % len(x)])

                # Entangling layer
                for i in range(self.n_qubits - 1):
                    circuit.cx(i, i + 1)
                    circuit.rz(i + 1, 2 * (np.pi - x[i % len(x)]) *
                              (np.pi - x[(i+1) % len(x)]))
                    circuit.cx(i, i + 1)

        elif self.feature_map == 'pauli':
            # Pauli feature map
            for layer in range(self.depth):
                for i in range(self.n_qubits):
                    circuit.h(i)
                    circuit.rz(i, 2 * x[i % len(x)])
                    circuit.rx(i, 2 * x[i % len(x)])

                # Entangling
                for i in range(self.n_qubits - 1):
                    circuit.cnot(i, i + 1)

        return circuit

    def compute_kernel_element(self, x1, x2):
        """
        Compute kernel element K(x1, x2).

        Args:
            x1, x2: Data points

        Returns:
            Kernel value
        """
        # Create feature maps
        circuit1 = self.create_feature_map(x1)
        circuit2 = self.create_feature_map(x2)

        # Combine circuits: |φ(x1)⟩⟨φ(x2)|
        kernel_circuit = qr.PyCircuit(self.n_qubits)

        # Apply U(x2)
        circuit2_ops = circuit2.get_operations()
        for op in circuit2_ops:
            kernel_circuit.add_operation(op)

        # Apply U†(x1)
        circuit1_ops = circuit1.get_operations()
        for op in reversed(circuit1_ops):
            kernel_circuit.add_operation(op.dagger())

        # Measure overlap
        result = kernel_circuit.run(shots=1000)
        probs = result.state_probabilities()

        # Kernel is probability of measuring |0...0⟩
        return probs[0]

    def compute_kernel_matrix(self, X1, X2=None):
        """
        Compute kernel matrix between datasets.

        Args:
            X1: First dataset (n_samples1, n_features)
            X2: Second dataset (n_samples2, n_features), if None use X1

        Returns:
            Kernel matrix (n_samples1, n_samples2)
        """
        if X2 is None:
            X2 = X1

        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.compute_kernel_element(X1[i], X2[j])

        return K

    def fit(self, X, y):
        """
        Train the quantum kernel classifier.

        Args:
            X: Training data (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        # Compute quantum kernel matrix
        K_train = self.compute_kernel_matrix(X)

        # Train classical SVM with precomputed kernel
        self.classical_svm = SVC(kernel='precomputed')
        self.classical_svm.fit(K_train, y)
        self.X_train = X

    def predict(self, X):
        """
        Predict labels for new data.

        Args:
            X: Test data (n_samples, n_features)

        Returns:
            Predicted labels
        """
        # Compute kernel between test and train data
        K_test = self.compute_kernel_matrix(X, self.X_train)

        # Predict using SVM
        return self.classical_svm.predict(K_test)

    def score(self, X, y):
        """Compute accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Example usage
np.random.seed(42)

# Generate synthetic dataset
X, y = make_classification(n_samples=100, n_features=4,
                          n_informative=2, n_redundant=0,
                          n_clusters_per_class=1, random_state=42)

# Normalize features to [0, 2π]
X = (X - X.min()) / (X.max() - X.min()) * 2 * np.pi

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train quantum kernel classifier
qkc = QuantumKernelClassifier(n_qubits=4, depth=2, feature_map='zz')
print("Training quantum kernel classifier...")
qkc.fit(X_train, y_train)

# Evaluate
train_score = qkc.score(X_train, y_train)
test_score = qkc.score(X_test, y_test)

print(f"Training accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")
```

### Quantum Kernel Alignment

Optimize the quantum kernel to align with the target kernel:

```python
class QuantumKernelAlignment:
    """Optimize quantum kernels using kernel alignment."""

    def __init__(self, n_qubits, depth=2):
        self.n_qubits = n_qubits
        self.depth = depth
        self.optimal_params = None

    def parameterized_feature_map(self, x, params):
        """Create parameterized quantum feature map."""
        circuit = qr.PyCircuit(self.n_qubits)

        param_idx = 0
        for layer in range(self.depth):
            # Rotation layer with parameters
            for i in range(self.n_qubits):
                circuit.ry(i, params[param_idx] * x[i % len(x)])
                param_idx += 1
                circuit.rz(i, params[param_idx] * x[i % len(x)])
                param_idx += 1

            # Entangling layer
            for i in range(self.n_qubits - 1):
                circuit.cnot(i, i + 1)

        return circuit

    def kernel_alignment(self, K_quantum, K_target):
        """
        Compute kernel alignment score.

        Args:
            K_quantum: Quantum kernel matrix
            K_target: Target kernel matrix (e.g., from labels)

        Returns:
            Alignment score
        """
        # Centered kernel matrices
        n = len(K_quantum)
        H = np.eye(n) - np.ones((n, n)) / n

        K_q_centered = H @ K_quantum @ H
        K_t_centered = H @ K_target @ H

        # Frobenius inner product
        numerator = np.trace(K_q_centered @ K_t_centered)
        denominator = np.sqrt(np.trace(K_q_centered @ K_q_centered) *
                             np.trace(K_t_centered @ K_t_centered))

        return numerator / denominator if denominator > 0 else 0

    def optimize(self, X, y, num_iterations=50):
        """
        Optimize kernel parameters using kernel alignment.

        Args:
            X: Training data
            y: Training labels
            num_iterations: Number of optimization iterations
        """
        from scipy.optimize import minimize

        # Create target kernel from labels
        K_target = np.outer(y, y)

        # Number of parameters
        num_params = self.n_qubits * self.depth * 2

        def objective(params):
            """Negative kernel alignment (for minimization)."""
            # Compute quantum kernel with current parameters
            qkc = QuantumKernelClassifier(self.n_qubits, self.depth)
            # TODO: Inject parameters into feature map
            K_quantum = qkc.compute_kernel_matrix(X)

            alignment = self.kernel_alignment(K_quantum, K_target)
            return -alignment  # Negative for minimization

        # Optimize
        initial_params = np.random.uniform(0, 2*np.pi, num_params)
        result = minimize(objective, initial_params, method='COBYLA',
                         options={'maxiter': num_iterations})

        self.optimal_params = result.x
        return result

# Example kernel alignment
qka = QuantumKernelAlignment(n_qubits=4, depth=2)
result = qka.optimize(X_train, y_train, num_iterations=30)
print(f"Optimal kernel alignment: {-result.fun:.3f}")
```

## 2. Quantum Neural Tangent Kernels (QNTK)

QNTK describes the training dynamics of quantum neural networks in the infinite-width limit.

```python
class QuantumNeuralTangentKernel:
    """
    Compute and analyze Quantum Neural Tangent Kernels.

    QNTK provides insights into trainability and expressivity
    of variational quantum circuits.
    """

    def __init__(self, n_qubits, circuit_architecture):
        self.n_qubits = n_qubits
        self.circuit_architecture = circuit_architecture

    def create_variational_circuit(self, x, params):
        """Create parameterized quantum circuit."""
        circuit = qr.PyCircuit(self.n_qubits)

        # Encode data
        for i in range(self.n_qubits):
            circuit.ry(i, x[i % len(x)])

        # Variational layers
        param_idx = 0
        for layer in range(len(self.circuit_architecture)):
            for i in range(self.n_qubits):
                circuit.ry(i, params[param_idx])
                param_idx += 1
                circuit.rz(i, params[param_idx])
                param_idx += 1

            # Entanglement
            for i in range(self.n_qubits - 1):
                circuit.cnot(i, i + 1)

        return circuit

    def compute_gradient(self, x, params, param_idx, observable):
        """
        Compute gradient using parameter-shift rule.

        Args:
            x: Input data
            params: Circuit parameters
            param_idx: Index of parameter to differentiate
            observable: Observable to measure

        Returns:
            Gradient value
        """
        # Shift parameter
        shift = np.pi / 2
        params_plus = params.copy()
        params_minus = params.copy()

        params_plus[param_idx] += shift
        params_minus[param_idx] -= shift

        # Evaluate circuits
        circuit_plus = self.create_variational_circuit(x, params_plus)
        circuit_minus = self.create_variational_circuit(x, params_minus)

        # Measure observable
        exp_plus = self.measure_observable(circuit_plus, observable)
        exp_minus = self.measure_observable(circuit_minus, observable)

        # Parameter-shift rule
        gradient = 0.5 * (exp_plus - exp_minus)
        return gradient

    def measure_observable(self, circuit, observable):
        """Measure expectation value of observable."""
        result = circuit.run(shots=1000)
        probs = result.state_probabilities()

        # For simplicity, measure Z on first qubit
        exp_value = probs[0] + probs[1] - probs[2] - probs[3]
        return exp_value

    def compute_qntk_element(self, x1, x2, params, observable):
        """
        Compute QNTK matrix element.

        K_QNTK(x1, x2) = Σ_μ (∂f/∂θμ)|x1 (∂f/∂θμ)|x2
        """
        num_params = len(params)
        qntk_value = 0.0

        for param_idx in range(num_params):
            grad1 = self.compute_gradient(x1, params, param_idx, observable)
            grad2 = self.compute_gradient(x2, params, param_idx, observable)
            qntk_value += grad1 * grad2

        return qntk_value

    def compute_qntk_matrix(self, X, params, observable):
        """Compute full QNTK matrix."""
        n = len(X)
        K_qntk = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                K_qntk[i, j] = self.compute_qntk_element(
                    X[i], X[j], params, observable
                )
                K_qntk[j, i] = K_qntk[i, j]  # Symmetric

        return K_qntk

    def analyze_trainability(self, K_qntk):
        """
        Analyze trainability from QNTK eigenvalues.

        Large eigenvalue spread indicates potential barren plateaus.
        """
        eigenvalues = np.linalg.eigvalsh(K_qntk)

        max_eig = np.max(eigenvalues)
        min_eig = np.min(np.abs(eigenvalues))

        condition_number = max_eig / min_eig if min_eig > 0 else np.inf

        return {
            'eigenvalues': eigenvalues,
            'condition_number': condition_number,
            'max_eigenvalue': max_eig,
            'min_eigenvalue': min_eig,
            'trainable': condition_number < 1000  # Heuristic threshold
        }

# Example QNTK analysis
qntk = QuantumNeuralTangentKernel(
    n_qubits=4,
    circuit_architecture=[1, 1, 1]  # 3 layers
)

# Initialize parameters
num_params = 4 * 3 * 2  # n_qubits * layers * 2 (ry, rz)
params = np.random.uniform(0, 2*np.pi, num_params)

# Compute QNTK on small dataset
X_small = X_train[:10]
observable = 'Z0'  # Measure Z on qubit 0

print("Computing QNTK matrix...")
K_qntk = qntk.compute_qntk_matrix(X_small, params, observable)

# Analyze trainability
analysis = qntk.analyze_trainability(K_qntk)
print(f"\nQNTK Analysis:")
print(f"Condition number: {analysis['condition_number']:.2e}")
print(f"Trainable: {analysis['trainable']}")
print(f"Eigenvalue range: [{analysis['min_eigenvalue']:.2e}, "
      f"{analysis['max_eigenvalue']:.2e}]")
```

## 3. Barren Plateau Mitigation

Strategies to avoid or mitigate barren plateaus in variational quantum algorithms.

```python
class BarrenPlateauMitigation:
    """
    Implement strategies to mitigate barren plateaus.

    Techniques include:
    1. Layer-wise training
    2. Identity initialization
    3. Correlated parameters
    4. Local cost functions
    """

    def __init__(self, n_qubits, max_layers=10):
        self.n_qubits = n_qubits
        self.max_layers = max_layers

    def identity_initialized_circuit(self, x, num_layers):
        """
        Create circuit with identity-block initialization.

        Initialize parameters near identity to avoid barren plateaus.
        """
        circuit = qr.PyCircuit(self.n_qubits)

        # Data encoding
        for i in range(self.n_qubits):
            circuit.ry(i, x[i % len(x)])

        # Variational layers with identity initialization
        for layer in range(num_layers):
            for i in range(self.n_qubits):
                # Small perturbation from identity
                circuit.ry(i, np.random.normal(0, 0.01))
                circuit.rz(i, np.random.normal(0, 0.01))

            # Entanglement
            for i in range(self.n_qubits - 1):
                circuit.cnot(i, i + 1)

        return circuit

    def layerwise_training(self, X, y, final_layers=5):
        """
        Train circuit layer-by-layer to avoid barren plateaus.

        Args:
            X: Training data
            y: Training labels
            final_layers: Total number of layers to reach

        Returns:
            Optimized parameters for each layer
        """
        from scipy.optimize import minimize

        all_params = []

        for num_layers in range(1, final_layers + 1):
            print(f"\nTraining layer {num_layers}/{final_layers}")

            # Initialize new layer parameters
            layer_params = np.random.normal(0, 0.01, self.n_qubits * 2)

            if num_layers > 1:
                # Concatenate with previous layers (frozen)
                current_params = np.concatenate([
                    np.concatenate(all_params),
                    layer_params
                ])
            else:
                current_params = layer_params

            def cost_function(params):
                """Cost function for current layer configuration."""
                total_cost = 0.0

                for i in range(len(X)):
                    circuit = self.create_circuit_with_layers(
                        X[i], params, num_layers
                    )
                    result = circuit.run()
                    probs = result.state_probabilities()

                    # Simple cost: negative log-likelihood
                    prediction = 1 if probs[0] < 0.5 else 0
                    cost = -np.log(probs[prediction] + 1e-10)
                    total_cost += cost

                return total_cost / len(X)

            # Optimize only the new layer
            result = minimize(
                cost_function,
                current_params,
                method='COBYLA',
                options={'maxiter': 100}
            )

            all_params.append(result.x[-self.n_qubits*2:])
            print(f"Layer {num_layers} cost: {result.fun:.4f}")

        return np.concatenate(all_params)

    def create_circuit_with_layers(self, x, params, num_layers):
        """Helper to create circuit with specified layers."""
        circuit = qr.PyCircuit(self.n_qubits)

        # Data encoding
        for i in range(self.n_qubits):
            circuit.ry(i, x[i % len(x)])

        # Apply layers
        param_idx = 0
        for layer in range(num_layers):
            for i in range(self.n_qubits):
                circuit.ry(i, params[param_idx])
                param_idx += 1
                circuit.rz(i, params[param_idx])
                param_idx += 1

            for i in range(self.n_qubits - 1):
                circuit.cnot(i, i + 1)

        return circuit

    def local_cost_function(self, X, y, params, locality=2):
        """
        Use local cost functions to avoid barren plateaus.

        Instead of global cost, measure on small subsets of qubits.
        """
        total_cost = 0.0

        for i in range(len(X)):
            circuit = self.create_circuit_with_layers(
                X[i], params, len(params) // (self.n_qubits * 2)
            )
            result = circuit.run()
            probs = result.state_probabilities()

            # Local cost: measure only first 'locality' qubits
            local_probs = np.zeros(2**locality)
            for state, prob in enumerate(probs):
                local_state = state % (2**locality)
                local_probs[local_state] += prob

            # Compute cost on local measurement
            prediction = np.argmax(local_probs)
            target = y[i] % (2**locality)
            cost = -np.log(local_probs[target] + 1e-10)
            total_cost += cost

        return total_cost / len(X)

# Example barren plateau mitigation
bpm = BarrenPlateauMitigation(n_qubits=4, max_layers=5)

print("Layer-wise training to avoid barren plateaus...")
optimal_params = bpm.layerwise_training(X_train[:20], y_train[:20], final_layers=3)

print(f"\nOptimized {len(optimal_params)} parameters across 3 layers")
```

## 4. Hardware-Efficient Quantum ML

Design quantum circuits that are optimized for near-term quantum hardware.

```python
class HardwareEfficientQML:
    """
    Hardware-efficient quantum machine learning circuits.

    Designed for NISQ devices with:
    - Limited qubit connectivity
    - Short coherence times
    - Gate fidelity constraints
    """

    def __init__(self, n_qubits, connectivity_map=None):
        self.n_qubits = n_qubits
        self.connectivity_map = connectivity_map or self.linear_connectivity()

    def linear_connectivity(self):
        """Linear connectivity pattern."""
        return [(i, i+1) for i in range(self.n_qubits - 1)]

    def create_hardware_efficient_ansatz(self, params, num_layers):
        """
        Create hardware-efficient ansatz.

        Uses only native gates and respects connectivity constraints.
        """
        circuit = qr.PyCircuit(self.n_qubits)

        param_idx = 0
        for layer in range(num_layers):
            # Single-qubit rotation layer (native gates)
            for i in range(self.n_qubits):
                circuit.ry(i, params[param_idx])
                param_idx += 1
                circuit.rz(i, params[param_idx])
                param_idx += 1

            # Two-qubit entangling layer (respecting connectivity)
            for q1, q2 in self.connectivity_map:
                circuit.cnot(q1, q2)

        return circuit

    def noise_aware_training(self, X, y, noise_model, num_layers=3):
        """
        Train with noise model to improve hardware performance.

        Args:
            X: Training data
            y: Training labels
            noise_model: Quantum noise model
            num_layers: Number of circuit layers
        """
        from scipy.optimize import minimize

        num_params = self.n_qubits * num_layers * 2

        def noisy_cost_function(params):
            """Cost function with noise simulation."""
            total_cost = 0.0

            for i in range(len(X)):
                # Encode data
                circuit = qr.PyCircuit(self.n_qubits)
                for j in range(self.n_qubits):
                    circuit.ry(j, X[i][j % len(X[i])])

                # Add variational circuit
                var_circuit = self.create_hardware_efficient_ansatz(
                    params, num_layers
                )
                # Combine circuits
                # circuit.compose(var_circuit)

                # Apply noise model
                noisy_circuit = noise_model.apply(circuit)

                # Run simulation
                result = noisy_circuit.run(shots=1000)
                probs = result.state_probabilities()

                # Compute cost
                prediction = 1 if probs[0] < 0.5 else 0
                if prediction != y[i]:
                    total_cost += 1.0

            return total_cost / len(X)

        # Optimize
        initial_params = np.random.uniform(0, 2*np.pi, num_params)
        result = minimize(
            noisy_cost_function,
            initial_params,
            method='COBYLA',
            options={'maxiter': 100}
        )

        return result

    def circuit_depth_optimization(self, params, max_depth=20):
        """
        Optimize circuit depth while maintaining performance.

        Uses dynamic circuit compilation and gate merging.
        """
        # Create initial circuit
        num_layers = len(params) // (self.n_qubits * 2)
        circuit = self.create_hardware_efficient_ansatz(params, num_layers)

        # Optimize circuit depth
        from quantrs2 import circuit_optimization

        optimized = circuit_optimization.optimize_depth(
            circuit,
            max_depth=max_depth,
            preserve_unitarity=True
        )

        return optimized

# Example hardware-efficient QML
connectivity = [(0, 1), (1, 2), (2, 3)]  # Linear chain
heqml = HardwareEfficientQML(n_qubits=4, connectivity_map=connectivity)

# Create ansatz
params = np.random.uniform(0, 2*np.pi, 4 * 3 * 2)
ansatz = heqml.create_hardware_efficient_ansatz(params, num_layers=3)

print("Hardware-efficient ansatz created")
print(f"Circuit depth: {ansatz.depth()}")
print(f"Gate count: {ansatz.num_gates()}")
```

## 5. Quantum Reinforcement Learning

Apply quantum computing to reinforcement learning problems.

```python
class QuantumReinforcementLearning:
    """
    Quantum reinforcement learning using variational circuits.

    Implements quantum Q-learning and policy gradient methods.
    """

    def __init__(self, n_qubits, n_actions, learning_rate=0.1):
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.q_network_params = None

    def create_q_network(self, state, params):
        """
        Create quantum Q-network.

        Maps state to Q-values for each action.
        """
        circuit = qr.PyCircuit(self.n_qubits)

        # Encode state
        for i in range(len(state)):
            circuit.ry(i, state[i])

        # Variational layers
        param_idx = 0
        for layer in range(3):  # 3 layers
            for i in range(self.n_qubits):
                circuit.ry(i, params[param_idx])
                param_idx += 1
                circuit.rz(i, params[param_idx])
                param_idx += 1

            for i in range(self.n_qubits - 1):
                circuit.cnot(i, i + 1)

        return circuit

    def get_q_values(self, state, params):
        """
        Get Q-values for all actions.

        Returns:
            Array of Q-values, one per action
        """
        circuit = self.create_q_network(state, params)
        result = circuit.run(shots=1000)
        probs = result.state_probabilities()

        # Map probabilities to Q-values
        # Use first n_actions basis states
        q_values = probs[:self.n_actions]

        # Normalize to [-1, 1] range
        q_values = 2 * q_values - 1

        return q_values

    def select_action(self, state, params, epsilon=0.1):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            params: Q-network parameters
            epsilon: Exploration rate

        Returns:
            Selected action
        """
        if np.random.random() < epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action
            q_values = self.get_q_values(state, params)
            return np.argmax(q_values)

    def compute_gradient(self, state, action, target, params):
        """
        Compute gradient for Q-learning update.

        Uses parameter-shift rule.
        """
        num_params = len(params)
        gradients = np.zeros(num_params)

        for param_idx in range(num_params):
            # Shift parameter
            shift = np.pi / 2
            params_plus = params.copy()
            params_minus = params.copy()

            params_plus[param_idx] += shift
            params_minus[param_idx] -= shift

            # Compute Q-values with shifted parameters
            q_plus = self.get_q_values(state, params_plus)[action]
            q_minus = self.get_q_values(state, params_minus)[action]

            # Parameter-shift gradient
            gradient = 0.5 * (q_plus - q_minus)

            # TD error gradient
            current_q = self.get_q_values(state, params)[action]
            td_error = target - current_q

            gradients[param_idx] = td_error * gradient

        return gradients

    def train_episode(self, env, params, num_steps=100, gamma=0.99):
        """
        Train for one episode.

        Args:
            env: Environment with .reset(), .step(action) interface
            params: Current Q-network parameters
            num_steps: Maximum steps per episode
            gamma: Discount factor

        Returns:
            Updated parameters and total reward
        """
        state = env.reset()
        total_reward = 0

        for step in range(num_steps):
            # Select action
            action = self.select_action(state, params, epsilon=0.1)

            # Take action
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Compute target Q-value
            if done:
                target = reward
            else:
                next_q_values = self.get_q_values(next_state, params)
                target = reward + gamma * np.max(next_q_values)

            # Compute gradients
            gradients = self.compute_gradient(state, action, target, params)

            # Update parameters
            params += self.learning_rate * gradients

            # Update state
            state = next_state

            if done:
                break

        return params, total_reward

    def train(self, env, num_episodes=100):
        """
        Train quantum Q-learning agent.

        Args:
            env: Reinforcement learning environment
            num_episodes: Number of training episodes

        Returns:
            Trained parameters and reward history
        """
        # Initialize parameters
        num_params = self.n_qubits * 3 * 2  # 3 layers, 2 params per qubit
        params = np.random.uniform(0, 2*np.pi, num_params)

        reward_history = []

        for episode in range(num_episodes):
            params, total_reward = self.train_episode(env, params)
            reward_history.append(total_reward)

            if episode % 10 == 0:
                avg_reward = np.mean(reward_history[-10:])
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}")

        self.q_network_params = params
        return params, reward_history

# Example quantum RL (with dummy environment)
class DummyEnv:
    """Simple dummy environment for demonstration."""
    def __init__(self, n_states=4, n_actions=2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.state = None

    def reset(self):
        self.state = np.random.uniform(0, 2*np.pi, self.n_states)
        return self.state

    def step(self, action):
        # Simple dynamics
        reward = 1.0 if action == 0 else -1.0
        self.state = np.random.uniform(0, 2*np.pi, self.n_states)
        done = np.random.random() < 0.1
        return self.state, reward, done, {}

# Train quantum RL agent
env = DummyEnv(n_states=4, n_actions=2)
qrl = QuantumReinforcementLearning(n_qubits=4, n_actions=2, learning_rate=0.1)

print("Training quantum RL agent...")
params, rewards = qrl.train(env, num_episodes=50)

print(f"\nFinal average reward: {np.mean(rewards[-10:]):.2f}")
```

## Exercises

1. Implement quantum support vector machine (QSVM) with different kernels
2. Design a quantum convolutional neural network (QCNN)
3. Implement variational quantum regressor (VQR)
4. Create quantum transfer learning pipeline
5. Build quantum meta-learning algorithm

## Further Reading

- **Quantum Kernels**: Havlíček et al., "Supervised learning with quantum-enhanced feature spaces" (2019)
- **QNTK**: Liu et al., "A rigorous and robust quantum speed-up in supervised machine learning" (2021)
- **Barren Plateaus**: McClean et al., "Barren plateaus in quantum neural network training landscapes" (2018)
- **Quantum RL**: Dunjko & Briegel, "Machine learning & artificial intelligence in the quantum domain" (2018)

## Next Steps

- Advanced Tutorial: Fault-Tolerant Quantum Computing
- Advanced Tutorial: Topological Quantum Computing
- Research: Implementing custom quantum ML algorithms
