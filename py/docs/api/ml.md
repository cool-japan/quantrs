# Machine Learning API Reference

Complete reference for quantum machine learning algorithms and tools in QuantRS2, enabling hybrid quantum-classical ML applications.

## Overview

The QuantRS2 ML module provides quantum machine learning algorithms, data encoding methods, and hybrid optimization tools. These enable quantum advantage in machine learning tasks through exponential feature spaces and quantum interference.

```python
import quantrs2
from quantrs2.ml import (
    QuantumNeuralNetwork,
    VariationalQuantumClassifier,
    QuantumKernelMachine,
    AmplitudeEncoding,
    AngleEncoding
)

# Create quantum classifier
qnn = QuantumNeuralNetwork(num_qubits=4, num_layers=3)
qnn.fit(X_train, y_train)
predictions = qnn.predict(X_test)
```

## Data Encoding

### Amplitude Encoding

#### `AmplitudeEncoding(normalization='l2')`

Encode classical data into quantum amplitudes of a quantum state.

**Parameters:**
- `normalization` (str): Normalization method ('l2', 'l1', 'max')

**Methods:**

##### `encode(data, circuit, qubits=None)`
Encode data vector into quantum circuit amplitudes.

**Parameters:**
- `data` (np.ndarray): Classical data vector to encode
- `circuit` (Circuit): Target quantum circuit
- `qubits` (Optional[List[int]]): Qubits to use for encoding

**Example:**
```python
import numpy as np
from quantrs2.ml import AmplitudeEncoding

# Encode 4-dimensional data
data = np.array([0.5, 0.3, 0.2, 0.1])
encoder = AmplitudeEncoding(normalization='l2')

circuit = quantrs2.Circuit(2)  # 2 qubits for 4 amplitudes
encoder.encode(data, circuit)

# The quantum state now has amplitudes proportional to data
result = circuit.run()
print(f"Encoded state: {result.state_probabilities()}")
```

##### `encoding_dimension(num_qubits)`
Calculate maximum data dimension for given number of qubits.

**Parameters:**
- `num_qubits` (int): Number of qubits available

**Returns:**
- `int`: Maximum data dimension (2^num_qubits)

### Angle Encoding

#### `AngleEncoding(rotation_gate='ry', entangling_gate='cx')`

Encode classical data into rotation angles of quantum gates.

**Parameters:**
- `rotation_gate` (str): Single-qubit rotation gate ('rx', 'ry', 'rz')
- `entangling_gate` (str): Two-qubit entangling gate ('cx', 'cz')

**Methods:**

##### `encode(data, circuit, feature_map='linear')`
Encode data into quantum circuit using rotation angles.

**Parameters:**
- `data` (np.ndarray): Classical data vector
- `circuit` (Circuit): Target quantum circuit
- `feature_map` (str): Feature mapping ('linear', 'quadratic', 'fourier')

**Example:**
```python
from quantrs2.ml import AngleEncoding

# Encode 3D data point
data = np.array([0.2, 0.5, 0.8])
encoder = AngleEncoding(rotation_gate='ry')

circuit = quantrs2.Circuit(3)
encoder.encode(data, circuit, feature_map='linear')

# Each data feature is encoded as a rotation angle
result = circuit.run()
print(f"Encoded probabilities: {result.state_probabilities()}")
```

##### `create_feature_map(data_dim, circuit_qubits, map_type='pauli')`
Create quantum feature map for data encoding.

**Parameters:**
- `data_dim` (int): Dimension of input data
- `circuit_qubits` (int): Number of qubits in circuit
- `map_type` (str): Type of feature map ('pauli', 'zz_feature', 'fourier')

**Returns:**
- `FeatureMap`: Quantum feature map object

### Advanced Encoding Methods

#### `BasisEncoding(basis_states='computational')`
Encode data into computational basis states.

**Example:**
```python
from quantrs2.ml import BasisEncoding

# Encode binary data
binary_data = [1, 0, 1, 1]
encoder = BasisEncoding()

circuit = quantrs2.Circuit(4)
encoder.encode(binary_data, circuit)
# Creates state |1011⟩
```

#### `IQPEncoding(num_layers=1, entangling_pattern='linear')`
Instantaneous Quantum Polynomial (IQP) encoding for quantum advantage.

**Parameters:**
- `num_layers` (int): Number of encoding layers
- `entangling_pattern` (str): Pattern for entangling gates ('linear', 'circular', 'all_to_all')

## Quantum Feature Maps

### Pauli Feature Maps

#### `PauliFeatureMap(num_qubits, depth=2, paulis=['X', 'Y', 'Z'])`

Create feature maps using Pauli operators for quantum kernel methods.

**Parameters:**
- `num_qubits` (int): Number of qubits in feature map
- `depth` (int): Depth of feature map circuit
- `paulis` (List[str]): Pauli operators to use

**Methods:**

##### `create_circuit(data_point)`
Create quantum circuit for specific data point.

**Parameters:**
- `data_point` (np.ndarray): Input data vector

**Returns:**
- `Circuit`: Quantum circuit implementing feature map

**Example:**
```python
from quantrs2.ml import PauliFeatureMap

# Create Pauli feature map
feature_map = PauliFeatureMap(
    num_qubits=3,
    depth=2,
    paulis=['X', 'Y', 'Z']
)

# Encode data point
data = np.array([0.5, 0.3, 0.8])
circuit = feature_map.create_circuit(data)

print(f"Feature map circuit depth: {circuit.depth}")
print(f"Feature map gates: {circuit.gate_count}")
```

##### `compute_kernel(x1, x2)`
Compute quantum kernel between two data points.

**Parameters:**
- `x1` (np.ndarray): First data point
- `x2` (np.ndarray): Second data point

**Returns:**
- `float`: Quantum kernel value

### ZZ Feature Maps

#### `ZZFeatureMap(num_qubits, depth=2, data_map_func=None)`

Feature map with ZZ interactions for capturing data correlations.

**Parameters:**
- `num_qubits` (int): Number of qubits
- `depth` (int): Circuit depth
- `data_map_func` (Optional[Callable]): Custom data mapping function

**Example:**
```python
from quantrs2.ml import ZZFeatureMap

# Create ZZ feature map for pairwise interactions
zz_map = ZZFeatureMap(num_qubits=4, depth=3)

# Custom data mapping
def custom_map(x):
    return np.arctan(x)

zz_map_custom = ZZFeatureMap(
    num_qubits=4,
    depth=2,
    data_map_func=custom_map
)
```

## Quantum Classifiers

### Variational Quantum Classifier

#### `VariationalQuantumClassifier(num_qubits, num_layers, optimizer='adam')`

Quantum classifier using variational quantum circuits.

**Parameters:**
- `num_qubits` (int): Number of qubits in quantum circuit
- `num_layers` (int): Number of variational layers
- `optimizer` (str): Classical optimizer ('adam', 'sgd', 'cobyla')

**Methods:**

##### `fit(X, y, epochs=100, batch_size=32)`
Train the quantum classifier.

**Parameters:**
- `X` (np.ndarray): Training data features
- `y` (np.ndarray): Training labels
- `epochs` (int): Number of training epochs
- `batch_size` (int): Batch size for training

**Returns:**
- `TrainingHistory`: Training metrics and loss curves

**Example:**
```python
from quantrs2.ml import VariationalQuantumClassifier
from sklearn.datasets import make_classification

# Generate sample dataset
X, y = make_classification(
    n_samples=100,
    n_features=4,
    n_classes=2,
    random_state=42
)

# Create and train classifier
vqc = VariationalQuantumClassifier(
    num_qubits=4,
    num_layers=3,
    optimizer='adam'
)

# Train the model
history = vqc.fit(X, y, epochs=50)

print(f"Final training accuracy: {history.final_accuracy:.3f}")
print(f"Training time: {history.training_time:.2f}s")
```

##### `predict(X)`
Make predictions on new data.

**Parameters:**
- `X` (np.ndarray): Input data for prediction

**Returns:**
- `np.ndarray`: Predicted class labels

##### `predict_proba(X)`
Predict class probabilities.

**Parameters:**
- `X` (np.ndarray): Input data

**Returns:**
- `np.ndarray`: Predicted class probabilities

##### `score(X, y)`
Calculate accuracy score on test data.

**Parameters:**
- `X` (np.ndarray): Test features
- `y` (np.ndarray): True labels

**Returns:**
- `float`: Accuracy score

### Quantum Neural Network

#### `QuantumNeuralNetwork(num_qubits, num_layers, ansatz='hardware_efficient')`

General quantum neural network for regression and classification.

**Parameters:**
- `num_qubits` (int): Number of qubits
- `num_layers` (int): Number of variational layers
- `ansatz` (str): Type of variational ansatz ('hardware_efficient', 'real_amplitudes', 'efficient_su2')

**Methods:**

##### `add_layer(layer_type, **kwargs)`
Add custom layer to quantum neural network.

**Parameters:**
- `layer_type` (str): Type of layer ('rotation', 'entangling', 'data_encoding')
- `**kwargs`: Layer-specific parameters

**Example:**
```python
from quantrs2.ml import QuantumNeuralNetwork

# Create custom QNN architecture
qnn = QuantumNeuralNetwork(num_qubits=4, num_layers=0)

# Add custom layers
qnn.add_layer('data_encoding', encoding_type='angle')
qnn.add_layer('rotation', rotation_gates=['ry', 'rz'])
qnn.add_layer('entangling', pattern='circular')
qnn.add_layer('rotation', rotation_gates=['ry'])

print(f"QNN architecture: {qnn.num_parameters} parameters")
```

##### `compile(loss='mse', optimizer='adam', metrics=['accuracy'])`
Compile quantum neural network for training.

**Parameters:**
- `loss` (str): Loss function ('mse', 'cross_entropy', 'hinge')
- `optimizer` (str): Optimizer algorithm
- `metrics` (List[str]): Metrics to track during training

##### `forward(x, parameters)`
Forward pass through quantum neural network.

**Parameters:**
- `x` (np.ndarray): Input data
- `parameters` (np.ndarray): Current network parameters

**Returns:**
- `np.ndarray`: Network output

## Quantum Kernel Methods

### Quantum Kernel Machine

#### `QuantumKernelMachine(feature_map, classical_kernel='rbf')`

Quantum-enhanced kernel machine using quantum feature maps.

**Parameters:**
- `feature_map` (FeatureMap): Quantum feature map
- `classical_kernel` (str): Classical kernel for comparison ('rbf', 'linear', 'poly')

**Methods:**

##### `fit(X, y, alpha=1e-3)`
Train quantum kernel machine.

**Parameters:**
- `X` (np.ndarray): Training data
- `y` (np.ndarray): Training labels  
- `alpha` (float): Regularization parameter

**Example:**
```python
from quantrs2.ml import QuantumKernelMachine, PauliFeatureMap
from sklearn.datasets import make_moons

# Generate non-linear dataset
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# Create quantum feature map
feature_map = PauliFeatureMap(num_qubits=2, depth=3)

# Create quantum kernel machine
qkm = QuantumKernelMachine(
    feature_map=feature_map,
    classical_kernel='rbf'
)

# Train model
qkm.fit(X, y, alpha=1e-2)

# Make predictions
predictions = qkm.predict(X)
accuracy = qkm.score(X, y)

print(f"Quantum kernel accuracy: {accuracy:.3f}")
```

##### `compute_kernel_matrix(X1, X2=None)`
Compute quantum kernel matrix between data points.

**Parameters:**
- `X1` (np.ndarray): First set of data points
- `X2` (Optional[np.ndarray]): Second set of data points

**Returns:**
- `np.ndarray`: Quantum kernel matrix

##### `quantum_advantage_score(X, y)`
Estimate quantum advantage over classical kernels.

**Parameters:**
- `X` (np.ndarray): Dataset features
- `y` (np.ndarray): Dataset labels

**Returns:**
- `Dict[str, float]`: Quantum vs classical performance metrics

### Quantum Support Vector Machine

#### `QuantumSVM(feature_map, C=1.0, kernel='quantum')`

Quantum support vector machine with quantum kernels.

**Parameters:**
- `feature_map` (FeatureMap): Quantum feature map
- `C` (float): Regularization parameter
- `kernel` (str): Kernel type ('quantum', 'hybrid')

**Example:**
```python
from quantrs2.ml import QuantumSVM, ZZFeatureMap

# Create quantum SVM
feature_map = ZZFeatureMap(num_qubits=3, depth=2)
qsvm = QuantumSVM(feature_map=feature_map, C=10.0)

# Train and evaluate
qsvm.fit(X_train, y_train)
test_accuracy = qsvm.score(X_test, y_test)

print(f"Quantum SVM accuracy: {test_accuracy:.3f}")
```

## Quantum Generative Models

### Quantum Generative Adversarial Network

#### `QuantumGAN(generator_qubits, discriminator_qubits, latent_dim)`

Quantum generative adversarial network for quantum data generation.

**Parameters:**
- `generator_qubits` (int): Qubits in quantum generator
- `discriminator_qubits` (int): Qubits in quantum discriminator  
- `latent_dim` (int): Dimension of latent space

**Methods:**

##### `train(data, epochs=100, batch_size=16)`
Train quantum GAN on dataset.

**Parameters:**
- `data` (np.ndarray): Training data
- `epochs` (int): Training epochs
- `batch_size` (int): Batch size

**Example:**
```python
from quantrs2.ml import QuantumGAN

# Generate quantum data
qgan = QuantumGAN(
    generator_qubits=4,
    discriminator_qubits=4,
    latent_dim=2
)

# Train on quantum states
quantum_data = generate_quantum_training_data(num_samples=1000)
qgan.train(quantum_data, epochs=200)

# Generate new quantum states
generated_states = qgan.generate(num_samples=50)
```

##### `generate(num_samples, latent_input=None)`
Generate new data samples.

**Parameters:**
- `num_samples` (int): Number of samples to generate
- `latent_input` (Optional[np.ndarray]): Input latent vectors

**Returns:**
- `np.ndarray`: Generated data samples

### Quantum Boltzmann Machine

#### `QuantumBoltzmannMachine(num_visible, num_hidden, temperature=1.0)`

Quantum Boltzmann machine for unsupervised learning.

**Parameters:**
- `num_visible` (int): Number of visible units
- `num_hidden` (int): Number of hidden units
- `temperature` (float): Temperature parameter

**Example:**
```python
from quantrs2.ml import QuantumBoltzmannMachine

# Create QBM for feature learning
qbm = QuantumBoltzmannMachine(
    num_visible=6,
    num_hidden=4,
    temperature=0.5
)

# Train on data
qbm.fit(X_train, learning_rate=0.1, epochs=100)

# Extract learned features
features = qbm.transform(X_test)
```

## Quantum Reinforcement Learning

### Quantum Policy Gradient

#### `QuantumPolicyGradient(num_qubits, num_actions, policy_layers=3)`

Quantum policy gradient for reinforcement learning.

**Parameters:**
- `num_qubits` (int): Qubits for state encoding
- `num_actions` (int): Number of possible actions
- `policy_layers` (int): Layers in policy network

**Methods:**

##### `select_action(state, parameters)`
Select action using quantum policy.

**Parameters:**
- `state` (np.ndarray): Current state
- `parameters` (np.ndarray): Policy parameters

**Returns:**
- `Tuple[int, float]`: (action, action_probability)

**Example:**
```python
from quantrs2.ml import QuantumPolicyGradient

# Create quantum RL agent
agent = QuantumPolicyGradient(
    num_qubits=4,
    num_actions=2,
    policy_layers=3
)

# Training loop
for episode in range(1000):
    state = env.reset()
    episode_reward = 0
    
    while not done:
        # Select action using quantum policy
        action, prob = agent.select_action(state, agent.parameters)
        
        # Take action in environment
        next_state, reward, done, _ = env.step(action)
        
        # Update policy parameters
        agent.update_parameters(state, action, reward, prob)
        
        state = next_state
        episode_reward += reward
    
    print(f"Episode {episode}: Reward = {episode_reward}")
```

##### `update_parameters(state, action, reward, action_prob)`
Update policy parameters using policy gradient.

**Parameters:**
- `state` (np.ndarray): State where action was taken
- `action` (int): Action that was taken
- `reward` (float): Reward received
- `action_prob` (float): Probability of taken action

### Quantum Q-Learning

#### `QuantumQLearning(num_qubits, num_actions, learning_rate=0.1)`

Quantum Q-learning algorithm.

**Parameters:**
- `num_qubits` (int): Qubits for state representation
- `num_actions` (int): Number of actions
- `learning_rate` (float): Learning rate

**Example:**
```python
from quantrs2.ml import QuantumQLearning

# Create Q-learning agent
q_agent = QuantumQLearning(
    num_qubits=3,
    num_actions=4,
    learning_rate=0.05
)

# Train agent
q_agent.train(environment, episodes=5000)

# Evaluate policy
avg_reward = q_agent.evaluate(environment, episodes=100)
```

## Optimization and Training

### Quantum-Aware Optimizers

#### `QuantumAdam(learning_rate=0.01, beta1=0.9, beta2=0.999)`

Adam optimizer adapted for quantum parameter optimization.

**Parameters:**
- `learning_rate` (float): Learning rate
- `beta1` (float): First moment decay rate
- `beta2` (float): Second moment decay rate

**Methods:**

##### `step(gradients, parameters)`
Update parameters using quantum-aware Adam.

**Parameters:**
- `gradients` (np.ndarray): Parameter gradients
- `parameters` (np.ndarray): Current parameters

**Returns:**
- `np.ndarray`: Updated parameters

#### `SPSA(learning_rate=0.1, perturbation=0.01)`

Simultaneous Perturbation Stochastic Approximation for gradient-free optimization.

**Parameters:**
- `learning_rate` (float): Learning rate
- `perturbation` (float): Perturbation magnitude

**Example:**
```python
from quantrs2.ml import SPSA

# Use SPSA for noisy quantum circuits
optimizer = SPSA(learning_rate=0.05, perturbation=0.01)

def objective_function(parameters):
    circuit = create_variational_circuit(parameters)
    result = circuit.run()
    return calculate_cost(result)

# Optimize parameters
optimal_params = optimizer.minimize(
    objective_function,
    initial_parameters,
    max_iterations=1000
)
```

### Gradient Computation

#### `parameter_shift_gradient(circuit, parameters, observables)`

Compute gradients using parameter shift rule.

**Parameters:**
- `circuit` (Circuit): Parameterized quantum circuit
- `parameters` (np.ndarray): Current parameter values
- `observables` (List[Observable]): Observables to measure

**Returns:**
- `np.ndarray`: Gradient vector

**Example:**
```python
from quantrs2.ml import parameter_shift_gradient

# Define parameterized circuit
def create_circuit(params):
    circuit = quantrs2.Circuit(2)
    circuit.ry(0, params[0])
    circuit.ry(1, params[1])
    circuit.cx(0, 1)
    return circuit

# Compute gradients
params = np.array([0.5, 1.2])
gradients = parameter_shift_gradient(
    lambda p: create_circuit(p),
    params,
    observables=['Z0', 'Z1', 'Z0*Z1']
)

print(f"Gradients: {gradients}")
```

#### `finite_difference_gradient(objective_func, parameters, epsilon=1e-6)`

Compute gradients using finite differences.

**Parameters:**
- `objective_func` (Callable): Objective function
- `parameters` (np.ndarray): Parameters
- `epsilon` (float): Finite difference step size

## Hybrid Classical-Quantum Models

### Quantum Transfer Learning

#### `QuantumTransferLearning(pretrained_layers, quantum_layers)`

Transfer learning with quantum layers.

**Parameters:**
- `pretrained_layers` (List): Pretrained classical layers
- `quantum_layers` (List): Quantum processing layers

**Example:**
```python
from quantrs2.ml import QuantumTransferLearning
import torch.nn as nn

# Classical pretrained layers
classical_layers = [
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64)
]

# Quantum processing layers
quantum_layers = [
    QuantumNeuralNetwork(num_qubits=6, num_layers=3)
]

# Create hybrid model
hybrid_model = QuantumTransferLearning(
    pretrained_layers=classical_layers,
    quantum_layers=quantum_layers
)

# Fine-tune on quantum-enhanced features
hybrid_model.fit(X_train, y_train, quantum_advantage=True)
```

### Quantum-Classical Ensembles

#### `QuantumEnsemble(quantum_models, classical_models, voting='soft')`

Ensemble of quantum and classical models.

**Parameters:**
- `quantum_models` (List): List of quantum models
- `classical_models` (List): List of classical models
- `voting` (str): Voting strategy ('soft', 'hard', 'weighted')

**Example:**
```python
from quantrs2.ml import QuantumEnsemble
from sklearn.ensemble import RandomForestClassifier

# Create diverse model ensemble
quantum_models = [
    VariationalQuantumClassifier(num_qubits=4, num_layers=2),
    QuantumKernelMachine(PauliFeatureMap(num_qubits=4)),
    QuantumSVM(ZZFeatureMap(num_qubits=4))
]

classical_models = [
    RandomForestClassifier(n_estimators=100),
    # Add other classical models
]

# Create ensemble
ensemble = QuantumEnsemble(
    quantum_models=quantum_models,
    classical_models=classical_models,
    voting='weighted'
)

# Train ensemble
ensemble.fit(X_train, y_train)

# Make ensemble predictions
predictions = ensemble.predict(X_test)
confidence = ensemble.predict_proba(X_test)
```

## Performance Analysis

### Quantum Advantage Metrics

#### `quantum_advantage_analysis(quantum_model, classical_baseline, test_data)`

Analyze quantum advantage over classical methods.

**Parameters:**
- `quantum_model`: Trained quantum model
- `classical_baseline`: Classical baseline model
- `test_data` (Tuple): (X_test, y_test)

**Returns:**
- `AdvantageReport`: Comprehensive advantage analysis

**Example:**
```python
from quantrs2.ml import quantum_advantage_analysis
from sklearn.svm import SVC

# Train models
quantum_model = QuantumSVM(feature_map)
classical_model = SVC(kernel='rbf')

quantum_model.fit(X_train, y_train)
classical_model.fit(X_train, y_train)

# Analyze quantum advantage
advantage = quantum_advantage_analysis(
    quantum_model,
    classical_model,
    (X_test, y_test)
)

print(f"Quantum accuracy: {advantage.quantum_accuracy:.3f}")
print(f"Classical accuracy: {advantage.classical_accuracy:.3f}")
print(f"Advantage factor: {advantage.advantage_factor:.2f}")
print(f"Statistical significance: p = {advantage.p_value:.4f}")
```

### Model Interpretability

#### `explain_quantum_model(model, data_point, method='permutation')`

Explain quantum model predictions.

**Parameters:**
- `model`: Trained quantum model
- `data_point` (np.ndarray): Input to explain
- `method` (str): Explanation method ('permutation', 'gradient', 'quantum_shap')

**Returns:**
- `ExplanationResult`: Feature importance and explanations

**Example:**
```python
from quantrs2.ml import explain_quantum_model

# Explain model prediction
explanation = explain_quantum_model(
    quantum_model,
    X_test[0],
    method='quantum_shap'
)

print(f"Prediction: {explanation.prediction}")
print(f"Feature importance: {explanation.feature_importance}")
print(f"Quantum circuit contribution: {explanation.quantum_contribution}")
```

## Utilities and Helpers

### Data Preprocessing

#### `quantum_preprocess(X, method='standardize', target_range=(0, 1))`

Preprocess data for quantum machine learning.

**Parameters:**
- `X` (np.ndarray): Input data
- `method` (str): Preprocessing method ('standardize', 'normalize', 'minmax')
- `target_range` (Tuple[float, float]): Target range for scaling

**Returns:**
- `Tuple[np.ndarray, Dict]`: (processed_data, preprocessing_info)

#### `create_quantum_dataset(size, dim, task='classification', noise=0.1)`

Generate synthetic dataset suitable for quantum ML.

**Parameters:**
- `size` (int): Dataset size
- `dim` (int): Data dimension
- `task` (str): Task type ('classification', 'regression')
- `noise` (float): Noise level

**Example:**
```python
from quantrs2.ml import create_quantum_dataset

# Generate quantum-friendly dataset
X, y = create_quantum_dataset(
    size=200,
    dim=4,
    task='classification',
    noise=0.05
)

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")
```

### Visualization

#### `plot_quantum_decision_boundary(model, X, y, resolution=100)`

Plot decision boundary for 2D quantum classifier.

**Parameters:**
- `model`: Trained quantum model
- `X` (np.ndarray): 2D input data
- `y` (np.ndarray): Labels
- `resolution` (int): Plot resolution

#### `visualize_quantum_circuit_learning(circuit, parameters_history)`

Visualize how quantum circuit evolves during training.

**Parameters:**
- `circuit` (Circuit): Quantum circuit
- `parameters_history` (List[np.ndarray]): Parameter evolution

**Example:**
```python
from quantrs2.ml import plot_quantum_decision_boundary

# Plot decision boundary
plot_quantum_decision_boundary(
    trained_vqc,
    X_test[:, :2],  # Use first 2 features
    y_test,
    resolution=200
)
```

## Best Practices

### Model Selection

```python
def select_quantum_model(X, y, task_type='classification'):
    """Recommend quantum model based on data characteristics."""
    
    n_samples, n_features = X.shape
    
    recommendations = []
    
    # For small datasets with complex patterns
    if n_samples < 1000 and n_features <= 10:
        recommendations.append(QuantumKernelMachine)
    
    # For larger datasets
    if n_samples > 1000:
        recommendations.append(VariationalQuantumClassifier)
    
    # For high-dimensional data
    if n_features > 10:
        recommendations.append(QuantumNeuralNetwork)
    
    return recommendations
```

### Hyperparameter Optimization

```python
from quantrs2.ml import QuantumBayesianOptimization

def optimize_quantum_hyperparameters(model_class, X, y):
    """Optimize quantum model hyperparameters."""
    
    # Define search space
    search_space = {
        'num_layers': (1, 10),
        'learning_rate': (0.001, 0.1),
        'batch_size': (8, 64)
    }
    
    # Use quantum-aware Bayesian optimization
    optimizer = QuantumBayesianOptimization(
        model_class=model_class,
        search_space=search_space,
        acquisition_function='expected_improvement'
    )
    
    best_params = optimizer.optimize(X, y, n_trials=50)
    return best_params
```

### Error Handling

```python
def robust_quantum_training(model, X, y, max_retries=3):
    """Train quantum model with error handling."""
    
    for attempt in range(max_retries):
        try:
            # Train with noise-aware techniques
            history = model.fit(
                X, y,
                validation_split=0.2,
                early_stopping=True,
                noise_mitigation=True
            )
            
            # Validate training success
            if history.final_loss < 0.1:
                return model, history
                
        except QuantumTrainingError as e:
            print(f"Training attempt {attempt + 1} failed: {e}")
            
            # Adjust hyperparameters for next attempt
            model.learning_rate *= 0.5
            model.add_noise_regularization()
    
    raise RuntimeError("Quantum training failed after all retries")
```

## Related Documentation

- [Quantum Machine Learning Tutorial](../tutorials/beginner/05-quantum-machine-learning.md) - Learning QML fundamentals
- [Core API](core.md) - Basic circuit operations
- [Mitigation API](mitigation.md) - Error mitigation for QML
- [Examples Gallery](../examples/ml/) - Complete QML examples

## Examples

### Complete QML Pipeline

```python
import quantrs2
from quantrs2.ml import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def complete_qml_pipeline():
    """Complete quantum machine learning pipeline."""
    
    # 1. Generate/load data
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_classes=2,
        n_redundant=0,
        random_state=42
    )
    
    # 2. Preprocess for quantum ML
    X_processed, preprocessing_info = quantum_preprocess(
        X, 
        method='minmax',
        target_range=(0, 1)
    )
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42
    )
    
    # 4. Create quantum model
    feature_map = PauliFeatureMap(num_qubits=4, depth=2)
    
    quantum_model = QuantumKernelMachine(
        feature_map=feature_map
    )
    
    # 5. Train model
    quantum_model.fit(X_train, y_train)
    
    # 6. Make predictions
    y_pred_quantum = quantum_model.predict(X_test)
    quantum_accuracy = accuracy_score(y_test, y_pred_quantum)
    
    # 7. Compare with classical baseline
    from sklearn.svm import SVC
    classical_model = SVC(kernel='rbf')
    classical_model.fit(X_train, y_train)
    y_pred_classical = classical_model.predict(X_test)
    classical_accuracy = accuracy_score(y_test, y_pred_classical)
    
    # 8. Analyze quantum advantage
    advantage = quantum_advantage_analysis(
        quantum_model,
        classical_model,
        (X_test, y_test)
    )
    
    print("Quantum Machine Learning Results:")
    print(f"  Quantum accuracy: {quantum_accuracy:.3f}")
    print(f"  Classical accuracy: {classical_accuracy:.3f}")
    print(f"  Quantum advantage: {advantage.advantage_factor:.2f}x")
    print(f"  Statistical significance: p = {advantage.p_value:.4f}")
    
    return quantum_model, advantage

# Run complete pipeline
model, results = complete_qml_pipeline()
```

---

*Quantum machine learning opens new possibilities for AI.* Start with the [QML Tutorial](../tutorials/beginner/05-quantum-machine-learning.md) and explore [QML Examples](../examples/ml/)!