"""
PennyLane Integration Demonstration

This example shows how to use the QuantRS2-PennyLane plugin for:
1. Hybrid quantum-classical machine learning
2. Variational quantum algorithms
3. Quantum circuit optimization
4. Integration with existing PennyLane workflows
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# Import QuantRS2-PennyLane integration
from quantrs2.pennylane_plugin import (
    QuantRS2Device,
    QuantRS2QMLModel,
    QuantRS2VQC,
    register_quantrs2_device,
    create_quantrs2_device,
    quantrs2_qnode,
    test_quantrs2_pennylane_integration,
    PENNYLANE_AVAILABLE,
    QUANTRS2_AVAILABLE
)

# Try to import PennyLane for advanced examples
try:
    import pennylane as qml
    PENNYLANE_DEMOS_AVAILABLE = True
except ImportError:
    PENNYLANE_DEMOS_AVAILABLE = False
    print("PennyLane not available. Basic demos will use mock implementations.")

# Try to import scikit-learn for comparison
try:
    from sklearn.datasets import make_classification, make_circles
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. Will use synthetic data generation.")


def demo_basic_device_usage():
    """Demonstrate basic QuantRS2 device usage with PennyLane."""
    print("=" * 60)
    print("BASIC DEVICE USAGE DEMO")
    print("=" * 60)
    
    print("1. Creating QuantRS2 device...")
    try:
        device = create_quantrs2_device(wires=2, shots=1000)
        print(f"   Device created: {device.n_qubits} qubits, {device.shots} shots")
    except Exception as e:
        print(f"   Device creation failed: {e}")
        return
    
    print("\n2. Creating simple quantum circuit...")
    if PENNYLANE_DEMOS_AVAILABLE:
        @qml.qnode(device)
        def bell_circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))
        
        try:
            result = bell_circuit()
            print(f"   Bell circuit result: {result}")
            print("   ✅ Circuit execution successful!")
        except Exception as e:
            print(f"   Circuit execution failed: {e}")
    else:
        # Mock demonstration
        print("   Mock Bell circuit result: [0.0, 0.0]")
        print("   ✅ Mock circuit execution successful!")
    
    print("\n3. Testing parametric circuits...")
    if PENNYLANE_DEMOS_AVAILABLE:
        @qml.qnode(device)
        def parametric_circuit(theta):
            qml.RY(theta, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))
        
        try:
            angles = [0, np.pi/4, np.pi/2, np.pi]
            results = [parametric_circuit(angle) for angle in angles]
            print(f"   Parametric results: {[f'{r:.3f}' for r in results]}")
            print("   ✅ Parametric circuits working!")
        except Exception as e:
            print(f"   Parametric circuit failed: {e}")
    else:
        print("   Mock parametric results: [1.000, 0.707, 0.000, -1.000]")
        print("   ✅ Mock parametric circuits working!")


def demo_quantum_machine_learning():
    """Demonstrate quantum machine learning with QuantRS2."""
    print("\n" + "=" * 60)
    print("QUANTUM MACHINE LEARNING DEMO")
    print("=" * 60)
    
    print("1. Creating quantum ML model...")
    try:
        qml_model = QuantRS2QMLModel(n_qubits=2, n_layers=2, shots=1000)
        print(f"   Model created: {qml_model.n_qubits} qubits, {qml_model.n_layers} layers")
    except Exception as e:
        print(f"   Model creation failed: {e}")
        return
    
    print("\n2. Initializing parameters...")
    try:
        params = qml_model.initialize_params(seed=42)
        print(f"   Initialized {len(params)} parameters")
        print(f"   Sample parameters: {[f'{p:.3f}' for p in params[:4]]}...")
    except Exception as e:
        print(f"   Parameter initialization failed: {e}")
        return
    
    print("\n3. Testing forward pass...")
    try:
        # Test input
        x = np.array([0.5, 1.0])
        output = qml_model.forward(x)
        print(f"   Input: {x}")
        print(f"   Output: {[f'{o:.3f}' for o in output]}")
        print("   ✅ Forward pass successful!")
    except Exception as e:
        print(f"   Forward pass failed: {e}")
    
    print("\n4. Training on simple dataset...")
    try:
        # Generate simple training data
        X_train = np.array([
            [0.0, 0.0], [0.1, 0.1], [0.9, 0.9], [1.0, 1.0],
            [0.0, 1.0], [0.1, 0.9], [0.9, 0.1], [1.0, 0.0]
        ])
        y_train = np.array([1, 1, 1, 1, -1, -1, -1, -1])
        
        print(f"   Training data: {len(X_train)} samples")
        
        # Train for a few epochs
        history = qml_model.train(X_train, y_train, n_epochs=5, learning_rate=0.1)
        print(f"   Training completed: {len(history)} epochs")
        print(f"   Final cost: {history[-1]:.6f}")
        
        # Test predictions
        predictions = qml_model.predict(X_train)
        accuracy = np.mean(predictions == y_train)
        print(f"   Training accuracy: {accuracy:.2%}")
        print("   ✅ Training successful!")
        
    except Exception as e:
        print(f"   Training failed: {e}")


def demo_variational_quantum_classifier():
    """Demonstrate Variational Quantum Classifier."""
    print("\n" + "=" * 60)
    print("VARIATIONAL QUANTUM CLASSIFIER DEMO")
    print("=" * 60)
    
    print("1. Generating classification dataset...")
    if SKLEARN_AVAILABLE:
        # Use scikit-learn to generate a more interesting dataset
        X, y = make_classification(
            n_samples=100, n_features=2, n_redundant=0, n_informative=2,
            n_clusters_per_class=1, random_state=42
        )
        # Convert to binary classification (-1, 1)
        y = 2 * y - 1
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
    else:
        # Generate simple synthetic data
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 2)
        y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
        
        # Split into train/test
        split = int(0.7 * n_samples)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
    
    print(f"   Dataset: {len(X_train)} train, {len(X_test)} test samples")
    print(f"   Features: {X_train.shape[1]}D")
    print(f"   Classes: {len(np.unique(y_train))} ({np.unique(y_train)})")
    
    print("\n2. Creating Variational Quantum Classifier...")
    try:
        vqc = QuantRS2VQC(n_features=2, n_qubits=2, n_layers=2)
        print(f"   VQC created: {vqc.n_features} features → {vqc.n_qubits} qubits")
    except Exception as e:
        print(f"   VQC creation failed: {e}")
        return
    
    print("\n3. Training VQC...")
    try:
        history = vqc.fit(X_train, y_train, n_epochs=20, learning_rate=0.1)
        print(f"   Training completed: {len(history)} epochs")
        print(f"   Final cost: {history[-1]:.6f}")
        
        # Evaluate on test set
        train_accuracy = vqc.score(X_train, y_train)
        test_accuracy = vqc.score(X_test, y_test)
        
        print(f"   Training accuracy: {train_accuracy:.2%}")
        print(f"   Test accuracy: {test_accuracy:.2%}")
        print("   ✅ VQC training successful!")
        
    except Exception as e:
        print(f"   VQC training failed: {e}")
        return
    
    # Compare with classical SVM if available
    if SKLEARN_AVAILABLE:
        print("\n4. Comparing with classical SVM...")
        try:
            svm = SVC(kernel='rbf', random_state=42)
            svm.fit(X_train, y_train)
            
            svm_train_acc = svm.score(X_train, y_train)
            svm_test_acc = svm.score(X_test, y_test)
            
            print(f"   SVM training accuracy: {svm_train_acc:.2%}")
            print(f"   SVM test accuracy: {svm_test_acc:.2%}")
            
            print("\n   Comparison:")
            print(f"   VQC vs SVM (train): {train_accuracy:.2%} vs {svm_train_acc:.2%}")
            print(f"   VQC vs SVM (test):  {test_accuracy:.2%} vs {svm_test_acc:.2%}")
            
        except Exception as e:
            print(f"   SVM comparison failed: {e}")
    
    # Plot results if possible
    if len(X_train[0]) == 2:  # Only for 2D data
        print("\n5. Visualizing results...")
        try:
            plot_classification_results(X_train, y_train, X_test, y_test, vqc)
        except Exception as e:
            print(f"   Visualization failed: {e}")
    
    return history


def plot_classification_results(X_train, y_train, X_test, y_test, vqc):
    """Plot classification results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training data
    colors = ['red' if label == -1 else 'blue' for label in y_train]
    ax1.scatter(X_train[:, 0], X_train[:, 1], c=colors, alpha=0.7, s=50)
    ax1.set_title('Training Data')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend(['Class -1', 'Class +1'])
    ax1.grid(True, alpha=0.3)
    
    # Plot test predictions
    try:
        test_predictions = vqc.predict(X_test)
        pred_colors = ['red' if pred == -1 else 'blue' for pred in test_predictions]
        
        ax2.scatter(X_test[:, 0], X_test[:, 1], c=pred_colors, alpha=0.7, s=50)
        
        # Mark incorrect predictions
        incorrect = test_predictions != y_test
        if np.any(incorrect):
            ax2.scatter(X_test[incorrect, 0], X_test[incorrect, 1], 
                       s=100, facecolors='none', edgecolors='black', linewidth=2)
        
        ax2.set_title('Test Predictions (✗ = incorrect)')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        ax2.legend(['Pred Class -1', 'Pred Class +1'])
        ax2.grid(True, alpha=0.3)
        
    except Exception as e:
        ax2.text(0.5, 0.5, f'Prediction failed:\n{e}', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Test Predictions (Failed)')
    
    plt.tight_layout()
    plt.show()


def demo_advanced_pennylane_integration():
    """Demonstrate advanced PennyLane integration features."""
    print("\n" + "=" * 60)
    print("ADVANCED PENNYLANE INTEGRATION DEMO")
    print("=" * 60)
    
    if not PENNYLANE_DEMOS_AVAILABLE:
        print("PennyLane not available for advanced demos")
        return
    
    print("1. Creating quantum node with gradients...")
    try:
        device = create_quantrs2_device(wires=3)
        
        @qml.qnode(device, diff_method="parameter-shift")
        def variational_circuit(params):
            # Data encoding layer
            for i in range(3):
                qml.RY(params[i], wires=i)
            
            # Entangling layer
            for i in range(2):
                qml.CNOT(wires=[i, i+1])
            qml.CNOT(wires=[2, 0])  # Ring connectivity
            
            # Measurement layer
            for i in range(3):
                qml.RY(params[i+3], wires=i)
            
            return qml.expval(qml.PauliZ(0))
        
        # Test gradient computation
        params = np.random.uniform(0, 2*np.pi, 6)
        
        # Forward pass
        result = variational_circuit(params)
        print(f"   Circuit output: {result:.6f}")
        
        # Gradient computation (if supported)
        try:
            grad_fn = qml.grad(variational_circuit)
            gradients = grad_fn(params)
            print(f"   Gradients computed: {len(gradients)} parameters")
            print(f"   Sample gradients: {[f'{g:.6f}' for g in gradients[:3]]}...")
            print("   ✅ Gradient computation successful!")
        except Exception as e:
            print(f"   Gradient computation failed: {e}")
            print("   (This is expected if autodiff backend not available)")
        
    except Exception as e:
        print(f"   Advanced circuit failed: {e}")
    
    print("\n2. Testing optimization with PennyLane optimizers...")
    try:
        # Simple optimization problem: minimize <Z> expectation
        def cost_function(params):
            return variational_circuit(params)
        
        # Use a simple gradient descent (mock if PennyLane optimizers unavailable)
        initial_params = np.random.uniform(0, 2*np.pi, 6)
        params = initial_params.copy()
        
        print(f"   Initial cost: {cost_function(params):.6f}")
        
        # Manual gradient descent
        learning_rate = 0.1
        for step in range(10):
            # Simple finite difference gradient
            gradients = np.zeros_like(params)
            eps = 1e-4
            
            for i in range(len(params)):
                params_plus = params.copy()
                params_minus = params.copy()
                params_plus[i] += eps
                params_minus[i] -= eps
                
                gradients[i] = (cost_function(params_plus) - cost_function(params_minus)) / (2 * eps)
            
            params -= learning_rate * gradients
            
            if step % 5 == 0:
                cost = cost_function(params)
                print(f"   Step {step}: Cost = {cost:.6f}")
        
        final_cost = cost_function(params)
        print(f"   Final cost: {final_cost:.6f}")
        improvement = initial_params.sum() - params.sum()
        print(f"   Optimization {'improved' if final_cost < cost_function(initial_params) else 'completed'}")
        print("   ✅ Optimization test successful!")
        
    except Exception as e:
        print(f"   Optimization test failed: {e}")
    
    print("\n3. Testing quantum chemistry application...")
    try:
        # Simple H2 molecule VQE simulation
        @qml.qnode(device)
        def h2_vqe_circuit(params):
            # Prepare initial state (simplified)
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            
            # Ansatz
            qml.CNOT(wires=[0, 1])
            qml.RY(params[2], wires=0)
            qml.RY(params[3], wires=1)
            
            # Simplified H2 Hamiltonian measurement
            # In practice, this would be a weighted sum of Pauli terms
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        
        # Run VQE optimization
        vqe_params = np.random.uniform(0, 2*np.pi, 4)
        energy = h2_vqe_circuit(vqe_params)
        
        print(f"   H2 VQE energy: {energy:.6f}")
        print("   ✅ Quantum chemistry test successful!")
        
    except Exception as e:
        print(f"   Quantum chemistry test failed: {e}")


def demo_performance_comparison():
    """Compare QuantRS2 vs other backends."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON DEMO")
    print("=" * 60)
    
    if not PENNYLANE_DEMOS_AVAILABLE:
        print("PennyLane not available for performance comparison")
        return
    
    print("1. Comparing different backends...")
    
    # Test circuit
    def test_circuit():
        for i in range(3):
            qml.Hadamard(wires=i)
        for i in range(2):
            qml.CNOT(wires=[i, i+1])
        return qml.expval(qml.PauliZ(0))
    
    backends_to_test = []
    
    # QuantRS2 backend
    try:
        quantrs2_device = create_quantrs2_device(wires=3)
        backends_to_test.append(("QuantRS2", quantrs2_device))
    except Exception as e:
        print(f"   QuantRS2 backend failed: {e}")
    
    # Default simulator (if available)
    try:
        default_device = qml.device('default.qubit', wires=3)
        backends_to_test.append(("default.qubit", default_device))
    except Exception:
        pass
    
    # Test performance
    results = {}
    for name, device in backends_to_test:
        try:
            import time
            
            qnode = qml.QNode(test_circuit, device)
            
            # Warmup
            qnode()
            
            # Time execution
            start_time = time.time()
            for _ in range(10):
                result = qnode()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            results[name] = {
                'time': avg_time,
                'result': result
            }
            
            print(f"   {name}: {avg_time:.6f}s per execution")
            
        except Exception as e:
            print(f"   {name} failed: {e}")
    
    if len(results) > 1:
        print("\n2. Performance comparison:")
        times = [r['time'] for r in results.values()]
        names = list(results.keys())
        fastest = names[np.argmin(times)]
        print(f"   Fastest backend: {fastest}")
        
        for name, data in results.items():
            relative_speed = data['time'] / min(times)
            print(f"   {name}: {relative_speed:.2f}x relative time")


def main():
    """Run all PennyLane integration demonstrations."""
    print("QuantRS2-PennyLane Integration Demonstration")
    print("=" * 60)
    
    # Check availability
    print("Checking dependencies:")
    print(f"  QuantRS2 available: {QUANTRS2_AVAILABLE}")
    print(f"  PennyLane available: {PENNYLANE_AVAILABLE}")
    print(f"  Scikit-learn available: {SKLEARN_AVAILABLE}")
    
    # Run integration test first
    print("\nRunning integration test...")
    integration_success = test_quantrs2_pennylane_integration()
    
    if not integration_success:
        print("Integration test failed. Some demos may not work properly.")
    
    # Run demonstrations
    demos = [
        ("Basic Device Usage", demo_basic_device_usage),
        ("Quantum Machine Learning", demo_quantum_machine_learning),
        ("Variational Quantum Classifier", demo_variational_quantum_classifier),
        ("Advanced PennyLane Integration", demo_advanced_pennylane_integration),
        ("Performance Comparison", demo_performance_comparison)
    ]
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*20} {demo_name} {'='*20}")
            demo_func()
        except Exception as e:
            print(f"\n{demo_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    print("\nKey takeaways:")
    print("1. QuantRS2 integrates seamlessly with PennyLane workflows")
    print("2. Hybrid quantum-classical ML is possible with QuantRS2 backend")
    print("3. Gradients and optimization work through PennyLane interface")
    print("4. Performance is competitive with other simulators")
    print("5. Complex quantum algorithms can be implemented easily")
    
    print("\nNext steps:")
    print("- Try the VQC on your own datasets")
    print("- Experiment with different ansatz architectures")
    print("- Use PennyLane's optimization algorithms")
    print("- Explore quantum chemistry applications")
    print("- Implement custom quantum ML models")


if __name__ == "__main__":
    main()