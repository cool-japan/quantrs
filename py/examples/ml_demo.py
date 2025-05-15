#!/usr/bin/env python3
"""
Quantum Machine Learning Demo for QuantRS2.

This script demonstrates the machine learning capabilities of the QuantRS2
framework, including Quantum Neural Networks and Variational Quantum Algorithms.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import time

# Add parent directory to path for imports when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def print_section(title):
    """Print a section header."""
    print(f"\n{title}")
    print("="*len(title))

def qnn_classifier_demo():
    """Demonstrate a Quantum Neural Network for classification."""
    print_section("Quantum Neural Network Classification Demo")
    
    try:
        import quantrs2 as qr
        from quantrs2.ml import QNN
        
        # Load the iris dataset
        iris = datasets.load_iris()
        X = iris.data[:, :2]  # Use only the first two features
        y = iris.target
        
        # Normalize features
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        
        # Create a QNN with 4 qubits
        print("Creating a 4-qubit QNN...")
        qnn = QNN(n_qubits=4, n_layers=2)
        
        # Select a few samples to classify
        sample_indices = [0, 50, 100]  # One sample from each class
        samples = X[sample_indices]
        true_labels = y[sample_indices]
        
        print("\nRunning classification on sample data...")
        print("Sample | Features | Prediction")
        print("-" * 40)
        
        # Process each sample
        for i, sample in enumerate(samples):
            # Forward pass
            start_time = time.time()
            output = qnn.forward(sample.reshape(1, -1))
            elapsed = time.time() - start_time
            
            # Convert output to class prediction (simple thresholding)
            predicted_class = int(np.argmax(output[0, :3]))  # Use first 3 outputs for 3 classes
            
            # Print results
            print(f"{i} | {sample} | Class {predicted_class} (true: {true_labels[i]}) in {elapsed:.3f}s")
        
        print("\nThis demonstrates how a quantum neural network can be used")
        print("for classification tasks with minimal classical post-processing.")
        
    except ImportError as e:
        print(f"Error importing quantrs2: {e}")
        print("Make sure the package is properly installed.")

def vqe_demo():
    """Demonstrate a Variational Quantum Eigensolver."""
    print_section("Variational Quantum Eigensolver Demo")
    
    try:
        import quantrs2 as qr
        from quantrs2.ml import VQE
        
        # Create a simple 2-qubit Hamiltonian for demonstration
        # This represents a simple ZZ interaction
        print("Creating a 2-qubit VQE solver for a ZZ Hamiltonian...")
        vqe = VQE(n_qubits=2)
        
        # Calculate energy with random parameters
        initial_params = np.random.randn(4) * 0.1
        initial_energy = vqe.expectation(initial_params)
        print(f"Initial energy: {initial_energy:.6f}")
        
        # Run optimization
        print("\nRunning VQE optimization (simplified version)...")
        iterations = 10
        energy_history = []
        
        # Simulate optimization process
        parameters = initial_params.copy()
        for i in range(iterations):
            # Evaluate energy
            energy = vqe.expectation(parameters)
            energy_history.append(energy)
            
            # Update parameters (simple noise-based update for demonstration)
            parameters = parameters - 0.1 * np.sin(parameters)
            
            print(f"Iteration {i+1:2d}: Energy = {energy:.6f}")
        
        # Calculate final energy
        final_energy = vqe.expectation(parameters)
        print(f"\nFinal energy: {final_energy:.6f}")
        print(f"Energy improvement: {initial_energy - final_energy:.6f}")
        
        print("\nThis demonstrates how VQE can be used to find the")
        print("ground state energy of a quantum Hamiltonian.")
        
        # Plot energy history
        plt.figure(figsize=(8, 4))
        plt.plot(range(iterations), energy_history, 'b-o')
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('VQE Optimization')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('vqe_optimization.png')
        print("\nVQE optimization plot saved as 'vqe_optimization.png'")
        
    except ImportError as e:
        print(f"Error importing quantrs2: {e}")
        print("Make sure the package is properly installed.")

def hep_classification_demo():
    """Demonstrate High-Energy Physics classification."""
    print_section("High-Energy Physics Classification Demo")
    
    try:
        import quantrs2 as qr
        from quantrs2.ml import HEPClassifier
        
        print("Creating a HEP classifier for particle collision data...")
        classifier = HEPClassifier(n_qubits=5, n_features=4, n_classes=2)
        
        # Create synthetic HEP data (simplified)
        n_samples = 10
        np.random.seed(42)
        X = np.random.randn(n_samples, 4)  # 4 features
        y = np.random.randint(0, 2, size=n_samples)  # Binary labels
        
        # Train classifier (simplified)
        print("\nTraining classifier on synthetic collision data...")
        results = classifier.fit(X, y, iterations=5)
        
        # Show final accuracy
        final_accuracy = results['accuracy'][-1] if results['accuracy'] else 0
        print(f"Final training accuracy: {final_accuracy:.4f}")
        
        # Test on one sample
        test_sample = np.random.randn(4)
        prediction = classifier.predict_single(test_sample)
        print(f"\nSample prediction: Class {prediction}")
        
        print("\nThis demonstrates how quantum algorithms can be applied")
        print("to particle physics data classification tasks.")
        
    except ImportError as e:
        print(f"Error importing quantrs2: {e}")
        print("Make sure the package is properly installed.")

def quantum_gan_demo():
    """Demonstrate a Quantum Generative Adversarial Network."""
    print_section("Quantum GAN Demo")
    
    try:
        import quantrs2 as qr
        from quantrs2.ml import QuantumGAN
        
        print("Creating a Quantum GAN...")
        qgan = QuantumGAN(generator_qubits=3, discriminator_qubits=2, 
                          latent_dim=2, data_dim=4)
        
        # Create simple synthetic data (e.g., representing financial time series)
        n_samples = 20
        np.random.seed(42)
        real_data = np.random.randn(n_samples, 4)
        
        # Train the GAN (simplified)
        print("\nTraining QGAN on synthetic data...")
        history = qgan.train(real_data, iterations=5, batch_size=4)
        
        # Generate samples
        print("\nGenerating samples from trained generator...")
        gen_samples = qgan.generate_samples(3)
        
        print("Generated samples:")
        for i, sample in enumerate(gen_samples):
            print(f"Sample {i+1}: {sample}")
        
        print("\nThis demonstrates how quantum GANs can generate")
        print("synthetic data samples using quantum circuits.")
        
    except ImportError as e:
        print(f"Error importing quantrs2: {e}")
        print("Make sure the package is properly installed.")

def main():
    """Run all ML demonstrations."""
    print("QuantRS2 Machine Learning Demonstrations")
    print("=======================================\n")
    
    try:
        import quantrs2
        print(f"QuantRS2 version: {quantrs2.__version__}")
    except ImportError:
        print("QuantRS2 not found. Please install it first.")
        return
    
    # Run demonstrations
    qnn_classifier_demo()
    vqe_demo()
    hep_classification_demo()
    quantum_gan_demo()
    
    print("\nAll demonstrations completed.")

if __name__ == "__main__":
    main()