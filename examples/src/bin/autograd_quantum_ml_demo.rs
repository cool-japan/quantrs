//! Automatic Differentiation for Quantum Machine Learning
//!
//! This example demonstrates how to use `SciRS2`'s automatic differentiation
//! capabilities for computing gradients in quantum machine learning algorithms.
//!
//! Note: This requires the `scirs2-autograd` feature to be enabled.

use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use scirs2_core::Complex64;
use std::f64::consts::PI;

/// Example quantum neural network layer using parameterized gates
struct QuantumLayer {
    num_qubits: usize,
    params: Vec<f64>,
}

impl QuantumLayer {
    fn new(num_qubits: usize) -> Self {
        // Initialize with random parameters
        let num_params = num_qubits * 3; // RY, RZ per qubit + entangling
        let params = vec![0.1; num_params];
        Self { num_qubits, params }
    }

    /// Apply the quantum layer to a circuit
    fn apply<const N: usize>(&self, circuit: &mut Circuit<N>) -> Result<(), String> {
        if self.num_qubits > N {
            return Err("Circuit has fewer qubits than layer".to_string());
        }

        let mut param_idx = 0;

        // Apply parameterized single-qubit rotations
        for i in 0..self.num_qubits {
            if param_idx < self.params.len() {
                circuit
                    .ry(i, self.params[param_idx])
                    .map_err(|e| e.to_string())?;
                param_idx += 1;
            }
            if param_idx < self.params.len() {
                circuit
                    .rz(i, self.params[param_idx])
                    .map_err(|e| e.to_string())?;
                param_idx += 1;
            }
        }

        // Apply entangling gates
        for i in 0..self.num_qubits - 1 {
            circuit.cnot(i, i + 1).map_err(|e| e.to_string())?;
        }

        // Additional parameterized rotations
        for i in 0..self.num_qubits {
            if param_idx < self.params.len() {
                circuit
                    .ry(i, self.params[param_idx])
                    .map_err(|e| e.to_string())?;
                param_idx += 1;
            }
        }

        Ok(())
    }

    /// Compute gradients using parameter shift rule
    fn compute_gradients(&self, loss_fn: impl Fn(&[f64]) -> f64) -> Vec<f64> {
        let mut gradients = vec![0.0; self.params.len()];
        let shift = PI / 2.0;

        for i in 0..self.params.len() {
            // Positive shift
            let mut params_plus = self.params.clone();
            params_plus[i] += shift;
            let loss_plus = loss_fn(&params_plus);

            // Negative shift
            let mut params_minus = self.params.clone();
            params_minus[i] -= shift;
            let loss_minus = loss_fn(&params_minus);

            // Parameter shift gradient
            gradients[i] = (loss_plus - loss_minus) / 2.0;
        }

        gradients
    }
}

/// Example of using automatic differentiation concepts with quantum circuits
fn quantum_autograd_example() {
    println!("=== Quantum ML with Automatic Differentiation ===\n");

    // Create a quantum neural network with 4 qubits
    let mut qnn = QuantumLayer::new(4);

    // Define a simple loss function (in practice, this would involve circuit simulation)
    let loss_fn = |params: &[f64]| -> f64 {
        // Simulate running the circuit and computing expectation value
        // For demo purposes, we use a simple quadratic function
        params.iter().map(|&p| (p - 0.5).powi(2)).sum::<f64>()
    };

    // Training loop
    let learning_rate = 0.1;
    let num_epochs = 10;

    println!("Initial loss: {:.4}", loss_fn(&qnn.params));

    for epoch in 0..num_epochs {
        // Compute gradients
        let gradients = qnn.compute_gradients(loss_fn);

        // Update parameters (gradient descent)
        for i in 0..qnn.params.len() {
            qnn.params[i] -= learning_rate * gradients[i];
        }

        let loss = loss_fn(&qnn.params);
        println!("Epoch {}: loss = {:.4}", epoch + 1, loss);
    }

    println!("\nFinal parameters: {:?}", qnn.params);
}

/// Demonstrate hybrid quantum-classical optimization
fn hybrid_optimization_demo() {
    println!("\n=== Hybrid Quantum-Classical Optimization ===\n");

    // Define a variational quantum circuit
    const NUM_QUBITS: usize = 3;
    let mut params = vec![0.1; NUM_QUBITS * 2];

    // Cost function that simulates quantum circuit execution
    let cost_fn = |params: &[f64]| -> f64 {
        let mut circuit = Circuit::<NUM_QUBITS>::new();

        // Apply parameterized gates
        for i in 0..NUM_QUBITS {
            let _ = circuit.ry(i, params[i * 2]);
            let _ = circuit.rz(i, params[i * 2 + 1]);
        }

        // In practice, we would simulate the circuit and compute expectation value
        // For demo, use a simple function
        let expectation = params
            .iter()
            .enumerate()
            .map(|(i, &p)| (PI / 4.0).mul_add(-(i as f64 + 1.0), p).powi(2))
            .sum::<f64>();

        expectation
    };

    // Gradient computation using parameter shift rule
    let compute_gradient = |params: &[f64]| -> Vec<f64> {
        let mut grad = vec![0.0; params.len()];
        let epsilon = PI / 2.0;

        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            let mut params_minus = params.to_vec();

            params_plus[i] += epsilon;
            params_minus[i] -= epsilon;

            grad[i] = (cost_fn(&params_plus) - cost_fn(&params_minus)) / (2.0 * epsilon);
        }

        grad
    };

    // Optimization loop
    let learning_rate = 0.1;
    println!("Initial cost: {:.4}", cost_fn(&params));

    for iter in 0..20 {
        let grad = compute_gradient(&params);

        // Update parameters
        for i in 0..params.len() {
            params[i] -= learning_rate * grad[i];
        }

        if iter % 5 == 0 {
            println!("Iteration {}: cost = {:.4}", iter, cost_fn(&params));
        }
    }

    println!("\nOptimized parameters:");
    for (i, &p) in params.iter().enumerate() {
        println!(
            "  θ[{}] = {:.4} (target: {:.4})",
            i,
            p,
            PI / 4.0 * (i as f64 + 1.0)
        );
    }
}

/// Demonstrate gradient computation for quantum kernels
fn quantum_kernel_gradients() {
    println!("\n=== Quantum Kernel Gradient Computation ===\n");

    // Quantum feature map parameters
    let mut feature_params = vec![0.1, 0.2, 0.3];

    // Kernel function (simplified)
    let kernel_fn = |params: &[f64], x1: f64, x2: f64| -> f64 {
        // In practice, this would involve quantum circuit simulation
        let phi1 = params[0].mul_add(x1, params[1] * x1.powi(2));
        let phi2 = params[0].mul_add(x2, params[1] * x2.powi(2));
        let scale = params[2];

        scale * (phi1 - phi2).cos()
    };

    // Sample data points
    let x_train = [0.0, 0.5, 1.0];
    let y_train = [0.0, 1.0, 0.0];

    // Loss function for kernel alignment
    let loss_fn = |params: &[f64]| -> f64 {
        let mut loss = 0.0;

        for i in 0..x_train.len() {
            for j in 0..x_train.len() {
                let k_ij = kernel_fn(params, x_train[i], x_train[j]);
                let target = if y_train[i] == y_train[j] { 1.0 } else { -1.0 };
                loss += (k_ij - target).powi(2);
            }
        }

        loss / (x_train.len() * x_train.len()) as f64
    };

    // Compute gradients numerically
    let epsilon = 0.001;
    let mut gradients = vec![0.0; feature_params.len()];

    for i in 0..feature_params.len() {
        let mut params_plus = feature_params.clone();
        let mut params_minus = feature_params.clone();

        params_plus[i] += epsilon;
        params_minus[i] -= epsilon;

        gradients[i] = (loss_fn(&params_plus) - loss_fn(&params_minus)) / (2.0 * epsilon);
    }

    println!("Kernel parameters: {feature_params:?}");
    println!("Loss: {:.4}", loss_fn(&feature_params));
    println!("Gradients: {gradients:?}");

    // Optimize kernel parameters
    let learning_rate = 0.1;
    for _ in 0..10 {
        for i in 0..feature_params.len() {
            let mut params_plus = feature_params.clone();
            let mut params_minus = feature_params.clone();

            params_plus[i] += epsilon;
            params_minus[i] -= epsilon;

            let grad = (loss_fn(&params_plus) - loss_fn(&params_minus)) / (2.0 * epsilon);
            feature_params[i] -= learning_rate * grad;
        }
    }

    println!("\nOptimized kernel parameters: {feature_params:?}");
    println!("Final loss: {:.4}", loss_fn(&feature_params));
}

fn main() {
    quantum_autograd_example();
    hybrid_optimization_demo();
    quantum_kernel_gradients();

    println!("\n=== Summary ===");
    println!("Demonstrated automatic differentiation concepts for:");
    println!("- Quantum neural network training");
    println!("- Hybrid quantum-classical optimization");
    println!("- Quantum kernel gradient computation");
    println!("\nIn practice, these would be integrated with SciRS2's autograd");
    println!("module for more efficient and flexible gradient computation.");
}
