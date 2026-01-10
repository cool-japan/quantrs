//! QML Utilities and Benchmarks
//!
//! This module provides utility functions and benchmarking for quantum machine learning.

use super::config::{QMLArchitectureType, QMLConfig};
use super::framework::QuantumMLFramework;
use super::types::{QMLBenchmarkResults, QuantumAdvantageMetrics};
use crate::error::Result;
use scirs2_core::ndarray::Array1;
use scirs2_core::random::prelude::*;
use std::collections::HashMap;

/// Utility functions for QML
pub struct QMLUtils;

impl QMLUtils {
    /// Generate synthetic training data for testing
    #[must_use]
    pub fn generate_synthetic_data(
        num_samples: usize,
        input_dim: usize,
        output_dim: usize,
    ) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
        let mut rng = thread_rng();
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        for _ in 0..num_samples {
            let input: Array1<f64> = Array1::from_vec(
                (0..input_dim)
                    .map(|_| rng.random_range(-1.0_f64..1.0_f64))
                    .collect(),
            );

            // Generate output based on some function of input
            let output = Array1::from_vec(
                (0..output_dim)
                    .map(|i| {
                        if i < input_dim {
                            input[i].sin() // Simple nonlinear transformation
                        } else {
                            rng.random_range(-1.0_f64..1.0_f64)
                        }
                    })
                    .collect(),
            );

            inputs.push(input);
            outputs.push(output);
        }

        (inputs, outputs)
    }

    /// Split data into training and validation sets
    #[must_use]
    pub fn train_test_split(
        inputs: Vec<Array1<f64>>,
        outputs: Vec<Array1<f64>>,
        test_ratio: f64,
    ) -> (
        Vec<(Array1<f64>, Array1<f64>)>,
        Vec<(Array1<f64>, Array1<f64>)>,
    ) {
        let total_samples = inputs.len();
        let test_samples = ((total_samples as f64) * test_ratio) as usize;
        let train_samples = total_samples - test_samples;

        let mut combined: Vec<(Array1<f64>, Array1<f64>)> =
            inputs.into_iter().zip(outputs).collect();

        // Shuffle data
        let mut rng = thread_rng();
        for i in (1..combined.len()).rev() {
            let j = rng.random_range(0..=i);
            combined.swap(i, j);
        }

        let (train_data, test_data) = combined.split_at(train_samples);
        (train_data.to_vec(), test_data.to_vec())
    }

    /// Evaluate model accuracy
    #[must_use]
    pub fn evaluate_accuracy(
        predictions: &[Array1<f64>],
        targets: &[Array1<f64>],
        threshold: f64,
    ) -> f64 {
        let mut correct = 0;
        let total = predictions.len();

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let diff = pred - target;
            let mse = diff.iter().map(|x| x * x).sum::<f64>() / diff.len() as f64;
            if mse < threshold {
                correct += 1;
            }
        }

        f64::from(correct) / total as f64
    }

    /// Compute quantum circuit complexity metrics
    #[must_use]
    pub fn compute_circuit_complexity(
        num_qubits: usize,
        depth: usize,
        gate_count: usize,
    ) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        // State space size
        let state_space_size = 2.0_f64.powi(num_qubits as i32);
        metrics.insert("state_space_size".to_string(), state_space_size);

        // Circuit complexity (depth * gates)
        let circuit_complexity = (depth * gate_count) as f64;
        metrics.insert("circuit_complexity".to_string(), circuit_complexity);

        // Classical simulation cost estimate
        let classical_cost = state_space_size * gate_count as f64;
        metrics.insert("classical_simulation_cost".to_string(), classical_cost);

        // Quantum advantage estimate (log scale)
        let quantum_advantage = classical_cost.log(circuit_complexity);
        metrics.insert("quantum_advantage_estimate".to_string(), quantum_advantage);

        metrics
    }
}

/// Benchmark quantum machine learning implementations
pub fn benchmark_quantum_ml_layers(config: &QMLConfig) -> Result<QMLBenchmarkResults> {
    let mut results = QMLBenchmarkResults {
        training_times: HashMap::new(),
        final_accuracies: HashMap::new(),
        convergence_rates: HashMap::new(),
        memory_usage: HashMap::new(),
        quantum_advantage: HashMap::new(),
        parameter_counts: HashMap::new(),
        circuit_depths: HashMap::new(),
        gate_counts: HashMap::new(),
    };

    // Generate test data
    let (inputs, outputs) =
        QMLUtils::generate_synthetic_data(100, config.num_qubits, config.num_qubits);
    let (train_data, val_data) = QMLUtils::train_test_split(inputs, outputs, 0.2);

    // Benchmark different QML architectures
    let architectures = vec![
        QMLArchitectureType::VariationalQuantumCircuit,
        QMLArchitectureType::QuantumConvolutionalNN,
        // Add more architectures as needed
    ];

    for architecture in architectures {
        let arch_name = format!("{architecture:?}");

        // Create configuration for this architecture
        let mut arch_config = config.clone();
        arch_config.architecture_type = architecture;

        // Create and train model
        let start_time = std::time::Instant::now();
        let mut framework = QuantumMLFramework::new(arch_config)?;

        let training_result = framework.train(&train_data, Some(&val_data))?;
        let training_time = start_time.elapsed();

        // Evaluate final accuracy
        let final_accuracy = framework.evaluate(&val_data)?;

        // Store results
        results
            .training_times
            .insert(arch_name.clone(), training_time);
        results
            .final_accuracies
            .insert(arch_name.clone(), 1.0 / (1.0 + final_accuracy)); // Convert loss to accuracy
        results.convergence_rates.insert(
            arch_name.clone(),
            training_result.epochs_trained as f64 / config.training_config.epochs as f64,
        );
        results
            .memory_usage
            .insert(arch_name.clone(), framework.get_stats().peak_memory_usage);
        results
            .quantum_advantage
            .insert(arch_name.clone(), training_result.quantum_advantage_metrics);
        results.parameter_counts.insert(
            arch_name.clone(),
            framework
                .layers
                .iter()
                .map(|l| l.get_num_parameters())
                .sum(),
        );
        results.circuit_depths.insert(
            arch_name.clone(),
            framework.layers.iter().map(|l| l.get_depth()).sum(),
        );
        results.gate_counts.insert(
            arch_name.clone(),
            framework.layers.iter().map(|l| l.get_gate_count()).sum(),
        );
    }

    Ok(results)
}
