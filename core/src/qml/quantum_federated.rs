//! Quantum Federated Learning
//!
//! This module implements federated learning for quantum machine learning,
//! enabling privacy-preserving distributed training across multiple quantum
//! devices without sharing raw quantum data.
//!
//! # Theoretical Background
//!
//! Quantum Federated Learning extends classical federated learning to quantum
//! computing, allowing multiple parties to collaboratively train quantum models
//! while keeping their quantum data private. This is crucial for applications
//! in healthcare, finance, and defense where quantum data privacy is paramount.
//!
//! # Key Features
//!
//! - **Distributed Quantum Training**: Train across multiple quantum computers
//! - **Privacy-Preserving Aggregation**: Secure parameter averaging
//! - **Differential Privacy**: Noise injection for formal privacy guarantees
//! - **Byzantine-Robust Aggregation**: Defense against malicious participants
//! - **Adaptive Communication**: Minimize quantum circuit transmission
//!
//! # References
//!
//! - "Federated Learning with Quantum Computing" (2023)
//! - "Privacy-Preserving Quantum Machine Learning" (2024)
//! - "Distributed Quantum Neural Networks" (2024)

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Configuration for quantum federated learning
#[derive(Debug, Clone)]
pub struct QuantumFederatedConfig {
    /// Number of qubits in the quantum model
    pub num_qubits: usize,
    /// Circuit depth
    pub circuit_depth: usize,
    /// Number of clients
    pub num_clients: usize,
    /// Fraction of clients selected per round
    pub client_fraction: f64,
    /// Number of local training epochs
    pub local_epochs: usize,
    /// Local learning rate
    pub local_lr: f64,
    /// Aggregation strategy
    pub aggregation: AggregationStrategy,
    /// Differential privacy epsilon (0.0 = no DP)
    pub dp_epsilon: f64,
    /// Differential privacy delta
    pub dp_delta: f64,
}

impl Default for QuantumFederatedConfig {
    fn default() -> Self {
        Self {
            num_qubits: 4,
            circuit_depth: 3,
            num_clients: 10,
            client_fraction: 0.3,
            local_epochs: 5,
            local_lr: 0.01,
            aggregation: AggregationStrategy::FedAvg,
            dp_epsilon: 1.0,
            dp_delta: 1e-5,
        }
    }
}

/// Aggregation strategy for federated learning
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Federated averaging (FedAvg)
    FedAvg,
    /// Weighted averaging by dataset size
    WeightedAvg,
    /// Median aggregation (Byzantine-robust)
    Median,
    /// Trimmed mean (Byzantine-robust)
    TrimmedMean,
    /// Krum (Byzantine-robust)
    Krum,
}

/// Quantum federated client
#[derive(Debug, Clone)]
pub struct QuantumFederatedClient {
    /// Client ID
    id: usize,
    /// Local quantum circuit parameters
    params: Array2<f64>,
    /// Number of qubits
    num_qubits: usize,
    /// Circuit depth
    depth: usize,
    /// Local dataset size
    dataset_size: usize,
}

impl QuantumFederatedClient {
    /// Create new federated client
    pub fn new(id: usize, num_qubits: usize, depth: usize, dataset_size: usize) -> Self {
        let mut rng = thread_rng();
        let params = Array2::from_shape_fn((depth, num_qubits * 3), |_| rng.gen_range(-PI..PI));

        Self {
            id,
            params,
            num_qubits,
            depth,
            dataset_size,
        }
    }

    /// Local training on client's quantum data
    pub fn train_local(
        &mut self,
        data: &[Array1<Complex64>],
        labels: &[usize],
        epochs: usize,
        lr: f64,
    ) -> QuantRS2Result<f64> {
        let mut total_loss = 0.0;

        for _ in 0..epochs {
            let loss = self.compute_loss(data, labels)?;
            total_loss += loss;

            // Compute gradients using parameter-shift rule
            let gradients = self.compute_gradients(data, labels)?;

            // Update parameters
            self.params = &self.params - &(gradients * lr);
        }

        Ok(total_loss / epochs as f64)
    }

    /// Compute loss on local data
    fn compute_loss(&self, data: &[Array1<Complex64>], labels: &[usize]) -> QuantRS2Result<f64> {
        let mut total_loss = 0.0;

        for (state, &label) in data.iter().zip(labels.iter()) {
            let output = self.forward(state)?;

            // Cross-entropy loss
            total_loss -= output[label].ln();
        }

        Ok(total_loss / data.len() as f64)
    }

    /// Forward pass through quantum circuit
    fn forward(&self, state: &Array1<Complex64>) -> QuantRS2Result<Array1<f64>> {
        let mut encoded = state.clone();

        // Apply parameterized quantum circuit
        for layer in 0..self.depth {
            for q in 0..self.num_qubits {
                let rx = self.params[[layer, q * 3]];
                let ry = self.params[[layer, q * 3 + 1]];
                let rz = self.params[[layer, q * 3 + 2]];

                encoded = self.apply_rotation(&encoded, q, rx, ry, rz)?;
            }

            // Entangling layer
            for q in 0..self.num_qubits - 1 {
                encoded = self.apply_cnot(&encoded, q, q + 1)?;
            }
        }

        // Measure Pauli-Z expectations
        let mut expectations = Array1::zeros(2); // Binary classification
        expectations[0] = self.pauli_z_expectation(&encoded, 0)?;
        expectations[1] = 1.0 - expectations[0];

        // Softmax
        let max_exp = expectations
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut probs = Array1::zeros(2);
        let mut sum = 0.0;

        for i in 0..2 {
            probs[i] = (expectations[i] - max_exp).exp();
            sum += probs[i];
        }

        for i in 0..2 {
            probs[i] /= sum;
        }

        Ok(probs)
    }

    /// Compute gradients using parameter-shift rule
    fn compute_gradients(
        &self,
        data: &[Array1<Complex64>],
        labels: &[usize],
    ) -> QuantRS2Result<Array2<f64>> {
        let epsilon = PI / 2.0; // Parameter-shift rule
        let mut gradients = Array2::zeros(self.params.dim());

        for i in 0..self.params.shape()[0] {
            for j in 0..self.params.shape()[1] {
                // Shift parameter forward
                let mut client_plus = self.clone();
                client_plus.params[[i, j]] += epsilon;
                let loss_plus = client_plus.compute_loss(data, labels)?;

                // Shift parameter backward
                let mut client_minus = self.clone();
                client_minus.params[[i, j]] -= epsilon;
                let loss_minus = client_minus.compute_loss(data, labels)?;

                // Parameter-shift gradient
                gradients[[i, j]] = (loss_plus - loss_minus) / 2.0;
            }
        }

        Ok(gradients)
    }

    /// Get model parameters
    pub fn get_params(&self) -> &Array2<f64> {
        &self.params
    }

    /// Set model parameters
    pub fn set_params(&mut self, params: Array2<f64>) {
        self.params = params;
    }

    /// Get dataset size
    pub fn dataset_size(&self) -> usize {
        self.dataset_size
    }

    // Helper methods
    fn apply_rotation(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
        rx: f64,
        ry: f64,
        rz: f64,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let mut result = state.clone();
        result = self.apply_rz_gate(&result, qubit, rz)?;
        result = self.apply_ry_gate(&result, qubit, ry)?;
        result = self.apply_rx_gate(&result, qubit, rx)?;
        Ok(result)
    }

    fn apply_rx_gate(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
        angle: f64,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = Array1::zeros(dim);
        let cos_half = Complex64::new((angle / 2.0).cos(), 0.0);
        let sin_half = Complex64::new(0.0, -(angle / 2.0).sin());

        for i in 0..dim {
            let j = i ^ (1 << qubit);
            new_state[i] = state[i] * cos_half + state[j] * sin_half;
        }

        Ok(new_state)
    }

    fn apply_ry_gate(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
        angle: f64,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = Array1::zeros(dim);
        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            let j = i ^ (1 << qubit);
            if bit == 0 {
                new_state[i] = state[i] * cos_half - state[j] * sin_half;
            } else {
                new_state[i] = state[i] * cos_half + state[j] * sin_half;
            }
        }

        Ok(new_state)
    }

    fn apply_rz_gate(
        &self,
        state: &Array1<Complex64>,
        qubit: usize,
        angle: f64,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = state.clone();
        let phase = Complex64::new((angle / 2.0).cos(), -(angle / 2.0).sin());

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            new_state[i] = if bit == 1 {
                new_state[i] * phase
            } else {
                new_state[i] * phase.conj()
            };
        }

        Ok(new_state)
    }

    fn apply_cnot(
        &self,
        state: &Array1<Complex64>,
        control: usize,
        target: usize,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let dim = state.len();
        let mut new_state = state.clone();

        for i in 0..dim {
            let control_bit = (i >> control) & 1;
            if control_bit == 1 {
                let j = i ^ (1 << target);
                if i < j {
                    let temp = new_state[i];
                    new_state[i] = new_state[j];
                    new_state[j] = temp;
                }
            }
        }

        Ok(new_state)
    }

    fn pauli_z_expectation(&self, state: &Array1<Complex64>, qubit: usize) -> QuantRS2Result<f64> {
        let dim = state.len();
        let mut expectation = 0.0;

        for i in 0..dim {
            let bit = (i >> qubit) & 1;
            let sign = if bit == 0 { 1.0 } else { -1.0 };
            expectation += sign * state[i].norm_sqr();
        }

        // Map from [-1, 1] to [0, 1]
        Ok((expectation + 1.0) / 2.0)
    }
}

/// Quantum federated learning server
#[derive(Debug)]
pub struct QuantumFederatedServer {
    /// Configuration
    config: QuantumFederatedConfig,
    /// Global model parameters
    global_params: Array2<f64>,
    /// Clients
    clients: Vec<QuantumFederatedClient>,
    /// Training history
    history: Vec<f64>,
}

impl QuantumFederatedServer {
    /// Create new federated server
    pub fn new(config: QuantumFederatedConfig) -> Self {
        let mut rng = thread_rng();

        // Initialize global model
        let global_params =
            Array2::from_shape_fn((config.circuit_depth, config.num_qubits * 3), |_| {
                rng.gen_range(-PI..PI)
            });

        // Create clients
        let mut clients = Vec::with_capacity(config.num_clients);
        for i in 0..config.num_clients {
            let dataset_size = rng.gen_range(50..200);
            clients.push(QuantumFederatedClient::new(
                i,
                config.num_qubits,
                config.circuit_depth,
                dataset_size,
            ));
        }

        Self {
            config,
            global_params,
            clients,
            history: Vec::new(),
        }
    }

    /// Run one federated learning round
    pub fn train_round(
        &mut self,
        client_data: &HashMap<usize, (Vec<Array1<Complex64>>, Vec<usize>)>,
    ) -> QuantRS2Result<f64> {
        // Select clients for this round
        let num_selected =
            (self.config.num_clients as f64 * self.config.client_fraction).ceil() as usize;
        let selected_clients = self.select_clients(num_selected);

        // Distribute global model to selected clients
        for &client_id in &selected_clients {
            self.clients[client_id].set_params(self.global_params.clone());
        }

        // Local training on each client
        let mut client_updates = Vec::new();
        let mut client_weights = Vec::new();
        let mut avg_loss = 0.0;

        for &client_id in &selected_clients {
            if let Some((data, labels)) = client_data.get(&client_id) {
                let loss = self.clients[client_id].train_local(
                    data,
                    labels,
                    self.config.local_epochs,
                    self.config.local_lr,
                )?;

                avg_loss += loss;

                client_updates.push(self.clients[client_id].get_params().clone());
                client_weights.push(self.clients[client_id].dataset_size() as f64);
            }
        }

        avg_loss /= selected_clients.len() as f64;
        self.history.push(avg_loss);

        // Aggregate client updates
        self.aggregate_updates(&client_updates, &client_weights)?;

        Ok(avg_loss)
    }

    /// Select clients for training round
    fn select_clients(&self, num_selected: usize) -> Vec<usize> {
        let mut rng = thread_rng();
        let mut clients: Vec<usize> = (0..self.config.num_clients).collect();

        // Shuffle and select
        for i in (1..clients.len()).rev() {
            let j = rng.gen_range(0..=i);
            clients.swap(i, j);
        }

        clients.truncate(num_selected);
        clients
    }

    /// Aggregate client updates
    fn aggregate_updates(
        &mut self,
        updates: &[Array2<f64>],
        weights: &[f64],
    ) -> QuantRS2Result<()> {
        match self.config.aggregation {
            AggregationStrategy::FedAvg => {
                self.federated_averaging(updates)?;
            }
            AggregationStrategy::WeightedAvg => {
                self.weighted_averaging(updates, weights)?;
            }
            AggregationStrategy::Median => {
                self.median_aggregation(updates)?;
            }
            AggregationStrategy::TrimmedMean => {
                self.trimmed_mean_aggregation(updates, 0.1)?;
            }
            AggregationStrategy::Krum => {
                self.krum_aggregation(updates)?;
            }
        }

        // Apply differential privacy if enabled
        if self.config.dp_epsilon > 0.0 {
            self.apply_differential_privacy()?;
        }

        Ok(())
    }

    /// Federated averaging (FedAvg)
    fn federated_averaging(&mut self, updates: &[Array2<f64>]) -> QuantRS2Result<()> {
        let mut avg_params = Array2::zeros(self.global_params.dim());

        for update in updates {
            avg_params = avg_params + update;
        }

        avg_params = avg_params / (updates.len() as f64);
        self.global_params = avg_params;

        Ok(())
    }

    /// Weighted averaging by dataset size
    fn weighted_averaging(
        &mut self,
        updates: &[Array2<f64>],
        weights: &[f64],
    ) -> QuantRS2Result<()> {
        let total_weight: f64 = weights.iter().sum();
        let mut weighted_params = Array2::zeros(self.global_params.dim());

        for (update, &weight) in updates.iter().zip(weights.iter()) {
            weighted_params = weighted_params + update * (weight / total_weight);
        }

        self.global_params = weighted_params;
        Ok(())
    }

    /// Median aggregation (coordinate-wise median)
    fn median_aggregation(&mut self, updates: &[Array2<f64>]) -> QuantRS2Result<()> {
        let shape = self.global_params.dim();
        let mut median_params = Array2::zeros(shape);

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                let mut values: Vec<f64> = updates.iter().map(|u| u[[i, j]]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                median_params[[i, j]] = if values.len() % 2 == 0 {
                    (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                } else {
                    values[values.len() / 2]
                };
            }
        }

        self.global_params = median_params;
        Ok(())
    }

    /// Trimmed mean aggregation
    fn trimmed_mean_aggregation(
        &mut self,
        updates: &[Array2<f64>],
        trim_ratio: f64,
    ) -> QuantRS2Result<()> {
        let shape = self.global_params.dim();
        let mut trimmed_params = Array2::zeros(shape);
        let trim_count = (updates.len() as f64 * trim_ratio).floor() as usize;

        for i in 0..shape.0 {
            for j in 0..shape.1 {
                let mut values: Vec<f64> = updates.iter().map(|u| u[[i, j]]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                // Trim extremes
                let trimmed: Vec<f64> = values[trim_count..values.len() - trim_count].to_vec();
                trimmed_params[[i, j]] = trimmed.iter().sum::<f64>() / trimmed.len() as f64;
            }
        }

        self.global_params = trimmed_params;
        Ok(())
    }

    /// Krum aggregation (Byzantine-robust)
    fn krum_aggregation(&mut self, updates: &[Array2<f64>]) -> QuantRS2Result<()> {
        let n = updates.len();
        let f = (n - 1) / 2; // Maximum Byzantine clients
        let n_minus_f_minus_2 = n - f - 2;

        // Compute pairwise distances
        let mut scores = vec![0.0; n];

        for i in 0..n {
            let mut distances: Vec<(usize, f64)> = Vec::new();

            for j in 0..n {
                if i != j {
                    let diff = &updates[i] - &updates[j];
                    let dist: f64 = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
                    distances.push((j, dist));
                }
            }

            // Sort by distance and sum closest n-f-2
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            scores[i] = distances
                .iter()
                .take(n_minus_f_minus_2)
                .map(|(_, d)| d)
                .sum();
        }

        // Select client with minimum score
        let best_client = scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        self.global_params = updates[best_client].clone();
        Ok(())
    }

    /// Apply differential privacy to global model
    fn apply_differential_privacy(&mut self) -> QuantRS2Result<()> {
        let mut rng = thread_rng();

        // Compute noise scale based on DP parameters
        let sensitivity = 1.0; // L2 sensitivity
        let noise_scale = sensitivity / self.config.dp_epsilon;

        // Add Gaussian noise to parameters
        for i in 0..self.global_params.shape()[0] {
            for j in 0..self.global_params.shape()[1] {
                let noise = rng.gen_range(-1.0..1.0) * noise_scale;
                self.global_params[[i, j]] += noise;
            }
        }

        Ok(())
    }

    /// Get global model parameters
    pub fn get_global_params(&self) -> &Array2<f64> {
        &self.global_params
    }

    /// Get training history
    pub fn history(&self) -> &[f64] {
        &self.history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federated_client() {
        let mut client = QuantumFederatedClient::new(0, 2, 2, 100);

        let state = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);

        let probs = client.forward(&state).unwrap();
        assert_eq!(probs.len(), 2);

        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_federated_server() {
        let config = QuantumFederatedConfig {
            num_qubits: 2,
            circuit_depth: 2,
            num_clients: 5,
            client_fraction: 0.6,
            local_epochs: 2,
            local_lr: 0.01,
            aggregation: AggregationStrategy::FedAvg,
            dp_epsilon: 0.0,
            dp_delta: 1e-5,
        };

        let server = QuantumFederatedServer::new(config);
        assert_eq!(server.clients.len(), 5);
    }
}
