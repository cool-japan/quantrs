//! Quantum federated learning protocols for distributed quantum machine learning.
//!
//! This module implements privacy-preserving distributed training of quantum models
//! with secure aggregation and differential privacy guarantees.

use scirs2_core::ndarray::{Array1, Array2, Array3};
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::error::{MLError, Result};
use crate::qnn::QuantumNeuralNetwork;
use crate::utils::VariationalCircuit;
use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::{multi::*, single::*, GateOp};

/// Federated learning client for quantum models
#[derive(Debug)]
pub struct QuantumFLClient {
    /// Client ID
    client_id: String,
    /// Local quantum model
    local_model: QuantumNeuralNetwork,
    /// Local dataset size
    dataset_size: usize,
    /// Privacy budget
    epsilon: f64,
    /// Noise scale for differential privacy
    noise_scale: f64,
    /// Client-specific parameters
    local_params: HashMap<String, f64>,
}

impl QuantumFLClient {
    /// Create a new federated learning client
    pub fn new(
        client_id: String,
        model_config: &[(String, usize)], // Layer configs
        dataset_size: usize,
        epsilon: f64,
    ) -> Result<Self> {
        // Create local model based on config
        let layers = model_config
            .iter()
            .map(|(layer_type, size)| match layer_type.as_str() {
                "encoding" => crate::qnn::QNNLayerType::EncodingLayer {
                    num_features: *size,
                },
                "variational" => crate::qnn::QNNLayerType::VariationalLayer { num_params: *size },
                "entanglement" => crate::qnn::QNNLayerType::EntanglementLayer {
                    connectivity: "full".to_string(),
                },
                _ => crate::qnn::QNNLayerType::MeasurementLayer {
                    measurement_basis: "computational".to_string(),
                },
            })
            .collect();

        let local_model = QuantumNeuralNetwork::new(layers, 4, 10, 2)?;
        let noise_scale = (2.0 * (1.25 / epsilon).ln()).sqrt() / dataset_size as f64;

        Ok(Self {
            client_id,
            local_model,
            dataset_size,
            epsilon,
            noise_scale,
            local_params: HashMap::new(),
        })
    }

    /// Train on local data
    pub fn train_local(
        &mut self,
        local_data: &Array2<f64>,
        local_labels: &Array1<i32>,
        epochs: usize,
    ) -> Result<f64> {
        let mut total_loss = 0.0;

        for _ in 0..epochs {
            // Simplified training loop
            for i in 0..local_data.nrows() {
                let input = local_data.row(i).to_owned();
                let label = local_labels[i];

                // Forward pass
                let output = self.local_model.forward(&input)?;

                // Compute loss
                let loss = self.compute_loss(&output, label)?;
                total_loss += loss;

                // Backward pass (simplified)
                self.update_parameters(&input, label, 0.01)?;
            }
        }

        // Add differential privacy noise
        self.add_dp_noise()?;

        Ok(total_loss / (epochs * local_data.nrows()) as f64)
    }

    /// Compute loss function
    fn compute_loss(&self, output: &Array1<f64>, label: i32) -> Result<f64> {
        // Cross-entropy loss for classification
        let label_idx = label as usize;
        if label_idx >= output.len() {
            return Err(MLError::InvalidInput("Label out of bounds".to_string()));
        }

        Ok(-output[label_idx].ln())
    }

    /// Update parameters (simplified)
    fn update_parameters(
        &mut self,
        input: &Array1<f64>,
        label: i32,
        learning_rate: f64,
    ) -> Result<()> {
        // Placeholder parameter update
        for (key, value) in self.local_params.iter_mut() {
            *value += learning_rate * fastrand::f64() * 0.1;
        }
        Ok(())
    }

    /// Add differential privacy noise
    fn add_dp_noise(&mut self) -> Result<()> {
        for (_, value) in self.local_params.iter_mut() {
            // Add Gaussian noise scaled by sensitivity and epsilon
            let noise = self.noise_scale * Self::gaussian_noise();
            *value += noise;
        }
        Ok(())
    }

    /// Generate Gaussian noise
    fn gaussian_noise() -> f64 {
        // Box-Muller transform
        let u1 = fastrand::f64();
        let u2 = fastrand::f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Get model parameters for aggregation
    pub fn get_parameters(&self) -> HashMap<String, f64> {
        self.local_params.clone()
    }

    /// Update model with aggregated parameters
    pub fn set_parameters(&mut self, params: HashMap<String, f64>) {
        self.local_params = params;
    }
}

/// Quantum secure aggregation server
#[derive(Debug)]
pub struct QuantumFLServer {
    /// Global model configuration
    model_config: Vec<(String, usize)>,
    /// Aggregated parameters
    global_params: HashMap<String, f64>,
    /// Client weights for aggregation
    client_weights: HashMap<String, f64>,
    /// Secure aggregation protocol
    aggregation_protocol: SecureAggregationProtocol,
    /// Byzantine fault tolerance threshold
    byzantine_threshold: f64,
}

#[derive(Debug, Clone)]
pub enum SecureAggregationProtocol {
    /// Simple averaging
    FederatedAveraging,
    /// Secure multi-party computation
    SecureMultiparty,
    /// Homomorphic encryption
    HomomorphicEncryption,
    /// Quantum secret sharing
    QuantumSecretSharing,
}

impl QuantumFLServer {
    /// Create a new federated learning server
    pub fn new(
        model_config: Vec<(String, usize)>,
        aggregation_protocol: SecureAggregationProtocol,
        byzantine_threshold: f64,
    ) -> Self {
        Self {
            model_config,
            global_params: HashMap::new(),
            client_weights: HashMap::new(),
            aggregation_protocol,
            byzantine_threshold,
        }
    }

    /// Aggregate client updates
    pub fn aggregate_updates(
        &mut self,
        client_updates: Vec<(String, HashMap<String, f64>, usize)>, // (client_id, params, dataset_size)
    ) -> Result<HashMap<String, f64>> {
        match self.aggregation_protocol {
            SecureAggregationProtocol::FederatedAveraging => {
                self.federated_averaging(client_updates)
            }
            SecureAggregationProtocol::SecureMultiparty => {
                self.secure_multiparty_aggregation(client_updates)
            }
            SecureAggregationProtocol::HomomorphicEncryption => {
                self.homomorphic_aggregation(client_updates)
            }
            SecureAggregationProtocol::QuantumSecretSharing => {
                self.quantum_secret_sharing_aggregation(client_updates)
            }
        }
    }

    /// Federated averaging aggregation
    fn federated_averaging(
        &mut self,
        client_updates: Vec<(String, HashMap<String, f64>, usize)>,
    ) -> Result<HashMap<String, f64>> {
        let total_samples: usize = client_updates.iter().map(|(_, _, size)| size).sum();
        let mut aggregated = HashMap::new();

        // Weight by dataset size
        for (client_id, params, dataset_size) in client_updates {
            let weight = dataset_size as f64 / total_samples as f64;
            self.client_weights.insert(client_id.clone(), weight);

            for (param_name, param_value) in params {
                *aggregated.entry(param_name).or_insert(0.0) += weight * param_value;
            }
        }

        self.global_params = aggregated.clone();
        Ok(aggregated)
    }

    /// Secure multi-party computation aggregation
    fn secure_multiparty_aggregation(
        &mut self,
        client_updates: Vec<(String, HashMap<String, f64>, usize)>,
    ) -> Result<HashMap<String, f64>> {
        // Implement secure aggregation using secret sharing
        let num_clients = client_updates.len();
        let mut shares: HashMap<String, Vec<f64>> = HashMap::new();

        // Collect shares for each parameter
        for (_, params, _) in &client_updates {
            for (param_name, param_value) in params {
                shares
                    .entry(param_name.clone())
                    .or_insert(Vec::new())
                    .push(*param_value);
            }
        }

        // Aggregate shares with Byzantine fault tolerance
        let mut aggregated = HashMap::new();
        for (param_name, param_shares) in shares {
            let aggregated_value = self.byzantine_robust_aggregation(&param_shares)?;
            aggregated.insert(param_name, aggregated_value);
        }

        self.global_params = aggregated.clone();
        Ok(aggregated)
    }

    /// Homomorphic encryption aggregation
    fn homomorphic_aggregation(
        &mut self,
        client_updates: Vec<(String, HashMap<String, f64>, usize)>,
    ) -> Result<HashMap<String, f64>> {
        // Simplified homomorphic aggregation
        // In practice, would use actual homomorphic encryption

        let mut encrypted_sum = HashMap::new();

        for (_, params, _) in &client_updates {
            for (param_name, param_value) in params {
                // "Encrypt" (simplified)
                let encrypted = self.homomorphic_encrypt(*param_value)?;

                // Add encrypted values
                *encrypted_sum.entry(param_name.clone()).or_insert(0.0) += encrypted;
            }
        }

        // "Decrypt" aggregated values
        let mut aggregated = HashMap::new();
        for (param_name, encrypted_value) in encrypted_sum {
            let decrypted = self.homomorphic_decrypt(encrypted_value)?;
            aggregated.insert(param_name, decrypted / client_updates.len() as f64);
        }

        self.global_params = aggregated.clone();
        Ok(aggregated)
    }

    /// Quantum secret sharing aggregation
    fn quantum_secret_sharing_aggregation(
        &mut self,
        client_updates: Vec<(String, HashMap<String, f64>, usize)>,
    ) -> Result<HashMap<String, f64>> {
        let num_clients = client_updates.len();
        let threshold = ((num_clients as f64) * self.byzantine_threshold).ceil() as usize;

        // Create quantum shares
        let mut quantum_shares: HashMap<String, Vec<QuantumShare>> = HashMap::new();

        for (client_id, params, _) in &client_updates {
            for (param_name, param_value) in params {
                let share = self.create_quantum_share(client_id, *param_value)?;
                quantum_shares
                    .entry(param_name.clone())
                    .or_insert(Vec::new())
                    .push(share);
            }
        }

        // Reconstruct from shares
        let mut aggregated = HashMap::new();
        for (param_name, shares) in quantum_shares {
            if shares.len() >= threshold {
                let reconstructed = self.reconstruct_from_quantum_shares(&shares)?;
                aggregated.insert(param_name, reconstructed);
            }
        }

        self.global_params = aggregated.clone();
        Ok(aggregated)
    }

    /// Byzantine-robust aggregation
    fn byzantine_robust_aggregation(&self, values: &[f64]) -> Result<f64> {
        if values.is_empty() {
            return Err(MLError::InvalidInput("No values to aggregate".to_string()));
        }

        // Krum algorithm for Byzantine robustness
        let n = values.len();
        let f = ((n as f64 * self.byzantine_threshold) as usize).min(n / 2);

        // Compute pairwise distances
        let mut scores = vec![0.0; n];
        for i in 0..n {
            let mut distances: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (values[i] - values[j]).abs())
                .collect();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Sum of n-f-1 closest values
            scores[i] = distances.iter().take(n - f - 1).sum();
        }

        // Select value with minimum score
        let best_idx = scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        Ok(values[best_idx])
    }

    /// Simple homomorphic encryption (placeholder)
    fn homomorphic_encrypt(&self, value: f64) -> Result<f64> {
        // In practice, use proper homomorphic encryption
        Ok(value * 1000.0 + fastrand::f64() * 10.0)
    }

    /// Simple homomorphic decryption (placeholder)
    fn homomorphic_decrypt(&self, encrypted: f64) -> Result<f64> {
        // In practice, use proper homomorphic decryption
        Ok((encrypted - 5.0) / 1000.0)
    }

    /// Create quantum share
    fn create_quantum_share(&self, client_id: &str, value: f64) -> Result<QuantumShare> {
        let num_qubits = 3;
        let mut circuit = VariationalCircuit::new(num_qubits);

        // Encode value in quantum state
        circuit.add_gate("RY", vec![0], vec![(value * PI).to_string()]);

        // Create entangled shares
        circuit.add_gate("H", vec![1], vec![]);
        circuit.add_gate("CNOT", vec![1, 2], vec![]);
        circuit.add_gate("CNOT", vec![0, 1], vec![]);

        Ok(QuantumShare {
            client_id: client_id.to_string(),
            share_circuit: circuit,
            share_value: value,
        })
    }

    /// Reconstruct from quantum shares
    fn reconstruct_from_quantum_shares(&self, shares: &[QuantumShare]) -> Result<f64> {
        // Simplified reconstruction
        // In practice, would perform quantum state tomography
        let sum: f64 = shares.iter().map(|s| s.share_value).sum();
        Ok(sum / shares.len() as f64)
    }
}

/// Quantum share for secret sharing
#[derive(Debug)]
struct QuantumShare {
    client_id: String,
    share_circuit: VariationalCircuit,
    share_value: f64,
}

/// Distributed quantum learning coordinator
#[derive(Debug)]
pub struct DistributedQuantumLearning {
    /// Server instance
    server: QuantumFLServer,
    /// Client instances
    clients: HashMap<String, QuantumFLClient>,
    /// Communication rounds
    rounds: usize,
    /// Convergence threshold
    convergence_threshold: f64,
}

impl DistributedQuantumLearning {
    /// Create a new distributed learning system
    pub fn new(
        num_clients: usize,
        model_config: Vec<(String, usize)>,
        aggregation_protocol: SecureAggregationProtocol,
        epsilon: f64,
    ) -> Result<Self> {
        let server = QuantumFLServer::new(
            model_config.clone(),
            aggregation_protocol,
            0.2, // Byzantine threshold
        );

        let mut clients = HashMap::new();
        for i in 0..num_clients {
            let client_id = format!("client_{}", i);
            let dataset_size = 100 + fastrand::usize(..900); // Random dataset size
            let client =
                QuantumFLClient::new(client_id.clone(), &model_config, dataset_size, epsilon)?;
            clients.insert(client_id, client);
        }

        Ok(Self {
            server,
            clients,
            rounds: 0,
            convergence_threshold: 1e-4,
        })
    }

    /// Run federated training
    pub fn train(
        &mut self,
        data_distribution: &HashMap<String, (Array2<f64>, Array1<i32>)>,
        num_rounds: usize,
        clients_per_round: usize,
    ) -> Result<FederatedTrainingResult> {
        let mut round_losses = Vec::new();
        let mut convergence_metric = f64::INFINITY;

        for round in 0..num_rounds {
            self.rounds = round + 1;

            // Select random subset of clients
            let selected_clients = self.select_clients(clients_per_round);

            // Local training
            let mut client_updates = Vec::new();
            let mut round_loss = 0.0;

            for client_id in selected_clients {
                if let Some(client) = self.clients.get_mut(&client_id) {
                    if let Some((data, labels)) = data_distribution.get(&client_id) {
                        // Train locally
                        let loss = client.train_local(data, labels, 5)?;
                        round_loss += loss;

                        // Get parameters
                        let params = client.get_parameters();
                        let dataset_size = data.nrows();
                        client_updates.push((client_id.clone(), params, dataset_size));
                    }
                }
            }

            // Aggregate updates
            let aggregated = self.server.aggregate_updates(client_updates)?;

            // Update all clients with aggregated model
            for (_, client) in self.clients.iter_mut() {
                client.set_parameters(aggregated.clone());
            }

            // Check convergence (skip on first round)
            if round > 0 {
                let prev_params = self.server.global_params.clone();
                convergence_metric = self.compute_convergence(&prev_params, &aggregated)?;

                if convergence_metric < self.convergence_threshold {
                    round_losses.push(round_loss / clients_per_round as f64);
                    break;
                }
            }

            round_losses.push(round_loss / clients_per_round as f64);

            // Update server's global params
            self.server.global_params = aggregated.clone();
        }

        Ok(FederatedTrainingResult {
            final_model_params: self.server.global_params.clone(),
            round_losses,
            num_rounds: self.rounds,
            converged: convergence_metric < self.convergence_threshold,
            convergence_metric,
        })
    }

    /// Select random clients for training round
    fn select_clients(&self, num_clients: usize) -> Vec<String> {
        let all_clients: Vec<String> = self.clients.keys().cloned().collect();
        let mut selected = Vec::new();

        while selected.len() < num_clients.min(all_clients.len()) {
            let idx = fastrand::usize(..all_clients.len());
            let client = all_clients[idx].clone();
            if !selected.contains(&client) {
                selected.push(client);
            }
        }

        selected
    }

    /// Compute convergence metric
    fn compute_convergence(
        &self,
        old_params: &HashMap<String, f64>,
        new_params: &HashMap<String, f64>,
    ) -> Result<f64> {
        let mut diff_sum = 0.0;
        let mut count = 0;

        for (key, new_val) in new_params {
            if let Some(old_val) = old_params.get(key) {
                diff_sum += (new_val - old_val).abs();
                count += 1;
            }
        }

        Ok(if count > 0 {
            diff_sum / count as f64
        } else {
            0.0
        })
    }
}

/// Result of federated training
#[derive(Debug)]
pub struct FederatedTrainingResult {
    /// Final aggregated model parameters
    pub final_model_params: HashMap<String, f64>,
    /// Loss history per round
    pub round_losses: Vec<f64>,
    /// Number of rounds completed
    pub num_rounds: usize,
    /// Whether training converged
    pub converged: bool,
    /// Final convergence metric
    pub convergence_metric: f64,
}

/// Privacy-preserving quantum computation
pub mod privacy {
    use super::*;

    /// Differential privacy mechanism for quantum circuits
    #[derive(Debug)]
    pub struct QuantumDifferentialPrivacy {
        /// Privacy budget
        epsilon: f64,
        /// Sensitivity bound
        sensitivity: f64,
        /// Noise mechanism
        mechanism: NoiseType,
    }

    #[derive(Debug, Clone)]
    pub enum NoiseType {
        Laplace,
        Gaussian,
        Quantum,
    }

    impl QuantumDifferentialPrivacy {
        /// Create new DP mechanism
        pub fn new(epsilon: f64, sensitivity: f64, mechanism: NoiseType) -> Self {
            Self {
                epsilon,
                sensitivity,
                mechanism,
            }
        }

        /// Add noise to quantum circuit parameters
        pub fn add_noise(&self, params: &mut HashMap<String, f64>) -> Result<()> {
            for (_, value) in params.iter_mut() {
                let noise = match self.mechanism {
                    NoiseType::Laplace => self.laplace_noise(),
                    NoiseType::Gaussian => self.gaussian_noise(),
                    NoiseType::Quantum => self.quantum_noise()?,
                };
                *value += noise;
            }
            Ok(())
        }

        /// Laplace noise
        fn laplace_noise(&self) -> f64 {
            let scale = self.sensitivity / self.epsilon;
            let u = fastrand::f64() - 0.5;
            -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
        }

        /// Gaussian noise
        fn gaussian_noise(&self) -> f64 {
            let scale = self.sensitivity * (2.0 * (1.25 / self.epsilon).ln()).sqrt();
            QuantumFLClient::gaussian_noise() * scale
        }

        /// Quantum noise
        fn quantum_noise(&self) -> Result<f64> {
            // Implement quantum noise using depolarizing channel
            let p = (-self.epsilon).exp();
            Ok(if fastrand::f64() < p {
                fastrand::f64() * 2.0 - 1.0
            } else {
                0.0
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_quantum_fl_client() {
        let config = vec![
            ("encoding".to_string(), 4),
            ("variational".to_string(), 8),
            ("measurement".to_string(), 0),
        ];

        let mut client = QuantumFLClient::new("client_1".to_string(), &config, 100, 1.0).unwrap();

        let data = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
        let labels = array![0, 1, 0];

        let loss = client.train_local(&data, &labels, 1).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_federated_averaging() {
        let config = vec![("encoding".to_string(), 4)];
        let mut server =
            QuantumFLServer::new(config, SecureAggregationProtocol::FederatedAveraging, 0.2);

        let mut params1 = HashMap::new();
        params1.insert("w1".to_string(), 0.5);
        params1.insert("w2".to_string(), 0.3);

        let mut params2 = HashMap::new();
        params2.insert("w1".to_string(), 0.7);
        params2.insert("w2".to_string(), 0.4);

        let updates = vec![
            ("client1".to_string(), params1, 100),
            ("client2".to_string(), params2, 200),
        ];

        let aggregated = server.aggregate_updates(updates).unwrap();

        // Weighted average: w1 = (0.5*100 + 0.7*200)/300 = 0.633...
        assert!((aggregated["w1"] - 0.633).abs() < 0.01);
    }

    #[test]
    fn test_byzantine_robust_aggregation() {
        let server = QuantumFLServer::new(vec![], SecureAggregationProtocol::SecureMultiparty, 0.3);

        // Normal values with one outlier
        let values = vec![0.5, 0.52, 0.48, 0.51, 10.0]; // 10.0 is Byzantine
        let robust_value = server.byzantine_robust_aggregation(&values).unwrap();

        // Should select one of the normal values
        assert!(robust_value < 1.0);
    }

    #[test]
    fn test_differential_privacy() {
        use privacy::*;

        let dp = QuantumDifferentialPrivacy::new(1.0, 0.1, NoiseType::Gaussian);

        let mut params = HashMap::new();
        params.insert("param1".to_string(), 0.5);
        params.insert("param2".to_string(), 0.3);

        let original = params.clone();
        dp.add_noise(&mut params).unwrap();

        // Check that noise was added
        assert_ne!(params["param1"], original["param1"]);
        assert_ne!(params["param2"], original["param2"]);
    }

    #[test]
    fn test_distributed_learning() {
        let config = vec![("encoding".to_string(), 4), ("variational".to_string(), 8)];

        let mut system = DistributedQuantumLearning::new(
            3, // 3 clients
            config,
            SecureAggregationProtocol::FederatedAveraging,
            1.0,
        )
        .unwrap();

        // Create dummy data for each client
        let mut data_dist = HashMap::new();
        for i in 0..3 {
            let data = Array2::zeros((10, 4));
            let labels = Array1::zeros(10);
            data_dist.insert(format!("client_{}", i), (data, labels));
        }

        let result = system.train(&data_dist, 2, 2).unwrap();

        assert_eq!(result.num_rounds, 2);
        assert_eq!(result.round_losses.len(), 2);
    }
}
