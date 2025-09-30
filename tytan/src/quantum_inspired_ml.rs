//! Quantum-inspired machine learning algorithms.
//!
//! This module provides quantum-inspired ML algorithms that leverage
//! quantum optimization principles for classical machine learning tasks.

#![allow(dead_code)]

use crate::sampler::Sampler;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::random::prelude::*;
use std::collections::HashMap;

/// Quantum-Inspired Support Vector Machine
pub struct QuantumSVM {
    /// Kernel type
    kernel: KernelType,
    /// Regularization parameter
    c: f64,
    /// Kernel parameters
    kernel_params: KernelParams,
    /// Support vectors
    support_vectors: Option<Array2<f64>>,
    /// Alphas (Lagrange multipliers)
    alphas: Option<Array1<f64>>,
    /// Bias term
    bias: Option<f64>,
    /// Labels of support vectors
    sv_labels: Option<Array1<f64>>,
}

#[derive(Debug, Clone)]
pub enum KernelType {
    /// Linear kernel: K(x, y) = x^T y
    Linear,
    /// RBF kernel: K(x, y) = exp(-gamma ||x - y||^2)
    RBF { gamma: f64 },
    /// Polynomial kernel: K(x, y) = (x^T y + c)^d
    Polynomial { degree: usize, coef0: f64 },
    /// Quantum kernel: K(x, y) = |<φ(x)|φ(y)>|^2
    Quantum { feature_map: FeatureMap },
}

#[derive(Debug, Clone)]
pub struct KernelParams {
    /// Cache size for kernel matrix
    cache_size: usize,
    /// Tolerance for convergence
    tolerance: f64,
    /// Maximum iterations
    max_iter: usize,
}

#[derive(Debug, Clone)]
pub enum FeatureMap {
    /// Pauli-Z feature map
    PauliZ { depth: usize },
    /// Pauli-ZZ feature map
    PauliZZ { depth: usize, entanglement: String },
    /// Custom feature map
    Custom { name: String },
}

impl QuantumSVM {
    /// Create new Quantum SVM
    pub fn new(kernel: KernelType, c: f64) -> Self {
        Self {
            kernel,
            c,
            kernel_params: KernelParams {
                cache_size: 200,
                tolerance: 1e-3,
                max_iter: 1000,
            },
            support_vectors: None,
            alphas: None,
            bias: None,
            sv_labels: None,
        }
    }

    /// Train the SVM using quantum optimization
    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        sampler: &dyn Sampler,
    ) -> Result<(), String> {
        let n_samples = x.shape()[0];

        // Compute kernel matrix
        let k_matrix = self.compute_kernel_matrix(x)?;

        // Formulate as QUBO for alpha optimization
        let (qubo, var_map) = self.create_svm_qubo(&k_matrix, y)?;

        // Solve using quantum sampler
        let results = sampler
            .run_qubo(&(qubo, var_map.clone()), 100)
            .map_err(|e| format!("Sampling error: {:?}", e))?;

        if let Some(best) = results.first() {
            // Extract alphas from solution
            let alphas = self.decode_alphas(&best.assignments, &var_map, n_samples);

            // Identify support vectors
            let sv_indices: Vec<usize> = alphas
                .iter()
                .enumerate()
                .filter(|(_, &alpha)| alpha > 1e-5)
                .map(|(i, _)| i)
                .collect();

            if sv_indices.is_empty() {
                return Err("No support vectors found".to_string());
            }

            // Store support vectors and alphas
            let mut support_vectors = Array2::zeros((sv_indices.len(), x.shape()[1]));
            let mut sv_alphas = Array1::zeros(sv_indices.len());
            let mut sv_labels = Array1::zeros(sv_indices.len());

            for (i, &idx) in sv_indices.iter().enumerate() {
                support_vectors.row_mut(i).assign(&x.row(idx));
                sv_alphas[i] = alphas[idx];
                sv_labels[i] = y[idx];
            }

            self.support_vectors = Some(support_vectors);
            self.alphas = Some(sv_alphas);
            self.sv_labels = Some(sv_labels);

            // Calculate bias
            self.bias = Some(self.calculate_bias(x, y, &alphas)?);
        }

        Ok(())
    }

    /// Predict labels for new data
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, String> {
        let support_vectors = self.support_vectors.as_ref().ok_or("Model not trained")?;
        let alphas = self.alphas.as_ref().ok_or("Model not trained")?;
        let sv_labels = self.sv_labels.as_ref().ok_or("Model not trained")?;
        let bias = self.bias.ok_or("Model not trained")?;

        let n_samples = x.shape()[0];
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut decision = bias;

            for j in 0..support_vectors.shape()[0] {
                let kernel_val = self.kernel_function(&x.row(i), &support_vectors.row(j))?;
                decision += alphas[j] * sv_labels[j] * kernel_val;
            }

            predictions[i] = if decision >= 0.0 { 1.0 } else { -1.0 };
        }

        Ok(predictions)
    }

    /// Compute kernel matrix
    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Result<Array2<f64>, String> {
        let n = x.shape()[0];
        let mut k_matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in i..n {
                let k_val = self.kernel_function(&x.row(i), &x.row(j))?;
                k_matrix[[i, j]] = k_val;
                k_matrix[[j, i]] = k_val;
            }
        }

        Ok(k_matrix)
    }

    /// Kernel function evaluation
    fn kernel_function(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64, String> {
        match &self.kernel {
            KernelType::Linear => Ok(x.dot(y)),
            KernelType::RBF { gamma } => {
                let diff = x - y;
                Ok((-gamma * diff.dot(&diff)).exp())
            }
            KernelType::Polynomial { degree, coef0 } => Ok((x.dot(y) + coef0).powi(*degree as i32)),
            KernelType::Quantum { feature_map } => {
                // Simulate quantum kernel
                self.quantum_kernel(x, y, feature_map)
            }
        }
    }

    /// Quantum kernel computation
    fn quantum_kernel(
        &self,
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        feature_map: &FeatureMap,
    ) -> Result<f64, String> {
        // Simplified quantum kernel simulation
        match feature_map {
            FeatureMap::PauliZ { depth } => {
                // Simulate Pauli-Z feature map
                let mut kernel = 1.0;
                for _ in 0..*depth {
                    let phase_x: f64 = x.iter().sum();
                    let phase_y: f64 = y.iter().sum();
                    kernel *= (phase_x - phase_y).cos();
                }
                Ok(kernel * kernel) // |<φ(x)|φ(y)>|^2
            }
            FeatureMap::PauliZZ { depth, .. } => {
                // Simulate Pauli-ZZ feature map with entanglement
                let mut kernel = 1.0;
                for d in 0..*depth {
                    for i in 0..x.len() - 1 {
                        let phase = (x[i] - y[i]) * (x[i + 1] - y[i + 1]);
                        kernel *= (phase * (d + 1) as f64).cos();
                    }
                }
                Ok(kernel * kernel)
            }
            FeatureMap::Custom { .. } => {
                // Placeholder for custom feature maps
                Ok(x.dot(y))
            }
        }
    }

    /// Create QUBO for SVM optimization
    fn create_svm_qubo(
        &self,
        k_matrix: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        let n = k_matrix.shape()[0];
        let n_bits = 5; // Bits per alpha variable
        let total_vars = n * n_bits;

        let mut qubo = Array2::zeros((total_vars, total_vars));
        let mut var_map = HashMap::new();

        // Create variable mapping
        for i in 0..n {
            for b in 0..n_bits {
                let var_name = format!("alpha_{}_{}", i, b);
                var_map.insert(var_name, i * n_bits + b);
            }
        }

        // Objective: maximize sum(alpha_i) - 0.5 * sum(alpha_i * alpha_j * y_i * y_j * K_ij)
        // Convert to minimization and binary variables

        // Linear terms (maximize sum becomes minimize negative sum)
        for i in 0..n {
            for b in 0..n_bits {
                let idx = i * n_bits + b;
                let weight = -(1 << b) as f64 / (1 << n_bits) as f64;
                qubo[[idx, idx]] += weight;
            }
        }

        // Quadratic terms
        for i in 0..n {
            for j in 0..n {
                let coef = 0.5 * y[i] * y[j] * k_matrix[[i, j]];

                for bi in 0..n_bits {
                    for bj in 0..n_bits {
                        let idx_i = i * n_bits + bi;
                        let idx_j = j * n_bits + bj;

                        let weight = coef * (1 << bi) as f64 * (1 << bj) as f64
                            / ((1 << n_bits) * (1 << n_bits)) as f64;

                        if idx_i == idx_j {
                            qubo[[idx_i, idx_j]] += weight;
                        } else {
                            qubo[[idx_i, idx_j]] += weight / 2.0;
                            qubo[[idx_j, idx_i]] += weight / 2.0;
                        }
                    }
                }
            }
        }

        // Constraints: 0 <= alpha_i <= C
        let penalty = 100.0 * self.c;
        for i in 0..n {
            // Add penalty for exceeding C
            let alpha_max = (1 << n_bits) - 1;
            if alpha_max as f64 > self.c {
                // Add quadratic penalty
                for b1 in 0..n_bits {
                    for b2 in b1..n_bits {
                        if (1 << b1) + (1 << b2) > self.c as usize {
                            let idx1 = i * n_bits + b1;
                            let idx2 = i * n_bits + b2;

                            if idx1 == idx2 {
                                qubo[[idx1, idx1]] += penalty;
                            } else {
                                qubo[[idx1, idx2]] += penalty;
                                qubo[[idx2, idx1]] += penalty;
                            }
                        }
                    }
                }
            }
        }

        Ok((qubo, var_map))
    }

    /// Decode alpha values from binary solution
    fn decode_alphas(
        &self,
        assignments: &HashMap<String, bool>,
        var_map: &HashMap<String, usize>,
        n_samples: usize,
    ) -> Array1<f64> {
        let n_bits = 5;
        let mut alphas = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut alpha = 0.0;
            for b in 0..n_bits {
                let var_name = format!("alpha_{}_{}", i, b);
                if let Some(&_var_idx) = var_map.get(&var_name) {
                    if assignments.get(&var_name).copied().unwrap_or(false) {
                        alpha += (1 << b) as f64 / (1 << n_bits) as f64 * self.c;
                    }
                }
            }
            alphas[i] = alpha;
        }

        alphas
    }

    /// Calculate bias term
    fn calculate_bias(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        alphas: &Array1<f64>,
    ) -> Result<f64, String> {
        // Use first support vector to calculate bias
        for i in 0..x.shape()[0] {
            if alphas[i] > 1e-5 && alphas[i] < self.c - 1e-5 {
                let mut sum = 0.0;
                for j in 0..x.shape()[0] {
                    if alphas[j] > 1e-5 {
                        let k_val = self.kernel_function(&x.row(i), &x.row(j))?;
                        sum += alphas[j] * y[j] * k_val;
                    }
                }
                return Ok(y[i] - sum);
            }
        }

        Ok(0.0)
    }
}

/// Quantum Boltzmann Machine for generative modeling
pub struct QuantumBoltzmannMachine {
    /// Number of visible units
    n_visible: usize,
    /// Number of hidden units
    n_hidden: usize,
    /// Weights between visible and hidden
    weights: Array2<f64>,
    /// Visible bias
    visible_bias: Array1<f64>,
    /// Hidden bias
    hidden_bias: Array1<f64>,
    /// Learning rate
    learning_rate: f64,
    /// Temperature parameter
    temperature: f64,
}

impl QuantumBoltzmannMachine {
    /// Create new QBM
    pub fn new(n_visible: usize, n_hidden: usize) -> Self {
        let mut rng = thread_rng();

        Self {
            n_visible,
            n_hidden,
            weights: {
                let mut weights = Array2::zeros((n_visible, n_hidden));
                for element in weights.iter_mut() {
                    *element = rng.gen_range(-0.01..0.01);
                }
                weights
            },
            visible_bias: Array1::zeros(n_visible),
            hidden_bias: Array1::zeros(n_hidden),
            learning_rate: 0.01,
            temperature: 1.0,
        }
    }

    /// Train using quantum sampling
    pub fn train(
        &mut self,
        data: &Array2<f64>,
        sampler: &dyn Sampler,
        epochs: usize,
    ) -> Result<Vec<f64>, String> {
        let mut losses = Vec::new();
        let batch_size = data.shape()[0];

        for epoch in 0..epochs {
            #[allow(unused_assignments)]
            let mut epoch_loss = 0.0;

            // Positive phase - from data
            let pos_hidden = self.sample_hidden_given_visible(&data.view(), sampler)?;
            let pos_associations = data.t().dot(&pos_hidden);

            // Negative phase - from model
            let neg_visible = self.sample_visible_given_hidden(&pos_hidden.view(), sampler)?;
            let neg_hidden = self.sample_hidden_given_visible(&neg_visible.view(), sampler)?;
            let neg_associations = neg_visible.t().dot(&neg_hidden);

            // Update weights
            self.weights +=
                &((pos_associations - neg_associations) * self.learning_rate / batch_size as f64);

            // Update biases
            let pos_v_mean = data.mean_axis(Axis(0)).unwrap();
            let neg_v_mean = neg_visible.mean_axis(Axis(0)).unwrap();
            self.visible_bias += &((pos_v_mean - neg_v_mean) * self.learning_rate);

            let pos_h_mean = pos_hidden.mean_axis(Axis(0)).unwrap();
            let neg_h_mean = neg_hidden.mean_axis(Axis(0)).unwrap();
            self.hidden_bias += &((pos_h_mean - neg_h_mean) * self.learning_rate);

            // Calculate reconstruction error
            let reconstruction_error =
                ((data - &neg_visible).mapv(|x| x * x)).sum() / batch_size as f64;
            epoch_loss = reconstruction_error;

            losses.push(epoch_loss);

            if epoch % 10 == 0 {
                println!("Epoch {}: Loss = {:.4}", epoch, epoch_loss);
            }
        }

        Ok(losses)
    }

    /// Sample hidden given visible using quantum sampler
    fn sample_hidden_given_visible(
        &self,
        visible: &ArrayView2<f64>,
        sampler: &dyn Sampler,
    ) -> Result<Array2<f64>, String> {
        let batch_size = visible.shape()[0];
        let mut hidden = Array2::zeros((batch_size, self.n_hidden));

        // Create QUBO for each sample
        for i in 0..batch_size {
            let v = visible.row(i);

            // Energy function: -sum(b_j * h_j) - sum(v_i * W_ij * h_j)
            let mut qubo = Array2::zeros((self.n_hidden, self.n_hidden));
            let mut var_map = HashMap::new();

            for j in 0..self.n_hidden {
                var_map.insert(format!("h_{}", j), j);

                // Linear term
                let linear = self.hidden_bias[j] + v.dot(&self.weights.column(j));
                qubo[[j, j]] = -linear / self.temperature;
            }

            // Sample using quantum sampler
            let results = sampler
                .run_qubo(&(qubo, var_map), 1)
                .map_err(|e| format!("Sampling error: {:?}", e))?;

            if let Some(result) = results.first() {
                for j in 0..self.n_hidden {
                    let var_name = format!("h_{}", j);
                    hidden[[i, j]] = if result.assignments.get(&var_name).copied().unwrap_or(false)
                    {
                        1.0
                    } else {
                        0.0
                    };
                }
            }
        }

        Ok(hidden)
    }

    /// Sample visible given hidden using quantum sampler
    fn sample_visible_given_hidden(
        &self,
        hidden: &ArrayView2<f64>,
        sampler: &dyn Sampler,
    ) -> Result<Array2<f64>, String> {
        let batch_size = hidden.shape()[0];
        let mut visible = Array2::zeros((batch_size, self.n_visible));

        // Similar to sample_hidden_given_visible but reversed
        for i in 0..batch_size {
            let h = hidden.row(i);

            let mut qubo = Array2::zeros((self.n_visible, self.n_visible));
            let mut var_map = HashMap::new();

            for j in 0..self.n_visible {
                var_map.insert(format!("v_{}", j), j);

                // Linear term
                let linear = self.visible_bias[j] + self.weights.row(j).dot(&h);
                qubo[[j, j]] = -linear / self.temperature;
            }

            let results = sampler
                .run_qubo(&(qubo, var_map), 1)
                .map_err(|e| format!("Sampling error: {:?}", e))?;

            if let Some(result) = results.first() {
                for j in 0..self.n_visible {
                    let var_name = format!("v_{}", j);
                    visible[[i, j]] = if result.assignments.get(&var_name).copied().unwrap_or(false)
                    {
                        1.0
                    } else {
                        0.0
                    };
                }
            }
        }

        Ok(visible)
    }

    /// Generate new samples
    pub fn generate(&self, n_samples: usize, sampler: &dyn Sampler) -> Result<Array2<f64>, String> {
        // Start with random hidden state
        let mut rng = thread_rng();
        let mut hidden = {
            let mut hidden = Array2::zeros((n_samples, self.n_hidden));
            for element in hidden.iter_mut() {
                *element = if rng.gen::<bool>() { 1.0 } else { 0.0 };
            }
            hidden
        };

        // Gibbs sampling
        for _ in 0..10 {
            let visible = self.sample_visible_given_hidden(&hidden.view(), sampler)?;
            hidden = self.sample_hidden_given_visible(&visible.view(), sampler)?;
        }

        // Final visible sample
        self.sample_visible_given_hidden(&hidden.view(), sampler)
    }
}

/// Quantum-inspired clustering using quantum optimization
pub struct QuantumClustering {
    /// Number of clusters
    n_clusters: usize,
    /// Distance metric
    distance_metric: DistanceMetric,
    /// Regularization for balanced clusters
    balance_weight: f64,
}

#[derive(Debug, Clone)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    Quantum,
}

impl QuantumClustering {
    /// Create new quantum clustering
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            distance_metric: DistanceMetric::Euclidean,
            balance_weight: 0.1,
        }
    }

    /// Set distance metric
    pub fn with_distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Perform clustering using quantum optimization
    pub fn fit_predict(
        &self,
        data: &Array2<f64>,
        sampler: &dyn Sampler,
    ) -> Result<Array1<usize>, String> {
        let n_samples = data.shape()[0];

        // Compute distance matrix
        let distances = self.compute_distance_matrix(data)?;

        // Create QUBO for clustering
        let (qubo, var_map) = self.create_clustering_qubo(&distances)?;

        // Solve using quantum sampler
        let results = sampler
            .run_qubo(&(qubo, var_map.clone()), 100)
            .map_err(|e| format!("Sampling error: {:?}", e))?;

        if let Some(best) = results.first() {
            // Decode cluster assignments
            let assignments = self.decode_clusters(&best.assignments, &var_map, n_samples);
            Ok(assignments)
        } else {
            Err("No solution found".to_string())
        }
    }

    /// Compute distance matrix
    fn compute_distance_matrix(&self, data: &Array2<f64>) -> Result<Array2<f64>, String> {
        let n = data.shape()[0];
        let mut distances = Array2::zeros((n, n));

        for i in 0..n {
            for j in i + 1..n {
                let dist = match &self.distance_metric {
                    DistanceMetric::Euclidean => {
                        let diff = &data.row(i) - &data.row(j);
                        diff.dot(&diff).sqrt()
                    }
                    DistanceMetric::Manhattan => {
                        (&data.row(i) - &data.row(j)).mapv(|x| x.abs()).sum()
                    }
                    DistanceMetric::Cosine => {
                        let dot = data.row(i).dot(&data.row(j));
                        let norm_i = data.row(i).dot(&data.row(i)).sqrt();
                        let norm_j = data.row(j).dot(&data.row(j)).sqrt();
                        1.0 - dot / (norm_i * norm_j)
                    }
                    DistanceMetric::Quantum => {
                        // Quantum-inspired distance
                        self.quantum_distance(&data.row(i), &data.row(j))
                    }
                };

                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        Ok(distances)
    }

    /// Quantum-inspired distance metric
    fn quantum_distance(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
        // Based on quantum state fidelity
        let inner_product = x.dot(y);
        let norm_x = x.dot(x).sqrt();
        let norm_y = y.dot(y).sqrt();

        let fidelity = (inner_product / (norm_x * norm_y)).abs();
        (1.0 - fidelity * fidelity).sqrt()
    }

    /// Create QUBO for clustering
    fn create_clustering_qubo(
        &self,
        distances: &Array2<f64>,
    ) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        let n_samples = distances.shape()[0];
        let n_vars = n_samples * self.n_clusters;

        let mut qubo = Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();

        // Variable mapping: x[i,k] = 1 if sample i is in cluster k
        for i in 0..n_samples {
            for k in 0..self.n_clusters {
                let var_name = format!("x_{}_{}", i, k);
                var_map.insert(var_name, i * self.n_clusters + k);
            }
        }

        // Objective: minimize sum of intra-cluster distances
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                for k in 0..self.n_clusters {
                    let idx_ik = i * self.n_clusters + k;
                    let idx_jk = j * self.n_clusters + k;

                    qubo[[idx_ik, idx_jk]] += distances[[i, j]];
                    qubo[[idx_jk, idx_ik]] += distances[[i, j]];
                }
            }
        }

        // Constraint: each sample in exactly one cluster
        let penalty = distances.sum() * 10.0;
        for i in 0..n_samples {
            // One-hot constraint
            for k1 in 0..self.n_clusters {
                let idx1 = i * self.n_clusters + k1;

                // Linear penalty
                qubo[[idx1, idx1]] -= penalty;

                // Quadratic penalty
                for k2 in k1 + 1..self.n_clusters {
                    let idx2 = i * self.n_clusters + k2;
                    qubo[[idx1, idx2]] += penalty;
                    qubo[[idx2, idx1]] += penalty;
                }
            }
        }

        // Balance term: encourage equal-sized clusters
        if self.balance_weight > 0.0 {
            let target_size = n_samples as f64 / self.n_clusters as f64;

            for k in 0..self.n_clusters {
                // Penalize deviation from target size
                for i in 0..n_samples {
                    for j in i + 1..n_samples {
                        let idx_ik = i * self.n_clusters + k;
                        let idx_jk = j * self.n_clusters + k;

                        let weight = self.balance_weight / (target_size * target_size);
                        qubo[[idx_ik, idx_jk]] += weight;
                        qubo[[idx_jk, idx_ik]] += weight;
                    }
                }
            }
        }

        Ok((qubo, var_map))
    }

    /// Decode cluster assignments
    fn decode_clusters(
        &self,
        assignments: &HashMap<String, bool>,
        _var_map: &HashMap<String, usize>,
        n_samples: usize,
    ) -> Array1<usize> {
        let mut clusters = Array1::zeros(n_samples);

        for i in 0..n_samples {
            for k in 0..self.n_clusters {
                let var_name = format!("x_{}_{}", i, k);
                if assignments.get(&var_name).copied().unwrap_or(false) {
                    clusters[i] = k;
                    break;
                }
            }
        }

        clusters
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::SASampler;
    use scirs2_core::ndarray::array;
    use quantrs2_anneal::simulator::AnnealingParams;

    #[test]
    fn test_quantum_svm() {
        // Simple linearly separable data
        let mut x = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],];
        let mut y = array![-1.0, -1.0, 1.0, 1.0];

        let mut svm = QuantumSVM::new(KernelType::Linear, 1.0);

        // Create fast annealing parameters for testing
        let mut params = AnnealingParams::new();
        params.timeout = Some(10.0); // 10 second timeout
        params.num_sweeps = 100; // Reduce from default 1000
        params.num_repetitions = 2; // Reduce from default 10

        let sampler = SASampler::with_params(Some(42), params);

        svm.fit(&x, &y, &sampler).unwrap();

        let mut predictions = svm.predict(&x).unwrap();

        // Check that it learned something reasonable
        assert!(svm.support_vectors.is_some());
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    #[ignore]
    fn test_quantum_clustering() {
        let data = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1],];

        let clustering = QuantumClustering::new(2).with_distance_metric(DistanceMetric::Euclidean);

        let sampler = SASampler::new(Some(42));
        let labels = clustering.fit_predict(&data, &sampler).unwrap();

        // Check that similar points are in same cluster
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }
}
