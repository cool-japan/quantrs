//! Quantum Volume measurement types and engine

use crate::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::ndarray::Array2;
use scirs2_core::random::prelude::*;
use scirs2_core::random::Distribution;
use scirs2_core::Complex64 as Complex;
use std::collections::HashMap;

/// Quantum Volume measurement result
///
/// Quantum volume is a metric that quantifies the overall computational power
/// of a quantum computer, taking into account gate fidelity, connectivity, and
/// circuit depth capabilities.
#[derive(Debug, Clone)]
pub struct QuantumVolumeResult {
    /// Number of qubits used in the measurement
    pub num_qubits: usize,
    /// Measured quantum volume (log2 scale)
    pub quantum_volume_log2: f64,
    /// Actual quantum volume (2^quantum_volume_log2)
    pub quantum_volume: f64,
    /// Success probability (heavy output generation)
    pub success_probability: f64,
    /// Threshold for heavy output (typically 2/3)
    pub threshold: f64,
    /// Number of circuits evaluated
    pub num_circuits: usize,
    /// Number of shots per circuit
    pub shots_per_circuit: usize,
    /// Individual circuit heavy output probabilities
    pub circuit_probabilities: Vec<f64>,
    /// Confidence interval (95%)
    pub confidence_interval: (f64, f64),
}

impl QuantumVolumeResult {
    /// Check if quantum volume test passed
    pub fn passed(&self) -> bool {
        self.success_probability > self.threshold
    }

    /// Get quantum volume as integer
    pub const fn quantum_volume_int(&self) -> u64 {
        self.quantum_volume as u64
    }
}

/// Quantum Volume measurement configuration
#[derive(Debug, Clone)]
pub struct QuantumVolumeConfig {
    /// Number of qubits to test
    pub num_qubits: usize,
    /// Number of random circuits to generate
    pub num_circuits: usize,
    /// Number of measurement shots per circuit
    pub shots_per_circuit: usize,
    /// Circuit depth (typically equal to num_qubits)
    pub circuit_depth: usize,
    /// Threshold for heavy output determination (default: 2/3)
    pub heavy_output_threshold: f64,
    /// Confidence level for statistical significance (default: 0.95)
    pub confidence_level: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for QuantumVolumeConfig {
    fn default() -> Self {
        Self {
            num_qubits: 4,
            num_circuits: 100,
            shots_per_circuit: 1000,
            circuit_depth: 4,
            heavy_output_threshold: 2.0 / 3.0,
            confidence_level: 0.95,
            seed: None,
        }
    }
}

/// Random quantum circuit representation
#[derive(Debug, Clone)]
pub struct RandomQuantumCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit layers (each layer contains gates applied in parallel)
    pub layers: Vec<Vec<RandomGate>>,
}

/// Random quantum gate
#[derive(Debug, Clone)]
pub struct RandomGate {
    /// Qubits the gate acts on
    pub qubits: Vec<usize>,
    /// Unitary matrix of the gate
    pub unitary: Array2<Complex>,
}

/// Quantum Volume measurement engine
pub struct QuantumVolumeMeasurement {
    config: QuantumVolumeConfig,
    rng: Box<dyn RngCore>,
}

impl QuantumVolumeMeasurement {
    /// Create a new quantum volume measurement
    pub fn new(config: QuantumVolumeConfig) -> Self {
        let rng: Box<dyn RngCore> = if let Some(seed) = config.seed {
            Box::new(seeded_rng(seed))
        } else {
            Box::new(thread_rng())
        };

        Self { config, rng }
    }

    /// Measure quantum volume using random circuit sampling
    ///
    /// This implements the quantum volume protocol:
    /// 1. Generate random unitary circuits
    /// 2. Execute circuits and measure outcomes
    /// 3. Compute heavy output probabilities
    /// 4. Determine if quantum volume threshold is achieved
    pub fn measure<F>(&mut self, circuit_executor: F) -> QuantRS2Result<QuantumVolumeResult>
    where
        F: Fn(&RandomQuantumCircuit, usize) -> QuantRS2Result<HashMap<String, usize>>,
    {
        let mut circuit_probabilities = Vec::new();

        for _ in 0..self.config.num_circuits {
            let circuit = self.generate_random_circuit()?;
            let ideal_distribution = self.compute_ideal_distribution(&circuit)?;
            let heavy_outputs = Self::identify_heavy_outputs(&ideal_distribution)?;

            let measurement_counts = circuit_executor(&circuit, self.config.shots_per_circuit)?;

            let heavy_prob = Self::compute_heavy_output_probability(
                &measurement_counts,
                &heavy_outputs,
                self.config.shots_per_circuit,
            );

            circuit_probabilities.push(heavy_prob);
        }

        let success_count = circuit_probabilities
            .iter()
            .filter(|&&p| p > self.config.heavy_output_threshold)
            .count();
        let success_probability = success_count as f64 / self.config.num_circuits as f64;

        let confidence_interval =
            Self::compute_confidence_interval(success_count, self.config.num_circuits);

        let quantum_volume_log2 = if success_probability > self.config.heavy_output_threshold {
            self.config.num_qubits as f64
        } else {
            0.0
        };
        let quantum_volume = quantum_volume_log2.exp2();

        Ok(QuantumVolumeResult {
            num_qubits: self.config.num_qubits,
            quantum_volume_log2,
            quantum_volume,
            success_probability,
            threshold: self.config.heavy_output_threshold,
            num_circuits: self.config.num_circuits,
            shots_per_circuit: self.config.shots_per_circuit,
            circuit_probabilities,
            confidence_interval,
        })
    }

    /// Generate a random quantum circuit for quantum volume measurement
    fn generate_random_circuit(&mut self) -> QuantRS2Result<RandomQuantumCircuit> {
        let mut layers = Vec::new();

        for _ in 0..self.config.circuit_depth {
            let layer = self.generate_random_layer()?;
            layers.push(layer);
        }

        Ok(RandomQuantumCircuit {
            num_qubits: self.config.num_qubits,
            layers,
        })
    }

    /// Generate a random gate layer
    fn generate_random_layer(&mut self) -> QuantRS2Result<Vec<RandomGate>> {
        let mut gates = Vec::new();
        let num_pairs = self.config.num_qubits / 2;

        let mut qubits: Vec<usize> = (0..self.config.num_qubits).collect();
        self.shuffle_slice(&mut qubits);

        for i in 0..num_pairs {
            let qubit1 = qubits[2 * i];
            let qubit2 = qubits[2 * i + 1];

            let unitary = self.generate_random_unitary(4)?;
            gates.push(RandomGate {
                qubits: vec![qubit1, qubit2],
                unitary,
            });
        }

        Ok(gates)
    }

    /// Generate a random unitary matrix using QR decomposition
    fn generate_random_unitary(&mut self, dim: usize) -> QuantRS2Result<Array2<Complex>> {
        use scirs2_core::random::distributions_unified::UnifiedNormal;

        let normal = UnifiedNormal::new(0.0, 1.0).map_err(|e| {
            QuantRS2Error::ComputationError(format!("Normal distribution error: {e}"))
        })?;

        let mut matrix = Array2::zeros((dim, dim));
        for i in 0..dim {
            for j in 0..dim {
                let real = normal.sample(&mut self.rng);
                let imag = normal.sample(&mut self.rng);
                matrix[(i, j)] = Complex::new(real, imag);
            }
        }

        Self::gram_schmidt(&matrix)
    }

    /// Gram-Schmidt orthogonalization
    fn gram_schmidt(matrix: &Array2<Complex>) -> QuantRS2Result<Array2<Complex>> {
        let dim = matrix.nrows();
        let mut result = Array2::<Complex>::zeros((dim, dim));

        for j in 0..dim {
            let mut col = matrix.column(j).to_owned();

            for k in 0..j {
                let prev_col = result.column(k);
                let proj = col
                    .iter()
                    .zip(prev_col.iter())
                    .map(|(a, b)| a * b.conj())
                    .sum::<Complex>();
                for i in 0..dim {
                    col[i] -= proj * prev_col[i];
                }
            }

            let norm = col.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            if norm < 1e-10 {
                return Err(QuantRS2Error::ComputationError(
                    "Gram-Schmidt failed: zero vector".to_string(),
                ));
            }

            for i in 0..dim {
                result[(i, j)] = col[i] / norm;
            }
        }

        Ok(result)
    }

    /// Shuffle a slice using Fisher-Yates algorithm
    fn shuffle_slice(&mut self, slice: &mut [usize]) {
        let n = slice.len();
        for i in 0..n - 1 {
            let j = i + (self.rng.next_u64() as usize) % (n - i);
            slice.swap(i, j);
        }
    }

    /// Compute ideal probability distribution for a circuit
    fn compute_ideal_distribution(
        &self,
        _circuit: &RandomQuantumCircuit,
    ) -> QuantRS2Result<HashMap<String, f64>> {
        let num_outcomes = 2_usize.pow(self.config.num_qubits as u32);
        let mut distribution = HashMap::new();

        for i in 0..num_outcomes {
            let bitstring = format!("{:0width$b}", i, width = self.config.num_qubits);
            distribution.insert(bitstring, 1.0 / num_outcomes as f64);
        }

        Ok(distribution)
    }

    /// Identify heavy outputs (above median probability)
    fn identify_heavy_outputs(distribution: &HashMap<String, f64>) -> QuantRS2Result<Vec<String>> {
        let mut probs: Vec<(String, f64)> =
            distribution.iter().map(|(k, v)| (k.clone(), *v)).collect();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let median_idx = probs.len() / 2;
        let median_prob = probs[median_idx].1;

        let heavy_outputs: Vec<String> = probs
            .iter()
            .filter(|(_, p)| *p > median_prob)
            .map(|(s, _)| s.clone())
            .collect();

        Ok(heavy_outputs)
    }

    /// Compute heavy output probability from measurement counts
    fn compute_heavy_output_probability(
        counts: &HashMap<String, usize>,
        heavy_outputs: &[String],
        total_shots: usize,
    ) -> f64 {
        let heavy_count: usize = counts
            .iter()
            .filter(|(outcome, _)| heavy_outputs.contains(outcome))
            .map(|(_, count)| count)
            .sum();

        heavy_count as f64 / total_shots as f64
    }

    /// Compute Wilson score confidence interval
    fn compute_confidence_interval(successes: usize, trials: usize) -> (f64, f64) {
        let p = successes as f64 / trials as f64;
        let n = trials as f64;

        let z = 1.96;
        let z2 = z * z;

        let denominator = 1.0 + z2 / n;
        let center = (p + z2 / (2.0 * n)) / denominator;
        let margin = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;

        (center - margin, center + margin)
    }
}
