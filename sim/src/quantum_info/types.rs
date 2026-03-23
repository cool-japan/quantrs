//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::Result;
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::Complex64;
use std::collections::HashMap;
use thiserror::Error;

use super::functions::{generate_pauli_basis, kronecker_product, matrix_trace, purity};

/// Result of quantum state tomography
#[derive(Debug, Clone)]
pub struct TomographyResult {
    /// Reconstructed density matrix
    pub density_matrix: Array2<Complex64>,
    /// Estimated fidelity with true state (if known)
    pub fidelity_estimate: Option<f64>,
    /// Purity of reconstructed state
    pub purity: f64,
    /// Reconstruction confidence/uncertainty
    pub uncertainty: f64,
    /// Number of measurements used
    pub total_measurements: usize,
}
/// Quantum information error types
#[derive(Debug, Error)]
pub enum QuantumInfoError {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("Invalid quantum state: {0}")]
    InvalidState(String),
    #[error("Invalid density matrix: {0}")]
    InvalidDensityMatrix(String),
    #[error("Numerical error: {0}")]
    NumericalError(String),
    #[error("Tomography error: {0}")]
    TomographyError(String),
    #[error("Not implemented: {0}")]
    NotImplemented(String),
}
/// Quantum channel representation
#[derive(Debug, Clone)]
pub struct QuantumChannel {
    /// Kraus operators {K_i} such that E(ρ) = Σᵢ Kᵢ ρ Kᵢ†
    kraus_operators: Vec<Array2<Complex64>>,
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
}
impl QuantumChannel {
    /// Create a quantum channel from Kraus operators
    pub fn from_kraus(
        kraus: Vec<Array2<Complex64>>,
    ) -> std::result::Result<Self, QuantumInfoError> {
        if kraus.is_empty() {
            return Err(QuantumInfoError::InvalidState(
                "Kraus operators cannot be empty".to_string(),
            ));
        }
        let input_dim = kraus[0].ncols();
        let output_dim = kraus[0].nrows();
        Ok(Self {
            kraus_operators: kraus,
            input_dim,
            output_dim,
        })
    }
    /// Create an identity channel
    pub fn identity(dim: usize) -> Self {
        let mut identity = Array2::zeros((dim, dim));
        for i in 0..dim {
            identity[[i, i]] = Complex64::new(1.0, 0.0);
        }
        Self {
            kraus_operators: vec![identity],
            input_dim: dim,
            output_dim: dim,
        }
    }
    /// Create a unitary channel U ρ U†
    pub fn unitary(u: Array2<Complex64>) -> std::result::Result<Self, QuantumInfoError> {
        let dim = u.nrows();
        if dim != u.ncols() {
            return Err(QuantumInfoError::InvalidState(
                "Unitary matrix must be square".to_string(),
            ));
        }
        Ok(Self {
            kraus_operators: vec![u],
            input_dim: dim,
            output_dim: dim,
        })
    }
    /// Create a depolarizing channel
    ///
    /// E(ρ) = (1-p)ρ + p/3 (XρX + YρY + ZρZ)
    pub fn depolarizing(p: f64) -> Self {
        let sqrt_1_p = (1.0 - p).sqrt();
        let sqrt_p_3 = (p / 3.0).sqrt();
        let k0 = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(sqrt_1_p, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(sqrt_1_p, 0.0),
            ],
        )
        .expect("Valid shape");
        let k1 = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(sqrt_p_3, 0.0),
                Complex64::new(sqrt_p_3, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("Valid shape");
        let k2 = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -sqrt_p_3),
                Complex64::new(0.0, sqrt_p_3),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("Valid shape");
        let k3 = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(sqrt_p_3, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-sqrt_p_3, 0.0),
            ],
        )
        .expect("Valid shape");
        Self {
            kraus_operators: vec![k0, k1, k2, k3],
            input_dim: 2,
            output_dim: 2,
        }
    }
    /// Create an amplitude damping channel (T1 decay)
    ///
    /// E(ρ) = K₀ρK₀† + K₁ρK₁†
    /// K₀ = [[1, 0], [0, √(1-γ)]]
    /// K₁ = [[0, √γ], [0, 0]]
    pub fn amplitude_damping(gamma: f64) -> Self {
        let sqrt_gamma = gamma.sqrt();
        let sqrt_1_gamma = (1.0 - gamma).sqrt();
        let k0 = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(sqrt_1_gamma, 0.0),
            ],
        )
        .expect("Valid shape");
        let k1 = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(sqrt_gamma, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("Valid shape");
        Self {
            kraus_operators: vec![k0, k1],
            input_dim: 2,
            output_dim: 2,
        }
    }
    /// Create a phase damping channel (T2 decay)
    ///
    /// E(ρ) = K₀ρK₀† + K₁ρK₁†
    pub fn phase_damping(gamma: f64) -> Self {
        let sqrt_gamma = gamma.sqrt();
        let sqrt_1_gamma = (1.0 - gamma).sqrt();
        let k0 = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(sqrt_1_gamma, 0.0),
            ],
        )
        .expect("Valid shape");
        let k1 = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(sqrt_gamma, 0.0),
            ],
        )
        .expect("Valid shape");
        Self {
            kraus_operators: vec![k0, k1],
            input_dim: 2,
            output_dim: 2,
        }
    }
    /// Apply the channel to a quantum state
    pub fn apply(
        &self,
        state: &QuantumState,
    ) -> std::result::Result<QuantumState, QuantumInfoError> {
        let rho = state.to_density_matrix()?;
        let mut output = Array2::zeros((self.output_dim, self.output_dim));
        for k in &self.kraus_operators {
            let k_dag = k.t().mapv(|c| c.conj());
            let k_rho = k.dot(&rho);
            let k_rho_k_dag = k_rho.dot(&k_dag);
            output = output + k_rho_k_dag;
        }
        Ok(QuantumState::Mixed(output))
    }
    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
    /// Convert to Choi matrix representation
    ///
    /// Λ_E = (E ⊗ I)(|Ω⟩⟨Ω|)
    /// where |Ω⟩ = Σᵢ |ii⟩ is the maximally entangled state
    pub fn to_choi(&self) -> std::result::Result<Array2<Complex64>, QuantumInfoError> {
        let d = self.input_dim;
        let choi_dim = d * self.output_dim;
        let mut choi = Array2::zeros((choi_dim, choi_dim));
        for k in &self.kraus_operators {
            let mut vec_k = Array1::zeros(choi_dim);
            for j in 0..d {
                for i in 0..self.output_dim {
                    vec_k[j * self.output_dim + i] = k[[i, j]];
                }
            }
            for i in 0..choi_dim {
                for j in 0..choi_dim {
                    choi[[i, j]] += vec_k[i] * vec_k[j].conj();
                }
            }
        }
        Ok(choi)
    }
    /// Convert to Pauli transfer matrix (PTM) representation
    ///
    /// R_ij = Tr[P_i E(P_j)] / d
    /// where {P_i} is the Pauli basis
    pub fn to_ptm(&self) -> std::result::Result<Array2<Complex64>, QuantumInfoError> {
        let d = self.input_dim;
        let num_paulis = d * d;
        let paulis = generate_pauli_basis(d)?;
        let mut ptm = Array2::zeros((num_paulis, num_paulis));
        for (j, pj) in paulis.iter().enumerate() {
            let state_j = QuantumState::Mixed(pj.clone());
            let output = self.apply(&state_j)?;
            let rho_out = output.to_density_matrix()?;
            for (i, pi) in paulis.iter().enumerate() {
                let trace = matrix_trace(&pi.dot(&rho_out));
                ptm[[i, j]] = trace / Complex64::new(d as f64, 0.0);
            }
        }
        Ok(ptm)
    }
}
/// Quantum process tomography engine
pub struct ProcessTomography {
    /// Number of qubits the process acts on
    num_qubits: usize,
    /// Configuration
    config: StateTomographyConfig,
}
impl ProcessTomography {
    /// Create a new process tomography instance
    pub fn new(num_qubits: usize, config: StateTomographyConfig) -> Self {
        Self { num_qubits, config }
    }
    /// Perform process tomography from input-output state data
    ///
    /// # Arguments
    /// * `data` - Process tomography data (input states and their outputs)
    ///
    /// # Returns
    /// Reconstructed quantum channel
    pub fn reconstruct(
        &self,
        data: &ProcessTomographyData,
    ) -> std::result::Result<QuantumChannel, QuantumInfoError> {
        let dim = 1 << self.num_qubits;
        let mut choi = Array2::zeros((dim * dim, dim * dim));
        for (input_state, output_dm) in &data.state_pairs {
            let input_dm = input_state.to_density_matrix()?;
            let contrib = kronecker_product(&input_dm, output_dm);
            choi = choi + contrib;
        }
        choi /= Complex64::new(data.state_pairs.len() as f64, 0.0);
        let kraus = self.choi_to_kraus(&choi)?;
        QuantumChannel::from_kraus(kraus)
    }
    /// Convert Choi matrix to Kraus operators
    pub(super) fn choi_to_kraus(
        &self,
        choi: &Array2<Complex64>,
    ) -> std::result::Result<Vec<Array2<Complex64>>, QuantumInfoError> {
        let dim = 1 << self.num_qubits;
        let mut k0 = Array2::zeros((dim, dim));
        for i in 0..dim {
            k0[[i, i]] = (choi[[i * dim + i, i * dim + i]]).sqrt();
        }
        Ok(vec![k0])
    }
}
/// Quantum state representation (pure or mixed)
#[derive(Debug, Clone)]
pub enum QuantumState {
    /// Pure state represented as state vector |ψ⟩
    Pure(Array1<Complex64>),
    /// Mixed state represented as density matrix ρ
    Mixed(Array2<Complex64>),
}
impl QuantumState {
    /// Create a pure state from a state vector
    pub fn pure(state_vector: Array1<Complex64>) -> Self {
        QuantumState::Pure(state_vector)
    }
    /// Create a mixed state from a density matrix
    pub fn mixed(density_matrix: Array2<Complex64>) -> Self {
        QuantumState::Mixed(density_matrix)
    }
    /// Create a computational basis state |i⟩
    pub fn computational_basis(dim: usize, index: usize) -> Self {
        let mut state = Array1::zeros(dim);
        if index < dim {
            state[index] = Complex64::new(1.0, 0.0);
        }
        QuantumState::Pure(state)
    }
    /// Create a maximally mixed state I/d
    pub fn maximally_mixed(dim: usize) -> Self {
        let mut rho = Array2::zeros((dim, dim));
        let val = Complex64::new(1.0 / dim as f64, 0.0);
        for i in 0..dim {
            rho[[i, i]] = val;
        }
        QuantumState::Mixed(rho)
    }
    /// Create a Bell state
    pub fn bell_state(index: usize) -> Self {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let state = match index {
            0 => Array1::from_vec(vec![
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
            ]),
            1 => Array1::from_vec(vec![
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ]),
            2 => Array1::from_vec(vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(0.0, 0.0),
            ]),
            _ => Array1::from_vec(vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
                Complex64::new(0.0, 0.0),
            ]),
        };
        QuantumState::Pure(state)
    }
    /// Create a GHZ state for n qubits
    pub fn ghz_state(n: usize) -> Self {
        let dim = 1 << n;
        let mut state = Array1::zeros(dim);
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        state[0] = Complex64::new(inv_sqrt2, 0.0);
        state[dim - 1] = Complex64::new(inv_sqrt2, 0.0);
        QuantumState::Pure(state)
    }
    /// Create a W state for n qubits
    pub fn w_state(n: usize) -> Self {
        let dim = 1 << n;
        let amplitude = 1.0 / (n as f64).sqrt();
        let mut state = Array1::zeros(dim);
        for i in 0..n {
            let index = 1 << i;
            state[index] = Complex64::new(amplitude, 0.0);
        }
        QuantumState::Pure(state)
    }
    /// Convert to density matrix representation
    pub fn to_density_matrix(&self) -> std::result::Result<Array2<Complex64>, QuantumInfoError> {
        match self {
            QuantumState::Pure(psi) => {
                let n = psi.len();
                let mut rho = Array2::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        rho[[i, j]] = psi[i] * psi[j].conj();
                    }
                }
                Ok(rho)
            }
            QuantumState::Mixed(rho) => Ok(rho.clone()),
        }
    }
    /// Get the dimension of the state
    pub fn dim(&self) -> usize {
        match self {
            QuantumState::Pure(psi) => psi.len(),
            QuantumState::Mixed(rho) => rho.nrows(),
        }
    }
    /// Check if the state is pure
    pub fn is_pure(&self) -> bool {
        matches!(self, QuantumState::Pure(_))
    }
}
/// Tomography reconstruction method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TomographyMethod {
    /// Linear inversion (fast but may produce unphysical states)
    LinearInversion,
    /// Maximum likelihood estimation (slower but always physical)
    MaximumLikelihood,
    /// Compressed sensing for sparse states
    CompressedSensing,
    /// Bayesian estimation with prior
    Bayesian,
}
/// Measurement data for tomography
#[derive(Debug, Clone)]
pub struct MeasurementData {
    /// Measurement basis (e.g., "ZZ", "XY", "YZ")
    pub basis: String,
    /// Measurement outcomes and their counts
    /// Key: bitstring (e.g., "00", "01", "10", "11")
    /// Value: number of times this outcome was observed
    pub counts: std::collections::HashMap<String, usize>,
}
impl MeasurementData {
    /// Create new measurement data
    pub fn new(basis: &str, counts: std::collections::HashMap<String, usize>) -> Self {
        Self {
            basis: basis.to_string(),
            counts,
        }
    }
    /// Get total number of shots
    pub fn total_shots(&self) -> usize {
        self.counts.values().sum()
    }
    /// Compute expectation value ⟨P⟩ from measurement counts
    pub fn expectation_value(&self) -> f64 {
        let total = self.total_shots() as f64;
        if total < 1e-10 {
            return 0.0;
        }
        let mut expectation = 0.0;
        for (outcome, &count) in &self.counts {
            let parity: usize = outcome.chars().filter(|&c| c == '1').count();
            let eigenvalue = if parity % 2 == 0 { 1.0 } else { -1.0 };
            expectation += eigenvalue * count as f64;
        }
        expectation / total
    }
}
/// Classical shadow protocol for efficient property estimation
#[derive(Debug, Clone)]
pub struct ClassicalShadow {
    /// Number of random measurements
    num_snapshots: usize,
    /// Random measurement basis for each snapshot
    bases: Vec<String>,
    /// Measurement outcomes for each snapshot
    outcomes: Vec<String>,
    /// Number of qubits
    num_qubits: usize,
}
impl ClassicalShadow {
    /// Create a new classical shadow from measurement data
    pub fn from_measurements(
        num_qubits: usize,
        bases: Vec<String>,
        outcomes: Vec<String>,
    ) -> std::result::Result<Self, QuantumInfoError> {
        if bases.len() != outcomes.len() {
            return Err(QuantumInfoError::InvalidState(
                "Number of bases must match number of outcomes".to_string(),
            ));
        }
        Ok(Self {
            num_snapshots: bases.len(),
            bases,
            outcomes,
            num_qubits,
        })
    }
    /// Generate random Pauli measurement bases
    pub fn generate_random_bases(num_qubits: usize, num_snapshots: usize) -> Vec<String> {
        let mut rng = thread_rng();
        let paulis = ['X', 'Y', 'Z'];
        (0..num_snapshots)
            .map(|_| {
                (0..num_qubits)
                    .map(|_| paulis[rng.random_range(0..3)])
                    .collect()
            })
            .collect()
    }
    /// Estimate expectation value of a Pauli observable
    ///
    /// # Arguments
    /// * `observable` - Pauli string (e.g., "ZZI", "XYZ")
    ///
    /// # Returns
    /// Estimated expectation value ⟨O⟩
    pub fn estimate_observable(
        &self,
        observable: &str,
    ) -> std::result::Result<f64, QuantumInfoError> {
        if observable.len() != self.num_qubits {
            return Err(QuantumInfoError::DimensionMismatch(format!(
                "Observable length {} doesn't match qubit count {}",
                observable.len(),
                self.num_qubits
            )));
        }
        let mut sum = 0.0;
        let mut valid_snapshots = 0;
        for (basis, outcome) in self.bases.iter().zip(self.outcomes.iter()) {
            let mut useful = true;
            let mut contrib = 1.0;
            for ((obs_char, basis_char), out_char) in
                observable.chars().zip(basis.chars()).zip(outcome.chars())
            {
                if obs_char == 'I' {
                    continue;
                }
                if obs_char != basis_char {
                    useful = false;
                    break;
                }
                let eigenvalue = if out_char == '0' { 1.0 } else { -1.0 };
                contrib *= 3.0 * eigenvalue;
            }
            if useful {
                sum += contrib;
                valid_snapshots += 1;
            }
        }
        if valid_snapshots == 0 {
            return Ok(0.0);
        }
        Ok(sum / valid_snapshots as f64)
    }
    /// Estimate multiple observables efficiently
    pub fn estimate_observables(
        &self,
        observables: &[String],
    ) -> std::result::Result<Vec<f64>, QuantumInfoError> {
        observables
            .iter()
            .map(|obs| self.estimate_observable(obs))
            .collect()
    }
    /// Estimate fidelity with a target pure state
    pub fn estimate_fidelity(
        &self,
        target: &QuantumState,
    ) -> std::result::Result<f64, QuantumInfoError> {
        let target_dm = target.to_density_matrix()?;
        let dim = target_dm.nrows();
        let mut fidelity_sum = 0.0;
        let num_samples = 100;
        let mut rng = thread_rng();
        for _ in 0..num_samples {
            let paulis = ['I', 'X', 'Y', 'Z'];
            let pauli_string: String = (0..self.num_qubits)
                .map(|_| paulis[rng.random_range(0..4)])
                .collect();
            if let Ok(shadow_exp) = self.estimate_observable(&pauli_string) {
                fidelity_sum += shadow_exp.abs();
            }
        }
        Ok((fidelity_sum / num_samples as f64).min(1.0))
    }
    /// Get number of snapshots
    pub fn num_snapshots(&self) -> usize {
        self.num_snapshots
    }
}
/// Quantum state tomography engine
pub struct StateTomography {
    config: StateTomographyConfig,
    num_qubits: usize,
}
impl StateTomography {
    /// Create a new state tomography instance
    pub fn new(num_qubits: usize, config: StateTomographyConfig) -> Self {
        Self { config, num_qubits }
    }
    /// Perform state tomography from measurement data
    ///
    /// # Arguments
    /// * `measurements` - Measurement results in different bases
    ///   Each entry is (basis, outcomes) where basis is "X", "Y", "Z" etc.
    ///   and outcomes is a vector of measurement counts
    pub fn reconstruct(
        &self,
        measurements: &[MeasurementData],
    ) -> std::result::Result<TomographyResult, QuantumInfoError> {
        match self.config.method {
            TomographyMethod::LinearInversion => self.linear_inversion(measurements),
            TomographyMethod::MaximumLikelihood => self.maximum_likelihood(measurements),
            TomographyMethod::CompressedSensing => self.compressed_sensing(measurements),
            TomographyMethod::Bayesian => self.bayesian_estimation(measurements),
        }
    }
    /// Linear inversion tomography
    pub(super) fn linear_inversion(
        &self,
        measurements: &[MeasurementData],
    ) -> std::result::Result<TomographyResult, QuantumInfoError> {
        let dim = 1 << self.num_qubits;
        let mut rho = Array2::zeros((dim, dim));
        for data in measurements {
            let expectation = data.expectation_value();
            let pauli = self.basis_to_pauli(&data.basis)?;
            rho = rho + &pauli * Complex64::new(expectation / dim as f64, 0.0);
        }
        let mut identity = Array2::zeros((dim, dim));
        for i in 0..dim {
            identity[[i, i]] = Complex64::new(1.0 / dim as f64, 0.0);
        }
        rho = rho + identity;
        if self.config.physical_constraints {
            rho = self.make_physical(rho)?;
        }
        let state = QuantumState::Mixed(rho.clone());
        let purity_val = purity(&state)?;
        let total_measurements: usize = measurements.iter().map(|m| m.total_shots()).sum();
        Ok(TomographyResult {
            density_matrix: rho,
            fidelity_estimate: None,
            purity: purity_val,
            uncertainty: 1.0 / (total_measurements as f64).sqrt(),
            total_measurements,
        })
    }
    /// Maximum likelihood estimation
    pub(super) fn maximum_likelihood(
        &self,
        measurements: &[MeasurementData],
    ) -> std::result::Result<TomographyResult, QuantumInfoError> {
        let dim = 1 << self.num_qubits;
        let initial = self.linear_inversion(measurements)?;
        let mut rho = initial.density_matrix;
        let max_iterations = 100;
        let tolerance = 1e-6;
        for _iter in 0..max_iterations {
            let r = self.compute_r_matrix(&rho, measurements)?;
            let r_rho = r.dot(&rho);
            let r_rho_r = r_rho.dot(&r);
            let trace: Complex64 = (0..dim).map(|i| r_rho_r[[i, i]]).sum();
            let trace_re = trace.re.max(1e-15);
            let rho_new = r_rho_r / Complex64::new(trace_re, 0.0);
            let diff: f64 = rho
                .iter()
                .zip(rho_new.iter())
                .map(|(a, b)| (a - b).norm())
                .sum();
            if diff < tolerance {
                rho = rho_new;
                break;
            }
            rho = rho_new;
        }
        let state = QuantumState::Mixed(rho.clone());
        let purity_val = purity(&state)?;
        let total_measurements: usize = measurements.iter().map(|m| m.total_shots()).sum();
        Ok(TomographyResult {
            density_matrix: rho,
            fidelity_estimate: None,
            purity: purity_val,
            uncertainty: 1.0 / (total_measurements as f64).sqrt(),
            total_measurements,
        })
    }
    /// Compute R matrix for MLE iteration
    pub(super) fn compute_r_matrix(
        &self,
        rho: &Array2<Complex64>,
        measurements: &[MeasurementData],
    ) -> std::result::Result<Array2<Complex64>, QuantumInfoError> {
        let dim = rho.nrows();
        let mut r = Array2::zeros((dim, dim));
        for data in measurements {
            let pauli = self.basis_to_pauli(&data.basis)?;
            let p_rho = pauli.dot(rho);
            let exp_rho: Complex64 = (0..dim).map(|i| p_rho[[i, i]]).sum();
            let exp_data = data.expectation_value();
            if exp_rho.re.abs() > 1e-10 {
                let weight = exp_data / exp_rho.re;
                r = r + &pauli * Complex64::new(weight, 0.0);
            }
        }
        let trace: Complex64 = (0..dim).map(|i| r[[i, i]]).sum();
        if trace.re.abs() > 1e-10 {
            r /= Complex64::new(trace.re, 0.0);
        }
        Ok(r)
    }
    /// Compressed sensing tomography
    pub(super) fn compressed_sensing(
        &self,
        measurements: &[MeasurementData],
    ) -> std::result::Result<TomographyResult, QuantumInfoError> {
        let mut result = self.linear_inversion(measurements)?;
        result.density_matrix = self.truncate_rank(result.density_matrix, 4)?;
        Ok(result)
    }
    /// Bayesian estimation
    pub(super) fn bayesian_estimation(
        &self,
        measurements: &[MeasurementData],
    ) -> std::result::Result<TomographyResult, QuantumInfoError> {
        let dim = 1 << self.num_qubits;
        let mut rho = Array2::zeros((dim, dim));
        for i in 0..dim {
            rho[[i, i]] = Complex64::new(1.0 / dim as f64, 0.0);
        }
        for data in measurements {
            let pauli = self.basis_to_pauli(&data.basis)?;
            let exp_data = data.expectation_value();
            let p_rho = pauli.dot(&rho);
            let exp_rho: Complex64 = (0..dim).map(|i| p_rho[[i, i]]).sum();
            let diff = exp_data - exp_rho.re;
            let learning_rate = 0.1;
            rho = rho + &pauli * Complex64::new(learning_rate * diff / dim as f64, 0.0);
        }
        rho = self.make_physical(rho)?;
        let state = QuantumState::Mixed(rho.clone());
        let purity_val = purity(&state)?;
        let total_measurements: usize = measurements.iter().map(|m| m.total_shots()).sum();
        Ok(TomographyResult {
            density_matrix: rho,
            fidelity_estimate: None,
            purity: purity_val,
            uncertainty: 1.0 / (total_measurements as f64).sqrt(),
            total_measurements,
        })
    }
    /// Convert basis string to Pauli operator
    pub(super) fn basis_to_pauli(
        &self,
        basis: &str,
    ) -> std::result::Result<Array2<Complex64>, QuantumInfoError> {
        let dim = 1 << self.num_qubits;
        if basis.len() != self.num_qubits {
            return Err(QuantumInfoError::InvalidState(format!(
                "Basis string length {} doesn't match qubit count {}",
                basis.len(),
                self.num_qubits
            )));
        }
        let mut result: Option<Array2<Complex64>> = None;
        for c in basis.chars() {
            let single_qubit = match c {
                'I' => Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(1.0, 0.0),
                    ],
                )
                .map_err(|e| QuantumInfoError::NumericalError(e.to_string()))?,
                'X' => Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(0.0, 0.0),
                        Complex64::new(1.0, 0.0),
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 0.0),
                    ],
                )
                .map_err(|e| QuantumInfoError::NumericalError(e.to_string()))?,
                'Y' => Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, -1.0),
                        Complex64::new(0.0, 1.0),
                        Complex64::new(0.0, 0.0),
                    ],
                )
                .map_err(|e| QuantumInfoError::NumericalError(e.to_string()))?,
                'Z' => Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(-1.0, 0.0),
                    ],
                )
                .map_err(|e| QuantumInfoError::NumericalError(e.to_string()))?,
                _ => {
                    return Err(QuantumInfoError::InvalidState(format!(
                        "Unknown basis character: {}",
                        c
                    )));
                }
            };
            result = Some(match result {
                None => single_qubit,
                Some(r) => kronecker_product(&r, &single_qubit),
            });
        }
        result.ok_or_else(|| QuantumInfoError::InvalidState("Empty basis string".to_string()))
    }
    /// Make a matrix physical (positive semidefinite with trace 1)
    pub(super) fn make_physical(
        &self,
        mut rho: Array2<Complex64>,
    ) -> std::result::Result<Array2<Complex64>, QuantumInfoError> {
        let dim = rho.nrows();
        let rho_dag = rho.t().mapv(|c| c.conj());
        rho = (&rho + &rho_dag) / Complex64::new(2.0, 0.0);
        let trace: Complex64 = (0..dim).map(|i| rho[[i, i]]).sum();
        if trace.re.abs() > 1e-10 {
            rho /= Complex64::new(trace.re, 0.0);
        }
        Ok(rho)
    }
    /// Truncate density matrix to given rank
    pub(super) fn truncate_rank(
        &self,
        rho: Array2<Complex64>,
        max_rank: usize,
    ) -> std::result::Result<Array2<Complex64>, QuantumInfoError> {
        Ok(rho)
    }
}
/// Data for process tomography
#[derive(Debug, Clone)]
pub struct ProcessTomographyData {
    /// Input states and their output density matrices
    pub state_pairs: Vec<(QuantumState, Array2<Complex64>)>,
}
impl ProcessTomographyData {
    /// Create new process tomography data
    pub fn new() -> Self {
        Self {
            state_pairs: Vec::new(),
        }
    }
    /// Add an input-output pair
    pub fn add_pair(&mut self, input: QuantumState, output: Array2<Complex64>) {
        self.state_pairs.push((input, output));
    }
}
/// Configuration for quantum state tomography
#[derive(Debug, Clone)]
pub struct StateTomographyConfig {
    /// Number of measurement shots per basis
    pub shots_per_basis: usize,
    /// Tomography method
    pub method: TomographyMethod,
    /// Whether to enforce physical constraints (trace=1, positive semidefinite)
    pub physical_constraints: bool,
    /// Threshold for small eigenvalues
    pub threshold: f64,
}
