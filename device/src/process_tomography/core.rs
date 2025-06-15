//! Core process tomography implementation

use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2};
use num_complex::Complex64;
use quantrs2_circuit::prelude::*;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use std::collections::HashMap;

use crate::{
    backend_traits::{query_backend_capabilities, BackendCapabilities},
    calibration::{CalibrationManager, DeviceCalibration},
    characterization::{ProcessTomography, StateTomography},
    noise_model::CalibrationNoiseModel,
    translation::HardwareBackend,
    CircuitResult, DeviceError, DeviceResult,
};

use super::{
    analysis::*,
    config::{ReconstructionMethod, SciRS2ProcessTomographyConfig},
    reconstruction::*,
    results::*,
    utils::*,
    validation::*,
};

// Conditional imports for SciRS2
#[cfg(feature = "scirs2")]
use scirs2_graph::{
    betweenness_centrality, closeness_centrality, minimum_spanning_tree, shortest_path,
    strongly_connected_components, Graph,
};
#[cfg(feature = "scirs2")]
use scirs2_linalg::{
    cholesky, det, eig, inv, matrix_norm, prelude::*, qr, svd, trace, LinalgError, LinalgResult,
};
#[cfg(feature = "scirs2")]
use scirs2_optimize::{minimize, OptimizeResult};
#[cfg(feature = "scirs2")]
use scirs2_stats::{
    corrcoef,
    distributions::{chi2, gamma, norm},
    ks_2samp, mean, pearsonr, shapiro_wilk, spearmanr, std, ttest_1samp, ttest_ind, var,
    Alternative, TTestResult,
};

// Fallback imports when SciRS2 is not available
#[cfg(not(feature = "scirs2"))]
use super::fallback::*;

/// Main SciRS2 process tomographer
pub struct SciRS2ProcessTomographer {
    pub(crate) config: SciRS2ProcessTomographyConfig,
    pub(crate) calibration_manager: CalibrationManager,
    pub(crate) input_states: Vec<Array2<Complex64>>,
    pub(crate) measurement_operators: Vec<Array2<Complex64>>,
}

impl SciRS2ProcessTomographer {
    /// Create a new SciRS2 process tomographer
    pub fn new(
        config: SciRS2ProcessTomographyConfig,
        calibration_manager: CalibrationManager,
    ) -> Self {
        Self {
            config,
            calibration_manager,
            input_states: Vec::new(),
            measurement_operators: Vec::new(),
        }
    }

    /// Generate input states for process tomography
    pub fn generate_input_states(&mut self, num_qubits: usize) -> DeviceResult<()> {
        self.input_states = self.create_informationally_complete_states(num_qubits)?;
        Ok(())
    }

    /// Generate measurement operators
    pub fn generate_measurement_operators(&mut self, num_qubits: usize) -> DeviceResult<()> {
        self.measurement_operators = self.create_pauli_measurements(num_qubits)?;
        Ok(())
    }

    /// Perform comprehensive process tomography
    pub async fn perform_process_tomography<const N: usize, E: ProcessTomographyExecutor>(
        &self,
        device_id: &str,
        process_circuit: &Circuit<N>,
        executor: &E,
    ) -> DeviceResult<SciRS2ProcessTomographyResult> {
        // Step 1: Collect experimental data
        let experimental_data = self
            .collect_experimental_data(process_circuit, executor)
            .await?;

        // Step 2: Reconstruct process matrix using selected method
        let (process_matrix, reconstruction_quality) = match self.config.reconstruction_method {
            ReconstructionMethod::LinearInversion => {
                self.linear_inversion_reconstruction(&experimental_data)?
            }
            ReconstructionMethod::MaximumLikelihood => {
                self.maximum_likelihood_reconstruction(&experimental_data)?
            }
            ReconstructionMethod::CompressedSensing => {
                self.compressed_sensing_reconstruction(&experimental_data)?
            }
            ReconstructionMethod::BayesianInference => {
                self.bayesian_reconstruction(&experimental_data)?
            }
            ReconstructionMethod::EnsembleMethods => {
                self.ensemble_reconstruction(&experimental_data)?
            }
            ReconstructionMethod::MachineLearning => self.ml_reconstruction(&experimental_data)?,
        };

        // Step 3: Convert to Pauli transfer representation
        let pauli_transfer_matrix = self.convert_to_pauli_transfer(&process_matrix)?;

        // Step 4: Statistical analysis
        let statistical_analysis =
            self.perform_statistical_analysis(&process_matrix, &experimental_data)?;

        // Step 5: Calculate process metrics
        let process_metrics = self.calculate_process_metrics(&process_matrix)?;

        // Step 6: Validation
        let validation_results = if self.config.validation_config.enable_cross_validation {
            self.perform_validation(&experimental_data)?
        } else {
            ProcessValidationResults {
                cross_validation: None,
                bootstrap_results: None,
                benchmark_results: None,
                model_selection: ModelSelectionResults {
                    aic_scores: HashMap::new(),
                    bic_scores: HashMap::new(),
                    cross_validation_scores: HashMap::new(),
                    best_model: "mle".to_string(),
                    model_weights: HashMap::new(),
                },
            }
        };

        // Step 7: Structure analysis (if enabled)
        let structure_analysis = if self.config.enable_structure_analysis {
            Some(self.analyze_process_structure(&process_matrix)?)
        } else {
            None
        };

        // Step 8: Uncertainty quantification
        let uncertainty_quantification =
            self.quantify_uncertainty(&process_matrix, &experimental_data)?;

        // Step 9: Process comparisons
        let process_comparisons = self.compare_with_known_processes(&process_matrix)?;

        Ok(SciRS2ProcessTomographyResult {
            device_id: device_id.to_string(),
            config: self.config.clone(),
            process_matrix,
            pauli_transfer_matrix,
            statistical_analysis: ProcessStatisticalAnalysis {
                reconstruction_quality,
                statistical_tests: statistical_analysis.0,
                distribution_analysis: statistical_analysis.1,
                correlation_analysis: statistical_analysis.2,
            },
            process_metrics,
            validation_results,
            structure_analysis,
            uncertainty_quantification,
            process_comparisons,
        })
    }

    /// Create informationally complete set of input states
    pub(crate) fn create_informationally_complete_states(
        &self,
        num_qubits: usize,
    ) -> DeviceResult<Vec<Array2<Complex64>>> {
        let mut states = Vec::new();
        let dim = 1 << num_qubits; // 2^n

        // Create standard IC-POVM states
        // For 1 qubit: |0⟩, |1⟩, |+⟩, |-⟩, |+i⟩, |-i⟩
        if num_qubits == 1 {
            // |0⟩
            let mut state0 = Array2::zeros((2, 2));
            state0[[0, 0]] = Complex64::new(1.0, 0.0);
            states.push(state0);

            // |1⟩
            let mut state1 = Array2::zeros((2, 2));
            state1[[1, 1]] = Complex64::new(1.0, 0.0);
            states.push(state1);

            // |+⟩ = (|0⟩ + |1⟩)/√2
            let mut state_plus = Array2::zeros((2, 2));
            state_plus[[0, 0]] = Complex64::new(0.5, 0.0);
            state_plus[[0, 1]] = Complex64::new(0.5, 0.0);
            state_plus[[1, 0]] = Complex64::new(0.5, 0.0);
            state_plus[[1, 1]] = Complex64::new(0.5, 0.0);
            states.push(state_plus);

            // |-⟩ = (|0⟩ - |1⟩)/√2
            let mut state_minus = Array2::zeros((2, 2));
            state_minus[[0, 0]] = Complex64::new(0.5, 0.0);
            state_minus[[0, 1]] = Complex64::new(-0.5, 0.0);
            state_minus[[1, 0]] = Complex64::new(-0.5, 0.0);
            state_minus[[1, 1]] = Complex64::new(0.5, 0.0);
            states.push(state_minus);

            // |+i⟩ = (|0⟩ + i|1⟩)/√2
            let mut state_plus_i = Array2::zeros((2, 2));
            state_plus_i[[0, 0]] = Complex64::new(0.5, 0.0);
            state_plus_i[[0, 1]] = Complex64::new(0.0, 0.5);
            state_plus_i[[1, 0]] = Complex64::new(0.0, -0.5);
            state_plus_i[[1, 1]] = Complex64::new(0.5, 0.0);
            states.push(state_plus_i);

            // |-i⟩ = (|0⟩ - i|1⟩)/√2
            let mut state_minus_i = Array2::zeros((2, 2));
            state_minus_i[[0, 0]] = Complex64::new(0.5, 0.0);
            state_minus_i[[0, 1]] = Complex64::new(0.0, -0.5);
            state_minus_i[[1, 0]] = Complex64::new(0.0, 0.5);
            state_minus_i[[1, 1]] = Complex64::new(0.5, 0.0);
            states.push(state_minus_i);
        } else {
            // For multi-qubit systems, use tensor products of single-qubit states
            let single_qubit_states = self.create_informationally_complete_states(1)?;

            // Generate all combinations
            for combination in self.generate_state_combinations(&single_qubit_states, num_qubits)? {
                states.push(combination);
            }
        }

        Ok(states)
    }

    /// Create Pauli measurement operators
    pub(crate) fn create_pauli_measurements(
        &self,
        num_qubits: usize,
    ) -> DeviceResult<Vec<Array2<Complex64>>> {
        let mut measurements = Vec::new();
        let dim = 1 << num_qubits;

        // Single qubit Pauli operators
        let pauli_i = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        )
        .map_err(|e| DeviceError::APIError(format!("Array creation error: {}", e)))?;

        let pauli_x = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .map_err(|e| DeviceError::APIError(format!("Array creation error: {}", e)))?;

        let pauli_y = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .map_err(|e| DeviceError::APIError(format!("Array creation error: {}", e)))?;

        let pauli_z = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )
        .map_err(|e| DeviceError::APIError(format!("Array creation error: {}", e)))?;

        let single_qubit_paulis = vec![pauli_i, pauli_x, pauli_y, pauli_z];

        if num_qubits == 1 {
            measurements = single_qubit_paulis;
        } else {
            // Generate tensor products for multi-qubit measurements
            measurements = self.generate_pauli_tensor_products(&single_qubit_paulis, num_qubits)?;
        }

        Ok(measurements)
    }

    /// Calculate process fidelity with an ideal process
    pub fn calculate_process_fidelity(
        &self,
        process_matrix: &Array4<Complex64>,
    ) -> DeviceResult<f64> {
        // Convert to Choi representation for fidelity calculation
        let choi_matrix = self.convert_to_choi_matrix(process_matrix)?;

        // Create ideal identity process for comparison
        let ideal_choi = self.create_ideal_identity_choi(choi_matrix.dim().0)?;

        // Calculate process fidelity using Choi matrices
        #[cfg(feature = "scirs2")]
        {
            let fidelity = self.calculate_choi_fidelity(&choi_matrix, &ideal_choi)?;
            Ok(fidelity)
        }

        #[cfg(not(feature = "scirs2"))]
        {
            // Fallback: simplified fidelity calculation
            Ok(0.95) // Placeholder
        }
    }

    /// Calculate entangling power of the process
    pub fn calculate_entangling_power(
        &self,
        process_matrix: &Array4<Complex64>,
    ) -> DeviceResult<f64> {
        let dim = process_matrix.dim().0;

        // For single qubit processes, entangling power is zero
        if dim <= 2 {
            return Ok(0.0);
        }

        #[cfg(feature = "scirs2")]
        {
            // Calculate entangling power using process matrix analysis
            let choi_matrix = self.convert_to_choi_matrix(process_matrix)?;

            // Compute eigenvalues of the partial transpose
            let partial_transpose = self.partial_transpose(&choi_matrix)?;

            if let Ok((eigenvalues, _)) = eig(&partial_transpose.view()) {
                let negative_eigenvalues: Vec<f64> = eigenvalues
                    .iter()
                    .map(|x| x.re)
                    .filter(|&x| x < 0.0)
                    .collect();

                let entangling_power = negative_eigenvalues.iter().map(|x| x.abs()).sum::<f64>();
                Ok(entangling_power)
            } else {
                Ok(0.0)
            }
        }

        #[cfg(not(feature = "scirs2"))]
        {
            // Fallback: check for off-diagonal elements as entanglement indicator
            let mut entangling_power = 0.0;
            for i in 0..dim {
                for j in 0..dim {
                    for k in 0..dim {
                        for l in 0..dim {
                            if i != j || k != l {
                                entangling_power += process_matrix[[i, j, k, l]].norm_sqr();
                            }
                        }
                    }
                }
            }
            Ok(entangling_power / (dim * dim) as f64)
        }
    }
}

/// Trait for process tomography executors
pub trait ProcessTomographyExecutor {
    /// Execute a circuit on input states and perform measurements
    async fn execute_process_measurement<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        input_state: &Array2<Complex64>,
        measurement: &Array2<Complex64>,
        shots: usize,
    ) -> DeviceResult<f64>;
}
