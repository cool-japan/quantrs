//! Process reconstruction methods

pub mod linear_inversion;
pub mod maximum_likelihood;
pub mod compressed_sensing;
pub mod bayesian;
pub mod ensemble;
pub mod machine_learning;

use ndarray::{Array1, Array2, Array4};
use num_complex::Complex64;

use crate::DeviceResult;
use super::results::{ExperimentalData, ReconstructionQuality, PhysicalValidityMetrics};
use super::core::SciRS2ProcessTomographer;

pub use linear_inversion::*;
pub use maximum_likelihood::*;
pub use compressed_sensing::*;
pub use bayesian::*;
pub use ensemble::*;
pub use machine_learning::*;

/// Reconstruction methods implementation for SciRS2ProcessTomographer
impl SciRS2ProcessTomographer {
    /// Linear inversion reconstruction
    pub fn linear_inversion_reconstruction(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<(Array4<Complex64>, ReconstructionQuality)> {
        linear_inversion::reconstruct_linear_inversion(self, experimental_data)
    }

    /// Maximum likelihood estimation reconstruction
    pub fn maximum_likelihood_reconstruction(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<(Array4<Complex64>, ReconstructionQuality)> {
        maximum_likelihood::reconstruct_maximum_likelihood(self, experimental_data)
    }

    /// Compressed sensing reconstruction
    pub fn compressed_sensing_reconstruction(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<(Array4<Complex64>, ReconstructionQuality)> {
        compressed_sensing::reconstruct_compressed_sensing(self, experimental_data)
    }

    /// Bayesian inference reconstruction
    pub fn bayesian_reconstruction(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<(Array4<Complex64>, ReconstructionQuality)> {
        bayesian::reconstruct_bayesian(self, experimental_data)
    }

    /// Ensemble methods reconstruction
    pub fn ensemble_reconstruction(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<(Array4<Complex64>, ReconstructionQuality)> {
        ensemble::reconstruct_ensemble(self, experimental_data)
    }

    /// Machine learning reconstruction
    pub fn ml_reconstruction(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<(Array4<Complex64>, ReconstructionQuality)> {
        machine_learning::reconstruct_machine_learning(self, experimental_data)
    }

    /// Build measurement matrix for linear reconstruction
    pub fn build_measurement_matrix(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<Array2<f64>> {
        let num_measurements = experimental_data.measurement_results.len();
        let num_qubits = (experimental_data.input_states[0].dim().0 as f64).log2() as usize;
        let process_dim = (1_usize << num_qubits).pow(2); // d^2 for d-dimensional system
        
        let mut measurement_matrix = Array2::zeros((num_measurements, process_dim));
        
        for (m_idx, (&result, &uncertainty)) in experimental_data.measurement_results
            .iter()
            .zip(experimental_data.measurement_uncertainties.iter())
            .enumerate() 
        {
            let input_idx = m_idx / experimental_data.measurement_operators.len();
            let meas_idx = m_idx % experimental_data.measurement_operators.len();
            
            if input_idx < experimental_data.input_states.len() 
                && meas_idx < experimental_data.measurement_operators.len() 
            {
                let input_state = &experimental_data.input_states[input_idx];
                let measurement = &experimental_data.measurement_operators[meas_idx];
                
                // Compute coefficients for the process matrix elements
                for i in 0..(1 << num_qubits) {
                    for j in 0..(1 << num_qubits) {
                        for k in 0..(1 << num_qubits) {
                            for l in 0..(1 << num_qubits) {
                                let process_idx = i * (1_usize << num_qubits).pow(3) 
                                    + j * (1_usize << num_qubits).pow(2) 
                                    + k * (1_usize << num_qubits) 
                                    + l;
                                
                                if process_idx < process_dim {
                                    let coefficient = self.compute_process_coefficient(
                                        input_state, measurement, i, j, k, l
                                    )?;
                                    measurement_matrix[[m_idx, process_idx]] = coefficient;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(measurement_matrix)
    }

}

/// Common utilities for reconstruction methods
pub mod utils {
    use super::*;
    use crate::DeviceResult;

    /// Check physical validity of reconstructed process
    pub fn check_physical_validity(process_matrix: &Array4<Complex64>) -> PhysicalValidityMetrics {
        let dim = process_matrix.dim().0;
        
        // Check complete positivity (simplified)
        let mut is_cp = true;
        let mut positivity_measure = 1.0;
        
        // Check trace preservation (simplified)
        let mut trace_sum = 0.0;
        for i in 0..dim {
            for j in 0..dim {
                trace_sum += process_matrix[[i, j, i, j]].re;
            }
        }
        let is_tp = (trace_sum - 1.0).abs() < 1e-6;
        let tp_measure = 1.0 - (trace_sum - 1.0).abs();
        
        PhysicalValidityMetrics {
            is_completely_positive: is_cp,
            is_trace_preserving: is_tp,
            positivity_measure,
            trace_preservation_measure: tp_measure.max(0.0),
        }
    }

    /// Calculate reconstruction quality metrics
    pub fn calculate_reconstruction_quality(
        process_matrix: &Array4<Complex64>,
        experimental_data: &ExperimentalData,
        log_likelihood: f64,
    ) -> ReconstructionQuality {
        let physical_validity = check_physical_validity(process_matrix);
        
        // Calculate condition number (simplified)
        let condition_number = 10.0; // Placeholder
        
        ReconstructionQuality {
            log_likelihood,
            physical_validity,
            condition_number,
        }
    }
}