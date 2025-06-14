//! Utility functions for process tomography

use std::collections::HashMap;
use ndarray::{Array1, Array2, Array4};
use num_complex::Complex64;

use crate::DeviceResult;
use super::results::*;
use super::core::SciRS2ProcessTomographer;

impl SciRS2ProcessTomographer {
    /// Collect experimental data from device
    pub async fn collect_experimental_data<const N: usize, E: super::core::ProcessTomographyExecutor>(
        &self,
        process_circuit: &quantrs2_circuit::prelude::Circuit<N>,
        executor: &E,
    ) -> DeviceResult<ExperimentalData> {
        let mut measurement_results = Vec::new();
        let mut measurement_uncertainties = Vec::new();
        
        // Execute measurements for all input state and measurement operator combinations
        for input_state in &self.input_states {
            for measurement_op in &self.measurement_operators {
                let result = executor.execute_process_measurement(
                    process_circuit,
                    input_state,
                    measurement_op,
                    self.config.shots_per_state,
                ).await?;
                
                measurement_results.push(result);
                
                // Estimate measurement uncertainty based on Poisson statistics
                let uncertainty = if result > 0.0 {
                    (result / self.config.shots_per_state as f64).sqrt()
                } else {
                    1.0 / (self.config.shots_per_state as f64).sqrt()
                };
                measurement_uncertainties.push(uncertainty);
            }
        }
        
        Ok(ExperimentalData {
            input_states: self.input_states.clone(),
            measurement_operators: self.measurement_operators.clone(),
            measurement_results,
            measurement_uncertainties,
        })
    }

    /// Create ideal identity Choi matrix
    pub(crate) fn create_ideal_identity_choi(&self, dim: usize) -> DeviceResult<Array2<Complex64>> {
        let choi_dim = dim;
        let mut identity_choi = Array2::zeros((choi_dim, choi_dim));
        
        // Identity channel Choi matrix is the maximally entangled state
        for i in 0..choi_dim {
            identity_choi[[i, i]] = Complex64::new(1.0, 0.0);
        }
        
        Ok(identity_choi)
    }

    /// Calculate Choi fidelity between two Choi matrices
    pub(crate) fn calculate_choi_fidelity(
        &self,
        choi1: &Array2<Complex64>,
        choi2: &Array2<Complex64>,
    ) -> DeviceResult<f64> {
        let dim = choi1.nrows();
        let mut fidelity = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;
        
        for i in 0..dim {
            for j in 0..dim {
                let element1 = choi1[[i, j]];
                let element2 = choi2[[i, j]];
                
                fidelity += (element1.conj() * element2).re;
                norm1 += element1.norm_sqr();
                norm2 += element2.norm_sqr();
            }
        }
        
        if norm1 > 1e-12 && norm2 > 1e-12 {
            Ok(fidelity / (norm1 * norm2).sqrt())
        } else {
            Ok(0.0)
        }
    }

    /// Calculate partial transpose of a matrix
    pub(crate) fn partial_transpose(&self, matrix: &Array2<Complex64>) -> DeviceResult<Array2<f64>> {
        let dim = matrix.nrows();
        let sqrt_dim = (dim as f64).sqrt() as usize;
        
        if sqrt_dim * sqrt_dim != dim {
            return Ok(Array2::zeros((dim, dim)));
        }
        
        let mut pt_matrix = Array2::zeros((dim, dim));
        
        // Partial transpose operation
        for i in 0..sqrt_dim {
            for j in 0..sqrt_dim {
                for k in 0..sqrt_dim {
                    for l in 0..sqrt_dim {
                        let row1 = i * sqrt_dim + j;
                        let col1 = k * sqrt_dim + l;
                        let row2 = i * sqrt_dim + l;
                        let col2 = k * sqrt_dim + j;
                        
                        if row1 < dim && col1 < dim && row2 < dim && col2 < dim {
                            pt_matrix[[row2, col2]] = matrix[[row1, col1]].re;
                        }
                    }
                }
            }
        }
        
        Ok(pt_matrix)
    }

    /// Calculate process metrics from process matrix
    pub fn calculate_process_metrics(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<ProcessMetrics> {
        let process_fidelity = self.calculate_process_fidelity(process_matrix)?;
        let average_gate_fidelity = self.calculate_average_gate_fidelity(process_matrix)?;
        let unitarity = self.calculate_unitarity(process_matrix)?;
        let entangling_power = self.calculate_entangling_power(process_matrix)?;
        let non_unitality = self.calculate_non_unitality(process_matrix)?;
        let channel_capacity = self.calculate_channel_capacity(process_matrix)?;
        let coherent_information = self.calculate_coherent_information(process_matrix)?;
        let diamond_norm_distance = self.calculate_diamond_norm_distance(process_matrix)?;
        let process_spectrum = self.calculate_process_spectrum(process_matrix)?;
        
        Ok(ProcessMetrics {
            process_fidelity,
            average_gate_fidelity,
            unitarity,
            entangling_power,
            non_unitality,
            channel_capacity,
            coherent_information,
            diamond_norm_distance,
            process_spectrum,
        })
    }

    /// Calculate average gate fidelity
    fn calculate_average_gate_fidelity(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<f64> {
        let dim = process_matrix.dim().0;
        
        // AGF = (d * process_fidelity + 1) / (d + 1) where d is the dimension
        let process_fidelity = self.calculate_process_fidelity(process_matrix)?;
        let agf = (dim as f64 * process_fidelity + 1.0) / (dim as f64 + 1.0);
        
        Ok(agf)
    }

    /// Calculate unitarity measure
    fn calculate_unitarity(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<f64> {
        let dim = process_matrix.dim().0;
        let mut unitarity = 0.0;
        
        // Unitarity is the trace of chi^2 where chi is the process matrix
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        for m in 0..dim {
                            for n in 0..dim {
                                unitarity += (process_matrix[[i, j, k, l]].conj() * 
                                            process_matrix[[i, j, m, n]] * 
                                            process_matrix[[m, n, k, l]]).re;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(unitarity / (dim * dim) as f64)
    }

    /// Calculate non-unitality measure
    fn calculate_non_unitality(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<f64> {
        let dim = process_matrix.dim().0;
        let mut non_unitality = 0.0;
        
        // Non-unitality measures deviation from unital channels
        for i in 0..dim {
            for j in 0..dim {
                if i != j {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for k in 0..dim {
                        sum += process_matrix[[i, j, k, k]];
                    }
                    non_unitality += sum.norm_sqr();
                }
            }
        }
        
        Ok(non_unitality / (dim * dim) as f64)
    }

    /// Calculate channel capacity
    fn calculate_channel_capacity(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<f64> {
        // Simplified channel capacity calculation
        // In practice, this would involve optimization over input states
        let unitarity = self.calculate_unitarity(process_matrix)?;
        let capacity = unitarity * (process_matrix.dim().0 as f64).log2();
        
        Ok(capacity)
    }

    /// Calculate coherent information
    fn calculate_coherent_information(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<f64> {
        // Simplified coherent information calculation
        let unitarity = self.calculate_unitarity(process_matrix)?;
        let coherent_info = unitarity * 0.8; // Simplified approximation
        
        Ok(coherent_info)
    }

    /// Calculate diamond norm distance
    fn calculate_diamond_norm_distance(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<f64> {
        // Simplified diamond norm calculation
        // In practice, this would involve semidefinite programming
        let process_fidelity = self.calculate_process_fidelity(process_matrix)?;
        let diamond_distance = 2.0 * (1.0 - process_fidelity).sqrt();
        
        Ok(diamond_distance)
    }

    /// Calculate process spectrum (eigenvalues)
    fn calculate_process_spectrum(&self, process_matrix: &Array4<Complex64>) -> DeviceResult<Array1<f64>> {
        let choi_matrix = self.convert_to_choi_matrix(process_matrix)?;
        
        #[cfg(feature = "scirs2")]
        {
            use scirs2_linalg::eig;
            
            let real_choi = choi_matrix.mapv(|x| x.re);
            if let Ok((eigenvalues, _)) = eig(&real_choi.view()) {
                let spectrum = eigenvalues.mapv(|x| x.re);
                return Ok(spectrum);
            }
        }
        
        // Fallback: return uniform spectrum
        let dim = choi_matrix.nrows();
        Ok(Array1::from_elem(dim, 1.0 / dim as f64))
    }
}

/// Process tomography utility functions
pub mod process_utils {
    use super::*;
    
    /// Reshape process matrix for different representations
    pub fn reshape_process_matrix(
        process_matrix: &Array4<Complex64>,
        target_shape: (usize, usize),
    ) -> DeviceResult<Array2<Complex64>> {
        let (dim1, dim2, dim3, dim4) = process_matrix.dim();
        let total_elements = dim1 * dim2 * dim3 * dim4;
        
        if target_shape.0 * target_shape.1 != total_elements {
            return Err(crate::DeviceError::APIError(
                "Target shape incompatible with process matrix size".to_string()
            ));
        }
        
        let mut reshaped = Array2::zeros(target_shape);
        let mut idx = 0;
        
        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    for l in 0..dim4 {
                        let row = idx / target_shape.1;
                        let col = idx % target_shape.1;
                        
                        if row < target_shape.0 && col < target_shape.1 {
                            reshaped[[row, col]] = process_matrix[[i, j, k, l]];
                        }
                        idx += 1;
                    }
                }
            }
        }
        
        Ok(reshaped)
    }

    /// Vectorize process matrix
    pub fn vectorize_process_matrix(process_matrix: &Array4<Complex64>) -> Array1<Complex64> {
        let (dim1, dim2, dim3, dim4) = process_matrix.dim();
        let total_elements = dim1 * dim2 * dim3 * dim4;
        let mut vector = Array1::zeros(total_elements);
        
        let mut idx = 0;
        for i in 0..dim1 {
            for j in 0..dim2 {
                for k in 0..dim3 {
                    for l in 0..dim4 {
                        vector[idx] = process_matrix[[i, j, k, l]];
                        idx += 1;
                    }
                }
            }
        }
        
        vector
    }

    /// Convert vector back to process matrix
    pub fn devectorize_to_process_matrix(
        vector: &Array1<Complex64>,
        dim: usize,
    ) -> DeviceResult<Array4<Complex64>> {
        let expected_length = dim.pow(4);
        if vector.len() != expected_length {
            return Err(crate::DeviceError::APIError(
                format!("Vector length {} does not match expected length {} for dimension {}", 
                       vector.len(), expected_length, dim)
            ));
        }
        
        let mut process_matrix = Array4::zeros((dim, dim, dim, dim));
        let mut idx = 0;
        
        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        process_matrix[[i, j, k, l]] = vector[idx];
                        idx += 1;
                    }
                }
            }
        }
        
        Ok(process_matrix)
    }

    /// Calculate trace distance between two process matrices
    pub fn trace_distance(
        process1: &Array4<Complex64>,
        process2: &Array4<Complex64>,
    ) -> DeviceResult<f64> {
        if process1.dim() != process2.dim() {
            return Err(crate::DeviceError::APIError(
                "Process matrices must have the same dimensions".to_string()
            ));
        }
        
        let dim = process1.dim();
        let mut trace_distance = 0.0;
        
        for i in 0..dim.0 {
            for j in 0..dim.1 {
                for k in 0..dim.2 {
                    for l in 0..dim.3 {
                        let diff = process1[[i, j, k, l]] - process2[[i, j, k, l]];
                        trace_distance += diff.norm();
                    }
                }
            }
        }
        
        Ok(trace_distance / 2.0)
    }

    /// Check if process matrix satisfies trace preservation
    pub fn check_trace_preservation(process_matrix: &Array4<Complex64>, tolerance: f64) -> bool {
        let dim = process_matrix.dim().0;
        let mut trace = Complex64::new(0.0, 0.0);
        
        for i in 0..dim {
            for j in 0..dim {
                trace += process_matrix[[i, j, i, j]];
            }
        }
        
        (trace.re - 1.0).abs() < tolerance && trace.im.abs() < tolerance
    }

    /// Check if process matrix satisfies complete positivity (simplified)
    pub fn check_complete_positivity(process_matrix: &Array4<Complex64>, tolerance: f64) -> bool {
        let dim = process_matrix.dim().0;
        
        // Simplified check: all diagonal elements should be non-negative
        for i in 0..dim {
            for j in 0..dim {
                if process_matrix[[i, j, i, j]].re < -tolerance {
                    return false;
                }
            }
        }
        
        true
    }

    /// Normalize process matrix to satisfy trace preservation
    pub fn normalize_process_matrix(process_matrix: &mut Array4<Complex64>) {
        let dim = process_matrix.dim().0;
        let mut trace = Complex64::new(0.0, 0.0);
        
        // Calculate current trace
        for i in 0..dim {
            for j in 0..dim {
                trace += process_matrix[[i, j, i, j]];
            }
        }
        
        // Normalize if trace is non-zero
        if trace.norm() > 1e-12 {
            let scale_factor = Complex64::new(1.0, 0.0) / trace;
            for i in 0..dim {
                for j in 0..dim {
                    for k in 0..dim {
                        for l in 0..dim {
                            process_matrix[[i, j, k, l]] *= scale_factor;
                        }
                    }
                }
            }
        }
    }
}