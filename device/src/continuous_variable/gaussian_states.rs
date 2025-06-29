//! Gaussian states for continuous variable quantum computing
//!
//! This module implements Gaussian states, which are the foundation of CV quantum computing.
//! Gaussian states can be fully characterized by their first and second moments.

use super::{CVDeviceConfig, CVEntanglementMeasures, CVModeState, Complex};
use crate::{DeviceError, DeviceResult};
use rand::{random, rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Gaussian state representation using covariance matrix formalism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianState {
    /// Number of modes
    pub num_modes: usize,
    /// Mean vector (displacement amplitudes) - [x1, p1, x2, p2, ...]
    pub mean_vector: Vec<f64>,
    /// Covariance matrix (2N x 2N where N is number of modes)
    pub covariance_matrix: Vec<Vec<f64>>,
    /// Symplectic matrix for canonical commutation relations
    symplectic_matrix: Vec<Vec<f64>>,
}

impl GaussianState {
    /// Create vacuum state for N modes
    pub fn vacuum_state(num_modes: usize) -> Self {
        let vector_size = 2 * num_modes;
        let mean_vector = vec![0.0; vector_size];

        // Vacuum covariance matrix: I/2 (identity matrix scaled by 1/2)
        let mut covariance_matrix = vec![vec![0.0; vector_size]; vector_size];
        for i in 0..vector_size {
            covariance_matrix[i][i] = 0.5;
        }

        let symplectic_matrix = Self::build_symplectic_matrix(num_modes);

        Self {
            num_modes,
            mean_vector,
            covariance_matrix,
            symplectic_matrix,
        }
    }

    /// Create coherent state with given displacement
    pub fn coherent_state(num_modes: usize, displacements: Vec<Complex>) -> DeviceResult<Self> {
        if displacements.len() != num_modes {
            return Err(DeviceError::InvalidInput(
                "Number of displacements must match number of modes".to_string(),
            ));
        }

        let mut state = Self::vacuum_state(num_modes);

        // Set mean vector from displacements
        for (i, displacement) in displacements.iter().enumerate() {
            state.mean_vector[2 * i] = displacement.real * (2.0_f64).sqrt(); // x quadrature
            state.mean_vector[2 * i + 1] = displacement.imag * (2.0_f64).sqrt();
            // p quadrature
        }

        Ok(state)
    }

    /// Create squeezed vacuum state
    pub fn squeezed_vacuum_state(
        num_modes: usize,
        squeezing_params: Vec<f64>,
        squeezing_phases: Vec<f64>,
    ) -> DeviceResult<Self> {
        if squeezing_params.len() != num_modes || squeezing_phases.len() != num_modes {
            return Err(DeviceError::InvalidInput(
                "Squeezing parameters and phases must match number of modes".to_string(),
            ));
        }

        let mut state = Self::vacuum_state(num_modes);

        // Apply squeezing to each mode
        for i in 0..num_modes {
            state.apply_squeezing(i, squeezing_params[i], squeezing_phases[i])?;
        }

        Ok(state)
    }

    /// Build symplectic matrix for canonical commutation relations
    fn build_symplectic_matrix(num_modes: usize) -> Vec<Vec<f64>> {
        let size = 2 * num_modes;
        let mut omega = vec![vec![0.0; size]; size];

        for i in 0..num_modes {
            omega[2 * i][2 * i + 1] = 1.0;
            omega[2 * i + 1][2 * i] = -1.0;
        }

        omega
    }

    /// Apply displacement operation to a mode
    pub fn apply_displacement(&mut self, mode: usize, displacement: Complex) -> DeviceResult<()> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes",
                mode
            )));
        }

        // Update mean vector
        self.mean_vector[2 * mode] += displacement.real * (2.0_f64).sqrt();
        self.mean_vector[2 * mode + 1] += displacement.imag * (2.0_f64).sqrt();

        Ok(())
    }

    /// Apply squeezing operation to a mode
    pub fn apply_squeezing(&mut self, mode: usize, r: f64, phi: f64) -> DeviceResult<()> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes",
                mode
            )));
        }

        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        let cosh_r = r.cosh();
        let sinh_r = r.sinh();

        // Squeezing transformation matrix
        // Standard squeezing: S = [[e^(-r), 0], [0, e^r]] for φ=0
        let s11 = cosh_r - sinh_r * cos_phi; // For φ=0: e^(-r) (squeeze position)
        let s12 = -sinh_r * sin_phi; // For φ=0: 0
        let s21 = -sinh_r * sin_phi; // For φ=0: 0
        let s22 = cosh_r + sinh_r * cos_phi; // For φ=0: e^r (anti-squeeze momentum)

        // Apply to covariance matrix
        let i = 2 * mode;
        let j = 2 * mode + 1;

        let old_covar = self.covariance_matrix.clone();

        // Transform covariance matrix: S * V * S^T
        for a in 0..2 * self.num_modes {
            for b in 0..2 * self.num_modes {
                if (a == i || a == j) && (b == i || b == j) {
                    let mut new_val = 0.0;

                    for k in [i, j].iter() {
                        for l in [i, j].iter() {
                            let s_ak = if a == i {
                                if *k == i {
                                    s11
                                } else {
                                    s12
                                }
                            } else {
                                if *k == i {
                                    s21
                                } else {
                                    s22
                                }
                            };

                            let s_bl = if b == i {
                                if *l == i {
                                    s11
                                } else {
                                    s12
                                }
                            } else {
                                if *l == i {
                                    s21
                                } else {
                                    s22
                                }
                            };

                            new_val += s_ak * old_covar[*k][*l] * s_bl;
                        }
                    }

                    self.covariance_matrix[a][b] = new_val;
                } else if a == i || a == j {
                    // Mixed terms
                    let s_a = if a == i { [s11, s12] } else { [s21, s22] };
                    self.covariance_matrix[a][b] =
                        s_a[0] * old_covar[i][b] + s_a[1] * old_covar[j][b];
                } else if b == i || b == j {
                    // Mixed terms (transpose)
                    let s_b = if b == i { [s11, s21] } else { [s12, s22] };
                    self.covariance_matrix[a][b] =
                        old_covar[a][i] * s_b[0] + old_covar[a][j] * s_b[1];
                }
            }
        }

        Ok(())
    }

    /// Apply two-mode squeezing operation
    pub fn apply_two_mode_squeezing(
        &mut self,
        mode1: usize,
        mode2: usize,
        r: f64,
        phi: f64,
    ) -> DeviceResult<()> {
        if mode1 >= self.num_modes || mode2 >= self.num_modes {
            return Err(DeviceError::InvalidInput(
                "One or both modes exceed available modes".to_string(),
            ));
        }

        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        let cosh_r = r.cosh();
        let sinh_r = r.sinh();

        // Two-mode squeezing transformation
        let indices = [2 * mode1, 2 * mode1 + 1, 2 * mode2, 2 * mode2 + 1];
        let old_covar = self.covariance_matrix.clone();

        // Build 4x4 transformation matrix
        let mut transform = [[0.0; 4]; 4];
        transform[0][0] = cosh_r;
        transform[0][2] = sinh_r * cos_phi;
        transform[0][3] = sinh_r * sin_phi;
        transform[1][1] = cosh_r;
        transform[1][2] = sinh_r * sin_phi;
        transform[1][3] = -sinh_r * cos_phi;
        transform[2][0] = sinh_r * cos_phi;
        transform[2][1] = sinh_r * sin_phi;
        transform[2][2] = cosh_r;
        transform[3][0] = sinh_r * sin_phi;
        transform[3][1] = -sinh_r * cos_phi;
        transform[3][3] = cosh_r;

        // Apply transformation to relevant block of covariance matrix
        for i in 0..4 {
            for j in 0..4 {
                let mut new_val = 0.0;
                for k in 0..4 {
                    for l in 0..4 {
                        new_val +=
                            transform[i][k] * old_covar[indices[k]][indices[l]] * transform[j][l];
                    }
                }
                self.covariance_matrix[indices[i]][indices[j]] = new_val;
            }
        }

        Ok(())
    }

    /// Apply beamsplitter operation
    pub fn apply_beamsplitter(
        &mut self,
        mode1: usize,
        mode2: usize,
        transmittance: f64,
        phase: f64,
    ) -> DeviceResult<()> {
        if mode1 >= self.num_modes || mode2 >= self.num_modes {
            return Err(DeviceError::InvalidInput(
                "One or both modes exceed available modes".to_string(),
            ));
        }

        let t = transmittance.sqrt();
        let r = (1.0 - transmittance).sqrt();
        let cos_phi = phase.cos();
        let sin_phi = phase.sin();

        // Beamsplitter transformation matrix
        let indices = [2 * mode1, 2 * mode1 + 1, 2 * mode2, 2 * mode2 + 1];
        let old_mean = self.mean_vector.clone();
        let old_covar = self.covariance_matrix.clone();

        // Transform mean vector
        let mean1_x = old_mean[2 * mode1];
        let mean1_p = old_mean[2 * mode1 + 1];
        let mean2_x = old_mean[2 * mode2];
        let mean2_p = old_mean[2 * mode2 + 1];

        self.mean_vector[2 * mode1] = t * mean1_x + r * cos_phi * mean2_x - r * sin_phi * mean2_p;
        self.mean_vector[2 * mode1 + 1] =
            t * mean1_p + r * sin_phi * mean2_x + r * cos_phi * mean2_p;
        self.mean_vector[2 * mode2] = -r * cos_phi * mean1_x + r * sin_phi * mean1_p + t * mean2_x;
        self.mean_vector[2 * mode2 + 1] =
            r * sin_phi * mean1_x + r * cos_phi * mean1_p + t * mean2_p;

        // Build 4x4 transformation matrix
        let mut transform = [[0.0; 4]; 4];
        transform[0][0] = t;
        transform[0][2] = r * cos_phi;
        transform[0][3] = -r * sin_phi;
        transform[1][1] = t;
        transform[1][2] = r * sin_phi;
        transform[1][3] = r * cos_phi;
        transform[2][0] = -r * cos_phi;
        transform[2][1] = r * sin_phi;
        transform[2][2] = t;
        transform[3][0] = r * sin_phi;
        transform[3][1] = r * cos_phi;
        transform[3][3] = t;

        // Apply transformation to covariance matrix
        for i in 0..4 {
            for j in 0..4 {
                let mut new_val = 0.0;
                for k in 0..4 {
                    for l in 0..4 {
                        new_val +=
                            transform[i][k] * old_covar[indices[k]][indices[l]] * transform[j][l];
                    }
                }
                self.covariance_matrix[indices[i]][indices[j]] = new_val;
            }
        }

        Ok(())
    }

    /// Apply phase rotation
    pub fn apply_phase_rotation(&mut self, mode: usize, phi: f64) -> DeviceResult<()> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes",
                mode
            )));
        }

        let cos_phi = phi.cos();
        let sin_phi = phi.sin();

        // Rotate mean vector
        let mean_x = self.mean_vector[2 * mode];
        let mean_p = self.mean_vector[2 * mode + 1];

        self.mean_vector[2 * mode] = cos_phi * mean_x + sin_phi * mean_p;
        self.mean_vector[2 * mode + 1] = -sin_phi * mean_x + cos_phi * mean_p;

        // Rotation transformation matrix
        let indices = [2 * mode, 2 * mode + 1];
        let old_covar = self.covariance_matrix.clone();

        let transform = [[cos_phi, sin_phi], [-sin_phi, cos_phi]];

        // Apply rotation to covariance matrix
        for i in 0..2 {
            for j in 0..2 {
                let mut new_val = 0.0;
                for k in 0..2 {
                    for l in 0..2 {
                        new_val +=
                            transform[i][k] * old_covar[indices[k]][indices[l]] * transform[j][l];
                    }
                }
                self.covariance_matrix[indices[i]][indices[j]] = new_val;
            }
        }

        Ok(())
    }

    /// Perform homodyne measurement
    pub fn homodyne_measurement(
        &mut self,
        mode: usize,
        phase: f64,
        config: &CVDeviceConfig,
    ) -> DeviceResult<f64> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes",
                mode
            )));
        }

        // Measurement operator: x*cos(phi) + p*sin(phi)
        let cos_phi = phase.cos();
        let sin_phi = phase.sin();

        // Mean value
        let mean_result =
            cos_phi * self.mean_vector[2 * mode] + sin_phi * self.mean_vector[2 * mode + 1];

        // Variance
        let var_x = self.covariance_matrix[2 * mode][2 * mode];
        let var_p = self.covariance_matrix[2 * mode + 1][2 * mode + 1];
        let cov_xp = self.covariance_matrix[2 * mode][2 * mode + 1];

        let variance = cos_phi * cos_phi * var_x
            + sin_phi * sin_phi * var_p
            + 2.0 * cos_phi * sin_phi * cov_xp;

        // Add noise effects
        let noise_variance = self.calculate_measurement_noise(config);
        let total_variance = variance + noise_variance;

        // Sample from Gaussian distribution
        let mut rng = StdRng::seed_from_u64(random::<u64>());
        let noise: f64 = Normal::new(0.0, total_variance.sqrt())
            .map_err(|e| DeviceError::InvalidInput(format!("Distribution error: {}", e)))?
            .sample(&mut rng);

        let result = mean_result + noise;

        // Condition the state on the measurement result
        self.condition_on_homodyne_measurement(mode, phase, result)?;

        Ok(result)
    }

    /// Perform heterodyne measurement
    pub fn heterodyne_measurement(
        &mut self,
        mode: usize,
        config: &CVDeviceConfig,
    ) -> DeviceResult<Complex> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes",
                mode
            )));
        }

        // Heterodyne measures both quadratures simultaneously
        let mean_x = self.mean_vector[2 * mode];
        let mean_p = self.mean_vector[2 * mode + 1];

        let var_x = self.covariance_matrix[2 * mode][2 * mode];
        let var_p = self.covariance_matrix[2 * mode + 1][2 * mode + 1];

        // Add noise
        let noise_variance = self.calculate_measurement_noise(config);
        let mut rng = StdRng::seed_from_u64(random::<u64>());

        let noise_x: f64 = Normal::new(0.0, (var_x + noise_variance / 2.0).sqrt())
            .map_err(|e| DeviceError::InvalidInput(format!("Distribution error: {}", e)))?
            .sample(&mut rng);

        let noise_p: f64 = Normal::new(0.0, (var_p + noise_variance / 2.0).sqrt())
            .map_err(|e| DeviceError::InvalidInput(format!("Distribution error: {}", e)))?
            .sample(&mut rng);

        let result_x = mean_x + noise_x;
        let result_p = mean_p + noise_p;

        let result = Complex::new(
            (result_x + result_p * Complex::i().real) / (2.0_f64).sqrt(),
            (result_p - result_x * Complex::i().imag) / (2.0_f64).sqrt(),
        );

        // Condition state on measurement (destructive measurement)
        self.condition_on_heterodyne_measurement(mode, result)?;

        Ok(result)
    }

    /// Calculate measurement noise based on device configuration
    fn calculate_measurement_noise(&self, config: &CVDeviceConfig) -> f64 {
        // Include electronic noise, detection efficiency, and thermal noise
        let electronic_noise = 10.0_f64.powf(config.electronic_noise_db / 10.0);
        let efficiency_loss = 1.0 / config.detection_efficiency - 1.0;
        let thermal_noise = 2.0 * config.temperature_k / 0.01; // Relative to 10 mK

        electronic_noise + efficiency_loss + thermal_noise
    }

    /// Condition state on homodyne measurement result
    pub fn condition_on_homodyne_measurement(
        &mut self,
        mode: usize,
        phase: f64,
        result: f64,
    ) -> DeviceResult<()> {
        // This is a simplified conditioning - in practice would use full Kalman filter
        // For now, just reset the measured mode to vacuum and update correlations

        let cos_phi = phase.cos();
        let sin_phi = phase.sin();

        // Update mean vector
        self.mean_vector[2 * mode] = result * cos_phi / (2.0_f64).sqrt();
        self.mean_vector[2 * mode + 1] = result * sin_phi / (2.0_f64).sqrt();

        // Simplified: reduce variance in measured quadrature
        let measured_var = cos_phi * cos_phi * self.covariance_matrix[2 * mode][2 * mode]
            + sin_phi * sin_phi * self.covariance_matrix[2 * mode + 1][2 * mode + 1];

        let reduction_factor = 0.1; // Measurement significantly reduces uncertainty
        self.covariance_matrix[2 * mode][2 * mode] *= reduction_factor;
        self.covariance_matrix[2 * mode + 1][2 * mode + 1] *= reduction_factor;

        Ok(())
    }

    /// Condition state on heterodyne measurement result
    pub fn condition_on_heterodyne_measurement(
        &mut self,
        mode: usize,
        _result: Complex,
    ) -> DeviceResult<()> {
        // Heterodyne measurement destroys the mode - reset to vacuum
        self.reset_mode_to_vacuum(mode)
    }

    /// Reset a mode to vacuum state
    pub fn reset_mode_to_vacuum(&mut self, mode: usize) -> DeviceResult<()> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes",
                mode
            )));
        }

        // Reset mean
        self.mean_vector[2 * mode] = 0.0;
        self.mean_vector[2 * mode + 1] = 0.0;

        // Reset covariance to vacuum values
        self.covariance_matrix[2 * mode][2 * mode] = 0.5;
        self.covariance_matrix[2 * mode + 1][2 * mode + 1] = 0.5;
        self.covariance_matrix[2 * mode][2 * mode + 1] = 0.0;
        self.covariance_matrix[2 * mode + 1][2 * mode] = 0.0;

        // Remove correlations with other modes
        for i in 0..2 * self.num_modes {
            if i != 2 * mode && i != 2 * mode + 1 {
                self.covariance_matrix[2 * mode][i] = 0.0;
                self.covariance_matrix[2 * mode + 1][i] = 0.0;
                self.covariance_matrix[i][2 * mode] = 0.0;
                self.covariance_matrix[i][2 * mode + 1] = 0.0;
            }
        }

        Ok(())
    }

    /// Get mode state information
    pub fn get_mode_state(&self, mode: usize) -> DeviceResult<CVModeState> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes",
                mode
            )));
        }

        let mean_amplitude = Complex::new(
            self.mean_vector[2 * mode] / (2.0_f64).sqrt(),
            self.mean_vector[2 * mode + 1] / (2.0_f64).sqrt(),
        );

        let var_x = self.covariance_matrix[2 * mode][2 * mode];
        let var_p = self.covariance_matrix[2 * mode + 1][2 * mode + 1];

        // Calculate squeezing parameters
        let (squeezing_parameter, squeezing_phase) = self.calculate_squeezing(mode);

        // Calculate mode purity
        let purity = self.calculate_mode_purity(mode);

        Ok(CVModeState {
            mean_amplitude,
            quadrature_variances: (var_x, var_p),
            squeezing_parameter,
            squeezing_phase,
            purity,
        })
    }

    /// Calculate squeezing parameters for a mode
    fn calculate_squeezing(&self, mode: usize) -> (f64, f64) {
        let var_x = self.covariance_matrix[2 * mode][2 * mode];
        let var_p = self.covariance_matrix[2 * mode + 1][2 * mode + 1];
        let cov_xp = self.covariance_matrix[2 * mode][2 * mode + 1];

        // Find minimum variance quadrature
        let delta = (var_x - var_p).powi(2) + 4.0 * cov_xp.powi(2);
        let min_var = 0.5 * (var_x + var_p - delta.sqrt());
        let max_var = 0.5 * (var_x + var_p + delta.sqrt());

        let squeezing_parameter = if min_var < 0.5 {
            -0.5 * (2.0 * min_var).ln()
        } else {
            0.0
        };

        let squeezing_phase = if cov_xp.abs() > 1e-10 {
            0.5 * (2.0 * cov_xp / (var_x - var_p)).atan()
        } else {
            0.0
        };

        (squeezing_parameter, squeezing_phase)
    }

    /// Calculate mode purity
    fn calculate_mode_purity(&self, mode: usize) -> f64 {
        let var_x = self.covariance_matrix[2 * mode][2 * mode];
        let var_p = self.covariance_matrix[2 * mode + 1][2 * mode + 1];
        let cov_xp = self.covariance_matrix[2 * mode][2 * mode + 1];

        let det = var_x * var_p - cov_xp.powi(2);
        1.0 / (4.0 * det)
    }

    /// Calculate average squeezing across all modes
    pub fn calculate_average_squeezing(&self) -> f64 {
        let mut total_squeezing = 0.0;
        for mode in 0..self.num_modes {
            let (squeezing, _) = self.calculate_squeezing(mode);
            total_squeezing += squeezing;
        }
        total_squeezing / self.num_modes as f64
    }

    /// Calculate system purity
    pub fn calculate_purity(&self) -> f64 {
        // Simplified purity calculation
        let mut total_purity = 0.0;
        for mode in 0..self.num_modes {
            total_purity += self.calculate_mode_purity(mode);
        }
        total_purity / self.num_modes as f64
    }

    /// Calculate entanglement entropy
    pub fn calculate_entanglement_entropy(&self) -> f64 {
        // Simplified calculation based on covariance matrix eigenvalues
        let mut entropy = 0.0;

        for mode in 0..self.num_modes {
            let var_x = self.covariance_matrix[2 * mode][2 * mode];
            let var_p = self.covariance_matrix[2 * mode + 1][2 * mode + 1];
            let cov_xp = self.covariance_matrix[2 * mode][2 * mode + 1];

            let det = var_x * var_p - cov_xp.powi(2);
            if det > 0.25 {
                let eigenvalue = det.sqrt();
                entropy += (eigenvalue + 0.5) * (eigenvalue + 0.5).ln()
                    - (eigenvalue - 0.5) * (eigenvalue - 0.5).ln();
            }
        }

        entropy
    }

    /// Calculate entanglement measures
    pub fn calculate_entanglement_measures(&self) -> CVEntanglementMeasures {
        // Simplified calculations for demonstration
        let entropy = self.calculate_entanglement_entropy();

        CVEntanglementMeasures {
            logarithmic_negativity: entropy * 0.5,
            entanglement_of_formation: entropy * 0.7,
            mutual_information: entropy * 1.2,
            epr_correlation: self.calculate_epr_correlation(),
        }
    }

    /// Calculate EPR correlation
    fn calculate_epr_correlation(&self) -> f64 {
        if self.num_modes < 2 {
            return 0.0;
        }

        // Calculate correlation between first two modes
        let cov_x1x2 = self.covariance_matrix[0][2];
        let cov_p1p2 = self.covariance_matrix[1][3];

        (cov_x1x2.abs() + cov_p1p2.abs()) / 2.0
    }
}

// rand_distr imports are already available at the top of the file

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vacuum_state_creation() {
        let state = GaussianState::vacuum_state(2);
        assert_eq!(state.num_modes, 2);
        assert_eq!(state.mean_vector.len(), 4);
        assert_eq!(state.covariance_matrix.len(), 4);

        // Check vacuum variances
        assert!((state.covariance_matrix[0][0] - 0.5).abs() < 1e-10);
        assert!((state.covariance_matrix[1][1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_coherent_state_creation() {
        let displacements = vec![Complex::new(1.0, 0.5), Complex::new(0.0, 1.0)];
        let state = GaussianState::coherent_state(2, displacements).unwrap();

        assert!(state.mean_vector[0] > 0.0); // x quadrature of mode 0
        assert!(state.mean_vector[1] > 0.0); // p quadrature of mode 0
    }

    #[test]
    fn test_displacement_operation() {
        let mut state = GaussianState::vacuum_state(1);
        let displacement = Complex::new(2.0, 1.0);

        state.apply_displacement(0, displacement).unwrap();

        assert!((state.mean_vector[0] - 2.0 * (2.0_f64).sqrt()).abs() < 1e-10);
        assert!((state.mean_vector[1] - 1.0 * (2.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_squeezing_operation() {
        let mut state = GaussianState::vacuum_state(1);

        state.apply_squeezing(0, 1.0, 0.0).unwrap();

        // Check that one quadrature is squeezed
        assert!(state.covariance_matrix[0][0] < 0.5); // x should be squeezed
        assert!(state.covariance_matrix[1][1] > 0.5); // p should be antisqueezed
    }

    #[test]
    fn test_beamsplitter_operation() {
        let mut state =
            GaussianState::coherent_state(2, vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)])
                .unwrap();

        let initial_energy = state.mean_vector[0].powi(2)
            + state.mean_vector[1].powi(2)
            + state.mean_vector[2].powi(2)
            + state.mean_vector[3].powi(2);

        state.apply_beamsplitter(0, 1, 0.5, 0.0).unwrap();

        let final_energy = state.mean_vector[0].powi(2)
            + state.mean_vector[1].powi(2)
            + state.mean_vector[2].powi(2)
            + state.mean_vector[3].powi(2);

        // Energy should be conserved
        assert!((initial_energy - final_energy).abs() < 1e-10);
    }

    #[test]
    fn test_mode_state_calculation() {
        let state = GaussianState::squeezed_vacuum_state(1, vec![1.0], vec![0.0]).unwrap();

        let mode_state = state.get_mode_state(0).unwrap();
        assert!(mode_state.squeezing_parameter > 0.0);
        assert!((mode_state.squeezing_phase).abs() < 1e-10);
    }
}
