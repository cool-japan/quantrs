//! Continuous Variable Quantum Computing Implementation
//!
//! This module implements continuous variable (CV) quantum computing operations,
//! including Gaussian states, displacement operations, squeezing, and CV gates.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::{PI, SQRT_2};
use thiserror::Error;

use super::{PhotonicMode, PhotonicSystemType};
use crate::DeviceResult;

/// Errors specific to continuous variable operations
#[derive(Error, Debug)]
pub enum CVError {
    #[error("Invalid displacement parameter: {0}")]
    InvalidDisplacement(String),
    #[error("Invalid squeezing parameter: {0}")]
    InvalidSqueezing(String),
    #[error("Mode not found: {0}")]
    ModeNotFound(usize),
    #[error("Incompatible CV operation: {0}")]
    IncompatibleOperation(String),
    #[error("Matrix dimension mismatch: {0}")]
    MatrixDimensionMismatch(String),
}

pub type CVResult<T> = Result<T, CVError>;

/// Complex number representation for CV quantum computing
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl Complex {
    pub const fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn magnitude(&self) -> f64 {
        self.real.hypot(self.imag)
    }

    pub fn phase(&self) -> f64 {
        self.imag.atan2(self.real)
    }

    #[must_use]
    pub fn conj(&self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
        }
    }
}

/// Gaussian state representation in phase space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianState {
    /// Mean vector (displacement vector)
    pub mean: Vec<f64>,
    /// Covariance matrix
    pub covariance: Vec<Vec<f64>>,
    /// Number of modes
    pub num_modes: usize,
}

impl GaussianState {
    /// Create vacuum state
    pub fn vacuum(num_modes: usize) -> Self {
        let mean = vec![0.0; 2 * num_modes];
        let mut covariance = vec![vec![0.0; 2 * num_modes]; 2 * num_modes];

        // Initialize covariance matrix for vacuum state
        for i in 0..2 * num_modes {
            covariance[i][i] = 0.5; // Vacuum noise
        }

        Self {
            mean,
            covariance,
            num_modes,
        }
    }

    /// Create coherent state with displacement alpha
    pub fn coherent(alpha: Complex, mode: usize, num_modes: usize) -> CVResult<Self> {
        if mode >= num_modes {
            return Err(CVError::ModeNotFound(mode));
        }

        let mut state = Self::vacuum(num_modes);

        // Set displacement in position and momentum
        state.mean[2 * mode] = alpha.real * SQRT_2;
        state.mean[2 * mode + 1] = alpha.imag * SQRT_2;

        Ok(state)
    }

    /// Create squeezed vacuum state
    pub fn squeezed_vacuum(r: f64, phi: f64, mode: usize, num_modes: usize) -> CVResult<Self> {
        if mode >= num_modes {
            return Err(CVError::ModeNotFound(mode));
        }

        if r.abs() > 10.0 {
            return Err(CVError::InvalidSqueezing(
                "Squeezing parameter too large".to_string(),
            ));
        }

        let mut state = Self::vacuum(num_modes);

        // Apply squeezing to covariance matrix
        let cos_2phi = (2.0 * phi).cos();
        let sin_2phi = (2.0 * phi).sin();
        let exp_2r = (2.0 * r).exp();
        let exp_neg_2r = (-2.0 * r).exp();

        let x_idx = 2 * mode;
        let p_idx = 2 * mode + 1;

        // Update covariance matrix elements using correct squeezing formulas
        // For phi=0: x-variance = 0.5 * e^(-2r), p-variance = 0.5 * e^(2r)
        state.covariance[x_idx][x_idx] =
            0.5 * (exp_2r - exp_neg_2r).mul_add(-cos_2phi, exp_neg_2r + exp_2r) / 2.0;
        state.covariance[p_idx][p_idx] =
            0.5 * (exp_2r - exp_neg_2r).mul_add(cos_2phi, exp_2r + exp_neg_2r) / 2.0;
        state.covariance[x_idx][p_idx] = 0.5 * (exp_2r - exp_neg_2r) * sin_2phi / 2.0;
        state.covariance[p_idx][x_idx] = state.covariance[x_idx][p_idx];

        Ok(state)
    }

    /// Create thermal state
    pub fn thermal(n_bar: f64, mode: usize, num_modes: usize) -> CVResult<Self> {
        if mode >= num_modes {
            return Err(CVError::ModeNotFound(mode));
        }

        if n_bar < 0.0 {
            return Err(CVError::InvalidDisplacement(
                "Thermal photon number must be non-negative".to_string(),
            ));
        }

        let mut state = Self::vacuum(num_modes);

        // Add thermal noise
        let thermal_variance = 0.5 * 2.0f64.mul_add(n_bar, 1.0);
        state.covariance[2 * mode][2 * mode] = thermal_variance;
        state.covariance[2 * mode + 1][2 * mode + 1] = thermal_variance;

        Ok(state)
    }

    /// Apply displacement operation
    pub fn displace(&mut self, alpha: Complex, mode: usize) -> CVResult<()> {
        if mode >= self.num_modes {
            return Err(CVError::ModeNotFound(mode));
        }

        // Add displacement to mean vector
        self.mean[2 * mode] += alpha.real * SQRT_2;
        self.mean[2 * mode + 1] += alpha.imag * SQRT_2;

        Ok(())
    }

    /// Apply squeezing operation
    pub fn squeeze(&mut self, r: f64, phi: f64, mode: usize) -> CVResult<()> {
        if mode >= self.num_modes {
            return Err(CVError::ModeNotFound(mode));
        }

        if r.abs() > 10.0 {
            return Err(CVError::InvalidSqueezing(
                "Squeezing parameter too large".to_string(),
            ));
        }

        // Squeezing transformation matrix
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        let cosh_r = r.cosh();
        let sinh_r = r.sinh();

        let s_matrix = [
            [sinh_r.mul_add(-cos_phi, cosh_r), -sinh_r * sin_phi],
            [-sinh_r * sin_phi, sinh_r.mul_add(cos_phi, cosh_r)],
        ];

        // Apply transformation to covariance matrix
        let x_idx = 2 * mode;
        let p_idx = 2 * mode + 1;

        let old_cov = [
            [self.covariance[x_idx][x_idx], self.covariance[x_idx][p_idx]],
            [self.covariance[p_idx][x_idx], self.covariance[p_idx][p_idx]],
        ];

        // New covariance = S * old_cov * S^T
        for i in 0..2 {
            for j in 0..2 {
                let mut new_val = 0.0;
                for k in 0..2 {
                    for l in 0..2 {
                        new_val += s_matrix[i][k] * old_cov[k][l] * s_matrix[j][l];
                    }
                }
                let idx_i = x_idx + i;
                let idx_j = x_idx + j;
                self.covariance[idx_i][idx_j] = new_val;
            }
        }

        Ok(())
    }

    /// Apply two-mode squeezing operation
    pub fn two_mode_squeeze(
        &mut self,
        r: f64,
        phi: f64,
        mode1: usize,
        mode2: usize,
    ) -> CVResult<()> {
        if mode1 >= self.num_modes || mode2 >= self.num_modes {
            return Err(CVError::ModeNotFound(mode1.max(mode2)));
        }

        if mode1 == mode2 {
            return Err(CVError::IncompatibleOperation(
                "Two-mode squeezing requires different modes".to_string(),
            ));
        }

        // Two-mode squeezing transformation
        let cosh_r = r.cosh();
        let sinh_r = r.sinh();
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();

        // Update covariance matrix for two modes
        let x1_idx = 2 * mode1;
        let p1_idx = 2 * mode1 + 1;
        let x2_idx = 2 * mode2;
        let p2_idx = 2 * mode2 + 1;

        // Store original values
        let orig_cov = [
            [
                self.covariance[x1_idx][x1_idx],
                self.covariance[x1_idx][p1_idx],
                self.covariance[x1_idx][x2_idx],
                self.covariance[x1_idx][p2_idx],
            ],
            [
                self.covariance[p1_idx][x1_idx],
                self.covariance[p1_idx][p1_idx],
                self.covariance[p1_idx][x2_idx],
                self.covariance[p1_idx][p2_idx],
            ],
            [
                self.covariance[x2_idx][x1_idx],
                self.covariance[x2_idx][p1_idx],
                self.covariance[x2_idx][x2_idx],
                self.covariance[x2_idx][p2_idx],
            ],
            [
                self.covariance[p2_idx][x1_idx],
                self.covariance[p2_idx][p1_idx],
                self.covariance[p2_idx][x2_idx],
                self.covariance[p2_idx][p2_idx],
            ],
        ];

        // Two-mode squeezing matrix
        let s_matrix = [
            [cosh_r, 0.0, sinh_r * cos_phi, sinh_r * sin_phi],
            [0.0, cosh_r, sinh_r * sin_phi, -sinh_r * cos_phi],
            [sinh_r * cos_phi, sinh_r * sin_phi, cosh_r, 0.0],
            [sinh_r * sin_phi, -sinh_r * cos_phi, 0.0, cosh_r],
        ];

        // Apply transformation
        let indices = [x1_idx, p1_idx, x2_idx, p2_idx];
        for i in 0..4 {
            for j in 0..4 {
                let mut new_val = 0.0;
                for k in 0..4 {
                    for l in 0..4 {
                        new_val += s_matrix[i][k] * orig_cov[k][l] * s_matrix[j][l];
                    }
                }
                self.covariance[indices[i]][indices[j]] = new_val;
            }
        }

        Ok(())
    }

    /// Apply beamsplitter operation
    pub fn beamsplitter(
        &mut self,
        theta: f64,
        phi: f64,
        mode1: usize,
        mode2: usize,
    ) -> CVResult<()> {
        if mode1 >= self.num_modes || mode2 >= self.num_modes {
            return Err(CVError::ModeNotFound(mode1.max(mode2)));
        }

        if mode1 == mode2 {
            return Err(CVError::IncompatibleOperation(
                "Beamsplitter requires different modes".to_string(),
            ));
        }

        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();

        // Beamsplitter transformation matrix
        let bs_matrix = [
            [cos_theta, 0.0, sin_theta * cos_phi, sin_theta * sin_phi],
            [0.0, cos_theta, -sin_theta * sin_phi, sin_theta * cos_phi],
            [-sin_theta * cos_phi, sin_theta * sin_phi, cos_theta, 0.0],
            [-sin_theta * sin_phi, -sin_theta * cos_phi, 0.0, cos_theta],
        ];

        // Apply transformation to both mean and covariance
        let indices = [2 * mode1, 2 * mode1 + 1, 2 * mode2, 2 * mode2 + 1];

        // Transform mean vector
        let orig_mean = [
            self.mean[indices[0]],
            self.mean[indices[1]],
            self.mean[indices[2]],
            self.mean[indices[3]],
        ];

        for i in 0..4 {
            let mut new_mean = 0.0;
            for j in 0..4 {
                new_mean += bs_matrix[i][j] * orig_mean[j];
            }
            self.mean[indices[i]] = new_mean;
        }

        // Transform covariance matrix
        let orig_cov = [
            [
                self.covariance[indices[0]][indices[0]],
                self.covariance[indices[0]][indices[1]],
                self.covariance[indices[0]][indices[2]],
                self.covariance[indices[0]][indices[3]],
            ],
            [
                self.covariance[indices[1]][indices[0]],
                self.covariance[indices[1]][indices[1]],
                self.covariance[indices[1]][indices[2]],
                self.covariance[indices[1]][indices[3]],
            ],
            [
                self.covariance[indices[2]][indices[0]],
                self.covariance[indices[2]][indices[1]],
                self.covariance[indices[2]][indices[2]],
                self.covariance[indices[2]][indices[3]],
            ],
            [
                self.covariance[indices[3]][indices[0]],
                self.covariance[indices[3]][indices[1]],
                self.covariance[indices[3]][indices[2]],
                self.covariance[indices[3]][indices[3]],
            ],
        ];

        for i in 0..4 {
            for j in 0..4 {
                let mut new_val = 0.0;
                for k in 0..4 {
                    for l in 0..4 {
                        new_val += bs_matrix[i][k] * orig_cov[k][l] * bs_matrix[j][l];
                    }
                }
                self.covariance[indices[i]][indices[j]] = new_val;
            }
        }

        Ok(())
    }

    /// Apply phase rotation
    pub fn phase_rotation(&mut self, phi: f64, mode: usize) -> CVResult<()> {
        if mode >= self.num_modes {
            return Err(CVError::ModeNotFound(mode));
        }

        let cos_phi = phi.cos();
        let sin_phi = phi.sin();

        // Rotation matrix for phase space
        let rotation_matrix = [[cos_phi, -sin_phi], [sin_phi, cos_phi]];

        let x_idx = 2 * mode;
        let p_idx = 2 * mode + 1;

        // Transform mean vector
        let old_x = self.mean[x_idx];
        let old_p = self.mean[p_idx];

        self.mean[x_idx] = rotation_matrix[0][0].mul_add(old_x, rotation_matrix[0][1] * old_p);
        self.mean[p_idx] = rotation_matrix[1][0].mul_add(old_x, rotation_matrix[1][1] * old_p);

        // Transform covariance matrix
        let old_cov = [
            [self.covariance[x_idx][x_idx], self.covariance[x_idx][p_idx]],
            [self.covariance[p_idx][x_idx], self.covariance[p_idx][p_idx]],
        ];

        for i in 0..2 {
            for j in 0..2 {
                let mut new_val = 0.0;
                for k in 0..2 {
                    for l in 0..2 {
                        new_val += rotation_matrix[i][k] * old_cov[k][l] * rotation_matrix[j][l];
                    }
                }
                let idx_i = x_idx + i;
                let idx_j = x_idx + j;
                self.covariance[idx_i][idx_j] = new_val;
            }
        }

        Ok(())
    }

    /// Calculate fidelity with another Gaussian state
    pub fn fidelity(&self, other: &Self) -> CVResult<f64> {
        if self.num_modes != other.num_modes {
            return Err(CVError::MatrixDimensionMismatch(
                "States must have same number of modes".to_string(),
            ));
        }

        // Simplified fidelity calculation for Gaussian states
        // This is a basic implementation - full calculation involves matrix operations

        let mut mean_diff_squared = 0.0;
        for i in 0..self.mean.len() {
            let diff = self.mean[i] - other.mean[i];
            mean_diff_squared += diff * diff;
        }

        // Approximate fidelity based on overlap of means and covariances
        let overlap = (-0.5 * mean_diff_squared).exp();

        // Include covariance contribution (simplified)
        let mut cov_diff = 0.0;
        for i in 0..self.covariance.len() {
            for j in 0..self.covariance[i].len() {
                let diff = self.covariance[i][j] - other.covariance[i][j];
                cov_diff += diff * diff;
            }
        }

        let cov_overlap = (-0.1 * cov_diff).exp();

        Ok(overlap * cov_overlap)
    }

    /// Get average photon number for a mode
    pub fn average_photon_number(&self, mode: usize) -> CVResult<f64> {
        if mode >= self.num_modes {
            return Err(CVError::ModeNotFound(mode));
        }

        let x_idx = 2 * mode;
        let p_idx = 2 * mode + 1;

        // Average photon number = (Var(X) + Var(P) + <X>^2 + <P>^2)/2 - 1/2
        let var_x = self.covariance[x_idx][x_idx];
        let var_p = self.covariance[p_idx][p_idx];
        let mean_x_sq = self.mean[x_idx] * self.mean[x_idx];
        let mean_p_sq = self.mean[p_idx] * self.mean[p_idx];

        let n_avg = (var_x + var_p + mean_x_sq + mean_p_sq) / 2.0 - 0.5;

        Ok(n_avg.max(0.0)) // Ensure non-negative
    }

    /// Get squeezing parameter for a mode
    pub fn squeezing_parameter(&self, mode: usize) -> CVResult<f64> {
        if mode >= self.num_modes {
            return Err(CVError::ModeNotFound(mode));
        }

        let x_idx = 2 * mode;
        let p_idx = 2 * mode + 1;

        let var_x = self.covariance[x_idx][x_idx];
        let var_p = self.covariance[p_idx][p_idx];

        // Squeezing in dB: -10 * log10(min(Var(X), Var(P)) / 0.5)
        let min_variance = var_x.min(var_p);
        let squeezing_db = -10.0 * (min_variance / 0.5).log10();

        Ok(squeezing_db.max(0.0))
    }
}

/// CV gate operations
pub struct CVGateSet;

impl CVGateSet {
    /// Create displacement operation
    pub fn displacement(
        alpha: Complex,
        mode: usize,
    ) -> impl Fn(&mut GaussianState) -> CVResult<()> {
        move |state: &mut GaussianState| state.displace(alpha, mode)
    }

    /// Create squeezing operation
    pub fn squeezing(r: f64, phi: f64, mode: usize) -> impl Fn(&mut GaussianState) -> CVResult<()> {
        move |state: &mut GaussianState| state.squeeze(r, phi, mode)
    }

    /// Create two-mode squeezing operation
    pub fn two_mode_squeezing(
        r: f64,
        phi: f64,
        mode1: usize,
        mode2: usize,
    ) -> impl Fn(&mut GaussianState) -> CVResult<()> {
        move |state: &mut GaussianState| state.two_mode_squeeze(r, phi, mode1, mode2)
    }

    /// Create beamsplitter operation
    pub fn beamsplitter(
        theta: f64,
        phi: f64,
        mode1: usize,
        mode2: usize,
    ) -> impl Fn(&mut GaussianState) -> CVResult<()> {
        move |state: &mut GaussianState| state.beamsplitter(theta, phi, mode1, mode2)
    }

    /// Create phase rotation operation
    pub fn phase_rotation(phi: f64, mode: usize) -> impl Fn(&mut GaussianState) -> CVResult<()> {
        move |state: &mut GaussianState| state.phase_rotation(phi, mode)
    }
}

/// Measurement operations for CV systems
pub struct CVMeasurements;

impl CVMeasurements {
    /// Perform homodyne measurement
    pub fn homodyne(state: &GaussianState, mode: usize, phase: f64) -> CVResult<f64> {
        if mode >= state.num_modes {
            return Err(CVError::ModeNotFound(mode));
        }

        let x_idx = 2 * mode;
        let p_idx = 2 * mode + 1;

        // Quadrature operator: X_phi = X*cos(phi) + P*sin(phi)
        let cos_phi = phase.cos();
        let sin_phi = phase.sin();

        let mean_value = cos_phi.mul_add(state.mean[x_idx], sin_phi * state.mean[p_idx]);
        let variance = (2.0 * cos_phi * sin_phi).mul_add(
            state.covariance[x_idx][p_idx],
            (cos_phi * cos_phi).mul_add(
                state.covariance[x_idx][x_idx],
                sin_phi * sin_phi * state.covariance[p_idx][p_idx],
            ),
        );

        // Return mean value (in a real implementation, this would be sampled)
        Ok(mean_value)
    }

    /// Perform heterodyne measurement
    pub fn heterodyne(state: &GaussianState, mode: usize) -> CVResult<Complex> {
        if mode >= state.num_modes {
            return Err(CVError::ModeNotFound(mode));
        }

        let x_idx = 2 * mode;
        let p_idx = 2 * mode + 1;

        // Heterodyne measures both quadratures simultaneously
        let alpha_real = state.mean[x_idx] / SQRT_2;
        let alpha_imag = state.mean[p_idx] / SQRT_2;

        Ok(Complex::new(alpha_real, alpha_imag))
    }

    /// Perform photon number measurement (approximate for CV)
    pub fn photon_number(state: &GaussianState, mode: usize) -> CVResult<f64> {
        state.average_photon_number(mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vacuum_state() {
        let vacuum = GaussianState::vacuum(2);
        assert_eq!(vacuum.num_modes, 2);
        assert_eq!(vacuum.mean.len(), 4);
        assert_eq!(vacuum.covariance.len(), 4);

        // Check vacuum noise
        for i in 0..4 {
            assert_eq!(vacuum.mean[i], 0.0);
            assert_eq!(vacuum.covariance[i][i], 0.5);
        }
    }

    #[test]
    fn test_coherent_state() {
        let alpha = Complex::new(1.0, 0.5);
        let coherent =
            GaussianState::coherent(alpha, 0, 1).expect("Coherent state creation should succeed");

        assert!((coherent.mean[0] - alpha.real * SQRT_2).abs() < 1e-10);
        assert!((coherent.mean[1] - alpha.imag * SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_squeezed_vacuum() {
        let squeezed = GaussianState::squeezed_vacuum(1.0, 0.0, 0, 1)
            .expect("Squeezed vacuum creation should succeed");

        // For squeezing in X quadrature (phi=0), Var(X) should be reduced
        assert!(squeezed.covariance[0][0] < 0.5);
        assert!(squeezed.covariance[1][1] > 0.5);
    }

    #[test]
    fn test_displacement_operation() {
        let mut state = GaussianState::vacuum(1);
        let alpha = Complex::new(2.0, 1.0);

        state
            .displace(alpha, 0)
            .expect("Displacement operation should succeed");

        assert!((state.mean[0] - alpha.real * SQRT_2).abs() < 1e-10);
        assert!((state.mean[1] - alpha.imag * SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_beamsplitter() {
        let mut state = GaussianState::coherent(Complex::new(1.0, 0.0), 0, 2)
            .expect("Coherent state creation should succeed");

        // 50:50 beamsplitter
        state
            .beamsplitter(PI / 4.0, 0.0, 0, 1)
            .expect("Beamsplitter operation should succeed");

        // Check that amplitude is distributed between modes
        assert!(state.mean[0].abs() > 0.0);
        assert!(state.mean[2].abs() > 0.0);
    }

    #[test]
    fn test_average_photon_number() {
        let alpha = Complex::new(2.0, 0.0);
        let coherent =
            GaussianState::coherent(alpha, 0, 1).expect("Coherent state creation should succeed");

        let n_avg = coherent
            .average_photon_number(0)
            .expect("Photon number calculation should succeed");
        let expected = alpha.magnitude() * alpha.magnitude();

        assert!((n_avg - expected).abs() < 1e-10);
    }

    #[test]
    fn test_homodyne_measurement() {
        let alpha = Complex::new(2.0, 1.0);
        let coherent =
            GaussianState::coherent(alpha, 0, 1).expect("Coherent state creation should succeed");

        // Measure X quadrature
        let x_result = CVMeasurements::homodyne(&coherent, 0, 0.0)
            .expect("X quadrature homodyne measurement should succeed");
        assert!((x_result - alpha.real * SQRT_2).abs() < 1e-10);

        // Measure P quadrature
        let p_result = CVMeasurements::homodyne(&coherent, 0, PI / 2.0)
            .expect("P quadrature homodyne measurement should succeed");
        assert!((p_result - alpha.imag * SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_fidelity() {
        let state1 = GaussianState::vacuum(1);
        let state2 = GaussianState::vacuum(1);

        let fidelity = state1
            .fidelity(&state2)
            .expect("Fidelity calculation should succeed");
        assert!((fidelity - 1.0).abs() < 1e-6); // Identical states should have fidelity 1

        let coherent = GaussianState::coherent(Complex::new(1.0, 0.0), 0, 1)
            .expect("Coherent state creation should succeed");
        let fidelity_diff = state1
            .fidelity(&coherent)
            .expect("Fidelity calculation should succeed");
        assert!(fidelity_diff < 1.0); // Different states should have fidelity < 1
    }
}
