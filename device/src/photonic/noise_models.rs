//! Photonic Noise Models
//!
//! This module implements noise models specific to photonic quantum computing systems,
//! including loss, thermal noise, and detector inefficiencies.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use thiserror::Error;

use super::continuous_variable::{CVError, CVResult, Complex, GaussianState};
use super::{PhotonicMode, PhotonicSystemType};
use crate::DeviceResult;

/// Photonic noise model errors
#[derive(Error, Debug)]
pub enum PhotonicNoiseError {
    #[error("Invalid noise parameter: {0}")]
    InvalidParameter(String),
    #[error("Noise model not supported: {0}")]
    UnsupportedModel(String),
    #[error("Noise application failed: {0}")]
    ApplicationFailed(String),
}

/// Types of photonic noise
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PhotonicNoiseType {
    /// Photon loss
    Loss { rate: f64 },
    /// Thermal noise
    Thermal { temperature: f64, frequency: f64 },
    /// Detector inefficiency
    DetectorInefficiency { efficiency: f64 },
    /// Phase noise
    PhaseNoise { variance: f64 },
    /// Amplitude noise
    AmplitudeNoise { variance: f64 },
    /// Cross-talk between modes
    Crosstalk { coupling_strength: f64 },
    /// Nonlinear noise
    Nonlinear { coefficient: f64 },
}

/// Photonic noise model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicNoiseModel {
    /// Types of noise included
    pub noise_types: Vec<PhotonicNoiseType>,
    /// Mode-specific noise parameters
    pub mode_specific: HashMap<usize, Vec<PhotonicNoiseType>>,
    /// Global noise parameters
    pub global_parameters: NoiseParameters,
}

/// Global noise parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseParameters {
    /// Overall noise strength
    pub noise_strength: f64,
    /// Correlation time
    pub correlation_time: f64,
    /// Temperature of environment
    pub environment_temperature: f64,
    /// System bandwidth
    pub bandwidth: f64,
}

impl Default for NoiseParameters {
    fn default() -> Self {
        Self {
            noise_strength: 0.01,
            correlation_time: 1e-6,        // 1 microsecond
            environment_temperature: 0.01, // 10 mK in energy units
            bandwidth: 1e9,                // 1 GHz
        }
    }
}

/// Photonic noise simulator
pub struct PhotonicNoiseSimulator {
    /// Active noise models
    pub noise_models: Vec<PhotonicNoiseModel>,
    /// Noise realization cache
    pub noise_cache: HashMap<String, Vec<f64>>,
    /// Random number generator state
    rng_state: u64,
}

impl PhotonicNoiseSimulator {
    pub fn new() -> Self {
        Self {
            noise_models: Vec::new(),
            noise_cache: HashMap::new(),
            rng_state: 42, // Simple seed
        }
    }

    /// Apply noise to a Gaussian state
    pub fn apply_noise(
        &mut self,
        state: &mut GaussianState,
        noise_model: &PhotonicNoiseModel,
    ) -> Result<(), PhotonicNoiseError> {
        for noise_type in &noise_model.noise_types {
            self.apply_noise_type(state, noise_type)?;
        }

        // Apply mode-specific noise
        for (&mode, noise_types) in &noise_model.mode_specific {
            if mode < state.num_modes {
                for noise_type in noise_types {
                    self.apply_mode_specific_noise(state, mode, noise_type)?;
                }
            }
        }

        Ok(())
    }

    /// Apply a specific type of noise
    fn apply_noise_type(
        &mut self,
        state: &mut GaussianState,
        noise_type: &PhotonicNoiseType,
    ) -> Result<(), PhotonicNoiseError> {
        match noise_type {
            PhotonicNoiseType::Loss { rate } => {
                self.apply_loss_noise(state, *rate)?;
            }
            PhotonicNoiseType::Thermal {
                temperature,
                frequency,
            } => {
                self.apply_thermal_noise(state, *temperature, *frequency)?;
            }
            PhotonicNoiseType::PhaseNoise { variance } => {
                self.apply_phase_noise(state, *variance)?;
            }
            PhotonicNoiseType::AmplitudeNoise { variance } => {
                self.apply_amplitude_noise(state, *variance)?;
            }
            PhotonicNoiseType::DetectorInefficiency { efficiency } => {
                self.apply_detector_noise(state, *efficiency)?;
            }
            PhotonicNoiseType::Crosstalk { coupling_strength } => {
                self.apply_crosstalk_noise(state, *coupling_strength)?;
            }
            PhotonicNoiseType::Nonlinear { coefficient } => {
                self.apply_nonlinear_noise(state, *coefficient)?;
            }
        }
        Ok(())
    }

    /// Apply photon loss noise
    fn apply_loss_noise(
        &mut self,
        state: &mut GaussianState,
        loss_rate: f64,
    ) -> Result<(), PhotonicNoiseError> {
        if loss_rate < 0.0 || loss_rate > 1.0 {
            return Err(PhotonicNoiseError::InvalidParameter(format!(
                "Loss rate must be between 0 and 1, got {}",
                loss_rate
            )));
        }

        let transmission = 1.0 - loss_rate;

        // Apply loss by scaling the covariance matrix
        for mode in 0..state.num_modes {
            let x_idx = 2 * mode;
            let p_idx = 2 * mode + 1;

            // Scale mean values
            state.mean[x_idx] *= transmission.sqrt();
            state.mean[p_idx] *= transmission.sqrt();

            // Scale covariance matrix elements
            state.covariance[x_idx][x_idx] =
                transmission * state.covariance[x_idx][x_idx] + (1.0 - transmission) * 0.5;
            state.covariance[p_idx][p_idx] =
                transmission * state.covariance[p_idx][p_idx] + (1.0 - transmission) * 0.5;
            state.covariance[x_idx][p_idx] *= transmission;
            state.covariance[p_idx][x_idx] *= transmission;
        }

        Ok(())
    }

    /// Apply thermal noise
    fn apply_thermal_noise(
        &mut self,
        state: &mut GaussianState,
        temperature: f64,
        frequency: f64,
    ) -> Result<(), PhotonicNoiseError> {
        if temperature < 0.0 {
            return Err(PhotonicNoiseError::InvalidParameter(
                "Temperature must be non-negative".to_string(),
            ));
        }

        // Calculate thermal photon number: n_th = 1/(exp(hf/kT) - 1)
        // Simplified: n_th ≈ kT/hf for kT << hf
        let thermal_photons = temperature / frequency;

        // Add thermal noise to all modes
        for mode in 0..state.num_modes {
            let x_idx = 2 * mode;
            let p_idx = 2 * mode + 1;

            // Add thermal variance
            state.covariance[x_idx][x_idx] += thermal_photons;
            state.covariance[p_idx][p_idx] += thermal_photons;
        }

        Ok(())
    }

    /// Apply phase noise
    fn apply_phase_noise(
        &mut self,
        state: &mut GaussianState,
        variance: f64,
    ) -> Result<(), PhotonicNoiseError> {
        if variance < 0.0 {
            return Err(PhotonicNoiseError::InvalidParameter(
                "Phase noise variance must be non-negative".to_string(),
            ));
        }

        // Apply random phase rotation to each mode
        for mode in 0..state.num_modes {
            let phase_noise = self.generate_gaussian_noise(0.0, variance);
            if let Err(e) = state.phase_rotation(phase_noise, mode) {
                return Err(PhotonicNoiseError::ApplicationFailed(format!(
                    "Failed to apply phase noise: {:?}",
                    e
                )));
            }
        }

        Ok(())
    }

    /// Apply amplitude noise
    fn apply_amplitude_noise(
        &mut self,
        state: &mut GaussianState,
        variance: f64,
    ) -> Result<(), PhotonicNoiseError> {
        if variance < 0.0 {
            return Err(PhotonicNoiseError::InvalidParameter(
                "Amplitude noise variance must be non-negative".to_string(),
            ));
        }

        // Add amplitude fluctuations to quadratures
        for mode in 0..state.num_modes {
            let x_idx = 2 * mode;
            let p_idx = 2 * mode + 1;

            let x_noise = self.generate_gaussian_noise(0.0, variance);
            let p_noise = self.generate_gaussian_noise(0.0, variance);

            state.mean[x_idx] += x_noise;
            state.mean[p_idx] += p_noise;
        }

        Ok(())
    }

    /// Apply detector inefficiency noise
    fn apply_detector_noise(
        &mut self,
        state: &mut GaussianState,
        efficiency: f64,
    ) -> Result<(), PhotonicNoiseError> {
        if efficiency < 0.0 || efficiency > 1.0 {
            return Err(PhotonicNoiseError::InvalidParameter(format!(
                "Detector efficiency must be between 0 and 1, got {}",
                efficiency
            )));
        }

        // Detector inefficiency acts like loss but only affects measurements
        // For simulation purposes, we model it as additional loss
        self.apply_loss_noise(state, 1.0 - efficiency)
    }

    /// Apply crosstalk noise between modes
    fn apply_crosstalk_noise(
        &mut self,
        state: &mut GaussianState,
        coupling_strength: f64,
    ) -> Result<(), PhotonicNoiseError> {
        if state.num_modes < 2 {
            return Ok(()); // No crosstalk for single mode
        }

        // Apply weak coupling between adjacent modes
        for mode in 0..(state.num_modes - 1) {
            let coupling_angle = coupling_strength * PI / 4.0;
            if let Err(e) = state.beamsplitter(coupling_angle, 0.0, mode, mode + 1) {
                return Err(PhotonicNoiseError::ApplicationFailed(format!(
                    "Failed to apply crosstalk: {:?}",
                    e
                )));
            }
        }

        Ok(())
    }

    /// Apply nonlinear noise
    fn apply_nonlinear_noise(
        &mut self,
        state: &mut GaussianState,
        coefficient: f64,
    ) -> Result<(), PhotonicNoiseError> {
        // Nonlinear noise is typically small and breaks Gaussianity
        // For Gaussian states, we approximate with additional phase noise
        let phase_variance = coefficient.abs() * 0.1;
        self.apply_phase_noise(state, phase_variance)
    }

    /// Apply mode-specific noise
    fn apply_mode_specific_noise(
        &mut self,
        state: &mut GaussianState,
        mode: usize,
        noise_type: &PhotonicNoiseType,
    ) -> Result<(), PhotonicNoiseError> {
        match noise_type {
            PhotonicNoiseType::PhaseNoise { variance } => {
                let phase_noise = self.generate_gaussian_noise(0.0, *variance);
                if let Err(e) = state.phase_rotation(phase_noise, mode) {
                    return Err(PhotonicNoiseError::ApplicationFailed(format!(
                        "Mode-specific phase noise failed: {:?}",
                        e
                    )));
                }
            }
            PhotonicNoiseType::AmplitudeNoise { variance } => {
                let x_idx = 2 * mode;
                let p_idx = 2 * mode + 1;

                let x_noise = self.generate_gaussian_noise(0.0, *variance);
                let p_noise = self.generate_gaussian_noise(0.0, *variance);

                state.mean[x_idx] += x_noise;
                state.mean[p_idx] += p_noise;
            }
            _ => {
                // For other noise types, apply globally
                self.apply_noise_type(state, noise_type)?;
            }
        }
        Ok(())
    }

    /// Generate Gaussian noise (simplified implementation)
    fn generate_gaussian_noise(&mut self, mean: f64, variance: f64) -> f64 {
        // Simple Box-Muller transform
        self.rng_state = self
            .rng_state
            .wrapping_mul(1664525)
            .wrapping_add(1013904223);
        let u1 = (self.rng_state as f64) / (u64::MAX as f64);

        self.rng_state = self
            .rng_state
            .wrapping_mul(1664525)
            .wrapping_add(1013904223);
        let u2 = (self.rng_state as f64) / (u64::MAX as f64);

        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + variance.sqrt() * z
    }

    /// Create a realistic noise model for photonic systems
    pub fn create_realistic_noise_model(
        loss_rate: f64,
        thermal_temperature: f64,
        detector_efficiency: f64,
    ) -> PhotonicNoiseModel {
        let noise_types = vec![
            PhotonicNoiseType::Loss { rate: loss_rate },
            PhotonicNoiseType::Thermal {
                temperature: thermal_temperature,
                frequency: 1e14, // Optical frequency
            },
            PhotonicNoiseType::DetectorInefficiency {
                efficiency: detector_efficiency,
            },
            PhotonicNoiseType::PhaseNoise { variance: 0.001 },
            PhotonicNoiseType::AmplitudeNoise { variance: 0.0005 },
        ];

        PhotonicNoiseModel {
            noise_types,
            mode_specific: HashMap::new(),
            global_parameters: NoiseParameters::default(),
        }
    }

    /// Estimate noise impact on fidelity
    pub fn estimate_noise_impact(
        &self,
        initial_fidelity: f64,
        noise_model: &PhotonicNoiseModel,
    ) -> f64 {
        let mut fidelity = initial_fidelity;

        for noise_type in &noise_model.noise_types {
            match noise_type {
                PhotonicNoiseType::Loss { rate } => {
                    fidelity *= 1.0 - rate * 0.5; // Approximate impact
                }
                PhotonicNoiseType::Thermal { temperature, .. } => {
                    fidelity *= 1.0 - temperature * 0.1;
                }
                PhotonicNoiseType::PhaseNoise { variance } => {
                    fidelity *= 1.0 - variance * 2.0;
                }
                PhotonicNoiseType::DetectorInefficiency { efficiency } => {
                    fidelity *= efficiency;
                }
                _ => {
                    fidelity *= 0.99; // Small general impact
                }
            }
        }

        fidelity.max(0.0).min(1.0)
    }
}

impl Default for PhotonicNoiseSimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photonic::continuous_variable::GaussianState;

    #[test]
    fn test_noise_simulator_creation() {
        let simulator = PhotonicNoiseSimulator::new();
        assert_eq!(simulator.noise_models.len(), 0);
    }

    #[test]
    fn test_loss_noise() {
        let mut simulator = PhotonicNoiseSimulator::new();
        let mut state = GaussianState::coherent(Complex::new(2.0, 0.0), 0, 1).unwrap();

        let original_mean = state.mean[0];
        simulator.apply_loss_noise(&mut state, 0.1).unwrap();

        // Mean should be reduced due to loss
        assert!(state.mean[0] < original_mean);
    }

    #[test]
    fn test_thermal_noise() {
        let mut simulator = PhotonicNoiseSimulator::new();
        let mut state = GaussianState::vacuum(1);

        let original_variance = state.covariance[0][0];
        simulator
            .apply_thermal_noise(&mut state, 0.01, 1e14)
            .unwrap();

        // Variance should increase due to thermal noise
        assert!(state.covariance[0][0] > original_variance);
    }

    #[test]
    fn test_realistic_noise_model() {
        let model = PhotonicNoiseSimulator::create_realistic_noise_model(0.05, 0.01, 0.9);
        assert_eq!(model.noise_types.len(), 5);
    }

    #[test]
    fn test_noise_impact_estimation() {
        let simulator = PhotonicNoiseSimulator::new();
        let model = PhotonicNoiseSimulator::create_realistic_noise_model(0.1, 0.01, 0.95);

        let initial_fidelity = 0.99;
        let final_fidelity = simulator.estimate_noise_impact(initial_fidelity, &model);

        assert!(final_fidelity < initial_fidelity);
        assert!(final_fidelity > 0.0);
    }
}
