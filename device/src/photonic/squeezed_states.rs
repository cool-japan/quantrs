//! Squeezed State Generation and Manipulation
//!
//! This module implements advanced squeezed state operations for photonic quantum computing,
//! including generation, characterization, and application of squeezed light.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::{E, PI};
use thiserror::Error;

use super::continuous_variable::{CVError, CVResult, Complex, GaussianState};
use super::{PhotonicMode, PhotonicSystemType};
use crate::DeviceResult;
use scirs2_core::random::prelude::*;

/// Squeezed state operation errors
#[derive(Error, Debug)]
pub enum SqueezedStateError {
    #[error("Invalid squeezing parameters: {0}")]
    InvalidParameters(String),
    #[error("Squeezing generation failed: {0}")]
    GenerationFailed(String),
    #[error("State characterization failed: {0}")]
    CharacterizationFailed(String),
    #[error("Squeezing measurement error: {0}")]
    MeasurementError(String),
}

/// Types of squeezed states
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub enum SqueezedStateType {
    /// Single-mode squeezed vacuum
    SingleMode { squeezing_dB: f64, angle: f64 },
    /// Two-mode squeezed state
    TwoMode { squeezing_dB: f64, phase: f64 },
    /// Multimode squeezed state
    Multimode { squeezing_matrix: Vec<Vec<f64>> },
    /// Spin squeezed state
    SpinSqueezed {
        collective_spin: f64,
        squeezing_parameter: f64,
    },
    /// Amplitude squeezed state
    AmplitudeSqueezed { alpha: Complex, squeezing_dB: f64 },
    /// Phase squeezed state
    PhaseSqueezed { alpha: Complex, squeezing_dB: f64 },
}

/// Squeezing generation method
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SqueezingMethod {
    /// Parametric down-conversion
    ParametricDownConversion {
        pump_power: f64,
        crystal_length: f64,
        phase_matching: Phasematching,
    },
    /// Four-wave mixing
    FourWaveMixing {
        pump_powers: Vec<f64>,
        fiber_length: f64,
        nonlinearity: f64,
    },
    /// Kerr squeezing
    KerrSqueezing {
        kerr_coefficient: f64,
        interaction_time: f64,
    },
    /// Cavity squeezing
    CavitySqueezing {
        cavity_finesse: f64,
        pump_detuning: f64,
    },
}

/// Phase matching conditions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Phasematching {
    /// Type I phase matching
    TypeI { crystal_angle: f64 },
    /// Type II phase matching
    TypeII {
        crystal_angle: f64,
        temperature: f64,
    },
    /// Quasi-phase matching
    QuasiPhaseMatching { poling_period: f64 },
}

/// Squeezed state characterization
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct SqueezingCharacterization {
    /// Measured squeezing level (dB)
    pub measured_squeezing_dB: f64,
    /// Squeezing angle (radians)
    pub squeezing_angle: f64,
    /// Anti-squeezing level (dB)
    pub anti_squeezing_dB: f64,
    /// Noise figure
    pub noise_figure: f64,
    /// Squeezing bandwidth
    pub bandwidth_hz: f64,
    /// Purity measure
    pub purity: f64,
    /// Loss estimation
    pub estimated_loss: f64,
}

/// Squeezed state generator
pub struct SqueezedStateGenerator {
    /// Generation method
    pub method: SqueezingMethod,
    /// Operating parameters
    pub parameters: SqueezingParameters,
    /// Performance metrics
    pub performance: GeneratorPerformance,
    /// Calibration data
    pub calibration: GeneratorCalibration,
}

/// Squeezing generation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct SqueezingParameters {
    /// Target squeezing level (dB)
    pub target_squeezing_dB: f64,
    /// Operating wavelength (nm)
    pub wavelength_nm: f64,
    /// Pump power (mW)
    pub pump_power_mw: f64,
    /// Temperature (K)
    pub temperature_k: f64,
    /// Phase stabilization enabled
    pub phase_stabilization: bool,
    /// Feedback control enabled
    pub feedback_control: bool,
}

impl Default for SqueezingParameters {
    fn default() -> Self {
        Self {
            target_squeezing_dB: 10.0,
            wavelength_nm: 1550.0,
            pump_power_mw: 100.0,
            temperature_k: 293.0,
            phase_stabilization: true,
            feedback_control: true,
        }
    }
}

/// Generator performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(non_snake_case)]
pub struct GeneratorPerformance {
    /// Achieved squeezing level (dB)
    pub achieved_squeezing_dB: f64,
    /// Squeezing stability (%)
    pub stability_percent: f64,
    /// Generation efficiency (%)
    pub efficiency_percent: f64,
    /// Output power (μW)
    pub output_power_uw: f64,
    /// Noise floor (dB below shot noise)
    pub noise_floor_dB: f64,
}

impl Default for GeneratorPerformance {
    fn default() -> Self {
        Self {
            achieved_squeezing_dB: 8.5,
            stability_percent: 95.0,
            efficiency_percent: 85.0,
            output_power_uw: 50.0,
            noise_floor_dB: -15.0,
        }
    }
}

/// Generator calibration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorCalibration {
    /// Power calibration curve
    pub power_curve: Vec<(f64, f64)>, // (input_power, output_squeezing)
    /// Temperature dependence
    pub temperature_curve: Vec<(f64, f64)>, // (temperature, squeezing)
    /// Phase calibration
    pub phase_calibration: Vec<(f64, f64)>, // (phase_setting, actual_phase)
    /// Last calibration timestamp (Unix timestamp)
    #[serde(with = "instant_serde")]
    pub last_calibration: std::time::Instant,
}

mod instant_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(instant: &Instant, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration_since_epoch = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        duration_since_epoch.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Instant, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Instant::now()) // Simplified - in practice would need proper epoch handling
    }
}

impl Default for GeneratorCalibration {
    fn default() -> Self {
        Self {
            power_curve: vec![(0.0, 0.0), (50.0, 5.0), (100.0, 10.0), (200.0, 12.0)],
            temperature_curve: vec![(273.0, 8.0), (293.0, 10.0), (313.0, 9.0)],
            phase_calibration: vec![(0.0, 0.0), (PI / 2.0, PI / 2.0), (PI, PI)],
            last_calibration: std::time::Instant::now(),
        }
    }
}

#[allow(non_snake_case)]
impl SqueezedStateGenerator {
    pub fn new(method: SqueezingMethod) -> Self {
        Self {
            method,
            parameters: SqueezingParameters::default(),
            performance: GeneratorPerformance::default(),
            calibration: GeneratorCalibration::default(),
        }
    }

    /// Generate a squeezed state
    pub fn generate_squeezed_state(
        &mut self,
        state_type: SqueezedStateType,
        num_modes: usize,
    ) -> Result<GaussianState, SqueezedStateError> {
        match state_type {
            SqueezedStateType::SingleMode {
                squeezing_dB,
                angle,
            } => self.generate_single_mode_squeezed(squeezing_dB, angle, num_modes),
            SqueezedStateType::TwoMode {
                squeezing_dB,
                phase,
            } => self.generate_two_mode_squeezed(squeezing_dB, phase, num_modes),
            SqueezedStateType::AmplitudeSqueezed {
                alpha,
                squeezing_dB,
            } => self.generate_amplitude_squeezed(alpha, squeezing_dB, num_modes),
            SqueezedStateType::PhaseSqueezed {
                alpha,
                squeezing_dB,
            } => self.generate_phase_squeezed(alpha, squeezing_dB, num_modes),
            _ => Err(SqueezedStateError::GenerationFailed(format!(
                "State type {:?} not yet implemented",
                state_type
            ))),
        }
    }

    /// Generate single-mode squeezed vacuum
    fn generate_single_mode_squeezed(
        &mut self,
        squeezing_dB: f64,
        angle: f64,
        num_modes: usize,
    ) -> Result<GaussianState, SqueezedStateError> {
        if squeezing_dB < 0.0 {
            return Err(SqueezedStateError::InvalidParameters(
                "Squeezing must be non-negative".to_string(),
            ));
        }

        // Convert dB to linear squeezing parameter
        let squeezing_r = squeezing_dB * (10.0_f64.ln() / 20.0);

        // Account for realistic limitations
        let realistic_r = self.apply_realistic_limitations(squeezing_r)?;

        // Create squeezed vacuum state
        match GaussianState::squeezed_vacuum(realistic_r, angle, 0, num_modes) {
            Ok(state) => {
                self.update_performance_metrics(squeezing_dB, realistic_r);
                Ok(state)
            }
            Err(e) => Err(SqueezedStateError::GenerationFailed(format!(
                "Failed to generate squeezed state: {:?}",
                e
            ))),
        }
    }

    /// Generate two-mode squeezed state
    fn generate_two_mode_squeezed(
        &mut self,
        squeezing_dB: f64,
        phase: f64,
        num_modes: usize,
    ) -> Result<GaussianState, SqueezedStateError> {
        if num_modes < 2 {
            return Err(SqueezedStateError::InvalidParameters(
                "Two-mode squeezing requires at least 2 modes".to_string(),
            ));
        }

        let squeezing_r = squeezing_dB * (10.0_f64.ln() / 20.0);
        let realistic_r = self.apply_realistic_limitations(squeezing_r)?;

        let mut state = GaussianState::vacuum(num_modes);

        match state.two_mode_squeeze(realistic_r, phase, 0, 1) {
            Ok(_) => {
                self.update_performance_metrics(squeezing_dB, realistic_r);
                Ok(state)
            }
            Err(e) => Err(SqueezedStateError::GenerationFailed(format!(
                "Failed to generate two-mode squeezed state: {:?}",
                e
            ))),
        }
    }

    /// Generate amplitude squeezed state
    fn generate_amplitude_squeezed(
        &mut self,
        alpha: Complex,
        squeezing_dB: f64,
        num_modes: usize,
    ) -> Result<GaussianState, SqueezedStateError> {
        let squeezing_r = squeezing_dB * (10.0_f64.ln() / 20.0);
        let realistic_r = self.apply_realistic_limitations(squeezing_r)?;

        // Create coherent state
        let mut state = match GaussianState::coherent(alpha, 0, num_modes) {
            Ok(s) => s,
            Err(e) => {
                return Err(SqueezedStateError::GenerationFailed(format!(
                    "Failed to create coherent state: {:?}",
                    e
                )))
            }
        };

        // Apply amplitude squeezing (squeezing in phase with displacement)
        let squeezing_angle = alpha.phase();
        match state.squeeze(realistic_r, squeezing_angle, 0) {
            Ok(_) => {
                self.update_performance_metrics(squeezing_dB, realistic_r);
                Ok(state)
            }
            Err(e) => Err(SqueezedStateError::GenerationFailed(format!(
                "Failed to apply amplitude squeezing: {:?}",
                e
            ))),
        }
    }

    /// Generate phase squeezed state
    fn generate_phase_squeezed(
        &mut self,
        alpha: Complex,
        squeezing_dB: f64,
        num_modes: usize,
    ) -> Result<GaussianState, SqueezedStateError> {
        let squeezing_r = squeezing_dB * (10.0_f64.ln() / 20.0);
        let realistic_r = self.apply_realistic_limitations(squeezing_r)?;

        // Create coherent state
        let mut state = match GaussianState::coherent(alpha, 0, num_modes) {
            Ok(s) => s,
            Err(e) => {
                return Err(SqueezedStateError::GenerationFailed(format!(
                    "Failed to create coherent state: {:?}",
                    e
                )))
            }
        };

        // Apply phase squeezing (squeezing perpendicular to displacement)
        let squeezing_angle = alpha.phase() + PI / 2.0;
        match state.squeeze(realistic_r, squeezing_angle, 0) {
            Ok(_) => {
                self.update_performance_metrics(squeezing_dB, realistic_r);
                Ok(state)
            }
            Err(e) => Err(SqueezedStateError::GenerationFailed(format!(
                "Failed to apply phase squeezing: {:?}",
                e
            ))),
        }
    }

    /// Apply realistic limitations to squeezing
    fn apply_realistic_limitations(&self, ideal_r: f64) -> Result<f64, SqueezedStateError> {
        // Limit maximum squeezing based on method
        let max_r = match &self.method {
            SqueezingMethod::ParametricDownConversion { .. } => 2.3, // ~20 dB typical maximum
            SqueezingMethod::FourWaveMixing { .. } => 1.15,          // ~10 dB typical maximum
            SqueezingMethod::KerrSqueezing { .. } => 0.46,           // ~4 dB typical maximum
            SqueezingMethod::CavitySqueezing { .. } => 3.45,         // ~30 dB possible in cavities
        };

        if ideal_r > max_r {
            return Err(SqueezedStateError::InvalidParameters(format!(
                "Requested squeezing {} exceeds maximum {} for method {:?}",
                ideal_r, max_r, self.method
            )));
        }

        // Apply efficiency and loss factors
        let efficiency = self.performance.efficiency_percent / 100.0;
        let realistic_r = ideal_r * efficiency;

        // Add noise and imperfections
        let noise_factor = 1.0 - (self.performance.noise_floor_dB / 20.0);
        let final_r = realistic_r * noise_factor;

        Ok(final_r.max(0.0))
    }

    /// Update performance metrics based on generation
    fn update_performance_metrics(&mut self, target_dB: f64, achieved_r: f64) {
        let achieved_dB = achieved_r * (20.0 / 10.0_f64.ln());
        self.performance.achieved_squeezing_dB = achieved_dB;

        // Update efficiency estimate
        let efficiency = achieved_dB / target_dB;
        self.performance.efficiency_percent = (efficiency * 100.0).min(100.0);

        // Simulate stability fluctuations
        let stability_noise = (thread_rng().gen::<f64>() - 0.5) * 0.1;
        self.performance.stability_percent = (95.0 + stability_noise).max(90.0).min(99.0);
    }

    /// Characterize generated squeezed state
    pub fn characterize_state(
        &self,
        state: &GaussianState,
        mode: usize,
    ) -> Result<SqueezingCharacterization, SqueezedStateError> {
        if mode >= state.num_modes {
            return Err(SqueezedStateError::CharacterizationFailed(format!(
                "Mode {} does not exist",
                mode
            )));
        }

        // Extract squeezing information from covariance matrix
        let x_idx = 2 * mode;
        let p_idx = 2 * mode + 1;

        let var_x = state.covariance[x_idx][x_idx];
        let var_p = state.covariance[p_idx][p_idx];
        let cov_xp = state.covariance[x_idx][p_idx];

        // Calculate eigenvalues of covariance matrix
        let trace = var_x + var_p;
        let det = var_x * var_p - cov_xp * cov_xp;
        let discriminant = trace * trace - 4.0 * det;

        if discriminant < 0.0 {
            return Err(SqueezedStateError::CharacterizationFailed(
                "Invalid covariance matrix".to_string(),
            ));
        }

        let sqrt_discriminant = discriminant.sqrt();
        let eigenval1 = (trace + sqrt_discriminant) / 2.0;
        let eigenval2 = (trace - sqrt_discriminant) / 2.0;

        let min_eigenval = eigenval1.min(eigenval2);
        let max_eigenval = eigenval1.max(eigenval2);

        // Calculate squeezing and anti-squeezing in dB
        let squeezing_dB = -10.0 * (min_eigenval / 0.5).log10();
        let anti_squeezing_dB = 10.0 * (max_eigenval / 0.5).log10();

        // Calculate squeezing angle
        let squeezing_angle = if cov_xp.abs() > 1e-10 {
            0.5 * (2.0 * cov_xp / (var_x - var_p)).atan()
        } else {
            if var_x < var_p {
                0.0
            } else {
                PI / 2.0
            }
        };

        // Estimate purity
        let purity = 1.0 / (4.0 * det).sqrt();

        // Estimate loss (simplified)
        let theoretical_min =
            0.5 * (-self.performance.achieved_squeezing_dB * 10.0_f64.ln() / 10.0).exp();
        let estimated_loss = (min_eigenval - theoretical_min) / (0.5 - theoretical_min);

        Ok(SqueezingCharacterization {
            measured_squeezing_dB: squeezing_dB,
            squeezing_angle,
            anti_squeezing_dB,
            noise_figure: anti_squeezing_dB - squeezing_dB,
            bandwidth_hz: 1e6, // Placeholder - would depend on specific measurement
            purity: purity.min(1.0),
            estimated_loss: estimated_loss.max(0.0).min(1.0),
        })
    }

    /// Optimize generator parameters
    pub fn optimize_parameters(
        &mut self,
        target_squeezing_dB: f64,
    ) -> Result<SqueezingParameters, SqueezedStateError> {
        // Simple optimization based on calibration curves
        let mut optimized = self.parameters.clone();
        optimized.target_squeezing_dB = target_squeezing_dB;

        // Find optimal power from calibration curve
        let optimal_power = self.find_optimal_power(target_squeezing_dB);
        optimized.pump_power_mw = optimal_power;

        // Adjust temperature for stability
        let optimal_temp = self.find_optimal_temperature(target_squeezing_dB);
        optimized.temperature_k = optimal_temp;

        // Enable feedback for high squeezing levels
        optimized.feedback_control = target_squeezing_dB >= 15.0;
        optimized.phase_stabilization = target_squeezing_dB >= 10.0;

        self.parameters = optimized.clone();
        Ok(optimized)
    }

    /// Find optimal pump power from calibration
    fn find_optimal_power(&self, target_squeezing_dB: f64) -> f64 {
        for (power, squeezing) in &self.calibration.power_curve {
            if *squeezing >= target_squeezing_dB {
                return *power;
            }
        }
        // Extrapolate if beyond calibration range
        self.calibration
            .power_curve
            .last()
            .map_or(100.0, |(p, _)| *p * 1.5)
    }

    /// Find optimal temperature from calibration
    fn find_optimal_temperature(&self, _target_squeezing_dB: f64) -> f64 {
        // Find temperature with maximum squeezing
        self.calibration
            .temperature_curve
            .iter()
            .max_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(293.0, |(t, _)| *t)
    }

    /// Create multimode squeezed state
    pub fn create_multimode_squeezed(
        &mut self,
        num_modes: usize,
        squeezing_matrix: &[Vec<f64>],
    ) -> Result<GaussianState, SqueezedStateError> {
        if squeezing_matrix.len() != num_modes
            || squeezing_matrix.iter().any(|row| row.len() != num_modes)
        {
            return Err(SqueezedStateError::InvalidParameters(
                "Squeezing matrix dimensions don't match number of modes".to_string(),
            ));
        }

        let mut state = GaussianState::vacuum(num_modes);

        // Apply squeezing transformations based on matrix
        for i in 0..num_modes {
            for j in i + 1..num_modes {
                let squeezing_strength = squeezing_matrix[i][j];
                if squeezing_strength.abs() > 1e-10 {
                    let realistic_r = self.apply_realistic_limitations(squeezing_strength)?;
                    if let Err(e) = state.two_mode_squeeze(realistic_r, 0.0, i, j) {
                        return Err(SqueezedStateError::GenerationFailed(format!(
                            "Multimode squeezing failed: {:?}",
                            e
                        )));
                    }
                }
            }
        }

        Ok(state)
    }
}

/// Squeezed state measurement apparatus
pub struct SqueezedStateMeasurement {
    /// Homodyne detection efficiency
    pub homodyne_efficiency: f64,
    /// Electronic noise level
    pub electronic_noise: f64,
    /// Dark count rate
    pub dark_count_rate: f64,
    /// Measurement bandwidth
    pub bandwidth_hz: f64,
}

#[allow(non_snake_case)]
impl SqueezedStateMeasurement {
    pub fn new() -> Self {
        Self {
            homodyne_efficiency: 0.95,
            electronic_noise: 0.01,
            dark_count_rate: 1000.0, // Hz
            bandwidth_hz: 1e6,
        }
    }

    /// Measure squeezing via homodyne detection
    pub fn measure_squeezing_homodyne(
        &self,
        state: &GaussianState,
        mode: usize,
        local_oscillator_phase: f64,
        measurement_time: f64,
    ) -> Result<Vec<f64>, SqueezedStateError> {
        if mode >= state.num_modes {
            return Err(SqueezedStateError::MeasurementError(format!(
                "Mode {} does not exist",
                mode
            )));
        }

        let x_idx = 2 * mode;
        let p_idx = 2 * mode + 1;

        // Calculate quadrature being measured
        let cos_phi = local_oscillator_phase.cos();
        let sin_phi = local_oscillator_phase.sin();

        let mean_quad = cos_phi * state.mean[x_idx] + sin_phi * state.mean[p_idx];
        let var_quad = cos_phi * cos_phi * state.covariance[x_idx][x_idx]
            + sin_phi * sin_phi * state.covariance[p_idx][p_idx]
            + 2.0 * cos_phi * sin_phi * state.covariance[x_idx][p_idx];

        // Account for detection efficiency
        let effective_variance = self.homodyne_efficiency * var_quad
            + (1.0 - self.homodyne_efficiency) * 0.5
            + self.electronic_noise;

        // Generate measurement samples
        let num_samples = (measurement_time * self.bandwidth_hz) as usize;
        let mut measurements = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            // Add dark counts
            let dark_noise = if thread_rng().gen::<f64>() < self.dark_count_rate / self.bandwidth_hz {
                (thread_rng().gen::<f64>() - 0.5) * 0.1
            } else {
                0.0
            };

            // Generate Gaussian measurement result
            let measurement =
                self.generate_gaussian_sample(mean_quad, effective_variance) + dark_noise;
            measurements.push(measurement);
        }

        Ok(measurements)
    }

    /// Generate Gaussian random sample
    fn generate_gaussian_sample(&self, mean: f64, variance: f64) -> f64 {
        // Box-Muller transform
        let u1 = thread_rng().gen::<f64>();
        let u2 = thread_rng().gen::<f64>();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + variance.sqrt() * z
    }

    /// Analyze measurement data for squeezing
    pub fn analyze_squeezing(
        &self,
        measurements: &[f64],
        reference_measurements: &[f64],
    ) -> Result<f64, SqueezedStateError> {
        if measurements.is_empty() || reference_measurements.is_empty() {
            return Err(SqueezedStateError::MeasurementError(
                "Insufficient measurement data".to_string(),
            ));
        }

        // Calculate variances
        let squeezed_variance = self.calculate_variance(measurements);
        let reference_variance = self.calculate_variance(reference_measurements);

        // Squeezing in dB relative to reference (shot noise)
        let squeezing_dB = 10.0 * (squeezed_variance / reference_variance).log10();

        Ok(-squeezing_dB) // Negative because squeezing reduces noise
    }

    /// Calculate sample variance
    fn calculate_variance(&self, data: &[f64]) -> f64 {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        variance
    }
}

impl Default for SqueezedStateMeasurement {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squeezed_state_generator_creation() {
        let method = SqueezingMethod::ParametricDownConversion {
            pump_power: 100.0,
            crystal_length: 10.0,
            phase_matching: Phasematching::TypeI {
                crystal_angle: 45.0,
            },
        };

        let generator = SqueezedStateGenerator::new(method);
        assert_eq!(generator.parameters.target_squeezing_dB, 10.0);
    }

    #[test]
    fn test_single_mode_squeezed_generation() {
        let method = SqueezingMethod::ParametricDownConversion {
            pump_power: 100.0,
            crystal_length: 10.0,
            phase_matching: Phasematching::TypeI {
                crystal_angle: 45.0,
            },
        };

        let mut generator = SqueezedStateGenerator::new(method);

        let state_type = SqueezedStateType::SingleMode {
            squeezing_dB: 10.0,
            angle: 0.0,
        };

        let result = generator.generate_squeezed_state(state_type, 1);
        assert!(result.is_ok());

        let state = result.unwrap();
        assert_eq!(state.num_modes, 1);

        // Check that squeezing was applied (X quadrature should be squeezed)
        assert!(state.covariance[0][0] < 0.5); // Squeezed quadrature
        assert!(state.covariance[1][1] > 0.5); // Anti-squeezed quadrature
    }

    #[test]
    fn test_two_mode_squeezed_generation() {
        let method = SqueezingMethod::FourWaveMixing {
            pump_powers: vec![50.0, 50.0],
            fiber_length: 100.0,
            nonlinearity: 0.001,
        };

        let mut generator = SqueezedStateGenerator::new(method);

        let state_type = SqueezedStateType::TwoMode {
            squeezing_dB: 8.0,
            phase: 0.0,
        };

        let result = generator.generate_squeezed_state(state_type, 2);
        assert!(result.is_ok());

        let state = result.unwrap();
        assert_eq!(state.num_modes, 2);
    }

    #[test]
    fn test_squeezing_characterization() {
        let method = SqueezingMethod::ParametricDownConversion {
            pump_power: 100.0,
            crystal_length: 10.0,
            phase_matching: Phasematching::TypeI {
                crystal_angle: 45.0,
            },
        };

        let mut generator = SqueezedStateGenerator::new(method);

        let state_type = SqueezedStateType::SingleMode {
            squeezing_dB: 10.0,
            angle: 0.0,
        };

        let state = generator.generate_squeezed_state(state_type, 1).unwrap();
        let characterization = generator.characterize_state(&state, 0).unwrap();

        assert!(characterization.measured_squeezing_dB > 0.0);
        assert!(characterization.anti_squeezing_dB > 0.0);
        assert!(characterization.purity > 0.0 && characterization.purity <= 1.0);
    }

    #[test]
    fn test_measurement_apparatus() {
        let measurement = SqueezedStateMeasurement::new();

        // Create a simple squeezed state for testing
        let state = GaussianState::squeezed_vacuum(1.0, 0.0, 0, 1).unwrap();

        let measurements = measurement
            .measure_squeezing_homodyne(&state, 0, 0.0, 0.1)
            .unwrap();

        assert!(!measurements.is_empty());
        assert!(measurements.len() > 10000); // Should have many samples
    }

    #[test]
    fn test_parameter_optimization() {
        let method = SqueezingMethod::ParametricDownConversion {
            pump_power: 100.0,
            crystal_length: 10.0,
            phase_matching: Phasematching::TypeI {
                crystal_angle: 45.0,
            },
        };

        let mut generator = SqueezedStateGenerator::new(method);

        let optimized = generator.optimize_parameters(15.0).unwrap();
        assert_eq!(optimized.target_squeezing_dB, 15.0);
        assert!(optimized.feedback_control); // Should be enabled for high squeezing
        assert!(optimized.phase_stabilization);
    }
}
