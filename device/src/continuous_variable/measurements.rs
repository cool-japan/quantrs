//! Continuous variable quantum measurements
//!
//! This module implements various measurement schemes for CV quantum systems,
//! including homodyne, heterodyne, photon number, and parity measurements.

use super::{CVDeviceConfig, Complex, GaussianState};
use crate::{DeviceError, DeviceResult};
use rand::{random, rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Types of CV measurements
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CVMeasurementScheme {
    /// Homodyne detection at specific phase
    Homodyne { phase: f64 },
    /// Heterodyne detection (simultaneous x and p)
    Heterodyne,
    /// Photon number measurement
    PhotonNumber,
    /// Parity measurement (even/odd photon number)
    Parity,
    /// Bell measurement for entangled modes
    Bell { basis: BellBasis },
    /// Fock state projection
    FockProjection { n: usize },
    /// Coherent state projection
    CoherentProjection { alpha: Complex },
}

/// Bell measurement bases for CV systems
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BellBasis {
    /// X-X correlation measurement
    XBasis,
    /// P-P correlation measurement
    PBasis,
    /// Mixed X-P correlation
    MixedBasis { phase: f64 },
}

/// CV measurement result with statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVMeasurementResult {
    /// The measurement outcome
    pub outcome: CVMeasurementOutcome,
    /// Measurement fidelity/confidence
    pub fidelity: f64,
    /// Standard deviation of the measurement
    pub standard_deviation: f64,
    /// Number of samples used
    pub sample_count: usize,
    /// Timestamp
    pub timestamp: f64,
}

/// Different types of measurement outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CVMeasurementOutcome {
    /// Real-valued outcome (homodyne)
    Real(f64),
    /// Complex-valued outcome (heterodyne)
    Complex(Complex),
    /// Integer outcome (photon number)
    Integer(i32),
    /// Boolean outcome (parity)
    Boolean(bool),
    /// Bell correlation outcome
    BellCorrelation { correlation: f64, phase: f64 },
}

/// Configuration for CV measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVMeasurementConfig {
    /// Number of measurement samples
    pub num_samples: usize,
    /// Integration time (seconds)
    pub integration_time: f64,
    /// Local oscillator power (mW)
    pub lo_power_mw: f64,
    /// Detection bandwidth (Hz)
    pub bandwidth_hz: f64,
    /// Phase lock stability (rad RMS)
    pub phase_stability: f64,
    /// Enable post-processing
    pub enable_post_processing: bool,
    /// Calibration data
    pub calibration: MeasurementCalibration,
}

/// Calibration data for CV measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCalibration {
    /// Detection efficiency per mode
    pub efficiency: Vec<f64>,
    /// Electronic noise level (dB)
    pub electronic_noise_db: f64,
    /// Dark count rate (Hz)
    pub dark_count_rate: f64,
    /// Gain calibration factors
    pub gain_factors: Vec<f64>,
    /// Phase calibration offsets
    pub phase_offsets: Vec<f64>,
}

impl Default for CVMeasurementConfig {
    fn default() -> Self {
        Self {
            num_samples: 10000,
            integration_time: 0.001, // 1 ms
            lo_power_mw: 10.0,
            bandwidth_hz: 10e6,
            phase_stability: 0.01,
            enable_post_processing: true,
            calibration: MeasurementCalibration::default(),
        }
    }
}

impl Default for MeasurementCalibration {
    fn default() -> Self {
        Self {
            efficiency: vec![0.95; 10], // Default for up to 10 modes
            electronic_noise_db: -90.0,
            dark_count_rate: 100.0,
            gain_factors: vec![1.0; 10],
            phase_offsets: vec![0.0; 10],
        }
    }
}

/// CV measurement engine
pub struct CVMeasurementEngine {
    /// Measurement configuration
    config: CVMeasurementConfig,
    /// Current measurement results
    measurement_history: Vec<CVMeasurementResult>,
    /// Calibration state
    is_calibrated: bool,
}

impl CVMeasurementEngine {
    /// Create a new measurement engine
    pub fn new(config: CVMeasurementConfig) -> Self {
        Self {
            config,
            measurement_history: Vec::new(),
            is_calibrated: false,
        }
    }

    /// Calibrate the measurement system
    pub async fn calibrate(&mut self) -> DeviceResult<()> {
        println!("Calibrating CV measurement system...");

        // Simulate calibration process
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        // Update calibration factors
        for i in 0..self.config.calibration.efficiency.len() {
            self.config.calibration.efficiency[i] = 0.95 + 0.03 * rand::random::<f64>();
            self.config.calibration.gain_factors[i] = 1.0 + 0.1 * (rand::random::<f64>() - 0.5);
            self.config.calibration.phase_offsets[i] = 0.1 * (rand::random::<f64>() - 0.5);
        }

        self.is_calibrated = true;
        println!("Calibration complete");
        Ok(())
    }

    /// Perform homodyne measurement
    pub async fn homodyne_measurement(
        &mut self,
        state: &mut GaussianState,
        mode: usize,
        phase: f64,
    ) -> DeviceResult<CVMeasurementResult> {
        if !self.is_calibrated {
            return Err(DeviceError::DeviceNotInitialized(
                "Measurement system not calibrated".to_string(),
            ));
        }

        if mode >= state.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes",
                mode
            )));
        }

        // Apply phase offset calibration
        let calibrated_phase = phase
            + self
                .config
                .calibration
                .phase_offsets
                .get(mode)
                .unwrap_or(&0.0);

        // Get the theoretical mean and variance
        let cos_phi = calibrated_phase.cos();
        let sin_phi = calibrated_phase.sin();

        let mean_x = state.mean_vector[2 * mode];
        let mean_p = state.mean_vector[2 * mode + 1];
        let theoretical_mean = cos_phi * mean_x + sin_phi * mean_p;

        let var_x = state.covariancematrix[2 * mode][2 * mode];
        let var_p = state.covariancematrix[2 * mode + 1][2 * mode + 1];
        let cov_xp = state.covariancematrix[2 * mode][2 * mode + 1];

        let theoretical_variance =
            cos_phi.powi(2) * var_x + sin_phi.powi(2) * var_p + 2.0 * cos_phi * sin_phi * cov_xp;

        // Add noise effects
        let efficiency = self
            .config
            .calibration
            .efficiency
            .get(mode)
            .unwrap_or(&0.95);
        let gain = self
            .config
            .calibration
            .gain_factors
            .get(mode)
            .unwrap_or(&1.0);

        let noise_variance = self.calculate_detection_noise(mode);
        let total_variance = theoretical_variance / efficiency + noise_variance;

        // Perform multiple samples
        let mut samples = Vec::new();
        let distribution = Normal::new(theoretical_mean, total_variance.sqrt())
            .map_err(|e| DeviceError::InvalidInput(format!("Distribution error: {}", e)))?;

        let mut rng = StdRng::seed_from_u64(random::<u64>());
        for _ in 0..self.config.num_samples {
            let sample = distribution.sample(&mut rng) * gain;
            samples.push(sample);
        }

        // Calculate statistics
        let mean_outcome = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples
            .iter()
            .map(|x| (x - mean_outcome).powi(2))
            .sum::<f64>()
            / samples.len() as f64;
        let std_dev = variance.sqrt();

        // Estimate fidelity
        let fidelity =
            self.estimate_measurement_fidelity(&samples, theoretical_mean, total_variance);

        let result = CVMeasurementResult {
            outcome: CVMeasurementOutcome::Real(mean_outcome),
            fidelity,
            standard_deviation: std_dev,
            sample_count: samples.len(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        // Condition the state on the measurement
        state.condition_on_homodyne_measurement(mode, calibrated_phase, mean_outcome)?;

        self.measurement_history.push(result.clone());
        Ok(result)
    }

    /// Perform heterodyne measurement
    pub async fn heterodyne_measurement(
        &mut self,
        state: &mut GaussianState,
        mode: usize,
    ) -> DeviceResult<CVMeasurementResult> {
        if !self.is_calibrated {
            return Err(DeviceError::DeviceNotInitialized(
                "Measurement system not calibrated".to_string(),
            ));
        }

        if mode >= state.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes",
                mode
            )));
        }

        let efficiency = self
            .config
            .calibration
            .efficiency
            .get(mode)
            .unwrap_or(&0.95);
        let gain = self
            .config
            .calibration
            .gain_factors
            .get(mode)
            .unwrap_or(&1.0);

        // Get state parameters
        let mean_x = state.mean_vector[2 * mode];
        let mean_p = state.mean_vector[2 * mode + 1];
        let var_x = state.covariancematrix[2 * mode][2 * mode];
        let var_p = state.covariancematrix[2 * mode + 1][2 * mode + 1];

        // Add noise
        let noise_variance = self.calculate_detection_noise(mode);

        // Sample both quadratures
        let mut x_samples = Vec::new();
        let mut p_samples = Vec::new();

        let x_distribution =
            Normal::new(mean_x, (var_x / efficiency + noise_variance / 2.0).sqrt())
                .map_err(|e| DeviceError::InvalidInput(format!("Distribution error: {}", e)))?;
        let p_distribution =
            Normal::new(mean_p, (var_p / efficiency + noise_variance / 2.0).sqrt())
                .map_err(|e| DeviceError::InvalidInput(format!("Distribution error: {}", e)))?;

        let mut rng = StdRng::seed_from_u64(random::<u64>());
        for _ in 0..self.config.num_samples {
            x_samples.push(x_distribution.sample(&mut rng) * gain);
            p_samples.push(p_distribution.sample(&mut rng) * gain);
        }

        // Calculate complex outcome
        let mean_x_outcome = x_samples.iter().sum::<f64>() / x_samples.len() as f64;
        let mean_p_outcome = p_samples.iter().sum::<f64>() / p_samples.len() as f64;

        let complex_outcome = Complex::new(
            (mean_x_outcome + mean_p_outcome * Complex::i().real) / (2.0_f64).sqrt(),
            (mean_p_outcome - mean_x_outcome * Complex::i().imag) / (2.0_f64).sqrt(),
        );

        // Calculate standard deviation
        let x_var = x_samples
            .iter()
            .map(|x| (x - mean_x_outcome).powi(2))
            .sum::<f64>()
            / x_samples.len() as f64;
        let p_var = p_samples
            .iter()
            .map(|p| (p - mean_p_outcome).powi(2))
            .sum::<f64>()
            / p_samples.len() as f64;
        let std_dev = (x_var + p_var).sqrt();

        // Estimate fidelity
        let theoretical_variance = (var_x + var_p) / efficiency + noise_variance;
        let fidelity = 1.0 / (1.0 + theoretical_variance);

        let result = CVMeasurementResult {
            outcome: CVMeasurementOutcome::Complex(complex_outcome),
            fidelity,
            standard_deviation: std_dev,
            sample_count: self.config.num_samples,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        // Condition state (heterodyne destroys the mode)
        state.condition_on_heterodyne_measurement(mode, complex_outcome)?;

        self.measurement_history.push(result.clone());
        Ok(result)
    }

    /// Perform photon number measurement (simplified for Gaussian states)
    pub async fn photon_number_measurement(
        &mut self,
        state: &GaussianState,
        mode: usize,
    ) -> DeviceResult<CVMeasurementResult> {
        if !self.is_calibrated {
            return Err(DeviceError::DeviceNotInitialized(
                "Measurement system not calibrated".to_string(),
            ));
        }

        if mode >= state.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes",
                mode
            )));
        }

        // For Gaussian states, estimate photon number from second moments
        let mean_x = state.mean_vector[2 * mode];
        let mean_p = state.mean_vector[2 * mode + 1];
        let var_x = state.covariancematrix[2 * mode][2 * mode];
        let var_p = state.covariancematrix[2 * mode + 1][2 * mode + 1];

        // Average photon number for Gaussian state
        let mean_n = 0.5 * ((mean_x.powi(2) + mean_p.powi(2)) / 2.0 + (var_x + var_p) - 1.0);

        // For thermal/squeezed states, photon number follows geometric/negative binomial distribution
        // Simplified: just add Poissonian noise around the mean
        let efficiency = self
            .config
            .calibration
            .efficiency
            .get(mode)
            .unwrap_or(&0.95);
        let dark_counts = self.config.calibration.dark_count_rate * self.config.integration_time;

        let detected_n = mean_n * efficiency + dark_counts;

        // Sample from Poisson-like distribution
        let mut samples = Vec::new();
        let mut rng = StdRng::seed_from_u64(random::<u64>());
        for _ in 0..self.config.num_samples {
            let sample = if detected_n > 0.0 {
                let poisson = rand_distr::Poisson::new(detected_n)
                    .map_err(|e| DeviceError::InvalidInput(format!("Poisson error: {}", e)))?;
                poisson.sample(&mut rng) as f64
            } else {
                0.0
            };
            samples.push(sample);
        }

        let mean_outcome = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples
            .iter()
            .map(|x| (x - mean_outcome).powi(2))
            .sum::<f64>()
            / samples.len() as f64;

        let rounded_outcome = mean_outcome.round() as i32;
        let fidelity = 1.0 / (1.0 + variance / (mean_outcome + 1.0));

        let result = CVMeasurementResult {
            outcome: CVMeasurementOutcome::Integer(rounded_outcome),
            fidelity,
            standard_deviation: variance.sqrt(),
            sample_count: samples.len(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        self.measurement_history.push(result.clone());
        Ok(result)
    }

    /// Perform parity measurement
    pub async fn parity_measurement(
        &mut self,
        state: &GaussianState,
        mode: usize,
    ) -> DeviceResult<CVMeasurementResult> {
        // Parity measurement for Gaussian states
        let photon_result = self.photon_number_measurement(state, mode).await?;

        let parity = match photon_result.outcome {
            CVMeasurementOutcome::Integer(n) => n % 2 == 0,
            _ => {
                return Err(DeviceError::InvalidInput(
                    "Invalid photon number result".to_string(),
                ))
            }
        };

        let result = CVMeasurementResult {
            outcome: CVMeasurementOutcome::Boolean(parity),
            fidelity: photon_result.fidelity,
            standard_deviation: if parity { 0.0 } else { 1.0 },
            sample_count: photon_result.sample_count,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        self.measurement_history.push(result.clone());
        Ok(result)
    }

    /// Perform Bell measurement on two modes
    pub async fn bell_measurement(
        &mut self,
        state: &mut GaussianState,
        mode1: usize,
        mode2: usize,
        basis: BellBasis,
    ) -> DeviceResult<CVMeasurementResult> {
        if mode1 >= state.num_modes || mode2 >= state.num_modes {
            return Err(DeviceError::InvalidInput(
                "One or both modes exceed available modes".to_string(),
            ));
        }

        let (phase1, phase2) = match basis {
            BellBasis::XBasis => (0.0, 0.0),
            BellBasis::PBasis => (PI / 2.0, PI / 2.0),
            BellBasis::MixedBasis { phase } => (0.0, phase),
        };

        // Measure both modes
        let result1 = self.homodyne_measurement(state, mode1, phase1).await?;
        let result2 = self.homodyne_measurement(state, mode2, phase2).await?;

        // Calculate correlation
        let (val1, val2) = match (result1.outcome, result2.outcome) {
            (CVMeasurementOutcome::Real(v1), CVMeasurementOutcome::Real(v2)) => (v1, v2),
            _ => {
                return Err(DeviceError::InvalidInput(
                    "Invalid homodyne results".to_string(),
                ))
            }
        };

        let correlation = val1 * val2;
        let avg_phase = (phase1 + phase2) / 2.0;

        let result = CVMeasurementResult {
            outcome: CVMeasurementOutcome::BellCorrelation {
                correlation,
                phase: avg_phase,
            },
            fidelity: (result1.fidelity + result2.fidelity) / 2.0,
            standard_deviation: (result1.standard_deviation.powi(2)
                + result2.standard_deviation.powi(2))
            .sqrt(),
            sample_count: result1.sample_count.min(result2.sample_count),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        };

        self.measurement_history.push(result.clone());
        Ok(result)
    }

    /// Calculate detection noise
    fn calculate_detection_noise(&self, mode: usize) -> f64 {
        let electronic_noise = 10.0_f64.powf(self.config.calibration.electronic_noise_db / 10.0);
        let efficiency = self
            .config
            .calibration
            .efficiency
            .get(mode)
            .unwrap_or(&0.95);
        let phase_noise = self.config.phase_stability.powi(2);

        electronic_noise + (1.0 - efficiency) / efficiency + phase_noise
    }

    /// Estimate measurement fidelity based on sample statistics
    fn estimate_measurement_fidelity(
        &self,
        samples: &[f64],
        theoretical_mean: f64,
        theoretical_variance: f64,
    ) -> f64 {
        let sample_mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let sample_variance = samples
            .iter()
            .map(|x| (x - sample_mean).powi(2))
            .sum::<f64>()
            / samples.len() as f64;

        let mean_error = (sample_mean - theoretical_mean).abs();
        let variance_error = (sample_variance - theoretical_variance).abs();

        // Simple fidelity estimate
        let fidelity = 1.0 / (1.0 + mean_error + variance_error);
        fidelity.clamp(0.0, 1.0)
    }

    /// Get measurement history
    pub fn get_measurement_history(&self) -> &[CVMeasurementResult] {
        &self.measurement_history
    }

    /// Clear measurement history
    pub fn clear_history(&mut self) {
        self.measurement_history.clear();
    }

    /// Get measurement statistics
    pub fn get_measurement_statistics(&self) -> MeasurementStatistics {
        if self.measurement_history.is_empty() {
            return MeasurementStatistics::default();
        }

        let total_measurements = self.measurement_history.len();
        let avg_fidelity = self
            .measurement_history
            .iter()
            .map(|r| r.fidelity)
            .sum::<f64>()
            / total_measurements as f64;

        let avg_std_dev = self
            .measurement_history
            .iter()
            .map(|r| r.standard_deviation)
            .sum::<f64>()
            / total_measurements as f64;

        let total_samples = self
            .measurement_history
            .iter()
            .map(|r| r.sample_count)
            .sum::<usize>();

        MeasurementStatistics {
            total_measurements,
            average_fidelity: avg_fidelity,
            average_standard_deviation: avg_std_dev,
            total_samples,
            is_calibrated: self.is_calibrated,
        }
    }
}

/// Statistics for measurement performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementStatistics {
    pub total_measurements: usize,
    pub average_fidelity: f64,
    pub average_standard_deviation: f64,
    pub total_samples: usize,
    pub is_calibrated: bool,
}

impl Default for MeasurementStatistics {
    fn default() -> Self {
        Self {
            total_measurements: 0,
            average_fidelity: 0.0,
            average_standard_deviation: 0.0,
            total_samples: 0,
            is_calibrated: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_measurement_engine_creation() {
        let config = CVMeasurementConfig::default();
        let engine = CVMeasurementEngine::new(config);
        assert!(!engine.is_calibrated);
        assert_eq!(engine.measurement_history.len(), 0);
    }

    #[tokio::test]
    async fn test_calibration() {
        let config = CVMeasurementConfig::default();
        let mut engine = CVMeasurementEngine::new(config);

        engine.calibrate().await.unwrap();
        assert!(engine.is_calibrated);
    }

    #[tokio::test]
    async fn test_homodyne_measurement() {
        let config = CVMeasurementConfig::default();
        let mut engine = CVMeasurementEngine::new(config);
        engine.calibrate().await.unwrap();

        let mut state =
            GaussianState::coherent_state(2, vec![Complex::new(2.0, 0.0), Complex::new(0.0, 0.0)])
                .unwrap();

        let result = engine
            .homodyne_measurement(&mut state, 0, 0.0)
            .await
            .unwrap();

        match result.outcome {
            CVMeasurementOutcome::Real(value) => {
                assert!(value > 0.0); // Should measure positive x quadrature
            }
            _ => panic!("Expected real outcome"),
        }

        assert!(result.fidelity > 0.0);
        assert_eq!(engine.measurement_history.len(), 1);
    }

    #[tokio::test]
    async fn test_heterodyne_measurement() {
        let config = CVMeasurementConfig::default();
        let mut engine = CVMeasurementEngine::new(config);
        engine.calibrate().await.unwrap();

        let mut state = GaussianState::coherent_state(1, vec![Complex::new(1.0, 0.5)]).unwrap();

        let result = engine.heterodyne_measurement(&mut state, 0).await.unwrap();

        match result.outcome {
            CVMeasurementOutcome::Complex(z) => {
                assert!(z.magnitude() > 0.0);
            }
            _ => panic!("Expected complex outcome"),
        }
    }

    #[tokio::test]
    async fn test_photon_number_measurement() {
        let config = CVMeasurementConfig::default();
        let mut engine = CVMeasurementEngine::new(config);
        engine.calibrate().await.unwrap();

        let state = GaussianState::coherent_state(1, vec![Complex::new(2.0, 0.0)]).unwrap();

        let result = engine.photon_number_measurement(&state, 0).await.unwrap();

        match result.outcome {
            CVMeasurementOutcome::Integer(n) => {
                assert!(n >= 0); // Photon number should be non-negative
            }
            _ => panic!("Expected integer outcome"),
        }
    }

    #[tokio::test]
    async fn test_parity_measurement() {
        let config = CVMeasurementConfig::default();
        let mut engine = CVMeasurementEngine::new(config);
        engine.calibrate().await.unwrap();

        let state = GaussianState::vacuum_state(1);

        let result = engine.parity_measurement(&state, 0).await.unwrap();

        match result.outcome {
            CVMeasurementOutcome::Boolean(parity) => {
                // Vacuum state should have even parity (0 photons)
                assert!(parity);
            }
            _ => panic!("Expected boolean outcome"),
        }
    }

    #[test]
    fn test_measurement_config_defaults() {
        let config = CVMeasurementConfig::default();
        assert_eq!(config.num_samples, 10000);
        assert_eq!(config.integration_time, 0.001);
        assert!(config.enable_post_processing);
    }

    #[test]
    fn test_measurement_statistics() {
        let config = CVMeasurementConfig::default();
        let engine = CVMeasurementEngine::new(config);

        let stats = engine.get_measurement_statistics();
        assert_eq!(stats.total_measurements, 0);
        assert!(!stats.is_calibrated);
    }
}
