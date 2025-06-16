//! Homodyne detection for continuous variable quantum systems
//!
//! This module implements homodyne detection, which measures a specific quadrature
//! of the quantum field by interfering the signal with a strong local oscillator.

use super::{CVDeviceConfig, Complex, GaussianState};
use crate::{DeviceError, DeviceResult};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Homodyne detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomodyneDetectorConfig {
    /// Local oscillator power (mW)
    pub lo_power_mw: f64,
    /// Detection efficiency
    pub efficiency: f64,
    /// Electronic noise (V²/Hz)
    pub electronic_noise: f64,
    /// Detector bandwidth (Hz)
    pub bandwidth_hz: f64,
    /// Saturation level (mW)
    pub saturation_power_mw: f64,
    /// Phase lock loop parameters
    pub pll_config: PLLConfig,
    /// Photodiode specifications
    pub photodiode_config: PhotodiodeConfig,
}

/// Phase-locked loop configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PLLConfig {
    /// Loop bandwidth (Hz)
    pub loop_bandwidth_hz: f64,
    /// Phase noise (rad²/Hz)
    pub phase_noise_density: f64,
    /// Lock range (rad)
    pub lock_range: f64,
    /// Acquisition time (ms)
    pub acquisition_time_ms: f64,
}

/// Photodiode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotodiodeConfig {
    /// Responsivity (A/W)
    pub responsivity: f64,
    /// Dark current (nA)
    pub dark_current_na: f64,
    /// NEP (noise equivalent power) (W/√Hz)
    pub nep: f64,
    /// Active area (mm²)
    pub active_area_mm2: f64,
}

impl Default for HomodyneDetectorConfig {
    fn default() -> Self {
        Self {
            lo_power_mw: 10.0,
            efficiency: 0.95,
            electronic_noise: 1e-12, // V²/Hz
            bandwidth_hz: 10e6,
            saturation_power_mw: 100.0,
            pll_config: PLLConfig::default(),
            photodiode_config: PhotodiodeConfig::default(),
        }
    }
}

impl Default for PLLConfig {
    fn default() -> Self {
        Self {
            loop_bandwidth_hz: 1000.0,
            phase_noise_density: 1e-8, // rad²/Hz at 1 kHz
            lock_range: PI,
            acquisition_time_ms: 10.0,
        }
    }
}

impl Default for PhotodiodeConfig {
    fn default() -> Self {
        Self {
            responsivity: 0.8, // A/W at 1550 nm
            dark_current_na: 10.0,
            nep: 1e-14, // W/√Hz
            active_area_mm2: 1.0,
        }
    }
}

/// Homodyne detection result with detailed statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomodyneResult {
    /// Measured quadrature value
    pub quadrature_value: f64,
    /// Measurement phase
    pub phase: f64,
    /// Shot noise level
    pub shot_noise_level: f64,
    /// Electronic noise contribution
    pub electronic_noise_level: f64,
    /// Signal-to-noise ratio (dB)
    pub snr_db: f64,
    /// Clearance above shot noise (dB)
    pub squeezing_db: f64,
    /// Phase lock stability
    pub phase_stability: f64,
    /// Measurement fidelity
    pub fidelity: f64,
    /// Raw detector currents
    pub detector_currents: DetectorCurrents,
}

/// Raw currents from balanced detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorCurrents {
    /// Current from detector A (mA)
    pub current_a: f64,
    /// Current from detector B (mA)
    pub current_b: f64,
    /// Difference current (mA)
    pub difference_current: f64,
    /// Common mode current (mA)
    pub common_mode_current: f64,
}

/// Homodyne detector system
pub struct HomodyneDetector {
    /// Detector configuration
    config: HomodyneDetectorConfig,
    /// Current local oscillator phase
    lo_phase: f64,
    /// Phase lock status
    is_phase_locked: bool,
    /// Calibration data
    calibration: HomodyneCalibration,
    /// Measurement history
    measurement_history: Vec<HomodyneResult>,
}

/// Calibration data for homodyne detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomodyneCalibration {
    /// Visibility of the interference
    pub visibility: f64,
    /// DC offset in difference signal
    pub dc_offset: f64,
    /// Gain imbalance between detectors
    pub gain_imbalance: f64,
    /// Phase offset correction
    pub phase_offset: f64,
    /// Common mode rejection ratio (dB)
    pub cmrr_db: f64,
}

impl Default for HomodyneCalibration {
    fn default() -> Self {
        Self {
            visibility: 0.99,
            dc_offset: 0.001,      // mA
            gain_imbalance: 0.005, // 0.5%
            phase_offset: 0.02,    // rad
            cmrr_db: 60.0,
        }
    }
}

impl HomodyneDetector {
    /// Create a new homodyne detector
    pub fn new(config: HomodyneDetectorConfig) -> Self {
        Self {
            config,
            lo_phase: 0.0,
            is_phase_locked: false,
            calibration: HomodyneCalibration::default(),
            measurement_history: Vec::new(),
        }
    }

    /// Initialize and calibrate the detector
    pub async fn initialize(&mut self) -> DeviceResult<()> {
        println!("Initializing homodyne detector...");

        // Simulate initialization process
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Calibrate the system
        self.calibrate().await?;

        // Lock to local oscillator phase
        self.acquire_phase_lock().await?;

        println!("Homodyne detector initialized successfully");
        Ok(())
    }

    /// Calibrate the homodyne detector
    async fn calibrate(&mut self) -> DeviceResult<()> {
        println!("Calibrating homodyne detector...");

        // Simulate calibration measurements
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // Measure visibility with known coherent state
        self.calibration.visibility = 0.99 - 0.02 * rand::random::<f64>();

        // Measure DC offsets
        self.calibration.dc_offset = 0.001 * (rand::random::<f64>() - 0.5);

        // Measure gain imbalance
        self.calibration.gain_imbalance = 0.01 * (rand::random::<f64>() - 0.5);

        // Measure phase offset
        self.calibration.phase_offset = 0.05 * (rand::random::<f64>() - 0.5);

        println!(
            "Calibration complete: visibility = {:.3}",
            self.calibration.visibility
        );
        Ok(())
    }

    /// Acquire phase lock to local oscillator
    async fn acquire_phase_lock(&mut self) -> DeviceResult<()> {
        println!("Acquiring phase lock...");

        // Simulate phase lock acquisition
        let acquisition_time =
            std::time::Duration::from_millis(self.config.pll_config.acquisition_time_ms as u64);
        tokio::time::sleep(acquisition_time).await;

        self.is_phase_locked = true;
        println!("Phase lock acquired");
        Ok(())
    }

    /// Set the local oscillator phase
    pub async fn set_lo_phase(&mut self, phase: f64) -> DeviceResult<()> {
        if !self.is_phase_locked {
            return Err(DeviceError::DeviceNotInitialized(
                "Phase lock not acquired".to_string(),
            ));
        }

        // Apply phase with correction
        self.lo_phase = phase + self.calibration.phase_offset;

        // Simulate PLL settling time
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;

        Ok(())
    }

    /// Perform homodyne measurement
    pub async fn measure(
        &mut self,
        state: &mut GaussianState,
        mode: usize,
        phase: f64,
    ) -> DeviceResult<HomodyneResult> {
        if !self.is_phase_locked {
            return Err(DeviceError::DeviceNotInitialized(
                "Detector not initialized or phase lock lost".to_string(),
            ));
        }

        if mode >= state.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes",
                mode
            )));
        }

        // Set measurement phase
        self.set_lo_phase(phase).await?;

        // Get state parameters
        let corrected_phase = phase + self.calibration.phase_offset;
        let cos_phi = corrected_phase.cos();
        let sin_phi = corrected_phase.sin();

        let mean_x = state.mean_vector[2 * mode];
        let mean_p = state.mean_vector[2 * mode + 1];
        let theoretical_mean = cos_phi * mean_x + sin_phi * mean_p;

        let var_x = state.covariance_matrix[2 * mode][2 * mode];
        let var_p = state.covariance_matrix[2 * mode + 1][2 * mode + 1];
        let cov_xp = state.covariance_matrix[2 * mode][2 * mode + 1];

        let theoretical_variance =
            cos_phi.powi(2) * var_x + sin_phi.powi(2) * var_p + 2.0 * cos_phi * sin_phi * cov_xp;

        // Calculate noise contributions
        let shot_noise = self.calculate_shot_noise_level();
        let electronic_noise = self.calculate_electronic_noise_level();
        let phase_noise = self.calculate_phase_noise_contribution(theoretical_mean);

        let total_noise_variance = theoretical_variance / self.config.efficiency
            + shot_noise
            + electronic_noise
            + phase_noise;

        // Simulate measurement
        let distribution = Normal::new(theoretical_mean, total_noise_variance.sqrt())
            .map_err(|e| DeviceError::InvalidInput(format!("Distribution error: {}", e)))?;

        let mut rng = StdRng::seed_from_u64(rand::random::<u64>());
        let measured_value = distribution.sample(&mut rng);

        // Calculate detector currents
        let detector_currents = self.calculate_detector_currents(measured_value, theoretical_mean);

        // Calculate signal-to-noise ratio
        let signal_power = theoretical_mean.powi(2);
        let noise_power = total_noise_variance;
        let snr_db = 10.0 * (signal_power / noise_power).log10();

        // Calculate squeezing (clearance above shot noise)
        let squeezing_db = 10.0 * (shot_noise / theoretical_variance).log10();

        // Estimate phase stability
        let phase_stability = self.estimate_phase_stability();

        // Calculate measurement fidelity
        let fidelity =
            self.calculate_measurement_fidelity(signal_power, noise_power, phase_stability);

        let result = HomodyneResult {
            quadrature_value: measured_value,
            phase: corrected_phase,
            shot_noise_level: shot_noise,
            electronic_noise_level: electronic_noise,
            snr_db,
            squeezing_db,
            phase_stability,
            fidelity,
            detector_currents,
        };

        // Update state (simplified conditioning)
        state.condition_on_homodyne_measurement(mode, corrected_phase, measured_value)?;

        self.measurement_history.push(result.clone());
        Ok(result)
    }

    /// Calculate shot noise level
    fn calculate_shot_noise_level(&self) -> f64 {
        // Shot noise for homodyne detection: 2ℏω (in natural units, this is just 1)
        // Including detection efficiency
        1.0 / self.config.efficiency
    }

    /// Calculate electronic noise level
    fn calculate_electronic_noise_level(&self) -> f64 {
        // Convert electronic noise to quadrature units
        let current_noise = (2.0
            * 1.602e-19
            * self.config.photodiode_config.dark_current_na
            * 1e-9
            * self.config.bandwidth_hz)
            .sqrt(); // Shot noise from dark current

        let thermal_noise = (4.0 * 1.381e-23 * 300.0 * self.config.bandwidth_hz / 50.0).sqrt(); // Johnson noise

        let total_electronic_noise =
            (current_noise.powi(2) + thermal_noise.powi(2) + self.config.electronic_noise).sqrt();

        // Convert to quadrature variance units (simplified)
        total_electronic_noise
            / (self.config.photodiode_config.responsivity * (self.config.lo_power_mw * 1e-3).sqrt())
    }

    /// Calculate phase noise contribution
    fn calculate_phase_noise_contribution(&self, signal_amplitude: f64) -> f64 {
        // Phase noise converts to amplitude noise: Δx = signal * Δφ
        let phase_noise_variance =
            self.config.pll_config.phase_noise_density * self.config.bandwidth_hz;

        signal_amplitude.powi(2) * phase_noise_variance
    }

    /// Calculate detector currents
    fn calculate_detector_currents(
        &self,
        measured_value: f64,
        mean_signal: f64,
    ) -> DetectorCurrents {
        let lo_current =
            self.config.photodiode_config.responsivity * self.config.lo_power_mw * 1e-3; // mA

        // Signal contribution to photocurrent
        let signal_current = 2.0
            * (lo_current * self.config.photodiode_config.responsivity * mean_signal.abs() * 1e-3)
                .sqrt();

        // Balanced detection
        let current_a = lo_current
            + signal_current * 0.5
            + self.config.photodiode_config.dark_current_na * 1e-6;
        let current_b = lo_current - signal_current * 0.5
            + self.config.photodiode_config.dark_current_na * 1e-6;

        let difference_current = current_a - current_b;
        let common_mode_current = (current_a + current_b) / 2.0;

        DetectorCurrents {
            current_a,
            current_b,
            difference_current,
            common_mode_current,
        }
    }

    /// Estimate phase stability
    fn estimate_phase_stability(&self) -> f64 {
        // Phase stability based on PLL performance
        let frequency_noise = self.config.pll_config.phase_noise_density;
        let loop_bandwidth = self.config.pll_config.loop_bandwidth_hz;

        // RMS phase error
        (frequency_noise * loop_bandwidth).sqrt()
    }

    /// Calculate measurement fidelity
    fn calculate_measurement_fidelity(
        &self,
        signal_power: f64,
        noise_power: f64,
        phase_stability: f64,
    ) -> f64 {
        let snr = signal_power / noise_power;
        let phase_penalty = 1.0 / (1.0 + phase_stability.powi(2));
        let efficiency_penalty = self.config.efficiency;

        let fidelity = (snr / (1.0 + snr)) * phase_penalty * efficiency_penalty;
        fidelity.clamp(0.0, 1.0)
    }

    /// Get measurement statistics
    pub fn get_measurement_statistics(&self) -> HomodyneStatistics {
        if self.measurement_history.is_empty() {
            return HomodyneStatistics::default();
        }

        let total_measurements = self.measurement_history.len();

        let avg_snr = self
            .measurement_history
            .iter()
            .map(|r| r.snr_db)
            .sum::<f64>()
            / total_measurements as f64;

        let avg_squeezing = self
            .measurement_history
            .iter()
            .map(|r| r.squeezing_db)
            .sum::<f64>()
            / total_measurements as f64;

        let avg_fidelity = self
            .measurement_history
            .iter()
            .map(|r| r.fidelity)
            .sum::<f64>()
            / total_measurements as f64;

        let avg_phase_stability = self
            .measurement_history
            .iter()
            .map(|r| r.phase_stability)
            .sum::<f64>()
            / total_measurements as f64;

        HomodyneStatistics {
            total_measurements,
            average_snr_db: avg_snr,
            average_squeezing_db: avg_squeezing,
            average_fidelity: avg_fidelity,
            average_phase_stability: avg_phase_stability,
            is_phase_locked: self.is_phase_locked,
            detector_efficiency: self.config.efficiency,
        }
    }

    /// Clear measurement history
    pub fn clear_history(&mut self) {
        self.measurement_history.clear();
    }

    /// Get calibration data
    pub fn get_calibration(&self) -> &HomodyneCalibration {
        &self.calibration
    }
}

/// Statistics for homodyne detector performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomodyneStatistics {
    pub total_measurements: usize,
    pub average_snr_db: f64,
    pub average_squeezing_db: f64,
    pub average_fidelity: f64,
    pub average_phase_stability: f64,
    pub is_phase_locked: bool,
    pub detector_efficiency: f64,
}

impl Default for HomodyneStatistics {
    fn default() -> Self {
        Self {
            total_measurements: 0,
            average_snr_db: 0.0,
            average_squeezing_db: 0.0,
            average_fidelity: 0.0,
            average_phase_stability: 0.0,
            is_phase_locked: false,
            detector_efficiency: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_homodyne_detector_creation() {
        let config = HomodyneDetectorConfig::default();
        let detector = HomodyneDetector::new(config);
        assert!(!detector.is_phase_locked);
        assert_eq!(detector.measurement_history.len(), 0);
    }

    #[tokio::test]
    async fn test_detector_initialization() {
        let config = HomodyneDetectorConfig::default();
        let mut detector = HomodyneDetector::new(config);

        detector.initialize().await.unwrap();
        assert!(detector.is_phase_locked);
    }

    #[tokio::test]
    async fn test_phase_setting() {
        let config = HomodyneDetectorConfig::default();
        let mut detector = HomodyneDetector::new(config);
        detector.initialize().await.unwrap();

        detector.set_lo_phase(PI / 4.0).await.unwrap();
        assert!((detector.lo_phase - PI / 4.0).abs() < 0.1); // Within calibration offset
    }

    #[tokio::test]
    async fn test_homodyne_measurement() {
        let config = HomodyneDetectorConfig::default();
        let mut detector = HomodyneDetector::new(config);
        detector.initialize().await.unwrap();

        let mut state = GaussianState::coherent_state(1, vec![Complex::new(2.0, 0.0)]).unwrap();

        let result = detector.measure(&mut state, 0, 0.0).await.unwrap();

        assert!(result.quadrature_value.is_finite());
        assert!(result.fidelity > 0.0);
        assert!(result.snr_db.is_finite());
        assert_eq!(detector.measurement_history.len(), 1);
    }

    #[tokio::test]
    async fn test_squeezing_measurement() {
        let config = HomodyneDetectorConfig::default();
        let mut detector = HomodyneDetector::new(config);
        detector.initialize().await.unwrap();

        let mut state = GaussianState::squeezed_vacuum_state(1, vec![1.0], vec![0.0]).unwrap();

        let result = detector.measure(&mut state, 0, 0.0).await.unwrap();

        // Should observe squeezing in x quadrature
        assert!(result.squeezing_db < 0.0); // Below shot noise
    }

    #[test]
    fn test_noise_calculations() {
        let config = HomodyneDetectorConfig::default();
        let detector = HomodyneDetector::new(config);

        let shot_noise = detector.calculate_shot_noise_level();
        assert!(shot_noise > 0.0);

        let electronic_noise = detector.calculate_electronic_noise_level();
        assert!(electronic_noise > 0.0);
    }

    #[test]
    fn test_detector_current_calculation() {
        let config = HomodyneDetectorConfig::default();
        let detector = HomodyneDetector::new(config);

        let currents = detector.calculate_detector_currents(1.0, 0.5);
        assert!(currents.current_a > 0.0);
        assert!(currents.current_b > 0.0);
        assert!(currents.difference_current != 0.0);
    }

    #[test]
    fn test_statistics() {
        let config = HomodyneDetectorConfig::default();
        let detector = HomodyneDetector::new(config);

        let stats = detector.get_measurement_statistics();
        assert_eq!(stats.total_measurements, 0);
        assert!(!stats.is_phase_locked);
    }
}
