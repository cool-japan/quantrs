//! Heterodyne detection for continuous variable quantum systems
//!
//! This module implements heterodyne detection, which simultaneously measures both
//! quadratures of the quantum field using two local oscillators in quadrature.

use super::{CVDeviceConfig, Complex, GaussianState};
use crate::{DeviceError, DeviceResult};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, RandNormal};
// Alias for backward compatibility
type Normal<T> = RandNormal<T>;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Heterodyne detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterodyneDetectorConfig {
    /// Local oscillator power (mW)
    pub lo_power_mw: f64,
    /// Detection efficiency
    pub efficiency: f64,
    /// Electronic noise (V²/Hz)
    pub electronic_noise: f64,
    /// Detector bandwidth (Hz)
    pub bandwidth_hz: f64,
    /// Intermediate frequency (Hz)
    pub intermediate_frequency_hz: f64,
    /// Phase lock loop configurations for both LOs
    pub pll_x_config: PLLConfig,
    pub pll_p_config: PLLConfig,
    /// Photodiode specifications
    pub photodiode_config: PhotodiodeConfig,
    /// IQ demodulator config
    pub iq_demod_config: IQDemodulatorConfig,
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

/// IQ demodulator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IQDemodulatorConfig {
    /// IQ imbalance (amplitude)
    pub amplitude_imbalance: f64,
    /// IQ imbalance (phase, rad)
    pub phase_imbalance: f64,
    /// DC offset I channel (V)
    pub dc_offset_i: f64,
    /// DC offset Q channel (V)
    pub dc_offset_q: f64,
    /// Low-pass filter cutoff (Hz)
    pub lpf_cutoff_hz: f64,
}

impl Default for HeterodyneDetectorConfig {
    fn default() -> Self {
        Self {
            lo_power_mw: 10.0,
            efficiency: 0.95,
            electronic_noise: 1e-12, // V²/Hz
            bandwidth_hz: 10e6,
            intermediate_frequency_hz: 100e6, // 100 MHz IF
            pll_x_config: PLLConfig::default(),
            pll_p_config: PLLConfig::default(),
            photodiode_config: PhotodiodeConfig::default(),
            iq_demod_config: IQDemodulatorConfig::default(),
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

impl Default for IQDemodulatorConfig {
    fn default() -> Self {
        Self {
            amplitude_imbalance: 0.01, // 1%
            phase_imbalance: 0.02,     // ~1.1 degrees
            dc_offset_i: 0.001,        // 1 mV
            dc_offset_q: 0.001,        // 1 mV
            lpf_cutoff_hz: 1e6,        // 1 MHz
        }
    }
}

/// Heterodyne detection result with detailed statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterodyneResult {
    /// Measured complex amplitude
    pub complex_amplitude: Complex,
    /// I and Q quadrature values
    pub i_quadrature: f64,
    pub q_quadrature: f64,
    /// Shot noise level (each quadrature)
    pub shot_noise_level: f64,
    /// Electronic noise contribution
    pub electronic_noise_level: f64,
    /// Signal-to-noise ratio (dB)
    pub snr_db: f64,
    /// Phase measurement uncertainty (rad)
    pub phase_uncertainty: f64,
    /// Amplitude measurement uncertainty
    pub amplitude_uncertainty: f64,
    /// IQ imbalance correction applied
    pub iq_correction: IQCorrection,
    /// Measurement fidelity
    pub fidelity: f64,
    /// Raw detector data
    pub detector_data: HeterodyneDetectorData,
}

/// IQ imbalance correction data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IQCorrection {
    /// Amplitude correction factor
    pub amplitude_correction: f64,
    /// Phase correction (rad)
    pub phase_correction: f64,
    /// DC offset correction I
    pub dc_offset_i: f64,
    /// DC offset correction Q
    pub dc_offset_q: f64,
}

/// Raw detector data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterodyneDetectorData {
    /// IF signal amplitude (V)
    pub if_amplitude: f64,
    /// IF signal phase (rad)
    pub if_phase: f64,
    /// Raw I signal (V)
    pub raw_i_signal: f64,
    /// Raw Q signal (V)
    pub raw_q_signal: f64,
    /// Local oscillator powers (mW)
    pub lo_powers: (f64, f64),
}

/// Heterodyne detector system
pub struct HeterodyneDetector {
    /// Detector configuration
    config: HeterodyneDetectorConfig,
    /// Local oscillator phases (x and p LOs)
    lo_phases: (f64, f64),
    /// Phase lock status for both PLLs
    phase_lock_status: (bool, bool),
    /// Calibration data
    calibration: HeterodyneCalibration,
    /// Measurement history
    measurement_history: Vec<HeterodyneResult>,
}

/// Calibration data for heterodyne detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterodyneCalibration {
    /// IQ calibration matrix [[a, b], [c, d]]
    pub iq_matrix: [[f64; 2]; 2],
    /// DC offset corrections
    pub dc_offsets: (f64, f64),
    /// Phase calibration between LOs
    pub relative_phase_offset: f64,
    /// Amplitude calibration factors
    pub amplitude_factors: (f64, f64),
    /// Common mode rejection ratio (dB)
    pub cmrr_db: f64,
}

impl Default for HeterodyneCalibration {
    fn default() -> Self {
        Self {
            iq_matrix: [[1.0, 0.0], [0.0, 1.0]], // Identity matrix
            dc_offsets: (0.0, 0.0),
            relative_phase_offset: 0.0,
            amplitude_factors: (1.0, 1.0),
            cmrr_db: 60.0,
        }
    }
}

impl HeterodyneDetector {
    /// Create a new heterodyne detector
    pub fn new(config: HeterodyneDetectorConfig) -> Self {
        Self {
            config,
            lo_phases: (0.0, PI / 2.0), // X and P quadratures
            phase_lock_status: (false, false),
            calibration: HeterodyneCalibration::default(),
            measurement_history: Vec::new(),
        }
    }

    /// Initialize and calibrate the detector
    pub async fn initialize(&mut self) -> DeviceResult<()> {
        println!("Initializing heterodyne detector...");

        // Simulate initialization process
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;

        // Calibrate the system
        self.calibrate().await?;

        // Lock both local oscillators
        self.acquire_phase_locks().await?;

        println!("Heterodyne detector initialized successfully");
        Ok(())
    }

    /// Calibrate the heterodyne detector
    async fn calibrate(&mut self) -> DeviceResult<()> {
        println!("Calibrating heterodyne detector...");

        // Simulate calibration measurements
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;

        // Measure IQ imbalance with known coherent states
        let amplitude_imbalance = self.config.iq_demod_config.amplitude_imbalance;
        let phase_imbalance = self.config.iq_demod_config.phase_imbalance;

        // Build correction matrix
        let cos_phi = phase_imbalance.cos();
        let sin_phi = phase_imbalance.sin();
        let alpha = 1.0 + amplitude_imbalance;

        // Inverse of imbalance matrix
        let det = alpha * cos_phi;
        self.calibration.iq_matrix = [[cos_phi / det, -sin_phi / det], [0.0, 1.0 / det]];

        // Measure DC offsets
        self.calibration.dc_offsets = (
            self.config.iq_demod_config.dc_offset_i + 0.0005 * (thread_rng().gen::<f64>() - 0.5),
            self.config.iq_demod_config.dc_offset_q + 0.0005 * (thread_rng().gen::<f64>() - 0.5),
        );

        // Measure relative phase between LOs
        self.calibration.relative_phase_offset = 0.02 * (thread_rng().gen::<f64>() - 0.5);

        // Measure amplitude factors
        self.calibration.amplitude_factors = (
            1.0 + 0.01 * (thread_rng().gen::<f64>() - 0.5),
            1.0 + 0.01 * (thread_rng().gen::<f64>() - 0.5),
        );

        println!("Calibration complete");
        Ok(())
    }

    /// Acquire phase lock for both local oscillators
    async fn acquire_phase_locks(&mut self) -> DeviceResult<()> {
        println!("Acquiring phase locks for both LOs...");

        // Lock X quadrature LO
        let x_acquisition_time =
            std::time::Duration::from_millis(self.config.pll_x_config.acquisition_time_ms as u64);
        tokio::time::sleep(x_acquisition_time).await;
        self.phase_lock_status.0 = true;
        println!("X quadrature LO locked");

        // Lock P quadrature LO (in quadrature)
        let p_acquisition_time =
            std::time::Duration::from_millis(self.config.pll_p_config.acquisition_time_ms as u64);
        tokio::time::sleep(p_acquisition_time).await;
        self.phase_lock_status.1 = true;
        println!("P quadrature LO locked");

        Ok(())
    }

    /// Check if both phase locks are acquired
    pub fn is_phase_locked(&self) -> bool {
        self.phase_lock_status.0 && self.phase_lock_status.1
    }

    /// Perform heterodyne measurement
    pub async fn measure(
        &mut self,
        state: &mut GaussianState,
        mode: usize,
    ) -> DeviceResult<HeterodyneResult> {
        if !self.is_phase_locked() {
            return Err(DeviceError::DeviceNotInitialized(
                "Phase locks not acquired".to_string(),
            ));
        }

        if mode >= state.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes",
                mode
            )));
        }

        // Get state parameters with phase corrections
        let mean_x = state.mean_vector[2 * mode];
        let mean_p = state.mean_vector[2 * mode + 1];
        let var_x = state.covariancematrix[2 * mode][2 * mode];
        let var_p = state.covariancematrix[2 * mode + 1][2 * mode + 1];
        let cov_xp = state.covariancematrix[2 * mode][2 * mode + 1];

        // Calculate noise contributions
        let shot_noise = self.calculate_shot_noise_level();
        let electronic_noise = self.calculate_electronic_noise_level();
        let phase_noise_x =
            self.calculate_phase_noise_contribution(mean_x, &self.config.pll_x_config);
        let phase_noise_p =
            self.calculate_phase_noise_contribution(mean_p, &self.config.pll_p_config);

        // Total noise variances for each quadrature
        let noise_var_x =
            var_x / self.config.efficiency + shot_noise + electronic_noise + phase_noise_x;
        let noise_var_p =
            var_p / self.config.efficiency + shot_noise + electronic_noise + phase_noise_p;

        // Sample both quadratures
        let dist_x = Normal::new(mean_x, noise_var_x.sqrt())
            .map_err(|e| DeviceError::InvalidInput(format!("Distribution error: {}", e)))?;
        let dist_p = Normal::new(mean_p, noise_var_p.sqrt())
            .map_err(|e| DeviceError::InvalidInput(format!("Distribution error: {}", e)))?;

        let mut rng = StdRng::seed_from_u64(thread_rng().gen::<u64>());
        let raw_i = dist_x.sample(&mut rng);
        let raw_q = dist_p.sample(&mut rng);

        // Apply IQ corrections
        let iq_correction = self.apply_iq_correction(raw_i, raw_q);
        let corrected_i = iq_correction.corrected_i;
        let corrected_q = iq_correction.corrected_q;

        // Convert to complex amplitude
        let complex_amplitude = Complex::new(
            (corrected_i + corrected_q * Complex::i().real) / (2.0_f64).sqrt(),
            (corrected_q - corrected_i * Complex::i().imag) / (2.0_f64).sqrt(),
        );

        // Calculate uncertainties
        let amplitude_uncertainty = (noise_var_x + noise_var_p).sqrt() / 2.0;
        let phase_uncertainty = if complex_amplitude.magnitude() > 0.0 {
            amplitude_uncertainty / complex_amplitude.magnitude()
        } else {
            PI // Maximum uncertainty for zero amplitude
        };

        // Calculate SNR
        let signal_power = complex_amplitude.magnitude().powi(2);
        let noise_power = (noise_var_x + noise_var_p) / 2.0;
        let snr_db = 10.0 * (signal_power / noise_power).log10();

        // Generate detector data
        let detector_data = self.generate_detector_data(raw_i, raw_q, complex_amplitude);

        // Calculate measurement fidelity
        let fidelity =
            self.calculate_measurement_fidelity(signal_power, noise_power, phase_uncertainty);

        let result = HeterodyneResult {
            complex_amplitude,
            i_quadrature: corrected_i,
            q_quadrature: corrected_q,
            shot_noise_level: shot_noise,
            electronic_noise_level: electronic_noise,
            snr_db,
            phase_uncertainty,
            amplitude_uncertainty,
            iq_correction: iq_correction.correction_data,
            fidelity,
            detector_data,
        };

        // Update state (heterodyne destroys the mode)
        state.condition_on_heterodyne_measurement(mode, complex_amplitude)?;

        self.measurement_history.push(result.clone());
        Ok(result)
    }

    /// Apply IQ correction
    fn apply_iq_correction(&self, raw_i: f64, raw_q: f64) -> IQCorrectionResult {
        // Remove DC offsets
        let dc_corrected_i = raw_i - self.calibration.dc_offsets.0;
        let dc_corrected_q = raw_q - self.calibration.dc_offsets.1;

        // Apply calibration matrix
        let matrix = &self.calibration.iq_matrix;
        let corrected_i = matrix[0][0] * dc_corrected_i + matrix[0][1] * dc_corrected_q;
        let corrected_q = matrix[1][0] * dc_corrected_i + matrix[1][1] * dc_corrected_q;

        // Apply amplitude corrections
        let final_i = corrected_i * self.calibration.amplitude_factors.0;
        let final_q = corrected_q * self.calibration.amplitude_factors.1;

        IQCorrectionResult {
            corrected_i: final_i,
            corrected_q: final_q,
            correction_data: IQCorrection {
                amplitude_correction: (self.calibration.amplitude_factors.0
                    + self.calibration.amplitude_factors.1)
                    / 2.0,
                phase_correction: self.calibration.relative_phase_offset,
                dc_offset_i: self.calibration.dc_offsets.0,
                dc_offset_q: self.calibration.dc_offsets.1,
            },
        }
    }

    /// Calculate shot noise level (heterodyne has 3dB penalty)
    fn calculate_shot_noise_level(&self) -> f64 {
        // Heterodyne has 3dB penalty compared to homodyne: 2ℏω per quadrature
        2.0 / self.config.efficiency
    }

    /// Calculate electronic noise level
    fn calculate_electronic_noise_level(&self) -> f64 {
        // Similar to homodyne but accounting for IF processing
        let current_noise = (2.0
            * 1.602e-19
            * self.config.photodiode_config.dark_current_na
            * 1e-9
            * self.config.bandwidth_hz)
            .sqrt();

        let thermal_noise = (4.0 * 1.381e-23 * 300.0 * self.config.bandwidth_hz / 50.0).sqrt();

        // Additional noise from IQ demodulation
        let iq_noise = self.config.electronic_noise.sqrt();

        let total_electronic_noise =
            (current_noise.powi(2) + thermal_noise.powi(2) + iq_noise.powi(2)).sqrt();

        // Convert to quadrature variance units
        total_electronic_noise
            / (self.config.photodiode_config.responsivity * (self.config.lo_power_mw * 1e-3).sqrt())
    }

    /// Calculate phase noise contribution
    fn calculate_phase_noise_contribution(
        &self,
        signal_amplitude: f64,
        pll_config: &PLLConfig,
    ) -> f64 {
        let phase_noise_variance = pll_config.phase_noise_density * self.config.bandwidth_hz;
        signal_amplitude.powi(2) * phase_noise_variance
    }

    /// Generate detector data
    fn generate_detector_data(
        &self,
        raw_i: f64,
        raw_q: f64,
        complex_amplitude: Complex,
    ) -> HeterodyneDetectorData {
        let if_amplitude = (raw_i.powi(2) + raw_q.powi(2)).sqrt();
        let if_phase = raw_q.atan2(raw_i);

        HeterodyneDetectorData {
            if_amplitude,
            if_phase,
            raw_i_signal: raw_i,
            raw_q_signal: raw_q,
            lo_powers: (self.config.lo_power_mw, self.config.lo_power_mw),
        }
    }

    /// Calculate measurement fidelity
    fn calculate_measurement_fidelity(
        &self,
        signal_power: f64,
        noise_power: f64,
        phase_uncertainty: f64,
    ) -> f64 {
        let snr = signal_power / noise_power;
        let phase_penalty = 1.0 / (1.0 + phase_uncertainty.powi(2));
        let efficiency_penalty = self.config.efficiency;

        // Additional penalty for heterodyne 3dB penalty
        let heterodyne_penalty = 0.5;

        let fidelity =
            (snr / (1.0 + snr)) * phase_penalty * efficiency_penalty * heterodyne_penalty;
        fidelity.clamp(0.0, 1.0)
    }

    /// Get measurement statistics
    pub fn get_measurement_statistics(&self) -> HeterodyneStatistics {
        if self.measurement_history.is_empty() {
            return HeterodyneStatistics::default();
        }

        let total_measurements = self.measurement_history.len();

        let avg_snr = self
            .measurement_history
            .iter()
            .map(|r| r.snr_db)
            .sum::<f64>()
            / total_measurements as f64;

        let avg_amplitude_uncertainty = self
            .measurement_history
            .iter()
            .map(|r| r.amplitude_uncertainty)
            .sum::<f64>()
            / total_measurements as f64;

        let avg_phase_uncertainty = self
            .measurement_history
            .iter()
            .map(|r| r.phase_uncertainty)
            .sum::<f64>()
            / total_measurements as f64;

        let avg_fidelity = self
            .measurement_history
            .iter()
            .map(|r| r.fidelity)
            .sum::<f64>()
            / total_measurements as f64;

        HeterodyneStatistics {
            total_measurements,
            average_snr_db: avg_snr,
            average_amplitude_uncertainty: avg_amplitude_uncertainty,
            average_phase_uncertainty: avg_phase_uncertainty,
            average_fidelity: avg_fidelity,
            phase_lock_status: self.phase_lock_status,
            detector_efficiency: self.config.efficiency,
        }
    }

    /// Clear measurement history
    pub fn clear_history(&mut self) {
        self.measurement_history.clear();
    }

    /// Get calibration data
    pub fn get_calibration(&self) -> &HeterodyneCalibration {
        &self.calibration
    }
}

/// Result of IQ correction
struct IQCorrectionResult {
    corrected_i: f64,
    corrected_q: f64,
    correction_data: IQCorrection,
}

/// Statistics for heterodyne detector performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterodyneStatistics {
    pub total_measurements: usize,
    pub average_snr_db: f64,
    pub average_amplitude_uncertainty: f64,
    pub average_phase_uncertainty: f64,
    pub average_fidelity: f64,
    pub phase_lock_status: (bool, bool),
    pub detector_efficiency: f64,
}

impl Default for HeterodyneStatistics {
    fn default() -> Self {
        Self {
            total_measurements: 0,
            average_snr_db: 0.0,
            average_amplitude_uncertainty: 0.0,
            average_phase_uncertainty: 0.0,
            average_fidelity: 0.0,
            phase_lock_status: (false, false),
            detector_efficiency: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_heterodyne_detector_creation() {
        let config = HeterodyneDetectorConfig::default();
        let detector = HeterodyneDetector::new(config);
        assert!(!detector.is_phase_locked());
        assert_eq!(detector.measurement_history.len(), 0);
    }

    #[tokio::test]
    async fn test_detector_initialization() {
        let config = HeterodyneDetectorConfig::default();
        let mut detector = HeterodyneDetector::new(config);

        detector.initialize().await.unwrap();
        assert!(detector.is_phase_locked());
    }

    #[tokio::test]
    async fn test_heterodyne_measurement() {
        let config = HeterodyneDetectorConfig::default();
        let mut detector = HeterodyneDetector::new(config);
        detector.initialize().await.unwrap();

        let mut state = GaussianState::coherent_state(1, vec![Complex::new(2.0, 1.0)]).unwrap();

        let result = detector.measure(&mut state, 0).await.unwrap();

        assert!(result.complex_amplitude.magnitude() > 0.0);
        assert!(result.fidelity > 0.0);
        assert!(result.snr_db.is_finite());
        assert_eq!(detector.measurement_history.len(), 1);
    }

    #[tokio::test]
    async fn test_vacuum_measurement() {
        let config = HeterodyneDetectorConfig::default();
        let mut detector = HeterodyneDetector::new(config);
        detector.initialize().await.unwrap();

        let mut state = GaussianState::vacuum_state(1);

        let result = detector.measure(&mut state, 0).await.unwrap();

        // Vacuum should have small but finite amplitude due to noise
        assert!(result.complex_amplitude.magnitude() >= 0.0);
        assert!(result.amplitude_uncertainty > 0.0);
        assert!(result.phase_uncertainty > 0.0);
    }

    #[test]
    fn test_iq_correction() {
        let config = HeterodyneDetectorConfig::default();
        let detector = HeterodyneDetector::new(config);

        let correction_result = detector.apply_iq_correction(1.0, 0.5);
        assert!(correction_result.corrected_i.is_finite());
        assert!(correction_result.corrected_q.is_finite());
    }

    #[test]
    fn test_noise_calculations() {
        let config = HeterodyneDetectorConfig::default();
        let efficiency = config.efficiency;
        let detector = HeterodyneDetector::new(config);

        let shot_noise = detector.calculate_shot_noise_level();
        assert!(shot_noise > 0.0);
        // Heterodyne should have higher shot noise than homodyne
        assert!(shot_noise >= 2.0 / efficiency);

        let electronic_noise = detector.calculate_electronic_noise_level();
        assert!(electronic_noise > 0.0);
    }

    #[test]
    fn test_detector_data_generation() {
        let config = HeterodyneDetectorConfig::default();
        let detector = HeterodyneDetector::new(config);

        let complex_amp = Complex::new(1.0, 0.5);
        let data = detector.generate_detector_data(1.0, 0.5, complex_amp);

        assert!(data.if_amplitude > 0.0);
        assert!(data.if_phase.is_finite());
        assert_eq!(data.raw_i_signal, 1.0);
        assert_eq!(data.raw_q_signal, 0.5);
    }

    #[test]
    fn test_statistics() {
        let config = HeterodyneDetectorConfig::default();
        let detector = HeterodyneDetector::new(config);

        let stats = detector.get_measurement_statistics();
        assert_eq!(stats.total_measurements, 0);
        assert_eq!(stats.phase_lock_status, (false, false));
    }
}
