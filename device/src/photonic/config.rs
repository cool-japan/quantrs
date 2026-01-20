//! Configuration management for photonic quantum devices

use crate::DeviceResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Photonic system types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhotonicSystem {
    /// Continuous variable quantum computing
    ContinuousVariable,
    /// Gate-based photonic quantum computing
    GateBased,
    /// Measurement-based quantum computing
    MeasurementBased,
    /// Hybrid photonic systems
    Hybrid,
}

/// Advanced configuration for photonic quantum devices
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhotonicConfig {
    /// Base system configuration
    pub system: SystemConfig,
    /// Hardware configuration
    pub hardware: HardwareConfig,
    /// Measurement configuration
    pub measurement: MeasurementConfig,
    /// Error correction settings
    pub error_correction: ErrorCorrectionConfig,
    /// Optimization settings
    pub optimization: OptimizationConfig,
}

/// System configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Photonic system type
    pub system_type: PhotonicSystem,
    /// Number of optical modes
    pub mode_count: usize,
    /// Cutoff dimension for Fock space
    pub cutoff_dimension: usize,
    /// Maximum photon number
    pub max_photon_number: usize,
    /// Squeezing parameter range
    pub squeezing_range: (f64, f64),
    /// Displacement amplitude range
    pub displacement_range: (f64, f64),
}

/// Hardware configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HardwareConfig {
    /// Laser configuration
    pub laser: LaserConfig,
    /// Detector configuration
    pub detector: DetectorConfig,
    /// Beam splitter configuration
    pub beam_splitter: BeamSplitterConfig,
    /// Phase shifter configuration
    pub phase_shifter: PhaseShifterConfig,
    /// Squeezer configuration
    pub squeezer: SqueezerConfig,
}

/// Laser configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserConfig {
    /// Wavelength (nm)
    pub wavelength: f64,
    /// Power (mW)
    pub power: f64,
    /// Linewidth (Hz)
    pub linewidth: f64,
    /// Coherence time (ns)
    pub coherence_time: f64,
    /// Intensity noise (dB/Hz)
    pub intensity_noise: f64,
    /// Phase noise (rad²/Hz)
    pub phase_noise: f64,
}

/// Detector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorConfig {
    /// Detection efficiency
    pub efficiency: f64,
    /// Dark count rate (Hz)
    pub dark_count_rate: f64,
    /// Dead time (ns)
    pub dead_time: f64,
    /// Timing jitter (ps)
    pub timing_jitter: f64,
    /// Number resolution
    pub number_resolution: bool,
    /// Quantum efficiency
    pub quantum_efficiency: f64,
}

/// Beam splitter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamSplitterConfig {
    /// Transmission coefficient
    pub transmission: f64,
    /// Reflection coefficient
    pub reflection: f64,
    /// Loss coefficient
    pub loss: f64,
    /// Phase shift (radians)
    pub phase_shift: f64,
    /// Bandwidth (Hz)
    pub bandwidth: f64,
}

/// Phase shifter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseShifterConfig {
    /// Maximum phase shift (radians)
    pub max_phase_shift: f64,
    /// Phase resolution (radians)
    pub phase_resolution: f64,
    /// Response time (ns)
    pub response_time: f64,
    /// Drift rate (rad/s)
    pub drift_rate: f64,
}

/// Squeezer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqueezerConfig {
    /// Maximum squeezing parameter
    pub max_squeezing: f64,
    /// Squeezing bandwidth (Hz)
    pub bandwidth: f64,
    /// Anti-squeezing penalty
    pub anti_squeezing_penalty: f64,
    /// Pump power (mW)
    pub pump_power: f64,
}

/// Measurement configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MeasurementConfig {
    /// Homodyne detection settings
    pub homodyne: HomodyneConfig,
    /// Heterodyne detection settings
    pub heterodyne: HeterodyneConfig,
    /// Photon counting settings
    pub photon_counting: PhotonCountingConfig,
    /// Tomography settings
    pub tomography: TomographyConfig,
}

/// Homodyne detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomodyneConfig {
    /// Local oscillator power (mW)
    pub lo_power: f64,
    /// Detection efficiency
    pub efficiency: f64,
    /// Electronic noise (V²/Hz)
    pub electronic_noise: f64,
    /// Shot noise clearance (dB)
    pub shot_noise_clearance: f64,
}

/// Heterodyne detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterodyneConfig {
    /// Local oscillator power (mW)
    pub lo_power: f64,
    /// Intermediate frequency (Hz)
    pub intermediate_frequency: f64,
    /// Detection efficiency
    pub efficiency: f64,
    /// Phase resolution (radians)
    pub phase_resolution: f64,
}

/// Photon counting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonCountingConfig {
    /// Maximum count rate (Hz)
    pub max_count_rate: f64,
    /// Detection window (ns)
    pub detection_window: f64,
    /// Coincidence window (ns)
    pub coincidence_window: f64,
    /// Background count rate (Hz)
    pub background_rate: f64,
}

/// Quantum state tomography configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TomographyConfig {
    /// Number of measurement settings
    pub measurement_settings: usize,
    /// Shots per setting
    pub shots_per_setting: usize,
    /// Reconstruction method
    pub reconstruction_method: TomographyMethod,
    /// Regularization parameter
    pub regularization: f64,
}

/// Tomography reconstruction methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TomographyMethod {
    /// Maximum likelihood estimation
    MaximumLikelihood,
    /// Linear inversion
    LinearInversion,
    /// Bayesian inference
    Bayesian,
    /// Compressed sensing
    CompressedSensing,
}

/// Error correction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCorrectionConfig {
    /// Enable error correction
    pub enabled: bool,
    /// Error correction scheme
    pub scheme: ErrorCorrectionScheme,
    /// Loss tolerance
    pub loss_tolerance: f64,
    /// Phase error tolerance
    pub phase_error_tolerance: f64,
    /// Syndrome extraction rounds
    pub syndrome_rounds: usize,
}

/// Error correction schemes for photonic systems
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCorrectionScheme {
    /// No error correction
    None,
    /// GKP (Gottesman-Kitaev-Preskill) codes
    GKP,
    /// Cat codes
    Cat,
    /// Binomial codes
    Binomial,
    /// Four-component cat codes
    FourComponentCat,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable gate optimization
    pub gate_optimization: bool,
    /// Enable measurement optimization
    pub measurement_optimization: bool,
    /// Enable loss compensation
    pub loss_compensation: bool,
    /// Optimization algorithm
    pub algorithm: OptimizationAlgorithm,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
}

/// Optimization algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationAlgorithm {
    /// Gradient descent
    GradientDescent,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Genetic algorithm
    Genetic,
    /// Particle swarm optimization
    ParticleSwarm,
    /// Bayesian optimization
    Bayesian,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            system_type: PhotonicSystem::ContinuousVariable,
            mode_count: 8,
            cutoff_dimension: 10,
            max_photon_number: 50,
            squeezing_range: (-2.0, 2.0),
            displacement_range: (-5.0, 5.0),
        }
    }
}

impl Default for LaserConfig {
    fn default() -> Self {
        Self {
            wavelength: 1550.0,
            power: 10.0,
            linewidth: 100.0,
            coherence_time: 10.0,
            intensity_noise: -140.0,
            phase_noise: 1e-6,
        }
    }
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            efficiency: 0.9,
            dark_count_rate: 100.0,
            dead_time: 50.0,
            timing_jitter: 100.0,
            number_resolution: true,
            quantum_efficiency: 0.85,
        }
    }
}

impl Default for BeamSplitterConfig {
    fn default() -> Self {
        Self {
            transmission: 0.5,
            reflection: 0.5,
            loss: 0.01,
            phase_shift: 0.0,
            bandwidth: 1e12,
        }
    }
}

impl Default for PhaseShifterConfig {
    fn default() -> Self {
        Self {
            max_phase_shift: 2.0 * std::f64::consts::PI,
            phase_resolution: 0.01,
            response_time: 1.0,
            drift_rate: 1e-6,
        }
    }
}

impl Default for SqueezerConfig {
    fn default() -> Self {
        Self {
            max_squeezing: 10.0,
            bandwidth: 1e9,
            anti_squeezing_penalty: 3.0,
            pump_power: 100.0,
        }
    }
}

impl Default for HomodyneConfig {
    fn default() -> Self {
        Self {
            lo_power: 1.0,
            efficiency: 0.9,
            electronic_noise: 1e-8,
            shot_noise_clearance: 10.0,
        }
    }
}

impl Default for HeterodyneConfig {
    fn default() -> Self {
        Self {
            lo_power: 1.0,
            intermediate_frequency: 1e6,
            efficiency: 0.85,
            phase_resolution: 0.01,
        }
    }
}

impl Default for PhotonCountingConfig {
    fn default() -> Self {
        Self {
            max_count_rate: 1e6,
            detection_window: 10.0,
            coincidence_window: 1.0,
            background_rate: 100.0,
        }
    }
}

impl Default for TomographyConfig {
    fn default() -> Self {
        Self {
            measurement_settings: 16,
            shots_per_setting: 10000,
            reconstruction_method: TomographyMethod::MaximumLikelihood,
            regularization: 1e-6,
        }
    }
}

impl Default for ErrorCorrectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            scheme: ErrorCorrectionScheme::None,
            loss_tolerance: 0.1,
            phase_error_tolerance: 0.05,
            syndrome_rounds: 3,
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            gate_optimization: true,
            measurement_optimization: true,
            loss_compensation: true,
            algorithm: OptimizationAlgorithm::GradientDescent,
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}

/// Configuration builder for photonic devices
pub struct PhotonicConfigBuilder {
    config: PhotonicConfig,
}

impl PhotonicConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: PhotonicConfig::default(),
        }
    }

    #[must_use]
    pub const fn system_type(mut self, system_type: PhotonicSystem) -> Self {
        self.config.system.system_type = system_type;
        self
    }

    #[must_use]
    pub const fn mode_count(mut self, count: usize) -> Self {
        self.config.system.mode_count = count;
        self
    }

    #[must_use]
    pub const fn cutoff_dimension(mut self, cutoff: usize) -> Self {
        self.config.system.cutoff_dimension = cutoff;
        self
    }

    #[must_use]
    pub const fn laser_wavelength(mut self, wavelength: f64) -> Self {
        self.config.hardware.laser.wavelength = wavelength;
        self
    }

    #[must_use]
    pub const fn detection_efficiency(mut self, efficiency: f64) -> Self {
        self.config.hardware.detector.efficiency = efficiency;
        self
    }

    #[must_use]
    pub const fn enable_error_correction(mut self, enabled: bool) -> Self {
        self.config.error_correction.enabled = enabled;
        self
    }

    #[must_use]
    pub const fn error_correction_scheme(mut self, scheme: ErrorCorrectionScheme) -> Self {
        self.config.error_correction.scheme = scheme;
        self
    }

    #[must_use]
    pub const fn optimization_algorithm(mut self, algorithm: OptimizationAlgorithm) -> Self {
        self.config.optimization.algorithm = algorithm;
        self
    }

    pub fn build(self) -> DeviceResult<PhotonicConfig> {
        self.validate()?;
        Ok(self.config)
    }

    fn validate(&self) -> DeviceResult<()> {
        if self.config.system.mode_count == 0 {
            return Err(crate::DeviceError::InvalidInput(
                "Mode count must be greater than 0".to_string(),
            ));
        }

        if self.config.system.cutoff_dimension == 0 {
            return Err(crate::DeviceError::InvalidInput(
                "Cutoff dimension must be greater than 0".to_string(),
            ));
        }

        if self.config.hardware.laser.wavelength <= 0.0 {
            return Err(crate::DeviceError::InvalidInput(
                "Laser wavelength must be positive".to_string(),
            ));
        }

        if self.config.hardware.detector.efficiency < 0.0
            || self.config.hardware.detector.efficiency > 1.0
        {
            return Err(crate::DeviceError::InvalidInput(
                "Detection efficiency must be between 0 and 1".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for PhotonicConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Predefined configurations for common photonic setups
pub struct PhotonicConfigurations;

impl PhotonicConfigurations {
    /// Configuration for continuous variable quantum computing
    pub fn cv_config() -> PhotonicConfig {
        PhotonicConfigBuilder::new()
            .system_type(PhotonicSystem::ContinuousVariable)
            .mode_count(8)
            .cutoff_dimension(20)
            .laser_wavelength(1550.0)
            .detection_efficiency(0.9)
            .build()
            .expect("CV config uses valid parameters")
    }

    /// Configuration for gate-based photonic quantum computing
    pub fn gate_based_config() -> PhotonicConfig {
        PhotonicConfigBuilder::new()
            .system_type(PhotonicSystem::GateBased)
            .mode_count(16)
            .cutoff_dimension(5)
            .laser_wavelength(780.0)
            .detection_efficiency(0.95)
            .enable_error_correction(true)
            .error_correction_scheme(ErrorCorrectionScheme::GKP)
            .build()
            .expect("Gate-based config uses valid parameters")
    }

    /// Configuration for measurement-based quantum computing
    pub fn mbqc_config() -> PhotonicConfig {
        PhotonicConfigBuilder::new()
            .system_type(PhotonicSystem::MeasurementBased)
            .mode_count(32)
            .cutoff_dimension(3)
            .laser_wavelength(532.0)
            .detection_efficiency(0.85)
            .optimization_algorithm(OptimizationAlgorithm::Bayesian)
            .build()
            .expect("MBQC config uses valid parameters")
    }

    /// Configuration for hybrid photonic systems
    pub fn hybrid_config() -> PhotonicConfig {
        PhotonicConfigBuilder::new()
            .system_type(PhotonicSystem::Hybrid)
            .mode_count(24)
            .cutoff_dimension(15)
            .laser_wavelength(1064.0)
            .detection_efficiency(0.92)
            .enable_error_correction(true)
            .error_correction_scheme(ErrorCorrectionScheme::Cat)
            .build()
            .expect("Hybrid config uses valid parameters")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = PhotonicConfigBuilder::new()
            .mode_count(10)
            .cutoff_dimension(15)
            .laser_wavelength(1550.0)
            .build();

        assert!(config.is_ok());
        let config = config.expect("Config should be valid");
        assert_eq!(config.system.mode_count, 10);
        assert_eq!(config.system.cutoff_dimension, 15);
        assert_eq!(config.hardware.laser.wavelength, 1550.0);
    }

    #[test]
    fn test_predefined_configs() {
        let cv = PhotonicConfigurations::cv_config();
        assert_eq!(cv.system.system_type, PhotonicSystem::ContinuousVariable);

        let gate_based = PhotonicConfigurations::gate_based_config();
        assert_eq!(gate_based.system.system_type, PhotonicSystem::GateBased);
        assert!(gate_based.error_correction.enabled);

        let mbqc = PhotonicConfigurations::mbqc_config();
        assert_eq!(mbqc.system.system_type, PhotonicSystem::MeasurementBased);

        let hybrid = PhotonicConfigurations::hybrid_config();
        assert_eq!(hybrid.system.system_type, PhotonicSystem::Hybrid);
    }

    #[test]
    fn test_invalid_config() {
        let config = PhotonicConfigBuilder::new().mode_count(0).build();

        assert!(config.is_err());
    }
}
