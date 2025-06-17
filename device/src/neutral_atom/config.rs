//! Configuration management for neutral atom quantum devices

use crate::DeviceResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Advanced configuration for neutral atom quantum devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedNeutralAtomConfig {
    /// Hardware-specific settings
    pub hardware_settings: HardwareSettings,
    /// Optimization parameters
    pub optimization: OptimizationConfig,
    /// Error correction settings
    pub error_correction: ErrorCorrectionConfig,
    /// Calibration parameters
    pub calibration: CalibrationConfig,
}

/// Hardware-specific settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSettings {
    /// Laser control parameters
    pub laser_control: LaserControlConfig,
    /// Trap configuration
    pub trap_config: TrapConfig,
    /// Detection settings
    pub detection: DetectionConfig,
}

/// Laser control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserControlConfig {
    /// Main trapping laser wavelength (nm)
    pub trapping_wavelength: f64,
    /// Rydberg excitation laser wavelength (nm)
    pub rydberg_wavelength: f64,
    /// Maximum laser power (mW)
    pub max_power: f64,
    /// Beam waist (μm)
    pub beam_waist: f64,
    /// Laser stability specifications
    pub stability: LaserStabilityConfig,
}

/// Laser stability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserStabilityConfig {
    /// Frequency stability (Hz)
    pub frequency_stability: f64,
    /// Power stability (%)
    pub power_stability: f64,
    /// Phase stability (mrad)
    pub phase_stability: f64,
}

/// Trap configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrapConfig {
    /// Trap depth (μK)
    pub trap_depth: f64,
    /// Trap frequency (Hz)
    pub trap_frequency: f64,
    /// Anharmonicity coefficient
    pub anharmonicity: f64,
    /// Loading rate (atoms/s)
    pub loading_rate: f64,
}

/// Detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    /// Detection efficiency
    pub efficiency: f64,
    /// Dark count rate (Hz)
    pub dark_count_rate: f64,
    /// Integration time (μs)
    pub integration_time: f64,
    /// Background subtraction
    pub background_subtraction: bool,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable gate optimization
    pub gate_optimization: bool,
    /// Enable circuit compilation optimization
    pub circuit_optimization: bool,
    /// Maximum optimization time (ms)
    pub max_optimization_time: Duration,
    /// Optimization target
    pub optimization_target: OptimizationTarget,
}

/// Optimization targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationTarget {
    /// Minimize execution time
    Speed,
    /// Maximize fidelity
    Fidelity,
    /// Balance speed and fidelity
    Balanced,
    /// Custom optimization function
    Custom(String),
}

/// Error correction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCorrectionConfig {
    /// Enable error correction
    pub enabled: bool,
    /// Error correction scheme
    pub scheme: ErrorCorrectionScheme,
    /// Syndrome detection parameters
    pub syndrome_detection: SyndromeDetectionConfig,
    /// Recovery protocol
    pub recovery_protocol: RecoveryProtocolConfig,
}

/// Error correction schemes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorCorrectionScheme {
    /// No error correction
    None,
    /// Repetition code
    Repetition,
    /// Surface code
    Surface,
    /// Color code
    Color,
    /// Custom scheme
    Custom(String),
}

/// Syndrome detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyndromeDetectionConfig {
    /// Detection threshold
    pub threshold: f64,
    /// Measurement rounds
    pub measurement_rounds: usize,
    /// Majority voting
    pub majority_voting: bool,
}

/// Recovery protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryProtocolConfig {
    /// Maximum correction attempts
    pub max_attempts: usize,
    /// Recovery timeout (ms)
    pub timeout: Duration,
    /// Adaptive recovery
    pub adaptive: bool,
}

/// Calibration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// Auto-calibration enabled
    pub auto_calibration: bool,
    /// Calibration interval (hours)
    pub calibration_interval: Duration,
    /// Calibration procedures
    pub procedures: Vec<CalibrationProcedure>,
    /// Reference measurements
    pub reference_measurements: HashMap<String, f64>,
}

/// Calibration procedures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibrationProcedure {
    /// Laser frequency calibration
    LaserFrequency,
    /// Power calibration
    PowerCalibration,
    /// Trap depth calibration
    TrapDepth,
    /// Gate fidelity calibration
    GateFidelity,
    /// Detection efficiency calibration
    DetectionEfficiency,
    /// Custom procedure
    Custom(String),
}

impl Default for AdvancedNeutralAtomConfig {
    fn default() -> Self {
        Self {
            hardware_settings: HardwareSettings::default(),
            optimization: OptimizationConfig::default(),
            error_correction: ErrorCorrectionConfig::default(),
            calibration: CalibrationConfig::default(),
        }
    }
}

impl Default for HardwareSettings {
    fn default() -> Self {
        Self {
            laser_control: LaserControlConfig::default(),
            trap_config: TrapConfig::default(),
            detection: DetectionConfig::default(),
        }
    }
}

impl Default for LaserControlConfig {
    fn default() -> Self {
        Self {
            trapping_wavelength: 852.0,
            rydberg_wavelength: 480.0,
            max_power: 100.0,
            beam_waist: 1.0,
            stability: LaserStabilityConfig::default(),
        }
    }
}

impl Default for LaserStabilityConfig {
    fn default() -> Self {
        Self {
            frequency_stability: 1000.0,
            power_stability: 0.1,
            phase_stability: 0.01,
        }
    }
}

impl Default for TrapConfig {
    fn default() -> Self {
        Self {
            trap_depth: 1000.0,
            trap_frequency: 100000.0,
            anharmonicity: 0.01,
            loading_rate: 1000.0,
        }
    }
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            efficiency: 0.95,
            dark_count_rate: 10.0,
            integration_time: 100.0,
            background_subtraction: true,
        }
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            gate_optimization: true,
            circuit_optimization: true,
            max_optimization_time: Duration::from_millis(1000),
            optimization_target: OptimizationTarget::Balanced,
        }
    }
}

impl Default for ErrorCorrectionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            scheme: ErrorCorrectionScheme::None,
            syndrome_detection: SyndromeDetectionConfig::default(),
            recovery_protocol: RecoveryProtocolConfig::default(),
        }
    }
}

impl Default for SyndromeDetectionConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            measurement_rounds: 3,
            majority_voting: true,
        }
    }
}

impl Default for RecoveryProtocolConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            timeout: Duration::from_millis(100),
            adaptive: true,
        }
    }
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            auto_calibration: true,
            calibration_interval: Duration::from_secs(3600),
            procedures: vec![
                CalibrationProcedure::LaserFrequency,
                CalibrationProcedure::PowerCalibration,
                CalibrationProcedure::TrapDepth,
            ],
            reference_measurements: HashMap::new(),
        }
    }
}

/// Validate advanced configuration
pub fn validate_advanced_config(config: &AdvancedNeutralAtomConfig) -> DeviceResult<()> {
    // Validate laser control
    if config.hardware_settings.laser_control.max_power <= 0.0 {
        return Err(crate::DeviceError::InvalidInput(
            "Maximum laser power must be positive".to_string(),
        ));
    }

    // Validate trap configuration
    if config.hardware_settings.trap_config.trap_depth <= 0.0 {
        return Err(crate::DeviceError::InvalidInput(
            "Trap depth must be positive".to_string(),
        ));
    }

    // Validate detection efficiency
    let efficiency = config.hardware_settings.detection.efficiency;
    if efficiency < 0.0 || efficiency > 1.0 {
        return Err(crate::DeviceError::InvalidInput(
            "Detection efficiency must be between 0 and 1".to_string(),
        ));
    }

    Ok(())
}
