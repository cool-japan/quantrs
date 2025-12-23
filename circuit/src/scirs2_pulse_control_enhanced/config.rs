//! Configuration types for enhanced pulse control

use super::pulses::PulseLibrary;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Enhanced pulse control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedPulseConfig {
    /// Base pulse control configuration
    pub base_config: PulseControlConfig,
    /// Enable ML-based pulse optimization
    pub enable_ml_optimization: bool,
    /// Enable real-time calibration
    pub enable_realtime_calibration: bool,
    /// Enable advanced waveform synthesis
    pub enable_advanced_synthesis: bool,
    /// Enable comprehensive error mitigation
    pub enable_error_mitigation: bool,
    /// Enable adaptive control
    pub enable_adaptive_control: bool,
    /// Enable visual pulse representation
    pub enable_visual_output: bool,
    /// Optimization objectives
    pub optimization_objectives: Vec<PulseOptimizationObjective>,
    /// Performance constraints
    pub performance_constraints: PulseConstraints,
    /// Signal processing options
    pub signal_processing: SignalProcessingConfig,
    /// Export formats
    pub export_formats: Vec<PulseExportFormat>,
}

impl Default for EnhancedPulseConfig {
    fn default() -> Self {
        Self {
            base_config: PulseControlConfig::default(),
            enable_ml_optimization: true,
            enable_realtime_calibration: true,
            enable_advanced_synthesis: true,
            enable_error_mitigation: true,
            enable_adaptive_control: true,
            enable_visual_output: true,
            optimization_objectives: vec![
                PulseOptimizationObjective::MinimizeInfidelity,
                PulseOptimizationObjective::MinimizeDuration,
                PulseOptimizationObjective::MinimizePower,
            ],
            performance_constraints: PulseConstraints::default(),
            signal_processing: SignalProcessingConfig::default(),
            export_formats: vec![
                PulseExportFormat::OpenPulse,
                PulseExportFormat::Qiskit,
                PulseExportFormat::Custom,
            ],
        }
    }
}

/// Base pulse control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PulseControlConfig {
    /// Sample rate in Hz
    pub sample_rate: f64,
    /// Maximum pulse amplitude
    pub max_amplitude: f64,
    /// Minimum pulse duration in seconds
    pub min_duration: f64,
    /// Maximum pulse duration in seconds
    pub max_duration: f64,
    /// Hardware constraints
    pub hardware_constraints: HardwareConstraints,
    /// Default pulse shapes
    pub pulse_library: PulseLibrary,
}

impl Default for PulseControlConfig {
    fn default() -> Self {
        Self {
            sample_rate: 1e9,
            max_amplitude: 1.0,
            min_duration: 1e-9,
            max_duration: 1e-6,
            hardware_constraints: HardwareConstraints::default(),
            pulse_library: PulseLibrary::default(),
        }
    }
}

/// Hardware constraints for pulse control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConstraints {
    /// AWG specifications
    pub awg_specs: AWGSpecifications,
    /// IQ mixer specifications
    pub iq_mixer_specs: IQMixerSpecifications,
    /// Control electronics bandwidth
    pub bandwidth: f64,
    /// Rise/fall time constraints
    pub rise_time: f64,
    /// Phase noise specifications
    pub phase_noise: PhaseNoiseSpec,
    /// Amplitude noise specifications
    pub amplitude_noise: AmplitudeNoiseSpec,
}

impl Default for HardwareConstraints {
    fn default() -> Self {
        Self {
            awg_specs: AWGSpecifications::default(),
            iq_mixer_specs: IQMixerSpecifications::default(),
            bandwidth: 500e6,
            rise_time: 2e-9,
            phase_noise: PhaseNoiseSpec::default(),
            amplitude_noise: AmplitudeNoiseSpec::default(),
        }
    }
}

/// AWG specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AWGSpecifications {
    pub resolution_bits: u8,
    pub max_sample_rate: f64,
    pub memory_depth: usize,
    pub channels: usize,
    pub voltage_range: (f64, f64),
}

impl Default for AWGSpecifications {
    fn default() -> Self {
        Self {
            resolution_bits: 16,
            max_sample_rate: 2.5e9,
            memory_depth: 16_000_000,
            channels: 4,
            voltage_range: (-1.0, 1.0),
        }
    }
}

/// IQ mixer specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IQMixerSpecifications {
    pub lo_frequency_range: (f64, f64),
    pub if_bandwidth: f64,
    pub isolation: f64,
    pub conversion_loss: f64,
}

impl Default for IQMixerSpecifications {
    fn default() -> Self {
        Self {
            lo_frequency_range: (1e9, 20e9),
            if_bandwidth: 1e9,
            isolation: 40.0,
            conversion_loss: 6.0,
        }
    }
}

/// Phase noise specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseNoiseSpec {
    pub offset_frequencies: Vec<f64>,
    pub noise_levels: Vec<f64>,
}

impl Default for PhaseNoiseSpec {
    fn default() -> Self {
        Self {
            offset_frequencies: vec![10.0, 100.0, 1e3, 10e3, 100e3, 1e6],
            noise_levels: vec![-80.0, -100.0, -110.0, -120.0, -130.0, -140.0],
        }
    }
}

/// Amplitude noise specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmplitudeNoiseSpec {
    pub rms_noise: f64,
    pub peak_to_peak_noise: f64,
    pub spectral_density: f64,
}

impl Default for AmplitudeNoiseSpec {
    fn default() -> Self {
        Self {
            rms_noise: 1e-6,
            peak_to_peak_noise: 6e-6,
            spectral_density: 1e-9,
        }
    }
}

/// Pulse optimization objectives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PulseOptimizationObjective {
    MinimizeInfidelity,
    MinimizeDuration,
    MinimizePower,
    MinimizeLeakage,
    MaximizeRobustness,
    MinimizeCrosstalk,
}

/// Pulse constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PulseConstraints {
    pub max_amplitude: Option<f64>,
    pub max_slew_rate: Option<f64>,
    pub max_frequency: Option<f64>,
    pub min_fidelity: Option<f64>,
    pub max_leakage: Option<f64>,
}

impl Default for PulseConstraints {
    fn default() -> Self {
        Self {
            max_amplitude: Some(1.0),
            max_slew_rate: Some(1e12),
            max_frequency: Some(500e6),
            min_fidelity: Some(0.999),
            max_leakage: Some(0.001),
        }
    }
}

/// Signal processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalProcessingConfig {
    pub filter_type: FilterType,
    pub windowing: WindowType,
    pub oversampling_factor: usize,
    pub enable_predistortion: bool,
    pub enable_feedback: bool,
}

impl Default for SignalProcessingConfig {
    fn default() -> Self {
        Self {
            filter_type: FilterType::Butterworth(4),
            windowing: WindowType::Hamming,
            oversampling_factor: 4,
            enable_predistortion: true,
            enable_feedback: true,
        }
    }
}

/// Filter types for signal processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterType {
    Butterworth(usize),
    Chebyshev(usize),
    Bessel(usize),
    FIR(usize),
}

/// Window types for spectral analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowType {
    Rectangular,
    Hamming,
    Hanning,
    Blackman,
    Kaiser(u32),
}

/// Pulse export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PulseExportFormat {
    OpenPulse,
    Qiskit,
    Custom,
}
