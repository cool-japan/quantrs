//! Annealing Integration Configuration Types

/// Annealing integration configuration
#[derive(Debug, Clone)]
pub struct AnnealingIntegration {
    /// Integration strategy
    pub integration_strategy: IntegrationStrategy,
    /// Logical annealing schedule
    pub logical_schedule: LogicalAnnealingSchedule,
    /// Error correction timing
    pub correction_timing: CorrectionTiming,
    /// Adaptation parameters
    pub adaptation_parameters: AdaptationParameters,
}

/// Integration strategies
#[derive(Debug, Clone, PartialEq)]
pub enum IntegrationStrategy {
    /// Interleaved correction and annealing
    Interleaved,
    /// Continuous error correction
    Continuous,
    /// Batch correction
    Batch,
    /// Adaptive correction
    Adaptive,
    /// Threshold-based correction
    ThresholdBased,
}

/// Logical annealing schedule
#[derive(Debug, Clone)]
pub struct LogicalAnnealingSchedule {
    /// Schedule for logical Hamiltonian
    pub logical_hamiltonian_schedule: Vec<(f64, f64)>, // (time, coefficient)
    /// Schedule for logical mixer
    pub logical_mixer_schedule: Vec<(f64, f64)>,
    /// Error correction intervals
    pub correction_intervals: Vec<f64>,
    /// Adaptation rate
    pub adaptation_rate: f64,
}

/// Correction timing configuration
#[derive(Debug, Clone)]
pub struct CorrectionTiming {
    /// Correction period
    pub correction_period: f64,
    /// Adaptive timing
    pub adaptive_timing: bool,
    /// Error rate threshold for timing adjustment
    pub error_rate_threshold: f64,
    /// Minimum correction interval
    pub min_correction_interval: f64,
    /// Maximum correction interval
    pub max_correction_interval: f64,
}

/// Adaptation parameters
#[derive(Debug, Clone)]
pub struct AdaptationParameters {
    /// Learning rate for adaptation
    pub learning_rate: f64,
    /// Performance history window
    pub history_window: usize,
    /// Adaptation threshold
    pub adaptation_threshold: f64,
    /// Maximum adaptation steps
    pub max_adaptation_steps: usize,
}
