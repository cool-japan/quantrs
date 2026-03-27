//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Duration, SystemTime};

use super::types::{
    AdvancedErrorMitigationManager, CalibrationError, CalibrationResult, DecodingError,
    ErrorMitigationConfig, MitigatedResult, MitigationError, NoiseModel, QECError,
    ResourceRequirements,
};

pub trait MitigationProtocolImpl: Send + Sync {
    fn name(&self) -> &str;
    fn apply(
        &self,
        circuit: &QuantumCircuit,
        noise_model: &NoiseModel,
    ) -> Result<MitigatedResult, MitigationError>;
    fn cost(&self) -> f64;
    fn effectiveness(&self, noise_model: &NoiseModel) -> f64;
    fn parameters(&self) -> HashMap<String, f64>;
    fn set_parameters(&mut self, params: HashMap<String, f64>) -> Result<(), MitigationError>;
}
pub trait CalibrationRoutine: Send + Sync {
    fn name(&self) -> &str;
    fn calibrate(
        &mut self,
        device: &mut QuantumDevice,
    ) -> Result<CalibrationResult, CalibrationError>;
    fn estimate_duration(&self) -> Duration;
    fn required_resources(&self) -> ResourceRequirements;
    fn dependencies(&self) -> Vec<String>;
}
pub trait ErrorCorrectionCode: Send + Sync {
    fn name(&self) -> &str;
    fn distance(&self) -> usize;
    fn encoding_rate(&self) -> f64;
    fn threshold(&self) -> f64;
    fn encode(&self, logical_state: &Array1<f64>) -> Result<Array1<f64>, QECError>;
    fn syndrome_extraction(&self, state: &Array1<f64>) -> Result<Array1<i32>, QECError>;
    fn error_lookup(&self, syndrome: &Array1<i32>) -> Result<Array1<i32>, QECError>;
}
pub trait Decoder: Send + Sync {
    fn name(&self) -> &str;
    fn decode(
        &self,
        syndrome: &Array1<i32>,
        code: &dyn ErrorCorrectionCode,
    ) -> Result<Array1<i32>, DecodingError>;
    fn confidence(&self) -> f64;
    fn computational_cost(&self) -> usize;
}
pub type QuantumCircuit = Vec<String>;
pub type QuantumDevice = HashMap<String, f64>;
pub type Pulse = (f64, f64, f64);
pub type PulseSequence = Vec<Pulse>;
/// Create a default advanced error mitigation manager
pub fn create_advanced_error_mitigation_manager() -> AdvancedErrorMitigationManager {
    AdvancedErrorMitigationManager::new(ErrorMitigationConfig::default())
}
/// Create a lightweight error mitigation manager for testing
pub fn create_lightweight_error_mitigation_manager() -> AdvancedErrorMitigationManager {
    let config = ErrorMitigationConfig {
        real_time_monitoring: false,
        adaptive_protocols: true,
        device_calibration: false,
        syndrome_prediction: false,
        qec_integration: false,
        monitoring_interval: Duration::from_secs(1),
        calibration_interval: Duration::from_secs(3600),
        noise_update_threshold: 0.1,
        mitigation_threshold: 0.2,
        history_retention: Duration::from_secs(3600),
    };
    AdvancedErrorMitigationManager::new(config)
}
