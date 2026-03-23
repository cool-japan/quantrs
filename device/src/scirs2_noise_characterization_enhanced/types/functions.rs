//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::{
    buffer_pool::BufferPool,
    error::{QuantRS2Error, QuantRS2Result},
    qubit::QubitId,
};

use super::types::{CalibrationData, DeviceTopology, DynamicCircuit, QuantumJob};

/// Helper trait for quantum devices
pub(crate) trait QuantumDevice: Sync {
    fn execute(&self, circuit: DynamicCircuit, shots: usize) -> QuantRS2Result<QuantumJob>;
    fn get_topology(&self) -> &DeviceTopology;
    fn get_calibration_data(&self) -> &CalibrationData;
}
