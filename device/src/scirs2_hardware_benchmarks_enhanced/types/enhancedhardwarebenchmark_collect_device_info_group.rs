//! # EnhancedHardwareBenchmark - collect_device_info_group Methods
//!
//! This module contains method implementations for `EnhancedHardwareBenchmark`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::functions::QuantumDevice;
use quantrs2_core::{
    buffer_pool::BufferPool,
    error::{QuantRS2Error, QuantRS2Result},
    qubit::QubitId,
};
use scirs2_core::parallel_ops::*;

use super::types::DeviceInfo;

use super::enhancedhardwarebenchmark_type::EnhancedHardwareBenchmark;

impl EnhancedHardwareBenchmark {
    /// Collect device information
    pub(super) fn collect_device_info(device: &impl QuantumDevice) -> QuantRS2Result<DeviceInfo> {
        Ok(DeviceInfo {
            name: device.get_name(),
            num_qubits: device.get_topology().num_qubits,
            connectivity: device.get_topology().connectivity.clone(),
            gate_set: device.get_native_gates(),
            calibration_timestamp: device.get_calibration_data().timestamp,
            backend_version: device.get_backend_version(),
        })
    }
}
