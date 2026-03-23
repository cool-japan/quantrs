//! # HardwareOptimizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `HardwareOptimizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{
    ConnectivityConstraints, ErrorMitigationMethod, HardwareOptimizationConfig, NativeGate,
    QuantumPlatform,
};

impl Default for HardwareOptimizationConfig {
    fn default() -> Self {
        Self {
            platform: QuantumPlatform::Simulator,
            enable_noise_aware: true,
            error_mitigation: vec![ErrorMitigationMethod::ZNE, ErrorMitigationMethod::PEC],
            enable_circuit_optimization: true,
            native_gate_set: vec![NativeGate::RZ, NativeGate::SX, NativeGate::CNOT],
            connectivity_constraints: ConnectivityConstraints::AllToAll,
            enable_calibration: false,
            calibration_frequency: 100,
            enable_monitoring: true,
            enable_hardware_adaptation: false,
        }
    }
}
