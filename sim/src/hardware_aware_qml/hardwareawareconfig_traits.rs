//! # HardwareAwareConfig - Trait Implementations
//!
//! This module contains trait implementations for `HardwareAwareConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::device_noise_models::DeviceTopology;

use super::types::{HardwareArchitecture, HardwareAwareConfig, HardwareOptimizationLevel};

impl Default for HardwareAwareConfig {
    fn default() -> Self {
        Self {
            target_architecture: HardwareArchitecture::Simulator,
            device_topology: DeviceTopology::default(),
            enable_noise_aware_optimization: true,
            enable_connectivity_optimization: true,
            enable_hardware_efficient_ansatz: true,
            enable_dynamic_adaptation: true,
            enable_cross_device_portability: false,
            optimization_level: HardwareOptimizationLevel::Balanced,
            max_compilation_time_ms: 30_000,
            enable_performance_monitoring: true,
        }
    }
}
