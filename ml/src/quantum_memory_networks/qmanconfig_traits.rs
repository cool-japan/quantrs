//! # QMANConfig - Trait Implementations
//!
//! This module contains trait implementations for `QMANConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    AddressingConfig, ControllerConfig, HeadConfig, MemoryInitialization, QMANConfig,
    QMANTrainingConfig,
};

impl Default for QMANConfig {
    fn default() -> Self {
        Self {
            controller_qubits: 6,
            memory_qubits: 4,
            memory_size: 128,
            addressing_config: AddressingConfig::default(),
            head_config: HeadConfig::default(),
            controller_config: ControllerConfig::default(),
            training_config: QMANTrainingConfig::default(),
            memory_init: MemoryInitialization::default(),
        }
    }
}
