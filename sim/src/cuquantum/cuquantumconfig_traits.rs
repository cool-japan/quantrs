//! # CuQuantumConfig - Trait Implementations
//!
//! This module contains trait implementations for `CuQuantumConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    ComputePrecision, CuQuantumConfig, GateFusionLevel, TensorContractionAlgorithm,
};

impl Default for CuQuantumConfig {
    fn default() -> Self {
        Self {
            device_id: -1,
            multi_gpu: false,
            num_gpus: 1,
            memory_pool_size: 0,
            async_execution: true,
            memory_optimization: true,
            precision: ComputePrecision::Double,
            gate_fusion_level: GateFusionLevel::Aggressive,
            enable_profiling: false,
            max_statevec_qubits: 30,
            tensor_contraction: TensorContractionAlgorithm::Auto,
            enable_tf32: true, // Enable TF32 by default on compatible hardware
        }
    }
}
