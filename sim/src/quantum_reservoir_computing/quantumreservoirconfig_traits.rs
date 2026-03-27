//! # QuantumReservoirConfig - Trait Implementations
//!
//! This module contains trait implementations for `QuantumReservoirConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{
    AdaptiveLearningConfig, AdvancedLearningConfig, BenchmarkingConfig, HardwareOptimizationConfig,
    InputEncoding, MemoryAnalysisConfig, OutputMeasurement, QuantumReservoirArchitecture,
    QuantumReservoirConfig, ReservoirDynamics, TimeSeriesConfig, TopologyConfig,
};

impl Default for QuantumReservoirConfig {
    fn default() -> Self {
        Self {
            num_qubits: 8,
            architecture: QuantumReservoirArchitecture::RandomCircuit,
            dynamics: ReservoirDynamics::Unitary,
            input_encoding: InputEncoding::Amplitude,
            output_measurement: OutputMeasurement::PauliExpectation,
            learning_config: AdvancedLearningConfig::default(),
            time_series_config: TimeSeriesConfig::default(),
            topology_config: TopologyConfig::default(),
            adaptive_config: AdaptiveLearningConfig::default(),
            memory_config: MemoryAnalysisConfig::default(),
            hardware_config: HardwareOptimizationConfig::default(),
            benchmark_config: BenchmarkingConfig::default(),
            time_step: 0.1,
            evolution_steps: 10,
            coupling_strength: 1.0,
            noise_level: 0.01,
            memory_capacity: 100,
            adaptive_learning: true,
            learning_rate: 0.01,
            washout_period: 50,
            random_seed: None,
            enable_qec: false,
            precision: 1e-8,
        }
    }
}
