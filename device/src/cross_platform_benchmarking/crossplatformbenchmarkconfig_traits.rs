//! # CrossPlatformBenchmarkConfig - Trait Implementations
//!
//! This module contains trait implementations for `CrossPlatformBenchmarkConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_circuit::prelude::*;
use std::time::{Duration, Instant, SystemTime};

use super::types::{
    ComplexityLevel, CrossPlatformBenchmarkConfig, ParallelBenchmarkConfig, QuantumPlatform,
    StatisticalAnalysisConfig,
};

impl Default for CrossPlatformBenchmarkConfig {
    fn default() -> Self {
        Self {
            target_platforms: vec![
                QuantumPlatform::IBMQuantum("ibmq_qasm_simulator".to_string()),
                QuantumPlatform::AWSBraket(
                    "arn:aws:braket:::device/quantum-simulator/amazon/sv1".to_string(),
                ),
            ],
            complexity_levels: vec![
                ComplexityLevel {
                    name: "Simple".to_string(),
                    qubit_count: 2,
                    circuit_depth: 5,
                    gate_count_range: (5, 15),
                    two_qubit_gate_ratio: 0.3,
                    description: "Basic circuits for connectivity testing".to_string(),
                },
                ComplexityLevel {
                    name: "Medium".to_string(),
                    qubit_count: 5,
                    circuit_depth: 20,
                    gate_count_range: (20, 50),
                    two_qubit_gate_ratio: 0.4,
                    description: "Intermediate circuits for performance assessment".to_string(),
                },
                ComplexityLevel {
                    name: "Complex".to_string(),
                    qubit_count: 10,
                    circuit_depth: 50,
                    gate_count_range: (50, 150),
                    two_qubit_gate_ratio: 0.5,
                    description: "Complex circuits for scalability testing".to_string(),
                },
            ],
            statistical_config: StatisticalAnalysisConfig::default(),
            parallel_config: ParallelBenchmarkConfig::default(),
            benchmark_timeout: Duration::from_secs(300),
            repetitions: 10,
            enable_cost_analysis: true,
            enable_latency_analysis: true,
            enable_reliability_analysis: true,
            custom_circuits: Vec::new(),
        }
    }
}
