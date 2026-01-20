//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::{
    buffer_pool::BufferPool,
    error::{QuantRS2Error, QuantRS2Result},
    qubit::QubitId,
};
use scirs2_core::parallel_ops::*;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::types::{
    BenchmarkSuite, BenchmarkSuiteResult, CalibrationData, DeviceTopology, EnhancedBenchmarkConfig,
    EnhancedHardwareBenchmark, ExecutionResult, QuantumCircuit, QuantumJob,
};

/// Quantum device trait
pub trait QuantumDevice: Sync {
    fn execute(&self, circuit: QuantumCircuit, shots: usize) -> QuantRS2Result<QuantumJob>;
    fn get_topology(&self) -> &DeviceTopology;
    fn get_calibration_data(&self) -> &CalibrationData;
    fn get_name(&self) -> String;
    fn get_native_gates(&self) -> Vec<String>;
    fn get_backend_version(&self) -> String;
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_benchmark_creation() {
        let config = EnhancedBenchmarkConfig::default();
        let benchmark = EnhancedHardwareBenchmark::new(config);
        assert!(benchmark.config.enable_ml_prediction);
    }
    #[test]
    fn test_benchmark_suite_result() {
        let mut result = BenchmarkSuiteResult::new(BenchmarkSuite::QuantumVolume);
        result.add_measurement(
            4,
            ExecutionResult {
                success_rate: 0.85,
                execution_time: Duration::from_millis(100),
                counts: HashMap::new(),
            },
        );
        assert_eq!(result.measurements.len(), 1);
        assert_eq!(result.measurements[&4].len(), 1);
    }
}
