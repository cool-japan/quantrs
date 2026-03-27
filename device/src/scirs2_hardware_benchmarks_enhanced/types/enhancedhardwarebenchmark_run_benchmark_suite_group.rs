//! # EnhancedHardwareBenchmark - run_benchmark_suite_group Methods
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

use super::types::{BenchmarkSuite, BenchmarkSuiteResult, LayerFidelity};

use super::enhancedhardwarebenchmark_type::EnhancedHardwareBenchmark;

impl EnhancedHardwareBenchmark {
    /// Run specific benchmark suite
    pub(super) fn run_benchmark_suite(
        &self,
        device: &impl QuantumDevice,
        suite: BenchmarkSuite,
    ) -> QuantRS2Result<BenchmarkSuiteResult> {
        match suite {
            BenchmarkSuite::QuantumVolume => self.run_quantum_volume_benchmark(device),
            BenchmarkSuite::RandomizedBenchmarking => Self::run_rb_benchmark(device),
            BenchmarkSuite::CrossEntropyBenchmarking => Self::run_xeb_benchmark(device),
            BenchmarkSuite::LayerFidelity => Self::run_layer_fidelity_benchmark(device),
            BenchmarkSuite::MirrorCircuits => self.run_mirror_circuit_benchmark(device),
            BenchmarkSuite::ProcessTomography => Self::run_process_tomography_benchmark(device),
            BenchmarkSuite::GateSetTomography => Self::run_gst_benchmark(device),
            BenchmarkSuite::Applications => Self::run_application_benchmark(device),
            BenchmarkSuite::Custom => Err(QuantRS2Error::InvalidOperation(
                "Custom benchmarks not yet implemented".to_string(),
            )),
        }
    }
}
