//! # EnhancedHardwareBenchmark - execute_and_measure_group Methods
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
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::types::{ExecutionResult, QuantumCircuit};

use super::enhancedhardwarebenchmark_type::EnhancedHardwareBenchmark;

impl EnhancedHardwareBenchmark {
    pub(super) fn execute_and_measure(
        &self,
        device: &impl QuantumDevice,
        circuit: &QuantumCircuit,
    ) -> QuantRS2Result<ExecutionResult> {
        let start = Instant::now();
        let job = device.execute(circuit.clone(), self.config.base_config.shots_per_circuit)?;
        let execution_time = start.elapsed();
        let counts = job.get_counts()?;
        let success_rate = Self::calculate_success_rate(&counts, circuit)?;
        Ok(ExecutionResult {
            success_rate,
            execution_time,
            counts,
        })
    }
    fn calculate_success_rate(
        _counts: &HashMap<Vec<bool>, usize>,
        _circuit: &QuantumCircuit,
    ) -> QuantRS2Result<f64> {
        Ok(0.67)
    }
}
