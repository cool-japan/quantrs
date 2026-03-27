//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use std::time::{Duration, Instant};

use super::autoparallelengine_type::AutoParallelEngine;
use super::types::{AutoParallelBenchmarkResults, AutoParallelConfig, CircuitParallelResult};

/// Benchmark automatic parallelization performance
pub fn benchmark_automatic_parallelization<const N: usize>(
    circuits: Vec<Circuit<N>>,
    config: AutoParallelConfig,
) -> QuantRS2Result<AutoParallelBenchmarkResults> {
    let engine = AutoParallelEngine::new(config);
    let mut results = Vec::new();
    let start_time = Instant::now();
    for circuit in circuits {
        let analysis_start = Instant::now();
        let analysis = engine.analyze_circuit(&circuit)?;
        let analysis_time = analysis_start.elapsed();
        results.push(CircuitParallelResult {
            circuit_size: circuit.num_gates(),
            num_qubits: circuit.num_qubits(),
            analysis_time,
            efficiency: analysis.efficiency,
            max_parallelism: analysis.max_parallelism,
            num_tasks: analysis.tasks.len(),
        });
    }
    let total_time = start_time.elapsed();
    Ok(AutoParallelBenchmarkResults {
        total_time,
        average_efficiency: results.iter().map(|r| r.efficiency).sum::<f64>()
            / results.len() as f64,
        average_parallelism: results.iter().map(|r| r.max_parallelism).sum::<usize>()
            / results.len(),
        circuit_results: results,
    })
}
