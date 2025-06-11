//! Batch circuit execution using SciRS2 parallel algorithms

use super::{BatchConfig, BatchExecutionResult, BatchStateVector};
use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    gpu::{GpuBackendFactory, GpuStateVector},
    qubit::QubitId,
};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

// Import SciRS2 parallel scheduler
extern crate scirs2_core;
use scirs2_core::parallel::{SchedulerConfig, WorkStealingScheduler};

/// Simple circuit representation for batch execution
pub struct BatchCircuit {
    /// Number of qubits
    pub n_qubits: usize,
    /// Gate sequence
    pub gates: Vec<Box<dyn GateOp>>,
}

impl BatchCircuit {
    /// Create a new circuit
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            gates: Vec::new(),
        }
    }

    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate: Box<dyn GateOp>) -> QuantRS2Result<()> {
        // Validate gate qubits
        for qubit in gate.qubits() {
            if qubit.0 as usize >= self.n_qubits {
                return Err(QuantRS2Error::InvalidQubitId(qubit.0));
            }
        }
        self.gates.push(gate);
        Ok(())
    }

    /// Get the gate sequence
    pub fn gate_sequence(&self) -> impl Iterator<Item = &Box<dyn GateOp>> {
        self.gates.iter()
    }

    /// Get the number of gates
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }
}

/// Batch circuit executor with parallel processing
pub struct BatchCircuitExecutor {
    /// Configuration for batch execution
    pub config: BatchConfig,
    /// Optional GPU backend
    pub gpu_backend: Option<Arc<dyn crate::gpu::GpuBackend>>,
    /// Work-stealing scheduler for parallel execution
    pub scheduler: Option<WorkStealingScheduler>,
}

impl BatchCircuitExecutor {
    /// Create a new batch circuit executor
    pub fn new(config: BatchConfig) -> QuantRS2Result<Self> {
        // Initialize GPU backend if requested
        let gpu_backend = if config.use_gpu {
            GpuBackendFactory::create_best_available().ok()
        } else {
            None
        };

        // Create work-stealing scheduler if parallel execution is enabled
        let scheduler = None; // Simplified - WorkStealingScheduler API needs review

        Ok(Self {
            config,
            gpu_backend,
            scheduler,
        })
    }

    /// Execute a circuit on a batch of initial states
    pub fn execute_batch(
        &self,
        circuit: &BatchCircuit,
        batch: &mut BatchStateVector,
    ) -> QuantRS2Result<BatchExecutionResult> {
        if batch.n_qubits != circuit.n_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Circuit has {} qubits but batch has {}",
                circuit.n_qubits, batch.n_qubits
            )));
        }

        let start_time = Instant::now();
        let gates_applied = circuit.num_gates();

        // Choose execution strategy based on batch size and configuration
        let used_gpu = if self.gpu_backend.is_some() && batch.batch_size() >= 64 {
            self.execute_with_gpu(circuit, batch)?;
            true
        } else if batch.batch_size() > self.config.max_batch_size {
            self.execute_chunked(circuit, batch)?;
            false
        } else {
            self.execute_parallel(circuit, batch)?;
            false
        };

        let execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(BatchExecutionResult {
            final_states: batch.states.clone(),
            execution_time_ms,
            gates_applied,
            used_gpu,
        })
    }

    /// Execute using GPU acceleration
    fn execute_with_gpu(
        &self,
        circuit: &BatchCircuit,
        batch: &mut BatchStateVector,
    ) -> QuantRS2Result<()> {
        // GPU execution is a placeholder for now
        // In practice, would need proper GPU state management
        // For now, fall back to CPU execution
        self.execute_parallel(circuit, batch)
    }

    /// Execute in chunks for large batches
    fn execute_chunked(
        &self,
        circuit: &BatchCircuit,
        batch: &mut BatchStateVector,
    ) -> QuantRS2Result<()> {
        let chunk_size = self.config.max_batch_size;
        let chunks = super::split_batch(batch, chunk_size);

        // Process chunks in parallel
        let processed_chunks: Vec<_> = chunks
            .into_par_iter()
            .map(|mut chunk| {
                self.execute_parallel(circuit, &mut chunk)?;
                Ok(chunk)
            })
            .collect::<QuantRS2Result<Vec<_>>>()?;

        // Merge results back
        let merged = super::merge_batches(processed_chunks, batch.config.clone())?;
        batch.states = merged.states;

        Ok(())
    }

    /// Execute using parallel processing
    fn execute_parallel(
        &self,
        circuit: &BatchCircuit,
        batch: &mut BatchStateVector,
    ) -> QuantRS2Result<()> {
        let batch_size = batch.batch_size();
        let gate_sequence: Vec<_> = circuit.gate_sequence().collect();

        if let Some(scheduler) = &self.scheduler {
            // Use work-stealing scheduler for better load balancing
            self.execute_with_scheduler(batch, &gate_sequence, scheduler)?;
        } else {
            // Use Rayon for simple parallel execution
            batch
                .states
                .axis_iter_mut(ndarray::Axis(0))
                .into_par_iter()
                .try_for_each(|mut state_row| -> QuantRS2Result<()> {
                    let mut state = state_row.to_owned();
                    apply_gates_to_state(&mut state, &gate_sequence, batch.n_qubits)?;
                    state_row.assign(&state);
                    Ok(())
                })?;
        }

        Ok(())
    }

    /// Execute using work-stealing scheduler
    fn execute_with_scheduler(
        &self,
        batch: &mut BatchStateVector,
        gates: &[&Box<dyn GateOp>],
        scheduler: &WorkStealingScheduler,
    ) -> QuantRS2Result<()> {
        // Create tasks for each state
        let batch_size = batch.batch_size();
        let n_qubits = batch.n_qubits;

        // Process states using simple parallel iteration
        let results: Vec<Array1<Complex64>> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let mut state = batch.states.row(i).to_owned();
                apply_gates_to_state(&mut state, gates, n_qubits).map(|_| state)
            })
            .collect::<QuantRS2Result<Vec<_>>>()?;

        // Update batch states
        for (i, state) in results.into_iter().enumerate() {
            batch.states.row_mut(i).assign(&state);
        }

        Ok(())
    }

    /// Execute multiple circuits on the same batch
    pub fn execute_multiple_circuits(
        &self,
        circuits: &[BatchCircuit],
        initial_batch: &BatchStateVector,
    ) -> QuantRS2Result<Vec<BatchExecutionResult>> {
        if circuits.is_empty() {
            return Ok(Vec::new());
        }

        // Execute circuits in parallel
        let results: Vec<_> = circuits
            .par_iter()
            .map(|circuit| {
                let mut batch_copy = BatchStateVector::from_states(
                    initial_batch.states.clone(),
                    initial_batch.config.clone(),
                )?;

                self.execute_batch(circuit, &mut batch_copy)
            })
            .collect::<QuantRS2Result<Vec<_>>>()?;

        Ok(results)
    }

    /// Execute a parameterized circuit with different parameter sets
    pub fn execute_parameterized_batch(
        &self,
        circuit_fn: impl Fn(&[f64]) -> QuantRS2Result<BatchCircuit> + Sync,
        parameter_sets: &[Vec<f64>],
        initial_state: &Array1<Complex64>,
    ) -> QuantRS2Result<Vec<Array1<Complex64>>> {
        // Create batch from single initial state
        let batch_size = parameter_sets.len();
        let mut states = Array2::zeros((batch_size, initial_state.len()));
        for i in 0..batch_size {
            states.row_mut(i).assign(initial_state);
        }

        let mut batch = BatchStateVector::from_states(states, self.config.clone())?;

        // Execute with different parameters in parallel
        let results: Vec<_> = parameter_sets
            .par_iter()
            .enumerate()
            .map(|(i, params)| {
                let circuit = circuit_fn(params)?;
                let mut state = batch.get_state(i)?;
                apply_gates_to_state(
                    &mut state,
                    &circuit.gate_sequence().collect::<Vec<_>>(),
                    circuit.n_qubits,
                )?;
                Ok(state)
            })
            .collect::<QuantRS2Result<Vec<_>>>()?;

        Ok(results)
    }
}

/// Apply gates to a single state
fn apply_gates_to_state(
    state: &mut Array1<Complex64>,
    gates: &[&Box<dyn GateOp>],
    n_qubits: usize,
) -> QuantRS2Result<()> {
    for gate in gates {
        let qubits = gate.qubits();
        let matrix = gate.matrix()?;

        match qubits.len() {
            1 => {
                apply_single_qubit_gate(state, &matrix, qubits[0], n_qubits)?;
            }
            2 => {
                apply_two_qubit_gate(state, &matrix, qubits[0], qubits[1], n_qubits)?;
            }
            _ => {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Gates with {} qubits not yet supported",
                    qubits.len()
                )));
            }
        }
    }

    Ok(())
}

/// Apply a single-qubit gate
fn apply_single_qubit_gate(
    state: &mut Array1<Complex64>,
    matrix: &[Complex64],
    target: QubitId,
    n_qubits: usize,
) -> QuantRS2Result<()> {
    let target_idx = target.0 as usize;
    let state_size = 1 << n_qubits;
    let target_mask = 1 << target_idx;

    for i in 0..state_size {
        if i & target_mask == 0 {
            let j = i | target_mask;

            let a = state[i];
            let b = state[j];

            state[i] = matrix[0] * a + matrix[1] * b;
            state[j] = matrix[2] * a + matrix[3] * b;
        }
    }

    Ok(())
}

/// Apply a two-qubit gate
fn apply_two_qubit_gate(
    state: &mut Array1<Complex64>,
    matrix: &[Complex64],
    control: QubitId,
    target: QubitId,
    n_qubits: usize,
) -> QuantRS2Result<()> {
    let control_idx = control.0 as usize;
    let target_idx = target.0 as usize;
    let state_size = 1 << n_qubits;
    let control_mask = 1 << control_idx;
    let target_mask = 1 << target_idx;

    for i in 0..state_size {
        if (i & control_mask == 0) && (i & target_mask == 0) {
            let i00 = i;
            let i01 = i | target_mask;
            let i10 = i | control_mask;
            let i11 = i | control_mask | target_mask;

            let a00 = state[i00];
            let a01 = state[i01];
            let a10 = state[i10];
            let a11 = state[i11];

            state[i00] = matrix[0] * a00 + matrix[1] * a01 + matrix[2] * a10 + matrix[3] * a11;
            state[i01] = matrix[4] * a00 + matrix[5] * a01 + matrix[6] * a10 + matrix[7] * a11;
            state[i10] = matrix[8] * a00 + matrix[9] * a01 + matrix[10] * a10 + matrix[11] * a11;
            state[i11] = matrix[12] * a00 + matrix[13] * a01 + matrix[14] * a10 + matrix[15] * a11;
        }
    }

    Ok(())
}

/// Create a batch executor with optimized settings
pub fn create_optimized_executor() -> QuantRS2Result<BatchCircuitExecutor> {
    let config = BatchConfig {
        num_workers: Some(8), // Default to 8 workers
        max_batch_size: 1024,
        use_gpu: true,
        memory_limit: Some(8 * 1024 * 1024 * 1024), // 8GB
        enable_cache: true,
    };

    BatchCircuitExecutor::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::single::Hadamard;

    #[test]
    fn test_batch_circuit_execution() {
        let config = BatchConfig {
            use_gpu: false,
            ..Default::default()
        };

        let executor = BatchCircuitExecutor::new(config).unwrap();

        // Create a simple circuit
        let mut circuit = BatchCircuit::new(2);
        circuit
            .add_gate(Box::new(Hadamard { target: QubitId(0) }))
            .unwrap();
        circuit
            .add_gate(Box::new(Hadamard { target: QubitId(1) }))
            .unwrap();

        // Create batch
        let mut batch = BatchStateVector::new(5, 2, Default::default()).unwrap();

        // Execute
        let result = executor.execute_batch(&circuit, &mut batch).unwrap();

        assert_eq!(result.gates_applied, 2);
        assert!(!result.used_gpu);

        // Check all states are in superposition
        for i in 0..5 {
            let state = batch.get_state(i).unwrap();
            assert!((state[0].re - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_parallel_circuit_execution() {
        let config = BatchConfig {
            num_workers: Some(2),
            use_gpu: false,
            ..Default::default()
        };

        let executor = BatchCircuitExecutor::new(config).unwrap();

        // Create multiple circuits
        let mut circuits = Vec::new();
        for _ in 0..3 {
            let mut circuit = BatchCircuit::new(1);
            circuit
                .add_gate(Box::new(Hadamard { target: QubitId(0) }))
                .unwrap();
            circuits.push(circuit);
        }

        // Create initial batch
        let batch = BatchStateVector::new(10, 1, Default::default()).unwrap();

        // Execute multiple circuits
        let results = executor
            .execute_multiple_circuits(&circuits, &batch)
            .unwrap();

        assert_eq!(results.len(), 3);
        for result in results {
            assert_eq!(result.gates_applied, 1);
        }
    }
}
