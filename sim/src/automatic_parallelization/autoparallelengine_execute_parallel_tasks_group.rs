//! # AutoParallelEngine - execute_parallel_tasks_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::parallel_ops::{current_num_threads, IndexedParallelIterator, ParallelIterator};
use scirs2_core::Complex64;
use std::sync::{Arc, Barrier, Mutex, RwLock};

use super::types::ParallelTask;

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// Execute parallel tasks with proper synchronization
    pub(super) fn execute_parallel_tasks(
        &self,
        tasks: &[ParallelTask],
        shared_state: Arc<RwLock<Vec<Complex64>>>,
        results: Arc<Mutex<Vec<Complex64>>>,
        barrier: Arc<Barrier>,
    ) -> QuantRS2Result<()> {
        use scirs2_core::parallel_ops::{parallel_map, IndexedParallelIterator};
        let _ = parallel_map(tasks, |task| {
            barrier.wait();
            let mut state = shared_state
                .write()
                .expect("Failed to acquire write lock on shared state");
            for gate in &task.gates {
                let qubits = gate.qubits();
                match qubits.len() {
                    1 => {
                        Self::apply_single_qubit_gate_to_state(
                            &mut state,
                            gate.as_ref(),
                            qubits[0].0 as usize,
                        );
                    }
                    2 => {
                        Self::apply_two_qubit_gate_to_state(
                            &mut state,
                            gate.as_ref(),
                            qubits[0].0 as usize,
                            qubits[1].0 as usize,
                        );
                    }
                    _ => {
                        eprintln!(
                            "Warning: {}-qubit gates not optimized for parallel execution",
                            qubits.len()
                        );
                    }
                }
            }
            barrier.wait();
        });
        let final_state = shared_state
            .read()
            .expect("Failed to acquire read lock on shared state");
        let mut result_vec = results.lock().expect("Failed to acquire lock on results");
        result_vec.clone_from(&final_state);
        Ok(())
    }
    /// Apply a single-qubit gate to a state vector
    pub(super) fn apply_single_qubit_gate_to_state(
        state: &mut [Complex64],
        gate: &dyn GateOp,
        qubit: usize,
    ) {
        let num_qubits = (state.len() as f64).log2() as usize;
        let stride = 1 << qubit;
        for base in 0..state.len() {
            if (base & stride) == 0 {
                let idx0 = base;
                let idx1 = base | stride;
                let amp0 = state[idx0];
                let amp1 = state[idx1];
                state[idx0] = amp0;
                state[idx1] = amp1;
            }
        }
    }
    /// Apply a two-qubit gate to a state vector
    pub(super) fn apply_two_qubit_gate_to_state(
        state: &mut [Complex64],
        gate: &dyn GateOp,
        qubit1: usize,
        qubit2: usize,
    ) {
        let num_qubits = (state.len() as f64).log2() as usize;
        let stride1 = 1 << qubit1;
        let stride2 = 1 << qubit2;
        for base in 0..state.len() {
            if (base & stride1) == 0 && (base & stride2) == 0 {
                let idx00 = base;
                let idx01 = base | stride1;
                let idx10 = base | stride2;
                let idx11 = base | stride1 | stride2;
                let amp00 = state[idx00];
                let amp01 = state[idx01];
                let amp10 = state[idx10];
                let amp11 = state[idx11];
                state[idx00] = amp00;
                state[idx01] = amp01;
                state[idx10] = amp10;
                state[idx11] = amp11;
            }
        }
    }
}
