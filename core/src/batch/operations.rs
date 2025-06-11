//! Batch operations for quantum gates using SciRS2 parallel algorithms

use super::{BatchGateOp, BatchStateVector};
use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{single::*, GateOp},
    qubit::QubitId,
};
use ndarray::{s, Array1, Array2, Array3, ArrayView2, Axis};
use num_complex::Complex64;
use rayon::prelude::*;
use std::sync::Arc;

// Import SciRS2 batch operations
// Note: SciRS2 batch operations don't support Complex numbers yet
// extern crate scirs2_linalg;
// use scirs2_linalg::batch::{batch_matmul, batch_matvec};

/// Apply a single-qubit gate to all states in a batch
pub fn apply_single_qubit_gate_batch(
    batch: &mut BatchStateVector,
    gate_matrix: &[Complex64; 4],
    target: QubitId,
) -> QuantRS2Result<()> {
    let n_qubits = batch.n_qubits;
    let target_idx = target.0 as usize;

    if target_idx >= n_qubits {
        return Err(QuantRS2Error::InvalidQubitId(target.0));
    }

    let batch_size = batch.batch_size();
    let state_size = 1 << n_qubits;

    // Use parallel processing for large batches
    if batch_size > 32 {
        batch
            .states
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .try_for_each(|mut state_row| -> QuantRS2Result<()> {
                apply_single_qubit_to_state(
                    &mut state_row.to_owned(),
                    gate_matrix,
                    target_idx,
                    n_qubits,
                )?;
                Ok(())
            })?;
    } else {
        // Sequential for small batches
        for i in 0..batch_size {
            let mut state = batch.states.row(i).to_owned();
            apply_single_qubit_to_state(&mut state, gate_matrix, target_idx, n_qubits)?;
            batch.states.row_mut(i).assign(&state);
        }
    }

    Ok(())
}

/// Apply a two-qubit gate to all states in a batch
pub fn apply_two_qubit_gate_batch(
    batch: &mut BatchStateVector,
    gate_matrix: &[Complex64; 16],
    control: QubitId,
    target: QubitId,
) -> QuantRS2Result<()> {
    let n_qubits = batch.n_qubits;
    let control_idx = control.0 as usize;
    let target_idx = target.0 as usize;

    if control_idx >= n_qubits || target_idx >= n_qubits {
        return Err(QuantRS2Error::InvalidQubitId(if control_idx >= n_qubits {
            control.0
        } else {
            target.0
        }));
    }

    if control_idx == target_idx {
        return Err(QuantRS2Error::InvalidInput(
            "Control and target qubits must be different".to_string(),
        ));
    }

    let batch_size = batch.batch_size();

    // Use parallel processing for large batches
    if batch_size > 16 {
        batch
            .states
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .try_for_each(|mut state_row| -> QuantRS2Result<()> {
                apply_two_qubit_to_state(
                    &mut state_row.to_owned(),
                    gate_matrix,
                    control_idx,
                    target_idx,
                    n_qubits,
                )?;
                Ok(())
            })?;
    } else {
        // Sequential for small batches
        for i in 0..batch_size {
            let mut state = batch.states.row(i).to_owned();
            apply_two_qubit_to_state(&mut state, gate_matrix, control_idx, target_idx, n_qubits)?;
            batch.states.row_mut(i).assign(&state);
        }
    }

    Ok(())
}

/// Apply a single-qubit gate to a state vector
fn apply_single_qubit_to_state(
    state: &mut Array1<Complex64>,
    gate_matrix: &[Complex64; 4],
    target_idx: usize,
    n_qubits: usize,
) -> QuantRS2Result<()> {
    let state_size = 1 << n_qubits;
    let target_mask = 1 << target_idx;

    for i in 0..state_size {
        if i & target_mask == 0 {
            let j = i | target_mask;

            let a = state[i];
            let b = state[j];

            state[i] = gate_matrix[0] * a + gate_matrix[1] * b;
            state[j] = gate_matrix[2] * a + gate_matrix[3] * b;
        }
    }

    Ok(())
}

/// Apply a two-qubit gate to a state vector
fn apply_two_qubit_to_state(
    state: &mut Array1<Complex64>,
    gate_matrix: &[Complex64; 16],
    control_idx: usize,
    target_idx: usize,
    n_qubits: usize,
) -> QuantRS2Result<()> {
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

            state[i00] = gate_matrix[0] * a00
                + gate_matrix[1] * a01
                + gate_matrix[2] * a10
                + gate_matrix[3] * a11;
            state[i01] = gate_matrix[4] * a00
                + gate_matrix[5] * a01
                + gate_matrix[6] * a10
                + gate_matrix[7] * a11;
            state[i10] = gate_matrix[8] * a00
                + gate_matrix[9] * a01
                + gate_matrix[10] * a10
                + gate_matrix[11] * a11;
            state[i11] = gate_matrix[12] * a00
                + gate_matrix[13] * a01
                + gate_matrix[14] * a10
                + gate_matrix[15] * a11;
        }
    }

    Ok(())
}

/// Batch-optimized Hadamard gate using SciRS2
pub struct BatchHadamard;

impl BatchGateOp for Hadamard {
    fn apply_batch(
        &self,
        batch: &mut BatchStateVector,
        target_qubits: &[QubitId],
    ) -> QuantRS2Result<()> {
        if target_qubits.len() != 1 {
            return Err(QuantRS2Error::InvalidInput(
                "Hadamard gate requires exactly one target qubit".to_string(),
            ));
        }

        let gate_matrix = [
            Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0),
            Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0),
            Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0),
            Complex64::new(-1.0 / std::f64::consts::SQRT_2, 0.0),
        ];

        apply_single_qubit_gate_batch(batch, &gate_matrix, target_qubits[0])
    }
}

/// Batch-optimized Pauli-X gate
impl BatchGateOp for PauliX {
    fn apply_batch(
        &self,
        batch: &mut BatchStateVector,
        target_qubits: &[QubitId],
    ) -> QuantRS2Result<()> {
        if target_qubits.len() != 1 {
            return Err(QuantRS2Error::InvalidInput(
                "Pauli-X gate requires exactly one target qubit".to_string(),
            ));
        }

        let gate_matrix = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        apply_single_qubit_gate_batch(batch, &gate_matrix, target_qubits[0])
    }
}

/// Apply multiple gates to a batch using SciRS2 batch operations
pub fn apply_gate_sequence_batch(
    batch: &mut BatchStateVector,
    gates: &[(Box<dyn GateOp>, Vec<QubitId>)],
) -> QuantRS2Result<()> {
    // For gates that support batch operations, use them
    // Otherwise fall back to standard application

    for (gate, qubits) in gates {
        // For now, always use standard application
        // TODO: Add batch-optimized gate detection
        {
            // Fall back to standard application
            let matrix = gate.matrix()?;

            match qubits.len() {
                1 => {
                    let mut gate_array = [Complex64::new(0.0, 0.0); 4];
                    gate_array.copy_from_slice(&matrix[..4]);
                    apply_single_qubit_gate_batch(batch, &gate_array, qubits[0])?;
                }
                2 => {
                    let mut gate_array = [Complex64::new(0.0, 0.0); 16];
                    gate_array.copy_from_slice(&matrix[..16]);
                    apply_two_qubit_gate_batch(batch, &gate_array, qubits[0], qubits[1])?;
                }
                _ => {
                    return Err(QuantRS2Error::InvalidInput(
                        "Batch operations for gates with more than 2 qubits not yet supported"
                            .to_string(),
                    ));
                }
            }
        }
    }

    Ok(())
}

/// Batch matrix multiplication
/// Note: SciRS2 batch_matmul doesn't support Complex numbers, so we implement our own
pub fn batch_state_matrix_multiply(
    batch: &BatchStateVector,
    matrices: &Array3<Complex64>,
) -> QuantRS2Result<BatchStateVector> {
    let batch_size = batch.batch_size();
    let (num_matrices, rows, cols) = matrices.dim();

    if num_matrices != batch_size {
        return Err(QuantRS2Error::InvalidInput(format!(
            "Number of matrices {} doesn't match batch size {}",
            num_matrices, batch_size
        )));
    }

    if cols != batch.states.ncols() {
        return Err(QuantRS2Error::InvalidInput(format!(
            "Matrix columns {} don't match state size {}",
            cols,
            batch.states.ncols()
        )));
    }

    // Perform batch matrix multiplication manually
    let mut result_states = Array2::zeros((batch_size, rows));

    // Use parallel processing for large batches
    if batch_size > 16 {
        use rayon::prelude::*;

        let results: Vec<_> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let matrix = matrices.slice(s![i, .., ..]);
                let state = batch.states.row(i);
                matrix.dot(&state)
            })
            .collect();

        for (i, result) in results.into_iter().enumerate() {
            result_states.row_mut(i).assign(&result);
        }
    } else {
        // Sequential for small batches
        for i in 0..batch_size {
            let matrix = matrices.slice(s![i, .., ..]);
            let state = batch.states.row(i);
            let result = matrix.dot(&state);
            result_states.row_mut(i).assign(&result);
        }
    }

    BatchStateVector::from_states(result_states, batch.config.clone())
}

/// Parallel expectation value computation
pub fn compute_expectation_values_batch(
    batch: &BatchStateVector,
    observable_matrix: &Array2<Complex64>,
) -> QuantRS2Result<Vec<f64>> {
    let batch_size = batch.batch_size();

    // Use parallel computation for large batches
    if batch_size > 16 {
        let expectations: Vec<f64> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let state = batch.states.row(i);
                compute_expectation_value(&state.to_owned(), observable_matrix)
            })
            .collect();

        Ok(expectations)
    } else {
        // Sequential for small batches
        let mut expectations = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let state = batch.states.row(i);
            expectations.push(compute_expectation_value(
                &state.to_owned(),
                observable_matrix,
            ));
        }
        Ok(expectations)
    }
}

/// Compute expectation value for a single state
fn compute_expectation_value(state: &Array1<Complex64>, observable: &Array2<Complex64>) -> f64 {
    // <ψ|O|ψ>
    let temp = observable.dot(state);
    let expectation = state
        .iter()
        .zip(temp.iter())
        .map(|(a, b)| a.conj() * b)
        .sum::<Complex64>();

    expectation.re
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_batch_hadamard() {
        let mut batch = BatchStateVector::new(3, 1, Default::default()).unwrap();
        let h = Hadamard { target: QubitId(0) };

        h.apply_batch(&mut batch, &[QubitId(0)]).unwrap();

        // Check all states are in superposition
        for i in 0..3 {
            let state = batch.get_state(i).unwrap();
            assert!((state[0].re - 1.0 / std::f64::consts::SQRT_2).abs() < 1e-10);
            assert!((state[1].re - 1.0 / std::f64::consts::SQRT_2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_batch_pauli_x() {
        let mut batch = BatchStateVector::new(2, 1, Default::default()).unwrap();
        let x = PauliX { target: QubitId(0) };

        x.apply_batch(&mut batch, &[QubitId(0)]).unwrap();

        // Check all states are flipped
        for i in 0..2 {
            let state = batch.get_state(i).unwrap();
            assert_eq!(state[0], Complex64::new(0.0, 0.0));
            assert_eq!(state[1], Complex64::new(1.0, 0.0));
        }
    }

    #[test]
    fn test_expectation_values_batch() {
        let batch = BatchStateVector::new(5, 1, Default::default()).unwrap();

        // Pauli Z observable
        let z_observable = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
        ];

        let expectations = compute_expectation_values_batch(&batch, &z_observable).unwrap();

        // All states are |0>, so expectation of Z should be 1
        for exp in expectations {
            assert!((exp - 1.0).abs() < 1e-10);
        }
    }
}
