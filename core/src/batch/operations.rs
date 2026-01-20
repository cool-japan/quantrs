//! Batch operations for quantum gates using SciRS2 parallel algorithms

use super::{BatchGateOp, BatchStateVector};
use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{single::*, GateOp},
    qubit::QubitId,
};
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis};
use scirs2_core::Complex64;
// use scirs2_core::parallel_ops::*;
use crate::parallel_ops_stubs::*;
// use scirs2_core::simd_ops::SimdUnifiedOps;
use crate::simd_ops_stubs::{SimdComplex64, SimdF64};

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
    // let _state_size = 1 << n_qubits;

    // Use optimized SIMD batch processing for large batches
    if batch_size > 32 {
        apply_single_qubit_batch_simd(batch, gate_matrix, target_idx, n_qubits)?;
    } else if batch_size > 16 {
        // Use parallel processing for medium batches
        batch
            .states
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .try_for_each(|mut state_row| -> QuantRS2Result<()> {
                let mut state = state_row.to_owned();
                apply_single_qubit_to_state_optimized(
                    &mut state,
                    gate_matrix,
                    target_idx,
                    n_qubits,
                )?;
                state_row.assign(&state);
                Ok(())
            })?;
    } else {
        // Sequential for small batches
        for i in 0..batch_size {
            let mut state = batch.states.row(i).to_owned();
            apply_single_qubit_to_state_optimized(&mut state, gate_matrix, target_idx, n_qubits)?;
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
                let mut state = state_row.to_owned();
                apply_two_qubit_to_state(
                    &mut state,
                    gate_matrix,
                    control_idx,
                    target_idx,
                    n_qubits,
                )?;
                state_row.assign(&state);
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

/// Apply a single-qubit gate to a state vector (optimized version)
fn apply_single_qubit_to_state_optimized(
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

/// SIMD-optimized batch single-qubit gate application
fn apply_single_qubit_batch_simd(
    batch: &mut BatchStateVector,
    gate_matrix: &[Complex64; 4],
    target_idx: usize,
    n_qubits: usize,
) -> QuantRS2Result<()> {
    // use scirs2_core::simd_ops::SimdUnifiedOps;
    use scirs2_core::ndarray::ArrayView1;

    let batch_size = batch.batch_size();
    let state_size = 1 << n_qubits;
    let target_mask = 1 << target_idx;

    // Extract gate matrix components
    let g00 = gate_matrix[0];
    let g01 = gate_matrix[1];
    let g10 = gate_matrix[2];
    let g11 = gate_matrix[3];

    // Process using scirs2_core SIMD operations
    // We'll process multiple batch items simultaneously using SIMD
    // Collect all pairs of amplitudes that need to be transformed
    let _pairs_per_batch = state_size / 2;
    let total_pairs = batch_size * _pairs_per_batch;

    // For simpler implementation, process each batch item individually
    // but use SIMD within each batch item
    for batch_idx in 0..batch_size {
        // Collect indices and values for SIMD processing
        let mut idx_pairs = Vec::new();
        let mut a_values = Vec::new();
        let mut b_values = Vec::new();

        for i in 0..state_size {
            if i & target_mask == 0 {
                let j = i | target_mask;
                idx_pairs.push((i, j));
                a_values.push(batch.states[[batch_idx, i]]);
                b_values.push(batch.states[[batch_idx, j]]);
            }
        }

        if idx_pairs.is_empty() {
            continue;
        }

        // Apply gate transformation using SIMD
        // new_a = g00 * a + g01 * b
        // new_b = g10 * a + g11 * b

        // Extract real and imaginary parts
        let _len = a_values.len();
        let a_real: Vec<f64> = a_values.iter().map(|c| c.re).collect();
        let a_imag: Vec<f64> = a_values.iter().map(|c| c.im).collect();
        let b_real: Vec<f64> = b_values.iter().map(|c| c.re).collect();
        let b_imag: Vec<f64> = b_values.iter().map(|c| c.im).collect();

        // Compute new_a using SIMD
        let a_real_view = ArrayView1::from(&a_real);
        let a_imag_view = ArrayView1::from(&a_imag);
        let b_real_view = ArrayView1::from(&b_real);
        let b_imag_view = ArrayView1::from(&b_imag);

        // new_a_real = g00.re * a.re - g00.im * a.im + g01.re * b.re - g01.im * b.im
        let term1 = <f64 as SimdF64>::simd_scalar_mul(&a_real_view, g00.re);
        let term2 = <f64 as SimdF64>::simd_scalar_mul(&a_imag_view, g00.im);
        let term3 = <f64 as SimdF64>::simd_scalar_mul(&b_real_view, g01.re);
        let term4 = <f64 as SimdF64>::simd_scalar_mul(&b_imag_view, g01.im);

        let temp1 = <f64 as SimdF64>::simd_sub_arrays(&term1.view(), &term2.view());
        let temp2 = <f64 as SimdF64>::simd_sub_arrays(&term3.view(), &term4.view());
        let new_a_real = <f64 as SimdF64>::simd_add_arrays(&temp1.view(), &temp2.view());

        // new_a_imag = g00.re * a.im + g00.im * a.re + g01.re * b.im + g01.im * b.re
        let term5 = <f64 as SimdF64>::simd_scalar_mul(&a_imag_view, g00.re);
        let term6 = <f64 as SimdF64>::simd_scalar_mul(&a_real_view, g00.im);
        let term7 = <f64 as SimdF64>::simd_scalar_mul(&b_imag_view, g01.re);
        let term8 = <f64 as SimdF64>::simd_scalar_mul(&b_real_view, g01.im);

        let temp3 = <f64 as SimdF64>::simd_add_arrays(&term5.view(), &term6.view());
        let temp4 = <f64 as SimdF64>::simd_add_arrays(&term7.view(), &term8.view());
        let new_a_imag = <f64 as SimdF64>::simd_add_arrays(&temp3.view(), &temp4.view());

        // Compute new_b using SIMD (similar process)
        let term9 = <f64 as SimdF64>::simd_scalar_mul(&a_real_view, g10.re);
        let term10 = <f64 as SimdF64>::simd_scalar_mul(&a_imag_view, g10.im);
        let term11 = <f64 as SimdF64>::simd_scalar_mul(&b_real_view, g11.re);
        let term12 = <f64 as SimdF64>::simd_scalar_mul(&b_imag_view, g11.im);

        let temp5 = <f64 as SimdF64>::simd_sub_arrays(&term9.view(), &term10.view());
        let temp6 = <f64 as SimdF64>::simd_sub_arrays(&term11.view(), &term12.view());
        let new_b_real = <f64 as SimdF64>::simd_add_arrays(&temp5.view(), &temp6.view());

        let term13 = <f64 as SimdF64>::simd_scalar_mul(&a_imag_view, g10.re);
        let term14 = <f64 as SimdF64>::simd_scalar_mul(&a_real_view, g10.im);
        let term15 = <f64 as SimdF64>::simd_scalar_mul(&b_imag_view, g11.re);
        let term16 = <f64 as SimdF64>::simd_scalar_mul(&b_real_view, g11.im);

        let temp7 = <f64 as SimdF64>::simd_add_arrays(&term13.view(), &term14.view());
        let temp8 = <f64 as SimdF64>::simd_add_arrays(&term15.view(), &term16.view());
        let new_b_imag = <f64 as SimdF64>::simd_add_arrays(&temp7.view(), &temp8.view());

        // Write back results
        for (idx, &(i, j)) in idx_pairs.iter().enumerate() {
            batch.states[[batch_idx, i]] = Complex64::new(new_a_real[idx], new_a_imag[idx]);
            batch.states[[batch_idx, j]] = Complex64::new(new_b_real[idx], new_b_imag[idx]);
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
            "Number of matrices {num_matrices} doesn't match batch size {batch_size}"
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
        // use scirs2_core::parallel_ops::*;
        use crate::parallel_ops_stubs::*;

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
    use scirs2_core::ndarray::array;

    #[test]
    fn test_batch_hadamard() {
        let mut batch = BatchStateVector::new(3, 1, Default::default())
            .expect("Failed to create batch state vector for Hadamard test");
        let h = Hadamard { target: QubitId(0) };

        h.apply_batch(&mut batch, &[QubitId(0)])
            .expect("Failed to apply Hadamard gate to batch");

        // Check all states are in superposition
        for i in 0..3 {
            let state = batch.get_state(i).expect("Failed to get state from batch");
            assert!((state[0].re - 1.0 / std::f64::consts::SQRT_2).abs() < 1e-10);
            assert!((state[1].re - 1.0 / std::f64::consts::SQRT_2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_batch_pauli_x() {
        let mut batch = BatchStateVector::new(2, 1, Default::default())
            .expect("Failed to create batch state vector for Pauli X test");
        let x = PauliX { target: QubitId(0) };

        x.apply_batch(&mut batch, &[QubitId(0)])
            .expect("Failed to apply Pauli X gate to batch");

        // Check all states are flipped
        for i in 0..2 {
            let state = batch.get_state(i).expect("Failed to get state from batch");
            assert_eq!(state[0], Complex64::new(0.0, 0.0));
            assert_eq!(state[1], Complex64::new(1.0, 0.0));
        }
    }

    #[test]
    fn test_expectation_values_batch() {
        let batch = BatchStateVector::new(5, 1, Default::default())
            .expect("Failed to create batch state vector for expectation test");

        // Pauli Z observable
        let z_observable = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
        ];

        let expectations = compute_expectation_values_batch(&batch, &z_observable)
            .expect("Failed to compute expectation values");

        // All states are |0>, so expectation of Z should be 1
        for exp in expectations {
            assert!((exp - 1.0).abs() < 1e-10);
        }
    }
}
