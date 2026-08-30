//! Lightweight exact state-vector simulator for QML circuits.
//!
//! QML primitives (variational classifiers, reinforcement-learning value/policy
//! circuits, NLP models) build sequences of [`GateOp`] objects and need their
//! *real* output statistics (measurement probabilities, Pauli-Z expectations)
//! to compute losses and parameter-shift gradients. This module provides a
//! small, dependency-free, exact simulator that applies arbitrary one- and
//! two-qubit gates (via their dense `matrix()`) to a full `2^n` amplitude
//! vector.
//!
//! # Qubit convention
//!
//! Qubit `q` corresponds to bit `q` of the basis-state index (little-endian),
//! i.e. amplitude index `i` has qubit `q` in state `(i >> q) & 1`. This matches
//! the convention used by the per-gate applicators in
//! [`crate::qml::advanced_algorithms`] and the gate `matrix()` definitions in
//! [`crate::gate::functions`].

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::gate::GateOp;
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;

/// Allocate the `|0…0⟩` computational-basis state for `num_qubits` qubits.
#[must_use]
pub fn zero_state(num_qubits: usize) -> Array1<Complex64> {
    let dim = 1usize << num_qubits;
    let mut state = Array1::zeros(dim);
    state[0] = Complex64::new(1.0, 0.0);
    state
}

/// Apply a single one- or two-qubit gate to `state` in place.
///
/// The gate's dense matrix (row-major, `2^k × 2^k` for a `k`-qubit gate) is
/// applied to the sub-amplitudes selected by the gate's target qubits. Only
/// one- and two-qubit gates are supported; larger gates return an honest error
/// rather than silently leaving the state unchanged.
pub fn apply_gate(state: &mut Array1<Complex64>, gate: &dyn GateOp) -> QuantRS2Result<()> {
    let qubits = gate.qubits();
    let matrix = gate.matrix()?;

    match qubits.len() {
        1 => apply_one_qubit(state, qubits[0].0 as usize, &matrix),
        2 => apply_two_qubit(state, qubits[0].0 as usize, qubits[1].0 as usize, &matrix),
        k => Err(QuantRS2Error::UnsupportedOperation(format!(
            "state-vector simulator only supports 1- and 2-qubit gates, got {k}-qubit gate '{}'",
            gate.name()
        ))),
    }
}

/// Apply a sequence of gates to a fresh `|0…0⟩` state and return the final
/// amplitude vector.
pub fn simulate(num_qubits: usize, gates: &[Box<dyn GateOp>]) -> QuantRS2Result<Array1<Complex64>> {
    let mut state = zero_state(num_qubits);
    for gate in gates {
        apply_gate(&mut state, gate.as_ref())?;
    }
    Ok(state)
}

/// Apply a `2x2` matrix `[[m0, m1], [m2, m3]]` (row-major) to qubit `target`.
fn apply_one_qubit(
    state: &mut Array1<Complex64>,
    target: usize,
    matrix: &[Complex64],
) -> QuantRS2Result<()> {
    if matrix.len() != 4 {
        return Err(QuantRS2Error::InvalidInput(format!(
            "one-qubit gate matrix must have 4 entries, got {}",
            matrix.len()
        )));
    }
    let dim = state.len();
    let bit = 1usize << target;
    if bit >= dim {
        return Err(QuantRS2Error::InvalidInput(format!(
            "qubit index {target} out of range for {dim}-amplitude state"
        )));
    }

    let mut idx = 0;
    while idx < dim {
        if idx & bit == 0 {
            let i0 = idx;
            let i1 = idx | bit;
            let a = state[i0];
            let b = state[i1];
            state[i0] = matrix[0] * a + matrix[1] * b;
            state[i1] = matrix[2] * a + matrix[3] * b;
        }
        idx += 1;
    }
    Ok(())
}

/// Apply a `4x4` matrix (row-major) to qubits `q_high`/`q_low`.
///
/// The matrix is indexed by the two-bit value `(b_first << 1) | b_second`,
/// where `b_first` is the bit of `q_first` and `b_second` is the bit of
/// `q_second` (matching the ordering produced by `gate.qubits()`).
fn apply_two_qubit(
    state: &mut Array1<Complex64>,
    q_first: usize,
    q_second: usize,
    matrix: &[Complex64],
) -> QuantRS2Result<()> {
    if matrix.len() != 16 {
        return Err(QuantRS2Error::InvalidInput(format!(
            "two-qubit gate matrix must have 16 entries, got {}",
            matrix.len()
        )));
    }
    if q_first == q_second {
        return Err(QuantRS2Error::InvalidInput(
            "two-qubit gate requires two distinct qubits".to_string(),
        ));
    }
    let dim = state.len();
    let bit_first = 1usize << q_first;
    let bit_second = 1usize << q_second;
    if bit_first >= dim || bit_second >= dim {
        return Err(QuantRS2Error::InvalidInput(format!(
            "qubit index ({q_first},{q_second}) out of range for {dim}-amplitude state"
        )));
    }

    let mut idx = 0;
    while idx < dim {
        // Only process indices where both target bits are 0; the other three
        // members of the 2-qubit subspace are derived from this base.
        if idx & bit_first == 0 && idx & bit_second == 0 {
            let i00 = idx;
            let i01 = idx | bit_second;
            let i10 = idx | bit_first;
            let i11 = idx | bit_first | bit_second;
            let amps = [state[i00], state[i01], state[i10], state[i11]];
            for (row, target_idx) in [i00, i01, i10, i11].into_iter().enumerate() {
                let mut acc = Complex64::new(0.0, 0.0);
                for (col, amp) in amps.iter().enumerate() {
                    acc += matrix[row * 4 + col] * *amp;
                }
                state[target_idx] = acc;
            }
        }
        idx += 1;
    }
    Ok(())
}

/// Probability of measuring qubit `target` in state `|1⟩`.
#[must_use]
pub fn probability_one(state: &Array1<Complex64>, target: usize) -> f64 {
    let bit = 1usize << target;
    state
        .iter()
        .enumerate()
        .filter(|(i, _)| i & bit != 0)
        .map(|(_, a)| a.norm_sqr())
        .sum()
}

/// Expectation value of the Pauli-Z operator on qubit `target`,
/// `⟨Z⟩ = P(|0⟩) − P(|1⟩) ∈ [−1, 1]`.
#[must_use]
pub fn expectation_z(state: &Array1<Complex64>, target: usize) -> f64 {
    let p1 = probability_one(state, target);
    1.0 - 2.0 * p1
}

/// Full probability distribution over all `2^n` computational basis states.
#[must_use]
pub fn probabilities(state: &Array1<Complex64>) -> Vec<f64> {
    state.iter().map(scirs2_core::Complex::norm_sqr).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::multi::CNOT;
    use crate::gate::single::{Hadamard, PauliX, RotationY};
    use crate::qubit::QubitId;

    #[test]
    fn test_pauli_x_flips_qubit() {
        let mut state = zero_state(1);
        let x = PauliX { target: QubitId(0) };
        apply_gate(&mut state, &x).expect("apply X");
        // |0> -> |1>
        assert!((state[1].norm() - 1.0).abs() < 1e-12);
        assert!(state[0].norm() < 1e-12);
        assert!((expectation_z(&state, 0) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_hadamard_superposition() {
        let mut state = zero_state(1);
        let h = Hadamard { target: QubitId(0) };
        apply_gate(&mut state, &h).expect("apply H");
        assert!((probability_one(&state, 0) - 0.5).abs() < 1e-12);
        assert!(expectation_z(&state, 0).abs() < 1e-12);
    }

    #[test]
    fn test_bell_state_entanglement() {
        // H on q0, CNOT(0->1) yields (|00> + |11>)/sqrt(2)
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: QubitId(0) }),
            Box::new(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            }),
        ];
        let state = simulate(2, &gates).expect("simulate bell");
        let probs = probabilities(&state);
        assert!((probs[0b00] - 0.5).abs() < 1e-12);
        assert!((probs[0b11] - 0.5).abs() < 1e-12);
        assert!(probs[0b01] < 1e-12);
        assert!(probs[0b10] < 1e-12);
    }

    #[test]
    fn test_rotation_y_expectation_is_continuous() {
        // RY(theta)|0> gives <Z> = cos(theta); verify it is a real, theta-dependent value.
        let theta = 0.7;
        let mut state = zero_state(1);
        let ry = RotationY {
            target: QubitId(0),
            theta,
        };
        apply_gate(&mut state, &ry).expect("apply RY");
        assert!((expectation_z(&state, 0) - theta.cos()).abs() < 1e-10);
    }

    #[test]
    fn test_two_qubit_gate_on_high_index_qubits() {
        // CNOT with control=1, target=0 on |10> (q1=1) -> |11>.
        let mut state = zero_state(2);
        // set |10>: q1 = 1
        state[0] = Complex64::new(0.0, 0.0);
        state[0b10] = Complex64::new(1.0, 0.0);
        let cnot = CNOT {
            control: QubitId(1),
            target: QubitId(0),
        };
        apply_gate(&mut state, &cnot).expect("apply CNOT");
        assert!((state[0b11].norm() - 1.0).abs() < 1e-12);
    }
}
