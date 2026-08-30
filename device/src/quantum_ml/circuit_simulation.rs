//! Self-contained dense state-vector simulator for QML circuits.
//!
//! The quantum-ML algorithms in this crate (VQE, QAOA, QNNs, gradient
//! estimators, trainers) operate on [`ParameterizedQuantumCircuit`] values
//! and need *real* measurement statistics / expectation values to drive their
//! optimisation loops.  Historically the `execute_circuit_helper` methods in
//! those modules returned a hard-coded 50/50 split of `|0…0⟩` and `|1…1⟩`
//! counts, which silently fed fabricated data into every gradient and training
//! computation.
//!
//! This module provides a small, exact, in-crate state-vector engine so those
//! helpers can produce genuine results.  It is intentionally self-contained:
//! `quantrs2-device` must *not* depend on `quantrs2-sim` (that crate is a
//! sibling consumer, and adding it here would create a cross-dependency), so we
//! implement the few gates that [`QuantumGate`] can express directly.
//!
//! For paths that genuinely require execution on remote hardware (with
//! credentials / network access), callers should return an honest
//! [`DeviceError`] instead of using this local simulator.

use std::collections::HashMap;

use scirs2_core::Complex64;

use super::variational_algorithms::{ParameterizedQuantumCircuit, QuantumGate};
use crate::{CircuitResult, DeviceError, DeviceResult};

/// Maximum number of qubits this exact simulator will allocate a state vector
/// for.  `2^30` complex amplitudes is already 16 GiB, so we cap well below that
/// and return an honest error rather than attempting an impossible allocation.
const MAX_SIMULATED_QUBITS: usize = 26;

/// Simulate a [`ParameterizedQuantumCircuit`] from the all-zero state and
/// return the resulting amplitude vector of length `2^num_qubits`.
///
/// Amplitudes are indexed in the little-endian convention used throughout the
/// framework: qubit `q` is bit `q` of the basis index (so qubit 0 is the least
/// significant bit).
pub fn simulate_statevector(circuit: &ParameterizedQuantumCircuit) -> DeviceResult<Vec<Complex64>> {
    let num_qubits = circuit.num_qubits();
    if num_qubits > MAX_SIMULATED_QUBITS {
        return Err(DeviceError::InvalidInput(format!(
            "Local state-vector simulation supports at most {MAX_SIMULATED_QUBITS} qubits, \
             but circuit has {num_qubits}"
        )));
    }

    let dim = 1usize << num_qubits;
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    state[0] = Complex64::new(1.0, 0.0);

    for gate in circuit.gates() {
        apply_gate(&mut state, num_qubits, gate)?;
    }

    Ok(state)
}

/// Apply one [`QuantumGate`] to `state` in place.
fn apply_gate(state: &mut [Complex64], num_qubits: usize, gate: &QuantumGate) -> DeviceResult<()> {
    match *gate {
        QuantumGate::H(q) => {
            let s = std::f64::consts::FRAC_1_SQRT_2;
            apply_single_qubit(
                state,
                num_qubits,
                q,
                [
                    Complex64::new(s, 0.0),
                    Complex64::new(s, 0.0),
                    Complex64::new(s, 0.0),
                    Complex64::new(-s, 0.0),
                ],
            )
        }
        QuantumGate::X(q) => apply_single_qubit(
            state,
            num_qubits,
            q,
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        ),
        QuantumGate::Y(q) => apply_single_qubit(
            state,
            num_qubits,
            q,
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
        ),
        QuantumGate::Z(q) => apply_single_qubit(
            state,
            num_qubits,
            q,
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        ),
        QuantumGate::SDagger(q) => apply_single_qubit(
            state,
            num_qubits,
            q,
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
            ],
        ),
        QuantumGate::RX(q, theta) => {
            let c = (theta / 2.0).cos();
            let s = (theta / 2.0).sin();
            apply_single_qubit(
                state,
                num_qubits,
                q,
                [
                    Complex64::new(c, 0.0),
                    Complex64::new(0.0, -s),
                    Complex64::new(0.0, -s),
                    Complex64::new(c, 0.0),
                ],
            )
        }
        QuantumGate::RY(q, theta) => {
            let c = (theta / 2.0).cos();
            let s = (theta / 2.0).sin();
            apply_single_qubit(
                state,
                num_qubits,
                q,
                [
                    Complex64::new(c, 0.0),
                    Complex64::new(-s, 0.0),
                    Complex64::new(s, 0.0),
                    Complex64::new(c, 0.0),
                ],
            )
        }
        QuantumGate::RZ(q, theta) => {
            let phase_neg = Complex64::from_polar(1.0, -theta / 2.0);
            let phase_pos = Complex64::from_polar(1.0, theta / 2.0);
            apply_single_qubit(
                state,
                num_qubits,
                q,
                [
                    phase_neg,
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    phase_pos,
                ],
            )
        }
        QuantumGate::CNOT(control, target) => {
            apply_controlled_x(state, num_qubits, control, target)
        }
        QuantumGate::CZ(control, target) => apply_controlled_z(state, num_qubits, control, target),
    }
}

/// Apply a 2x2 unitary (row-major `[m00, m01, m10, m11]`) to qubit `q`.
fn apply_single_qubit(
    state: &mut [Complex64],
    num_qubits: usize,
    q: usize,
    matrix: [Complex64; 4],
) -> DeviceResult<()> {
    if q >= num_qubits {
        return Err(DeviceError::InvalidInput(format!(
            "Gate targets qubit {q} but circuit only has {num_qubits} qubits"
        )));
    }
    let bit = 1usize << q;
    let dim = state.len();
    for base in 0..dim {
        if base & bit != 0 {
            continue;
        }
        let i0 = base;
        let i1 = base | bit;
        let a0 = state[i0];
        let a1 = state[i1];
        state[i0] = matrix[0] * a0 + matrix[1] * a1;
        state[i1] = matrix[2] * a0 + matrix[3] * a1;
    }
    Ok(())
}

/// Apply a CNOT: flip `target` when `control` is set.
fn apply_controlled_x(
    state: &mut [Complex64],
    num_qubits: usize,
    control: usize,
    target: usize,
) -> DeviceResult<()> {
    validate_two_qubit(num_qubits, control, target)?;
    let control_bit = 1usize << control;
    let target_bit = 1usize << target;
    let dim = state.len();
    for base in 0..dim {
        // Only act when control is set and target is 0, swapping with the
        // target-is-1 partner exactly once.
        if (base & control_bit != 0) && (base & target_bit == 0) {
            let partner = base | target_bit;
            state.swap(base, partner);
        }
    }
    Ok(())
}

/// Apply a controlled-Z: phase of -1 on `|11⟩` of (control, target).
fn apply_controlled_z(
    state: &mut [Complex64],
    num_qubits: usize,
    control: usize,
    target: usize,
) -> DeviceResult<()> {
    validate_two_qubit(num_qubits, control, target)?;
    let control_bit = 1usize << control;
    let target_bit = 1usize << target;
    let dim = state.len();
    for (idx, amp) in state.iter_mut().enumerate().take(dim) {
        if (idx & control_bit != 0) && (idx & target_bit != 0) {
            *amp = -*amp;
        }
    }
    Ok(())
}

fn validate_two_qubit(num_qubits: usize, control: usize, target: usize) -> DeviceResult<()> {
    if control >= num_qubits || target >= num_qubits {
        return Err(DeviceError::InvalidInput(format!(
            "Two-qubit gate on ({control}, {target}) but circuit only has {num_qubits} qubits"
        )));
    }
    if control == target {
        return Err(DeviceError::InvalidInput(
            "Two-qubit gate requires distinct control and target qubits".to_string(),
        ));
    }
    Ok(())
}

/// Compute the exact probability of each computational-basis outcome.
///
/// Returns a vector of length `2^num_qubits` where index `i` is `|⟨i|ψ⟩|²`.
pub fn outcome_probabilities(state: &[Complex64]) -> Vec<f64> {
    state.iter().map(|amp| amp.norm_sqr()).collect()
}

/// Render a basis index as a bitstring with qubit 0 as the **leftmost**
/// character (matching the `"0".repeat(n)` / `"1".repeat(n)` convention the
/// previous mock used and that the expectation helpers parse).
fn index_to_bitstring(index: usize, num_qubits: usize) -> String {
    let mut s = String::with_capacity(num_qubits);
    for q in 0..num_qubits {
        if index & (1usize << q) != 0 {
            s.push('1');
        } else {
            s.push('0');
        }
    }
    s
}

/// Simulate `circuit` and sample `shots` measurement outcomes from the exact
/// output distribution, returning a [`CircuitResult`] with real counts.
///
/// Sampling uses [`fastrand`] (the crate's existing RNG dependency).  The
/// returned counts are genuine multinomial draws from `|⟨i|ψ⟩|²`, so they
/// reflect the true circuit (e.g. a Bell circuit yields only correlated
/// `00`/`11` outcomes, never the uniform spread the old mock produced).
pub fn simulate_and_sample(
    circuit: &ParameterizedQuantumCircuit,
    shots: usize,
) -> DeviceResult<CircuitResult> {
    let num_qubits = circuit.num_qubits();
    let state = simulate_statevector(circuit)?;
    let probabilities = outcome_probabilities(&state);

    // Build a cumulative distribution for inverse-transform sampling.
    let total: f64 = probabilities.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return Err(DeviceError::ExecutionFailed(
            "Circuit produced a non-normalizable state (zero or non-finite total probability)"
                .to_string(),
        ));
    }

    let mut cumulative = Vec::with_capacity(probabilities.len());
    let mut running = 0.0;
    for p in &probabilities {
        running += p / total;
        cumulative.push(running);
    }
    // Guard the last bin against floating-point shortfall.
    if let Some(last) = cumulative.last_mut() {
        *last = 1.0;
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    for _ in 0..shots {
        let r = fastrand::f64();
        let idx = match cumulative
            .binary_search_by(|probe| probe.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(i) | Err(i) => i.min(cumulative.len().saturating_sub(1)),
        };
        *counts
            .entry(index_to_bitstring(idx, num_qubits))
            .or_insert(0) += 1;
    }

    let mut metadata = HashMap::new();
    metadata.insert("backend".to_string(), "local_statevector".to_string());
    metadata.insert("num_qubits".to_string(), num_qubits.to_string());

    Ok(CircuitResult {
        counts,
        shots,
        metadata,
    })
}

/// Compute the exact expectation value of the total-spin (number-of-ones)
/// observable `Σ_q (1 - Z_q)/2 = Σ_q n_q`, i.e. the expected Hamming weight of
/// a measurement outcome, directly from the state vector.
///
/// This is the noiseless counterpart of the count-based estimator used by the
/// gradient/training code and is convenient for analytic tests.
pub fn expected_hamming_weight(circuit: &ParameterizedQuantumCircuit) -> DeviceResult<f64> {
    let num_qubits = circuit.num_qubits();
    let state = simulate_statevector(circuit)?;
    let mut expectation = 0.0;
    for (idx, amp) in state.iter().enumerate() {
        let weight = (idx.count_ones()) as f64;
        expectation += weight * amp.norm_sqr();
    }
    Ok(expectation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bell_state_is_correlated_not_uniform() {
        // |00> -> H on q0 -> CNOT(0,1) gives (|00> + |11>)/sqrt(2).
        let mut circuit = ParameterizedQuantumCircuit::new(2);
        circuit.add_h_gate(0).unwrap();
        circuit.add_cnot_gate(0, 1).unwrap();

        let probs = outcome_probabilities(&simulate_statevector(&circuit).unwrap());
        // Indices: 00 -> 0, 11 -> 3.
        assert!((probs[0] - 0.5).abs() < 1e-9, "P(00) should be 0.5");
        assert!((probs[3] - 0.5).abs() < 1e-9, "P(11) should be 0.5");
        assert!(probs[1].abs() < 1e-9, "P(01) should be 0");
        assert!(probs[2].abs() < 1e-9, "P(10) should be 0");

        // Sampled counts must respect the correlation (no 01/10 outcomes), and
        // must NOT be the old fabricated 50/50 of 00/11-as-all-zeros/all-ones
        // uniform mock — here all weight is on the two correlated strings.
        let result = simulate_and_sample(&circuit, 4096).unwrap();
        let c01 = result.counts.get("10").copied().unwrap_or(0); // qubit0=1,qubit1=0
        let c10 = result.counts.get("01").copied().unwrap_or(0);
        assert_eq!(c01, 0, "Bell state must never measure 01");
        assert_eq!(c10, 0, "Bell state must never measure 10");
        let c00 = result.counts.get("00").copied().unwrap_or(0);
        let c11 = result.counts.get("11").copied().unwrap_or(0);
        assert_eq!(c00 + c11, 4096);
        // Both should appear with finite frequency (probabilistic but extremely
        // unlikely to be 0 over 4096 shots).
        assert!(c00 > 0 && c11 > 0, "both correlated outcomes should appear");
    }

    #[test]
    fn x_gate_flips_qubit() {
        let mut circuit = ParameterizedQuantumCircuit::new(1);
        circuit.add_x_gate(0).unwrap();
        let probs = outcome_probabilities(&simulate_statevector(&circuit).unwrap());
        assert!(probs[1] > 0.999, "X|0> = |1>");
        assert_eq!(expected_hamming_weight(&circuit).unwrap().round() as i64, 1);
    }

    #[test]
    fn ry_rotation_matches_analytic_probability() {
        // RY(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>.
        let theta = 0.7;
        let mut circuit = ParameterizedQuantumCircuit::new(1);
        circuit.add_ry_gate(0, theta).unwrap();
        let probs = outcome_probabilities(&simulate_statevector(&circuit).unwrap());
        let expected_p1 = (theta / 2.0).sin().powi(2);
        assert!((probs[1] - expected_p1).abs() < 1e-9);
        let weight = expected_hamming_weight(&circuit).unwrap();
        assert!((weight - expected_p1).abs() < 1e-9);
    }

    #[test]
    fn rejects_oversized_circuit() {
        let circuit = ParameterizedQuantumCircuit::new(MAX_SIMULATED_QUBITS + 1);
        assert!(simulate_statevector(&circuit).is_err());
    }
}
