//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::*;
use scirs2_core::{Complex64, Float};

use super::types::QulacsStateVector;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qulacs_backend::circuit_api::{self, QulacsCircuit};
    use crate::qulacs_backend::gates;
    use crate::qulacs_backend::observable;
    use crate::qulacs_backend::observable::PauliObservable;
    use scirs2_core::Float;
    use std::f64::consts::PI;
    use std::sync::Arc;
    #[test]
    fn test_state_creation() {
        let state = QulacsStateVector::new(2).unwrap();
        assert_eq!(state.num_qubits(), 2);
        assert_eq!(state.dim(), 4);
        assert!((state.norm_squared() - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_hadamard_gate() {
        let mut state = QulacsStateVector::new(1).unwrap();
        gates::hadamard(&mut state, 0).unwrap();
        let expected_amp = 1.0 / 2.0f64.sqrt();
        assert!((state.amplitudes()[0].re - expected_amp).abs() < 1e-10);
        assert!((state.amplitudes()[1].re - expected_amp).abs() < 1e-10);
    }
    #[test]
    fn test_pauli_x_gate() {
        let mut state = QulacsStateVector::new(1).unwrap();
        gates::pauli_x(&mut state, 0).unwrap();
        assert!((state.amplitudes()[0].norm() - 0.0).abs() < 1e-10);
        assert!((state.amplitudes()[1].norm() - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_cnot_gate() {
        let mut state = QulacsStateVector::new(2).unwrap();
        gates::pauli_x(&mut state, 0).unwrap();
        gates::pauli_x(&mut state, 1).unwrap();
        assert!((state.amplitudes()[3].norm() - 1.0).abs() < 1e-10);
        gates::cnot(&mut state, 0, 1).unwrap();
        assert!((state.amplitudes()[0].norm() - 0.0).abs() < 1e-10);
        assert!((state.amplitudes()[1].norm() - 1.0).abs() < 1e-10);
        assert!((state.amplitudes()[2].norm() - 0.0).abs() < 1e-10);
        assert!((state.amplitudes()[3].norm() - 0.0).abs() < 1e-10);
    }
    #[test]
    fn test_bell_state() {
        let mut state = QulacsStateVector::new(2).unwrap();
        gates::hadamard(&mut state, 0).unwrap();
        gates::cnot(&mut state, 0, 1).unwrap();
        let expected_amp = 1.0 / 2.0f64.sqrt();
        assert!((state.amplitudes()[0].re - expected_amp).abs() < 1e-10);
        assert!((state.amplitudes()[1].norm() - 0.0).abs() < 1e-10);
        assert!((state.amplitudes()[2].norm() - 0.0).abs() < 1e-10);
        assert!((state.amplitudes()[3].re - expected_amp).abs() < 1e-10);
    }
    #[test]
    fn test_norm_squared() {
        let state = QulacsStateVector::new(3).unwrap();
        assert!((state.norm_squared() - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_inner_product() {
        let state1 = QulacsStateVector::new(2).unwrap();
        let state2 = QulacsStateVector::new(2).unwrap();
        let inner = state1.inner_product(&state2).unwrap();
        assert!((inner.re - 1.0).abs() < 1e-10);
        assert!(inner.im.abs() < 1e-10);
    }
    #[test]
    fn test_rx_gate() {
        use std::f64::consts::PI;
        let mut state1 = QulacsStateVector::new(1).unwrap();
        gates::rx(&mut state1, 0, PI).unwrap();
        let mut state2 = QulacsStateVector::new(1).unwrap();
        gates::pauli_x(&mut state2, 0).unwrap();
        assert!(state1.amplitudes()[0].norm() < 1e-10);
        assert!((state1.amplitudes()[1].norm() - 1.0).abs() < 1e-10);
        let mut state3 = QulacsStateVector::new(1).unwrap();
        gates::rx(&mut state3, 0, PI / 2.0).unwrap();
        let expected = 1.0 / 2.0f64.sqrt();
        assert!((state3.amplitudes()[0].re - expected).abs() < 1e-10);
        assert!(state3.amplitudes()[0].im.abs() < 1e-10);
        assert!(state3.amplitudes()[1].re.abs() < 1e-10);
        assert!((state3.amplitudes()[1].im + expected).abs() < 1e-10);
    }
    #[test]
    fn test_ry_gate() {
        let mut state = QulacsStateVector::new(1).unwrap();
        gates::ry(&mut state, 0, PI / 2.0).unwrap();
        let expected = 1.0 / 2.0f64.sqrt();
        assert!((state.amplitudes()[0].re - expected).abs() < 1e-10);
        assert!((state.amplitudes()[1].re - expected).abs() < 1e-10);
        assert!(state.amplitudes()[0].im.abs() < 1e-10);
        assert!(state.amplitudes()[1].im.abs() < 1e-10);
        let mut bell_state = QulacsStateVector::new(2).unwrap();
        gates::ry(&mut bell_state, 0, PI / 2.0).unwrap();
        gates::cnot(&mut bell_state, 0, 1).unwrap();
        assert!((bell_state.amplitudes()[0].norm() - expected).abs() < 1e-10);
        assert!((bell_state.amplitudes()[3].norm() - expected).abs() < 1e-10);
    }
    #[test]
    fn test_rz_gate() {
        let mut state1 = QulacsStateVector::new(1).unwrap();
        gates::pauli_x(&mut state1, 0).unwrap();
        gates::rz(&mut state1, 0, PI).unwrap();
        assert!(state1.amplitudes()[0].norm() < 1e-10);
        assert!((state1.amplitudes()[1].norm() - 1.0).abs() < 1e-10);
        let mut state2 = QulacsStateVector::new(1).unwrap();
        gates::hadamard(&mut state2, 0).unwrap();
        gates::rz(&mut state2, 0, PI / 4.0).unwrap();
        assert!((state2.amplitudes()[0].norm() - 1.0 / 2.0f64.sqrt()).abs() < 1e-10);
        assert!((state2.amplitudes()[1].norm() - 1.0 / 2.0f64.sqrt()).abs() < 1e-10);
    }
    #[test]
    fn test_phase_gate() {
        let mut state = QulacsStateVector::new(1).unwrap();
        gates::pauli_x(&mut state, 0).unwrap();
        gates::phase(&mut state, 0, PI / 2.0).unwrap();
        assert!(state.amplitudes()[0].norm() < 1e-10);
        assert!((state.amplitudes()[1].re).abs() < 1e-10);
        assert!((state.amplitudes()[1].im - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_u3_gate() {
        let mut state1 = QulacsStateVector::new(1).unwrap();
        gates::u3(&mut state1, 0, PI / 2.0, 0.0, PI).unwrap();
        let mut state2 = QulacsStateVector::new(1).unwrap();
        gates::hadamard(&mut state2, 0).unwrap();
        let ratio = state1.amplitudes()[0] / state2.amplitudes()[0];
        assert!((state1.amplitudes()[0] / ratio - state2.amplitudes()[0]).norm() < 1e-10);
        assert!((state1.amplitudes()[1] / ratio - state2.amplitudes()[1]).norm() < 1e-10);
        let mut state3 = QulacsStateVector::new(1).unwrap();
        gates::u3(&mut state3, 0, PI, 0.0, PI).unwrap();
        let mut state4 = QulacsStateVector::new(1).unwrap();
        gates::pauli_x(&mut state4, 0).unwrap();
        let ratio2 = state3.amplitudes()[1] / state4.amplitudes()[1];
        assert!((state3.amplitudes()[0] / ratio2 - state4.amplitudes()[0]).norm() < 1e-10);
        assert!((state3.amplitudes()[1] / ratio2 - state4.amplitudes()[1]).norm() < 1e-10);
    }
    #[test]
    fn test_rotation_gates_on_multi_qubit_state() {
        let mut state = QulacsStateVector::new(3).unwrap();
        gates::ry(&mut state, 0, PI / 2.0).unwrap();
        gates::rx(&mut state, 1, PI / 4.0).unwrap();
        gates::rz(&mut state, 2, PI / 3.0).unwrap();
        assert!((state.norm_squared() - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_rotation_composition() {
        let mut state1 = QulacsStateVector::new(1).unwrap();
        gates::rx(&mut state1, 0, PI / 4.0).unwrap();
        gates::ry(&mut state1, 0, PI / 4.0).unwrap();
        gates::rz(&mut state1, 0, PI / 4.0).unwrap();
        assert!((state1.norm_squared() - 1.0).abs() < 1e-10);
        let mut state2 = QulacsStateVector::new(1).unwrap();
        gates::u3(&mut state2, 0, PI / 4.0, PI / 4.0, PI / 4.0).unwrap();
        assert!((state2.norm_squared() - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_probability_calculation() {
        let state0 = QulacsStateVector::new(1).unwrap();
        assert!((state0.probability_zero(0).unwrap() - 1.0).abs() < 1e-10);
        assert!(state0.probability_one(0).unwrap().abs() < 1e-10);
        let mut state1 = QulacsStateVector::new(1).unwrap();
        gates::pauli_x(&mut state1, 0).unwrap();
        assert!(state1.probability_zero(0).unwrap().abs() < 1e-10);
        assert!((state1.probability_one(0).unwrap() - 1.0).abs() < 1e-10);
        let mut state_plus = QulacsStateVector::new(1).unwrap();
        gates::hadamard(&mut state_plus, 0).unwrap();
        assert!((state_plus.probability_zero(0).unwrap() - 0.5).abs() < 1e-10);
        assert!((state_plus.probability_one(0).unwrap() - 0.5).abs() < 1e-10);
    }
    #[test]
    fn test_measurement_collapse() {
        let mut outcomes_0 = 0;
        let mut outcomes_1 = 0;
        let num_trials = 1000;
        for _ in 0..num_trials {
            let mut state = QulacsStateVector::new(1).unwrap();
            gates::hadamard(&mut state, 0).unwrap();
            let outcome = state.measure(0).unwrap();
            if outcome {
                outcomes_1 += 1;
            } else {
                outcomes_0 += 1;
            }
            assert!((state.norm_squared() - 1.0).abs() < 1e-10);
            if outcome {
                assert!((state.probability_one(0).unwrap() - 1.0).abs() < 1e-10);
            } else {
                assert!((state.probability_zero(0).unwrap() - 1.0).abs() < 1e-10);
            }
        }
        let ratio = outcomes_1 as f64 / num_trials as f64;
        assert!(ratio > 0.4 && ratio < 0.6, "Ratio: {}", ratio);
    }
    #[test]
    fn test_sampling() {
        let mut bell_state = QulacsStateVector::new(2).unwrap();
        gates::hadamard(&mut bell_state, 0).unwrap();
        gates::cnot(&mut bell_state, 0, 1).unwrap();
        let samples = bell_state.sample(1000).unwrap();
        assert_eq!(samples.len(), 1000);
        let mut count_00 = 0;
        let mut count_11 = 0;
        for bitstring in &samples {
            assert_eq!(bitstring.len(), 2);
            if !bitstring[0] && !bitstring[1] {
                count_00 += 1;
            } else if bitstring[0] && bitstring[1] {
                count_11 += 1;
            } else {
                panic!("Unexpected outcome: {:?}", bitstring);
            }
        }
        let ratio = count_00 as f64 / 1000.0;
        assert!(ratio > 0.4 && ratio < 0.6, "Ratio |00⟩: {}", ratio);
        assert!((bell_state.norm_squared() - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_get_counts() {
        let mut state = QulacsStateVector::new(1).unwrap();
        gates::hadamard(&mut state, 0).unwrap();
        let counts = state.get_counts(1000).unwrap();
        assert!(counts.len() <= 2);
        let count_0 = *counts.get(&vec![false]).unwrap_or(&0);
        let count_1 = *counts.get(&vec![true]).unwrap_or(&0);
        assert_eq!(count_0 + count_1, 1000);
        let ratio = count_1 as f64 / 1000.0;
        assert!(ratio > 0.4 && ratio < 0.6, "Ratio |1⟩: {}", ratio);
    }
    #[test]
    fn test_sample_qubits() {
        let mut bell = QulacsStateVector::new(2).unwrap();
        gates::hadamard(&mut bell, 0).unwrap();
        gates::cnot(&mut bell, 0, 1).unwrap();
        let samples = bell.sample_qubits(&[0], 1000).unwrap();
        let mut count_0 = 0;
        let mut count_1 = 0;
        for bitstring in &samples {
            assert_eq!(bitstring.len(), 1);
            if bitstring[0] {
                count_1 += 1;
            } else {
                count_0 += 1;
            }
        }
        let ratio = count_1 as f64 / 1000.0;
        assert!(ratio > 0.4 && ratio < 0.6, "Ratio: {}", ratio);
    }
    #[test]
    fn test_measurement_multi_qubit() {
        let mut bell_state = QulacsStateVector::new(2).unwrap();
        gates::hadamard(&mut bell_state, 0).unwrap();
        gates::cnot(&mut bell_state, 0, 1).unwrap();
        let outcome0 = bell_state.measure(0).unwrap();
        let outcome1 = bell_state.measure(1).unwrap();
        assert_eq!(outcome0, outcome1);
    }
    #[test]
    fn test_toffoli_gate() {
        let mut state = QulacsStateVector::new(3).unwrap();
        gates::toffoli(&mut state, 0, 1, 2).unwrap();
        assert!((state.amplitudes()[0].norm() - 1.0).abs() < 1e-10);
        assert!(state.amplitudes()[7].norm() < 1e-10);
        let mut state2 = QulacsStateVector::new(3).unwrap();
        gates::pauli_x(&mut state2, 0).unwrap();
        gates::pauli_x(&mut state2, 1).unwrap();
        gates::toffoli(&mut state2, 0, 1, 2).unwrap();
        assert!((state2.amplitudes()[7].norm() - 1.0).abs() < 1e-10);
        assert!(state2.amplitudes()[3].norm() < 1e-10);
        let mut state3 = QulacsStateVector::new(3).unwrap();
        gates::hadamard(&mut state3, 0).unwrap();
        gates::pauli_x(&mut state3, 0).unwrap();
        gates::pauli_x(&mut state3, 1).unwrap();
        gates::pauli_x(&mut state3, 2).unwrap();
        gates::toffoli(&mut state3, 0, 1, 2).unwrap();
        assert!((state3.norm_squared() - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_toffoli_reversibility() {
        let mut state1 = QulacsStateVector::new(3).unwrap();
        gates::hadamard(&mut state1, 0).unwrap();
        gates::hadamard(&mut state1, 1).unwrap();
        gates::hadamard(&mut state1, 2).unwrap();
        let original_state = state1.clone();
        gates::toffoli(&mut state1, 0, 1, 2).unwrap();
        gates::toffoli(&mut state1, 0, 1, 2).unwrap();
        for i in 0..8 {
            let diff = (state1.amplitudes()[i] - original_state.amplitudes()[i]).norm();
            assert!(diff < 1e-10, "Difference at index {}: {}", i, diff);
        }
    }
    #[test]
    fn test_fredkin_gate() {
        let mut state = QulacsStateVector::new(3).unwrap();
        gates::fredkin(&mut state, 0, 1, 2).unwrap();
        assert!((state.amplitudes()[0].norm() - 1.0).abs() < 1e-10);
        let mut state2 = QulacsStateVector::new(3).unwrap();
        gates::pauli_x(&mut state2, 0).unwrap();
        gates::pauli_x(&mut state2, 2).unwrap();
        assert!((state2.amplitudes()[0b101].norm() - 1.0).abs() < 1e-10);
        gates::fredkin(&mut state2, 0, 1, 2).unwrap();
        assert!((state2.amplitudes()[0b011].norm() - 1.0).abs() < 1e-10);
        assert!(state2.amplitudes()[0b101].norm() < 1e-10);
        let mut state3 = QulacsStateVector::new(3).unwrap();
        gates::pauli_x(&mut state3, 1).unwrap();
        gates::pauli_x(&mut state3, 2).unwrap();
        let before = state3.clone();
        gates::fredkin(&mut state3, 0, 1, 2).unwrap();
        for i in 0..8 {
            let diff = (state3.amplitudes()[i] - before.amplitudes()[i]).norm();
            assert!(diff < 1e-10);
        }
    }
    #[test]
    fn test_fredkin_reversibility() {
        let mut state1 = QulacsStateVector::new(3).unwrap();
        gates::hadamard(&mut state1, 0).unwrap();
        gates::hadamard(&mut state1, 1).unwrap();
        gates::hadamard(&mut state1, 2).unwrap();
        let original_state = state1.clone();
        gates::fredkin(&mut state1, 0, 1, 2).unwrap();
        gates::fredkin(&mut state1, 0, 1, 2).unwrap();
        for i in 0..8 {
            let diff = (state1.amplitudes()[i] - original_state.amplitudes()[i]).norm();
            assert!(diff < 1e-10, "Difference at index {}: {}", i, diff);
        }
    }
    #[test]
    fn test_toffoli_error_cases() {
        let mut state = QulacsStateVector::new(3).unwrap();
        assert!(gates::toffoli(&mut state, 0, 1, 5).is_err());
        assert!(gates::toffoli(&mut state, 5, 1, 2).is_err());
        assert!(gates::toffoli(&mut state, 0, 0, 2).is_err());
        assert!(gates::toffoli(&mut state, 0, 1, 1).is_err());
        assert!(gates::toffoli(&mut state, 0, 1, 0).is_err());
    }
    #[test]
    fn test_fredkin_error_cases() {
        let mut state = QulacsStateVector::new(3).unwrap();
        assert!(gates::fredkin(&mut state, 5, 1, 2).is_err());
        assert!(gates::fredkin(&mut state, 0, 5, 2).is_err());
        assert!(gates::fredkin(&mut state, 0, 1, 5).is_err());
        assert!(gates::fredkin(&mut state, 0, 0, 2).is_err());
        assert!(gates::fredkin(&mut state, 0, 1, 1).is_err());
        assert!(gates::fredkin(&mut state, 0, 1, 0).is_err());
    }
    #[test]
    fn test_s_gate() {
        let mut state = QulacsStateVector::new(1).unwrap();
        gates::pauli_x(&mut state, 0).unwrap();
        gates::s(&mut state, 0).unwrap();
        assert!((state.amplitudes()[0].norm() - 0.0).abs() < 1e-10);
        assert!((state.amplitudes()[1].re - 0.0).abs() < 1e-10);
        assert!((state.amplitudes()[1].im - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_s_dag_gate() {
        let mut state = QulacsStateVector::new(1).unwrap();
        gates::pauli_x(&mut state, 0).unwrap();
        gates::sdg(&mut state, 0).unwrap();
        assert!((state.amplitudes()[0].norm() - 0.0).abs() < 1e-10);
        assert!((state.amplitudes()[1].re - 0.0).abs() < 1e-10);
        assert!((state.amplitudes()[1].im + 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_s_s_dag_identity() {
        let mut state = QulacsStateVector::new(1).unwrap();
        gates::hadamard(&mut state, 0).unwrap();
        let original = state.clone();
        gates::s(&mut state, 0).unwrap();
        gates::sdg(&mut state, 0).unwrap();
        for i in 0..2 {
            let diff = (state.amplitudes()[i] - original.amplitudes()[i]).norm();
            assert!(diff < 1e-10);
        }
    }
    #[test]
    fn test_t_gate() {
        let mut state = QulacsStateVector::new(1).unwrap();
        gates::pauli_x(&mut state, 0).unwrap();
        gates::t(&mut state, 0).unwrap();
        let expected = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
        assert!((state.amplitudes()[0].norm() - 0.0).abs() < 1e-10);
        assert!((state.amplitudes()[1].re - expected.re).abs() < 1e-10);
        assert!((state.amplitudes()[1].im - expected.im).abs() < 1e-10);
    }
    #[test]
    fn test_t_dag_gate() {
        let mut state = QulacsStateVector::new(1).unwrap();
        gates::pauli_x(&mut state, 0).unwrap();
        gates::tdg(&mut state, 0).unwrap();
        let expected = Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_4);
        assert!((state.amplitudes()[0].norm() - 0.0).abs() < 1e-10);
        assert!((state.amplitudes()[1].re - expected.re).abs() < 1e-10);
        assert!((state.amplitudes()[1].im - expected.im).abs() < 1e-10);
    }
    #[test]
    fn test_t_t_dag_identity() {
        let mut state = QulacsStateVector::new(1).unwrap();
        gates::hadamard(&mut state, 0).unwrap();
        let original = state.clone();
        gates::t(&mut state, 0).unwrap();
        gates::tdg(&mut state, 0).unwrap();
        for i in 0..2 {
            let diff = (state.amplitudes()[i] - original.amplitudes()[i]).norm();
            assert!(diff < 1e-10);
        }
    }
    #[test]
    fn test_s_equals_two_t() {
        let mut state1 = QulacsStateVector::new(1).unwrap();
        let mut state2 = QulacsStateVector::new(1).unwrap();
        gates::hadamard(&mut state1, 0).unwrap();
        gates::hadamard(&mut state2, 0).unwrap();
        gates::s(&mut state1, 0).unwrap();
        gates::t(&mut state2, 0).unwrap();
        gates::t(&mut state2, 0).unwrap();
        for i in 0..2 {
            let diff = (state1.amplitudes()[i] - state2.amplitudes()[i]).norm();
            assert!(diff < 1e-10);
        }
    }
    #[test]
    fn test_pauli_operator_matrices() {
        use observable::PauliOperator;
        let i_mat = PauliOperator::I.matrix();
        assert_eq!(i_mat[[0, 0]], Complex64::new(1.0, 0.0));
        assert_eq!(i_mat[[1, 1]], Complex64::new(1.0, 0.0));
        let x_mat = PauliOperator::X.matrix();
        assert_eq!(x_mat[[0, 1]], Complex64::new(1.0, 0.0));
        assert_eq!(x_mat[[1, 0]], Complex64::new(1.0, 0.0));
        let z_mat = PauliOperator::Z.matrix();
        assert_eq!(z_mat[[0, 0]], Complex64::new(1.0, 0.0));
        assert_eq!(z_mat[[1, 1]], Complex64::new(-1.0, 0.0));
    }
    #[test]
    fn test_pauli_observable_creation() {
        use observable::PauliObservable;
        let obs_z = PauliObservable::pauli_z(&[0]);
        assert_eq!(obs_z.coefficient, 1.0);
        assert_eq!(obs_z.operators.len(), 1);
        let obs_x = PauliObservable::pauli_x(&[0, 1]);
        assert_eq!(obs_x.operators.len(), 2);
    }
    #[test]
    fn test_pauli_z_expectation_value() {
        let state = QulacsStateVector::new(1).unwrap();
        let obs = PauliObservable::pauli_z(&[0]);
        let exp_val = obs.expectation_value(&state);
        assert!((exp_val - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_pauli_z_expectation_value_excited() {
        let mut state = QulacsStateVector::new(1).unwrap();
        gates::pauli_x(&mut state, 0).unwrap();
        let obs = PauliObservable::pauli_z(&[0]);
        let exp_val = obs.expectation_value(&state);
        assert!((exp_val - (-1.0)).abs() < 1e-10);
    }
    #[test]
    fn test_pauli_z_expectation_value_superposition() {
        let mut state = QulacsStateVector::new(1).unwrap();
        gates::hadamard(&mut state, 0).unwrap();
        let obs = PauliObservable::pauli_z(&[0]);
        let exp_val = obs.expectation_value(&state);
        assert!(exp_val.abs() < 1e-10);
    }
    #[test]
    fn test_hermitian_observable() {
        use observable::HermitianObservable;
        let mut matrix = Array2::zeros((2, 2));
        matrix[[0, 0]] = Complex64::new(1.0, 0.0);
        matrix[[1, 1]] = Complex64::new(-1.0, 0.0);
        let obs = HermitianObservable::new(matrix).unwrap();
        assert_eq!(obs.num_qubits, 1);
        let state = QulacsStateVector::new(1).unwrap();
        let exp_val = obs.expectation_value(&state).unwrap();
        assert!((exp_val - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_composite_observable() {
        use observable::{CompositeObservable, PauliObservable};
        let obs1 = PauliObservable::pauli_z(&[0]).with_coefficient(0.5);
        let obs2 = PauliObservable::pauli_z(&[1]).with_coefficient(0.3);
        let composite = CompositeObservable::new().add_term(obs1).add_term(obs2);
        assert_eq!(composite.num_terms(), 2);
        let state = QulacsStateVector::new(2).unwrap();
        let exp_val = composite.expectation_value(&state);
        assert!((exp_val - 0.8).abs() < 1e-10);
    }
    #[test]
    fn test_observable_weight() {
        use observable::{PauliObservable, PauliOperator};
        use std::collections::HashMap;
        let mut operators = HashMap::new();
        operators.insert(0, PauliOperator::X);
        operators.insert(1, PauliOperator::Y);
        operators.insert(2, PauliOperator::I);
        let obs = PauliObservable::new(operators, 1.0);
        assert_eq!(obs.weight(), 2);
    }
    #[test]
    fn test_circuit_api_basic() {
        use circuit_api::QulacsCircuit;
        let mut circuit = QulacsCircuit::new(2).unwrap();
        assert_eq!(circuit.num_qubits(), 2);
        assert_eq!(circuit.gate_count(), 0);
        circuit.h(0).x(1);
        assert_eq!(circuit.gate_count(), 2);
    }
    #[test]
    fn test_circuit_api_bell_state() {
        let mut circuit = QulacsCircuit::new(2).unwrap();
        circuit.bell_pair(0, 1);
        assert_eq!(circuit.gate_count(), 2);
        let probs = circuit.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!(probs[1].abs() < 1e-10);
        assert!(probs[2].abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }
    #[test]
    fn test_circuit_api_run_shots() {
        let mut circuit = QulacsCircuit::new(2).unwrap();
        circuit.bell_pair(0, 1);
        let counts = circuit.run(100).unwrap();
        assert!(counts.contains_key("00") || counts.contains_key("11"));
        let total: usize = counts.values().sum();
        assert_eq!(total, 100);
    }
    #[test]
    fn test_circuit_api_reset() {
        let mut circuit = QulacsCircuit::new(2).unwrap();
        circuit.h(0).cnot(0, 1);
        assert_eq!(circuit.gate_count(), 2);
        circuit.reset().unwrap();
        assert_eq!(circuit.gate_count(), 0);
        let probs = circuit.probabilities();
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_circuit_api_rotation_gates() {
        let mut circuit = QulacsCircuit::new(1).unwrap();
        circuit.rx(0, PI);
        let probs = circuit.probabilities();
        assert!(probs[0].abs() < 1e-10);
        assert!((probs[1] - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_circuit_api_phase_gates() {
        let mut circuit = QulacsCircuit::new(1).unwrap();
        circuit.s(0).s(0);
        assert_eq!(circuit.gate_count(), 2);
        let probs = circuit.probabilities();
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_circuit_api_two_qubit_gates() {
        let mut circuit = QulacsCircuit::new(2).unwrap();
        circuit.x(0).cnot(0, 1);
        let probs = circuit.probabilities();
        assert!(probs[0].abs() < 1e-10);
        assert!(probs[1].abs() < 1e-10);
        assert!(probs[2].abs() < 1e-10);
        assert!((probs[3] - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_circuit_api_observable() {
        use circuit_api::{Observable, QulacsCircuit};
        let mut circuit = QulacsCircuit::new(2).unwrap();
        circuit.h(0).h(1);
        let obs = PauliObservable::pauli_z(&[0]);
        let exp_val = circuit.expectation(&obs).unwrap();
        assert!(exp_val.abs() < 1e-10);
    }
    #[test]
    fn test_circuit_api_gate_record() {
        let mut circuit = QulacsCircuit::new(2).unwrap();
        circuit.h(0).cnot(0, 1).rx(1, 1.5);
        let gates = circuit.gates();
        assert_eq!(gates.len(), 3);
        assert_eq!(gates[0].name, "H");
        assert_eq!(gates[0].qubits, vec![0]);
        assert_eq!(gates[1].name, "CNOT");
        assert_eq!(gates[1].qubits, vec![0, 1]);
        assert_eq!(gates[2].name, "RX");
        assert_eq!(gates[2].params[0], 1.5);
    }
    #[test]
    fn test_circuit_api_noise_model() {
        use crate::noise_models::{DepolarizingNoise, NoiseModel as KrausNoiseModel};
        use std::sync::Arc;
        let mut circuit = QulacsCircuit::new(2).unwrap();
        assert!(!circuit.has_noise_model());
        let mut noise_model = KrausNoiseModel::new();
        noise_model.add_channel(Arc::new(DepolarizingNoise::new(0.01)));
        circuit.set_noise_model(noise_model);
        assert!(circuit.has_noise_model());
        circuit.h(0).cnot(0, 1);
        assert_eq!(circuit.gate_count(), 2);
        circuit.clear_noise_model();
        assert!(!circuit.has_noise_model());
    }
    #[test]
    fn test_circuit_api_run_with_noise() {
        use crate::noise_models::{BitFlipNoise, NoiseModel as KrausNoiseModel};
        let mut circuit = QulacsCircuit::new(1).unwrap();
        let mut noise_model = KrausNoiseModel::new();
        noise_model.add_channel(Arc::new(BitFlipNoise::new(0.1)));
        circuit.set_noise_model(noise_model);
        let counts = circuit.run_with_noise(100).unwrap();
        let total: usize = counts.values().sum();
        assert_eq!(total, 100);
        assert!(counts.contains_key("0"));
    }
}
