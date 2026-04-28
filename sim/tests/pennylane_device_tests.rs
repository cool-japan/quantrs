//! Integration tests for the PennyLane device backend.

use quantrs2_sim::pennylane::device::{PennyLaneCircuit, PennyLaneObservable, PennyLaneOperation, QuantRS2Device};

fn make_hadamard_op(wire: usize) -> PennyLaneOperation {
    PennyLaneOperation {
        name: "Hadamard".to_string(),
        wires: vec![wire],
        params: vec![],
    }
}

fn make_cnot_op(control: usize, target: usize) -> PennyLaneOperation {
    PennyLaneOperation {
        name: "CNOT".to_string(),
        wires: vec![control, target],
        params: vec![],
    }
}

fn make_rx_op(wire: usize, theta: f64) -> PennyLaneOperation {
    PennyLaneOperation {
        name: "RX".to_string(),
        wires: vec![wire],
        params: vec![theta],
    }
}

fn make_pauliz_obs(wire: usize) -> PennyLaneObservable {
    PennyLaneObservable {
        name: "PauliZ".to_string(),
        wires: vec![wire],
    }
}

// ─── basic execution tests ────────────────────────────────────────────────────

#[test]
fn test_bell_state_probabilities() {
    let circuit = PennyLaneCircuit {
        num_wires: 2,
        operations: vec![make_hadamard_op(0), make_cnot_op(0, 1)],
        observables: vec![],
    };

    let device = QuantRS2Device::new();
    let result = device.execute(&circuit).expect("bell state");

    // |Φ+⟩ = (|00⟩ + |11⟩) / √2
    assert_eq!(result.probabilities.len(), 4, "wrong number of basis states");
    assert!(
        (result.probabilities[0] - 0.5).abs() < 1e-9,
        "P(|00⟩) = {} (expected 0.5)",
        result.probabilities[0]
    );
    assert!(
        result.probabilities[1].abs() < 1e-9,
        "P(|01⟩) = {} (expected 0)",
        result.probabilities[1]
    );
    assert!(
        result.probabilities[2].abs() < 1e-9,
        "P(|10⟩) = {} (expected 0)",
        result.probabilities[2]
    );
    assert!(
        (result.probabilities[3] - 0.5).abs() < 1e-9,
        "P(|11⟩) = {} (expected 0.5)",
        result.probabilities[3]
    );
}

#[test]
fn test_bell_state_amplitude_norms() {
    let circuit = PennyLaneCircuit {
        num_wires: 2,
        operations: vec![make_hadamard_op(0), make_cnot_op(0, 1)],
        observables: vec![],
    };

    let device = QuantRS2Device::new();
    let result = device.execute(&circuit).expect("bell state amplitudes");

    // Verify |ψ|² = 1
    let norm_sq: f64 = result
        .state_re
        .iter()
        .zip(result.state_im.iter())
        .map(|(r, i)| r * r + i * i)
        .sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-9,
        "state vector norm² = {norm_sq} (expected 1.0)"
    );
}

#[test]
fn test_bell_state_expval_pauliz() {
    let circuit = PennyLaneCircuit {
        num_wires: 2,
        operations: vec![make_hadamard_op(0), make_cnot_op(0, 1)],
        observables: vec![make_pauliz_obs(0)],
    };

    let device = QuantRS2Device::new();
    let result = device.execute(&circuit).expect("bell state expval");

    // ⟨Z⊗I⟩ for |Φ+⟩ = 0 (equal mix of |0⟩ and |1⟩ on wire 0)
    assert_eq!(result.expval.len(), 1);
    assert!(
        result.expval[0].abs() < 1e-9,
        "⟨Z⟩ = {} (expected 0)",
        result.expval[0]
    );
}

#[test]
fn test_identity_expval() {
    let circuit = PennyLaneCircuit {
        num_wires: 2,
        operations: vec![make_hadamard_op(0)],
        observables: vec![PennyLaneObservable {
            name: "Identity".to_string(),
            wires: vec![0],
        }],
    };

    let device = QuantRS2Device::new();
    let result = device.execute(&circuit).expect("identity expval");
    assert_eq!(result.expval.len(), 1);
    assert!((result.expval[0] - 1.0).abs() < 1e-9, "⟨I⟩ should be 1.0");
}

#[test]
fn test_all_zero_state_probabilities() {
    // No gates → |00⟩ state
    let circuit = PennyLaneCircuit {
        num_wires: 2,
        operations: vec![],
        observables: vec![make_pauliz_obs(0)],
    };

    let device = QuantRS2Device::new();
    let result = device.execute(&circuit).expect("|00> state");

    // Should have P(|00⟩) = 1.0
    assert!(
        (result.probabilities[0] - 1.0).abs() < 1e-9,
        "P(|00⟩) should be 1.0, got {}",
        result.probabilities[0]
    );

    // ⟨Z⟩ on qubit 0 of |0⟩ = +1
    assert!(
        (result.expval[0] - 1.0).abs() < 1e-9,
        "⟨Z⟩ on |0⟩ should be +1, got {}",
        result.expval[0]
    );
}

#[test]
fn test_rx_gate_half_pi() {
    // RX(π/2)|00⟩ on qubit 0 of a 2-qubit circuit
    let circuit = PennyLaneCircuit {
        num_wires: 2,
        operations: vec![make_rx_op(0, std::f64::consts::FRAC_PI_2)],
        observables: vec![make_pauliz_obs(0)],
    };

    let device = QuantRS2Device::new();
    let result = device.execute(&circuit).expect("rx(pi/2)");

    // RX(π/2) on qubit 0 of 2-qubit system (initial state |00⟩):
    // State → cos(π/4)|00⟩ - i·sin(π/4)|01⟩
    //
    // Using QuantRS2 LSB convention: state index bit 0 = qubit 0.
    // P(qubit0=0) = P(index 0) + P(index 2) (states where bit 0 = 0)
    // P(qubit0=1) = P(index 1) + P(index 3) (states where bit 0 = 1)
    assert_eq!(result.probabilities.len(), 4);
    let p_qubit0_zero = result.probabilities[0] + result.probabilities[2];
    let p_qubit0_one = result.probabilities[1] + result.probabilities[3];
    assert!(
        (p_qubit0_zero - 0.5).abs() < 1e-9,
        "P(qubit0=0) = {p_qubit0_zero} (expected 0.5)"
    );
    assert!(
        (p_qubit0_one - 0.5).abs() < 1e-9,
        "P(qubit0=1) = {p_qubit0_one} (expected 0.5)"
    );
}

// ─── JSON protocol tests ─────────────────────────────────────────────────────

#[test]
fn test_json_input_bell_state() {
    let device = QuantRS2Device::new();
    let json = r#"{"num_wires":2,"operations":[{"name":"Hadamard","wires":[0],"params":[]},{"name":"CNOT","wires":[0,1],"params":[]}],"observables":[]}"#;

    let result_json = device.execute_json(json).expect("json execution");

    // Deserialize and check
    use quantrs2_sim::pennylane::device::PennyLaneResult;
    let result: PennyLaneResult =
        serde_json::from_str(&result_json).expect("deserialize result");

    assert_eq!(result.probabilities.len(), 4);
    assert!((result.probabilities[0] - 0.5).abs() < 1e-9);
    assert!((result.probabilities[3] - 0.5).abs() < 1e-9);
}

#[test]
fn test_json_input_with_rotation_params() {
    let device = QuantRS2Device::new();
    let pi_str = std::f64::consts::PI.to_string();
    let json = format!(
        r#"{{"num_wires":2,"operations":[{{"name":"RX","wires":[0],"params":[{pi_str}]}}],"observables":[{{"name":"PauliZ","wires":[0]}}]}}"#
    );

    let result_json = device.execute_json(&json).expect("json rx execution");
    use quantrs2_sim::pennylane::device::PennyLaneResult;
    let result: PennyLaneResult = serde_json::from_str(&result_json).expect("deserialize");

    // RX(π)|0⟩ = -i|1⟩, so ⟨Z⟩ ≈ -1
    assert_eq!(result.expval.len(), 1);
    assert!(
        (result.expval[0] + 1.0).abs() < 1e-9,
        "⟨Z⟩ after RX(π) = {} (expected -1)",
        result.expval[0]
    );
}

#[test]
fn test_json_error_unknown_gate() {
    let device = QuantRS2Device::new();
    let json = r#"{"num_wires":2,"operations":[{"name":"UnsupportedGate","wires":[0,1],"params":[]}],"observables":[]}"#;

    let result = device.execute_json(json);
    assert!(result.is_err(), "should fail for unknown gate");
}

#[test]
fn test_json_error_malformed() {
    let device = QuantRS2Device::new();
    let bad_json = r#"{"num_wires": "not_a_number"}"#;
    let result = device.execute_json(bad_json);
    assert!(result.is_err(), "should fail for malformed JSON");
}

// ─── multi-qubit circuit tests ────────────────────────────────────────────────

#[test]
fn test_toffoli_gate() {
    let circuit = PennyLaneCircuit {
        num_wires: 3,
        operations: vec![
            PennyLaneOperation {
                name: "PauliX".to_string(),
                wires: vec![0],
                params: vec![],
            },
            PennyLaneOperation {
                name: "PauliX".to_string(),
                wires: vec![1],
                params: vec![],
            },
            PennyLaneOperation {
                name: "Toffoli".to_string(),
                wires: vec![0, 1, 2],
                params: vec![],
            },
        ],
        observables: vec![make_pauliz_obs(2)],
    };

    let device = QuantRS2Device::new();
    let result = device.execute(&circuit).expect("toffoli gate");

    // X⊗X⊗I |000⟩ = |110⟩, then Toffoli(0,1,2)|110⟩ = |111⟩
    // ⟨Z⟩ on qubit 2 of |111⟩ = -1
    assert_eq!(result.expval.len(), 1);
    assert!(
        (result.expval[0] + 1.0).abs() < 1e-9,
        "⟨Z⟩ on target qubit after Toffoli = {} (expected -1)",
        result.expval[0]
    );
}

#[test]
fn test_swap_gate() {
    let circuit = PennyLaneCircuit {
        num_wires: 2,
        operations: vec![
            // Put qubit 0 in |1⟩ state
            PennyLaneOperation {
                name: "PauliX".to_string(),
                wires: vec![0],
                params: vec![],
            },
            // SWAP(0,1) should move the |1⟩ to qubit 1
            PennyLaneOperation {
                name: "SWAP".to_string(),
                wires: vec![0, 1],
                params: vec![],
            },
        ],
        observables: vec![
            make_pauliz_obs(0), // should be +1 (|0⟩ after swap)
            make_pauliz_obs(1), // should be -1 (|1⟩ after swap)
        ],
    };

    let device = QuantRS2Device::new();
    let result = device.execute(&circuit).expect("swap gate");

    assert_eq!(result.expval.len(), 2);
    assert!(
        (result.expval[0] - 1.0).abs() < 1e-9,
        "⟨Z⟩ on qubit 0 after SWAP = {} (expected +1)",
        result.expval[0]
    );
    assert!(
        (result.expval[1] + 1.0).abs() < 1e-9,
        "⟨Z⟩ on qubit 1 after SWAP = {} (expected -1)",
        result.expval[1]
    );
}

// ─── PauliX/Y expectation value tests ────────────────────────────────────────

#[test]
fn test_paulix_expval_plus_state() {
    // H|0⟩ = |+⟩, so ⟨X⟩ = 1 on qubit 0
    let circuit = PennyLaneCircuit {
        num_wires: 2,
        operations: vec![PennyLaneOperation {
            name: "Hadamard".to_string(),
            wires: vec![0],
            params: vec![],
        }],
        observables: vec![PennyLaneObservable {
            name: "PauliX".to_string(),
            wires: vec![0],
        }],
    };

    let device = QuantRS2Device::new();
    let result = device.execute(&circuit).expect("H|0⟩ PauliX expval");

    assert_eq!(result.expval.len(), 1);
    assert!(
        (result.expval[0] - 1.0).abs() < 1e-9,
        "⟨X⟩ for H|0⟩ = |+⟩ should be 1.0, got {}",
        result.expval[0]
    );
}

#[test]
fn test_paulix_expval_zero_state() {
    // |0⟩ state: ⟨X⟩ = 0 (equal superposition of |+⟩ and |-⟩)
    let circuit = PennyLaneCircuit {
        num_wires: 2,
        operations: vec![],
        observables: vec![PennyLaneObservable {
            name: "PauliX".to_string(),
            wires: vec![0],
        }],
    };

    let device = QuantRS2Device::new();
    let result = device.execute(&circuit).expect("|0⟩ PauliX expval");

    assert_eq!(result.expval.len(), 1);
    assert!(
        result.expval[0].abs() < 1e-9,
        "⟨X⟩ for |0⟩ should be 0.0, got {}",
        result.expval[0]
    );
}
