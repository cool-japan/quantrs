//! Integration tests for the QEC surface code pipeline.

use quantrs2_core::error_correction::{
    MwpmSurfaceDecoder, PauliFrame, PauliString, RotatedSurfaceCode, SyndromeDecoder,
    UnionFindDecoder,
};
use quantrs2_core::Pauli;

/// Check whether the composed Pauli (error * correction) is a logical error.
/// A logical error anticommutes with at least one of the logical operators.
fn is_logical_error(composed: &PauliString, code: &RotatedSurfaceCode) -> bool {
    let lx = code.logical_x_operator();
    let lz = code.logical_z_operator();
    let anticommutes_lx = composed.commutes_with(&lz).is_ok_and(|c| !c);
    let anticommutes_lz = composed.commutes_with(&lx).is_ok_and(|c| !c);
    anticommutes_lx || anticommutes_lz
}

#[test]
fn test_qec_smoke_d3_zero_errors() {
    let code = RotatedSurfaceCode::new(3);
    let n = code.n_data_qubits();

    let identity = PauliString::identity(n);
    let syndrome = code.syndrome(&identity).expect("syndrome failed");
    assert!(syndrome.iter().all(|&s| !s), "Zero errors → zero syndrome");

    let mwpm = MwpmSurfaceDecoder::new(code.clone());
    let correction = mwpm.decode(&syndrome).expect("decode failed");
    assert_eq!(correction.weight(), 0, "Zero errors → identity correction");

    let uf = UnionFindDecoder::new(code);
    let uf_correction = uf.decode(&syndrome).expect("uf decode failed");
    assert_eq!(
        uf_correction.weight(),
        0,
        "UF: Zero errors → identity correction"
    );
}

#[test]
fn test_qec_smoke_d3_single_x_error() {
    let code = RotatedSurfaceCode::new(3);
    let n = code.n_data_qubits();

    // Try X errors on each data qubit
    for qubit in 0..n {
        let mut paulis = vec![Pauli::I; n];
        paulis[qubit] = Pauli::X;
        let error = PauliString::new(paulis);

        let syndrome = code.syndrome(&error).expect("syndrome failed");
        let defect_count = syndrome.iter().filter(|&&s| s).count();
        assert!(
            defect_count > 0,
            "X error on qubit {qubit} must produce defects"
        );

        let mwpm = MwpmSurfaceDecoder::new(code.clone());
        let correction = mwpm.decode(&syndrome).expect("decode failed");
        let composed = error.multiply(&correction).expect("multiply failed");
        assert!(
            !is_logical_error(&composed, &code),
            "MWPM: X error on qubit {qubit} should not cause logical error"
        );
    }
}

#[test]
fn test_qec_smoke_d3_single_z_error() {
    let code = RotatedSurfaceCode::new(3);
    let n = code.n_data_qubits();

    for qubit in 0..n {
        let mut paulis = vec![Pauli::I; n];
        paulis[qubit] = Pauli::Z;
        let error = PauliString::new(paulis);

        let syndrome = code.syndrome(&error).expect("syndrome failed");
        let defect_count = syndrome.iter().filter(|&&s| s).count();
        assert!(
            defect_count > 0,
            "Z error on qubit {qubit} must produce defects"
        );

        let mwpm = MwpmSurfaceDecoder::new(code.clone());
        let correction = mwpm.decode(&syndrome).expect("decode failed");
        let composed = error.multiply(&correction).expect("multiply failed");
        assert!(
            !is_logical_error(&composed, &code),
            "MWPM: Z error on qubit {qubit} should not cause logical error"
        );
    }
}

#[test]
fn test_qec_uf_smoke_d3_single_error() {
    let code = RotatedSurfaceCode::new(3);
    let n = code.n_data_qubits();

    for qubit in [0, 4, 8] {
        // corner, center, corner
        let mut paulis = vec![Pauli::I; n];
        paulis[qubit] = Pauli::X;
        let error = PauliString::new(paulis);

        let syndrome = code.syndrome(&error).expect("syndrome failed");

        let uf = UnionFindDecoder::new(code.clone());
        let correction = uf.decode(&syndrome).expect("UF decode failed");
        let composed = error.multiply(&correction).expect("multiply failed");
        assert!(
            !is_logical_error(&composed, &code),
            "UF: X error on qubit {qubit} should not cause logical error"
        );
    }
}

#[test]
fn test_qec_pauli_frame_no_errors() {
    let code = RotatedSurfaceCode::new(3);
    let n = code.n_data_qubits();

    let identity = PauliString::identity(n);
    let syndrome = code.syndrome(&identity).expect("syndrome failed");

    let mwpm = MwpmSurfaceDecoder::new(code.clone());
    let correction = mwpm.decode(&syndrome).expect("decode failed");

    let mut frame = PauliFrame::new(n);
    frame.apply_pauli_string(&correction);

    assert!(
        frame.is_identity(),
        "No-error case: PauliFrame should remain identity"
    );
    assert!(
        !frame.measure_logical_x(&code.logical_x_qubits()),
        "No logical X error"
    );
    assert!(
        !frame.measure_logical_z(&code.logical_z_qubits()),
        "No logical Z error"
    );
}
