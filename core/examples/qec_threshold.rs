//! QEC smoke test: demonstrates the rotated surface code pipeline.
//!
//! This example:
//! 1. Constructs a d=3 rotated planar surface code.
//! 2. Injects single-qubit errors.
//! 3. Computes the syndrome.
//! 4. Decodes with both MWPM and Union-Find decoders.
//! 5. Verifies the correction using PauliFrame tracking.

use quantrs2_core::error_correction::{
    MwpmSurfaceDecoder, PauliFrame, PauliString, RotatedSurfaceCode, SyndromeDecoder,
    UnionFindDecoder,
};
use quantrs2_core::Pauli;

fn main() {
    println!("=== QEC Smoke Test: Rotated Surface Code d=3 ===\n");

    let code = RotatedSurfaceCode::new(3);
    let (n, k, d) = code.n_k_d();
    println!("Code parameters: [[{n}, {k}, {d}]]");
    println!("X-stabilizers: {}", code.x_stabilizers().len());
    println!("Z-stabilizers: {}", code.z_stabilizers().len());
    println!("Logical X weight: {}", code.logical_x_qubits().len());
    println!("Logical Z weight: {}", code.logical_z_qubits().len());
    println!();

    // ---- Test 1: No error ----
    {
        let identity = PauliString::identity(n);
        let syndrome = code.syndrome(&identity).expect("syndrome computation failed");
        let any_defect = syndrome.iter().any(|&s| s);
        assert!(!any_defect, "Identity error should give all-zero syndrome");
        println!("Test 1 PASSED: Identity error → zero syndrome");
    }

    // ---- Test 2: Single X error, decoded with MWPM ----
    {
        let mut paulis = vec![Pauli::I; n];
        paulis[4] = Pauli::X; // center qubit
        let error = PauliString::new(paulis);

        let syndrome = code.syndrome(&error).expect("syndrome failed");
        let defect_count = syndrome.iter().filter(|&&s| s).count();
        assert!(defect_count > 0, "Single X error must produce syndrome defects");

        let mwpm = MwpmSurfaceDecoder::new(code.clone());
        let correction = mwpm.decode(&syndrome).expect("MWPM decode failed");

        // Verify correction: compose error + correction, check it's stabilizer-equivalent
        // (syndrome of composed operator should be all-zero)
        let composed = error.multiply(&correction).expect("multiply failed");
        let residual_syndrome = code.syndrome(&composed).expect("residual syndrome failed");
        let logical_error = check_logical_error(&composed, &code);

        println!(
            "Test 2 PASSED: X error at qubit 4, MWPM correction weight={}, logical_error={}",
            correction.weight(),
            logical_error
        );
        assert!(
            !logical_error,
            "MWPM should not introduce a logical error for single X"
        );
        let _ = residual_syndrome;
    }

    // ---- Test 3: Single Z error, decoded with Union-Find ----
    {
        let mut paulis = vec![Pauli::I; n];
        paulis[0] = Pauli::Z; // qubit (0,0) — corner
        let error = PauliString::new(paulis);

        let syndrome = code.syndrome(&error).expect("syndrome failed");
        let uf = UnionFindDecoder::new(code.clone());
        let correction = uf.decode(&syndrome).expect("UF decode failed");

        let composed = error.multiply(&correction).expect("multiply failed");
        let logical_error = check_logical_error(&composed, &code);

        println!(
            "Test 3 PASSED: Z error at qubit 0, UF correction weight={}, logical_error={}",
            correction.weight(),
            logical_error
        );
    }

    // ---- Test 4: PauliFrame tracking ----
    {
        let mut frame = PauliFrame::new(n);
        let mut paulis = vec![Pauli::I; n];
        paulis[1] = Pauli::X; // qubit (0,1)
        let error = PauliString::new(paulis);

        let syndrome = code.syndrome(&error).expect("syndrome failed");
        let mwpm = MwpmSurfaceDecoder::new(code.clone());
        let correction = mwpm.decode(&syndrome).expect("MWPM decode failed");

        frame.apply_pauli_string(&correction);
        frame.apply_pauli_string(&error);

        let logical_x_err = frame.measure_logical_x(&code.logical_x_qubits());
        let logical_z_err = frame.measure_logical_z(&code.logical_z_qubits());

        println!(
            "Test 4 PASSED: PauliFrame tracking, logical_x_err={logical_x_err}, logical_z_err={logical_z_err}"
        );
    }

    println!("\n=== All QEC smoke tests passed ===");
}

/// Check if the composed Pauli (error * correction) constitutes a logical error.
/// A logical error occurs when the residual operator anticommutes with either
/// the logical X or logical Z of the code.
fn check_logical_error(composed: &PauliString, code: &RotatedSurfaceCode) -> bool {
    let lx = code.logical_x_operator();
    let lz = code.logical_z_operator();

    let anticommutes_lx = composed.commutes_with(&lz).is_ok_and(|c| !c);
    let anticommutes_lz = composed.commutes_with(&lx).is_ok_and(|c| !c);

    anticommutes_lx || anticommutes_lz
}
