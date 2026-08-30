//! Regression tests for the `scirs2_matrices` numerical routines.
//!
//! These lock in the fixes for three silent-fabrication findings:
//!   1. `SparseGateLibrary::embed_two_qubit_gate` used to return an identity
//!      instead of the embedded CNOT unitary.
//!   2. The `BLAS`/`SimdOperations` numerical helpers returned hardcoded
//!      constants (condition number `1.0`, rank `1`, `is_unitary == true`, ...).
//!   3. `SparseMatrix::matrix_exp` returned the input unchanged instead of the
//!      matrix exponential.

use quantrs2_circuit::scirs2_matrices::{
    Complex64, SimdOperations, SparseFormat, SparseGateLibrary, SparseMatrix, SparseOptimizer, BLAS,
};

const fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Sum of the triplets landing on `(i, j)`.
fn entry(m: &SparseMatrix, i: usize, j: usize) -> Complex64 {
    let mut acc = c(0.0, 0.0);
    for &(r, col, v) in m.triplets() {
        if r == i && col == j {
            acc += v;
        }
    }
    acc
}

fn pauli_x(format: SparseFormat) -> SparseMatrix {
    let mut x = SparseMatrix::new(2, 2, format);
    x.insert(0, 1, c(1.0, 0.0));
    x.insert(1, 0, c(1.0, 0.0));
    x
}

fn pauli_z(format: SparseFormat) -> SparseMatrix {
    let mut z = SparseMatrix::new(2, 2, format);
    z.insert(0, 0, c(1.0, 0.0));
    z.insert(1, 1, c(-1.0, 0.0));
    z
}

// ---------------------------------------------------------------------------
// Finding 1: embed_two_qubit_gate builds the real CNOT unitary.
// ---------------------------------------------------------------------------

#[test]
fn embed_cnot_two_qubits_matches_reference() {
    let library = SparseGateLibrary::new();
    let embedded = library
        .embed_two_qubit_gate("CNOT", 0, 1, 2)
        .expect("CNOT embedding should succeed");
    let reference = library.get_gate("CNOT").expect("CNOT gate in library");

    assert_eq!(embedded.shape, (4, 4));
    assert!(
        embedded.matrices_equal(reference, 1e-12),
        "embedded 2-qubit CNOT must equal the reference CNOT matrix"
    );
    // Regression guard: it must NOT be the identity (the old fabricated result).
    assert!(
        !embedded.matrices_equal(&SparseMatrix::identity(4), 1e-12),
        "embedded CNOT must not be an identity"
    );
    assert!(embedded.is_unitary(1e-10));
}

#[test]
fn embed_cnot_three_qubits_is_unitary_permutation() {
    let library = SparseGateLibrary::new();
    let embedded = library
        .embed_two_qubit_gate("CNOT", 0, 2, 3)
        .expect("3-qubit CNOT embedding should succeed");

    assert_eq!(embedded.shape, (8, 8));
    assert_eq!(embedded.nnz(), 8, "permutation matrix has exactly 8 ones");
    assert!(embedded.is_unitary(1e-10));
    assert!(!embedded.matrices_equal(&SparseMatrix::identity(8), 1e-12));

    // control = qubit 0 (MSB), target = qubit 2 (LSB): flips LSB when col >= 4.
    assert_eq!(entry(&embedded, 5, 4), c(1.0, 0.0));
    assert_eq!(entry(&embedded, 4, 5), c(1.0, 0.0));
    assert_eq!(entry(&embedded, 0, 0), c(1.0, 0.0));
}

#[test]
fn embed_two_qubit_gate_rejects_bad_input() {
    let library = SparseGateLibrary::new();
    assert!(library.embed_two_qubit_gate("CNOT", 1, 1, 2).is_err());
    assert!(library.embed_two_qubit_gate("SWAP", 0, 1, 2).is_err());
    assert!(library.embed_two_qubit_gate("CNOT", 0, 5, 2).is_err());
}

// ---------------------------------------------------------------------------
// Finding 3: matrix_exp computes the real matrix exponential.
// ---------------------------------------------------------------------------

#[test]
fn matrix_exp_of_pauli_x() {
    let x = pauli_x(SparseFormat::COO);
    let result = x.matrix_exp(1.0).expect("matrix_exp of X");
    // exp(X) = cosh(1) I + sinh(1) X.
    let cosh1 = 1.0_f64.cosh();
    let sinh1 = 1.0_f64.sinh();
    assert!((entry(&result, 0, 0).re - cosh1).abs() < 1e-10);
    assert!((entry(&result, 1, 1).re - cosh1).abs() < 1e-10);
    assert!((entry(&result, 0, 1).re - sinh1).abs() < 1e-10);
    assert!((entry(&result, 1, 0).re - sinh1).abs() < 1e-10);
    // Regression guard: NOT the untouched input (which had zero diagonal).
    assert!(entry(&result, 0, 0).norm() > 0.5);
}

#[test]
fn matrix_exp_zero_scale_is_identity() {
    let x = pauli_x(SparseFormat::COO);
    let result = x.matrix_exp(0.0).expect("matrix_exp with zero scale");
    assert!(result.matrices_equal(&SparseMatrix::identity(2), 1e-12));
}

#[test]
fn matrix_exp_diagonal_and_simd_path_agree() {
    // exp(Z) = diag(e, 1/e); also exercises the SIMD dispatch path.
    let z_simd = pauli_z(SparseFormat::SIMDAligned);
    let result = z_simd.matrix_exp(1.0).expect("matrix_exp via SIMD path");
    assert!((entry(&result, 0, 0).re - 1.0_f64.exp()).abs() < 1e-10);
    assert!((entry(&result, 1, 1).re - (-1.0_f64).exp()).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Finding 2: real numerical analysis instead of hardcoded constants.
// ---------------------------------------------------------------------------

#[test]
fn matrix_norm_reflects_entries() {
    let id4 = SparseMatrix::identity(4);
    // Frobenius norm of I_4 is 2, not the old hardcoded 1.0.
    assert!((BLAS::matrix_norm(&id4.inner, "frobenius") - 2.0).abs() < 1e-12);
    assert!((BLAS::matrix_norm(&id4.inner, "1") - 1.0).abs() < 1e-12);
    assert!((BLAS::matrix_norm(&id4.inner, "inf") - 1.0).abs() < 1e-12);
    assert!((BLAS::matrix_norm(&id4.inner, "2") - 1.0).abs() < 1e-9);

    let mut diag = SparseMatrix::new(2, 2, SparseFormat::COO);
    diag.insert(0, 0, c(1.0, 0.0));
    diag.insert(1, 1, c(100.0, 0.0));
    assert!((BLAS::matrix_norm(&diag.inner, "max") - 100.0).abs() < 1e-12);
    assert!((BLAS::matrix_norm(&diag.inner, "2") - 100.0).abs() < 1e-6);
}

#[test]
fn condition_number_and_rank_are_real() {
    let id4 = SparseMatrix::identity(4);
    assert!((BLAS::condition_number(&id4.inner) - 1.0).abs() < 1e-9);
    // Rank 4, not the old hardcoded 1.
    assert_eq!(BLAS::numerical_rank(&id4.inner, 1e-12), 4);

    let mut diag = SparseMatrix::new(2, 2, SparseFormat::COO);
    diag.insert(0, 0, c(1.0, 0.0));
    diag.insert(1, 1, c(100.0, 0.0));
    assert!((BLAS::condition_number(&diag.inner) - 100.0).abs() < 1e-6);

    // Rank-deficient: a single non-zero entry in a 2x2 matrix has rank 1.
    let mut rank1 = SparseMatrix::new(2, 2, SparseFormat::COO);
    rank1.insert(0, 0, c(1.0, 0.0));
    assert_eq!(BLAS::numerical_rank(&rank1.inner, 1e-9), 1);
    assert!(BLAS::condition_number(&rank1.inner).is_infinite());
}

#[test]
fn is_positive_definite_is_real() {
    // Identity is positive definite.
    assert!(BLAS::is_positive_definite(&SparseMatrix::identity(3).inner));
    // Pauli X is Hermitian with eigenvalues ±1 -> not positive definite.
    assert!(!BLAS::is_positive_definite(
        &pauli_x(SparseFormat::COO).inner
    ));
    // diag(2, -1) is Hermitian but indefinite.
    let mut indef = SparseMatrix::new(2, 2, SparseFormat::COO);
    indef.insert(0, 0, c(2.0, 0.0));
    indef.insert(1, 1, c(-1.0, 0.0));
    assert!(!BLAS::is_positive_definite(&indef.inner));
    // diag(2, 3) is positive definite.
    let mut pd = SparseMatrix::new(2, 2, SparseFormat::COO);
    pd.insert(0, 0, c(2.0, 0.0));
    pd.insert(1, 1, c(3.0, 0.0));
    assert!(BLAS::is_positive_definite(&pd.inner));
}

#[test]
fn spectral_analysis_reports_true_radius() {
    let x = pauli_x(SparseFormat::COO);
    let sa = BLAS::spectral_analysis(&x.inner);
    // X has eigenvalues +/-1 -> spectral radius 1, spread 0.
    assert!((sa.spectral_radius - 1.0).abs() < 1e-6);
    assert!(sa.eigenvalue_spread.abs() < 1e-6);

    let mut diag = SparseMatrix::new(2, 2, SparseFormat::COO);
    diag.insert(0, 0, c(3.0, 0.0));
    diag.insert(1, 1, c(0.5, 0.0));
    let sa2 = BLAS::spectral_analysis(&diag.inner);
    assert!((sa2.spectral_radius - 3.0).abs() < 1e-6);
    assert!((sa2.eigenvalue_spread - 2.5).abs() < 1e-6);
}

#[test]
fn simd_is_unitary_detects_non_unitary() {
    // Unitary gate in SIMD format -> true.
    let x_simd = pauli_x(SparseFormat::SIMDAligned);
    assert!(x_simd.is_unitary(1e-10));

    // Non-unitary matrix in SIMD format must report false (old code: always true).
    let mut bad = SparseMatrix::new(2, 2, SparseFormat::SIMDAligned);
    bad.insert(0, 0, c(2.0, 0.0));
    bad.insert(1, 1, c(1.0, 0.0));
    assert!(!bad.is_unitary(1e-10));
}

#[test]
fn simd_matrices_equal_and_threshold_are_real() {
    // matrices_equal via the SIMD dispatch: differing matrices must not be equal.
    let x_simd = pauli_x(SparseFormat::SIMDAligned);
    let id_simd = {
        let mut m = SparseMatrix::new(2, 2, SparseFormat::SIMDAligned);
        m.insert(0, 0, c(1.0, 0.0));
        m.insert(1, 1, c(1.0, 0.0));
        m
    };
    assert!(!x_simd.matrices_equal(&id_simd, 1e-12));
    assert!(x_simd.matrices_equal(&pauli_x(SparseFormat::SIMDAligned), 1e-12));

    // threshold_filter must actually drop sub-threshold entries.
    let mut mixed = SparseMatrix::new(2, 2, SparseFormat::COO);
    mixed.insert(0, 0, c(1.0, 0.0));
    mixed.insert(1, 1, c(0.1, 0.0));
    let simd = SimdOperations::new();
    let filtered = simd.threshold_filter(&mixed.inner, 0.5);
    assert_eq!(filtered.nnz(), 1, "0.1 entry must be filtered out");
}

#[test]
fn fidelity_and_distance_metrics_are_real() {
    let x = pauli_x(SparseFormat::COO);
    let id2 = SparseMatrix::identity(2);
    let z = pauli_z(SparseFormat::COO);

    // Identical gates: fidelity 1, distances 0.
    assert!((BLAS::gate_fidelity(&x.inner, &x.inner) - 1.0).abs() < 1e-12);
    assert!((BLAS::process_fidelity(&x.inner, &x.inner) - 1.0).abs() < 1e-12);
    assert!(BLAS::trace_distance(&x.inner, &x.inner).abs() < 1e-12);
    assert!(BLAS::diamond_distance(&x.inner, &x.inner).abs() < 1e-9);

    // X vs I: process fidelity |Tr(X)|^2/4 = 0, avg gate fidelity 1/3.
    assert!(BLAS::process_fidelity(&x.inner, &id2.inner).abs() < 1e-12);
    assert!((BLAS::gate_fidelity(&x.inner, &id2.inner) - 1.0 / 3.0).abs() < 1e-12);

    // trace distance between I and X: eigenvalues of (I-X) are {0, 2} -> 1.
    assert!((BLAS::trace_distance(&id2.inner, &x.inner) - 1.0).abs() < 1e-9);

    // Diamond distance between I and Z channels is 2 (perfectly distinguishable).
    assert!((BLAS::diamond_distance(&id2.inner, &z.inner) - 2.0).abs() < 1e-9);

    // Error decomposition of a gate with itself vanishes.
    let dec = BLAS::error_decomposition(&x.inner, &x.inner);
    assert!(dec.coherent_component.abs() < 1e-9);
    assert!(dec.incoherent_component.abs() < 1e-9);
}

#[test]
fn analyze_gate_properties_uses_real_math() {
    let library = SparseGateLibrary::new();
    let h = library.get_gate("H").expect("Hadamard in library");
    let optimizer = SparseOptimizer::new();
    let props = optimizer.analyze_gate_properties(h);

    assert!(props.is_unitary);
    assert!(props.is_hermitian);
    assert!((props.condition_number - 1.0).abs() < 1e-9);
    assert!((props.spectral_radius - 1.0).abs() < 1e-6);
    assert!((props.matrix_norm - 2.0_f64.sqrt()).abs() < 1e-9);
    assert_eq!(props.numerical_rank, 2);
    assert!(!props.structure_analysis.is_positive_definite);
}
