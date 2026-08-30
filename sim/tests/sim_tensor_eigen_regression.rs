#![allow(clippy::pedantic)]
//! Regression tests for the `sim-tensor-eigen` bundle.
//!
//! These lock in the fixes that replaced fabricated/placeholder returns with
//! real computation in:
//!   * `tensor_network::mod` — `TensorNetwork::contract_to_statevector` now
//!     performs a genuine sequential tensor contraction of the applied gate
//!     tensors instead of returning a canned circuit-type state.
//!   * `tensor_network::contraction` — `contract_network_along_path` now
//!     executes the contraction path with real pairwise tensor contraction.
//!   * `scirs2_sparse` — the Arnoldi/Lanczos eigensolvers return genuine Ritz
//!     values and (non-zero) Ritz vectors.
//!   * `scirs2_eigensolvers` — the entanglement spectrum uses the full Hermitian
//!     eigendecomposition (not a single power-iteration eigenvalue).

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_sim::scirs2_eigensolvers::{SciRS2SpectralAnalyzer, SpectralConfig};
use quantrs2_sim::scirs2_sparse::{
    SciRS2SparseSolver, SparseMatrix, SparseSolverConfig, SparseSolverMethod,
};
use quantrs2_sim::statevector::StateVectorSimulator;
use quantrs2_sim::tensor_network::contraction::{contract_network_along_path, ContractionPath};
use quantrs2_sim::tensor_network::tensor::{Tensor, TensorIndex};
use quantrs2_sim::tensor_network::TensorNetworkSimulator;
use scirs2_core::ndarray::{Array, Array1, IxDyn};
use scirs2_core::Complex64;
use std::collections::HashMap;

const TOL: f64 = 1e-9;

const fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

// ============================================================
// Tensor-network simulator: real contraction, not canned states
// ============================================================

/// The tensor-network simulator must agree with the (independently trusted)
/// state-vector simulator amplitude-by-amplitude. The old code returned a
/// hardcoded state chosen from a circuit-type heuristic, which ignored gate
/// parameters entirely.
fn assert_tn_matches_statevector<const N: usize>(circuit: &Circuit<N>) {
    let sv = StateVectorSimulator::new();
    let tn = TensorNetworkSimulator::new();

    let sv_reg = sv.run(circuit).expect("state-vector simulation");
    let tn_reg = tn.run(circuit).expect("tensor-network simulation");

    let sv_amps = sv_reg.amplitudes();
    let tn_amps = tn_reg.amplitudes();

    assert_eq!(
        sv_amps.len(),
        tn_amps.len(),
        "amplitude vector length mismatch"
    );
    for (i, (a, b)) in sv_amps.iter().zip(tn_amps.iter()).enumerate() {
        assert!(
            (a - b).norm() < 1e-8,
            "amplitude {i} mismatch: statevector {a:?}, tensornetwork {b:?}"
        );
    }
}

#[test]
fn test_tensor_network_bell_state() {
    let mut circuit = Circuit::<2>::new();
    circuit.h(0).expect("h");
    circuit.cnot(0, 1).expect("cnot");

    let tn = TensorNetworkSimulator::new();
    let reg = tn.run(&circuit).expect("tensor-network simulation");
    let amps = reg.amplitudes();

    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let expected = [
        c(inv_sqrt2, 0.0),
        c(0.0, 0.0),
        c(0.0, 0.0),
        c(inv_sqrt2, 0.0),
    ];
    for (i, (got, want)) in amps.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).norm() < TOL,
            "Bell amplitude {i}: got {got:?}, want {want:?}"
        );
    }
}

#[test]
fn test_tensor_network_matches_statevector_ghz() {
    let mut circuit = Circuit::<3>::new();
    circuit.h(0).expect("h");
    circuit.cnot(0, 1).expect("cnot01");
    circuit.cnot(1, 2).expect("cnot12");
    assert_tn_matches_statevector(&circuit);
}

/// The decisive test: a circuit whose result depends on the actual rotation
/// angles. The old fabrication ignored angles, so this would have failed.
#[test]
fn test_tensor_network_matches_statevector_parametric() {
    let mut circuit = Circuit::<3>::new();
    circuit.h(0).expect("h");
    circuit.ry(1, 0.7).expect("ry");
    circuit.cnot(0, 1).expect("cnot01");
    circuit.rz(0, 0.3).expect("rz");
    circuit.cnot(1, 2).expect("cnot12");
    circuit.rx(2, 1.1).expect("rx");
    circuit.rz(2, -0.42).expect("rz2");
    assert_tn_matches_statevector(&circuit);
}

/// Non-adjacent control/target must be handled correctly by the contraction.
#[test]
fn test_tensor_network_matches_statevector_nonadjacent_cnot() {
    let mut circuit = Circuit::<3>::new();
    circuit.h(0).expect("h");
    circuit.cnot(0, 2).expect("cnot02");
    circuit.ry(1, 0.9).expect("ry");
    assert_tn_matches_statevector(&circuit);
}

#[test]
fn test_tensor_network_output_normalized() {
    let mut circuit = Circuit::<4>::new();
    circuit.h(0).expect("h");
    circuit.ry(1, 0.55).expect("ry");
    circuit.cnot(0, 1).expect("cnot01");
    circuit.cnot(1, 2).expect("cnot12");
    circuit.rx(3, 0.8).expect("rx");
    circuit.cnot(2, 3).expect("cnot23");

    let tn = TensorNetworkSimulator::new();
    let reg = tn.run(&circuit).expect("tensor-network simulation");
    let norm_sq: f64 = reg.amplitudes().iter().map(|a| a.norm_sqr()).sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-9,
        "tensor-network output not normalized: |ψ|² = {norm_sq}"
    );
}

// ============================================================
// contract_network_along_path: real contraction path execution
// ============================================================

/// Executing a single-step path over a 2-tensor network bonded along one index
/// must perform genuine matrix multiplication, not return an existing tensor.
#[test]
fn test_contract_network_along_path_matrix_multiply() {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]; contract A's columns with B's rows.
    // Expected A·B = [[19,22],[43,50]].
    let a = Tensor::new(
        Array::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)],
        )
        .expect("tensor a"),
    );
    let b = Tensor::new(
        Array::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![c(5.0, 0.0), c(6.0, 0.0), c(7.0, 0.0), c(8.0, 0.0)],
        )
        .expect("tensor b"),
    );

    let mut tensors: HashMap<usize, Tensor> = HashMap::new();
    tensors.insert(0, a);
    tensors.insert(1, b);

    // Bond: axis 1 of tensor 0 to axis 0 of tensor 1.
    let mut connections = vec![(
        TensorIndex {
            tensor_id: 0,
            index: 1,
        },
        TensorIndex {
            tensor_id: 1,
            index: 0,
        },
    )];

    let path = ContractionPath::new(vec![(0, 1)], 8.0);
    let mut next_id = 2usize;

    let result = contract_network_along_path(&mut tensors, &mut connections, &path, &mut next_id)
        .expect("path contraction");

    assert_eq!(result.dimensions, vec![2, 2], "result should be 2x2");
    let expected = [[19.0, 22.0], [43.0, 50.0]];
    for (i, row) in expected.iter().enumerate() {
        for (j, expected_val) in row.iter().enumerate() {
            let got = result.data[IxDyn(&[i, j])];
            assert!(
                (got - c(*expected_val, 0.0)).norm() < TOL,
                "product[{i},{j}]: got {got:?}, want {expected_val}"
            );
        }
    }
}

/// An empty network is an honest error, not a fabricated qubit-zero tensor.
#[test]
fn test_contract_network_along_path_empty_errors() {
    let mut tensors: HashMap<usize, Tensor> = HashMap::new();
    let mut connections: Vec<(TensorIndex, TensorIndex)> = Vec::new();
    let path = ContractionPath::new(vec![], 0.0);
    let mut next_id = 0usize;
    let result = contract_network_along_path(&mut tensors, &mut connections, &path, &mut next_id);
    assert!(result.is_err(), "empty network should error");
}

// ============================================================
// Sparse Lanczos eigensolver: real Ritz values and vectors
// ============================================================

/// Build the 4x4 1-D Laplacian tridiag(-1, 2, -1) as a CSR `SparseMatrix`.
fn laplacian_4x4() -> SparseMatrix {
    // Row-compressed storage.
    let row_ptr = vec![0, 2, 5, 8, 10];
    let col_indices = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let values: Vec<Complex64> = vec![
        c(2.0, 0.0),
        c(-1.0, 0.0),
        c(-1.0, 0.0),
        c(2.0, 0.0),
        c(-1.0, 0.0),
        c(-1.0, 0.0),
        c(2.0, 0.0),
        c(-1.0, 0.0),
        c(-1.0, 0.0),
        c(2.0, 0.0),
    ];
    let mut m = SparseMatrix::from_csr((4, 4), row_ptr, col_indices, values);
    m.is_hermitian = true;
    m.is_positive_definite = true;
    m
}

#[test]
fn test_sparse_lanczos_returns_real_eigenpairs() {
    let matrix = laplacian_4x4();

    let config = SparseSolverConfig {
        method: SparseSolverMethod::Lanczos,
        ..SparseSolverConfig::default()
    };
    let mut solver = SciRS2SparseSolver::new(config).expect("solver");
    let result = solver
        .solve_eigenvalue_problem(&matrix, 2, "smallest")
        .expect("eigenvalue problem");

    // Analytic smallest eigenvalues: 2 - 2cos(kπ/5) for k = 1, 2.
    let lambda1 = 2.0f64.mul_add(-(std::f64::consts::PI / 5.0).cos(), 2.0);
    let lambda2 = 2.0f64.mul_add(-(2.0 * std::f64::consts::PI / 5.0).cos(), 2.0);

    assert_eq!(result.eigenvalues.len(), 2, "should return two eigenvalues");
    assert!(
        (result.eigenvalues[0] - lambda1).abs() < 1e-6,
        "smallest eigenvalue: got {}, want {lambda1}",
        result.eigenvalues[0]
    );
    assert!(
        (result.eigenvalues[1] - lambda2).abs() < 1e-6,
        "second eigenvalue: got {}, want {lambda2}",
        result.eigenvalues[1]
    );

    // Eigenvectors must be genuine (non-zero) and satisfy A v = λ v.
    assert_eq!(result.eigenvectors.shape(), &[4, 2]);
    for (j, &lambda) in result.eigenvalues.iter().enumerate() {
        let v = result.eigenvectors.column(j).to_owned();
        let vnorm = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            vnorm > 0.5,
            "eigenvector {j} is (near) zero — fabricated placeholder"
        );

        let av = matrix.matvec(&v).expect("matvec");
        let mut residual = 0.0_f64;
        for i in 0..4 {
            residual += (av[i] - Complex64::new(lambda, 0.0) * v[i]).norm_sqr();
        }
        assert!(
            residual.sqrt() < 1e-6,
            "‖A v - λ v‖ too large for eigenpair {j}: {}",
            residual.sqrt()
        );
    }
}

#[test]
fn test_sparse_arnoldi_non_hermitian_real_spectrum() {
    // Non-symmetric 4x4 with real eigenvalues {1, 2, 5, 7}:
    //   [[0, 1, 0, 0],
    //    [-2, 3, 0, 0],   (2x2 block eigenvalues 1, 2)
    //    [0, 0, 5, 0],
    //    [0, 0, 0, 7]]
    let row_ptr = vec![0, 1, 3, 4, 5];
    let col_indices = vec![1, 0, 1, 2, 3];
    let values: Vec<Complex64> = vec![
        c(1.0, 0.0),
        c(-2.0, 0.0),
        c(3.0, 0.0),
        c(5.0, 0.0),
        c(7.0, 0.0),
    ];
    let matrix = SparseMatrix::from_csr((4, 4), row_ptr, col_indices, values);
    // is_hermitian stays false -> Arnoldi path.

    let config = SparseSolverConfig {
        method: SparseSolverMethod::Arnoldi,
        ..SparseSolverConfig::default()
    };
    let mut solver = SciRS2SparseSolver::new(config).expect("solver");
    let result = solver
        .solve_eigenvalue_problem(&matrix, 2, "smallest")
        .expect("arnoldi eigenvalue problem");

    assert_eq!(result.eigenvalues.len(), 2);
    // Smallest two eigenvalues are 1 and 2.
    assert!(
        (result.eigenvalues[0] - 1.0).abs() < 1e-4,
        "smallest eigenvalue: got {}, want 1.0",
        result.eigenvalues[0]
    );
    assert!(
        (result.eigenvalues[1] - 2.0).abs() < 1e-4,
        "second eigenvalue: got {}, want 2.0",
        result.eigenvalues[1]
    );

    // Genuine, non-zero eigenvectors satisfying A v = λ v.
    for (j, &lambda) in result.eigenvalues.iter().enumerate() {
        let v = result.eigenvectors.column(j).to_owned();
        let vnorm = v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        assert!(vnorm > 0.5, "eigenvector {j} is (near) zero");
        let av = matrix.matvec(&v).expect("matvec");
        let mut residual = 0.0_f64;
        for i in 0..4 {
            residual += (av[i] - Complex64::new(lambda, 0.0) * v[i]).norm_sqr();
        }
        assert!(
            residual.sqrt() < 1e-4,
            "‖A v - λ v‖ too large for Arnoldi eigenpair {j}: {}",
            residual.sqrt()
        );
    }
}

// ============================================================
// Entanglement spectrum: full Hermitian spectrum, not one eigenvalue
// ============================================================

/// Build an `n`-qubit GHZ state (|0…0⟩ + |1…1⟩)/√2 as an amplitude vector.
fn ghz_state(n: usize) -> Array1<Complex64> {
    let dim = 1usize << n;
    let mut amps = vec![c(0.0, 0.0); dim];
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    amps[0] = c(inv_sqrt2, 0.0);
    amps[dim - 1] = c(inv_sqrt2, 0.0);
    Array1::from_vec(amps)
}

/// For a GHZ state the reduced density matrix of any nontrivial bipartition has
/// spectrum {1/2, 1/2}, so the von-Neumann entropy is ln 2. This exercises the
/// 4x4 (subsystem_dim = 4) reduced-density-matrix case that the old
/// power-iteration code got wrong (it returned only a single eigenvalue).
#[test]
fn test_entanglement_spectrum_ghz_two_qubit_cut() {
    let mut analyzer = SciRS2SpectralAnalyzer::new(SpectralConfig::default()).expect("analyzer");
    let state = ghz_state(3);
    let result = analyzer
        .calculate_entanglement_spectrum(&state, &[0, 1])
        .expect("entanglement spectrum");

    assert_eq!(
        result.eigenvalues.len(),
        2,
        "GHZ 2-qubit cut must yield two non-zero Schmidt values, got {:?}",
        result.eigenvalues
    );
    for &ev in &result.eigenvalues {
        assert!(
            (ev - 0.5).abs() < 1e-9,
            "Schmidt value should be 0.5, got {ev}"
        );
    }
    assert!(
        (result.entropy - std::f64::consts::LN_2).abs() < 1e-9,
        "entanglement entropy should be ln 2, got {}",
        result.entropy
    );
}

/// Strong test: a 3-qubit state whose 2-qubit reduced density matrix is
/// genuinely NON-diagonal with UNEQUAL Schmidt weights {0.7, 0.3}. A GHZ state's
/// reduced density matrix is diagonal, so it does not exercise the off-diagonal
/// diagonalization; this state does. The old single-eigenvalue power iteration
/// could not produce the correct two-value spectrum, and scirs2's complex_eigh
/// gets non-diagonal matrices wrong — this locks in the Jacobi-based fix.
///
/// State (in the crate's big-endian encoding, dim 8):
///   ampl[0] = ampl[6] = sqrt(0.35),  ampl[1] = sqrt(0.15),  ampl[7] = -sqrt(0.15)
/// which is √0.7·|u0⟩|0⟩ + √0.3·|u1⟩|1⟩ with |u0⟩=(|00⟩+|11⟩)/√2,
/// |u1⟩=(|00⟩-|11⟩)/√2 on the {0,1} subsystem — Schmidt spectrum {0.7, 0.3}.
#[test]
fn test_entanglement_spectrum_nondiagonal_unequal_weights() {
    let mut analyzer = SciRS2SpectralAnalyzer::new(SpectralConfig::default()).expect("analyzer");

    let p0 = 0.7_f64;
    let p1 = 0.3_f64;
    let mut amps = vec![c(0.0, 0.0); 8];
    amps[0] = c((p0 / 2.0).sqrt(), 0.0);
    amps[6] = c((p0 / 2.0).sqrt(), 0.0);
    amps[1] = c((p1 / 2.0).sqrt(), 0.0);
    amps[7] = c(-(p1 / 2.0).sqrt(), 0.0);
    let state = Array1::from_vec(amps);

    let result = analyzer
        .calculate_entanglement_spectrum(&state, &[0, 1])
        .expect("entanglement spectrum");

    let mut evs = result.eigenvalues.clone();
    evs.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert_eq!(
        evs.len(),
        2,
        "expected two non-zero Schmidt values, got {evs:?}"
    );
    assert!(
        (evs[0] - 0.7).abs() < 1e-6 && (evs[1] - 0.3).abs() < 1e-6,
        "Schmidt spectrum should be {{0.7, 0.3}}, got {evs:?}"
    );

    let expected_entropy = -(0.7f64.mul_add(0.7_f64.ln(), 0.3 * 0.3_f64.ln()));
    assert!(
        (result.entropy - expected_entropy).abs() < 1e-6,
        "entropy should be {expected_entropy}, got {}",
        result.entropy
    );
}

/// The 8x8 (subsystem_dim = 8) case — a 3-qubit cut of a 4-qubit GHZ state.
#[test]
fn test_entanglement_spectrum_ghz_three_qubit_cut() {
    let mut analyzer = SciRS2SpectralAnalyzer::new(SpectralConfig::default()).expect("analyzer");
    let state = ghz_state(4);
    let result = analyzer
        .calculate_entanglement_spectrum(&state, &[0, 1, 2])
        .expect("entanglement spectrum");

    assert_eq!(
        result.eigenvalues.len(),
        2,
        "GHZ 3-qubit cut must yield two non-zero Schmidt values, got {:?}",
        result.eigenvalues
    );
    assert!(
        (result.entropy - std::f64::consts::LN_2).abs() < 1e-9,
        "entanglement entropy should be ln 2, got {}",
        result.entropy
    );
}
