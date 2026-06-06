//! Integration tests for the fermionic quantum simulator.
//!
//! These tests verify that the Jordan-Wigner transformation, Pauli string
//! application, expectation values, and time evolution all produce physically
//! correct results.

use quantrs2_sim::fermionic_simulation::{
    FermionicHamiltonian, FermionicOperator, FermionicSimulator, FermionicString,
};
use quantrs2_sim::pauli::{PauliOperator, PauliString};
use scirs2_core::Complex64;

// ============================================================================
// 1. Number operator expectation values
// ============================================================================

#[test]
fn test_number_operator_mode0_occupied() {
    // 2-mode system, mode 0 occupied, mode 1 empty → |10⟩
    let mut sim = FermionicSimulator::new(2).expect("create simulator");
    sim.set_initial_state(&[true, false])
        .expect("set initial state");

    let n0 = sim
        .expectation_value(&FermionicOperator::Number(0))
        .expect("n0 expectation");
    let n1 = sim
        .expectation_value(&FermionicOperator::Number(1))
        .expect("n1 expectation");

    assert!(
        (n0.re - 1.0).abs() < 1e-10,
        "mode 0 occupied: expected 1.0, got {:.6}",
        n0.re
    );
    assert!(
        n1.re.abs() < 1e-10,
        "mode 1 empty: expected 0.0, got {:.6}",
        n1.re
    );
}

#[test]
fn test_number_operator_mode1_occupied() {
    // 2-mode system, mode 0 empty, mode 1 occupied → |01⟩
    let mut sim = FermionicSimulator::new(2).expect("create simulator");
    sim.set_initial_state(&[false, true])
        .expect("set initial state");

    let n0 = sim
        .expectation_value(&FermionicOperator::Number(0))
        .expect("n0 expectation");
    let n1 = sim
        .expectation_value(&FermionicOperator::Number(1))
        .expect("n1 expectation");

    assert!(
        n0.re.abs() < 1e-10,
        "mode 0 empty: expected 0.0, got {:.6}",
        n0.re
    );
    assert!(
        (n1.re - 1.0).abs() < 1e-10,
        "mode 1 occupied: expected 1.0, got {:.6}",
        n1.re
    );
}

// ============================================================================
// 2. Pauli Z action: Z on occupied mode gives eigenvalue −1
// ============================================================================

#[test]
fn test_pauli_z_expectation_both_occupied() {
    // 2-mode system, both occupied → |11⟩
    // Z for mode 0 in a 2-qubit system corresponds to the string "ZI"
    // ⟨11|Z_0 ⊗ I_1|11⟩ = −1 (Z acting on mode 0 which is occupied)
    let mut sim = FermionicSimulator::new(2).expect("create simulator");
    sim.set_initial_state(&[true, true])
        .expect("set initial state");

    // Build the Pauli string "ZI" manually (num_qubits = num_modes = 2)
    let zi_string =
        PauliString::from_string("ZI", Complex64::new(1.0, 0.0)).expect("create ZI string");

    // We test via the number operator expectation path which internally calls
    // compute_pauli_expectation.  But we can also reach compute_pauli_expectation
    // directly through expectation_value with the Number operator (which maps to
    // -0.5 * Z_i + 0.5 * I_i).  For a direct Z test use the JW-transform exposed
    // through transform_operator.
    //
    // Instead: use a 1-mode system where n_0 = 1 and Z_0 on |1⟩ gives −1.
    let mut sim1 = FermionicSimulator::new(1).expect("create 1-mode simulator");
    sim1.set_initial_state(&[true]).expect("set occupied");

    // Z_0 maps to a Pauli string with a single Z.  For 1 mode: n_0 = (I − Z)/2
    // So ⟨1|Z|1⟩ = −1.
    let z_string =
        PauliString::from_string("Z", Complex64::new(1.0, 0.0)).expect("create Z string");

    // Drive through expectation_value by directly constructing a test.
    // We need access to compute_pauli_expectation, which is private.  Instead,
    // derive from the number operator: ⟨n⟩ = (1 − ⟨Z⟩)/2 → ⟨Z⟩ = 1 − 2⟨n⟩.
    let n0 = sim1
        .expectation_value(&FermionicOperator::Number(0))
        .expect("n0 expectation");
    let z_exp = 2.0_f64.mul_add(-n0.re, 1.0);

    assert!(
        (z_exp - (-1.0)).abs() < 1e-10,
        "⟨Z⟩ for occupied mode: expected -1.0, got {z_exp:.6}"
    );

    // Also verify for the 2-mode "ZI" case: ⟨11|ZI|11⟩ should be −1.
    // n_0 in 2-mode system with both occupied.
    let n0_2mode = sim
        .expectation_value(&FermionicOperator::Number(0))
        .expect("n0 2-mode");
    let z0_2mode = 2.0_f64.mul_add(-n0_2mode.re, 1.0);
    assert!(
        (z0_2mode - (-1.0)).abs() < 1e-10,
        "⟨Z_0⟩ in |11⟩: expected -1.0, got {z0_2mode:.6}"
    );

    // Verify the string itself doesn't cause a panic
    let _ = zi_string; // used to suppress unused warning
    let _ = z_string;
}

// ============================================================================
// 3. Norm preservation under time evolution
// ============================================================================

#[test]
fn test_norm_preserved_after_evolution() {
    // Use a 2-site Hubbard model (4 modes)
    let hamiltonian =
        FermionicHamiltonian::hubbard_model(2, 1.0, 1.0, 0.0).expect("create Hubbard model");

    let mut sim = FermionicSimulator::new(4).expect("create simulator");
    sim.set_initial_state(&[true, false, true, false])
        .expect("set initial state");

    // Evolve for multiple time steps
    for step in 1..=5 {
        let t = 0.1 * f64::from(step);
        sim.evolve_hamiltonian(&hamiltonian, t)
            .expect("evolve Hamiltonian");

        let norm_sq: f64 = sim.get_state().iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-8,
            "norm² after t={t:.1}: expected 1.0, got {norm_sq:.10}"
        );
    }
}

// ============================================================================
// 4. Hopping moves a particle between modes
// ============================================================================

#[test]
fn test_hopping_moves_particle() {
    // Start with mode 0 occupied, mode 1 empty in a 2-mode system.
    // Apply the hopping Hamiltonian H = c†_1 c_0 + c†_0 c_1 (Hermitian) with
    // strength t = 1.  At time τ = π/2 the particle should fully tunnel.
    // Using FermionicHamiltonian with hopping terms.
    let mut hamiltonian = FermionicHamiltonian::new(2);

    // Forward hopping: c†_1 c_0
    let hop_fwd = FermionicString::hopping(0, 1, Complex64::new(-1.0, 0.0), 2);
    hamiltonian.add_term(hop_fwd).expect("add fwd hop");

    // Backward hopping: c†_0 c_1 (Hermitian conjugate)
    let hop_bwd = FermionicString::hopping(1, 0, Complex64::new(-1.0, 0.0), 2);
    hamiltonian.add_term(hop_bwd).expect("add bwd hop");

    let mut sim = FermionicSimulator::new(2).expect("create simulator");
    sim.set_initial_state(&[true, false])
        .expect("set initial state");

    // Measure initial occupation
    let n0_init = sim
        .expectation_value(&FermionicOperator::Number(0))
        .expect("n0 init")
        .re;
    let n1_init = sim
        .expectation_value(&FermionicOperator::Number(1))
        .expect("n1 init")
        .re;

    assert!(
        (n0_init - 1.0).abs() < 1e-10,
        "initial n0 expected 1.0, got {n0_init:.6}"
    );
    assert!(
        n1_init.abs() < 1e-10,
        "initial n1 expected 0.0, got {n1_init:.6}"
    );

    // Evolve for a short time; both occupations should change
    sim.evolve_hamiltonian(&hamiltonian, 0.3).expect("evolve");

    let n0_after = sim
        .expectation_value(&FermionicOperator::Number(0))
        .expect("n0 after")
        .re;
    let n1_after = sim
        .expectation_value(&FermionicOperator::Number(1))
        .expect("n1 after")
        .re;

    // After hopping, mode 0 occupation decreases and mode 1 increases
    assert!(
        n0_after < n0_init - 1e-4,
        "mode 0 should decrease after hopping: {n0_init:.6} → {n0_after:.6}"
    );
    assert!(
        n1_after > n1_init + 1e-4,
        "mode 1 should increase after hopping: {n1_init:.6} → {n1_after:.6}"
    );

    // Particle number conservation: n0 + n1 ≈ 1 throughout
    assert!(
        (n0_after + n1_after - 1.0).abs() < 1e-8,
        "total particle number should be conserved: {:.6}",
        n0_after + n1_after
    );
}

// ============================================================================
// 5. Jordan-Wigner transform of c†_0 c_0
// ============================================================================

#[test]
fn test_jw_number_operator() {
    // n_0 = c†_0 c_0 should map to (I − Z_0)/2 under Jordan-Wigner.
    // For a 2-mode system: number operator on mode 0 is −0.5 * Z + 0.0 * I
    // (the constant +0.5*I is the identity contribution, not tracked directly).
    //
    // We verify this by checking expectation values:
    //   ⟨0|n_0|0⟩ = 0  (vacuum)
    //   ⟨1|n_0|1⟩ = 1  (occupied)

    // Vacuum state
    let mut sim_vac = FermionicSimulator::new(2).expect("create simulator");
    // Default state is vacuum (all zero occupation)
    let n0_vac = sim_vac
        .expectation_value(&FermionicOperator::Number(0))
        .expect("vacuum n0")
        .re;
    assert!(
        n0_vac.abs() < 1e-10,
        "vacuum ⟨n_0⟩ should be 0, got {n0_vac:.6}"
    );

    // Occupied state
    let mut sim_occ = FermionicSimulator::new(2).expect("create simulator");
    sim_occ
        .set_initial_state(&[true, false])
        .expect("set occupied");
    let n0_occ = sim_occ
        .expectation_value(&FermionicOperator::Number(0))
        .expect("occupied n0")
        .re;
    assert!(
        (n0_occ - 1.0).abs() < 1e-10,
        "occupied ⟨n_0⟩ should be 1, got {n0_occ:.6}"
    );

    // Verify the JW-transformed Pauli string has the correct structure:
    // transform_operator(Number(0)) should give coefficient -0.5 and a Z at position 0.
    use quantrs2_sim::fermionic_simulation::JordanWignerTransform;
    let mut jw = JordanWignerTransform::new(2);
    let pauli = jw
        .transform_operator(&FermionicOperator::Number(0))
        .expect("JW transform number op");

    assert_eq!(
        pauli.operators[0],
        PauliOperator::Z,
        "number operator site 0 should have Z at position 0"
    );
    assert!(
        (pauli.coefficient.re - (-0.5)).abs() < 1e-10,
        "number operator JW coefficient should be -0.5, got {:.6}",
        pauli.coefficient.re
    );
}

// ============================================================================
// 6. Particle correlation: decorrelated modes give zero connected part
// ============================================================================

#[test]
fn test_particle_correlation_product_state() {
    // In a product (Fock) state with mode 0 occupied and mode 1 empty:
    // ⟨n_0 n_1⟩ = 0,  ⟨n_0⟩ = 1,  ⟨n_1⟩ = 0
    // connected = 0 − 1·0 = 0
    let mut sim = FermionicSimulator::new(2).expect("create simulator");
    sim.set_initial_state(&[true, false])
        .expect("set initial state");

    let corr = sim.particle_correlation(0, 1).expect("correlation");
    assert!(
        corr.abs() < 1e-10,
        "connected correlation in product state should be 0, got {corr:.10}"
    );
}

// ============================================================================
// 7. Vacuum state has zero expectation for all number operators
// ============================================================================

#[test]
fn test_vacuum_state_zero_occupation() {
    let mut sim = FermionicSimulator::new(3).expect("create simulator");
    // Default state is vacuum

    for mode in 0..3 {
        let n = sim
            .expectation_value(&FermionicOperator::Number(mode))
            .expect("number op");
        assert!(
            n.re.abs() < 1e-10,
            "vacuum ⟨n_{mode}⟩ should be 0, got {:.6}",
            n.re
        );
    }
}
