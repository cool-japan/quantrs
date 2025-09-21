//! Demonstration of circuit equivalence checking
//!
//! This example shows how to verify that different quantum circuits
//! produce equivalent results.

use quantrs2_circuit::prelude::*;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Quantum Circuit Equivalence Checking Demo ===\n");

    // Example 1: Structural Equivalence
    demo_structural_equivalence()?;
    println!();

    // Example 2: Algebraic Equivalence
    demo_algebraic_equivalence()?;
    println!();

    // Example 3: Phase Equivalence
    demo_phase_equivalence()?;
    println!();

    // Example 4: Non-Equivalence Detection
    demo_non_equivalence()?;
    println!();

    // Example 5: Custom Tolerance
    demo_custom_tolerance()?;

    Ok(())
}

/// Demonstrate checking structural equivalence
fn demo_structural_equivalence() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Structural Equivalence Check");
    println!("   Checking if two circuits have identical gate sequences");

    // Create two identical Bell state circuits
    let mut circuit1 = Circuit::<2>::new();
    circuit1.h(0)?;
    circuit1.cnot(0, 1)?;

    let mut circuit2 = Circuit::<2>::new();
    circuit2.h(0)?;
    circuit2.cnot(0, 1)?;

    // Check structural equivalence
    let checker = EquivalenceChecker::default();
    let result = checker.check_structural_equivalence(&circuit1, &circuit2)?;

    println!("   Circuit 1: H(0), CNOT(0,1)");
    println!("   Circuit 2: H(0), CNOT(0,1)");
    println!("   Structurally equivalent: {}", result.equivalent);
    println!("   Details: {}", result.details);

    Ok(())
}

/// Demonstrate algebraic equivalence (different gates, same result)
fn demo_algebraic_equivalence() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Algebraic Equivalence Check");
    println!("   Different gate sequences that produce the same result");

    // Circuit 1: X followed by X (should equal identity)
    let mut circuit1 = Circuit::<1>::new();
    circuit1.x(0)?;
    circuit1.x(0)?;

    // Circuit 2: Empty circuit (identity)
    let circuit2 = Circuit::<1>::new();

    // Check equivalence
    let mut checker = EquivalenceChecker::default();
    let result = checker.check_equivalence(&circuit1, &circuit2)?;

    println!("   Circuit 1: X(0), X(0)");
    println!("   Circuit 2: (empty)");
    println!("   Equivalent: {}", result.equivalent);
    println!("   Check type: {:?}", result.check_type);

    // Another example: HZH = X
    let mut circuit3 = Circuit::<1>::new();
    circuit3.h(0)?;
    circuit3.z(0)?;
    circuit3.h(0)?;

    let mut circuit4 = Circuit::<1>::new();
    circuit4.x(0)?;

    let result2 = checker.check_structural_equivalence(&circuit3, &circuit4)?;

    println!("\n   Circuit 3: H(0), Z(0), H(0)");
    println!("   Circuit 4: X(0)");
    println!("   Structurally equivalent: {}", result2.equivalent);
    println!("   (Note: They are algebraically equivalent but not structurally)");

    Ok(())
}

/// Demonstrate phase equivalence
fn demo_phase_equivalence() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Global Phase Equivalence");
    println!("   Circuits that differ only by a global phase");

    // Circuit 1: S gate (phase π/2)
    let mut circuit1 = Circuit::<1>::new();
    circuit1.s(0)?;

    // Circuit 2: Z followed by T (phase π + π/4 = 5π/4)
    // This gives the same result up to global phase
    let mut circuit2 = Circuit::<1>::new();
    circuit2.z(0)?;
    circuit2.t(0)?;

    // Check with phase ignored
    let mut checker_phase = EquivalenceChecker::new(EquivalenceOptions {
        tolerance: 1e-10,
        ignore_global_phase: true,
        check_all_states: true,
        max_unitary_qubits: 10,
        enable_adaptive_tolerance: true,
        enable_statistical_analysis: true,
        enable_stability_analysis: true,
        enable_graph_comparison: false,
        confidence_level: 0.95,
        max_condition_number: 1e12,
        scirs2_config: None,
        complex_tolerance: 1e-14,
        enable_parallel_computation: true,
    });

    // Check without phase ignored
    let mut checker_no_phase = EquivalenceChecker::new(EquivalenceOptions {
        tolerance: 1e-10,
        ignore_global_phase: false,
        check_all_states: true,
        max_unitary_qubits: 10,
        enable_adaptive_tolerance: true,
        enable_statistical_analysis: true,
        enable_stability_analysis: true,
        enable_graph_comparison: false,
        confidence_level: 0.95,
        max_condition_number: 1e12,
        scirs2_config: None,
        complex_tolerance: 1e-14,
        enable_parallel_computation: true,
    });

    println!("   Circuit 1: S(0)");
    println!("   Circuit 2: Z(0), T(0)");

    // Note: This example is simplified - actual phase relationship may differ
    println!("   With global phase ignored: (would check if implemented)");
    println!("   Without global phase: (would check if implemented)");

    Ok(())
}

/// Demonstrate detection of non-equivalent circuits
fn demo_non_equivalence() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Non-Equivalence Detection");
    println!("   Detecting when circuits are NOT equivalent");

    // Circuit 1: Bell state |00> + |11>
    let mut circuit1 = Circuit::<2>::new();
    circuit1.h(0)?;
    circuit1.cnot(0, 1)?;

    // Circuit 2: Different entangled state |01> + |10>
    let mut circuit2 = Circuit::<2>::new();
    circuit2.h(0)?;
    circuit2.x(1)?;
    circuit2.cnot(0, 1)?;

    let checker = EquivalenceChecker::default();
    let result = checker.check_structural_equivalence(&circuit1, &circuit2)?;

    println!("   Circuit 1: H(0), CNOT(0,1)");
    println!("   Circuit 2: H(0), X(1), CNOT(0,1)");
    println!("   Equivalent: {}", result.equivalent);
    println!("   Details: {}", result.details);

    Ok(())
}

/// Demonstrate custom tolerance settings
fn demo_custom_tolerance() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Custom Tolerance Settings");
    println!("   Using different tolerance levels for approximate equivalence");

    // Create circuits that might have small numerical differences
    let mut circuit1 = Circuit::<1>::new();
    circuit1.rx(0, PI / 4.0)?;

    let mut circuit2 = Circuit::<1>::new();
    circuit2.rx(0, PI / 4.0 + 1e-12)?; // Tiny difference

    // Strict tolerance
    let mut strict_checker = EquivalenceChecker::new(EquivalenceOptions {
        tolerance: 1e-15,
        ignore_global_phase: false,
        check_all_states: true,
        max_unitary_qubits: 10,
        enable_adaptive_tolerance: true,
        enable_statistical_analysis: true,
        enable_stability_analysis: true,
        enable_graph_comparison: false,
        confidence_level: 0.95,
        max_condition_number: 1e12,
        scirs2_config: None,
        complex_tolerance: 1e-14,
        enable_parallel_computation: true,
    });

    // Relaxed tolerance
    let mut relaxed_checker = EquivalenceChecker::new(EquivalenceOptions {
        tolerance: 1e-10,
        ignore_global_phase: false,
        check_all_states: true,
        max_unitary_qubits: 10,
        enable_adaptive_tolerance: true,
        enable_statistical_analysis: true,
        enable_stability_analysis: true,
        enable_graph_comparison: false,
        confidence_level: 0.95,
        max_condition_number: 1e12,
        scirs2_config: None,
        complex_tolerance: 1e-14,
        enable_parallel_computation: true,
    });

    println!("   Circuit 1: RX(0, π/4)");
    println!("   Circuit 2: RX(0, π/4 + 1e-12)");
    println!("   With strict tolerance (1e-15): (would check if implemented)");
    println!("   With relaxed tolerance (1e-10): (would check if implemented)");

    Ok(())
}

/// Additional example: Verify circuit optimization preserves behavior
fn verify_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nBonus: Verifying Circuit Optimization");

    // Original circuit with redundant gates
    let mut original = Circuit::<2>::new();
    original.h(0)?;
    original.cnot(0, 1)?;
    original.cnot(0, 1)?; // This cancels the previous CNOT
    original.h(0)?;

    // Optimized circuit (manually optimized for this example)
    let optimized = Circuit::<2>::new();
    // Empty circuit since H-CNOT-CNOT-H = H-H = I

    let checker = EquivalenceChecker::default();
    println!("   Original: H(0), CNOT(0,1), CNOT(0,1), H(0)");
    println!("   Optimized: (empty)");

    // In practice, you would use the circuit optimizer and then verify:
    // let optimized = original.optimize()?;
    // let result = checker.check_equivalence(&original, &optimized)?;

    Ok(())
}
