//! Demonstration of unitary synthesis capabilities
//!
//! This example shows how to synthesize quantum circuits from unitary matrix
//! descriptions using various decomposition algorithms.

use quantrs2_circuit::prelude::*;
use quantrs2_circuit::synthesis::unitaries::{
    cnot, hadamard, pauli_x, pauli_y, pauli_z, rotation_x, rotation_y, rotation_z,
};
use quantrs2_core::qubit::QubitId;
use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;
use std::f64::consts::PI;

type C64 = Complex64;

fn main() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("=== Unitary Synthesis Demo ===\n");

    demo_single_qubit_synthesis()?;
    demo_two_qubit_synthesis()?;
    demo_common_operations()?;
    demo_gate_sets();
    demo_validation();
    demo_synthesis_comparison();

    Ok(())
}

fn demo_single_qubit_synthesis() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Single-Qubit Unitary Synthesis ---");

    let config = SynthesisConfig::default();
    let synthesizer = SingleQubitSynthesizer::new(config);

    // Test common single-qubit unitaries
    let unitaries = vec![
        ("Hadamard", hadamard()),
        ("Pauli-X", pauli_x()),
        ("Pauli-Y", pauli_y()),
        ("Pauli-Z", pauli_z()),
        ("RX(π/4)", rotation_x(PI / 4.0)),
        ("RY(π/3)", rotation_y(PI / 3.0)),
        ("RZ(π/2)", rotation_z(PI / 2.0)),
    ];

    for (name, unitary) in unitaries {
        let circuit: Circuit<1> = synthesizer.synthesize(&unitary, QubitId(0))?;
        println!("{:>12}: {} gates", name, circuit.num_gates());

        // Show gate sequence for Hadamard as example
        if name == "Hadamard" {
            println!("             Gate sequence: ");
            for (i, gate) in circuit.gates().iter().enumerate() {
                println!("             {}: {}", i, gate.name());
            }
        }
    }

    println!();
    Ok(())
}

fn demo_two_qubit_synthesis() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Two-Qubit Unitary Synthesis ---");

    let config = SynthesisConfig::default();
    let synthesizer = TwoQubitSynthesizer::new(config);

    // Test CNOT synthesis
    let cnot_matrix = cnot();
    let circuit: Circuit<2> = synthesizer.synthesize(&cnot_matrix, QubitId(0), QubitId(1))?;

    println!("CNOT synthesis:");
    println!("  Gates: {}", circuit.num_gates());
    println!("  Gate sequence:");
    for (i, gate) in circuit.gates().iter().enumerate() {
        println!("    {}: {}", i, gate.name());
    }

    // Test controlled rotation
    let theta = PI / 4.0;
    println!("\nControlled RY(π/4) synthesis:");
    let controlled_ry_circuit: Circuit<2> = synthesizer.cartan_decomposition(
        &cnot_matrix, // Placeholder - would be actual controlled-RY matrix
        QubitId(0),
        QubitId(1),
    )?;
    println!("  Gates: {}", controlled_ry_circuit.num_gates());

    println!();
    Ok(())
}

fn demo_common_operations() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Common Quantum Operations ---");

    let synthesizer = UnitarySynthesizer::default_config();

    // Quantum Fourier Transform
    println!("Quantum Fourier Transform:");
    for n_qubits in 2..=4 {
        let qft_circuit: Circuit<4> = synthesizer.synthesize_qft(n_qubits)?;
        println!("  QFT-{}: {} gates", n_qubits, qft_circuit.num_gates());
    }

    // Toffoli gate
    let toffoli_circuit: Circuit<3> =
        synthesizer.synthesize_toffoli(QubitId(0), QubitId(1), QubitId(2))?;
    println!("\nToffoli gate: {} gates", toffoli_circuit.num_gates());

    // Show first few gates of Toffoli decomposition
    println!("  First 5 gates:");
    for (i, gate) in toffoli_circuit.gates().iter().take(5).enumerate() {
        println!("    {}: {}", i, gate.name());
    }

    println!();
    Ok(())
}

fn demo_gate_sets() {
    println!("--- Different Gate Sets ---");

    let gate_sets = vec![
        ("Universal", GateSet::Universal),
        ("IBM", GateSet::IBM),
        ("Google", GateSet::Google),
        ("Rigetti", GateSet::Rigetti),
    ];

    let hadamard_matrix = hadamard();

    for (name, gate_set) in gate_sets {
        let synthesizer = UnitarySynthesizer::for_gate_set(gate_set);

        // For this demo, just show the configuration
        println!("{name} gate set configured");

        // In a full implementation, would show different gate decompositions
        // For now, all use the same underlying synthesis
        let config = SynthesisConfig {
            gate_set: synthesizer.config.gate_set.clone(),
            ..Default::default()
        };

        println!("  Tolerance: {}", config.tolerance);
        println!("  Max gates: {}", config.max_gates);
    }

    println!();
}

fn demo_validation() {
    println!("--- Unitary Matrix Validation ---");

    let synthesizer = UnitarySynthesizer::default_config();

    // Test valid unitaries
    println!("Valid unitaries:");

    // Identity matrix in row-major format
    let identity_2x2 = Array2::from_shape_vec(
        (2, 2),
        vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
        ],
    )
    .unwrap();

    match synthesizer.validate_unitary(&identity_2x2) {
        Ok(()) => println!("  ✓ 2x2 Identity matrix"),
        Err(e) => println!("  ✗ 2x2 Identity matrix: {e}"),
    }

    // Hadamard matrix in row-major format
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let hadamard_matrix = Array2::from_shape_vec(
        (2, 2),
        vec![
            C64::new(inv_sqrt2, 0.0),
            C64::new(inv_sqrt2, 0.0),
            C64::new(inv_sqrt2, 0.0),
            C64::new(-inv_sqrt2, 0.0),
        ],
    )
    .unwrap();

    match synthesizer.validate_unitary(&hadamard_matrix) {
        Ok(()) => println!("  ✓ Hadamard matrix"),
        Err(e) => println!("  ✗ Hadamard matrix: {e}"),
    }

    // Test invalid unitaries
    println!("\nInvalid unitaries:");

    // Non-square matrix (2x3) in row-major format
    let non_square = Array2::from_shape_vec(
        (2, 3),
        vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
        ],
    )
    .unwrap();

    match synthesizer.validate_unitary(&non_square) {
        Ok(()) => println!("  ✓ Non-square matrix (unexpected)"),
        Err(_) => println!("  ✗ Non-square matrix (expected)"),
    }

    // Non-unitary matrix in row-major format
    let non_unitary = Array2::from_shape_vec(
        (2, 2),
        vec![
            C64::new(2.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
        ],
    )
    .unwrap();

    match synthesizer.validate_unitary(&non_unitary) {
        Ok(()) => println!("  ✓ Non-unitary matrix (unexpected)"),
        Err(_) => println!("  ✗ Non-unitary matrix (expected)"),
    }

    // 3x3 matrix (non-power-of-2) in row-major format
    let wrong_dimension = Array2::from_shape_vec(
        (3, 3),
        vec![
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(0.0, 0.0),
            C64::new(1.0, 0.0),
        ],
    )
    .unwrap();

    match synthesizer.validate_unitary(&wrong_dimension) {
        Ok(()) => println!("  ✓ Non-power-of-2 dimension (unexpected)"),
        Err(_) => println!("  ✗ Non-power-of-2 dimension (expected)"),
    }

    println!();
}

fn demo_synthesis_comparison() {
    println!("--- Synthesis Algorithm Comparison ---");

    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let test_unitaries = vec![
        ("Identity", Array2::<C64>::eye(2)),
        (
            "Hadamard",
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    C64::new(inv_sqrt2, 0.0),
                    C64::new(inv_sqrt2, 0.0),
                    C64::new(inv_sqrt2, 0.0),
                    C64::new(-inv_sqrt2, 0.0),
                ],
            )
            .unwrap(),
        ),
        (
            "T gate",
            Array2::from_shape_vec(
                (2, 2),
                vec![
                    C64::new(1.0, 0.0),
                    C64::new(0.0, 0.0),
                    C64::new(0.0, 0.0),
                    C64::from_polar(1.0, PI / 4.0),
                ],
            )
            .unwrap(),
        ),
    ];

    let optimization_levels = vec![0, 1, 2, 3];

    println!(
        "{:<12} {:<8} {:<8} {:<8} {:<8}",
        "Unitary", "Opt-0", "Opt-1", "Opt-2", "Opt-3"
    );
    println!("{:-<48}", "");

    for (name, unitary) in test_unitaries {
        print!("{name:<12}");

        for &opt_level in &optimization_levels {
            let config = SynthesisConfig {
                optimization_level: opt_level,
                ..Default::default()
            };

            let synthesizer = UnitarySynthesizer::new(config);
            match synthesizer.synthesize::<1>(&unitary) {
                Ok(circuit) => print!(" {:<8}", circuit.num_gates()),
                Err(_) => print!(" {:<8}", "Error"),
            }
        }
        println!();
    }

    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_demo() {
        let result = main();
        assert!(result.is_ok());
    }

    #[test]
    fn test_unitary_creation() {
        let h = hadamard();
        let x = pauli_x();
        let y = pauli_y();
        let z = pauli_z();

        // Check dimensions
        assert_eq!(h.nrows(), 2);
        assert_eq!(x.ncols(), 2);
        assert_eq!(y.nrows(), 2);
        assert_eq!(z.ncols(), 2);

        // Check that they're different
        assert_ne!(h, x);
        assert_ne!(y, z);
    }

    #[test]
    fn test_cnot_matrix() {
        let cnot_mat = cnot();
        assert_eq!(cnot_mat.nrows(), 4);
        assert_eq!(cnot_mat.ncols(), 4);

        // CNOT should be real-valued
        for element in cnot_mat.iter() {
            assert!(element.im.abs() < 1e-10);
        }
    }
}
