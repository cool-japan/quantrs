//! Quantum Error Correction Example
//!
//! This example demonstrates quantum error correction codes available in QuantRS2-Core.
//!
//! Run with: cargo run --example error_correction

use quantrs2_core::{
    error::QuantRS2Result,
    error_correction::{ColorCode, Pauli, PauliString, SurfaceCode, ToricCode},
};

fn main() -> QuantRS2Result<()> {
    println!("=================================================================");
    println!("   QuantRS2-Core: Quantum Error Correction");
    println!("=================================================================\n");

    demonstrate_error_correction_codes();
    println!();

    demonstrate_pauli_operations();
    println!();

    println!("=================================================================");
    println!("   Example Complete!");
    println!("=================================================================");

    Ok(())
}

/// Demonstrate error correction code structures
fn demonstrate_error_correction_codes() {
    println!("ERROR CORRECTION CODES");
    println!("-----------------------------------------------------------------");

    println!("QuantRS2-Core includes several quantum error correction codes:\n");

    // Surface Code
    println!("1. Surface Code");
    let surface_code = SurfaceCode::new(5, 5);
    println!("   Lattice: 5×5");
    println!("   Properties:");
    println!("     • 2D nearest-neighbor connectivity");
    println!("     • Suitable for planar quantum architectures");
    println!("     • Error threshold ~1% with efficient decoder");
    println!("     • Uses stabilizer formalism");

    // Color Code
    println!("\n2. Color Code");
    let color_code = ColorCode::triangular(5);
    println!("   Size: 5 (triangular lattice)");
    println!("   Properties:");
    println!("     • Transversal Clifford gates");
    println!("     • Triangular/hexagonal lattice");
    println!("     • Three-coloring for stabilizers");
    println!("     • Better gate implementations vs surface codes");

    // Toric Code
    println!("\n3. Toric Code");
    let toric_code = ToricCode::new(5, 5);
    println!("   Lattice: 5×5 on a torus");
    println!("   Properties:");
    println!("     • Topological protection");
    println!("     • Periodic boundary conditions");
    println!("     • Star and plaquette stabilizers");
    println!("     • Fundamental example of topological order");

    println!("\nAll codes use the stabilizer formalism:");
    println!("  • Stabilizers: Operators that leave code space invariant");
    println!("  • Syndrome: Measurement outcomes of stabilizers");
    println!("  • Decoding: Determining error from syndrome");
    println!("  • Correction: Applying recovery operations");

    println!("\n  ✓ Error correction codes available");
}

/// Demonstrate Pauli operations
fn demonstrate_pauli_operations() {
    println!("PAULI OPERATIONS");
    println!("-----------------------------------------------------------------");

    println!("Pauli operators form the basis for quantum errors:\n");

    // Individual Pauli operators
    println!("Individual Pauli operators:");
    println!("  I (Identity): No error");
    println!("  X (Bit-flip): |0⟩ ↔ |1⟩");
    println!("  Y (Bit-phase flip): |0⟩ → i|1⟩, |1⟩ → -i|0⟩");
    println!("  Z (Phase-flip): |1⟩ → -|1⟩");

    // Pauli string example
    println!("\nPauli strings (multi-qubit errors):");

    let error1 = PauliString::new(vec![Pauli::X]);
    println!("  Single X error: X on qubit 0");

    let error2 = PauliString::new(vec![Pauli::X, Pauli::Z]);
    println!("  Two-qubit error: X on qubit 0, Z on qubit 1");

    let error3 = PauliString::new(vec![Pauli::Y, Pauli::Y, Pauli::X]);
    println!("  Three-qubit error: Y on qubit 0, Y on qubit 1, X on qubit 2");

    println!("\nError correction workflow:");
    println!("  1. Encode logical qubit(s) into physical qubits");
    println!("  2. Perform quantum computation");
    println!("  3. Physical errors occur (modeled as Pauli operators)");
    println!("  4. Measure stabilizers → syndrome");
    println!("  5. Classical decoder: syndrome → error location");
    println!("  6. Apply correction (Pauli operations)");
    println!("  7. Continue computation or measure");

    println!("\nKey concepts:");
    println!("  • Distance d: minimum weight of logical operator");
    println!("  • Can correct t = ⌊(d-1)/2⌋ errors");
    println!("  • Syndrome measurement: non-destructive");
    println!("  • Fault-tolerant threshold: error rate below which");
    println!("    adding more qubits decreases logical error rate");

    println!("\n  ✓ Pauli operations explained");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_runs() {
        assert!(main().is_ok());
    }

    #[test]
    fn test_surface_code_creation() {
        let code = SurfaceCode::new(3, 3);
        // Code created successfully
    }

    #[test]
    fn test_color_code_creation() {
        let code = ColorCode::triangular(3);
        // Code created successfully
    }

    #[test]
    fn test_toric_code_creation() {
        let code = ToricCode::new(3, 3);
        // Code created successfully
    }

    #[test]
    fn test_pauli_string_creation() {
        let pauli_string = PauliString::new(vec![Pauli::X, Pauli::Z, Pauli::Y]);
        // Pauli string created successfully
    }
}
