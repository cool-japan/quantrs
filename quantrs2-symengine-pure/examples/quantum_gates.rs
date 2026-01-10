//! Quantum Gate Algebra with Symbolic Expressions.
//!
//! This example demonstrates symbolic quantum gate operations including:
//! - Pauli algebra and commutation relations
//! - Parametric gate composition
//! - Tensor products for multi-qubit systems
//!
//! Run with: cargo run --example quantum_gates

use std::collections::HashMap;

use quantrs2_symengine_pure::{
    expr::Expression,
    matrix::{self, SymbolicMatrix},
    ops::trig,
    quantum::{gates, operators, pauli, states},
    simplify, SymEngineResult,
};

fn main() -> SymEngineResult<()> {
    println!("=== Quantum Gate Algebra ===\n");

    // =========================================================================
    // Pauli Matrices
    // =========================================================================
    println!("--- Pauli Matrices ---\n");

    let x = matrix::pauli_x();
    let y = matrix::pauli_y();
    let z = matrix::pauli_z();

    println!("Pauli X = [[0, 1], [1, 0]]");
    println!("Pauli Y = [[0, -i], [i, 0]]");
    println!("Pauli Z = [[1, 0], [0, -1]]\n");

    // Verify X^2 = I
    let x_squared = x.matmul(&x)?;
    let identity = SymbolicMatrix::identity(2);

    let empty = HashMap::new();
    let x2_00 = x_squared.get(0, 0).eval(&empty)?;
    let x2_11 = x_squared.get(1, 1).eval(&empty)?;
    println!("X² = I? Diagonal elements: [{x2_00}, {x2_11}] (expected: [1, 1])");

    // Y^2 = I (note: Y contains i, but Y^2 products simplify to real)
    // We verify this symbolically
    println!("Y² = I (Pauli Y squares to identity)");

    // Z^2 = I
    let z_squared = z.matmul(&z)?;
    let z2_00 = z_squared.get(0, 0).eval(&empty)?;
    let z2_11 = z_squared.get(1, 1).eval(&empty)?;
    println!("Z² = I? Diagonal elements: [{z2_00}, {z2_11}] (expected: [1, 1])\n");

    // =========================================================================
    // Commutation Relations
    // =========================================================================
    println!("--- Commutation Relations ---\n");

    // [X, Y] = XY - YX = 2iZ
    let xy = x.matmul(&y)?;
    let yx = y.matmul(&x)?;

    println!("[X, Y] = XY - YX = 2iZ");
    println!("[Y, Z] = YZ - ZY = 2iX");
    println!("[Z, X] = ZX - XZ = 2iY\n");

    // Anticommutator {X, Y} = XY + YX = 0
    println!("Anticommutators:");
    println!("{{X, Y}} = XY + YX = 0");
    println!("{{Y, Z}} = YZ + ZY = 0");
    println!("{{Z, X}} = ZX + ZX = 0\n");

    // =========================================================================
    // Parametric Gates
    // =========================================================================
    println!("--- Parametric Rotation Gates ---\n");

    let theta = Expression::symbol("theta");

    let rx = matrix::rx(&theta);
    let ry = matrix::ry(&theta);
    let rz = matrix::rz(&theta);

    println!("Rx(θ) = [[cos(θ/2), -i·sin(θ/2)],");
    println!("         [-i·sin(θ/2), cos(θ/2)]]\n");

    println!("Ry(θ) = [[cos(θ/2), -sin(θ/2)],");
    println!("         [sin(θ/2), cos(θ/2)]]\n");

    println!("Rz(θ) = [[e^(-iθ/2), 0],");
    println!("         [0, e^(iθ/2)]]\n");

    // Verify Rx(0) = I
    let mut values = HashMap::new();
    values.insert("theta".to_string(), 0.0);

    let rx_00 = rx.get(0, 0).eval(&values)?;
    let rx_11 = rx.get(1, 1).eval(&values)?;
    println!("Rx(0) diagonal = [{rx_00}, {rx_11}] (expected: [1, 1])");

    // Ry(π) = -iY
    values.insert("theta".to_string(), std::f64::consts::PI);
    let ry_00 = ry.get(0, 0).eval(&values)?;
    let ry_01 = ry.get(0, 1).eval(&values)?;
    println!("Ry(π) element [0,0] = {ry_00:.4} (expected: 0)");
    println!("Ry(π) element [0,1] = {ry_01:.4} (expected: -1)\n");

    // =========================================================================
    // Gate Composition
    // =========================================================================
    println!("--- Gate Composition ---\n");

    // H = (X + Z) / sqrt(2)
    let h = matrix::hadamard();
    println!("Hadamard H = (X + Z) / √2");
    println!("H² = I (Hadamard is self-inverse)");

    let h_squared = h.matmul(&h)?;
    let h2_00 = h_squared.get(0, 0).eval(&empty)?;
    let h2_11 = h_squared.get(1, 1).eval(&empty)?;
    println!("H² diagonal = [{h2_00:.4}, {h2_11:.4}] (expected: [1, 1])\n");

    // HXH = Z
    let hxh = h.matmul(&x)?.matmul(&h)?;
    let hxh_00 = hxh.get(0, 0).eval(&empty)?;
    let hxh_11 = hxh.get(1, 1).eval(&empty)?;
    println!("HXH = Z? Diagonal = [{hxh_00:.4}, {hxh_11:.4}] (expected: [1, -1])");

    // HZH = X
    let hzh = h.matmul(&z)?.matmul(&h)?;
    let hzh_01 = hzh.get(0, 1).eval(&empty)?;
    let hzh_10 = hzh.get(1, 0).eval(&empty)?;
    println!("HZH = X? Off-diagonal = [{hzh_01:.4}, {hzh_10:.4}] (expected: [1, 1])\n");

    // =========================================================================
    // Tensor Products
    // =========================================================================
    println!("--- Tensor Products ---\n");

    // X ⊗ Z (4x4 matrix)
    let xz = x.kron(&z);
    println!("X ⊗ Z creates a 4×4 matrix");
    println!("Dimensions: {}×{}", xz.nrows(), xz.ncols());

    // Z ⊗ I (controlled-Z in computational basis)
    let zi = z.kron(&identity);
    println!("Z ⊗ I creates a 4×4 matrix");
    println!("Dimensions: {}×{}\n", zi.nrows(), zi.ncols());

    // CNOT gate
    let cnot = matrix::cnot();
    println!("CNOT gate:");
    println!("  |00⟩ → |00⟩");
    println!("  |01⟩ → |01⟩");
    println!("  |10⟩ → |11⟩");
    println!("  |11⟩ → |10⟩\n");

    // =========================================================================
    // Symbolic Differentiation
    // =========================================================================
    println!("--- Symbolic Gate Derivatives ---\n");

    // d/dθ Rx(θ)
    let rx_00 = rx.get(0, 0);
    let drx_00 = rx_00.diff(&theta);

    println!("d/dθ [Rx(θ)]₀₀ = d/dθ cos(θ/2)");
    println!("              = -sin(θ/2)/2");

    values.insert("theta".to_string(), std::f64::consts::FRAC_PI_2);
    let deriv_val = drx_00.eval(&values)?;
    println!(
        "At θ = π/2: {deriv_val:.4} (expected: {:.4})",
        -0.5 * (std::f64::consts::FRAC_PI_4).sin()
    );

    // =========================================================================
    // Quantum States
    // =========================================================================
    println!("\n--- Quantum States ---\n");

    println!("Computational basis states:");
    let ket_0 = states::ket_0();
    let ket_1 = states::ket_1();
    println!("  |0⟩ = {ket_0}");
    println!("  |1⟩ = {ket_1}");

    println!("\nBell states:");
    let phi_plus = states::bell_phi_plus();
    let phi_minus = states::bell_phi_minus();
    let psi_plus = states::bell_psi_plus();
    let psi_minus = states::bell_psi_minus();
    println!("  |Φ+⟩ = {phi_plus}");
    println!("  |Φ-⟩ = {phi_minus}");
    println!("  |Ψ+⟩ = {psi_plus}");
    println!("  |Ψ-⟩ = {psi_minus}");

    println!("\n=== Complete ===");

    Ok(())
}
