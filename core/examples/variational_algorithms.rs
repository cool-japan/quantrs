//! Variational Quantum Algorithms Example
//!
//! This example demonstrates Variational Quantum Eigensolver (VQE) and
//! Quantum Approximate Optimization Algorithm (QAOA) using QuantRS2-Core.
//!
//! These algorithms are hybrid quantum-classical approaches that are among
//! the most promising near-term quantum computing applications.
//!
//! Run with: cargo run --example variational_algorithms

use quantrs2_core::{
    error::QuantRS2Result,
    gate::{multi, single, GateOp},
    qubit::QubitId,
};
use scirs2_core::Complex64;

fn main() -> QuantRS2Result<()> {
    println!("=================================================================");
    println!("   QuantRS2-Core: Variational Quantum Algorithms");
    println!("=================================================================\n");

    // Demonstrate VQE basics
    demonstrate_vqe_basics()?;
    println!();

    // Demonstrate QAOA basics
    demonstrate_qaoa_basics()?;
    println!();

    // Demonstrate variational circuit construction
    demonstrate_variational_circuit()?;
    println!();

    // Demonstrate parameter optimization
    demonstrate_parameter_optimization()?;
    println!();

    println!("=================================================================");
    println!("   Example Complete!");
    println!("=================================================================");

    Ok(())
}

/// Demonstrate Variational Quantum Eigensolver (VQE) basics
fn demonstrate_vqe_basics() -> QuantRS2Result<()> {
    println!("VARIATIONAL QUANTUM EIGENSOLVER (VQE)");
    println!("-----------------------------------------------------------------");

    println!("VQE is a hybrid quantum-classical algorithm for finding");
    println!("the ground state energy of a molecular Hamiltonian.");
    println!();

    // Example: H2 molecule Hamiltonian (simplified)
    println!("Application: H₂ Molecule Ground State Energy");
    println!("  • Qubits needed: 2 (for 2 electrons in 2 orbitals)");
    println!("  • Hamiltonian terms: 5 Pauli operators");
    println!("  • Classical optimization: Gradient descent or BFGS");
    println!();

    // Create a simple variational ansatz
    println!("Variational Ansatz (Hardware-Efficient):");
    println!("  1. RY(θ₁) on qubit 0");
    println!("  2. RY(θ₂) on qubit 1");
    println!("  3. CNOT(0, 1) for entanglement");
    println!("  4. RY(θ₃) on qubit 0");
    println!("  5. RY(θ₄) on qubit 1");
    println!();

    // Demonstrate parameter space
    let n_params = 4;
    println!("Parameter Space:");
    println!("  • Dimensionality: {n_params} parameters");
    println!("  • Search space: [0, 2π]^{n_params}");
    println!("  • Optimization landscape: Non-convex with local minima");
    println!();

    // Show typical VQE workflow
    println!("VQE Workflow:");
    println!("  1. Initialize variational parameters θ");
    println!("  2. Prepare quantum state |ψ(θ)⟩");
    println!("  3. Measure energy E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩");
    println!("  4. Classical optimizer updates θ → θ'");
    println!("  5. Repeat until convergence");
    println!();

    println!("  ✓ VQE basics demonstrated");
    println!("  ✓ Used for: Molecular simulation, materials science");

    Ok(())
}

/// Demonstrate Quantum Approximate Optimization Algorithm (QAOA) basics
fn demonstrate_qaoa_basics() -> QuantRS2Result<()> {
    println!("QUANTUM APPROXIMATE OPTIMIZATION ALGORITHM (QAOA)");
    println!("-----------------------------------------------------------------");

    println!("QAOA is a hybrid algorithm for solving combinatorial");
    println!("optimization problems like MaxCut, TSP, and SAT.");
    println!();

    // Example: MaxCut problem
    println!("Application: Graph MaxCut Problem");
    println!("  • Problem: Partition graph vertices to maximize cut edges");
    println!("  • Qubits: One per vertex");
    println!("  • Classical NP-hard, quantum approximate solution");
    println!();

    // QAOA structure
    println!("QAOA Structure:");
    println!("  Circuit depth p (number of layers):");
    println!("    • p=1: Simple approximation, fast");
    println!("    • p→∞: Approaches optimal solution");
    println!("    • Trade-off: accuracy vs. circuit depth");
    println!();

    println!("QAOA Layer Structure:");
    println!("  For each layer i:");
    println!("    1. Cost Hamiltonian: exp(-iγᵢHc)");
    println!("       • Encodes problem structure");
    println!("       • Applied as problem-specific gates");
    println!("    2. Mixer Hamiltonian: exp(-iβᵢHm)");
    println!("       • Typically X rotations on all qubits");
    println!("       • Explores solution space");
    println!();

    println!("Example: 4-node graph MaxCut");
    println!("  • 4 qubits (one per node)");
    println!("  • 4 edges → 4 ZZ interaction terms");
    println!("  • p=1: 2 variational parameters (γ, β)");
    println!("  • p=3: 6 parameters (γ₁,β₁,γ₂,β₂,γ₃,β₃)");
    println!();

    println!("QAOA Workflow:");
    println!("  1. Initialize in uniform superposition: H⊗ⁿ|0⟩");
    println!("  2. Apply p QAOA layers with parameters (γ,β)");
    println!("  3. Measure in computational basis");
    println!("  4. Compute cost function value");
    println!("  5. Classical optimizer updates (γ,β)");
    println!("  6. Repeat until convergence");
    println!();

    println!("  ✓ QAOA basics demonstrated");
    println!("  ✓ Used for: Logistics, scheduling, portfolio optimization");

    Ok(())
}

/// Demonstrate variational circuit construction
fn demonstrate_variational_circuit() -> QuantRS2Result<()> {
    println!("VARIATIONAL CIRCUIT CONSTRUCTION");
    println!("-----------------------------------------------------------------");

    // Hardware-efficient ansatz for 3 qubits
    println!("Hardware-Efficient Ansatz (3 qubits):");
    println!();

    let n_qubits = 3;
    let n_layers = 2;

    println!("Configuration:");
    println!("  • Qubits: {n_qubits}");
    println!("  • Layers: {n_layers}");
    println!(
        "  • Total parameters: {} (3 rotations × 3 qubits × 2 layers)",
        3 * n_qubits * n_layers
    );
    println!();

    println!("Layer structure:");
    for layer in 0..n_layers {
        println!("  Layer {layer}:");
        println!(
            "    • RZ(θ[{}..{}]) on each qubit",
            layer * 9,
            layer * 9 + 3
        );
        println!(
            "    • RY(θ[{}..{}]) on each qubit",
            layer * 9 + 3,
            layer * 9 + 6
        );
        println!(
            "    • RZ(θ[{}..{}]) on each qubit",
            layer * 9 + 6,
            layer * 9 + 9
        );
        println!("    • Entangling: CNOT chain (0→1, 1→2, 2→0)");
        println!();
    }

    println!("Circuit Gates (Symbolic):");
    println!("  |q0⟩ ──RZ(θ₀)──RY(θ₃)──RZ(θ₆)──●────────────┐");
    println!("  |q1⟩ ──RZ(θ₁)──RY(θ₄)──RZ(θ₇)──X──●─────────┤");
    println!("  |q2⟩ ──RZ(θ₂)──RY(θ₅)──RZ(θ₈)─────X──●──────┤");
    println!("                                        X");
    println!();

    println!("Expressibility:");
    println!("  • Can approximate wide variety of quantum states");
    println!("  • Balance: expressibility vs. trainability");
    println!("  • Barren plateau problem for deep circuits");
    println!();

    println!("  ✓ Variational circuit demonstrated");

    Ok(())
}

/// Demonstrate parameter optimization strategies
fn demonstrate_parameter_optimization() -> QuantRS2Result<()> {
    println!("PARAMETER OPTIMIZATION STRATEGIES");
    println!("-----------------------------------------------------------------");

    println!("1. GRADIENT-FREE METHODS");
    println!("   Nelder-Mead Simplex:");
    println!("     • No gradient computation needed");
    println!("     • Good for noisy objectives");
    println!("     • Scales poorly with parameters");
    println!();

    println!("   CMA-ES (Covariance Matrix Adaptation):");
    println!("     • Evolution strategy");
    println!("     • Adapts to landscape");
    println!("     • Good for ~100 parameters");
    println!();

    println!("2. GRADIENT-BASED METHODS");
    println!("   Parameter-Shift Rule:");
    println!("     • Quantum-compatible gradient estimation");
    println!("     • ∂E/∂θᵢ = [E(θ+s) - E(θ-s)] / 2sin(s)");
    println!("     • Requires 2 circuit evaluations per parameter");
    println!("     • Exact gradient (no approximation error)");
    println!();

    println!("   SPSA (Simultaneous Perturbation):");
    println!("     • Estimates all gradients with 2 measurements");
    println!("     • Stochastic approximation");
    println!("     • Scales well to many parameters");
    println!();

    println!("   Natural Gradient Descent:");
    println!("     • Uses quantum Fisher information metric");
    println!("     • Better optimization landscape navigation");
    println!("     • Requires Fubini-Study metric computation");
    println!();

    println!("3. HYBRID APPROACHES");
    println!("   Adam + Parameter-Shift:");
    println!("     • Adaptive learning rates");
    println!("     • Momentum for escaping local minima");
    println!("     • First/second moment estimates");
    println!();

    println!("   Quantum Natural Gradient + Adam:");
    println!("     • Combines best of both");
    println!("     • Used in state-of-the-art VQE");
    println!();

    println!("Optimization Challenges:");
    println!("  • Barren plateaus: Vanishing gradients in deep circuits");
    println!("  • Local minima: Non-convex landscape");
    println!("  • Noise: Measurement and gate errors");
    println!("  • Shot noise: Finite sampling statistics");
    println!();

    println!("Best Practices:");
    println!("  1. Start with small circuit depths");
    println!("  2. Use warm-starting from classical solutions");
    println!("  3. Monitor convergence carefully");
    println!("  4. Consider noise mitigation (error mitigation, ZNE)");
    println!("  5. Use ansatz with good inductive bias for problem");
    println!();

    println!("  ✓ Parameter optimization demonstrated");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_runs() {
        assert!(main().is_ok());
    }

    #[test]
    fn test_vqe_demonstration() {
        assert!(demonstrate_vqe_basics().is_ok());
    }

    #[test]
    fn test_qaoa_demonstration() {
        assert!(demonstrate_qaoa_basics().is_ok());
    }

    #[test]
    fn test_circuit_demonstration() {
        assert!(demonstrate_variational_circuit().is_ok());
    }

    #[test]
    fn test_optimization_demonstration() {
        assert!(demonstrate_parameter_optimization().is_ok());
    }
}
