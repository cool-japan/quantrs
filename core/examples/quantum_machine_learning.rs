//! Quantum Machine Learning Example
//!
//! This example demonstrates quantum machine learning concepts including
//! Quantum Neural Networks (QNNs), Quantum Kernels, and Quantum Feature Maps.
//!
//! These techniques leverage quantum computing for machine learning tasks
//! and may provide advantages for certain classification and regression problems.
//!
//! Run with: cargo run --example quantum_machine_learning

use quantrs2_core::{
    error::QuantRS2Result,
    gate::{multi, single, GateOp},
    qubit::QubitId,
};
use scirs2_core::Complex64;

fn main() {
    println!("=================================================================");
    println!("   QuantRS2-Core: Quantum Machine Learning");
    println!("=================================================================\n");

    // Demonstrate quantum feature maps
    demonstrate_feature_maps();
    println!();

    // Demonstrate quantum kernels
    demonstrate_quantum_kernels();
    println!();

    // Demonstrate quantum neural networks
    demonstrate_quantum_neural_networks();
    println!();

    // Demonstrate quantum-classical hybrid learning
    demonstrate_hybrid_learning();
    println!();

    println!("=================================================================");
    println!("   Example Complete!");
    println!("=================================================================");
}

/// Demonstrate quantum feature maps for encoding classical data
fn demonstrate_feature_maps() {
    println!("QUANTUM FEATURE MAPS");
    println!("-----------------------------------------------------------------");

    println!("Feature maps encode classical data into quantum states.");
    println!("Good feature maps enable quantum advantage for ML tasks.");
    println!();

    println!("1. PAULI-Z FEATURE MAP");
    println!("   Structure: U(x) = exp(-i Σᵢ xᵢ Zᵢ)");
    println!("   Properties:");
    println!("     • Simple, product state");
    println!("     • No entanglement");
    println!("     • Linear kernel");
    println!("     • Good for linearly separable data");
    println!();

    println!("   Example circuit (2 features, 2 qubits):");
    println!("     |q0⟩ ──RZ(2x₀)──");
    println!("     |q1⟩ ──RZ(2x₁)──");
    println!();

    println!("2. ZZ FEATURE MAP");
    println!("   Structure: U(x) = exp(-i Σᵢⱼ (π-xᵢ)(π-xⱼ) ZᵢZⱼ)");
    println!("   Properties:");
    println!("     • Entangling feature map");
    println!("     • Non-linear kernel");
    println!("     • Captures feature correlations");
    println!("     • Used in quantum SVM");
    println!();

    println!("   Example circuit (2 features, 2 qubits):");
    println!("     |q0⟩ ──H──RZ(2x₀)──●────────");
    println!("                        │");
    println!("     |q1⟩ ──H──RZ(2x₁)──●──ZZ(x₀,x₁)──");
    println!();

    println!("3. IQP (INSTANTANEOUS QUANTUM POLYNOMIAL) FEATURE MAP");
    println!("   Structure: Hadamards + diagonal unitaries");
    println!("   Properties:");
    println!("     • Exponentially hard to simulate classically");
    println!("     • Deep quantum kernel");
    println!("     • Potential quantum advantage");
    println!();

    println!("   Example circuit:");
    println!("     |q0⟩ ──H──RZ(x₀)──●────────H──RZ(x₀²)──");
    println!("                       │");
    println!("     |q1⟩ ──H──RZ(x₁)──●──ZZ──H──RZ(x₁²)──");
    println!();

    println!("4. HARDWARE-EFFICIENT FEATURE MAP");
    println!("   Structure: Native gates of quantum hardware");
    println!("   Properties:");
    println!("     • Minimal circuit depth");
    println!("     • Reduced error from gates");
    println!("     • Device-specific optimization");
    println!();

    println!("Feature Map Selection:");
    println!("  • Linear data → Pauli-Z feature map");
    println!("  • Non-linear data → ZZ or IQP feature map");
    println!("  • High-dimensional → Hardware-efficient");
    println!("  • Quantum advantage needs → IQP-type maps");
    println!();

    println!("  ✓ Quantum feature maps demonstrated");
}

/// Demonstrate quantum kernel methods
fn demonstrate_quantum_kernels() {
    println!("QUANTUM KERNEL METHODS");
    println!("-----------------------------------------------------------------");

    println!("Quantum kernels measure similarity between quantum states.");
    println!("Can be used with classical SVM for quantum-enhanced classification.");
    println!();

    println!("QUANTUM KERNEL DEFINITION");
    println!("  K(xᵢ, xⱼ) = |⟨Φ(xᵢ)|Φ(xⱼ)⟩|²");
    println!();
    println!("  where Φ(x) = U(x)|0⟩ is the feature-mapped state");
    println!();

    println!("COMPUTING QUANTUM KERNELS");
    println!("  1. Prepare state |Φ(xᵢ)⟩ = U(xᵢ)|0⟩");
    println!("  2. Apply inverse U†(xⱼ)");
    println!("  3. Measure probability of |0⟩");
    println!("  4. Kernel value = P(|0⟩)");
    println!();

    println!("Example Circuit:");
    println!("  |0⟩ ──U(xᵢ)──U†(xⱼ)──M");
    println!("                       ↓");
    println!("                    P(|0⟩) = K(xᵢ,xⱼ)");
    println!();

    println!("QUANTUM SVM WORKFLOW");
    println!("  Training:");
    println!("    1. Compute quantum kernel matrix K[i,j] for all training pairs");
    println!("    2. Solve SVM dual problem classically");
    println!("    3. Find support vectors and coefficients");
    println!();
    println!("  Prediction:");
    println!("    1. For new point x, compute K(x, xᵢ) for all support vectors");
    println!("    2. Classify: sign(Σᵢ αᵢyᵢK(x,xᵢ) + b)");
    println!();

    println!("QUANTUM KERNEL ADVANTAGES");
    println!("  • Exponentially large feature space");
    println!("  • No explicit feature computation");
    println!("  • Potential speedup for certain problems");
    println!("  • Natural handling of quantum data");
    println!();

    println!("PRACTICAL CONSIDERATIONS");
    println!("  • Kernel matrix computation: O(N²) quantum circuit evaluations");
    println!("  • Shot noise: Need many measurements per kernel element");
    println!("  • NISQ limitations: Shallow circuits, limited qubits");
    println!("  • Kernel design: Critical for performance");
    println!();

    println!("Example Applications:");
    println!("  • Image classification with quantum feature extraction");
    println!("  • Molecular property prediction");
    println!("  • Anomaly detection in high-dimensional data");
    println!("  • Quantum state classification");
    println!();

    println!("  ✓ Quantum kernel methods demonstrated");
}

/// Demonstrate quantum neural networks (QNNs)
fn demonstrate_quantum_neural_networks() {
    println!("QUANTUM NEURAL NETWORKS (QNNs)");
    println!("-----------------------------------------------------------------");

    println!("QNNs are parameterized quantum circuits trained via");
    println!("backpropagation to minimize a cost function.");
    println!();

    println!("QNN ARCHITECTURE");
    println!("  Components:");
    println!("    1. Input encoding layer (feature map)");
    println!("    2. Variational layers (trainable gates)");
    println!("    3. Measurement layer (observables)");
    println!();

    println!("Example 3-Qubit QNN:");
    println!("  Input Layer:");
    println!("    |0⟩ ──RY(x₀)──");
    println!("    |0⟩ ──RY(x₁)──");
    println!("    |0⟩ ──RY(x₂)──");
    println!();
    println!("  Variational Layer 1:");
    println!("    ──RY(θ₀)──RZ(θ₃)──●───────");
    println!("    ──RY(θ₁)──RZ(θ₄)──X──●────");
    println!("    ──RY(θ₂)──RZ(θ₅)─────X────");
    println!();
    println!("  Variational Layer 2:");
    println!("    ──RY(θ₆)──RZ(θ₉)──●───────");
    println!("    ──RY(θ₇)──RZ(θ₁₀)──X──●───");
    println!("    ──RY(θ₈)──RZ(θ₁₁)─────X───");
    println!();
    println!("  Measurement:");
    println!("    ──⟨Z⟩── → output");
    println!();

    println!("TRAINING QNNs");
    println!("  Forward Pass:");
    println!("    1. Encode input data x into quantum state");
    println!("    2. Apply parameterized quantum circuit U(θ)");
    println!("    3. Measure observable ⟨O⟩ = ⟨ψ(θ)|O|ψ(θ)⟩");
    println!("    4. Compute loss L(θ) = Σᵢ (⟨O⟩ᵢ - yᵢ)²");
    println!();

    println!("  Backward Pass (Parameter-Shift Rule):");
    println!("    ∂⟨O⟩/∂θⱼ = [⟨O⟩(θ+π/2) - ⟨O⟩(θ-π/2)] / 2");
    println!("    • Exact gradient (not finite difference)");
    println!("    • Requires 2 circuit evaluations per parameter");
    println!();

    println!("  Gradient Descent:");
    println!("    θ ← θ - η∇L(θ)");
    println!("    • Learning rate η typically 0.01-0.1");
    println!("    • Can use Adam, RMSprop, etc.");
    println!();

    println!("QNN TYPES");
    println!();
    println!("  1. Quantum Convolutional Neural Networks (QCNNs)");
    println!("     • Quantum analog of CNNs");
    println!("     • Pooling via measurement");
    println!("     • Good for quantum phase recognition");
    println!();

    println!("  2. Quantum Recurrent Neural Networks (QRNNs)");
    println!("     • Memory via quantum states");
    println!("     • Sequential data processing");
    println!("     • Temporal pattern recognition");
    println!();

    println!("  3. Quantum Graph Neural Networks (QGNNs)");
    println!("     • Process graph-structured data");
    println!("     • Node embeddings via quantum states");
    println!("     • Molecular property prediction");
    println!();

    println!("APPLICATIONS");
    println!("  • Classification: Handwritten digits, medical imaging");
    println!("  • Regression: Quantum chemistry, materials properties");
    println!("  • Generative: Quantum GANs for data generation");
    println!("  • Reinforcement Learning: Quantum policy optimization");
    println!();

    println!("CHALLENGES");
    println!("  • Barren plateaus: Vanishing gradients");
    println!("  • Training time: Many circuit evaluations needed");
    println!("  • Scalability: Limited qubits in NISQ era");
    println!("  • Noise: Measurement and gate errors");
    println!();

    println!("  ✓ Quantum neural networks demonstrated");
}

/// Demonstrate quantum-classical hybrid learning
fn demonstrate_hybrid_learning() {
    println!("QUANTUM-CLASSICAL HYBRID LEARNING");
    println!("-----------------------------------------------------------------");

    println!("Hybrid models combine quantum and classical processing");
    println!("to leverage strengths of both paradigms.");
    println!();

    println!("HYBRID ARCHITECTURES");
    println!();

    println!("1. QUANTUM LAYERS IN CLASSICAL NETWORKS");
    println!("   Classical → Quantum → Classical");
    println!();
    println!("   Input (classical)");
    println!("       ↓");
    println!("   Classical NN layers");
    println!("       ↓");
    println!("   Quantum circuit layer");
    println!("       ↓");
    println!("   Classical NN layers");
    println!("       ↓");
    println!("   Output (classical)");
    println!();
    println!("   Benefits:");
    println!("     • Use classical layers for feature extraction");
    println!("     • Quantum layer for non-linear transformations");
    println!("     • End-to-end differentiable");
    println!();

    println!("2. QUANTUM FEATURE EXTRACTION");
    println!("   Quantum → Classical ML");
    println!();
    println!("   Raw data");
    println!("       ↓");
    println!("   Quantum feature map");
    println!("       ↓");
    println!("   Quantum measurements");
    println!("       ↓");
    println!("   Classical SVM/RF/XGBoost");
    println!("       ↓");
    println!("   Predictions");
    println!();
    println!("   Benefits:");
    println!("     • Quantum provides rich features");
    println!("     • Classical ML is mature and fast");
    println!("     • Easy to deploy");
    println!();

    println!("3. QUANTUM ENSEMBLE METHODS");
    println!("   Multiple QNNs → Classical Aggregation");
    println!();
    println!("   QNN₁, QNN₂, ..., QNNₙ (different initializations)");
    println!("       ↓       ↓            ↓");
    println!("       └───────┴────────────┘");
    println!("              ↓");
    println!("         Voting/Averaging");
    println!("              ↓");
    println!("         Final prediction");
    println!();
    println!("   Benefits:");
    println!("     • Reduces variance");
    println!("     • More robust predictions");
    println!("     • Leverages multiple quantum runs");
    println!();

    println!("TRAINING STRATEGIES");
    println!();

    println!("  Transfer Learning:");
    println!("    1. Pre-train QNN on large dataset");
    println!("    2. Fine-tune on specific task");
    println!("    3. Faster convergence, better generalization");
    println!();

    println!("  Co-training:");
    println!("    1. Train quantum and classical parts jointly");
    println!("    2. Gradient flows through entire hybrid model");
    println!("    3. Optimize for end-to-end performance");
    println!();

    println!("  Multi-task Learning:");
    println!("    1. Shared quantum encoder");
    println!("    2. Task-specific classical decoders");
    println!("    3. Learn general quantum features");
    println!();

    println!("PRACTICAL WORKFLOW");
    println!("  1. Data preprocessing (classical)");
    println!("  2. Encode batch of data (quantum)");
    println!("  3. Forward pass through hybrid model");
    println!("  4. Compute loss (classical)");
    println!("  5. Backprop through classical layers");
    println!("  6. Parameter-shift for quantum layers");
    println!("  7. Update all parameters");
    println!("  8. Repeat until convergence");
    println!();

    println!("ADVANTAGES OF HYBRID APPROACH");
    println!("  • Best of both worlds: quantum + classical");
    println!("  • Quantum handles hard sub-problems");
    println!("  • Classical provides stability and efficiency");
    println!("  • Easier to scale than pure quantum");
    println!("  • More suitable for NISQ devices");
    println!();

    println!("REAL-WORLD APPLICATIONS");
    println!("  • Drug discovery: Molecular property prediction");
    println!("  • Finance: Portfolio optimization with QML");
    println!("  • Computer vision: Quantum-enhanced image classification");
    println!("  • NLP: Quantum text embedding for semantic search");
    println!();

    println!("  ✓ Hybrid quantum-classical learning demonstrated");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_runs() {
        main();
    }

    #[test]
    fn test_feature_maps() {
        demonstrate_feature_maps();
    }

    #[test]
    fn test_quantum_kernels() {
        demonstrate_quantum_kernels();
    }

    #[test]
    fn test_quantum_neural_networks() {
        demonstrate_quantum_neural_networks();
    }

    #[test]
    fn test_hybrid_learning() {
        demonstrate_hybrid_learning();
    }
}
