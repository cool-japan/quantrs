//! Advanced Quantum Machine Learning Features Demonstration
//!
//! This example showcases the cutting-edge quantum machine learning capabilities
//! added to QuantRS2-Core, including:
//!
//! 1. **Quantum Transformers**: Attention-based quantum neural networks
//! 2. **Quantum Reservoir Computing**: Time-series processing with quantum dynamics
//! 3. **Quantum Memory Networks**: Memory-augmented quantum learning
//! 4. **Quantum Contrastive Learning**: Self-supervised representation learning
//! 5. **Quantum Meta-Learning**: Few-shot learning with quantum circuits
//!
//! # Scientific Background
//!
//! These implementations represent state-of-the-art quantum ML research:
//!
//! - **Quantum Transformers** leverage quantum attention mechanisms for enhanced
//!   sequence modeling beyond classical transformers
//! - **Quantum Reservoir Computing** exploits natural quantum dynamics for
//!   computational memory without training the reservoir
//! - **Quantum Memory Networks** provide external quantum memory for complex
//!   reasoning tasks requiring long-term dependencies
//! - **Quantum Contrastive Learning** enables unsupervised quantum representation
//!   learning using quantum fidelity measures
//! - **Quantum Meta-Learning** allows rapid adaptation to new tasks with minimal
//!   quantum training data
//!
//! # Performance Insights
//!
//! These quantum ML techniques offer potential advantages:
//! - Exponentially large feature spaces (quantum transformers, contrastive learning)
//! - Intrinsic quantum memory effects (reservoir computing, memory networks)
//! - Natural few-shot learning from quantum interference (meta-learning)
//! - Hardware-efficient implementations on NISQ devices
//!
//! # Usage
//!
//! ```bash
//! cargo run --example advanced_qml_features --release
//! ```

use quantrs2_core::qml::{
    QuantumAttention, QuantumAugmentation, QuantumContrastiveConfig, QuantumContrastiveLearner,
    QuantumMAML, QuantumMemoryConfig, QuantumMemoryNetwork, QuantumMetaLearningConfig,
    QuantumReservoirComputer, QuantumReservoirConfig, QuantumTask, QuantumTransformer,
    QuantumTransformerConfig,
};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║     Advanced Quantum Machine Learning Features - QuantRS2-Core          ║");
    println!("║                                                                          ║");
    println!("║  Demonstrating cutting-edge quantum ML algorithms                        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Part 1: Quantum Transformers with Attention
    println!("═══════════════════════════════════════════════════════════════════");
    println!("Part 1: Quantum Transformers with Attention Mechanisms");
    println!("═══════════════════════════════════════════════════════════════════\n");

    demo_quantum_transformer()?;

    // Part 2: Quantum Reservoir Computing
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("Part 2: Quantum Reservoir Computing for Time-Series");
    println!("═══════════════════════════════════════════════════════════════════\n");

    demo_quantum_reservoir()?;

    // Part 3: Quantum Memory Networks
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("Part 3: Quantum Memory Networks");
    println!("═══════════════════════════════════════════════════════════════════\n");

    demo_quantum_memory_network()?;

    // Part 4: Quantum Contrastive Learning
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("Part 4: Quantum Contrastive Learning");
    println!("═══════════════════════════════════════════════════════════════════\n");

    demo_quantum_contrastive()?;

    // Part 5: Quantum Meta-Learning
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("Part 5: Quantum Meta-Learning (MAML)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    demo_quantum_meta_learning()?;

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("Summary: Advanced QML Capabilities");
    println!("═══════════════════════════════════════════════════════════════════\n");

    print_summary();

    Ok(())
}

/// Demonstrate quantum transformer with attention
fn demo_quantum_transformer() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔷 Quantum Transformers enable attention-based sequence processing");
    println!("   on quantum computers, leveraging quantum superposition for");
    println!("   enhanced representation learning.\n");

    let config = QuantumTransformerConfig {
        num_qubits: 4,
        num_heads: 2,
        head_dim: 2,
        num_layers: 2,
        ffn_dim: 8,
        dropout_rate: 0.1,
        max_seq_length: 16,
        use_layer_norm: true,
    };

    println!("Configuration:");
    println!(
        "  • Qubits: {} (Hilbert space dimension: 2^{} = {})",
        config.num_qubits,
        config.num_qubits,
        1 << config.num_qubits
    );
    println!("  • Attention heads: {}", config.num_heads);
    println!("  • Transformer layers: {}", config.num_layers);
    println!("  • Feed-forward dimension: {}", config.ffn_dim);
    println!("  • Layer normalization: {}\n", config.use_layer_norm);

    let transformer = QuantumTransformer::new(config)?;

    // Create test sequence of quantum states
    println!("Creating test sequence of 3 quantum states...");
    let mut input = Array2::zeros((3, 4));
    for i in 0..3 {
        for j in 0..4 {
            input[[i, j]] = Complex64::new((i + j) as f64 * 0.1, 0.0);
        }
    }

    // Process through transformer
    println!("Processing through quantum transformer...");
    let output = transformer.forward(&input)?;

    println!("✓ Successfully processed sequence");
    println!(
        "  Output shape: {} states × {} qubits",
        output.shape()[0],
        output.shape()[1]
    );
    println!("\n  Key Features:");
    println!("    → Multi-head quantum attention for pattern recognition");
    println!("    → Quantum positional encoding preserves sequence information");
    println!("    → Feed-forward quantum networks for non-linear transformations");
    println!("    → Layer normalization maintains quantum state properties");

    println!("\n  Applications:");
    println!("    • Quantum natural language processing");
    println!("    • Quantum time-series prediction");
    println!("    • Quantum molecular sequence analysis");

    Ok(())
}

/// Demonstrate quantum reservoir computing
fn demo_quantum_reservoir() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔷 Quantum Reservoir Computing leverages the natural dynamics of");
    println!("   quantum systems as computational resources without training");
    println!("   the reservoir itself.\n");

    let config = QuantumReservoirConfig {
        num_qubits: 6,
        depth: 8,
        spectral_radius: 0.9,
        input_scaling: 1.0,
        leak_rate: 0.3,
        use_entanglement: true,
        seed: Some(42),
    };

    println!("Configuration:");
    println!("  • Reservoir qubits: {}", config.num_qubits);
    println!("  • Reservoir depth: {} (circuit layers)", config.depth);
    println!(
        "  • Spectral radius: {:.2} (controls dynamics)",
        config.spectral_radius
    );
    println!(
        "  • Leak rate: {:.2} (memory fading factor)",
        config.leak_rate
    );
    println!("  • Entanglement: {}\n", config.use_entanglement);

    let mut qrc = QuantumReservoirComputer::new(config, 2)?;

    // Create test time-series
    println!("Creating test time-series (10 time steps)...");
    let inputs = Array2::from_shape_fn((10, 6), |(i, j)| (i + j) as f64 * 0.1);

    // Process through reservoir
    println!("Processing through quantum reservoir...");
    let outputs = qrc.process_sequence(&inputs)?;

    println!("✓ Successfully processed time-series");
    println!(
        "  Output shape: {} steps × {} outputs",
        outputs.shape()[0],
        outputs.shape()[1]
    );
    println!("\n  Key Features:");
    println!("    → Fixed random quantum circuit (no training needed)");
    println!("    → Quantum echo state property for temporal patterns");
    println!("    → Pauli expectation features (3 per qubit)");
    println!("    → Linear readout layer (trainable)");

    println!("\n  Echo State Property:");
    println!("    The quantum reservoir projects input sequences into a high-");
    println!("    dimensional Hilbert space where temporal patterns become");
    println!("    linearly separable for the readout layer.");

    println!("\n  Applications:");
    println!("    • Quantum time-series forecasting");
    println!("    • Chaotic system prediction");
    println!("    • Real-time quantum signal processing");

    Ok(())
}

/// Demonstrate quantum memory networks
fn demo_quantum_memory_network() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔷 Quantum Memory Networks augment quantum neural networks with");
    println!("   external quantum memory for complex reasoning tasks.\n");

    let config = QuantumMemoryConfig {
        memory_slots: 32,
        qubits_per_slot: 3,
        controller_size: 16,
        num_read_heads: 1,
        num_write_heads: 1,
        init_strategy: quantrs2_core::qml::MemoryInitStrategy::Zero,
    };

    println!("Configuration:");
    println!("  • Memory slots: {}", config.memory_slots);
    println!(
        "  • Qubits per slot: {} (2^{} = {} dimensional states)",
        config.qubits_per_slot,
        config.qubits_per_slot,
        1 << config.qubits_per_slot
    );
    println!("  • Controller size: {} neurons", config.controller_size);
    println!(
        "  • Read/Write heads: {}/{}\n",
        config.num_read_heads, config.num_write_heads
    );

    let mut network = QuantumMemoryNetwork::new(8, config);

    // Process a sequence
    println!("Processing input sequence...");
    let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    let outputs = network.step(&input)?;

    println!("✓ Successfully performed memory operations");
    println!("  Read outputs: {} quantum states", outputs.len());
    println!("  State dimension: {}", outputs[0].len());

    println!("\n  Memory Operations:");
    println!("    1. Controller processes input → generates attention weights");
    println!("    2. Read: Weighted sum of memory slots using attention");
    println!("    3. Write: Erase-then-add operations on memory");
    println!("    4. Usage tracking for least-used slot allocation");

    println!("\n  Architecture:");
    println!("    • Attention-based addressing (quantum fidelity metric)");
    println!("    • Differentiable read/write operations");
    println!("    • Quantum state normalization preservation");
    println!("    • Neural Turing Machine-inspired design");

    println!("\n  Applications:");
    println!("    • Question answering with quantum reasoning");
    println!("    • Quantum program synthesis");
    println!("    • Long-term quantum dependency learning");

    Ok(())
}

/// Demonstrate quantum contrastive learning
fn demo_quantum_contrastive() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔷 Quantum Contrastive Learning enables self-supervised quantum");
    println!("   representation learning without labeled data.\n");

    let config = QuantumContrastiveConfig {
        num_qubits: 3,
        encoder_depth: 3,
        temperature: 0.5,
        momentum: 0.999,
        batch_size: 4,
        num_views: 2,
    };

    println!("Configuration:");
    println!("  • Encoder qubits: {}", config.num_qubits);
    println!("  • Encoder depth: {} layers", config.encoder_depth);
    println!(
        "  • Temperature: {:.2} (contrastive loss scaling)",
        config.temperature
    );
    println!(
        "  • Momentum: {:.4} (for momentum encoder)",
        config.momentum
    );
    println!("  • Batch size: {}\n", config.batch_size);

    let mut learner = QuantumContrastiveLearner::new(config);

    // Create batch of quantum states
    println!("Creating batch of quantum states...");
    let mut batch = Vec::new();
    for i in 0..4 {
        let state = Array1::from_vec(vec![
            Complex64::new(((i + 1) as f64 * 0.3).cos(), 0.0),
            Complex64::new(((i + 1) as f64 * 0.3).sin(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        batch.push(state);
    }

    println!("Training one contrastive learning step...");
    let loss = learner.train_step(&batch, 0.01)?;

    println!("✓ Successfully performed contrastive learning");
    println!("  Loss: {loss:.6}");

    println!("\n  Contrastive Learning Pipeline:");
    println!("    1. Quantum data augmentation (rotations, noise)");
    println!("    2. Encode multiple views through quantum circuits");
    println!("    3. Maximize agreement between views of same state");
    println!("    4. Minimize agreement with different states");
    println!("    5. NT-Xent loss with quantum fidelity metric");

    println!("\n  Quantum Augmentations:");
    println!("    • Random unitary rotations");
    println!("    • Depolarizing noise");
    println!("    • Amplitude/phase damping");
    println!("    • Random Pauli gates");

    println!("\n  Applications:");
    println!("    • Unsupervised quantum feature learning");
    println!("    • Robust quantum representations for NISQ devices");
    println!("    • Pre-training for downstream quantum tasks");

    Ok(())
}

/// Demonstrate quantum meta-learning
fn demo_quantum_meta_learning() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔷 Quantum Meta-Learning (MAML) enables rapid adaptation to new");
    println!("   tasks with minimal quantum training data.\n");

    let config = QuantumMetaLearningConfig {
        num_qubits: 2,
        circuit_depth: 2,
        inner_lr: 0.01,
        outer_lr: 0.001,
        inner_steps: 3,
        n_support: 2,
        n_query: 4,
        n_way: 2,
        meta_batch_size: 2,
    };

    println!("Configuration:");
    println!("  • Circuit qubits: {}", config.num_qubits);
    println!("  • Circuit depth: {} layers", config.circuit_depth);
    println!(
        "  • Inner/Outer learning rate: {}/{}",
        config.inner_lr, config.outer_lr
    );
    println!("  • Inner adaptation steps: {}", config.inner_steps);
    println!(
        "  • {}-way {}-shot classification\n",
        config.n_way, config.n_support
    );

    let mut maml = QuantumMAML::new(config.clone());

    // Create random task
    println!("Creating random few-shot learning task...");
    let task = QuantumTask::random(
        config.num_qubits,
        config.n_way,
        config.n_support,
        config.n_query,
    );

    println!(
        "  Support set: {} examples ({} per class)",
        task.support_states.len(),
        config.n_support
    );
    println!(
        "  Query set: {} examples ({} per class)",
        task.query_states.len(),
        config.n_query
    );

    // Adapt to task
    println!("\nAdapting quantum circuit to new task...");
    let adapted_model = maml.adapt(&task)?;

    println!(
        "✓ Successfully adapted to task in {} gradient steps",
        config.inner_steps
    );

    // Evaluate
    let accuracy = maml.evaluate(&task)?;
    println!("  Adaptation accuracy: {:.1}%", accuracy * 100.0);

    println!("\n  MAML Training Loop:");
    println!("    1. Sample batch of tasks from task distribution");
    println!("    2. For each task:");
    println!("       a) Clone meta-parameters θ");
    println!("       b) Adapt: θ' = θ - α∇L_support(θ)  [inner loop]");
    println!("       c) Compute loss on query set: L_query(θ')");
    println!("    3. Meta-update: θ = θ - β∇Σ L_query(θ')  [outer loop]");

    println!("\n  Key Advantages:");
    println!("    • Rapid task adaptation (few gradient steps)");
    println!("    • Learns good initialization for quantum parameters");
    println!("    • Efficient use of limited quantum data");
    println!("    • Task-agnostic meta-learning framework");

    println!("\n  Applications:");
    println!("    • Few-shot quantum classification");
    println!("    • Fast quantum state tomography");
    println!("    • Adaptive quantum control");
    println!("    • Drug discovery with limited molecular data");

    Ok(())
}

/// Print summary of all advanced QML capabilities
fn print_summary() {
    println!("🌟 QuantRS2-Core now includes 5 cutting-edge QML algorithms:");
    println!();
    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!("│ 1. Quantum Transformers                                          │");
    println!("│    ✓ Multi-head quantum attention mechanisms                     │");
    println!("│    ✓ Quantum positional encoding for sequences                   │");
    println!("│    ✓ Feed-forward quantum networks with layer norm               │");
    println!("│    → Applications: NLP, time-series, molecular sequences         │");
    println!("├──────────────────────────────────────────────────────────────────┤");
    println!("│ 2. Quantum Reservoir Computing                                   │");
    println!("│    ✓ Fixed random quantum circuits (no training)                 │");
    println!("│    ✓ Quantum echo state property for temporal memory             │");
    println!("│    ✓ Pauli expectation feature extraction                        │");
    println!("│    → Applications: Time-series forecasting, signal processing    │");
    println!("├──────────────────────────────────────────────────────────────────┤");
    println!("│ 3. Quantum Memory Networks                                       │");
    println!("│    ✓ External quantum memory with addressable slots              │");
    println!("│    ✓ Attention-based read/write operations                       │");
    println!("│    ✓ Neural Turing Machine architecture                          │");
    println!("│    → Applications: Q&A, reasoning, program synthesis             │");
    println!("├──────────────────────────────────────────────────────────────────┤");
    println!("│ 4. Quantum Contrastive Learning                                  │");
    println!("│    ✓ Self-supervised representation learning                     │");
    println!("│    ✓ Quantum data augmentation strategies                        │");
    println!("│    ✓ NT-Xent loss with quantum fidelity                          │");
    println!("│    → Applications: Unsupervised learning, robust features        │");
    println!("├──────────────────────────────────────────────────────────────────┤");
    println!("│ 5. Quantum Meta-Learning (MAML & Reptile)                        │");
    println!("│    ✓ Model-agnostic few-shot learning                            │");
    println!("│    ✓ Rapid task adaptation with minimal data                     │");
    println!("│    ✓ Bi-level optimization for quantum circuits                  │");
    println!("│    → Applications: Few-shot classification, drug discovery       │");
    println!("└──────────────────────────────────────────────────────────────────┘");
    println!();
    println!("📊 Performance Characteristics:");
    println!("   • Hilbert space scaling: Exponential in qubit count");
    println!("   • Quantum memory: Intrinsic quantum dynamics");
    println!("   • Few-shot learning: Quantum interference patterns");
    println!("   • NISQ-friendly: Designed for current quantum hardware");
    println!();
    println!("🔬 Research Impact:");
    println!("   These implementations represent state-of-the-art quantum ML");
    println!("   research, providing researchers and developers");
    println!("   with production-ready quantum learning algorithms.");
    println!();
    println!("📚 For detailed documentation and theory:");
    println!("   See individual module documentation in src/qml/");
    println!();
    println!("✨ QuantRS2: Pushing the boundaries of quantum machine learning!");
}
