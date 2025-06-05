//! Barren Plateau Detection Demonstration

use quantrs2_circuit::prelude::*;
use quantrs2_ml::barren_plateau::{
    BarrenPlateauDetector, BarrenPlateauMitigation, VarianceScalingAnalyzer,
};
use quantrs2_ml::error::MLError;
use std::f64::consts::PI;

fn main() -> Result<(), MLError> {
    println!("Barren Plateau Detection Demo");
    println!("============================\n");

    // Example 1: Analyze a deep parameterized circuit
    println!("1. Analyzing a Deep Parameterized Circuit");
    println!("-----------------------------------------");

    let detector = BarrenPlateauDetector::new(50);

    // Create a circuit builder for a 6-qubit system with 4 layers
    let num_qubits = 6;
    let num_layers = 4;
    let params_per_layer = num_qubits * 3; // 3 rotation gates per qubit
    let total_params = num_layers * params_per_layer;

    let circuit_builder = move |params: &[f64]| -> Result<Circuit<6>, MLError> {
        let mut circuit = Circuit::<6>::new();

        for layer in 0..num_layers {
            let layer_start = layer * params_per_layer;

            // Single-qubit rotations
            for q in 0..num_qubits {
                let idx = layer_start + q * 3;
                if idx + 2 < params.len() {
                    circuit.rx(q, params[idx])?;
                    circuit.ry(q, params[idx + 1])?;
                    circuit.rz(q, params[idx + 2])?;
                }
            }

            // Entangling gates
            for q in 0..num_qubits - 1 {
                circuit.cnot(q, q + 1)?;
            }
            if num_qubits > 2 {
                circuit.cnot(num_qubits - 1, 0)?; // Circular connectivity
            }
        }

        Ok(circuit)
    };

    println!("Circuit configuration:");
    println!("  - Qubits: {}", num_qubits);
    println!("  - Layers: {}", num_layers);
    println!("  - Total parameters: {}", total_params);
    println!("  - Entanglement: Circular connectivity");

    println!("\nAnalyzing for barren plateaus...");
    let analysis = detector.analyze_circuit(circuit_builder, total_params, num_layers)?;

    println!("\nAnalysis Results:");
    println!(
        "  - Overall gradient variance: {:.2e}",
        analysis.overall_variance
    );
    println!("  - Is in barren plateau: {}", analysis.is_barren);

    println!("\n  Layer-wise variances:");
    for (i, &var) in analysis.layer_variances.iter().enumerate() {
        println!(
            "    Layer {}: {:.2e} {}",
            i + 1,
            var,
            if analysis.problematic_layers.contains(&i) {
                "⚠️  (problematic)"
            } else {
                "✓"
            }
        );
    }

    if !analysis.mitigation_strategies.is_empty() {
        println!("\n  Suggested mitigation strategies:");
        for (i, strategy) in analysis.mitigation_strategies.iter().enumerate() {
            println!("    {}. {}", i + 1, strategy);
        }
    }

    // Example 2: Variance scaling analysis
    println!("\n\n2. Variance Scaling with System Size");
    println!("------------------------------------");

    let scaling_analyzer = VarianceScalingAnalyzer::new(30);
    let scaling_results = scaling_analyzer.analyze_scaling(2, 10, 2)?;

    println!("Gradient variance vs number of qubits:");
    println!("Qubits | Variance    | Log(Variance)");
    println!("-------|-------------|---------------");

    for (n, var) in &scaling_results {
        println!("{:6} | {:.4e} | {:.2}", n, var, var.ln());
    }

    // Check for exponential decay
    if scaling_results.len() >= 2 {
        let first = scaling_results[0].1.ln();
        let last = scaling_results.last().unwrap().1.ln();
        let decay_rate = (last - first) / (scaling_results.len() as f64 - 1.0);

        println!("\nExponential decay rate: {:.3} per qubit", decay_rate);
        if decay_rate < -0.5 {
            println!("⚠️  Warning: Exponential gradient suppression detected!");
        }
    }

    // Example 3: Mitigation strategies
    println!("\n\n3. Barren Plateau Mitigation");
    println!("----------------------------");

    let mitigation = BarrenPlateauMitigation::new(100, 0.01);

    // Smart initialization
    let smart_params = mitigation.smart_initialization(total_params);
    println!("\nSmart initialization (first 10 parameters):");
    for (i, &p) in smart_params.iter().take(10).enumerate() {
        println!("  param[{}] = {:.4}", i, p);
    }

    // Layer-wise pre-training
    println!("\nPerforming layer-wise pre-training...");
    let pretrained_params =
        mitigation.layer_wise_pretrain(circuit_builder, total_params, num_layers)?;

    println!("Pre-training complete!");
    println!("  - Parameters initialized layer by layer");
    println!("  - Each layer trained independently before combining");

    // Example 4: Architecture recommendations
    println!("\n\n4. Architecture Recommendations");
    println!("-------------------------------");

    println!("\nBased on the analysis, here are general recommendations:");
    println!("\n  For avoiding barren plateaus:");
    println!("  • Use shallow circuits (depth ≤ number of qubits)");
    println!("  • Limit entanglement to nearest neighbors");
    println!("  • Initialize parameters near zero (σ ≈ 0.1)");
    println!("  • Use hardware-efficient ansätze");
    println!("  • Consider local cost functions for large systems");

    println!("\n  For deep circuits:");
    println!("  • Implement identity blocks or skip connections");
    println!("  • Use parameter sharing across layers");
    println!("  • Apply layer-wise or block-wise training");
    println!("  • Monitor gradient norms during training");

    println!("\n  Alternative approaches:");
    println!("  • Quantum Natural Gradient descent");
    println!("  • Parameter shift rules with rescaling");
    println!("  • Adaptive circuit structure");
    println!("  • Classical pre-training with tensor networks");

    Ok(())
}
