//! Quantum Convolutional Neural Network demonstration

use quantrs2_ml::error::MLError;
use quantrs2_ml::qcnn::{QuantumImageEncoder, QCNN};
use scirs2_core::Complex64 as Complex;

fn main() -> Result<(), MLError> {
    println!("Quantum Convolutional Neural Network Demo");
    println!("=========================================");

    // Create a simple 4x4 image
    let image = vec![
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.5, 0.6, 0.7, 0.8],
        vec![0.8, 0.7, 0.6, 0.5],
        vec![0.4, 0.3, 0.2, 0.1],
    ];

    println!("\nInput image (4x4):");
    for row in &image {
        for &val in row {
            print!("{val:.2} ");
        }
        println!();
    }

    // Create quantum image encoder
    let encoder = QuantumImageEncoder::new(4, 4, 4); // 4 qubits for 16 pixels
    let encoded_state = encoder.encode(&image)?;

    println!("\nEncoded quantum state (first 8 amplitudes):");
    for i in 0..8.min(encoded_state.len()) {
        let amp = encoded_state[i];
        println!(
            "  |{:04b}⟩: {:.4} + {:.4}i (magnitude: {:.4})",
            i,
            amp.re,
            amp.im,
            amp.norm()
        );
    }

    // Create QCNN with:
    // - 8 qubits total
    // - 2 convolutional layers: (4 qubits, stride 2) and (2 qubits, stride 1)
    // - Pooling sizes: 2 and 2
    // - 4 parameters in fully connected layer
    let qcnn = QCNN::new(8, vec![(4, 2), (2, 1)], vec![2, 2], 4)?;

    println!("\nQCNN Architecture:");
    println!("  Input: 8 qubits");
    println!("  Conv Layer 1: 4-qubit filter, stride 2");
    println!("  Pooling 1: size 2");
    println!("  Conv Layer 2: 2-qubit filter, stride 1");
    println!("  Pooling 2: size 2");
    println!("  FC Layer: 4 parameters");

    // Get trainable parameters
    let params = qcnn.get_parameters();
    println!("\nTotal trainable parameters: {}", params.len());

    // Demonstrate forward pass with a simple input
    let input_size = 1 << 8; // 2^8 = 256
    let mut input_state = vec![Complex::new(0.0, 0.0); input_size];
    // Initialize with a simple superposition
    for i in 0..16 {
        input_state[i] = Complex::new(1.0 / 4.0, 0.0);
    }

    println!("\nPerforming forward pass...");
    let output = qcnn.forward(&input_state)?;

    println!("\nOutput state statistics:");
    let output_norm: f64 = output.iter().map(scirs2_core::Complex::norm_sqr).sum();
    println!("  Total probability: {output_norm:.6}");

    // Find the most probable outcomes
    let mut outcomes: Vec<(usize, f64)> = output
        .iter()
        .enumerate()
        .map(|(i, c)| (i, c.norm_sqr()))
        .filter(|(_, p)| *p > 1e-6)
        .collect();
    outcomes.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).expect(
            "Failed to compare outcome probabilities (NaN encountered in QCNN forward pass)",
        )
    });

    println!("\nTop 5 measurement outcomes:");
    for (idx, (state, prob)) in outcomes.iter().take(5).enumerate() {
        println!("  {}. |{:08b}⟩: {:.4}", idx + 1, state, prob);
    }

    // Demonstrate gradient computation
    println!("\nDemonstrating gradient computation...");
    let target_state = vec![Complex::new(0.0, 0.0); output.len()];

    // Simple loss function: negative fidelity with target
    let loss_fn = |output: &Vec<Complex>, target: &Vec<Complex>| -> f64 {
        let fidelity: Complex = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| o.conj() * t)
            .sum();
        -fidelity.norm()
    };

    // Note: Gradient computation is computationally intensive for demonstration
    println!(
        "  (Skipping actual gradient computation for demo - would compute {} gradients)",
        params.len()
    );

    // Show how QCNN processes image-like data
    println!("\nQuantum Image Processing Pipeline:");
    println!("  1. Classical image → Quantum state encoding");
    println!("  2. Quantum convolutions extract local features");
    println!("  3. Quantum pooling reduces dimensionality");
    println!("  4. Fully connected layer produces final quantum state");
    println!("  5. Measurement extracts classical predictions");

    println!("\nPotential Applications:");
    println!("  - Quantum image classification");
    println!("  - Quantum feature extraction");
    println!("  - Hybrid classical-quantum computer vision");
    println!("  - Quantum data compression");

    Ok(())
}
