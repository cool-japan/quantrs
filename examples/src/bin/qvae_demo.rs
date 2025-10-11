//! Quantum Variational Autoencoder Demonstration
//!
//! This example demonstrates quantum data compression using a QVAE.

use quantrs2_ml::vae::{ClassicalAutoencoder, HybridAutoencoder, QVAE};
use scirs2_core::Complex64 as Complex;
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Quantum Variational Autoencoder Demo ===\n");

    // 1. Create a QVAE for 4-qubit data compressed to 2-qubit latent space
    println!("1. Creating QVAE: 4 qubits → 2 qubits (latent)");
    let mut qvae = QVAE::new(4, 2, 0)?;
    println!("   Total qubits required: {}", qvae.total_qubits());
    println!("   Compression ratio: 4:2 = 2:1");

    // 2. Show parameter structure
    println!("\n2. Parameter Structure:");
    println!("   Encoder parameters: {}", qvae.encoder_params.len());
    println!("   Decoder parameters: {}", qvae.decoder_params.len());
    println!("   Total parameters: {}", qvae.get_parameters().len());

    // 3. Build the circuit
    println!("\n3. Building Autoencoder Circuit:");
    let circuit = qvae.build_circuit::<10>()?;
    println!("   Circuit created successfully");

    // 4. Test reconstruction fidelity
    println!("\n4. Testing Reconstruction Fidelity:");

    // Create a test quantum state (normalized)
    let test_state = vec![
        Complex::new(0.5, 0.0),
        Complex::new(0.5, 0.0),
        Complex::new(0.5, 0.0),
        Complex::new(0.5, 0.0),
    ];

    // Perfect reconstruction test
    let fidelity = qvae.reconstruction_fidelity(&test_state, &test_state)?;
    println!("   Perfect reconstruction fidelity: {fidelity:.6}");

    // Imperfect reconstruction test
    let noisy_state = vec![
        Complex::new(0.48, 0.0),
        Complex::new(0.52, 0.0),
        Complex::new(0.49, 0.0),
        Complex::new(0.51, 0.0),
    ];
    let noisy_fidelity = qvae.reconstruction_fidelity(&test_state, &noisy_state)?;
    println!("   Noisy reconstruction fidelity: {noisy_fidelity:.6}");

    // 5. Classical autoencoder comparison
    println!("\n5. Classical Autoencoder Comparison:");
    let classical_ae = ClassicalAutoencoder::new(8, 3);

    let classical_input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let latent_rep = classical_ae.encode(&classical_input);
    let reconstructed = classical_ae.decode(&latent_rep);

    println!("   Input dimension: {}", classical_input.len());
    println!("   Latent dimension: {}", latent_rep.len());
    println!(
        "   Compression ratio: {}:{} = {:.1}:1",
        classical_input.len(),
        latent_rep.len(),
        classical_input.len() as f64 / latent_rep.len() as f64
    );

    // Compute reconstruction error
    let mse: f64 = classical_input
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / classical_input.len() as f64;
    println!("   Mean squared error: {mse:.6}");

    // 6. Hybrid quantum-classical autoencoder
    println!("\n6. Hybrid Quantum-Classical Autoencoder:");
    let hybrid = HybridAutoencoder::new(3, 2, 4)?;
    println!("   Quantum encoder: 3 → 2 qubits");
    println!("   Classical decoder: 4 → 4 dimensions");

    // 7. Parameter optimization simulation
    println!("\n7. Simulating Parameter Optimization:");
    let initial_params = qvae.get_parameters();
    println!(
        "   Initial parameter norm: {:.6}",
        initial_params.iter().map(|p| p * p).sum::<f64>().sqrt()
    );

    // Simulate optimization step
    let mut optimized_params = initial_params.clone();
    for (i, p) in optimized_params.iter_mut().enumerate() {
        // Simple gradient descent step simulation
        *p -= 0.01 * (i as f64 / initial_params.len() as f64 - 0.5);
    }

    qvae.set_parameters(&optimized_params)?;
    println!(
        "   Updated parameter norm: {:.6}",
        optimized_params.iter().map(|p| p * p).sum::<f64>().sqrt()
    );

    // 8. Loss computation
    println!("\n8. Loss Function Evaluation:");
    let test_inputs = vec![test_state; 5];
    let lambda = 0.01; // Regularization parameter
    let loss = qvae.compute_loss(&test_inputs, lambda)?;
    println!("   Average loss (with L2 regularization λ={lambda}): {loss:.6}");

    // 9. Compression analysis
    println!("\n9. Quantum Data Compression Analysis:");
    let data_bits = 1 << qvae.num_data_qubits;
    let latent_bits = 1 << qvae.num_latent_qubits;
    println!("   Data space dimension: {data_bits}");
    println!("   Latent space dimension: {latent_bits}");
    println!("   Compression factor: {}x", data_bits / latent_bits);
    println!(
        "   Information retention: {:.1}%",
        (f64::from(latent_bits) / f64::from(data_bits)) * 100.0
    );

    // 10. Quantum advantage discussion
    println!("\n10. Quantum Advantage:");
    println!("   - Exponential compression in qubit count");
    println!("   - Quantum superposition enables richer latent representations");
    println!("   - Entanglement captures non-classical correlations");
    println!("   - Parameter count scales polynomially with qubits");

    println!("\n✅ QVAE demonstration completed successfully!");

    Ok(())
}
