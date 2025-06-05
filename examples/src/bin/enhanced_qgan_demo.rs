//! Enhanced Quantum GAN Demonstration
//!
//! This example demonstrates advanced quantum GAN implementations including
//! Wasserstein QGAN and Conditional QGAN.

use ndarray::{Array1, Array2};
use quantrs2_ml::enhanced_gan::{
    ConditionalQGAN, EnhancedQuantumDiscriminator, EnhancedQuantumGenerator, WassersteinQGAN,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Enhanced Quantum GAN Demo ===\n");

    // 1. Basic Enhanced Generator
    println!("1. Enhanced Quantum Generator:");
    let generator = EnhancedQuantumGenerator::new(
        4, // num_qubits
        2, // latent_dim
        8, // output_dim
        3, // depth
    )?;

    println!("   Qubits: {}", generator.num_qubits);
    println!("   Latent dimension: {}", generator.latent_dim);
    println!("   Output dimension: {}", generator.output_dim);
    println!("   Circuit depth: {}", generator.depth);
    println!("   Parameters: {}", generator.params.len());

    // Generate samples
    let latent_vectors =
        Array2::from_shape_fn((5, 2), |(i, j)| (i as f64 * 0.2 + j as f64 * 0.1) - 0.5);

    let samples = generator.generate(&latent_vectors)?;
    println!(
        "\n   Generated {} samples of dimension {}",
        samples.nrows(),
        samples.ncols()
    );

    // 2. Enhanced Discriminator
    println!("\n2. Enhanced Quantum Discriminator:");
    let discriminator = EnhancedQuantumDiscriminator::new(
        4, // num_qubits
        8, // input_dim
        3, // depth
    )?;

    println!("   Qubits: {}", discriminator.num_qubits);
    println!("   Input dimension: {}", discriminator.input_dim);
    println!("   Circuit depth: {}", discriminator.depth);
    println!("   Parameters: {}", discriminator.params.len());

    // Discriminate samples
    let scores = discriminator.discriminate(&samples)?;
    println!("\n   Discrimination scores:");
    for (i, &score) in scores.iter().enumerate() {
        println!(
            "     Sample {}: {:.4} ({})",
            i,
            score,
            if score > 0.5 { "real" } else { "fake" }
        );
    }

    // 3. Wasserstein QGAN
    println!("\n3. Wasserstein Quantum GAN:");
    let wqgan = WassersteinQGAN::new(
        4, // num_qubits_gen
        4, // num_qubits_critic
        2, // latent_dim
        4, // data_dim
        2, // depth
    )?;

    println!("   Generator qubits: {}", wqgan.generator.num_qubits);
    println!("   Critic qubits: {}", wqgan.critic.num_qubits);
    println!("   Gradient penalty λ: {}", wqgan.lambda_gp);
    println!("   Critic iterations: {}", wqgan.n_critic);

    // Simulate Wasserstein loss calculation
    let real_scores = Array1::from_vec(vec![0.8, 0.85, 0.9, 0.75, 0.82]);
    let fake_scores = Array1::from_vec(vec![0.2, 0.15, 0.25, 0.3, 0.18]);

    let w_loss = wqgan.wasserstein_loss(&real_scores, &fake_scores);
    println!("\n   Wasserstein distance: {:.4}", w_loss);

    // Generate fake samples for gradient penalty
    let real_samples =
        Array2::from_shape_fn((5, 4), |(i, j)| ((i + j) as f64 * 0.1).sin() * 0.5 + 0.5);
    let fake_samples = wqgan.generator.generate(&latent_vectors)?;

    let gp = wqgan.gradient_penalty(&real_samples, &fake_samples)?;
    println!("   Gradient penalty: {:.4}", gp);
    println!(
        "   Total critic loss: {:.4}",
        -w_loss + wqgan.lambda_gp * gp
    );

    // 4. Conditional QGAN
    println!("\n4. Conditional Quantum GAN:");
    let cqgan = ConditionalQGAN::new(
        5, // num_qubits_gen
        5, // num_qubits_disc
        3, // latent_dim
        8, // data_dim
        4, // num_classes
        2, // depth
    )?;

    println!("   Number of classes: {}", cqgan.num_classes);
    println!(
        "   Generator latent+class dim: {}",
        cqgan.generator.latent_dim
    );
    println!(
        "   Discriminator input+class dim: {}",
        cqgan.discriminator.input_dim
    );

    // Generate samples for each class
    println!("\n   Class-conditional generation:");
    for class in 0..cqgan.num_classes {
        let class_samples = cqgan.generate_class(class, 3)?;
        println!(
            "     Class {}: Generated {} samples",
            class,
            class_samples.nrows()
        );

        // Show sample statistics
        let mean = class_samples.mean().unwrap_or(0.0);
        let max = class_samples.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min = class_samples.fold(f64::INFINITY, |a, &b| a.min(b));

        println!("       Mean: {:.4}, Min: {:.4}, Max: {:.4}", mean, min, max);
    }

    // 5. Circuit Analysis
    println!("\n5. Quantum Circuit Analysis:");

    // Build a sample generator circuit
    let latent = vec![0.5, -0.3];
    let gen_circuit = generator.build_circuit::<10>(&latent)?;
    println!("   Generator circuit created successfully");

    // Build a sample discriminator circuit
    let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let disc_circuit = discriminator.build_circuit::<10>(&input)?;
    println!("   Discriminator circuit created successfully");

    // 6. Training Simulation
    println!("\n6. Training Dynamics Simulation:");

    // Simulate training metrics over epochs
    let epochs = 10;
    println!("   Epoch | Gen Loss | Disc Loss | W-Distance");
    println!("   ------|----------|-----------|------------");

    for epoch in 0..epochs {
        // Simulate losses
        let gen_loss = 2.0 * (-(epoch as f64 / epochs as f64)).exp();
        let disc_loss = 0.5 + 0.3 * (-(epoch as f64 / epochs as f64 * 2.0)).exp();
        let w_dist = 1.0 - 0.8 * (epoch as f64 / epochs as f64);

        println!(
            "   {:5} | {:8.4} | {:9.4} | {:10.4}",
            epoch + 1,
            gen_loss,
            disc_loss,
            w_dist
        );
    }

    // 7. Mode Coverage Analysis
    println!("\n7. Mode Coverage Analysis:");
    println!("   Testing generator's ability to cover multiple data modes");

    let num_modes = 4;
    let samples_per_mode = 25;
    let total_samples = num_modes * samples_per_mode;

    // Generate many samples
    let large_latent =
        Array2::from_shape_fn((total_samples, 2), |(_, j)| fastrand::f64() * 2.0 - 1.0);
    let generated = generator.generate(&large_latent)?;

    // Simple mode detection (based on which quadrant the samples fall into)
    let mut mode_counts = vec![0; num_modes];
    for i in 0..generated.nrows() {
        let x = generated[[i, 0]] - 0.5;
        let y = generated[[i, 1]] - 0.5;

        let mode = match (x > 0.0, y > 0.0) {
            (true, true) => 0,
            (false, true) => 1,
            (false, false) => 2,
            (true, false) => 3,
        };
        mode_counts[mode] += 1;
    }

    println!("   Mode distribution (ideal: ~25% each):");
    for (i, &count) in mode_counts.iter().enumerate() {
        let percentage = count as f64 / total_samples as f64 * 100.0;
        println!("     Mode {}: {} samples ({:.1}%)", i, count, percentage);
    }

    // 8. Quantum Advantage Discussion
    println!("\n8. Quantum Advantages in GANs:");
    println!("   - Exponential parameter compression in quantum circuits");
    println!("   - Natural ability to generate quantum states");
    println!("   - Potential for capturing quantum correlations");
    println!("   - Hardware acceleration on quantum devices");
    println!("   - Novel loss landscapes due to quantum interference");

    println!("\n✅ Enhanced QGAN demonstration completed successfully!");

    Ok(())
}
