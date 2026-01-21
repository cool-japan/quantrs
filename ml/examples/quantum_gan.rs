#![allow(
    clippy::pedantic,
    clippy::unnecessary_wraps,
    clippy::needless_range_loop,
    clippy::useless_vec,
    clippy::needless_collect,
    clippy::too_many_arguments
)]
use quantrs2_ml::gan::{DiscriminatorType, GANEvaluationMetrics, GeneratorType, QuantumGAN};
use quantrs2_ml::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("Quantum Generative Adversarial Network Example");
    println!("=============================================");

    // GAN parameters
    let num_qubits_gen = 6;
    let num_qubits_disc = 6;
    let latent_dim = 4;
    let data_dim = 8;

    println!("Creating Quantum GAN...");
    println!("  Generator: {num_qubits_gen} qubits");
    println!("  Discriminator: {num_qubits_disc} qubits");
    println!("  Latent dimension: {latent_dim}");
    println!("  Data dimension: {data_dim}");

    // Create quantum GAN
    let mut qgan = QuantumGAN::new(
        num_qubits_gen,
        num_qubits_disc,
        latent_dim,
        data_dim,
        GeneratorType::HybridClassicalQuantum,
        DiscriminatorType::HybridQuantumFeatures,
    )?;

    // Generate synthetic data for training
    println!("Generating synthetic data for training...");
    let real_data = generate_sine_wave_data(500, data_dim);

    // Train GAN
    println!("Training quantum GAN...");
    let training_params = [
        (50, 32, 0.01, 0.01, 1), // (epochs, batch_size, lr_gen, lr_disc, disc_steps)
    ];

    for (epochs, batch_size, lr_gen, lr_disc, disc_steps) in training_params {
        println!("Training with parameters:");
        println!("  Epochs: {epochs}");
        println!("  Batch size: {batch_size}");
        println!("  Generator learning rate: {lr_gen}");
        println!("  Discriminator learning rate: {lr_disc}");
        println!("  Discriminator steps per iteration: {disc_steps}");

        let start = Instant::now();
        let history = qgan.train(&real_data, epochs, batch_size, lr_gen, lr_disc, disc_steps)?;

        println!("Training completed in {:.2?}", start.elapsed());
        println!("Final losses:");
        println!(
            "  Generator: {:.4}",
            history.gen_losses.last().unwrap_or(&0.0)
        );
        println!(
            "  Discriminator: {:.4}",
            history.disc_losses.last().unwrap_or(&0.0)
        );
    }

    // Generate samples
    println!("\nGenerating samples from trained GAN...");
    let num_samples = 10;
    let generated_samples = qgan.generate(num_samples)?;

    println!("Generated {num_samples} samples");
    println!("First sample:");
    print_sample(
        &generated_samples
            .slice(scirs2_core::ndarray::s![0, ..])
            .to_owned(),
    );

    // Evaluate GAN
    println!("\nEvaluating GAN quality...");
    let eval_metrics = qgan.evaluate(&real_data, num_samples)?;

    println!("Evaluation metrics:");
    println!(
        "  Real data accuracy: {:.2}%",
        eval_metrics.real_accuracy * 100.0
    );
    println!(
        "  Fake data accuracy: {:.2}%",
        eval_metrics.fake_accuracy * 100.0
    );
    println!(
        "  Overall discriminator accuracy: {:.2}%",
        eval_metrics.overall_accuracy * 100.0
    );
    println!("  JS Divergence: {:.4}", eval_metrics.js_divergence);

    // Use physics-specific GAN
    println!("\nCreating specialized particle physics GAN...");
    let particle_gan = quantrs2_ml::gan::physics_gan::ParticleGAN::new(
        num_qubits_gen,
        num_qubits_disc,
        latent_dim,
        data_dim,
    )?;

    println!("Particle GAN created successfully");

    Ok(())
}

// Generate synthetic sine wave data
fn generate_sine_wave_data(num_samples: usize, data_dim: usize) -> Array2<f64> {
    let mut data = Array2::zeros((num_samples, data_dim));

    for i in 0..num_samples {
        let x = (i as f64) / (num_samples as f64) * 2.0 * std::f64::consts::PI;

        for j in 0..data_dim {
            let freq = (j as f64 + 1.0) * 0.5;
            data[[i, j]] = 0.1f64.mul_add(thread_rng().gen::<f64>(), (x * freq).sin());
        }
    }

    data
}

// Print a sample vector
fn print_sample(sample: &Array1<f64>) {
    print!("  [");
    for (i, &val) in sample.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{val:.4}");
    }
    println!("]");
}
