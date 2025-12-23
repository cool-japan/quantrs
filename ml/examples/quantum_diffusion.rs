//! Quantum Diffusion Model Example
//!
//! This example demonstrates quantum diffusion models for generative modeling,
//! including DDPM-style models and score-based diffusion.

use quantrs2_ml::autodiff::optimizers::Adam;
use quantrs2_ml::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::prelude::*;

fn main() -> Result<()> {
    println!("=== Quantum Diffusion Model Demo ===\n");

    // Step 1: Demonstrate noise schedules
    println!("1. Comparing Noise Schedules...");
    compare_noise_schedules()?;

    // Step 2: Train quantum diffusion model on simple data
    println!("\n2. Training Quantum Diffusion Model...");
    train_diffusion_model()?;

    // Step 3: Generate samples
    println!("\n3. Generating New Samples...");
    generate_samples()?;

    // Step 4: Score-based diffusion
    println!("\n4. Score-Based Diffusion Demo...");
    score_diffusion_demo()?;

    // Step 5: Demonstrate diffusion process
    println!("\n5. Visualizing Diffusion Process...");
    visualize_diffusion_process()?;

    println!("\n=== Diffusion Model Demo Complete ===");

    Ok(())
}

/// Compare different noise schedules
fn compare_noise_schedules() -> Result<()> {
    let num_timesteps = 100;

    let schedules = vec![
        (
            "Linear",
            NoiseSchedule::Linear {
                beta_start: 0.0001,
                beta_end: 0.02,
            },
        ),
        ("Cosine", NoiseSchedule::Cosine { s: 0.008 }),
        (
            "Quadratic",
            NoiseSchedule::Quadratic {
                beta_start: 0.0001,
                beta_end: 0.02,
            },
        ),
        (
            "Sigmoid",
            NoiseSchedule::Sigmoid {
                beta_start: 0.0001,
                beta_end: 0.02,
            },
        ),
    ];

    println!("   Noise levels at different timesteps:");
    println!("   Time     Linear   Cosine   Quadratic  Sigmoid");

    for t in (0..=100).step_by(20) {
        let t_idx = (t * (num_timesteps - 1) / 100).min(num_timesteps - 1);
        print!("   t={t:3}%: ");

        for (_, schedule) in &schedules {
            let model = QuantumDiffusionModel::new(2, 4, num_timesteps, *schedule)?;
            print!("{:8.4} ", model.betas()[t_idx]);
        }
        println!();
    }

    Ok(())
}

/// Train a quantum diffusion model
fn train_diffusion_model() -> Result<()> {
    // Generate synthetic 2D data (two moons)
    let num_samples = 200;
    let data = generate_two_moons(num_samples);

    println!("   Generated {num_samples} samples of 2D two-moons data");

    // Create diffusion model
    let mut model = QuantumDiffusionModel::new(
        2,  // data dimension
        4,  // num qubits
        50, // timesteps
        NoiseSchedule::Cosine { s: 0.008 },
    )?;

    println!("   Created quantum diffusion model:");
    println!("   - Data dimension: 2");
    println!("   - Qubits: 4");
    println!("   - Timesteps: 50");
    println!("   - Schedule: Cosine");

    // Train model
    let mut optimizer = Adam::new(0.001);
    let epochs = 100;
    let batch_size = 32;

    println!("\n   Training for {epochs} epochs...");
    let losses = model.train(&data, &mut optimizer, epochs, batch_size)?;

    // Print training statistics
    println!("\n   Training Statistics:");
    println!("   - Initial loss: {:.4}", losses[0]);
    println!("   - Final loss: {:.4}", losses.last().unwrap());
    println!(
        "   - Improvement: {:.2}%",
        (1.0 - losses.last().unwrap() / losses[0]) * 100.0
    );

    Ok(())
}

/// Generate samples from trained model
fn generate_samples() -> Result<()> {
    // Create a simple trained model
    let model = QuantumDiffusionModel::new(
        2,  // data dimension
        4,  // num qubits
        50, // timesteps
        NoiseSchedule::Linear {
            beta_start: 0.0001,
            beta_end: 0.02,
        },
    )?;

    // Generate samples
    let num_samples = 10;
    println!("   Generating {num_samples} samples...");

    let samples = model.generate(num_samples)?;

    println!("\n   Generated samples:");
    for i in 0..num_samples.min(5) {
        println!(
            "   Sample {}: [{:.3}, {:.3}]",
            i + 1,
            samples[[i, 0]],
            samples[[i, 1]]
        );
    }

    // Compute statistics
    let mean = samples.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
    let std = samples.std_axis(scirs2_core::ndarray::Axis(0), 0.0);

    println!("\n   Sample statistics:");
    println!("   - Mean: [{:.3}, {:.3}]", mean[0], mean[1]);
    println!("   - Std:  [{:.3}, {:.3}]", std[0], std[1]);

    Ok(())
}

/// Score-based diffusion demonstration
fn score_diffusion_demo() -> Result<()> {
    // Create score-based model
    let model = QuantumScoreDiffusion::new(
        2,  // data dimension
        4,  // num qubits
        10, // noise levels
    )?;

    println!("   Created quantum score-based diffusion model");
    println!("   - Noise levels: {:?}", model.noise_levels());

    // Test score estimation
    let x = Array1::from_vec(vec![0.5, -0.3]);
    let noise_level = 0.1;

    let score = model.estimate_score(&x, noise_level)?;
    println!("\n   Score estimation:");
    println!("   - Input: [{:.3}, {:.3}]", x[0], x[1]);
    println!("   - Noise level: {noise_level:.3}");
    println!("   - Estimated score: [{:.3}, {:.3}]", score[0], score[1]);

    // Langevin sampling
    println!("\n   Langevin sampling:");
    let init = Array1::from_vec(vec![2.0, 2.0]);
    let num_steps = 100;
    let step_size = 0.01;

    let sample = model.langevin_sample(init.clone(), noise_level, num_steps, step_size)?;

    println!("   - Initial: [{:.3}, {:.3}]", init[0], init[1]);
    println!(
        "   - After {} steps: [{:.3}, {:.3}]",
        num_steps, sample[0], sample[1]
    );
    println!(
        "   - Distance moved: {:.3}",
        (sample[0] - init[0]).hypot(sample[1] - init[1])
    );

    Ok(())
}

/// Visualize the diffusion process
fn visualize_diffusion_process() -> Result<()> {
    let model = QuantumDiffusionModel::new(
        2,  // data dimension
        4,  // num qubits
        20, // fewer timesteps for visualization
        NoiseSchedule::Linear {
            beta_start: 0.0001,
            beta_end: 0.02,
        },
    )?;

    // Start with a clear data point
    let x0 = Array1::from_vec(vec![1.0, 0.5]);

    println!("   Forward diffusion process:");
    println!("   t=0 (original): [{:.3}, {:.3}]", x0[0], x0[1]);

    // Show forward diffusion at different timesteps
    for t in [5, 10, 15, 19] {
        let (xt, _) = model.forward_diffusion(&x0, t)?;
        let noise_level = (1.0 - model.alphas_cumprod()[t]).sqrt();
        println!(
            "   t={:2} (noise={:.3}): [{:.3}, {:.3}]",
            t, noise_level, xt[0], xt[1]
        );
    }

    println!("\n   Reverse diffusion process:");

    // Start from noise
    let mut xt = Array1::from_vec(vec![
        2.0f64.mul_add(thread_rng().gen::<f64>(), -1.0),
        2.0f64.mul_add(thread_rng().gen::<f64>(), -1.0),
    ]);

    println!("   t=19 (pure noise): [{:.3}, {:.3}]", xt[0], xt[1]);

    // Show reverse diffusion
    for t in [15, 10, 5, 0] {
        xt = model.reverse_diffusion_step(&xt, t)?;
        println!("   t={:2} (denoised): [{:.3}, {:.3}]", t, xt[0], xt[1]);
    }

    println!("\n   This demonstrates how diffusion models:");
    println!("   1. Gradually add noise to data (forward process)");
    println!("   2. Learn to reverse this process (backward process)");
    println!("   3. Generate new samples by denoising random noise");

    Ok(())
}

/// Generate two-moons dataset
fn generate_two_moons(n_samples: usize) -> Array2<f64> {
    let mut data = Array2::zeros((n_samples, 2));
    let n_samples_per_moon = n_samples / 2;

    // First moon
    for i in 0..n_samples_per_moon {
        let angle = std::f64::consts::PI * i as f64 / n_samples_per_moon as f64;
        data[[i, 0]] = 0.1f64.mul_add(2.0f64.mul_add(thread_rng().gen::<f64>(), -1.0), angle.cos());
        data[[i, 1]] = 0.1f64.mul_add(2.0f64.mul_add(thread_rng().gen::<f64>(), -1.0), angle.sin());
    }

    // Second moon (shifted and flipped)
    for i in 0..n_samples_per_moon {
        let idx = n_samples_per_moon + i;
        let angle = std::f64::consts::PI * i as f64 / n_samples_per_moon as f64;
        data[[idx, 0]] = 0.1f64.mul_add(
            2.0f64.mul_add(thread_rng().gen::<f64>(), -1.0),
            1.0 - angle.cos(),
        );
        data[[idx, 1]] = 0.1f64.mul_add(
            2.0f64.mul_add(thread_rng().gen::<f64>(), -1.0),
            0.5 - angle.sin(),
        );
    }

    data
}

/// Advanced diffusion techniques demonstration
fn advanced_diffusion_demo() -> Result<()> {
    println!("\n6. Advanced Diffusion Techniques:");

    // Conditional generation
    println!("\n   a) Conditional Generation:");
    let model = QuantumDiffusionModel::new(4, 4, 50, NoiseSchedule::Cosine { s: 0.008 })?;
    let condition = Array1::from_vec(vec![0.5, -0.5]);
    let conditional_samples = model.conditional_generate(&condition, 5)?;

    println!(
        "   Generated {} conditional samples",
        conditional_samples.nrows()
    );
    println!("   Condition: [{:.3}, {:.3}]", condition[0], condition[1]);

    // Variational diffusion
    println!("\n   b) Variational Diffusion Model:");
    let vdm = QuantumVariationalDiffusion::new(
        4, // data_dim
        2, // latent_dim
        4, // num_qubits
    )?;

    let x = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
    let (mean, log_var) = vdm.encode(&x)?;

    println!("   Encoded data to latent space:");
    println!("   - Input: {:?}", x.as_slice().unwrap());
    println!("   - Latent mean: [{:.3}, {:.3}]", mean[0], mean[1]);
    println!(
        "   - Latent log_var: [{:.3}, {:.3}]",
        log_var[0], log_var[1]
    );

    Ok(())
}
