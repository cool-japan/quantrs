use ndarray::{Array1, Array2};
use quantrs2_ml::prelude::*;
use quantrs2_ml::gan::{QuantumGAN, GeneratorType, DiscriminatorType, GANTrainingHistory};
use std::time::Instant;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("QuantRS2 Quantum Generative Adversarial Network Example");
    println!("====================================================");

    // Set up parameters for our GAN
    let num_qubits_gen = 6;  // Qubits for generator
    let num_qubits_disc = 6; // Qubits for discriminator
    let latent_dim = 4;      // Input noise dimension
    let data_dim = 8;        // Output data dimension
    
    println!("Creating Quantum GAN with configuration:");
    println!("  Generator: {} qubits (type: HybridClassicalQuantum)", num_qubits_gen);
    println!("  Discriminator: {} qubits (type: HybridQuantumFeatures)", num_qubits_disc);
    println!("  Latent dimension: {}", latent_dim);
    println!("  Data dimension: {}", data_dim);
    
    // Create the quantum GAN
    let mut qgan = QuantumGAN::new(
        num_qubits_gen,
        num_qubits_disc,
        latent_dim,
        data_dim,
        GeneratorType::HybridClassicalQuantum,
        DiscriminatorType::HybridQuantumFeatures,
    )?;
    
    // Generate a dataset to learn (we'll use a sine wave with multiple frequencies)
    println!("\nGenerating training data (multi-frequency sine waves)...");
    let (real_data, x_values) = generate_multifrequency_sine_data(200, data_dim)?;
    
    // Display some statistics about our dataset
    let mean = real_data.mean_axis(ndarray::Axis(0))?;
    let std_dev = real_data.std_axis(ndarray::Axis(0), 0.0);
    
    println!("Dataset statistics:");
    println!("  Samples: {}", real_data.shape()[0]);
    println!("  Mean: {:.4}", mean[0]);
    println!("  Standard deviation: {:.4}", std_dev[0]);
    
    // Train the GAN
    println!("\nTraining the Quantum GAN...");
    let start = Instant::now();
    
    let history = qgan.train(
        &real_data,
        50,    // epochs
        16,    // batch size
        0.01,  // generator learning rate
        0.01,  // discriminator learning rate
        1,     // discriminator steps per generator step
    )?;
    
    println!("Training completed in {:.2?}", start.elapsed());
    
    // Show training progress
    println!("\nTraining progression (selected epochs):");
    print_training_progress(&history);
    
    // Generate some samples
    println!("\nGenerating samples from trained GAN...");
    let num_samples = 10;
    let generated_samples = qgan.generate(num_samples)?;
    
    // Compare some samples with real data
    println!("\nComparison of real vs. generated data (first 3 samples):");
    for i in 0..3.min(num_samples) {
        // Take a real data point
        let real_idx = i * real_data.shape()[0] / num_samples;
        let real_sample = real_data.slice(ndarray::s![real_idx, ..]);
        
        // Get a generated sample
        let gen_sample = generated_samples.slice(ndarray::s![i, ..]);
        
        println!("Sample {}:", i);
        println!("  Real:       [{}]", format_sample(&real_sample));
        println!("  Generated:  [{}]", format_sample(&gen_sample));
    }
    
    // Calculate similarity metrics
    let similarity = calculate_similarity(&real_data, &generated_samples)?;
    println!("\nSimilarity between real and generated distributions: {:.4}", similarity);
    
    // Demonstrate using the GAN for generating specific patterns
    println!("\nGenerating specific patterns (conditioned generation)...");
    
    // Condition on the first feature being high
    let conditions = vec![(0, 0.8)]; // Feature 0 should be around 0.8
    let conditional_samples = qgan.generate_conditional(5, &conditions)?;
    
    println!("Samples conditioned on feature 0 ≈ 0.8:");
    for i in 0..3.min(conditional_samples.shape()[0]) {
        let sample = conditional_samples.slice(ndarray::s![i, ..]);
        println!("  Sample {}: [{}]", i, format_sample(&sample));
    }
    
    println!("\nQuantum GAN example completed successfully!");
    Ok(())
}

// Helper function to generate sine wave data with multiple frequencies
fn generate_multifrequency_sine_data(num_samples: usize, data_dim: usize) 
    -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    
    let mut data = Array2::zeros((num_samples, data_dim));
    let x_values = Array1::linspace(0.0, 2.0 * std::f64::consts::PI, num_samples);
    
    for i in 0..num_samples {
        let x = x_values[i];
        
        for j in 0..data_dim {
            // Each dimension has a different frequency
            let freq = 1.0 + (j as f64) * 0.2;
            let phase = (j as f64) * 0.1;
            data[[i, j]] = (x * freq + phase).sin() * 0.5 + 0.5;
            
            // Add some noise
            data[[i, j]] += (rand::random::<f64>() - 0.5) * 0.05;
            
            // Clamp to [0, 1]
            data[[i, j]] = data[[i, j]].max(0.0).min(1.0);
        }
    }
    
    Ok((data, x_values))
}

// Helper function to print training progress
fn print_training_progress(history: &GANTrainingHistory) {
    let epochs = history.gen_losses.len();
    let intervals = [0, epochs/4, epochs/2, 3*epochs/4, epochs-1];
    
    println!("{:<10} {:<15} {:<15}", "Epoch", "Generator Loss", "Discriminator Loss");
    
    for &epoch in intervals.iter() {
        if epoch < epochs {
            println!("{:<10} {:<15.6} {:<15.6}", 
                    epoch + 1, 
                    history.gen_losses[epoch], 
                    history.disc_losses[epoch]);
        }
    }
}

// Helper function to format a sample for display
fn format_sample(sample: &ndarray::ArrayView1<f64>) -> String {
    let mut result = String::new();
    
    for (i, &value) in sample.iter().enumerate() {
        if i > 0 {
            result.push_str(", ");
        }
        
        if i >= 5 && i < sample.len() - 1 {
            result.push_str("...");
            break;
        }
        
        result.push_str(&format!("{:.3}", value));
    }
    
    if sample.len() > 6 {
        result.push_str(", ");
        result.push_str(&format!("{:.3}", sample[sample.len() - 1]));
    }
    
    result
}

// Helper function to calculate similarity between distributions
fn calculate_similarity(real: &Array2<f64>, generated: &Array2<f64>) -> Result<f64, Box<dyn Error>> {
    // Calculate means
    let real_mean = real.mean_axis(ndarray::Axis(0))?;
    let gen_mean = generated.mean_axis(ndarray::Axis(0))?;
    
    // Calculate mean difference
    let mut mean_diff = 0.0;
    for i in 0..real_mean.len() {
        mean_diff += (real_mean[i] - gen_mean[i]).abs();
    }
    mean_diff /= real_mean.len() as f64;
    
    // Calculate similarity score (1 = identical, 0 = completely different)
    let similarity = (1.0 - mean_diff).max(0.0);
    
    Ok(similarity)
}