//! Example demonstrating energy landscape visualization
//!
//! This example shows how to:
//! 1. Analyze energy landscapes of different problem types
//! 2. Generate landscape statistics
//! 3. Export visualization data
//! 4. Compare problem difficulty using landscape features

use quantrs2_anneal::{
    ising::IsingModel,
    qubo::QuboBuilder,
    visualization::{
        calculate_landscape_stats, plot_energy_histogram, plot_energy_landscape, BasinAnalyzer,
        LandscapeAnalyzer,
    },
};
use std::fs;
use scirs2_core::random::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Energy Landscape Visualization Demo ===\n");

    // Create output directory
    fs::create_dir_all("landscape_outputs")?;

    // Example 1: Simple ferromagnetic chain
    println!("Example 1: Ferromagnetic Chain");
    let ferro_model = create_ferromagnetic_chain(8)?;
    analyze_and_visualize(&ferro_model, "ferromagnetic_chain", "Ferromagnetic Chain")?;

    // Example 2: Frustrated triangular lattice
    println!("\nExample 2: Frustrated System");
    let frustrated_model = create_frustrated_triangle(6)?;
    analyze_and_visualize(
        &frustrated_model,
        "frustrated_triangle",
        "Frustrated Triangle",
    )?;

    // Example 3: Random coupling problem
    println!("\nExample 3: Random Couplings");
    let random_model = create_random_problem(10, 0.5)?;
    analyze_and_visualize(&random_model, "random_problem", "Random Problem")?;

    // Example 4: MaxCut problem
    println!("\nExample 4: MaxCut Problem");
    let maxcut_model = create_maxcut_problem(8)?;
    analyze_and_visualize(&maxcut_model, "maxcut_problem", "MaxCut Problem")?;

    // Example 5: Compare landscape features
    println!("\nExample 5: Landscape Comparison");
    compare_landscapes()?;

    println!("\nVisualization files saved to 'landscape_outputs/' directory");
    println!("You can plot these CSV files using your favorite plotting tool:");
    println!("  - *_landscape.csv: Energy vs configuration data");
    println!("  - *_histogram.csv: Energy distribution histograms");

    Ok(())
}

fn analyze_and_visualize(
    model: &IsingModel,
    name: &str,
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create analyzer
    let analyzer = LandscapeAnalyzer::new()
        .with_max_samples(1000)
        .with_seed(42);

    // Sample or enumerate landscape
    let mut points = if model.num_qubits <= 12 {
        println!("  Enumerating complete landscape...");
        analyzer.enumerate_landscape(model)?
    } else {
        println!("  Sampling landscape...");
        analyzer.sample_landscape(model)?
    };

    println!("  Found {} configurations", points.len());

    // Calculate statistics
    let stats = calculate_landscape_stats(&points);
    println!(
        "  Energy range: [{:.4}, {:.4}]",
        stats.min_energy, stats.max_energy
    );
    println!(
        "  Mean energy: {:.4} ± {:.4}",
        stats.mean_energy, stats.energy_std
    );
    println!(
        "  Ground state degeneracy: {}",
        stats.ground_state_degeneracy
    );

    if let Some(gap) = stats.energy_gap {
        println!("  Energy gap: {:.4}", gap);
    }

    // Identify basins
    let basin_analyzer = BasinAnalyzer::new(0.1, 2);
    let num_basins = basin_analyzer.identify_basins(&mut points)?;
    println!("  Number of energy basins: {}", num_basins);

    // Export landscape data
    let landscape_path = format!("landscape_outputs/{}_landscape.csv", name);
    plot_energy_landscape(&points, &landscape_path, title)?;

    // Export energy histogram
    let histogram_path = format!("landscape_outputs/{}_histogram.csv", name);
    plot_energy_histogram(&points, &histogram_path, 20)?;

    Ok(())
}

fn compare_landscapes() -> Result<(), Box<dyn std::error::Error>> {
    println!("Comparing landscape characteristics:");

    let problems = vec![
        ("Ferromagnetic", create_ferromagnetic_chain(8)?),
        ("Frustrated", create_frustrated_triangle(6)?),
        ("Random", create_random_problem(8, 0.3)?),
    ];

    println!(
        "{:<15} {:<10} {:<10} {:<10} {:<10} {:<8}",
        "Problem", "Min E", "Max E", "Mean E", "Std E", "Gap"
    );
    println!("{}", "-".repeat(70));

    for (name, model) in problems {
        let analyzer = LandscapeAnalyzer::new().with_max_samples(500).with_seed(42);

        let points = if model.num_qubits <= 12 {
            analyzer.enumerate_landscape(&model)?
        } else {
            analyzer.sample_landscape(&model)?
        };

        let stats = calculate_landscape_stats(&points);

        println!(
            "{:<15} {:<10.3} {:<10.3} {:<10.3} {:<10.3} {:<8.3}",
            name,
            stats.min_energy,
            stats.max_energy,
            stats.mean_energy,
            stats.energy_std,
            stats.energy_gap.unwrap_or(0.0)
        );
    }

    Ok(())
}

/// Create a ferromagnetic chain model
fn create_ferromagnetic_chain(n: usize) -> Result<IsingModel, Box<dyn std::error::Error>> {
    let mut model = IsingModel::new(n);

    // Chain with ferromagnetic couplings
    for i in 0..(n - 1) {
        model.set_coupling(i, i + 1, -1.0)?;
    }

    Ok(model)
}

/// Create a frustrated triangular system
fn create_frustrated_triangle(n: usize) -> Result<IsingModel, Box<dyn std::error::Error>> {
    let mut model = IsingModel::new(n);

    // Create triangles with antiferromagnetic couplings
    for i in 0..(n / 3) {
        let base = i * 3;
        if base + 2 < n {
            model.set_coupling(base, base + 1, 1.0)?;
            model.set_coupling(base + 1, base + 2, 1.0)?;
            model.set_coupling(base, base + 2, 1.0)?;
        }
    }

    Ok(model)
}

/// Create a random coupling problem
fn create_random_problem(n: usize, density: f64) -> Result<IsingModel, Box<dyn std::error::Error>> {
    let mut model = IsingModel::new(n);

    for i in 0..n {
        for j in (i + 1)..n {
            if thread_rng().gen::<f64>() < density {
                let coupling = (thread_rng().gen::<f64>() - 0.5) * 2.0; // Range [-1, 1]
                model.set_coupling(i, j, coupling)?;
            }
        }
    }

    Ok(model)
}

/// Create a MaxCut problem instance
fn create_maxcut_problem(n: usize) -> Result<IsingModel, Box<dyn std::error::Error>> {
    let mut model = IsingModel::new(n);

    // Create a random graph with negative couplings for MaxCut
    for i in 0..n {
        for j in (i + 1)..n {
            if thread_rng().gen::<f64>() < 0.4 {
                // 40% edge probability
                model.set_coupling(i, j, -1.0)?;
            }
        }
    }

    Ok(model)
}

/// Create a Number Partitioning problem as Ising model
fn _create_number_partitioning() -> Result<IsingModel, Box<dyn std::error::Error>> {
    let numbers = vec![3.0, 1.0, 1.0, 2.0, 2.0, 1.0];
    let n = numbers.len();
    let target = numbers.iter().sum::<f64>() / 2.0;

    let mut model = IsingModel::new(n);

    // Set biases
    for i in 0..n {
        let bias = numbers[i] - target;
        model.set_bias(i, bias)?;
    }

    // Set quadratic terms
    for i in 0..n {
        for j in (i + 1)..n {
            model.set_coupling(i, j, numbers[i] * numbers[j])?;
        }
    }

    Ok(model)
}
