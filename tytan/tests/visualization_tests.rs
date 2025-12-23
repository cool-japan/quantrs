//! Tests and examples for advanced visualization functionality

use quantrs2_tytan::analysis::visualization::*;
use quantrs2_tytan::sampler::{SASampler, SampleResult, Sampler};
use scirs2_core::ndarray::Array;
use scirs2_core::random::prelude::*;
use std::collections::HashMap;

/// Create dummy sample results for testing
fn create_test_results(n: usize) -> Vec<SampleResult> {
    let mut results = Vec::new();

    for i in 0..n {
        let mut assignments = HashMap::new();

        // Create some binary variables
        assignments.insert("x0".to_string(), i % 2 == 0);
        assignments.insert("x1".to_string(), i % 3 == 0);
        assignments.insert("x2".to_string(), i % 5 == 0);
        assignments.insert("x3".to_string(), i % 7 == 0);

        // Energy is lower for more "aligned" solutions
        let energy = (i as f64).mul_add(
            0.1,
            if i % 2 == 0 { -2.0 } else { -1.0 }
                + if i % 3 == 0 { -1.5 } else { 0.0 }
                + if i % 5 == 0 { -1.0 } else { 0.5 },
        ); // Add some variation

        results.push(SampleResult {
            assignments,
            energy,
            occurrences: 1,
        });
    }

    results
}

#[test]
fn test_energy_landscape_preparation() {
    let results = create_test_results(100);

    // Test with default config
    let landscape_data = prepare_energy_landscape(&results, None).unwrap();

    assert_eq!(landscape_data.indices.len(), 100);
    assert_eq!(landscape_data.energies.len(), 100);
    assert!(landscape_data.energies[0] <= landscape_data.energies[99]);
    assert_eq!(landscape_data.histogram_bins.len(), 51); // num_bins + 1
    assert_eq!(landscape_data.histogram_counts.len(), 50);

    // Check KDE was computed
    assert!(landscape_data.kde_x.is_some());
    assert!(landscape_data.kde_y.is_some());
    assert_eq!(landscape_data.kde_x.as_ref().unwrap().len(), 200);
}

#[test]
fn test_solution_distribution_analysis() {
    let results = create_test_results(50);

    // Test with default config
    let dist_data = analyze_solution_distribution(&results, None).unwrap();

    assert_eq!(dist_data.variable_names.len(), 4);
    assert_eq!(dist_data.variable_frequencies.len(), 4);

    // Check that frequencies are in [0, 1]
    for &freq in dist_data.variable_frequencies.values() {
        assert!((0.0..=1.0).contains(&freq));
    }

    // Check correlations were computed
    assert!(dist_data.correlations.is_some());

    // Check solution matrix dimensions
    assert_eq!(dist_data.solution_matrix.nrows(), 50);
    assert_eq!(dist_data.solution_matrix.ncols(), 4);
}

#[test]
fn test_tsp_tour_extraction() {
    // Create a TSP solution for 4 cities
    let mut assignments = HashMap::new();
    // Tour: 0 -> 1 -> 3 -> 2 -> 0
    assignments.insert("x_0_1".to_string(), true);
    assignments.insert("x_1_3".to_string(), true);
    assignments.insert("x_3_2".to_string(), true);
    assignments.insert("x_2_0".to_string(), true);

    // Add false edges
    assignments.insert("x_0_2".to_string(), false);
    assignments.insert("x_0_3".to_string(), false);
    assignments.insert("x_1_0".to_string(), false);
    assignments.insert("x_1_2".to_string(), false);

    let result = SampleResult {
        assignments,
        energy: -10.0,
        occurrences: 1,
    };

    let tour = extract_tsp_tour(&result, 4).unwrap();
    assert_eq!(tour.len(), 4);
    assert_eq!(tour[0], 0); // Start at city 0

    // Check that all cities are visited
    let mut visited = [false; 4];
    for &city in &tour {
        visited[city] = true;
    }
    assert!(visited.iter().all(|&v| v));
}

#[test]
fn test_graph_coloring_extraction() {
    // Create a graph coloring solution
    let mut assignments = HashMap::new();

    // 4 nodes, 3 colors
    assignments.insert("x_0_0".to_string(), true); // Node 0 -> Color 0
    assignments.insert("x_0_1".to_string(), false);
    assignments.insert("x_0_2".to_string(), false);

    assignments.insert("x_1_1".to_string(), true); // Node 1 -> Color 1
    assignments.insert("x_1_0".to_string(), false);
    assignments.insert("x_1_2".to_string(), false);

    assignments.insert("x_2_0".to_string(), true); // Node 2 -> Color 0
    assignments.insert("x_2_1".to_string(), false);
    assignments.insert("x_2_2".to_string(), false);

    assignments.insert("x_3_2".to_string(), true); // Node 3 -> Color 2
    assignments.insert("x_3_0".to_string(), false);
    assignments.insert("x_3_1".to_string(), false);

    let result = SampleResult {
        assignments,
        energy: -5.0,
        occurrences: 1,
    };

    // Edges: 0-1, 1-2, 2-3
    let edges = vec![(0, 1), (1, 2), (2, 3)];

    let (colors, conflicts) = extract_graph_coloring(&result, 4, 3, &edges).unwrap();

    assert_eq!(colors.len(), 4);
    assert_eq!(colors[0], 0);
    assert_eq!(colors[1], 1);
    assert_eq!(colors[2], 0);
    assert_eq!(colors[3], 2);

    // Check conflicts - nodes 0 and 2 have same color but no edge between them
    assert_eq!(conflicts.len(), 0);
}

#[test]
fn test_convergence_analysis() {
    // Create iteration results with improving energy
    let mut iteration_results = Vec::new();

    for i in 0..10 {
        let mut iter_samples = Vec::new();

        for j in 0..20 {
            let energy = 0.1f64.mul_add(-f64::from(j), -f64::from(i)) + thread_rng().gen::<f64>();

            iter_samples.push(SampleResult {
                assignments: HashMap::new(),
                energy,
                occurrences: 1,
            });
        }

        iteration_results.push(iter_samples);
    }

    // Analyze convergence
    let conv_data = analyze_convergence(&iteration_results, Some(3)).unwrap();

    assert_eq!(conv_data.iterations.len(), 10);
    assert_eq!(conv_data.best_energies.len(), 10);
    assert_eq!(conv_data.avg_energies.len(), 10);
    assert_eq!(conv_data.std_devs.len(), 10);

    // Check that best energy generally decreases
    assert!(conv_data.best_energies[0] > conv_data.best_energies[9]);

    // Check moving averages
    assert!(conv_data.ma_best.is_some());
    assert_eq!(conv_data.ma_best.as_ref().unwrap().len(), 8); // 10 - 3 + 1
}

#[test]
fn test_csv_export() {
    let results = create_test_results(10);
    let landscape_data = prepare_energy_landscape(&results, None).unwrap();

    // Export to temporary file
    let temp_path = "/tmp/test_energy_landscape.csv";
    export_to_csv(&landscape_data, temp_path).unwrap();

    // Read back and verify
    let contents = std::fs::read_to_string(temp_path).unwrap();
    let lines: Vec<&str> = contents.trim().split('\n').collect();

    assert_eq!(lines[0], "index,original_index,energy");
    assert_eq!(lines.len(), 11); // header + 10 data rows

    // Clean up
    std::fs::remove_file(temp_path).ok();
}

#[test]
fn test_spring_layout() {
    // Create a simple graph
    let n_nodes = 5;
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)];

    let positions = spring_layout(n_nodes, &edges);

    assert_eq!(positions.len(), n_nodes);

    // Check that all positions are in [0, 1]
    for &(x, y) in &positions {
        assert!((0.0..=1.0).contains(&x));
        assert!((0.0..=1.0).contains(&y));
    }
}

/// Example: Complete visualization workflow
#[test]
fn example_complete_visualization_workflow() {
    println!("\n=== Advanced Visualization Example ===\n");

    // Step 1: Create or load sample results
    println!("1. Creating sample results...");
    let results = create_test_results(100);
    println!("   Created {} sample results", results.len());

    // Step 2: Analyze energy landscape
    println!("\n2. Analyzing energy landscape...");
    let landscape_config = EnergyLandscapeConfig {
        num_bins: 20,
        compute_kde: true,
        kde_points: 100,
    };

    let landscape_data = prepare_energy_landscape(&results, Some(landscape_config)).unwrap();

    println!(
        "   Energy range: {:.2} to {:.2}",
        landscape_data.energies.first().unwrap(),
        landscape_data.energies.last().unwrap()
    );
    println!("   Histogram bins: {}", landscape_data.histogram_bins.len());

    // Export for external plotting
    export_to_csv(&landscape_data, "/tmp/energy_landscape.csv").ok();
    println!("   Exported to /tmp/energy_landscape.csv");

    // Step 3: Analyze solution distribution
    println!("\n3. Analyzing solution distribution...");
    let dist_config = SolutionDistributionConfig {
        compute_correlations: true,
        compute_pca: true,
        n_components: 2,
    };

    let dist_data = analyze_solution_distribution(&results, Some(dist_config)).unwrap();

    println!("   Variables: {:?}", dist_data.variable_names);
    println!("   Variable frequencies:");
    for (var, freq) in &dist_data.variable_frequencies {
        println!("     {}: {:.2}%", var, freq * 100.0);
    }

    if let Some(correlations) = &dist_data.correlations {
        println!("   Significant correlations:");
        for ((var1, var2), corr) in correlations {
            if corr.abs() > 0.3 {
                println!("     {var1} <-> {var2}: {corr:.3}");
            }
        }
    }

    // Export solution matrix
    export_solution_matrix(&dist_data, "/tmp/solution_matrix.csv").ok();
    println!("   Exported solution matrix to /tmp/solution_matrix.csv");

    // Step 4: Problem-specific visualization (TSP example)
    println!("\n4. TSP visualization example...");
    let cities = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];

    let mut tsp_assignments = HashMap::new();
    tsp_assignments.insert("x_0_1".to_string(), true);
    tsp_assignments.insert("x_1_2".to_string(), true);
    tsp_assignments.insert("x_2_3".to_string(), true);
    tsp_assignments.insert("x_3_0".to_string(), true);

    let tsp_result = SampleResult {
        assignments: tsp_assignments,
        energy: -4.0,
        occurrences: 1,
    };

    let tour = extract_tsp_tour(&tsp_result, 4).unwrap();
    let tour_length = calculate_tour_length(&tour, &cities);

    println!("   Tour: {tour:?}");
    println!("   Tour length: {tour_length:.2}");

    // Step 5: Convergence analysis
    println!("\n5. Simulating convergence analysis...");
    let mut iteration_results = Vec::new();

    for i in 0..20 {
        let mut iter_samples = create_test_results(50);
        // Simulate improvement over iterations
        for sample in &mut iter_samples {
            sample.energy -= f64::from(i) * 0.5;
        }
        iteration_results.push(iter_samples);
    }

    let conv_data = analyze_convergence(&iteration_results, Some(5)).unwrap();

    println!("   Iterations: {}", conv_data.iterations.len());
    println!("   Initial best energy: {:.2}", conv_data.best_energies[0]);
    println!(
        "   Final best energy: {:.2}",
        conv_data.best_energies.last().unwrap()
    );
    println!(
        "   Improvement: {:.2}",
        conv_data.best_energies[0] - conv_data.best_energies.last().unwrap()
    );

    println!("\n=== Visualization data prepared successfully ===");
    println!("\nNote: The prepared data can be used with external plotting libraries");
    println!("such as matplotlib, plotly, or gnuplot for actual visualization.");
}

/// Example: Using the visualization data with external tools
#[test]
fn example_python_plotting_script() {
    println!("\n=== Example Python Plotting Script ===\n");

    let python_script = r"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load energy landscape data
df_energy = pd.read_csv('/tmp/energy_landscape.csv')

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Energy landscape
ax1.scatter(df_energy['index'], df_energy['energy'], alpha=0.6)
ax1.set_xlabel('Solution Index (sorted)')
ax1.set_ylabel('Energy')
ax1.set_title('Energy Landscape')

# Plot 2: Solution matrix heatmap
df_solutions = pd.read_csv('/tmp/solution_matrix.csv', index_col=0)
im = ax2.imshow(df_solutions.values.T, aspect='auto', cmap='RdBu')
ax2.set_xlabel('Sample')
ax2.set_ylabel('Variable')
ax2.set_title('Solution Matrix')
ax2.set_yticks(range(len(df_solutions.columns)))
ax2.set_yticklabels(df_solutions.columns)
plt.colorbar(im, ax=ax2)

plt.tight_layout()
plt.savefig('/tmp/quantum_annealing_visualization.png', dpi=150)
plt.show()
";

    println!("{python_script}");

    // Save the script for user convenience
    std::fs::write("/tmp/plot_visualization.py", python_script).ok();
    println!("\nScript saved to: /tmp/plot_visualization.py");
    println!("Run with: python /tmp/plot_visualization.py");
}
