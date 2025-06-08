//! Demonstration of advanced visualization capabilities in QuantRS2-Tytan
//!
//! This example shows how to use the visualization module to analyze and
//! visualize quantum annealing results including energy landscapes,
//! solution distributions, and convergence plots.

use quantrs2_tytan::{
    sampler::{SASampler, Sampler, SampleResult},
    visualization::{
        energy_landscape::{EnergyLandscape, EnergyLandscapeConfig, ProjectionMethod},
        solution_analysis::{SolutionDistribution, DistributionConfig, ClusteringMethod},
        convergence::{ConvergencePlot, ConvergenceConfig},
        problem_specific::{ProblemVisualizer, VisualizationType, VisualizationConfig},
        export::{VisualizationExporter, ExportFormat},
    },
};
use ndarray::Array2;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("QuantRS2-Tytan Visualization Demo");
    println!("=================================\n");
    
    // Create a sample problem (Max-Cut on a small graph)
    let problem = create_max_cut_problem();
    
    // Solve the problem
    println!("Solving Max-Cut problem...");
    let samples = solve_problem(&problem)?;
    println!("Found {} solutions\n", samples.len());
    
    // 1. Energy Landscape Visualization
    println!("1. Energy Landscape Visualization");
    println!("---------------------------------");
    visualize_energy_landscape(&samples)?;
    
    // 2. Solution Distribution Analysis
    println!("\n2. Solution Distribution Analysis");
    println!("---------------------------------");
    analyze_solution_distribution(&samples)?;
    
    // 3. Convergence Tracking (simulated)
    println!("\n3. Convergence Visualization");
    println!("-----------------------------");
    visualize_convergence(&samples)?;
    
    // 4. Problem-Specific Visualization
    println!("\n4. Problem-Specific Visualization");
    println!("---------------------------------");
    visualize_max_cut_solution(&samples, &problem)?;
    
    // 5. Export Visualization Data
    println!("\n5. Exporting Visualization Data");
    println!("-------------------------------");
    export_visualization_data(&samples)?;
    
    println!("\nVisualization demo completed!");
    
    Ok(())
}

/// Create a Max-Cut problem
fn create_max_cut_problem() -> MaxCutProblem {
    // Create a simple graph with 6 nodes
    let n_nodes = 6;
    let mut adjacency = Array2::<f64>::zeros((n_nodes, n_nodes));
    
    // Add edges (symmetric)
    let edges = vec![
        (0, 1, 1.0), (0, 2, 1.0),
        (1, 2, 1.0), (1, 3, 1.0),
        (2, 4, 1.0), (3, 4, 1.0),
        (3, 5, 1.0), (4, 5, 1.0),
    ];
    
    for (i, j, weight) in edges {
        adjacency[[i, j]] = weight;
        adjacency[[j, i]] = weight;
    }
    
    // Convert to QUBO
    let mut qubo = Array2::<f64>::zeros((n_nodes, n_nodes));
    
    // Max-Cut QUBO: minimize -sum_{(i,j) in E} w_ij * (x_i + x_j - 2*x_i*x_j)
    for i in 0..n_nodes {
        for j in i+1..n_nodes {
            if adjacency[[i, j]] > 0.0 {
                let w = adjacency[[i, j]];
                // Add linear terms
                qubo[[i, i]] -= w;
                qubo[[j, j]] -= w;
                // Add quadratic term
                qubo[[i, j]] += 2.0 * w;
                qubo[[j, i]] += 2.0 * w;
            }
        }
    }
    
    // Create variable map
    let mut var_map = HashMap::new();
    for i in 0..n_nodes {
        var_map.insert(format!("x_{}", i), i);
    }
    
    MaxCutProblem {
        qubo,
        var_map,
        adjacency,
        node_names: vec!["A", "B", "C", "D", "E", "F"]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
    }
}

/// Solve the problem
fn solve_problem(problem: &MaxCutProblem) -> Result<Vec<SampleResult>, Box<dyn std::error::Error>> {
    let solver = SASampler::new(None);
    let qubo_dyn = problem.qubo.clone().into_dyn();
    let samples = solver.run_hobo(&(qubo_dyn, problem.var_map.clone()), 500)?;
    Ok(samples)
}

/// Visualize energy landscape
fn visualize_energy_landscape(samples: &[SampleResult]) -> Result<(), Box<dyn std::error::Error>> {
    let config = EnergyLandscapeConfig {
        bins: 30,
        projection: ProjectionMethod::PCA,
        colormap: "viridis".to_string(),
        include_density: true,
        resolution: 50,
        energy_limits: None,
    };
    
    let mut landscape = EnergyLandscape::new(config);
    landscape.add_samples(samples.to_vec());
    
    // Generate energy histogram
    let histogram = landscape.energy_histogram()?;
    println!("Energy distribution:");
    println!("  Min energy: {:.3}", samples.iter().map(|s| s.energy).fold(f64::INFINITY, f64::min));
    println!("  Max energy: {:.3}", samples.iter().map(|s| s.energy).fold(f64::NEG_INFINITY, f64::max));
    println!("  Total samples: {}", histogram.total_samples);
    
    // Generate 2D projection
    let projection = landscape.project_2d()?;
    println!("2D projection generated using {}", projection.projection_info);
    
    // Export for external plotting
    #[cfg(not(feature = "scirs"))]
    {
        use quantrs2_tytan::visualization::export::EnergyLandscapeData;
        
        let export_data = EnergyLandscapeData {
            x_coords: projection.x_coords,
            y_coords: projection.y_coords,
            energies: projection.energies,
            density: projection.density_map.map(|d| d.density),
            projection_method: projection.projection_info,
            metadata: HashMap::new(),
        };
        
        let exporter = VisualizationExporter::new(ExportFormat::JSON);
        exporter.export_energy_landscape(&export_data, "energy_landscape.json")?;
        println!("Energy landscape exported to energy_landscape.json");
    }
    
    Ok(())
}

/// Analyze solution distribution
fn analyze_solution_distribution(samples: &[SampleResult]) -> Result<(), Box<dyn std::error::Error>> {
    let config = DistributionConfig {
        clustering_method: ClusteringMethod::KMeans,
        n_clusters: Some(3),
        epsilon: Some(0.5),
        min_samples: Some(5),
        compute_distances: true,
        distance_metric: quantrs2_tytan::visualization::solution_analysis::DistanceMetric::Hamming,
    };
    
    let mut analyzer = SolutionDistribution::new(config);
    analyzer.add_samples(samples.to_vec());
    
    let analysis = analyzer.analyze()?;
    
    println!("Solution statistics:");
    println!("  Total samples: {}", analysis.statistics.n_samples);
    println!("  Unique solutions: {}", analysis.statistics.n_unique);
    println!("  Mean energy: {:.3}", analysis.statistics.mean_energy);
    println!("  Std energy: {:.3}", analysis.statistics.std_energy);
    
    println!("\nDiversity metrics:");
    println!("  Average distance: {:.3}", analysis.diversity_metrics.average_distance);
    println!("  Diversity index: {:.3}", analysis.diversity_metrics.diversity_index);
    println!("  Solution entropy: {:.3}", analysis.diversity_metrics.entropy);
    
    if let Some(cluster_info) = &analysis.cluster_info {
        println!("\nCluster analysis:");
        println!("  Number of clusters: {}", cluster_info.n_clusters);
        for (i, cluster_energy) in cluster_info.cluster_energies.iter().enumerate() {
            println!("  Cluster {}: size={}, mean_energy={:.3}", 
                     i, cluster_info.cluster_sizes[i], cluster_energy.mean_energy);
        }
    }
    
    Ok(())
}

/// Visualize convergence (simulated)
fn visualize_convergence(samples: &[SampleResult]) -> Result<(), Box<dyn std::error::Error>> {
    let config = ConvergenceConfig {
        smoothing_window: 10,
        show_confidence: true,
        confidence_level: 0.95,
        show_constraints: false,
        log_scale: false,
        show_best: true,
        show_bounds: false,
    };
    
    let mut plotter = ConvergencePlot::new(config);
    
    // Simulate convergence by sorting samples by discovery order
    let mut objectives = Vec::new();
    let mut best_so_far = f64::INFINITY;
    
    for (i, sample) in samples.iter().enumerate() {
        best_so_far = best_so_far.min(sample.energy);
        objectives.push(best_so_far);
        
        plotter.add_iteration(
            sample.energy,
            HashMap::new(),
            HashMap::new(),
            std::time::Duration::from_millis(i as u64 * 10),
        );
    }
    
    let analysis = plotter.analyze()?;
    
    println!("Convergence analysis:");
    println!("  Final objective: {:.3}", analysis.final_objective);
    println!("  Best objective: {:.3}", analysis.best_objective);
    println!("  Improvement rate: {:.1}%", analysis.improvement_rate * 100.0);
    println!("  Total iterations: {}", analysis.total_iterations);
    
    if let Some(conv_iter) = analysis.convergence_iteration {
        println!("  Converged at iteration: {}", conv_iter);
    }
    
    println!("  Convergence rate: {:.4}", analysis.convergence_metrics.convergence_rate);
    println!("  Oscillation index: {:.3}", analysis.convergence_metrics.oscillation_index);
    
    // Export convergence data
    plotter.export_plot_data("convergence_data.json")?;
    println!("Convergence data exported to convergence_data.json");
    
    Ok(())
}

/// Visualize Max-Cut solution
fn visualize_max_cut_solution(
    samples: &[SampleResult],
    problem: &MaxCutProblem,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = VisualizationConfig {
        best_only: true,
        top_k: 3,
        color_scheme: "viridis".to_string(),
        node_size: 50.0,
        edge_width: 2.0,
        animate: false,
        animation_speed: 2.0,
    };
    
    let problem_type = VisualizationType::MaxCut {
        adjacency_matrix: problem.adjacency.clone(),
        node_names: Some(problem.node_names.clone()),
    };
    
    let mut visualizer = ProblemVisualizer::new(problem_type, config);
    visualizer.add_samples(samples.to_vec());
    
    // Find best solution
    let best_sample = samples.iter()
        .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
        .unwrap();
    
    // Extract partition
    let mut partition = Vec::new();
    for i in 0..problem.node_names.len() {
        let var_name = format!("x_{}", i);
        partition.push(best_sample.assignments.get(&var_name).copied().unwrap_or(false));
    }
    
    // Calculate cut weight
    let mut cut_weight = 0.0;
    for i in 0..problem.node_names.len() {
        for j in i+1..problem.node_names.len() {
            if problem.adjacency[[i, j]] > 0.0 && partition[i] != partition[j] {
                cut_weight += problem.adjacency[[i, j]];
            }
        }
    }
    
    println!("Best Max-Cut solution:");
    println!("  Cut weight: {:.1}", cut_weight);
    println!("  Partition A: {:?}", 
             problem.node_names.iter().enumerate()
             .filter(|(i, _)| partition[*i])
             .map(|(_, n)| n)
             .collect::<Vec<_>>());
    println!("  Partition B: {:?}",
             problem.node_names.iter().enumerate()
             .filter(|(i, _)| !partition[*i])
             .map(|(_, n)| n)
             .collect::<Vec<_>>());
    
    // Export solution
    #[cfg(not(feature = "scirs"))]
    {
        let export = MaxCutExport {
            adjacency: problem.adjacency.clone(),
            node_names: problem.node_names.clone(),
            best_partition: partition,
            cut_weight,
        };
        
        let json = serde_json::to_string_pretty(&export)?;
        std::fs::write("max_cut_solution.json", json)?;
        println!("Max-Cut solution exported to max_cut_solution.json");
    }
    
    Ok(())
}

/// Export visualization data in multiple formats
fn export_visualization_data(samples: &[SampleResult]) -> Result<(), Box<dyn std::error::Error>> {
    // Export sample data as CSV
    use std::io::Write;
    let mut csv_file = std::fs::File::create("samples.csv")?;
    
    // Write header
    writeln!(csv_file, "sample_id,energy,num_variables")?;
    
    // Write samples
    for (i, sample) in samples.iter().enumerate() {
        writeln!(csv_file, "{},{},{}", i, sample.energy, sample.assignments.len())?;
    }
    
    println!("Sample data exported to samples.csv");
    
    // Create HTML visualization template
    let html_template = r#"<!DOCTYPE html>
<html>
<head>
    <title>QuantRS2-Tytan Visualization Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #333; }
        .info { background: #f0f0f0; padding: 10px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>QuantRS2-Tytan Visualization Results</h1>
    
    <h2>Energy Landscape</h2>
    <div class="info">
        <p>Energy landscape data has been exported to <code>energy_landscape.json</code></p>
        <p>Use your preferred plotting library to visualize the 2D projection.</p>
    </div>
    
    <h2>Solution Distribution</h2>
    <div class="info">
        <p>Solution clustering and diversity analysis completed.</p>
        <p>Check the console output for detailed statistics.</p>
    </div>
    
    <h2>Convergence Analysis</h2>
    <div class="info">
        <p>Convergence data exported to <code>convergence_data.json</code></p>
        <p>The optimization showed good convergence behavior.</p>
    </div>
    
    <h2>Problem-Specific Visualization</h2>
    <div class="info">
        <p>Max-Cut solution exported to <code>max_cut_solution.json</code></p>
        <p>The solution partitions the graph into two sets to maximize the cut weight.</p>
    </div>
</body>
</html>"#;
    
    std::fs::write("visualization_results.html", html_template)?;
    println!("HTML summary exported to visualization_results.html");
    
    Ok(())
}

// Helper structures

struct MaxCutProblem {
    qubo: Array2<f64>,
    var_map: HashMap<String, usize>,
    adjacency: Array2<f64>,
    node_names: Vec<String>,
}

#[derive(serde::Serialize)]
struct MaxCutExport {
    adjacency: Array2<f64>,
    node_names: Vec<String>,
    best_partition: Vec<bool>,
    cut_weight: f64,
}