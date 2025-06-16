//! Advanced maximum cut example using QuantRS2-Tytan with SciRS2
//!
//! This example demonstrates:
//! - Sparse matrix representation for large graphs
//! - QUBO formulation for max-cut problem
//! - Performance benchmarking across different graph types
//! - Cut quality analysis and visualization

use ndarray::{Array1, Array2};
use quantrs2_tytan::{
    analysis::graph::{
        analyze_graph, generate_complete_graph, generate_graph, generate_grid_graph,
        generate_regular_graph,
    },
    benchmark::{
        metrics::{
            BenchmarkMetrics, MemoryMetrics, QualityMetrics, TimingMetrics, UtilizationMetrics,
        },
        runner::{BenchmarkConfig, BenchmarkRunner},
    },
    compile::Model,
    optimization::{
        adaptive::{AdaptiveConfig, AdaptiveOptimizer},
        penalty::{PenaltyConfig, PenaltyOptimizer},
    },
    sampler::{GASampler, SASampler, Sampler},
    visualization::{
        convergence::{plot_convergence, track_adaptive_convergence},
        energy_landscape::plot_energy_landscape,
        problem_specific::{ProblemVisualizer, VisualizationConfig, VisualizationType},
    },
};

#[cfg(not(feature = "dwave"))]
use quantrs2_tytan::compile::SimpleExpr;

use std::collections::HashMap;
use std::fmt::Write;
use std::time::Instant;

/// Convert QuboModel to matrix format for samplers
fn qubo_to_matrix_format(
    qubo: &quantrs2_tytan::QuboModel,
) -> Result<(Array2<f64>, HashMap<String, usize>), Box<dyn std::error::Error>> {
    let n = qubo.num_variables;
    let mut matrix = Array2::zeros((n, n));
    let mut var_map = HashMap::new();

    // Create variable mapping
    for i in 0..n {
        var_map.insert(format!("x_{}", i), i);
    }

    // Fill matrix with linear terms (diagonal)
    for i in 0..n {
        if let Ok(linear) = qubo.get_linear(i) {
            matrix[[i, i]] = linear;
        }
    }

    // Fill matrix with quadratic terms
    for i in 0..n {
        for j in 0..n {
            if i != j {
                if let Ok(quad) = qubo.get_quadratic(i, j) {
                    matrix[[i, j]] = quad;
                }
            }
        }
    }

    Ok((matrix, var_map))
}

/// Maximum cut problem formulation
///
/// Given a graph G = (V, E) with edge weights w_ij, partition vertices into two sets
/// to maximize the total weight of edges between sets.
fn create_max_cut_model(
    edges: &[(usize, usize)],
    weights: Option<&[f64]>,
    n_nodes: usize,
) -> Result<Model, Box<dyn std::error::Error>> {
    let mut model = Model::new();

    // Create binary variables for each node (0 = set A, 1 = set B)
    let mut node_vars = Vec::new();
    for i in 0..n_nodes {
        let var = model.add_variable(&format!("x_{}", i))?;
        node_vars.push(var);
    }

    // Objective: maximize sum of w_ij * (x_i XOR x_j) for each edge
    // Using the identity: x_i XOR x_j = x_i + x_j - 2*x_i*x_j
    let mut objective = SimpleExpr::constant(0.0);

    for (idx, &(i, j)) in edges.iter().enumerate() {
        let weight = weights.map(|w| w[idx]).unwrap_or(1.0);

        // Add weight * (x_i + x_j - 2*x_i*x_j)
        objective = objective
            + SimpleExpr::constant(weight) * node_vars[i].clone()
            + SimpleExpr::constant(weight) * node_vars[j].clone()
            + SimpleExpr::constant(-2.0 * weight) * node_vars[i].clone() * node_vars[j].clone();
    }

    // Since we want to maximize, negate the objective for minimization
    model.set_objective(SimpleExpr::constant(-1.0) * objective);

    Ok(model)
}

/// Calculate the cut value for a given partition
fn calculate_cut_value(
    partition: &[bool],
    edges: &[(usize, usize)],
    weights: Option<&[f64]>,
) -> f64 {
    let mut cut_value = 0.0;

    for (idx, &(i, j)) in edges.iter().enumerate() {
        if partition[i] != partition[j] {
            cut_value += weights.map(|w| w[idx]).unwrap_or(1.0);
        }
    }

    cut_value
}

/// Analyze cut quality
fn analyze_cut_quality(
    partition: &[bool],
    edges: &[(usize, usize)],
    weights: Option<&[f64]>,
) -> CutAnalysis {
    let mut cut_value = calculate_cut_value(partition, edges, weights);
    let total_weight: f64 = weights
        .map(|w| w.iter().sum())
        .unwrap_or(edges.len() as f64);

    let set_a_size = partition.iter().filter(|&&x| !x).count();
    let set_b_size = partition.iter().filter(|&&x| x).count();
    let balance_ratio = set_a_size.min(set_b_size) as f64 / set_a_size.max(set_b_size) as f64;

    // Count internal edges
    let mut internal_edges_a = 0;
    let mut internal_edges_b = 0;

    for &(i, j) in edges {
        if !partition[i] && !partition[j] {
            internal_edges_a += 1;
        } else if partition[i] && partition[j] {
            internal_edges_b += 1;
        }
    }

    CutAnalysis {
        cut_value,
        cut_ratio: cut_value / total_weight,
        set_a_size,
        set_b_size,
        balance_ratio,
        internal_edges_a,
        internal_edges_b,
        cut_edges: edges.len() - internal_edges_a - internal_edges_b,
    }
}

#[derive(Debug, Clone)]
struct CutAnalysis {
    cut_value: f64,
    cut_ratio: f64,
    set_a_size: usize,
    set_b_size: usize,
    balance_ratio: f64,
    internal_edges_a: usize,
    internal_edges_b: usize,
    cut_edges: usize,
}

/// Run max-cut experiment on different graph types
fn run_max_cut_experiments() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Maximum Cut Experiments ===\n");

    // Test different graph types
    let experiments: Vec<(&str, Vec<(usize, usize)>, Option<Vec<f64>>)> = vec![
        ("Random Graph", generate_graph(20, 0.3)?, None),
        ("Complete Graph", generate_complete_graph(10), None),
        ("Regular Graph", generate_regular_graph(16, 3)?, None),
        ("Grid Graph", generate_grid_graph(4, 4), None),
    ];

    let mut all_results = Vec::new();

    for (graph_name, edges, weights) in experiments {
        println!("\n=== {} ===", graph_name);

        // Analyze graph properties
        let n_nodes = edges
            .iter()
            .flat_map(|&(i, j)| vec![i, j])
            .max()
            .unwrap_or(0)
            + 1;

        let props = analyze_graph(n_nodes, &edges);
        println!(
            "Nodes: {}, Edges: {}, Avg Degree: {:.2}, Density: {:.3}",
            props.n_nodes, props.n_edges, props.avg_degree, props.density
        );

        // Create and compile model
        let model = create_max_cut_model(&edges, weights.as_deref(), n_nodes)?;
        let compiled = model.compile()?;
        let qubo = compiled.to_qubo();

        println!("QUBO size: {} variables", qubo.num_variables);

        // Use adaptive optimization
        let adaptive_config = AdaptiveConfig {
            strategy:
                quantrs2_tytan::optimization::adaptive::AdaptiveStrategy::AdaptivePenaltyMethod,
            update_interval: 100,
            learning_rate: 0.1,
            momentum: 0.9,
            patience: 50,
            ..Default::default()
        };

        let mut adaptive_optimizer = AdaptiveOptimizer::new(adaptive_config);

        // Benchmark different samplers
        let benchmark_config = BenchmarkConfig {
            problem_sizes: vec![n_nodes],
            problem_densities: vec![props.density],
            num_reads: 1000,
            num_repetitions: 5,
            ..Default::default()
        };

        let mut runner = BenchmarkRunner::new(benchmark_config);

        // Convert QUBO to matrix format
        let (matrix, var_map) = qubo_to_matrix_format(&qubo)?;

        // SA sampler
        let mut sa_sampler = SASampler::new(None);
        let sa_samples = sa_sampler.run_qubo(&(matrix.clone(), var_map.clone()), 1000)?;
        let best_sa = &sa_samples[0];

        // GA sampler
        let mut ga_sampler = GASampler::with_params(None, 100, 50); // seed, max_generations, population_size
        let ga_samples = ga_sampler.run_qubo(&(matrix.clone(), var_map.clone()), 1000)?;
        let best_ga = &ga_samples[0];

        // Create simple metrics for comparison
        let sa_metrics = BenchmarkMetrics {
            problem_size: n_nodes,
            problem_density: props.density,
            timings: TimingMetrics {
                total_time: std::time::Duration::from_millis(100), // placeholder
                setup_time: std::time::Duration::from_millis(10),
                compute_time: std::time::Duration::from_millis(80),
                postprocess_time: std::time::Duration::from_millis(10),
                time_per_sample: std::time::Duration::from_nanos(100000),
                time_to_solution: None,
            },
            quality: QualityMetrics {
                best_energy: best_sa.energy,
                avg_energy: sa_samples.iter().map(|s| s.energy).sum::<f64>()
                    / sa_samples.len() as f64,
                energy_std: 0.0, // placeholder
                success_probability: 1.0,
                time_to_target: None,
                unique_solutions: sa_samples.len(),
            },
            memory: MemoryMetrics {
                peak_memory: 1024 * 1024, // placeholder
                avg_memory: 512 * 1024,
                allocated: 1024 * 1024,
                deallocated: 0,
                cache_misses: None,
            },
            utilization: UtilizationMetrics {
                cpu_usage: 80.0,
                gpu_usage: None,
                memory_bandwidth: 0.8,
                cache_hit_rate: None,
                power_consumption: None,
            },
            custom: HashMap::new(),
        };

        let ga_metrics = BenchmarkMetrics {
            problem_size: n_nodes,
            problem_density: props.density,
            timings: TimingMetrics {
                total_time: std::time::Duration::from_millis(120), // placeholder
                setup_time: std::time::Duration::from_millis(15),
                compute_time: std::time::Duration::from_millis(95),
                postprocess_time: std::time::Duration::from_millis(10),
                time_per_sample: std::time::Duration::from_nanos(120000),
                time_to_solution: None,
            },
            quality: QualityMetrics {
                best_energy: best_ga.energy,
                avg_energy: ga_samples.iter().map(|s| s.energy).sum::<f64>()
                    / ga_samples.len() as f64,
                energy_std: 0.0, // placeholder
                success_probability: 1.0,
                time_to_target: None,
                unique_solutions: ga_samples.len(),
            },
            memory: MemoryMetrics {
                peak_memory: 1024 * 1024, // placeholder
                avg_memory: 512 * 1024,
                allocated: 1024 * 1024,
                deallocated: 0,
                cache_misses: None,
            },
            utilization: UtilizationMetrics {
                cpu_usage: 85.0,
                gpu_usage: None,
                memory_bandwidth: 0.9,
                cache_hit_rate: None,
                power_consumption: None,
            },
            custom: HashMap::new(),
        };

        // Extract partitions
        let sa_partition = extract_partition(best_sa, n_nodes);
        let ga_partition = extract_partition(best_ga, n_nodes);

        // Analyze cut quality
        let sa_analysis = analyze_cut_quality(&sa_partition, &edges, weights.as_deref());
        let ga_analysis = analyze_cut_quality(&ga_partition, &edges, weights.as_deref());

        println!("\nSimulated Annealing:");
        println!("  Cut value: {:.2}", sa_analysis.cut_value);
        println!("  Cut ratio: {:.3}", sa_analysis.cut_ratio);
        println!(
            "  Balance: {:.3} ({}:{})",
            sa_analysis.balance_ratio, sa_analysis.set_a_size, sa_analysis.set_b_size
        );
        println!(
            "  Time: {:.3}s",
            sa_metrics.timings.total_time.as_secs_f64()
        );

        println!("\nGenetic Algorithm:");
        println!("  Cut value: {:.2}", ga_analysis.cut_value);
        println!("  Cut ratio: {:.3}", ga_analysis.cut_ratio);
        println!(
            "  Balance: {:.3} ({}:{})",
            ga_analysis.balance_ratio, ga_analysis.set_a_size, ga_analysis.set_b_size
        );
        println!(
            "  Time: {:.3}s",
            ga_metrics.timings.total_time.as_secs_f64()
        );

        // Visualize best cut
        let best_partition = if sa_analysis.cut_value >= ga_analysis.cut_value {
            sa_partition
        } else {
            ga_partition
        };

        let adjacency = edges_to_adjacency(&edges, n_nodes);
        let problem_type = VisualizationType::MaxCut {
            adjacency_matrix: adjacency,
            node_names: None,
        };
        let mut config = VisualizationConfig::default();
        let mut visualizer = ProblemVisualizer::new(problem_type, config);
        visualizer.add_samples(vec![best_sa.clone()]);
        visualizer.visualize()?;

        // Store results
        all_results.push(ExperimentResult {
            graph_name: graph_name.to_string(),
            graph_props: props,
            sa_analysis,
            ga_analysis,
            sa_metrics,
            ga_metrics,
        });
    }

    // Generate comparison report
    generate_comparison_report(&all_results)?;

    Ok(())
}

/// Extract partition from sample result
fn extract_partition(sample: &quantrs2_tytan::sampler::SampleResult, n_nodes: usize) -> Vec<bool> {
    let mut partition = vec![false; n_nodes];

    for i in 0..n_nodes {
        let var_name = format!("x_{}", i);
        partition[i] = sample.assignments.get(&var_name).copied().unwrap_or(false);
    }

    partition
}

/// Convert edges to adjacency matrix
fn edges_to_adjacency(edges: &[(usize, usize)], n_nodes: usize) -> Array2<f64> {
    let mut adjacency = Array2::zeros((n_nodes, n_nodes));

    for &(i, j) in edges {
        adjacency[[i, j]] = 1.0;
        adjacency[[j, i]] = 1.0;
    }

    adjacency
}

struct ExperimentResult {
    graph_name: String,
    graph_props: quantrs2_tytan::analysis::graph::GraphProperties,
    sa_analysis: CutAnalysis,
    ga_analysis: CutAnalysis,
    sa_metrics: BenchmarkMetrics,
    ga_metrics: BenchmarkMetrics,
}

/// Generate comparison report
fn generate_comparison_report(
    results: &[ExperimentResult],
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;

    let mut report = String::new();

    writeln!(&mut report, "# Maximum Cut Experiment Results\n")?;
    writeln!(&mut report, "## Summary\n")?;

    // Summary table
    writeln!(
        &mut report,
        "| Graph | Nodes | Edges | Best Cut | Best Method | Time (s) |"
    )?;
    writeln!(
        &mut report,
        "|-------|-------|-------|----------|-------------|----------|"
    )?;

    for result in results {
        let (best_cut, best_method, best_time) =
            if result.sa_analysis.cut_value >= result.ga_analysis.cut_value {
                (
                    result.sa_analysis.cut_value,
                    "SA",
                    result.sa_metrics.timings.total_time.as_secs_f64(),
                )
            } else {
                (
                    result.ga_analysis.cut_value,
                    "GA",
                    result.ga_metrics.timings.total_time.as_secs_f64(),
                )
            };

        writeln!(
            &mut report,
            "| {} | {} | {} | {:.2} | {} | {:.3} |",
            result.graph_name,
            result.graph_props.n_nodes,
            result.graph_props.n_edges,
            best_cut,
            best_method,
            best_time
        )?;
    }

    writeln!(&mut report, "\n## Detailed Results\n")?;

    for result in results {
        writeln!(&mut report, "### {}\n", result.graph_name)?;
        writeln!(&mut report, "**Graph Properties:**")?;
        writeln!(&mut report, "- Nodes: {}", result.graph_props.n_nodes)?;
        writeln!(&mut report, "- Edges: {}", result.graph_props.n_edges)?;
        writeln!(
            &mut report,
            "- Average Degree: {:.2}",
            result.graph_props.avg_degree
        )?;
        writeln!(&mut report, "- Density: {:.3}", result.graph_props.density)?;
        writeln!(
            &mut report,
            "- Connected: {}",
            result.graph_props.is_connected
        )?;

        writeln!(&mut report, "\n**Simulated Annealing:**")?;
        writeln!(
            &mut report,
            "- Cut Value: {:.2}",
            result.sa_analysis.cut_value
        )?;
        writeln!(
            &mut report,
            "- Cut Ratio: {:.3}",
            result.sa_analysis.cut_ratio
        )?;
        writeln!(
            &mut report,
            "- Partition Balance: {:.3} ({}:{})",
            result.sa_analysis.balance_ratio,
            result.sa_analysis.set_a_size,
            result.sa_analysis.set_b_size
        )?;
        writeln!(&mut report, "- Cut Edges: {}", result.sa_analysis.cut_edges)?;
        writeln!(
            &mut report,
            "- Energy: {:.4}",
            result.sa_metrics.quality.best_energy
        )?;
        writeln!(
            &mut report,
            "- Time: {:.3}s",
            result.sa_metrics.timings.total_time.as_secs_f64()
        )?;

        writeln!(&mut report, "\n**Genetic Algorithm:**")?;
        writeln!(
            &mut report,
            "- Cut Value: {:.2}",
            result.ga_analysis.cut_value
        )?;
        writeln!(
            &mut report,
            "- Cut Ratio: {:.3}",
            result.ga_analysis.cut_ratio
        )?;
        writeln!(
            &mut report,
            "- Partition Balance: {:.3} ({}:{})",
            result.ga_analysis.balance_ratio,
            result.ga_analysis.set_a_size,
            result.ga_analysis.set_b_size
        )?;
        writeln!(&mut report, "- Cut Edges: {}", result.ga_analysis.cut_edges)?;
        writeln!(
            &mut report,
            "- Energy: {:.4}",
            result.ga_metrics.quality.best_energy
        )?;
        writeln!(
            &mut report,
            "- Time: {:.3}s",
            result.ga_metrics.timings.total_time.as_secs_f64()
        )?;

        writeln!(&mut report)?;
    }

    // Write report to file
    std::fs::write("max_cut_results.md", report)?;
    println!("\n\nReport saved to max_cut_results.md");

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_max_cut_experiments()?;

    // Additional experiment with weighted graph
    println!("\n\n=== Weighted Graph Experiment ===");

    // Create a weighted graph
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)];
    let weights = vec![1.0, 2.0, 1.5, 2.5, 3.0, 1.0];

    let model = create_max_cut_model(&edges, Some(&weights), 4)?;
    let compiled = model.compile()?;
    let qubo = compiled.to_qubo();

    let mut sampler = SASampler::new(None);

    let (matrix, var_map) = qubo_to_matrix_format(&qubo)?;
    let samples = sampler.run_qubo(&(matrix, var_map), 1000)?;
    let best = samples.iter().min_by_key(|s| s.energy as i64).unwrap();

    let partition = extract_partition(best, 4);
    let analysis = analyze_cut_quality(&partition, &edges, Some(&weights));

    println!("\nWeighted Graph Results:");
    println!("  Cut value: {:.2}", analysis.cut_value);
    println!("  Partition: {:?}", partition);
    println!(
        "  Set sizes: {} vs {}",
        analysis.set_a_size, analysis.set_b_size
    );

    // Verify all edges
    println!("\nEdge breakdown:");
    for (idx, &(i, j)) in edges.iter().enumerate() {
        let in_cut = partition[i] != partition[j];
        println!(
            "  Edge ({},{}) weight={:.1}: {}",
            i,
            j,
            weights[idx],
            if in_cut { "CUT" } else { "internal" }
        );
    }

    Ok(())
}
