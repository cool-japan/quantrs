//! Advanced graph coloring example using QuantRS2-Tytan with SciRS2
//!
//! This example demonstrates:
//! - Graph generation using SciRS2 graph algorithms
//! - QUBO formulation for graph coloring
//! - Advanced visualization and analysis
//! - Performance comparison across different samplers

use quantrs2_tytan::{
    compile::{Compile, Model},
    sampler::{Sampler, SASampler, GaSampler},
    auto_array::auto_array,
    visualization::{
        problem_specific::{ProblemVisualizer, ProblemType},
        solution_analysis::{analyze_solution_distribution, DistributionConfig},
        convergence::plot_convergence,
        energy_landscape::plot_energy_landscape,
    },
    optimization::{
        penalty::{PenaltyOptimizer, PenaltyConfig, PenaltyFunction},
        tuning::{ParameterTuner, TuningConfig, ParameterBounds, ParameterScale},
    },
    analysis::graph::generate_graph,
};
use std::collections::HashMap;
use std::time::Instant;

/// Generate a graph coloring problem
fn create_graph_coloring_model(
    n_nodes: usize, 
    n_colors: usize,
    edge_probability: f64,
) -> Result<(Model, Vec<(usize, usize)>), Box<dyn std::error::Error>> {
    let mut model = Model::new();
    
    // Generate random graph using SciRS2
    let edges = generate_graph(n_nodes, edge_probability)?;
    
    // Create color variables for each node
    let mut color_vars = HashMap::new();
    for node in 0..n_nodes {
        for color in 0..n_colors {
            let var_name = format!("x_{node}_{color}");
            color_vars.insert((node, color), model.add_variable(&var_name)?);
        }
    }
    
    // One-hot constraint: each node must have exactly one color
    for node in 0..n_nodes {
        let constraint_name = format!("one_color_{node}");
        let terms: Vec<_> = (0..n_colors)
            .map(|c| color_vars[&(node, c)].clone())
            .collect();
        model.add_constraint_eq_one(&constraint_name, terms)?;
    }
    
    // Edge constraint: adjacent nodes must have different colors
    for (i, j) in &edges {
        for color in 0..n_colors {
            let constraint_name = format!("edge_{i}_{j}_color_{color}");
            model.add_constraint_at_most_one(
                &constraint_name,
                vec![
                    color_vars[&(*i, color)].clone(),
                    color_vars[&(*j, color)].clone(),
                ],
            )?;
        }
    }
    
    // Minimize the number of colors used (optional objective)
    let mut color_usage_vars = Vec::new();
    for color in 0..n_colors {
        let usage_var = model.add_variable(&format!("color_used_{color}"))?;
        color_usage_vars.push(usage_var.clone());
        
        // If any node uses this color, the usage variable should be 1
        let mut or_terms = Vec::new();
        for node in 0..n_nodes {
            or_terms.push(color_vars[&(node, color)].clone());
        }
        
        // Add implication: if any node uses color, then color_used = 1
        model.add_constraint_implies_any(&format!("color_{color}_usage"), or_terms, usage_var)?;
    }
    
    // Minimize total colors used
    model.set_objective(color_usage_vars.into_iter().sum());
    
    Ok((model, edges))
}

/// Run graph coloring experiment
fn run_graph_coloring_experiment(
    n_nodes: usize,
    n_colors: usize,
    edge_probability: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Advanced Graph Coloring Example ===");
    println!("Nodes: {}, Max Colors: {}, Edge Probability: {:.2}", 
             n_nodes, n_colors, edge_probability);
    
    // Create the model
    let (mut model, edges) = create_graph_coloring_model(n_nodes, n_colors, edge_probability)?;
    
    // Optimize penalty weights using Bayesian optimization
    println!("\nOptimizing penalty weights...");
    let penalty_config = PenaltyConfig {
        penalty_function: PenaltyFunction::Quadratic,
        initial_penalty: 10.0,
        adaptive: true,
        ..Default::default()
    };
    
    let mut penalty_optimizer = PenaltyOptimizer::new(penalty_config);
    let optimized_model = penalty_optimizer.optimize_model(&model)?;
    
    // Compile to QUBO
    let compiled = optimized_model.compile()?;
    let qubo = compiled.to_qubo();
    
    println!("Problem size: {} variables", qubo.size());
    println!("Number of edges: {}", edges.len());
    
    // Parameter tuning for SA sampler
    println!("\nTuning sampler parameters...");
    let parameter_bounds = vec![
        ParameterBounds {
            name: "initial_temp".to_string(),
            min: 1.0,
            max: 100.0,
            scale: ParameterScale::Logarithmic,
            integer: false,
        },
        ParameterBounds {
            name: "final_temp".to_string(),
            min: 0.001,
            max: 1.0,
            scale: ParameterScale::Logarithmic,
            integer: false,
        },
        ParameterBounds {
            name: "num_sweeps".to_string(),
            min: 100.0,
            max: 10000.0,
            scale: ParameterScale::Logarithmic,
            integer: true,
        },
    ];
    
    let tuning_config = TuningConfig {
        max_evaluations: 30,
        initial_samples: 10,
        ..Default::default()
    };
    
    let mut tuner = ParameterTuner::new(tuning_config);
    tuner.add_parameters(parameter_bounds);
    
    let tuning_result = tuner.tune_sampler(
        |params| {
            let mut sampler = SASampler::new();
            sampler.set_parameters(params);
            sampler
        },
        &compiled,
        |samples| {
            // Objective: minimize average energy and maximize solution diversity
            let avg_energy = samples.iter().map(|s| s.energy).sum::<f64>() / samples.len() as f64;
            let unique_solutions = samples.iter()
                .map(|s| format!("{:?}", s.assignments))
                .collect::<std::collections::HashSet<_>>()
                .len();
            let diversity_bonus = (unique_solutions as f64 / samples.len() as f64).min(0.5);
            avg_energy - diversity_bonus
        },
    )?;
    
    println!("Best parameters found:");
    for (name, value) in &tuning_result.best_parameters {
        println!("  {}: {:.4}", name, value);
    }
    println!("Improvement over default: {:.2}%", tuning_result.improvement_over_default * 100.0);
    
    // Run optimized sampler
    println!("\nRunning optimized sampler...");
    let mut sa_sampler = SASampler::new();
    sa_sampler.set_parameters(tuning_result.best_parameters.clone());
    
    let start = Instant::now();
    let sa_samples = sa_sampler.run_qubo(&qubo, 1000)?;
    let sa_time = start.elapsed();
    
    // Also run GA sampler for comparison
    let mut ga_sampler = GaSampler::new();
    ga_sampler.set_parameters(HashMap::from([
        ("population_size".to_string(), 100.0),
        ("mutation_rate".to_string(), 0.1),
        ("crossover_rate".to_string(), 0.8),
        ("generations".to_string(), 200.0),
    ]));
    
    let start = Instant::now();
    let ga_samples = ga_sampler.run_qubo(&qubo, 1000)?;
    let ga_time = start.elapsed();
    
    // Analyze results
    println!("\n=== Results Analysis ===");
    
    // Best solutions
    let sa_best = sa_samples.iter().min_by_key(|s| s.energy as i64).unwrap();
    let ga_best = ga_samples.iter().min_by_key(|s| s.energy as i64).unwrap();
    
    println!("\nSimulated Annealing:");
    println!("  Best energy: {:.4}", sa_best.energy);
    println!("  Time: {:.2}s", sa_time.as_secs_f64());
    println!("  Samples/sec: {:.0}", 1000.0 / sa_time.as_secs_f64());
    
    println!("\nGenetic Algorithm:");
    println!("  Best energy: {:.4}", ga_best.energy);
    println!("  Time: {:.2}s", ga_time.as_secs_f64());
    println!("  Samples/sec: {:.0}", 1000.0 / ga_time.as_secs_f64());
    
    // Extract coloring from best solution
    let best_sample = if sa_best.energy <= ga_best.energy { sa_best } else { ga_best };
    let coloring = extract_coloring(best_sample, n_nodes, n_colors)?;
    
    // Verify solution
    let (valid, colors_used) = verify_coloring(&coloring, &edges, n_colors);
    println!("\nSolution validation:");
    println!("  Valid coloring: {}", valid);
    println!("  Colors used: {}", colors_used);
    
    // Visualizations
    println!("\n=== Generating Visualizations ===");
    
    // 1. Problem-specific visualization
    let mut visualizer = ProblemVisualizer::new();
    visualizer.visualize(
        ProblemType::GraphColoring {
            adjacency: edges_to_adjacency(&edges, n_nodes),
            node_names: Some((0..n_nodes).map(|i| format!("N{}", i)).collect()),
            max_colors: n_colors,
        },
        &sa_samples,
    )?;
    
    // 2. Solution distribution analysis
    println!("\nAnalyzing solution distribution...");
    let distribution_analysis = analyze_solution_distribution(
        sa_samples.clone(),
        Some(DistributionConfig {
            clustering_method: quantrs2_tytan::visualization::solution_analysis::ClusteringMethod::KMeans,
            n_clusters: Some(5),
            ..Default::default()
        }),
    )?;
    
    println!("Solution diversity metrics:");
    println!("  Unique solutions: {}", distribution_analysis.statistics.n_unique);
    println!("  Diversity index: {:.4}", distribution_analysis.diversity_metrics.diversity_index);
    println!("  Entropy: {:.4}", distribution_analysis.diversity_metrics.entropy);
    
    // 3. Energy landscape visualization
    println!("\nVisualizing energy landscape...");
    plot_energy_landscape(sa_samples.clone(), None)?;
    
    // 4. Convergence analysis
    println!("\nGenerating convergence plot...");
    let objectives: Vec<f64> = sa_samples.iter().map(|s| s.energy).collect();
    plot_convergence(objectives, None, None)?;
    
    // Export results
    let results = GraphColoringResults {
        n_nodes,
        n_colors,
        edge_probability,
        edges: edges.clone(),
        best_coloring: coloring,
        colors_used,
        sa_best_energy: sa_best.energy,
        ga_best_energy: ga_best.energy,
        sa_time: sa_time.as_secs_f64(),
        ga_time: ga_time.as_secs_f64(),
        tuned_parameters: tuning_result.best_parameters,
        distribution_analysis: Some(distribution_analysis),
    };
    
    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write("graph_coloring_results.json", json)?;
    println!("\nResults exported to graph_coloring_results.json");
    
    Ok(())
}

/// Extract coloring from solution
fn extract_coloring(
    sample: &quantrs2_tytan::sampler::SampleResult,
    n_nodes: usize,
    n_colors: usize,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut coloring = vec![0; n_nodes];
    
    for node in 0..n_nodes {
        for color in 0..n_colors {
            let var_name = format!("x_{node}_{color}");
            if sample.assignments.get(&var_name).copied().unwrap_or(false) {
                coloring[node] = color;
                break;
            }
        }
    }
    
    Ok(coloring)
}

/// Verify if coloring is valid
fn verify_coloring(
    coloring: &[usize],
    edges: &[(usize, usize)],
    max_colors: usize,
) -> (bool, usize) {
    let mut valid = true;
    
    // Check edge constraints
    for (i, j) in edges {
        if coloring[*i] == coloring[*j] {
            valid = false;
            break;
        }
    }
    
    // Count colors used
    let colors_used = coloring.iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    
    // Check if within color limit
    if colors_used > max_colors {
        valid = false;
    }
    
    (valid, colors_used)
}

/// Convert edge list to adjacency matrix
fn edges_to_adjacency(edges: &[(usize, usize)], n_nodes: usize) -> ndarray::Array2<f64> {
    use ndarray::Array2;
    let mut adjacency = Array2::zeros((n_nodes, n_nodes));
    
    for (i, j) in edges {
        adjacency[[*i, *j]] = 1.0;
        adjacency[[*j, *i]] = 1.0;
    }
    
    adjacency
}

#[derive(serde::Serialize)]
struct GraphColoringResults {
    n_nodes: usize,
    n_colors: usize,
    edge_probability: f64,
    edges: Vec<(usize, usize)>,
    best_coloring: Vec<usize>,
    colors_used: usize,
    sa_best_energy: f64,
    ga_best_energy: f64,
    sa_time: f64,
    ga_time: f64,
    tuned_parameters: HashMap<String, f64>,
    distribution_analysis: Option<quantrs2_tytan::visualization::solution_analysis::DistributionAnalysis>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Small example
    run_graph_coloring_experiment(10, 3, 0.3)?;
    
    // Medium example
    run_graph_coloring_experiment(20, 4, 0.2)?;
    
    // Larger example (if time permits)
    // run_graph_coloring_experiment(30, 5, 0.15)?;
    
    Ok(())
}