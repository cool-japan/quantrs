//! Interactive debugging demonstration.

use quantrs2_tytan::sampler::{SASampler, Sampler};
use quantrs2_tytan::solution_debugger::{
    ConstraintInfo, ConstraintType, DebuggerConfig, ProblemInfo, Solution, SolutionDebugger,
};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Interactive Solution Debugger Demo ===\n");

    // Create a constraint satisfaction problem
    let problem_info = create_test_problem();

    // Create interactive debugger
    let mut config = DebuggerConfig::default();
    let mut debugger = SolutionDebugger::new(problem_info, config);

    println!("Problem: Graph 3-Coloring");
    println!("Variables: 4 nodes × 3 colors = 12 binary variables");
    println!("Constraints: Adjacent nodes must have different colors\n");

    // Solve the problem (get a sample solution)
    let solution = solve_problem()?;

    // Debug the solution
    println!("Analyzing solution...");
    let debug_report = debugger.debug_solution(&solution);

    // Print debug results
    println!("\n=== Debug Report ===");
    println!("Overall Score: {:.2}", debug_report.summary.overall_score);

    if let Some(constraint_analysis) = &debug_report.constraint_analysis {
        println!("\nConstraint Analysis:");
        println!(
            "  Total constraints: {}",
            constraint_analysis.total_constraints
        );
        println!("  Satisfied: {}", constraint_analysis.satisfied);
        println!("  Violated: {}", constraint_analysis.violated);
        println!(
            "  Satisfaction rate: {:.1}%",
            constraint_analysis.satisfaction_rate * 100.0
        );
        println!(
            "  Penalty incurred: {:.4}",
            constraint_analysis.penalty_incurred
        );
    }

    if let Some(energy_analysis) = &debug_report.energy_analysis {
        println!("\nEnergy Analysis:");
        println!("  Total energy: {:.4}", energy_analysis.total_energy);
        println!(
            "  Improvement potential: {:.2}%",
            energy_analysis.improvement_potential * 100.0
        );

        println!("\n  Critical variables:");
        for (i, (var, contrib)) in energy_analysis
            .critical_variables
            .iter()
            .take(5)
            .enumerate()
        {
            println!("    {}: {} (contribution: {:.4})", i + 1, var, contrib);
        }
    }

    println!("\nIssues found: {}", debug_report.issues.len());
    for (i, issue) in debug_report.issues.iter().enumerate() {
        println!(
            "  {}: [{:?}] {} - {}",
            i + 1,
            issue.severity,
            issue.category,
            issue.description
        );
    }

    println!("\nSuggestions: {}", debug_report.suggestions.len());
    for (i, suggestion) in debug_report.suggestions.iter().enumerate() {
        println!(
            "  {}: {} (impact: {:.4})",
            i + 1,
            suggestion.description,
            suggestion.impact
        );
    }

    Ok(())
}

fn create_test_problem() -> ProblemInfo {
    // Graph coloring: 4 nodes, 3 colors
    // Graph structure: 0-1-2-3 with 0-2 edge (square)

    let n_nodes = 4;
    let n_colors = 3;
    let n_vars = n_nodes * n_colors;

    // Create QUBO matrix
    let mut qubo = Array2::zeros((n_vars, n_vars));

    // Penalty for not choosing exactly one color per node
    let penalty = 10.0;
    for node in 0..n_nodes {
        // Quadratic penalties for choosing multiple colors
        for c1 in 0..n_colors {
            for c2 in c1 + 1..n_colors {
                let var1 = node * n_colors + c1;
                let var2 = node * n_colors + c2;
                qubo[[var1, var2]] = penalty;
                qubo[[var2, var1]] = penalty;
            }
            // Linear penalty for not choosing any color
            let var = node * n_colors + c1;
            qubo[[var, var]] = -penalty;
        }
    }

    // Penalty for adjacent nodes having same color
    let edges = vec![(0, 1), (1, 2), (2, 3), (0, 2)];
    for (n1, n2) in &edges {
        for color in 0..n_colors {
            let var1 = n1 * n_colors + color;
            let var2 = n2 * n_colors + color;
            qubo[[var1, var2]] += penalty;
            qubo[[var2, var1]] += penalty;
        }
    }

    // Create variable mapping
    let mut var_map = HashMap::new();
    let mut reverse_var_map = HashMap::new();
    for node in 0..n_nodes {
        for color in 0..n_colors {
            let var_name = format!("x_{node}_{color}");
            let idx = node * n_colors + color;
            var_map.insert(var_name.clone(), idx);
            reverse_var_map.insert(idx, var_name);
        }
    }

    // Create constraints
    let mut constraints = Vec::new();

    // One color per node constraints
    for node in 0..n_nodes {
        let variables: Vec<String> = (0..n_colors).map(|c| format!("x_{node}_{c}")).collect();

        constraints.push(ConstraintInfo {
            name: Some(format!("one_color_node_{node}")),
            constraint_type: ConstraintType::ExactlyOne,
            variables,
            parameters: HashMap::new(),
            penalty,
            description: Some("Each node must have exactly one color".to_string()),
        });
    }

    // Adjacent nodes different colors
    for (n1, n2) in &edges {
        for color in 0..n_colors {
            constraints.push(ConstraintInfo {
                name: Some(format!("edge_{n1}_{n2}_{color}")),
                constraint_type: ConstraintType::AtMostOne,
                variables: vec![format!("x_{}_{}", n1, color), format!("x_{}_{}", n2, color)],
                parameters: HashMap::new(),
                penalty,
                description: Some("Adjacent nodes cannot have the same color".to_string()),
            });
        }
    }

    ProblemInfo {
        name: "Graph 3-Coloring".to_string(),
        problem_type: "CSP".to_string(),
        num_variables: n_vars,
        var_map,
        reverse_var_map,
        qubo,
        constraints,
        optimal_solution: None,
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("nodes".to_string(), n_nodes.to_string());
            meta.insert("colors".to_string(), n_colors.to_string());
            meta.insert("edges".to_string(), edges.len().to_string());
            meta
        },
    }
}

fn solve_problem() -> Result<Solution, Box<dyn std::error::Error>> {
    // Create a simple example solution for the graph coloring problem
    let mut assignments = HashMap::new();

    // Simple valid coloring: node 0=color0, node 1=color1, node 2=color2, node 3=color0
    for node in 0..4 {
        for color in 0..3 {
            let var_name = format!("x_{node}_{color}");
            let should_assign = match node {
                0 => color == 0, // node 0 gets color 0
                1 => color == 1, // node 1 gets color 1
                2 => color == 2, // node 2 gets color 2
                3 => color == 0, // node 3 gets color 0 (different from neighbors 1,2)
                _ => false,
            };
            assignments.insert(var_name, should_assign);
        }
    }

    Ok(Solution {
        assignments,
        energy: -1.0, // placeholder energy
        quality_metrics: HashMap::new(),
        metadata: HashMap::new(),
        sampling_stats: None,
    })
}

// Example session demonstrating features:
/*
> help
Available commands:
  analyze      - Analyze current solution
  constraints  - Show problem constraints
  energy       - Show energy breakdown
  flip <var>   - Flip variable value
  compare      - Compare solutions
  suggest      - Show improvement suggestions
  watch [var]  - Add/show watch variables
  break [type] - Add/show breakpoints
  history      - Show command history
  undo         - Undo last change
  path         - Analyze solution path
  sensitivity  - Run sensitivity analysis
  export <fmt> - Export analysis (json/csv/html)
  help         - Show this help message

> analyze
=== Solution Debug Report ===
Problem: Graph 3-Coloring
Variables: 12

Summary:
  Quality: Good
  Energy: -120.0000
  Constraint satisfaction: 100.0%

> constraints
Constraints:
  1. one_color_node_0 (OneHot)
  2. one_color_node_1 (OneHot)
  3. one_color_node_2 (OneHot)
  4. one_color_node_3 (OneHot)
  5. edge_0_1_0 (AtMostK { k: 1 })
  ...

> watch x_0_0
Added 'x_0_0' to watch list

> flip x_0_0
Flipped x_0_0 from true to false. New energy: -100.0000

Watched variables:
  x_0_0: false

> suggest
Suggestions:
  1. Fix constraint 'one_color_node_0'
     Variable 'x_0_0' contributes -10.00 to energy
  2. Flip variable 'x_0_1'
     Variable 'x_0_1' contributes 10.00 to energy

> undo
Undid last change

> sensitivity
Sensitivity analysis:
  x_0_0: impact = +20.0000 (current: true)
  x_0_1: impact = -10.0000 (current: false)
  x_0_2: impact = -10.0000 (current: false)
  ...

> path
Solution path analysis:
Starting energy: -120.0000
  Step 1: Energy = -100.0000 (change: +20.0000)
Current energy: -120.0000
Total change: +0.0000

> export json
Analysis exported to JSON format

> quit
*/
