//! Interactive debugging demonstration.

use quantrs2_tytan::solution_debugger::*;
use quantrs2_tytan::sampler::{SASampler, Sampler};
use std::collections::HashMap;
use ndarray::Array2;
use std::io::{self, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Interactive Solution Debugger Demo ===\n");

    // Create a constraint satisfaction problem
    let problem_info = create_test_problem();
    
    // Create interactive debugger
    let mut debugger = InteractiveDebugger::new(problem_info);
    
    // Start recording session
    debugger.start_recording();
    
    println!("Problem: Graph 3-Coloring");
    println!("Variables: 4 nodes × 3 colors = 12 binary variables");
    println!("Constraints: Adjacent nodes must have different colors\n");
    
    // Solve the problem
    let solution = solve_problem(&debugger.debugger.problem_info)?;
    debugger.load_solution(solution);
    
    println!("Initial solution loaded. Type 'help' for commands.\n");
    
    // Interactive loop
    loop {
        print!("> ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input == "quit" || input == "exit" {
            break;
        }
        
        let output = debugger.execute_command(input);
        println!("{}\n", output);
    }
    
    // Stop recording and save session
    if let Some(session) = debugger.stop_recording() {
        println!("\nSession recorded:");
        println!("  Commands: {}", session.commands.len());
        println!("  Events: {}", session.events.len());
        println!("  Snapshots: {}", session.snapshots.len());
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
            for c2 in c1+1..n_colors {
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
    for (n1, n2) in edges.iter() {
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
            let var_name = format!("x_{}_{}", node, color);
            let idx = node * n_colors + color;
            var_map.insert(var_name.clone(), idx);
            reverse_var_map.insert(idx, var_name);
        }
    }
    
    // Create constraints
    let mut constraints = Vec::new();
    
    // One color per node constraints
    for node in 0..n_nodes {
        let variables: Vec<String> = (0..n_colors)
            .map(|c| format!("x_{}_{}", node, c))
            .collect();
        
        constraints.push(ConstraintInfo {
            name: format!("one_color_node_{}", node),
            constraint_type: ConstraintType::OneHot,
            variables,
            parameters: HashMap::new(),
            penalty: penalty,
            is_hard: true,
        });
    }
    
    // Adjacent nodes different colors
    for (n1, n2) in edges {
        for color in 0..n_colors {
            constraints.push(ConstraintInfo {
                name: format!("edge_{}_{}_{}", n1, n2, color),
                constraint_type: ConstraintType::AtMostK { k: 1 },
                variables: vec![
                    format!("x_{}_{}", n1, color),
                    format!("x_{}_{}", n2, color),
                ],
                parameters: HashMap::new(),
                penalty: penalty,
                is_hard: true,
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

fn solve_problem(problem_info: &ProblemInfo) -> Result<Solution, Box<dyn std::error::Error>> {
    // Use SA to find a solution
    let sampler = SASampler::new(Some(42));
    let results = sampler.run_qubo(
        &(problem_info.qubo.clone(), problem_info.var_map.clone()),
        1000
    )?;
    
    if let Some(best) = results.first() {
        Ok(Solution {
            assignments: best.assignments.clone(),
            objective_value: best.energy,
            timestamp: Some(std::time::SystemTime::now()),
            solver: Some("SA".to_string()),
        })
    } else {
        Err("No solution found".into())
    }
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