// Graph Coloring problem example using quantum annealing
//
// This example demonstrates how to formulate and solve a Graph Coloring problem
// using the quantrs-anneal framework with simulated quantum annealing.
//
// The Graph Coloring problem: Assign colors to vertices of a graph such that
// no two adjacent vertices have the same color, using the minimum number of colors.

use std::time::Instant;
use std::collections::HashMap;

use quantrs_anneal::{
    ising::{IsingModel, QuboModel},
    qubo::{QuboBuilder, Variable},
    simulator::{ClassicalAnnealingSimulator, QuantumAnnealingSimulator, AnnealingParams}
};

fn main() {
    println!("Quantum Annealing - Graph Coloring Problem Example");
    println!("==================================================\n");

    // Create a graph representation for the graph coloring problem
    let graph = create_example_graph();
    println!("Graph has {} vertices and {} edges", graph.num_vertices, graph.edges.len());
    
    // Set the number of colors to use
    let num_colors = 3;
    println!("Attempting to color the graph with {} colors", num_colors);
    
    // Formulate the graph coloring problem as a QUBO
    let (qubo_model, var_map) = formulate_graph_coloring_qubo(&graph, num_colors);
    println!("\nFormulated Graph Coloring problem as QUBO with {} variables", qubo_model.num_variables);
    
    // Convert the QUBO model to an Ising model
    let (ising_model, offset) = qubo_model.to_ising();
    println!("Converted QUBO to Ising model with {} qubits", ising_model.num_qubits);
    println!("Offset from QUBO to Ising conversion: {:.4}\n", offset);
    
    // Solve using classical simulated annealing
    println!("Solving with Classical Simulated Annealing...");
    
    let mut params = AnnealingParams::new();
    params.num_sweeps = 1000;
    params.num_repetitions = 20;
    params.initial_temperature = 5.0; // Higher temperature for harder problems
    
    match ClassicalAnnealingSimulator::new(params) {
        Ok(simulator) => {
            match simulator.solve(&ising_model) {
                Ok(result) => {
                    println!("Classical annealing completed in {:?}", result.runtime);
                    println!("Best energy found: {:.4}", result.best_energy + offset);
                    
                    // Convert spins to binary variables for interpretation
                    let binary_solution = result.best_spins.iter()
                        .map(|&spin| spin > 0)
                        .collect::<Vec<_>>();
                    
                    // Interpret the solution as vertex colors
                    let coloring = interpret_coloring(&graph, &binary_solution, num_colors, &var_map);
                    
                    // Verify the coloring is valid
                    let is_valid = validate_coloring(&graph, &coloring);
                    println!("Valid coloring: {}", is_valid);
                    
                    // Print the coloring
                    println!("\nVertex coloring:");
                    for (vertex_name, color) in &coloring {
                        println!("  {} -> Color {}", vertex_name, color);
                    }
                    
                    // Count violations
                    let violations = count_violations(&graph, &coloring);
                    println!("\nConstraint violations: {}", violations);
                },
                Err(err) => println!("Error solving with classical annealing: {}", err),
            }
        },
        Err(err) => println!("Error creating classical annealing simulator: {}", err),
    }
    
    println!("\nSolving with Quantum Simulated Annealing...");
    
    // Configure parameters for quantum annealing
    let mut params = AnnealingParams::new();
    params.num_sweeps = 500;
    params.num_repetitions = 10;
    params.initial_temperature = 4.0;
    params.initial_transverse_field = 4.0;
    params.trotter_slices = 10;
    
    match QuantumAnnealingSimulator::new(params) {
        Ok(simulator) => {
            match simulator.solve(&ising_model) {
                Ok(result) => {
                    println!("Quantum annealing completed in {:?}", result.runtime);
                    println!("Best energy found: {:.4}", result.best_energy + offset);
                    
                    // Convert spins to binary variables for interpretation
                    let binary_solution = result.best_spins.iter()
                        .map(|&spin| spin > 0)
                        .collect::<Vec<_>>();
                    
                    // Interpret the solution as vertex colors
                    let coloring = interpret_coloring(&graph, &binary_solution, num_colors, &var_map);
                    
                    // Verify the coloring is valid
                    let is_valid = validate_coloring(&graph, &coloring);
                    println!("Valid coloring: {}", is_valid);
                    
                    // Print the coloring
                    println!("\nVertex coloring:");
                    for (vertex_name, color) in &coloring {
                        println!("  {} -> Color {}", vertex_name, color);
                    }
                    
                    // Count violations
                    let violations = count_violations(&graph, &coloring);
                    println!("\nConstraint violations: {}", violations);
                },
                Err(err) => println!("Error solving with quantum annealing: {}", err),
            }
        },
        Err(err) => println!("Error creating quantum annealing simulator: {}", err),
    }
    
    println!("\nNote: The quantum annealing simulation is just a demonstration.");
    println!("For real quantum hardware, enable the 'dwave' feature and use the DWaveClient.");
}

// Define a simple graph structure
struct Graph {
    num_vertices: usize,
    vertices: Vec<String>,
    edges: Vec<(usize, usize)>, // (from, to)
}

// Create an example graph for graph coloring
// This creates a small graph that requires at least 3 colors
fn create_example_graph() -> Graph {
    let vertices = vec![
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
        "D".to_string(),
        "E".to_string(),
    ];
    
    // This graph is essentially a wheel with 5 vertices
    // (A is connected to all others, and the others form a cycle)
    let edges = vec![
        (0, 1), // A-B
        (0, 2), // A-C
        (0, 3), // A-D
        (0, 4), // A-E
        (1, 2), // B-C
        (2, 3), // C-D
        (3, 4), // D-E
        (4, 1), // E-B
    ];
    
    Graph {
        num_vertices: vertices.len(),
        vertices,
        edges,
    }
}

// Formulate the graph coloring problem as a QUBO
fn formulate_graph_coloring_qubo(graph: &Graph, num_colors: usize) -> (QuboModel, HashMap<String, usize>) {
    let mut builder = QuboBuilder::new();
    
    // Create binary variables: x_{v,c} = 1 if vertex v has color c, otherwise 0
    let mut vertex_color_vars: Vec<Vec<Variable>> = Vec::new();
    
    for (v_idx, vertex) in graph.vertices.iter().enumerate() {
        let mut color_vars = Vec::new();
        
        for c in 0..num_colors {
            let var_name = format!("{}_{}", vertex, c);
            let var = builder.add_variable(var_name).unwrap();
            color_vars.push(var);
        }
        
        vertex_color_vars.push(color_vars);
    }
    
    // Set constraint weight
    builder.set_constraint_weight(5.0).unwrap();
    
    // Constraint 1: Each vertex must have exactly one color
    for vertex_vars in &vertex_color_vars {
        builder.constrain_one_hot(vertex_vars).unwrap();
    }
    
    // Constraint 2: Adjacent vertices must have different colors
    for &(v1, v2) in &graph.edges {
        for c in 0..num_colors {
            // Penalize if both endpoints of an edge have the same color
            builder.minimize_quadratic(
                &vertex_color_vars[v1][c],
                &vertex_color_vars[v2][c],
                10.0 // High penalty for color conflicts
            ).unwrap();
        }
    }
    
    // Build the QUBO model
    let model = builder.build();
    let var_map = builder.variable_map();
    
    (model, var_map)
}

// Interpret the binary solution as vertex colors
fn interpret_coloring(
    graph: &Graph,
    solution: &[bool],
    num_colors: usize,
    var_map: &HashMap<String, usize>
) -> HashMap<String, usize> {
    let mut coloring = HashMap::new();
    
    for (v_idx, vertex) in graph.vertices.iter().enumerate() {
        for c in 0..num_colors {
            let var_name = format!("{}_{}", vertex, c);
            let var_idx = var_map[&var_name];
            
            if solution[var_idx] {
                coloring.insert(vertex.clone(), c);
                break;
            }
        }
        
        // If no color was assigned, default to color 0
        if !coloring.contains_key(vertex) {
            coloring.insert(vertex.clone(), 0);
        }
    }
    
    coloring
}

// Validate that the coloring is proper
fn validate_coloring(graph: &Graph, coloring: &HashMap<String, usize>) -> bool {
    for &(v1, v2) in &graph.edges {
        let vertex1 = &graph.vertices[v1];
        let vertex2 = &graph.vertices[v2];
        
        let color1 = *coloring.get(vertex1).unwrap_or(&0);
        let color2 = *coloring.get(vertex2).unwrap_or(&0);
        
        if color1 == color2 {
            return false;
        }
    }
    
    true
}

// Count the number of constraint violations
fn count_violations(graph: &Graph, coloring: &HashMap<String, usize>) -> usize {
    let mut violations = 0;
    
    for &(v1, v2) in &graph.edges {
        let vertex1 = &graph.vertices[v1];
        let vertex2 = &graph.vertices[v2];
        
        let color1 = *coloring.get(vertex1).unwrap_or(&0);
        let color2 = *coloring.get(vertex2).unwrap_or(&0);
        
        if color1 == color2 {
            violations += 1;
        }
    }
    
    violations
}