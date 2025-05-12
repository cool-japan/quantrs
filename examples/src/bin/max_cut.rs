// Maximum Cut (MaxCut) problem example using quantum annealing
//
// This example demonstrates how to formulate and solve a MaxCut problem
// using the quantrs-anneal framework with both simulated quantum annealing
// and classical annealing.
//
// The MaxCut problem: Given a graph, partition the vertices into two sets
// such that the number of edges with endpoints in different partitions is maximized.

use quantrs_anneal::{
    ising::QuboModel,
    qubo::QuboBuilder,
    simulator::{AnnealingParams, ClassicalAnnealingSimulator, QuantumAnnealingSimulator},
};

fn main() {
    println!("Quantum Annealing - Maximum Cut (MaxCut) Problem Example");
    println!("=======================================================\n");

    // Create a graph representation for the MaxCut problem
    let graph = create_example_graph();
    println!(
        "Graph has {} vertices and {} edges",
        graph.num_vertices,
        graph.edges.len()
    );

    // Formulate the MaxCut problem as a QUBO
    let (qubo_model, _var_map) = formulate_maxcut_qubo(&graph);
    println!(
        "\nFormulated MaxCut problem as QUBO with {} variables",
        qubo_model.num_variables
    );

    // Convert the QUBO model to an Ising model
    let (ising_model, offset) = qubo_model.to_ising();
    println!(
        "Converted QUBO to Ising model with {} qubits",
        ising_model.num_qubits
    );
    println!("Offset from QUBO to Ising conversion: {:.4}\n", offset);

    // Solve using classical simulated annealing
    println!("Solving with Classical Simulated Annealing...");
    let _start_time = std::time::Instant::now();

    let mut params = AnnealingParams::new();
    params.num_sweeps = 1000;
    params.num_repetitions = 20;
    params.initial_temperature = 2.0;

    match ClassicalAnnealingSimulator::new(params) {
        Ok(simulator) => {
            match simulator.solve(&ising_model) {
                Ok(result) => {
                    println!("Classical annealing completed in {:?}", result.runtime);
                    println!("Best energy found: {:.4}", result.best_energy + offset);

                    // Convert spins to binary variables for interpretation
                    let binary_solution = result
                        .best_spins
                        .iter()
                        .map(|&spin| spin > 0)
                        .collect::<Vec<_>>();

                    // Calculate the cut size (number of edges across the partition)
                    let cut_size = calculate_cut_size(&graph, &binary_solution);

                    println!(
                        "Cut size: {} out of {} total edges",
                        cut_size,
                        graph.edges.len()
                    );
                    println!(
                        "Partition A: {:?}",
                        graph
                            .vertices
                            .iter()
                            .enumerate()
                            .filter(|&(i, _)| binary_solution[i])
                            .map(|(_, v)| v)
                            .collect::<Vec<_>>()
                    );
                    println!(
                        "Partition B: {:?}",
                        graph
                            .vertices
                            .iter()
                            .enumerate()
                            .filter(|&(i, _)| !binary_solution[i])
                            .map(|(_, v)| v)
                            .collect::<Vec<_>>()
                    );
                }
                Err(err) => println!("Error solving with classical annealing: {}", err),
            }
        }
        Err(err) => println!("Error creating classical annealing simulator: {}", err),
    }

    println!("\nSolving with Quantum Simulated Annealing...");

    // Configure parameters for quantum annealing
    let mut params = AnnealingParams::new();
    params.num_sweeps = 500;
    params.num_repetitions = 10;
    params.initial_temperature = 2.0;
    params.initial_transverse_field = 3.0;
    params.trotter_slices = 10;

    match QuantumAnnealingSimulator::new(params) {
        Ok(simulator) => {
            match simulator.solve(&ising_model) {
                Ok(result) => {
                    println!("Quantum annealing completed in {:?}", result.runtime);
                    println!("Best energy found: {:.4}", result.best_energy + offset);

                    // Convert spins to binary variables for interpretation
                    let binary_solution = result
                        .best_spins
                        .iter()
                        .map(|&spin| spin > 0)
                        .collect::<Vec<_>>();

                    // Calculate the cut size (number of edges across the partition)
                    let cut_size = calculate_cut_size(&graph, &binary_solution);

                    println!(
                        "Cut size: {} out of {} total edges",
                        cut_size,
                        graph.edges.len()
                    );
                    println!(
                        "Partition A: {:?}",
                        graph
                            .vertices
                            .iter()
                            .enumerate()
                            .filter(|&(i, _)| binary_solution[i])
                            .map(|(_, v)| v)
                            .collect::<Vec<_>>()
                    );
                    println!(
                        "Partition B: {:?}",
                        graph
                            .vertices
                            .iter()
                            .enumerate()
                            .filter(|&(i, _)| !binary_solution[i])
                            .map(|(_, v)| v)
                            .collect::<Vec<_>>()
                    );
                }
                Err(err) => println!("Error solving with quantum annealing: {}", err),
            }
        }
        Err(err) => println!("Error creating quantum annealing simulator: {}", err),
    }

    println!("\nNote: The quantum annealing simulation is just a demonstration.");
    println!("For real quantum hardware, enable the 'dwave' feature and use the DWaveClient.");
}

// Define a simple graph structure for MaxCut
struct Graph {
    num_vertices: usize,
    vertices: Vec<String>,
    edges: Vec<(usize, usize, f64)>, // (from, to, weight)
}

// Create an example graph for MaxCut
// This creates a small graph with 5 vertices and 7 edges
fn create_example_graph() -> Graph {
    let vertices = vec![
        "A".to_string(),
        "B".to_string(),
        "C".to_string(),
        "D".to_string(),
        "E".to_string(),
    ];

    let edges = vec![
        (0, 1, 1.0), // A-B
        (0, 2, 1.0), // A-C
        (0, 4, 1.0), // A-E
        (1, 2, 1.0), // B-C
        (1, 3, 1.0), // B-D
        (2, 3, 1.0), // C-D
        (3, 4, 1.0), // D-E
    ];

    Graph {
        num_vertices: vertices.len(),
        vertices,
        edges,
    }
}

// Formulate the MaxCut problem as a QUBO
fn formulate_maxcut_qubo(graph: &Graph) -> (QuboModel, std::collections::HashMap<String, usize>) {
    let mut builder = QuboBuilder::new();

    // Add variables for each vertex
    // Each variable represents whether the vertex is in set 0 or set 1
    let mut variables = Vec::new();
    for vertex in &graph.vertices {
        let var = builder.add_variable(vertex.clone()).unwrap();
        variables.push(var);
    }

    // For each edge (i,j), add the term: -w_ij * (x_i - x_j)^2
    // This encourages vertices connected by an edge to be in different sets
    // Expanding: -w_ij * (x_i - x_j)^2 = -w_ij * (x_i + x_j - 2*x_i*x_j)
    //                                    = -w_ij*x_i - w_ij*x_j + 2*w_ij*x_i*x_j
    for &(i, j, weight) in &graph.edges {
        // Add the linear terms
        builder.minimize_linear(&variables[i], -weight).unwrap();
        builder.minimize_linear(&variables[j], -weight).unwrap();

        // Add the quadratic term
        builder
            .minimize_quadratic(&variables[i], &variables[j], 2.0 * weight)
            .unwrap();
    }

    // Build the QUBO model
    let model = builder.build();
    let var_map = builder.variable_map();

    (model, var_map)
}

// Calculate the cut size for a given solution
fn calculate_cut_size(graph: &Graph, solution: &[bool]) -> usize {
    let mut cut_size = 0;

    for &(i, j, _) in &graph.edges {
        // If the vertices are in different sets, increment the cut size
        if solution[i] != solution[j] {
            cut_size += 1;
        }
    }

    cut_size
}
