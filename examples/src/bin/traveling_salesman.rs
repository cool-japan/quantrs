// Traveling Salesman Problem (TSP) example using quantum annealing
//
// This example demonstrates how to formulate and solve a TSP
// using the quantrs-anneal framework with simulated quantum annealing.
//
// The TSP: Find the shortest possible route that visits each city exactly once
// and returns to the starting city.

use quantrs2_anneal::{
    ising::QuboModel,
    qubo::{QuboBuilder, Variable},
    simulator::{AnnealingParams, ClassicalAnnealingSimulator, QuantumAnnealingSimulator},
};
use std::collections::HashMap;

fn main() {
    println!("Quantum Annealing - Traveling Salesman Problem Example");
    println!("=====================================================\n");

    // Create a TSP instance
    let tsp = create_example_tsp();
    println!("TSP instance with {} cities", tsp.num_cities);

    // Print the distance matrix
    println!("\nDistance matrix:");
    for i in 0..tsp.num_cities {
        let row = (0..tsp.num_cities)
            .map(|j| format!("{:4.1}", tsp.distances[i][j]))
            .collect::<Vec<_>>()
            .join(" ");
        println!("  {}", row);
    }

    // Formulate the TSP as a QUBO
    let (qubo_model, var_map) = formulate_tsp_qubo(&tsp);
    println!(
        "\nFormulated TSP as QUBO with {} variables",
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

    let mut params = AnnealingParams::new();
    params.num_sweeps = 2000;
    params.num_repetitions = 30;
    params.initial_temperature = 10.0; // Higher temperature for harder problems

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

                    // Interpret the solution as a tour
                    let tour = interpret_tour(&tsp, &binary_solution, &var_map);

                    // Validate the tour
                    let (is_valid, violations) = validate_tour(&tsp, &tour);
                    println!("Valid tour: {}", is_valid);

                    // Print the tour
                    if !tour.is_empty() {
                        print_tour(&tsp, &tour);
                        // Calculate tour length
                        let tour_length = calculate_tour_length(&tsp, &tour);
                        println!("Tour length: {:.2}", tour_length);
                    } else {
                        println!("No valid tour found");
                    }

                    if violations > 0 {
                        println!("Constraint violations: {}", violations);
                    }
                }
                Err(err) => println!("Error solving with classical annealing: {}", err),
            }
        }
        Err(err) => println!("Error creating classical annealing simulator: {}", err),
    }

    println!("\nSolving with Quantum Simulated Annealing...");

    // Configure parameters for quantum annealing
    let mut params = AnnealingParams::new();
    params.num_sweeps = 1000;
    params.num_repetitions = 15;
    params.initial_temperature = 8.0;
    params.initial_transverse_field = 5.0;
    params.trotter_slices = 8;

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

                    // Interpret the solution as a tour
                    let tour = interpret_tour(&tsp, &binary_solution, &var_map);

                    // Validate the tour
                    let (is_valid, violations) = validate_tour(&tsp, &tour);
                    println!("Valid tour: {}", is_valid);

                    // Print the tour
                    if !tour.is_empty() {
                        print_tour(&tsp, &tour);
                        // Calculate tour length
                        let tour_length = calculate_tour_length(&tsp, &tour);
                        println!("Tour length: {:.2}", tour_length);
                    } else {
                        println!("No valid tour found");
                    }

                    if violations > 0 {
                        println!("Constraint violations: {}", violations);
                    }
                }
                Err(err) => println!("Error solving with quantum annealing: {}", err),
            }
        }
        Err(err) => println!("Error creating quantum annealing simulator: {}", err),
    }

    println!("\nNote: The quantum annealing simulation is just a demonstration.");
    println!("The TSP is a hard problem that requires many qubits for larger instances.");
    println!("For real quantum hardware, enable the 'dwave' feature and use the DWaveClient.");
}

// TSP instance
struct Tsp {
    num_cities: usize,
    city_names: Vec<String>,
    distances: Vec<Vec<f64>>, // distance matrix
}

// Create a small example TSP instance
fn create_example_tsp() -> Tsp {
    // We'll create a small 4-city problem
    let city_names = vec![
        "City A".to_string(),
        "City B".to_string(),
        "City C".to_string(),
        "City D".to_string(),
    ];

    // Distance matrix (symmetric)
    let distances = vec![
        vec![0.0, 10.0, 15.0, 20.0],
        vec![10.0, 0.0, 35.0, 25.0],
        vec![15.0, 35.0, 0.0, 30.0],
        vec![20.0, 25.0, 30.0, 0.0],
    ];

    Tsp {
        num_cities: city_names.len(),
        city_names,
        distances,
    }
}

// Formulate the TSP as a QUBO
fn formulate_tsp_qubo(tsp: &Tsp) -> (QuboModel, HashMap<String, usize>) {
    let mut builder = QuboBuilder::new();

    // We'll use the binary variables x_{i,p} where:
    // x_{i,p} = 1 if city i is visited at position p in the tour, 0 otherwise
    let n = tsp.num_cities;

    // Create binary variables
    let mut variables: Vec<Vec<Variable>> = Vec::new();

    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let mut city_vars = Vec::new();

        for p in 0..n {
            let var_name = format!("x_{}_{}", i, p);
            let var = builder.add_variable(var_name).unwrap();
            city_vars.push(var);
        }

        variables.push(city_vars);
    }

    // Set constraint weight
    let constraint_weight = 10.0 * n as f64; // Make constraints stronger than objective
    builder.set_constraint_weight(constraint_weight).unwrap();

    // Constraint 1: Each city must be visited exactly once
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let city_vars = variables[i].clone();
        builder.constrain_one_hot(&city_vars).unwrap();
    }

    // Constraint 2: Each position in the tour must be occupied by exactly one city
    for p in 0..n {
        let position_vars: Vec<Variable> = (0..n).map(|i| variables[i][p].clone()).collect();
        builder.constrain_one_hot(&position_vars).unwrap();
    }

    // Objective: Minimize the total distance
    // For each consecutive positions p and p+1 (with wraparound), add the term:
    // \sum_{i,j} distance[i][j] * x_{i,p} * x_{j,p+1}
    for p in 0..n {
        let p_next = (p + 1) % n;

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let distance = tsp.distances[i][j];
                    builder
                        .minimize_quadratic(&variables[i][p], &variables[j][p_next], distance)
                        .unwrap();
                }
            }
        }
    }

    // Build the QUBO model
    let model = builder.build();
    let var_map = builder.variable_map();

    (model, var_map)
}

// Interpret the binary solution as a tour
fn interpret_tour(tsp: &Tsp, solution: &[bool], var_map: &HashMap<String, usize>) -> Vec<usize> {
    let n = tsp.num_cities;
    let mut tour = vec![0; n];
    let mut is_valid = true;

    // For each position in the tour
    #[allow(clippy::needless_range_loop)]
    for p in 0..n {
        let mut found_city = false;

        // Find which city is at this position
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let var_name = format!("x_{}_{}", i, p);
            if let Some(&var_idx) = var_map.get(&var_name) {
                if solution[var_idx] {
                    if found_city {
                        // Multiple cities at the same position
                        is_valid = false;
                    } else {
                        tour[p] = i;
                        found_city = true;
                    }
                }
            }
        }

        if !found_city {
            // No city at this position
            is_valid = false;
        }
    }

    // Check if each city appears exactly once
    let mut city_count = vec![0; n];
    for &city in &tour {
        city_count[city] += 1;
    }

    for count in city_count {
        if count != 1 {
            is_valid = false;
            break;
        }
    }

    if is_valid {
        tour
    } else {
        Vec::new() // Return empty tour if invalid
    }
}

// Validate that the tour meets the TSP constraints
fn validate_tour(tsp: &Tsp, tour: &[usize]) -> (bool, usize) {
    let n = tsp.num_cities;

    if tour.is_empty() {
        return (false, n); // Empty tour is invalid
    }

    // Check if the tour length is correct
    if tour.len() != n {
        return (false, n - tour.len());
    }

    // Check if each city appears exactly once
    let mut city_visited = vec![false; n];

    for &city in tour {
        if city >= n || city_visited[city] {
            return (false, 1);
        }
        city_visited[city] = true;
    }

    for visited in city_visited {
        if !visited {
            return (false, 1);
        }
    }

    (true, 0)
}

// Print the tour
fn print_tour(tsp: &Tsp, tour: &[usize]) {
    println!("\nTour:");
    for (idx, &city) in tour.iter().enumerate() {
        print!("  {} ({}) ", tsp.city_names[city], city);
        if idx < tour.len() - 1 {
            print!("→");
        }
    }
    println!("  → {} ({})", tsp.city_names[tour[0]], tour[0]);
}

// Calculate the length of a tour
fn calculate_tour_length(tsp: &Tsp, tour: &[usize]) -> f64 {
    let n = tour.len();
    let mut length = 0.0;

    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let city1 = tour[i];
        let city2 = tour[(i + 1) % n];
        length += tsp.distances[city1][city2];
    }

    length
}
