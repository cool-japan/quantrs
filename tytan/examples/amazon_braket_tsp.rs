#![allow(
    clippy::pedantic,
    clippy::unnecessary_wraps,
    clippy::needless_range_loop,
    clippy::useless_vec,
    clippy::needless_collect,
    clippy::too_many_arguments
)]
//! Amazon Braket Example: Traveling Salesman Problem
//!
//! This example demonstrates how to use the Amazon Braket sampler
//! to solve a Traveling Salesman Problem (TSP) using various devices.
//!
//! TSP: Find the shortest route that visits all cities exactly once
//! and returns to the starting city.

use quantrs2_tytan::sampler::hardware::{AmazonBraketConfig, AmazonBraketSampler, BraketDevice};
use quantrs2_tytan::sampler::Sampler;
use scirs2_core::ndarray::{Array, Array2};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Amazon Braket Traveling Salesman Problem Example ===\n");

    // Define cities and distances
    let num_cities = 4;
    let cities = vec!["Seattle", "Portland", "San Francisco", "Los Angeles"];

    // Distance matrix (simplified for demonstration)
    let distances = vec![
        vec![0.0, 174.0, 808.0, 1135.0],
        vec![174.0, 0.0, 635.0, 961.0],
        vec![808.0, 635.0, 0.0, 382.0],
        vec![1135.0, 961.0, 382.0, 0.0],
    ];

    println!("Cities: {cities:?}");
    println!("Traveling between {num_cities} cities\n");

    // Convert TSP to QUBO formulation
    // Variables: x_{i,t} = 1 if city i is visited at time t
    let num_vars = num_cities * num_cities;
    let mut qubo_matrix = Array2::<f64>::zeros((num_vars, num_vars));

    // Helper function to get variable index
    let var_idx = |city: usize, time: usize| city * num_cities + time;

    // Objective: Minimize total distance
    for t in 0..num_cities {
        let next_t = (t + 1) % num_cities;
        for i in 0..num_cities {
            for j in 0..num_cities {
                if i != j {
                    let idx_i = var_idx(i, t);
                    let idx_j = var_idx(j, next_t);
                    qubo_matrix[[idx_i, idx_j]] += distances[i][j];
                }
            }
        }
    }

    // Constraint: Each city visited exactly once
    let penalty = 1000.0;
    for i in 0..num_cities {
        for t1 in 0..num_cities {
            let idx1 = var_idx(i, t1);
            qubo_matrix[[idx1, idx1]] -= penalty;
            for t2 in (t1 + 1)..num_cities {
                let idx2 = var_idx(i, t2);
                qubo_matrix[[idx1, idx2]] += 2.0 * penalty;
            }
        }
    }

    // Constraint: Each time slot has exactly one city
    for t in 0..num_cities {
        for i1 in 0..num_cities {
            let idx1 = var_idx(i1, t);
            qubo_matrix[[idx1, idx1]] -= penalty;
            for i2 in (i1 + 1)..num_cities {
                let idx2 = var_idx(i2, t);
                qubo_matrix[[idx1, idx2]] += 2.0 * penalty;
            }
        }
    }

    // Create variable mapping
    let var_map: HashMap<String, usize> = (0..num_cities)
        .flat_map(|city| {
            (0..num_cities).map(move |time| (format!("x_{city}_{time}"), var_idx(city, time)))
        })
        .collect();

    println!("QUBO formulation complete with {num_vars} variables.\n");

    // Example 1: Using Local Simulator (fastest, for development)
    println!("--- Example 1: Local Simulator ---");
    let local_sampler = AmazonBraketSampler::with_s3("my-braket-bucket", "us-east-1")
        .with_device(BraketDevice::LocalSimulator);

    println!("Running on local simulator...");
    let local_results = local_sampler.run_qubo(&(qubo_matrix.clone(), var_map), 100)?;

    println!("Top 3 routes from local simulator:");
    for (idx, result) in local_results.iter().take(3).enumerate() {
        if let Some(route) = extract_route(&result.assignments, num_cities) {
            let total_distance = calculate_route_distance(&route, &distances);
            println!(
                "  {}. Distance: {:.1} km, Route: {:?}",
                idx + 1,
                total_distance,
                route.iter().map(|&i| cities[i]).collect::<Vec<_>>()
            );
        }
    }
    println!();

    // Example 2: Using State Vector Simulator (managed)
    println!("--- Example 2: Managed State Vector Simulator ---");
    let sv_sampler = AmazonBraketSampler::with_s3("my-braket-bucket", "us-east-1")
        .with_device(BraketDevice::StateVectorSimulator)
        .with_max_parallel(5);

    println!("Configuration:");
    println!("  - Device: Managed State Vector Simulator (SV1)");
    println!("  - Max parallel tasks: 5");
    println!("  - S3 bucket: my-braket-bucket");
    println!("  - Region: us-east-1");
    println!();

    // Example 3: Using Tensor Network Simulator (for larger problems)
    println!("--- Example 3: Tensor Network Simulator ---");
    let tn_sampler = AmazonBraketSampler::with_s3("my-braket-bucket", "us-east-1")
        .with_device(BraketDevice::TensorNetworkSimulator)
        .with_poll_interval(10);

    println!("Configuration:");
    println!("  - Device: Tensor Network Simulator (TN1)");
    println!("  - Best for: Problems with up to 50 qubits");
    println!("  - Poll interval: 10 seconds");
    println!();

    // Example 4: Using IonQ Quantum Computer
    println!("--- Example 4: IonQ Quantum Computer ---");
    let ionq_sampler = AmazonBraketSampler::with_s3("my-braket-bucket", "us-east-1")
        .with_device(BraketDevice::IonQDevice)
        .with_max_parallel(3);

    println!("Configuration:");
    println!("  - Device: IonQ Harmony/Aria");
    println!("  - Qubit count: Up to 29 qubits");
    println!("  - Gate fidelity: 99.5%+");
    println!("  - Note: Requires on-demand pricing");
    println!();

    // Example 5: Using Rigetti Quantum Computer
    println!("--- Example 5: Rigetti Quantum Computer ---");
    let rigetti_sampler = AmazonBraketSampler::with_s3("my-braket-bucket", "us-west-2")
        .with_device(BraketDevice::RigettiDevice("Aspen-M-3".to_string()))
        .with_max_parallel(5);

    println!("Configuration:");
    println!("  - Device: Rigetti Aspen-M-3");
    println!("  - Qubit count: 40 qubits");
    println!("  - Region: us-west-2 (required for Rigetti)");
    println!("  - Best for: QAOA and VQE algorithms");
    println!();

    // Example 6: Using D-Wave Advantage via Braket
    println!("--- Example 6: D-Wave Advantage Quantum Annealer ---");
    let dwave_sampler = AmazonBraketSampler::with_s3("my-braket-bucket", "us-west-2")
        .with_device(BraketDevice::DWaveAdvantage)
        .with_max_parallel(10);

    println!("Configuration:");
    println!("  - Device: D-Wave Advantage");
    println!("  - Qubit count: 5000+ qubits");
    println!("  - Best for: Native QUBO/Ising problems");
    println!("  - Very fast sampling (microseconds)");
    println!();

    // Device Comparison
    println!("--- Device Comparison ---");
    println!();
    println!("Local Simulator:");
    println!("  Pros: Free, instant results, good for development");
    println!("  Cons: Limited to ~25 qubits on standard hardware");
    println!("  Cost: Free");
    println!();
    println!("State Vector Simulator (SV1):");
    println!("  Pros: Up to 34 qubits, exact simulation");
    println!("  Cons: Expensive for large circuits");
    println!("  Cost: $0.075 per minute");
    println!();
    println!("Tensor Network Simulator (TN1):");
    println!("  Pros: Up to 50 qubits, efficient for sparse circuits");
    println!("  Cons: Approximate for dense circuits");
    println!("  Cost: $0.275 per minute");
    println!();
    println!("IonQ Harmony/Aria:");
    println!("  Pros: High fidelity, all-to-all connectivity");
    println!("  Cons: Limited qubits, queue times");
    println!("  Cost: $0.30 per task + $0.01 per shot (Harmony)");
    println!();
    println!("Rigetti Aspen-M-3:");
    println!("  Pros: More qubits, fast gate operations");
    println!("  Cons: Limited connectivity, noise");
    println!("  Cost: $0.35 per task + $0.00035 per shot");
    println!();
    println!("D-Wave Advantage:");
    println!("  Pros: Thousands of qubits, very fast");
    println!("  Cons: Limited to QUBO/Ising, limited connectivity");
    println!("  Cost: $0.30 per task + $0.00019 per shot");

    println!("\n--- Setup Instructions ---");
    println!("1. Install AWS CLI: https://aws.amazon.com/cli/");
    println!("2. Configure AWS credentials: aws configure");
    println!("3. Create S3 bucket for Braket results");
    println!("4. Enable Amazon Braket in AWS Console");
    println!("5. Request access to quantum devices (if needed)");
    println!("6. Monitor usage in Braket console to manage costs");

    println!("\n--- Best Practices ---");
    println!("1. Start with local simulator for development");
    println!("2. Use managed simulators for validation");
    println!("3. Test on real hardware only when ready");
    println!("4. Use hybrid algorithms for better results");
    println!("5. Monitor S3 costs (results are stored there)");
    println!("6. Set up AWS Budgets to avoid unexpected costs");

    Ok(())
}

/// Extract route from QUBO solution
fn extract_route(assignments: &HashMap<String, bool>, num_cities: usize) -> Option<Vec<usize>> {
    let mut route = vec![None; num_cities];

    for (var_name, &selected) in assignments {
        if selected {
            if let Some(parts) = var_name.strip_prefix("x_") {
                let parts: Vec<&str> = parts.split('_').collect();
                if parts.len() == 2 {
                    if let (Ok(city), Ok(time)) =
                        (parts[0].parse::<usize>(), parts[1].parse::<usize>())
                    {
                        if time < num_cities {
                            route[time] = Some(city);
                        }
                    }
                }
            }
        }
    }

    // Check if route is complete
    if route.iter().all(|r| r.is_some()) {
        Some(route.into_iter().map(|r| r.unwrap()).collect())
    } else {
        None
    }
}

/// Calculate total distance for a route
fn calculate_route_distance(route: &[usize], distances: &[Vec<f64>]) -> f64 {
    let mut total = 0.0;
    for i in 0..route.len() {
        let from = route[i];
        let to = route[(i + 1) % route.len()];
        total += distances[from][to];
    }
    total
}
