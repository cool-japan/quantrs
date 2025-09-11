//! Advanced Number Partitioning example using QuantRS2-Tytan
//!
//! This example demonstrates:
//! - Multiple variants of number partitioning (2-way, k-way, balanced)
//! - Dynamic programming comparison for validation
//! - Constraint formulations for different objectives
//! - Performance analysis on various problem instances

use ndarray::Array2;
use quantrs2_tytan::{
    compile::Model,
    constraints::PenaltyFunction,
    optimization::{
        constraints::{ConstraintHandler, ConstraintType},
        penalty::{PenaltyConfig, PenaltyOptimizer, PenaltyType},
    },
    sampler::{GASampler, SASampler, Sampler},
    visualization::{
        convergence::plot_convergence,
        solution_analysis::{analyze_solution_distribution, ClusteringMethod, DistributionConfig},
    },
};

use quantrs2_tytan::compile::expr::{constant, Expr};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::time::Instant;

/// Different types of partitioning problems
#[derive(Debug, Clone, Copy)]
enum PartitioningType {
    /// Classic 2-way partition (equal sum)
    TwoWayEqual,
    /// K-way partition with equal sums
    KWayEqual(usize),
    /// Minimize difference between largest and smallest partition
    BalancedKWay(usize),
    /// Constrained partition (sum within bounds)
    ConstrainedPartition,
}

/// Generate problem instance
fn generate_instance(n: usize, distribution: &str, seed: u64) -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(seed);

    match distribution {
        "uniform" => (0..n).map(|_| rng.gen_range(1..100)).collect(),
        "exponential" => (0..n).map(|i| 2_i32.pow(i as u32 % 10)).collect(),
        "gaussian" => {
            let mean = 50.0;
            let std = 15.0;
            (0..n)
                .map(|_| {
                    let val: f64 = rng.gen::<f64>() * std * 2.0 - std + mean;
                    val.max(1.0) as i32
                })
                .collect()
        }
        "mixed" => {
            // Mix of small and large numbers
            (0..n)
                .map(|i| {
                    if i % 3 == 0 {
                        rng.gen_range(100..1000)
                    } else {
                        rng.gen_range(1..50)
                    }
                })
                .collect()
        }
        _ => panic!("Unknown distribution"),
    }
}

/// Create 2-way equal partition model
fn create_two_way_partition_model(numbers: &[i32]) -> Result<Model, Box<dyn std::error::Error>> {
    let n = numbers.len();
    let total_sum: i32 = numbers.iter().sum();
    let target_sum = total_sum as f64 / 2.0;

    let mut model = Model::new();

    // Binary variables: x_i = 1 if number i is in partition 0
    let mut vars = Vec::new();
    for i in 0..n {
        vars.push(model.add_variable(&format!("x_{}", i))?);
    }

    // Objective: minimize (sum_partition_0 - target)^2
    let mut partition_sum = constant(0.0);
    for i in 0..n {
        partition_sum = partition_sum + constant(numbers[i] as f64) * vars[i].clone();
    }

    // (partition_sum - target)^2
    let diff = partition_sum + constant(-target_sum);
    // Expand: diff^2 = partition_sum^2 - 2*partition_sum*target + target^2
    let mut objective = constant(target_sum * target_sum);

    // Add partition_sum^2 term
    for i in 0..n {
        for j in 0..n {
            let coeff = numbers[i] as f64 * numbers[j] as f64;
            if i == j {
                // x_i^2 = x_i for binary
                objective = objective + constant(coeff) * vars[i].clone();
            } else {
                objective = objective + constant(coeff) * vars[i].clone() * vars[j].clone();
            }
        }
    }

    // Add -2*partition_sum*target term
    for i in 0..n {
        objective = objective + constant(-2.0 * numbers[i] as f64 * target_sum) * vars[i].clone();
    }

    model.set_objective(objective);

    Ok(model)
}

/// Create k-way partition model
fn create_k_way_partition_model(
    numbers: &[i32],
    k: usize,
) -> Result<Model, Box<dyn std::error::Error>> {
    let n = numbers.len();
    let total_sum: i32 = numbers.iter().sum();
    let target_sum = total_sum as f64 / k as f64;

    let mut model = Model::new();

    // Binary variables: x_i_p = 1 if number i is in partition p
    let mut vars = HashMap::new();
    for i in 0..n {
        for p in 0..k {
            let var = model.add_variable(&format!("x_{}_{}", i, p))?;
            vars.insert((i, p), var);
        }
    }

    // Constraint: each number must be in exactly one partition
    for i in 0..n {
        let mut partition_vars = Vec::new();
        for p in 0..k {
            partition_vars.push(vars[&(i, p)].clone());
        }
        model.add_constraint_eq_one(&format!("assign_{}", i), partition_vars)?;
    }

    // Objective: minimize sum of squared deviations from target
    let mut objective = constant(0.0);

    for p in 0..k {
        // Calculate partition sum
        let mut partition_sum = constant(0.0);
        for i in 0..n {
            partition_sum = partition_sum + constant(numbers[i] as f64) * vars[&(i, p)].clone();
        }

        // Add (partition_sum - target)^2 to objective
        // Expand as before
        objective = objective + constant(target_sum * target_sum);

        for i in 0..n {
            for j in 0..n {
                let coeff = numbers[i] as f64 * numbers[j] as f64;
                if i == j {
                    objective = objective + constant(coeff) * vars[&(i, p)].clone();
                } else {
                    objective =
                        objective + constant(coeff) * vars[&(i, p)].clone() * vars[&(j, p)].clone();
                }
            }
        }

        for i in 0..n {
            objective =
                objective + constant(-2.0 * numbers[i] as f64 * target_sum) * vars[&(i, p)].clone();
        }
    }

    model.set_objective(objective);

    Ok(model)
}

/// Dynamic programming solution for 2-way partition
fn solve_partition_dp(numbers: &[i32]) -> Option<(Vec<bool>, i32)> {
    let n = numbers.len();
    let total_sum: i32 = numbers.iter().sum();

    if total_sum % 2 != 0 {
        return None; // Can't partition into equal sums
    }

    let target = total_sum / 2;

    // DP table: dp[i][j] = can we achieve sum j using first i numbers
    let mut dp = vec![vec![false; (target + 1) as usize]; n + 1];
    dp[0][0] = true;

    for i in 1..=n {
        for j in 0..=target {
            dp[i][j as usize] = dp[i - 1][j as usize];
            if j >= numbers[i - 1] {
                dp[i][j as usize] |= dp[i - 1][(j - numbers[i - 1]) as usize];
            }
        }
    }

    if !dp[n][target as usize] {
        return None;
    }

    // Reconstruct solution
    let mut partition = vec![false; n];
    let mut current_sum = target;

    for i in (1..=n).rev() {
        if current_sum >= numbers[i - 1] && dp[i - 1][(current_sum - numbers[i - 1]) as usize] {
            partition[i - 1] = true;
            current_sum -= numbers[i - 1];
        }
    }

    Some((partition, target))
}

/// Extract partition from QUBO solution
fn extract_partition(
    solution: &quantrs2_tytan::sampler::SampleResult,
    n: usize,
    partition_type: PartitioningType,
) -> Vec<usize> {
    match partition_type {
        PartitioningType::TwoWayEqual => {
            let mut partition = vec![0; n];
            for i in 0..n {
                let var_name = format!("x_{}", i);
                if solution
                    .assignments
                    .get(&var_name)
                    .copied()
                    .unwrap_or(false)
                {
                    partition[i] = 0;
                } else {
                    partition[i] = 1;
                }
            }
            partition
        }
        PartitioningType::KWayEqual(k) | PartitioningType::BalancedKWay(k) => {
            let mut partition = vec![0; n];
            for i in 0..n {
                for p in 0..k {
                    let var_name = format!("x_{}_{}", i, p);
                    if solution
                        .assignments
                        .get(&var_name)
                        .copied()
                        .unwrap_or(false)
                    {
                        partition[i] = p;
                        break;
                    }
                }
            }
            partition
        }
        _ => vec![0; n], // Default
    }
}

/// Analyze partition quality
fn analyze_partition(numbers: &[i32], partition: &[usize], k: usize) -> PartitionAnalysis {
    let mut sums = vec![0i32; k];
    let mut counts = vec![0; k];

    for (i, &p) in partition.iter().enumerate() {
        if p < k {
            sums[p] += numbers[i];
            counts[p] += 1;
        }
    }

    let total_sum: i32 = numbers.iter().sum();
    let target_sum = total_sum as f64 / k as f64;

    let max_sum = *sums.iter().max().unwrap_or(&0);
    let min_sum = *sums.iter().min().unwrap_or(&0);
    let imbalance = (max_sum - min_sum) as f64;

    let deviation: f64 = sums
        .iter()
        .map(|&s| (s as f64 - target_sum).powi(2))
        .sum::<f64>()
        .sqrt();

    PartitionAnalysis {
        partition_sums: sums,
        partition_counts: counts,
        total_sum,
        target_sum,
        max_sum,
        min_sum,
        imbalance,
        deviation,
    }
}

#[derive(Debug)]
struct PartitionAnalysis {
    partition_sums: Vec<i32>,
    partition_counts: Vec<usize>,
    total_sum: i32,
    target_sum: f64,
    max_sum: i32,
    min_sum: i32,
    imbalance: f64,
    deviation: f64,
}

/// Run partitioning experiment
fn run_partition_experiment(
    name: &str,
    numbers: &[i32],
    partition_type: PartitioningType,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== {} ===", name);
    println!("Numbers: {:?}", numbers);
    println!("Total sum: {}", numbers.iter().sum::<i32>());

    // Create model based on type
    let model = match partition_type {
        PartitioningType::TwoWayEqual => create_two_way_partition_model(numbers)?,
        PartitioningType::KWayEqual(k) | PartitioningType::BalancedKWay(k) => {
            create_k_way_partition_model(numbers, k)?
        }
        _ => return Err("Unsupported partition type".into()),
    };

    // Optimize and compile
    let penalty_config = PenaltyConfig {
        initial_weight: 10.0,
        min_weight: 0.1,
        max_weight: 100.0,
        adjustment_factor: 1.5,
        violation_tolerance: 1e-4,
        max_iterations: 20,
        adaptive_scaling: true,
        penalty_type: PenaltyType::Quadratic,
    };

    let mut penalty_optimizer = PenaltyOptimizer::new(penalty_config);
    let compiled = model.compile()?;
    let qubo = compiled.to_qubo();

    println!("QUBO variables: {}", qubo.num_variables);

    // Convert QUBO to matrix format
    let n_vars = qubo.num_variables;
    let mut matrix = ndarray::Array2::zeros((n_vars, n_vars));
    let mut var_map = HashMap::new();

    // Create variable mapping and fill matrix
    for i in 0..n_vars {
        var_map.insert(format!("x_{}", i), i);

        // Get linear term (diagonal)
        if let Ok(linear) = qubo.get_linear(i) {
            matrix[[i, i]] = linear;
        }

        // Get quadratic terms
        for j in 0..n_vars {
            if i != j {
                if let Ok(quad) = qubo.get_quadratic(i, j) {
                    matrix[[i, j]] = quad;
                }
            }
        }
    }

    // Compare SA and GA
    let mut sa_sampler = SASampler::new(None);
    let mut ga_sampler = GASampler::with_params(None, 200, 100); // seed, max_generations, population_size

    // Run samplers
    println!("\nRunning samplers...");
    let mut sa_start = Instant::now();
    let sa_samples = sa_sampler.run_qubo(&(matrix.clone(), var_map.clone()), 1000)?;
    let sa_time = sa_start.elapsed();

    let mut ga_start = Instant::now();
    let ga_samples = ga_sampler.run_qubo(&(matrix, var_map), 1000)?;
    let ga_time = ga_start.elapsed();

    // Find best solutions
    let sa_best = sa_samples.iter().min_by_key(|s| s.energy as i64).unwrap();
    let ga_best = ga_samples.iter().min_by_key(|s| s.energy as i64).unwrap();

    // Extract partitions
    let k = match partition_type {
        PartitioningType::TwoWayEqual => 2,
        PartitioningType::KWayEqual(k) | PartitioningType::BalancedKWay(k) => k,
        _ => 2,
    };

    let sa_partition = extract_partition(sa_best, numbers.len(), partition_type);
    let ga_partition = extract_partition(ga_best, numbers.len(), partition_type);

    // Analyze results
    let sa_analysis = analyze_partition(numbers, &sa_partition, k);
    let ga_analysis = analyze_partition(numbers, &ga_partition, k);

    println!("\nSimulated Annealing:");
    println!("  Energy: {:.4}", sa_best.energy);
    println!("  Partition sums: {:?}", sa_analysis.partition_sums);
    println!("  Deviation: {:.2}", sa_analysis.deviation);
    println!("  Imbalance: {}", sa_analysis.imbalance);
    println!("  Time: {:.3}s", sa_time.as_secs_f64());

    println!("\nGenetic Algorithm:");
    println!("  Energy: {:.4}", ga_best.energy);
    println!("  Partition sums: {:?}", ga_analysis.partition_sums);
    println!("  Deviation: {:.2}", ga_analysis.deviation);
    println!("  Imbalance: {}", ga_analysis.imbalance);
    println!("  Time: {:.3}s", ga_time.as_secs_f64());

    // For 2-way partition, compare with DP solution
    if let PartitioningType::TwoWayEqual = partition_type {
        println!("\nDynamic Programming comparison:");
        let mut dp_start = Instant::now();
        match solve_partition_dp(numbers) {
            Some((dp_partition, target)) => {
                let dp_time = dp_start.elapsed();
                let dp_sums = [
                    numbers
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| dp_partition[*i])
                        .map(|(_, &n)| n)
                        .sum::<i32>(),
                    numbers
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !dp_partition[*i])
                        .map(|(_, &n)| n)
                        .sum::<i32>(),
                ];

                println!("  Optimal partition exists!");
                println!("  Partition sums: {:?}", dp_sums);
                println!("  Time: {:.6}s", dp_time.as_secs_f64());

                let sa_optimal = sa_analysis
                    .partition_sums
                    .iter()
                    .all(|&s| s == target || s == sa_analysis.total_sum - target);
                let ga_optimal = ga_analysis
                    .partition_sums
                    .iter()
                    .all(|&s| s == target || s == ga_analysis.total_sum - target);

                println!("  SA found optimal: {}", sa_optimal);
                println!("  GA found optimal: {}", ga_optimal);
            }
            None => {
                println!("  No equal partition exists");
            }
        }
    }

    // Analyze solution distribution
    println!("\nSolution distribution analysis...");
    let distribution_analysis = analyze_solution_distribution(
        sa_samples.clone(),
        Some(DistributionConfig {
            clustering_method: ClusteringMethod::KMeans,
            n_clusters: Some(5),
            ..Default::default()
        }),
    )?;

    println!(
        "  Unique solutions: {}",
        distribution_analysis.statistics.n_unique
    );
    println!(
        "  Mean energy: {:.4}",
        distribution_analysis.statistics.mean_energy
    );
    println!(
        "  Energy std dev: {:.4}",
        distribution_analysis.statistics.std_energy
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Number Partitioning Examples ===");

    // Example 1: Small instance with exact solution
    let small_numbers = vec![3, 1, 4, 2, 5, 5];
    run_partition_experiment(
        "Small 2-Way Partition",
        &small_numbers,
        PartitioningType::TwoWayEqual,
    )?;

    // Example 2: Larger uniform distribution
    let uniform_numbers = generate_instance(20, "uniform", 42);
    run_partition_experiment(
        "Uniform Distribution 2-Way",
        &uniform_numbers,
        PartitioningType::TwoWayEqual,
    )?;

    // Example 3: Exponential distribution (harder)
    let exp_numbers = generate_instance(15, "exponential", 123);
    run_partition_experiment(
        "Exponential Distribution 2-Way",
        &exp_numbers,
        PartitioningType::TwoWayEqual,
    )?;

    // Example 4: 3-way partition
    let three_way_numbers = vec![10, 20, 30, 15, 25, 5, 35, 40, 20];
    run_partition_experiment(
        "3-Way Equal Partition",
        &three_way_numbers,
        PartitioningType::KWayEqual(3),
    )?;

    // Example 5: Mixed distribution 4-way
    let mixed_numbers = generate_instance(16, "mixed", 456);
    run_partition_experiment(
        "Mixed Distribution 4-Way",
        &mixed_numbers,
        PartitioningType::KWayEqual(4),
    )?;

    // Scalability analysis
    println!("\n\n=== Scalability Analysis ===");
    println!("Testing 2-way partition with increasing problem size:");

    let sizes = vec![10, 15, 20, 25, 30];
    let mut results = Vec::new();

    for &n in &sizes {
        let numbers = generate_instance(n, "uniform", n as u64 * 100);

        // Create and solve
        let model = create_two_way_partition_model(&numbers)?;
        let compiled = model.compile()?;
        let qubo = compiled.to_qubo();

        // Convert QUBO to matrix format
        let n_vars = qubo.num_variables;
        let mut matrix = ndarray::Array2::zeros((n_vars, n_vars));
        let mut var_map = HashMap::new();

        for i in 0..n_vars {
            var_map.insert(format!("x_{}", i), i);
            if let Ok(linear) = qubo.get_linear(i) {
                matrix[[i, i]] = linear;
            }
            for j in 0..n_vars {
                if i != j {
                    if let Ok(quad) = qubo.get_quadratic(i, j) {
                        matrix[[i, j]] = quad;
                    }
                }
            }
        }

        let mut sampler = SASampler::new(None);
        let mut start = Instant::now();
        let samples = sampler.run_qubo(&(matrix, var_map), 100)?;
        let time = start.elapsed();

        let best = samples.iter().min_by_key(|s| s.energy as i64).unwrap();
        let partition = extract_partition(best, n, PartitioningType::TwoWayEqual);
        let analysis = analyze_partition(&numbers, &partition, 2);

        results.push((n, qubo.num_variables, analysis.deviation, time));
    }

    println!(
        "\n{:<10} | {:>10} | {:>12} | {:>10}",
        "Size", "QUBO Vars", "Deviation", "Time (s)"
    );
    println!("{:-<10}-+-{:-<10}-+-{:-<12}-+-{:-<10}", "", "", "", "");

    for (n, vars, dev, time) in results {
        println!(
            "{:<10} | {:>10} | {:>12.2} | {:>10.3}",
            n,
            vars,
            dev,
            time.as_secs_f64()
        );
    }

    Ok(())
}
