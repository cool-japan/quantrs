//! Simulated Bifurcation demo on a 4-variable Max-Cut problem (K4 graph).
//!
//! This example demonstrates using the `SBSampler` with the discrete
//! Simulated Bifurcation (dSB) variant to solve a Max-Cut problem on
//! the complete graph K4.
//!
//! # K4 Max-Cut
//!
//! The complete graph K4 has 4 vertices and 6 edges.
//! Maximum cut = 4 edges (partition {0,1} vs {2,3} or similar).
//!
//! QUBO formulation for Max-Cut on K4:
//! Minimize E = Σ_{(i,j) in E} (2*x_i*x_j - x_i - x_j) + const
//! = Σ_{i} (-3*x_i) + Σ_{i<j} 2*x_i*x_j
//!
//! Optimal QUBO energy = -4 (corresponding to max-cut of 4 edges).

use quantrs2_tytan::sampler::{SBSampler, SBVariant, Sampler};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

fn main() {
    println!("=== Simulated Bifurcation Demo: K4 Max-Cut ===\n");

    // Build K4 Max-Cut QUBO
    // E = sum_{i<j} (2*x_i*x_j - x_i - x_j)
    // For K4: 6 edges, each vertex appears in 3 edges
    // Q[i,i] = -3 (linear), Q[i,j] = 2 for i < j (quadratic)
    let n = 4;
    let mut q = Array2::<f64>::zeros((n, n));

    // Linear terms (diagonal): -3 from 3 incident edges per vertex
    for i in 0..n {
        q[[i, i]] = -3.0;
    }

    // Quadratic terms: +2 for each edge in K4
    for i in 0..n {
        for j in (i + 1)..n {
            q[[i, j]] = 2.0;
        }
    }

    // Variable map
    let mut var_map = HashMap::new();
    for i in 0..n {
        var_map.insert(format!("x{i}"), i);
    }

    println!("K4 Max-Cut QUBO matrix:");
    for i in 0..n {
        let row: Vec<String> = (0..n).map(|j| format!("{:5.1}", q[[i, j]])).collect();
        println!("  [{}]", row.join(", "));
    }
    println!();

    // Run discrete SB
    let sampler = SBSampler::new()
        .with_seed(42)
        .with_variant(SBVariant::Discrete)
        .with_time_steps(1000)
        .with_dt(0.5)
        .with_c0(0.5);

    println!("Running dSB with 30 shots...");
    let results = sampler
        .run_qubo(&(q.clone(), var_map.clone()), 30)
        .expect("SB run_qubo failed");

    println!("Top 5 results:");
    for (i, r) in results.iter().take(5).enumerate() {
        let mut assignments: Vec<(String, bool)> = r.assignments.iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        assignments.sort_by_key(|(k, _)| k.clone());
        let assignment_str: Vec<String> = assignments.iter()
            .map(|(k, v)| format!("{k}={}", i32::from(*v)))
            .collect();
        println!(
            "  [{}] energy={:.3}, occurrences={}, assignments=[{}]",
            i,
            r.energy,
            r.occurrences,
            assignment_str.join(", ")
        );
    }

    println!();
    println!("Best energy found: {:.3}", results[0].energy);
    println!("Known optimal energy for K4 Max-Cut QUBO: -4.0");

    // Run ballistic SB for comparison
    let sampler_ballistic = SBSampler::new()
        .with_seed(42)
        .with_variant(SBVariant::Ballistic)
        .with_time_steps(1000);

    println!("\nRunning bSB (ballistic) with 30 shots...");
    let results_b = sampler_ballistic
        .run_qubo(&(q, var_map), 30)
        .expect("bSB failed");

    println!("bSB best energy: {:.3}", results_b[0].energy);

    println!("\nSB demo complete");
}
