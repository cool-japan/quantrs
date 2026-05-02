//! Number Partitioning via QUBO with Tabu Search
//!
//! The number partitioning problem asks: given a set of integers, can they be
//! divided into two subsets with equal (or near-equal) sums?
//!
//! QUBO formulation: minimise (Σ w_i * x_i - T/2)² where T = Σ w_i,
//! expanded to: Σ_i w_i(w_i - T) * x_i + 2 Σ_{i<j} w_i * w_j * x_i * x_j
//!
//! A solution with energy ≈ 0 achieves a perfect split.
//!
//! Run with:
//!   cargo run --example number_partition -p quantrs2-tytan --all-features

use quantrs2_tytan::sampler::{Sampler, SBSampler, TabuSampler};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

fn main() {
    println!("=== Number Partitioning via QUBO (Tabu Search) ===\n");

    // ---- Problem definition ----
    // Eight weights that sum to 40; perfect split = 20 vs 20
    let weights: [f64; 8] = [3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0, 4.0]; // sum = 40
    let n = weights.len();
    let total: f64 = weights.iter().sum();

    println!("Weights : {:?}", &weights[..]);
    println!("Total   : {total}  (target split = {})", total / 2.0);
    println!("Variables: {n}\n");

    // ---- Build QUBO matrix ----
    // Q[i,i] = w_i * (w_i - total)
    // Q[i,j] = 2 * w_i * w_j   (i < j; upper triangle)
    let mut q = Array2::<f64>::zeros((n, n));
    for (i, &wi) in weights.iter().enumerate().take(n) {
        q[(i, i)] = wi * (wi - total);
        for j in (i + 1)..n {
            q[(i, j)] = 2.0 * wi * weights[j];
        }
    }

    // Variable map: x0 .. x7
    let mut var_map = HashMap::new();
    for i in 0..n {
        var_map.insert(format!("x{i}"), i);
    }

    // ---- Solve with TabuSampler ----
    println!("Running Tabu Search (100 shots)…");
    let tabu = TabuSampler::new().with_seed(42).with_max_iter(500);
    let results = tabu
        .run_qubo(&(q.clone(), var_map.clone()), 100)
        .expect("Tabu search failed");

    // Results are sorted by energy (ascending)
    let best = &results[0];
    println!("Best energy (Tabu) : {:.4}  (ideal = 0)", best.energy);

    // Reconstruct the two subsets from the best assignment
    let mut set_a: Vec<f64> = Vec::new();
    let mut set_b: Vec<f64> = Vec::new();
    for (i, &wi) in weights.iter().enumerate().take(n) {
        let key = format!("x{i}");
        if *best.assignments.get(&key).expect("key missing") {
            set_a.push(wi);
        } else {
            set_b.push(wi);
        }
    }

    let sum_a: f64 = set_a.iter().sum();
    let sum_b: f64 = set_b.iter().sum();
    println!("Partition A : {set_a:?}  sum = {sum_a}");
    println!("Partition B : {set_b:?}  sum = {sum_b}");
    println!("Imbalance   : {:.1}", (sum_a - sum_b).abs());

    // ---- Compare with SBSampler ----
    println!("\nRunning Simulated Bifurcation (100 shots)…");
    let sb = SBSampler::new();
    let sb_results = sb
        .run_qubo(&(q, var_map), 100)
        .expect("SB sampler failed");
    println!(
        "Best energy (SB)   : {:.4}  (ideal = 0)",
        sb_results[0].energy
    );

    // ---- Verify correctness ----
    // A perfect split (sum_a == sum_b) minimises the QUBO. Confirm imbalance is zero.
    assert!(
        (sum_a - sum_b).abs() < 1.0,
        "Expected a balanced partition; got imbalance {}",
        (sum_a - sum_b).abs()
    );
    println!("\nOK — perfect partition found.");
}
