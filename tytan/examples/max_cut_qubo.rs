//! Max-Cut on Cycle C_8 via QUBO — comparing SA vs Tabu Search
//!
//! The Max-Cut problem: given a graph G = (V, E), find a partition (S, V\S)
//! that maximises the number of edges crossing the cut.
//!
//! QUBO formulation for Max-Cut (minimisation):
//!   H = Σ_{(i,j)∈E}  (x_i * x_j - x_i - x_j)
//!       = Σ_{(i,j)∈E}  x_i * x_j  -  Σ_{(i,j)∈E} (x_i + x_j)
//!
//! Minimum of H corresponds to maximum cut.
//! For C_8 the maximum cut is 8 (all 8 edges are cut for alternating colouring).
//!
//! Run with:
//!   cargo run --example max_cut_qubo -p quantrs2-tytan --all-features

use quantrs2_tytan::sampler::{SASampler, Sampler, TabuSampler};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Max-Cut on Cycle C_8 via QUBO ===\n");

    // ---- Graph definition: cycle C_8 with 8 vertices ----
    let n = 8usize;
    // Edges of C_8: (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0)
    let edges: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();

    println!("Graph : cycle C_{n}");
    println!("Edges : {:?}", edges);
    println!("Max possible cut : {n}  (alternating colouring)\n");

    // ---- Build QUBO ----
    // Q[i,j] +=  1  for (i,j) edge (quadratic term)
    // Q[i,i] += -1  for each edge incident to i (linear term)
    let mut q = Array2::<f64>::zeros((n, n));
    for &(i, j) in &edges {
        q[(i, i)] -= 1.0; // -x_i contribution from edge (i,j)
        q[(j, j)] -= 1.0; // -x_j contribution from edge (i,j)
        // Use upper triangle for quadratic terms
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        q[(lo, hi)] += 1.0;
    }

    // Variable map
    let mut var_map = HashMap::new();
    for i in 0..n {
        var_map.insert(format!("x{i}"), i);
    }

    // ---- Solve with SASampler ----
    println!("Running Simulated Annealing (50 shots)…");
    let sa = SASampler::new(Some(42));
    let sa_results = sa
        .run_qubo(&(q.clone(), var_map.clone()), 50)
        .expect("SA sampler failed");
    let sa_best = &sa_results[0];

    let sa_cut = compute_cut(&sa_best.assignments, &edges);
    println!("SA  best energy : {:.4}", sa_best.energy);
    println!("SA  cut size    : {sa_cut}");
    print_assignment(&sa_best.assignments, n);

    // ---- Solve with TabuSampler ----
    println!("\nRunning Tabu Search (50 shots)…");
    let tabu = TabuSampler::new().with_seed(42).with_max_iter(300);
    let tabu_results = tabu
        .run_qubo(&(q, var_map), 50)
        .expect("Tabu sampler failed");
    let tabu_best = &tabu_results[0];

    let tabu_cut = compute_cut(&tabu_best.assignments, &edges);
    println!("Tabu best energy: {:.4}", tabu_best.energy);
    println!("Tabu cut size   : {tabu_cut}");
    print_assignment(&tabu_best.assignments, n);

    // ---- Report ----
    let achieved = sa_cut.max(tabu_cut);
    println!("\nBest cut achieved: {achieved} / {n}");
    assert!(
        achieved >= 7,
        "Expected to find a cut of at least 7 on C_8"
    );
    println!("OK");

    Ok(())
}

/// Count how many edges cross the cut defined by the boolean assignment
fn compute_cut(assignments: &HashMap<String, bool>, edges: &[(usize, usize)]) -> usize {
    edges
        .iter()
        .filter(|&&(i, j)| {
            let xi = *assignments
                .get(&format!("x{i}"))
                .expect("variable missing");
            let xj = *assignments
                .get(&format!("x{j}"))
                .expect("variable missing");
            xi != xj
        })
        .count()
}

/// Pretty-print the binary assignment as a 0/1 string
fn print_assignment(assignments: &HashMap<String, bool>, n: usize) {
    let bits: String = (0..n)
        .map(|i| {
            if *assignments
                .get(&format!("x{i}"))
                .expect("variable missing")
            {
                '1'
            } else {
                '0'
            }
        })
        .collect();
    println!("Assignment      : {bits}");
}
