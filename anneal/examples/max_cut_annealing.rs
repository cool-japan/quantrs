//! Max-Cut via Ising Simulated Annealing
//!
//! The Max-Cut problem on a graph G=(V,E) asks for a partition (S, V\S)
//! maximising |{(u,v) ∈ E : u ∈ S, v ∉ S}|.
//!
//! Ising formulation: spin s_i ∈ {-1,+1} encodes set membership.
//! The number of cut edges is:
//!   |cut| = (|E| - Σ_{(i,j)∈E} s_i*s_j) / 2
//! Minimising H = Σ_{(i,j)∈E} s_i*s_j maximises the cut.
//!
//! We test on the Petersen graph (10 vertices, 15 edges), whose max-cut is 12.
//!
//! Run with:
//!   cargo run --example max_cut_annealing -p quantrs2-anneal --all-features

use quantrs2_anneal::{
    ising::IsingModel,
    simulator::{AnnealingParams, ClassicalAnnealingSimulator},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Max-Cut on Petersen Graph via Ising Simulated Annealing ===\n");

    // ---- Petersen graph definition ----
    // Outer 5-cycle: 0-1-2-3-4-0
    // Inner pentagram: 5-7-9-6-8-5
    // Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    let edges: &[(usize, usize)] = &[
        // Outer cycle
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        // Inner pentagram
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        // Spokes
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    ];
    let n_vertices = 10usize;
    let n_edges = edges.len();
    let max_cut_known = 12usize; // known optimal for Petersen graph

    println!("Graph: Petersen (V={n_vertices}, E={n_edges})");
    println!("Known max-cut: {max_cut_known}\n");

    // ---- Build Ising model ----
    // Coupling J_{ij} = +1 for each edge (i,j): minimising Σ s_i*s_j maximises cut
    let mut model = IsingModel::new(n_vertices);
    for &(i, j) in edges {
        model.set_coupling(i, j, 1.0)?;
    }

    // ---- Annealing parameters ----
    let mut params = AnnealingParams::new();
    params.num_sweeps = 5000;
    params.num_repetitions = 50; // many restarts for a hard graph
    params.initial_temperature = 3.0;
    params.final_temperature = 0.005;
    params.seed = Some(7);

    // ---- Solve ----
    println!("Running Simulated Annealing ({} repetitions)…", params.num_repetitions);
    let solver = ClassicalAnnealingSimulator::new(params)?;
    let solution = solver.solve(&model)?;

    println!("Best Ising energy: {:.4}", solution.best_energy);
    println!("Runtime          : {:.2?}", solution.runtime);

    // ---- Compute cut size from spins ----
    // cut = (|E| - Σ_{(i,j)∈E} s_i*s_j) / 2
    let spin_product_sum: f64 = edges
        .iter()
        .map(|&(i, j)| {
            (solution.best_spins[i] as f64) * (solution.best_spins[j] as f64)
        })
        .sum();
    let cut_size = ((n_edges as f64 - spin_product_sum) / 2.0).round() as usize;

    println!("Cut size found   : {cut_size} / {max_cut_known} (optimal)");

    // ---- Pretty-print partition ----
    let set_s: Vec<usize> = solution
        .best_spins
        .iter()
        .enumerate()
        .filter(|&(_, &s)| s == 1)
        .map(|(i, _)| i)
        .collect();
    let set_t: Vec<usize> = solution
        .best_spins
        .iter()
        .enumerate()
        .filter(|&(_, &s)| s != 1)
        .map(|(i, _)| i)
        .collect();

    println!("\nPartition S (spin=+1): {set_s:?}");
    println!("Partition T (spin=-1): {set_t:?}");

    // List the cut edges
    let cut_edges: Vec<(usize, usize)> = edges
        .iter()
        .filter(|&&(i, j)| solution.best_spins[i] != solution.best_spins[j])
        .copied()
        .collect();
    println!("Cut edges ({}/{}): {:?}", cut_edges.len(), n_edges, cut_edges);

    // ---- Assertion ----
    assert!(
        cut_size >= 10,
        "Expected at least 10-edge cut on Petersen graph, got {cut_size}"
    );
    println!("\nOK — good cut found.");

    Ok(())
}
