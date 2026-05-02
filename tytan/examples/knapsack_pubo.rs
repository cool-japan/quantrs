//! 0/1 Knapsack Problem solved as QUBO with Simulated Bifurcation
//!
//! The knapsack problem: choose a subset of items with maximum total value
//! subject to a weight capacity constraint W.
//!
//! QUBO/PUBO penalty formulation:
//!   Objective: -Σ v_i * x_i        (maximise value → minimise negative value)
//!   Constraint: Σ w_i * x_i ≤ W    encoded as penalty P * (Σ w_i * x_i - W)²
//!
//! We use ancilla slack variables s_k (k = 0..K) to convert the inequality
//! into an equality: Σ w_i * x_i + Σ 2^k * s_k = W.
//!
//! Run with:
//!   cargo run --example knapsack_pubo -p quantrs2-tytan --all-features

use quantrs2_tytan::sampler::{SBSampler, Sampler};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

fn main() {
    println!("=== 0/1 Knapsack Problem via QUBO (Simulated Bifurcation) ===\n");

    // ---- Problem data ----
    // 8 items: (value, weight)
    let items: [(f64, f64); 8] = [
        (10.0, 6.0),
        (6.0, 4.0),
        (5.0, 5.0),
        (7.0, 3.0),
        (8.0, 2.0),
        (4.0, 3.0),
        (3.0, 1.0),
        (9.0, 5.0),
    ];
    let capacity: f64 = 15.0;
    let penalty: f64 = 10.0; // Lagrange multiplier for the capacity constraint

    println!("Items (value, weight):");
    for (i, &(v, w)) in items.iter().enumerate() {
        println!("  item[{i}]: value={v}, weight={w}");
    }
    println!("Capacity: {capacity}");
    println!("Penalty  : {penalty}\n");

    // ---- Slack variables ----
    // Need slack s.t. Σ w_i * x_i + Σ 2^k * s_k = W
    // Number of slack bits: ceil(log2(capacity + 1))
    let n_items = items.len();
    let n_slack = (capacity as usize + 1).next_power_of_two().trailing_zeros() as usize;
    let n_total = n_items + n_slack; // total binary variables

    println!("Slack bits: {n_slack}");
    println!("Total QUBO variables: {n_total}\n");

    // Build variable map: x0..x7 for items, s0..s{n_slack-1} for slack
    let mut var_map = HashMap::new();
    for i in 0..n_items {
        var_map.insert(format!("x{i}"), i);
    }
    for k in 0..n_slack {
        var_map.insert(format!("s{k}"), n_items + k);
    }

    // ---- Build QUBO matrix ----
    let mut q = Array2::<f64>::zeros((n_total, n_total));

    // Objective: -Σ v_i * x_i  (linear terms on item variables)
    for i in 0..n_items {
        q[(i, i)] -= items[i].0;
    }

    // Constraint penalty: P * (Σ w_i*x_i + Σ 2^k*s_k - W)²
    // = P * [(Σ w_i*x_i)² + (Σ 2^k*s_k)² + W²
    //        + 2*(Σ w_i*x_i)*(Σ 2^k*s_k)
    //        - 2*W*(Σ w_i*x_i) - 2*W*(Σ 2^k*s_k)]
    //
    // Expanding into QUBO coefficients:
    //   diagonal (linear): P * (coeff^2 - 2*W*coeff) for each variable
    //   off-diagonal (quadratic): P * 2 * coeff_a * coeff_b for each pair

    // Coefficient vector: w_i for items, 2^k for slacks
    let mut coeffs = vec![0.0f64; n_total];
    for i in 0..n_items {
        coeffs[i] = items[i].1; // weight of item i
    }
    for k in 0..n_slack {
        coeffs[n_items + k] = (1u64 << k) as f64;
    }

    // Add penalty terms
    for i in 0..n_total {
        // Linear part: P * (c_i^2 - 2*W*c_i)
        q[(i, i)] = (2.0 * capacity).mul_add(-coeffs[i], coeffs[i] * coeffs[i]).mul_add(penalty, q[(i, i)]);
        for j in (i + 1)..n_total {
            // Quadratic part: P * 2 * c_i * c_j (upper triangle)
            q[(i, j)] = (penalty * 2.0 * coeffs[i]).mul_add(coeffs[j], q[(i, j)]);
        }
    }

    // ---- Solve with SBSampler ----
    println!("Running Simulated Bifurcation (100 shots)…");
    let sb = SBSampler::new();
    let results = sb
        .run_qubo(&(q, var_map), 100)
        .expect("SB sampler failed");

    // ---- Decode best result ----
    let best = &results[0];
    let mut selected_value = 0.0f64;
    let mut selected_weight = 0.0f64;
    let mut selected_items: Vec<usize> = Vec::new();

    for (i, &(val, wt)) in items.iter().enumerate().take(n_items) {
        let key = format!("x{i}");
        if *best.assignments.get(&key).expect("key missing") {
            selected_items.push(i);
            selected_value += val;
            selected_weight += wt;
        }
    }

    println!("Best QUBO energy  : {:.4}", best.energy);
    println!("Selected items    : {selected_items:?}");
    println!("Total value       : {selected_value}");
    println!("Total weight      : {selected_weight} / {capacity}");

    // ---- Brute-force optimal for comparison ----
    let (opt_val, opt_weight, opt_items) = brute_force_knapsack(&items, capacity);
    println!("\nBrute-force optimum:");
    println!("  Items   : {opt_items:?}");
    println!("  Value   : {opt_val}");
    println!("  Weight  : {opt_weight} / {capacity}");

    // Verify constraint feasibility
    // Note: metaheuristic solvers on penalty QUBO may not always find the global optimum.
    // The important check is that the returned solution is *feasible* (weight ≤ capacity).
    assert!(
        selected_weight <= capacity + 1e-6,
        "Capacity constraint violated: {selected_weight} > {capacity}"
    );

    if (selected_value - opt_val).abs() < 1e-6 {
        println!("\nOK — optimal solution found.");
    } else {
        println!(
            "\nFeasible solution found (value={selected_value}, gap={:.1} from optimum).",
            opt_val - selected_value
        );
    }
}

/// Brute-force exact solution for verification
fn brute_force_knapsack(
    items: &[(f64, f64)],
    capacity: f64,
) -> (f64, f64, Vec<usize>) {
    let n = items.len();
    let mut best_value = 0.0f64;
    let mut best_weight = 0.0f64;
    let mut best_mask = 0usize;

    for mask in 0..(1usize << n) {
        let mut value = 0.0f64;
        let mut weight = 0.0f64;
        for (i, &(v, w)) in items.iter().enumerate().take(n) {
            if (mask >> i) & 1 == 1 {
                value += v;
                weight += w;
            }
        }
        if weight <= capacity && value > best_value {
            best_value = value;
            best_weight = weight;
            best_mask = mask;
        }
    }

    let selected: Vec<usize> = (0..n).filter(|&i| (best_mask >> i) & 1 == 1).collect();
    (best_value, best_weight, selected)
}
