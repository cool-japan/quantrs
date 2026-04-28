//! ZX-Calculus T-Gate Optimization
//!
//! Demonstrates using ZX-calculus graph rewrites to reduce the T-gate count
//! in a quantum circuit. T-gates are expensive in fault-tolerant quantum
//! computing (they require magic state distillation), so minimising them is
//! critical.
//!
//! Pipeline:
//!   1. Build a circuit with several T gates
//!   2. Convert it to a ZX-diagram
//!   3. Apply spider fusion + Hadamard cancellation rewrites
//!   4. Report T-count before and after
//!
//! Run with:
//!   cargo run --example zx_optimize -p quantrs2-circuit --all-features

use quantrs2_circuit::prelude::*;

fn main() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("=== ZX-Calculus T-Gate Optimization ===\n");

    // ---- Step 1: Build a T-heavy circuit ----
    // A typical circuit from quantum chemistry / Clifford+T synthesis
    // We build a 4-qubit circuit with many T and H gates
    let mut circuit = Circuit::<4>::new();

    // Layer 1: Hadamards
    for q in 0..4u32 {
        circuit.h(q)?;
    }
    // Layer 2: T gates
    for q in 0..4u32 {
        circuit.t(q)?;
    }
    // Layer 3: CNOT entanglement
    circuit.cnot(0, 1)?;
    circuit.cnot(2, 3)?;
    // Layer 4: More T gates
    for q in 0..4u32 {
        circuit.t(q)?;
    }
    // Layer 5: Hadamards + T (common in phase gadget synthesis)
    circuit.h(0)?;
    circuit.h(2)?;
    circuit.t(0)?;
    circuit.t(2)?;
    circuit.h(0)?;
    circuit.h(2)?;
    // Layer 6: Another round of CNOTs + T
    circuit.cnot(1, 2)?;
    circuit.cnot(0, 3)?;
    circuit.t(1)?;
    circuit.t(3)?;
    // Layer 7: T†T pairs (should cancel)
    circuit.t(0)?;
    circuit.tdg(0)?; // T† ≈ T^{-1}; T·T† = I
    circuit.t(2)?;
    circuit.tdg(2)?;

    let original_gates = circuit.num_gates();
    let original_depth = circuit.calculate_depth();

    // Count T gates manually (before ZX)
    let t_count_before = count_t_gates(&circuit);

    println!("=== Original Circuit ===");
    println!("  Total gates: {original_gates}");
    println!("  Depth      : {original_depth}");
    println!("  T-gate count: {t_count_before}");

    // ---- Step 2: Convert to ZX diagram ----
    let optimizer = ZXOptimizer::new();
    let mut zx_diagram = optimizer.circuit_to_zx(&circuit)?;

    let t_count_zx_before = zx_diagram.t_count();
    println!("\n=== ZX Diagram (before optimization) ===");
    println!("  Nodes: {}", zx_diagram.nodes.len());
    println!("  Edges: {}", zx_diagram.edges.len());
    println!("  T-gate count: {t_count_zx_before}");

    // ---- Step 3: Apply ZX-calculus optimizations ----
    println!("\n=== Applying ZX-Calculus Rewrites ===");
    let result = zx_diagram.optimize();

    println!("  Optimization complete:");
    println!("  Iterations        : {}", result.iterations);
    println!("  Initial nodes     : {}", result.initial_node_count);
    println!("  Final nodes       : {}", result.final_node_count);
    println!("  Initial T-count   : {}", result.initial_t_count);
    println!("  Final T-count     : {}", result.final_t_count);
    println!("  Converged         : {}", result.converged);

    let t_count_after = zx_diagram.t_count();
    let nodes_after = zx_diagram.nodes.len();
    let edges_after = zx_diagram.edges.len();

    println!("\n=== ZX Diagram (after optimization) ===");
    println!("  Nodes: {nodes_after}");
    println!("  Edges: {edges_after}");
    println!("  T-gate count: {t_count_after}");

    // ---- Summary ----
    println!("\n=== Summary ===");
    println!("  T-count before : {t_count_before}");
    println!("  T-count in ZX  : {t_count_zx_before} → {t_count_after}");
    if t_count_zx_before > 0 {
        let reduction = (t_count_zx_before - t_count_after) as f64 / t_count_zx_before as f64;
        println!("  T-count reduction: {:.1}%", reduction * 100.0);
    }

    // ---- Different circuit profiles ----
    println!("\n=== Benchmark: T-count on different circuits ===");
    for (label, t_cnt) in benchmark_t_counts()? {
        println!("  {label:<30}: T-count = {t_cnt}");
    }

    println!("\nOK — ZX-calculus optimization completed.");

    Ok(())
}

/// Count T and T† gates in a circuit
fn count_t_gates<const N: usize>(circuit: &Circuit<N>) -> usize {
    circuit
        .gates()
        .iter()
        .filter(|g| g.name() == "T" || g.name() == "T†")
        .count()
}

/// Build and optimize several circuits to compare T-counts
fn benchmark_t_counts() -> quantrs2_core::error::QuantRS2Result<Vec<(String, usize)>> {
    let optimizer = ZXOptimizer::new();
    let mut results = Vec::new();

    // All-T circuit
    let mut c_all_t = Circuit::<3>::new();
    for _ in 0..3 {
        for q in 0..3u32 { c_all_t.t(q)?; }
    }
    let zx1 = optimizer.circuit_to_zx(&c_all_t)?;
    results.push(("All-T (9 gates)".to_string(), zx1.t_count()));

    // T+H+T pattern (canonical phase gadget)
    let mut c_phase = Circuit::<2>::new();
    c_phase.h(0)?; c_phase.t(0)?; c_phase.h(0)?;
    c_phase.h(1)?; c_phase.t(1)?; c_phase.h(1)?;
    let zx2 = optimizer.circuit_to_zx(&c_phase)?;
    results.push(("H·T·H (phase gadget)".to_string(), zx2.t_count()));

    // Clifford-only (no T gates)
    let mut c_clifford = Circuit::<2>::new();
    c_clifford.h(0)?;
    c_clifford.cnot(0, 1)?;
    c_clifford.s(0)?;
    let zx3 = optimizer.circuit_to_zx(&c_clifford)?;
    results.push(("Clifford only".to_string(), zx3.t_count()));

    Ok(results)
}
