//! Circuit Compilation and Routing — 5-Qubit QFT with SABRE Routing
//!
//! Demonstrates the full pipeline:
//!   1. Build a 5-qubit Quantum Fourier Transform (QFT) circuit
//!   2. Route it onto a linear 5-qubit coupling map using SABRE
//!   3. Report gate count, SWAP overhead, and routing statistics
//!
//! The QFT circuit requires all-to-all connectivity (every qubit interacts with
//! every other), making it a good stress-test for routing algorithms. On a linear
//! device (0─1─2─3─4), only adjacent qubits can interact, so SWAPs must be inserted.
//!
//! Run with:
//!   cargo run --example compile_and_route -p quantrs2-circuit --all-features

use quantrs2_circuit::prelude::*;

fn main() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("=== 5-Qubit QFT Circuit Compilation and Routing ===\n");

    const N: usize = 5;

    // ---- Step 1: Build the QFT circuit ----
    let mut qft = Circuit::<N>::new();
    build_qft(&mut qft)?;

    let original_gates = qft.num_gates();
    let original_depth = qft.calculate_depth();
    let original_2q = qft.count_two_qubit_gates();

    println!("=== Original QFT Circuit (5 qubits) ===");
    println!("  Total gates    : {original_gates}");
    println!("  Circuit depth  : {original_depth}");
    println!("  Two-qubit gates: {original_2q}");
    println!("  (Assumes all-to-all connectivity)");

    // ---- Step 2: Route onto a linear device ----
    println!("\n=== Routing onto Linear Device (0─1─2─3─4) ===");
    let linear_map = CouplingMap::linear(N);
    println!("Coupling map: {:?}", linear_map.edges());

    let router = CircuitRouter::new(RoutingStrategy::Sabre, linear_map.clone());
    let routed_sabre = router.route(&qft)?;

    println!("\n-- SABRE Routing --");
    println!("  Gates after routing: {}", routed_sabre.num_gates());
    println!("  SWAP gates inserted: {}", routed_sabre.num_swaps());
    println!("  Routing overhead   : {:.1}%", routed_sabre.routing_overhead() * 100.0);

    let stats = routed_sabre.statistics();
    println!("  Circuit depth  : {}", stats.circuit_depth);
    println!(
        "  Gate breakdown : {} single-qubit, {} two-qubit, {} SWAP",
        stats.single_qubit_gates, stats.two_qubit_gates, stats.swap_gates
    );

    // ---- Step 3: Lookahead routing comparison ----
    println!("\n-- Lookahead Routing (depth=5) --");
    let router_la = CircuitRouter::new(RoutingStrategy::Lookahead { depth: 5 }, linear_map);
    let routed_la = router_la.route(&qft)?;

    println!("  Gates after routing: {}", routed_la.num_gates());
    println!("  SWAP gates inserted: {}", routed_la.num_swaps());
    println!("  Routing overhead   : {:.1}%", routed_la.routing_overhead() * 100.0);

    // ---- Step 4: Ring device (better for QFT) ----
    println!("\n=== Routing onto Ring Device (0─1─2─3─4─0) ===");
    let ring_map = CouplingMap::ring(N);
    let router_ring = CircuitRouter::new(RoutingStrategy::Sabre, ring_map);
    let routed_ring = router_ring.route(&qft)?;

    println!("  Gates after routing: {}", routed_ring.num_gates());
    println!("  SWAP gates inserted: {}", routed_ring.num_swaps());
    println!("  Routing overhead   : {:.1}%", routed_ring.routing_overhead() * 100.0);

    // ---- Step 5: IBM Lagos (7-qubit device, use 5 qubits) ----
    println!("\n=== Routing onto IBM Lagos (7-qubit, using 5) ===");
    let ibm_router = CircuitRouter::for_backend("ibm_lagos");
    // IBM Lagos has 7 qubits; our 5-qubit QFT fits within it
    match ibm_router.route(&qft) {
        Ok(routed_ibm) => {
            println!("  Gates after routing: {}", routed_ibm.num_gates());
            println!("  SWAP gates inserted: {}", routed_ibm.num_swaps());
            println!("  Routing overhead   : {:.1}%", routed_ibm.routing_overhead() * 100.0);
        }
        Err(e) => {
            println!("  IBM Lagos routing: {e}  (may fail for this circuit)");
        }
    }

    // ---- Summary table ----
    println!("\n=== Summary ===");
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8}",
        "Device", "Gates", "SWAPs", "Overhead"
    );
    println!("  {}", "-".repeat(45));
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8.1}%",
        "Original", original_gates, 0, 0.0
    );
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8.1}%",
        "Linear+SABRE",
        routed_sabre.num_gates(),
        routed_sabre.num_swaps(),
        routed_sabre.routing_overhead() * 100.0
    );
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8.1}%",
        "Linear+Lookahead",
        routed_la.num_gates(),
        routed_la.num_swaps(),
        routed_la.routing_overhead() * 100.0
    );
    println!(
        "  {:>12}  {:>8}  {:>8}  {:>8.1}%",
        "Ring+SABRE",
        routed_ring.num_gates(),
        routed_ring.num_swaps(),
        routed_ring.routing_overhead() * 100.0
    );

    // ---- Assertions ----
    assert!(
        routed_sabre.num_gates() >= original_gates,
        "Routing should not reduce gate count"
    );
    assert!(
        routed_sabre.num_gates() <= original_gates * 4,
        "SABRE overhead should be reasonable (< 4x)"
    );

    println!("\nOK — routing completed successfully.");

    Ok(())
}

/// Build a 5-qubit QFT-style circuit in-place.
///
/// We decompose controlled-phase gates CRZ(θ) into CNOT + Rz form,
/// since the SABRE router only supports CNOT, CZ, SWAP for two-qubit gates.
///
/// Decomposition: CRZ(θ, ctrl, tgt) =
///   Rz(θ/2) on ctrl
///   CNOT(ctrl, tgt)
///   Rz(-θ/2) on tgt
///   CNOT(ctrl, tgt)
///   Rz(θ/2) on tgt
fn build_qft<const N: usize>(circuit: &mut Circuit<N>) -> quantrs2_core::error::QuantRS2Result<()> {
    for k in 0..N {
        // Hadamard on qubit k
        circuit.h(k as u32)?;

        // Controlled-phase rotations from higher qubits (decomposed into CNOT+Rz)
        for j in 1..(N - k) {
            let theta = std::f64::consts::PI / (1u32 << j) as f64;
            let ctrl = (k + j) as u32;
            let tgt = k as u32;
            // CRZ(theta, ctrl, tgt) decomposition:
            circuit.rz(ctrl, theta / 2.0)?;
            circuit.cnot(ctrl, tgt)?;
            circuit.rz(tgt, -theta / 2.0)?;
            circuit.cnot(ctrl, tgt)?;
            circuit.rz(tgt, theta / 2.0)?;
        }
    }

    // Reverse qubit order (SWAP pairs)
    for i in 0..(N / 2) {
        circuit.swap(i as u32, (N - 1 - i) as u32)?;
    }

    Ok(())
}
