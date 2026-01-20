//! Demonstration of circuit routing algorithms
//!
//! This example shows how to use SABRE and lookahead routing algorithms
//! to map logical qubits to physical qubits on devices with limited connectivity.

use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::{multi::CNOT, single::Hadamard};
use quantrs2_core::qubit::QubitId;

fn main() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("=== Quantum Circuit Routing Demo ===\n");

    // Create a test circuit with gates that require routing
    let mut circuit = Circuit::<4>::new();

    // Add gates that create connectivity issues on linear devices
    circuit.add_gate(Hadamard { target: QubitId(0) })?;
    circuit.add_gate(Hadamard { target: QubitId(1) })?;
    circuit.add_gate(CNOT {
        control: QubitId(0),
        target: QubitId(3),
    })?; // Non-adjacent!
    circuit.add_gate(CNOT {
        control: QubitId(1),
        target: QubitId(2),
    })?;
    circuit.add_gate(CNOT {
        control: QubitId(2),
        target: QubitId(0),
    })?; // Another non-adjacent!

    println!("Original circuit:");
    println!("H(0), H(1), CNOT(0,3), CNOT(1,2), CNOT(2,0)");
    println!("Total gates: {}\n", circuit.num_gates());

    // Test different coupling maps
    demo_linear_coupling(&circuit)?;
    demo_grid_coupling(&circuit)?;
    demo_custom_device(&circuit)?;

    Ok(())
}

fn demo_linear_coupling(circuit: &Circuit<4>) -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Linear Device (0-1-2-3) ---");

    let coupling_map = CouplingMap::linear(4);
    println!(
        "Coupling map: linear chain with {} qubits",
        coupling_map.num_qubits()
    );
    println!("Edges: {:?}", coupling_map.edges());

    // Test SABRE routing
    let router = CircuitRouter::new(RoutingStrategy::Sabre, coupling_map.clone());
    let routed = router.route(circuit)?;

    println!("\nSABRE Routing Results:");
    println!("  Total gates after routing: {}", routed.num_gates());
    println!("  SWAP gates inserted: {}", routed.num_swaps());
    println!(
        "  Routing overhead: {:.2}%",
        routed.routing_overhead() * 100.0
    );
    println!("  Final mapping: {:?}", routed.get_mapping());

    let stats = routed.statistics();
    println!("  Circuit depth: {}", stats.circuit_depth);
    println!(
        "  Gate breakdown: {} single, {} two-qubit, {} SWAPs",
        stats.single_qubit_gates, stats.two_qubit_gates, stats.swap_gates
    );

    // Test Lookahead routing
    let lookahead_router =
        CircuitRouter::new(RoutingStrategy::Lookahead { depth: 5 }, coupling_map);
    let routed_lookahead = lookahead_router.route(circuit)?;

    println!("\nLookahead Routing Results:");
    println!(
        "  Total gates after routing: {}",
        routed_lookahead.num_gates()
    );
    println!("  SWAP gates inserted: {}", routed_lookahead.num_swaps());
    println!(
        "  Routing overhead: {:.2}%",
        routed_lookahead.routing_overhead() * 100.0
    );

    println!();
    Ok(())
}

fn demo_grid_coupling(circuit: &Circuit<4>) -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- 2x2 Grid Device ---");

    let coupling_map = CouplingMap::grid(2, 2);
    println!(
        "Coupling map: 2x2 grid with {} qubits",
        coupling_map.num_qubits()
    );
    println!("Edges: {:?}", coupling_map.edges());

    let router = CircuitRouter::new(RoutingStrategy::Sabre, coupling_map);
    let routed = router.route(circuit)?;

    println!("\nSABRE Routing Results:");
    println!("  Total gates after routing: {}", routed.num_gates());
    println!("  SWAP gates inserted: {}", routed.num_swaps());
    println!(
        "  Routing overhead: {:.2}%",
        routed.routing_overhead() * 100.0
    );
    println!("  Final mapping: {:?}", routed.get_mapping());

    println!();
    Ok(())
}

fn demo_custom_device(circuit: &Circuit<4>) -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Custom Device (Star topology) ---");

    // Create a star topology: center qubit (0) connected to all others
    let edges = [(0, 1), (0, 2), (0, 3)];
    let coupling_map = CouplingMap::from_edges(4, &edges);

    println!(
        "Coupling map: star topology with {} qubits",
        coupling_map.num_qubits()
    );
    println!("Edges: {:?}", coupling_map.edges());
    println!("Diameter: {}", coupling_map.diameter());

    let router = CircuitRouter::new(RoutingStrategy::Sabre, coupling_map);
    let routed = router.route(circuit)?;

    println!("\nSABRE Routing Results:");
    println!("  Total gates after routing: {}", routed.num_gates());
    println!("  SWAP gates inserted: {}", routed.num_swaps());
    println!(
        "  Routing overhead: {:.2}%",
        routed.routing_overhead() * 100.0
    );
    println!("  Final mapping: {:?}", routed.get_mapping());

    // Test stochastic routing for comparison
    let stochastic_router = CircuitRouter::new(
        RoutingStrategy::Stochastic { trials: 3 },
        CouplingMap::from_edges(4, &edges),
    );
    let routed_stochastic = stochastic_router.route(circuit)?;

    println!("\nStochastic Routing Results (3 trials):");
    println!(
        "  Total gates after routing: {}",
        routed_stochastic.num_gates()
    );
    println!("  SWAP gates inserted: {}", routed_stochastic.num_swaps());
    println!(
        "  Routing overhead: {:.2}%",
        routed_stochastic.routing_overhead() * 100.0
    );

    println!();
    Ok(())
}
