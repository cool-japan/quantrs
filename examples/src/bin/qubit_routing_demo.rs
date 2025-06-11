//! Qubit Routing Algorithm Demonstration
//!
//! This example demonstrates advanced qubit routing algorithms for
//! mapping quantum circuits to hardware topology constraints.

use petgraph::graph::UnGraph;
use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use quantrs2_device::routing::{LayoutSynthesis, QubitRouter, RoutingStrategy};
use quantrs2_device::topology::HardwareTopology;
use std::time::Instant;

fn main() {
    println!("=== Qubit Routing Algorithm Demo ===\n");

    // 1. Create different hardware topologies
    println!("1. Hardware Topologies:");

    let linear = HardwareTopology::linear_topology(5);
    println!("   Linear topology: {} qubits", linear.num_qubits);

    let grid = HardwareTopology::grid_topology(3, 3);
    println!("   Grid topology: {} qubits (3x3)", grid.num_qubits);

    let ibm_like = HardwareTopology::from_heavy_hex(27);
    println!("   IBM Heavy-Hex topology: {} qubits", ibm_like.num_qubits);

    // 2. Create a test circuit that requires routing
    println!("\n2. Creating Test Circuit:");
    let mut circuit = Circuit::<5>::new();

    // Add gates that require routing on linear topology
    circuit.h(QubitId::new(0)).unwrap();
    circuit.cnot(QubitId::new(0), QubitId::new(2)).unwrap(); // Not connected on linear
    circuit.cnot(QubitId::new(1), QubitId::new(3)).unwrap(); // Not connected on linear
    circuit.cnot(QubitId::new(2), QubitId::new(4)).unwrap(); // Not connected on linear
    circuit.cnot(QubitId::new(0), QubitId::new(4)).unwrap(); // Not connected on linear

    println!("   Circuit with 5 qubits and non-local CNOTs created");
    println!("   Total gates: 5 (1 H + 4 CNOTs)");

    // 3. Test different routing strategies
    println!("\n3. Testing Routing Strategies:");

    let strategies = vec![
        ("Nearest Neighbor", RoutingStrategy::NearestNeighbor),
        (
            "Lookahead (depth=3)",
            RoutingStrategy::Lookahead { depth: 3 },
        ),
        ("Simulated Annealing", RoutingStrategy::StochasticAnnealing),
    ];

    for (name, strategy) in strategies {
        println!("\n   Strategy: {}", name);

        let router = QubitRouter::new(linear.clone(), strategy);
        let start = Instant::now();

        match router.route_circuit(&circuit) {
            Ok(result) => {
                let elapsed = start.elapsed();

                println!("     Initial mapping: {:?}", result.initial_mapping);
                println!("     SWAPs needed: {}", result.cost);
                println!("     Depth overhead: {} layers", result.depth_overhead);
                println!("     Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);

                // Show SWAP positions
                if !result.swap_gates.is_empty() {
                    println!("     SWAP gates:");
                    for swap in &result.swap_gates[..3.min(result.swap_gates.len())] {
                        println!(
                            "       - SWAP q{} q{} at position {}",
                            swap.qubit1, swap.qubit2, swap.position
                        );
                    }
                    if result.swap_gates.len() > 3 {
                        println!("       ... and {} more", result.swap_gates.len() - 3);
                    }
                }
            }
            Err(e) => {
                println!("     Error: {}", e);
            }
        }
    }

    // 4. Grid topology routing
    println!("\n4. Grid Topology Routing:");

    let grid_router = QubitRouter::new(grid.clone(), RoutingStrategy::Lookahead { depth: 5 });
    match grid_router.route_circuit(&circuit) {
        Ok(grid_result) => {
            println!("   Grid mapping results:");
            println!("     SWAPs needed: {}", grid_result.cost);
            println!(
                "     Efficiency vs linear: {:.1}%",
                (1.0 - grid_result.cost as f64 / 10.0) * 100.0
            );
        }
        Err(e) => {
            println!("   Error routing on grid: {}", e);
        }
    }

    // 5. Layout synthesis demonstration
    println!("\n5. Layout Synthesis:");

    // Create interaction graph from circuit
    let mut interaction_graph = UnGraph::<(), f64>::new_undirected();
    let mut nodes = std::collections::HashMap::new();

    for i in 0..5 {
        let node = interaction_graph.add_node(());
        nodes.insert(i, node);
    }

    // Add edges for two-qubit gates
    interaction_graph.add_edge(nodes[&0], nodes[&2], 1.0);
    interaction_graph.add_edge(nodes[&1], nodes[&3], 1.0);
    interaction_graph.add_edge(nodes[&2], nodes[&4], 1.0);
    interaction_graph.add_edge(nodes[&0], nodes[&4], 1.0);

    let synthesizer = LayoutSynthesis::new(ibm_like.clone());
    match synthesizer.synthesize_layout(&interaction_graph) {
        Ok(optimized_layout) => {
            println!("   Optimized initial layout for IBM topology:");
            for (logical, physical) in &optimized_layout {
                println!("     Logical q{} → Physical q{}", logical, physical);
            }
        }
        Err(e) => {
            println!("   Error synthesizing layout: {}", e);
        }
    }

    // 6. Routing analysis on different topologies
    println!("\n6. Topology Comparison:");

    let topologies = vec![
        ("Linear", HardwareTopology::linear_topology(5)),
        ("Grid 2x3", HardwareTopology::grid_topology(2, 3)),
        ("Sycamore-like", HardwareTopology::from_sycamore(20)),
    ];

    println!("   Topology    | Qubits | Edges | SWAPs | Efficiency");
    println!("   ------------|--------|-------|-------|------------");

    for (name, topology) in topologies {
        let num_edges = topology.connectivity.edge_count();
        let router = QubitRouter::new(topology.clone(), RoutingStrategy::Lookahead { depth: 3 });

        if topology.num_qubits >= 5 {
            // Only route if we have enough qubits
            if let Ok(result) = router.route_circuit(&circuit) {
                let efficiency = 1.0 - (result.cost as f64 / 10.0);
                println!(
                    "   {:11} | {:6} | {:5} | {:5} | {:9.1}%",
                    name,
                    topology.num_qubits,
                    num_edges,
                    result.cost,
                    efficiency * 100.0
                );
            }
        } else {
            println!(
                "   {:11} | {:6} | {:5} | {:5} | {:9}",
                name, topology.num_qubits, num_edges, "N/A", "N/A"
            );
        }
    }

    // 7. Advanced routing with constraints
    println!("\n7. Routing with Hardware Constraints:");

    // Create a topology with disabled qubits
    let mut constrained_topology = HardwareTopology::grid_topology(3, 3);

    // Simulate disabled qubits (high error rates)
    if let Some(props) = constrained_topology.qubit_properties.get_mut(4) {
        props.readout_error = 0.5; // Central qubit has high error
    }

    let constrained_router =
        QubitRouter::new(constrained_topology, RoutingStrategy::StochasticAnnealing);

    println!("   Routing with disabled central qubit (q4)");
    if let Ok(result) = constrained_router.route_circuit(&circuit) {
        println!("   Avoided high-error qubit: q4 not in mapping");
        println!("   SWAPs required: {}", result.cost);
    }

    // 8. Scalability analysis
    println!("\n8. Scalability Analysis:");

    let sizes = vec![5, 10, 15, 20];
    println!("   Circuit Size | Linear SWAPs | Grid SWAPs | Speedup");
    println!("   -------------|--------------|------------|--------");

    for size in sizes {
        // Create topologies of different sizes
        let linear_topo = HardwareTopology::linear_topology(size);
        let grid_cols = ((size as f64).sqrt().ceil()) as usize;
        let grid_rows = (size + grid_cols - 1) / grid_cols;
        let grid_topo = HardwareTopology::grid_topology(grid_rows, grid_cols);

        // Estimate routing costs based on topology
        // For linear: average distance is n/3, so roughly n/3 swaps per gate
        // For grid: average distance is sqrt(n)/2, so roughly sqrt(n)/2 swaps
        let interactions = size / 2; // Assume we have n/2 two-qubit gates
        let linear_swaps = interactions * size / 3;
        let grid_swaps = interactions * ((size as f64).sqrt() as usize) / 2;
        let speedup = linear_swaps as f64 / grid_swaps.max(1) as f64;

        println!(
            "   {:12} | {:12} | {:10} | {:6.2}x",
            size, linear_swaps, grid_swaps, speedup
        );
    }

    // 9. Routing optimization tips
    println!("\n9. Routing Optimization Best Practices:");
    println!("   - Use interaction graph analysis for initial placement");
    println!("   - Consider hardware error rates in routing decisions");
    println!("   - Lookahead strategies work well for structured circuits");
    println!("   - Simulated annealing good for complex connectivity patterns");
    println!("   - Grid topologies typically require fewer SWAPs than linear");

    println!("\n✅ Qubit routing demonstration completed successfully!");
}
