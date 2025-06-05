//! Demonstration of hardware topology analysis using SciRS2 graphs
//!
//! This example shows how to analyze quantum hardware topologies,
//! find optimal qubit subsets, and identify critical connections.

// Import topology module directly to avoid feature requirements
#[path = "../../../device/src/topology.rs"]
mod topology;

use topology::{GateProperties, HardwareTopology, QubitProperties};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Hardware Topology Analysis Demo ===\n");

    // Test different hardware topologies
    let topologies = vec![
        ("linear", "Linear Chain (5 qubits)"),
        ("grid_2x3", "2x3 Grid"),
        ("ibm_5q", "IBM 5-qubit Bow-tie"),
        ("google_sycamore", "Google Sycamore Subset (3x3)"),
    ];

    for (name, description) in topologies {
        println!("Testing {}", description);
        println!("{}", "-".repeat(50));

        let topology = HardwareTopology::load_standard(name)?;
        let analysis = topology.analyze();

        println!("{}", analysis.report());

        // Find critical qubits
        let critical = topology.find_critical_qubits();
        if !critical.is_empty() {
            println!("Critical qubits (removal disconnects): {:?}", critical);
        }

        // Find optimal subset for 3-qubit circuit
        if topology.num_qubits >= 3 {
            let subset = topology.find_optimal_subset(3)?;
            println!("Optimal 3-qubit subset: {:?}", subset);
        }

        println!();
    }

    // Create a custom topology
    println!("Creating Custom Heavy-Hex Topology");
    println!("{}", "=".repeat(50));

    let mut heavy_hex = HardwareTopology::new(12);

    // Add qubits with varying properties
    for i in 0..12 {
        heavy_hex.add_qubit(QubitProperties {
            id: i as u32,
            index: i as u32,
            t1: 30.0 + 20.0 * fastrand::f64(),
            t2: 40.0 + 30.0 * fastrand::f64(),
            single_qubit_gate_error: 0.001 * (1.0 + 0.5 * fastrand::f64()),
            gate_error_1q: 0.001 * (1.0 + 0.5 * fastrand::f64()),
            readout_error: 0.02 * (1.0 + 0.3 * fastrand::f64()),
            frequency: 4.5 + 0.5 * fastrand::f64(),
        });
    }

    // Create heavy-hex connectivity pattern
    let connections = vec![
        // Hexagon 1
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        // Hexagon 2
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 6),
        // Bridge connections
        (2, 7),
        (4, 9),
    ];

    for (q1, q2) in connections {
        heavy_hex.add_connection(
            q1,
            q2,
            GateProperties {
                error_rate: 0.005 * (1.0 + 0.5 * fastrand::f64()),
                duration: 150.0 + 50.0 * fastrand::f64(),
                gate_type: "CZ".to_string(),
            },
        );
    }

    let analysis = heavy_hex.analyze();
    println!("{}", analysis.report());

    // Find best subset for different circuit sizes
    println!("\nFinding optimal subsets for different circuit sizes:");
    for size in [4, 6, 8] {
        let subset = heavy_hex.find_optimal_subset(size)?;
        println!("  {} qubits: {:?}", size, subset);

        // Calculate subset quality
        let mut total_error = 0.0;
        let mut connections = 0;
        for i in 0..subset.len() {
            if let Some(props) = heavy_hex.qubit_properties.get(subset[i] as usize) {
                total_error += props.gate_error_1q + props.readout_error;
            }
            for j in i + 1..subset.len() {
                if heavy_hex
                    .gate_properties
                    .contains_key(&(subset[i], subset[j]))
                {
                    connections += 1;
                }
            }
        }
        println!(
            "    Average error: {:.4}, Connections: {}",
            total_error / subset.len() as f64,
            connections
        );
    }

    // Demonstrate path finding
    println!("\n=== Path Finding Demo ===");
    let linear = HardwareTopology::load_standard("linear")?;

    println!("Linear topology shortest paths:");
    // In a real implementation, we would expose path finding methods
    // For now, we'll just show the concept
    println!("  Path 0 → 4: 0 → 1 → 2 → 3 → 4 (4 hops)");
    println!("  Path 1 → 3: 1 → 2 → 3 (2 hops)");

    println!("\nDemo completed successfully!");
    Ok(())
}
