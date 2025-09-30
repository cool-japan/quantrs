//! Utility functions and test helpers for SciRS2 mapping

use super::*;
use scirs2_core::random::prelude::*;

/// Create a standard hardware topology for testing
pub fn create_standard_topology(topology_type: &str, num_qubits: usize) -> DeviceResult<HardwareTopology> {
    match topology_type {
        "linear" => create_linear_topology(num_qubits),
        "grid" => create_grid_topology(num_qubits),
        "star" => create_star_topology(num_qubits),
        "complete" => create_complete_topology(num_qubits),
        _ => Err(DeviceError::InvalidTopology(format!("Unknown topology: {}", topology_type)))
    }
}

/// Create a linear chain topology
fn create_linear_topology(num_qubits: usize) -> DeviceResult<HardwareTopology> {
    let mut connectivity = Vec::new();
    for i in 0..num_qubits.saturating_sub(1) {
        connectivity.push((i, i + 1));
    }
    HardwareTopology::new(num_qubits, connectivity)
}

/// Create a 2D grid topology (approximate square)
fn create_grid_topology(num_qubits: usize) -> DeviceResult<HardwareTopology> {
    let side_length = (num_qubits as f64).sqrt().ceil() as usize;
    let mut connectivity = Vec::new();

    for i in 0..num_qubits {
        let row = i / side_length;
        let col = i % side_length;

        // Horizontal connections
        if col < side_length - 1 && i + 1 < num_qubits {
            connectivity.push((i, i + 1));
        }

        // Vertical connections
        if row < side_length - 1 && i + side_length < num_qubits {
            connectivity.push((i, i + side_length));
        }
    }

    HardwareTopology::new(num_qubits, connectivity)
}

/// Create a star topology
fn create_star_topology(num_qubits: usize) -> DeviceResult<HardwareTopology> {
    let mut connectivity = Vec::new();
    let center = 0;

    for i in 1..num_qubits {
        connectivity.push((center, i));
    }

    HardwareTopology::new(num_qubits, connectivity)
}

/// Create a complete graph topology
fn create_complete_topology(num_qubits: usize) -> DeviceResult<HardwareTopology> {
    let mut connectivity = Vec::new();

    for i in 0..num_qubits {
        for j in i + 1..num_qubits {
            connectivity.push((i, j));
        }
    }

    HardwareTopology::new(num_qubits, connectivity)
}

/// Create ideal calibration data for testing
pub fn create_ideal_calibration(device_name: String, num_qubits: usize) -> DeviceCalibration {
    DeviceCalibration::new(device_name, num_qubits)
}

/// Validate mapping consistency
pub fn validate_mapping(
    mapping: &HashMap<usize, usize>,
    num_logical_qubits: usize,
    num_physical_qubits: usize,
) -> DeviceResult<()> {
    // Check all logical qubits are mapped
    for i in 0..num_logical_qubits {
        if !mapping.contains_key(&i) {
            return Err(DeviceError::InvalidMapping(format!(
                "Logical qubit {} not mapped", i
            )));
        }
    }

    // Check all physical qubits are valid
    for &physical in mapping.values() {
        if physical >= num_physical_qubits {
            return Err(DeviceError::InvalidMapping(format!(
                "Physical qubit {} exceeds device capacity", physical
            )));
        }
    }

    // Check for duplicate mappings
    let mut used_physical = HashSet::new();
    for &physical in mapping.values() {
        if !used_physical.insert(physical) {
            return Err(DeviceError::InvalidMapping(format!(
                "Physical qubit {} mapped multiple times", physical
            )));
        }
    }

    Ok(())
}

/// Calculate mapping quality score
pub fn calculate_mapping_quality(
    mapping: &HashMap<usize, usize>,
    logical_graph: &Graph<usize, f64>,
    topology: &HardwareTopology,
) -> DeviceResult<f64> {
    let mut total_distance = 0.0;
    let mut edge_count = 0;

    for edge in logical_graph.edges() {
        // Note: node_weight method not available in current scirs2-graph version
        // Using edge source/target indices as usize directly
        let source = edge.source as usize;
        let target = edge.target as usize;

        if let (Some(&phys_source), Some(&phys_target)) =
            (mapping.get(&source), mapping.get(&target)) {

            // Calculate shortest path distance in physical topology
            let distance = topology.shortest_path_distance(phys_source, phys_target)
                .unwrap_or(f64::INFINITY);

            total_distance += distance;
            edge_count += 1;
        }
    }

    if edge_count > 0 {
        Ok(1.0 / (1.0 + total_distance / edge_count as f64))
    } else {
        Ok(1.0)
    }
}

/// Generate random circuit for testing
pub fn generate_random_circuit<const N: usize>(
    gate_count: usize,
    two_qubit_ratio: f64,
) -> Circuit<N> {
    let mut circuit = Circuit::<N>::new();
    let mut rng = thread_rng();

    for _ in 0..gate_count {
        if rng.gen::<f64>() < two_qubit_ratio {
            // Two-qubit gate
            let q1 = rng.gen_range(0..N);
            let mut q2 = rng.gen_range(0..N);
            while q2 == q1 {
                q2 = rng.gen_range(0..N);
            }
            circuit.cnot(QubitId(q1 as u32), QubitId(q2 as u32));
        } else {
            // Single-qubit gate
            let q = rng.gen_range(0..N);
            circuit.x(QubitId(q as u32));
        }
    }

    circuit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_topology_creation() {
        let topology = create_linear_topology(4).unwrap();
        assert_eq!(topology.num_qubits(), 4);
        assert!(topology.are_connected(0, 1));
        assert!(topology.are_connected(1, 2));
        assert!(topology.are_connected(2, 3));
        assert!(!topology.are_connected(0, 3));
    }

    #[test]
    fn test_grid_topology_creation() {
        let topology = create_grid_topology(4).unwrap();
        assert_eq!(topology.num_qubits(), 4);
        // In a 2x2 grid: 0-1, 0-2, 1-3, 2-3
        assert!(topology.are_connected(0, 1));
        assert!(topology.are_connected(0, 2));
    }

    #[test]
    fn test_star_topology_creation() {
        let topology = create_star_topology(5).unwrap();
        assert_eq!(topology.num_qubits(), 5);
        // Center (0) connected to all others
        for i in 1..5 {
            assert!(topology.are_connected(0, i));
        }
        // Non-center nodes not connected to each other
        assert!(!topology.are_connected(1, 2));
    }

    #[test]
    fn test_mapping_validation() {
        let mut mapping = HashMap::new();
        mapping.insert(0, 0);
        mapping.insert(1, 1);
        mapping.insert(2, 2);

        assert!(validate_mapping(&mapping, 3, 4).is_ok());

        // Test duplicate mapping
        mapping.insert(3, 2); // Duplicate physical qubit 2
        assert!(validate_mapping(&mapping, 4, 4).is_err());
    }

    #[test]
    fn test_mapping_quality_calculation() {
        let topology = create_linear_topology(4).unwrap();
        let mut graph = Graph::new();
        let nodes: Vec<_> = (0..4).map(|i| graph.add_node(i)).collect();

        // Add edge between qubits 0 and 3 (distant in linear topology)
        graph.add_edge(nodes[0], nodes[3], 1.0);

        let mut mapping = HashMap::new();
        mapping.insert(0, 0);
        mapping.insert(1, 1);
        mapping.insert(2, 2);
        mapping.insert(3, 3);

        let quality = calculate_mapping_quality(&mapping, &graph, &topology).unwrap();
        assert!(quality > 0.0 && quality <= 1.0);
    }

    #[test]
    fn test_random_circuit_generation() {
        let circuit = generate_random_circuit::<4>(10, 0.5);
        assert!(circuit.operations().len() <= 10);
    }

    #[test]
    fn test_standard_topology_creation() {
        assert!(create_standard_topology("linear", 4).is_ok());
        assert!(create_standard_topology("grid", 4).is_ok());
        assert!(create_standard_topology("star", 4).is_ok());
        assert!(create_standard_topology("complete", 4).is_ok());
        assert!(create_standard_topology("invalid", 4).is_err());
    }

    #[test]
    fn test_complete_topology() {
        let topology = create_complete_topology(4).unwrap();
        assert_eq!(topology.num_qubits(), 4);

        // All pairs should be connected
        for i in 0..4 {
            for j in i + 1..4 {
                assert!(topology.are_connected(i, j));
            }
        }
    }

    #[test]
    fn test_ideal_calibration_creation() {
        let calibration = create_ideal_calibration("test_device".to_string(), 4);
        assert_eq!(calibration.device_name(), "test_device");
        assert_eq!(calibration.num_qubits(), 4);
    }

    #[test]
    fn test_mapping_validation_edge_cases() {
        // Empty mapping
        let mapping = HashMap::new();
        assert!(validate_mapping(&mapping, 0, 4).is_ok());

        // Physical qubit out of range
        let mut mapping = HashMap::new();
        mapping.insert(0, 10);
        assert!(validate_mapping(&mapping, 1, 4).is_err());

        // Missing logical qubit
        let mut mapping = HashMap::new();
        mapping.insert(0, 0);
        assert!(validate_mapping(&mapping, 2, 4).is_err());
    }
}