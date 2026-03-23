//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{GeographicLocation, QuantumInternet, QuantumRouting, QuantumRoutingAlgorithm};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_quantum_internet_creation() {
        let quantum_internet = QuantumInternet::new();
        assert_eq!(
            quantum_internet
                .quantum_network_infrastructure
                .quantum_nodes
                .len(),
            0
        );
    }
    #[test]
    fn test_global_network_deployment() {
        let mut quantum_internet = QuantumInternet::new();
        let result = quantum_internet.deploy_global_quantum_network();
        assert!(result.is_ok());
        let deployment_result = result.expect("global network deployment should succeed");
        assert!(deployment_result.total_nodes > 0);
        assert!(deployment_result.satellite_coverage > 90.0);
        assert!(deployment_result.network_reliability > 99.0);
    }
    #[test]
    fn test_quantum_internet_advantages() {
        let mut quantum_internet = QuantumInternet::new();
        let report = quantum_internet.demonstrate_quantum_internet_advantages();
        assert!(report.communication_advantage > 1.0);
        assert!(report.distributed_computing_advantage > 1.0);
        assert!(report.sensing_advantage > 1.0);
        assert!(report.security_advantage > 1.0);
        assert!(report.scalability_advantage > 1.0);
        assert!(report.overall_advantage > 1.0);
    }
    #[test]
    fn test_global_qkd() {
        let mut quantum_internet = QuantumInternet::new();
        quantum_internet
            .deploy_global_quantum_network()
            .expect("network deployment should succeed for QKD test");
        let source = GeographicLocation {
            latitude: 40.7128,
            longitude: -74.0060,
            altitude: 0.0,
            country: "USA".to_string(),
            city: "New York".to_string(),
        };
        let destination = GeographicLocation {
            latitude: 51.5074,
            longitude: -0.1278,
            altitude: 0.0,
            country: "UK".to_string(),
            city: "London".to_string(),
        };
        let result = quantum_internet.execute_global_qkd(source, destination, 256);
        assert!(result.is_ok());
        let qkd_result = result.expect("global QKD should succeed");
        assert_eq!(qkd_result.distributed_key.key_length, 256);
        assert!(qkd_result.quantum_advantage > 1.0);
    }
    #[test]
    fn test_quantum_routing() {
        let routing = QuantumRouting::new();
        assert!(matches!(
            routing.routing_algorithm,
            QuantumRoutingAlgorithm::MultiObjective
        ));
    }
}
