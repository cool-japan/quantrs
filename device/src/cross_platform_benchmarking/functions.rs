//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_circuit::prelude::*;

use super::types::{ComplexityLevel, CrossPlatformBenchmarkConfig, QuantumPlatform};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_cross_platform_config_default() {
        let config = CrossPlatformBenchmarkConfig::default();
        assert_eq!(config.target_platforms.len(), 2);
        assert_eq!(config.complexity_levels.len(), 3);
        assert!(config.enable_cost_analysis);
    }
    #[test]
    fn test_complexity_level_creation() {
        let complexity = ComplexityLevel {
            name: "Test".to_string(),
            qubit_count: 5,
            circuit_depth: 10,
            gate_count_range: (10, 30),
            two_qubit_gate_ratio: 0.4,
            description: "Test complexity level".to_string(),
        };
        assert_eq!(complexity.name, "Test");
        assert_eq!(complexity.qubit_count, 5);
        assert_eq!(complexity.two_qubit_gate_ratio, 0.4);
    }
    #[test]
    fn test_platform_enum() {
        let ibm_platform = QuantumPlatform::IBMQuantum("ibmq_qasm_simulator".to_string());
        let aws_platform = QuantumPlatform::AWSBraket(
            "arn:aws:braket:::device/quantum-simulator/amazon/sv1".to_string(),
        );
        assert_ne!(ibm_platform, aws_platform);
    }
}
