//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::qnn::{QNNLayerType, QuantumNeuralNetwork};
use std::collections::HashMap;
use std::f64::consts::PI;

use super::types::{CircuitOptimizationLevel, ClientSelectionStrategy, ConvergenceCriteria, LocalTrainingConfig, MaliciousDetectionConfig, NoiseHandlingStrategy, PrivacyConfig, QuantumAggregationStrategy, QuantumDeviceInfo, QuantumDeviceType, QuantumFederatedClient, QuantumFederatedServer, QuantumPrivacyTechnique, ServerConfig, ServerSecurityConfig, ValidationConfig, ValidationDataConfig, WeightingType};

fn random_gaussian() -> f64 {
    static mut SPARE: Option<f64> = None;
    unsafe {
        if let Some(val) = SPARE {
            SPARE = None;
            return val;
        }
    }
    let u1 = fastrand::f64();
    let u2 = fastrand::f64();
    let mag = 1.0 * (-2.0 * u1.ln()).sqrt();
    let z0 = mag * (2.0 * PI * u2).cos();
    let z1 = mag * (2.0 * PI * u2).sin();
    unsafe {
        SPARE = Some(z1);
    }
    z0
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::qnn::QNNLayerType;
    #[test]
    fn test_quantum_federated_client_creation() {
        let layers = vec![
            QNNLayerType::EncodingLayer { num_features : 4 },
            QNNLayerType::VariationalLayer { num_params : 8 },
        ];
        let model = QuantumNeuralNetwork::new(layers, 4, 4, 2)
            .expect("Failed to create model");
        let device_info = QuantumDeviceInfo {
            num_qubits: 4,
            coherence_time: 100.0,
            gate_error_rates: HashMap::new(),
            readout_error: 0.01,
            connectivity: vec![(0, 1), (1, 2), (2, 3)],
            device_type: QuantumDeviceType::Simulator {
                noise_model: None,
            },
            last_calibration: 0,
        };
        let local_config = LocalTrainingConfig {
            local_epochs: 5,
            batch_size: 32,
            learning_rate: 0.01,
            circuit_optimization: CircuitOptimizationLevel::Basic,
            error_mitigation: Vec::new(),
            noise_handling: NoiseHandlingStrategy::Ignore,
        };
        let client = QuantumFederatedClient::new(
            "client_1".to_string(),
            model,
            device_info,
            local_config,
        );
        assert_eq!(client.client_id, "client_1");
        assert_eq!(client.device_info.num_qubits, 4);
    }
    #[test]
    fn test_quantum_federated_server_creation() {
        let layers = vec![
            QNNLayerType::EncodingLayer { num_features : 4 },
            QNNLayerType::VariationalLayer { num_params : 8 },
        ];
        let global_model = QuantumNeuralNetwork::new(layers, 4, 4, 2)
            .expect("Failed to create global model");
        let aggregation_strategy = QuantumAggregationStrategy::QuantumFedAvg {
            weight_type: WeightingType::DataSize,
        };
        let server_config = ServerConfig {
            global_rounds: 10,
            min_clients_per_round: 2,
            client_selection: ClientSelectionStrategy::Random {
                fraction: 0.5,
            },
            validation_config: ValidationConfig {
                validation_frequency: 5,
                validation_data: ValidationDataConfig::ServerSide,
                quantum_benchmarks: Vec::new(),
                classical_benchmarks: Vec::new(),
            },
            convergence_criteria: ConvergenceCriteria {
                loss_threshold: 0.01,
                patience: 10,
                quantum_fidelity_threshold: 0.95,
                parameter_change_threshold: 0.001,
                quantum_early_stopping: false,
            },
            security_config: ServerSecurityConfig {
                require_authentication: false,
                malicious_detection: MaliciousDetectionConfig {
                    byzantine_detection: false,
                    statistical_detection: false,
                    quantum_signature_verification: false,
                    reputation_system: false,
                },
                audit_logging: false,
                secure_aggregation_protocols: Vec::new(),
            },
        };
        let server = QuantumFederatedServer::new(
            global_model,
            aggregation_strategy,
            server_config,
        );
        assert_eq!(server.server_config.global_rounds, 10);
        assert_eq!(server.clients.len(), 0);
    }
    #[test]
    fn test_client_registration() {
        let layers = vec![
            QNNLayerType::EncodingLayer { num_features : 4 },
            QNNLayerType::VariationalLayer { num_params : 8 },
        ];
        let global_model = QuantumNeuralNetwork::new(layers.clone(), 4, 4, 2)
            .expect("Failed to create global model");
        let local_model = QuantumNeuralNetwork::new(layers, 4, 4, 2)
            .expect("Failed to create local model");
        let aggregation_strategy = QuantumAggregationStrategy::QuantumFedAvg {
            weight_type: WeightingType::Uniform,
        };
        let server_config = ServerConfig {
            global_rounds: 5,
            min_clients_per_round: 1,
            client_selection: ClientSelectionStrategy::Random {
                fraction: 1.0,
            },
            validation_config: ValidationConfig {
                validation_frequency: 5,
                validation_data: ValidationDataConfig::ServerSide,
                quantum_benchmarks: Vec::new(),
                classical_benchmarks: Vec::new(),
            },
            convergence_criteria: ConvergenceCriteria {
                loss_threshold: 0.01,
                patience: 10,
                quantum_fidelity_threshold: 0.95,
                parameter_change_threshold: 0.001,
                quantum_early_stopping: false,
            },
            security_config: ServerSecurityConfig {
                require_authentication: false,
                malicious_detection: MaliciousDetectionConfig {
                    byzantine_detection: false,
                    statistical_detection: false,
                    quantum_signature_verification: false,
                    reputation_system: false,
                },
                audit_logging: false,
                secure_aggregation_protocols: Vec::new(),
            },
        };
        let mut server = QuantumFederatedServer::new(
            global_model,
            aggregation_strategy,
            server_config,
        );
        let device_info = QuantumDeviceInfo {
            num_qubits: 4,
            coherence_time: 100.0,
            gate_error_rates: HashMap::new(),
            readout_error: 0.01,
            connectivity: vec![(0, 1), (1, 2), (2, 3)],
            device_type: QuantumDeviceType::Simulator {
                noise_model: None,
            },
            last_calibration: 0,
        };
        let local_config = LocalTrainingConfig {
            local_epochs: 5,
            batch_size: 32,
            learning_rate: 0.01,
            circuit_optimization: CircuitOptimizationLevel::Basic,
            error_mitigation: Vec::new(),
            noise_handling: NoiseHandlingStrategy::Ignore,
        };
        let client = QuantumFederatedClient::new(
            "client_1".to_string(),
            local_model,
            device_info,
            local_config,
        );
        server.register_client(client);
        assert_eq!(server.clients.len(), 1);
        assert!(server.clients.contains_key("client_1"));
    }
    #[test]
    fn test_aggregation_strategies() {
        let strategies = vec![
            QuantumAggregationStrategy::QuantumFedAvg { weight_type :
            WeightingType::Uniform, }, QuantumAggregationStrategy::QuantumFedProx { mu :
            0.1 }, QuantumAggregationStrategy::QuantumFedNova { tau_eff : 1.0 },
            QuantumAggregationStrategy::QuantumSCAFFOLD { learning_rate : 0.01 },
        ];
        assert_eq!(strategies.len(), 4);
    }
    #[test]
    fn test_device_types() {
        let device_types = vec![
            QuantumDeviceType::Superconducting { frequency_range : (4.0, 6.0) },
            QuantumDeviceType::TrappedIon { ion_species : "Ca+".to_string() },
            QuantumDeviceType::Photonic { wavelength : 1550.0 },
            QuantumDeviceType::Simulator { noise_model : None },
        ];
        assert_eq!(device_types.len(), 4);
    }
    #[test]
    fn test_privacy_config() {
        let privacy_config = PrivacyConfig {
            differential_privacy: true,
            privacy_budget: 1.0,
            delta: 1e-5,
            secure_aggregation: true,
            quantum_privacy: vec![
                QuantumPrivacyTechnique::QuantumDP { quantum_epsilon : 0.5 },
            ],
            data_minimization: true,
        };
        assert!(privacy_config.differential_privacy);
        assert_eq!(privacy_config.privacy_budget, 1.0);
    }
}
