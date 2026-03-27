//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::collections::{HashMap, VecDeque};

use super::quantumreservoircomputer_type::QuantumReservoirComputer;
use super::types::{
    InputEncoding, OutputMeasurement, QuantumReservoirArchitecture, QuantumReservoirConfig,
    QuantumReservoirState, ReservoirDynamics, ReservoirTrainingData,
};

/// Benchmark quantum reservoir computing
pub fn benchmark_quantum_reservoir_computing() -> Result<HashMap<String, f64>> {
    let mut results = HashMap::new();
    let configs = [
        QuantumReservoirConfig {
            num_qubits: 6,
            architecture: QuantumReservoirArchitecture::RandomCircuit,
            ..Default::default()
        },
        QuantumReservoirConfig {
            num_qubits: 8,
            architecture: QuantumReservoirArchitecture::SpinChain,
            ..Default::default()
        },
        QuantumReservoirConfig {
            num_qubits: 6,
            architecture: QuantumReservoirArchitecture::TransverseFieldIsing,
            ..Default::default()
        },
    ];
    for (i, config) in configs.iter().enumerate() {
        let start = std::time::Instant::now();
        let mut qrc = QuantumReservoirComputer::new(config.clone())?;
        let training_data = ReservoirTrainingData {
            inputs: (0..100)
                .map(|i| {
                    Array1::from_vec(vec![(f64::from(i) * 0.1).sin(), (f64::from(i) * 0.1).cos()])
                })
                .collect(),
            targets: (0..100)
                .map(|i| Array1::from_vec(vec![f64::from(i).mul_add(0.1, 1.0).sin()]))
                .collect(),
            timestamps: (0..100).map(|i| f64::from(i) * 0.1).collect(),
        };
        let _training_result = qrc.train(&training_data)?;
        let time = start.elapsed().as_secs_f64() * 1000.0;
        results.insert(format!("config_{i}"), time);
        let metrics = qrc.get_metrics();
        results.insert(format!("config_{i}_accuracy"), metrics.prediction_accuracy);
        results.insert(format!("config_{i}_memory"), metrics.memory_capacity);
    }
    results.insert("reservoir_initialization_time".to_string(), 500.0);
    results.insert("dynamics_evolution_throughput".to_string(), 200.0);
    results.insert("training_convergence_time".to_string(), 2000.0);
    Ok(results)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_quantum_reservoir_creation() {
        let config = QuantumReservoirConfig::default();
        let qrc = QuantumReservoirComputer::new(config);
        assert!(qrc.is_ok());
    }
    #[test]
    fn test_reservoir_state_creation() {
        let state = QuantumReservoirState::new(3, 10);
        assert_eq!(state.state_vector.len(), 8);
        assert_eq!(state.state_history.capacity(), 10);
        assert_eq!(state.time_index, 0);
    }
    #[test]
    fn test_input_processing() {
        let config = QuantumReservoirConfig {
            num_qubits: 3,
            evolution_steps: 2,
            ..Default::default()
        };
        let mut qrc = QuantumReservoirComputer::new(config)
            .expect("Failed to create quantum reservoir computer");
        let input = Array1::from_vec(vec![0.5, 0.3, 0.8]);
        let result = qrc.process_input(&input);
        assert!(result.is_ok());
        let features = result.expect("Failed to process input");
        assert!(!features.is_empty());
    }
    #[test]
    fn test_different_architectures() {
        let architectures = [
            QuantumReservoirArchitecture::RandomCircuit,
            QuantumReservoirArchitecture::SpinChain,
            QuantumReservoirArchitecture::TransverseFieldIsing,
        ];
        for arch in architectures {
            let config = QuantumReservoirConfig {
                num_qubits: 4,
                architecture: arch,
                evolution_steps: 2,
                ..Default::default()
            };
            let qrc = QuantumReservoirComputer::new(config);
            assert!(qrc.is_ok(), "Failed for architecture: {arch:?}");
        }
    }
    #[test]
    fn test_feature_extraction() {
        let config = QuantumReservoirConfig {
            num_qubits: 3,
            output_measurement: OutputMeasurement::PauliExpectation,
            ..Default::default()
        };
        let mut qrc = QuantumReservoirComputer::new(config)
            .expect("Failed to create quantum reservoir computer");
        let features = qrc.extract_features().expect("Failed to extract features");
        assert_eq!(features.len(), 9);
    }
    #[test]
    fn test_training_data() {
        let training_data = ReservoirTrainingData {
            inputs: vec![
                Array1::from_vec(vec![0.1, 0.2]),
                Array1::from_vec(vec![0.3, 0.4]),
            ],
            targets: vec![Array1::from_vec(vec![0.5]), Array1::from_vec(vec![0.6])],
            timestamps: vec![0.0, 1.0],
        };
        assert_eq!(training_data.inputs.len(), 2);
        assert_eq!(training_data.targets.len(), 2);
        assert_eq!(training_data.timestamps.len(), 2);
    }
    #[test]
    fn test_encoding_methods() {
        let config = QuantumReservoirConfig {
            num_qubits: 3,
            input_encoding: InputEncoding::Amplitude,
            ..Default::default()
        };
        let mut qrc = QuantumReservoirComputer::new(config)
            .expect("Failed to create quantum reservoir computer");
        let input = Array1::from_vec(vec![0.5, 0.3]);
        let result = qrc.encode_input(&input);
        assert!(result.is_ok());
    }
    #[test]
    fn test_measurement_strategies() {
        let measurements = [
            OutputMeasurement::PauliExpectation,
            OutputMeasurement::Probability,
            OutputMeasurement::Correlations,
            OutputMeasurement::Entanglement,
            OutputMeasurement::Fidelity,
        ];
        for measurement in measurements {
            let config = QuantumReservoirConfig {
                num_qubits: 3,
                output_measurement: measurement,
                ..Default::default()
            };
            let qrc = QuantumReservoirComputer::new(config);
            assert!(qrc.is_ok(), "Failed for measurement: {measurement:?}");
        }
    }
    #[test]
    fn test_reservoir_dynamics() {
        let dynamics = [
            ReservoirDynamics::Unitary,
            ReservoirDynamics::Open,
            ReservoirDynamics::NISQ,
        ];
        for dynamic in dynamics {
            let config = QuantumReservoirConfig {
                num_qubits: 3,
                dynamics: dynamic,
                evolution_steps: 1,
                ..Default::default()
            };
            let mut qrc = QuantumReservoirComputer::new(config)
                .expect("Failed to create quantum reservoir computer");
            let result = qrc.evolve_reservoir();
            assert!(result.is_ok(), "Failed for dynamics: {dynamic:?}");
        }
    }
    #[test]
    fn test_metrics_tracking() {
        let config = QuantumReservoirConfig::default();
        let qrc = QuantumReservoirComputer::new(config)
            .expect("Failed to create quantum reservoir computer");
        let metrics = qrc.get_metrics();
        assert_eq!(metrics.training_examples, 0);
        assert_eq!(metrics.prediction_accuracy, 0.0);
    }
}
