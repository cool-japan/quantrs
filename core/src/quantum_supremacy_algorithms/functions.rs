//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    ManyBodySystemType, Photon, PhotonSource, QuantumGateType, QuantumSupremacyEngine,
    RandomCircuitParameters, SamplingParameters,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_quantum_supremacy_engine_creation() {
        let engine = QuantumSupremacyEngine::new(50);
        assert_eq!(
            engine.random_circuit_sampling.circuit_generator.qubit_count,
            50
        );
    }
    #[test]
    fn test_random_circuit_sampling() {
        let mut engine = QuantumSupremacyEngine::new(20);
        let circuit_params = RandomCircuitParameters {
            qubit_count: 20,
            depth: 20,
            gate_set: vec![QuantumGateType::Hadamard, QuantumGateType::CNOT],
        };
        let sampling_params = SamplingParameters {
            sample_count: 1000,
            error_mitigation: true,
        };
        let result = engine.execute_random_circuit_sampling(circuit_params, sampling_params);
        assert!(result.is_ok());
        let supremacy_result = result.expect("Random circuit sampling should succeed");
        assert!(supremacy_result.supremacy_factor > 1.0);
        assert!(supremacy_result.verification_confidence > 0.5);
    }
    #[test]
    fn test_boson_sampling() {
        let mut engine = QuantumSupremacyEngine::new(20);
        let result = engine.execute_boson_sampling(6, 20, 10000);
        assert!(result.is_ok());
        let boson_result = result.expect("Boson sampling should succeed");
        assert_eq!(boson_result.photon_count, 6);
        assert_eq!(boson_result.mode_count, 20);
        assert!(boson_result.supremacy_factor > 1.0);
    }
    #[test]
    fn test_iqp_sampling() {
        let mut engine = QuantumSupremacyEngine::new(30);
        let result = engine.execute_iqp_sampling(30, 10, 100000);
        assert!(result.is_ok());
        let iqp_result = result.expect("IQP sampling should succeed");
        assert_eq!(iqp_result.circuit_depth, 10);
        assert!(iqp_result.computational_advantage > 1.0);
        assert!(iqp_result.hardness_verified);
    }
    #[test]
    fn test_quantum_simulation_advantage() {
        let mut engine = QuantumSupremacyEngine::new(40);
        let result =
            engine.execute_quantum_simulation_advantage(ManyBodySystemType::Hubbard, 40, 1.0);
        assert!(result.is_ok());
        let simulation_result = result.expect("Quantum simulation should succeed");
        assert_eq!(simulation_result.system_size, 40);
        assert!(simulation_result.advantage_factor > 1.0);
        assert!(simulation_result.verification_passed);
    }
    #[test]
    fn test_supremacy_benchmarking() {
        let mut engine = QuantumSupremacyEngine::new(50);
        let report = engine.benchmark_quantum_supremacy();
        assert!(report.random_circuit_advantage > 1e6);
        assert!(report.boson_sampling_advantage > 1e6);
        assert!(report.iqp_sampling_advantage > 1e6);
        assert!(report.quantum_simulation_advantage > 1e6);
        assert!(report.verification_efficiency > 1.0);
        assert!(report.overall_supremacy_factor > 1e6);
    }
    #[test]
    fn test_photon_source_generation() {
        let source = PhotonSource::spdc();
        let photon = source.generate_photon();
        assert!(photon.is_ok());
        let p = photon.expect("Photon generation should succeed");
        assert_eq!(p.wavelength, 800e-9);
    }
}
