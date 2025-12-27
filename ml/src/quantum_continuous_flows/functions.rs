//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use crate::quantum_continuous_flows::types::*;
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::random::ChaCha20Rng;
use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::Complex64;
use std::f64::consts::PI;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_quantum_continuous_flow_creation() {
        let config = QuantumContinuousFlowConfig::default();
        let flow = QuantumContinuousFlow::new(config);
        assert!(flow.is_ok());
    }
    #[test]
    fn test_flow_forward_pass() {
        let config = QuantumContinuousFlowConfig {
            input_dim: 4,
            latent_dim: 4,
            num_qubits: 4,
            num_flow_layers: 2,
            ..Default::default()
        };
        let flow = QuantumContinuousFlow::new(config).expect("Flow creation should succeed");
        let x = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let result = flow.forward(&x);
        assert!(result.is_ok());
        let output = result.expect("Forward pass should succeed");
        assert_eq!(output.latent_sample.len(), 4);
        assert!(output.quantum_enhancement.quantum_advantage_ratio >= 1.0);
    }
    #[test]
    fn test_flow_inverse_pass() {
        let config = QuantumContinuousFlowConfig {
            input_dim: 4,
            latent_dim: 4,
            num_qubits: 4,
            ..Default::default()
        };
        let flow = QuantumContinuousFlow::new(config).expect("Flow creation should succeed");
        let z = Array1::from_vec(vec![0.5, -0.3, 0.8, -0.1]);
        let result = flow.inverse(&z);
        assert!(result.is_ok());
        let output = result.expect("Inverse pass should succeed");
        assert_eq!(output.data_sample.len(), 4);
    }
    #[test]
    fn test_quantum_sampling() {
        let config = QuantumContinuousFlowConfig {
            input_dim: 2,
            latent_dim: 2,
            num_qubits: 3,
            ..Default::default()
        };
        let flow = QuantumContinuousFlow::new(config).expect("Flow creation should succeed");
        let result = flow.sample(5);
        assert!(result.is_ok());
        let output = result.expect("Sampling should succeed");
        assert_eq!(output.samples.shape(), &[5, 2]);
        assert_eq!(output.quantum_metrics.len(), 5);
    }
    #[test]
    fn test_quantum_coupling_types() {
        let config = QuantumContinuousFlowConfig {
            flow_architecture: FlowArchitecture::QuantumRealNVP {
                hidden_dims: vec![32, 32],
                num_coupling_layers: 2,
                quantum_coupling_type: QuantumCouplingType::QuantumEntangledCoupling,
            },
            ..Default::default()
        };
        let flow = QuantumContinuousFlow::new(config);
        assert!(flow.is_ok());
    }
    #[test]
    fn test_quantum_neural_ode_flow() {
        let config = QuantumContinuousFlowConfig {
            flow_architecture: FlowArchitecture::QuantumContinuousNormalizing {
                ode_net_dims: vec![16, 16],
                quantum_ode_solver: QuantumODESolver::QuantumRungeKutta4,
                trace_estimation_method: TraceEstimationMethod::EntanglementBasedTrace,
            },
            ..Default::default()
        };
        let flow = QuantumContinuousFlow::new(config);
        assert!(flow.is_ok());
    }
    #[test]
    fn test_quantum_base_distributions() {
        let config = QuantumContinuousFlowConfig {
            latent_dim: 3,
            ..Default::default()
        };
        let flow = QuantumContinuousFlow::new(config).expect("Flow creation should succeed");
        let sample = flow.sample_base_distribution();
        assert!(sample.is_ok());
        assert_eq!(
            sample
                .expect("Base distribution sample should succeed")
                .len(),
            3
        );
    }
    #[test]
    #[ignore]
    fn test_invertibility_guarantees() {
        let config = QuantumContinuousFlowConfig {
            input_dim: 4,
            latent_dim: 4,
            num_qubits: 4,
            num_flow_layers: 2,
            invertibility_tolerance: 1e-8,
            ..Default::default()
        };
        let flow = QuantumContinuousFlow::new(config).expect("Flow creation should succeed");
        let x = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let forward_output = flow.forward(&x).expect("Forward pass should succeed");
        let inverse_output = flow
            .inverse(&forward_output.latent_sample)
            .expect("Inverse pass should succeed");
        let error = (&x - &inverse_output.data_sample)
            .mapv(|x: f64| x.abs())
            .sum();
        assert!(error < 1.0);
    }
}
