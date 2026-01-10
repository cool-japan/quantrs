//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use crate::simulator_backends::{DynamicCircuit, Observable, SimulationResult, SimulatorBackend};
use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4, ArrayD, Axis};
use std::sync::Arc;

use super::types::{
    DataEncodingType, DifferentiationMethod, PQCLayer, ParameterInitStrategy, QuantumCircuitLayer,
    QuantumDataset, TFQLossFunction, TFQModel, TFQOptimizer,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulator_backends::{BackendCapabilities, StatevectorBackend};
    use crate::tensorflow_compatibility::tfq_utils;
    #[test]
    #[ignore]
    fn test_quantum_circuit_layer() {
        let mut builder = CircuitBuilder::new();
        builder.ry(0, 0.0).expect("RY gate should succeed");
        builder.ry(1, 0.0).expect("RY gate should succeed");
        builder.cnot(0, 1).expect("CNOT gate should succeed");
        let circuit = builder.build();
        let symbols = vec!["theta1".to_string(), "theta2".to_string()];
        let observable = Observable::PauliZ(vec![0, 1]);
        let backend = Arc::new(StatevectorBackend::new(8));
        let layer = QuantumCircuitLayer::new(circuit, symbols, observable, backend);
        let inputs = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4])
            .expect("Valid shape for inputs");
        let parameters = Array2::from_shape_vec((2, 2), vec![0.5, 0.6, 0.7, 0.8])
            .expect("Valid shape for parameters");
        let result = layer.forward(&inputs, &parameters);
        assert!(result.is_ok());
    }
    #[test]
    fn test_pqc_layer_initialization() -> Result<()> {
        let mut builder = CircuitBuilder::new();
        builder.h(0)?;
        let circuit = builder.build();
        let symbols = vec!["param1".to_string()];
        let observable = Observable::PauliZ(vec![0]);
        let backend = Arc::new(StatevectorBackend::new(8));
        let pqc = PQCLayer::new(circuit, symbols, observable, backend).with_initialization(
            ParameterInitStrategy::RandomNormal {
                mean: 0.0,
                std: 0.1,
            },
        );
        let params = pqc.initialize_parameters(5, 3);
        assert_eq!(params.shape(), &[5, 3]);
        Ok(())
    }
    #[test]
    fn test_glorot_uniform_initialization() -> Result<()> {
        let mut builder = CircuitBuilder::new();
        builder.h(0)?;
        let circuit = builder.build();
        let symbols = vec!["param1".to_string()];
        let observable = Observable::PauliZ(vec![0]);
        let backend = Arc::new(StatevectorBackend::new(8));
        let pqc = PQCLayer::new(circuit, symbols, observable, backend)
            .with_initialization(ParameterInitStrategy::GlorotUniform);
        let params = pqc.initialize_parameters(10, 6);
        assert_eq!(params.shape(), &[10, 6]);
        let limit = (6.0_f64 / (2.0_f64 * 6.0_f64)).sqrt();
        for &val in params.iter() {
            assert!(
                val >= -limit && val <= limit,
                "Parameter {} outside range [-{}, {}]",
                val,
                limit,
                limit
            );
        }
        let mean = params.mean().expect("Mean calculation should succeed");
        let variance = params
            .mapv(|x| (x - mean).powi(2))
            .mean()
            .expect("Variance calculation should succeed");
        assert!(variance > 0.0, "Parameters should have non-zero variance");
        Ok(())
    }
    #[test]
    fn test_glorot_normal_initialization() -> Result<()> {
        let mut builder = CircuitBuilder::new();
        builder.h(0)?;
        let circuit = builder.build();
        let symbols = vec!["param1".to_string()];
        let observable = Observable::PauliZ(vec![0]);
        let backend = Arc::new(StatevectorBackend::new(8));
        let pqc = PQCLayer::new(circuit, symbols, observable, backend)
            .with_initialization(ParameterInitStrategy::GlorotNormal);
        let params = pqc.initialize_parameters(100, 10);
        assert_eq!(params.shape(), &[100, 10]);
        let mean = params.mean().expect("Mean calculation should succeed");
        assert!(mean.abs() < 0.1, "Mean {} should be close to 0", mean);
        let expected_std = (2.0_f64 / (2.0_f64 * 10.0_f64)).sqrt();
        let variance = params
            .mapv(|x| (x - mean).powi(2))
            .mean()
            .expect("Variance calculation should succeed");
        let actual_std = variance.sqrt();
        assert!(
            (actual_std - expected_std).abs() / expected_std < 0.2,
            "Std {} should be close to expected {}",
            actual_std,
            expected_std
        );
        Ok(())
    }
    #[test]
    fn test_elasticnet_regularization() -> Result<()> {
        let l1_ratio = 0.5_f64;
        let alpha = 0.01_f64;
        let parameters = Array2::from_shape_vec((2, 2), vec![1.0, -1.0, 2.0, -2.0])
            .expect("Valid shape for parameters");
        let expected_l1 = parameters.mapv(|x: f64| alpha * l1_ratio * x.signum());
        let expected_l2 = &parameters * (2.0 * alpha * (1.0 - l1_ratio));
        let expected_total = &expected_l1 + &expected_l2;
        assert!((expected_total[[0, 0]] - 0.015_f64).abs() < 1e-10);
        assert!((expected_total[[0, 1]] - (-0.015_f64)).abs() < 1e-10);
        assert!((expected_total[[1, 0]] - 0.025_f64).abs() < 1e-10);
        assert!((expected_total[[1, 1]] - (-0.025_f64)).abs() < 1e-10);
        Ok(())
    }
    #[test]
    fn test_elasticnet_extreme_cases() -> Result<()> {
        let alpha = 0.01_f64;
        let parameters = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).expect("Valid shape");
        let l1_ratio = 0.0_f64;
        let l1_part = parameters.mapv(|x: f64| alpha * l1_ratio * x.signum());
        let l2_part = &parameters * (2.0 * alpha * (1.0 - l1_ratio));
        let total = &l1_part + &l2_part;
        assert!((total[[0, 0]] - 0.02_f64).abs() < 1e-10);
        assert!((total[[0, 1]] - 0.04_f64).abs() < 1e-10);
        let l1_ratio = 1.0_f64;
        let l1_part = parameters.mapv(|x: f64| alpha * l1_ratio * x.signum());
        let l2_part = &parameters * (2.0 * alpha * (1.0 - l1_ratio));
        let total = &l1_part + &l2_part;
        assert!((total[[0, 0]] - 0.01_f64).abs() < 1e-10);
        assert!((total[[0, 1]] - 0.01_f64).abs() < 1e-10);
        Ok(())
    }
    #[test]
    #[ignore]
    fn test_tfq_utils() {
        let circuit = tfq_utils::create_data_encoding_circuit(3, DataEncodingType::Angle)
            .expect("Data encoding circuit creation should succeed");
        assert_eq!(circuit.num_qubits(), 3);
        let ansatz = tfq_utils::create_hardware_efficient_ansatz(4, 2)
            .expect("Hardware efficient ansatz creation should succeed");
        assert_eq!(ansatz.num_qubits(), 4);
    }
    #[test]
    fn test_quantum_dataset() -> Result<()> {
        let circuits = vec![CircuitBuilder::new().build(), CircuitBuilder::new().build()];
        let parameters = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Valid shape for parameters");
        let labels = Array1::from_vec(vec![0.0, 1.0]);
        let dataset = QuantumDataset::new(circuits, parameters, labels, 1);
        let dataset = dataset?;
        let batches: Vec<_> = dataset.batches().collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].0.len(), 1);
        Ok(())
    }
    #[test]
    #[ignore]
    fn test_tfq_model() {
        let mut model = TFQModel::new(vec![2, 2])
            .set_loss(TFQLossFunction::MeanSquaredError)
            .set_optimizer(TFQOptimizer::Adam {
                learning_rate: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            });
        assert!(model.compile().is_ok());
    }
    #[test]
    fn test_cirq_circuit_creation() {
        use crate::tensorflow_compatibility::cirq_converter::*;
        let circuit = CirqCircuit::new(4);
        assert_eq!(circuit.num_qubits, 4);
        assert_eq!(circuit.gates.len(), 0);
        assert_eq!(circuit.param_symbols.len(), 0);
    }
    #[test]
    fn test_cirq_bell_circuit() -> Result<()> {
        use crate::tensorflow_compatibility::cirq_converter::*;
        let cirq_circuit = create_bell_circuit();
        assert_eq!(cirq_circuit.num_qubits, 2);
        assert_eq!(cirq_circuit.gates.len(), 2);
        let quantrs_circuit = cirq_circuit.to_quantrs2_circuit::<2>()?;
        assert_eq!(quantrs_circuit.num_qubits(), 2);
        assert_eq!(quantrs_circuit.gates().len(), 2);
        Ok(())
    }
    #[test]
    fn test_cirq_gate_conversion() -> Result<()> {
        use crate::tensorflow_compatibility::cirq_converter::*;
        let mut cirq_circuit = CirqCircuit::new(3);
        cirq_circuit.add_gate(CirqGate::H { qubit: 0 });
        cirq_circuit.add_gate(CirqGate::X { qubit: 1 });
        cirq_circuit.add_gate(CirqGate::Y { qubit: 2 });
        cirq_circuit.add_gate(CirqGate::CNOT {
            control: 0,
            target: 1,
        });
        cirq_circuit.add_gate(CirqGate::CZ {
            control: 1,
            target: 2,
        });
        assert_eq!(cirq_circuit.gates.len(), 5);
        let quantrs_circuit = cirq_circuit.to_quantrs2_circuit::<3>()?;
        assert_eq!(quantrs_circuit.gates().len(), 5);
        Ok(())
    }
    #[test]
    fn test_cirq_rotation_gates() -> Result<()> {
        use crate::tensorflow_compatibility::cirq_converter::*;
        let mut cirq_circuit = CirqCircuit::new(2);
        cirq_circuit.add_gate(CirqGate::Rx {
            qubit: 0,
            angle: std::f64::consts::PI / 2.0,
        });
        cirq_circuit.add_gate(CirqGate::Ry {
            qubit: 1,
            angle: std::f64::consts::PI / 4.0,
        });
        cirq_circuit.add_gate(CirqGate::Rz {
            qubit: 0,
            angle: std::f64::consts::PI,
        });
        let quantrs_circuit = cirq_circuit.to_quantrs2_circuit::<2>()?;
        assert_eq!(quantrs_circuit.gates().len(), 3);
        Ok(())
    }
    #[test]
    fn test_cirq_pow_gates() -> Result<()> {
        use crate::tensorflow_compatibility::cirq_converter::*;
        let mut cirq_circuit = CirqCircuit::new(3);
        cirq_circuit.add_gate(CirqGate::XPowGate {
            qubit: 0,
            exponent: 0.5,
            global_shift: 0.0,
        });
        cirq_circuit.add_gate(CirqGate::YPowGate {
            qubit: 1,
            exponent: 1.0,
            global_shift: 0.0,
        });
        cirq_circuit.add_gate(CirqGate::ZPowGate {
            qubit: 2,
            exponent: 0.25,
            global_shift: 0.0,
        });
        let quantrs_circuit = cirq_circuit.to_quantrs2_circuit::<3>()?;
        assert_eq!(quantrs_circuit.gates().len(), 3);
        Ok(())
    }
    #[test]
    fn test_cirq_parametric_circuit() -> Result<()> {
        use crate::tensorflow_compatibility::cirq_converter::*;
        let cirq_circuit = create_parametric_circuit(4, 2);
        assert_eq!(cirq_circuit.num_qubits, 4);
        assert_eq!(cirq_circuit.gates.len(), 14);
        assert_eq!(cirq_circuit.param_symbols.len(), 8);
        let quantrs_circuit = cirq_circuit.to_quantrs2_circuit::<4>()?;
        assert_eq!(quantrs_circuit.gates().len(), 14);
        Ok(())
    }
    #[test]
    fn test_cirq_to_dynamic_circuit() -> Result<()> {
        use crate::tensorflow_compatibility::cirq_converter::*;
        let cirq_circuit = create_bell_circuit();
        let dynamic_circuit = cirq_circuit.to_dynamic_circuit()?;
        assert_eq!(dynamic_circuit.num_qubits(), 2);
        Ok(())
    }
    #[test]
    fn test_cirq_u3_gate() -> Result<()> {
        use crate::tensorflow_compatibility::cirq_converter::*;
        let mut cirq_circuit = CirqCircuit::new(1);
        cirq_circuit.add_gate(CirqGate::U3 {
            qubit: 0,
            theta: std::f64::consts::PI / 2.0,
            phi: std::f64::consts::PI / 4.0,
            lambda: std::f64::consts::PI / 3.0,
        });
        let quantrs_circuit = cirq_circuit.to_quantrs2_circuit::<1>()?;
        assert_eq!(quantrs_circuit.gates().len(), 1);
        Ok(())
    }
    #[test]
    fn test_cirq_pow_gate_to_angle() {
        use crate::tensorflow_compatibility::cirq_converter::*;
        let angle = pow_gate_to_angle(1.0);
        assert!((angle - std::f64::consts::PI).abs() < 1e-10);
        let angle = pow_gate_to_angle(0.5);
        assert!((angle - std::f64::consts::PI / 2.0).abs() < 1e-10);
        let angle = pow_gate_to_angle(0.25);
        assert!((angle - std::f64::consts::PI / 4.0).abs() < 1e-10);
    }
    #[test]
    fn test_differentiation_method_enum() {
        use DifferentiationMethod::*;
        assert_eq!(ParameterShift, ParameterShift);
        assert_eq!(Adjoint, Adjoint);
        assert_ne!(ParameterShift, Adjoint);
    }
    #[test]
    fn test_pqc_with_adjoint_method() -> Result<()> {
        let mut builder = CircuitBuilder::new();
        builder.h(0)?;
        let circuit = builder.build();
        let symbols = vec!["param1".to_string()];
        let observable = Observable::PauliZ(vec![0]);
        let backend = Arc::new(StatevectorBackend::new(8));
        let pqc = PQCLayer::new(circuit, symbols, observable, backend)
            .with_differentiation(DifferentiationMethod::Adjoint);
        assert_eq!(pqc.differentiation_method, DifferentiationMethod::Adjoint);
        Ok(())
    }
    #[test]
    fn test_pqc_with_parameter_shift_default() -> Result<()> {
        let mut builder = CircuitBuilder::new();
        builder.h(0)?;
        let circuit = builder.build();
        let symbols = vec!["param1".to_string()];
        let observable = Observable::PauliZ(vec![0]);
        let backend = Arc::new(StatevectorBackend::new(8));
        let pqc = PQCLayer::new(circuit, symbols, observable, backend);
        assert_eq!(
            pqc.differentiation_method,
            DifferentiationMethod::ParameterShift
        );
        Ok(())
    }
    #[test]
    fn test_differentiation_method_switching() -> Result<()> {
        let mut builder = CircuitBuilder::new();
        builder.h(0)?;
        let circuit = builder.build();
        let symbols = vec!["param1".to_string()];
        let observable = Observable::PauliZ(vec![0]);
        let backend = Arc::new(StatevectorBackend::new(8));
        let pqc = PQCLayer::new(
            circuit.clone(),
            symbols.clone(),
            observable.clone(),
            backend.clone(),
        )
        .with_differentiation(DifferentiationMethod::ParameterShift);
        assert_eq!(
            pqc.differentiation_method,
            DifferentiationMethod::ParameterShift
        );
        let pqc_adjoint = PQCLayer::new(circuit, symbols, observable, backend)
            .with_differentiation(DifferentiationMethod::Adjoint);
        assert_eq!(
            pqc_adjoint.differentiation_method,
            DifferentiationMethod::Adjoint
        );
        Ok(())
    }
}
