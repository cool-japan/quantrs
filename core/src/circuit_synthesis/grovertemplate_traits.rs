//! # GroverTemplate - Trait Implementations
//!
//! This module contains trait implementations for `GroverTemplate`.
//!
//! ## Implemented Traits
//!
//! - `AlgorithmTemplate`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::AlgorithmTemplate;
use super::types::{
    AlgorithmSpecification, GateMetadata, GroverTemplate, QAOATemplate, QuantumAlgorithmType,
    ResourceEstimates, SynthesizedCircuit, SynthesizedGate, TemplateInfo,
};
use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::QubitId;
use std::collections::HashMap;
use std::time::Duration;

impl AlgorithmTemplate for GroverTemplate {
    fn synthesize(&self, spec: &AlgorithmSpecification) -> QuantRS2Result<SynthesizedCircuit> {
        let num_qubits = spec.parameters.num_qubits;
        let mut gates = Vec::new();
        for i in 0..num_qubits {
            gates.push(SynthesizedGate {
                name: "H".to_string(),
                qubits: vec![QubitId::new(i as u32)],
                parameters: vec![],
                matrix: None,
                metadata: GateMetadata {
                    layer: 0,
                    purpose: "Initial superposition".to_string(),
                    hints: vec!["single_qubit".to_string()],
                    hardware_preferences: vec!["any".to_string()],
                },
            });
        }
        QAOATemplate::new().create_circuit_from_gates(
            gates,
            num_qubits,
            QuantumAlgorithmType::Grover,
        )
    }
    fn estimate_resources(
        &self,
        spec: &AlgorithmSpecification,
    ) -> QuantRS2Result<ResourceEstimates> {
        let num_qubits = spec.parameters.num_qubits;
        let num_items = 2_usize.pow(num_qubits as u32);
        let iterations = (std::f64::consts::PI / 4.0 * (num_items as f64).sqrt()) as usize;
        Ok(ResourceEstimates {
            gate_count: num_qubits + iterations * (num_qubits + 1),
            circuit_depth: 1 + iterations * 2,
            qubit_count: num_qubits,
            gate_breakdown: HashMap::new(),
            estimated_execution_time: Duration::from_micros((iterations * num_qubits * 100) as u64),
            memory_requirements: 1 << num_qubits,
            parallelization_factor: 0.3,
        })
    }
    fn get_template_info(&self) -> TemplateInfo {
        TemplateInfo {
            name: "Grover".to_string(),
            supported_parameters: vec!["num_qubits".to_string(), "oracle".to_string()],
            required_parameters: vec!["num_qubits".to_string()],
            complexity_scaling: "O(√N)".to_string(),
            hardware_compatibility: vec!["all".to_string()],
        }
    }
    fn validate_specification(&self, spec: &AlgorithmSpecification) -> QuantRS2Result<()> {
        if spec.parameters.num_qubits == 0 {
            return Err(QuantRS2Error::InvalidParameter(
                "num_qubits must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}
