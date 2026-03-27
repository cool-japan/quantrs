//! # ShorTemplate - Trait Implementations
//!
//! This module contains trait implementations for `ShorTemplate`.
//!
//! ## Implemented Traits
//!
//! - `AlgorithmTemplate`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::AlgorithmTemplate;
use super::types::{
    AlgorithmSpecification, QAOATemplate, QuantumAlgorithmType, ResourceEstimates, ShorTemplate,
    SynthesizedCircuit, TemplateInfo,
};
use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::QubitId;
use std::collections::HashMap;
use std::time::Duration;

impl AlgorithmTemplate for ShorTemplate {
    fn synthesize(&self, spec: &AlgorithmSpecification) -> QuantRS2Result<SynthesizedCircuit> {
        QAOATemplate::new().create_circuit_from_gates(
            vec![],
            spec.parameters.num_qubits,
            QuantumAlgorithmType::Shor,
        )
    }
    fn estimate_resources(
        &self,
        spec: &AlgorithmSpecification,
    ) -> QuantRS2Result<ResourceEstimates> {
        let n = spec.parameters.num_qubits;
        Ok(ResourceEstimates {
            gate_count: n.pow(3),
            circuit_depth: n.pow(2),
            qubit_count: n,
            gate_breakdown: HashMap::new(),
            estimated_execution_time: Duration::from_millis((n.pow(3) / 1000) as u64),
            memory_requirements: 1 << n,
            parallelization_factor: 0.6,
        })
    }
    fn get_template_info(&self) -> TemplateInfo {
        TemplateInfo {
            name: "Shor".to_string(),
            supported_parameters: vec![
                "num_qubits".to_string(),
                "factorization_target".to_string(),
            ],
            required_parameters: vec!["num_qubits".to_string()],
            complexity_scaling: "O((log N)^3)".to_string(),
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
