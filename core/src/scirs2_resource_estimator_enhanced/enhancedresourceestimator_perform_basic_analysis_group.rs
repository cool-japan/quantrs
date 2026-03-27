//! # EnhancedResourceEstimator - perform_basic_analysis_group Methods
//!
//! This module contains method implementations for `EnhancedResourceEstimator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::QuantRS2Error;
use crate::gate_translation::GateType;
use crate::parallel_ops_stubs::*;
use crate::resource_estimator::{
    ErrorCorrectionCode, EstimationMode, HardwarePlatform, QuantumGate, ResourceEstimationConfig,
};

use super::types::{
    BasicResourceAnalysis, CircuitTopology, ComplexityMetrics, GateStatistics, ResourceRequirements,
};

use super::enhancedresourceestimator_type::EnhancedResourceEstimator;

impl EnhancedResourceEstimator {
    /// Perform basic resource analysis
    pub(super) fn perform_basic_analysis(
        &self,
        circuit: &[QuantumGate],
        num_qubits: usize,
    ) -> Result<BasicResourceAnalysis, QuantRS2Error> {
        let gate_stats = self.analyze_gate_statistics(circuit)?;
        let topology = self.analyze_circuit_topology(circuit, num_qubits)?;
        let requirements = self.calculate_resource_requirements(&gate_stats, &topology)?;
        let complexity = self.calculate_complexity_metrics(circuit, &topology)?;
        Ok(BasicResourceAnalysis {
            gate_statistics: gate_stats,
            circuit_topology: topology,
            resource_requirements: requirements,
            complexity_metrics: complexity,
            num_qubits,
            circuit_size: circuit.len(),
        })
    }
    /// Analyze circuit topology
    pub(super) fn analyze_circuit_topology(
        &self,
        circuit: &[QuantumGate],
        num_qubits: usize,
    ) -> Result<CircuitTopology, QuantRS2Error> {
        let mut connectivity = vec![vec![0; num_qubits]; num_qubits];
        let mut interaction_count = 0;
        for gate in circuit {
            if gate.target_qubits().len() >= 2 {
                let q1 = gate.target_qubits()[0];
                let q2 = gate.target_qubits()[1];
                connectivity[q1][q2] += 1;
                connectivity[q2][q1] += 1;
                interaction_count += 1;
            }
        }
        let connectivity_density = if num_qubits > 1 {
            interaction_count as f64 / ((num_qubits * (num_qubits - 1)) / 2) as f64
        } else {
            0.0
        };
        let max_connections = connectivity
            .iter()
            .map(|row| row.iter().filter(|&&x| x > 0).count())
            .max()
            .unwrap_or(0);
        let critical_qubits = Self::identify_critical_qubits(&connectivity)?;
        Ok(CircuitTopology {
            num_qubits,
            connectivity_matrix: connectivity.clone(),
            connectivity_density,
            max_connections,
            critical_qubits,
            topology_type: Self::classify_topology(&connectivity, connectivity_density),
        })
    }
    /// Calculate resource requirements
    pub(super) fn calculate_resource_requirements(
        &self,
        gate_stats: &GateStatistics,
        topology: &CircuitTopology,
    ) -> Result<ResourceRequirements, QuantRS2Error> {
        let code_distance = self.estimate_code_distance()?;
        let physical_qubits = self.estimate_physical_qubits(topology.num_qubits, code_distance)?;
        let execution_time = self.estimate_execution_time(gate_stats)?;
        let memory_requirements =
            self.estimate_memory_requirements(topology.num_qubits, gate_stats)?;
        let magic_states = self.estimate_magic_states(gate_stats)?;
        Ok(ResourceRequirements {
            logical_qubits: topology.num_qubits,
            physical_qubits,
            code_distance,
            execution_time,
            memory_requirements,
            magic_states,
            error_budget: self.calculate_error_budget()?,
        })
    }
    /// Calculate complexity metrics
    pub(super) fn calculate_complexity_metrics(
        &self,
        circuit: &[QuantumGate],
        topology: &CircuitTopology,
    ) -> Result<ComplexityMetrics, QuantRS2Error> {
        let t_complexity = circuit
            .iter()
            .filter(|g| matches!(g.gate_type(), GateType::T))
            .count();
        let t_depth = self.calculate_t_depth(circuit)?;
        let circuit_volume = topology.num_qubits * circuit.len();
        let communication_complexity = topology.connectivity_density * topology.num_qubits as f64;
        let entanglement_complexity = self.estimate_entanglement_complexity(circuit)?;
        Ok(ComplexityMetrics {
            t_complexity,
            t_depth,
            circuit_volume,
            communication_complexity,
            entanglement_complexity,
            algorithmic_complexity: self.classify_algorithmic_complexity(circuit)?,
        })
    }
}
