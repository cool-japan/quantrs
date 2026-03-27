//! # IBMOptimizer - Trait Implementations
//!
//! Implements `ProviderOptimizer` for IBM Quantum.  The optimizer:
//!
//! - Prefers the CNOT + Rz + √X (Sx) native gate set exposed by all IBM Eagle
//!   and Heron processors.
//! - Recommends Sabre qubit-routing to minimise SWAP overhead on heavy-hex
//!   topologies.
//! - Estimates cost using IBM Quantum's runtime-second pricing (free-tier ≤ 600 s/month;
//!   premium plan at $1.60 / runtime-second).
//! - Selects the least-busy backend with sufficient qubit count.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::traits::ProviderOptimizer;
use super::types::*;
use crate::prelude::CloudProvider;
use crate::DeviceResult;
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// IBM Quantum pricing constants (USD, 2024 schedule)
// ---------------------------------------------------------------------------

/// Free-tier monthly runtime budget (seconds).  Jobs that fit within this
/// window cost nothing.
const IBM_FREE_TIER_SECONDS_PER_MONTH: f64 = 600.0;

/// Premium pricing per runtime-second (USD).
const IBM_PREMIUM_PER_RUNTIME_SECOND_USD: f64 = 1.60;

// ---------------------------------------------------------------------------
// IBM backend catalogue (representative subset, 2024 Q3)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct IbmBackendSpec {
    name: &'static str,
    qubit_count: usize,
    /// Current (simulated) queue depth as a fraction of daily capacity [0,1].
    /// A lower value means the backend is less busy and a job will start sooner.
    relative_queue_depth: f64,
    /// Representative CNOT fidelity.
    cx_fidelity: f64,
    /// Typical runtime per 1 000 shots (seconds).
    runtime_per_1k_shots_s: f64,
}

const IBM_BACKENDS: &[IbmBackendSpec] = &[
    IbmBackendSpec {
        name: "ibm_brisbane",
        qubit_count: 127,
        relative_queue_depth: 0.55,
        cx_fidelity: 0.985,
        runtime_per_1k_shots_s: 2.5,
    },
    IbmBackendSpec {
        name: "ibm_kyoto",
        qubit_count: 127,
        relative_queue_depth: 0.40,
        cx_fidelity: 0.982,
        runtime_per_1k_shots_s: 2.6,
    },
    IbmBackendSpec {
        name: "ibm_osaka",
        qubit_count: 127,
        relative_queue_depth: 0.70,
        cx_fidelity: 0.987,
        runtime_per_1k_shots_s: 2.4,
    },
    IbmBackendSpec {
        name: "ibm_sherbrooke",
        qubit_count: 127,
        relative_queue_depth: 0.30,
        cx_fidelity: 0.983,
        runtime_per_1k_shots_s: 2.5,
    },
    IbmBackendSpec {
        name: "ibm_torino",
        qubit_count: 133,
        relative_queue_depth: 0.25,
        cx_fidelity: 0.991,
        runtime_per_1k_shots_s: 2.2,
    },
];

/// Select the least-busy IBM backend that can host `qubit_count` qubits
/// and satisfies `min_fidelity`.
fn select_ibm_backend(qubit_count: usize, min_fidelity: f64) -> Option<&'static IbmBackendSpec> {
    IBM_BACKENDS
        .iter()
        .filter(|b| b.qubit_count >= qubit_count && b.cx_fidelity >= min_fidelity)
        .min_by(|a, b| {
            a.relative_queue_depth
                .partial_cmp(&b.relative_queue_depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

// ---------------------------------------------------------------------------
// ProviderOptimizer implementation
// ---------------------------------------------------------------------------

impl ProviderOptimizer for IBMOptimizer {
    /// Produce an `OptimizationRecommendation` for executing the workload on
    /// IBM Quantum.
    ///
    /// Strategy:
    /// 1. Select the least-busy backend that meets the qubit and fidelity
    ///    requirements.
    /// 2. Enable CNOT+Rz+Sx gate set via aggressive transpilation.
    /// 3. Apply readout error mitigation (TREX / M3) and ZNE if fidelity is
    ///    critical (depth > 50).
    fn optimize_workload(
        &self,
        workload: &WorkloadSpec,
    ) -> DeviceResult<OptimizationRecommendation> {
        let qubit_count = workload.circuit_characteristics.qubit_count;
        let shots = workload.execution_requirements.shots;
        let min_fidelity = workload
            .circuit_characteristics
            .coherence_requirements
            .min_gate_fidelity;
        let depth = workload.circuit_characteristics.circuit_depth;

        let primary = select_ibm_backend(qubit_count, min_fidelity).ok_or_else(|| {
            crate::DeviceError::InvalidInput(format!(
                "No IBM Quantum backend can accommodate {qubit_count} qubits \
                 with CNOT fidelity ≥ {min_fidelity:.3}"
            ))
        })?;

        // Enable ZNE for deep circuits (depth > 50) where decoherence matters.
        let use_zne = depth > 50;

        let recommended_config = ExecutionConfig {
            provider: CloudProvider::IBM,
            backend: primary.name.to_string(),
            optimization_settings: OptimizationSettings {
                circuit_optimization: CircuitOptimizationSettings {
                    gate_fusion: true,
                    gate_cancellation: true,
                    circuit_compression: true,
                    transpilation_level: TranspilationLevel::Advanced,
                    error_mitigation: ErrorMitigationSettings {
                        zero_noise_extrapolation: use_zne,
                        readout_error_mitigation: true,
                        gate_error_mitigation: false,
                        decoherence_mitigation: use_zne,
                        crosstalk_mitigation: false,
                    },
                },
                hardware_optimization: HardwareOptimizationSettings {
                    qubit_mapping: QubitMappingStrategy::NoiseAdaptive,
                    routing_optimization: RoutingOptimizationStrategy::MinimumSwaps,
                    calibration_optimization: CalibrationOptimizationStrategy::Dynamic,
                    noise_adaptation: NoiseAdaptationStrategy::ModelBased,
                },
                ..OptimizationSettings::default()
            },
            ..ExecutionConfig::default()
        };

        let cost_estimate = compute_ibm_cost(shots, primary);
        let perf_prediction = compute_ibm_performance(workload, primary);

        let alternatives: Vec<AlternativeRecommendation> = IBM_BACKENDS
            .iter()
            .filter(|b| {
                b.name != primary.name
                    && b.qubit_count >= qubit_count
                    && b.cx_fidelity >= min_fidelity
            })
            .map(|b| {
                let alt_cost = compute_ibm_cost(shots, b);
                let alt_perf = compute_ibm_performance(workload, b);
                AlternativeRecommendation {
                    alternative_id: Uuid::new_v4().to_string(),
                    config: ExecutionConfig {
                        provider: CloudProvider::IBM,
                        backend: b.name.to_string(),
                        optimization_settings: OptimizationSettings::default(),
                        ..ExecutionConfig::default()
                    },
                    trade_offs: TradeOffAnalysis {
                        performance_impact: alt_perf.expected_fidelity
                            - perf_prediction.expected_fidelity,
                        cost_impact: alt_cost.total_cost - cost_estimate.total_cost,
                        reliability_impact: 0.0,
                        complexity_impact: 0.0,
                        trade_off_summary: format!(
                            "{} — queue depth {:.0}%, Δcost ${:+.2}, Δfidelity {:+.3}",
                            b.name,
                            b.relative_queue_depth * 100.0,
                            alt_cost.total_cost - cost_estimate.total_cost,
                            alt_perf.expected_fidelity - perf_prediction.expected_fidelity
                        ),
                    },
                    use_case_suitability: alt_perf.success_probability,
                }
            })
            .collect();

        let rationale = format!(
            "Selected {} as the least-busy IBM Quantum backend with ≥ {qubit_count} qubits \
             and CNOT fidelity ≥ {min_fidelity:.3}. \
             Current relative queue depth: {:.0}%. \
             ZNE mitigation: {}. Estimated cost: ${:.2}.",
            primary.name,
            primary.relative_queue_depth * 100.0,
            if use_zne { "enabled" } else { "disabled" },
            cost_estimate.total_cost,
        );

        Ok(OptimizationRecommendation {
            recommendation_id: Uuid::new_v4().to_string(),
            workload_id: workload.workload_id.clone(),
            provider: CloudProvider::IBM,
            recommended_config,
            optimization_strategies: self.get_optimization_strategies(),
            expected_performance: perf_prediction,
            cost_estimate,
            confidence_score: 0.88,
            rationale,
            alternative_recommendations: alternatives,
        })
    }

    fn get_provider(&self) -> CloudProvider {
        CloudProvider::IBM
    }

    fn get_optimization_strategies(&self) -> Vec<OptimizationStrategy> {
        vec![
            OptimizationStrategy::CircuitOptimization,
            OptimizationStrategy::HardwareSelection,
            OptimizationStrategy::ErrorMitigation,
        ]
    }

    fn predict_performance(
        &self,
        workload: &WorkloadSpec,
        config: &ExecutionConfig,
    ) -> DeviceResult<PerformancePrediction> {
        let backend = IBM_BACKENDS
            .iter()
            .find(|b| b.name == config.backend)
            .unwrap_or(&IBM_BACKENDS[0]);
        Ok(compute_ibm_performance(workload, backend))
    }

    fn estimate_cost(
        &self,
        workload: &WorkloadSpec,
        config: &ExecutionConfig,
    ) -> DeviceResult<CostEstimate> {
        let backend = IBM_BACKENDS
            .iter()
            .find(|b| b.name == config.backend)
            .unwrap_or(&IBM_BACKENDS[0]);
        Ok(compute_ibm_cost(
            workload.execution_requirements.shots,
            backend,
        ))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn compute_ibm_cost(shots: usize, backend: &IbmBackendSpec) -> CostEstimate {
    // Runtime (seconds) = shots / 1000 × time-per-1k-shots
    let runtime_s = (shots as f64 / 1000.0) * backend.runtime_per_1k_shots_s;

    // Apply free tier: first 600 s/month are free.
    let billable_s = (runtime_s - IBM_FREE_TIER_SECONDS_PER_MONTH).max(0.0);
    let execution_cost = billable_s * IBM_PREMIUM_PER_RUNTIME_SECOND_USD;
    let total = execution_cost;
    let uncertainty = (total * 0.20).max(0.01);

    CostEstimate {
        total_cost: total,
        cost_breakdown: CostBreakdown {
            execution_cost,
            queue_cost: 0.0,
            storage_cost: 0.0,
            network_cost: 0.0,
            overhead_cost: 0.0,
            discount_applied: (runtime_s.min(IBM_FREE_TIER_SECONDS_PER_MONTH))
                * IBM_PREMIUM_PER_RUNTIME_SECOND_USD,
        },
        cost_model: CostModel::PayPerUse,
        uncertainty_range: (total - uncertainty, total + uncertainty),
        cost_optimization_opportunities: vec![CostOptimizationOpportunity {
            opportunity_type: CostOptimizationType::SchedulingOptimization,
            potential_savings: total * 0.15,
            implementation_effort: 0.3,
            description: "Schedule during off-peak hours to reduce queue wait time \
                           and stay within the monthly free-tier budget."
                .to_string(),
        }],
    }
}

fn compute_ibm_performance(
    workload: &WorkloadSpec,
    backend: &IbmBackendSpec,
) -> PerformancePrediction {
    let cc = &workload.circuit_characteristics;
    let shots = workload.execution_requirements.shots;

    // CNOT error accumulates per two-qubit layer.
    let cx_layers = (cc.circuit_depth / 3).max(1) as f64;
    let expected_fidelity = backend.cx_fidelity.powf(cx_layers).clamp(0.01, 1.0);

    // Queue wait: proportional to relative queue depth (0–300 s range).
    let queue_s = backend.relative_queue_depth * 300.0;

    // Execution time modelled as runtime per 1k shots.
    let exec_s = (shots as f64 / 1000.0) * backend.runtime_per_1k_shots_s;

    let success_probability = (expected_fidelity * 0.98).clamp(0.0, 1.0);

    PerformancePrediction {
        execution_time: Duration::from_secs_f64(exec_s),
        queue_time: Duration::from_secs_f64(queue_s),
        total_time: Duration::from_secs_f64(exec_s + queue_s),
        success_probability,
        expected_fidelity,
        resource_utilization: ResourceUtilizationPrediction {
            cpu_utilization: 0.03,
            memory_utilization: 0.08,
            quantum_resource_utilization: cc.qubit_count as f64 / backend.qubit_count as f64,
            network_utilization: 0.01,
            storage_utilization: 0.005,
        },
        bottlenecks: Vec::new(),
        confidence_interval: (success_probability * 0.93, success_probability * 1.04),
    }
}
