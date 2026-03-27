//! # AWSOptimizer - Trait Implementations
//!
//! Implements `ProviderOptimizer` for AWS Braket. The optimizer:
//!
//! - Selects the cheapest backend that meets fidelity requirements (IonQ, Rigetti OQC).
//! - Estimates cost using AWS Braket per-task and per-shot pricing (as of 2024).
//! - Predicts performance by modelling gate-count, circuit depth, and shot count
//!   against empirical backend characteristics.
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
// AWS Braket pricing constants (USD, 2024 schedule)
// ---------------------------------------------------------------------------

/// Per-task base fee (USD) charged regardless of shot count.
const AWS_PER_TASK_FEE_USD: f64 = 0.30;

/// Per-shot price on IonQ hardware (USD).
const AWS_IONQ_PER_SHOT_USD: f64 = 0.00035;

/// Per-shot price on Rigetti hardware (USD).
const AWS_RIGETTI_PER_SHOT_USD: f64 = 0.00075;

/// Per-shot price on OQC hardware (USD).
const AWS_OQC_PER_SHOT_USD: f64 = 0.00035;

// ---------------------------------------------------------------------------
// Backend fidelity reference values (empirical, dimensionless ∈ [0,1])
// ---------------------------------------------------------------------------

/// Expected two-qubit gate fidelity on IonQ Harmony.
const IONQ_TWO_QUBIT_FIDELITY: f64 = 0.965;

/// Expected two-qubit gate fidelity on Rigetti Aspen-M series.
const RIGETTI_TWO_QUBIT_FIDELITY: f64 = 0.935;

/// Expected two-qubit gate fidelity on OQC Lucy.
const OQC_TWO_QUBIT_FIDELITY: f64 = 0.958;

// ---------------------------------------------------------------------------
// Helper: choose the cheapest backend that meets minimum fidelity
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct BackendSpec {
    name: &'static str,
    per_shot_usd: f64,
    two_qubit_fidelity: f64,
    max_qubits: usize,
    /// Typical per-shot queue latency (seconds).
    queue_latency_s: f64,
}

const AWS_BACKENDS: &[BackendSpec] = &[
    BackendSpec {
        name: "ionq.harmony",
        per_shot_usd: AWS_IONQ_PER_SHOT_USD,
        two_qubit_fidelity: IONQ_TWO_QUBIT_FIDELITY,
        max_qubits: 11,
        queue_latency_s: 30.0,
    },
    BackendSpec {
        name: "ionq.aria-1",
        per_shot_usd: AWS_IONQ_PER_SHOT_USD,
        two_qubit_fidelity: 0.975,
        max_qubits: 25,
        queue_latency_s: 45.0,
    },
    BackendSpec {
        name: "rigetti.aspen-m-3",
        per_shot_usd: AWS_RIGETTI_PER_SHOT_USD,
        two_qubit_fidelity: RIGETTI_TWO_QUBIT_FIDELITY,
        max_qubits: 79,
        queue_latency_s: 20.0,
    },
    BackendSpec {
        name: "oqc.lucy",
        per_shot_usd: AWS_OQC_PER_SHOT_USD,
        two_qubit_fidelity: OQC_TWO_QUBIT_FIDELITY,
        max_qubits: 8,
        queue_latency_s: 25.0,
    },
];

/// Select the cheapest backend that satisfies qubit count and fidelity
/// constraints.  Returns `None` when no backend can accommodate the workload.
fn select_aws_backend(qubit_count: usize, min_fidelity: f64) -> Option<&'static BackendSpec> {
    AWS_BACKENDS
        .iter()
        .filter(|b| b.max_qubits >= qubit_count && b.two_qubit_fidelity >= min_fidelity)
        .min_by(|a, b| {
            a.per_shot_usd
                .partial_cmp(&b.per_shot_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

// ---------------------------------------------------------------------------
// ProviderOptimizer implementation
// ---------------------------------------------------------------------------

impl ProviderOptimizer for AWSOptimizer {
    /// Produce an `OptimizationRecommendation` for executing the workload on
    /// AWS Braket.
    ///
    /// The recommendation:
    /// 1. Selects the cheapest AWS Braket backend that fits the qubit count
    ///    and coherence requirements.
    /// 2. Recommends native-gate transpilation (IonQ: MS + Rz; Rigetti: CZ + Rz;
    ///    OQC: ECR + Rz) to minimise two-qubit gate overhead.
    /// 3. Scores cost and performance and populates `alternative_recommendations`
    ///    with other feasible backends for comparison.
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

        // Select the primary (cheapest-qualifying) backend.
        let primary = select_aws_backend(qubit_count, min_fidelity).ok_or_else(|| {
            crate::DeviceError::InvalidInput(format!(
                "No AWS Braket backend can accommodate {qubit_count} qubits \
                 with fidelity ≥ {min_fidelity:.3}"
            ))
        })?;

        // Build the recommended ExecutionConfig.
        let recommended_config = ExecutionConfig {
            provider: CloudProvider::AWS,
            backend: primary.name.to_string(),
            optimization_settings: OptimizationSettings {
                circuit_optimization: CircuitOptimizationSettings {
                    gate_fusion: true,
                    gate_cancellation: true,
                    circuit_compression: true,
                    transpilation_level: TranspilationLevel::Advanced,
                    error_mitigation: ErrorMitigationSettings {
                        zero_noise_extrapolation: false,
                        readout_error_mitigation: true,
                        gate_error_mitigation: false,
                        decoherence_mitigation: false,
                        crosstalk_mitigation: false,
                    },
                },
                ..OptimizationSettings::default()
            },
            ..ExecutionConfig::default()
        };

        // Build cost and performance predictions.
        let cost_estimate = compute_aws_cost(shots, primary);
        let perf_prediction = compute_aws_performance(workload, primary);

        // Build alternative recommendations from other feasible backends.
        let alternatives: Vec<AlternativeRecommendation> = AWS_BACKENDS
            .iter()
            .filter(|b| {
                b.name != primary.name
                    && b.max_qubits >= qubit_count
                    && b.two_qubit_fidelity >= min_fidelity
            })
            .map(|b| {
                let alt_cost = compute_aws_cost(shots, b);
                let alt_perf = compute_aws_performance(workload, b);
                AlternativeRecommendation {
                    alternative_id: Uuid::new_v4().to_string(),
                    config: ExecutionConfig {
                        provider: CloudProvider::AWS,
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
                            "{} — Δcost ${:+.4}, Δfidelity {:+.3}",
                            b.name,
                            alt_cost.total_cost - cost_estimate.total_cost,
                            alt_perf.expected_fidelity - perf_prediction.expected_fidelity
                        ),
                    },
                    use_case_suitability: alt_perf.success_probability,
                }
            })
            .collect();

        let rationale = format!(
            "Selected {} as the cheapest AWS Braket backend that supports {qubit_count} qubits \
             with two-qubit gate fidelity ≥ {min_fidelity:.3}. \
             Estimated cost: ${:.4} for {shots} shots.",
            primary.name, cost_estimate.total_cost,
        );

        Ok(OptimizationRecommendation {
            recommendation_id: Uuid::new_v4().to_string(),
            workload_id: workload.workload_id.clone(),
            provider: CloudProvider::AWS,
            recommended_config,
            optimization_strategies: self.get_optimization_strategies(),
            expected_performance: perf_prediction,
            cost_estimate,
            confidence_score: 0.82,
            rationale,
            alternative_recommendations: alternatives,
        })
    }

    fn get_provider(&self) -> CloudProvider {
        CloudProvider::AWS
    }

    fn get_optimization_strategies(&self) -> Vec<OptimizationStrategy> {
        vec![
            OptimizationStrategy::CostOptimization,
            OptimizationStrategy::LoadBalancing,
            OptimizationStrategy::ResourceProvisioning,
        ]
    }

    /// Predict execution performance for a given workload/config pair on
    /// AWS Braket.
    ///
    /// The model accounts for:
    /// - Gate count relative to coherence budget (decoherence penalty).
    /// - Qubit count vs backend capacity (routing overhead).
    /// - Shot count vs typical queue depth.
    fn predict_performance(
        &self,
        workload: &WorkloadSpec,
        config: &ExecutionConfig,
    ) -> DeviceResult<PerformancePrediction> {
        // Look up the backend spec; fall back to Rigetti if not found.
        let backend = AWS_BACKENDS
            .iter()
            .find(|b| b.name == config.backend)
            .unwrap_or(&AWS_BACKENDS[2]);

        Ok(compute_aws_performance(workload, backend))
    }

    /// Estimate cost for a given workload/config pair on AWS Braket.
    ///
    /// Pricing model:
    /// - `total_cost = per_task_fee + shots × per_shot_price`
    fn estimate_cost(
        &self,
        workload: &WorkloadSpec,
        config: &ExecutionConfig,
    ) -> DeviceResult<CostEstimate> {
        let backend = AWS_BACKENDS
            .iter()
            .find(|b| b.name == config.backend)
            .unwrap_or(&AWS_BACKENDS[2]);
        Ok(compute_aws_cost(
            workload.execution_requirements.shots,
            backend,
        ))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn compute_aws_cost(shots: usize, backend: &BackendSpec) -> CostEstimate {
    let execution_cost = shots as f64 * backend.per_shot_usd;
    let total = execution_cost + AWS_PER_TASK_FEE_USD;
    // 15 % uncertainty band on quantum hardware.
    let uncertainty = total * 0.15;

    CostEstimate {
        total_cost: total,
        cost_breakdown: CostBreakdown {
            execution_cost,
            queue_cost: 0.0,
            storage_cost: 0.0,
            network_cost: 0.0,
            overhead_cost: AWS_PER_TASK_FEE_USD,
            discount_applied: 0.0,
        },
        cost_model: CostModel::PayPerUse,
        uncertainty_range: (total - uncertainty, total + uncertainty),
        cost_optimization_opportunities: vec![CostOptimizationOpportunity {
            opportunity_type: CostOptimizationType::VolumeDiscount,
            potential_savings: total * 0.10,
            implementation_effort: 0.2,
            description: "Consolidate multiple small jobs into a single batch task to \
                           amortise the per-task fee."
                .to_string(),
        }],
    }
}

fn compute_aws_performance(
    workload: &WorkloadSpec,
    backend: &BackendSpec,
) -> PerformancePrediction {
    let cc = &workload.circuit_characteristics;
    let shots = workload.execution_requirements.shots;

    // Decoherence penalty: fidelity degrades roughly as (fidelity)^(2q_gates).
    // We count two-qubit gates as circuit_depth / 2 (rough approximation).
    let two_qubit_gates = (cc.circuit_depth / 2).max(1) as f64;
    let expected_fidelity = (backend.two_qubit_fidelity.powf(two_qubit_gates)).clamp(0.01, 1.0);

    // Routing overhead: if qubit count > half capacity, add a depth penalty.
    let routing_penalty = if cc.qubit_count as f64 > backend.max_qubits as f64 * 0.6 {
        0.90
    } else {
        1.0
    };
    let adjusted_fidelity = (expected_fidelity * routing_penalty).clamp(0.0, 1.0);

    // Success probability: product of readout fidelity (0.97) and gate fidelity.
    let success_probability = (adjusted_fidelity * 0.97).clamp(0.0, 1.0);

    // Execution time: each shot takes ~1 ms on trapped-ion, plus queue wait.
    let exec_time_ms = shots as f64 * 1.5;
    let queue_time_s = backend.queue_latency_s;

    PerformancePrediction {
        execution_time: Duration::from_millis(exec_time_ms as u64),
        queue_time: Duration::from_secs_f64(queue_time_s),
        total_time: Duration::from_millis(exec_time_ms as u64)
            + Duration::from_secs_f64(queue_time_s),
        success_probability,
        expected_fidelity: adjusted_fidelity,
        resource_utilization: ResourceUtilizationPrediction {
            cpu_utilization: 0.05,
            memory_utilization: 0.10,
            quantum_resource_utilization: cc.qubit_count as f64 / backend.max_qubits as f64,
            network_utilization: 0.02,
            storage_utilization: 0.01,
        },
        bottlenecks: Vec::new(),
        confidence_interval: (success_probability * 0.92, success_probability * 1.05),
    }
}
