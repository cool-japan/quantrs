//! # GoogleOptimizer - Trait Implementations
//!
//! Implements `ProviderOptimizer` for Google Quantum AI.  The optimizer:
//!
//! - Targets the Sycamore processor family (Weber 53-qubit / Rainbow 23-qubit).
//! - Recommends CZ + √iSWAP native gate transpilation (the two native two-qubit
//!   gates on Sycamore hardware).
//! - Estimates cost using Google's research-access pricing model; commercial
//!   pricing is modelled conservatively as $0.00090 per gate-qubit operation.
//! - Selects the best processor based on qubit count and fidelity requirements.
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
// Pricing constants (USD, 2024 Google Cloud Quantum AI commercial estimate)
// ---------------------------------------------------------------------------

/// Estimated per-gate-qubit cost on Google Quantum AI hardware (USD).
/// Note: public commercial pricing is not yet officially published; this is
/// based on published research-access estimates for academic/commercial users.
const GOOGLE_PER_GATE_QUBIT_USD: f64 = 0.00090;

/// Flat per-job overhead fee (USD).
const GOOGLE_PER_JOB_FEE_USD: f64 = 0.50;

// ---------------------------------------------------------------------------
// Processor catalogue
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct GoogleProcessorSpec {
    name: &'static str,
    qubit_count: usize,
    /// Typical CZ gate fidelity.
    cz_fidelity: f64,
    /// Typical √iSWAP gate fidelity.
    sqrt_iswap_fidelity: f64,
    /// Relative queue depth (0–1).
    relative_queue_depth: f64,
    /// Typical queue latency (seconds).
    queue_latency_s: f64,
}

const GOOGLE_PROCESSORS: &[GoogleProcessorSpec] = &[
    GoogleProcessorSpec {
        name: "google_sycamore_weber",
        qubit_count: 53,
        cz_fidelity: 0.994,
        sqrt_iswap_fidelity: 0.992,
        relative_queue_depth: 0.60,
        queue_latency_s: 120.0,
    },
    GoogleProcessorSpec {
        name: "google_sycamore_rainbow",
        qubit_count: 23,
        cz_fidelity: 0.991,
        sqrt_iswap_fidelity: 0.989,
        relative_queue_depth: 0.45,
        queue_latency_s: 90.0,
    },
    GoogleProcessorSpec {
        name: "google_willow",
        qubit_count: 105,
        cz_fidelity: 0.997,
        sqrt_iswap_fidelity: 0.996,
        relative_queue_depth: 0.80,
        queue_latency_s: 180.0,
    },
];

/// Return the best processor for the given qubit count and minimum fidelity
/// based on a fidelity-weighted selection.
fn select_google_processor(
    qubit_count: usize,
    min_fidelity: f64,
) -> Option<&'static GoogleProcessorSpec> {
    GOOGLE_PROCESSORS
        .iter()
        .filter(|p| {
            p.qubit_count >= qubit_count
                && p.cz_fidelity >= min_fidelity
                && p.sqrt_iswap_fidelity >= min_fidelity
        })
        .max_by(|a, b| {
            // Prefer highest CZ fidelity; break ties by lower queue depth.
            let score_a = a.cz_fidelity - a.relative_queue_depth * 0.05;
            let score_b = b.cz_fidelity - b.relative_queue_depth * 0.05;
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

// ---------------------------------------------------------------------------
// ProviderOptimizer implementation
// ---------------------------------------------------------------------------

impl ProviderOptimizer for GoogleOptimizer {
    /// Produce an `OptimizationRecommendation` for executing the workload on
    /// Google Quantum AI.
    ///
    /// Strategy:
    /// 1. Select the best Sycamore processor meeting qubit and fidelity needs.
    /// 2. Recommend CZ + √iSWAP native gate transpilation (cirq-core equivalent).
    /// 3. Apply readout error mitigation for circuits with depth > 20.
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
        let circuit_depth = workload.circuit_characteristics.circuit_depth;

        let primary = select_google_processor(qubit_count, min_fidelity).ok_or_else(|| {
            crate::DeviceError::InvalidInput(format!(
                "No Google Quantum AI processor can accommodate {qubit_count} qubits \
                 with CZ / √iSWAP fidelity ≥ {min_fidelity:.3}"
            ))
        })?;

        let use_rem = circuit_depth > 20;

        let recommended_config = ExecutionConfig {
            provider: CloudProvider::Google,
            backend: primary.name.to_string(),
            optimization_settings: OptimizationSettings {
                circuit_optimization: CircuitOptimizationSettings {
                    gate_fusion: true,
                    gate_cancellation: true,
                    circuit_compression: true,
                    transpilation_level: TranspilationLevel::Advanced,
                    error_mitigation: ErrorMitigationSettings {
                        zero_noise_extrapolation: false,
                        readout_error_mitigation: use_rem,
                        gate_error_mitigation: false,
                        decoherence_mitigation: false,
                        crosstalk_mitigation: false,
                    },
                },
                hardware_optimization: HardwareOptimizationSettings {
                    qubit_mapping: QubitMappingStrategy::TopologyAware,
                    routing_optimization: RoutingOptimizationStrategy::NoiseAware,
                    calibration_optimization: CalibrationOptimizationStrategy::RealTime,
                    noise_adaptation: NoiseAdaptationStrategy::ModelBased,
                },
                ..OptimizationSettings::default()
            },
            ..ExecutionConfig::default()
        };

        let cost_estimate = compute_google_cost(shots, circuit_depth, qubit_count, primary);
        let perf_prediction = compute_google_performance(workload, primary);

        let alternatives: Vec<AlternativeRecommendation> = GOOGLE_PROCESSORS
            .iter()
            .filter(|p| {
                p.name != primary.name
                    && p.qubit_count >= qubit_count
                    && p.cz_fidelity >= min_fidelity
            })
            .map(|p| {
                let alt_cost = compute_google_cost(shots, circuit_depth, qubit_count, p);
                let alt_perf = compute_google_performance(workload, p);
                AlternativeRecommendation {
                    alternative_id: Uuid::new_v4().to_string(),
                    config: ExecutionConfig {
                        provider: CloudProvider::Google,
                        backend: p.name.to_string(),
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
                            "{} — {}-qubit, queue depth {:.0}%, Δcost ${:+.4}, Δfidelity {:+.4}",
                            p.name,
                            p.qubit_count,
                            p.relative_queue_depth * 100.0,
                            alt_cost.total_cost - cost_estimate.total_cost,
                            alt_perf.expected_fidelity - perf_prediction.expected_fidelity
                        ),
                    },
                    use_case_suitability: alt_perf.success_probability,
                }
            })
            .collect();

        let rationale = format!(
            "Selected Google Quantum AI processor {} ({} qubits) as the best option \
             for {qubit_count} qubits with CZ fidelity ≥ {min_fidelity:.3}. \
             CZ fidelity: {:.4}, √iSWAP fidelity: {:.4}. \
             REM: {}. Estimated cost: ${:.4}.",
            primary.name,
            primary.qubit_count,
            primary.cz_fidelity,
            primary.sqrt_iswap_fidelity,
            if use_rem { "enabled" } else { "disabled" },
            cost_estimate.total_cost,
        );

        Ok(OptimizationRecommendation {
            recommendation_id: Uuid::new_v4().to_string(),
            workload_id: workload.workload_id.clone(),
            provider: CloudProvider::Google,
            recommended_config,
            optimization_strategies: self.get_optimization_strategies(),
            expected_performance: perf_prediction,
            cost_estimate,
            confidence_score: 0.80,
            rationale,
            alternative_recommendations: alternatives,
        })
    }

    fn get_provider(&self) -> CloudProvider {
        CloudProvider::Google
    }

    fn get_optimization_strategies(&self) -> Vec<OptimizationStrategy> {
        vec![
            OptimizationStrategy::CircuitOptimization,
            OptimizationStrategy::PerformanceOptimization,
            OptimizationStrategy::ResourceProvisioning,
        ]
    }

    fn predict_performance(
        &self,
        workload: &WorkloadSpec,
        config: &ExecutionConfig,
    ) -> DeviceResult<PerformancePrediction> {
        let processor = GOOGLE_PROCESSORS
            .iter()
            .find(|p| p.name == config.backend)
            .unwrap_or(&GOOGLE_PROCESSORS[0]);
        Ok(compute_google_performance(workload, processor))
    }

    fn estimate_cost(
        &self,
        workload: &WorkloadSpec,
        config: &ExecutionConfig,
    ) -> DeviceResult<CostEstimate> {
        let processor = GOOGLE_PROCESSORS
            .iter()
            .find(|p| p.name == config.backend)
            .unwrap_or(&GOOGLE_PROCESSORS[0]);
        Ok(compute_google_cost(
            workload.execution_requirements.shots,
            workload.circuit_characteristics.circuit_depth,
            workload.circuit_characteristics.qubit_count,
            processor,
        ))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn compute_google_cost(
    shots: usize,
    circuit_depth: usize,
    qubit_count: usize,
    processor: &GoogleProcessorSpec,
) -> CostEstimate {
    // Cost model: per-job fee + shots × depth × qubits × per_gate_qubit_rate.
    let gate_qubit_ops = shots as f64 * circuit_depth as f64 * qubit_count as f64;
    let execution_cost = gate_qubit_ops * GOOGLE_PER_GATE_QUBIT_USD;
    let total = execution_cost + GOOGLE_PER_JOB_FEE_USD;
    let uncertainty = total * 0.25; // Wide uncertainty — pricing not fully public.

    CostEstimate {
        total_cost: total,
        cost_breakdown: CostBreakdown {
            execution_cost,
            queue_cost: 0.0,
            storage_cost: 0.0,
            network_cost: 0.0,
            overhead_cost: GOOGLE_PER_JOB_FEE_USD,
            discount_applied: 0.0,
        },
        cost_model: CostModel::PayPerUse,
        uncertainty_range: (total - uncertainty, total + uncertainty),
        cost_optimization_opportunities: vec![CostOptimizationOpportunity {
            opportunity_type: CostOptimizationType::ResourceRightSizing,
            potential_savings: total * 0.20,
            implementation_effort: 0.30,
            description: "Reduce circuit depth with Cirq optimisation passes \
                           (gate cancellation, two-qubit compilation) to lower the \
                           gate-qubit operation count."
                .to_string(),
        }],
    }
}

fn compute_google_performance(
    workload: &WorkloadSpec,
    processor: &GoogleProcessorSpec,
) -> PerformancePrediction {
    let cc = &workload.circuit_characteristics;
    let shots = workload.execution_requirements.shots;

    // Fidelity model: CZ error accumulates per two-qubit layer.
    let cz_layers = (cc.circuit_depth / 2).max(1) as f64;
    let expected_fidelity = processor.cz_fidelity.powf(cz_layers).clamp(0.01, 1.0);
    let success_probability = (expected_fidelity * 0.975).clamp(0.0, 1.0);

    // Sycamore executes at ~25 ns per CZ; 1 000 shots ≈ 1 s for typical depth.
    let exec_s = shots as f64 * 0.001;

    PerformancePrediction {
        execution_time: Duration::from_secs_f64(exec_s),
        queue_time: Duration::from_secs_f64(processor.queue_latency_s),
        total_time: Duration::from_secs_f64(exec_s + processor.queue_latency_s),
        success_probability,
        expected_fidelity,
        resource_utilization: ResourceUtilizationPrediction {
            cpu_utilization: 0.04,
            memory_utilization: 0.10,
            quantum_resource_utilization: cc.qubit_count as f64 / processor.qubit_count as f64,
            network_utilization: 0.02,
            storage_utilization: 0.01,
        },
        bottlenecks: Vec::new(),
        confidence_interval: (success_probability * 0.90, success_probability * 1.05),
    }
}
