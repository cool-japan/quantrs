//! # AzureOptimizer - Trait Implementations
//!
//! Implements `ProviderOptimizer` for Azure Quantum.  The optimizer:
//!
//! - Supports IonQ and Quantinuum (formerly Cambridge Quantum/Honeywell) targets.
//! - Uses IonQ native gates (MS + R_zz + GPi) or Quantinuum TKET-compiled ZZ + Rz.
//! - Estimates cost using Azure Quantum pricing:
//!   - IonQ: $0.00097 per gate-qubit (circuit resource units).
//!   - Quantinuum: HQC credit packs (100 HQC credits ≈ $200; 1 HQC ≈ depth × qubits).
//! - Selects the provider by performance/cost tradeoff based on circuit
//!   characteristics.
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
// Pricing constants
// ---------------------------------------------------------------------------

/// IonQ: price per circuit resource unit (gate × qubit) in USD.
const AZURE_IONQ_PER_CRU_USD: f64 = 0.00097;

/// Quantinuum HQC credit pack: 100 credits for $200 USD.
const AZURE_QUANTINUUM_USD_PER_HQC: f64 = 2.00;

/// HQC formula coefficient: H-credit = circuit_depth × qubit_count / constant.
const AZURE_QUANTINUUM_HQC_DEPTH_CONSTANT: f64 = 5000.0;

// ---------------------------------------------------------------------------
// Backend catalogue
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AzureTarget {
    IonQAria,
    IonQHarmony,
    QuantinuumH1,
    QuantinuumH2,
}

#[derive(Debug, Clone, Copy)]
struct AzureBackendSpec {
    target: AzureTarget,
    name: &'static str,
    max_qubits: usize,
    two_qubit_fidelity: f64,
    /// Typical queue latency in seconds.
    queue_latency_s: f64,
}

const AZURE_BACKENDS: &[AzureBackendSpec] = &[
    AzureBackendSpec {
        target: AzureTarget::IonQAria,
        name: "ionq.simulator",
        max_qubits: 29,
        two_qubit_fidelity: 0.972,
        queue_latency_s: 20.0,
    },
    AzureBackendSpec {
        target: AzureTarget::IonQHarmony,
        name: "ionq.qpu",
        max_qubits: 11,
        two_qubit_fidelity: 0.965,
        queue_latency_s: 40.0,
    },
    AzureBackendSpec {
        target: AzureTarget::QuantinuumH1,
        name: "quantinuum.hqs-lt-s1",
        max_qubits: 20,
        two_qubit_fidelity: 0.997,
        queue_latency_s: 60.0,
    },
    AzureBackendSpec {
        target: AzureTarget::QuantinuumH2,
        name: "quantinuum.hqs-lt-s2",
        max_qubits: 32,
        two_qubit_fidelity: 0.998,
        queue_latency_s: 90.0,
    },
];

/// Estimate the cost for a single backend.
fn estimate_azure_cost_for_backend(
    shots: usize,
    circuit_depth: usize,
    qubit_count: usize,
    backend: &AzureBackendSpec,
) -> f64 {
    match backend.target {
        AzureTarget::IonQAria | AzureTarget::IonQHarmony => {
            // IonQ charges per circuit resource unit (CRU = gate_count × qubit_targets).
            // Approximation: gate_count ≈ circuit_depth × qubit_count / 2.
            let approx_gates = (circuit_depth * qubit_count) as f64 / 2.0;
            let cru = approx_gates * qubit_count as f64;
            cru * AZURE_IONQ_PER_CRU_USD * shots as f64
        }
        AzureTarget::QuantinuumH1 | AzureTarget::QuantinuumH2 => {
            // Quantinuum charges HQC credits proportional to circuit size.
            let hqc = (circuit_depth as f64 * qubit_count as f64)
                / AZURE_QUANTINUUM_HQC_DEPTH_CONSTANT
                * shots as f64;
            hqc * AZURE_QUANTINUUM_USD_PER_HQC
        }
    }
}

/// Select backend by best performance/cost ratio.
fn select_azure_backend(
    qubit_count: usize,
    min_fidelity: f64,
    circuit_depth: usize,
    shots: usize,
) -> Option<&'static AzureBackendSpec> {
    // Gather feasible candidates.
    let candidates: Vec<&AzureBackendSpec> = AZURE_BACKENDS
        .iter()
        .filter(|b| b.max_qubits >= qubit_count && b.two_qubit_fidelity >= min_fidelity)
        .collect();

    // Score each candidate by fidelity/cost ratio.
    candidates.into_iter().max_by(|a, b| {
        let cost_a =
            estimate_azure_cost_for_backend(shots, circuit_depth, qubit_count, a).max(1e-9);
        let cost_b =
            estimate_azure_cost_for_backend(shots, circuit_depth, qubit_count, b).max(1e-9);
        let score_a = a.two_qubit_fidelity / cost_a;
        let score_b = b.two_qubit_fidelity / cost_b;
        score_a
            .partial_cmp(&score_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

// ---------------------------------------------------------------------------
// ProviderOptimizer implementation
// ---------------------------------------------------------------------------

impl ProviderOptimizer for AzureOptimizer {
    /// Produce an `OptimizationRecommendation` for executing the workload on
    /// Azure Quantum.
    ///
    /// Selects the target with the best fidelity/cost tradeoff, recommends
    /// native-gate transpilation for the selected provider, and exposes
    /// alternative targets.
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

        let primary = select_azure_backend(qubit_count, min_fidelity, circuit_depth, shots)
            .ok_or_else(|| {
                crate::DeviceError::InvalidInput(format!(
                    "No Azure Quantum backend can accommodate {qubit_count} qubits \
                     with two-qubit fidelity ≥ {min_fidelity:.3}"
                ))
            })?;

        // Transpilation level depends on the target technology.
        let (gate_fusion, zne) = match primary.target {
            AzureTarget::QuantinuumH1 | AzureTarget::QuantinuumH2 => (true, circuit_depth > 40),
            _ => (true, false),
        };

        let recommended_config = ExecutionConfig {
            provider: CloudProvider::Azure,
            backend: primary.name.to_string(),
            optimization_settings: OptimizationSettings {
                circuit_optimization: CircuitOptimizationSettings {
                    gate_fusion,
                    gate_cancellation: true,
                    circuit_compression: true,
                    transpilation_level: TranspilationLevel::Aggressive,
                    error_mitigation: ErrorMitigationSettings {
                        zero_noise_extrapolation: zne,
                        readout_error_mitigation: true,
                        gate_error_mitigation: false,
                        decoherence_mitigation: false,
                        crosstalk_mitigation: false,
                    },
                },
                hardware_optimization: HardwareOptimizationSettings {
                    qubit_mapping: QubitMappingStrategy::FidelityOptimized,
                    routing_optimization: RoutingOptimizationStrategy::FidelityAware,
                    calibration_optimization: CalibrationOptimizationStrategy::Dynamic,
                    noise_adaptation: NoiseAdaptationStrategy::Statistical,
                },
                ..OptimizationSettings::default()
            },
            ..ExecutionConfig::default()
        };

        let cost_estimate = compute_azure_cost(shots, circuit_depth, qubit_count, primary);
        let perf_prediction = compute_azure_performance(workload, primary);

        let alternatives: Vec<AlternativeRecommendation> = AZURE_BACKENDS
            .iter()
            .filter(|b| {
                b.name != primary.name
                    && b.max_qubits >= qubit_count
                    && b.two_qubit_fidelity >= min_fidelity
            })
            .map(|b| {
                let alt_cost = compute_azure_cost(shots, circuit_depth, qubit_count, b);
                let alt_perf = compute_azure_performance(workload, b);
                AlternativeRecommendation {
                    alternative_id: Uuid::new_v4().to_string(),
                    config: ExecutionConfig {
                        provider: CloudProvider::Azure,
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
                            "{} — Δcost ${:+.4}, Δfidelity {:+.4}",
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
            "Selected {} on Azure Quantum as the best fidelity/cost tradeoff \
             for {qubit_count} qubits at ≥{min_fidelity:.3} two-qubit fidelity. \
             Estimated cost: ${:.4}. ZNE: {}.",
            primary.name,
            cost_estimate.total_cost,
            if zne { "enabled" } else { "disabled" }
        );

        Ok(OptimizationRecommendation {
            recommendation_id: Uuid::new_v4().to_string(),
            workload_id: workload.workload_id.clone(),
            provider: CloudProvider::Azure,
            recommended_config,
            optimization_strategies: self.get_optimization_strategies(),
            expected_performance: perf_prediction,
            cost_estimate,
            confidence_score: 0.85,
            rationale,
            alternative_recommendations: alternatives,
        })
    }

    fn get_provider(&self) -> CloudProvider {
        CloudProvider::Azure
    }

    fn get_optimization_strategies(&self) -> Vec<OptimizationStrategy> {
        vec![
            OptimizationStrategy::SchedulingOptimization,
            OptimizationStrategy::HardwareSelection,
            OptimizationStrategy::CacheOptimization,
        ]
    }

    fn predict_performance(
        &self,
        workload: &WorkloadSpec,
        config: &ExecutionConfig,
    ) -> DeviceResult<PerformancePrediction> {
        let backend = AZURE_BACKENDS
            .iter()
            .find(|b| b.name == config.backend)
            .unwrap_or(&AZURE_BACKENDS[0]);
        Ok(compute_azure_performance(workload, backend))
    }

    fn estimate_cost(
        &self,
        workload: &WorkloadSpec,
        config: &ExecutionConfig,
    ) -> DeviceResult<CostEstimate> {
        let backend = AZURE_BACKENDS
            .iter()
            .find(|b| b.name == config.backend)
            .unwrap_or(&AZURE_BACKENDS[0]);
        Ok(compute_azure_cost(
            workload.execution_requirements.shots,
            workload.circuit_characteristics.circuit_depth,
            workload.circuit_characteristics.qubit_count,
            backend,
        ))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn compute_azure_cost(
    shots: usize,
    circuit_depth: usize,
    qubit_count: usize,
    backend: &AzureBackendSpec,
) -> CostEstimate {
    let total = estimate_azure_cost_for_backend(shots, circuit_depth, qubit_count, backend);
    let uncertainty = total * 0.18;

    CostEstimate {
        total_cost: total,
        cost_breakdown: CostBreakdown {
            execution_cost: total,
            queue_cost: 0.0,
            storage_cost: 0.0,
            network_cost: 0.0,
            overhead_cost: 0.0,
            discount_applied: 0.0,
        },
        cost_model: CostModel::PayPerUse,
        uncertainty_range: (total - uncertainty, total + uncertainty),
        cost_optimization_opportunities: vec![CostOptimizationOpportunity {
            opportunity_type: CostOptimizationType::ResourceRightSizing,
            potential_savings: total * 0.12,
            implementation_effort: 0.25,
            description: "Reduce circuit depth via aggressive TKET optimisation to lower \
                           HQC credit consumption on Quantinuum targets."
                .to_string(),
        }],
    }
}

fn compute_azure_performance(
    workload: &WorkloadSpec,
    backend: &AzureBackendSpec,
) -> PerformancePrediction {
    let cc = &workload.circuit_characteristics;
    let shots = workload.execution_requirements.shots;

    // Fidelity model: fidelity^(depth/4) approximates accumulated two-qubit errors.
    let two_qubit_layers = (cc.circuit_depth / 4).max(1) as f64;
    let expected_fidelity = backend
        .two_qubit_fidelity
        .powf(two_qubit_layers)
        .clamp(0.01, 1.0);
    let success_probability = (expected_fidelity * 0.99).clamp(0.0, 1.0);

    // Execution time: trapped-ion gates are slow (~250 µs per CX); assume 2 ms/shot.
    let exec_s = shots as f64 * 0.002;

    PerformancePrediction {
        execution_time: Duration::from_secs_f64(exec_s),
        queue_time: Duration::from_secs_f64(backend.queue_latency_s),
        total_time: Duration::from_secs_f64(exec_s + backend.queue_latency_s),
        success_probability,
        expected_fidelity,
        resource_utilization: ResourceUtilizationPrediction {
            cpu_utilization: 0.03,
            memory_utilization: 0.06,
            quantum_resource_utilization: cc.qubit_count as f64 / backend.max_qubits as f64,
            network_utilization: 0.01,
            storage_utilization: 0.005,
        },
        bottlenecks: Vec::new(),
        confidence_interval: (success_probability * 0.94, success_probability * 1.03),
    }
}
