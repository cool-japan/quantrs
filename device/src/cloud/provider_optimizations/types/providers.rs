//! Auto-generated module - providers
//!
//! 🤖 Generated with split_types_final.py

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock as TokioRwLock;
use uuid::Uuid;

use super::super::super::super::{DeviceError, DeviceResult, QuantumDevice};
use super::super::super::{CloudProvider, QuantumCloudConfig};
use crate::algorithm_marketplace::{ScalingBehavior, ValidationResult};
use crate::prelude::DeploymentStatus;

// Import ProviderOptimizer trait from parent module
use super::super::traits::ProviderOptimizer;

// Cross-module imports from sibling modules
use super::{cost::*, execution::*, optimization::*, profiling::*, tracking::*, workload::*};

pub struct AWSOptimizer {
    pub config: ProviderOptimizationConfig,
}
impl AWSOptimizer {
    pub fn new(config: &ProviderOptimizationConfig) -> DeviceResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

pub struct AzureOptimizer {
    pub config: ProviderOptimizationConfig,
}
impl AzureOptimizer {
    pub fn new(config: &ProviderOptimizationConfig) -> DeviceResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

pub struct GoogleOptimizer {
    pub config: ProviderOptimizationConfig,
}
impl GoogleOptimizer {
    pub fn new(config: &ProviderOptimizationConfig) -> DeviceResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

pub struct IBMOptimizer {
    pub config: ProviderOptimizationConfig,
}
impl IBMOptimizer {
    pub fn new(config: &ProviderOptimizationConfig) -> DeviceResult<Self> {
        Ok(Self {
            config: config.clone(),
        })
    }
}

#[derive(Clone)]
pub struct ProviderOptimizationConfig {
    pub enabled: bool,
    pub optimization_level: OptimizationLevel,
    pub target_metrics: Vec<OptimizationMetric>,
    pub cost_constraints: CostConstraints,
    pub performance_targets: PerformanceTargets,
    pub caching_enabled: bool,
    pub adaptive_optimization: bool,
    pub real_time_optimization: bool,
}

pub struct ProviderOptimizationEngine {
    config: ProviderOptimizationConfig,
    optimizers: HashMap<CloudProvider, Box<dyn ProviderOptimizer + Send + Sync>>,
    performance_tracker: Arc<TokioRwLock<PerformanceTracker>>,
    cost_analyzer: Arc<TokioRwLock<CostAnalyzer>>,
    workload_profiler: Arc<TokioRwLock<WorkloadProfiler>>,
    optimization_cache: Arc<TokioRwLock<OptimizationCache>>,
}
impl ProviderOptimizationEngine {
    /// Create a new provider optimization engine
    pub async fn new(config: ProviderOptimizationConfig) -> DeviceResult<Self> {
        let performance_tracker = Arc::new(TokioRwLock::new(PerformanceTracker::new()?));
        let cost_analyzer = Arc::new(TokioRwLock::new(CostAnalyzer::new()?));
        let workload_profiler = Arc::new(TokioRwLock::new(WorkloadProfiler::new()?));
        let optimization_cache = Arc::new(TokioRwLock::new(OptimizationCache::new()?));
        let mut optimizers: HashMap<CloudProvider, Box<dyn ProviderOptimizer + Send + Sync>> =
            HashMap::new();
        optimizers.insert(CloudProvider::IBM, Box::new(IBMOptimizer::new(&config)?));
        optimizers.insert(CloudProvider::AWS, Box::new(AWSOptimizer::new(&config)?));
        optimizers.insert(
            CloudProvider::Azure,
            Box::new(AzureOptimizer::new(&config)?),
        );
        optimizers.insert(
            CloudProvider::Google,
            Box::new(GoogleOptimizer::new(&config)?),
        );
        Ok(Self {
            config,
            optimizers,
            performance_tracker,
            cost_analyzer,
            workload_profiler,
            optimization_cache,
        })
    }
    /// Initialize the optimization engine
    pub async fn initialize(&mut self) -> DeviceResult<()> {
        for optimizer in self.optimizers.values_mut() {}
        self.load_historical_performance_data().await?;
        self.load_cost_models().await?;
        self.load_workload_profiles().await?;
        Ok(())
    }
    /// Optimize workload execution
    pub async fn optimize_workload(
        &self,
        workload: &WorkloadSpec,
    ) -> DeviceResult<OptimizationRecommendation> {
        if let Some(cached_result) = self.check_optimization_cache(workload).await? {
            return Ok(cached_result);
        }
        let workload_profile = self.profile_workload(workload).await?;
        let mut recommendations = Vec::new();
        for (provider, optimizer) in &self.optimizers {
            if self.is_provider_applicable(&workload.resource_constraints, provider) {
                match optimizer.optimize_workload(workload) {
                    Ok(recommendation) => recommendations.push(recommendation),
                    Err(e) => {
                        eprintln!("Error optimizing for provider {provider:?}: {e}");
                    }
                }
            }
        }
        let best_recommendation = self
            .select_best_recommendation(recommendations, &workload.resource_constraints)
            .await?;
        self.cache_optimization_result(workload, &best_recommendation)
            .await?;
        Ok(best_recommendation)
    }
    /// Update performance data
    pub async fn update_performance_data(
        &self,
        performance_record: PerformanceRecord,
    ) -> DeviceResult<()> {
        let mut tracker = self.performance_tracker.write().await;
        tracker.add_performance_record(performance_record).await?;
        if self.config.real_time_optimization {
            self.update_performance_models().await?;
        }
        Ok(())
    }
    /// Get provider comparison
    pub async fn get_provider_comparison(
        &self,
        workload: &WorkloadSpec,
    ) -> DeviceResult<ProviderComparison> {
        let mut comparison_results = HashMap::new();
        for (provider, optimizer) in &self.optimizers {
            if self.is_provider_applicable(&workload.resource_constraints, provider) {
                let prediction =
                    optimizer.predict_performance(workload, &ExecutionConfig::default())?;
                let cost_estimate =
                    optimizer.estimate_cost(workload, &ExecutionConfig::default())?;
                comparison_results.insert(provider.clone(), (prediction, cost_estimate));
            }
        }
        self.generate_provider_comparison(comparison_results).await
    }
    /// Shutdown optimization engine
    pub async fn shutdown(&self) -> DeviceResult<()> {
        self.save_optimization_cache().await?;
        self.save_performance_data().await?;
        Ok(())
    }
    async fn check_optimization_cache(
        &self,
        workload: &WorkloadSpec,
    ) -> DeviceResult<Option<OptimizationRecommendation>> {
        if !self.config.caching_enabled {
            return Ok(None);
        }
        let cache = self.optimization_cache.read().await;
        let workload_signature = self.generate_workload_signature(workload);
        if let Some(entry) = cache.get_entry(&workload_signature) {
            if entry.is_valid() {
                return Ok(Some(entry.optimization_result.clone()));
            }
        }
        Ok(None)
    }
    async fn profile_workload(&self, workload: &WorkloadSpec) -> DeviceResult<WorkloadProfile> {
        let profiler = self.workload_profiler.read().await;
        profiler.profile_workload(workload).await
    }
    fn is_provider_applicable(
        &self,
        constraints: &ResourceConstraints,
        provider: &CloudProvider,
    ) -> bool {
        !constraints.excluded_providers.contains(provider)
            && (constraints.preferred_providers.is_empty()
                || constraints.preferred_providers.contains(provider))
    }
    async fn select_best_recommendation(
        &self,
        recommendations: Vec<OptimizationRecommendation>,
        constraints: &ResourceConstraints,
    ) -> DeviceResult<OptimizationRecommendation> {
        if recommendations.is_empty() {
            return Err(DeviceError::InvalidInput(
                "No valid recommendations found".to_string(),
            ));
        }
        let scored_recommendations = self
            .score_recommendations(&recommendations, constraints)
            .await?;
        scored_recommendations
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(rec, _)| rec)
            .ok_or_else(|| {
                DeviceError::OptimizationError("Failed to select best recommendation".to_string())
            })
    }
    async fn score_recommendations(
        &self,
        recommendations: &[OptimizationRecommendation],
        constraints: &ResourceConstraints,
    ) -> DeviceResult<Vec<(OptimizationRecommendation, f64)>> {
        let mut scored = Vec::new();
        for recommendation in recommendations {
            let mut score = 0.0;
            if recommendation.cost_estimate.total_cost <= constraints.max_cost {
                score +=
                    0.3 * (1.0 - recommendation.cost_estimate.total_cost / constraints.max_cost);
            }
            score += 0.4 * recommendation.expected_performance.success_probability;
            score += 0.2 * recommendation.confidence_score;
            if constraints
                .preferred_providers
                .contains(&recommendation.provider)
            {
                score += 0.1;
            }
            scored.push((recommendation.clone(), score));
        }
        Ok(scored)
    }
    async fn cache_optimization_result(
        &self,
        workload: &WorkloadSpec,
        recommendation: &OptimizationRecommendation,
    ) -> DeviceResult<()> {
        if !self.config.caching_enabled {
            return Ok(());
        }
        let mut cache = self.optimization_cache.write().await;
        let workload_signature = self.generate_workload_signature(workload);
        cache
            .insert_entry(workload_signature, recommendation.clone())
            .await
    }
    fn generate_workload_signature(&self, workload: &WorkloadSpec) -> String {
        format!(
            "{}_{}_{}_{}",
            workload.workload_type.as_u8(),
            workload.circuit_characteristics.qubit_count,
            workload.circuit_characteristics.gate_count,
            workload.execution_requirements.shots
        )
    }
    async fn load_historical_performance_data(&self) -> DeviceResult<()> {
        Ok(())
    }
    async fn load_cost_models(&self) -> DeviceResult<()> {
        Ok(())
    }
    async fn load_workload_profiles(&self) -> DeviceResult<()> {
        Ok(())
    }
    async fn update_performance_models(&self) -> DeviceResult<()> {
        Ok(())
    }
    /// Generate a `ProviderComparison` from per-provider prediction / cost
    /// estimates gathered by `get_provider_comparison`.
    ///
    /// The comparison:
    /// 1. Identifies the provider with the best overall score (weighted sum of
    ///    fidelity, cost-efficiency, and success probability).
    /// 2. Builds normalised performance and cost metric maps so callers can
    ///    rank providers on individual dimensions.
    /// 3. Scores each provider's suitability for each `WorkloadType` (0–1).
    async fn generate_provider_comparison(
        &self,
        comparison_results: HashMap<CloudProvider, (PerformancePrediction, CostEstimate)>,
    ) -> DeviceResult<ProviderComparison> {
        use std::collections::HashMap;

        if comparison_results.is_empty() {
            return Err(DeviceError::InvalidInput(
                "No provider results available for comparison".to_string(),
            ));
        }

        // ----------------------------------------------------------------
        // Score each provider
        // ----------------------------------------------------------------
        // Weights: fidelity (40%), cost-efficiency (35%), success prob (25%).
        const W_FIDELITY: f64 = 0.40;
        const W_COST: f64 = 0.35;
        const W_SUCCESS: f64 = 0.25;

        // Gather raw metrics.
        let fidelities: HashMap<CloudProvider, f64> = comparison_results
            .iter()
            .map(|(p, (perf, _))| (p.clone(), perf.expected_fidelity))
            .collect();

        let total_costs: HashMap<CloudProvider, f64> = comparison_results
            .iter()
            .map(|(p, (_, cost))| (p.clone(), cost.total_cost.max(1e-9)))
            .collect();

        let success_probs: HashMap<CloudProvider, f64> = comparison_results
            .iter()
            .map(|(p, (perf, _))| (p.clone(), perf.success_probability))
            .collect();

        // Normalise costs so cheaper is a higher score (invert and scale).
        let max_cost = total_costs
            .values()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_cost = total_costs.values().cloned().fold(f64::INFINITY, f64::min);
        let cost_range = (max_cost - min_cost).max(1e-9);

        let cost_scores: HashMap<CloudProvider, f64> = total_costs
            .iter()
            .map(|(p, &c)| (p.clone(), 1.0 - (c - min_cost) / cost_range))
            .collect();

        // Composite scores.
        let mut provider_scores: Vec<(CloudProvider, f64)> = comparison_results
            .keys()
            .map(|p| {
                let f = fidelities.get(p).copied().unwrap_or(0.0);
                let c = cost_scores.get(p).copied().unwrap_or(0.0);
                let s = success_probs.get(p).copied().unwrap_or(0.0);
                let score = W_FIDELITY * f + W_COST * c + W_SUCCESS * s;
                (p.clone(), score)
            })
            .collect();

        provider_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Best provider = highest composite score.
        let (provider_a, _best_score) = provider_scores
            .first()
            .cloned()
            .ok_or_else(|| DeviceError::InvalidInput("Empty scoring result".to_string()))?;

        // Second provider for pair comparison (or same if only one).
        let provider_b = provider_scores
            .get(1)
            .map(|(p, _)| p.clone())
            .unwrap_or_else(|| provider_a.clone());

        // ----------------------------------------------------------------
        // Build performance comparison metrics
        // ----------------------------------------------------------------
        let mut performance_comparison: HashMap<String, f64> = HashMap::new();

        // Absolute metrics for each provider, keyed as "metric_PROVIDER_NAME".
        for (provider, (perf, _)) in &comparison_results {
            let label = format!("{provider:?}");
            performance_comparison
                .insert(format!("expected_fidelity_{label}"), perf.expected_fidelity);
            performance_comparison.insert(
                format!("success_probability_{label}"),
                perf.success_probability,
            );
            performance_comparison.insert(
                format!("execution_time_s_{label}"),
                perf.execution_time.as_secs_f64(),
            );
            performance_comparison.insert(
                format!("queue_time_s_{label}"),
                perf.queue_time.as_secs_f64(),
            );
        }

        // ----------------------------------------------------------------
        // Build cost comparison metrics
        // ----------------------------------------------------------------
        let mut cost_comparison: HashMap<String, f64> = HashMap::new();
        for (provider, (_, cost)) in &comparison_results {
            let label = format!("{provider:?}");
            cost_comparison.insert(format!("total_cost_{label}"), cost.total_cost);
            cost_comparison.insert(
                format!("execution_cost_{label}"),
                cost.cost_breakdown.execution_cost,
            );
            cost_comparison.insert(
                format!("overhead_cost_{label}"),
                cost.cost_breakdown.overhead_cost,
            );
        }

        // ----------------------------------------------------------------
        // Feature comparison: unique strengths of provider_a vs provider_b
        // ----------------------------------------------------------------
        let a_perf = comparison_results.get(&provider_a).map(|(p, _)| p).cloned();
        let b_perf = comparison_results.get(&provider_b).map(|(p, _)| p).cloned();

        let mut feature_scores: HashMap<String, (f64, f64)> = HashMap::new();
        if let (Some(ap), Some(bp)) = (a_perf, b_perf) {
            feature_scores.insert(
                "expected_fidelity".to_string(),
                (ap.expected_fidelity, bp.expected_fidelity),
            );
            feature_scores.insert(
                "success_probability".to_string(),
                (ap.success_probability, bp.success_probability),
            );
            feature_scores.insert(
                "total_execution_time_s".to_string(),
                (
                    ap.execution_time.as_secs_f64(),
                    bp.execution_time.as_secs_f64(),
                ),
            );
        }

        let a_cost_score = cost_scores.get(&provider_a).copied().unwrap_or(0.0);
        let b_cost_score = cost_scores.get(&provider_b).copied().unwrap_or(0.0);
        feature_scores.insert("cost_efficiency".to_string(), (a_cost_score, b_cost_score));

        let unique_a: Vec<String> = if a_cost_score > b_cost_score {
            vec!["Lower cost".to_string()]
        } else {
            vec![]
        };
        let unique_b: Vec<String> = if b_cost_score > a_cost_score {
            vec!["Lower cost".to_string()]
        } else {
            vec![]
        };

        let compatibility_scores: HashMap<String, f64> = comparison_results
            .keys()
            .map(|p| {
                (
                    format!("{p:?}"),
                    provider_scores
                        .iter()
                        .find(|(pp, _)| pp == p)
                        .map(|(_, s)| *s)
                        .unwrap_or(0.0),
                )
            })
            .collect();

        let feature_comparison = FeatureComparison {
            feature_scores,
            unique_features: (unique_a, unique_b),
            compatibility_scores,
        };

        // ----------------------------------------------------------------
        // Workload-type suitability (heuristic based on score)
        // ----------------------------------------------------------------
        let best_score = provider_scores.first().map(|(_, s)| *s).unwrap_or(0.0);

        let use_case_suitability: HashMap<WorkloadType, f64> = vec![
            (WorkloadType::Simulation, best_score),
            (WorkloadType::Optimization, best_score * 0.95),
            (WorkloadType::MachineLearning, best_score * 0.90),
            (WorkloadType::Cryptography, best_score * 0.85),
            (WorkloadType::Chemistry, best_score * 0.92),
            (WorkloadType::Research, best_score),
            (WorkloadType::Production, best_score * 0.88),
        ]
        .into_iter()
        .collect();

        Ok(ProviderComparison {
            provider_a,
            provider_b,
            performance_comparison,
            cost_comparison,
            feature_comparison,
            use_case_suitability,
        })
    }
    async fn save_optimization_cache(&self) -> DeviceResult<()> {
        Ok(())
    }
    async fn save_performance_data(&self) -> DeviceResult<()> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ProviderComparison {
    pub provider_a: CloudProvider,
    pub provider_b: CloudProvider,
    pub performance_comparison: HashMap<String, f64>,
    pub cost_comparison: HashMap<String, f64>,
    pub feature_comparison: FeatureComparison,
    pub use_case_suitability: HashMap<WorkloadType, f64>,
}
