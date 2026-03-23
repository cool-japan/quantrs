//! # PerformanceProfiler - generate_recommendations_group Methods
//!
//! This module contains method implementations for `PerformanceProfiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;

use super::types::{
    AnalysisReport, BottleneckType, ImplementationEffort, OptimizationRecommendation, Profile,
    RecommendationCategory, RecommendationImpact,
};

use super::performanceprofiler_type::PerformanceProfiler;

impl PerformanceProfiler {
    /// Generate optimization recommendations
    pub fn generate_recommendations(&self, profile: &Profile) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        let analysis = self.analyze_profile(profile);
        for bottleneck in &analysis.bottlenecks {
            match bottleneck.bottleneck_type {
                BottleneckType::CPU => {
                    if bottleneck.impact > 0.3 {
                        recommendations.push(OptimizationRecommendation {
                            title: format!("Optimize hot function: {}", bottleneck.location),
                            description:
                                "Consider algorithmic improvements, caching, or parallelization"
                                    .to_string(),
                            category: RecommendationCategory::Algorithm,
                            impact: RecommendationImpact::High,
                            effort: ImplementationEffort::Medium,
                            estimated_improvement: bottleneck.impact * 0.5,
                            code_suggestions: vec![
                                "Add memoization for expensive calculations".to_string(),
                                "Consider parallel processing".to_string(),
                                "Profile inner loops for micro-optimizations".to_string(),
                            ],
                        });
                    }
                }
                BottleneckType::Memory => {
                    recommendations.push(OptimizationRecommendation {
                        title: "Memory usage optimization".to_string(),
                        description: "Reduce memory allocations and improve data locality"
                            .to_string(),
                        category: RecommendationCategory::Memory,
                        impact: RecommendationImpact::Medium,
                        effort: ImplementationEffort::Low,
                        estimated_improvement: 0.2,
                        code_suggestions: vec![
                            "Use object pooling for frequently allocated objects".to_string(),
                            "Consider more compact data structures".to_string(),
                            "Implement streaming for large datasets".to_string(),
                        ],
                    });
                }
                _ => {}
            }
        }
        if profile.metrics.computation_metrics.cache_hit_rate < 0.8 {
            recommendations.push(OptimizationRecommendation {
                title: "Improve cache locality".to_string(),
                description: "Restructure data access patterns for better cache performance"
                    .to_string(),
                category: RecommendationCategory::Memory,
                impact: RecommendationImpact::Medium,
                effort: ImplementationEffort::High,
                estimated_improvement: 0.15,
                code_suggestions: vec![
                    "Use structure-of-arrays instead of array-of-structures".to_string(),
                    "Implement cache-oblivious algorithms".to_string(),
                    "Add data prefetching hints".to_string(),
                ],
            });
        }
        recommendations.sort_by(|a, b| {
            b.estimated_improvement
                .partial_cmp(&a.estimated_improvement)
                .expect("Failed to compare estimated improvements in recommendation sorting")
        });
        recommendations
    }
    /// Analyze profile
    pub fn analyze_profile(&self, profile: &Profile) -> AnalysisReport {
        self.analyzer.analyze(profile)
    }
}
