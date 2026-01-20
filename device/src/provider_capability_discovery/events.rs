//! Events, commands, and results for provider capability discovery.
//!
//! This module contains discovery events, commands, comparison results,
//! and recommendations.

use std::collections::HashMap;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use super::capabilities::ProviderCapabilities;
use super::config::{ComparisonCriterion, FilteringConfig};
use super::types::VerificationStatus;

/// Discovery events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiscoveryEvent {
    ProviderDiscovered {
        provider_id: String,
        capabilities: ProviderCapabilities,
        timestamp: SystemTime,
    },
    CapabilityUpdated {
        provider_id: String,
        updated_capabilities: ProviderCapabilities,
        timestamp: SystemTime,
    },
    ProviderUnavailable {
        provider_id: String,
        reason: String,
        timestamp: SystemTime,
    },
    VerificationCompleted {
        provider_id: String,
        status: VerificationStatus,
        timestamp: SystemTime,
    },
    ComparisonCompleted {
        providers: Vec<String>,
        results: ComparisonResults,
        timestamp: SystemTime,
    },
}

/// Discovery commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiscoveryCommand {
    DiscoverProviders,
    VerifyProvider(String),
    UpdateCapabilities(String),
    CompareProviders(Vec<String>),
    FilterProviders(FilteringConfig),
    GetProviderInfo(String),
    GetProviderRanking(Vec<ComparisonCriterion>),
    GenerateReport(ReportType),
}

/// Report types for discovery
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportType {
    ProviderSummary,
    CapabilityMatrix,
    PerformanceComparison,
    CostAnalysis,
    SecurityAssessment,
    ComprehensiveReport,
}

/// Comparison results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResults {
    /// Provider rankings
    pub rankings: Vec<ProviderRanking>,
    /// Comparison matrix
    pub comparison_matrix: HashMap<String, HashMap<String, f64>>,
    /// Analysis summary
    pub analysis_summary: AnalysisSummary,
    /// Recommendations
    pub recommendations: Vec<ProviderRecommendation>,
}

/// Provider ranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderRanking {
    /// Provider ID
    pub provider_id: String,
    /// Overall score
    pub overall_score: f64,
    /// Category scores
    pub category_scores: HashMap<String, f64>,
    /// Rank position
    pub rank: usize,
    /// Strengths
    pub strengths: Vec<String>,
    /// Weaknesses
    pub weaknesses: Vec<String>,
}

/// Analysis summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisSummary {
    /// Key findings
    pub key_findings: Vec<String>,
    /// Market insights
    pub market_insights: Vec<String>,
    /// Trends identified
    pub trends: Vec<String>,
    /// Risk factors
    pub risk_factors: Vec<String>,
}

/// Provider recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderRecommendation {
    /// Provider ID
    pub provider_id: String,
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Use case
    pub use_case: String,
    /// Confidence score
    pub confidence: f64,
    /// Reasoning
    pub reasoning: String,
    /// Cost estimate
    pub cost_estimate: Option<CostEstimate>,
}

/// Recommendation types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendationType {
    BestOverall,
    BestValue,
    BestPerformance,
    BestSecurity,
    BestSupport,
    BestForBeginners,
    BestForResearch,
    BestForProduction,
    Custom(String),
}

/// Cost estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Estimated monthly cost
    pub monthly_cost: f64,
    /// Cost breakdown
    pub cost_breakdown: HashMap<String, f64>,
    /// Currency
    pub currency: String,
    /// Confidence level
    pub confidence: f64,
}

/// Verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Verification status
    pub status: VerificationStatus,
    /// Confidence score
    pub confidence: f64,
    /// Verification details
    pub details: HashMap<String, String>,
    /// Verified at
    pub verified_at: SystemTime,
    /// Verification method
    pub verification_method: String,
}
