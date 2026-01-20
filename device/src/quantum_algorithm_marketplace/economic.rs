//! Economic Configuration Types

use serde::{Deserialize, Serialize};

/// Economic configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceEconomicConfig {
    /// Enable economic features
    pub enable_economics: bool,
    /// Incentive mechanisms
    pub incentive_mechanisms: Vec<IncentiveMechanism>,
    /// Reputation system configuration
    pub reputation_system: ReputationSystemConfig,
    /// Contribution rewards
    pub contribution_rewards: ContributionRewardConfig,
    /// Marketplace economics
    pub marketplace_economics: EconomicModelConfig,
    /// Resource allocation economics
    pub resource_economics: ResourceEconomicsConfig,
}

/// Incentive mechanisms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IncentiveMechanism {
    /// Contribution points
    ContributionPoints,
    /// Reputation tokens
    ReputationTokens,
    /// Access privileges
    AccessPrivileges,
    /// Resource credits
    ResourceCredits,
    /// Recognition badges
    RecognitionBadges,
    /// Collaborative rewards
    CollaborativeRewards,
    /// Custom incentive
    Custom(String),
}

/// Reputation system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationSystemConfig {
    pub enable_reputation: bool,
    pub reputation_metrics: Vec<String>,
}

/// Contribution reward configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionRewardConfig {
    pub reward_system: String,
    pub reward_thresholds: Vec<f64>,
}

/// Economic model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomicModelConfig {
    pub model_type: String,
    pub pricing_strategy: String,
}

/// Resource economics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEconomicsConfig {
    pub allocation_method: String,
    pub cost_model: String,
}
