//! Collaboration Configuration Types

use serde::{Deserialize, Serialize};

/// Collaboration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationConfig {
    /// Team management configuration
    pub team_management: TeamManagementConfig,
    /// Contribution tracking configuration
    pub contribution_tracking: ContributionTrackingConfig,
    /// Code review configuration
    pub code_review: CodeReviewConfig,
    /// Community configuration
    pub community_config: CommunityConfig,
    /// Knowledge sharing configuration
    pub knowledge_sharing: KnowledgeSharingConfig,
}

/// Team management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamManagementConfig {
    pub max_team_size: usize,
    pub roles: Vec<String>,
}

/// Contribution tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContributionTrackingConfig {
    pub track_contributions: bool,
    pub metrics: Vec<String>,
}

/// Code review configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeReviewConfig {
    pub require_review: bool,
    pub min_reviewers: usize,
}

/// Community configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityConfig {
    pub enable_forums: bool,
    pub moderation_enabled: bool,
}

/// Knowledge sharing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeSharingConfig {
    pub enable_sharing: bool,
    pub sharing_formats: Vec<String>,
}
