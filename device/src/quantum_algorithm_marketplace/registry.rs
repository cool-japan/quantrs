//! Algorithm Registry Configuration Types

use serde::{Deserialize, Serialize};

/// Algorithm registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmRegistryConfig {
    /// Enable algorithm registry
    pub enable_registry: bool,
    /// Algorithm storage configuration
    pub storage_config: AlgorithmStorageConfig,
    /// Algorithm categorization settings
    pub categorization_config: AlgorithmCategorizationConfig,
    /// Version control settings
    pub version_control: AlgorithmVersionControlConfig,
    /// Metadata management
    pub metadata_config: AlgorithmMetadataConfig,
    /// Algorithm validation settings
    pub validation_config: AlgorithmValidationConfig,
    /// Search and indexing configuration
    pub search_config: AlgorithmSearchConfig,
}

/// Algorithm storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmStorageConfig {
    pub storage_type: String,
    pub storage_path: String,
    pub compression_enabled: bool,
}

/// Algorithm categorization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmCategorizationConfig {
    pub enable_auto_categorization: bool,
    pub category_tags: Vec<String>,
}

/// Algorithm version control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmVersionControlConfig {
    pub enable_versioning: bool,
    pub max_versions: usize,
}

/// Algorithm metadata configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmMetadataConfig {
    pub required_fields: Vec<String>,
    pub optional_fields: Vec<String>,
}

/// Algorithm validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmValidationConfig {
    pub enable_validation: bool,
    pub validation_timeout: u64,
}

/// Algorithm search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmSearchConfig {
    pub enable_fuzzy_search: bool,
    pub max_results: usize,
}
