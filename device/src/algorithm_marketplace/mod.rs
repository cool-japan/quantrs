//! Quantum Algorithm Marketplace Integration
//!
//! This module provides a comprehensive marketplace for quantum algorithms,
//! including algorithm discovery, deployment, monetization, and optimization
//! across multiple quantum computing platforms.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock as TokioRwLock;
use uuid::Uuid;

use crate::{DeviceError, DeviceResult, QuantumDevice};

pub mod algorithm_registry;
pub mod deployment;
pub mod discovery;
pub mod marketplace_api;
pub mod monetization;
pub mod optimization;
pub mod validation;
pub mod versioning;

pub use algorithm_registry::*;
pub use deployment::*;
pub use discovery::*;
pub use marketplace_api::*;
pub use monetization::*;
pub use optimization::*;
pub use validation::*;
pub use versioning::*;

/// Quantum Algorithm Marketplace main manager
pub struct QuantumAlgorithmMarketplace {
    config: MarketplaceConfig,
    registry: Arc<TokioRwLock<AlgorithmRegistry>>,
    discovery_engine: Arc<TokioRwLock<AlgorithmDiscoveryEngine>>,
    deployment_manager: Arc<TokioRwLock<AlgorithmDeploymentManager>>,
    monetization_system: Arc<TokioRwLock<MonetizationSystem>>,
    optimization_engine: Arc<TokioRwLock<AlgorithmOptimizationEngine>>,
    validation_service: Arc<TokioRwLock<AlgorithmValidationService>>,
    versioning_system: Arc<TokioRwLock<AlgorithmVersioningSystem>>,
    marketplace_api: Arc<TokioRwLock<MarketplaceAPI>>,
    active_deployments: Arc<TokioRwLock<HashMap<String, ActiveDeployment>>>,
    user_sessions: Arc<TokioRwLock<HashMap<String, UserSession>>>,
}

/// Marketplace configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceConfig {
    pub enabled: bool,
    pub registry_config: RegistryConfig,
    pub discovery_config: DiscoveryConfig,
    pub deployment_config: DeploymentConfig,
    pub monetization_config: MonetizationConfig,
    pub optimization_config: OptimizationConfig,
    pub validation_config: ValidationConfig,
    pub versioning_config: VersioningConfig,
    pub api_config: APIConfig,
    pub security_config: MarketplaceSecurityConfig,
}

/// Registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    pub max_algorithms: usize,
    pub max_algorithm_size: usize,
    pub supported_languages: Vec<String>,
    pub supported_frameworks: Vec<String>,
    pub metadata_validation: bool,
    pub content_filtering: bool,
}

/// Discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    pub enable_semantic_search: bool,
    pub enable_recommendation_engine: bool,
    pub enable_popularity_ranking: bool,
    pub enable_performance_ranking: bool,
    pub caching_enabled: bool,
    pub cache_ttl: Duration,
}

/// Deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    pub max_concurrent_deployments: usize,
    pub deployment_timeout: Duration,
    pub auto_scaling_enabled: bool,
    pub resource_limits: ResourceLimits,
    pub monitoring_enabled: bool,
    pub rollback_enabled: bool,
}

/// Resource limits for deployments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_qubits: usize,
    pub max_circuit_depth: usize,
    pub max_execution_time: Duration,
    pub max_memory_usage: usize,
    pub max_classical_processing: f64,
}

/// Monetization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonetizationConfig {
    pub enabled: bool,
    pub supported_payment_methods: Vec<PaymentMethod>,
    pub commission_rate: f64,
    pub revenue_sharing_enabled: bool,
    pub subscription_models: Vec<SubscriptionModel>,
    pub pricing_strategies: Vec<PricingStrategy>,
}

/// Payment methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaymentMethod {
    CreditCard,
    DigitalWallet,
    Cryptocurrency,
    QuantumCredits,
    InstitutionalBilling,
    Custom(String),
}

/// Subscription models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubscriptionModel {
    PayPerUse,
    Monthly,
    Annual,
    Enterprise,
    Academic,
    FreeTier,
    Custom(String),
}

/// Pricing strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PricingStrategy {
    Fixed,
    Dynamic,
    AuctionBased,
    PerformanceBased,
    ResourceBased,
    TieredPricing,
    Custom(String),
}

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIConfig {
    pub rest_api_enabled: bool,
    pub graphql_api_enabled: bool,
    pub websocket_api_enabled: bool,
    pub rate_limiting: RateLimitingConfig,
    pub authentication_required: bool,
    pub api_versioning: bool,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitingConfig {
    pub enabled: bool,
    pub requests_per_minute: usize,
    pub burst_limit: usize,
    pub per_user_limits: HashMap<String, usize>,
}

/// Marketplace security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceSecurityConfig {
    pub code_scanning_enabled: bool,
    pub vulnerability_checking: bool,
    pub access_control_enabled: bool,
    pub audit_logging: bool,
    pub encryption_at_rest: bool,
    pub encryption_in_transit: bool,
}

/// Active deployment information
#[derive(Debug, Clone)]
pub struct ActiveDeployment {
    pub deployment_id: String,
    pub algorithm_id: String,
    pub user_id: String,
    pub deployment_config: DeploymentConfig,
    pub status: DeploymentStatus,
    pub started_at: SystemTime,
    pub resource_usage: ResourceUsage,
    pub performance_metrics: DeploymentMetrics,
}

/// Deployment status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeploymentStatus {
    Pending,
    Deploying,
    Running,
    Paused,
    Stopping,
    Stopped,
    Failed,
    Scaling,
}

/// Deployment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfiguration {
    pub algorithm_version: String,
    pub target_platforms: Vec<String>,
    pub resource_requirements: ResourceRequirements,
    pub scaling_policy: ScalingPolicy,
    pub monitoring_config: MonitoringConfig,
    pub environment_variables: HashMap<String, String>,
}

/// Resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub min_qubits: usize,
    pub preferred_qubits: usize,
    pub min_fidelity: f64,
    pub min_coherence_time: Duration,
    pub classical_cpu_cores: usize,
    pub memory_gb: f64,
    pub storage_gb: f64,
}

/// Scaling policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPolicy {
    pub auto_scaling: bool,
    pub min_instances: usize,
    pub max_instances: usize,
    pub scale_up_threshold: f64,
    pub scale_down_threshold: f64,
    pub scale_up_cooldown: Duration,
    pub scale_down_cooldown: Duration,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub metrics_collection: bool,
    pub real_time_monitoring: bool,
    pub alerting_enabled: bool,
    pub log_retention_days: u32,
    pub performance_tracking: bool,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub qubits_used: usize,
    pub circuit_executions: u64,
    pub classical_compute_hours: f64,
    pub memory_peak_usage_gb: f64,
    pub storage_used_gb: f64,
    pub network_bandwidth_gb: f64,
    pub quantum_volume_consumed: f64,
}

/// Deployment metrics
#[derive(Debug, Clone)]
pub struct DeploymentMetrics {
    pub uptime_percentage: f64,
    pub average_response_time: Duration,
    pub request_count: u64,
    pub error_count: u64,
    pub throughput_requests_per_second: f64,
    pub fidelity_achieved: f64,
    pub cost_per_execution: f64,
}

/// User session for marketplace
#[derive(Debug, Clone)]
pub struct UserSession {
    pub session_id: String,
    pub user_id: String,
    pub user_type: UserType,
    pub permissions: Vec<Permission>,
    pub created_at: SystemTime,
    pub last_activity: SystemTime,
    pub session_data: HashMap<String, String>,
}

/// User types in the marketplace
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UserType {
    Individual,
    Academic,
    Enterprise,
    Developer,
    Researcher,
    Student,
    Administrator,
}

/// Permissions for marketplace actions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Permission {
    ViewAlgorithms,
    DownloadAlgorithms,
    UploadAlgorithms,
    ModifyAlgorithms,
    DeleteAlgorithms,
    DeployAlgorithms,
    ManageDeployments,
    ViewAnalytics,
    ManageUsers,
    ManagePayments,
    AdminAccess,
}

impl QuantumAlgorithmMarketplace {
    /// Create a new quantum algorithm marketplace
    pub async fn new(config: MarketplaceConfig) -> DeviceResult<Self> {
        let registry = Arc::new(TokioRwLock::new(AlgorithmRegistry::new(
            &config.registry_config,
        )?));
        let discovery_engine = Arc::new(TokioRwLock::new(AlgorithmDiscoveryEngine::new(
            &config.discovery_config,
        )?));
        let deployment_manager = Arc::new(TokioRwLock::new(AlgorithmDeploymentManager::new(
            &config.deployment_config,
        )?));
        let monetization_system = Arc::new(TokioRwLock::new(MonetizationSystem::new(
            &config.monetization_config,
        )?));
        let optimization_engine = Arc::new(TokioRwLock::new(AlgorithmOptimizationEngine::new(
            &config.optimization_config,
        )?));
        let validation_service = Arc::new(TokioRwLock::new(AlgorithmValidationService::new(
            &config.validation_config,
        )?));
        let versioning_system = Arc::new(TokioRwLock::new(AlgorithmVersioningSystem::new(
            &config.versioning_config,
        )?));
        let marketplace_api = Arc::new(TokioRwLock::new(MarketplaceAPI::new(&config.api_config)?));
        let active_deployments = Arc::new(TokioRwLock::new(HashMap::new()));
        let user_sessions = Arc::new(TokioRwLock::new(HashMap::new()));

        Ok(Self {
            config,
            registry,
            discovery_engine,
            deployment_manager,
            monetization_system,
            optimization_engine,
            validation_service,
            versioning_system,
            marketplace_api,
            active_deployments,
            user_sessions,
        })
    }

    /// Initialize the marketplace
    pub async fn initialize(&mut self) -> DeviceResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Initialize all subsystems
        self.registry.read().await.initialize().await?;
        self.discovery_engine.read().await.initialize().await?;
        self.deployment_manager.read().await.initialize().await?;
        self.monetization_system.read().await.initialize().await?;
        self.optimization_engine.read().await.initialize().await?;
        self.validation_service.read().await.initialize().await?;
        self.versioning_system.read().await.initialize().await?;
        self.marketplace_api.read().await.initialize().await?;

        Ok(())
    }

    /// Create user session
    pub async fn create_user_session(
        &self,
        user_id: &str,
        user_type: UserType,
    ) -> DeviceResult<String> {
        let session_id = Uuid::new_v4().to_string();

        let permissions = self.get_user_permissions(&user_type);

        let session = UserSession {
            session_id: session_id.clone(),
            user_id: user_id.to_string(),
            user_type,
            permissions,
            created_at: SystemTime::now(),
            last_activity: SystemTime::now(),
            session_data: HashMap::new(),
        };

        self.user_sessions
            .write()
            .await
            .insert(session_id.clone(), session);
        Ok(session_id)
    }

    /// Discover algorithms based on criteria
    pub async fn discover_algorithms(
        &self,
        criteria: DiscoveryCriteria,
    ) -> DeviceResult<Vec<AlgorithmInfo>> {
        let discovery_engine = self.discovery_engine.read().await;
        discovery_engine.search_algorithms(criteria).await
    }

    /// Register new algorithm
    pub async fn register_algorithm(
        &self,
        algorithm: AlgorithmRegistration,
    ) -> DeviceResult<String> {
        // Validate algorithm
        let validation_service = self.validation_service.read().await;
        validation_service.validate_algorithm(&algorithm).await?;

        // Register in registry
        let mut registry = self.registry.write().await;
        let algorithm_id = registry.register_algorithm(algorithm).await?;

        Ok(algorithm_id)
    }

    /// Deploy algorithm
    pub async fn deploy_algorithm(
        &self,
        deployment_request: DeploymentRequest,
    ) -> DeviceResult<String> {
        let deployment_id = Uuid::new_v4().to_string();

        // Create deployment
        let mut deployment_manager = self.deployment_manager.write().await;
        let deployment = deployment_manager
            .create_deployment(deployment_request)
            .await?;

        // Track active deployment
        let active_deployment = ActiveDeployment {
            deployment_id: deployment_id.clone(),
            algorithm_id: deployment.algorithm_id,
            user_id: deployment.user_id,
            deployment_config: deployment.configuration,
            status: DeploymentStatus::Pending,
            started_at: SystemTime::now(),
            resource_usage: ResourceUsage::default(),
            performance_metrics: DeploymentMetrics::default(),
        };

        self.active_deployments
            .write()
            .await
            .insert(deployment_id.clone(), active_deployment);

        Ok(deployment_id)
    }

    /// Get deployment status
    pub async fn get_deployment_status(
        &self,
        deployment_id: &str,
    ) -> DeviceResult<DeploymentStatus> {
        let deployments = self.active_deployments.read().await;
        let deployment = deployments.get(deployment_id).ok_or_else(|| {
            DeviceError::InvalidInput(format!("Deployment {deployment_id} not found"))
        })?;

        Ok(deployment.status.clone())
    }

    /// Update deployment metrics
    pub async fn update_deployment_metrics(
        &self,
        deployment_id: &str,
        metrics: DeploymentMetrics,
    ) -> DeviceResult<()> {
        let mut deployments = self.active_deployments.write().await;
        if let Some(deployment) = deployments.get_mut(deployment_id) {
            deployment.performance_metrics = metrics;
        }
        Ok(())
    }

    /// Stop deployment
    pub async fn stop_deployment(&self, deployment_id: &str) -> DeviceResult<()> {
        let mut deployments = self.active_deployments.write().await;
        if let Some(deployment) = deployments.get_mut(deployment_id) {
            deployment.status = DeploymentStatus::Stopping;
        }

        // Cleanup deployment resources
        let deployment_manager = self.deployment_manager.read().await;
        deployment_manager.stop_deployment(deployment_id).await?;

        // Remove from active deployments
        deployments.remove(deployment_id);
        Ok(())
    }

    /// Get marketplace analytics
    pub async fn get_marketplace_analytics(&self) -> DeviceResult<MarketplaceAnalytics> {
        let registry = self.registry.read().await;
        let deployments = self.active_deployments.read().await;

        let analytics = MarketplaceAnalytics {
            total_algorithms: registry.get_algorithm_count().await?,
            active_deployments: deployments.len(),
            total_users: self.user_sessions.read().await.len(),
            platform_usage: self.get_platform_usage_stats().await?,
            revenue_metrics: self.get_revenue_metrics().await?,
            performance_metrics: self.get_performance_metrics().await?,
        };

        Ok(analytics)
    }

    /// Shutdown marketplace
    pub async fn shutdown(&self) -> DeviceResult<()> {
        // Stop all active deployments
        let deployment_ids: Vec<String> = self
            .active_deployments
            .read()
            .await
            .keys()
            .cloned()
            .collect();
        for deployment_id in deployment_ids {
            self.stop_deployment(&deployment_id).await?;
        }

        // Clear user sessions
        self.user_sessions.write().await.clear();

        Ok(())
    }

    // Helper methods
    fn get_user_permissions(&self, user_type: &UserType) -> Vec<Permission> {
        match user_type {
            UserType::Individual => vec![
                Permission::ViewAlgorithms,
                Permission::DownloadAlgorithms,
                Permission::DeployAlgorithms,
            ],
            UserType::Academic => vec![
                Permission::ViewAlgorithms,
                Permission::DownloadAlgorithms,
                Permission::UploadAlgorithms,
                Permission::DeployAlgorithms,
                Permission::ViewAnalytics,
            ],
            UserType::Enterprise => vec![
                Permission::ViewAlgorithms,
                Permission::DownloadAlgorithms,
                Permission::UploadAlgorithms,
                Permission::DeployAlgorithms,
                Permission::ManageDeployments,
                Permission::ViewAnalytics,
                Permission::ManagePayments,
            ],
            UserType::Developer => vec![
                Permission::ViewAlgorithms,
                Permission::DownloadAlgorithms,
                Permission::UploadAlgorithms,
                Permission::ModifyAlgorithms,
                Permission::DeployAlgorithms,
                Permission::ManageDeployments,
                Permission::ViewAnalytics,
            ],
            UserType::Researcher => vec![
                Permission::ViewAlgorithms,
                Permission::DownloadAlgorithms,
                Permission::UploadAlgorithms,
                Permission::DeployAlgorithms,
                Permission::ViewAnalytics,
            ],
            UserType::Student => vec![
                Permission::ViewAlgorithms,
                Permission::DownloadAlgorithms,
                Permission::DeployAlgorithms,
            ],
            UserType::Administrator => vec![
                Permission::ViewAlgorithms,
                Permission::DownloadAlgorithms,
                Permission::UploadAlgorithms,
                Permission::ModifyAlgorithms,
                Permission::DeleteAlgorithms,
                Permission::DeployAlgorithms,
                Permission::ManageDeployments,
                Permission::ViewAnalytics,
                Permission::ManageUsers,
                Permission::ManagePayments,
                Permission::AdminAccess,
            ],
        }
    }

    async fn get_platform_usage_stats(&self) -> DeviceResult<PlatformUsageStats> {
        // Implement platform usage statistics collection
        Ok(PlatformUsageStats::default())
    }

    async fn get_revenue_metrics(&self) -> DeviceResult<RevenueMetrics> {
        // Implement revenue metrics collection
        Ok(RevenueMetrics::default())
    }

    async fn get_performance_metrics(&self) -> DeviceResult<PerformanceMetrics> {
        // Implement performance metrics collection
        Ok(PerformanceMetrics::default())
    }
}

/// Marketplace analytics
#[derive(Debug, Clone)]
pub struct MarketplaceAnalytics {
    pub total_algorithms: usize,
    pub active_deployments: usize,
    pub total_users: usize,
    pub platform_usage: PlatformUsageStats,
    pub revenue_metrics: RevenueMetrics,
    pub performance_metrics: PerformanceMetrics,
}

/// Platform usage statistics
#[derive(Debug, Clone, Default)]
pub struct PlatformUsageStats {
    pub algorithms_deployed_per_day: f64,
    pub average_deployment_duration: Duration,
    pub most_popular_algorithms: Vec<String>,
    pub platform_distribution: HashMap<String, usize>,
}

/// Revenue metrics
#[derive(Debug, Clone, Default)]
pub struct RevenueMetrics {
    pub total_revenue: f64,
    pub revenue_per_algorithm: HashMap<String, f64>,
    pub revenue_by_platform: HashMap<String, f64>,
    pub subscription_revenue: f64,
    pub pay_per_use_revenue: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub average_algorithm_performance: f64,
    pub deployment_success_rate: f64,
    pub average_resource_utilization: f64,
    pub customer_satisfaction_score: f64,
}

// Default implementations
impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            qubits_used: 0,
            circuit_executions: 0,
            classical_compute_hours: 0.0,
            memory_peak_usage_gb: 0.0,
            storage_used_gb: 0.0,
            network_bandwidth_gb: 0.0,
            quantum_volume_consumed: 0.0,
        }
    }
}

// Default implementation is already provided in deployment.rs

impl Default for MarketplaceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            registry_config: RegistryConfig {
                max_algorithms: 10000,
                max_algorithm_size: 100 * 1024 * 1024, // 100MB
                supported_languages: vec![
                    "Python".to_string(),
                    "Rust".to_string(),
                    "C++".to_string(),
                    "Julia".to_string(),
                ],
                supported_frameworks: vec![
                    "Qiskit".to_string(),
                    "Cirq".to_string(),
                    "PennyLane".to_string(),
                    "QuantRS2".to_string(),
                ],
                metadata_validation: true,
                content_filtering: true,
            },
            discovery_config: DiscoveryConfig {
                enable_semantic_search: true,
                enable_recommendation_engine: true,
                enable_popularity_ranking: true,
                enable_performance_ranking: true,
                caching_enabled: true,
                cache_ttl: Duration::from_secs(3600),
            },
            deployment_config: DeploymentConfig {
                max_concurrent_deployments: 100,
                deployment_timeout: Duration::from_secs(1800),
                auto_scaling_enabled: true,
                resource_limits: ResourceLimits {
                    max_qubits: 1000,
                    max_circuit_depth: 10000,
                    max_execution_time: Duration::from_secs(3600),
                    max_memory_usage: 32 * 1024 * 1024 * 1024, // 32GB
                    max_classical_processing: 16.0,            // 16 CPU cores
                },
                monitoring_enabled: true,
                rollback_enabled: true,
            },
            monetization_config: MonetizationConfig {
                enabled: true,
                supported_payment_methods: vec![
                    PaymentMethod::CreditCard,
                    PaymentMethod::DigitalWallet,
                    PaymentMethod::QuantumCredits,
                ],
                commission_rate: 0.15, // 15%
                revenue_sharing_enabled: true,
                subscription_models: vec![
                    SubscriptionModel::PayPerUse,
                    SubscriptionModel::Monthly,
                    SubscriptionModel::Annual,
                ],
                pricing_strategies: vec![
                    PricingStrategy::Fixed,
                    PricingStrategy::Dynamic,
                    PricingStrategy::PerformanceBased,
                ],
            },
            optimization_config: OptimizationConfig::default(),
            validation_config: ValidationConfig::default(),
            versioning_config: VersioningConfig::default(),
            api_config: APIConfig {
                rest_api_enabled: true,
                graphql_api_enabled: false,
                websocket_api_enabled: true,
                rate_limiting: RateLimitingConfig {
                    enabled: true,
                    requests_per_minute: 1000,
                    burst_limit: 100,
                    per_user_limits: HashMap::new(),
                },
                authentication_required: true,
                api_versioning: true,
            },
            security_config: MarketplaceSecurityConfig {
                code_scanning_enabled: true,
                vulnerability_checking: true,
                access_control_enabled: true,
                audit_logging: true,
                encryption_at_rest: true,
                encryption_in_transit: true,
            },
        }
    }
}

// ============================================================================
// AlgorithmMarketplace — lightweight synchronous facade
// ============================================================================
//
// The `QuantumAlgorithmMarketplace` above is a fully-async, subsystem-rich
// manager.  The `AlgorithmMarketplace` below is a simpler, *synchronous*
// facade that is easier to use in non-async contexts (CLI tools, unit tests,
// benchmarks) while still delegating to the same underlying data model.

/// A simplified, synchronous listing for an algorithm in the marketplace.
///
/// This is a thin wrapper around `RegisteredAlgorithm` / `AlgorithmMetadata`
/// that surfaces only the fields needed by the synchronous API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmListing {
    /// Stable, unique identifier for this algorithm (UUID v4 string).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Short description.
    pub description: String,
    /// Author or organization name.
    pub author: String,
    /// Semantic version string, e.g. `"1.0.0"`.
    pub version: String,
    /// High-level category.
    pub category: AlgorithmCategory,
    /// Free-form tags for search.
    pub tags: Vec<String>,
    /// Minimum qubit count required to run.
    pub min_qubits: usize,
    /// QASM3 / pseudo-code source snippet (may be empty for proprietary algorithms).
    pub source_snippet: String,
}

/// Query parameters for `AlgorithmMarketplace::search_algorithms`.
#[derive(Debug, Clone, Default)]
pub struct AlgorithmQuery {
    /// Filter by name substring (case-insensitive).
    pub name_contains: Option<String>,
    /// Filter by category.
    pub category: Option<AlgorithmCategory>,
    /// All of these tags must be present.
    pub required_tags: Vec<String>,
    /// Maximum qubit count (inclusive).  Useful for constrained hardware.
    pub max_qubits: Option<usize>,
    /// Author substring filter (case-insensitive).
    pub author_contains: Option<String>,
}

/// Input parameters for `AlgorithmMarketplace::execute_algorithm`.
#[derive(Debug, Clone)]
pub struct AlgorithmParams {
    /// Number of measurement shots.
    pub shots: usize,
    /// Named scalar parameters forwarded to the algorithm (e.g. rotation angles).
    pub scalar_params: HashMap<String, f64>,
    /// Arbitrary string metadata forwarded without interpretation.
    pub metadata: HashMap<String, String>,
}

impl Default for AlgorithmParams {
    fn default() -> Self {
        Self {
            shots: 1024,
            scalar_params: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Result returned by `AlgorithmMarketplace::execute_algorithm`.
#[derive(Debug, Clone)]
pub struct AlgorithmResult {
    /// Algorithm identifier that was executed.
    pub algorithm_id: String,
    /// Measurement outcome counts keyed by bitstring (e.g. `"001" -> 372`).
    pub counts: HashMap<String, usize>,
    /// Estimated success probability or fidelity (provider-dependent).
    pub estimated_fidelity: f64,
    /// Wall-clock execution time.
    pub execution_time: Duration,
    /// Arbitrary metadata returned by the execution backend.
    pub metadata: HashMap<String, String>,
}

/// Lightweight synchronous marketplace facade.
///
/// Stores algorithm listings in memory and provides CRUD + search + mock
/// execution without requiring an async runtime.
pub struct AlgorithmMarketplace {
    /// All registered algorithms keyed by their ID.
    listings: std::sync::RwLock<HashMap<String, AlgorithmListing>>,
}

impl AlgorithmMarketplace {
    /// Create an empty marketplace.
    pub fn new() -> Self {
        Self {
            listings: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Register a new algorithm.
    ///
    /// Fails if an algorithm with the same `id` is already registered.
    pub fn register_algorithm(&self, algo: AlgorithmListing) -> DeviceResult<()> {
        let mut listings = self
            .listings
            .write()
            .map_err(|e| DeviceError::LockError(format!("marketplace write lock: {e}")))?;
        if listings.contains_key(&algo.id) {
            return Err(DeviceError::InvalidInput(format!(
                "algorithm '{}' is already registered",
                algo.id
            )));
        }
        listings.insert(algo.id.clone(), algo);
        Ok(())
    }

    /// Search for algorithms matching `query`.
    ///
    /// Returns a cloned `Vec` so the caller does not need to hold any locks.
    pub fn search_algorithms(&self, query: &AlgorithmQuery) -> DeviceResult<Vec<AlgorithmListing>> {
        let listings = self
            .listings
            .read()
            .map_err(|e| DeviceError::LockError(format!("marketplace read lock: {e}")))?;
        let results = listings
            .values()
            .filter(|a| Self::matches(a, query))
            .cloned()
            .collect();
        Ok(results)
    }

    /// Retrieve an algorithm by its exact identifier.
    pub fn get_algorithm(&self, id: &str) -> DeviceResult<Option<AlgorithmListing>> {
        let listings = self
            .listings
            .read()
            .map_err(|e| DeviceError::LockError(format!("marketplace read lock: {e}")))?;
        Ok(listings.get(id).cloned())
    }

    /// Remove a previously registered algorithm.
    pub fn deregister_algorithm(&self, id: &str) -> DeviceResult<()> {
        let mut listings = self
            .listings
            .write()
            .map_err(|e| DeviceError::LockError(format!("marketplace write lock: {e}")))?;
        if listings.remove(id).is_none() {
            return Err(DeviceError::DeviceNotFound(format!(
                "algorithm '{}' not found",
                id
            )));
        }
        Ok(())
    }

    /// Execute an algorithm by ID with the given parameters.
    ///
    /// This is a *mock* executor: it simulates a uniform distribution over
    /// all 2^n bitstrings (where n is `min_qubits` for the listing) and
    /// records counts that sum to `params.shots`.  A real implementation
    /// would dispatch to a backend simulator or quantum hardware.
    pub fn execute_algorithm(
        &self,
        id: &str,
        params: AlgorithmParams,
    ) -> DeviceResult<AlgorithmResult> {
        let listing = self
            .get_algorithm(id)?
            .ok_or_else(|| DeviceError::DeviceNotFound(format!("algorithm '{}' not found", id)))?;

        let start = std::time::Instant::now();

        // Simulate a uniform distribution over bitstrings.
        // We use a simple deterministic PRNG (xorshift64) seeded from the
        // algorithm ID so that results are reproducible without pulling in a
        // full RNG library.
        let n_qubits = listing.min_qubits.clamp(1, 16);
        let n_states = 1usize << n_qubits;
        let shots = params.shots.max(1);

        let mut counts: HashMap<String, usize> = HashMap::with_capacity(n_states);
        let mut rng: u64 = {
            // Seed from id bytes using djb2 hash.
            let mut h: u64 = 5381;
            for b in id.bytes() {
                h = h.wrapping_mul(33).wrapping_add(b as u64);
            }
            h | 1 // ensure non-zero seed
        };

        for _ in 0..shots {
            // xorshift64
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let state_idx = (rng as usize) % n_states;
            let bitstring = format!("{:0width$b}", state_idx, width = n_qubits);
            *counts.entry(bitstring).or_insert(0) += 1;
        }

        let elapsed = start.elapsed();

        Ok(AlgorithmResult {
            algorithm_id: id.to_string(),
            counts,
            estimated_fidelity: 0.95, // Placeholder — real backend would populate this.
            execution_time: elapsed,
            metadata: params.metadata,
        })
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn matches(listing: &AlgorithmListing, query: &AlgorithmQuery) -> bool {
        if let Some(ref name_substr) = query.name_contains {
            if !listing
                .name
                .to_lowercase()
                .contains(&name_substr.to_lowercase())
            {
                return false;
            }
        }
        if let Some(ref cat) = query.category {
            if &listing.category != cat {
                return false;
            }
        }
        for tag in &query.required_tags {
            if !listing.tags.iter().any(|t| t.eq_ignore_ascii_case(tag)) {
                return false;
            }
        }
        if let Some(max_q) = query.max_qubits {
            if listing.min_qubits > max_q {
                return false;
            }
        }
        if let Some(ref author_substr) = query.author_contains {
            if !listing
                .author
                .to_lowercase()
                .contains(&author_substr.to_lowercase())
            {
                return false;
            }
        }
        true
    }
}

impl Default for AlgorithmMarketplace {
    fn default() -> Self {
        Self::new()
    }
}
