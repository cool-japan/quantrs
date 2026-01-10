//! Basic types for provider capability discovery.
//!
//! This module contains provider information, endpoint types,
//! pricing models, and basic capability types.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use url::Url;

use super::config::ComplianceStandard;

/// Topology types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyType {
    Linear,
    Grid,
    Heavy,
    Falcon,
    Hummingbird,
    Eagle,
    Custom(String),
}

/// Provider features
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProviderFeature {
    QuantumComputing,
    QuantumSimulation,
    NoiseModeling,
    ErrorCorrection,
    MidCircuitMeasurement,
    ParametricCircuits,
    PulseControl,
    RealTimeControl,
    HybridAlgorithms,
    QuantumNetworking,
    Custom(String),
}

/// Provider information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderInfo {
    /// Provider ID
    pub provider_id: String,
    /// Provider name
    pub name: String,
    /// Provider description
    pub description: String,
    /// Provider type
    pub provider_type: ProviderType,
    /// Contact information
    pub contact_info: ContactInfo,
    /// Service endpoints
    pub endpoints: Vec<ServiceEndpoint>,
    /// Supported regions
    pub supported_regions: Vec<String>,
    /// Pricing model
    pub pricing_model: PricingModel,
    /// Terms of service
    pub terms_of_service: Option<String>,
    /// Privacy policy
    pub privacy_policy: Option<String>,
    /// Compliance certifications
    pub compliance_certifications: Vec<ComplianceStandard>,
    /// Last updated
    pub last_updated: SystemTime,
}

/// Provider types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderType {
    /// Cloud-based quantum computing provider
    CloudProvider,
    /// Hardware manufacturer
    HardwareManufacturer,
    /// Software platform
    SoftwarePlatform,
    /// Research institution
    ResearchInstitution,
    /// Service integrator
    ServiceIntegrator,
    /// Custom provider type
    Custom(String),
}

/// Contact information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactInfo {
    /// Support email
    pub support_email: Option<String>,
    /// Support phone
    pub support_phone: Option<String>,
    /// Support website
    pub support_website: Option<Url>,
    /// Technical contact
    pub technical_contact: Option<String>,
    /// Business contact
    pub business_contact: Option<String>,
    /// Emergency contact
    pub emergency_contact: Option<String>,
}

/// Service endpoint information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    /// Endpoint URL
    pub url: Url,
    /// Endpoint type
    pub endpoint_type: EndpointType,
    /// API version
    pub api_version: String,
    /// Authentication methods
    pub auth_methods: Vec<AuthenticationMethod>,
    /// Rate limits
    pub rate_limits: Option<RateLimits>,
    /// Health status
    pub health_status: EndpointHealth,
    /// Response time statistics
    pub response_time_stats: ResponseTimeStats,
}

/// Endpoint types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndpointType {
    REST,
    GraphQL,
    WebSocket,
    GRpc,
    Custom(String),
}

/// Authentication methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthenticationMethod {
    APIKey,
    OAuth2,
    JWT,
    BasicAuth,
    Certificate,
    Custom(String),
}

/// Rate limit information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimits {
    /// Requests per minute
    pub requests_per_minute: u32,
    /// Requests per hour
    pub requests_per_hour: u32,
    /// Requests per day
    pub requests_per_day: u32,
    /// Burst limit
    pub burst_limit: u32,
    /// Concurrent requests
    pub concurrent_requests: u32,
}

/// Endpoint health status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndpointHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Response time statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTimeStats {
    /// Average response time
    pub average_ms: f64,
    /// Median response time
    pub median_ms: f64,
    /// 95th percentile
    pub p95_ms: f64,
    /// 99th percentile
    pub p99_ms: f64,
    /// Standard deviation
    pub std_dev_ms: f64,
}

/// Pricing model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingModel {
    /// Pricing type
    pub pricing_type: PricingType,
    /// Cost per shot
    pub cost_per_shot: Option<f64>,
    /// Cost per circuit
    pub cost_per_circuit: Option<f64>,
    /// Cost per hour
    pub cost_per_hour: Option<f64>,
    /// Monthly subscription
    pub monthly_subscription: Option<f64>,
    /// Free tier limits
    pub free_tier: Option<FreeTierLimits>,
    /// Currency
    pub currency: String,
    /// Billing model
    pub billing_model: BillingModel,
}

/// Pricing types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PricingType {
    PayPerUse,
    Subscription,
    Hybrid,
    Enterprise,
    Academic,
    Free,
    Custom,
}

/// Free tier limitations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeTierLimits {
    /// Maximum shots per month
    pub max_shots_per_month: Option<u64>,
    /// Maximum circuits per month
    pub max_circuits_per_month: Option<u64>,
    /// Maximum queue time
    pub max_queue_time: Option<Duration>,
    /// Feature limitations
    pub feature_limitations: Vec<String>,
}

/// Billing models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BillingModel {
    Prepaid,
    Postpaid,
    Credit,
    Invoice,
    Custom,
}

/// Cached capability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedCapability {
    /// Provider ID
    pub provider_id: String,
    /// Capabilities
    pub capabilities: super::capabilities::ProviderCapabilities,
    /// Cache timestamp
    pub cached_at: SystemTime,
    /// Expiration time
    pub expires_at: SystemTime,
    /// Verification status
    pub verification_status: VerificationStatus,
    /// Access count
    pub access_count: u64,
}

/// Verification status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStatus {
    Verified,
    PartiallyVerified,
    Unverified,
    Failed,
    Pending,
}

/// Connectivity graph representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityGraph {
    /// Adjacency list
    pub adjacency_list: HashMap<usize, Vec<usize>>,
    /// Edge weights (if applicable)
    pub edge_weights: Option<HashMap<(usize, usize), f64>>,
    /// Topology type
    pub topology_type: TopologyType,
    /// Connectivity metrics
    pub metrics: ConnectivityMetrics,
}

/// Connectivity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityMetrics {
    /// Average degree
    pub average_degree: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Diameter
    pub diameter: usize,
    /// Density
    pub density: f64,
    /// Number of connected components
    pub connected_components: usize,
}

/// Measurement types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementType {
    ComputationalBasis,
    Pauli,
    POVM,
    Projective,
    Weak,
    Custom(String),
}

/// Quantum frameworks
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantumFramework {
    Qiskit,
    Cirq,
    QSharp,
    Braket,
    Pennylane,
    Strawberry,
    Tket,
    Forest,
    ProjectQ,
    Custom(String),
}

/// Authorization models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthorizationModel {
    RBAC, // Role-Based Access Control
    ABAC, // Attribute-Based Access Control
    ACL,  // Access Control List
    MAC,  // Mandatory Access Control
    DAC,  // Discretionary Access Control
    Custom(String),
}

/// Support channels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SupportChannel {
    Email,
    Phone,
    Chat,
    Forum,
    Documentation,
    VideoCall,
    OnSite,
    Custom(String),
}

/// Maintenance frequency
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaintenanceFrequency {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annually,
    AsNeeded,
}

/// Impact levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImpactLevel {
    None,
    Low,
    Medium,
    High,
    Critical,
}
