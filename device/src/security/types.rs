//! Security types and enums

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

use crate::{DeviceError, DeviceResult};

/// Security classification levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum SecurityClassification {
    #[default]
    Public,
    Internal,
    Confidential,
    Secret,
    TopSecret,
    QuantumProtected,
    Custom(String),
}

/// Security objectives
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecurityObjective {
    Confidentiality,
    Integrity,
    Availability,
    Authentication,
    Authorization,
    NonRepudiation,
    Privacy,
    Compliance,
    QuantumSafety,
    Custom(String),
}

/// Security standards
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecurityStandard {
    ISO27001,
    NistCsf,
    SOC2,
    FedRAMP,
    GDPR,
    HIPAA,
    PciDss,
    Fips140_2,
    CommonCriteria,
    QuantumSafeNist,
    Custom(String),
}

/// Post-quantum algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PostQuantumAlgorithm {
    // NIST Post-Quantum Cryptography Standards
    Kyber,
    Dilithium,
    Falcon,
    SphincsPlus,
    // Additional algorithms
    NTRU,
    McEliece,
    Rainbow,
    SIDH,
    SIKE,
    NewHope,
    FrodoKEM,
    Custom(String),
}

impl Default for PostQuantumAlgorithm {
    fn default() -> Self {
        PostQuantumAlgorithm::Kyber
    }
}

/// Authentication methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuthenticationMethod {
    Password,
    Biometric,
    SmartCard,
    QuantumKey,
    CertificateBased,
    TokenBased,
    BehavioralBiometrics,
    ZeroKnowledgeProof,
    QuantumSignature,
    Custom(String),
}

impl Default for AuthenticationMethod {
    fn default() -> Self {
        AuthenticationMethod::Password
    }
}

/// Authorization models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuthorizationModel {
    RBAC,       // Role-Based Access Control
    ABAC,       // Attribute-Based Access Control
    DAC,        // Discretionary Access Control
    MAC,        // Mandatory Access Control
    PBAC,       // Policy-Based Access Control
    QuantumACL, // Quantum Access Control List
    ZeroTrust,
    Custom(String),
}

impl Default for AuthorizationModel {
    fn default() -> Self {
        AuthorizationModel::RBAC
    }
}

/// Threat detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ThreatDetectionAlgorithm {
    SignatureBased,
    BehaviorBased,
    MachineLearning,
    StatisticalAnalysis,
    AnomalyDetection,
    HeuristicAnalysis,
    QuantumStateAnalysis,
    QuantumNoiseAnalysis,
    Custom(String),
}

/// Security analytics engines
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecurityAnalyticsEngine {
    SIEM, // Security Information and Event Management
    SOAR, // Security Orchestration, Automation and Response
    UEBA, // User and Entity Behavior Analytics
    ThreatIntelligence,
    QuantumSecurityAnalytics,
    MLSecurityAnalytics,
    RiskAnalytics,
    Custom(String),
}

/// Regulatory frameworks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RegulatoryFramework {
    GDPR,
    CCPA,
    HIPAA,
    SOX,
    PciDss,
    FISMA,
    ITAR,
    EAR,
    QuantumRegulations,
    Custom(String),
}

/// Compliance standards
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComplianceStandard {
    ISO27001,
    Soc2Type1,
    Soc2Type2,
    FedRampLow,
    FedRampModerate,
    FedRampHigh,
    Nist800_53,
    CisControls,
    QuantumCompliance,
    Custom(String),
}

/// Encryption protocols
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EncryptionProtocol {
    Tls1_3,
    IPSec,
    WireGuard,
    QuantumSafeTLS,
    QuantumKeyDistribution,
    QuantumTunneling,
    PostQuantumVPN,
    Custom(String),
}

/// Security ML models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecurityMLModel {
    AnomalyDetection,
    ThreatClassification,
    BehaviorProfiling,
    RiskScoring,
    FraudDetection,
    IntrusionDetection,
    QuantumAnomalyDetection,
    QuantumThreatClassification,
    Custom(String),
}

/// Security operation types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecurityOperationType {
    Authentication,
    Authorization,
    Encryption,
    Decryption,
    ThreatDetection,
    RiskAssessment,
    ComplianceAudit,
    IncidentResponse,
    SecurityAnalytics,
    PolicyEnforcement,
    DataProtection,
    HardwareSecurity,
    CommunicationSecurity,
    Custom(String),
}

/// Quantum security execution status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantumSecurityExecutionStatus {
    Pending,
    Initializing,
    AuthenticatingUsers,
    DetectingThreats,
    AnalyzingRisks,
    EnforcingPolicies,
    MonitoringCompliance,
    RespondingToIncidents,
    AnalyzingPerformance,
    Completed,
    Failed,
    PartiallyCompleted,
    ComplianceViolation,
    SecurityThreatDetected,
}

/// Threat severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ThreatSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl Default for ThreatSeverity {
    fn default() -> Self {
        ThreatSeverity::Low
    }
}

/// Incident severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IncidentSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl Default for IncidentSeverity {
    fn default() -> Self {
        IncidentSeverity::Low
    }
}

/// Data protection event types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataProtectionEventType {
    AccessRequest,
    DataModification,
    DataDeletion,
    SecurityViolation,
}

impl Default for DataProtectionEventType {
    fn default() -> Self {
        DataProtectionEventType::AccessRequest
    }
}

/// Security report types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecurityReportType {
    Summary,
    Detailed,
    Compliance,
    ThreatAnalysis,
}

impl Default for SecurityReportType {
    fn default() -> Self {
        SecurityReportType::Summary
    }
}

/// Security level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecurityLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl Default for SecurityLevel {
    fn default() -> Self {
        SecurityLevel::Medium
    }
}

/// Security recommendation category
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecurityRecommendationCategory {
    ThreatDetection,
    Cryptography,
    AccessControl,
    Compliance,
}

impl Default for SecurityRecommendationCategory {
    fn default() -> Self {
        SecurityRecommendationCategory::ThreatDetection
    }
}

/// Recommendation priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

impl Default for RecommendationPriority {
    fn default() -> Self {
        RecommendationPriority::Medium
    }
}

/// Implementation effort
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ImplementationEffort {
    Low,
    Medium,
    High,
}

impl Default for ImplementationEffort {
    fn default() -> Self {
        ImplementationEffort::Medium
    }
}

/// Security maturity level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecurityMaturityLevel {
    Basic,
    Intermediate,
    Advanced,
    Expert,
}

impl Default for SecurityMaturityLevel {
    fn default() -> Self {
        SecurityMaturityLevel::Basic
    }
}

/// Helper trait for duration extensions
pub trait DurationExt {
    fn from_weeks(weeks: u64) -> Duration;
    fn from_hours(hours: u64) -> Duration;
    fn from_minutes(minutes: u64) -> Duration;
}

impl DurationExt for Duration {
    fn from_weeks(weeks: u64) -> Duration {
        Duration::from_secs(weeks * 7 * 24 * 3600)
    }

    fn from_hours(hours: u64) -> Duration {
        Duration::from_secs(hours * 3600)
    }

    fn from_minutes(minutes: u64) -> Duration {
        Duration::from_secs(minutes * 60)
    }
}
