//! Security types and enums

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use uuid::Uuid;

use crate::{DeviceError, DeviceResult};

/// Security classification levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum PostQuantumAlgorithm {
    // NIST Post-Quantum Cryptography Standards
    #[default]
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

/// Authentication methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum AuthenticationMethod {
    #[default]
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

/// Authorization models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum AuthorizationModel {
    #[default]
    RBAC, // Role-Based Access Control
    ABAC,       // Attribute-Based Access Control
    DAC,        // Discretionary Access Control
    MAC,        // Mandatory Access Control
    PBAC,       // Policy-Based Access Control
    QuantumACL, // Quantum Access Control List
    ZeroTrust,
    Custom(String),
}

/// Threat detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ThreatSeverity {
    #[default]
    Low,
    Medium,
    High,
    Critical,
}

/// Incident severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum IncidentSeverity {
    #[default]
    Low,
    Medium,
    High,
    Critical,
}

/// Data protection event types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum DataProtectionEventType {
    #[default]
    AccessRequest,
    DataModification,
    DataDeletion,
    SecurityViolation,
}

/// Security report types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum SecurityReportType {
    #[default]
    Summary,
    Detailed,
    Compliance,
    ThreatAnalysis,
}

/// Security level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum SecurityLevel {
    Low,
    #[default]
    Medium,
    High,
    Critical,
}

/// Security recommendation category
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum SecurityRecommendationCategory {
    #[default]
    ThreatDetection,
    Cryptography,
    AccessControl,
    Compliance,
}

/// Recommendation priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum RecommendationPriority {
    Low,
    #[default]
    Medium,
    High,
    Critical,
}

/// Implementation effort
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ImplementationEffort {
    Low,
    #[default]
    Medium,
    High,
}

/// Security maturity level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum SecurityMaturityLevel {
    #[default]
    Basic,
    Intermediate,
    Advanced,
    Expert,
}

/// Helper trait for duration extensions
pub trait DurationExt {
    fn from_weeks(weeks: u64) -> Duration;
    fn from_hours(hours: u64) -> Duration;
    fn from_minutes(minutes: u64) -> Duration;
}

impl DurationExt for Duration {
    fn from_weeks(weeks: u64) -> Duration {
        Self::from_secs(weeks * 7 * 24 * 3600)
    }

    fn from_hours(hours: u64) -> Duration {
        Self::from_secs(hours * 3600)
    }

    fn from_minutes(minutes: u64) -> Duration {
        Self::from_secs(minutes * 60)
    }
}
