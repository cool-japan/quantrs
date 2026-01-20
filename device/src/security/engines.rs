//! Security engines and result types

use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};
use uuid::Uuid;

use super::config::*;
use super::types::*;
use crate::{DeviceError, DeviceResult};

/// Data structures for security operations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecurityOperationInfo {
    pub operation_id: String,
}

impl SecurityOperationInfo {
    pub fn from_operation(operation: SecurityOperation) -> Self {
        Self {
            operation_id: operation.operation_id,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThreatDetectionResults {
    pub threats_detected: usize,
    pub high_severity_threats: usize,
    pub medium_severity_threats: usize,
    pub low_severity_threats: usize,
    pub false_positives: usize,
    pub detection_algorithms_used: Vec<ThreatDetectionAlgorithm>,
    pub overall_risk_score: f64,
    pub detection_time: Duration,
}

impl ThreatDetectionResults {
    pub fn merge(traditional: Self, ml: Self) -> Self {
        Self {
            threats_detected: traditional.threats_detected + ml.threats_detected,
            high_severity_threats: traditional.high_severity_threats + ml.high_severity_threats,
            medium_severity_threats: traditional.medium_severity_threats
                + ml.medium_severity_threats,
            low_severity_threats: traditional.low_severity_threats + ml.low_severity_threats,
            false_positives: traditional.false_positives + ml.false_positives,
            detection_algorithms_used: [
                traditional.detection_algorithms_used,
                ml.detection_algorithms_used,
            ]
            .concat(),
            overall_risk_score: f64::midpoint(
                traditional.overall_risk_score,
                ml.overall_risk_score,
            ),
            detection_time: traditional.detection_time + ml.detection_time,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AccessControlAuditResults {
    pub audit_id: String,
    pub access_attempts: usize,
    pub successful_authentications: usize,
    pub failed_authentications: usize,
    pub authorization_decisions: usize,
    pub policy_violations: usize,
    pub anomalous_access_patterns: usize,
    pub risk_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CryptographicOperationResults {
    pub operations_executed: usize,
    pub algorithms_used: Vec<PostQuantumAlgorithm>,
    pub key_operations: usize,
    pub signature_operations: usize,
    pub encryption_operations: usize,
    pub performance_metrics: CryptographicPerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComplianceAssessmentResults {
    pub compliance_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecurityAnalyticsResults {
    pub analytics_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RiskAssessmentResults {
    pub risk_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IncidentResponseAction {
    pub action_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecurityMetrics {
    pub overall_security_score: f64,
    pub threat_detection_rate: f64,
    pub false_positive_rate: f64,
    pub incident_response_time: Duration,
    pub compliance_score: f64,
    pub risk_score: f64,
    pub encryption_coverage: f64,
    pub access_control_effectiveness: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecurityPerformanceImpact {
    pub performance_overhead: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecurityRecommendation {
    pub recommendation_id: String,
    pub category: SecurityRecommendationCategory,
    pub priority: RecommendationPriority,
    pub title: String,
    pub description: String,
    pub expected_impact: f64,
    pub implementation_effort: ImplementationEffort,
    pub timeline: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityExecutionMetadata {
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub total_duration: Option<Duration>,
    pub security_config: QuantumSecurityConfig,
    pub threat_landscape: ThreatLandscape,
    pub compliance_status: ComplianceStatus,
    pub security_posture: SecurityPosture,
}

impl Default for SecurityExecutionMetadata {
    fn default() -> Self {
        Self {
            start_time: SystemTime::UNIX_EPOCH,
            end_time: None,
            total_duration: None,
            security_config: QuantumSecurityConfig::default(),
            threat_landscape: ThreatLandscape::default(),
            compliance_status: ComplianceStatus::default(),
            security_posture: SecurityPosture::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuthenticationCredentials {
    pub credential_type: String,
    pub credential_data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecurityPolicyParameters {
    pub policy_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IncidentResponseParameters {
    pub incident_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    pub data: Vec<u8>,
    pub algorithm: PostQuantumAlgorithm,
    pub classification: SecurityClassification,
    pub key_id: String,
    pub timestamp: SystemTime,
    pub integrity_hash: Vec<u8>,
}

impl Default for EncryptedData {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            algorithm: PostQuantumAlgorithm::default(),
            classification: SecurityClassification::default(),
            key_id: String::new(),
            timestamp: SystemTime::UNIX_EPOCH,
            integrity_hash: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationResult {
    pub user_id: String,
    pub authenticated: bool,
    pub authentication_method: AuthenticationMethod,
    pub security_level: SecurityLevel,
    pub session_token: String,
    pub expires_at: SystemTime,
    pub additional_verification_required: bool,
}

impl Default for AuthenticationResult {
    fn default() -> Self {
        Self {
            user_id: String::new(),
            authenticated: false,
            authentication_method: AuthenticationMethod::default(),
            security_level: SecurityLevel::default(),
            session_token: String::new(),
            expires_at: SystemTime::UNIX_EPOCH,
            additional_verification_required: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationResult {
    pub user_id: String,
    pub resource: String,
    pub authorized: bool,
    pub granted_permissions: Vec<String>,
    pub authorization_model: AuthorizationModel,
    pub decision_factors: Vec<String>,
    pub expires_at: SystemTime,
}

impl Default for AuthorizationResult {
    fn default() -> Self {
        Self {
            user_id: String::new(),
            resource: String::new(),
            authorized: false,
            granted_permissions: Vec::new(),
            authorization_model: AuthorizationModel::default(),
            decision_factors: Vec::new(),
            expires_at: SystemTime::UNIX_EPOCH,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantumThreatDetectionResult {
    pub target_system: String,
    pub threats_detected: usize,
    pub quantum_state_anomalies: usize,
    pub quantum_noise_anomalies: usize,
    pub decoherence_attacks: usize,
    pub side_channel_attacks: usize,
    pub risk_score: f64,
    pub detection_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantumRiskAssessment {
    pub asset_id: String,
    pub risk_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecurityReport {
    pub report_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecurityOperationContext {
    pub context_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecurityRequirement {
    pub requirement_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CryptographicPerformanceMetrics {
    pub operation_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThreatLandscape {
    pub threat_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComplianceStatus {
    pub compliance_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SecurityPosture {
    pub overall_posture_score: f64,
    pub maturity_level: SecurityMaturityLevel,
    pub key_strengths: Vec<String>,
    pub areas_for_improvement: Vec<String>,
    pub risk_exposure: f64,
}

/// Security operation definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityOperation {
    /// Operation identifier
    pub operation_id: String,
    /// Operation type
    pub operation_type: SecurityOperationType,
    /// Target system or resource
    pub target: String,
    /// Requesting user
    pub user_id: String,
    /// Security requirements
    pub security_requirements: Vec<SecurityRequirement>,
    /// Classification level
    pub classification: SecurityClassification,
    /// Operation context
    pub context: SecurityOperationContext,
    /// Risk tolerance
    pub risk_tolerance: f64,
}

/// Comprehensive execution result for quantum security operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSecurityExecutionResult {
    /// Execution identifier
    pub execution_id: String,
    /// Security execution status
    pub status: QuantumSecurityExecutionStatus,
    /// Security operations performed
    pub security_operations: Vec<SecurityOperationInfo>,
    /// Threat detection results
    pub threat_detection_results: ThreatDetectionResults,
    /// Access control audit results
    pub access_control_results: AccessControlAuditResults,
    /// Cryptographic operations results
    pub cryptographic_results: CryptographicOperationResults,
    /// Compliance assessment results
    pub compliance_results: ComplianceAssessmentResults,
    /// Security analytics results
    pub security_analytics: SecurityAnalyticsResults,
    /// Risk assessment results
    pub risk_assessment: RiskAssessmentResults,
    /// Incident response actions
    pub incident_response_actions: Vec<IncidentResponseAction>,
    /// Security metrics
    pub security_metrics: SecurityMetrics,
    /// Performance impact analysis
    pub performance_impact: SecurityPerformanceImpact,
    /// Recommendations
    pub security_recommendations: Vec<SecurityRecommendation>,
    /// Execution metadata
    pub execution_metadata: SecurityExecutionMetadata,
}
