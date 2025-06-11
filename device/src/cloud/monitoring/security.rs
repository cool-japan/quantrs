//! Security monitoring and threat detection configuration.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Security monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudSecurityMonitoringConfig {
    /// Enable security monitoring
    pub enabled: bool,
    /// Security events to monitor
    pub events: Vec<SecurityEvent>,
    /// Threat detection
    pub threat_detection: ThreatDetectionConfig,
    /// Incident response
    pub incident_response: IncidentResponseConfig,
    /// Compliance monitoring
    pub compliance: ComplianceMonitoringConfig,
}

/// Security events
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SecurityEvent {
    UnauthorizedAccess,
    SuspiciousActivity,
    DataBreach,
    MalwareDetection,
    NetworkIntrusion,
    PrivilegeEscalation,
    QuantumCircuitTampering,
    ConfigurationChanges,
    Custom(String),
}

/// Threat detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatDetectionConfig {
    /// Enable threat detection
    pub enabled: bool,
    /// Detection methods
    pub methods: Vec<ThreatDetectionMethod>,
    /// Response policies
    pub response_policies: Vec<ThreatResponsePolicy>,
    /// Intelligence feeds
    pub intelligence: ThreatIntelligenceConfig,
}

/// Threat detection methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ThreatDetectionMethod {
    SignatureBased,
    BehaviorBased,
    MachineLearning,
    AnomalyDetection,
    HybridApproach,
    Custom(String),
}

/// Threat response policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatResponsePolicy {
    /// Policy name
    pub name: String,
    /// Threat types this policy applies to
    pub threat_types: Vec<ThreatType>,
    /// Response actions
    pub actions: Vec<ThreatResponseAction>,
    /// Escalation rules
    pub escalation: ThreatEscalationRules,
}

/// Threat types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ThreatType {
    Malware,
    Phishing,
    DDoS,
    DataExfiltration,
    InsiderThreat,
    QuantumThreat,
    Custom(String),
}

/// Threat response actions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ThreatResponseAction {
    Alert,
    Block,
    Quarantine,
    Isolate,
    Investigate,
    Escalate,
    Custom(String),
}

/// Threat escalation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatEscalationRules {
    /// Escalation triggers
    pub triggers: Vec<EscalationTrigger>,
    /// Escalation levels
    pub levels: Vec<ThreatEscalationLevel>,
}

/// Escalation triggers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EscalationTrigger {
    ThreatSeverity,
    ResponseTimeout,
    FailedMitigation,
    Custom(String),
}

/// Threat escalation level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatEscalationLevel {
    /// Level name
    pub name: String,
    /// Severity threshold
    pub severity_threshold: f64,
    /// Response time
    pub response_time: Duration,
    /// Escalation contacts
    pub contacts: Vec<String>,
}

/// Threat intelligence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatIntelligenceConfig {
    /// Enable threat intelligence
    pub enabled: bool,
    /// Intelligence sources
    pub sources: Vec<IntelligenceSource>,
    /// Feed integration
    pub feeds: Vec<FeedIntegrationConfig>,
    /// Update frequency
    pub update_frequency: Duration,
}

/// Intelligence sources
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IntelligenceSource {
    Commercial,
    OpenSource,
    Government,
    Community,
    Internal,
    Custom(String),
}

/// Feed integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedIntegrationConfig {
    /// Feed name
    pub name: String,
    /// Feed type
    pub feed_type: String,
    /// API endpoint
    pub endpoint: String,
    /// Authentication
    pub auth: FeedAuthConfig,
}

/// Feed authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedAuthConfig {
    /// Authentication type
    pub auth_type: AuthType,
    /// Credentials
    pub credentials: std::collections::HashMap<String, String>,
}

/// Authentication types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuthType {
    ApiKey,
    OAuth,
    Basic,
    Certificate,
    Custom(String),
}

/// Incident response configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncidentResponseConfig {
    /// Enable automated response
    pub automated: bool,
    /// Response playbooks
    pub playbooks: Vec<ResponsePlaybook>,
    /// Communication plan
    pub communication: CommunicationPlan,
}

/// Response playbook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsePlaybook {
    /// Playbook name
    pub name: String,
    /// Trigger conditions
    pub triggers: Vec<String>,
    /// Response steps
    pub steps: Vec<ResponseStep>,
}

/// Response step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseStep {
    /// Step name
    pub name: String,
    /// Action type
    pub action_type: ResponseActionType,
    /// Parameters
    pub parameters: std::collections::HashMap<String, String>,
}

/// Response action types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResponseActionType {
    Notify,
    Execute,
    Isolate,
    Investigate,
    Document,
    Custom(String),
}

/// Communication plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPlan {
    /// Notification channels
    pub channels: Vec<String>,
    /// Stakeholder groups
    pub stakeholders: Vec<StakeholderGroup>,
    /// Escalation matrix
    pub escalation_matrix: Vec<EscalationContact>,
}

/// Stakeholder group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakeholderGroup {
    /// Group name
    pub name: String,
    /// Contact methods
    pub contacts: Vec<String>,
    /// Notification conditions
    pub conditions: Vec<String>,
}

/// Escalation contact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationContact {
    /// Contact name
    pub name: String,
    /// Contact method
    pub method: String,
    /// Escalation level
    pub level: i32,
    /// Response time requirement
    pub response_time: Duration,
}

/// Compliance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceMonitoringConfig {
    /// Enable compliance monitoring
    pub enabled: bool,
    /// Compliance frameworks
    pub frameworks: Vec<ComplianceFramework>,
    /// Audit configuration
    pub audit: AuditConfig,
    /// Reporting
    pub reporting: ComplianceReportingConfig,
}

/// Compliance frameworks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ComplianceFramework {
    SOC2,
    ISO27001,
    GDPR,
    HIPAA,
    PciDss,
    FedRAMP,
    Custom(String),
}

/// Audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Log retention period
    pub retention: Duration,
    /// Audit events
    pub events: Vec<AuditEvent>,
}

/// Audit events
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuditEvent {
    UserLogin,
    UserLogout,
    DataAccess,
    ConfigChange,
    PermissionChange,
    SystemAccess,
    Custom(String),
}

/// Compliance reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReportingConfig {
    /// Enable automated reporting
    pub automated: bool,
    /// Report frequency
    pub frequency: Duration,
    /// Report types
    pub types: Vec<ReportType>,
}

/// Report types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReportType {
    ComplianceStatus,
    VulnerabilityAssessment,
    SecurityPosture,
    IncidentSummary,
    Custom(String),
}

impl Default for CloudSecurityMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            events: vec![
                SecurityEvent::UnauthorizedAccess,
                SecurityEvent::SuspiciousActivity,
                SecurityEvent::ConfigurationChanges,
            ],
            threat_detection: ThreatDetectionConfig::default(),
            incident_response: IncidentResponseConfig::default(),
            compliance: ComplianceMonitoringConfig::default(),
        }
    }
}

impl Default for ThreatDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            methods: vec![ThreatDetectionMethod::SignatureBased],
            response_policies: vec![],
            intelligence: ThreatIntelligenceConfig::default(),
        }
    }
}

impl Default for ThreatIntelligenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            sources: vec![IntelligenceSource::OpenSource],
            feeds: vec![],
            update_frequency: Duration::from_secs(3600), // hourly
        }
    }
}

impl Default for IncidentResponseConfig {
    fn default() -> Self {
        Self {
            automated: false,
            playbooks: vec![],
            communication: CommunicationPlan::default(),
        }
    }
}

impl Default for CommunicationPlan {
    fn default() -> Self {
        Self {
            channels: vec!["email".to_string()],
            stakeholders: vec![],
            escalation_matrix: vec![],
        }
    }
}

impl Default for ComplianceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            frameworks: vec![],
            audit: AuditConfig::default(),
            reporting: ComplianceReportingConfig::default(),
        }
    }
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            retention: Duration::from_secs(86400 * 365), // 1 year
            events: vec![
                AuditEvent::UserLogin,
                AuditEvent::DataAccess,
                AuditEvent::ConfigChange,
            ],
        }
    }
}

impl Default for ComplianceReportingConfig {
    fn default() -> Self {
        Self {
            automated: false,
            frequency: Duration::from_secs(86400 * 30), // monthly
            types: vec![ReportType::ComplianceStatus],
        }
    }
}
