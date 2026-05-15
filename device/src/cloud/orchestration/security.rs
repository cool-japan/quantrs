//! Cloud Orchestration — Security Configuration
//!
//! Authentication, authorization, encryption, network-security, and compliance
//! configuration types for cloud orchestration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Cloud security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudSecurityConfig {
    /// Authentication configuration
    pub authentication: AuthenticationConfig,
    /// Authorization configuration
    pub authorization: AuthorizationConfig,
    /// Encryption configuration
    pub encryption: EncryptionConfig,
    /// Network security
    pub network_security: NetworkSecurityConfig,
    /// Compliance configuration
    pub compliance: ComplianceConfig,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    /// Authentication methods
    pub methods: Vec<AuthMethod>,
    /// Multi-factor authentication
    pub mfa: MFAConfig,
    /// Single sign-on
    pub sso: SSOConfig,
}

/// Authentication methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthMethod {
    Password,
    APIKey,
    Certificate,
    OAuth2,
    SAML,
    Custom(String),
}

/// MFA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MFAConfig {
    /// Enable MFA
    pub enabled: bool,
    /// MFA methods
    pub methods: Vec<MFAMethod>,
    /// Backup codes
    pub backup_codes: bool,
}

/// MFA methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MFAMethod {
    TOTP,
    SMS,
    Email,
    PushNotification,
    Hardware,
}

/// SSO configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSOConfig {
    /// Enable SSO
    pub enabled: bool,
    /// SSO provider
    pub provider: SSOProvider,
    /// SAML configuration
    pub saml: SAMLConfig,
    /// OIDC configuration
    pub oidc: OIDCConfig,
}

/// SSO providers
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SSOProvider {
    SAML,
    OIDC,
    LDAP,
    ActiveDirectory,
    Custom(String),
}

/// SAML configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SAMLConfig {
    /// Identity provider URL
    pub idp_url: String,
    /// Service provider ID
    pub sp_id: String,
    /// Certificate
    pub certificate: String,
}

/// OIDC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OIDCConfig {
    /// Client ID
    pub client_id: String,
    /// Client secret
    pub client_secret: String,
    /// Discovery URL
    pub discovery_url: String,
}

/// Authorization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationConfig {
    /// Authorization model
    pub model: AuthorizationModel,
    /// Role definitions
    pub roles: Vec<RoleDefinition>,
    /// Permission system
    pub permissions: PermissionSystem,
}

/// Authorization models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthorizationModel {
    RBAC,
    ABAC,
    DAC,
    MAC,
    Custom(String),
}

/// Role definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleDefinition {
    /// Role name
    pub name: String,
    /// Description
    pub description: String,
    /// Permissions
    pub permissions: Vec<String>,
    /// Role hierarchy
    pub parent_roles: Vec<String>,
}

/// Permission system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionSystem {
    /// Permission model
    pub model: PermissionModel,
    /// Resource definitions
    pub resources: Vec<ResourceDefinition>,
    /// Action definitions
    pub actions: Vec<ActionDefinition>,
}

/// Permission models
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionModel {
    ResourceAction,
    CapabilityBased,
    AttributeBased,
    Custom(String),
}

/// Resource definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceDefinition {
    /// Resource type
    pub resource_type: String,
    /// Resource attributes
    pub attributes: HashMap<String, String>,
    /// Access patterns
    pub access_patterns: Vec<AccessPattern>,
}

/// Access pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPattern {
    /// Pattern name
    pub name: String,
    /// Allowed actions
    pub actions: Vec<String>,
    /// Conditions
    pub conditions: Vec<AccessCondition>,
}

/// Access condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessCondition {
    /// Attribute
    pub attribute: String,
    /// Operator
    pub operator: String,
    /// Value
    pub value: String,
}

/// Action definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionDefinition {
    /// Action name
    pub name: String,
    /// Description
    pub description: String,
    /// Required permissions
    pub required_permissions: Vec<String>,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Encryption at rest
    pub at_rest: EncryptionAtRestConfig,
    /// Encryption in transit
    pub in_transit: EncryptionInTransitConfig,
    /// Key management
    pub key_management: EncryptionKeyManagementConfig,
}

/// Encryption at rest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionAtRestConfig {
    /// Enable encryption
    pub enabled: bool,
    /// Encryption algorithm
    pub algorithm: String,
    /// Key size
    pub key_size: usize,
}

/// Encryption in transit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionInTransitConfig {
    /// Enable encryption
    pub enabled: bool,
    /// TLS version
    pub tls_version: String,
    /// Cipher suites
    pub cipher_suites: Vec<String>,
}

/// Encryption key management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionKeyManagementConfig {
    /// Key management service
    pub service: KeyManagementService,
    /// Key rotation policy
    pub rotation_policy: EncryptionKeyRotationPolicy,
    /// Key backup policy
    pub backup_policy: EncryptionKeyBackupPolicy,
}

/// Key management services
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeyManagementService {
    AwsKms,
    AzureKeyVault,
    GoogleKms,
    HashiCorpVault,
    Custom(String),
}

/// Key rotation policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionKeyRotationPolicy {
    /// Enable rotation
    pub enabled: bool,
    /// Rotation frequency
    pub frequency: Duration,
    /// Automatic rotation
    pub automatic: bool,
}

/// Key backup policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionKeyBackupPolicy {
    /// Enable backup
    pub enabled: bool,
    /// Backup frequency
    pub frequency: Duration,
    /// Backup locations
    pub locations: Vec<String>,
}

/// Network security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSecurityConfig {
    /// Firewall configuration
    pub firewall: FirewallConfig,
    /// VPN configuration
    pub vpn: VPNConfig,
    /// DDoS protection
    pub ddos_protection: DDoSProtectionConfig,
}

/// Firewall configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallConfig {
    /// Enable firewall
    pub enabled: bool,
    /// Firewall rules
    pub rules: Vec<FirewallRule>,
    /// Default policy
    pub default_policy: FirewallPolicy,
}

/// Firewall rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirewallRule {
    /// Rule name
    pub name: String,
    /// Source
    pub source: String,
    /// Destination
    pub destination: String,
    /// Port
    pub port: String,
    /// Protocol
    pub protocol: String,
    /// Action
    pub action: FirewallAction,
}

/// Firewall policies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FirewallPolicy {
    Allow,
    Deny,
    Log,
}

/// Firewall actions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FirewallAction {
    Allow,
    Deny,
    Log,
}

/// VPN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VPNConfig {
    /// Enable VPN
    pub enabled: bool,
    /// VPN type
    pub vpn_type: VPNType,
    /// Connection settings
    pub connection: VPNConnectionConfig,
}

/// VPN types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VPNType {
    SiteToSite,
    PointToSite,
    PointToPoint,
    Custom(String),
}

/// VPN connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VPNConnectionConfig {
    /// Gateway address
    pub gateway: String,
    /// Pre-shared key
    pub psk: String,
    /// Encryption
    pub encryption: String,
}

/// DDoS protection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DDoSProtectionConfig {
    /// Enable protection
    pub enabled: bool,
    /// Protection level
    pub level: DDoSProtectionLevel,
    /// Rate limiting
    pub rate_limiting: RateLimitingConfig,
}

/// DDoS protection levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DDoSProtectionLevel {
    Basic,
    Standard,
    Premium,
    Custom(String),
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitingConfig {
    /// Enable rate limiting
    pub enabled: bool,
    /// Request limits
    pub limits: HashMap<String, usize>,
    /// Time windows
    pub windows: HashMap<String, Duration>,
}

/// Compliance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    /// Compliance frameworks
    pub frameworks: Vec<ComplianceFramework>,
    /// Audit configuration
    pub audit: AuditConfig,
    /// Data governance
    pub data_governance: DataGovernanceConfig,
}

/// Compliance frameworks
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceFramework {
    SOC2,
    ISO27001,
    GDPR,
    HIPAA,
    PciDss,
    Custom(String),
}

/// Audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable auditing
    pub enabled: bool,
    /// Audit events
    pub events: Vec<AuditEvent>,
    /// Log retention
    pub retention: Duration,
}

/// Audit events
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditEvent {
    Authentication,
    Authorization,
    DataAccess,
    ConfigChange,
    SecurityEvent,
    Custom(String),
}

/// Data governance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataGovernanceConfig {
    /// Data classification
    pub classification: DataClassificationConfig,
    /// Data retention
    pub retention: DataRetentionConfig,
    /// Data privacy
    pub privacy: DataPrivacyConfig,
}

/// Data classification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataClassificationConfig {
    /// Classification levels
    pub levels: Vec<ClassificationLevel>,
    /// Auto-classification
    pub auto_classification: bool,
}

/// Classification level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationLevel {
    /// Level name
    pub name: String,
    /// Sensitivity score
    pub sensitivity: u8,
    /// Handling requirements
    pub requirements: Vec<String>,
}

/// Data retention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRetentionConfig {
    /// Retention policies
    pub policies: Vec<RetentionPolicy>,
    /// Default retention
    pub default_retention: Duration,
}

/// Retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Data type
    pub data_type: String,
    /// Retention period
    pub retention_period: Duration,
    /// Disposal method
    pub disposal_method: DisposalMethod,
}

/// Disposal methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisposalMethod {
    Delete,
    Archive,
    Anonymize,
    Custom(String),
}

/// Data privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPrivacyConfig {
    /// Privacy controls
    pub controls: Vec<PrivacyControl>,
    /// Consent management
    pub consent: ConsentManagementConfig,
}

/// Privacy controls
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyControl {
    Anonymization,
    Pseudonymization,
    DataMinimization,
    AccessControl,
    Custom(String),
}

/// Consent management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentManagementConfig {
    /// Enable consent management
    pub enabled: bool,
    /// Consent types
    pub consent_types: Vec<String>,
    /// Withdrawal process
    pub withdrawal_process: WithdrawalProcess,
}

/// Withdrawal process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithdrawalProcess {
    /// Methods available
    pub methods: Vec<WithdrawalMethod>,
    /// Processing time
    pub processing_time: Duration,
    /// Confirmation required
    pub confirmation_required: bool,
}

/// Withdrawal methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WithdrawalMethod {
    Online,
    Email,
    Phone,
    Mail,
    Custom(String),
}
