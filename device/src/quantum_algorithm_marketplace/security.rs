//! Marketplace Security Configuration Types

use serde::{Deserialize, Serialize};

/// Marketplace security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceSecurityConfig {
    /// Enable security features
    pub enable_security: bool,
    /// Access control configuration
    pub access_control: AccessControlConfig,
    /// Intellectual property protection
    pub ip_protection: IPProtectionConfig,
    /// Code verification and validation
    pub code_verification: CodeVerificationConfig,
    /// Audit and compliance settings
    pub audit_config: AuditConfig,
    /// Privacy protection
    pub privacy_config: PrivacyConfig,
}

/// Access control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlConfig {
    pub authentication_required: bool,
    pub access_levels: Vec<String>,
}

/// IP protection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IPProtectionConfig {
    pub enable_ip_protection: bool,
    pub protection_methods: Vec<String>,
}

/// Code verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeVerificationConfig {
    pub enable_verification: bool,
    pub verification_methods: Vec<String>,
}

/// Audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    pub enable_auditing: bool,
    pub audit_frequency: u64,
}

/// Privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    pub privacy_level: String,
    pub data_retention_days: u64,
}
