//! Integration Configuration Types

use serde::{Deserialize, Serialize};

/// Marketplace integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceIntegrationConfig {
    /// Enable external integrations
    pub enable_integrations: bool,
    /// Cloud provider integrations
    pub cloud_integrations: Vec<CloudIntegration>,
    /// Hardware platform integrations
    pub hardware_integrations: Vec<HardwareIntegration>,
    /// Development tool integrations
    pub development_tools: Vec<DevelopmentToolIntegration>,
    /// API and SDK configuration
    pub api_config: APIConfig,
    /// Workflow integration settings
    pub workflow_integration: WorkflowIntegrationConfig,
}

/// Cloud integration options
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CloudIntegration {
    AWS,
    Azure,
    Google,
    IBM,
    Alibaba,
    Oracle,
    DigitalOcean,
}

/// Hardware integration options
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HardwareIntegration {
    IBMQuantum,
    GoogleQuantumAI,
    IonQ,
    Rigetti,
    Honeywell,
    Oxford,
    PsiQuantum,
    Xanadu,
    Atos,
    Cambridge,
}

/// Development tool integration options
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DevelopmentToolIntegration {
    Jupyter,
    VSCode,
    GitHub,
    GitLab,
    Docker,
    Kubernetes,
    JupyterLab,
    Colab,
    Eclipse,
    IntelliJ,
}

/// API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIConfig {
    pub api_version: String,
    pub rate_limits: Vec<String>,
}

/// Workflow integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowIntegrationConfig {
    pub enable_workflows: bool,
    pub supported_workflows: Vec<String>,
}
