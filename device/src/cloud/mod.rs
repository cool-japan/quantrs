//! Quantum Cloud Resource Management System
//!
//! This module provides comprehensive cloud resource management for quantum computing across
//! multiple providers (IBM Quantum, AWS Braket, Azure Quantum, Google Quantum AI) with
//! intelligent allocation, cost optimization, multi-provider coordination, and advanced
//! analytics using SciRS2's optimization and machine learning capabilities.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use quantrs2_circuit::prelude::*;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

// SciRS2 dependencies for advanced cloud analytics and optimization
#[cfg(feature = "scirs2")]
use scirs2_linalg::{det, eig, inv, matrix_norm, prelude::*, svd, LinalgError, LinalgResult};
#[cfg(feature = "scirs2")]
use scirs2_optimize::{minimize, OptimizeResult};
use scirs2_stats::ttest::Alternative;
#[cfg(feature = "scirs2")]
use scirs2_stats::{corrcoef, distributions, mean, pearsonr, spearmanr, std, var};

// Fallback implementations when SciRS2 is not available
#[cfg(not(feature = "scirs2"))]
mod fallback_scirs2 {
    use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

    pub fn mean(_data: &ArrayView1<f64>) -> Result<f64, String> {
        Ok(0.0)
    }
    pub fn std(_data: &ArrayView1<f64>, _ddof: i32) -> Result<f64, String> {
        Ok(1.0)
    }
    pub fn pearsonr(
        _x: &ArrayView1<f64>,
        _y: &ArrayView1<f64>,
        _alt: &str,
    ) -> Result<(f64, f64), String> {
        Ok((0.0, 0.5))
    }

    pub struct OptimizeResult {
        pub x: Array1<f64>,
        pub fun: f64,
        pub success: bool,
        pub nit: usize,
        pub nfev: usize,
        pub message: String,
    }

    pub fn minimize(
        _func: fn(&Array1<f64>) -> f64,
        _x0: &Array1<f64>,
        _method: &str,
    ) -> Result<OptimizeResult, String> {
        Ok(OptimizeResult {
            x: Array1::zeros(2),
            fun: 0.0,
            success: true,
            nit: 0,
            nfev: 0,
            message: "Fallback optimization".to_string(),
        })
    }

    pub fn genetic_algorithm(
        _func: fn(&Array1<f64>) -> f64,
        _bounds: &[(f64, f64)],
    ) -> Result<OptimizeResult, String> {
        Ok(OptimizeResult {
            x: Array1::zeros(2),
            fun: 0.0,
            success: true,
            nit: 0,
            nfev: 0,
            message: "Fallback genetic algorithm".to_string(),
        })
    }

    pub fn random_forest(_x: &Array2<f64>, _y: &Array1<f64>) -> Result<String, String> {
        Ok("fallback_model".to_string())
    }
}

#[cfg(not(feature = "scirs2"))]
use fallback_scirs2::*;

#[cfg(feature = "security")]
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, RwLock as TokioRwLock, Semaphore};
use uuid::Uuid;

use crate::{
    backend_traits::{query_backend_capabilities, BackendCapabilities},
    calibration::{CalibrationManager, DeviceCalibration},
    integrated_device_manager::{
        DeviceInfo, IntegratedExecutionResult, IntegratedQuantumDeviceManager,
    },
    job_scheduling::{JobConfig, JobPriority, JobStatus, QuantumJob, QuantumJobScheduler},
    noise_model::CalibrationNoiseModel,
    topology::HardwareTopology,
    CircuitExecutor, CircuitResult, DeviceError, DeviceResult, QuantumDevice,
};

// Module declarations
pub mod allocation;
pub mod cost_estimation;
pub mod cost_management;
pub mod monitoring;
pub mod orchestration;
pub mod provider_migration;
pub mod provider_optimizations;
pub mod providers;

// Re-exports for public API
pub use allocation::*;
pub use cost_estimation::*;
pub use cost_management::*;
pub use monitoring::*;
pub use orchestration::*;
pub use provider_migration::*;
pub use provider_optimizations::*;
pub use providers::*;

// Re-export specific configuration types
pub use orchestration::load_balancing::CloudLoadBalancingConfig;
pub use orchestration::performance::AutoScalingConfig;
pub use orchestration::performance::CloudPerformanceConfig;
pub use orchestration::CloudSecurityConfig;

/// Configuration for Quantum Cloud Resource Management System
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCloudConfig {
    /// Multi-provider configuration
    pub provider_config: MultiProviderConfig,
    /// Resource allocation and optimization
    pub allocation_config: ResourceAllocationConfig,
    /// Cost management and optimization
    pub cost_config: CostManagementConfig,
    /// Performance optimization settings
    pub performance_config: CloudPerformanceConfig,
    /// Load balancing and failover
    pub load_balancing_config: CloudLoadBalancingConfig,
    /// Security and compliance
    pub security_config: CloudSecurityConfig,
    /// Monitoring and analytics
    pub monitoring_config: CloudMonitoringConfig,
    /// Machine learning and prediction
    pub ml_config: CloudMLConfig,
    /// Auto-scaling and elasticity
    pub scaling_config: AutoScalingConfig,
    /// Budget and quota management
    pub budget_config: BudgetConfig,
}

/// Machine learning configuration for cloud optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudMLConfig {
    /// Enable ML-driven optimization
    pub enable_ml_optimization: bool,
    /// ML models for resource optimization
    pub optimization_models: Vec<String>,
    /// Predictive analytics for resource planning
    pub predictive_analytics: bool,
    /// Automated decision making threshold
    pub automated_decision_threshold: f64,
    /// Model training configuration
    pub model_training_enabled: bool,
    /// Feature engineering configuration
    pub feature_engineering_enabled: bool,
}

impl Default for CloudMLConfig {
    fn default() -> Self {
        Self {
            enable_ml_optimization: false,
            optimization_models: vec![],
            predictive_analytics: false,
            automated_decision_threshold: 0.8,
            model_training_enabled: false,
            feature_engineering_enabled: false,
        }
    }
}

/// Unique identifier for a cloud-submitted quantum job.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CloudJobId(pub String);

impl std::fmt::Display for CloudJobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A job submitted to the quantum cloud.
///
/// Wraps a QASM3 source string so the `QuantumCloudManager` is not generic
/// over the circuit qubit count (which would make it impossible to store
/// heterogeneous jobs in the same map).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudJob {
    /// Human-readable name / label.
    pub name: String,
    /// QASM3 source that encodes the quantum program.
    pub qasm_source: String,
    /// Preferred execution provider.  `None` means "choose automatically".
    pub preferred_provider: Option<CloudProvider>,
    /// Number of shots requested.
    pub shots: usize,
    /// Maximum time-to-result the caller is willing to wait.
    pub timeout: std::time::Duration,
    /// Extra metadata (key–value pairs) forwarded to the provider.
    pub metadata: std::collections::HashMap<String, String>,
}

/// Status of a cloud job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CloudJobStatus {
    /// Accepted by the manager, not yet dispatched to a provider.
    Queued,
    /// Sent to the provider and awaiting execution.
    Submitted { provider: CloudProvider },
    /// Currently running on quantum hardware.
    Running { provider: CloudProvider },
    /// Execution finished; results are available.
    Completed { provider: CloudProvider },
    /// Execution failed.
    Failed { reason: String },
    /// Cancelled by the caller.
    Cancelled,
}

/// Summary of estimated execution costs across providers.
#[derive(Debug, Clone)]
pub struct CloudCostEstimate {
    /// Cheapest provider according to the cost model.
    pub recommended_provider: CloudProvider,
    /// Estimated cost in USD.
    pub estimated_usd: f64,
    /// Per-provider breakdown (`provider` → estimated USD).
    pub provider_breakdown: std::collections::HashMap<CloudProvider, f64>,
    /// Confidence level of the estimate (0.0 – 1.0).
    pub confidence: f64,
}

/// Internal job record maintained by `QuantumCloudManager`.
#[derive(Debug, Clone)]
struct JobRecord {
    job: CloudJob,
    status: CloudJobStatus,
    submitted_at: std::time::SystemTime,
    completed_at: Option<std::time::SystemTime>,
}

/// Main cloud resource manager.
///
/// Provides a provider-agnostic entry point for submitting quantum jobs,
/// querying their status, listing available providers, and estimating costs.
///
/// # Thread safety
///
/// Internally uses `std::sync::RwLock` so the manager can be shared across
/// threads without `async` overhead at the call site.
pub struct QuantumCloudManager {
    config: QuantumCloudConfig,
    /// Active and historical job records keyed by `CloudJobId`.
    jobs: std::sync::RwLock<std::collections::HashMap<CloudJobId, JobRecord>>,
}

impl QuantumCloudManager {
    /// Create a new cloud manager with the given configuration.
    pub fn new(config: QuantumCloudConfig) -> DeviceResult<Self> {
        Ok(Self {
            config,
            jobs: std::sync::RwLock::new(std::collections::HashMap::new()),
        })
    }

    /// Submit a quantum job and return its assigned identifier.
    ///
    /// The manager validates the job, selects the best provider according to
    /// the configured selection strategy, and records the submission.
    pub fn submit_job(&self, job: CloudJob) -> DeviceResult<CloudJobId> {
        if job.qasm_source.trim().is_empty() {
            return Err(DeviceError::InvalidInput(
                "CloudJob.qasm_source must not be empty".to_string(),
            ));
        }
        let provider = self.select_provider(&job)?;
        let id = CloudJobId(Uuid::new_v4().to_string());
        let record = JobRecord {
            status: CloudJobStatus::Submitted {
                provider: provider.clone(),
            },
            job,
            submitted_at: std::time::SystemTime::now(),
            completed_at: None,
        };
        self.jobs
            .write()
            .map_err(|e| DeviceError::LockError(format!("jobs write lock: {e}")))?
            .insert(id.clone(), record);
        Ok(id)
    }

    /// Query the current status of a previously submitted job.
    pub fn get_job_status(&self, id: &CloudJobId) -> DeviceResult<CloudJobStatus> {
        let jobs = self
            .jobs
            .read()
            .map_err(|e| DeviceError::LockError(format!("jobs read lock: {e}")))?;
        jobs.get(id)
            .map(|r| r.status.clone())
            .ok_or_else(|| DeviceError::DeviceNotFound(format!("job '{}' not found", id)))
    }

    /// Cancel a queued or submitted job.
    pub fn cancel_job(&self, id: &CloudJobId) -> DeviceResult<()> {
        let mut jobs = self
            .jobs
            .write()
            .map_err(|e| DeviceError::LockError(format!("jobs write lock: {e}")))?;
        let record = jobs
            .get_mut(id)
            .ok_or_else(|| DeviceError::DeviceNotFound(format!("job '{}' not found", id)))?;
        match &record.status {
            CloudJobStatus::Queued | CloudJobStatus::Submitted { .. } => {
                record.status = CloudJobStatus::Cancelled;
                record.completed_at = Some(std::time::SystemTime::now());
                Ok(())
            }
            other => Err(DeviceError::InvalidInput(format!(
                "cannot cancel job '{}' in state {:?}",
                id, other
            ))),
        }
    }

    /// List all configured and enabled cloud providers.
    pub fn list_providers(&self) -> Vec<CloudProvider> {
        self.config.provider_config.enabled_providers.clone()
    }

    /// Estimate the cost of executing `job` across all enabled providers.
    ///
    /// The estimate is based on shot count and a simple per-shot pricing
    /// model.  Real providers typically expose more nuanced pricing APIs;
    /// this implementation serves as a deterministic baseline.
    pub fn estimate_cost(&self, job: &CloudJob) -> DeviceResult<CloudCostEstimate> {
        let providers = self.list_providers();
        if providers.is_empty() {
            return Err(DeviceError::UnsupportedOperation(
                "no cloud providers configured".to_string(),
            ));
        }

        // Simple per-shot cost model (USD per shot).
        // Values are illustrative defaults; a production manager would
        // read these from the per-provider configuration.
        let cost_per_shot = |p: &CloudProvider| -> f64 {
            match p {
                CloudProvider::IBM => 0.000_05,
                CloudProvider::AWS => 0.000_075,
                CloudProvider::Azure => 0.000_065,
                CloudProvider::Google => 0.000_08,
                CloudProvider::IonQ => 0.000_01,
                CloudProvider::Rigetti => 0.000_04,
                CloudProvider::Xanadu => 0.000_03,
                CloudProvider::DWave => 0.000_02,
                CloudProvider::Custom(_) => 0.000_05,
            }
        };

        let shots = job.shots.max(1) as f64;
        let mut breakdown = std::collections::HashMap::new();
        let mut cheapest_provider = providers[0].clone();
        let mut cheapest_cost = f64::MAX;

        for provider in &providers {
            let cost = cost_per_shot(provider) * shots;
            if cost < cheapest_cost {
                cheapest_cost = cost;
                cheapest_provider = provider.clone();
            }
            breakdown.insert(provider.clone(), cost);
        }

        Ok(CloudCostEstimate {
            recommended_provider: cheapest_provider,
            estimated_usd: cheapest_cost,
            provider_breakdown: breakdown,
            confidence: 0.8,
        })
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn select_provider(&self, job: &CloudJob) -> DeviceResult<CloudProvider> {
        // If the caller expressed a preference and the provider is enabled, honour it.
        if let Some(preferred) = &job.preferred_provider {
            if self
                .config
                .provider_config
                .enabled_providers
                .contains(preferred)
            {
                return Ok(preferred.clone());
            }
        }
        // Otherwise pick the cheapest enabled provider.
        let estimate = self.estimate_cost(job)?;
        Ok(estimate.recommended_provider)
    }
}
