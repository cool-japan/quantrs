//! Unified Error Handling for Quantum Device Providers
//!
//! This module provides a comprehensive, unified error handling system for all quantum
//! device providers, including sophisticated retry logic, error classification, and
//! recovery strategies.

use std::collections::HashMap;
use std::fmt::{self, Display};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::time::sleep;

use crate::DeviceError;

/// Unified error classification system for quantum device operations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    /// Network connectivity issues
    Network,
    /// Authentication and authorization errors
    Authentication,
    /// API rate limiting or quota exceeded
    RateLimit,
    /// Invalid request parameters or circuit
    Validation,
    /// Hardware-specific errors (calibration, decoherence, etc.)
    Hardware,
    /// Service temporarily unavailable
    ServiceUnavailable,
    /// Internal server errors from provider
    ServerError,
    /// Resource not found (backend, job, etc.)
    NotFound,
    /// Operation timed out
    Timeout,
    /// Insufficient permissions or credits
    Insufficient,
    /// Data parsing or serialization errors
    DataFormat,
    /// Unsupported operation for this provider/backend
    Unsupported,
    /// Circuit execution failed
    Execution,
    /// Critical system error requiring manual intervention
    Critical,
}

/// Error severity levels for prioritization and alerting
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ErrorSeverity {
    /// Informational - operation can continue
    Info,
    /// Warning - might affect performance but operation can continue
    Warning,
    /// Error - operation failed but system is stable
    Error,
    /// Critical - system stability affected, immediate attention required
    Critical,
}

/// Recovery strategies for different error types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Retry the operation immediately
    RetryImmediate,
    /// Retry with exponential backoff
    RetryWithBackoff {
        initial_delay: Duration,
        max_delay: Duration,
        multiplier: f64,
        max_attempts: u32,
    },
    /// Switch to alternative provider/backend
    Fallback { alternatives: Vec<String> },
    /// Wait for a specific condition before retrying
    WaitAndRetry {
        wait_duration: Duration,
        condition: String,
    },
    /// Circuit modification may resolve the issue
    CircuitModification { suggestions: Vec<String> },
    /// Manual intervention required
    ManualIntervention {
        instructions: String,
        contact: String,
    },
    /// Operation should be abandoned
    Abort,
}

/// Comprehensive error context with provider-specific details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedErrorContext {
    /// Error category for classification
    pub category: ErrorCategory,
    /// Severity level for prioritization
    pub severity: ErrorSeverity,
    /// Provider that generated this error
    pub provider: String,
    /// Specific backend/device if applicable
    pub backend: Option<String>,
    /// Original error code from provider
    pub error_code: Option<String>,
    /// Human-readable error message
    pub message: String,
    /// Additional context and details
    pub details: HashMap<String, String>,
    /// Timestamp when error occurred
    pub timestamp: std::time::SystemTime,
    /// Suggested recovery strategy
    pub recovery_strategy: RecoveryStrategy,
    /// Whether this error can be retried
    pub retryable: bool,
    /// Request ID for tracking
    pub request_id: Option<String>,
    /// User-friendly explanation of what went wrong
    pub user_message: String,
    /// Actionable steps the user can take
    pub suggested_actions: Vec<String>,
}

/// Enhanced device error with unified context
#[derive(Error, Debug, Clone)]
pub struct UnifiedDeviceError {
    /// The underlying device error
    pub device_error: DeviceError,
    /// Rich error context
    pub context: UnifiedErrorContext,
    /// Chain of related errors (for error propagation)
    pub error_chain: Vec<UnifiedErrorContext>,
}

impl Display for UnifiedDeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}:{}] {} - {}",
            self.context.provider,
            self.context.category.to_string(),
            self.context.message,
            self.context.user_message
        )
    }
}

/// Retry configuration for different error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedRetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Initial delay between attempts
    pub initial_delay: Duration,
    /// Maximum delay between attempts
    pub max_delay: Duration,
    /// Backoff multiplier (exponential backoff)
    pub backoff_multiplier: f64,
    /// Jitter factor to avoid thundering herd (0.0 to 1.0)
    pub jitter_factor: f64,
    /// Timeout for individual retry attempts
    pub attempt_timeout: Duration,
}

impl Default for UnifiedRetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
            attempt_timeout: Duration::from_secs(60),
        }
    }
}

/// Comprehensive error handling and recovery system
pub struct UnifiedErrorHandler {
    /// Provider-specific error mappings
    error_mappings: HashMap<String, HashMap<String, ErrorCategory>>,
    /// Retry configurations by error category
    retry_configs: HashMap<ErrorCategory, UnifiedRetryConfig>,
    /// Circuit modification suggestions
    circuit_suggestions: HashMap<ErrorCategory, Vec<String>>,
    /// Provider fallback chains
    fallback_chains: HashMap<String, Vec<String>>,
    /// Error statistics for monitoring
    error_stats: HashMap<ErrorCategory, u64>,
}

impl UnifiedErrorHandler {
    /// Create a new unified error handler with default configurations
    pub fn new() -> Self {
        let mut handler = Self {
            error_mappings: HashMap::new(),
            retry_configs: HashMap::new(),
            circuit_suggestions: HashMap::new(),
            fallback_chains: HashMap::new(),
            error_stats: HashMap::new(),
        };

        handler.setup_default_mappings();
        handler.setup_default_retry_configs();
        handler.setup_circuit_suggestions();
        handler.setup_fallback_chains();

        handler
    }

    /// Convert a provider-specific error to a unified error
    pub fn unify_error(
        &mut self,
        provider: &str,
        error: DeviceError,
        request_id: Option<String>,
    ) -> UnifiedDeviceError {
        let category = self.classify_error(provider, &error);
        let severity = self.determine_severity(&category, &error);
        let recovery_strategy = self.determine_recovery_strategy(&category, provider);

        // Update statistics
        *self.error_stats.entry(category.clone()).or_insert(0) += 1;

        let context = UnifiedErrorContext {
            category: category.clone(),
            severity,
            provider: provider.to_string(),
            backend: None, // Can be set by caller
            error_code: self.extract_error_code(&error),
            message: error.to_string(),
            details: self.extract_error_details(&error),
            timestamp: std::time::SystemTime::now(),
            recovery_strategy,
            retryable: self.is_retryable(&category),
            request_id,
            user_message: self.generate_user_message(&category, &error),
            suggested_actions: self.generate_suggested_actions(&category),
        };

        UnifiedDeviceError {
            device_error: error,
            context,
            error_chain: vec![],
        }
    }

    /// Execute an operation with automatic retry and error handling
    pub async fn execute_with_retry<F, T, E>(
        &mut self,
        operation: F,
        category: ErrorCategory,
    ) -> Result<T, UnifiedDeviceError>
    where
        F: Fn() -> Result<T, E>,
        E: Into<DeviceError>,
    {
        let default_config = UnifiedRetryConfig::default();
        let retry_config = self.retry_configs.get(&category).unwrap_or(&default_config);
        let mut delay = retry_config.initial_delay;
        let max_attempts = retry_config.max_attempts;
        let max_delay = retry_config.max_delay;
        let backoff_multiplier = retry_config.backoff_multiplier;
        let jitter_factor = retry_config.jitter_factor;

        for attempt in 1..=max_attempts {
            match operation() {
                Ok(result) => return Ok(result),
                Err(error) => {
                    let device_error = error.into();
                    let unified_error = self.unify_error("unknown", device_error, None);

                    if attempt == max_attempts || !unified_error.context.retryable {
                        return Err(unified_error);
                    }

                    // Apply jitter to delay
                    let jitter = delay.as_millis() as f64 * jitter_factor;
                    let jittered_delay = delay
                        + Duration::from_millis(
                            (0.5 * jitter) as u64, // Use fixed jitter instead of random for now
                        );

                    sleep(jittered_delay).await;

                    // Exponential backoff
                    delay = std::cmp::min(
                        Duration::from_millis(
                            (delay.as_millis() as f64 * backoff_multiplier) as u64,
                        ),
                        max_delay,
                    );
                }
            }
        }

        unreachable!()
    }

    /// Get error statistics for monitoring and alerting
    pub fn get_error_statistics(&self) -> &HashMap<ErrorCategory, u64> {
        &self.error_stats
    }

    /// Clear error statistics
    pub fn clear_statistics(&mut self) {
        self.error_stats.clear();
    }

    /// Set custom retry configuration for a specific error category
    pub fn set_retry_config(&mut self, category: ErrorCategory, config: UnifiedRetryConfig) {
        self.retry_configs.insert(category, config);
    }

    /// Add provider-specific error mapping
    pub fn add_error_mapping(&mut self, provider: &str, error_code: &str, category: ErrorCategory) {
        self.error_mappings
            .entry(provider.to_string())
            .or_insert_with(HashMap::new)
            .insert(error_code.to_string(), category);
    }

    /// Set fallback provider chain
    pub fn set_fallback_chain(&mut self, primary_provider: &str, fallbacks: Vec<String>) {
        self.fallback_chains
            .insert(primary_provider.to_string(), fallbacks);
    }

    // Private helper methods

    fn setup_default_mappings(&mut self) {
        // IBM Quantum error mappings
        let mut ibm_mappings = HashMap::new();
        ibm_mappings.insert(
            "AUTHENTICATION_ERROR".to_string(),
            ErrorCategory::Authentication,
        );
        ibm_mappings.insert("RATE_LIMIT_EXCEEDED".to_string(), ErrorCategory::RateLimit);
        ibm_mappings.insert(
            "BACKEND_NOT_AVAILABLE".to_string(),
            ErrorCategory::ServiceUnavailable,
        );
        ibm_mappings.insert("INVALID_CIRCUIT".to_string(), ErrorCategory::Validation);
        ibm_mappings.insert(
            "INSUFFICIENT_CREDITS".to_string(),
            ErrorCategory::Insufficient,
        );
        self.error_mappings.insert("ibm".to_string(), ibm_mappings);

        // AWS Braket error mappings
        let mut aws_mappings = HashMap::new();
        aws_mappings.insert(
            "AccessDeniedException".to_string(),
            ErrorCategory::Authentication,
        );
        aws_mappings.insert("ThrottlingException".to_string(), ErrorCategory::RateLimit);
        aws_mappings.insert(
            "ServiceUnavailableException".to_string(),
            ErrorCategory::ServiceUnavailable,
        );
        aws_mappings.insert("ValidationException".to_string(), ErrorCategory::Validation);
        aws_mappings.insert(
            "DeviceOfflineException".to_string(),
            ErrorCategory::Hardware,
        );
        self.error_mappings.insert("aws".to_string(), aws_mappings);

        // Azure Quantum error mappings
        let mut azure_mappings = HashMap::new();
        azure_mappings.insert("Unauthorized".to_string(), ErrorCategory::Authentication);
        azure_mappings.insert("TooManyRequests".to_string(), ErrorCategory::RateLimit);
        azure_mappings.insert(
            "ServiceUnavailable".to_string(),
            ErrorCategory::ServiceUnavailable,
        );
        azure_mappings.insert("BadRequest".to_string(), ErrorCategory::Validation);
        azure_mappings.insert("InsufficientQuota".to_string(), ErrorCategory::Insufficient);
        self.error_mappings
            .insert("azure".to_string(), azure_mappings);
    }

    fn setup_default_retry_configs(&mut self) {
        // Network errors - aggressive retry
        self.retry_configs.insert(
            ErrorCategory::Network,
            UnifiedRetryConfig {
                max_attempts: 5,
                initial_delay: Duration::from_millis(50),
                max_delay: Duration::from_secs(10),
                backoff_multiplier: 2.0,
                jitter_factor: 0.2,
                attempt_timeout: Duration::from_secs(30),
            },
        );

        // Rate limiting - gradual backoff
        self.retry_configs.insert(
            ErrorCategory::RateLimit,
            UnifiedRetryConfig {
                max_attempts: 3,
                initial_delay: Duration::from_secs(1),
                max_delay: Duration::from_secs(60),
                backoff_multiplier: 3.0,
                jitter_factor: 0.3,
                attempt_timeout: Duration::from_secs(120),
            },
        );

        // Service unavailable - patient retry
        self.retry_configs.insert(
            ErrorCategory::ServiceUnavailable,
            UnifiedRetryConfig {
                max_attempts: 3,
                initial_delay: Duration::from_secs(5),
                max_delay: Duration::from_secs(120),
                backoff_multiplier: 2.5,
                jitter_factor: 0.4,
                attempt_timeout: Duration::from_secs(180),
            },
        );

        // Server errors - moderate retry
        self.retry_configs.insert(
            ErrorCategory::ServerError,
            UnifiedRetryConfig {
                max_attempts: 3,
                initial_delay: Duration::from_millis(500),
                max_delay: Duration::from_secs(30),
                backoff_multiplier: 2.0,
                jitter_factor: 0.15,
                attempt_timeout: Duration::from_secs(60),
            },
        );
    }

    fn setup_circuit_suggestions(&mut self) {
        self.circuit_suggestions.insert(
            ErrorCategory::Hardware,
            vec![
                "Reduce circuit depth to minimize decoherence effects".to_string(),
                "Add error mitigation techniques like ZNE or PEC".to_string(),
                "Use native gates for the target hardware".to_string(),
                "Implement dynamical decoupling sequences".to_string(),
            ],
        );

        self.circuit_suggestions.insert(
            ErrorCategory::Validation,
            vec![
                "Check circuit connectivity matches hardware topology".to_string(),
                "Verify all gates are supported by the target backend".to_string(),
                "Ensure qubit indices are within hardware limits".to_string(),
                "Add necessary SWAP gates for qubit routing".to_string(),
            ],
        );
    }

    fn setup_fallback_chains(&mut self) {
        self.fallback_chains.insert(
            "ibm".to_string(),
            vec!["aws".to_string(), "azure".to_string()],
        );
        self.fallback_chains.insert(
            "aws".to_string(),
            vec!["ibm".to_string(), "azure".to_string()],
        );
        self.fallback_chains.insert(
            "azure".to_string(),
            vec!["ibm".to_string(), "aws".to_string()],
        );
    }

    fn classify_error(&self, provider: &str, error: &DeviceError) -> ErrorCategory {
        // Try provider-specific mapping first
        if let Some(provider_mappings) = self.error_mappings.get(provider) {
            let error_string = error.to_string();
            for (error_code, category) in provider_mappings {
                if error_string.contains(error_code) {
                    return category.clone();
                }
            }
        }

        // Fallback to general classification
        match error {
            DeviceError::ExecutionFailed(_) => ErrorCategory::Execution,
            DeviceError::Connection(_) => ErrorCategory::Network,
            DeviceError::Authentication(_) => ErrorCategory::Authentication,
            DeviceError::APIError(msg) => {
                if msg.contains("rate limit") || msg.contains("quota") {
                    ErrorCategory::RateLimit
                } else if msg.contains("unavailable") || msg.contains("maintenance") {
                    ErrorCategory::ServiceUnavailable
                } else if msg.contains("timeout") {
                    ErrorCategory::Timeout
                } else {
                    ErrorCategory::ServerError
                }
            }
            DeviceError::Deserialization(_) => ErrorCategory::DataFormat,
            DeviceError::UnsupportedDevice(_) => ErrorCategory::Unsupported,
            DeviceError::UnsupportedOperation(_) => ErrorCategory::Unsupported,
            DeviceError::InvalidInput(_) => ErrorCategory::Validation,
            DeviceError::CircuitConversion(_) => ErrorCategory::Validation,
            DeviceError::InsufficientQubits { .. } => ErrorCategory::Hardware,
            DeviceError::RoutingError(_) => ErrorCategory::Hardware,
            DeviceError::OptimizationError(_) => ErrorCategory::ServerError,
            DeviceError::GraphAnalysisError(_) => ErrorCategory::ServerError,
            DeviceError::NotImplemented(_) => ErrorCategory::Unsupported,
            DeviceError::InvalidMapping(_) => ErrorCategory::Validation,
            DeviceError::DeviceNotFound(_) => ErrorCategory::NotFound,
            DeviceError::JobSubmission(_) => ErrorCategory::ServerError,
            DeviceError::JobExecution(_) => ErrorCategory::Hardware,
            DeviceError::Timeout(_) => ErrorCategory::Timeout,
            DeviceError::DeviceNotInitialized(_) => ErrorCategory::Hardware,
            DeviceError::JobExecutionFailed(_) => ErrorCategory::Hardware,
            DeviceError::InvalidResponse(_) => ErrorCategory::DataFormat,
            DeviceError::UnknownJobStatus(_) => ErrorCategory::ServerError,
        }
    }

    fn determine_severity(&self, category: &ErrorCategory, _error: &DeviceError) -> ErrorSeverity {
        match category {
            ErrorCategory::Critical => ErrorSeverity::Critical,
            ErrorCategory::Authentication
            | ErrorCategory::Hardware
            | ErrorCategory::ServiceUnavailable => ErrorSeverity::Error,
            ErrorCategory::RateLimit | ErrorCategory::Timeout | ErrorCategory::Network => {
                ErrorSeverity::Warning
            }
            _ => ErrorSeverity::Info,
        }
    }

    fn determine_recovery_strategy(
        &self,
        category: &ErrorCategory,
        provider: &str,
    ) -> RecoveryStrategy {
        match category {
            ErrorCategory::Network | ErrorCategory::Timeout => RecoveryStrategy::RetryWithBackoff {
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(10),
                multiplier: 2.0,
                max_attempts: 5,
            },
            ErrorCategory::RateLimit => RecoveryStrategy::WaitAndRetry {
                wait_duration: Duration::from_secs(60),
                condition: "Rate limit reset".to_string(),
            },
            ErrorCategory::ServiceUnavailable => {
                if let Some(fallbacks) = self.fallback_chains.get(provider) {
                    RecoveryStrategy::Fallback {
                        alternatives: fallbacks.clone(),
                    }
                } else {
                    RecoveryStrategy::WaitAndRetry {
                        wait_duration: Duration::from_secs(300),
                        condition: "Service restoration".to_string(),
                    }
                }
            }
            ErrorCategory::Validation | ErrorCategory::Hardware => {
                if let Some(suggestions) = self.circuit_suggestions.get(category) {
                    RecoveryStrategy::CircuitModification {
                        suggestions: suggestions.clone(),
                    }
                } else {
                    RecoveryStrategy::Abort
                }
            }
            ErrorCategory::Authentication | ErrorCategory::Insufficient => {
                RecoveryStrategy::ManualIntervention {
                    instructions: "Check credentials and account status".to_string(),
                    contact: "support@quantumcloud.com".to_string(),
                }
            }
            _ => RecoveryStrategy::Abort,
        }
    }

    fn is_retryable(&self, category: &ErrorCategory) -> bool {
        matches!(
            category,
            ErrorCategory::Network
                | ErrorCategory::RateLimit
                | ErrorCategory::ServiceUnavailable
                | ErrorCategory::ServerError
                | ErrorCategory::Timeout
        )
    }

    fn extract_error_code(&self, error: &DeviceError) -> Option<String> {
        // Extract structured error codes if available
        match error {
            DeviceError::APIError(msg) => {
                // Try to extract error codes from common formats
                if let Some(start) = msg.find("Error:") {
                    if let Some(end) = msg[start..].find(" ") {
                        return Some(msg[start + 6..start + end].to_string());
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn extract_error_details(&self, error: &DeviceError) -> HashMap<String, String> {
        let mut details = HashMap::new();
        details.insert("error_type".to_string(), format!("{:?}", error));
        details.insert("error_message".to_string(), error.to_string());
        details
    }

    fn generate_user_message(&self, category: &ErrorCategory, _error: &DeviceError) -> String {
        match category {
            ErrorCategory::Network => {
                "Network connectivity issue. Please check your internet connection.".to_string()
            }
            ErrorCategory::Authentication => {
                "Authentication failed. Please verify your credentials.".to_string()
            }
            ErrorCategory::RateLimit => {
                "Rate limit exceeded. Please wait before making more requests.".to_string()
            }
            ErrorCategory::Validation => {
                "Invalid request. Please check your circuit and parameters.".to_string()
            }
            ErrorCategory::Hardware => {
                "Hardware issue detected. The quantum device may be calibrating.".to_string()
            }
            ErrorCategory::ServiceUnavailable => {
                "Quantum service temporarily unavailable. Please try again later.".to_string()
            }
            ErrorCategory::ServerError => {
                "Internal server error. The issue has been reported to our team.".to_string()
            }
            ErrorCategory::NotFound => {
                "Requested resource not found. Please check the identifier.".to_string()
            }
            ErrorCategory::Timeout => {
                "Operation timed out. Please try again or reduce complexity.".to_string()
            }
            ErrorCategory::Insufficient => {
                "Insufficient resources or credits. Please check your account.".to_string()
            }
            ErrorCategory::DataFormat => {
                "Data format error. Please check your input data.".to_string()
            }
            ErrorCategory::Unsupported => "Operation not supported on this platform.".to_string(),
            ErrorCategory::Execution => {
                "Circuit execution failed. Please check your circuit and try again.".to_string()
            }
            ErrorCategory::Critical => {
                "Critical system error. Please contact support immediately.".to_string()
            }
        }
    }

    fn generate_suggested_actions(&self, category: &ErrorCategory) -> Vec<String> {
        match category {
            ErrorCategory::Network => vec![
                "Check internet connectivity".to_string(),
                "Try again in a few moments".to_string(),
                "Switch to a different network".to_string(),
            ],
            ErrorCategory::Authentication => vec![
                "Verify API credentials".to_string(),
                "Check token expiration".to_string(),
                "Refresh authentication".to_string(),
            ],
            ErrorCategory::RateLimit => vec![
                "Wait before retrying".to_string(),
                "Reduce request frequency".to_string(),
                "Implement request batching".to_string(),
            ],
            ErrorCategory::Validation => vec![
                "Review circuit structure".to_string(),
                "Check parameter ranges".to_string(),
                "Validate against backend specifications".to_string(),
            ],
            ErrorCategory::Hardware => vec![
                "Try a different backend".to_string(),
                "Reduce circuit complexity".to_string(),
                "Wait for calibration to complete".to_string(),
            ],
            _ => vec![
                "Try again later".to_string(),
                "Contact support if issue persists".to_string(),
            ],
        }
    }
}

impl Default for UnifiedErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let display = match self {
            ErrorCategory::Network => "Network",
            ErrorCategory::Authentication => "Authentication",
            ErrorCategory::RateLimit => "RateLimit",
            ErrorCategory::Validation => "Validation",
            ErrorCategory::Hardware => "Hardware",
            ErrorCategory::ServiceUnavailable => "ServiceUnavailable",
            ErrorCategory::ServerError => "ServerError",
            ErrorCategory::NotFound => "NotFound",
            ErrorCategory::Timeout => "Timeout",
            ErrorCategory::Insufficient => "Insufficient",
            ErrorCategory::DataFormat => "DataFormat",
            ErrorCategory::Unsupported => "Unsupported",
            ErrorCategory::Execution => "Execution",
            ErrorCategory::Critical => "Critical",
        };
        write!(f, "{}", display)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_handler_creation() {
        let handler = UnifiedErrorHandler::new();
        assert!(!handler.error_mappings.is_empty());
        assert!(!handler.retry_configs.is_empty());
    }

    #[test]
    fn test_error_classification() {
        let mut handler = UnifiedErrorHandler::new();
        let error = DeviceError::Connection("Network timeout".to_string());
        let unified_error = handler.unify_error("ibm", error, None);

        assert_eq!(unified_error.context.category, ErrorCategory::Network);
        assert_eq!(unified_error.context.provider, "ibm");
        assert!(unified_error.context.retryable);
    }

    #[test]
    fn test_retry_config() {
        let config = UnifiedRetryConfig::default();
        assert_eq!(config.max_attempts, 3);
        assert!(config.initial_delay > Duration::ZERO);
    }

    #[test]
    fn test_error_statistics() {
        let mut handler = UnifiedErrorHandler::new();
        let error = DeviceError::APIError("Test error".to_string());
        handler.unify_error("test", error, None);

        let stats = handler.get_error_statistics();
        assert!(stats.len() > 0);
    }
}
