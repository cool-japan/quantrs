use quantrs2_circuit::prelude::Circuit;
#[cfg(feature = "ibm")]
use std::collections::HashMap;
#[cfg(feature = "ibm")]
use std::sync::Arc;
#[cfg(feature = "ibm")]
use std::thread::sleep;
#[cfg(feature = "ibm")]
use std::time::{Duration, Instant, SystemTime};
#[cfg(feature = "ibm")]
use tokio::sync::RwLock;

#[cfg(feature = "ibm")]
use reqwest::{header, Client};
#[cfg(feature = "ibm")]
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::DeviceError;
use crate::DeviceResult;

#[cfg(feature = "ibm")]
const IBM_QUANTUM_API_URL: &str = "https://api.quantum-computing.ibm.com/api";
#[cfg(feature = "ibm")]
const IBM_AUTH_URL: &str = "https://auth.quantum-computing.ibm.com/api";
#[cfg(feature = "ibm")]
const DEFAULT_TIMEOUT_SECS: u64 = 90;
/// Token validity buffer in seconds (refresh 5 minutes before expiry)
#[cfg(feature = "ibm")]
const TOKEN_REFRESH_BUFFER_SECS: u64 = 300;
/// Default token validity period in seconds (1 hour)
#[cfg(feature = "ibm")]
const DEFAULT_TOKEN_VALIDITY_SECS: u64 = 3600;
/// Default maximum retry attempts
#[cfg(feature = "ibm")]
const DEFAULT_MAX_RETRIES: u32 = 3;
/// Default initial retry delay in milliseconds
#[cfg(feature = "ibm")]
const DEFAULT_INITIAL_RETRY_DELAY_MS: u64 = 100;
/// Default maximum retry delay in milliseconds
#[cfg(feature = "ibm")]
const DEFAULT_MAX_RETRY_DELAY_MS: u64 = 30000;
/// Default backoff multiplier
#[cfg(feature = "ibm")]
const DEFAULT_BACKOFF_MULTIPLIER: f64 = 2.0;

/// Retry configuration for IBM Quantum API calls
#[cfg(feature = "ibm")]
#[derive(Debug, Clone)]
pub struct IBMRetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Jitter factor (0.0 to 1.0) to randomize delays
    pub jitter_factor: f64,
}

#[cfg(feature = "ibm")]
impl Default for IBMRetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_MAX_RETRIES,
            initial_delay: Duration::from_millis(DEFAULT_INITIAL_RETRY_DELAY_MS),
            max_delay: Duration::from_millis(DEFAULT_MAX_RETRY_DELAY_MS),
            backoff_multiplier: DEFAULT_BACKOFF_MULTIPLIER,
            jitter_factor: 0.1,
        }
    }
}

#[cfg(feature = "ibm")]
impl IBMRetryConfig {
    /// Create a configuration for aggressive retries (good for transient network errors)
    pub fn aggressive() -> Self {
        Self {
            max_attempts: 5,
            initial_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            jitter_factor: 0.2,
        }
    }

    /// Create a configuration for patient retries (good for rate limiting)
    pub fn patient() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            backoff_multiplier: 3.0,
            jitter_factor: 0.3,
        }
    }
}

/// Token information including expiration tracking
#[cfg(feature = "ibm")]
#[derive(Debug, Clone)]
pub struct TokenInfo {
    /// The access token
    pub access_token: String,
    /// When the token was obtained
    pub obtained_at: Instant,
    /// Token validity period in seconds
    pub valid_for_secs: u64,
}

#[cfg(feature = "ibm")]
impl TokenInfo {
    /// Check if the token is expired or about to expire
    pub fn is_expired(&self) -> bool {
        let elapsed = self.obtained_at.elapsed().as_secs();
        elapsed + TOKEN_REFRESH_BUFFER_SECS >= self.valid_for_secs
    }

    /// Get remaining validity time in seconds
    pub fn remaining_secs(&self) -> u64 {
        let elapsed = self.obtained_at.elapsed().as_secs();
        if elapsed >= self.valid_for_secs {
            0
        } else {
            self.valid_for_secs - elapsed
        }
    }
}

/// Response from IBM Quantum authentication endpoint
#[cfg(feature = "ibm")]
#[derive(Debug, Deserialize)]
struct AuthResponse {
    /// The access token
    id: String,
    /// Token TTL in seconds (if provided)
    ttl: Option<u64>,
}

/// Authentication configuration for IBM Quantum
#[cfg(feature = "ibm")]
#[derive(Debug, Clone)]
pub struct IBMAuthConfig {
    /// The API key (used to obtain access tokens)
    pub api_key: String,
    /// Whether to automatically refresh expired tokens
    pub auto_refresh: bool,
    /// Custom token validity period (if known)
    pub token_validity_secs: Option<u64>,
}

/// Represents the available backends on IBM Quantum
#[derive(Debug, Clone)]
#[cfg_attr(feature = "ibm", derive(serde::Deserialize))]
pub struct IBMBackend {
    /// Unique identifier for the backend
    pub id: String,
    /// Name of the backend
    pub name: String,
    /// Whether the backend is a simulator or real quantum hardware
    pub simulator: bool,
    /// Number of qubits on the backend
    pub n_qubits: usize,
    /// Status of the backend (e.g., "active", "maintenance")
    pub status: String,
    /// Description of the backend
    pub description: String,
    /// Version of the backend
    pub version: String,
}

/// Configuration for a quantum circuit to be submitted to IBM Quantum
#[derive(Debug, Clone)]
#[cfg_attr(feature = "ibm", derive(Serialize))]
pub struct IBMCircuitConfig {
    /// Name of the circuit
    pub name: String,
    /// QASM representation of the circuit
    pub qasm: String,
    /// Number of shots to run
    pub shots: usize,
    /// Optional optimization level (0-3)
    pub optimization_level: Option<usize>,
    /// Optional initial layout mapping
    pub initial_layout: Option<std::collections::HashMap<String, usize>>,
}

/// Status of a job in IBM Quantum
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "ibm", derive(Deserialize))]
pub enum IBMJobStatus {
    #[cfg_attr(feature = "ibm", serde(rename = "CREATING"))]
    Creating,
    #[cfg_attr(feature = "ibm", serde(rename = "CREATED"))]
    Created,
    #[cfg_attr(feature = "ibm", serde(rename = "VALIDATING"))]
    Validating,
    #[cfg_attr(feature = "ibm", serde(rename = "VALIDATED"))]
    Validated,
    #[cfg_attr(feature = "ibm", serde(rename = "QUEUED"))]
    Queued,
    #[cfg_attr(feature = "ibm", serde(rename = "RUNNING"))]
    Running,
    #[cfg_attr(feature = "ibm", serde(rename = "COMPLETED"))]
    Completed,
    #[cfg_attr(feature = "ibm", serde(rename = "CANCELLED"))]
    Cancelled,
    #[cfg_attr(feature = "ibm", serde(rename = "ERROR"))]
    Error,
}

/// Response from submitting a job to IBM Quantum
#[cfg(feature = "ibm")]
#[derive(Debug, Deserialize)]
pub struct IBMJobResponse {
    /// Job ID
    pub id: String,
    /// Status of the job
    pub status: IBMJobStatus,
    /// Number of shots
    pub shots: usize,
    /// Backend used for the job
    pub backend: IBMBackend,
}

#[cfg(not(feature = "ibm"))]
#[derive(Debug)]
pub struct IBMJobResponse {
    /// Job ID
    pub id: String,
    /// Status of the job
    pub status: IBMJobStatus,
    /// Number of shots
    pub shots: usize,
}

/// Results from a completed job
#[cfg(feature = "ibm")]
#[derive(Debug, Deserialize)]
pub struct IBMJobResult {
    /// Counts of each basis state
    pub counts: HashMap<String, usize>,
    /// Total number of shots executed
    pub shots: usize,
    /// Status of the job
    pub status: IBMJobStatus,
    /// Error message, if any
    pub error: Option<String>,
}

#[cfg(not(feature = "ibm"))]
#[derive(Debug)]
pub struct IBMJobResult {
    /// Counts of each basis state
    pub counts: std::collections::HashMap<String, usize>,
    /// Total number of shots executed
    pub shots: usize,
    /// Status of the job
    pub status: IBMJobStatus,
    /// Error message, if any
    pub error: Option<String>,
}

/// Errors specific to IBM Quantum
#[derive(Error, Debug)]
pub enum IBMQuantumError {
    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("API error: {0}")]
    API(String),

    #[error("Backend not available: {0}")]
    BackendUnavailable(String),

    #[error("QASM conversion error: {0}")]
    QasmConversion(String),

    #[error("Job submission error: {0}")]
    JobSubmission(String),

    #[error("Timeout waiting for job completion")]
    Timeout,
}

/// Client for interacting with IBM Quantum
#[cfg(feature = "ibm")]
pub struct IBMQuantumClient {
    /// HTTP client for making API requests
    client: Client,
    /// Base URL for the IBM Quantum API
    api_url: String,
    /// Authentication URL
    auth_url: String,
    /// Current token information (protected by RwLock for thread-safe refresh)
    token_info: Arc<RwLock<TokenInfo>>,
    /// Authentication configuration
    auth_config: IBMAuthConfig,
    /// Retry configuration for API calls
    retry_config: IBMRetryConfig,
}

#[cfg(feature = "ibm")]
impl Clone for IBMQuantumClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            api_url: self.api_url.clone(),
            auth_url: self.auth_url.clone(),
            token_info: Arc::clone(&self.token_info),
            auth_config: self.auth_config.clone(),
            retry_config: self.retry_config.clone(),
        }
    }
}

#[cfg(not(feature = "ibm"))]
#[derive(Clone)]
pub struct IBMQuantumClient;

#[cfg(feature = "ibm")]
impl IBMQuantumClient {
    /// Create a new IBM Quantum client with the given access token (legacy method)
    ///
    /// Note: This method does not support automatic token refresh.
    /// For production use, prefer `new_with_api_key` which supports auto-refresh.
    pub fn new(token: &str) -> DeviceResult<Self> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );

        let client = Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| DeviceError::Connection(e.to_string()))?;

        let token_info = TokenInfo {
            access_token: token.to_string(),
            obtained_at: Instant::now(),
            valid_for_secs: DEFAULT_TOKEN_VALIDITY_SECS,
        };

        Ok(Self {
            client,
            api_url: IBM_QUANTUM_API_URL.to_string(),
            auth_url: IBM_AUTH_URL.to_string(),
            token_info: Arc::new(RwLock::new(token_info)),
            auth_config: IBMAuthConfig {
                api_key: String::new(), // No API key for legacy token-based auth
                auto_refresh: false,
                token_validity_secs: None,
            },
            retry_config: IBMRetryConfig::default(),
        })
    }

    /// Create a new IBM Quantum client with an API key
    ///
    /// This method exchanges the API key for an access token and supports
    /// automatic token refresh when the token expires.
    pub async fn new_with_api_key(api_key: &str) -> DeviceResult<Self> {
        Self::new_with_config(IBMAuthConfig {
            api_key: api_key.to_string(),
            auto_refresh: true,
            token_validity_secs: None,
        })
        .await
    }

    /// Create a new IBM Quantum client with full authentication configuration
    pub async fn new_with_config(config: IBMAuthConfig) -> DeviceResult<Self> {
        Self::new_with_config_and_retry(config, IBMRetryConfig::default()).await
    }

    /// Create a new IBM Quantum client with authentication and retry configuration
    pub async fn new_with_config_and_retry(
        config: IBMAuthConfig,
        retry_config: IBMRetryConfig,
    ) -> DeviceResult<Self> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );

        let client = Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| DeviceError::Connection(e.to_string()))?;

        // Exchange API key for access token
        let token_info = Self::exchange_api_key_for_token(&client, &config.api_key).await?;

        Ok(Self {
            client,
            api_url: IBM_QUANTUM_API_URL.to_string(),
            auth_url: IBM_AUTH_URL.to_string(),
            token_info: Arc::new(RwLock::new(token_info)),
            auth_config: config,
            retry_config,
        })
    }

    /// Set retry configuration
    pub fn set_retry_config(&mut self, config: IBMRetryConfig) {
        self.retry_config = config;
    }

    /// Get current retry configuration
    pub fn retry_config(&self) -> &IBMRetryConfig {
        &self.retry_config
    }

    /// Execute an async operation with exponential backoff retry
    async fn with_retry<F, Fut, T>(&self, operation: F) -> DeviceResult<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = DeviceResult<T>>,
    {
        use scirs2_core::random::prelude::*;

        let mut attempt = 0;
        let mut delay = self.retry_config.initial_delay;

        loop {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(err) => {
                    attempt += 1;

                    // Check if error is retryable
                    let is_retryable = match &err {
                        DeviceError::Connection(_) => true,
                        DeviceError::Timeout(_) => true,
                        DeviceError::APIError(msg) => {
                            msg.contains("rate") || msg.contains("5") || msg.contains("503")
                        }
                        _ => false,
                    };

                    if !is_retryable || attempt >= self.retry_config.max_attempts {
                        return Err(err);
                    }

                    // Calculate delay with jitter
                    let jitter = if self.retry_config.jitter_factor > 0.0 {
                        let mut rng = thread_rng();
                        let jitter_range =
                            delay.as_millis() as f64 * self.retry_config.jitter_factor;
                        Duration::from_millis((rng.gen::<f64>() * jitter_range) as u64)
                    } else {
                        Duration::ZERO
                    };

                    let actual_delay = delay + jitter;
                    tokio::time::sleep(actual_delay).await;

                    // Calculate next delay with exponential backoff
                    delay = Duration::from_millis(
                        (delay.as_millis() as f64 * self.retry_config.backoff_multiplier) as u64,
                    )
                    .min(self.retry_config.max_delay);
                }
            }
        }
    }

    /// Exchange an API key for an access token
    async fn exchange_api_key_for_token(client: &Client, api_key: &str) -> DeviceResult<TokenInfo> {
        let response = client
            .post(&format!("{}/users/loginWithToken", IBM_AUTH_URL))
            .json(&serde_json::json!({ "apiToken": api_key }))
            .send()
            .await
            .map_err(|e| {
                DeviceError::Connection(format!("Authentication request failed: {}", e))
            })?;

        if !response.status().is_success() {
            let error_msg = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown authentication error".to_string());
            return Err(DeviceError::Authentication(error_msg));
        }

        let auth_response: AuthResponse = response.json().await.map_err(|e| {
            DeviceError::Deserialization(format!("Failed to parse auth response: {}", e))
        })?;

        let valid_for_secs = auth_response.ttl.unwrap_or(DEFAULT_TOKEN_VALIDITY_SECS);

        Ok(TokenInfo {
            access_token: auth_response.id,
            obtained_at: Instant::now(),
            valid_for_secs,
        })
    }

    /// Refresh the access token if it's expired or about to expire
    pub async fn refresh_token(&self) -> DeviceResult<()> {
        if self.auth_config.api_key.is_empty() {
            return Err(DeviceError::Authentication(
                "Cannot refresh token: no API key configured. Use new_with_api_key() for auto-refresh support.".to_string()
            ));
        }

        let new_token_info =
            Self::exchange_api_key_for_token(&self.client, &self.auth_config.api_key).await?;

        let mut token_guard = self.token_info.write().await;
        *token_guard = new_token_info;

        Ok(())
    }

    /// Get a valid access token, refreshing if necessary
    async fn get_valid_token(&self) -> DeviceResult<String> {
        // First check if refresh is needed
        let needs_refresh = {
            let token_guard = self.token_info.read().await;
            token_guard.is_expired()
        };

        if needs_refresh && self.auth_config.auto_refresh {
            self.refresh_token().await?;
        }

        let token_guard = self.token_info.read().await;

        // If still expired after refresh attempt (or auto_refresh disabled), warn but continue
        if token_guard.is_expired() && !self.auth_config.auto_refresh {
            // Token is expired but auto-refresh is disabled
            // Let the API call fail and return appropriate error
        }

        Ok(token_guard.access_token.clone())
    }

    /// Check if the current token is valid
    pub async fn is_token_valid(&self) -> bool {
        let token_guard = self.token_info.read().await;
        !token_guard.is_expired()
    }

    /// Get token expiration information
    pub async fn token_info(&self) -> TokenInfo {
        let token_guard = self.token_info.read().await;
        token_guard.clone()
    }

    /// List all available backends with automatic retry
    pub async fn list_backends_with_retry(&self) -> DeviceResult<Vec<IBMBackend>> {
        self.with_retry(|| async { self.list_backends().await })
            .await
    }

    /// List all available backends
    pub async fn list_backends(&self) -> DeviceResult<Vec<IBMBackend>> {
        let token = self.get_valid_token().await?;

        let response = self
            .client
            .get(&format!("{}/backends", self.api_url))
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
            .map_err(|e| DeviceError::Connection(e.to_string()))?;

        if !response.status().is_success() {
            let error_msg = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(DeviceError::APIError(error_msg));
        }

        let backends: Vec<IBMBackend> = response
            .json()
            .await
            .map_err(|e| DeviceError::Deserialization(e.to_string()))?;

        Ok(backends)
    }

    /// Get details about a specific backend
    pub async fn get_backend(&self, backend_name: &str) -> DeviceResult<IBMBackend> {
        let token = self.get_valid_token().await?;

        let response = self
            .client
            .get(&format!("{}/backends/{}", self.api_url, backend_name))
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
            .map_err(|e| DeviceError::Connection(e.to_string()))?;

        if !response.status().is_success() {
            let error_msg = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(DeviceError::APIError(error_msg));
        }

        let backend: IBMBackend = response
            .json()
            .await
            .map_err(|e| DeviceError::Deserialization(e.to_string()))?;

        Ok(backend)
    }

    /// Submit a circuit to be executed on an IBM Quantum backend
    pub async fn submit_circuit(
        &self,
        backend_name: &str,
        config: IBMCircuitConfig,
    ) -> DeviceResult<String> {
        #[cfg(feature = "ibm")]
        {
            use serde_json::json;

            let token = self.get_valid_token().await?;

            let payload = json!({
                "backend": backend_name,
                "name": config.name,
                "qasm": config.qasm,
                "shots": config.shots,
                "optimization_level": config.optimization_level.unwrap_or(1),
                "initial_layout": config.initial_layout.unwrap_or_default(),
            });

            let response = self
                .client
                .post(&format!("{}/jobs", self.api_url))
                .header("Authorization", format!("Bearer {}", token))
                .json(&payload)
                .send()
                .await
                .map_err(|e| DeviceError::Connection(e.to_string()))?;

            if !response.status().is_success() {
                let error_msg = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                return Err(DeviceError::JobSubmission(error_msg));
            }

            let job_response: IBMJobResponse = response
                .json()
                .await
                .map_err(|e| DeviceError::Deserialization(e.to_string()))?;

            Ok(job_response.id)
        }

        #[cfg(not(feature = "ibm"))]
        Err(DeviceError::UnsupportedDevice(
            "IBM Quantum support not enabled".to_string(),
        ))
    }

    /// Get the status of a job
    pub async fn get_job_status(&self, job_id: &str) -> DeviceResult<IBMJobStatus> {
        let token = self.get_valid_token().await?;

        let response = self
            .client
            .get(&format!("{}/jobs/{}", self.api_url, job_id))
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
            .map_err(|e| DeviceError::Connection(e.to_string()))?;

        if !response.status().is_success() {
            let error_msg = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(DeviceError::APIError(error_msg));
        }

        let job: IBMJobResponse = response
            .json()
            .await
            .map_err(|e| DeviceError::Deserialization(e.to_string()))?;

        Ok(job.status)
    }

    /// Get the results of a completed job
    pub async fn get_job_result(&self, job_id: &str) -> DeviceResult<IBMJobResult> {
        let token = self.get_valid_token().await?;

        let response = self
            .client
            .get(&format!("{}/jobs/{}/result", self.api_url, job_id))
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
            .map_err(|e| DeviceError::Connection(e.to_string()))?;

        if !response.status().is_success() {
            let error_msg = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(DeviceError::APIError(error_msg));
        }

        let result: IBMJobResult = response
            .json()
            .await
            .map_err(|e| DeviceError::Deserialization(e.to_string()))?;

        Ok(result)
    }

    /// Wait for a job to complete with timeout
    pub async fn wait_for_job(
        &self,
        job_id: &str,
        timeout_secs: Option<u64>,
    ) -> DeviceResult<IBMJobResult> {
        let timeout = timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS);
        let mut elapsed = 0;
        let interval = 5; // Check status every 5 seconds

        while elapsed < timeout {
            let status = self.get_job_status(job_id).await?;

            match status {
                IBMJobStatus::Completed => {
                    return self.get_job_result(job_id).await;
                }
                IBMJobStatus::Error => {
                    return Err(DeviceError::JobExecution(format!(
                        "Job {} encountered an error",
                        job_id
                    )));
                }
                IBMJobStatus::Cancelled => {
                    return Err(DeviceError::JobExecution(format!(
                        "Job {} was cancelled",
                        job_id
                    )));
                }
                _ => {
                    // Still in progress, wait and check again
                    sleep(Duration::from_secs(interval));
                    elapsed += interval;
                }
            }
        }

        Err(DeviceError::Timeout(format!(
            "Timed out waiting for job {} to complete",
            job_id
        )))
    }

    /// Submit multiple circuits in parallel
    pub async fn submit_circuits_parallel(
        &self,
        backend_name: &str,
        configs: Vec<IBMCircuitConfig>,
    ) -> DeviceResult<Vec<String>> {
        #[cfg(feature = "ibm")]
        {
            use tokio::task;

            let client = Arc::new(self.clone());

            let mut handles = vec![];

            for config in configs {
                let client_clone = client.clone();
                let backend_name = backend_name.to_string();

                let handle =
                    task::spawn(
                        async move { client_clone.submit_circuit(&backend_name, config).await },
                    );

                handles.push(handle);
            }

            let mut job_ids = vec![];

            for handle in handles {
                match handle.await {
                    Ok(result) => match result {
                        Ok(job_id) => job_ids.push(job_id),
                        Err(e) => return Err(e),
                    },
                    Err(e) => {
                        return Err(DeviceError::JobSubmission(format!(
                            "Failed to join task: {}",
                            e
                        )));
                    }
                }
            }

            Ok(job_ids)
        }

        #[cfg(not(feature = "ibm"))]
        Err(DeviceError::UnsupportedDevice(
            "IBM Quantum support not enabled".to_string(),
        ))
    }

    /// Convert a Quantrs circuit to QASM
    pub fn circuit_to_qasm<const N: usize>(
        _circuit: &Circuit<N>,
        _initial_layout: Option<std::collections::HashMap<String, usize>>,
    ) -> DeviceResult<String> {
        // This is a placeholder for the actual conversion logic
        // In a complete implementation, this would translate our circuit representation
        // to OpenQASM format compatible with IBM Quantum

        let mut qasm = String::from("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\n");

        // Define the quantum and classical registers
        qasm.push_str(&format!("qreg q[{}];\n", N));
        qasm.push_str(&format!("creg c[{}];\n\n", N));

        // Implement conversion of gates to QASM here
        // For example:
        // - X gate: x q[i];
        // - H gate: h q[i];
        // - CNOT gate: cx q[i], q[j];

        // For now, just return placeholder QASM
        Ok(qasm)
    }
}

#[cfg(not(feature = "ibm"))]
impl IBMQuantumClient {
    pub fn new(_token: &str) -> DeviceResult<Self> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Quantum support not enabled. Recompile with the 'ibm' feature.".to_string(),
        ))
    }

    pub async fn list_backends(&self) -> DeviceResult<Vec<IBMBackend>> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Quantum support not enabled".to_string(),
        ))
    }

    pub async fn get_backend(&self, _backend_name: &str) -> DeviceResult<IBMBackend> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Quantum support not enabled".to_string(),
        ))
    }

    pub async fn submit_circuit(
        &self,
        _backend_name: &str,
        _config: IBMCircuitConfig,
    ) -> DeviceResult<String> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Quantum support not enabled".to_string(),
        ))
    }

    pub async fn get_job_status(&self, _job_id: &str) -> DeviceResult<IBMJobStatus> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Quantum support not enabled".to_string(),
        ))
    }

    pub async fn get_job_result(&self, _job_id: &str) -> DeviceResult<IBMJobResult> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Quantum support not enabled".to_string(),
        ))
    }

    pub async fn wait_for_job(
        &self,
        _job_id: &str,
        _timeout_secs: Option<u64>,
    ) -> DeviceResult<IBMJobResult> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Quantum support not enabled".to_string(),
        ))
    }

    pub async fn submit_circuits_parallel(
        &self,
        _backend_name: &str,
        _configs: Vec<IBMCircuitConfig>,
    ) -> DeviceResult<Vec<String>> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Quantum support not enabled".to_string(),
        ))
    }

    pub fn circuit_to_qasm<const N: usize>(
        _circuit: &Circuit<N>,
        _initial_layout: Option<std::collections::HashMap<String, usize>>,
    ) -> DeviceResult<String> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Quantum support not enabled".to_string(),
        ))
    }
}
