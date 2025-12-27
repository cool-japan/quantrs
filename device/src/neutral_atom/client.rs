//! Neutral atom quantum computing client implementation
//!
//! This module provides client connectivity for neutral atom quantum computers,
//! supporting various neutral atom platforms and hardware providers.

use crate::{DeviceError, DeviceResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::timeout;

/// Client for neutral atom quantum computing systems
#[derive(Debug, Clone)]
pub struct NeutralAtomClient {
    /// Base URL for the neutral atom service
    pub base_url: String,
    /// Authentication token
    pub auth_token: String,
    /// HTTP client for API requests
    pub client: reqwest::Client,
    /// Request timeout
    pub timeout: Duration,
    /// Additional headers for requests
    pub headers: HashMap<String, String>,
}

impl NeutralAtomClient {
    /// Create a new neutral atom client
    pub fn new(base_url: String, auth_token: String) -> DeviceResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| DeviceError::Connection(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            base_url,
            auth_token,
            client,
            timeout: Duration::from_secs(300),
            headers: HashMap::new(),
        })
    }

    /// Create a new neutral atom client with custom configuration
    pub fn with_config(
        base_url: String,
        auth_token: String,
        timeout_secs: u64,
        headers: HashMap<String, String>,
    ) -> DeviceResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| DeviceError::Connection(format!("Failed to create HTTP client: {e}")))?;

        Ok(Self {
            base_url,
            auth_token,
            client,
            timeout: Duration::from_secs(timeout_secs),
            headers,
        })
    }

    /// Get available neutral atom devices
    pub async fn get_devices(&self) -> DeviceResult<Vec<NeutralAtomDeviceInfo>> {
        let url = format!("{}/devices", self.base_url);
        let response = timeout(self.timeout, self.get_request(&url))
            .await
            .map_err(|_| DeviceError::Timeout("Request timed out".to_string()))?
            .map_err(|e| DeviceError::APIError(format!("Failed to get devices: {e}")))?;

        response
            .json::<Vec<NeutralAtomDeviceInfo>>()
            .await
            .map_err(|e| DeviceError::Deserialization(format!("Failed to parse devices: {e}")))
    }

    /// Get device information by ID
    pub async fn get_device(&self, device_id: &str) -> DeviceResult<NeutralAtomDeviceInfo> {
        let url = format!("{}/devices/{}", self.base_url, device_id);
        let response = timeout(self.timeout, self.get_request(&url))
            .await
            .map_err(|_| DeviceError::Timeout("Request timed out".to_string()))?
            .map_err(|e| DeviceError::APIError(format!("Failed to get device: {e}")))?;

        response
            .json::<NeutralAtomDeviceInfo>()
            .await
            .map_err(|e| DeviceError::Deserialization(format!("Failed to parse device: {e}")))
    }

    /// Submit a job to a neutral atom device
    pub async fn submit_job(&self, job_request: &NeutralAtomJobRequest) -> DeviceResult<String> {
        let url = format!("{}/jobs", self.base_url);
        let response = timeout(self.timeout, self.post_request(&url, job_request))
            .await
            .map_err(|_| DeviceError::Timeout("Request timed out".to_string()))?
            .map_err(|e| DeviceError::JobSubmission(format!("Failed to submit job: {e}")))?;

        let job_response: NeutralAtomJobResponse = response.json().await.map_err(|e| {
            DeviceError::Deserialization(format!("Failed to parse job response: {e}"))
        })?;

        Ok(job_response.job_id)
    }

    /// Get job status
    pub async fn get_job_status(&self, job_id: &str) -> DeviceResult<NeutralAtomJobStatus> {
        let url = format!("{}/jobs/{}", self.base_url, job_id);
        let response = timeout(self.timeout, self.get_request(&url))
            .await
            .map_err(|_| DeviceError::Timeout("Request timed out".to_string()))?
            .map_err(|e| DeviceError::APIError(format!("Failed to get job status: {e}")))?;

        response
            .json::<NeutralAtomJobStatus>()
            .await
            .map_err(|e| DeviceError::Deserialization(format!("Failed to parse job status: {e}")))
    }

    /// Get job results
    pub async fn get_job_results(&self, job_id: &str) -> DeviceResult<NeutralAtomJobResult> {
        let url = format!("{}/jobs/{}/results", self.base_url, job_id);
        let response = timeout(self.timeout, self.get_request(&url))
            .await
            .map_err(|_| DeviceError::Timeout("Request timed out".to_string()))?
            .map_err(|e| DeviceError::APIError(format!("Failed to get job results: {e}")))?;

        response
            .json::<NeutralAtomJobResult>()
            .await
            .map_err(|e| DeviceError::Deserialization(format!("Failed to parse job results: {e}")))
    }

    /// Cancel a job
    pub async fn cancel_job(&self, job_id: &str) -> DeviceResult<()> {
        let url = format!("{}/jobs/{}/cancel", self.base_url, job_id);
        timeout(self.timeout, self.delete_request(&url))
            .await
            .map_err(|_| DeviceError::Timeout("Request timed out".to_string()))?
            .map_err(|e| DeviceError::APIError(format!("Failed to cancel job: {e}")))?;

        Ok(())
    }

    /// Perform GET request
    async fn get_request(&self, url: &str) -> Result<reqwest::Response, reqwest::Error> {
        let mut request = self.client.get(url).bearer_auth(&self.auth_token);

        for (key, value) in &self.headers {
            request = request.header(key, value);
        }

        request.send().await?.error_for_status()
    }

    /// Perform POST request
    async fn post_request<T: Serialize>(
        &self,
        url: &str,
        body: &T,
    ) -> Result<reqwest::Response, reqwest::Error> {
        let mut request = self
            .client
            .post(url)
            .bearer_auth(&self.auth_token)
            .json(body);

        for (key, value) in &self.headers {
            request = request.header(key, value);
        }

        request.send().await?.error_for_status()
    }

    /// Perform DELETE request
    async fn delete_request(&self, url: &str) -> Result<reqwest::Response, reqwest::Error> {
        let mut request = self.client.delete(url).bearer_auth(&self.auth_token);

        for (key, value) in &self.headers {
            request = request.header(key, value);
        }

        request.send().await?.error_for_status()
    }
}

/// Information about a neutral atom device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeutralAtomDeviceInfo {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub system_type: String,
    pub atom_count: usize,
    pub atom_spacing: f64,
    pub state_encoding: String,
    pub blockade_radius: Option<f64>,
    pub loading_efficiency: f64,
    pub gate_fidelity: f64,
    pub measurement_fidelity: f64,
    pub is_available: bool,
    pub queue_length: usize,
    pub estimated_wait_time: Option<Duration>,
    pub capabilities: Vec<String>,
    pub properties: HashMap<String, String>,
}

/// Request to submit a job to a neutral atom device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeutralAtomJobRequest {
    pub device_id: String,
    pub circuit: String, // Serialized circuit
    pub shots: usize,
    pub config: Option<HashMap<String, serde_json::Value>>,
    pub priority: Option<String>,
    pub tags: Option<HashMap<String, String>>,
}

/// Response from job submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeutralAtomJobResponse {
    pub job_id: String,
    pub status: String,
    pub estimated_execution_time: Option<Duration>,
    pub queue_position: Option<usize>,
}

/// Status of a neutral atom job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeutralAtomJobStatus {
    pub job_id: String,
    pub status: String,
    pub created_at: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub progress: Option<f64>,
    pub queue_position: Option<usize>,
    pub estimated_completion: Option<String>,
    pub error_message: Option<String>,
}

/// Results from a neutral atom job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeutralAtomJobResult {
    pub job_id: String,
    pub device_id: String,
    pub status: String,
    pub results: HashMap<String, serde_json::Value>,
    pub metadata: HashMap<String, String>,
    pub execution_time: Duration,
    pub shots_completed: usize,
    pub fidelity_estimate: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neutral_atom_client_creation() {
        let client = NeutralAtomClient::new(
            "https://api.neutralatom.example.com".to_string(),
            "test_token".to_string(),
        );
        assert!(client.is_ok());
    }

    #[test]
    fn test_neutral_atom_client_with_config() {
        let mut headers = HashMap::new();
        headers.insert("User-Agent".to_string(), "QuantRS2".to_string());

        let client = NeutralAtomClient::with_config(
            "https://api.neutralatom.example.com".to_string(),
            "test_token".to_string(),
            60,
            headers,
        );
        assert!(client.is_ok());
    }

    #[test]
    fn test_neutral_atom_device_info_serialization() {
        let device_info = NeutralAtomDeviceInfo {
            id: "neutral_atom_1".to_string(),
            name: "Test Neutral Atom Device".to_string(),
            provider: "TestProvider".to_string(),
            system_type: "Rydberg".to_string(),
            atom_count: 100,
            atom_spacing: 5.0,
            state_encoding: "GroundExcited".to_string(),
            blockade_radius: Some(8.0),
            loading_efficiency: 0.95,
            gate_fidelity: 0.995,
            measurement_fidelity: 0.99,
            is_available: true,
            queue_length: 0,
            estimated_wait_time: None,
            capabilities: vec!["rydberg_gates".to_string()],
            properties: HashMap::new(),
        };

        let serialized = serde_json::to_string(&device_info);
        assert!(serialized.is_ok());

        let deserialized: Result<NeutralAtomDeviceInfo, _> =
            serde_json::from_str(&serialized.expect("serialization should succeed"));
        assert!(deserialized.is_ok());
    }
}
