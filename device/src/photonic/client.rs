//! Photonic quantum computing client implementation
//!
//! This module provides client connectivity for photonic quantum computers,
//! supporting various photonic platforms and hardware providers.

use crate::{DeviceError, DeviceResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::timeout;

/// Client for photonic quantum computing systems
#[derive(Debug, Clone)]
pub struct PhotonicClient {
    /// Base URL for the photonic service
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

impl PhotonicClient {
    /// Create a new photonic client
    pub fn new(base_url: String, auth_token: String) -> DeviceResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| DeviceError::Connection(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            base_url,
            auth_token,
            client,
            timeout: Duration::from_secs(300),
            headers: HashMap::new(),
        })
    }

    /// Create a new photonic client with custom configuration
    pub fn with_config(
        base_url: String,
        auth_token: String,
        timeout_secs: u64,
        headers: HashMap<String, String>,
    ) -> DeviceResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| DeviceError::Connection(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            base_url,
            auth_token,
            client,
            timeout: Duration::from_secs(timeout_secs),
            headers,
        })
    }

    /// Get available photonic devices
    pub async fn get_devices(&self) -> DeviceResult<Vec<PhotonicDeviceInfo>> {
        let url = format!("{}/devices", self.base_url);
        let response = timeout(self.timeout, self.get_request(&url)).await
            .map_err(|_| DeviceError::Timeout("Request timed out".to_string()))?
            .map_err(|e| DeviceError::APIError(format!("Failed to get devices: {}", e)))?;

        response.json::<Vec<PhotonicDeviceInfo>>().await
            .map_err(|e| DeviceError::Deserialization(format!("Failed to parse devices: {}", e)))
    }

    /// Get device information by ID
    pub async fn get_device(&self, device_id: &str) -> DeviceResult<PhotonicDeviceInfo> {
        let url = format!("{}/devices/{}", self.base_url, device_id);
        let response = timeout(self.timeout, self.get_request(&url)).await
            .map_err(|_| DeviceError::Timeout("Request timed out".to_string()))?
            .map_err(|e| DeviceError::APIError(format!("Failed to get device: {}", e)))?;

        response.json::<PhotonicDeviceInfo>().await
            .map_err(|e| DeviceError::Deserialization(format!("Failed to parse device: {}", e)))
    }

    /// Submit a job to a photonic device
    pub async fn submit_job(&self, job_request: &PhotonicJobRequest) -> DeviceResult<String> {
        let url = format!("{}/jobs", self.base_url);
        let response = timeout(self.timeout, self.post_request(&url, job_request)).await
            .map_err(|_| DeviceError::Timeout("Request timed out".to_string()))?
            .map_err(|e| DeviceError::JobSubmission(format!("Failed to submit job: {}", e)))?;

        let job_response: PhotonicJobResponse = response.json().await
            .map_err(|e| DeviceError::Deserialization(format!("Failed to parse job response: {}", e)))?;

        Ok(job_response.job_id)
    }

    /// Get job status
    pub async fn get_job_status(&self, job_id: &str) -> DeviceResult<PhotonicJobStatus> {
        let url = format!("{}/jobs/{}", self.base_url, job_id);
        let response = timeout(self.timeout, self.get_request(&url)).await
            .map_err(|_| DeviceError::Timeout("Request timed out".to_string()))?
            .map_err(|e| DeviceError::APIError(format!("Failed to get job status: {}", e)))?;

        response.json::<PhotonicJobStatus>().await
            .map_err(|e| DeviceError::Deserialization(format!("Failed to parse job status: {}", e)))
    }

    /// Get job results
    pub async fn get_job_results(&self, job_id: &str) -> DeviceResult<PhotonicJobResult> {
        let url = format!("{}/jobs/{}/results", self.base_url, job_id);
        let response = timeout(self.timeout, self.get_request(&url)).await
            .map_err(|_| DeviceError::Timeout("Request timed out".to_string()))?
            .map_err(|e| DeviceError::APIError(format!("Failed to get job results: {}", e)))?;

        response.json::<PhotonicJobResult>().await
            .map_err(|e| DeviceError::Deserialization(format!("Failed to parse job results: {}", e)))
    }

    /// Cancel a job
    pub async fn cancel_job(&self, job_id: &str) -> DeviceResult<()> {
        let url = format!("{}/jobs/{}/cancel", self.base_url, job_id);
        timeout(self.timeout, self.delete_request(&url)).await
            .map_err(|_| DeviceError::Timeout("Request timed out".to_string()))?
            .map_err(|e| DeviceError::APIError(format!("Failed to cancel job: {}", e)))?;

        Ok(())
    }

    /// Perform quadrature measurement
    pub async fn measure_quadratures(
        &self,
        device_id: &str,
        modes: &[usize],
        phases: &[f64],
    ) -> DeviceResult<Vec<(f64, f64)>> {
        let url = format!("{}/devices/{}/measure/quadratures", self.base_url, device_id);
        let request = QuadratureMeasurementRequest {
            modes: modes.to_vec(),
            phases: phases.to_vec(),
        };

        let response = timeout(self.timeout, self.post_request(&url, &request)).await
            .map_err(|_| DeviceError::Timeout("Request timed out".to_string()))?
            .map_err(|e| DeviceError::APIError(format!("Failed to measure quadratures: {}", e)))?;

        let measurement_response: QuadratureMeasurementResponse = response.json().await
            .map_err(|e| DeviceError::Deserialization(format!("Failed to parse measurement: {}", e)))?;

        Ok(measurement_response.quadratures)
    }

    /// Perform photon number measurement
    pub async fn measure_photon_numbers(
        &self,
        device_id: &str,
        modes: &[usize],
    ) -> DeviceResult<Vec<usize>> {
        let url = format!("{}/devices/{}/measure/photons", self.base_url, device_id);
        let request = PhotonMeasurementRequest {
            modes: modes.to_vec(),
        };

        let response = timeout(self.timeout, self.post_request(&url, &request)).await
            .map_err(|_| DeviceError::Timeout("Request timed out".to_string()))?
            .map_err(|e| DeviceError::APIError(format!("Failed to measure photons: {}", e)))?;

        let measurement_response: PhotonMeasurementResponse = response.json().await
            .map_err(|e| DeviceError::Deserialization(format!("Failed to parse measurement: {}", e)))?;

        Ok(measurement_response.photon_numbers)
    }

    /// Perform homodyne detection
    pub async fn homodyne_detection(
        &self,
        device_id: &str,
        mode: usize,
        phase: f64,
        shots: usize,
    ) -> DeviceResult<Vec<f64>> {
        let url = format!("{}/devices/{}/measure/homodyne", self.base_url, device_id);
        let request = HomodyneMeasurementRequest {
            mode,
            phase,
            shots,
        };

        let response = timeout(self.timeout, self.post_request(&url, &request)).await
            .map_err(|_| DeviceError::Timeout("Request timed out".to_string()))?
            .map_err(|e| DeviceError::APIError(format!("Failed to perform homodyne: {}", e)))?;

        let measurement_response: HomodyneMeasurementResponse = response.json().await
            .map_err(|e| DeviceError::Deserialization(format!("Failed to parse homodyne: {}", e)))?;

        Ok(measurement_response.measurements)
    }

    /// Perform heterodyne detection
    pub async fn heterodyne_detection(
        &self,
        device_id: &str,
        mode: usize,
        shots: usize,
    ) -> DeviceResult<Vec<(f64, f64)>> {
        let url = format!("{}/devices/{}/measure/heterodyne", self.base_url, device_id);
        let request = HeterodyneMeasurementRequest {
            mode,
            shots,
        };

        let response = timeout(self.timeout, self.post_request(&url, &request)).await
            .map_err(|_| DeviceError::Timeout("Request timed out".to_string()))?
            .map_err(|e| DeviceError::APIError(format!("Failed to perform heterodyne: {}", e)))?;

        let measurement_response: HeterodyneMeasurementResponse = response.json().await
            .map_err(|e| DeviceError::Deserialization(format!("Failed to parse heterodyne: {}", e)))?;

        Ok(measurement_response.measurements)
    }

    /// Perform GET request
    async fn get_request(&self, url: &str) -> Result<reqwest::Response, reqwest::Error> {
        let mut request = self.client.get(url)
            .bearer_auth(&self.auth_token);

        for (key, value) in &self.headers {
            request = request.header(key, value);
        }

        request.send().await?.error_for_status()
    }

    /// Perform POST request
    async fn post_request<T: Serialize>(&self, url: &str, body: &T) -> Result<reqwest::Response, reqwest::Error> {
        let mut request = self.client.post(url)
            .bearer_auth(&self.auth_token)
            .json(body);

        for (key, value) in &self.headers {
            request = request.header(key, value);
        }

        request.send().await?.error_for_status()
    }

    /// Perform DELETE request
    async fn delete_request(&self, url: &str) -> Result<reqwest::Response, reqwest::Error> {
        let mut request = self.client.delete(url)
            .bearer_auth(&self.auth_token);

        for (key, value) in &self.headers {
            request = request.header(key, value);
        }

        request.send().await?.error_for_status()
    }
}

/// Information about a photonic device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicDeviceInfo {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub system_type: String,
    pub mode_count: usize,
    pub cutoff_dimension: Option<usize>,
    pub squeezing_range: Option<(f64, f64)>,
    pub loss_rate: f64,
    pub detection_efficiency: f64,
    pub gate_fidelity: f64,
    pub is_available: bool,
    pub queue_length: usize,
    pub estimated_wait_time: Option<Duration>,
    pub capabilities: Vec<String>,
    pub properties: HashMap<String, String>,
}

/// Request to submit a job to a photonic device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicJobRequest {
    pub device_id: String,
    pub circuit: String, // Serialized circuit
    pub shots: usize,
    pub config: Option<HashMap<String, serde_json::Value>>,
    pub priority: Option<String>,
    pub tags: Option<HashMap<String, String>>,
}

/// Response from job submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicJobResponse {
    pub job_id: String,
    pub status: String,
    pub estimated_execution_time: Option<Duration>,
    pub queue_position: Option<usize>,
}

/// Status of a photonic job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicJobStatus {
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

/// Results from a photonic job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicJobResult {
    pub job_id: String,
    pub device_id: String,
    pub status: String,
    pub results: HashMap<String, serde_json::Value>,
    pub metadata: HashMap<String, String>,
    pub execution_time: Duration,
    pub shots_completed: usize,
    pub fidelity_estimate: Option<f64>,
}

/// Quadrature measurement request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadratureMeasurementRequest {
    pub modes: Vec<usize>,
    pub phases: Vec<f64>,
}

/// Quadrature measurement response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadratureMeasurementResponse {
    pub quadratures: Vec<(f64, f64)>,
}

/// Photon number measurement request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonMeasurementRequest {
    pub modes: Vec<usize>,
}

/// Photon number measurement response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonMeasurementResponse {
    pub photon_numbers: Vec<usize>,
}

/// Homodyne measurement request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomodyneMeasurementRequest {
    pub mode: usize,
    pub phase: f64,
    pub shots: usize,
}

/// Homodyne measurement response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomodyneMeasurementResponse {
    pub measurements: Vec<f64>,
}

/// Heterodyne measurement request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterodyneMeasurementRequest {
    pub mode: usize,
    pub shots: usize,
}

/// Heterodyne measurement response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterodyneMeasurementResponse {
    pub measurements: Vec<(f64, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_photonic_client_creation() {
        let client = PhotonicClient::new(
            "https://api.photonic.example.com".to_string(),
            "test_token".to_string(),
        );
        assert!(client.is_ok());
    }

    #[test]
    fn test_photonic_client_with_config() {
        let mut headers = HashMap::new();
        headers.insert("User-Agent".to_string(), "QuantRS2".to_string());

        let client = PhotonicClient::with_config(
            "https://api.photonic.example.com".to_string(),
            "test_token".to_string(),
            60,
            headers,
        );
        assert!(client.is_ok());
    }

    #[test]
    fn test_photonic_device_info_serialization() {
        let device_info = PhotonicDeviceInfo {
            id: "photonic_1".to_string(),
            name: "Test Photonic Device".to_string(),
            provider: "TestProvider".to_string(),
            system_type: "ContinuousVariable".to_string(),
            mode_count: 8,
            cutoff_dimension: Some(10),
            squeezing_range: Some((-2.0, 2.0)),
            loss_rate: 0.01,
            detection_efficiency: 0.9,
            gate_fidelity: 0.99,
            is_available: true,
            queue_length: 0,
            estimated_wait_time: None,
            capabilities: vec!["cv_operations".to_string()],
            properties: HashMap::new(),
        };

        let serialized = serde_json::to_string(&device_info);
        assert!(serialized.is_ok());

        let deserialized: Result<PhotonicDeviceInfo, _> = serde_json::from_str(&serialized.unwrap());
        assert!(deserialized.is_ok());
    }
}