use quantrs2_circuit::prelude::Circuit;
use std::collections::HashMap;
#[cfg(feature = "aws")]
use std::sync::Arc;
#[cfg(feature = "aws")]
use std::thread::sleep;
#[cfg(feature = "aws")]
use std::time::Duration;

#[cfg(feature = "aws")]
use reqwest::{header, Client};
#[cfg(feature = "aws")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "aws")]
use serde_json;
use thiserror::Error;

use crate::DeviceError;
use crate::DeviceResult;

#[cfg(feature = "aws")]
const AWS_BRAKET_API_URL: &str = "https://braket.{region}.amazonaws.com";
#[cfg(feature = "aws")]
const DEFAULT_TIMEOUT_SECS: u64 = 120;
#[cfg(feature = "aws")]
const DEFAULT_REGION: &str = "us-east-1";

/// Represents the available devices on AWS Braket
#[derive(Debug, Clone)]
#[cfg_attr(feature = "aws", derive(serde::Deserialize))]
pub struct AWSDevice {
    /// Device ARN
    pub device_arn: String,
    /// Name of the device
    pub name: String,
    /// Type of device (QPU or SIMULATOR)
    pub device_type: String,
    /// Device provider
    pub provider_name: String,
    /// Status of the device
    pub status: String,
    /// Number of qubits on the device
    pub num_qubits: usize,
    /// Device capabilities
    #[cfg(feature = "aws")]
    pub device_capabilities: serde_json::Value,
    /// Device properties
    #[cfg(not(feature = "aws"))]
    pub device_capabilities: (),
}

/// Configuration for a quantum circuit to be submitted to AWS Braket
#[derive(Debug, Clone)]
#[cfg_attr(feature = "aws", derive(Serialize))]
pub struct AWSCircuitConfig {
    /// Name of the job/task
    pub name: String,
    /// AWS Braket IR (ABIR) representation of the circuit
    pub ir: String,
    /// Type of IR (e.g., "OPENQASM", "BRAKET")
    pub ir_type: String,
    /// Number of shots to run
    pub shots: usize,
    /// AWS S3 bucket for results
    pub s3_bucket: String,
    /// AWS S3 key prefix for results
    pub s3_key_prefix: String,
    /// Device-specific parameters
    #[cfg(feature = "aws")]
    pub device_parameters: Option<serde_json::Value>,
    /// Device-specific parameters
    #[cfg(not(feature = "aws"))]
    pub device_parameters: Option<()>,
}

/// Status of a task in AWS Braket
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "aws", derive(Deserialize))]
pub enum AWSTaskStatus {
    #[cfg_attr(feature = "aws", serde(rename = "CREATED"))]
    Created,
    #[cfg_attr(feature = "aws", serde(rename = "QUEUED"))]
    Queued,
    #[cfg_attr(feature = "aws", serde(rename = "RUNNING"))]
    Running,
    #[cfg_attr(feature = "aws", serde(rename = "COMPLETED"))]
    Completed,
    #[cfg_attr(feature = "aws", serde(rename = "FAILED"))]
    Failed,
    #[cfg_attr(feature = "aws", serde(rename = "CANCELLING"))]
    Cancelling,
    #[cfg_attr(feature = "aws", serde(rename = "CANCELLED"))]
    Cancelled,
}

/// Response from submitting a task to AWS Braket
#[cfg(feature = "aws")]
#[derive(Debug, Deserialize)]
pub struct AWSTaskResponse {
    /// Task ARN
    pub quantum_task_arn: String,
    /// Status of the task
    pub status: AWSTaskStatus,
    /// Creation time
    pub creation_time: String,
    /// Device ARN
    pub device_arn: String,
    /// S3 bucket for results
    pub output_s3_bucket: String,
    /// S3 key prefix for results
    pub output_s3_key_prefix: String,
    /// Shots
    pub shots: usize,
}

#[cfg(not(feature = "aws"))]
#[derive(Debug)]
pub struct AWSTaskResponse {
    /// Task ARN
    pub quantum_task_arn: String,
    /// Status of the task
    pub status: AWSTaskStatus,
}

/// Results from a completed task
#[cfg(feature = "aws")]
#[derive(Debug, Deserialize)]
pub struct AWSTaskResult {
    /// Measurement counts
    pub measurements: HashMap<String, usize>,
    /// Measurement probabilities
    pub measurement_probabilities: HashMap<String, f64>,
    /// Number of shots
    pub shots: usize,
    /// Task metadata
    pub task_metadata: HashMap<String, serde_json::Value>,
    /// Additional results
    pub additional_metadata: HashMap<String, serde_json::Value>,
}

#[cfg(not(feature = "aws"))]
#[derive(Debug)]
pub struct AWSTaskResult {
    /// Measurement counts
    pub measurements: HashMap<String, usize>,
    /// Measurement probabilities
    pub measurement_probabilities: HashMap<String, f64>,
    /// Number of shots
    pub shots: usize,
}

/// Errors specific to AWS Braket
#[derive(Error, Debug)]
pub enum AWSBraketError {
    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("API error: {0}")]
    API(String),

    #[error("Device not available: {0}")]
    DeviceUnavailable(String),

    #[error("Circuit conversion error: {0}")]
    CircuitConversion(String),

    #[error("Task submission error: {0}")]
    TaskSubmission(String),

    #[error("Timeout waiting for task completion")]
    Timeout,

    #[error("S3 error: {0}")]
    S3Error(String),
}

/// Client for interacting with AWS Braket
#[cfg(feature = "aws")]
#[derive(Clone)]
pub struct AWSBraketClient {
    /// HTTP client for making API requests
    client: Client,
    /// Base URL for the AWS Braket API
    api_url: String,
    /// AWS region
    region: String,
    /// AWS access key
    access_key: String,
    /// AWS secret key
    secret_key: String,
    /// AWS S3 bucket for results
    s3_bucket: String,
    /// AWS S3 key prefix for results
    s3_key_prefix: String,
}

#[cfg(not(feature = "aws"))]
#[derive(Clone)]
pub struct AWSBraketClient;

#[cfg(feature = "aws")]
impl AWSBraketClient {
    /// Create a new AWS Braket client with the given credentials
    pub fn new(
        access_key: &str,
        secret_key: &str,
        region: Option<&str>,
        s3_bucket: &str,
        s3_key_prefix: Option<&str>,
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

        let region = region.unwrap_or(DEFAULT_REGION).to_string();
        let api_url = AWS_BRAKET_API_URL.replace("{region}", &region);
        let s3_key_prefix = s3_key_prefix.unwrap_or("quantrs").to_string();

        Ok(Self {
            client,
            api_url,
            region,
            access_key: access_key.to_string(),
            secret_key: secret_key.to_string(),
            s3_bucket: s3_bucket.to_string(),
            s3_key_prefix,
        })
    }

    /// Generate AWS signature for API requests
    fn generate_aws_v4_signature(
        &self,
        request_method: &str,
        path: &str,
        body: &str,
    ) -> reqwest::header::HeaderMap {
        use crate::aws_auth::{AwsRegion, AwsSignatureV4};
        use chrono::Utc;

        let mut headers = reqwest::header::HeaderMap::new();

        // Add required headers
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );

        let host = format!("braket.{}.amazonaws.com", self.region);
        headers.insert(
            reqwest::header::HOST,
            reqwest::header::HeaderValue::from_str(&host).unwrap(),
        );

        let now = Utc::now();
        headers.insert(
            reqwest::header::HeaderName::from_static("x-amz-date"),
            reqwest::header::HeaderValue::from_str(&now.format("%Y%m%dT%H%M%SZ").to_string())
                .unwrap(),
        );

        // Create region information
        let region = AwsRegion {
            name: self.region.clone(),
            service: "braket".to_string(),
        };

        // Sign the request
        AwsSignatureV4::sign_request(
            request_method,
            path,
            "", // No query string
            &mut headers,
            body.as_bytes(),
            &self.access_key,
            &self.secret_key,
            &region,
            &now,
        );

        headers
    }

    /// List all available devices
    pub async fn list_devices(&self) -> DeviceResult<Vec<AWSDevice>> {
        let path = "/devices";
        let url = format!("{}{}", self.api_url, path);
        let body = "{}";

        let headers = self.generate_aws_v4_signature("GET", path, body);

        let mut request = self.client.get(&url);
        for (key, value) in headers.iter() {
            request = request.header(key, value);
        }

        let response = request
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

        let devices: Vec<AWSDevice> = response
            .json()
            .await
            .map_err(|e| DeviceError::Deserialization(e.to_string()))?;

        Ok(devices)
    }

    /// Get details about a specific device
    pub async fn get_device(&self, device_arn: &str) -> DeviceResult<AWSDevice> {
        let path = format!("/device/{}", device_arn);
        let url = format!("{}{}", self.api_url, path);
        let body = "{}";

        let headers = self.generate_aws_v4_signature("GET", &path, body);

        let mut request = self.client.get(&url);
        for (key, value) in headers.iter() {
            request = request.header(key, value);
        }

        let response = request
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

        let device: AWSDevice = response
            .json()
            .await
            .map_err(|e| DeviceError::Deserialization(e.to_string()))?;

        Ok(device)
    }

    /// Submit a circuit to be executed on an AWS Braket device
    pub async fn submit_circuit(
        &self,
        device_arn: &str,
        config: AWSCircuitConfig,
    ) -> DeviceResult<String> {
        let path = "/quantum-task";
        let url = format!("{}{}", self.api_url, path);

        use serde_json::json;

        let payload = json!({
            "action": config.ir,
            "deviceArn": device_arn,
            "shots": config.shots,
            "outputS3Bucket": config.s3_bucket,
            "outputS3KeyPrefix": config.s3_key_prefix,
            "deviceParameters": config.device_parameters,
            "name": config.name,
            "irType": config.ir_type
        });

        let body = payload.to_string();
        let headers = self.generate_aws_v4_signature("POST", path, &body);

        let mut request = self.client.post(&url);
        for (key, value) in headers.iter() {
            request = request.header(key, value);
        }

        let response = request
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

        let task_response: AWSTaskResponse = response
            .json()
            .await
            .map_err(|e| DeviceError::Deserialization(e.to_string()))?;

        Ok(task_response.quantum_task_arn)
    }

    /// Get the status of a task
    pub async fn get_task_status(&self, task_arn: &str) -> DeviceResult<AWSTaskStatus> {
        let path = format!("/quantum-task/{}", task_arn);
        let url = format!("{}{}", self.api_url, path);
        let body = "{}";

        let headers = self.generate_aws_v4_signature("GET", &path, body);

        let mut request = self.client.get(&url);
        for (key, value) in headers.iter() {
            request = request.header(key, value);
        }

        let response = request
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

        let task: AWSTaskResponse = response
            .json()
            .await
            .map_err(|e| DeviceError::Deserialization(e.to_string()))?;

        Ok(task.status)
    }

    /// Get the results of a completed task
    pub async fn get_task_result(&self, task_arn: &str) -> DeviceResult<AWSTaskResult> {
        // For AWS Braket, we need to:
        // 1. Get the task status to confirm it's completed
        // 2. Fetch the results from S3

        let status = self.get_task_status(task_arn).await?;

        if status != AWSTaskStatus::Completed {
            return Err(DeviceError::JobExecution(format!(
                "Task {} is not completed, current status: {:?}",
                task_arn, status
            )));
        }

        // In a real implementation, this would fetch the result from S3
        // For now, this is a placeholder
        // The actual S3 fetching would use the aws-sdk-s3 crate

        let dummy_result = AWSTaskResult {
            measurements: HashMap::new(),
            measurement_probabilities: HashMap::new(),
            shots: 0,
            task_metadata: HashMap::new(),
            additional_metadata: HashMap::new(),
        };

        Ok(dummy_result)
    }

    /// Wait for a task to complete with timeout
    pub async fn wait_for_task(
        &self,
        task_arn: &str,
        timeout_secs: Option<u64>,
    ) -> DeviceResult<AWSTaskResult> {
        let timeout = timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS);
        let mut elapsed = 0;
        let interval = 5; // Check status every 5 seconds

        while elapsed < timeout {
            let status = self.get_task_status(task_arn).await?;

            match status {
                AWSTaskStatus::Completed => {
                    return self.get_task_result(task_arn).await;
                }
                AWSTaskStatus::Failed => {
                    return Err(DeviceError::JobExecution(format!(
                        "Task {} failed",
                        task_arn
                    )));
                }
                AWSTaskStatus::Cancelled => {
                    return Err(DeviceError::JobExecution(format!(
                        "Task {} was cancelled",
                        task_arn
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
            "Timed out waiting for task {} to complete",
            task_arn
        )))
    }

    /// Submit multiple circuits in parallel
    pub async fn submit_circuits_parallel(
        &self,
        device_arn: &str,
        configs: Vec<AWSCircuitConfig>,
    ) -> DeviceResult<Vec<String>> {
        use tokio::task;

        let client = Arc::new(self.clone());

        let mut handles = vec![];

        for config in configs {
            let client_clone = client.clone();
            let device_arn = device_arn.to_string();

            let handle =
                task::spawn(async move { client_clone.submit_circuit(&device_arn, config).await });

            handles.push(handle);
        }

        let mut task_arns = vec![];

        for handle in handles {
            match handle.await {
                Ok(result) => match result {
                    Ok(task_arn) => task_arns.push(task_arn),
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

        Ok(task_arns)
    }

    /// Convert a Quantrs circuit to Braket IR JSON
    pub fn circuit_to_braket_ir<const N: usize>(circuit: &Circuit<N>) -> DeviceResult<String> {
        use crate::aws_conversion;
        aws_conversion::circuit_to_braket_ir(circuit)
    }

    /// Convert a Quantrs circuit to OpenQASM
    pub fn circuit_to_qasm<const N: usize>(circuit: &Circuit<N>) -> DeviceResult<String> {
        use crate::aws_conversion;
        aws_conversion::circuit_to_qasm(circuit)
    }
}

#[cfg(not(feature = "aws"))]
impl AWSBraketClient {
    pub fn new(
        _access_key: &str,
        _secret_key: &str,
        _region: Option<&str>,
        _s3_bucket: &str,
        _s3_key_prefix: Option<&str>,
    ) -> DeviceResult<Self> {
        Err(DeviceError::UnsupportedDevice(
            "AWS Braket support not enabled. Recompile with the 'aws' feature.".to_string(),
        ))
    }

    pub async fn list_devices(&self) -> DeviceResult<Vec<AWSDevice>> {
        Err(DeviceError::UnsupportedDevice(
            "AWS Braket support not enabled".to_string(),
        ))
    }

    pub async fn get_device(&self, _device_arn: &str) -> DeviceResult<AWSDevice> {
        Err(DeviceError::UnsupportedDevice(
            "AWS Braket support not enabled".to_string(),
        ))
    }

    pub async fn submit_circuit(
        &self,
        _device_arn: &str,
        _config: AWSCircuitConfig,
    ) -> DeviceResult<String> {
        Err(DeviceError::UnsupportedDevice(
            "AWS Braket support not enabled".to_string(),
        ))
    }

    pub async fn get_task_status(&self, _task_arn: &str) -> DeviceResult<AWSTaskStatus> {
        Err(DeviceError::UnsupportedDevice(
            "AWS Braket support not enabled".to_string(),
        ))
    }

    pub async fn get_task_result(&self, _task_arn: &str) -> DeviceResult<AWSTaskResult> {
        Err(DeviceError::UnsupportedDevice(
            "AWS Braket support not enabled".to_string(),
        ))
    }

    pub async fn wait_for_task(
        &self,
        _task_arn: &str,
        _timeout_secs: Option<u64>,
    ) -> DeviceResult<AWSTaskResult> {
        Err(DeviceError::UnsupportedDevice(
            "AWS Braket support not enabled".to_string(),
        ))
    }

    pub async fn submit_circuits_parallel(
        &self,
        _device_arn: &str,
        _configs: Vec<AWSCircuitConfig>,
    ) -> DeviceResult<Vec<String>> {
        Err(DeviceError::UnsupportedDevice(
            "AWS Braket support not enabled".to_string(),
        ))
    }

    pub fn circuit_to_braket_ir<const N: usize>(_circuit: &Circuit<N>) -> DeviceResult<String> {
        Err(DeviceError::UnsupportedDevice(
            "AWS Braket support not enabled".to_string(),
        ))
    }

    pub fn circuit_to_qasm<const N: usize>(_circuit: &Circuit<N>) -> DeviceResult<String> {
        Err(DeviceError::UnsupportedDevice(
            "AWS Braket support not enabled".to_string(),
        ))
    }
}
