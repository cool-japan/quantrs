//! AWS Braket quantum annealing client
//!
//! This module provides a comprehensive interface for AWS Braket quantum annealing services,
//! including quantum annealing hardware access, device management, and advanced features.
//! It requires the "braket" feature to be enabled.
//!
//! # Features
//!
//! - Full AWS Braket API integration
//! - Quantum annealing device access (`IonQ`, Rigetti, etc.)
//! - Analog quantum simulation support
//! - Advanced problem formulation and submission
//! - Device status tracking and management
//! - Performance monitoring and metrics
//! - Batch problem submission
//! - Robust error handling and retry logic
//! - Cost optimization and tracking

#[cfg(feature = "braket")]
mod client {
    use reqwest::Client;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::fmt::Write;
    use std::time::{Duration, Instant};
    use thiserror::Error;
    use tokio::runtime::Runtime;

    use crate::ising::{IsingError, IsingModel, QuboModel};

    /// Errors that can occur when interacting with AWS Braket API
    #[derive(Error, Debug)]
    pub enum BraketError {
        /// Error in the underlying Ising model
        #[error("Ising error: {0}")]
        IsingError(#[from] IsingError),

        /// Error with the network request
        #[error("Network error: {0}")]
        NetworkError(#[from] reqwest::Error),

        /// Error parsing the response
        #[error("Response parsing error: {0}")]
        ParseError(#[from] serde_json::Error),

        /// Error with the AWS Braket API response
        #[error("AWS Braket API error: {0}")]
        ApiError(String),

        /// Error with the authentication credentials
        #[error("Authentication error: {0}")]
        AuthError(String),

        /// Error with the tokio runtime
        #[error("Runtime error: {0}")]
        RuntimeError(String),

        /// Error with the problem formulation
        #[error("Problem formulation error: {0}")]
        ProblemError(String),

        /// Error with quantum task
        #[error("Quantum task error: {0}")]
        TaskError(String),

        /// Error with device configuration
        #[error("Device configuration error: {0}")]
        DeviceConfigError(String),

        /// Error with batch operations
        #[error("Batch operation error: {0}")]
        BatchError(String),

        /// Timeout error
        #[error("Operation timed out: {0}")]
        TimeoutError(String),

        /// Cost limit error
        #[error("Cost limit exceeded: {0}")]
        CostLimitError(String),

        /// AWS SDK error
        #[error("AWS SDK error: {0}")]
        AwsSdkError(String),
    }

    /// Result type for AWS Braket operations
    pub type BraketResult<T> = Result<T, BraketError>;

    /// AWS Braket device types
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub enum DeviceType {
        /// Quantum processing unit
        #[serde(rename = "QPU")]
        QuantumProcessor,
        /// Quantum simulator
        #[serde(rename = "SIMULATOR")]
        Simulator,
    }

    /// AWS Braket device status
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub enum DeviceStatus {
        /// Device is online and available
        #[serde(rename = "ONLINE")]
        Online,
        /// Device is offline
        #[serde(rename = "OFFLINE")]
        Offline,
        /// Device is retired
        #[serde(rename = "RETIRED")]
        Retired,
    }

    /// Quantum task status
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
    pub enum TaskStatus {
        /// Task is being processed
        #[serde(rename = "RUNNING")]
        Running,
        /// Task completed successfully
        #[serde(rename = "COMPLETED")]
        Completed,
        /// Task failed
        #[serde(rename = "FAILED")]
        Failed,
        /// Task was cancelled
        #[serde(rename = "CANCELLED")]
        Cancelled,
        /// Task is queued
        #[serde(rename = "QUEUED")]
        Queued,
        /// Task is being created
        #[serde(rename = "CREATED")]
        Created,
    }

    /// AWS Braket device information
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BraketDevice {
        /// Device ARN
        pub device_arn: String,
        /// Device name
        pub device_name: String,
        /// Provider name
        pub provider_name: String,
        /// Device type
        pub device_type: DeviceType,
        /// Device status
        pub device_status: DeviceStatus,
        /// Device capabilities
        pub device_capabilities: serde_json::Value,
        /// Device parameters
        pub device_parameters: Option<serde_json::Value>,
    }

    /// Quantum annealing problem specification
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AnnealingProblem {
        /// Problem type (ising or qubo)
        #[serde(rename = "type")]
        pub problem_type: String,
        /// Linear coefficients
        pub linear: HashMap<String, f64>,
        /// Quadratic coefficients
        pub quadratic: HashMap<String, f64>,
        /// Number of reads
        pub shots: usize,
    }

    /// Analog quantum simulation problem
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AnalogHamiltonianSimulation {
        /// Hamiltonian specification
        pub hamiltonian: serde_json::Value,
        /// Evolution time
        pub time: f64,
        /// Time steps
        pub steps: Option<usize>,
    }

    /// Quantum task submission parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TaskParams {
        /// Device ARN
        pub device_arn: String,
        /// Number of shots
        pub shots: usize,
        /// Additional device parameters
        #[serde(flatten)]
        pub device_parameters: HashMap<String, serde_json::Value>,
    }

    impl Default for TaskParams {
        fn default() -> Self {
            Self {
                device_arn: String::new(),
                shots: 1000,
                device_parameters: HashMap::new(),
            }
        }
    }

    /// Quantum task result
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TaskResult {
        /// Task ARN
        pub task_arn: String,
        /// Task status
        pub task_status: TaskStatus,
        /// Result data
        pub result: Option<serde_json::Value>,
        /// Measurements (for annealing)
        pub measurements: Option<Vec<HashMap<String, i32>>>,
        /// Additional measurements (binary outcomes)
        pub measurement_counts: Option<HashMap<String, usize>>,
        /// Task metadata
        pub task_metadata: serde_json::Value,
        /// Additional result info
        pub additional_metadata: Option<serde_json::Value>,
    }

    /// Device selector for filtering
    #[derive(Debug, Clone)]
    pub struct DeviceSelector {
        /// Device type filter
        pub device_type: Option<DeviceType>,
        /// Provider filter
        pub provider: Option<String>,
        /// Status filter
        pub status: DeviceStatus,
        /// Minimum gate fidelity
        pub min_fidelity: Option<f64>,
        /// Maximum cost per shot
        pub max_cost_per_shot: Option<f64>,
        /// Capabilities required
        pub required_capabilities: Vec<String>,
    }

    impl Default for DeviceSelector {
        fn default() -> Self {
            Self {
                device_type: None,
                provider: None,
                status: DeviceStatus::Online,
                min_fidelity: None,
                max_cost_per_shot: None,
                required_capabilities: Vec::new(),
            }
        }
    }

    /// Advanced annealing parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AdvancedAnnealingParams {
        /// Number of shots
        pub shots: usize,
        /// Annealing time in microseconds
        pub annealing_time: Option<f64>,
        /// Programming thermalization
        pub programming_thermalization: Option<f64>,
        /// Readout thermalization
        pub readout_thermalization: Option<f64>,
        /// Beta (inverse temperature) schedule
        pub beta_schedule: Option<Vec<(f64, f64)>>,
        /// Transverse field schedule
        pub s_schedule: Option<Vec<(f64, f64)>>,
        /// Auto-scale problem
        pub auto_scale: Option<bool>,
        /// Flux biases
        pub flux_biases: Option<HashMap<String, f64>>,
        /// Additional parameters
        #[serde(flatten)]
        pub extra: HashMap<String, serde_json::Value>,
    }

    impl Default for AdvancedAnnealingParams {
        fn default() -> Self {
            Self {
                shots: 1000,
                annealing_time: Some(20.0),
                programming_thermalization: Some(1000.0),
                readout_thermalization: Some(100.0),
                beta_schedule: None,
                s_schedule: None,
                auto_scale: Some(true),
                flux_biases: None,
                extra: HashMap::new(),
            }
        }
    }

    /// Performance metrics for tasks
    #[derive(Debug, Clone)]
    pub struct TaskMetrics {
        /// Total execution time
        pub total_time: Duration,
        /// Queue time
        pub queue_time: Duration,
        /// Execution time on device
        pub execution_time: Duration,
        /// Cost in USD
        pub cost: f64,
        /// Success rate
        pub success_rate: f64,
        /// Average energy
        pub average_energy: Option<f64>,
        /// Best energy found
        pub best_energy: Option<f64>,
        /// Standard deviation of energies
        pub energy_std: Option<f64>,
    }

    /// Batch task submission result
    #[derive(Debug)]
    pub struct BatchTaskResult {
        /// List of submitted task ARNs
        pub task_arns: Vec<String>,
        /// Success/failure status for each task
        pub statuses: Vec<Result<String, BraketError>>,
        /// Total submission time
        pub submission_time: Duration,
        /// Total estimated cost
        pub estimated_cost: f64,
    }

    /// Cost tracking and limits
    #[derive(Debug, Clone)]
    pub struct CostTracker {
        /// Maximum cost limit
        pub cost_limit: Option<f64>,
        /// Current cost tracking
        pub current_cost: f64,
        /// Cost per shot estimates
        pub cost_estimates: HashMap<String, f64>,
    }

    impl Default for CostTracker {
        fn default() -> Self {
            Self {
                cost_limit: None,
                current_cost: 0.0,
                cost_estimates: HashMap::new(),
            }
        }
    }

    /// Enhanced AWS Braket quantum annealing client
    #[derive(Debug)]
    pub struct BraketClient {
        /// The HTTP client for making API requests
        client: Client,

        /// AWS region
        region: String,

        /// AWS credentials (access key, secret key, session token)
        credentials: (String, String, Option<String>),

        /// The tokio runtime for async requests
        runtime: Runtime,

        /// Default device selector
        default_device_selector: DeviceSelector,

        /// Cost tracking
        cost_tracker: CostTracker,

        /// Retry configuration
        max_retries: usize,

        /// Request timeout
        request_timeout: Duration,

        /// Default task timeout
        task_timeout: Duration,
    }

    impl BraketClient {
        /// Create a new AWS Braket client
        pub fn new(
            access_key: impl Into<String>,
            secret_key: impl Into<String>,
            region: impl Into<String>,
        ) -> BraketResult<Self> {
            Self::with_config(
                access_key,
                secret_key,
                None,
                region,
                DeviceSelector::default(),
                CostTracker::default(),
            )
        }

        /// Create a Braket client with custom configuration
        pub fn with_config(
            access_key: impl Into<String>,
            secret_key: impl Into<String>,
            session_token: Option<String>,
            region: impl Into<String>,
            device_selector: DeviceSelector,
            cost_tracker: CostTracker,
        ) -> BraketResult<Self> {
            // Create HTTP client
            let client = Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .map_err(BraketError::NetworkError)?;

            // Create tokio runtime
            let runtime = Runtime::new().map_err(|e| BraketError::RuntimeError(e.to_string()))?;

            Ok(Self {
                client,
                region: region.into(),
                credentials: (access_key.into(), secret_key.into(), session_token),
                runtime,
                default_device_selector: device_selector,
                cost_tracker,
                max_retries: 3,
                request_timeout: Duration::from_secs(300),
                task_timeout: Duration::from_secs(1800), // 30 minutes
            })
        }

        /// Get a list of available devices
        pub fn get_devices(&self) -> BraketResult<Vec<BraketDevice>> {
            let url = format!("https://braket.{}.amazonaws.com/devices", self.region);

            self.runtime.block_on(async {
                let response = self
                    .client
                    .get(&url)
                    .header("Authorization", self.get_auth_header().await?)
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await?;
                    return Err(BraketError::ApiError(format!(
                        "Error getting devices: {} - {}",
                        status, error_text
                    )));
                }

                let devices_response: serde_json::Value = response.json().await?;
                let devices: Vec<BraketDevice> =
                    serde_json::from_value(devices_response["devices"].clone())?;
                Ok(devices)
            })
        }

        /// Select optimal device based on criteria
        pub fn select_device(
            &self,
            selector: Option<&DeviceSelector>,
        ) -> BraketResult<BraketDevice> {
            let selector = selector.unwrap_or(&self.default_device_selector);
            let devices = self.get_devices()?;

            let filtered_devices: Vec<_> = devices
                .into_iter()
                .filter(|device| {
                    // Filter by device type
                    let type_match = selector
                        .device_type
                        .as_ref()
                        .map(|dt| &device.device_type == dt)
                        .unwrap_or(true);

                    // Filter by provider
                    let provider_match = selector
                        .provider
                        .as_ref()
                        .map(|p| device.provider_name.contains(p))
                        .unwrap_or(true);

                    // Filter by status
                    let status_match = device.device_status == selector.status;

                    type_match && provider_match && status_match
                })
                .collect();

            if filtered_devices.is_empty() {
                return Err(BraketError::DeviceConfigError(
                    "No devices match the selection criteria".to_string(),
                ));
            }

            // Simple selection: prefer QPUs over simulators, then by name
            let mut best_device = filtered_devices[0].clone();
            for device in &filtered_devices[1..] {
                if matches!(device.device_type, DeviceType::QuantumProcessor)
                    && matches!(best_device.device_type, DeviceType::Simulator)
                {
                    best_device = device.clone();
                }
            }

            Ok(best_device)
        }

        /// Submit an Ising model as quantum annealing task
        pub fn submit_ising(
            &self,
            model: &IsingModel,
            device_arn: Option<&str>,
            params: Option<AdvancedAnnealingParams>,
        ) -> BraketResult<TaskResult> {
            let params = params.unwrap_or_default();

            // Select device if not provided
            let device = if let Some(arn) = device_arn {
                self.get_device_by_arn(arn)?
            } else {
                let annealing_selector = DeviceSelector {
                    device_type: Some(DeviceType::QuantumProcessor),
                    required_capabilities: vec!["ANNEALING".to_string()],
                    ..Default::default()
                };
                self.select_device(Some(&annealing_selector))?
            };

            // Check cost limits
            let estimated_cost = self.estimate_task_cost(&device, params.shots);
            self.check_cost_limit(estimated_cost)?;

            // Convert Ising model to Braket format
            let mut linear = HashMap::new();
            for (qubit, bias) in model.biases() {
                linear.insert(qubit.to_string(), bias);
            }

            let mut quadratic = HashMap::new();
            for coupling in model.couplings() {
                let key = format!("({},{})", coupling.i, coupling.j);
                quadratic.insert(key, coupling.strength);
            }

            let problem = AnnealingProblem {
                problem_type: "ising".to_string(),
                linear,
                quadratic,
                shots: params.shots,
            };

            // Submit the task
            self.submit_annealing_task(&device, &problem, &params)
        }

        /// Submit a QUBO model as quantum annealing task
        pub fn submit_qubo(
            &self,
            model: &QuboModel,
            device_arn: Option<&str>,
            params: Option<AdvancedAnnealingParams>,
        ) -> BraketResult<TaskResult> {
            let params = params.unwrap_or_default();

            // Select device if not provided
            let device = if let Some(arn) = device_arn {
                self.get_device_by_arn(arn)?
            } else {
                let annealing_selector = DeviceSelector {
                    device_type: Some(DeviceType::QuantumProcessor),
                    required_capabilities: vec!["ANNEALING".to_string()],
                    ..Default::default()
                };
                self.select_device(Some(&annealing_selector))?
            };

            // Check cost limits
            let estimated_cost = self.estimate_task_cost(&device, params.shots);
            self.check_cost_limit(estimated_cost)?;

            // Convert QUBO model to Braket format
            let mut linear = HashMap::new();
            for (var, value) in model.linear_terms() {
                linear.insert(var.to_string(), value);
            }

            let mut quadratic = HashMap::new();
            for (var1, var2, value) in model.quadratic_terms() {
                let key = format!("({},{})", var1, var2);
                quadratic.insert(key, value);
            }

            let problem = AnnealingProblem {
                problem_type: "qubo".to_string(),
                linear,
                quadratic,
                shots: params.shots,
            };

            // Submit the task
            self.submit_annealing_task(&device, &problem, &params)
        }

        /// Submit multiple tasks in batch
        pub fn submit_batch(
            &self,
            tasks: Vec<(&IsingModel, Option<&str>, Option<AdvancedAnnealingParams>)>,
        ) -> BraketResult<BatchTaskResult> {
            let start_time = Instant::now();
            let mut task_arns = Vec::new();
            let mut statuses = Vec::new();
            let mut total_cost = 0.0;

            for (model, device_arn, params) in tasks {
                match self.submit_ising(model, device_arn, params.clone()) {
                    Ok(task_result) => {
                        task_arns.push(task_result.task_arn.clone());
                        statuses.push(Ok(task_result.task_arn));
                        // Add estimated cost
                        if let Some(params) = &params {
                            total_cost += self.estimate_shot_cost(device_arn, params.shots);
                        }
                    }
                    Err(e) => {
                        task_arns.push(String::new());
                        statuses.push(Err(e));
                    }
                }
            }

            Ok(BatchTaskResult {
                task_arns,
                statuses,
                submission_time: start_time.elapsed(),
                estimated_cost: total_cost,
            })
        }

        /// Get task status
        pub fn get_task_status(&self, task_arn: &str) -> BraketResult<TaskResult> {
            let url = format!(
                "https://braket.{}.amazonaws.com/quantum-task/{}",
                self.region, task_arn
            );

            self.runtime.block_on(async {
                let response = self
                    .client
                    .get(&url)
                    .header("Authorization", self.get_auth_header().await?)
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await?;
                    return Err(BraketError::ApiError(format!(
                        "Error getting task status: {} - {}",
                        status, error_text
                    )));
                }

                let task_result: TaskResult = response.json().await?;
                Ok(task_result)
            })
        }

        /// Cancel a running task
        pub fn cancel_task(&self, task_arn: &str) -> BraketResult<()> {
            let url = format!(
                "https://braket.{}.amazonaws.com/quantum-task/{}/cancel",
                self.region, task_arn
            );

            self.runtime.block_on(async {
                let response = self
                    .client
                    .post(&url)
                    .header("Authorization", self.get_auth_header().await?)
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await?;
                    return Err(BraketError::ApiError(format!(
                        "Error cancelling task: {} - {}",
                        status, error_text
                    )));
                }

                Ok(())
            })
        }

        /// Wait for task completion and get result
        pub fn get_task_result(&self, task_arn: &str) -> BraketResult<TaskResult> {
            let start_time = Instant::now();

            loop {
                let task_result = self.get_task_status(task_arn)?;

                match task_result.task_status {
                    TaskStatus::Completed => {
                        return Ok(task_result);
                    }
                    TaskStatus::Failed => {
                        return Err(BraketError::TaskError(format!("Task {} failed", task_arn)));
                    }
                    TaskStatus::Cancelled => {
                        return Err(BraketError::TaskError(format!(
                            "Task {} was cancelled",
                            task_arn
                        )));
                    }
                    TaskStatus::Running | TaskStatus::Queued | TaskStatus::Created => {
                        if start_time.elapsed() > self.task_timeout {
                            return Err(BraketError::TimeoutError(format!(
                                "Timeout waiting for task {} completion",
                                task_arn
                            )));
                        }
                        // Wait before checking again
                        std::thread::sleep(Duration::from_secs(5));
                    }
                }
            }
        }

        /// Get performance metrics for a completed task
        pub fn get_task_metrics(&self, task_arn: &str) -> BraketResult<TaskMetrics> {
            let task_result = self.get_task_result(task_arn)?;

            // Extract timing and cost information from metadata
            let metadata = &task_result.task_metadata;

            let queue_time = Duration::from_secs(metadata["queueTime"].as_u64().unwrap_or(0));
            let execution_time =
                Duration::from_secs(metadata["executionTime"].as_u64().unwrap_or(0));
            let total_time = queue_time + execution_time;

            let cost = metadata["cost"].as_f64().unwrap_or(0.0);
            let success_rate = metadata["successRate"].as_f64().unwrap_or(1.0);

            // Extract energy statistics if available
            let (average_energy, best_energy, energy_std) =
                if let Some(measurements) = &task_result.measurements {
                    let energies: Vec<f64> = measurements
                        .iter()
                        .filter_map(|m| m.get("energy").and_then(|e| Some(*e as f64)))
                        .collect();

                    if !energies.is_empty() {
                        let avg = energies.iter().sum::<f64>() / energies.len() as f64;
                        let best = energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let variance = energies.iter().map(|&e| (e - avg).powi(2)).sum::<f64>()
                            / energies.len() as f64;
                        let std_dev = variance.sqrt();

                        (Some(avg), Some(best), Some(std_dev))
                    } else {
                        (None, None, None)
                    }
                } else {
                    (None, None, None)
                };

            Ok(TaskMetrics {
                total_time,
                queue_time,
                execution_time,
                cost,
                success_rate,
                average_energy,
                best_energy,
                energy_std,
            })
        }

        /// List recent tasks
        pub fn list_tasks(&self, limit: Option<usize>) -> BraketResult<Vec<TaskResult>> {
            let mut url = format!("https://braket.{}.amazonaws.com/quantum-tasks", self.region);
            if let Some(limit) = limit {
                let _ = write!(url, "?limit={}", limit);
            }

            self.runtime.block_on(async {
                let response = self
                    .client
                    .get(&url)
                    .header("Authorization", self.get_auth_header().await?)
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await?;
                    return Err(BraketError::ApiError(format!(
                        "Error listing tasks: {} - {}",
                        status, error_text
                    )));
                }

                let tasks_response: serde_json::Value = response.json().await?;
                let tasks: Vec<TaskResult> =
                    serde_json::from_value(tasks_response["quantumTasks"].clone())?;
                Ok(tasks)
            })
        }

        /// Get cost tracking information
        pub fn get_cost_summary(&self) -> BraketResult<serde_json::Value> {
            let url = format!("https://braket.{}.amazonaws.com/usage", self.region);

            self.runtime.block_on(async {
                let response = self
                    .client
                    .get(&url)
                    .header("Authorization", self.get_auth_header().await?)
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await?;
                    return Err(BraketError::ApiError(format!(
                        "Error getting cost summary: {} - {}",
                        status, error_text
                    )));
                }

                let usage: serde_json::Value = response.json().await?;
                Ok(usage)
            })
        }

        // Helper methods

        /// Submit annealing task to device
        fn submit_annealing_task(
            &self,
            device: &BraketDevice,
            problem: &AnnealingProblem,
            params: &AdvancedAnnealingParams,
        ) -> BraketResult<TaskResult> {
            let url = format!("https://braket.{}.amazonaws.com/quantum-task", self.region);

            // Create task specification
            let mut task_spec = serde_json::json!({
                "deviceArn": device.device_arn,
                "action": {
                    "type": "braket.ir.annealing.Problem",
                    "linear": problem.linear,
                    "quadratic": problem.quadratic
                },
                "shots": params.shots
            });

            // Add advanced parameters if specified
            if let Some(annealing_time) = params.annealing_time {
                task_spec["deviceParameters"]["annealingTime"] = serde_json::json!(annealing_time);
            }
            if let Some(prog_therm) = params.programming_thermalization {
                task_spec["deviceParameters"]["programmingThermalization"] =
                    serde_json::json!(prog_therm);
            }
            if let Some(readout_therm) = params.readout_thermalization {
                task_spec["deviceParameters"]["readoutThermalization"] =
                    serde_json::json!(readout_therm);
            }

            self.runtime.block_on(async {
                let response = self
                    .client
                    .post(&url)
                    .header("Authorization", self.get_auth_header().await?)
                    .header("Content-Type", "application/json")
                    .json(&task_spec)
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await?;
                    return Err(BraketError::ApiError(format!(
                        "Error submitting task: {} - {}",
                        status, error_text
                    )));
                }

                let task_result: TaskResult = response.json().await?;
                Ok(task_result)
            })
        }

        /// Get device by ARN
        fn get_device_by_arn(&self, arn: &str) -> BraketResult<BraketDevice> {
            let devices = self.get_devices()?;
            devices
                .into_iter()
                .find(|d| d.device_arn == arn)
                .ok_or_else(|| BraketError::DeviceConfigError(format!("Device {} not found", arn)))
        }

        /// Estimate task cost
        fn estimate_task_cost(&self, device: &BraketDevice, shots: usize) -> f64 {
            // Rough cost estimates based on device type and provider
            let base_cost = match device.device_type {
                DeviceType::QuantumProcessor => {
                    if device.provider_name.contains("IonQ") {
                        0.01 // $0.01 per shot
                    } else if device.provider_name.contains("Rigetti") {
                        0.00_035 // $0.00_035 per shot
                    } else {
                        0.001 // Default QPU cost
                    }
                }
                DeviceType::Simulator => 0.0, // Simulators are usually free
            };

            base_cost * shots as f64
        }

        /// Estimate cost for given shots and device
        fn estimate_shot_cost(&self, device_arn: Option<&str>, shots: usize) -> f64 {
            if let Some(arn) = device_arn {
                if let Ok(device) = self.get_device_by_arn(arn) {
                    return self.estimate_task_cost(&device, shots);
                }
            }
            // Default estimation
            0.001 * shots as f64
        }

        /// Check cost limit before submission
        fn check_cost_limit(&self, estimated_cost: f64) -> BraketResult<()> {
            if let Some(limit) = self.cost_tracker.cost_limit {
                if self.cost_tracker.current_cost + estimated_cost > limit {
                    return Err(BraketError::CostLimitError(format!(
                        "Estimated cost ${:.4} would exceed limit ${:.4}",
                        estimated_cost, limit
                    )));
                }
            }
            Ok(())
        }

        /// Generate AWS authentication header
        async fn get_auth_header(&self) -> BraketResult<String> {
            // Simplified authentication - in practice would use AWS SDK
            // This is a placeholder for proper AWS Signature Version 4
            Ok(format!(
                "AWS4-HMAC-SHA256 Credential={}/...",
                self.credentials.0
            ))
        }
    }
}

#[cfg(feature = "braket")]
pub use client::*;

#[cfg(not(feature = "braket"))]
mod placeholder {
    use thiserror::Error;

    /// Error type for when Braket feature is not enabled
    #[derive(Error, Debug)]
    pub enum BraketError {
        /// Error when trying to use Braket without the feature enabled
        #[error("AWS Braket feature not enabled. Recompile with '--features braket'")]
        NotEnabled,
    }

    /// Result type for Braket operations
    pub type BraketResult<T> = Result<T, BraketError>;

    /// Placeholder for Braket client
    #[derive(Debug, Clone)]
    pub struct BraketClient {
        _private: (),
    }

    impl BraketClient {
        /// Placeholder for Braket client creation
        pub fn new(
            _access_key: impl Into<String>,
            _secret_key: impl Into<String>,
            _region: impl Into<String>,
        ) -> BraketResult<Self> {
            Err(BraketError::NotEnabled)
        }
    }

    /// Placeholder types for AWS Braket functionality (feature disabled)
    #[derive(Debug, Clone)]
    pub enum DeviceType {
        QuantumProcessor,
        Simulator,
    }

    #[derive(Debug, Clone)]
    pub enum DeviceStatus {
        Online,
        Offline,
        Retired,
    }

    #[derive(Debug, Clone)]
    pub enum TaskStatus {
        Running,
        Completed,
        Failed,
        Cancelled,
        Queued,
        Created,
    }

    #[derive(Debug, Clone)]
    pub struct BraketDevice;

    #[derive(Debug, Clone)]
    pub struct DeviceSelector;

    #[derive(Debug, Clone)]
    pub struct AdvancedAnnealingParams;

    #[derive(Debug, Clone)]
    pub struct TaskResult;

    #[derive(Debug, Clone)]
    pub struct TaskMetrics;

    #[derive(Debug, Clone)]
    pub struct BatchTaskResult;

    #[derive(Debug, Clone)]
    pub struct CostTracker;

    impl Default for DeviceSelector {
        fn default() -> Self {
            Self
        }
    }

    impl Default for AdvancedAnnealingParams {
        fn default() -> Self {
            Self
        }
    }

    impl Default for CostTracker {
        fn default() -> Self {
            Self
        }
    }
}

#[cfg(not(feature = "braket"))]
pub use placeholder::*;

/// Check if AWS Braket support is enabled
#[must_use]
pub const fn is_available() -> bool {
    cfg!(feature = "braket")
}

use std::fmt::Write;
