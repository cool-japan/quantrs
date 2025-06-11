//! D-Wave Leap cloud service client
//!
//! This module provides a comprehensive interface for D-Wave Leap cloud services,
//! including quantum annealing hardware, hybrid solvers, and advanced features.
//! It requires the "dwave" feature to be enabled.
//!
//! # Features
//!
//! - Full D-Wave Leap API integration
//! - Quantum annealing hardware access
//! - Hybrid classical-quantum solvers
//! - Advanced embedding integration
//! - Problem status tracking and management
//! - Performance monitoring and metrics
//! - Custom annealing schedules
//! - Batch problem submission
//! - Robust error handling and retry logic

#[cfg(feature = "dwave")]
mod client {
    use reqwest::Client;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::time::{Duration, Instant};
    use thiserror::Error;
    use tokio::runtime::Runtime;

    use crate::embedding::{Embedding, HardwareGraph, MinorMiner};
    use crate::ising::{IsingError, IsingModel, QuboModel};

    /// Errors that can occur when interacting with D-Wave API
    #[derive(Error, Debug)]
    pub enum DWaveError {
        /// Error in the underlying Ising model
        #[error("Ising error: {0}")]
        IsingError(#[from] IsingError),

        /// Error with the network request
        #[error("Network error: {0}")]
        NetworkError(#[from] reqwest::Error),

        /// Error parsing the response
        #[error("Response parsing error: {0}")]
        ParseError(#[from] serde_json::Error),

        /// Error with the D-Wave API response
        #[error("D-Wave API error: {0}")]
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

        /// Error with embedding
        #[error("Embedding error: {0}")]
        EmbeddingError(String),

        /// Error with hybrid solver
        #[error("Hybrid solver error: {0}")]
        HybridSolverError(String),

        /// Error with problem status
        #[error("Problem status error: {0}")]
        StatusError(String),

        /// Error with batch operations
        #[error("Batch operation error: {0}")]
        BatchError(String),

        /// Error with solver configuration
        #[error("Solver configuration error: {0}")]
        SolverConfigError(String),

        /// Timeout error
        #[error("Operation timed out: {0}")]
        TimeoutError(String),
    }

    /// Result type for D-Wave operations
    pub type DWaveResult<T> = Result<T, DWaveError>;

    /// D-Wave solver information
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SolverInfo {
        /// ID of the solver
        pub id: String,

        /// Name of the solver
        pub name: String,

        /// Description of the solver
        pub description: String,

        /// Number of qubits
        pub num_qubits: usize,

        /// Connectivity information
        pub connectivity: SolverConnectivity,

        /// Properties of the solver
        pub properties: SolverProperties,
    }

    /// D-Wave solver connectivity
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SolverConnectivity {
        /// Type of connectivity (e.g., "chimera", "pegasus")
        #[serde(rename = "type")]
        pub type_: String,

        /// Parameters for the connectivity
        #[serde(flatten)]
        pub params: serde_json::Value,
    }

    /// D-Wave solver properties
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SolverProperties {
        /// Supported parameters
        pub parameters: serde_json::Value,

        /// Additional properties
        #[serde(flatten)]
        pub other: serde_json::Value,
    }

    /// D-Wave problem submission parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ProblemParams {
        /// Number of reads/samples to take
        pub num_reads: usize,

        /// Annealing time in microseconds
        pub annealing_time: usize,

        /// Programming thermalization in microseconds
        #[serde(rename = "programming_thermalization")]
        pub programming_therm: usize,

        /// Read-out thermalization in microseconds
        #[serde(rename = "readout_thermalization")]
        pub readout_therm: usize,

        /// Flux biases for each qubit (optional)
        #[serde(rename = "flux_biases", skip_serializing_if = "Option::is_none")]
        pub flux_biases: Option<Vec<f64>>,

        /// Per-qubit flux bias values (optional, alternative format)
        #[serde(rename = "flux_bias", skip_serializing_if = "Option::is_none")]
        pub flux_bias_map: Option<serde_json::Map<String, serde_json::Value>>,

        /// Additional parameters
        #[serde(flatten)]
        pub other: serde_json::Value,
    }

    impl Default for ProblemParams {
        fn default() -> Self {
            Self {
                num_reads: 1000,
                annealing_time: 20,
                programming_therm: 1000,
                readout_therm: 0,
                flux_biases: None,
                flux_bias_map: None,
                other: serde_json::Value::Object(serde_json::Map::new()),
            }
        }
    }

    /// D-Wave problem submission
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Problem {
        /// The linear terms (h_i values for Ising, Q_ii for QUBO)
        #[serde(rename = "linear")]
        pub linear_terms: serde_json::Value,

        /// The quadratic terms (J_ij values for Ising, Q_ij for QUBO)
        #[serde(rename = "quadratic")]
        pub quadratic_terms: serde_json::Value,

        /// The type of problem (ising or qubo)
        #[serde(rename = "type")]
        pub type_: String,

        /// The solver to use
        pub solver: String,

        /// The parameters for the problem
        pub params: ProblemParams,
    }

    /// D-Wave problem solution
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Solution {
        /// The energy of each sample
        pub energies: Vec<f64>,

        /// The occurrences of each sample
        pub occurrences: Vec<usize>,

        /// The solutions (spin values for Ising, binary values for QUBO)
        pub solutions: Vec<Vec<i8>>,

        /// The number of samples
        pub num_samples: usize,

        /// The problem ID
        pub problem_id: String,

        /// The solver used
        pub solver: String,

        /// The timing information
        pub timing: serde_json::Value,
    }

    /// Enhanced solver types for Leap
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum SolverType {
        /// Quantum annealing processor
        #[serde(rename = "qpu")]
        QuantumProcessor,
        /// Hybrid classical-quantum solver
        #[serde(rename = "hybrid")]
        Hybrid,
        /// Discrete Quadratic Model solver
        #[serde(rename = "dqm")]
        DiscreteQuadraticModel,
        /// Constrained Quadratic Model solver
        #[serde(rename = "cqm")]
        ConstrainedQuadraticModel,
        /// Software solver/simulator
        #[serde(rename = "software")]
        Software,
    }

    /// Solver category for filtering
    #[derive(Debug, Clone, PartialEq)]
    pub enum SolverCategory {
        /// Quantum processing units
        QPU,
        /// Hybrid solvers
        Hybrid,
        /// Software-based solvers
        Software,
        /// All solver types
        All,
    }

    /// Problem status tracking
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    pub enum ProblemStatus {
        /// Problem is being processed
        #[serde(rename = "IN_PROGRESS")]
        InProgress,
        /// Problem completed successfully
        #[serde(rename = "COMPLETED")]
        Completed,
        /// Problem failed
        #[serde(rename = "FAILED")]
        Failed,
        /// Problem was cancelled
        #[serde(rename = "CANCELLED")]
        Cancelled,
        /// Problem is pending
        #[serde(rename = "PENDING")]
        Pending,
    }

    /// Problem metadata and tracking
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ProblemInfo {
        /// Problem ID
        pub id: String,
        /// Problem status
        pub status: ProblemStatus,
        /// Submission time
        pub submitted_on: String,
        /// Solver used
        pub solver: String,
        /// Problem type
        #[serde(rename = "type")]
        pub problem_type: String,
        /// Parameters used
        pub params: serde_json::Value,
        /// Additional metadata
        #[serde(flatten)]
        pub metadata: serde_json::Value,
    }

    /// Enhanced solver information with Leap features
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LeapSolverInfo {
        /// Solver ID
        pub id: String,
        /// Solver name
        pub name: String,
        /// Solver description
        pub description: String,
        /// Solver type
        #[serde(rename = "category")]
        pub solver_type: SolverType,
        /// Status (online/offline)
        pub status: String,
        /// Solver properties
        pub properties: serde_json::Value,
        /// Supported problem types
        pub problem_types: Vec<String>,
        /// Average queue time
        pub avg_load: Option<f64>,
        /// Whether solver is available
        pub available: bool,
    }

    /// Custom annealing schedule
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AnnealingSchedule {
        /// Time points in microseconds
        pub schedule: Vec<(f64, f64)>,
    }

    impl AnnealingSchedule {
        /// Create a linear annealing schedule
        pub fn linear(annealing_time: f64) -> Self {
            Self {
                schedule: vec![
                    (0.0, 1.0),            // Start with full transverse field
                    (annealing_time, 0.0), // End with no transverse field
                ],
            }
        }

        /// Create a pause-and-ramp schedule
        pub fn pause_and_ramp(annealing_time: f64, pause_start: f64, pause_duration: f64) -> Self {
            Self {
                schedule: vec![
                    (0.0, 1.0),
                    (pause_start, 1.0 - pause_start / annealing_time),
                    (
                        pause_start + pause_duration,
                        1.0 - pause_start / annealing_time,
                    ),
                    (annealing_time, 0.0),
                ],
            }
        }

        /// Create a custom schedule from points
        pub fn custom(points: Vec<(f64, f64)>) -> Self {
            Self { schedule: points }
        }
    }

    /// Advanced problem parameters for Leap
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AdvancedProblemParams {
        /// Number of reads/samples
        pub num_reads: usize,
        /// Custom annealing schedule
        #[serde(skip_serializing_if = "Option::is_none")]
        pub anneal_schedule: Option<AnnealingSchedule>,
        /// Programming thermalization
        #[serde(skip_serializing_if = "Option::is_none")]
        pub programming_thermalization: Option<usize>,
        /// Readout thermalization
        #[serde(skip_serializing_if = "Option::is_none")]
        pub readout_thermalization: Option<usize>,
        /// Auto-scale flag
        #[serde(skip_serializing_if = "Option::is_none")]
        pub auto_scale: Option<bool>,
        /// Chain strength
        #[serde(skip_serializing_if = "Option::is_none")]
        pub chain_strength: Option<f64>,
        /// Flux biases
        #[serde(skip_serializing_if = "Option::is_none")]
        pub flux_biases: Option<HashMap<String, f64>>,
        /// Additional parameters
        #[serde(flatten)]
        pub extra: HashMap<String, serde_json::Value>,
    }

    impl Default for AdvancedProblemParams {
        fn default() -> Self {
            Self {
                num_reads: 1000,
                anneal_schedule: None,
                programming_thermalization: Some(1000),
                readout_thermalization: Some(0),
                auto_scale: Some(true),
                chain_strength: None,
                flux_biases: None,
                extra: HashMap::new(),
            }
        }
    }

    /// Hybrid solver parameters
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct HybridSolverParams {
        /// Time limit in seconds
        #[serde(skip_serializing_if = "Option::is_none")]
        pub time_limit: Option<f64>,
        /// Maximum number of variables
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_variables: Option<usize>,
        /// Additional solver-specific parameters
        #[serde(flatten)]
        pub extra: HashMap<String, serde_json::Value>,
    }

    impl Default for HybridSolverParams {
        fn default() -> Self {
            Self {
                time_limit: Some(5.0),
                max_variables: None,
                extra: HashMap::new(),
            }
        }
    }

    /// Performance metrics for problems
    #[derive(Debug, Clone)]
    pub struct ProblemMetrics {
        /// Total execution time
        pub total_time: Duration,
        /// Queue time
        pub queue_time: Duration,
        /// Solver access time
        pub access_time: Duration,
        /// Programming time
        pub programming_time: Duration,
        /// Sampling time
        pub sampling_time: Duration,
        /// Readout time
        pub readout_time: Duration,
        /// Best energy found
        pub best_energy: f64,
        /// Number of valid solutions
        pub num_valid_solutions: usize,
        /// Chain break fraction
        pub chain_break_fraction: Option<f64>,
    }

    /// Batch problem submission result
    #[derive(Debug)]
    pub struct BatchSubmissionResult {
        /// List of submitted problem IDs
        pub problem_ids: Vec<String>,
        /// Success/failure status for each problem
        pub statuses: Vec<Result<String, DWaveError>>,
        /// Total submission time
        pub submission_time: Duration,
    }

    /// Solver selection criteria
    #[derive(Debug, Clone)]
    pub struct SolverSelector {
        /// Solver category filter
        pub category: SolverCategory,
        /// Minimum number of qubits
        pub min_qubits: Option<usize>,
        /// Maximum queue time preference
        pub max_queue_time: Option<f64>,
        /// Prefer online solvers
        pub online_only: bool,
        /// Specific solver name pattern
        pub name_pattern: Option<String>,
        /// Topology preference
        pub topology_preference: Option<String>,
    }

    impl Default for SolverSelector {
        fn default() -> Self {
            Self {
                category: SolverCategory::All,
                min_qubits: None,
                max_queue_time: None,
                online_only: true,
                name_pattern: None,
                topology_preference: None,
            }
        }
    }

    /// Problem embedding configuration
    #[derive(Debug, Clone)]
    pub struct EmbeddingConfig {
        /// Use automatic embedding
        pub auto_embed: bool,
        /// Embedding timeout
        pub timeout: Duration,
        /// Chain strength calculation method
        pub chain_strength_method: ChainStrengthMethod,
        /// Custom embedding (if not auto-embedding)
        pub custom_embedding: Option<Embedding>,
        /// Embedding optimization level
        pub optimization_level: usize,
    }

    /// Chain strength calculation methods
    #[derive(Debug, Clone)]
    pub enum ChainStrengthMethod {
        /// Automatic calculation
        Auto,
        /// Fixed value
        Fixed(f64),
        /// Based on problem coupling strengths
        Adaptive(f64), // multiplier
    }

    impl Default for EmbeddingConfig {
        fn default() -> Self {
            Self {
                auto_embed: true,
                timeout: Duration::from_secs(30),
                chain_strength_method: ChainStrengthMethod::Auto,
                custom_embedding: None,
                optimization_level: 1,
            }
        }
    }

    /// Enhanced D-Wave Leap cloud service client
    #[derive(Debug)]
    pub struct DWaveClient {
        /// The HTTP client for making API requests
        client: Client,

        /// The API endpoint
        endpoint: String,

        /// The API token
        token: String,

        /// The tokio runtime for async requests
        runtime: Runtime,

        /// Default solver selector
        default_solver_selector: SolverSelector,

        /// Default embedding configuration
        default_embedding_config: EmbeddingConfig,

        /// Retry configuration
        max_retries: usize,

        /// Request timeout
        request_timeout: Duration,

        /// Default problem timeout
        problem_timeout: Duration,
    }

    impl DWaveClient {
        /// Create a new enhanced D-Wave Leap client
        pub fn new(token: impl Into<String>, endpoint: Option<String>) -> DWaveResult<Self> {
            Self::with_config(
                token,
                endpoint,
                SolverSelector::default(),
                EmbeddingConfig::default(),
            )
        }

        /// Create a D-Wave client with custom configuration
        pub fn with_config(
            token: impl Into<String>,
            endpoint: Option<String>,
            solver_selector: SolverSelector,
            embedding_config: EmbeddingConfig,
        ) -> DWaveResult<Self> {
            // Create HTTP client with appropriate timeout
            let client = Client::builder()
                .timeout(Duration::from_secs(300)) // Increased timeout for large problems
                .build()
                .map_err(DWaveError::NetworkError)?;

            // Create tokio runtime
            let runtime = Runtime::new().map_err(|e| DWaveError::RuntimeError(e.to_string()))?;

            // Default Leap endpoint if not provided
            let endpoint =
                endpoint.unwrap_or_else(|| "https://cloud.dwavesys.com/sapi/v2".to_string());

            Ok(Self {
                client,
                endpoint,
                token: token.into(),
                runtime,
                default_solver_selector: solver_selector,
                default_embedding_config: embedding_config,
                max_retries: 3,
                request_timeout: Duration::from_secs(300),
                problem_timeout: Duration::from_secs(1800), // 30 minutes
            })
        }

        /// Get a list of available solvers
        pub fn get_solvers(&self) -> DWaveResult<Vec<SolverInfo>> {
            // Create the URL
            let url = format!("{}/solvers/remote", self.endpoint);

            // Execute the request
            self.runtime.block_on(async {
                let response = self
                    .client
                    .get(&url)
                    .header("Authorization", format!("token {}", self.token))
                    .send()
                    .await?;

                // Check for errors
                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await?;
                    return Err(DWaveError::ApiError(format!(
                        "Error getting solvers: {} - {}",
                        status, error_text
                    )));
                }

                // Parse the response
                let solvers: Vec<SolverInfo> = response.json().await?;
                Ok(solvers)
            })
        }

        /// Submit an Ising model to D-Wave
        pub fn submit_ising(
            &self,
            model: &IsingModel,
            solver_id: &str,
            params: ProblemParams,
        ) -> DWaveResult<Solution> {
            // Convert the Ising model to the format expected by D-Wave
            let mut linear_terms = serde_json::Map::new();
            for (qubit, bias) in model.biases() {
                linear_terms.insert(qubit.to_string(), serde_json::to_value(bias).unwrap());
            }

            let mut quadratic_terms = serde_json::Map::new();
            for coupling in model.couplings() {
                let key = format!("{},{}", coupling.i, coupling.j);
                quadratic_terms.insert(key, serde_json::to_value(coupling.strength).unwrap());
            }

            // Create the problem
            let problem = Problem {
                linear_terms: serde_json::Value::Object(linear_terms),
                quadratic_terms: serde_json::Value::Object(quadratic_terms),
                type_: "ising".to_string(),
                solver: solver_id.to_string(),
                params,
            };

            // Submit the problem
            self.submit_problem(&problem)
        }

        /// Submit a QUBO model to D-Wave
        pub fn submit_qubo(
            &self,
            model: &QuboModel,
            solver_id: &str,
            params: ProblemParams,
        ) -> DWaveResult<Solution> {
            // Convert the QUBO model to the format expected by D-Wave
            let mut linear_terms = serde_json::Map::new();
            for (var, value) in model.linear_terms() {
                linear_terms.insert(var.to_string(), serde_json::to_value(value).unwrap());
            }

            let mut quadratic_terms = serde_json::Map::new();
            for (var1, var2, value) in model.quadratic_terms() {
                let key = format!("{},{}", var1, var2);
                quadratic_terms.insert(key, serde_json::to_value(value).unwrap());
            }

            // Create the problem
            let problem = Problem {
                linear_terms: serde_json::Value::Object(linear_terms),
                quadratic_terms: serde_json::Value::Object(quadratic_terms),
                type_: "qubo".to_string(),
                solver: solver_id.to_string(),
                params,
            };

            // Submit the problem
            self.submit_problem(&problem)
        }

        /// Submit an Ising model with flux bias optimization
        pub fn submit_ising_with_flux_bias(
            &self,
            model: &IsingModel,
            solver_id: &str,
            params: ProblemParams,
            flux_biases: &std::collections::HashMap<usize, f64>,
        ) -> DWaveResult<Solution> {
            let mut params_with_flux = params;

            // Convert flux biases to the format expected by D-Wave
            let mut flux_map = serde_json::Map::new();
            for (qubit, &flux_bias) in flux_biases {
                flux_map.insert(qubit.to_string(), serde_json::to_value(flux_bias).unwrap());
            }
            params_with_flux.flux_bias_map = Some(flux_map);

            self.submit_ising(model, solver_id, params_with_flux)
        }

        /// Submit a problem to D-Wave
        fn submit_problem(&self, problem: &Problem) -> DWaveResult<Solution> {
            // Create the URL
            let url = format!("{}/problems", self.endpoint);

            // Execute the request
            self.runtime.block_on(async {
                // Submit the problem
                let response = self
                    .client
                    .post(&url)
                    .header("Authorization", format!("token {}", self.token))
                    .header("Content-Type", "application/json")
                    .json(problem)
                    .send()
                    .await?;

                // Check for errors
                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await?;
                    return Err(DWaveError::ApiError(format!(
                        "Error submitting problem: {} - {}",
                        status, error_text
                    )));
                }

                // Get the problem ID
                let submit_response: serde_json::Value = response.json().await?;
                let problem_id = submit_response["id"].as_str().ok_or_else(|| {
                    // Create the error string first
                    let error_msg = String::from("Failed to extract problem ID from response");
                    // Then return the error itself
                    DWaveError::ApiError(error_msg)
                })?;

                // Poll for the result
                let result_url = format!("{}/problems/{}", self.endpoint, problem_id);
                let mut attempts = 0;
                const MAX_ATTEMPTS: usize = 60; // 5 minutes with 5-second delay

                while attempts < MAX_ATTEMPTS {
                    // Get the problem status
                    let status_response = self
                        .client
                        .get(&result_url)
                        .header("Authorization", format!("token {}", self.token))
                        .send()
                        .await?;

                    // Check for errors
                    if !status_response.status().is_success() {
                        let status = status_response.status();
                        let error_text = status_response.text().await?;
                        return Err(DWaveError::ApiError(format!(
                            "Error getting problem status: {} - {}",
                            status, error_text
                        )));
                    }

                    // Parse the response
                    let status: serde_json::Value = status_response.json().await?;

                    // Check if the problem is done
                    if let Some(state) = status["state"].as_str() {
                        if state == "COMPLETED" {
                            // Get the solution
                            return Ok(Solution {
                                energies: serde_json::from_value(status["energies"].clone())?,
                                occurrences: serde_json::from_value(status["occurrences"].clone())?,
                                solutions: serde_json::from_value(status["solutions"].clone())?,
                                num_samples: status["num_samples"].as_u64().unwrap_or(0) as usize,
                                problem_id: problem_id.to_string(),
                                solver: problem.solver.clone(),
                                timing: status["timing"].clone(),
                            });
                        } else if state == "FAILED" {
                            let error = status["error"].as_str().unwrap_or("Unknown error");
                            return Err(DWaveError::ApiError(format!("Problem failed: {}", error)));
                        }
                    }

                    // Sleep and try again
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    attempts += 1;
                }

                Err(DWaveError::ApiError(
                    "Timeout waiting for problem solution".into(),
                ))
            })
        }

        /// Get enhanced Leap solver information
        pub fn get_leap_solvers(&self) -> DWaveResult<Vec<LeapSolverInfo>> {
            let url = format!("{}/solvers/remote", self.endpoint);

            self.runtime.block_on(async {
                let response = self
                    .client
                    .get(&url)
                    .header("Authorization", format!("token {}", self.token))
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await?;
                    return Err(DWaveError::ApiError(format!(
                        "Error getting Leap solvers: {} - {}",
                        status, error_text
                    )));
                }

                let solvers: Vec<LeapSolverInfo> = response.json().await?;
                Ok(solvers)
            })
        }

        /// Select optimal solver based on criteria
        pub fn select_solver(
            &self,
            selector: Option<&SolverSelector>,
        ) -> DWaveResult<LeapSolverInfo> {
            let selector = selector.unwrap_or(&self.default_solver_selector);
            let solvers = self.get_leap_solvers()?;

            let filtered_solvers: Vec<_> = solvers
                .into_iter()
                .filter(|solver| {
                    // Filter by category
                    let category_match = match selector.category {
                        SolverCategory::QPU => {
                            matches!(solver.solver_type, SolverType::QuantumProcessor)
                        }
                        SolverCategory::Hybrid => matches!(solver.solver_type, SolverType::Hybrid),
                        SolverCategory::Software => {
                            matches!(solver.solver_type, SolverType::Software)
                        }
                        SolverCategory::All => true,
                    };

                    // Filter by availability
                    let availability_match = !selector.online_only || solver.available;

                    // Filter by name pattern
                    let name_match = selector
                        .name_pattern
                        .as_ref()
                        .map(|pattern| solver.name.contains(pattern))
                        .unwrap_or(true);

                    // Filter by queue time
                    let queue_match = selector
                        .max_queue_time
                        .map(|max_time| solver.avg_load.unwrap_or(0.0) <= max_time)
                        .unwrap_or(true);

                    category_match && availability_match && name_match && queue_match
                })
                .collect();

            if filtered_solvers.is_empty() {
                return Err(DWaveError::SolverConfigError(
                    "No solvers match the selection criteria".to_string(),
                ));
            }

            // Sort by preference (lowest queue time first)
            let mut best_solver = filtered_solvers[0].clone();
            for solver in &filtered_solvers[1..] {
                let current_load = best_solver.avg_load.unwrap_or(f64::INFINITY);
                let candidate_load = solver.avg_load.unwrap_or(f64::INFINITY);
                if candidate_load < current_load {
                    best_solver = solver.clone();
                }
            }

            Ok(best_solver)
        }

        /// Submit problem with automatic embedding
        pub fn submit_ising_with_embedding(
            &self,
            model: &IsingModel,
            solver_id: Option<&str>,
            params: Option<AdvancedProblemParams>,
            embedding_config: Option<&EmbeddingConfig>,
        ) -> DWaveResult<Solution> {
            let embedding_config = embedding_config.unwrap_or(&self.default_embedding_config);

            // Select solver if not provided
            let solver = if let Some(id) = solver_id {
                self.get_leap_solvers()?
                    .into_iter()
                    .find(|s| s.id == id)
                    .ok_or_else(|| {
                        DWaveError::SolverConfigError(format!("Solver {} not found", id))
                    })?
            } else {
                self.select_solver(None)?
            };

            // Check if embedding is needed
            if matches!(solver.solver_type, SolverType::QuantumProcessor) {
                self.submit_with_auto_embedding(model, &solver, params, embedding_config)
            } else {
                // For hybrid solvers, convert to standard submission
                let legacy_params = params
                    .map(|p| ProblemParams {
                        num_reads: p.num_reads,
                        annealing_time: 20,
                        programming_therm: p.programming_thermalization.unwrap_or(1000),
                        readout_therm: p.readout_thermalization.unwrap_or(0),
                        flux_biases: None, // Use flux_bias_map instead
                        flux_bias_map: p.flux_biases.clone().map(|fb| {
                            fb.into_iter()
                                .map(|(k, v)| (k, serde_json::to_value(v).unwrap()))
                                .collect()
                        }),
                        other: serde_json::Value::Object(serde_json::Map::new()),
                    })
                    .unwrap_or_default();

                self.submit_ising(model, &solver.id, legacy_params)
            }
        }

        /// Submit with automatic embedding for QPU solvers
        fn submit_with_auto_embedding(
            &self,
            model: &IsingModel,
            solver: &LeapSolverInfo,
            params: Option<AdvancedProblemParams>,
            embedding_config: &EmbeddingConfig,
        ) -> DWaveResult<Solution> {
            let params = params.unwrap_or_default();

            // Create logical problem graph
            let mut logical_edges = Vec::new();
            for coupling in model.couplings() {
                logical_edges.push((coupling.i, coupling.j));
            }

            // Get solver topology and create hardware graph
            let hardware_graph = self.get_solver_topology(&solver.id)?;

            // Find embedding
            let embedding = if let Some(custom_emb) = &embedding_config.custom_embedding {
                custom_emb.clone()
            } else {
                let embedder = MinorMiner {
                    max_tries: 10 * embedding_config.optimization_level,
                    ..Default::default()
                };
                embedder
                    .find_embedding(&logical_edges, model.num_qubits, &hardware_graph)
                    .map_err(|e| DWaveError::EmbeddingError(e.to_string()))?
            };

            // Calculate chain strength
            let chain_strength =
                self.calculate_chain_strength(model, &embedding_config.chain_strength_method);

            // Create embedded problem
            let embedded_problem = self.embed_problem(model, &embedding, chain_strength)?;

            // Submit embedded problem
            self.submit_embedded_problem(&embedded_problem, solver, params)
        }

        /// Get solver topology information
        fn get_solver_topology(&self, solver_id: &str) -> DWaveResult<HardwareGraph> {
            let url = format!("{}/solvers/remote/{}", self.endpoint, solver_id);

            let topology_info = self.runtime.block_on(async {
                let response = self
                    .client
                    .get(&url)
                    .header("Authorization", format!("token {}", self.token))
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await?;
                    return Err(DWaveError::ApiError(format!(
                        "Error getting solver topology: {} - {}",
                        status, error_text
                    )));
                }

                let solver_data: serde_json::Value = response.json().await?;
                Ok(solver_data)
            })?;

            // Parse topology information
            let properties = &topology_info["properties"];

            if let Some(edges) = properties["couplers"].as_array() {
                let mut hardware_edges = Vec::new();
                for edge in edges {
                    if let (Some(i), Some(j)) = (edge[0].as_u64(), edge[1].as_u64()) {
                        hardware_edges.push((i as usize, j as usize));
                    }
                }

                let num_qubits = properties["qubits"]
                    .as_array()
                    .map(|arr| arr.len())
                    .unwrap_or(0);

                Ok(HardwareGraph::new_custom(num_qubits, hardware_edges))
            } else {
                Err(DWaveError::SolverConfigError(
                    "Could not parse solver topology".to_string(),
                ))
            }
        }

        /// Calculate appropriate chain strength
        fn calculate_chain_strength(
            &self,
            model: &IsingModel,
            method: &ChainStrengthMethod,
        ) -> f64 {
            match method {
                ChainStrengthMethod::Auto => {
                    // Calculate based on maximum coupling strength
                    let max_coupling = model
                        .couplings()
                        .iter()
                        .map(|c| c.strength.abs())
                        .fold(0.0, f64::max);

                    let max_bias = (0..model.num_qubits)
                        .filter_map(|i| model.get_bias(i).ok())
                        .fold(0.0_f64, |acc, bias| acc.max(bias.abs()));

                    2.0 * (max_coupling.max(max_bias))
                }
                ChainStrengthMethod::Fixed(value) => *value,
                ChainStrengthMethod::Adaptive(multiplier) => {
                    let avg_coupling = model
                        .couplings()
                        .iter()
                        .map(|c| c.strength.abs())
                        .sum::<f64>()
                        / model.couplings().len().max(1) as f64;

                    multiplier * avg_coupling
                }
            }
        }

        /// Embed logical problem onto physical hardware
        fn embed_problem(
            &self,
            model: &IsingModel,
            embedding: &Embedding,
            chain_strength: f64,
        ) -> DWaveResult<IsingModel> {
            let mut embedded_model = IsingModel::new(0); // Will be resized

            // Find maximum physical qubit index
            let max_qubit = embedding
                .chains
                .values()
                .flat_map(|chain| chain.iter())
                .max()
                .copied()
                .unwrap_or(0);

            embedded_model = IsingModel::new(max_qubit + 1);

            // Embed linear terms (biases)
            for (var, chain) in &embedding.chains {
                if let Ok(bias) = model.get_bias(*var) {
                    if bias != 0.0 {
                        // Distribute bias evenly across chain
                        let bias_per_qubit = bias / chain.len() as f64;
                        for &qubit in chain {
                            embedded_model
                                .set_bias(qubit, bias_per_qubit)
                                .map_err(|e| DWaveError::EmbeddingError(e.to_string()))?;
                        }
                    }
                }
            }

            // Embed quadratic terms (couplings)
            for coupling in model.couplings() {
                if let (Some(chain1), Some(chain2)) = (
                    embedding.chains.get(&coupling.i),
                    embedding.chains.get(&coupling.j),
                ) {
                    // Create couplings between all pairs of qubits in the chains
                    for &q1 in chain1 {
                        for &q2 in chain2 {
                            embedded_model
                                .set_coupling(q1, q2, coupling.strength)
                                .map_err(|e| DWaveError::EmbeddingError(e.to_string()))?;
                        }
                    }
                }
            }

            // Add chain couplings (ferromagnetic couplings within chains)
            for chain in embedding.chains.values() {
                for window in chain.windows(2) {
                    if let [q1, q2] = window {
                        embedded_model
                            .set_coupling(*q1, *q2, -chain_strength)
                            .map_err(|e| DWaveError::EmbeddingError(e.to_string()))?;
                    }
                }
            }

            Ok(embedded_model)
        }

        /// Submit an embedded problem
        fn submit_embedded_problem(
            &self,
            embedded_model: &IsingModel,
            solver: &LeapSolverInfo,
            params: AdvancedProblemParams,
        ) -> DWaveResult<Solution> {
            // Convert advanced parameters to legacy format
            let legacy_params = ProblemParams {
                num_reads: params.num_reads,
                annealing_time: params
                    .anneal_schedule
                    .as_ref()
                    .and_then(|schedule| schedule.schedule.last())
                    .map(|(time, _)| *time as usize)
                    .unwrap_or(20),
                programming_therm: params.programming_thermalization.unwrap_or(1000),
                readout_therm: params.readout_thermalization.unwrap_or(0),
                flux_biases: None, // Use flux_bias_map instead
                flux_bias_map: params.flux_biases.clone().map(|fb| {
                    fb.into_iter()
                        .map(|(k, v)| (k, serde_json::to_value(v).unwrap()))
                        .collect()
                }),
                other: serde_json::Value::Object(serde_json::Map::new()),
            };

            self.submit_ising(embedded_model, &solver.id, legacy_params)
        }

        /// Submit hybrid solver problem
        pub fn submit_hybrid(
            &self,
            model: &IsingModel,
            solver_id: Option<&str>,
            params: Option<HybridSolverParams>,
        ) -> DWaveResult<Solution> {
            let params = params.unwrap_or_default();

            // Select hybrid solver if not specified
            let solver = if let Some(id) = solver_id {
                self.get_leap_solvers()?
                    .into_iter()
                    .find(|s| s.id == id)
                    .ok_or_else(|| {
                        DWaveError::SolverConfigError(format!("Solver {} not found", id))
                    })?
            } else {
                let hybrid_selector = SolverSelector {
                    category: SolverCategory::Hybrid,
                    ..Default::default()
                };
                self.select_solver(Some(&hybrid_selector))?
            };

            // Convert to appropriate format for hybrid submission
            let mut linear_terms = serde_json::Map::new();
            for (qubit, bias) in model.biases() {
                linear_terms.insert(qubit.to_string(), serde_json::to_value(bias).unwrap());
            }

            let mut quadratic_terms = serde_json::Map::new();
            for coupling in model.couplings() {
                let key = format!("{},{}", coupling.i, coupling.j);
                quadratic_terms.insert(key, serde_json::to_value(coupling.strength).unwrap());
            }

            // Add hybrid-specific parameters
            let mut hybrid_params = params.extra.clone();
            if let Some(time_limit) = params.time_limit {
                hybrid_params.insert(
                    "time_limit".to_string(),
                    serde_json::to_value(time_limit).unwrap(),
                );
            }

            let problem = Problem {
                linear_terms: serde_json::Value::Object(linear_terms),
                quadratic_terms: serde_json::Value::Object(quadratic_terms),
                type_: "ising".to_string(),
                solver: solver.id.clone(),
                params: ProblemParams {
                    num_reads: 1, // Hybrid solvers typically return one solution
                    annealing_time: 1,
                    programming_therm: 0,
                    readout_therm: 0,
                    flux_biases: None,
                    flux_bias_map: None,
                    other: serde_json::Value::Object(
                        hybrid_params.into_iter().map(|(k, v)| (k, v)).collect(),
                    ),
                },
            };

            self.submit_problem(&problem)
        }

        /// Get problem status
        pub fn get_problem_status(&self, problem_id: &str) -> DWaveResult<ProblemInfo> {
            let url = format!("{}/problems/{}", self.endpoint, problem_id);

            self.runtime.block_on(async {
                let response = self
                    .client
                    .get(&url)
                    .header("Authorization", format!("token {}", self.token))
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await?;
                    return Err(DWaveError::ApiError(format!(
                        "Error getting problem status: {} - {}",
                        status, error_text
                    )));
                }

                let problem_info: ProblemInfo = response.json().await?;
                Ok(problem_info)
            })
        }

        /// Cancel a running problem
        pub fn cancel_problem(&self, problem_id: &str) -> DWaveResult<()> {
            let url = format!("{}/problems/{}/cancel", self.endpoint, problem_id);

            self.runtime.block_on(async {
                let response = self
                    .client
                    .delete(&url)
                    .header("Authorization", format!("token {}", self.token))
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await?;
                    return Err(DWaveError::ApiError(format!(
                        "Error cancelling problem: {} - {}",
                        status, error_text
                    )));
                }

                Ok(())
            })
        }

        /// Submit multiple problems in batch
        pub fn submit_batch(
            &self,
            problems: Vec<(&IsingModel, Option<&str>, Option<AdvancedProblemParams>)>,
        ) -> DWaveResult<BatchSubmissionResult> {
            let start_time = Instant::now();
            let mut problem_ids = Vec::new();
            let mut statuses = Vec::new();

            for (model, solver_id, params) in problems {
                match self.submit_ising_with_embedding(model, solver_id, params, None) {
                    Ok(solution) => {
                        problem_ids.push(solution.problem_id.clone());
                        statuses.push(Ok(solution.problem_id));
                    }
                    Err(e) => {
                        problem_ids.push(String::new());
                        statuses.push(Err(e));
                    }
                }
            }

            Ok(BatchSubmissionResult {
                problem_ids,
                statuses,
                submission_time: start_time.elapsed(),
            })
        }

        /// Get performance metrics for a completed problem
        pub fn get_problem_metrics(&self, problem_id: &str) -> DWaveResult<ProblemMetrics> {
            let solution = self.get_problem_result(problem_id)?;
            let timing = &solution.timing;

            // Extract timing information
            let queue_time =
                Duration::from_micros(timing["qpu_access_overhead_time"].as_u64().unwrap_or(0));
            let programming_time =
                Duration::from_micros(timing["qpu_programming_time"].as_u64().unwrap_or(0));
            let sampling_time =
                Duration::from_micros(timing["qpu_sampling_time"].as_u64().unwrap_or(0));
            let readout_time =
                Duration::from_micros(timing["qpu_readout_time"].as_u64().unwrap_or(0));

            let total_time = queue_time + programming_time + sampling_time + readout_time;
            let access_time = programming_time + sampling_time + readout_time;

            let best_energy = solution
                .energies
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
                .unwrap_or(f64::INFINITY);

            Ok(ProblemMetrics {
                total_time,
                queue_time,
                access_time,
                programming_time,
                sampling_time,
                readout_time,
                best_energy,
                num_valid_solutions: solution.solutions.len(),
                chain_break_fraction: timing["chain_break_fraction"].as_f64(),
            })
        }

        /// Get problem result (blocking until completion)
        pub fn get_problem_result(&self, problem_id: &str) -> DWaveResult<Solution> {
            let start_time = Instant::now();

            loop {
                let status = self.get_problem_status(problem_id)?;

                match status.status {
                    ProblemStatus::Completed => {
                        // Get the full solution
                        return self.get_solution_data(problem_id);
                    }
                    ProblemStatus::Failed => {
                        return Err(DWaveError::StatusError(format!(
                            "Problem {} failed",
                            problem_id
                        )));
                    }
                    ProblemStatus::Cancelled => {
                        return Err(DWaveError::StatusError(format!(
                            "Problem {} was cancelled",
                            problem_id
                        )));
                    }
                    ProblemStatus::InProgress | ProblemStatus::Pending => {
                        if start_time.elapsed() > self.problem_timeout {
                            return Err(DWaveError::TimeoutError(format!(
                                "Timeout waiting for problem {} completion",
                                problem_id
                            )));
                        }
                        // Wait before checking again
                        std::thread::sleep(Duration::from_secs(2));
                    }
                }
            }
        }

        /// Get solution data for a completed problem
        fn get_solution_data(&self, problem_id: &str) -> DWaveResult<Solution> {
            let url = format!("{}/problems/{}", self.endpoint, problem_id);

            self.runtime.block_on(async {
                let response = self
                    .client
                    .get(&url)
                    .header("Authorization", format!("token {}", self.token))
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await?;
                    return Err(DWaveError::ApiError(format!(
                        "Error getting solution data: {} - {}",
                        status, error_text
                    )));
                }

                let data: serde_json::Value = response.json().await?;

                Ok(Solution {
                    energies: serde_json::from_value(data["energies"].clone())?,
                    occurrences: serde_json::from_value(data["occurrences"].clone())?,
                    solutions: serde_json::from_value(data["solutions"].clone())?,
                    num_samples: data["num_samples"].as_u64().unwrap_or(0) as usize,
                    problem_id: problem_id.to_string(),
                    solver: data["solver"].as_str().unwrap_or("unknown").to_string(),
                    timing: data["timing"].clone(),
                })
            })
        }

        /// List recent problems
        pub fn list_problems(&self, limit: Option<usize>) -> DWaveResult<Vec<ProblemInfo>> {
            let mut url = format!("{}/problems", self.endpoint);
            if let Some(limit) = limit {
                url.push_str(&format!("?limit={}", limit));
            }

            self.runtime.block_on(async {
                let response = self
                    .client
                    .get(&url)
                    .header("Authorization", format!("token {}", self.token))
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await?;
                    return Err(DWaveError::ApiError(format!(
                        "Error listing problems: {} - {}",
                        status, error_text
                    )));
                }

                let problems: Vec<ProblemInfo> = response.json().await?;
                Ok(problems)
            })
        }

        /// Get account usage information
        pub fn get_usage_info(&self) -> DWaveResult<serde_json::Value> {
            let url = format!("{}/usage", self.endpoint);

            self.runtime.block_on(async {
                let response = self
                    .client
                    .get(&url)
                    .header("Authorization", format!("token {}", self.token))
                    .send()
                    .await?;

                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response.text().await?;
                    return Err(DWaveError::ApiError(format!(
                        "Error getting usage info: {} - {}",
                        status, error_text
                    )));
                }

                let usage: serde_json::Value = response.json().await?;
                Ok(usage)
            })
        }
    }
}

#[cfg(feature = "dwave")]
pub use client::*;

#[cfg(not(feature = "dwave"))]
mod placeholder {
    use thiserror::Error;

    /// Error type for when D-Wave feature is not enabled
    #[derive(Error, Debug)]
    pub enum DWaveError {
        /// Error when trying to use D-Wave without the feature enabled
        #[error("D-Wave feature not enabled. Recompile with '--features dwave'")]
        NotEnabled,
    }

    /// Result type for D-Wave operations
    pub type DWaveResult<T> = Result<T, DWaveError>;

    /// Placeholder for D-Wave client
    #[derive(Debug, Clone)]
    pub struct DWaveClient {
        _private: (),
    }

    impl DWaveClient {
        /// Placeholder for D-Wave client creation
        pub fn new(_token: impl Into<String>, _endpoint: Option<String>) -> DWaveResult<Self> {
            Err(DWaveError::NotEnabled)
        }
    }

    /// Placeholder for D-Wave problem submission parameters
    #[derive(Debug, Clone)]
    pub struct ProblemParams {
        /// Number of reads/samples to take
        pub num_reads: usize,
        /// Annealing time in microseconds
        pub annealing_time: usize,
        /// Programming thermalization in microseconds
        pub programming_therm: usize,
        /// Read-out thermalization in microseconds
        pub readout_therm: usize,
    }

    /// Placeholder types for enhanced Leap functionality (feature disabled)
    #[derive(Debug, Clone)]
    pub enum SolverType {
        QuantumProcessor,
        Hybrid,
        Software,
    }

    #[derive(Debug, Clone)]
    pub enum SolverCategory {
        QPU,
        Hybrid,
        Software,
        All,
    }

    #[derive(Debug, Clone)]
    pub enum ProblemStatus {
        InProgress,
        Completed,
        Failed,
        Cancelled,
        Pending,
    }

    #[derive(Debug, Clone)]
    pub struct SolverSelector;

    #[derive(Debug, Clone)]
    pub struct EmbeddingConfig;

    #[derive(Debug, Clone)]
    pub struct AdvancedProblemParams;

    #[derive(Debug, Clone)]
    pub struct HybridSolverParams;

    #[derive(Debug, Clone)]
    pub struct LeapSolverInfo;

    #[derive(Debug, Clone)]
    pub struct ProblemInfo;

    #[derive(Debug, Clone)]
    pub struct AnnealingSchedule;

    #[derive(Debug, Clone)]
    pub struct ProblemMetrics;

    #[derive(Debug, Clone)]
    pub struct BatchSubmissionResult;

    #[derive(Debug, Clone)]
    pub enum ChainStrengthMethod {
        Auto,
        Fixed(f64),
        Adaptive(f64),
    }

    impl Default for ProblemParams {
        fn default() -> Self {
            Self {
                num_reads: 1000,
                annealing_time: 20,
                programming_therm: 1000,
                readout_therm: 0,
            }
        }
    }
}

#[cfg(not(feature = "dwave"))]
pub use placeholder::*;

/// Check if D-Wave API support is enabled
pub fn is_available() -> bool {
    cfg!(feature = "dwave")
}
