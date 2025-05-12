//! D-Wave quantum annealing API client
//!
//! This module provides an interface for submitting problems to D-Wave quantum annealing hardware.
//! It requires the "dwave" feature to be enabled.

#[cfg(feature = "dwave")]
mod client {
    use reqwest::Client;
    use serde::{Deserialize, Serialize};
    use std::time::Duration;
    use thiserror::Error;
    use tokio::runtime::Runtime;

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

    /// D-Wave API client
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
    }

    impl DWaveClient {
        /// Create a new D-Wave client with the given API token
        pub fn new(token: impl Into<String>, endpoint: Option<String>) -> DWaveResult<Self> {
            // Create HTTP client
            let client = Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .map_err(DWaveError::NetworkError)?;

            // Create tokio runtime
            let runtime = Runtime::new().map_err(|e| DWaveError::RuntimeError(e.to_string()))?;

            // Default endpoint if not provided
            let endpoint =
                endpoint.unwrap_or_else(|| "https://cloud.dwavesys.com/sapi/v2".to_string());

            Ok(Self {
                client,
                endpoint,
                token: token.into(),
                runtime,
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
}

#[cfg(not(feature = "dwave"))]
pub use placeholder::*;

/// Check if D-Wave API support is enabled
pub fn is_available() -> bool {
    cfg!(feature = "dwave")
}
