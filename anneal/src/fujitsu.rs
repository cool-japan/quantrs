//! Fujitsu Digital Annealer interface
//!
//! This module provides support for solving optimization problems on
//! Fujitsu's Digital Annealer Unit (DAU) hardware.

use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use thiserror::Error;

use crate::ising::{IsingError, IsingModel, QuboModel};
use crate::simulator::AnnealingSolution;

/// Errors that can occur when using the Fujitsu Digital Annealer
#[derive(Error, Debug)]
pub enum FujitsuError {
    /// API authentication failed
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    /// Problem submission failed
    #[error("Problem submission failed: {0}")]
    SubmissionError(String),

    /// Result retrieval failed
    #[error("Result retrieval failed: {0}")]
    RetrievalError(String),

    /// Network or HTTP error
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    /// JSON parsing error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Hardware constraint violation
    #[error("Hardware constraint: {0}")]
    HardwareConstraint(String),

    /// Timeout waiting for results
    #[error("Timeout after {0:?}")]
    Timeout(Duration),
}

/// Result type for Fujitsu operations
pub type FujitsuResult<T> = Result<T, FujitsuError>;

/// Fujitsu Digital Annealer hardware specifications
#[derive(Debug, Clone)]
pub struct FujitsuHardwareSpec {
    /// Maximum number of variables supported
    pub max_variables: usize,

    /// Maximum number of connections per variable
    pub max_connections_per_variable: usize,

    /// Range of allowed coefficient values
    pub coefficient_range: (i32, i32),

    /// Available annealing parameters
    pub available_parameters: Vec<String>,
}

impl Default for FujitsuHardwareSpec {
    fn default() -> Self {
        Self {
            max_variables: 8192,                // Generation 2 DAU
            max_connections_per_variable: 8192, // Fully connected
            coefficient_range: (-65536, 65535), // 17-bit signed integers
            available_parameters: vec![
                "number_iterations".to_string(),
                "number_runs".to_string(),
                "temperature_start".to_string(),
                "temperature_end".to_string(),
                "temperature_mode".to_string(),
                "auto_scale".to_string(),
            ],
        }
    }
}

/// Parameters for Fujitsu Digital Annealer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FujitsuAnnealingParams {
    /// Number of iterations per run
    pub number_iterations: Option<u32>,

    /// Number of parallel runs
    pub number_runs: Option<u32>,

    /// Starting temperature
    pub temperature_start: Option<f64>,

    /// Ending temperature
    pub temperature_end: Option<f64>,

    /// Temperature scheduling mode
    pub temperature_mode: Option<String>,

    /// Enable automatic scaling of coefficients
    pub auto_scale: Option<bool>,

    /// Time limit in seconds
    pub time_limit_sec: Option<u32>,

    /// Guidance mode configuration
    pub guidance_config: Option<GuidanceConfig>,
}

impl Default for FujitsuAnnealingParams {
    fn default() -> Self {
        Self {
            number_iterations: Some(10_000_000),
            number_runs: Some(16),
            temperature_start: Some(100.0),
            temperature_end: Some(0.01),
            temperature_mode: Some("EXPONENTIAL".to_string()),
            auto_scale: Some(true),
            time_limit_sec: Some(60),
            guidance_config: None,
        }
    }
}

/// Guidance mode configuration for improved solution quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidanceConfig {
    /// Enable guidance mode
    pub enabled: bool,

    /// Initial solution for guidance (optional)
    pub initial_solution: Option<Vec<i8>>,

    /// Guidance strength parameter
    pub guidance_strength: f64,
}

/// Problem format for submission to Fujitsu DAU
#[derive(Debug, Serialize)]
struct FujitsuProblem {
    /// Binary or Ising mode
    pub binary_polynomial: BinaryPolynomial,

    /// Annealing parameters
    pub parameters: FujitsuAnnealingParams,
}

/// Binary polynomial representation
#[derive(Debug, Serialize)]
struct BinaryPolynomial {
    /// Number of variables
    pub num_variables: usize,

    /// Linear terms
    pub linear_terms: Vec<LinearTerm>,

    /// Quadratic terms
    pub quadratic_terms: Vec<QuadraticTerm>,

    /// Constant offset
    pub constant: f64,
}

#[derive(Debug, Serialize)]
struct LinearTerm {
    pub index: usize,
    pub coefficient: i32,
}

#[derive(Debug, Serialize)]
struct QuadraticTerm {
    pub index_i: usize,
    pub index_j: usize,
    pub coefficient: i32,
}

/// Result from Fujitsu DAU
#[derive(Debug, Deserialize)]
struct FujitsuSolutionResponse {
    pub solutions: Vec<FujitsuSolution>,
    pub timing: FujitsuTiming,
    pub info: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct FujitsuSolution {
    pub configuration: Vec<i8>,
    pub energy: f64,
    pub frequency: u32,
}

#[derive(Debug, Deserialize)]
struct FujitsuTiming {
    pub total_time_ms: u64,
    pub annealing_time_ms: u64,
    pub queue_time_ms: u64,
}

/// Fujitsu Digital Annealer client
pub struct FujitsuClient {
    /// HTTP client
    client: Client,

    /// API endpoint
    endpoint: String,

    /// API key
    api_key: String,

    /// Hardware specifications
    hardware_spec: FujitsuHardwareSpec,
}

impl FujitsuClient {
    /// Create a new Fujitsu Digital Annealer client
    pub fn new(endpoint: String, api_key: String) -> Self {
        Self {
            client: Client::new(),
            endpoint,
            api_key,
            hardware_spec: FujitsuHardwareSpec::default(),
        }
    }

    /// Create a client with custom hardware specifications
    pub fn with_hardware_spec(
        endpoint: String,
        api_key: String,
        hardware_spec: FujitsuHardwareSpec,
    ) -> Self {
        Self {
            client: Client::new(),
            endpoint,
            api_key,
            hardware_spec,
        }
    }

    /// Solve an Ising model on Fujitsu hardware
    pub async fn solve_ising(
        &self,
        model: &IsingModel,
        params: FujitsuAnnealingParams,
    ) -> FujitsuResult<AnnealingSolution> {
        // Validate the model against hardware constraints
        self.validate_model(model)?;

        // Convert to QUBO format (Fujitsu uses binary variables)
        let qubo = model.to_qubo();

        // Convert to Fujitsu problem format
        let problem = self.format_qubo_problem(&qubo, params)?;

        // Submit the problem
        let job_id = self.submit_problem(problem).await?;

        // Wait for and retrieve results
        let response = self.wait_for_results(&job_id).await?;

        // Convert results back to our format
        self.convert_results(response, model)
    }

    /// Solve a QUBO problem on Fujitsu hardware
    pub async fn solve_qubo(
        &self,
        qubo: &QuboModel,
        params: FujitsuAnnealingParams,
    ) -> FujitsuResult<Vec<bool>> {
        // Validate the model against hardware constraints
        self.validate_qubo(qubo)?;

        // Convert to Fujitsu problem format
        let problem = self.format_qubo_problem(qubo, params)?;

        // Submit the problem
        let job_id = self.submit_problem(problem).await?;

        // Wait for and retrieve results
        let response = self.wait_for_results(&job_id).await?;

        // Return best solution
        let best_solution = response
            .solutions
            .into_iter()
            .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
            .ok_or_else(|| FujitsuError::RetrievalError("No solutions returned".to_string()))?;

        Ok(best_solution
            .configuration
            .into_iter()
            .map(|x| x > 0)
            .collect())
    }

    /// Validate an Ising model against hardware constraints
    fn validate_model(&self, model: &IsingModel) -> FujitsuResult<()> {
        if model.num_qubits > self.hardware_spec.max_variables {
            return Err(FujitsuError::HardwareConstraint(format!(
                "Model has {} variables, but hardware supports at most {}",
                model.num_qubits, self.hardware_spec.max_variables
            )));
        }

        // Check connectivity constraints
        let mut connections_per_qubit = vec![0; model.num_qubits];
        for coupling in model.couplings() {
            connections_per_qubit[coupling.i] += 1;
            connections_per_qubit[coupling.j] += 1;
        }

        for (qubit, &count) in connections_per_qubit.iter().enumerate() {
            if count > self.hardware_spec.max_connections_per_variable {
                return Err(FujitsuError::HardwareConstraint(format!(
                    "Qubit {} has {} connections, but hardware supports at most {}",
                    qubit, count, self.hardware_spec.max_connections_per_variable
                )));
            }
        }

        Ok(())
    }

    /// Validate a QUBO model against hardware constraints
    fn validate_qubo(&self, qubo: &QuboModel) -> FujitsuResult<()> {
        if qubo.num_variables > self.hardware_spec.max_variables {
            return Err(FujitsuError::HardwareConstraint(format!(
                "Model has {} variables, but hardware supports at most {}",
                qubo.num_variables, self.hardware_spec.max_variables
            )));
        }

        Ok(())
    }

    /// Convert QUBO model to Fujitsu problem format
    fn format_qubo_problem(
        &self,
        qubo: &QuboModel,
        params: FujitsuAnnealingParams,
    ) -> FujitsuResult<FujitsuProblem> {
        let mut linear_terms = Vec::new();
        let mut quadratic_terms = Vec::new();

        // Convert linear terms
        for (var, value) in qubo.linear_terms() {
            let coefficient = self.scale_coefficient(value)?;
            linear_terms.push(LinearTerm {
                index: var,
                coefficient,
            });
        }

        // Convert quadratic terms
        for (var1, var2, value) in qubo.quadratic_terms() {
            let coefficient = self.scale_coefficient(value)?;
            quadratic_terms.push(QuadraticTerm {
                index_i: var1,
                index_j: var2,
                coefficient,
            });
        }

        Ok(FujitsuProblem {
            binary_polynomial: BinaryPolynomial {
                num_variables: qubo.num_variables,
                linear_terms,
                quadratic_terms,
                constant: qubo.offset,
            },
            parameters: params,
        })
    }

    /// Scale a floating-point coefficient to hardware integer range
    fn scale_coefficient(&self, value: f64) -> FujitsuResult<i32> {
        let (min_coeff, max_coeff) = self.hardware_spec.coefficient_range;

        // Simple scaling - in practice, this should be more sophisticated
        let scaled = (value * 1000.0).round() as i32;

        if scaled < min_coeff || scaled > max_coeff {
            return Err(FujitsuError::HardwareConstraint(format!(
                "Coefficient {} is outside hardware range [{}, {}]",
                scaled, min_coeff, max_coeff
            )));
        }

        Ok(scaled)
    }

    /// Submit a problem to the Fujitsu DAU
    async fn submit_problem(&self, problem: FujitsuProblem) -> FujitsuResult<String> {
        let response = self
            .client
            .post(&format!("{}/problems", self.endpoint))
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .json(&problem)
            .send()
            .await?;

        match response.status() {
            StatusCode::OK | StatusCode::ACCEPTED => {
                let job_response: serde_json::Value = response.json().await?;
                job_response["job_id"]
                    .as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| FujitsuError::SubmissionError("No job ID returned".to_string()))
            }
            StatusCode::UNAUTHORIZED => Err(FujitsuError::AuthenticationError(
                "Invalid API key".to_string(),
            )),
            status => {
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                Err(FujitsuError::SubmissionError(format!(
                    "Submission failed with status {}: {}",
                    status, error_text
                )))
            }
        }
    }

    /// Wait for results from a submitted job
    async fn wait_for_results(&self, job_id: &str) -> FujitsuResult<FujitsuSolutionResponse> {
        let mut retry_count = 0;
        let max_retries = 60; // 5 minutes with 5-second intervals

        loop {
            let response = self
                .client
                .get(&format!("{}/problems/{}/result", self.endpoint, job_id))
                .header("Authorization", &format!("Bearer {}", self.api_key))
                .send()
                .await?;

            match response.status() {
                StatusCode::OK => {
                    return Ok(response.json().await?);
                }
                StatusCode::ACCEPTED => {
                    // Job still running
                    if retry_count >= max_retries {
                        return Err(FujitsuError::Timeout(Duration::from_secs(300)));
                    }
                    retry_count += 1;
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
                status => {
                    let error_text = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());
                    return Err(FujitsuError::RetrievalError(format!(
                        "Result retrieval failed with status {}: {}",
                        status, error_text
                    )));
                }
            }
        }
    }

    /// Convert Fujitsu results back to our format
    fn convert_results(
        &self,
        response: FujitsuSolutionResponse,
        model: &IsingModel,
    ) -> FujitsuResult<AnnealingSolution> {
        // Find the best solution
        let best_solution = response
            .solutions
            .into_iter()
            .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
            .ok_or_else(|| FujitsuError::RetrievalError("No solutions returned".to_string()))?;

        // Convert from binary (0/1) to spin (-1/+1)
        let spins: Vec<i8> = best_solution
            .configuration
            .into_iter()
            .map(|x| if x == 0 { -1 } else { 1 })
            .collect();

        // Calculate energy in Ising model
        let energy = model.energy(&spins)?;

        Ok(AnnealingSolution {
            best_spins: spins,
            best_energy: energy,
            repetitions: 1,
            total_sweeps: response.timing.annealing_time_ms as usize,
            runtime: Duration::from_millis(response.timing.total_time_ms),
            info: format!(
                "Solved on Fujitsu Digital Annealer in {:.1}s (queue: {:.1}s, annealing: {:.1}s)",
                response.timing.total_time_ms as f64 / 1000.0,
                response.timing.queue_time_ms as f64 / 1000.0,
                response.timing.annealing_time_ms as f64 / 1000.0,
            ),
        })
    }
}

/// Check if Fujitsu Digital Annealer support is available
pub fn is_available() -> bool {
    // This would check for API credentials in environment variables
    std::env::var("FUJITSU_DAU_API_KEY").is_ok() && std::env::var("FUJITSU_DAU_ENDPOINT").is_ok()
}

/// Create a Fujitsu client from environment variables
pub fn from_env() -> FujitsuResult<FujitsuClient> {
    let api_key = std::env::var("FUJITSU_DAU_API_KEY").map_err(|_| {
        FujitsuError::AuthenticationError("FUJITSU_DAU_API_KEY not set".to_string())
    })?;
    let endpoint = std::env::var("FUJITSU_DAU_ENDPOINT").map_err(|_| {
        FujitsuError::AuthenticationError("FUJITSU_DAU_ENDPOINT not set".to_string())
    })?;

    Ok(FujitsuClient::new(endpoint, api_key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_spec() {
        let spec = FujitsuHardwareSpec::default();
        assert_eq!(spec.max_variables, 8192);
        assert_eq!(spec.coefficient_range, (-65536, 65535));
    }

    #[test]
    fn test_parameter_defaults() {
        let params = FujitsuAnnealingParams::default();
        assert_eq!(params.number_iterations, Some(10_000_000));
        assert_eq!(params.number_runs, Some(16));
        assert_eq!(params.temperature_mode, Some("EXPONENTIAL".to_string()));
    }

    #[test]
    fn test_model_validation() {
        let client = FujitsuClient::new("http://test".to_string(), "test_key".to_string());

        // Test valid model
        let mut model = IsingModel::new(100);
        model.set_coupling(0, 1, -1.0).unwrap();
        assert!(client.validate_model(&model).is_ok());

        // Test model too large
        let large_model = IsingModel::new(10000);
        assert!(client.validate_model(&large_model).is_err());
    }

    #[test]
    fn test_coefficient_scaling() {
        let client = FujitsuClient::new("http://test".to_string(), "test_key".to_string());

        // Test valid coefficient
        assert_eq!(client.scale_coefficient(1.5).unwrap(), 1500);
        assert_eq!(client.scale_coefficient(-2.7).unwrap(), -2700);

        // Test coefficient out of range
        assert!(client.scale_coefficient(100.0).is_err());
        assert!(client.scale_coefficient(1000000.0).is_err());
    }
}
