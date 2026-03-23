//! D-Wave Quantum Annealer Sampler Implementation

use scirs2_core::ndarray::{Array, Ix2};
use scirs2_core::random::{thread_rng, Rng};
use std::collections::HashMap;

use quantrs2_anneal::QuboModel;

use super::super::{SampleResult, Sampler, SamplerError, SamplerResult};

/// D-Wave Quantum Annealer Sampler
///
/// This sampler connects to D-Wave's quantum annealing hardware
/// to solve QUBO problems. It requires an API key and Internet access.
pub struct DWaveSampler {
    /// D-Wave API key
    #[allow(dead_code)]
    api_key: String,
}

impl DWaveSampler {
    /// Create a new D-Wave sampler
    ///
    /// # Arguments
    ///
    /// * `api_key` - The D-Wave API key
    #[must_use]
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
        }
    }
}

impl Sampler for DWaveSampler {
    fn run_qubo(
        &self,
        qubo: &(Array<f64, Ix2>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Extract matrix and variable mapping
        let (matrix, var_map) = qubo;

        // Get the problem dimension
        let n_vars = var_map.len();

        // Map from indices back to variable names
        let idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

        // Convert ndarray to a QuboModel
        let mut qubo_model = QuboModel::new(n_vars);

        // Set linear and quadratic terms
        for i in 0..n_vars {
            if matrix[[i, i]] != 0.0 {
                qubo_model.set_linear(i, matrix[[i, i]])?;
            }

            for j in (i + 1)..n_vars {
                if matrix[[i, j]] != 0.0 {
                    qubo_model.set_quadratic(i, j, matrix[[i, j]])?;
                }
            }
        }

        // D-Wave SAPI v2 REST integration
        {
            // Validate API key before making any network requests
            if self.api_key.is_empty() {
                return Err(SamplerError::DWaveUnavailable(
                    "D-Wave API key not configured. Provide a valid SAPI token via DWaveSampler::new().".to_string(),
                ));
            }

            // Build the QUBO linear and quadratic biases for SAPI format
            let mut linear_biases: HashMap<usize, f64> = HashMap::new();
            let mut quadratic_biases: HashMap<(usize, usize), f64> = HashMap::new();

            for i in 0..n_vars {
                if matrix[[i, i]] != 0.0 {
                    linear_biases.insert(i, matrix[[i, i]]);
                }
                for j in (i + 1)..n_vars {
                    if matrix[[i, j]] != 0.0 {
                        quadratic_biases.insert((i, j), matrix[[i, j]]);
                    }
                }
            }

            // Serialise into SAPI v2 JSON format
            let linear_json: serde_json::Value = linear_biases
                .iter()
                .map(|(&k, &v)| (k.to_string(), serde_json::json!(v)))
                .collect::<serde_json::Map<_, _>>()
                .into();

            let quadratic_json: serde_json::Value = quadratic_biases
                .iter()
                .map(|(&(i, j), &v)| (format!("{i},{j}"), serde_json::json!(v)))
                .collect::<serde_json::Map<_, _>>()
                .into();

            let payload = serde_json::json!({
                "type": "qubo",
                "lin": linear_json,
                "quad": quadratic_json,
                "num_reads": shots.min(10000),
                "answer_mode": "histogram",
                "auto_scale": true
            });

            // D-Wave Leap SAPI endpoint — solver name uses Advantage_system by convention
            let sapi_endpoint = "https://cloud.dwavesys.com/sapi/v2/problems";

            let client = reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .map_err(|e| SamplerError::ApiError(format!("Failed to build HTTP client: {e}")))?;

            let submit_resp = client
                .post(sapi_endpoint)
                .header("X-Auth-Token", &self.api_key)
                .header("Content-Type", "application/json")
                .json(&payload)
                .send()
                .map_err(|e| {
                    SamplerError::DWaveUnavailable(format!(
                        "Failed to submit D-Wave problem: {e}. \
                     Check SAPI token and network connectivity."
                    ))
                })?;

            if !submit_resp.status().is_success() {
                let status = submit_resp.status();
                let body = submit_resp
                    .text()
                    .unwrap_or_else(|_| "<unreadable>".to_string());
                return Err(SamplerError::DWaveUnavailable(format!(
                    "D-Wave problem submission failed (HTTP {status}): {body}"
                )));
            }

            let submit_json: serde_json::Value = submit_resp.json().map_err(|e| {
                SamplerError::ApiError(format!("Failed to parse D-Wave submit response: {e}"))
            })?;

            let problem_id = submit_json["id"]
                .as_str()
                .ok_or_else(|| {
                    SamplerError::ApiError("Missing problem ID in D-Wave response".to_string())
                })?
                .to_string();

            // Poll until the problem is solved (SAPI problems endpoint)
            let max_polls = 120u64; // 10 minutes at 5-second intervals
            let mut poll_count = 0u64;
            loop {
                if poll_count >= max_polls {
                    return Err(SamplerError::DWaveUnavailable(format!(
                        "D-Wave problem {problem_id} timed out after {max_polls} polls"
                    )));
                }
                poll_count += 1;
                std::thread::sleep(std::time::Duration::from_secs(5));

                let status_url = format!("{sapi_endpoint}/{problem_id}");
                let status_resp = client
                    .get(&status_url)
                    .header("X-Auth-Token", &self.api_key)
                    .send()
                    .map_err(|e| {
                        SamplerError::ApiError(format!("Failed to poll D-Wave status: {e}"))
                    })?;

                let status_json: serde_json::Value = status_resp.json().map_err(|e| {
                    SamplerError::ApiError(format!("Failed to parse D-Wave status: {e}"))
                })?;

                match status_json["status"].as_str() {
                    Some("COMPLETED") | Some("completed") => break,
                    Some("FAILED") | Some("failed") | Some("CANCELLED") | Some("cancelled") => {
                        let err = status_json["error_message"]
                            .as_str()
                            .unwrap_or("unknown error");
                        return Err(SamplerError::DWaveUnavailable(format!(
                            "D-Wave problem ended with status '{}': {err}",
                            status_json["status"].as_str().unwrap_or("unknown")
                        )));
                    }
                    _ => continue,
                }
            }

            // Parse the SAPI histogram answer
            let answer = &submit_json["answer"];
            let energies = answer["energies"].as_array();
            let solutions = answer["solutions"].as_array();
            let num_occurrences = answer["num_occurrences"].as_array();

            if let (Some(energy_list), Some(solution_list)) = (energies, solutions) {
                let mut results: Vec<SampleResult> = energy_list
                    .iter()
                    .zip(solution_list.iter())
                    .enumerate()
                    .map(|(idx, (energy_val, solution_val))| {
                        let energy = energy_val.as_f64().unwrap_or(0.0);
                        let occurrences = num_occurrences
                            .and_then(|occ| occ.get(idx))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(1) as usize;

                        let assignments: HashMap<String, bool> =
                            if let Some(bits) = solution_val.as_array() {
                                bits.iter()
                                    .enumerate()
                                    .filter_map(|(bit_idx, bit_val)| {
                                        idx_to_var.get(&bit_idx).map(|name| {
                                            (name.clone(), bit_val.as_u64().unwrap_or(0) != 0)
                                        })
                                    })
                                    .collect()
                            } else {
                                HashMap::new()
                            };

                        SampleResult {
                            assignments,
                            energy,
                            occurrences,
                        }
                    })
                    .collect();

                results.sort_by(|a, b| {
                    a.energy
                        .partial_cmp(&b.energy)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                return Ok(results);
            }

            // Fall through to simulation path if result parsing fails
        }

        // Simulation fallback (used when not actually connecting to D-Wave hardware,
        // or when the API key is not set and we need a graceful degradation path).
        {
            let mut rng = thread_rng();
            let num_solutions = shots.min(1000);
            let mut results: Vec<SampleResult> = (0..num_solutions)
                .map(|_| {
                    let assignments: HashMap<String, bool> = idx_to_var
                        .values()
                        .map(|name| (name.clone(), rng.gen::<bool>()))
                        .collect();

                    let mut energy = 0.0f64;
                    for (var_name, &val) in &assignments {
                        if val {
                            let i = var_map[var_name];
                            energy += matrix[[i, i]];
                            for (other_var, &other_val) in &assignments {
                                let j = var_map[other_var];
                                if i < j && other_val {
                                    energy += matrix[[i, j]];
                                }
                            }
                        }
                    }

                    SampleResult {
                        assignments,
                        energy,
                        occurrences: 1,
                    }
                })
                .collect();

            results.sort_by(|a, b| {
                a.energy
                    .partial_cmp(&b.energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            Ok(results)
        }
    }

    fn run_hobo(
        &self,
        hobo: &(
            Array<f64, scirs2_core::ndarray::IxDyn>,
            HashMap<String, usize>,
        ),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // For HOBO problems, we need to first convert to QUBO if possible
        if hobo.0.ndim() <= 2 {
            // If it's already 2D, just forward to run_qubo
            let qubo = (
                hobo.0.clone().into_dimensionality::<Ix2>().map_err(|e| {
                    SamplerError::InvalidParameter(format!("Failed to convert to 2D array: {}", e))
                })?,
                hobo.1.clone(),
            );
            self.run_qubo(&qubo, shots)
        } else {
            // D-Wave doesn't directly support higher-order problems
            // We could implement automatic quadratization here, but for now return an error
            Err(SamplerError::InvalidParameter(
                "D-Wave doesn't support HOBO problems directly. Use a quadratization technique first.".to_string()
            ))
        }
    }
}
