//! Validation methods for process tomography

use scirs2_core::ndarray::{Array1, Array2, Array4};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;
use std::collections::HashMap;

use super::core::SciRS2ProcessTomographer;
use super::results::*;
use crate::DeviceResult;

impl SciRS2ProcessTomographer {
    /// Perform comprehensive validation
    pub fn perform_validation(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<ProcessValidationResults> {
        let cross_validation = if self.config.validation_config.enable_cross_validation {
            Some(self.perform_cross_validation(experimental_data)?)
        } else {
            None
        };

        let bootstrap_results = if self.config.validation_config.enable_bootstrap {
            Some(self.perform_bootstrap_validation(experimental_data)?)
        } else {
            None
        };

        let benchmark_results = if self.config.validation_config.enable_benchmarking {
            Some(self.perform_benchmark_validation(experimental_data)?)
        } else {
            None
        };

        let model_selection = self.perform_model_selection(experimental_data)?;

        Ok(ProcessValidationResults {
            cross_validation,
            bootstrap_results,
            benchmark_results,
            model_selection,
        })
    }

    /// Perform k-fold cross-validation
    fn perform_cross_validation(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<CrossValidationResults> {
        let num_folds = self.config.validation_config.cv_folds;
        let data_size = experimental_data.measurement_results.len();
        let fold_size = data_size / num_folds;

        let mut fold_scores = Vec::new();

        for fold_idx in 0..num_folds {
            // Create training and validation sets
            let val_start = fold_idx * fold_size;
            let val_end = if fold_idx == num_folds - 1 {
                data_size
            } else {
                (fold_idx + 1) * fold_size
            };

            let training_data = self.create_training_fold(experimental_data, val_start, val_end)?;
            let validation_data =
                self.create_validation_fold(experimental_data, val_start, val_end)?;

            // Train on training set
            let (process_matrix, _) = self.linear_inversion_reconstruction(&training_data)?;

            // Evaluate on validation set
            let score = self.evaluate_process_quality(&process_matrix, &validation_data)?;
            fold_scores.push(score);
        }

        let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        let variance = fold_scores
            .iter()
            .map(|&x| (x - mean_score).powi(2))
            .sum::<f64>()
            / fold_scores.len() as f64;
        let std_score = variance.sqrt();

        // 95% confidence interval
        let margin = 1.96 * std_score / (fold_scores.len() as f64).sqrt();
        let confidence_interval = (mean_score - margin, mean_score + margin);

        Ok(CrossValidationResults {
            fold_scores,
            mean_score,
            std_score,
            confidence_interval,
        })
    }

    /// Perform bootstrap validation
    fn perform_bootstrap_validation(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<BootstrapResults> {
        let num_bootstrap = self.config.validation_config.bootstrap_samples;
        let data_size = experimental_data.measurement_results.len();

        let mut bootstrap_samples = Vec::new();
        let mut bootstrap_metrics = HashMap::new();

        for _ in 0..num_bootstrap {
            // Create bootstrap sample
            let bootstrap_data = self.create_bootstrap_sample(experimental_data)?;

            // Reconstruct process
            let (process_matrix, _) = self.linear_inversion_reconstruction(&bootstrap_data)?;

            // Calculate metrics
            let metrics = self.calculate_process_metrics(&process_matrix)?;
            bootstrap_samples.push(metrics.clone());

            // Collect metrics for confidence intervals
            self.collect_bootstrap_metrics(&metrics, &mut bootstrap_metrics);
        }

        // Calculate confidence intervals
        let confidence_intervals =
            self.calculate_bootstrap_confidence_intervals(&bootstrap_metrics)?;

        // Calculate bias estimates
        let bias_estimates = self.calculate_bootstrap_bias(&bootstrap_samples)?;

        Ok(BootstrapResults {
            bootstrap_samples,
            confidence_intervals,
            bias_estimates,
        })
    }

    /// Perform benchmark validation
    fn perform_benchmark_validation(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<BenchmarkResults> {
        let benchmarks = &self.config.validation_config.benchmark_processes;
        let mut benchmark_scores = HashMap::new();
        let mut rankings = HashMap::new();

        // Reconstruct experimental process
        let (experimental_process, _) = self.linear_inversion_reconstruction(experimental_data)?;

        for (idx, benchmark_name) in benchmarks.iter().enumerate() {
            let benchmark_process = self.create_benchmark_process(benchmark_name)?;
            let fidelity =
                self.calculate_process_fidelity_between(&experimental_process, &benchmark_process)?;

            benchmark_scores.insert(benchmark_name.clone(), fidelity);
            rankings.insert(benchmark_name.clone(), idx + 1);
        }

        Ok(BenchmarkResults {
            benchmark_scores,
            rankings,
        })
    }

    /// Perform model selection
    fn perform_model_selection(
        &self,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<ModelSelectionResults> {
        let reconstruction_methods = vec![
            "linear_inversion",
            "maximum_likelihood",
            "compressed_sensing",
            "bayesian",
        ];

        let mut aic_scores = HashMap::new();
        let mut bic_scores = HashMap::new();
        let mut cv_scores = HashMap::new();
        let mut model_weights = HashMap::new();

        let mut best_score = f64::NEG_INFINITY;
        let mut best_model = "linear_inversion".to_string();

        for method in &reconstruction_methods {
            // Reconstruct using method
            let (process_matrix, quality) = match method.as_ref() {
                "linear_inversion" => self.linear_inversion_reconstruction(experimental_data)?,
                "maximum_likelihood" => {
                    self.maximum_likelihood_reconstruction(experimental_data)?
                }
                "compressed_sensing" => {
                    self.compressed_sensing_reconstruction(experimental_data)?
                }
                "bayesian" => self.bayesian_reconstruction(experimental_data)?,
                _ => self.linear_inversion_reconstruction(experimental_data)?,
            };

            // Calculate model selection criteria
            let num_params = self.estimate_num_parameters(&process_matrix);
            let log_likelihood = quality.log_likelihood;
            let n_data = experimental_data.measurement_results.len() as f64;

            let aic = -2.0 * log_likelihood + 2.0 * num_params as f64;
            let bic = -2.0 * log_likelihood + num_params as f64 * n_data.ln();

            // Cross-validation score
            let cv_score = self.calculate_cv_score_for_method(method, experimental_data)?;

            aic_scores.insert(method.to_string(), aic);
            bic_scores.insert(method.to_string(), bic);
            cv_scores.insert(method.to_string(), cv_score);

            // Track best model (lowest AIC)
            if -aic > best_score {
                best_score = -aic;
                best_model = method.to_string();
            }
        }

        // Calculate model weights (Akaike weights)
        let min_aic = aic_scores.values().cloned().fold(f64::INFINITY, f64::min);
        let mut weight_sum = 0.0;

        for (model, &aic) in &aic_scores {
            let delta_aic = aic - min_aic;
            let weight = (-0.5 * delta_aic).exp();
            model_weights.insert(model.clone(), weight);
            weight_sum += weight;
        }

        // Normalize weights
        for weight in model_weights.values_mut() {
            *weight /= weight_sum;
        }

        Ok(ModelSelectionResults {
            aic_scores,
            bic_scores,
            cross_validation_scores: cv_scores,
            best_model,
            model_weights,
        })
    }

    /// Create training fold for cross-validation
    fn create_training_fold(
        &self,
        data: &ExperimentalData,
        val_start: usize,
        val_end: usize,
    ) -> DeviceResult<ExperimentalData> {
        let mut training_results = Vec::new();
        let mut training_uncertainties = Vec::new();

        for (i, (&result, &uncertainty)) in data
            .measurement_results
            .iter()
            .zip(data.measurement_uncertainties.iter())
            .enumerate()
        {
            if i < val_start || i >= val_end {
                training_results.push(result);
                training_uncertainties.push(uncertainty);
            }
        }

        Ok(ExperimentalData {
            input_states: data.input_states.clone(),
            measurement_operators: data.measurement_operators.clone(),
            measurement_results: training_results,
            measurement_uncertainties: training_uncertainties,
        })
    }

    /// Create validation fold for cross-validation
    fn create_validation_fold(
        &self,
        data: &ExperimentalData,
        val_start: usize,
        val_end: usize,
    ) -> DeviceResult<ExperimentalData> {
        let validation_results = data.measurement_results[val_start..val_end].to_vec();
        let validation_uncertainties = data.measurement_uncertainties[val_start..val_end].to_vec();

        Ok(ExperimentalData {
            input_states: data.input_states.clone(),
            measurement_operators: data.measurement_operators.clone(),
            measurement_results: validation_results,
            measurement_uncertainties: validation_uncertainties,
        })
    }

    /// Create bootstrap sample
    fn create_bootstrap_sample(&self, data: &ExperimentalData) -> DeviceResult<ExperimentalData> {
        use scirs2_core::random::prelude::*;
        let mut rng = thread_rng();

        let data_size = data.measurement_results.len();
        let mut bootstrap_results = Vec::new();
        let mut bootstrap_uncertainties = Vec::new();

        for _ in 0..data_size {
            let idx = rng.gen_range(0..data_size);
            bootstrap_results.push(data.measurement_results[idx]);
            bootstrap_uncertainties.push(data.measurement_uncertainties[idx]);
        }

        Ok(ExperimentalData {
            input_states: data.input_states.clone(),
            measurement_operators: data.measurement_operators.clone(),
            measurement_results: bootstrap_results,
            measurement_uncertainties: bootstrap_uncertainties,
        })
    }

    /// Evaluate process quality
    fn evaluate_process_quality(
        &self,
        process_matrix: &Array4<Complex64>,
        validation_data: &ExperimentalData,
    ) -> DeviceResult<f64> {
        // Calculate log-likelihood on validation data
        let mut log_likelihood = 0.0;

        for (observed, &uncertainty) in validation_data
            .measurement_results
            .iter()
            .zip(validation_data.measurement_uncertainties.iter())
        {
            let predicted = 0.5; // Placeholder prediction
            let diff = observed - predicted;
            let variance = uncertainty * uncertainty;
            log_likelihood -= 0.5 * (diff * diff / variance);
        }

        Ok(log_likelihood)
    }

    /// Collect bootstrap metrics
    fn collect_bootstrap_metrics(
        &self,
        metrics: &ProcessMetrics,
        bootstrap_metrics: &mut HashMap<String, Vec<f64>>,
    ) {
        bootstrap_metrics
            .entry("process_fidelity".to_string())
            .or_insert_with(Vec::new)
            .push(metrics.process_fidelity);

        bootstrap_metrics
            .entry("average_gate_fidelity".to_string())
            .or_insert_with(Vec::new)
            .push(metrics.average_gate_fidelity);

        bootstrap_metrics
            .entry("unitarity".to_string())
            .or_insert_with(Vec::new)
            .push(metrics.unitarity);

        bootstrap_metrics
            .entry("entangling_power".to_string())
            .or_insert_with(Vec::new)
            .push(metrics.entangling_power);
    }

    /// Calculate bootstrap confidence intervals
    fn calculate_bootstrap_confidence_intervals(
        &self,
        bootstrap_metrics: &HashMap<String, Vec<f64>>,
    ) -> DeviceResult<HashMap<String, (f64, f64)>> {
        let mut confidence_intervals = HashMap::new();

        for (metric_name, values) in bootstrap_metrics {
            let mut sorted_values = values.clone();
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let n = sorted_values.len();
            let lower_idx = (0.025 * n as f64) as usize;
            let upper_idx = (0.975 * n as f64) as usize;

            let lower_bound = sorted_values[lower_idx.min(n - 1)];
            let upper_bound = sorted_values[upper_idx.min(n - 1)];

            confidence_intervals.insert(metric_name.clone(), (lower_bound, upper_bound));
        }

        Ok(confidence_intervals)
    }

    /// Calculate bootstrap bias estimates
    fn calculate_bootstrap_bias(
        &self,
        bootstrap_samples: &[ProcessMetrics],
    ) -> DeviceResult<HashMap<String, f64>> {
        let mut bias_estimates = HashMap::new();

        if bootstrap_samples.is_empty() {
            return Ok(bias_estimates);
        }

        // Calculate means
        let mean_fidelity = bootstrap_samples
            .iter()
            .map(|m| m.process_fidelity)
            .sum::<f64>()
            / bootstrap_samples.len() as f64;

        let mean_unitarity = bootstrap_samples.iter().map(|m| m.unitarity).sum::<f64>()
            / bootstrap_samples.len() as f64;

        // Bias is difference from theoretical expectation (assuming ideal = 1.0)
        bias_estimates.insert("process_fidelity".to_string(), mean_fidelity - 1.0);
        bias_estimates.insert("unitarity".to_string(), mean_unitarity - 1.0);

        Ok(bias_estimates)
    }

    /// Create benchmark process
    fn create_benchmark_process(&self, benchmark_name: &str) -> DeviceResult<Array4<Complex64>> {
        let dim = 2; // Assume single qubit for simplicity
        let mut process = Array4::zeros((dim, dim, dim, dim));

        match benchmark_name {
            "identity" => {
                // Identity process
                for i in 0..dim {
                    process[[i, i, i, i]] = Complex64::new(1.0, 0.0);
                }
            }
            "pauli_x" => {
                // Pauli-X process
                process[[0, 0, 1, 1]] = Complex64::new(1.0, 0.0);
                process[[1, 1, 0, 0]] = Complex64::new(1.0, 0.0);
                process[[0, 1, 1, 0]] = Complex64::new(1.0, 0.0);
                process[[1, 0, 0, 1]] = Complex64::new(1.0, 0.0);
            }
            "pauli_y" => {
                // Pauli-Y process
                process[[0, 0, 1, 1]] = Complex64::new(1.0, 0.0);
                process[[1, 1, 0, 0]] = Complex64::new(1.0, 0.0);
                process[[0, 1, 1, 0]] = Complex64::new(0.0, -1.0);
                process[[1, 0, 0, 1]] = Complex64::new(0.0, 1.0);
            }
            "pauli_z" => {
                // Pauli-Z process
                process[[0, 0, 0, 0]] = Complex64::new(1.0, 0.0);
                process[[1, 1, 1, 1]] = Complex64::new(-1.0, 0.0);
            }
            "hadamard" => {
                // Hadamard process
                let factor = 1.0 / (2.0_f64).sqrt();
                process[[0, 0, 0, 0]] = Complex64::new(factor, 0.0);
                process[[0, 0, 1, 1]] = Complex64::new(factor, 0.0);
                process[[1, 1, 0, 0]] = Complex64::new(factor, 0.0);
                process[[1, 1, 1, 1]] = Complex64::new(-factor, 0.0);
                process[[0, 1, 0, 1]] = Complex64::new(factor, 0.0);
                process[[0, 1, 1, 0]] = Complex64::new(factor, 0.0);
                process[[1, 0, 0, 1]] = Complex64::new(factor, 0.0);
                process[[1, 0, 1, 0]] = Complex64::new(-factor, 0.0);
            }
            _ => {
                // Default to identity
                for i in 0..dim {
                    process[[i, i, i, i]] = Complex64::new(1.0, 0.0);
                }
            }
        }

        Ok(process)
    }

    /// Calculate process fidelity between two processes
    fn calculate_process_fidelity_between(
        &self,
        process1: &Array4<Complex64>,
        process2: &Array4<Complex64>,
    ) -> DeviceResult<f64> {
        let dim = process1.dim().0;
        let mut fidelity = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        for i in 0..dim {
            for j in 0..dim {
                for k in 0..dim {
                    for l in 0..dim {
                        let element1 = process1[[i, j, k, l]];
                        let element2 = process2[[i, j, k, l]];

                        fidelity += (element1.conj() * element2).re;
                        norm1 += element1.norm_sqr();
                        norm2 += element2.norm_sqr();
                    }
                }
            }
        }

        if norm1 > 1e-12 && norm2 > 1e-12 {
            Ok(fidelity / (norm1 * norm2).sqrt())
        } else {
            Ok(0.0)
        }
    }

    /// Estimate number of parameters in process matrix
    fn estimate_num_parameters(&self, process_matrix: &Array4<Complex64>) -> usize {
        let dim = process_matrix.dim().0;
        // Complex process matrix has 2 * dim^4 real parameters
        // But with constraints (trace preservation, etc.), effective parameters are fewer
        2 * dim * dim * dim * dim - dim * dim // Subtract constraints
    }

    /// Calculate cross-validation score for specific method
    fn calculate_cv_score_for_method(
        &self,
        method: &str,
        experimental_data: &ExperimentalData,
    ) -> DeviceResult<f64> {
        // Simplified CV score calculation
        let (process_matrix, quality) = match method.as_ref() {
            "linear_inversion" => self.linear_inversion_reconstruction(experimental_data)?,
            "maximum_likelihood" => self.maximum_likelihood_reconstruction(experimental_data)?,
            "compressed_sensing" => self.compressed_sensing_reconstruction(experimental_data)?,
            "bayesian" => self.bayesian_reconstruction(experimental_data)?,
            _ => self.linear_inversion_reconstruction(experimental_data)?,
        };

        Ok(quality.log_likelihood)
    }
}
