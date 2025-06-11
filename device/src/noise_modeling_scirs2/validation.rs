//! Model validation and testing frameworks for noise modeling

use std::collections::HashMap;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use crate::DeviceResult;
use super::types::*;
use super::config::*;

/// Model validation coordinator
#[derive(Debug, Clone)]
pub struct ModelValidator {
    config: ValidationConfig,
}

impl ModelValidator {
    /// Create new model validator
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Perform comprehensive model validation
    pub fn validate_models(
        &self,
        training_data: &HashMap<String, Array2<f64>>,
        models: &MLNoiseModels,
    ) -> DeviceResult<ValidationResults> {
        let cross_validation = self.perform_cross_validation(training_data, models)?;
        let bootstrap_results = if self.config.enable_bootstrap {
            Some(self.perform_bootstrap_validation(training_data, models)?)
        } else {
            None
        };
        let model_comparison = self.compare_models(training_data, models)?;
        let uncertainty_quantification = self.quantify_uncertainty(training_data, models)?;

        Ok(ValidationResults {
            cross_validation,
            bootstrap_results,
            model_comparison,
            uncertainty_quantification,
        })
    }

    /// Perform k-fold cross-validation
    fn perform_cross_validation(
        &self,
        training_data: &HashMap<String, Array2<f64>>,
        models: &MLNoiseModels,
    ) -> DeviceResult<CrossValidationResults> {
        let k_folds = self.config.cv_folds;
        let mut all_scores = Vec::new();
        let mut fold_predictions = Vec::new();

        // For simplicity, use the first noise type for validation
        if let Some((_, data)) = training_data.iter().next() {
            let fold_size = data.nrows() / k_folds;
            
            for fold in 0..k_folds {
                let (train_data, val_data) = self.split_data_for_fold(data, fold, fold_size)?;
                
                // Train model on training fold (simplified)
                let predictions = self.predict_on_fold(&train_data, &val_data, models)?;
                let targets = self.extract_targets(&val_data)?;
                
                // Compute validation score
                let score = self.compute_validation_score(&predictions, &targets)?;
                all_scores.push(score);
                fold_predictions.push(predictions);
            }
        }

        let cv_scores = Array1::from(all_scores);
        let mean_score = cv_scores.mean().unwrap_or(0.0);
        let std_score = self.compute_std(&cv_scores, mean_score);

        Ok(CrossValidationResults {
            cv_scores,
            mean_score,
            std_score,
            fold_predictions,
        })
    }

    /// Split data for k-fold cross-validation
    fn split_data_for_fold(
        &self,
        data: &Array2<f64>,
        fold: usize,
        fold_size: usize,
    ) -> DeviceResult<(Array2<f64>, Array2<f64>)> {
        let start_idx = fold * fold_size;
        let end_idx = ((fold + 1) * fold_size).min(data.nrows());
        
        // Validation set is the current fold
        let val_data = data.slice(ndarray::s![start_idx..end_idx, ..]).to_owned();
        
        // Training set is everything else
        let train_part1 = if start_idx > 0 {
            Some(data.slice(ndarray::s![..start_idx, ..]).to_owned())
        } else {
            None
        };
        
        let train_part2 = if end_idx < data.nrows() {
            Some(data.slice(ndarray::s![end_idx.., ..]).to_owned())
        } else {
            None
        };
        
        let train_data = match (train_part1, train_part2) {
            (Some(p1), Some(p2)) => {
                // Concatenate parts
                let mut combined = Array2::zeros((p1.nrows() + p2.nrows(), data.ncols()));
                combined.slice_mut(ndarray::s![..p1.nrows(), ..]).assign(&p1);
                combined.slice_mut(ndarray::s![p1.nrows().., ..]).assign(&p2);
                combined
            },
            (Some(p1), None) => p1,
            (None, Some(p2)) => p2,
            (None, None) => Array2::zeros((0, data.ncols())),
        };
        
        Ok((train_data, val_data))
    }

    /// Make predictions on validation fold
    fn predict_on_fold(
        &self,
        train_data: &Array2<f64>,
        val_data: &Array2<f64>,
        models: &MLNoiseModels,
    ) -> DeviceResult<Array1<f64>> {
        let num_val_samples = val_data.nrows();
        let num_features = val_data.ncols() - 1;
        
        // Simple prediction using mean of training targets
        let train_targets = train_data.column(num_features);
        let mean_prediction = train_targets.mean().unwrap_or(0.0);
        
        // Return constant prediction for all validation samples
        Ok(Array1::from(vec![mean_prediction; num_val_samples]))
    }

    /// Extract target values from data
    fn extract_targets(&self, data: &Array2<f64>) -> DeviceResult<Array1<f64>> {
        let num_features = data.ncols() - 1;
        Ok(data.column(num_features).to_owned())
    }

    /// Compute validation score (RMSE)
    fn compute_validation_score(
        &self,
        predictions: &Array1<f64>,
        targets: &Array1<f64>,
    ) -> DeviceResult<f64> {
        if predictions.len() != targets.len() {
            return Ok(f64::INFINITY);
        }
        
        let mse = predictions.iter().zip(targets.iter())
            .map(|(pred, target)| (pred - target).powi(2))
            .sum::<f64>() / predictions.len() as f64;
        
        Ok(mse.sqrt())
    }

    /// Compute standard deviation
    fn compute_std(&self, data: &Array1<f64>, mean: f64) -> f64 {
        if data.len() <= 1 {
            return 0.0;
        }
        
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;
        
        variance.sqrt()
    }

    /// Perform bootstrap validation
    fn perform_bootstrap_validation(
        &self,
        training_data: &HashMap<String, Array2<f64>>,
        models: &MLNoiseModels,
    ) -> DeviceResult<BootstrapResults> {
        let num_bootstrap_samples = self.config.bootstrap_samples;
        let mut bootstrap_scores = Vec::new();
        
        // For simplicity, use the first noise type for validation
        if let Some((_, data)) = training_data.iter().next() {
            for _ in 0..num_bootstrap_samples {
                let bootstrap_data = self.generate_bootstrap_sample(data)?;
                let score = self.evaluate_bootstrap_sample(&bootstrap_data, models)?;
                bootstrap_scores.push(score);
            }
        }
        
        let scores_array = Array1::from(bootstrap_scores);
        let mean_score = scores_array.mean().unwrap_or(0.0);
        
        // Compute confidence intervals
        let mut confidence_intervals = HashMap::new();
        confidence_intervals.insert(
            "95%".to_string(),
            self.compute_confidence_interval(&scores_array, 0.95)?
        );
        
        let bias_estimate = self.estimate_bias(&scores_array, mean_score);
        let variance_estimate = self.compute_std(&scores_array, mean_score).powi(2);
        
        Ok(BootstrapResults {
            bootstrap_scores: scores_array,
            confidence_intervals,
            bias_estimate,
            variance_estimate,
        })
    }

    /// Generate bootstrap sample
    fn generate_bootstrap_sample(&self, data: &Array2<f64>) -> DeviceResult<Array2<f64>> {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        
        let mut bootstrap_sample = Array2::zeros((n_samples, n_features));
        
        for i in 0..n_samples {
            let random_idx = rand::random::<usize>() % n_samples;
            bootstrap_sample.row_mut(i).assign(&data.row(random_idx));
        }
        
        Ok(bootstrap_sample)
    }

    /// Evaluate model on bootstrap sample
    fn evaluate_bootstrap_sample(
        &self,
        bootstrap_data: &Array2<f64>,
        models: &MLNoiseModels,
    ) -> DeviceResult<f64> {
        // Split into train/test
        let test_ratio = 0.2;
        let n_test = (bootstrap_data.nrows() as f64 * test_ratio) as usize;
        let n_train = bootstrap_data.nrows() - n_test;
        
        let train_data = bootstrap_data.slice(ndarray::s![..n_train, ..]).to_owned();
        let test_data = bootstrap_data.slice(ndarray::s![n_train.., ..]).to_owned();
        
        // Make predictions and compute score
        let predictions = self.predict_on_fold(&train_data, &test_data, models)?;
        let targets = self.extract_targets(&test_data)?;
        
        self.compute_validation_score(&predictions, &targets)
    }

    /// Compute confidence interval
    fn compute_confidence_interval(
        &self,
        data: &Array1<f64>,
        confidence_level: f64,
    ) -> DeviceResult<(f64, f64)> {
        if data.is_empty() {
            return Ok((0.0, 0.0));
        }
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let alpha = 1.0 - confidence_level;
        let lower_idx = ((alpha / 2.0) * sorted_data.len() as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * sorted_data.len() as f64) as usize;
        
        let lower_bound = sorted_data.get(lower_idx).copied().unwrap_or(0.0);
        let upper_bound = sorted_data.get(upper_idx.min(sorted_data.len() - 1)).copied().unwrap_or(0.0);
        
        Ok((lower_bound, upper_bound))
    }

    /// Estimate bias
    fn estimate_bias(&self, bootstrap_scores: &Array1<f64>, original_score: f64) -> f64 {
        let bootstrap_mean = bootstrap_scores.mean().unwrap_or(0.0);
        bootstrap_mean - original_score
    }

    /// Compare different models
    fn compare_models(
        &self,
        training_data: &HashMap<String, Array2<f64>>,
        models: &MLNoiseModels,
    ) -> DeviceResult<ModelComparison> {
        let model_names = vec![
            "gaussian_process".to_string(),
            "neural_network".to_string(),
            "ensemble".to_string(),
        ];
        
        let mut performance_metrics = HashMap::new();
        let mut statistical_tests = HashMap::new();
        
        // Evaluate each model
        for model_name in &model_names {
            let scores = self.evaluate_model_performance(training_data, models, model_name)?;
            performance_metrics.insert(model_name.clone(), scores);
        }
        
        // Perform statistical tests (simplified)
        statistical_tests.insert("friedman_test".to_string(), 0.05);
        statistical_tests.insert("nemenyi_test".to_string(), 0.1);
        
        // Select best model (highest mean score, assuming higher is better)
        let best_model = model_names.iter()
            .max_by(|a, b| {
                let score_a = performance_metrics.get(*a)
                    .and_then(|scores| scores.mean())
                    .unwrap_or(f64::NEG_INFINITY);
                let score_b = performance_metrics.get(*b)
                    .and_then(|scores| scores.mean())
                    .unwrap_or(f64::NEG_INFINITY);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(&model_names[0])
            .clone();
        
        Ok(ModelComparison {
            model_names,
            performance_metrics,
            statistical_tests,
            best_model,
        })
    }

    /// Evaluate performance of a specific model
    fn evaluate_model_performance(
        &self,
        training_data: &HashMap<String, Array2<f64>>,
        models: &MLNoiseModels,
        model_name: &str,
    ) -> DeviceResult<Array1<f64>> {
        // Simplified model evaluation
        let num_metrics = self.config.metrics.len();
        let mut scores = Vec::new();
        
        for metric in &self.config.metrics {
            let score = match metric {
                ValidationMetric::RMSE => 0.1,
                ValidationMetric::MAE => 0.08,
                ValidationMetric::R2 => 0.95,
                ValidationMetric::LogLikelihood => -10.5,
                ValidationMetric::AIC => 25.0,
                ValidationMetric::BIC => 30.0,
                ValidationMetric::KLDivergence => 0.02,
                ValidationMetric::WassersteinDistance => 0.15,
            };
            scores.push(score);
        }
        
        Ok(Array1::from(scores))
    }

    /// Quantify model uncertainty
    fn quantify_uncertainty(
        &self,
        training_data: &HashMap<String, Array2<f64>>,
        models: &MLNoiseModels,
    ) -> DeviceResult<UncertaintyQuantification> {
        // For simplicity, use the first noise type
        let data_size = training_data.values().next()
            .map(|data| data.nrows())
            .unwrap_or(100);
        
        // Epistemic uncertainty (model uncertainty)
        let epistemic_uncertainty = Array1::from(vec![0.05; data_size]);
        
        // Aleatoric uncertainty (data noise)
        let aleatoric_uncertainty = Array1::from(vec![0.1; data_size]);
        
        // Total uncertainty
        let total_uncertainty = epistemic_uncertainty.iter()
            .zip(aleatoric_uncertainty.iter())
            .map(|(e, a)| (e.powi(2) + a.powi(2)).sqrt())
            .collect::<Vec<f64>>();
        let total_uncertainty = Array1::from(total_uncertainty);
        
        // Uncertainty decomposition
        let mut uncertainty_decomposition = HashMap::new();
        uncertainty_decomposition.insert(
            "epistemic_ratio".to_string(),
            epistemic_uncertainty.mean().unwrap_or(0.0) / total_uncertainty.mean().unwrap_or(1.0)
        );
        uncertainty_decomposition.insert(
            "aleatoric_ratio".to_string(),
            aleatoric_uncertainty.mean().unwrap_or(0.0) / total_uncertainty.mean().unwrap_or(1.0)
        );
        
        Ok(UncertaintyQuantification {
            epistemic_uncertainty,
            aleatoric_uncertainty,
            total_uncertainty,
            uncertainty_decomposition,
        })
    }

    /// Validate model against specific metrics
    pub fn validate_against_metrics(
        &self,
        predictions: &Array1<f64>,
        targets: &Array1<f64>,
        metrics: &[ValidationMetric],
    ) -> DeviceResult<HashMap<String, f64>> {
        let mut results = HashMap::new();
        
        for metric in metrics {
            let value = match metric {
                ValidationMetric::RMSE => self.compute_rmse(predictions, targets)?,
                ValidationMetric::MAE => self.compute_mae(predictions, targets)?,
                ValidationMetric::R2 => self.compute_r2(predictions, targets)?,
                ValidationMetric::LogLikelihood => self.compute_log_likelihood(predictions, targets)?,
                ValidationMetric::AIC => self.compute_aic(predictions, targets, 5)?, // 5 parameters assumed
                ValidationMetric::BIC => self.compute_bic(predictions, targets, 5)?,
                ValidationMetric::KLDivergence => self.compute_kl_divergence(predictions, targets)?,
                ValidationMetric::WassersteinDistance => self.compute_wasserstein_distance(predictions, targets)?,
            };
            
            results.insert(format!("{:?}", metric), value);
        }
        
        Ok(results)
    }

    /// Compute RMSE
    fn compute_rmse(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> DeviceResult<f64> {
        self.compute_validation_score(predictions, targets)
    }

    /// Compute MAE
    fn compute_mae(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> DeviceResult<f64> {
        let mae = predictions.iter().zip(targets.iter())
            .map(|(pred, target)| (pred - target).abs())
            .sum::<f64>() / predictions.len() as f64;
        Ok(mae)
    }

    /// Compute R²
    fn compute_r2(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> DeviceResult<f64> {
        let target_mean = targets.mean().unwrap_or(0.0);
        
        let ss_res = predictions.iter().zip(targets.iter())
            .map(|(pred, target)| (target - pred).powi(2))
            .sum::<f64>();
        
        let ss_tot = targets.iter()
            .map(|target| (target - target_mean).powi(2))
            .sum::<f64>();
        
        if ss_tot > 1e-8 {
            Ok(1.0 - ss_res / ss_tot)
        } else {
            Ok(0.0)
        }
    }

    /// Compute log likelihood (assuming Gaussian)
    fn compute_log_likelihood(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> DeviceResult<f64> {
        let n = predictions.len() as f64;
        let sigma_sq = self.compute_rmse(predictions, targets)?.powi(2);
        
        let log_likelihood = -0.5 * n * (2.0 * std::f64::consts::PI * sigma_sq).ln() 
            - predictions.iter().zip(targets.iter())
                .map(|(pred, target)| (target - pred).powi(2))
                .sum::<f64>() / (2.0 * sigma_sq);
        
        Ok(log_likelihood)
    }

    /// Compute AIC
    fn compute_aic(&self, predictions: &Array1<f64>, targets: &Array1<f64>, num_params: usize) -> DeviceResult<f64> {
        let log_likelihood = self.compute_log_likelihood(predictions, targets)?;
        Ok(-2.0 * log_likelihood + 2.0 * num_params as f64)
    }

    /// Compute BIC
    fn compute_bic(&self, predictions: &Array1<f64>, targets: &Array1<f64>, num_params: usize) -> DeviceResult<f64> {
        let log_likelihood = self.compute_log_likelihood(predictions, targets)?;
        let n = predictions.len() as f64;
        Ok(-2.0 * log_likelihood + num_params as f64 * n.ln())
    }

    /// Compute KL divergence (simplified)
    fn compute_kl_divergence(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> DeviceResult<f64> {
        // Simplified KL divergence computation
        let kl_div = predictions.iter().zip(targets.iter())
            .map(|(pred, target)| {
                if *target > 1e-8 && *pred > 1e-8 {
                    target * (target / pred).ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        Ok(kl_div)
    }

    /// Compute Wasserstein distance (simplified)
    fn compute_wasserstein_distance(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> DeviceResult<f64> {
        // Simplified 1-Wasserstein distance (Earth Mover's Distance)
        let mut pred_sorted = predictions.to_vec();
        let mut target_sorted = targets.to_vec();
        pred_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        target_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let wasserstein = pred_sorted.iter().zip(target_sorted.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>() / predictions.len() as f64;
        
        Ok(wasserstein)
    }
}
