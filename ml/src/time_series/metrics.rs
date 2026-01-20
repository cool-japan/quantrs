//! Performance metrics and evaluation tools for time series forecasting

use super::config::AnomalyType;
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive forecast metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastMetrics {
    /// Mean Absolute Error
    pub mae: f64,

    /// Mean Squared Error
    pub mse: f64,

    /// Root Mean Squared Error
    pub rmse: f64,

    /// Mean Absolute Percentage Error
    pub mape: f64,

    /// Symmetric MAPE
    pub smape: f64,

    /// Directional accuracy
    pub directional_accuracy: f64,

    /// Quantum fidelity of predictions
    pub quantum_fidelity: f64,

    /// Coverage of prediction intervals
    pub coverage: f64,

    /// Additional custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Forecast result with predictions and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    /// Point predictions
    pub predictions: Array2<f64>,

    /// Lower prediction interval
    pub lower_bound: Array2<f64>,

    /// Upper prediction interval
    pub upper_bound: Array2<f64>,

    /// Detected anomalies
    pub anomalies: Vec<AnomalyPoint>,

    /// Confidence scores for each prediction
    pub confidence_scores: Array1<f64>,

    /// Quantum uncertainty measure
    pub quantum_uncertainty: f64,
}

/// Anomaly point in forecasts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyPoint {
    /// Time index
    pub timestamp: usize,

    /// Anomalous value
    pub value: f64,

    /// Anomaly score
    pub anomaly_score: f64,

    /// Type of anomaly
    pub anomaly_type: AnomalyType,
}

/// Training history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Loss values per epoch
    pub losses: Vec<f64>,

    /// Validation losses per epoch
    pub val_losses: Vec<f64>,

    /// Metrics per epoch
    pub metrics: Vec<HashMap<String, f64>>,

    /// Best model parameters
    pub best_params: Option<Array1<f64>>,

    /// Total training time in seconds
    pub training_time: f64,

    /// Learning curves
    pub learning_curves: LearningCurves,
}

/// Learning curves for training analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningCurves {
    /// Training accuracy over epochs
    pub training_accuracy: Vec<f64>,

    /// Validation accuracy over epochs
    pub validation_accuracy: Vec<f64>,

    /// Learning rate schedule
    pub learning_rates: Vec<f64>,

    /// Quantum coherence measures
    pub quantum_coherence: Vec<f64>,
}

/// Model evaluation suite
#[derive(Debug, Clone)]
pub struct ModelEvaluator {
    /// Evaluation metrics to compute
    metrics: Vec<MetricType>,

    /// Cross-validation configuration
    cv_config: CrossValidationConfig,

    /// Statistical test configuration
    statistical_tests: StatisticalTestConfig,
}

/// Types of evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    MAE,
    MSE,
    RMSE,
    MAPE,
    SMAPE,
    DirectionalAccuracy,
    QuantumFidelity,
    Coverage,
    MSIS, // Mean Scaled Interval Score
    CRPS, // Continuous Ranked Probability Score
    Custom(String),
}

/// Cross-validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationConfig {
    /// Number of folds
    pub n_folds: usize,

    /// Validation strategy
    pub strategy: ValidationStrategy,

    /// Time series specific settings
    pub time_series_split: TimeSeriesSplitConfig,
}

/// Validation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStrategy {
    KFold,
    TimeSeriesSplit,
    WalkForward,
    BlockingTimeSeriesSplit,
    QuantumBootstrap,
}

/// Time series splitting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesSplitConfig {
    /// Test size as fraction of total data
    pub test_size: f64,

    /// Minimum training size
    pub min_train_size: Option<usize>,

    /// Gap between train and test
    pub gap: usize,

    /// Enable expanding window
    pub expanding_window: bool,
}

/// Statistical test configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestConfig {
    /// Significance level
    pub alpha: f64,

    /// Tests to perform
    pub tests: Vec<StatisticalTest>,

    /// Multiple comparison correction
    pub correction: MultipleComparisonCorrection,
}

/// Statistical tests for model comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalTest {
    DieboldMariano,
    WilcoxonSignedRank,
    PairedTTest,
    McNemar,
    QuantumSignificanceTest,
}

/// Multiple comparison corrections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultipleComparisonCorrection {
    None,
    Bonferroni,
    BenjaminiHochberg,
    Holm,
    QuantumCorrection,
}

/// Benchmark suite for model comparison
#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    /// Benchmark datasets
    datasets: Vec<BenchmarkDataset>,

    /// Models to compare
    models: Vec<String>,

    /// Evaluation metrics
    metrics: Vec<MetricType>,

    /// Results storage
    results: BenchmarkResults,
}

/// Benchmark dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkDataset {
    /// Dataset name
    pub name: String,

    /// Dataset description
    pub description: String,

    /// Data characteristics
    pub characteristics: DataCharacteristics,

    /// Source reference
    pub source: String,
}

/// Data characteristics for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCharacteristics {
    /// Number of time series
    pub n_series: usize,

    /// Length of time series
    pub series_length: usize,

    /// Frequency of observations
    pub frequency: String,

    /// Seasonality periods
    pub seasonality: Vec<usize>,

    /// Trend characteristics
    pub trend_type: TrendType,

    /// Noise level
    pub noise_level: f64,
}

/// Types of trends in data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendType {
    None,
    Linear,
    Exponential,
    Polynomial,
    Cyclic,
    Random,
    QuantumSuperposition,
}

/// Benchmark results storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Results by model and dataset
    pub results: HashMap<String, HashMap<String, ModelPerformance>>,

    /// Statistical comparisons
    pub statistical_comparisons: Vec<StatisticalComparison>,

    /// Rankings
    pub rankings: HashMap<String, Vec<String>>,

    /// Summary statistics
    pub summary: BenchmarkSummary,
}

/// Model performance on a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformance {
    /// Metric values
    pub metrics: HashMap<String, f64>,

    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>,

    /// Execution time
    pub execution_time: f64,

    /// Memory usage
    pub memory_usage: f64,

    /// Quantum enhancement factor
    pub quantum_enhancement: f64,
}

/// Statistical comparison between models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalComparison {
    /// Model names being compared
    pub models: (String, String),

    /// Test type
    pub test_type: StatisticalTest,

    /// Test statistic
    pub statistic: f64,

    /// p-value
    pub p_value: f64,

    /// Effect size
    pub effect_size: f64,

    /// Significant difference
    pub is_significant: bool,
}

/// Benchmark summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    /// Best performing model overall
    pub best_model: String,

    /// Average performance by model
    pub average_performance: HashMap<String, f64>,

    /// Win rates by model
    pub win_rates: HashMap<String, f64>,

    /// Quantum advantage metrics
    pub quantum_advantage: HashMap<String, f64>,
}

impl ForecastMetrics {
    /// Create new forecast metrics
    pub fn new() -> Self {
        Self {
            mae: 0.0,
            mse: 0.0,
            rmse: 0.0,
            mape: 0.0,
            smape: 0.0,
            directional_accuracy: 0.0,
            quantum_fidelity: 0.0,
            coverage: 0.0,
            custom_metrics: HashMap::new(),
        }
    }

    /// Calculate all metrics from predictions and actuals
    pub fn calculate_metrics(
        &mut self,
        predictions: &Array2<f64>,
        actuals: &Array2<f64>,
    ) -> Result<()> {
        if predictions.shape() != actuals.shape() {
            return Err(MLError::DimensionMismatch(
                "Predictions and actuals must have the same shape".to_string(),
            ));
        }

        let n = predictions.len() as f64;

        // Reset metrics
        self.mae = 0.0;
        self.mse = 0.0;
        self.mape = 0.0;
        self.smape = 0.0;

        // Calculate basic metrics
        for (pred, actual) in predictions.iter().zip(actuals.iter()) {
            let error = pred - actual;
            let abs_error = error.abs();

            self.mae += abs_error;
            self.mse += error * error;

            // MAPE calculation (avoid division by zero)
            if actual.abs() > 1e-10 {
                self.mape += (abs_error / actual.abs()) * 100.0;
            }

            // SMAPE calculation
            let denominator = (pred.abs() + actual.abs()) / 2.0;
            if denominator > 1e-10 {
                self.smape += (abs_error / denominator) * 100.0;
            }
        }

        // Normalize by number of observations
        self.mae /= n;
        self.mse /= n;
        self.mape /= n;
        self.smape /= n;

        // Calculate RMSE
        self.rmse = self.mse.sqrt();

        // Calculate directional accuracy
        self.directional_accuracy = self.calculate_directional_accuracy(predictions, actuals)?;

        // Calculate quantum fidelity (simplified)
        self.quantum_fidelity = self.calculate_quantum_fidelity(predictions, actuals)?;

        // Coverage would be calculated if prediction intervals are available
        self.coverage = 0.95; // Placeholder

        Ok(())
    }

    /// Calculate directional accuracy
    fn calculate_directional_accuracy(
        &self,
        predictions: &Array2<f64>,
        actuals: &Array2<f64>,
    ) -> Result<f64> {
        if predictions.nrows() < 2 {
            return Ok(0.0);
        }

        let mut correct_directions = 0;
        let mut total_directions = 0;

        for i in 1..predictions.nrows() {
            let pred_change = predictions[[i, 0]] - predictions[[i - 1, 0]];
            let actual_change = actuals[[i, 0]] - actuals[[i - 1, 0]];

            if pred_change * actual_change > 0.0 {
                correct_directions += 1;
            }
            total_directions += 1;
        }

        Ok(if total_directions > 0 {
            correct_directions as f64 / total_directions as f64
        } else {
            0.0
        })
    }

    /// Calculate quantum fidelity measure
    fn calculate_quantum_fidelity(
        &self,
        predictions: &Array2<f64>,
        actuals: &Array2<f64>,
    ) -> Result<f64> {
        // Simplified quantum fidelity based on normalized correlation
        let pred_flat: Vec<f64> = predictions.iter().cloned().collect();
        let actual_flat: Vec<f64> = actuals.iter().cloned().collect();

        let pred_mean = pred_flat.iter().sum::<f64>() / pred_flat.len() as f64;
        let actual_mean = actual_flat.iter().sum::<f64>() / actual_flat.len() as f64;

        let mut numerator = 0.0;
        let mut pred_sq_sum = 0.0;
        let mut actual_sq_sum = 0.0;

        for (pred, actual) in pred_flat.iter().zip(actual_flat.iter()) {
            let pred_dev = pred - pred_mean;
            let actual_dev = actual - actual_mean;

            numerator += pred_dev * actual_dev;
            pred_sq_sum += pred_dev * pred_dev;
            actual_sq_sum += actual_dev * actual_dev;
        }

        let denominator = (pred_sq_sum * actual_sq_sum).sqrt();
        let correlation = if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };

        // Convert correlation to fidelity-like measure
        Ok((correlation + 1.0) / 2.0)
    }

    /// Add custom metric
    pub fn add_custom_metric(&mut self, name: String, value: f64) {
        self.custom_metrics.insert(name, value);
    }

    /// Get summary of all metrics
    pub fn get_summary(&self) -> HashMap<String, f64> {
        let mut summary = HashMap::new();

        summary.insert("MAE".to_string(), self.mae);
        summary.insert("MSE".to_string(), self.mse);
        summary.insert("RMSE".to_string(), self.rmse);
        summary.insert("MAPE".to_string(), self.mape);
        summary.insert("SMAPE".to_string(), self.smape);
        summary.insert("DirectionalAccuracy".to_string(), self.directional_accuracy);
        summary.insert("QuantumFidelity".to_string(), self.quantum_fidelity);
        summary.insert("Coverage".to_string(), self.coverage);

        // Add custom metrics
        for (name, value) in &self.custom_metrics {
            summary.insert(name.clone(), *value);
        }

        summary
    }
}

impl TrainingHistory {
    /// Create new training history
    pub fn new() -> Self {
        Self {
            losses: Vec::new(),
            val_losses: Vec::new(),
            metrics: Vec::new(),
            best_params: None,
            training_time: 0.0,
            learning_curves: LearningCurves::new(),
        }
    }

    /// Add metrics for an epoch
    pub fn add_epoch_metrics(&mut self, metrics: HashMap<String, f64>, loss: f64, val_loss: f64) {
        self.losses.push(loss);
        self.val_losses.push(val_loss);
        self.metrics.push(metrics);

        // Update learning curves
        if let Some(accuracy) = self.metrics.last().and_then(|m| m.get("accuracy")) {
            self.learning_curves.training_accuracy.push(*accuracy);
        }

        if let Some(val_accuracy) = self.metrics.last().and_then(|m| m.get("val_accuracy")) {
            self.learning_curves.validation_accuracy.push(*val_accuracy);
        }
    }

    /// Get best epoch based on validation loss
    pub fn get_best_epoch(&self) -> Option<usize> {
        if self.val_losses.is_empty() {
            return None;
        }

        self.val_losses
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
    }

    /// Check if training is converged
    pub fn is_converged(&self, patience: usize, min_delta: f64) -> bool {
        if self.val_losses.len() < patience {
            return false;
        }

        let recent_losses = &self.val_losses[self.val_losses.len() - patience..];
        let min_recent = recent_losses.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        // Check if there's been improvement recently
        recent_losses
            .iter()
            .any(|&loss| min_recent - loss > min_delta)
    }
}

impl LearningCurves {
    /// Create new learning curves
    pub fn new() -> Self {
        Self {
            training_accuracy: Vec::new(),
            validation_accuracy: Vec::new(),
            learning_rates: Vec::new(),
            quantum_coherence: Vec::new(),
        }
    }

    /// Add learning rate
    pub fn add_learning_rate(&mut self, lr: f64) {
        self.learning_rates.push(lr);
    }

    /// Add quantum coherence measure
    pub fn add_quantum_coherence(&mut self, coherence: f64) {
        self.quantum_coherence.push(coherence);
    }
}

impl ModelEvaluator {
    /// Create new model evaluator
    pub fn new() -> Self {
        Self {
            metrics: vec![
                MetricType::MAE,
                MetricType::MSE,
                MetricType::RMSE,
                MetricType::MAPE,
                MetricType::DirectionalAccuracy,
            ],
            cv_config: CrossValidationConfig::default(),
            statistical_tests: StatisticalTestConfig::default(),
        }
    }

    /// Evaluate model performance
    pub fn evaluate(
        &self,
        predictions: &Array2<f64>,
        actuals: &Array2<f64>,
    ) -> Result<HashMap<String, f64>> {
        let mut results = HashMap::new();

        for metric_type in &self.metrics {
            let value = self.calculate_metric(metric_type, predictions, actuals)?;
            let metric_name = format!("{:?}", metric_type);
            results.insert(metric_name, value);
        }

        Ok(results)
    }

    /// Calculate specific metric
    fn calculate_metric(
        &self,
        metric_type: &MetricType,
        predictions: &Array2<f64>,
        actuals: &Array2<f64>,
    ) -> Result<f64> {
        match metric_type {
            MetricType::MAE => self.calculate_mae(predictions, actuals),
            MetricType::MSE => self.calculate_mse(predictions, actuals),
            MetricType::RMSE => {
                let mse = self.calculate_mse(predictions, actuals)?;
                Ok(mse.sqrt())
            }
            MetricType::MAPE => self.calculate_mape(predictions, actuals),
            MetricType::SMAPE => self.calculate_smape(predictions, actuals),
            MetricType::DirectionalAccuracy => {
                self.calculate_directional_accuracy(predictions, actuals)
            }
            MetricType::QuantumFidelity => self.calculate_quantum_fidelity(predictions, actuals),
            _ => Ok(0.0), // Placeholder for other metrics
        }
    }

    /// Calculate Mean Absolute Error
    fn calculate_mae(&self, predictions: &Array2<f64>, actuals: &Array2<f64>) -> Result<f64> {
        if predictions.shape() != actuals.shape() {
            return Err(MLError::DimensionMismatch(
                "Predictions and actuals must have same shape".to_string(),
            ));
        }

        let mae = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(p, a)| (p - a).abs())
            .sum::<f64>()
            / predictions.len() as f64;

        Ok(mae)
    }

    /// Calculate Mean Squared Error
    fn calculate_mse(&self, predictions: &Array2<f64>, actuals: &Array2<f64>) -> Result<f64> {
        if predictions.shape() != actuals.shape() {
            return Err(MLError::DimensionMismatch(
                "Predictions and actuals must have same shape".to_string(),
            ));
        }

        let mse = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;

        Ok(mse)
    }

    /// Calculate Mean Absolute Percentage Error
    fn calculate_mape(&self, predictions: &Array2<f64>, actuals: &Array2<f64>) -> Result<f64> {
        if predictions.shape() != actuals.shape() {
            return Err(MLError::DimensionMismatch(
                "Predictions and actuals must have same shape".to_string(),
            ));
        }

        let mut mape_sum = 0.0;
        let mut count = 0;

        for (pred, actual) in predictions.iter().zip(actuals.iter()) {
            if actual.abs() > 1e-10 {
                mape_sum += ((pred - actual) / actual).abs() * 100.0;
                count += 1;
            }
        }

        Ok(if count > 0 {
            mape_sum / count as f64
        } else {
            0.0
        })
    }

    /// Calculate Symmetric Mean Absolute Percentage Error
    fn calculate_smape(&self, predictions: &Array2<f64>, actuals: &Array2<f64>) -> Result<f64> {
        if predictions.shape() != actuals.shape() {
            return Err(MLError::DimensionMismatch(
                "Predictions and actuals must have same shape".to_string(),
            ));
        }

        let smape = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(pred, actual)| {
                let numerator = (pred - actual).abs();
                let denominator = (pred.abs() + actual.abs()) / 2.0;
                if denominator > 1e-10 {
                    numerator / denominator * 100.0
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / predictions.len() as f64;

        Ok(smape)
    }

    /// Calculate directional accuracy
    fn calculate_directional_accuracy(
        &self,
        predictions: &Array2<f64>,
        actuals: &Array2<f64>,
    ) -> Result<f64> {
        if predictions.nrows() < 2 {
            return Ok(0.0);
        }

        let mut correct = 0;
        let mut total = 0;

        for i in 1..predictions.nrows() {
            let pred_change = predictions[[i, 0]] - predictions[[i - 1, 0]];
            let actual_change = actuals[[i, 0]] - actuals[[i - 1, 0]];

            if pred_change * actual_change > 0.0 {
                correct += 1;
            }
            total += 1;
        }

        Ok(correct as f64 / total as f64)
    }

    /// Calculate quantum fidelity
    fn calculate_quantum_fidelity(
        &self,
        predictions: &Array2<f64>,
        actuals: &Array2<f64>,
    ) -> Result<f64> {
        // Simplified quantum fidelity calculation
        let mae = self.calculate_mae(predictions, actuals)?;
        let max_possible_error = actuals.iter().map(|x| x.abs()).fold(0.0, f64::max);

        let fidelity = if max_possible_error > 1e-10 {
            1.0 - (mae / max_possible_error).min(1.0)
        } else {
            1.0
        };

        Ok(fidelity)
    }
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            strategy: ValidationStrategy::TimeSeriesSplit,
            time_series_split: TimeSeriesSplitConfig::default(),
        }
    }
}

impl Default for TimeSeriesSplitConfig {
    fn default() -> Self {
        Self {
            test_size: 0.2,
            min_train_size: None,
            gap: 0,
            expanding_window: false,
        }
    }
}

impl Default for StatisticalTestConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            tests: vec![StatisticalTest::DieboldMariano],
            correction: MultipleComparisonCorrection::BenjaminiHochberg,
        }
    }
}

/// Utility functions for metrics calculation

/// Calculate prediction interval coverage
pub fn calculate_coverage(
    actuals: &Array2<f64>,
    lower_bounds: &Array2<f64>,
    upper_bounds: &Array2<f64>,
) -> Result<f64> {
    if actuals.shape() != lower_bounds.shape() || actuals.shape() != upper_bounds.shape() {
        return Err(MLError::DimensionMismatch(
            "All arrays must have the same shape".to_string(),
        ));
    }

    let mut covered = 0;
    let total = actuals.len();

    for ((actual, lower), upper) in actuals
        .iter()
        .zip(lower_bounds.iter())
        .zip(upper_bounds.iter())
    {
        if actual >= lower && actual <= upper {
            covered += 1;
        }
    }

    Ok(covered as f64 / total as f64)
}

/// Calculate Mean Scaled Interval Score (MSIS)
pub fn calculate_msis(
    actuals: &Array2<f64>,
    predictions: &Array2<f64>,
    lower_bounds: &Array2<f64>,
    upper_bounds: &Array2<f64>,
    alpha: f64,
    seasonal_period: Option<usize>,
) -> Result<f64> {
    // Simplified MSIS calculation
    let coverage = calculate_coverage(actuals, lower_bounds, upper_bounds)?;
    let interval_width: f64 = upper_bounds
        .iter()
        .zip(lower_bounds.iter())
        .map(|(u, l)| u - l)
        .sum::<f64>()
        / upper_bounds.len() as f64;

    // Simplified scaling factor
    let scaling_factor = if let Some(period) = seasonal_period {
        period as f64
    } else {
        1.0
    };

    Ok(interval_width / scaling_factor * (1.0 - coverage + alpha))
}
