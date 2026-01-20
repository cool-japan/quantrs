//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2};
/// Calculate calibration curve (reliability diagram)
/// Returns (mean_predicted_prob, fraction_of_positives) for each bin
pub fn calibration_curve(
    probabilities: &Array1<f64>,
    labels: &Array1<usize>,
    n_bins: usize,
) -> Result<(Array1<f64>, Array1<f64>)> {
    if probabilities.len() != labels.len() {
        return Err(MLError::InvalidInput(
            "Probabilities and labels must have same length".to_string(),
        ));
    }
    if n_bins < 2 {
        return Err(MLError::InvalidInput(
            "Number of bins must be at least 2".to_string(),
        ));
    }
    let mut bins = vec![Vec::new(); n_bins];
    for (i, &prob) in probabilities.iter().enumerate() {
        let bin_idx = ((prob * n_bins as f64).floor() as usize).min(n_bins - 1);
        bins[bin_idx].push((prob, labels[i]));
    }
    let mut mean_predicted = Vec::new();
    let mut fraction_positives = Vec::new();
    for bin in bins {
        if !bin.is_empty() {
            let sum_prob: f64 = bin.iter().map(|(p, _)| p).sum();
            let sum_labels: f64 = bin.iter().map(|(_, l)| *l as f64).sum();
            mean_predicted.push(sum_prob / bin.len() as f64);
            fraction_positives.push(sum_labels / bin.len() as f64);
        }
    }
    Ok((
        Array1::from_vec(mean_predicted),
        Array1::from_vec(fraction_positives),
    ))
}
/// Calibration visualization and analysis utilities
pub mod visualization {
    use super::*;
    /// Calibration plot data for reliability diagrams
    #[derive(Debug, Clone)]
    pub struct CalibrationPlotData {
        /// Mean predicted probabilities in each bin
        pub mean_predicted: Array1<f64>,
        /// Fraction of positives in each bin
        pub fraction_positives: Array1<f64>,
        /// Number of samples in each bin
        pub bin_counts: Array1<usize>,
        /// Bin edges
        pub bin_edges: Vec<f64>,
    }
    /// Generate comprehensive calibration plot data
    pub fn generate_calibration_plot_data(
        probabilities: &Array1<f64>,
        labels: &Array1<usize>,
        n_bins: usize,
    ) -> Result<CalibrationPlotData> {
        if probabilities.len() != labels.len() {
            return Err(MLError::InvalidInput(
                "Probabilities and labels must have same length".to_string(),
            ));
        }
        if n_bins < 2 {
            return Err(MLError::InvalidInput(
                "Number of bins must be at least 2".to_string(),
            ));
        }
        let mut bins = vec![Vec::new(); n_bins];
        let bin_edges: Vec<f64> = (0..=n_bins).map(|i| i as f64 / n_bins as f64).collect();
        for (i, &prob) in probabilities.iter().enumerate() {
            let bin_idx = ((prob * n_bins as f64).floor() as usize).min(n_bins - 1);
            bins[bin_idx].push((prob, labels[i]));
        }
        let mut mean_predicted = Vec::new();
        let mut fraction_positives = Vec::new();
        let mut bin_counts = Vec::new();
        for bin in bins {
            if !bin.is_empty() {
                let sum_prob: f64 = bin.iter().map(|(p, _)| p).sum();
                let sum_labels: f64 = bin.iter().map(|(_, l)| *l as f64).sum();
                mean_predicted.push(sum_prob / bin.len() as f64);
                fraction_positives.push(sum_labels / bin.len() as f64);
                bin_counts.push(bin.len());
            } else {
                mean_predicted.push(
                    (bin_edges[mean_predicted.len()] + bin_edges[mean_predicted.len() + 1]) / 2.0,
                );
                fraction_positives.push(0.0);
                bin_counts.push(0);
            }
        }
        Ok(CalibrationPlotData {
            mean_predicted: Array1::from_vec(mean_predicted),
            fraction_positives: Array1::from_vec(fraction_positives),
            bin_counts: Array1::from_vec(bin_counts),
            bin_edges,
        })
    }
    /// Comprehensive calibration analysis report
    #[derive(Debug, Clone)]
    pub struct CalibrationAnalysis {
        /// Expected Calibration Error
        pub ece: f64,
        /// Maximum Calibration Error
        pub mce: f64,
        /// Brier score
        pub brier_score: f64,
        /// Negative log-likelihood
        pub nll: f64,
        /// Number of bins used
        pub n_bins: usize,
        /// Per-bin calibration errors
        pub bin_errors: Array1<f64>,
        /// Interpretation of calibration quality
        pub interpretation: String,
    }
    impl CalibrationAnalysis {
        /// Generate interpretation based on ECE
        fn interpret_ece(ece: f64) -> String {
            if ece < 0.01 {
                "Excellent calibration - predictions are highly reliable".to_string()
            } else if ece < 0.05 {
                "Good calibration - predictions are generally reliable".to_string()
            } else if ece < 0.10 {
                "Moderate calibration - some miscalibration present".to_string()
            } else if ece < 0.20 {
                "Poor calibration - significant miscalibration detected".to_string()
            } else {
                "Very poor calibration - predictions are unreliable".to_string()
            }
        }
    }
    /// Perform comprehensive calibration analysis
    pub fn analyze_calibration(
        probabilities: &Array1<f64>,
        labels: &Array1<usize>,
        n_bins: usize,
    ) -> Result<CalibrationAnalysis> {
        let plot_data = generate_calibration_plot_data(probabilities, labels, n_bins)?;
        let mut ece = 0.0;
        let total_samples = probabilities.len() as f64;
        for i in 0..plot_data.mean_predicted.len() {
            let bin_error = (plot_data.mean_predicted[i] - plot_data.fraction_positives[i]).abs();
            let bin_weight = plot_data.bin_counts[i] as f64 / total_samples;
            ece += bin_weight * bin_error;
        }
        let bin_errors: Array1<f64> =
            (&plot_data.mean_predicted - &plot_data.fraction_positives).mapv(|x| x.abs());
        let mce = bin_errors.iter().cloned().fold(0.0f64, f64::max);
        let mut brier_score = 0.0;
        for (i, &prob) in probabilities.iter().enumerate() {
            let true_label = labels[i] as f64;
            brier_score += (prob - true_label).powi(2);
        }
        brier_score /= probabilities.len() as f64;
        let mut nll = 0.0;
        for (i, &prob) in probabilities.iter().enumerate() {
            let true_label = labels[i];
            let prob_clamped = prob.max(1e-10).min(1.0 - 1e-10);
            if true_label == 1 {
                nll -= prob_clamped.ln();
            } else {
                nll -= (1.0 - prob_clamped).ln();
            }
        }
        nll /= probabilities.len() as f64;
        let interpretation = CalibrationAnalysis::interpret_ece(ece);
        Ok(CalibrationAnalysis {
            ece,
            mce,
            brier_score,
            nll,
            n_bins,
            bin_errors,
            interpretation,
        })
    }
    /// Compare multiple calibration methods
    #[derive(Debug, Clone)]
    pub struct CalibrationComparison {
        /// Method name
        pub method_name: String,
        /// Calibration analysis
        pub analysis: CalibrationAnalysis,
        /// Calibrated probabilities
        pub calibrated_probs: Array1<f64>,
    }
    /// Compare multiple calibration methods on the same dataset
    pub fn compare_calibration_methods(
        uncalibrated_probs: &Array1<f64>,
        labels: &Array1<usize>,
        n_bins: usize,
    ) -> Result<Vec<CalibrationComparison>> {
        let mut comparisons = Vec::new();
        let uncal_analysis = analyze_calibration(uncalibrated_probs, labels, n_bins)?;
        comparisons.push(CalibrationComparison {
            method_name: "Uncalibrated".to_string(),
            analysis: uncal_analysis,
            calibrated_probs: uncalibrated_probs.clone(),
        });
        if labels.iter().max().unwrap_or(&0) == &1 {
            let mut platt = PlattScaler::new();
            if let Ok(calibrated) = platt.fit_transform(uncalibrated_probs, labels) {
                let analysis = analyze_calibration(&calibrated, labels, n_bins)?;
                comparisons.push(CalibrationComparison {
                    method_name: "Platt Scaling".to_string(),
                    analysis,
                    calibrated_probs: calibrated,
                });
            }
        }
        if labels.iter().max().unwrap_or(&0) == &1 {
            let mut isotonic = IsotonicRegression::new();
            if let Ok(calibrated) = isotonic.fit_transform(uncalibrated_probs, labels) {
                let analysis = analyze_calibration(&calibrated, labels, n_bins)?;
                comparisons.push(CalibrationComparison {
                    method_name: "Isotonic Regression".to_string(),
                    analysis,
                    calibrated_probs: calibrated,
                });
            }
        }
        if labels.iter().max().unwrap_or(&0) == &1 {
            let mut bbq = BayesianBinningQuantiles::new(n_bins);
            if let Ok(calibrated) = bbq.fit_transform(uncalibrated_probs, labels) {
                let analysis = analyze_calibration(&calibrated, labels, n_bins)?;
                comparisons.push(CalibrationComparison {
                    method_name: "Bayesian Binning (BBQ)".to_string(),
                    analysis,
                    calibrated_probs: calibrated,
                });
            }
        }
        Ok(comparisons)
    }
    /// Generate a text report comparing calibration methods
    pub fn generate_comparison_report(comparisons: &[CalibrationComparison]) -> String {
        let mut report = String::new();
        report.push_str("=== Calibration Methods Comparison Report ===\n\n");
        let mut best_ece_idx = 0;
        let mut best_mce_idx = 0;
        let mut best_brier_idx = 0;
        let mut best_nll_idx = 0;
        for (i, comp) in comparisons.iter().enumerate() {
            if comp.analysis.ece < comparisons[best_ece_idx].analysis.ece {
                best_ece_idx = i;
            }
            if comp.analysis.mce < comparisons[best_mce_idx].analysis.mce {
                best_mce_idx = i;
            }
            if comp.analysis.brier_score < comparisons[best_brier_idx].analysis.brier_score {
                best_brier_idx = i;
            }
            if comp.analysis.nll < comparisons[best_nll_idx].analysis.nll {
                best_nll_idx = i;
            }
        }
        for (i, comp) in comparisons.iter().enumerate() {
            report.push_str(&format!("\n{}\n", comp.method_name));
            report.push_str(&format!("{}\n", "=".repeat(comp.method_name.len())));
            report.push_str(&format!(
                "ECE: {:.4}{}\n",
                comp.analysis.ece,
                if i == best_ece_idx { " ⭐ BEST" } else { "" }
            ));
            report.push_str(&format!(
                "MCE: {:.4}{}\n",
                comp.analysis.mce,
                if i == best_mce_idx { " ⭐ BEST" } else { "" }
            ));
            report.push_str(&format!(
                "Brier Score: {:.4}{}\n",
                comp.analysis.brier_score,
                if i == best_brier_idx { " ⭐ BEST" } else { "" }
            ));
            report.push_str(&format!(
                "NLL: {:.4}{}\n",
                comp.analysis.nll,
                if i == best_nll_idx { " ⭐ BEST" } else { "" }
            ));
            report.push_str(&format!(
                "Interpretation: {}\n",
                comp.analysis.interpretation
            ));
        }
        report.push_str("\n=== Recommendations ===\n");
        report.push_str(&format!(
            "Best overall (ECE): {}\n",
            comparisons[best_ece_idx].method_name
        ));
        report.push_str(&format!(
            "Most reliable (MCE): {}\n",
            comparisons[best_mce_idx].method_name
        ));
        report.push_str(&format!(
            "Best probability estimates (Brier): {}\n",
            comparisons[best_brier_idx].method_name
        ));
        report
    }
}
/// Post-hoc calibration for Quantum Neural Networks
/// Provides specialized calibration methods for quantum ML models
pub mod quantum_calibration {
    use super::*;
    /// Quantum-aware calibration configuration
    #[derive(Debug, Clone)]
    pub struct QuantumCalibrationConfig {
        /// Number of bins for histogram-based methods
        pub n_bins: usize,
        /// Whether to use quantum-aware error mitigation
        pub use_error_mitigation: bool,
        /// Confidence level for uncertainty quantification
        pub confidence_level: f64,
        /// Whether to account for shot noise
        pub account_shot_noise: bool,
    }
    impl Default for QuantumCalibrationConfig {
        fn default() -> Self {
            Self {
                n_bins: 10,
                use_error_mitigation: true,
                confidence_level: 0.95,
                account_shot_noise: true,
            }
        }
    }
    /// Quantum Neural Network Calibrator
    /// Specialized calibration for quantum ML models accounting for:
    /// - Quantum measurement noise
    /// - Shot noise from finite sampling
    /// - Hardware-specific errors
    #[derive(Debug, Clone)]
    pub struct QuantumNeuralNetworkCalibrator {
        /// Base calibration method
        method: CalibrationMethod,
        /// Configuration
        config: QuantumCalibrationConfig,
        /// Shot noise estimates per prediction
        shot_noise_estimates: Option<Array1<f64>>,
        /// Whether calibrator is fitted
        fitted: bool,
    }
    /// Calibration method selection
    #[derive(Debug, Clone)]
    pub enum CalibrationMethod {
        /// Temperature scaling (for multi-class)
        Temperature(TemperatureScaler),
        /// Vector scaling (for multi-class with class-specific parameters)
        Vector(VectorScaler),
        /// Platt scaling (for binary)
        Platt(PlattScaler),
        /// Isotonic regression (for binary)
        Isotonic(IsotonicRegression),
        /// Bayesian Binning (for binary with uncertainty)
        BayesianBinning(BayesianBinningQuantiles),
    }
    impl QuantumNeuralNetworkCalibrator {
        /// Create new quantum calibrator with default temperature scaling
        pub fn new() -> Self {
            Self {
                method: CalibrationMethod::Temperature(TemperatureScaler::new()),
                config: QuantumCalibrationConfig::default(),
                shot_noise_estimates: None,
                fitted: false,
            }
        }
        /// Create calibrator with specific method
        pub fn with_method(method: CalibrationMethod) -> Self {
            Self {
                method,
                config: QuantumCalibrationConfig::default(),
                shot_noise_estimates: None,
                fitted: false,
            }
        }
        /// Set configuration
        pub fn with_config(mut self, config: QuantumCalibrationConfig) -> Self {
            self.config = config;
            self
        }
        /// Fit calibrator for binary classification
        pub fn fit_binary(
            &mut self,
            probabilities: &Array1<f64>,
            labels: &Array1<usize>,
            shot_counts: Option<&Array1<usize>>,
        ) -> Result<()> {
            if let Some(shots) = shot_counts {
                if self.config.account_shot_noise {
                    self.shot_noise_estimates =
                        Some(self.estimate_shot_noise(probabilities, shots));
                }
            }
            match &mut self.method {
                CalibrationMethod::Platt(scaler) => {
                    scaler.fit(probabilities, labels)?;
                }
                CalibrationMethod::Isotonic(scaler) => {
                    scaler.fit(probabilities, labels)?;
                }
                CalibrationMethod::BayesianBinning(scaler) => {
                    scaler.fit(probabilities, labels)?;
                }
                _ => {
                    return Err(MLError::InvalidInput(
                        "Binary calibration requires Platt, Isotonic, or BBQ method".to_string(),
                    ));
                }
            }
            self.fitted = true;
            Ok(())
        }
        /// Fit calibrator for multi-class classification
        pub fn fit_multiclass(
            &mut self,
            logits: &Array2<f64>,
            labels: &Array1<usize>,
            shot_counts: Option<&Array1<usize>>,
        ) -> Result<()> {
            if let Some(shots) = shot_counts {
                if self.config.account_shot_noise {
                    let avg_probs = logits
                        .mean_axis(scirs2_core::ndarray::Axis(1))
                        .expect("logits should have valid axis");
                    self.shot_noise_estimates = Some(self.estimate_shot_noise(&avg_probs, shots));
                }
            }
            match &mut self.method {
                CalibrationMethod::Temperature(scaler) => {
                    scaler.fit(logits, labels)?;
                }
                CalibrationMethod::Vector(scaler) => {
                    scaler.fit(logits, labels)?;
                }
                _ => {
                    return Err(MLError::InvalidInput(
                        "Multi-class calibration requires Temperature or Vector method".to_string(),
                    ));
                }
            }
            self.fitted = true;
            Ok(())
        }
        /// Estimate shot noise for each probability
        fn estimate_shot_noise(
            &self,
            probabilities: &Array1<f64>,
            shot_counts: &Array1<usize>,
        ) -> Array1<f64> {
            probabilities
                .iter()
                .zip(shot_counts.iter())
                .map(|(&p, &n)| {
                    if n > 0 {
                        (p * (1.0 - p) / n as f64).sqrt()
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>()
                .into()
        }
        /// Transform binary probabilities
        pub fn transform_binary(&self, probabilities: &Array1<f64>) -> Result<Array1<f64>> {
            if !self.fitted {
                return Err(MLError::InvalidInput(
                    "Calibrator must be fitted before transform".to_string(),
                ));
            }
            match &self.method {
                CalibrationMethod::Platt(scaler) => scaler.transform(probabilities),
                CalibrationMethod::Isotonic(scaler) => scaler.transform(probabilities),
                CalibrationMethod::BayesianBinning(scaler) => scaler.transform(probabilities),
                _ => Err(MLError::InvalidInput(
                    "Method does not support binary transformation".to_string(),
                )),
            }
        }
        /// Transform multi-class logits
        pub fn transform_multiclass(&self, logits: &Array2<f64>) -> Result<Array2<f64>> {
            if !self.fitted {
                return Err(MLError::InvalidInput(
                    "Calibrator must be fitted before transform".to_string(),
                ));
            }
            match &self.method {
                CalibrationMethod::Temperature(scaler) => scaler.transform(logits),
                CalibrationMethod::Vector(scaler) => scaler.transform(logits),
                _ => Err(MLError::InvalidInput(
                    "Method does not support multi-class transformation".to_string(),
                )),
            }
        }
        /// Transform with uncertainty quantification (binary only)
        pub fn transform_with_uncertainty(
            &self,
            probabilities: &Array1<f64>,
        ) -> Result<Vec<(f64, f64, f64)>> {
            if !self.fitted {
                return Err(MLError::InvalidInput(
                    "Calibrator must be fitted before transform".to_string(),
                ));
            }
            match &self.method {
                CalibrationMethod::BayesianBinning(scaler) => {
                    scaler.predict_with_uncertainty(probabilities, self.config.confidence_level)
                }
                _ => {
                    let calibrated = self.transform_binary(probabilities)?;
                    if let Some(noise) = &self.shot_noise_estimates {
                        let results = calibrated
                            .iter()
                            .zip(noise.iter())
                            .map(|(&p, &sigma)| {
                                let z = 1.96;
                                let lower = (p - z * sigma).max(0.0);
                                let upper = (p + z * sigma).min(1.0);
                                (p, lower, upper)
                            })
                            .collect();
                        Ok(results)
                    } else {
                        Ok(calibrated.iter().map(|&p| (p, p, p)).collect())
                    }
                }
            }
        }
        /// Get calibration quality metrics for quantum model
        pub fn evaluate_quantum_calibration(
            &self,
            probabilities: &Array1<f64>,
            labels: &Array1<usize>,
        ) -> Result<QuantumCalibrationMetrics> {
            let calibrated = self.transform_binary(probabilities)?;
            let analysis =
                visualization::analyze_calibration(&calibrated, labels, self.config.n_bins)?;
            let shot_noise_impact = if let Some(noise) = &self.shot_noise_estimates {
                noise.mean().unwrap_or(0.0)
            } else {
                0.0
            };
            Ok(QuantumCalibrationMetrics {
                ece: analysis.ece,
                mce: analysis.mce,
                brier_score: analysis.brier_score,
                nll: analysis.nll,
                shot_noise_impact,
                interpretation: analysis.interpretation,
            })
        }
    }
    impl Default for QuantumNeuralNetworkCalibrator {
        fn default() -> Self {
            Self::new()
        }
    }
    /// Quantum calibration metrics
    #[derive(Debug, Clone)]
    pub struct QuantumCalibrationMetrics {
        /// Expected Calibration Error
        pub ece: f64,
        /// Maximum Calibration Error
        pub mce: f64,
        /// Brier score
        pub brier_score: f64,
        /// Negative log-likelihood
        pub nll: f64,
        /// Average shot noise impact
        pub shot_noise_impact: f64,
        /// Interpretation
        pub interpretation: String,
    }
    /// Quantum-aware ensemble calibration
    /// Combines multiple calibration methods with quantum circuit execution results
    pub fn quantum_ensemble_calibration(
        probabilities: &Array1<f64>,
        labels: &Array1<usize>,
        shot_counts: &Array1<usize>,
        n_bins: usize,
    ) -> Result<(Array1<f64>, QuantumCalibrationMetrics)> {
        let mut platt_cal = QuantumNeuralNetworkCalibrator::with_method(CalibrationMethod::Platt(
            PlattScaler::new(),
        ));
        platt_cal.fit_binary(probabilities, labels, Some(shot_counts))?;
        let mut isotonic_cal = QuantumNeuralNetworkCalibrator::with_method(
            CalibrationMethod::Isotonic(IsotonicRegression::new()),
        );
        isotonic_cal.fit_binary(probabilities, labels, Some(shot_counts))?;
        let mut bbq_cal = QuantumNeuralNetworkCalibrator::with_method(
            CalibrationMethod::BayesianBinning(BayesianBinningQuantiles::new(n_bins)),
        );
        bbq_cal.fit_binary(probabilities, labels, Some(shot_counts))?;
        let platt_probs = platt_cal.transform_binary(probabilities)?;
        let isotonic_probs = isotonic_cal.transform_binary(probabilities)?;
        let bbq_probs = bbq_cal.transform_binary(probabilities)?;
        let platt_metrics = platt_cal.evaluate_quantum_calibration(probabilities, labels)?;
        let isotonic_metrics = isotonic_cal.evaluate_quantum_calibration(probabilities, labels)?;
        let bbq_metrics = bbq_cal.evaluate_quantum_calibration(probabilities, labels)?;
        let platt_weight = 1.0 / (platt_metrics.ece + 1e-6);
        let isotonic_weight = 1.0 / (isotonic_metrics.ece + 1e-6);
        let bbq_weight = 1.0 / (bbq_metrics.ece + 1e-6);
        let total_weight = platt_weight + isotonic_weight + bbq_weight;
        let ensemble_probs = (&platt_probs * (platt_weight / total_weight))
            + (&isotonic_probs * (isotonic_weight / total_weight))
            + (&bbq_probs * (bbq_weight / total_weight));
        let ensemble_analysis =
            visualization::analyze_calibration(&ensemble_probs, labels, n_bins)?;
        let metrics = QuantumCalibrationMetrics {
            ece: ensemble_analysis.ece,
            mce: ensemble_analysis.mce,
            brier_score: ensemble_analysis.brier_score,
            nll: ensemble_analysis.nll,
            shot_noise_impact: platt_metrics.shot_noise_impact,
            interpretation: ensemble_analysis.interpretation,
        };
        Ok((ensemble_probs, metrics))
    }
}
/// Ensemble selection and calibration-aware model selection
pub mod ensemble_selection {
    use super::*;
    use crate::utils::split::KFold;
    /// Ensemble calibration method with metadata
    #[derive(Debug, Clone)]
    pub struct CalibratorCandidate {
        /// Name of the calibration method
        pub name: String,
        /// Cross-validation ECE scores
        pub cv_ece_scores: Vec<f64>,
        /// Mean ECE across folds
        pub mean_ece: f64,
        /// Standard deviation of ECE
        pub std_ece: f64,
        /// Whether this is for binary or multiclass
        pub is_binary: bool,
    }
    /// Ensemble selection strategy
    #[derive(Debug, Clone)]
    pub enum SelectionStrategy {
        /// Select single best method by mean ECE
        BestSingle,
        /// Select top K methods
        TopK(usize),
        /// Select all methods with ECE below threshold
        Threshold(f64),
        /// Weighted ensemble of all methods
        WeightedAll,
    }
    /// Result of ensemble selection
    #[derive(Debug, Clone)]
    pub struct EnsembleSelectionResult {
        /// Selected calibrator names
        pub selected_methods: Vec<String>,
        /// Weights for ensemble (if applicable)
        pub weights: Vec<f64>,
        /// Performance metrics for each method
        pub method_performances: Vec<CalibratorCandidate>,
        /// Best individual method
        pub best_method: String,
        /// Ensemble expected ECE
        pub ensemble_ece: f64,
    }
    /// Perform cross-validated ensemble selection for binary calibration
    pub fn select_binary_ensemble(
        probabilities: &Array1<f64>,
        labels: &Array1<usize>,
        n_folds: usize,
        strategy: SelectionStrategy,
    ) -> Result<EnsembleSelectionResult> {
        if n_folds < 2 {
            return Err(MLError::InvalidInput(
                "Need at least 2 folds for cross-validation".to_string(),
            ));
        }
        let kfold = KFold::new(probabilities.len(), n_folds, true)?;
        let method_names = vec!["Platt", "Isotonic", "BBQ-5", "BBQ-10"];
        let mut candidates = Vec::new();
        for method_name in method_names {
            let mut cv_ece_scores = Vec::new();
            for fold in 0..n_folds {
                let (train_indices, val_indices) = kfold.get_fold(fold)?;
                let train_probs: Array1<f64> =
                    train_indices.iter().map(|&i| probabilities[i]).collect();
                let train_labels: Array1<usize> =
                    train_indices.iter().map(|&i| labels[i]).collect();
                let val_probs: Array1<f64> =
                    val_indices.iter().map(|&i| probabilities[i]).collect();
                let val_labels: Array1<usize> = val_indices.iter().map(|&i| labels[i]).collect();
                let calibrated_val = match method_name {
                    "Platt" => {
                        let mut scaler = PlattScaler::new();
                        scaler.fit(&train_probs, &train_labels)?;
                        scaler.transform(&val_probs)?
                    }
                    "Isotonic" => {
                        let mut scaler = IsotonicRegression::new();
                        scaler.fit(&train_probs, &train_labels)?;
                        scaler.transform(&val_probs)?
                    }
                    "BBQ-5" => {
                        let mut scaler = BayesianBinningQuantiles::new(5);
                        scaler.fit(&train_probs, &train_labels)?;
                        scaler.transform(&val_probs)?
                    }
                    "BBQ-10" => {
                        let mut scaler = BayesianBinningQuantiles::new(10);
                        scaler.fit(&train_probs, &train_labels)?;
                        scaler.transform(&val_probs)?
                    }
                    _ => {
                        return Err(MLError::InvalidInput(format!(
                            "Unknown method: {}",
                            method_name
                        )));
                    }
                };
                let analysis =
                    visualization::analyze_calibration(&calibrated_val, &val_labels, 10)?;
                cv_ece_scores.push(analysis.ece);
            }
            let mean_ece = cv_ece_scores.iter().sum::<f64>() / cv_ece_scores.len() as f64;
            let variance = cv_ece_scores
                .iter()
                .map(|&x| (x - mean_ece).powi(2))
                .sum::<f64>()
                / cv_ece_scores.len() as f64;
            let std_ece = variance.sqrt();
            candidates.push(CalibratorCandidate {
                name: method_name.to_string(),
                cv_ece_scores,
                mean_ece,
                std_ece,
                is_binary: true,
            });
        }
        candidates.sort_by(|a, b| {
            a.mean_ece
                .partial_cmp(&b.mean_ece)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let (selected_methods, weights) = match strategy {
            SelectionStrategy::BestSingle => (vec![candidates[0].name.clone()], vec![1.0]),
            SelectionStrategy::TopK(k) => {
                let k = k.min(candidates.len());
                let methods: Vec<String> = candidates[..k].iter().map(|c| c.name.clone()).collect();
                let weights = vec![1.0 / k as f64; k];
                (methods, weights)
            }
            SelectionStrategy::Threshold(threshold) => {
                let selected: Vec<_> = candidates
                    .iter()
                    .filter(|c| c.mean_ece < threshold)
                    .map(|c| c.name.clone())
                    .collect();
                if selected.is_empty() {
                    (vec![candidates[0].name.clone()], vec![1.0])
                } else {
                    let n = selected.len();
                    let weights = vec![1.0 / n as f64; n];
                    (selected, weights)
                }
            }
            SelectionStrategy::WeightedAll => {
                let methods: Vec<String> = candidates.iter().map(|c| c.name.clone()).collect();
                let inv_eces: Vec<f64> = candidates
                    .iter()
                    .map(|c| 1.0 / (c.mean_ece + 1e-6))
                    .collect();
                let sum_inv: f64 = inv_eces.iter().sum();
                let weights: Vec<f64> = inv_eces.iter().map(|&w| w / sum_inv).collect();
                (methods, weights)
            }
        };
        let best_method = candidates[0].name.clone();
        let ensemble_ece = if weights.len() == 1 {
            candidates[0].mean_ece
        } else {
            candidates
                .iter()
                .zip(&weights)
                .map(|(c, &w)| c.mean_ece * w)
                .sum()
        };
        Ok(EnsembleSelectionResult {
            selected_methods,
            weights,
            method_performances: candidates,
            best_method,
            ensemble_ece,
        })
    }
    /// Calibration-aware model selection
    /// Selects the best calibration method for a given model based on validation performance
    #[derive(Debug, Clone)]
    pub struct CalibrationAwareSelector {
        /// Selection strategy
        strategy: SelectionStrategy,
        /// Number of cross-validation folds
        n_folds: usize,
        /// Whether to use binary or multiclass calibration
        is_binary: bool,
    }
    impl CalibrationAwareSelector {
        /// Create a new calibration-aware selector
        pub fn new(n_folds: usize, is_binary: bool) -> Self {
            Self {
                strategy: SelectionStrategy::BestSingle,
                n_folds,
                is_binary,
            }
        }
        /// Set selection strategy
        pub fn with_strategy(mut self, strategy: SelectionStrategy) -> Self {
            self.strategy = strategy;
            self
        }
        /// Select best calibration method for binary classification
        pub fn select_binary(
            &self,
            probabilities: &Array1<f64>,
            labels: &Array1<usize>,
        ) -> Result<EnsembleSelectionResult> {
            select_binary_ensemble(probabilities, labels, self.n_folds, self.strategy.clone())
        }
        /// Generate a detailed report of calibration method comparison
        pub fn generate_selection_report(&self, result: &EnsembleSelectionResult) -> String {
            let mut report = String::new();
            report.push_str("=== Calibration Method Selection Report ===\n\n");
            report.push_str("Cross-Validation Results:\n");
            report.push_str(&format!("{:-<60}\n", ""));
            for method in &result.method_performances {
                report.push_str(&format!(
                    "{:<15} | Mean ECE: {:.4} ± {:.4}\n",
                    method.name, method.mean_ece, method.std_ece
                ));
            }
            report.push_str(&format!("\n{:-<60}\n", ""));
            report.push_str(&format!(
                "\nBest Individual Method: {}\n",
                result.best_method
            ));
            report.push_str(&format!(
                "Expected Ensemble ECE: {:.4}\n\n",
                result.ensemble_ece
            ));
            report.push_str("Selected Ensemble:\n");
            for (method, weight) in result.selected_methods.iter().zip(&result.weights) {
                report.push_str(&format!("  {} (weight: {:.3})\n", method, weight));
            }
            report.push_str("\nRecommendation:\n");
            if result.selected_methods.len() == 1 {
                report.push_str(&format!(
                    "Use {} for best calibration performance.\n",
                    result.selected_methods[0]
                ));
            } else {
                report.push_str(
                    "Use weighted ensemble of selected methods for robust calibration.\n",
                );
            }
            report
        }
    }
}
