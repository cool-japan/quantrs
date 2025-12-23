//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2};

/// Matrix Scaling - full affine transformation for maximum calibration flexibility
/// Uses full weight matrix W and bias vector b: calibrated = softmax(W @ logits + b)
/// More expressive than vector scaling but requires more data to avoid overfitting
#[derive(Debug, Clone)]
pub struct MatrixScaler {
    /// Weight matrix (n_classes × n_classes)
    weight_matrix: Option<Array2<f64>>,
    /// Bias vector (n_classes)
    bias_vector: Option<Array1<f64>>,
    /// Whether the scaler has been fitted
    fitted: bool,
    /// Regularization strength (L2 penalty on off-diagonal elements)
    regularization: f64,
}
impl MatrixScaler {
    /// Create a new matrix scaler
    pub fn new() -> Self {
        Self {
            weight_matrix: None,
            bias_vector: None,
            fitted: false,
            regularization: 0.01,
        }
    }
    /// Create matrix scaler with custom regularization
    pub fn with_regularization(regularization: f64) -> Self {
        Self {
            weight_matrix: None,
            bias_vector: None,
            fitted: false,
            regularization,
        }
    }
    /// Fit the matrix scaler to logits and true labels
    /// Uses gradient descent with L2 regularization on off-diagonal weights
    pub fn fit(&mut self, logits: &Array2<f64>, labels: &Array1<usize>) -> Result<()> {
        if logits.nrows() != labels.len() {
            return Err(MLError::InvalidInput(
                "Logits and labels must have same number of samples".to_string(),
            ));
        }
        let n_samples = logits.nrows();
        let n_classes = logits.ncols();
        if n_samples < n_classes * 2 {
            return Err(MLError::InvalidInput(format!(
                "Need at least {} samples for {} classes (matrix calibration)",
                n_classes * 2,
                n_classes
            )));
        }
        let mut weight_matrix = Array2::eye(n_classes);
        let mut bias_vector = Array1::zeros(n_classes);
        let learning_rate = 0.001;
        let max_iter = 300;
        let tolerance = 1e-7;
        let mut prev_nll = f64::INFINITY;
        for _iter in 0..max_iter {
            let (nll, reg_term) =
                self.compute_nll_with_reg(logits, labels, &weight_matrix, &bias_vector)?;
            let total_loss = nll + reg_term;
            if (prev_nll - total_loss).abs() < tolerance {
                break;
            }
            prev_nll = total_loss;
            let epsilon = 1e-6;
            let mut weight_grads = Array2::zeros((n_classes, n_classes));
            let mut bias_grads = Array1::zeros(n_classes);
            for i in 0..n_classes {
                for j in 0..n_classes {
                    let mut weight_plus = weight_matrix.clone();
                    weight_plus[(i, j)] += epsilon;
                    let (nll_plus, reg_plus) =
                        self.compute_nll_with_reg(logits, labels, &weight_plus, &bias_vector)?;
                    weight_grads[(i, j)] = (nll_plus + reg_plus - total_loss) / epsilon;
                }
            }
            for j in 0..n_classes {
                let mut bias_plus = bias_vector.clone();
                bias_plus[j] += epsilon;
                let (nll_plus, reg_plus) =
                    self.compute_nll_with_reg(logits, labels, &weight_matrix, &bias_plus)?;
                bias_grads[j] = (nll_plus + reg_plus - total_loss) / epsilon;
            }
            weight_matrix = &weight_matrix - &weight_grads.mapv(|g| learning_rate * g);
            bias_vector = &bias_vector - &bias_grads.mapv(|g| learning_rate * g);
            for i in 0..n_classes {
                weight_matrix[(i, i)] = weight_matrix[(i, i)].max(0.01);
            }
            let grad_norm = weight_grads.iter().map(|&g| g * g).sum::<f64>().sqrt()
                + bias_grads.iter().map(|&g| g * g).sum::<f64>().sqrt();
            if grad_norm < tolerance {
                break;
            }
        }
        self.weight_matrix = Some(weight_matrix);
        self.bias_vector = Some(bias_vector);
        self.fitted = true;
        Ok(())
    }
    /// Compute NLL with L2 regularization on off-diagonal weights
    fn compute_nll_with_reg(
        &self,
        logits: &Array2<f64>,
        labels: &Array1<usize>,
        weight_matrix: &Array2<f64>,
        bias_vector: &Array1<f64>,
    ) -> Result<(f64, f64)> {
        let mut nll = 0.0;
        let n_samples = logits.nrows();
        let n_classes = logits.ncols();
        for i in 0..n_samples {
            let logits_row = logits.row(i);
            let mut scaled_logits = Array1::zeros(n_classes);
            for j in 0..n_classes {
                let mut val = bias_vector[j];
                for k in 0..n_classes {
                    val += weight_matrix[(j, k)] * logits_row[k];
                }
                scaled_logits[j] = val;
            }
            let max_logit = scaled_logits
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_logits: Vec<f64> = scaled_logits
                .iter()
                .map(|&x| (x - max_logit).exp())
                .collect();
            let sum_exp: f64 = exp_logits.iter().sum();
            let true_label = labels[i];
            if true_label >= exp_logits.len() {
                return Err(MLError::InvalidInput(format!(
                    "Label {} out of bounds for {} classes",
                    true_label,
                    exp_logits.len()
                )));
            }
            let prob = exp_logits[true_label] / sum_exp;
            nll -= prob.max(1e-10).ln();
        }
        nll /= n_samples as f64;
        let mut reg_term = 0.0;
        for i in 0..n_classes {
            for j in 0..n_classes {
                if i != j {
                    reg_term += weight_matrix[(i, j)].powi(2);
                }
            }
        }
        reg_term *= self.regularization;
        Ok((nll, reg_term))
    }
    /// Transform logits to calibrated probabilities using matrix scaling
    pub fn transform(&self, logits: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(MLError::InvalidInput(
                "Scaler must be fitted before transform".to_string(),
            ));
        }
        let weight_matrix = self.weight_matrix.as_ref().unwrap();
        let bias_vector = self.bias_vector.as_ref().unwrap();
        let n_classes = logits.ncols();
        let mut calibrated_probs = Array2::zeros((logits.nrows(), logits.ncols()));
        for i in 0..logits.nrows() {
            let logits_row = logits.row(i);
            let mut scaled_logits = Array1::zeros(n_classes);
            for j in 0..n_classes {
                let mut val = bias_vector[j];
                for k in 0..n_classes {
                    val += weight_matrix[(j, k)] * logits_row[k];
                }
                scaled_logits[j] = val;
            }
            let max_logit = scaled_logits
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_logits: Vec<f64> = scaled_logits
                .iter()
                .map(|&x| (x - max_logit).exp())
                .collect();
            let sum_exp: f64 = exp_logits.iter().sum();
            for j in 0..logits.ncols() {
                calibrated_probs[(i, j)] = exp_logits[j] / sum_exp;
            }
        }
        Ok(calibrated_probs)
    }
    /// Fit and transform in one step
    pub fn fit_transform(
        &mut self,
        logits: &Array2<f64>,
        labels: &Array1<usize>,
    ) -> Result<Array2<f64>> {
        self.fit(logits, labels)?;
        self.transform(logits)
    }
    /// Get the fitted parameters (weight_matrix, bias_vector)
    pub fn parameters(&self) -> Option<(Array2<f64>, Array1<f64>)> {
        if self.fitted {
            Some((
                self.weight_matrix.as_ref().unwrap().clone(),
                self.bias_vector.as_ref().unwrap().clone(),
            ))
        } else {
            None
        }
    }
    /// Get the condition number of the weight matrix (for diagnostics)
    /// Higher values indicate potential numerical instability
    pub fn condition_number(&self) -> Option<f64> {
        if !self.fitted {
            return None;
        }
        let w = self.weight_matrix.as_ref().unwrap();
        let norm = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
        Some(norm)
    }
}
/// Isotonic Regression - non-parametric calibration using monotonic transformation
/// More flexible than Platt scaling but requires more data
#[derive(Debug, Clone)]
pub struct IsotonicRegression {
    /// X values (decision scores)
    x_thresholds: Vec<f64>,
    /// Y values (calibrated probabilities)
    y_thresholds: Vec<f64>,
    /// Whether the regressor has been fitted
    fitted: bool,
}
impl IsotonicRegression {
    /// Create a new isotonic regression calibrator
    pub fn new() -> Self {
        Self {
            x_thresholds: Vec::new(),
            y_thresholds: Vec::new(),
            fitted: false,
        }
    }
    /// Fit isotonic regression to scores and labels
    pub fn fit(&mut self, scores: &Array1<f64>, labels: &Array1<usize>) -> Result<()> {
        if scores.len() != labels.len() {
            return Err(MLError::InvalidInput(
                "Scores and labels must have same length".to_string(),
            ));
        }
        let n = scores.len();
        if n < 2 {
            return Err(MLError::InvalidInput(
                "Need at least 2 samples for calibration".to_string(),
            ));
        }
        let mut pairs: Vec<(f64, f64)> = scores
            .iter()
            .zip(labels.iter())
            .map(|(&s, &l)| (s, l as f64))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut weights = Vec::new();
        for (score, label) in pairs {
            x.push(score);
            y.push(label);
            weights.push(1.0);
        }
        let mut i = 0;
        while i < y.len() - 1 {
            if y[i] > y[i + 1] {
                let w1 = weights[i];
                let w2 = weights[i + 1];
                let total_weight = w1 + w2;
                y[i] = (y[i] * w1 + y[i + 1] * w2) / total_weight;
                weights[i] = total_weight;
                y.remove(i + 1);
                x.remove(i + 1);
                weights.remove(i + 1);
                if i > 0 {
                    i -= 1;
                }
            } else {
                i += 1;
            }
        }
        self.x_thresholds = x;
        self.y_thresholds = y;
        self.fitted = true;
        Ok(())
    }
    /// Transform decision scores to calibrated probabilities
    pub fn transform(&self, scores: &Array1<f64>) -> Result<Array1<f64>> {
        if !self.fitted {
            return Err(MLError::InvalidInput(
                "Regressor must be fitted before transform".to_string(),
            ));
        }
        let mut calibrated = Array1::zeros(scores.len());
        for (i, &score) in scores.iter().enumerate() {
            let pos = self
                .x_thresholds
                .binary_search_by(|&x| x.partial_cmp(&score).unwrap_or(std::cmp::Ordering::Less))
                .unwrap_or_else(|e| e);
            if pos == 0 {
                calibrated[i] = self.y_thresholds[0];
            } else if pos >= self.x_thresholds.len() {
                calibrated[i] = *self.y_thresholds.last().unwrap();
            } else {
                let x0 = self.x_thresholds[pos - 1];
                let x1 = self.x_thresholds[pos];
                let y0 = self.y_thresholds[pos - 1];
                let y1 = self.y_thresholds[pos];
                if (x1 - x0).abs() < 1e-10 {
                    calibrated[i] = (y0 + y1) / 2.0;
                } else {
                    let alpha = (score - x0) / (x1 - x0);
                    calibrated[i] = y0 + alpha * (y1 - y0);
                }
            }
        }
        Ok(calibrated)
    }
    /// Fit and transform in one step
    pub fn fit_transform(
        &mut self,
        scores: &Array1<f64>,
        labels: &Array1<usize>,
    ) -> Result<Array1<f64>> {
        self.fit(scores, labels)?;
        self.transform(scores)
    }
}
/// Bayesian Binning into Quantiles (BBQ) - sophisticated histogram-based calibration
/// Bins predictions into quantiles and learns Bayesian posterior for each bin
/// Uses Beta distribution for robust probability estimation with uncertainty quantification
#[derive(Debug, Clone)]
pub struct BayesianBinningQuantiles {
    /// Number of bins
    n_bins: usize,
    /// Bin edges (quantile thresholds)
    bin_edges: Option<Vec<f64>>,
    /// Alpha parameters for Beta distribution in each bin
    alphas: Option<Array1<f64>>,
    /// Beta parameters for Beta distribution in each bin
    betas: Option<Array1<f64>>,
    /// Whether the calibrator has been fitted
    fitted: bool,
}
impl BayesianBinningQuantiles {
    /// Create a new BBQ calibrator with specified number of bins
    pub fn new(n_bins: usize) -> Self {
        Self {
            n_bins,
            bin_edges: None,
            alphas: None,
            betas: None,
            fitted: false,
        }
    }
    /// Fit the BBQ calibrator to probabilities and true labels
    pub fn fit(&mut self, probabilities: &Array1<f64>, labels: &Array1<usize>) -> Result<()> {
        if probabilities.len() != labels.len() {
            return Err(MLError::InvalidInput(
                "Probabilities and labels must have same length".to_string(),
            ));
        }
        let n_samples = probabilities.len();
        if n_samples < self.n_bins {
            return Err(MLError::InvalidInput(format!(
                "Need at least {} samples for {} bins, got {}",
                self.n_bins, self.n_bins, n_samples
            )));
        }
        let mut sorted_probs = probabilities.to_vec();
        sorted_probs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut bin_edges = vec![0.0];
        for i in 1..self.n_bins {
            let quantile_idx = (i as f64 / self.n_bins as f64 * n_samples as f64) as usize;
            let quantile_idx = quantile_idx.min(sorted_probs.len() - 1);
            bin_edges.push(sorted_probs[quantile_idx]);
        }
        bin_edges.push(1.0);
        let mut bin_positives = vec![0.0; self.n_bins];
        let mut bin_negatives = vec![0.0; self.n_bins];
        for (i, &prob) in probabilities.iter().enumerate() {
            let bin_idx = self.find_bin(&bin_edges, prob);
            let label = labels[i];
            if label == 1 {
                bin_positives[bin_idx] += 1.0;
            } else {
                bin_negatives[bin_idx] += 1.0;
            }
        }
        let prior_alpha = 0.5;
        let prior_beta = 0.5;
        let mut alphas = Array1::zeros(self.n_bins);
        let mut betas = Array1::zeros(self.n_bins);
        for i in 0..self.n_bins {
            alphas[i] = prior_alpha + bin_positives[i];
            betas[i] = prior_beta + bin_negatives[i];
        }
        self.bin_edges = Some(bin_edges);
        self.alphas = Some(alphas);
        self.betas = Some(betas);
        self.fitted = true;
        Ok(())
    }
    /// Find which bin a probability belongs to
    fn find_bin(&self, bin_edges: &[f64], prob: f64) -> usize {
        for i in 0..bin_edges.len() - 1 {
            if prob >= bin_edges[i] && prob < bin_edges[i + 1] {
                return i;
            }
        }
        bin_edges.len() - 2
    }
    /// Transform probabilities to calibrated probabilities using BBQ
    pub fn transform(&self, probabilities: &Array1<f64>) -> Result<Array1<f64>> {
        if !self.fitted {
            return Err(MLError::InvalidInput(
                "Calibrator must be fitted before transform".to_string(),
            ));
        }
        let bin_edges = self.bin_edges.as_ref().unwrap();
        let alphas = self.alphas.as_ref().unwrap();
        let betas = self.betas.as_ref().unwrap();
        let mut calibrated = Array1::zeros(probabilities.len());
        for (i, &prob) in probabilities.iter().enumerate() {
            let bin_idx = self.find_bin(bin_edges, prob);
            let alpha = alphas[bin_idx];
            let beta = betas[bin_idx];
            calibrated[i] = alpha / (alpha + beta);
        }
        Ok(calibrated)
    }
    /// Fit and transform in one step
    pub fn fit_transform(
        &mut self,
        probabilities: &Array1<f64>,
        labels: &Array1<usize>,
    ) -> Result<Array1<f64>> {
        self.fit(probabilities, labels)?;
        self.transform(probabilities)
    }
    /// Get calibrated probability with uncertainty bounds (credible interval)
    /// Returns (mean, lower_bound, upper_bound) for given confidence level
    pub fn predict_with_uncertainty(
        &self,
        probabilities: &Array1<f64>,
        confidence: f64,
    ) -> Result<Vec<(f64, f64, f64)>> {
        if !self.fitted {
            return Err(MLError::InvalidInput(
                "Calibrator must be fitted before prediction".to_string(),
            ));
        }
        if confidence <= 0.0 || confidence >= 1.0 {
            return Err(MLError::InvalidInput(
                "Confidence must be between 0 and 1".to_string(),
            ));
        }
        let bin_edges = self.bin_edges.as_ref().unwrap();
        let alphas = self.alphas.as_ref().unwrap();
        let betas = self.betas.as_ref().unwrap();
        let lower_quantile = (1.0 - confidence) / 2.0;
        let upper_quantile = 1.0 - lower_quantile;
        let mut results = Vec::new();
        for &prob in probabilities.iter() {
            let bin_idx = self.find_bin(bin_edges, prob);
            let alpha = alphas[bin_idx];
            let beta = betas[bin_idx];
            let mean = alpha / (alpha + beta);
            let n = alpha + beta - 1.0;
            let p = alpha / (alpha + beta);
            if n > 0.0 {
                let z = 1.96;
                let denominator = 1.0 + z * z / n;
                let center = (p + z * z / (2.0 * n)) / denominator;
                let margin = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / denominator;
                let lower = (center - margin).max(0.0);
                let upper = (center + margin).min(1.0);
                results.push((mean, lower, upper));
            } else {
                results.push((mean, 0.0, 1.0));
            }
        }
        Ok(results)
    }
    /// Get the number of bins
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }
    /// Get bin statistics (edges, alphas, betas)
    pub fn bin_statistics(&self) -> Option<(Vec<f64>, Array1<f64>, Array1<f64>)> {
        if self.fitted {
            Some((
                self.bin_edges.as_ref().unwrap().clone(),
                self.alphas.as_ref().unwrap().clone(),
                self.betas.as_ref().unwrap().clone(),
            ))
        } else {
            None
        }
    }
}
/// Platt Scaling - fits a logistic regression on decision scores
/// Calibrates binary classifier outputs to produce better probability estimates
#[derive(Debug, Clone)]
pub struct PlattScaler {
    /// Slope parameter of logistic function
    a: f64,
    /// Intercept parameter of logistic function
    b: f64,
    /// Whether the scaler has been fitted
    fitted: bool,
}
impl PlattScaler {
    /// Create a new Platt scaler
    pub fn new() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            fitted: false,
        }
    }
    /// Fit the Platt scaler to decision scores and true labels
    /// Uses maximum likelihood estimation to find optimal sigmoid parameters
    pub fn fit(&mut self, scores: &Array1<f64>, labels: &Array1<usize>) -> Result<()> {
        if scores.len() != labels.len() {
            return Err(MLError::InvalidInput(
                "Scores and labels must have same length".to_string(),
            ));
        }
        let n = scores.len();
        if n < 2 {
            return Err(MLError::InvalidInput(
                "Need at least 2 samples for calibration".to_string(),
            ));
        }
        let y: Array1<f64> = labels
            .iter()
            .map(|&l| if l == 1 { 1.0 } else { -1.0 })
            .collect();
        let mut a = 0.0;
        let mut b = 0.0;
        let n_pos = labels.iter().filter(|&&l| l == 1).count() as f64;
        let n_neg = n as f64 - n_pos;
        let prior_pos = (n_pos + 1.0) / (n as f64 + 2.0);
        b = (prior_pos / (1.0 - prior_pos)).ln();
        for _ in 0..100 {
            let mut fval = 0.0;
            let mut fpp = 0.0;
            for i in 0..n {
                let fapb = scores[i] * a + b;
                let p = 1.0 / (1.0 + (-fapb).exp());
                let t = if y[i] > 0.0 { 1.0 } else { 0.0 };
                fval += scores[i] * (t - p);
                fpp += scores[i] * scores[i] * p * (1.0 - p);
            }
            if fpp.abs() < 1e-12 {
                break;
            }
            let delta = fval / fpp;
            a += delta;
            if delta.abs() < 1e-7 {
                break;
            }
        }
        for _ in 0..100 {
            let mut fval = 0.0;
            let mut fpp = 0.0;
            for i in 0..n {
                let fapb = scores[i] * a + b;
                let p = 1.0 / (1.0 + (-fapb).exp());
                let t = if y[i] > 0.0 { 1.0 } else { 0.0 };
                fval += t - p;
                fpp += p * (1.0 - p);
            }
            if fpp.abs() < 1e-12 {
                break;
            }
            let delta = fval / fpp;
            b += delta;
            if delta.abs() < 1e-7 {
                break;
            }
        }
        self.a = a;
        self.b = b;
        self.fitted = true;
        Ok(())
    }
    /// Transform decision scores to calibrated probabilities
    pub fn transform(&self, scores: &Array1<f64>) -> Result<Array1<f64>> {
        if !self.fitted {
            return Err(MLError::InvalidInput(
                "Scaler must be fitted before transform".to_string(),
            ));
        }
        let probs = scores.mapv(|s| {
            let fapb = s * self.a + self.b;
            1.0 / (1.0 + (-fapb).exp())
        });
        Ok(probs)
    }
    /// Fit and transform in one step
    pub fn fit_transform(
        &mut self,
        scores: &Array1<f64>,
        labels: &Array1<usize>,
    ) -> Result<Array1<f64>> {
        self.fit(scores, labels)?;
        self.transform(scores)
    }
    /// Get the fitted parameters
    pub fn parameters(&self) -> Option<(f64, f64)> {
        if self.fitted {
            Some((self.a, self.b))
        } else {
            None
        }
    }
}
/// Temperature Scaling - simple and effective multi-class calibration
/// Scales logits by a single learned temperature parameter
/// Particularly effective for neural network outputs
#[derive(Debug, Clone)]
pub struct TemperatureScaler {
    /// Temperature parameter (T > 0)
    temperature: f64,
    /// Whether the scaler has been fitted
    fitted: bool,
}
impl TemperatureScaler {
    /// Create a new temperature scaler
    pub fn new() -> Self {
        Self {
            temperature: 1.0,
            fitted: false,
        }
    }
    /// Fit the temperature scaler to logits and true labels
    /// Uses negative log-likelihood minimization
    pub fn fit(&mut self, logits: &Array2<f64>, labels: &Array1<usize>) -> Result<()> {
        if logits.nrows() != labels.len() {
            return Err(MLError::InvalidInput(
                "Logits and labels must have same number of samples".to_string(),
            ));
        }
        let n_samples = logits.nrows();
        if n_samples < 2 {
            return Err(MLError::InvalidInput(
                "Need at least 2 samples for calibration".to_string(),
            ));
        }
        let mut best_temp = 1.0;
        let mut best_nll = f64::INFINITY;
        for t_candidate in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0] {
            let nll = self.compute_nll(logits, labels, t_candidate)?;
            if nll < best_nll {
                best_nll = nll;
                best_temp = t_candidate;
            }
        }
        let mut temperature = best_temp;
        let learning_rate = 0.01;
        for _ in 0..100 {
            let nll_current = self.compute_nll(logits, labels, temperature)?;
            let nll_plus = self.compute_nll(logits, labels, temperature + 0.01)?;
            let gradient = (nll_plus - nll_current) / 0.01;
            let new_temp = temperature - learning_rate * gradient;
            if new_temp <= 0.01 {
                break;
            }
            temperature = new_temp;
            if gradient.abs() < 1e-5 {
                break;
            }
        }
        self.temperature = temperature.max(0.01);
        self.fitted = true;
        Ok(())
    }
    /// Compute negative log-likelihood for given temperature
    fn compute_nll(
        &self,
        logits: &Array2<f64>,
        labels: &Array1<usize>,
        temperature: f64,
    ) -> Result<f64> {
        let mut nll = 0.0;
        let n_samples = logits.nrows();
        for i in 0..n_samples {
            let scaled_logits = logits.row(i).mapv(|x| x / temperature);
            let max_logit = scaled_logits
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_logits: Vec<f64> = scaled_logits
                .iter()
                .map(|&x| (x - max_logit).exp())
                .collect();
            let sum_exp: f64 = exp_logits.iter().sum();
            let true_label = labels[i];
            if true_label >= exp_logits.len() {
                return Err(MLError::InvalidInput(format!(
                    "Label {} out of bounds for {} classes",
                    true_label,
                    exp_logits.len()
                )));
            }
            let prob = exp_logits[true_label] / sum_exp;
            nll -= prob.max(1e-10).ln();
        }
        Ok(nll / n_samples as f64)
    }
    /// Transform logits to calibrated probabilities using temperature scaling
    pub fn transform(&self, logits: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(MLError::InvalidInput(
                "Scaler must be fitted before transform".to_string(),
            ));
        }
        let mut calibrated_probs = Array2::zeros((logits.nrows(), logits.ncols()));
        for i in 0..logits.nrows() {
            let scaled_logits = logits.row(i).mapv(|x| x / self.temperature);
            let max_logit = scaled_logits
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_logits: Vec<f64> = scaled_logits
                .iter()
                .map(|&x| (x - max_logit).exp())
                .collect();
            let sum_exp: f64 = exp_logits.iter().sum();
            for j in 0..logits.ncols() {
                calibrated_probs[(i, j)] = exp_logits[j] / sum_exp;
            }
        }
        Ok(calibrated_probs)
    }
    /// Fit and transform in one step
    pub fn fit_transform(
        &mut self,
        logits: &Array2<f64>,
        labels: &Array1<usize>,
    ) -> Result<Array2<f64>> {
        self.fit(logits, labels)?;
        self.transform(logits)
    }
    /// Get the fitted temperature parameter
    pub fn temperature(&self) -> Option<f64> {
        if self.fitted {
            Some(self.temperature)
        } else {
            None
        }
    }
}
/// Vector Scaling - extension of temperature scaling with class-specific parameters
/// Uses diagonal weight matrix and bias vector for more flexible calibration
/// Particularly effective when different classes have different calibration needs
#[derive(Debug, Clone)]
pub struct VectorScaler {
    /// Diagonal weight matrix (one parameter per class)
    weights: Option<Array1<f64>>,
    /// Bias vector (one parameter per class)
    biases: Option<Array1<f64>>,
    /// Whether the scaler has been fitted
    fitted: bool,
}
impl VectorScaler {
    /// Create a new vector scaler
    pub fn new() -> Self {
        Self {
            weights: None,
            biases: None,
            fitted: false,
        }
    }
    /// Fit the vector scaler to logits and true labels
    /// Uses negative log-likelihood minimization with L-BFGS-B optimization
    pub fn fit(&mut self, logits: &Array2<f64>, labels: &Array1<usize>) -> Result<()> {
        if logits.nrows() != labels.len() {
            return Err(MLError::InvalidInput(
                "Logits and labels must have same number of samples".to_string(),
            ));
        }
        let n_samples = logits.nrows();
        let n_classes = logits.ncols();
        if n_samples < 2 {
            return Err(MLError::InvalidInput(
                "Need at least 2 samples for calibration".to_string(),
            ));
        }
        let mut weights = Array1::ones(n_classes);
        let mut biases = Array1::zeros(n_classes);
        let learning_rate = 0.01;
        let max_iter = 200;
        let tolerance = 1e-6;
        let mut prev_nll = f64::INFINITY;
        for iter in 0..max_iter {
            let nll = self.compute_nll_vec(logits, labels, &weights, &biases)?;
            if (prev_nll - nll).abs() < tolerance {
                break;
            }
            prev_nll = nll;
            let epsilon = 1e-6;
            let mut weight_grads = Array1::zeros(n_classes);
            let mut bias_grads = Array1::zeros(n_classes);
            for j in 0..n_classes {
                let mut weights_plus = weights.clone();
                weights_plus[j] += epsilon;
                let nll_plus = self.compute_nll_vec(logits, labels, &weights_plus, &biases)?;
                weight_grads[j] = (nll_plus - nll) / epsilon;
                let mut biases_plus = biases.clone();
                biases_plus[j] += epsilon;
                let nll_plus = self.compute_nll_vec(logits, labels, &weights, &biases_plus)?;
                bias_grads[j] = (nll_plus - nll) / epsilon;
            }
            weights = &weights - &weight_grads.mapv(|g| learning_rate * g);
            biases = &biases - &bias_grads.mapv(|g| learning_rate * g);
            weights.mapv_inplace(|w| w.max(0.01));
            if weight_grads.iter().all(|&g| g.abs() < tolerance)
                && bias_grads.iter().all(|&g| g.abs() < tolerance)
            {
                break;
            }
        }
        self.weights = Some(weights);
        self.biases = Some(biases);
        self.fitted = true;
        Ok(())
    }
    /// Compute negative log-likelihood for given weights and biases
    fn compute_nll_vec(
        &self,
        logits: &Array2<f64>,
        labels: &Array1<usize>,
        weights: &Array1<f64>,
        biases: &Array1<f64>,
    ) -> Result<f64> {
        let mut nll = 0.0;
        let n_samples = logits.nrows();
        for i in 0..n_samples {
            let scaled_logits = logits.row(i).to_owned() * weights + biases;
            let max_logit = scaled_logits
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_logits: Vec<f64> = scaled_logits
                .iter()
                .map(|&x| (x - max_logit).exp())
                .collect();
            let sum_exp: f64 = exp_logits.iter().sum();
            let true_label = labels[i];
            if true_label >= exp_logits.len() {
                return Err(MLError::InvalidInput(format!(
                    "Label {} out of bounds for {} classes",
                    true_label,
                    exp_logits.len()
                )));
            }
            let prob = exp_logits[true_label] / sum_exp;
            nll -= prob.max(1e-10).ln();
        }
        Ok(nll / n_samples as f64)
    }
    /// Transform logits to calibrated probabilities using vector scaling
    pub fn transform(&self, logits: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(MLError::InvalidInput(
                "Scaler must be fitted before transform".to_string(),
            ));
        }
        let weights = self.weights.as_ref().unwrap();
        let biases = self.biases.as_ref().unwrap();
        let mut calibrated_probs = Array2::zeros((logits.nrows(), logits.ncols()));
        for i in 0..logits.nrows() {
            let scaled_logits = logits.row(i).to_owned() * weights + biases;
            let max_logit = scaled_logits
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_logits: Vec<f64> = scaled_logits
                .iter()
                .map(|&x| (x - max_logit).exp())
                .collect();
            let sum_exp: f64 = exp_logits.iter().sum();
            for j in 0..logits.ncols() {
                calibrated_probs[(i, j)] = exp_logits[j] / sum_exp;
            }
        }
        Ok(calibrated_probs)
    }
    /// Fit and transform in one step
    pub fn fit_transform(
        &mut self,
        logits: &Array2<f64>,
        labels: &Array1<usize>,
    ) -> Result<Array2<f64>> {
        self.fit(logits, labels)?;
        self.transform(logits)
    }
    /// Get the fitted parameters (weights, biases)
    pub fn parameters(&self) -> Option<(Array1<f64>, Array1<f64>)> {
        if self.fitted {
            Some((
                self.weights.as_ref().unwrap().clone(),
                self.biases.as_ref().unwrap().clone(),
            ))
        } else {
            None
        }
    }
}
