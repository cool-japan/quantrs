//! Sklearn-compatible feature selection algorithms

use super::SklearnEstimator;
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Select K Best features (sklearn-compatible)
pub struct SelectKBest {
    score_func: String,
    k: usize,
    fitted: bool,
    selected_features_: Option<Vec<usize>>,
}

impl SelectKBest {
    pub fn new(score_func: &str, k: usize) -> Self {
        Self {
            score_func: score_func.to_string(),
            k,
            fitted: false,
            selected_features_: None,
        }
    }

    /// Get selected features
    pub fn get_support(&self) -> Option<&Vec<usize>> {
        self.selected_features_.as_ref()
    }
}

impl SklearnEstimator for SelectKBest {
    #[allow(non_snake_case)]
    fn fit(&mut self, X: &Array2<f64>, _y: Option<&Array1<f64>>) -> Result<()> {
        // Mock implementation - select first k features
        let features: Vec<usize> = (0..self.k.min(X.ncols())).collect();
        self.selected_features_ = Some(features);
        self.fitted = true;
        Ok(())
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("score_func".to_string(), self.score_func.clone());
        params.insert("k".to_string(), self.k.to_string());
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            match key.as_str() {
                "k" => {
                    self.k = value.parse().map_err(|_| {
                        MLError::InvalidConfiguration(format!("Invalid k parameter: {}", value))
                    })?;
                }
                "score_func" => {
                    self.score_func = value;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

/// Variance threshold feature selector
pub struct VarianceThreshold {
    /// Threshold
    threshold: f64,
    /// Variances
    variances: Option<Array1<f64>>,
    /// Mask of selected features
    mask: Option<Vec<bool>>,
}

impl VarianceThreshold {
    /// Create new VarianceThreshold
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            variances: None,
            mask: None,
        }
    }

    /// Fit the selector
    #[allow(non_snake_case)]
    pub fn fit(&mut self, X: &Array2<f64>) -> Result<()> {
        let n_features = X.ncols();
        let n_samples = X.nrows() as f64;
        let mut variances = Array1::zeros(n_features);
        let mut mask = vec![false; n_features];

        for j in 0..n_features {
            // Compute mean
            let mean = X.column(j).sum() / n_samples;
            // Compute variance
            let var = X.column(j).mapv(|x| (x - mean).powi(2)).sum() / n_samples;
            variances[j] = var;
            mask[j] = var > self.threshold;
        }

        self.variances = Some(variances);
        self.mask = Some(mask);
        Ok(())
    }

    /// Transform the data
    #[allow(non_snake_case)]
    pub fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let mask = self
            .mask
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("VarianceThreshold not fitted".to_string()))?;

        let selected_cols: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();

        if selected_cols.is_empty() {
            return Err(MLError::InvalidConfiguration(
                "No features selected".to_string(),
            ));
        }

        let mut result = Array2::zeros((X.nrows(), selected_cols.len()));
        for (new_j, &old_j) in selected_cols.iter().enumerate() {
            for i in 0..X.nrows() {
                result[[i, new_j]] = X[[i, old_j]];
            }
        }

        Ok(result)
    }

    /// Fit and transform
    #[allow(non_snake_case)]
    pub fn fit_transform(&mut self, X: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(X)?;
        self.transform(X)
    }

    /// Get variances
    pub fn variances(&self) -> Option<&Array1<f64>> {
        self.variances.as_ref()
    }

    /// Get feature mask
    pub fn get_support(&self) -> Option<&Vec<bool>> {
        self.mask.as_ref()
    }
}
