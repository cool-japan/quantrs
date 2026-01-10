//! Sklearn-compatible decomposition algorithms

use super::SklearnEstimator;
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

/// Principal Component Analysis
pub struct PCA {
    n_components: usize,
    components: Option<Array2<f64>>,
    mean: Option<Array1<f64>>,
    explained_variance: Option<Array1<f64>>,
    explained_variance_ratio: Option<Array1<f64>>,
    fitted: bool,
}

impl PCA {
    /// Create new PCA
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            components: None,
            mean: None,
            explained_variance: None,
            explained_variance_ratio: None,
            fitted: false,
        }
    }

    /// Fit PCA
    #[allow(non_snake_case)]
    pub fn fit(&mut self, X: &Array2<f64>) -> Result<()> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Center data
        let mean = X
            .mean_axis(Axis(0))
            .ok_or_else(|| MLError::InvalidConfiguration("Failed to compute mean".to_string()))?;

        let mut centered = X.clone();
        for i in 0..n_samples {
            for j in 0..n_features {
                centered[[i, j]] -= mean[j];
            }
        }

        // Compute covariance matrix (simplified)
        let cov = centered.t().dot(&centered) / (n_samples - 1) as f64;

        // Power iteration for eigendecomposition (simplified)
        let n_comp = self.n_components.min(n_features);
        let mut components = Array2::zeros((n_comp, n_features));
        let mut variances = Array1::zeros(n_comp);

        for k in 0..n_comp {
            // Initialize random vector
            let mut v = Array1::from_vec((0..n_features).map(|i| ((i + k) as f64).sin()).collect());
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v.mapv_inplace(|x| x / norm);

            // Power iteration
            for _ in 0..100 {
                let mut new_v = cov.dot(&v);

                // Orthogonalize against previous components
                for prev_k in 0..k {
                    let prev_comp = components.row(prev_k);
                    let proj: f64 = new_v.iter().zip(prev_comp.iter()).map(|(a, b)| a * b).sum();
                    for (i, val) in new_v.iter_mut().enumerate() {
                        *val -= proj * prev_comp[i];
                    }
                }

                let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-10 {
                    new_v.mapv_inplace(|x| x / norm);
                }
                v = new_v;
            }

            // Store component and compute variance
            for j in 0..n_features {
                components[[k, j]] = v[j];
            }
            variances[k] = cov.dot(&v).dot(&v);
        }

        let total_var: f64 = variances.sum();
        let variance_ratio = variances.mapv(|v| v / total_var);

        self.components = Some(components);
        self.mean = Some(mean);
        self.explained_variance = Some(variances);
        self.explained_variance_ratio = Some(variance_ratio);
        self.fitted = true;

        Ok(())
    }

    /// Transform data
    #[allow(non_snake_case)]
    pub fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("PCA not fitted".to_string()))?;
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("PCA not fitted".to_string()))?;

        // Center and project
        let n_samples = X.nrows();
        let mut centered = X.clone();
        for i in 0..n_samples {
            for j in 0..X.ncols() {
                centered[[i, j]] -= mean[j];
            }
        }

        Ok(centered.dot(&components.t()))
    }

    /// Fit and transform
    #[allow(non_snake_case)]
    pub fn fit_transform(&mut self, X: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(X)?;
        self.transform(X)
    }

    /// Get explained variance ratio
    pub fn explained_variance_ratio(&self) -> Option<&Array1<f64>> {
        self.explained_variance_ratio.as_ref()
    }

    /// Get components
    pub fn components(&self) -> Option<&Array2<f64>> {
        self.components.as_ref()
    }
}

impl SklearnEstimator for PCA {
    #[allow(non_snake_case)]
    fn fit(&mut self, X: &Array2<f64>, _y: Option<&Array1<f64>>) -> Result<()> {
        PCA::fit(self, X)
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("n_components".to_string(), self.n_components.to_string());
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        if let Some(n) = params.get("n_components") {
            self.n_components = n
                .parse()
                .map_err(|_| MLError::InvalidConfiguration("Invalid n_components".to_string()))?;
        }
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}
