//! Scikit-learn compatibility layer for QuantRS2-ML
//!
//! This module provides a compatibility layer that mimics scikit-learn APIs,
//! allowing easy integration of quantum ML models with existing scikit-learn
//! workflows and pipelines.

mod classifiers;
mod clustering;
mod decomposition;
mod feature_selection;
pub mod metrics;
pub mod model_selection;
pub mod pipeline;
mod preprocessing;
mod regressors;

pub use classifiers::*;
pub use clustering::*;
pub use decomposition::*;
pub use feature_selection::*;
pub use model_selection::*;
pub use pipeline::*;
pub use preprocessing::*;
pub use regressors::*;

use crate::error::Result;
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Base estimator trait following scikit-learn conventions
pub trait SklearnEstimator: Send + Sync {
    /// Fit the model to training data
    #[allow(non_snake_case)]
    fn fit(&mut self, X: &Array2<f64>, y: Option<&Array1<f64>>) -> Result<()>;

    /// Get model parameters
    fn get_params(&self) -> HashMap<String, String>;

    /// Set model parameters
    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()>;

    /// Check if model is fitted
    fn is_fitted(&self) -> bool;

    /// Get feature names
    fn get_feature_names_out(&self) -> Vec<String> {
        vec![]
    }
}

/// Classifier mixin trait
pub trait SklearnClassifier: SklearnEstimator {
    /// Predict class labels
    #[allow(non_snake_case)]
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>>;

    /// Predict class probabilities
    #[allow(non_snake_case)]
    fn predict_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>>;

    /// Get unique class labels
    fn classes(&self) -> &[i32];

    /// Score the model (accuracy by default)
    #[allow(non_snake_case)]
    fn score(&self, X: &Array2<f64>, y: &Array1<i32>) -> Result<f64> {
        let predictions = self.predict(X)?;
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_label)| pred == true_label)
            .count();
        Ok(correct as f64 / y.len() as f64)
    }

    /// Get feature importances (optional)
    fn feature_importances(&self) -> Option<Array1<f64>> {
        None
    }

    /// Save model to file (optional)
    fn save(&self, _path: &str) -> Result<()> {
        Ok(())
    }
}

/// Regressor mixin trait
pub trait SklearnRegressor: SklearnEstimator {
    /// Predict continuous values
    #[allow(non_snake_case)]
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<f64>>;

    /// Score the model (R² by default)
    #[allow(non_snake_case)]
    fn score(&self, X: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let predictions = self.predict(X)?;
        let y_mean = y.mean().unwrap_or(0.0);

        let ss_res: f64 = y
            .iter()
            .zip(predictions.iter())
            .map(|(&true_val, &pred)| (true_val - pred).powi(2))
            .sum();

        let ss_tot: f64 = y.iter().map(|&val| (val - y_mean).powi(2)).sum();

        Ok(1.0 - ss_res / ss_tot)
    }
}

/// Extension trait for fitting with Array1<f64> directly
pub trait SklearnFit {
    #[allow(non_snake_case)]
    fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) -> Result<()>;
}

/// Clusterer mixin trait
pub trait SklearnClusterer: SklearnEstimator {
    /// Predict cluster labels
    #[allow(non_snake_case)]
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>>;

    /// Fit and predict in one step
    #[allow(non_snake_case)]
    fn fit_predict(&mut self, X: &Array2<f64>) -> Result<Array1<i32>> {
        SklearnEstimator::fit(self, X, None)?;
        self.predict(X)
    }

    /// Get cluster centers (if applicable)
    fn cluster_centers(&self) -> Option<&Array2<f64>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_standard_scaler() {
        let mut scaler = StandardScaler::new();

        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        scaler.fit(&X, None).expect("Fit should succeed");

        assert!(scaler.is_fitted());
    }

    #[test]
    fn test_minmax_scaler() {
        let scaler = MinMaxScaler::new();
        let params = scaler.get_params();
        assert!(params.contains_key("feature_range_min"));
    }

    #[test]
    fn test_label_encoder() {
        let encoder = LabelEncoder::new();
        assert!(!encoder.is_fitted());
    }

    #[test]
    fn test_pca() {
        let pca = PCA::new(2);
        let params = pca.get_params();
        assert_eq!(params.get("n_components"), Some(&"2".to_string()));
    }
}
