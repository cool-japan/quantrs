//! Sklearn-compatible preprocessing utilities

use super::SklearnEstimator;
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

/// Standard Scaler (sklearn-compatible)
pub struct StandardScaler {
    mean_: Option<Array1<f64>>,
    scale_: Option<Array1<f64>>,
    fitted: bool,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            mean_: None,
            scale_: None,
            fitted: false,
        }
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl SklearnEstimator for StandardScaler {
    #[allow(non_snake_case)]
    fn fit(&mut self, X: &Array2<f64>, _y: Option<&Array1<f64>>) -> Result<()> {
        let mean = X.mean_axis(Axis(0)).ok_or_else(|| {
            MLError::InvalidInput("Cannot compute mean of empty array".to_string())
        })?;
        let std = X.std_axis(Axis(0), 0.0);

        self.mean_ = Some(mean);
        self.scale_ = Some(std);
        self.fitted = true;

        Ok(())
    }

    fn get_params(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    fn set_params(&mut self, _params: HashMap<String, String>) -> Result<()> {
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

/// MinMax scaler
pub struct MinMaxScaler {
    min: Option<Array1<f64>>,
    max: Option<Array1<f64>>,
    feature_range: (f64, f64),
    fitted: bool,
}

impl MinMaxScaler {
    /// Create new MinMaxScaler
    pub fn new() -> Self {
        Self {
            min: None,
            max: None,
            feature_range: (0.0, 1.0),
            fitted: false,
        }
    }

    /// Set feature range
    pub fn feature_range(mut self, min_val: f64, max_val: f64) -> Self {
        self.feature_range = (min_val, max_val);
        self
    }

    /// Fit the scaler
    #[allow(non_snake_case)]
    pub fn fit(&mut self, X: &Array2<f64>) -> Result<()> {
        let n_features = X.ncols();
        let mut min = Array1::zeros(n_features);
        let mut max = Array1::zeros(n_features);

        for j in 0..n_features {
            let col = X.column(j);
            min[j] = col.iter().cloned().fold(f64::INFINITY, f64::min);
            max[j] = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        }

        self.min = Some(min);
        self.max = Some(max);
        self.fitted = true;
        Ok(())
    }

    /// Transform data
    #[allow(non_snake_case)]
    pub fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let min = self
            .min
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Scaler not fitted".to_string()))?;
        let max = self
            .max
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Scaler not fitted".to_string()))?;

        let (range_min, range_max) = self.feature_range;
        let mut result = X.clone();

        for j in 0..X.ncols() {
            let scale = if (max[j] - min[j]).abs() > 1e-10 {
                (range_max - range_min) / (max[j] - min[j])
            } else {
                1.0
            };

            for i in 0..X.nrows() {
                result[[i, j]] = (X[[i, j]] - min[j]) * scale + range_min;
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
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl SklearnEstimator for MinMaxScaler {
    #[allow(non_snake_case)]
    fn fit(&mut self, X: &Array2<f64>, _y: Option<&Array1<f64>>) -> Result<()> {
        MinMaxScaler::fit(self, X)
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "feature_range_min".to_string(),
            self.feature_range.0.to_string(),
        );
        params.insert(
            "feature_range_max".to_string(),
            self.feature_range.1.to_string(),
        );
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        if let Some(min_str) = params.get("feature_range_min") {
            if let Some(max_str) = params.get("feature_range_max") {
                let min_val: f64 = min_str.parse().map_err(|_| {
                    MLError::InvalidConfiguration("Invalid feature_range_min".to_string())
                })?;
                let max_val: f64 = max_str.parse().map_err(|_| {
                    MLError::InvalidConfiguration("Invalid feature_range_max".to_string())
                })?;
                self.feature_range = (min_val, max_val);
            }
        }
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

/// Robust scaler (uses median and IQR)
pub struct RobustScaler {
    center: Option<Array1<f64>>,
    scale: Option<Array1<f64>>,
    with_centering: bool,
    with_scaling: bool,
    fitted: bool,
}

impl RobustScaler {
    /// Create new RobustScaler
    pub fn new() -> Self {
        Self {
            center: None,
            scale: None,
            with_centering: true,
            with_scaling: true,
            fitted: false,
        }
    }

    /// Fit the scaler
    #[allow(non_snake_case)]
    pub fn fit(&mut self, X: &Array2<f64>) -> Result<()> {
        let n_features = X.ncols();
        let mut center = Array1::zeros(n_features);
        let mut scale = Array1::zeros(n_features);

        for j in 0..n_features {
            let mut col: Vec<f64> = X.column(j).iter().cloned().collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let n = col.len();
            center[j] = col[n / 2]; // Median

            let q1 = col[n / 4];
            let q3 = col[3 * n / 4];
            scale[j] = if (q3 - q1).abs() > 1e-10 {
                q3 - q1
            } else {
                1.0
            };
        }

        self.center = Some(center);
        self.scale = Some(scale);
        self.fitted = true;
        Ok(())
    }

    /// Transform data
    #[allow(non_snake_case)]
    pub fn transform(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let center = self
            .center
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Scaler not fitted".to_string()))?;
        let scale = self
            .scale
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Scaler not fitted".to_string()))?;

        let mut result = X.clone();

        for j in 0..X.ncols() {
            for i in 0..X.nrows() {
                result[[i, j]] = if self.with_centering {
                    (X[[i, j]] - center[j]) / scale[j]
                } else {
                    X[[i, j]] / scale[j]
                };
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
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// Label encoder
pub struct LabelEncoder {
    classes: Vec<String>,
    fitted: bool,
}

impl LabelEncoder {
    /// Create new LabelEncoder
    pub fn new() -> Self {
        Self {
            classes: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the encoder
    pub fn fit(&mut self, y: &[String]) {
        let mut classes: Vec<String> = y.iter().cloned().collect();
        classes.sort();
        classes.dedup();
        self.classes = classes;
        self.fitted = true;
    }

    /// Transform labels to integers
    pub fn transform(&self, y: &[String]) -> Result<Array1<i32>> {
        if !self.fitted {
            return Err(MLError::ModelNotTrained("Encoder not fitted".to_string()));
        }

        let encoded: Vec<i32> = y
            .iter()
            .map(|label| {
                self.classes
                    .iter()
                    .position(|c| c == label)
                    .map(|p| p as i32)
                    .unwrap_or(-1)
            })
            .collect();

        Ok(Array1::from_vec(encoded))
    }

    /// Inverse transform
    pub fn inverse_transform(&self, y: &Array1<i32>) -> Result<Vec<String>> {
        if !self.fitted {
            return Err(MLError::ModelNotTrained("Encoder not fitted".to_string()));
        }

        let decoded: Vec<String> = y
            .iter()
            .map(|&idx| {
                if idx >= 0 && (idx as usize) < self.classes.len() {
                    self.classes[idx as usize].clone()
                } else {
                    "unknown".to_string()
                }
            })
            .collect();

        Ok(decoded)
    }

    /// Fit and transform
    pub fn fit_transform(&mut self, y: &[String]) -> Result<Array1<i32>> {
        self.fit(y);
        self.transform(y)
    }

    /// Get classes
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// Check if fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// One-hot encoder
pub struct OneHotEncoder {
    categories: Vec<Vec<String>>,
    fitted: bool,
    sparse: bool,
}

impl OneHotEncoder {
    /// Create new OneHotEncoder
    pub fn new() -> Self {
        Self {
            categories: Vec::new(),
            fitted: false,
            sparse: false,
        }
    }

    /// Set sparse output
    pub fn sparse(mut self, sparse: bool) -> Self {
        self.sparse = sparse;
        self
    }

    /// Fit the encoder
    #[allow(non_snake_case)]
    pub fn fit(&mut self, X: &Array2<String>) {
        self.categories = Vec::new();

        for j in 0..X.ncols() {
            let mut cats: Vec<String> = X.column(j).iter().cloned().collect();
            cats.sort();
            cats.dedup();
            self.categories.push(cats);
        }

        self.fitted = true;
    }

    /// Transform to one-hot encoding
    #[allow(non_snake_case)]
    pub fn transform(&self, X: &Array2<String>) -> Result<Array2<f64>> {
        if !self.fitted {
            return Err(MLError::ModelNotTrained("Encoder not fitted".to_string()));
        }

        let total_cols: usize = self.categories.iter().map(|c| c.len()).sum();
        let mut result = Array2::zeros((X.nrows(), total_cols));

        let mut col_offset = 0;
        for j in 0..X.ncols() {
            let cats = &self.categories[j];
            for i in 0..X.nrows() {
                if let Some(idx) = cats.iter().position(|c| c == &X[[i, j]]) {
                    result[[i, col_offset + idx]] = 1.0;
                }
            }
            col_offset += cats.len();
        }

        Ok(result)
    }

    /// Fit and transform
    #[allow(non_snake_case)]
    pub fn fit_transform(&mut self, X: &Array2<String>) -> Result<Array2<f64>> {
        self.fit(X);
        self.transform(X)
    }
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Quantum Feature Encoder (sklearn-compatible)
pub struct QuantumFeatureEncoder {
    encoding_type: String,
    normalization: String,
    fitted: bool,
}

impl QuantumFeatureEncoder {
    pub fn new(encoding_type: &str, normalization: &str) -> Self {
        Self {
            encoding_type: encoding_type.to_string(),
            normalization: normalization.to_string(),
            fitted: false,
        }
    }
}

impl SklearnEstimator for QuantumFeatureEncoder {
    #[allow(non_snake_case)]
    fn fit(&mut self, _X: &Array2<f64>, _y: Option<&Array1<f64>>) -> Result<()> {
        self.fitted = true;
        Ok(())
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("encoding_type".to_string(), self.encoding_type.clone());
        params.insert("normalization".to_string(), self.normalization.clone());
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> Result<()> {
        for (key, value) in params {
            match key.as_str() {
                "encoding_type" => {
                    self.encoding_type = value;
                }
                "normalization" => {
                    self.normalization = value;
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
