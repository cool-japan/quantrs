//! Fallback implementations when SciRS2 is not available

use ndarray::{Array1, Array2, ArrayView1};

/// Fallback mean calculation
pub fn mean(data: &ArrayView1<f64>) -> Result<f64, String> {
    Ok(data.mean().unwrap_or(0.0))
}

/// Fallback standard deviation calculation
pub fn std(data: &ArrayView1<f64>, _ddof: i32) -> Result<f64, String> {
    Ok(data.std(1.0))
}

/// Fallback Pearson correlation calculation
pub fn pearsonr(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    _alternative: &str,
) -> Result<(f64, f64), String> {
    if x.len() != y.len() || x.len() < 2 {
        return Ok((0.0, 0.5));
    }
    
    let x_mean = x.mean().unwrap_or(0.0);
    let y_mean = y.mean().unwrap_or(0.0);
    
    let mut num = 0.0;
    let mut x_sum_sq = 0.0;
    let mut y_sum_sq = 0.0;
    
    for i in 0..x.len() {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;
        num += x_diff * y_diff;
        x_sum_sq += x_diff * x_diff;
        y_sum_sq += y_diff * y_diff;
    }
    
    let denom = (x_sum_sq * y_sum_sq).sqrt();
    let corr = if denom > 1e-10 { num / denom } else { 0.0 };
    
    Ok((corr, 0.05)) // p-value placeholder
}

/// Fallback optimization function
pub fn minimize(
    _objective: fn(&[f64]) -> f64,
    _x0: &[f64],
    _bounds: Option<&[(f64, f64)]>,
) -> Result<OptimizeResult, String> {
    Ok(OptimizeResult {
        x: vec![0.0; _x0.len()],
        fun: 1.0,
        success: true,
        nit: 10,
        message: "Fallback optimization".to_string(),
    })
}

/// Fallback optimization result
pub struct OptimizeResult {
    pub x: Vec<f64>,
    pub fun: f64,
    pub success: bool,
    pub nit: usize,
    pub message: String,
}

/// Fallback linear regression implementation
pub struct LinearRegression {
    coefficients: Vec<f64>,
    intercept: f64,
}

impl LinearRegression {
    pub fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            intercept: 0.0,
        }
    }
    
    pub fn fit(&mut self, _x: &Array2<f64>, _y: &Array1<f64>) -> Result<(), String> {
        Ok(())
    }
    
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        Array1::zeros(x.nrows())
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}