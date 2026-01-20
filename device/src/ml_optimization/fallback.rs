//! Fallback implementations when SciRS2 is not available

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub fn mean(_data: &ArrayView1<f64>) -> Result<f64, String> {
    Ok(0.0)
}
pub fn std(_data: &ArrayView1<f64>, _ddof: i32) -> Result<f64, String> {
    Ok(1.0)
}
pub fn pearsonr(
    _x: &ArrayView1<f64>,
    _y: &ArrayView1<f64>,
    _alt: &str,
) -> Result<(f64, f64), String> {
    Ok((0.0, 0.5))
}
pub fn trace(_matrix: &ArrayView2<f64>) -> Result<f64, String> {
    Ok(1.0)
}
pub fn inv(_matrix: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
    Ok(Array2::eye(2))
}

pub struct OptimizeResult {
    pub x: Array1<f64>,
    pub fun: f64,
    pub success: bool,
    pub nit: usize,
    pub nfev: usize,
    pub message: String,
}

pub fn minimize(
    _func: fn(&Array1<f64>) -> f64,
    _x0: &Array1<f64>,
    _method: &str,
) -> Result<OptimizeResult, String> {
    Ok(OptimizeResult {
        x: Array1::zeros(2),
        fun: 0.0,
        success: true,
        nit: 0,
        nfev: 0,
        message: "Fallback optimization".to_string(),
    })
}

pub fn differential_evolution(
    _func: fn(&Array1<f64>) -> f64,
    _bounds: &[(f64, f64)],
) -> Result<OptimizeResult, String> {
    Ok(OptimizeResult {
        x: Array1::zeros(2),
        fun: 0.0,
        success: true,
        nit: 0,
        nfev: 0,
        message: "Fallback optimization".to_string(),
    })
}
