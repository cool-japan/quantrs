//! Data preprocessing utilities for QML

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2, Axis};

use super::*;
/// Normalize features to zero mean and unit variance
pub fn standardize(data: &Array2<f64>) -> Result<Array2<f64>> {
    let mean = data
        .mean_axis(Axis(0))
        .ok_or_else(|| MLError::ComputationError("Failed to compute mean".to_string()))?;
    let std = data.std_axis(Axis(0), 0.0);
    let mut normalized = data.clone();
    for i in 0..data.nrows() {
        for j in 0..data.ncols() {
            normalized[(i, j)] = (data[(i, j)] - mean[j]) / (std[j] + 1e-8);
        }
    }
    Ok(normalized)
}
/// Min-max normalization to [0, 1] range
pub fn min_max_normalize(data: &Array2<f64>) -> Result<Array2<f64>> {
    let mut normalized = data.clone();
    for j in 0..data.ncols() {
        let column = data.column(j);
        let min_val = column.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = column.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;
        if range > 1e-10 {
            for i in 0..data.nrows() {
                normalized[(i, j)] = (data[(i, j)] - min_val) / range;
            }
        }
    }
    Ok(normalized)
}
/// Robust scaling using median and IQR (Interquartile Range)
/// More robust to outliers than standardization
pub fn robust_scale(data: &Array2<f64>) -> Result<Array2<f64>> {
    let mut normalized = data.clone();
    for j in 0..data.ncols() {
        let mut column_values: Vec<f64> = data.column(j).to_vec();
        column_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = column_values.len();
        let median = if n % 2 == 0 {
            (column_values[n / 2 - 1] + column_values[n / 2]) / 2.0
        } else {
            column_values[n / 2]
        };
        let q1_idx = n / 4;
        let q3_idx = (3 * n) / 4;
        let q1 = column_values[q1_idx];
        let q3 = column_values[q3_idx];
        let iqr = q3 - q1;
        if iqr > 1e-10 {
            for i in 0..data.nrows() {
                normalized[(i, j)] = (data[(i, j)] - median) / iqr;
            }
        } else {
            for i in 0..data.nrows() {
                normalized[(i, j)] = data[(i, j)] - median;
            }
        }
    }
    Ok(normalized)
}
/// Quantile normalization - forces features to have the same distribution
/// Useful when features should be on the same scale but have different distributions
pub fn quantile_normalize(data: &Array2<f64>) -> Result<Array2<f64>> {
    let nrows = data.nrows();
    let ncols = data.ncols();
    let mut normalized = Array2::zeros((nrows, ncols));
    let mut sorted_columns: Vec<Vec<(usize, f64)>> = Vec::with_capacity(ncols);
    for j in 0..ncols {
        let mut indexed_values: Vec<(usize, f64)> = data
            .column(j)
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_columns.push(indexed_values);
    }
    let mut rank_means = vec![0.0; nrows];
    for i in 0..nrows {
        let mut sum = 0.0;
        for col in &sorted_columns {
            sum += col[i].1;
        }
        rank_means[i] = sum / ncols as f64;
    }
    for j in 0..ncols {
        for (rank, &(original_idx, _)) in sorted_columns[j].iter().enumerate() {
            normalized[(original_idx, j)] = rank_means[rank];
        }
    }
    Ok(normalized)
}
/// Max absolute scaling - scales features by maximum absolute value
/// Useful for sparse data where centering would destroy sparsity
pub fn max_abs_scale(data: &Array2<f64>) -> Result<Array2<f64>> {
    let mut normalized = data.clone();
    for j in 0..data.ncols() {
        let max_abs = data
            .column(j)
            .iter()
            .map(|x| x.abs())
            .fold(0.0f64, f64::max);
        if max_abs > 1e-10 {
            for i in 0..data.nrows() {
                normalized[(i, j)] = data[(i, j)] / max_abs;
            }
        }
    }
    Ok(normalized)
}
/// L1 normalization - normalize each sample (row) to unit L1 norm
/// Each row will sum to 1, useful for probability-like features
pub fn l1_normalize(data: &Array2<f64>) -> Result<Array2<f64>> {
    let mut normalized = data.clone();
    for i in 0..data.nrows() {
        let l1_norm: f64 = data.row(i).iter().map(|x| x.abs()).sum();
        if l1_norm > 1e-10 {
            for j in 0..data.ncols() {
                normalized[(i, j)] = data[(i, j)] / l1_norm;
            }
        }
    }
    Ok(normalized)
}
/// L2 normalization - normalize each sample (row) to unit L2 norm
/// Each row will have length 1, useful for cosine similarity
pub fn l2_normalize(data: &Array2<f64>) -> Result<Array2<f64>> {
    let mut normalized = data.clone();
    for i in 0..data.nrows() {
        let l2_norm: f64 = data.row(i).iter().map(|x| x * x).sum::<f64>().sqrt();
        if l2_norm > 1e-10 {
            for j in 0..data.ncols() {
                normalized[(i, j)] = data[(i, j)] / l2_norm;
            }
        }
    }
    Ok(normalized)
}
