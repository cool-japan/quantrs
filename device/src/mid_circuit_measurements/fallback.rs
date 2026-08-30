//! Fallback implementations when SciRS2 is not available

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

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

/// Fallback optimization function: a real (if simple) derivative-free
/// coordinate/pattern search ("compass search"). Repeatedly tries stepping
/// each coordinate up or down by the current step size, accepting any move
/// that reduces the objective, and halves the step size once a full sweep
/// finds no improvement; stops once the step size is negligible or the
/// iteration budget is exhausted. `objective` is genuinely evaluated (many
/// times), so `fun`/`x` reflect real optimization rather than a fixed
/// constant.
pub fn minimize(
    objective: fn(&[f64]) -> f64,
    x0: &[f64],
    bounds: Option<&[(f64, f64)]>,
) -> Result<OptimizeResult, String> {
    if x0.is_empty() {
        return Err("minimize: x0 must not be empty".to_string());
    }

    let n = x0.len();
    let mut x = x0.to_vec();
    let mut best = objective(&x);
    let mut step = 1.0_f64;
    const MAX_ITERATIONS: usize = 100;
    const MIN_STEP: f64 = 1e-8;
    let mut iterations_used = 0;

    let clamp_dim = |i: usize, val: f64| -> f64 {
        bounds
            .and_then(|b| b.get(i))
            .map_or(val, |&(lo, hi)| val.clamp(lo, hi))
    };

    while step > MIN_STEP && iterations_used < MAX_ITERATIONS {
        let mut improved = false;
        for i in 0..n {
            for delta in [step, -step] {
                let mut candidate = x.clone();
                candidate[i] = clamp_dim(i, candidate[i] + delta);
                let value = objective(&candidate);
                if value < best {
                    best = value;
                    x = candidate;
                    improved = true;
                }
            }
        }
        iterations_used += 1;
        if !improved {
            step *= 0.5;
        }
    }

    Ok(OptimizeResult {
        x,
        fun: best,
        success: true,
        nit: iterations_used,
        message: "Fallback coordinate/pattern search (no SciRS2 optimizer available)".to_string(),
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

/// Solve the square linear system `a * x = b` via Gaussian elimination with
/// partial pivoting. Used by [`LinearRegression::fit`] to solve the normal
/// equations; returns an honest error for a singular/near-singular system
/// rather than a fabricated solution.
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err(format!(
            "solve_linear_system: dimension mismatch (A is {}x{}, b has length {})",
            a.nrows(),
            a.ncols(),
            b.len()
        ));
    }

    let mut aug = a.clone();
    let mut rhs = b.clone();

    for col in 0..n {
        let mut pivot_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                pivot_row = row;
            }
        }
        if max_val < 1e-12 {
            return Err(format!(
                "solve_linear_system: matrix is singular or nearly singular (pivot magnitude {max_val:.3e} at column {col})"
            ));
        }
        if pivot_row != col {
            for k in 0..n {
                aug.swap((col, k), (pivot_row, k));
            }
            rhs.swap(col, pivot_row);
        }

        let pivot_val = aug[[col, col]];
        for k in 0..n {
            aug[[col, k]] /= pivot_val;
        }
        rhs[col] /= pivot_val;

        for row in 0..n {
            if row != col {
                let factor = aug[[row, col]];
                if factor != 0.0 {
                    for k in 0..n {
                        let aug_col_k = aug[[col, k]];
                        aug[[row, k]] -= factor * aug_col_k;
                    }
                    let rhs_col = rhs[col];
                    rhs[row] -= factor * rhs_col;
                }
            }
        }
    }

    Ok(rhs)
}

/// Fallback linear regression implementation: real ordinary-least-squares
/// fitting via the normal equations (`(X^T X) beta = X^T y`, solved with
/// [`solve_linear_system`]), not a no-op. The design matrix is augmented
/// with a column of ones so the intercept is fitted along with the feature
/// coefficients.
pub struct LinearRegression {
    coefficients: Vec<f64>,
    intercept: f64,
}

impl LinearRegression {
    pub const fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            intercept: 0.0,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        if n_samples != y.len() {
            return Err(format!(
                "LinearRegression::fit: x has {n_samples} rows but y has {} elements",
                y.len()
            ));
        }
        if n_samples == 0 || n_features == 0 {
            return Err("LinearRegression::fit: x must be non-empty".to_string());
        }

        // Augment X with a column of ones (last column) for the intercept.
        let mut x_aug = Array2::<f64>::ones((n_samples, n_features + 1));
        for i in 0..n_samples {
            for j in 0..n_features {
                x_aug[[i, j]] = x[[i, j]];
            }
        }

        let xt = x_aug.t();
        let xtx = xt.dot(&x_aug);
        let xty = xt.dot(y);

        let beta = solve_linear_system(&xtx, &xty)?;

        self.coefficients = beta.iter().take(n_features).copied().collect();
        self.intercept = beta[n_features];
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let n_samples = x.nrows();
        let mut predictions = Array1::<f64>::zeros(n_samples);
        for i in 0..n_samples {
            let mut value = self.intercept;
            for (j, &coefficient) in self.coefficients.iter().enumerate() {
                if j < x.ncols() {
                    value += coefficient * x[[i, j]];
                }
            }
            predictions[i] = value;
        }
        predictions
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimize_actually_evaluates_objective_and_reduces_it() {
        // f(x) = (x0 - 3)^2 + (x1 + 2)^2, minimized at (3, -2) with f = 0.
        fn objective(x: &[f64]) -> f64 {
            (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2)
        }

        let x0 = [0.0, 0.0];
        let initial_value = objective(&x0);

        let result = minimize(objective, &x0, None).expect("minimize should succeed");

        assert!(
            result.fun < initial_value,
            "fallback optimizer must actually reduce the objective (got {} vs initial {})",
            result.fun,
            initial_value
        );
        assert!(
            (result.x[0] - 3.0).abs() < 0.1,
            "expected x0 near 3.0, got {}",
            result.x[0]
        );
        assert!(
            (result.x[1] + 2.0).abs() < 0.1,
            "expected x1 near -2.0, got {}",
            result.x[1]
        );
    }

    #[test]
    fn test_minimize_respects_bounds() {
        fn objective(x: &[f64]) -> f64 {
            (x[0] - 100.0).powi(2)
        }

        let x0 = [0.0];
        let bounds = [(-1.0, 1.0)];
        let result = minimize(objective, &x0, Some(&bounds)).expect("minimize should succeed");

        assert!(
            (-1.0..=1.0).contains(&result.x[0]),
            "optimizer must respect bounds, got x={}",
            result.x[0]
        );
    }

    #[test]
    fn test_linear_regression_recovers_known_line() {
        // y = 2*x + 1
        let x = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 3.0, 5.0, 7.0]);

        let mut model = LinearRegression::new();
        model
            .fit(&x, &y)
            .expect("fit should succeed for a well-posed system");

        let predictions = model.predict(&x);
        for i in 0..4 {
            assert!(
                (predictions[i] - y[i]).abs() < 1e-6,
                "prediction {} should match training target {} at index {i}",
                predictions[i],
                y[i]
            );
        }
    }

    #[test]
    fn test_linear_regression_rejects_mismatched_shapes() {
        let x = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0]); // wrong length

        let mut model = LinearRegression::new();
        assert!(
            model.fit(&x, &y).is_err(),
            "fit must honestly error on mismatched shapes rather than silently succeeding"
        );
    }

    #[test]
    fn test_solve_linear_system_basic() {
        // [[2, 0], [0, 3]] * x = [4, 9] => x = [2, 3]
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 3.0]).unwrap();
        let b = Array1::from_vec(vec![4.0, 9.0]);
        let x = solve_linear_system(&a, &b).expect("solve should succeed");
        assert!((x[0] - 2.0).abs() < 1e-9);
        assert!((x[1] - 3.0).abs() < 1e-9);
    }
}
