//! Utility functions and helper types for advanced quantum algorithms

use scirs2_core::Complex64;
use std::f64::consts::PI;

/// Helper function for complex phase calculation
pub fn complex_phase(phase: f64) -> Complex {
    Complex {
        re: phase.cos(),
        im: phase.sin(),
    }
}

/// Simple complex number representation
#[derive(Debug, Clone)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn abs(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn norm_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

/// Validate parameter ranges for quantum algorithms
pub fn validate_parameters(params: &[f64]) -> bool {
    params.iter().all(|&p| p >= 0.0 && p <= 2.0 * PI)
}

/// Normalize parameter values to valid range
pub fn normalize_parameters(params: &mut [f64]) {
    for param in params.iter_mut() {
        *param = param.max(0.0).min(2.0 * PI);
    }
}

/// Estimate problem complexity based on size and structure
pub fn estimate_problem_complexity(num_variables: usize, density: f64) -> f64 {
    let size_factor = (num_variables as f64).log2();
    let density_factor = density.max(0.1).min(1.0);
    size_factor * density_factor
}

/// Calculate relative improvement between values
pub fn calculate_relative_improvement(old_value: f64, new_value: f64) -> f64 {
    if old_value.abs() < 1e-8 {
        if new_value.abs() < 1e-8 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (old_value - new_value) / old_value.abs()
    }
}

/// Linear interpolation between two values
pub fn linear_interpolate(start: f64, end: f64, fraction: f64) -> f64 {
    start + fraction * (end - start)
}

/// Compute running average with decay factor
pub fn running_average(current_avg: f64, new_value: f64, decay_factor: f64) -> f64 {
    decay_factor * current_avg + (1.0 - decay_factor) * new_value
}

/// Check if a value has converged within tolerance
pub fn has_converged(current: f64, previous: f64, tolerance: f64) -> bool {
    (current - previous).abs() < tolerance
}

/// Compute exponential moving average
pub fn exponential_moving_average(values: &[f64], alpha: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let mut ema = Vec::with_capacity(values.len());
    ema.push(values[0]);

    for i in 1..values.len() {
        let new_ema = alpha * values[i] + (1.0 - alpha) * ema[i - 1];
        ema.push(new_ema);
    }

    ema
}

/// Generate Fibonacci sequence for adaptive depth selection
pub fn fibonacci_sequence(n: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1];
    }

    let mut fib = vec![1, 1];
    for i in 2..n {
        let next = fib[i - 1] + fib[i - 2];
        fib.push(next);
    }
    fib
}

/// Calculate golden ratio increment
pub fn golden_ratio_increment(current: usize) -> usize {
    ((current as f64) * 1.618) as usize
}

/// Compute autocorrelation at lag
pub fn autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag {
        return 0.0;
    }

    let n = data.len() - lag;
    let mean = data.iter().sum::<f64>() / data.len() as f64;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..n {
        let x_i = data[i] - mean;
        let x_lag = data[i + lag] - mean;
        numerator += x_i * x_lag;
        denominator += x_i * x_i;
    }

    if denominator > 1e-8 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Compute moving window statistics
#[derive(Debug, Clone)]
pub struct WindowStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

impl WindowStats {
    pub fn new(data: &[f64]) -> Self {
        if data.is_empty() {
            return Self {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std = variance.sqrt();
        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        Self {
            mean,
            std,
            min,
            max,
        }
    }
}

/// Performance tracking utilities
#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    values: Vec<f64>,
    max_size: usize,
}

impl PerformanceTracker {
    pub fn new(max_size: usize) -> Self {
        Self {
            values: Vec::new(),
            max_size,
        }
    }

    pub fn add_value(&mut self, value: f64) {
        self.values.push(value);
        if self.values.len() > self.max_size {
            self.values.remove(0);
        }
    }

    pub fn recent_improvement(&self, window_size: usize) -> f64 {
        if self.values.len() < window_size {
            return 0.0;
        }

        let recent_start = self.values.len() - window_size;
        let recent_values = &self.values[recent_start..];

        if recent_values.len() < 2 {
            return 0.0;
        }

        let initial = recent_values[0];
        let final_val = recent_values[recent_values.len() - 1];

        calculate_relative_improvement(initial, final_val)
    }

    pub fn is_stagnating(&self, threshold: f64, window_size: usize) -> bool {
        let improvement = self.recent_improvement(window_size);
        improvement.abs() < threshold
    }

    pub fn get_trend(&self, window_size: usize) -> f64 {
        if self.values.len() < window_size {
            return 0.0;
        }

        let recent_start = self.values.len() - window_size;
        let recent_values = &self.values[recent_start..];

        // Simple linear trend calculation
        let n = recent_values.len() as f64;
        let sum_x = (0..recent_values.len()).sum::<usize>() as f64;
        let sum_y = recent_values.iter().sum::<f64>();
        let sum_xy = recent_values
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum::<f64>();
        let sum_x2 = (0..recent_values.len())
            .map(|i| (i as f64).powi(2))
            .sum::<f64>();

        let denominator = n * sum_x2 - sum_x * sum_x;
        if denominator.abs() < 1e-8 {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / denominator
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_phase() {
        let phase = PI / 4.0;
        let complex_val = complex_phase(phase);

        assert!((complex_val.re - (PI / 4.0).cos()).abs() < 1e-10);
        assert!((complex_val.im - (PI / 4.0).sin()).abs() < 1e-10);
    }

    #[test]
    fn test_parameter_validation() {
        let valid_params = vec![0.0, PI, 2.0 * PI];
        let invalid_params = vec![-1.0, 3.0 * PI];

        assert!(validate_parameters(&valid_params));
        assert!(!validate_parameters(&invalid_params));
    }

    #[test]
    fn test_parameter_normalization() {
        let mut params = vec![-1.0, PI, 3.0 * PI];
        normalize_parameters(&mut params);

        assert!(validate_parameters(&params));
        assert_eq!(params[0], 0.0);
        assert_eq!(params[1], PI);
        assert_eq!(params[2], 2.0 * PI);
    }

    #[test]
    fn test_relative_improvement() {
        assert_eq!(calculate_relative_improvement(10.0, 8.0), 0.2);
        assert_eq!(calculate_relative_improvement(8.0, 10.0), -0.25);
        assert_eq!(calculate_relative_improvement(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_fibonacci_sequence() {
        let fib = fibonacci_sequence(5);
        assert_eq!(fib, vec![1, 1, 2, 3, 5]);
    }

    #[test]
    fn test_exponential_moving_average() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = exponential_moving_average(&values, 0.5);

        assert_eq!(ema.len(), values.len());
        assert_eq!(ema[0], 1.0);
        // Each subsequent value should be a weighted average
        assert!(ema[1] > 1.0 && ema[1] < 2.0);
    }

    #[test]
    fn test_window_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = WindowStats::new(&data);

        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!(stats.std > 0.0);
    }

    #[test]
    fn test_performance_tracker() {
        let mut tracker = PerformanceTracker::new(5);

        tracker.add_value(10.0);
        tracker.add_value(8.0);
        tracker.add_value(6.0);

        let improvement = tracker.recent_improvement(3);
        assert!(improvement > 0.0); // Decreasing values = improvement

        let trend = tracker.get_trend(3);
        assert!(trend < 0.0); // Negative trend = decreasing values
    }
}
