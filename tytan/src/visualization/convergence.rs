//! Convergence visualization for optimization algorithms
//!
//! This module provides tools for tracking and visualizing the convergence
//! of quantum annealing and optimization algorithms.

use crate::optimization::{
    adaptive::PerformanceMetrics, penalty::ConstraintViolation, tuning::TuningEvaluation,
};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "scirs")]
use crate::scirs_stub::scirs2_plot::{Annotation, MultiPlot, Plot2D};

/// Convergence plot configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceConfig {
    /// Window size for moving average
    pub smoothing_window: usize,
    /// Show confidence intervals
    pub show_confidence: bool,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Show constraint violations
    pub show_constraints: bool,
    /// Logarithmic y-axis for objective
    pub log_scale: bool,
    /// Show best so far line
    pub show_best: bool,
    /// Show theoretical bounds if available
    pub show_bounds: bool,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            smoothing_window: 10,
            show_confidence: true,
            confidence_level: 0.95,
            show_constraints: true,
            log_scale: false,
            show_best: true,
            show_bounds: false,
        }
    }
}

/// Convergence data tracker
pub struct ConvergencePlot {
    config: ConvergenceConfig,
    objective_history: Vec<f64>,
    constraint_history: Vec<HashMap<String, f64>>,
    parameter_history: Vec<HashMap<String, f64>>,
    time_history: Vec<std::time::Duration>,
    metadata: HashMap<String, String>,
}

/// Convergence analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceAnalysis {
    pub final_objective: f64,
    pub best_objective: f64,
    pub improvement_rate: f64,
    pub convergence_iteration: Option<usize>,
    pub total_iterations: usize,
    pub total_time: std::time::Duration,
    pub constraint_satisfaction: ConstraintSatisfaction,
    pub parameter_stability: ParameterStability,
    pub convergence_metrics: ConvergenceMetrics,
}

/// Constraint satisfaction metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintSatisfaction {
    pub final_violations: HashMap<String, f64>,
    pub satisfaction_rate: f64,
    pub convergence_iterations: HashMap<String, usize>,
    pub max_violation: f64,
}

/// Parameter stability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterStability {
    pub final_values: HashMap<String, f64>,
    pub variation_coefficients: HashMap<String, f64>,
    pub convergence_windows: HashMap<String, (usize, usize)>,
    pub stable_parameters: Vec<String>,
}

/// Convergence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetrics {
    pub convergence_rate: f64,
    pub oscillation_index: f64,
    pub plateau_regions: Vec<(usize, usize)>,
    pub improvement_events: Vec<ImprovementEvent>,
    pub estimated_optimum: Option<f64>,
}

/// Improvement event in optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementEvent {
    pub iteration: usize,
    pub old_value: f64,
    pub new_value: f64,
    pub improvement: f64,
    pub relative_improvement: f64,
}

impl ConvergencePlot {
    /// Create new convergence plotter
    pub fn new(config: ConvergenceConfig) -> Self {
        Self {
            config,
            objective_history: Vec::new(),
            constraint_history: Vec::new(),
            parameter_history: Vec::new(),
            time_history: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add optimization iteration data
    pub fn add_iteration(
        &mut self,
        objective: f64,
        constraints: HashMap<String, f64>,
        parameters: HashMap<String, f64>,
        elapsed: std::time::Duration,
    ) {
        self.objective_history.push(objective);
        self.constraint_history.push(constraints);
        self.parameter_history.push(parameters);
        self.time_history.push(elapsed);
    }

    /// Add metadata
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Analyze convergence
    pub fn analyze(&self) -> Result<ConvergenceAnalysis, Box<dyn std::error::Error>> {
        if self.objective_history.is_empty() {
            return Err("No convergence data to analyze".into());
        }

        let final_objective = *self.objective_history.last().unwrap();
        let best_objective = self
            .objective_history
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        // Calculate improvement rate
        let initial_objective = self.objective_history[0];
        let improvement = initial_objective - final_objective;
        let improvement_rate = improvement / initial_objective.abs();

        // Find convergence iteration
        let convergence_iteration = self.find_convergence_iteration()?;

        // Analyze constraints
        let constraint_satisfaction = self.analyze_constraint_satisfaction()?;

        // Analyze parameter stability
        let parameter_stability = self.analyze_parameter_stability()?;

        // Calculate convergence metrics
        let convergence_metrics = self.calculate_convergence_metrics()?;

        let total_time = self
            .time_history
            .last()
            .copied()
            .unwrap_or(std::time::Duration::from_secs(0));

        Ok(ConvergenceAnalysis {
            final_objective,
            best_objective,
            improvement_rate,
            convergence_iteration,
            total_iterations: self.objective_history.len(),
            total_time,
            constraint_satisfaction,
            parameter_stability,
            convergence_metrics,
        })
    }

    /// Find iteration where convergence occurred
    fn find_convergence_iteration(&self) -> Result<Option<usize>, Box<dyn std::error::Error>> {
        let tolerance = 1e-6;
        let window = self.config.smoothing_window;

        if self.objective_history.len() < window {
            return Ok(None);
        }

        // Look for plateau in moving average
        let smoothed = self.moving_average(&self.objective_history, window);

        for i in window..smoothed.len() {
            let recent_values = &smoothed[i - window..i];
            let mean = recent_values.iter().sum::<f64>() / window as f64;
            let variance = recent_values
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>()
                / window as f64;

            if variance.sqrt() < tolerance * mean.abs() {
                return Ok(Some(i));
            }
        }

        Ok(None)
    }

    /// Analyze constraint satisfaction over time
    fn analyze_constraint_satisfaction(
        &self,
    ) -> Result<ConstraintSatisfaction, Box<dyn std::error::Error>> {
        let final_violations = self.constraint_history.last().cloned().unwrap_or_default();

        let satisfied_count = final_violations
            .values()
            .filter(|&&v| v.abs() < 1e-6)
            .count();
        let total_constraints = final_violations.len().max(1);
        let satisfaction_rate = satisfied_count as f64 / total_constraints as f64;

        // Find when each constraint converged
        let mut convergence_iterations = HashMap::new();
        let tolerance = 1e-6;

        for (name, _) in &final_violations {
            for (i, constraints) in self.constraint_history.iter().enumerate() {
                if let Some(&violation) = constraints.get(name) {
                    if violation.abs() < tolerance {
                        convergence_iterations.insert(name.clone(), i);
                        break;
                    }
                }
            }
        }

        let max_violation = final_violations
            .values()
            .map(|v| v.abs())
            .fold(0.0, f64::max);

        Ok(ConstraintSatisfaction {
            final_violations,
            satisfaction_rate,
            convergence_iterations,
            max_violation,
        })
    }

    /// Analyze parameter stability
    fn analyze_parameter_stability(
        &self,
    ) -> Result<ParameterStability, Box<dyn std::error::Error>> {
        if self.parameter_history.is_empty() {
            return Ok(ParameterStability {
                final_values: HashMap::new(),
                variation_coefficients: HashMap::new(),
                convergence_windows: HashMap::new(),
                stable_parameters: Vec::new(),
            });
        }

        let final_values = self.parameter_history.last().cloned().unwrap_or_default();

        let mut variation_coefficients = HashMap::new();
        let mut convergence_windows = HashMap::new();
        let mut stable_parameters = Vec::new();

        for (param_name, _) in &final_values {
            // Extract parameter values over time
            let values: Vec<f64> = self
                .parameter_history
                .iter()
                .filter_map(|p| p.get(param_name).copied())
                .collect();

            if values.len() > 1 {
                // Calculate coefficient of variation
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
                let cv = if mean.abs() > 1e-10 {
                    variance.sqrt() / mean.abs()
                } else {
                    0.0
                };

                variation_coefficients.insert(param_name.clone(), cv);

                // Find convergence window
                let window = self.find_parameter_convergence_window(&values)?;
                if let Some((start, end)) = window {
                    convergence_windows.insert(param_name.clone(), (start, end));

                    // Consider stable if converged in first half
                    if end < values.len() / 2 {
                        stable_parameters.push(param_name.clone());
                    }
                }
            }
        }

        Ok(ParameterStability {
            final_values,
            variation_coefficients,
            convergence_windows,
            stable_parameters,
        })
    }

    /// Find convergence window for a parameter
    fn find_parameter_convergence_window(
        &self,
        values: &[f64],
    ) -> Result<Option<(usize, usize)>, Box<dyn std::error::Error>> {
        if values.len() < self.config.smoothing_window {
            return Ok(None);
        }

        let tolerance = 0.01; // 1% variation
        let window = self.config.smoothing_window;

        for start in 0..values.len() - window {
            let window_values = &values[start..start + window];
            let mean = window_values.iter().sum::<f64>() / window as f64;
            let max_dev = window_values
                .iter()
                .map(|v| (v - mean).abs())
                .fold(0.0, f64::max);

            if max_dev < tolerance * mean.abs() {
                // Found stable window, now find where it ends
                let mut end = start + window;
                while end < values.len() {
                    if (values[end] - mean).abs() > tolerance * mean.abs() {
                        break;
                    }
                    end += 1;
                }
                return Ok(Some((start, end)));
            }
        }

        Ok(None)
    }

    /// Calculate convergence metrics
    fn calculate_convergence_metrics(
        &self,
    ) -> Result<ConvergenceMetrics, Box<dyn std::error::Error>> {
        let objectives = &self.objective_history;

        // Convergence rate (exponential fit)
        let convergence_rate = self.estimate_convergence_rate(objectives)?;

        // Oscillation index
        let oscillation_index = self.calculate_oscillation_index(objectives)?;

        // Find plateau regions
        let plateau_regions = self.find_plateau_regions(objectives)?;

        // Identify improvement events
        let improvement_events = self.identify_improvement_events(objectives)?;

        // Estimate optimum (extrapolation)
        let estimated_optimum = self.estimate_optimum(objectives)?;

        Ok(ConvergenceMetrics {
            convergence_rate,
            oscillation_index,
            plateau_regions,
            improvement_events,
            estimated_optimum,
        })
    }

    /// Estimate convergence rate
    fn estimate_convergence_rate(
        &self,
        objectives: &[f64],
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if objectives.len() < 10 {
            return Ok(0.0);
        }

        // Fit exponential decay: f(t) = a * exp(-rate * t) + c
        // Using simple linear regression on log scale
        let best_so_far = self.best_so_far(objectives);
        let final_value = best_so_far.last().copied().unwrap_or(0.0);

        let mut x_sum = 0.0;
        let mut y_sum = 0.0;
        let mut xy_sum = 0.0;
        let mut xx_sum = 0.0;
        let mut count = 0;

        for (i, &value) in best_so_far.iter().enumerate() {
            let diff = (value - final_value).abs();
            if diff > 1e-10 {
                let x = i as f64;
                let y = diff.ln();

                x_sum += x;
                y_sum += y;
                xy_sum += x * y;
                xx_sum += x * x;
                count += 1;
            }
        }

        if count < 2 {
            return Ok(0.0);
        }

        let n = count as f64;
        let rate = -(n * xy_sum - x_sum * y_sum) / (n * xx_sum - x_sum * x_sum);

        Ok(rate.max(0.0))
    }

    /// Calculate oscillation index
    fn calculate_oscillation_index(
        &self,
        objectives: &[f64],
    ) -> Result<f64, Box<dyn std::error::Error>> {
        if objectives.len() < 3 {
            return Ok(0.0);
        }

        let mut direction_changes = 0;
        let mut total_variation = 0.0;

        for i in 1..objectives.len() - 1 {
            let diff1 = objectives[i] - objectives[i - 1];
            let diff2 = objectives[i + 1] - objectives[i];

            if diff1 * diff2 < 0.0 {
                direction_changes += 1;
            }

            total_variation += diff1.abs();
        }

        let path_length = total_variation;
        let direct_distance = (objectives.last().unwrap() - objectives[0]).abs();

        if path_length > 0.0 {
            Ok((path_length - direct_distance) / path_length)
        } else {
            Ok(0.0)
        }
    }

    /// Find plateau regions
    fn find_plateau_regions(
        &self,
        objectives: &[f64],
    ) -> Result<Vec<(usize, usize)>, Box<dyn std::error::Error>> {
        let mut plateaus = Vec::new();
        let tolerance = 1e-6;
        let min_length = 5;

        let mut start = 0;
        while start < objectives.len() {
            let base_value = objectives[start];
            let mut end = start + 1;

            while end < objectives.len()
                && (objectives[end] - base_value).abs() < tolerance * base_value.abs()
            {
                end += 1;
            }

            if end - start >= min_length {
                plateaus.push((start, end));
            }

            start = end;
        }

        Ok(plateaus)
    }

    /// Identify significant improvement events
    fn identify_improvement_events(
        &self,
        objectives: &[f64],
    ) -> Result<Vec<ImprovementEvent>, Box<dyn std::error::Error>> {
        let mut events = Vec::new();
        let threshold = 0.01; // 1% improvement

        let best_so_far = self.best_so_far(objectives);

        for i in 1..best_so_far.len() {
            if best_so_far[i] < best_so_far[i - 1] {
                let improvement = best_so_far[i - 1] - best_so_far[i];
                let relative = improvement / best_so_far[i - 1].abs();

                if relative > threshold {
                    events.push(ImprovementEvent {
                        iteration: i,
                        old_value: best_so_far[i - 1],
                        new_value: best_so_far[i],
                        improvement,
                        relative_improvement: relative,
                    });
                }
            }
        }

        Ok(events)
    }

    /// Estimate optimum value
    fn estimate_optimum(
        &self,
        objectives: &[f64],
    ) -> Result<Option<f64>, Box<dyn std::error::Error>> {
        if objectives.len() < 20 {
            return Ok(None);
        }

        // Use exponential extrapolation on best-so-far
        let best_so_far = self.best_so_far(objectives);
        let rate = self.estimate_convergence_rate(objectives)?;

        if rate > 0.0 {
            let current = best_so_far.last().copied().unwrap();
            let initial = best_so_far[0];

            // Estimate asymptotic value
            let estimated = current - (initial - current) * (-rate).exp();
            Ok(Some(estimated))
        } else {
            Ok(None)
        }
    }

    /// Calculate best-so-far trajectory
    fn best_so_far(&self, objectives: &[f64]) -> Vec<f64> {
        let mut best = Vec::with_capacity(objectives.len());
        let mut current_best = f64::INFINITY;

        for &obj in objectives {
            current_best = current_best.min(obj);
            best.push(current_best);
        }

        best
    }

    /// Calculate moving average
    fn moving_average(&self, data: &[f64], window: usize) -> Vec<f64> {
        if data.len() < window {
            return data.to_vec();
        }

        let mut result = Vec::with_capacity(data.len());

        // Fill initial values
        for i in 0..window / 2 {
            result.push(data[i]);
        }

        // Calculate moving average
        for i in window / 2..data.len() - window / 2 {
            let sum: f64 = data[i - window / 2..i + window / 2].iter().sum();
            result.push(sum / window as f64);
        }

        // Fill final values
        for i in data.len() - window / 2..data.len() {
            result.push(data[i]);
        }

        result
    }

    /// Generate convergence plot
    pub fn plot(&self) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "scirs")]
        {
            use crate::scirs_stub::scirs2_plot::{Figure, Subplot};

            let mut fig = Figure::new();
            let iterations: Vec<f64> = (0..self.objective_history.len())
                .map(|i| i as f64)
                .collect();

            // Main objective plot
            let subplot = fig.add_subplot(2, 2, 1)?;

            // Raw objectives
            subplot
                .plot(&iterations, &self.objective_history)
                .set_label("Objective")
                .set_alpha(0.5);

            // Best so far
            if self.config.show_best {
                let best_so_far = self.best_so_far(&self.objective_history);
                subplot
                    .plot(&iterations, &best_so_far)
                    .set_label("Best so far")
                    .set_linewidth(2.0);
            }

            // Smoothed line
            if self.config.smoothing_window > 1 {
                let smoothed =
                    self.moving_average(&self.objective_history, self.config.smoothing_window);
                subplot
                    .plot(&iterations, &smoothed)
                    .set_label("Smoothed")
                    .set_linestyle("--");
            }

            subplot
                .set_xlabel("Iteration")
                .set_ylabel("Objective")
                .set_title("Convergence Plot");
            subplot.legend();

            if self.config.log_scale {
                subplot.set_yscale("log");
            }

            // Constraint violations plot
            if self.config.show_constraints && !self.constraint_history.is_empty() {
                let subplot = fig.add_subplot(2, 2, 2)?;

                // Plot each constraint
                let mut constraint_names: Vec<_> = self
                    .constraint_history
                    .iter()
                    .flat_map(|c| c.keys())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                constraint_names.sort();

                for name in constraint_names {
                    let violations: Vec<f64> = self
                        .constraint_history
                        .iter()
                        .map(|c| c.get(name).copied().unwrap_or(0.0).abs())
                        .collect();

                    subplot.plot(&iterations, &violations).set_label(name);
                }

                subplot
                    .set_xlabel("Iteration")
                    .set_ylabel("Violation")
                    .set_title("Constraint Violations")
                    .set_yscale("log");
                subplot.legend();
            }

            // Parameter evolution
            if !self.parameter_history.is_empty() {
                let subplot = fig.add_subplot(2, 2, 3)?;

                let param_names: Vec<_> = self.parameter_history
                    .iter()
                    .flat_map(|p| p.keys())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .take(5) // Limit to 5 parameters for clarity
                    .collect();

                for name in param_names {
                    let values: Vec<f64> = self
                        .parameter_history
                        .iter()
                        .map(|p| p.get(name).copied().unwrap_or(0.0))
                        .collect();

                    // Normalize to [0, 1]
                    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                    if max_val > min_val {
                        let normalized: Vec<f64> = values
                            .iter()
                            .map(|v| (v - min_val) / (max_val - min_val))
                            .collect();

                        subplot.plot(&iterations, &normalized).set_label(name);
                    }
                }

                subplot
                    .set_xlabel("Iteration")
                    .set_ylabel("Normalized Value")
                    .set_title("Parameter Evolution");
                subplot.legend();
            }

            // Convergence metrics
            let analysis = self.analyze()?;
            let subplot = fig.add_subplot(2, 2, 4)?;

            // Display key metrics as text
            let text = format!(
                "Final objective: {:.4e}\n\
                 Best objective: {:.4e}\n\
                 Improvement: {:.2}%\n\
                 Convergence rate: {:.4}\n\
                 Oscillation: {:.2}\n\
                 Constraint satisfaction: {:.2}%",
                analysis.final_objective,
                analysis.best_objective,
                analysis.improvement_rate * 100.0,
                analysis.convergence_metrics.convergence_rate,
                analysis.convergence_metrics.oscillation_index,
                analysis.constraint_satisfaction.satisfaction_rate * 100.0
            );

            subplot
                .text(0.1, 0.9, &text)
                .set_fontsize(10)
                .set_verticalalignment("top");
            subplot.set_axis_off();

            fig.show()?;
        }

        #[cfg(not(feature = "scirs"))]
        {
            // Export plot data
            self.export_plot_data("convergence_plot.json")?;
            println!("Convergence data exported to convergence_plot.json");
        }

        Ok(())
    }

    /// Export plot data
    pub fn export_plot_data(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let export = ConvergenceExport {
            config: self.config.clone(),
            objective_history: self.objective_history.clone(),
            constraint_history: self.constraint_history.clone(),
            parameter_history: self.parameter_history.clone(),
            time_history: self.time_history.clone(),
            metadata: self.metadata.clone(),
            analysis: self.analyze().ok(),
            timestamp: std::time::SystemTime::now(),
        };

        let json = serde_json::to_string_pretty(&export)?;
        std::fs::write(path, json)?;

        Ok(())
    }
}

/// Convergence data export format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceExport {
    pub config: ConvergenceConfig,
    pub objective_history: Vec<f64>,
    pub constraint_history: Vec<HashMap<String, f64>>,
    pub parameter_history: Vec<HashMap<String, f64>>,
    pub time_history: Vec<std::time::Duration>,
    pub metadata: HashMap<String, String>,
    pub analysis: Option<ConvergenceAnalysis>,
    pub timestamp: std::time::SystemTime,
}

/// Create and plot convergence from optimization data
pub fn plot_convergence(
    objectives: Vec<f64>,
    constraints: Option<Vec<HashMap<String, f64>>>,
    config: Option<ConvergenceConfig>,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = config.unwrap_or_default();
    let mut plotter = ConvergencePlot::new(config);

    // Add data
    let constraints = constraints.unwrap_or_else(|| vec![HashMap::new(); objectives.len()]);
    let elapsed = std::time::Duration::from_secs(1);

    for (i, (obj, constr)) in objectives.iter().zip(constraints.iter()).enumerate() {
        plotter.add_iteration(*obj, constr.clone(), HashMap::new(), elapsed * i as u32);
    }

    plotter.plot()
}

/// Create convergence tracker from tuning evaluations
pub fn track_tuning_convergence(
    evaluations: &[TuningEvaluation],
    config: Option<ConvergenceConfig>,
) -> ConvergencePlot {
    let config = config.unwrap_or_default();
    let mut plotter = ConvergencePlot::new(config);

    for eval in evaluations {
        plotter.add_iteration(
            eval.objective_value,
            eval.constraint_violations.clone(),
            eval.parameters.clone(),
            eval.evaluation_time,
        );
    }

    plotter
}

/// Create convergence tracker from adaptive optimization
pub fn track_adaptive_convergence(
    performance_history: &[PerformanceMetrics],
    parameter_history: &[HashMap<String, f64>],
    config: Option<ConvergenceConfig>,
) -> ConvergencePlot {
    let config = config.unwrap_or_default();
    let mut plotter = ConvergencePlot::new(config);

    let elapsed = std::time::Duration::from_secs(1);

    for (i, (perf, params)) in performance_history
        .iter()
        .zip(parameter_history.iter())
        .enumerate()
    {
        plotter.add_iteration(
            perf.best_energy,
            perf.constraint_violations.clone(),
            params.clone(),
            elapsed * i as u32,
        );
    }

    plotter
}
