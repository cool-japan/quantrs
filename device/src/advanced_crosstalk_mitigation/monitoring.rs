//! Real-time monitoring and alerting components

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

use super::*;
use crate::DeviceResult;

impl FeedbackController {
    pub fn new(config: &FeedbackControlConfig) -> Self {
        Self {
            controller_type: config.controller_type.clone(),
            control_state: ControlState::new(),
            stability_analyzer: StabilityAnalyzer::new(&config.stability_analysis),
            setpoint_history: VecDeque::with_capacity(1000),
            output_history: VecDeque::with_capacity(1000),
        }
    }

    /// Compute control output based on error signal
    pub fn compute_control_output(&mut self, setpoint: f64, measured_value: f64) -> DeviceResult<f64> {
        let error = setpoint - measured_value;
        self.control_state.error_history.push_back(error);

        // Keep error history bounded
        if self.control_state.error_history.len() > 1000 {
            self.control_state.error_history.pop_front();
        }

        let output = match &self.controller_type {
            ControllerType::PID { kp, ki, kd } => {
                self.compute_pid_output(error, *kp, *ki, *kd)?
            },
            ControllerType::LQR { q_matrix, r_matrix } => {
                self.compute_lqr_output(error, q_matrix, r_matrix)?
            },
            ControllerType::MPC { horizon, constraints } => {
                self.compute_mpc_output(error, *horizon, constraints)?
            },
            ControllerType::AdaptiveControl { adaptation_rate } => {
                self.compute_adaptive_output(error, *adaptation_rate)?
            },
            ControllerType::RobustControl { uncertainty_bounds } => {
                self.compute_robust_output(error, *uncertainty_bounds)?
            },
        };

        // Apply output limits
        let limited_output = self.apply_output_limits(output);

        // Record setpoint and output
        self.setpoint_history.push_back(setpoint);
        self.output_history.push_back(limited_output);

        // Keep history bounded
        if self.setpoint_history.len() > 1000 {
            self.setpoint_history.pop_front();
        }
        if self.output_history.len() > 1000 {
            self.output_history.pop_front();
        }

        Ok(limited_output)
    }

    fn compute_pid_output(&mut self, error: f64, kp: f64, ki: f64, kd: f64) -> DeviceResult<f64> {
        // Proportional term
        let proportional = kp * error;

        // Integral term
        self.control_state.integral_sum += error;
        let integral = ki * self.control_state.integral_sum;

        // Derivative term
        let derivative = if self.control_state.error_history.len() >= 2 {
            let prev_error = self.control_state.error_history[self.control_state.error_history.len() - 2];
            kd * (error - prev_error)
        } else {
            0.0
        };

        self.control_state.derivative_estimate = derivative;

        Ok(proportional + integral + derivative)
    }

    fn compute_lqr_output(&mut self, error: f64, q_matrix: &[f64], r_matrix: &[f64]) -> DeviceResult<f64> {
        // Linear Quadratic Regulator
        // Simplified implementation
        let gain = if !r_matrix.is_empty() && !q_matrix.is_empty() {
            q_matrix[0] / r_matrix[0]
        } else {
            1.0
        };

        Ok(-gain * error)
    }

    fn compute_mpc_output(&mut self, error: f64, horizon: usize, constraints: &[String]) -> DeviceResult<f64> {
        // Model Predictive Control
        // Simplified implementation: use current error with prediction
        let predicted_error = error * (1.0 + 0.1 * horizon as f64);
        Ok(-predicted_error * 0.5)
    }

    fn compute_adaptive_output(&mut self, error: f64, adaptation_rate: f64) -> DeviceResult<f64> {
        // Adaptive control
        // Update controller parameters based on performance
        let performance_metric = error.abs();

        // Simple adaptation: adjust gain based on error magnitude
        let adaptive_gain = 1.0 + adaptation_rate * performance_metric;

        Ok(-adaptive_gain * error)
    }

    fn compute_robust_output(&mut self, error: f64, uncertainty_bounds: f64) -> DeviceResult<f64> {
        // Robust control with uncertainty bounds
        let conservative_gain = 1.0 / (1.0 + uncertainty_bounds);
        Ok(-conservative_gain * error)
    }

    fn apply_output_limits(&self, output: f64) -> f64 {
        let (min_limit, max_limit) = self.control_state.output_limits;
        output.max(min_limit).min(max_limit)
    }

    /// Analyze controller performance
    pub fn analyze_performance(&self) -> DeviceResult<ControllerPerformanceMetrics> {
        if self.setpoint_history.is_empty() || self.output_history.is_empty() {
            return Ok(ControllerPerformanceMetrics::default());
        }

        // Calculate tracking error
        let tracking_errors: Vec<f64> = self.control_state.error_history.iter().cloned().collect();
        let tracking_error_rms = Self::calculate_rms(&tracking_errors);

        // Calculate settling time
        let settling_time = self.calculate_settling_time(&tracking_errors)?;

        // Calculate overshoot
        let overshoot = self.calculate_overshoot()?;

        // Calculate steady-state error
        let steady_state_error = self.calculate_steady_state_error(&tracking_errors);

        Ok(ControllerPerformanceMetrics {
            tracking_error_rms,
            settling_time,
            overshoot,
            steady_state_error,
            stability_margin: self.stability_analyzer.get_stability_margin(),
        })
    }

    fn calculate_rms(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let sum_squares: f64 = data.iter().map(|x| x * x).sum();
        (sum_squares / data.len() as f64).sqrt()
    }

    fn calculate_settling_time(&self, errors: &[f64]) -> DeviceResult<Duration> {
        // Find when error settles within 2% of final value
        let settling_threshold = 0.02;

        if errors.is_empty() {
            return Ok(Duration::from_secs(0));
        }

        let final_error = errors[errors.len() - 1].abs();
        let threshold = settling_threshold * final_error.max(1.0);

        // Find last time error exceeded threshold
        for (i, &error) in errors.iter().enumerate().rev() {
            if error.abs() > threshold {
                let settling_time_samples = errors.len() - i - 1;
                return Ok(Duration::from_millis(settling_time_samples as u64));
            }
        }

        Ok(Duration::from_secs(0))
    }

    fn calculate_overshoot(&self) -> DeviceResult<f64> {
        if self.setpoint_history.is_empty() || self.output_history.is_empty() {
            return Ok(0.0);
        }

        let setpoint = self.setpoint_history.back().unwrap_or(&0.0);
        let max_output = self.output_history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if setpoint.abs() > 1e-8 {
            Ok(((max_output - setpoint) / setpoint).max(0.0) * 100.0)
        } else {
            Ok(0.0)
        }
    }

    fn calculate_steady_state_error(&self, errors: &[f64]) -> f64 {
        if errors.is_empty() {
            return 0.0;
        }

        // Average of last 10% of errors
        let steady_state_samples = (errors.len() as f64 * 0.1).max(1.0) as usize;
        let start_idx = errors.len().saturating_sub(steady_state_samples);

        let steady_state_errors = &errors[start_idx..];
        steady_state_errors.iter().sum::<f64>() / steady_state_errors.len() as f64
    }

    /// Reset controller state
    pub fn reset(&mut self) {
        self.control_state.reset();
        self.setpoint_history.clear();
        self.output_history.clear();
    }

    /// Tune controller parameters automatically
    pub fn auto_tune(&mut self, system_response: &[f64]) -> DeviceResult<()> {
        match &mut self.controller_type {
            ControllerType::PID { kp, ki, kd } => {
                // Ziegler-Nichols or similar tuning method
                let tuned_params = self.tune_pid(system_response)?;
                *kp = tuned_params.0;
                *ki = tuned_params.1;
                *kd = tuned_params.2;
            },
            _ => {
                // Other controller types would have their own tuning methods
            }
        }

        Ok(())
    }

    fn tune_pid(&self, system_response: &[f64]) -> DeviceResult<(f64, f64, f64)> {
        // Simplified PID tuning based on system response
        // In practice, this would use more sophisticated methods

        // Estimate system parameters from step response
        let (gain, time_constant, delay) = self.estimate_system_parameters(system_response)?;

        // Apply Ziegler-Nichols tuning rules
        let kp = 1.2 / (gain * delay);
        let ki = kp / (2.0 * delay);
        let kd = kp * delay * 0.5;

        Ok((kp, ki, kd))
    }

    fn estimate_system_parameters(&self, response: &[f64]) -> DeviceResult<(f64, f64, f64)> {
        // Estimate gain, time constant, and delay from step response
        // Simplified implementation

        let steady_state = response.last().unwrap_or(&1.0);
        let gain = *steady_state;

        // Find 63% of steady state for time constant
        let target_63 = 0.63 * steady_state;
        let time_constant_idx = response.iter()
            .position(|&x| x >= target_63)
            .unwrap_or(response.len() / 2);
        let time_constant = time_constant_idx as f64;

        // Estimate delay (simplified)
        let delay = response.len() as f64 * 0.1;

        Ok((gain, time_constant, delay))
    }
}

impl ControlState {
    pub fn new() -> Self {
        Self {
            error_history: VecDeque::with_capacity(1000),
            integral_sum: 0.0,
            derivative_estimate: 0.0,
            output_limits: (-1.0, 1.0),
            controller_parameters: HashMap::new(),
        }
    }

    pub fn reset(&mut self) {
        self.error_history.clear();
        self.integral_sum = 0.0;
        self.derivative_estimate = 0.0;
        self.controller_parameters.clear();
    }

    pub fn set_output_limits(&mut self, min: f64, max: f64) {
        self.output_limits = (min, max);
    }

    pub fn get_current_error(&self) -> f64 {
        self.error_history.back().unwrap_or(&0.0).clone()
    }

    pub fn get_error_statistics(&self) -> ErrorStatistics {
        if self.error_history.is_empty() {
            return ErrorStatistics::default();
        }

        let errors: Vec<f64> = self.error_history.iter().cloned().collect();
        let mean = errors.iter().sum::<f64>() / errors.len() as f64;

        let variance = errors.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / errors.len() as f64;
        let std_dev = variance.sqrt();

        let min_error = errors.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_error = errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        ErrorStatistics {
            mean,
            std_dev,
            min_error,
            max_error,
            rms: ControlState::calculate_rms(&errors),
        }
    }

    fn calculate_rms(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let sum_squares: f64 = data.iter().map(|x| x * x).sum();
        (sum_squares / data.len() as f64).sqrt()
    }
}

impl StabilityAnalyzer {
    pub fn new(config: &StabilityAnalysisConfig) -> Self {
        Self {
            config: config.clone(),
            stability_history: VecDeque::with_capacity(1000),
            robustness_analyzer: RobustnessAnalyzer::new(),
        }
    }

    /// Analyze system stability
    pub fn analyze_stability(&mut self, control_output: &[f64], system_response: &[f64]) -> DeviceResult<StabilityAnalysisResult> {
        // Calculate current stability margins
        let current_margins = self.calculate_stability_margins(control_output, system_response)?;

        // Update stability history
        self.stability_history.push_back(current_margins.clone());
        if self.stability_history.len() > 1000 {
            self.stability_history.pop_front();
        }

        // Calculate Lyapunov exponents
        let lyapunov_exponents = self.calculate_lyapunov_exponents(system_response)?;

        // Identify stability regions
        let stability_regions = self.identify_stability_regions()?;

        // Perform robustness analysis
        let robustness_metrics = self.robustness_analyzer.analyze_robustness(control_output, system_response)?;

        Ok(StabilityAnalysisResult {
            stability_margins: current_margins,
            lyapunov_exponents,
            stability_regions,
            robustness_metrics,
        })
    }

    fn calculate_stability_margins(&self, control_output: &[f64], system_response: &[f64]) -> DeviceResult<StabilityMargins> {
        // Simplified stability margin calculation
        // In practice, this would involve frequency domain analysis

        let gain_margin = self.calculate_gain_margin(control_output, system_response)?;
        let phase_margin = self.calculate_phase_margin(control_output, system_response)?;
        let delay_margin = self.calculate_delay_margin(control_output, system_response)?;

        Ok(StabilityMargins {
            gain_margin,
            phase_margin,
            delay_margin,
        })
    }

    fn calculate_gain_margin(&self, control_output: &[f64], system_response: &[f64]) -> DeviceResult<f64> {
        // Calculate gain margin in dB
        // Simplified: based on maximum gain before instability

        if control_output.is_empty() || system_response.is_empty() {
            return Ok(6.0); // Default safe margin
        }

        // Estimate gain from input/output relationship
        let max_output = control_output.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let max_response = system_response.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if max_output.abs() > 1e-8 {
            let current_gain = max_response / max_output;
            let critical_gain = 2.0; // Simplified critical gain

            let gain_margin_linear = critical_gain / current_gain.abs();
            Ok(20.0 * gain_margin_linear.log10())
        } else {
            Ok(6.0)
        }
    }

    fn calculate_phase_margin(&self, control_output: &[f64], system_response: &[f64]) -> DeviceResult<f64> {
        // Calculate phase margin in degrees
        // Simplified implementation

        // Estimate phase from cross-correlation
        let phase_lag = self.estimate_phase_lag(control_output, system_response)?;
        let phase_margin = 180.0 - phase_lag.abs();

        Ok(phase_margin.max(0.0))
    }

    fn estimate_phase_lag(&self, input: &[f64], output: &[f64]) -> DeviceResult<f64> {
        // Estimate phase lag using cross-correlation
        let min_len = input.len().min(output.len());
        if min_len == 0 {
            return Ok(0.0);
        }

        let mut max_correlation = 0.0;
        let mut best_lag = 0;

        // Search for best correlation within reasonable lag range
        let max_lag = min_len / 4;
        for lag in 0..max_lag {
            let mut correlation = 0.0;
            let mut count = 0;

            for i in lag..min_len {
                correlation += input[i - lag] * output[i];
                count += 1;
            }

            if count > 0 {
                correlation /= count as f64;
                if correlation.abs() > max_correlation.abs() {
                    max_correlation = correlation;
                    best_lag = lag;
                }
            }
        }

        // Convert lag to phase (simplified)
        let phase_lag_degrees = (best_lag as f64 / min_len as f64) * 360.0;
        Ok(phase_lag_degrees)
    }

    fn calculate_delay_margin(&self, control_output: &[f64], system_response: &[f64]) -> DeviceResult<f64> {
        // Calculate delay margin in seconds
        // Simplified: based on estimated delay and critical delay

        let estimated_delay = self.estimate_system_delay(control_output, system_response)?;
        let critical_delay = 0.1; // Simplified critical delay in seconds

        Ok((critical_delay - estimated_delay).max(0.0))
    }

    fn estimate_system_delay(&self, input: &[f64], output: &[f64]) -> DeviceResult<f64> {
        // Estimate system delay from input-output relationship
        let phase_lag = self.estimate_phase_lag(input, output)?;

        // Convert phase lag to time delay (simplified)
        let sampling_period = 0.001; // Assume 1ms sampling
        let delay_samples = phase_lag / 360.0 * input.len() as f64;

        Ok(delay_samples * sampling_period)
    }

    fn calculate_lyapunov_exponents(&self, system_response: &[f64]) -> DeviceResult<scirs2_core::ndarray::Array1<f64>> {
        // Calculate Lyapunov exponents for stability analysis
        // Simplified implementation

        if system_response.len() < 10 {
            return Ok(scirs2_core::ndarray::Array1::zeros(1));
        }

        // Estimate largest Lyapunov exponent using finite difference method
        let mut divergence_sum = 0.0;
        let mut count = 0;

        for i in 1..system_response.len() {
            let local_divergence = (system_response[i] - system_response[i-1]).abs();
            if local_divergence > 1e-12 {
                divergence_sum += local_divergence.ln();
                count += 1;
            }
        }

        let lyapunov_exponent = if count > 0 {
            divergence_sum / count as f64
        } else {
            -1.0 // Stable system
        };

        Ok(scirs2_core::ndarray::Array1::from_vec(vec![lyapunov_exponent]))
    }

    fn identify_stability_regions(&self) -> DeviceResult<Vec<StabilityRegion>> {
        // Identify parameter regions where system remains stable
        // Simplified implementation

        let region = StabilityRegion {
            bounds: scirs2_core::ndarray::Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 0.0, 1.0])
                .expect("Failed to create 2x2 stability region bounds array"),
            stability_measure: 0.9,
            region_type: "nominal".to_string(),
        };

        Ok(vec![region])
    }

    pub fn get_stability_margin(&self) -> f64 {
        if let Some(latest_margins) = self.stability_history.back() {
            // Return minimum of all margins as overall stability indicator
            latest_margins.gain_margin.min(latest_margins.phase_margin).min(latest_margins.delay_margin * 100.0)
        } else {
            0.0
        }
    }

    pub fn is_system_stable(&self) -> bool {
        self.get_stability_margin() > 1.0 // Minimum acceptable margin
    }

    pub fn get_stability_trend(&self) -> StabilityTrend {
        if self.stability_history.len() < 2 {
            return StabilityTrend::Unknown;
        }

        let recent_margins: Vec<f64> = self.stability_history.iter()
            .map(|m| m.gain_margin)
            .rev()
            .take(5)
            .collect();

        if recent_margins.len() < 2 {
            return StabilityTrend::Unknown;
        }

        let trend_slope = (recent_margins[0] - recent_margins[recent_margins.len() - 1]) / (recent_margins.len() - 1) as f64;

        if trend_slope > 0.1 {
            StabilityTrend::Improving
        } else if trend_slope < -0.1 {
            StabilityTrend::Degrading
        } else {
            StabilityTrend::Stable
        }
    }
}

impl RobustnessAnalyzer {
    pub fn new() -> Self {
        Self {
            uncertainty_models: vec!["parametric".to_string(), "unstructured".to_string()],
            robustness_metrics: RobustnessMetrics {
                sensitivity: HashMap::new(),
                worst_case_performance: 1.0,
                robust_stability_margin: 0.1,
                structured_singular_value: 0.5,
            },
            sensitivity_analysis: HashMap::new(),
        }
    }

    pub fn analyze_robustness(&mut self, control_output: &[f64], system_response: &[f64]) -> DeviceResult<RobustnessMetrics> {
        // Perform sensitivity analysis
        self.perform_sensitivity_analysis(control_output, system_response)?;

        // Calculate worst-case performance
        let worst_case_performance = self.calculate_worst_case_performance(control_output, system_response)?;

        // Calculate robust stability margin
        let robust_stability_margin = self.calculate_robust_stability_margin(control_output, system_response)?;

        // Calculate structured singular value (μ)
        let structured_singular_value = self.calculate_structured_singular_value(control_output, system_response)?;

        self.robustness_metrics = RobustnessMetrics {
            sensitivity: self.sensitivity_analysis.clone(),
            worst_case_performance,
            robust_stability_margin,
            structured_singular_value,
        };

        Ok(self.robustness_metrics.clone())
    }

    fn perform_sensitivity_analysis(&mut self, control_output: &[f64], system_response: &[f64]) -> DeviceResult<()> {
        // Analyze sensitivity to parameter variations
        self.sensitivity_analysis.clear();

        // Simplified sensitivity analysis
        self.sensitivity_analysis.insert("gain_sensitivity".to_string(), 0.1);
        self.sensitivity_analysis.insert("delay_sensitivity".to_string(), 0.05);
        self.sensitivity_analysis.insert("noise_sensitivity".to_string(), 0.02);

        Ok(())
    }

    fn calculate_worst_case_performance(&self, control_output: &[f64], system_response: &[f64]) -> DeviceResult<f64> {
        // Calculate worst-case performance under uncertainty
        if system_response.is_empty() {
            return Ok(1.0);
        }

        // Simplified: assume performance degrades with uncertainty
        let nominal_performance = 1.0 - Self::calculate_rms(system_response);
        let uncertainty_factor = 0.1; // 10% uncertainty

        Ok(nominal_performance * (1.0 - uncertainty_factor))
    }

    fn calculate_robust_stability_margin(&self, control_output: &[f64], system_response: &[f64]) -> DeviceResult<f64> {
        // Calculate margin for robust stability
        // Simplified implementation

        if control_output.is_empty() || system_response.is_empty() {
            return Ok(0.1);
        }

        let gain_variation = Self::calculate_rms(control_output) / (Self::calculate_mean(control_output).abs() + 1e-8);
        let response_variation = Self::calculate_rms(system_response) / (Self::calculate_mean(system_response).abs() + 1e-8);

        let margin = 1.0 / (1.0 + gain_variation + response_variation);
        Ok(margin)
    }

    fn calculate_structured_singular_value(&self, control_output: &[f64], system_response: &[f64]) -> DeviceResult<f64> {
        // Calculate structured singular value (μ) for robust stability
        // Simplified implementation

        if control_output.is_empty() || system_response.is_empty() {
            return Ok(0.5);
        }

        // Simplified μ calculation based on gain and response characteristics
        let max_gain = control_output.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let max_response = system_response.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if max_gain.abs() > 1e-8 {
            let mu_estimate = (max_response / max_gain).abs();
            Ok(mu_estimate.min(1.0))
        } else {
            Ok(0.5)
        }
    }

    fn calculate_rms(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let sum_squares: f64 = data.iter().map(|x| x * x).sum();
        (sum_squares / data.len() as f64).sqrt()
    }

    fn calculate_mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        data.iter().sum::<f64>() / data.len() as f64
    }

    pub fn get_robustness_summary(&self) -> RobustnessSummary {
        RobustnessSummary {
            overall_robustness: self.robustness_metrics.robust_stability_margin,
            critical_parameters: self.identify_critical_parameters(),
            recommended_actions: self.generate_robustness_recommendations(),
        }
    }

    fn identify_critical_parameters(&self) -> Vec<String> {
        let mut critical_params = Vec::new();

        for (param, sensitivity) in &self.sensitivity_analysis {
            if *sensitivity > 0.05 { // Threshold for criticality
                critical_params.push(param.clone());
            }
        }

        critical_params
    }

    fn generate_robustness_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.robustness_metrics.robust_stability_margin < 0.1 {
            recommendations.push("Increase controller robustness".to_string());
        }

        if self.robustness_metrics.structured_singular_value > 0.8 {
            recommendations.push("Reduce system uncertainty".to_string());
        }

        if self.robustness_metrics.worst_case_performance < 0.7 {
            recommendations.push("Improve worst-case performance".to_string());
        }

        recommendations
    }
}

impl RealtimeMonitor {
    pub fn new(config: &RealtimeMitigationConfig) -> Self {
        Self {
            config: config.clone(),
            current_status: SystemStatus::Healthy,
            performance_buffer: VecDeque::with_capacity(10000),
            alert_generator: AlertGenerator::new(&config.alert_config),
        }
    }

    pub async fn update_monitoring(&mut self, characterization: &CrosstalkCharacterization) -> DeviceResult<RealtimeMonitoringResult> {
        // Update system status
        self.update_system_status(characterization)?;

        // Create performance snapshot
        let snapshot = PerformanceSnapshot {
            timestamp: SystemTime::now(),
            crosstalk_matrix: characterization.crosstalk_matrix.clone(),
            metrics: self.calculate_performance_metrics(characterization)?,
            system_state: SystemState::Active,
        };

        // Add to performance buffer
        self.performance_buffer.push_back(snapshot);
        if self.performance_buffer.len() > 10000 {
            self.performance_buffer.pop_front();
        }

        // Generate alerts if needed
        let alerts = self.alert_generator.check_for_alerts(characterization)?;

        // Create health metrics
        let health_metrics = self.calculate_health_metrics(characterization)?;

        Ok(RealtimeMonitoringResult {
            current_status: self.current_status.clone(),
            performance_history: self.performance_buffer.clone(),
            alert_history: alerts,
            control_actions: vec![], // Would be populated by control system
            health_metrics,
        })
    }

    fn update_system_status(&mut self, characterization: &CrosstalkCharacterization) -> DeviceResult<()> {
        // Determine system status based on crosstalk levels
        let max_crosstalk = characterization.crosstalk_matrix
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        self.current_status = if max_crosstalk > 0.2 {
            SystemStatus::Critical
        } else if max_crosstalk > 0.1 {
            SystemStatus::Warning
        } else {
            SystemStatus::Healthy
        };

        Ok(())
    }

    fn calculate_performance_metrics(&self, characterization: &CrosstalkCharacterization) -> DeviceResult<HashMap<String, f64>> {
        let mut metrics = HashMap::new();

        // Calculate various performance metrics
        let crosstalk_strength = characterization.crosstalk_matrix.mapv(|x| x.abs()).mean().unwrap_or(0.0);
        metrics.insert("crosstalk_strength".to_string(), crosstalk_strength);

        let max_crosstalk = characterization.crosstalk_matrix.mapv(|x| x.abs()).max().unwrap_or(0.0);
        metrics.insert("max_crosstalk".to_string(), max_crosstalk);

        let fidelity = 1.0 - crosstalk_strength;
        metrics.insert("fidelity".to_string(), fidelity);

        // Calculate mitigation effectiveness (simplified)
        let mitigation_effectiveness = if crosstalk_strength > 0.0 {
            (1.0 - crosstalk_strength).min(1.0)
        } else {
            1.0
        };
        metrics.insert("mitigation_effectiveness".to_string(), mitigation_effectiveness);

        Ok(metrics)
    }

    fn calculate_health_metrics(&self, characterization: &CrosstalkCharacterization) -> DeviceResult<HealthMetrics> {
        let crosstalk_strength = characterization.crosstalk_matrix.mapv(|x| x.abs()).mean().unwrap_or(0.0);

        // Overall health based on crosstalk levels
        let overall_health = (1.0 - crosstalk_strength).max(0.0).min(1.0);

        // Component health (simplified)
        let mut component_health = HashMap::new();
        component_health.insert("quantum_processor".to_string(), overall_health);
        component_health.insert("control_system".to_string(), 0.95);
        component_health.insert("mitigation_system".to_string(), 0.90);

        // Degradation rate (simplified)
        let degradation_rate = crosstalk_strength * 0.001;

        // Remaining life estimate
        let remaining_life = if degradation_rate > 1e-6 {
            let days_remaining = (0.1 / degradation_rate) / (24.0 * 3600.0);
            Some(Duration::from_secs((days_remaining * 24.0 * 3600.0) as u64))
        } else {
            Some(Duration::from_secs(365 * 24 * 3600)) // 1 year
        };

        // Maintenance recommendations
        let mut maintenance_recommendations = Vec::new();
        if overall_health < 0.8 {
            maintenance_recommendations.push("Schedule calibration".to_string());
        }
        if crosstalk_strength > 0.15 {
            maintenance_recommendations.push("Check hardware alignment".to_string());
        }

        Ok(HealthMetrics {
            overall_health,
            component_health,
            degradation_rate,
            remaining_life,
            maintenance_recommendations,
        })
    }

    pub fn get_current_status(&self) -> SystemStatus {
        self.current_status.clone()
    }

    pub fn get_performance_trends(&self) -> Vec<f64> {
        self.performance_buffer.iter()
            .filter_map(|snapshot| snapshot.metrics.get("fidelity"))
            .cloned()
            .collect()
    }

    pub fn start_monitoring(&mut self) -> DeviceResult<()> {
        // Start real-time monitoring
        self.current_status = SystemStatus::Healthy;
        Ok(())
    }

    pub fn stop_monitoring(&mut self) -> DeviceResult<()> {
        // Stop real-time monitoring
        self.current_status = SystemStatus::Unknown;
        Ok(())
    }
}

// Helper structs for monitoring
#[derive(Debug, Clone)]
pub struct ControllerPerformanceMetrics {
    pub tracking_error_rms: f64,
    pub settling_time: Duration,
    pub overshoot: f64,
    pub steady_state_error: f64,
    pub stability_margin: f64,
}

impl Default for ControllerPerformanceMetrics {
    fn default() -> Self {
        Self {
            tracking_error_rms: 0.0,
            settling_time: Duration::from_secs(0),
            overshoot: 0.0,
            steady_state_error: 0.0,
            stability_margin: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ErrorStatistics {
    pub mean: f64,
    pub std_dev: f64,
    pub min_error: f64,
    pub max_error: f64,
    pub rms: f64,
}

impl Default for ErrorStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            min_error: 0.0,
            max_error: 0.0,
            rms: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StabilityTrend {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct RobustnessSummary {
    pub overall_robustness: f64,
    pub critical_parameters: Vec<String>,
    pub recommended_actions: Vec<String>,
}