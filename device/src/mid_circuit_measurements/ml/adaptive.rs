//! Adaptive measurement management components

use super::super::config::AdaptiveConfig;
use super::super::results::*;
use crate::DeviceResult;
use std::collections::{HashMap, VecDeque};

/// Adaptive measurement manager with learning capabilities
pub struct AdaptiveMeasurementManager {
    config: AdaptiveConfig,
    adaptation_history: VecDeque<AdaptationEvent>,
    performance_baseline: Option<PerformanceBaseline>,
    learning_state: LearningState,
    drift_detector: DriftDetector,
    transfer_learning_engine: TransferLearningEngine,
}

impl AdaptiveMeasurementManager {
    /// Create new adaptive measurement manager
    pub fn new(config: &AdaptiveConfig) -> Self {
        Self {
            config: config.clone(),
            adaptation_history: VecDeque::with_capacity(1000),
            performance_baseline: None,
            learning_state: LearningState::new(),
            drift_detector: DriftDetector::new(),
            transfer_learning_engine: TransferLearningEngine::new(),
        }
    }

    /// Adapt measurement parameters based on performance
    pub async fn adapt_parameters(
        &mut self,
        current_performance: &PerformanceMetrics,
        measurement_history: &[MeasurementEvent],
    ) -> DeviceResult<AdaptiveLearningInsights> {
        if !self.config.enable_adaptive_scheduling {
            return Ok(AdaptiveLearningInsights::default());
        }

        // Update performance baseline
        self.update_performance_baseline(current_performance)?;

        // Detect performance drift
        let drift_detection = self.detect_performance_drift(current_performance)?;

        // Learn from recent performance
        let learning_progress = self
            .update_learning_state(current_performance, measurement_history)
            .await?;

        // Analyze performance trends
        let performance_trends = self.analyze_performance_trends()?;

        // Generate transfer learning insights
        let transfer_learning =
            self.analyze_transfer_learning_opportunities(measurement_history)?;

        // Record adaptation event
        self.record_adaptation_event(current_performance)?;

        Ok(AdaptiveLearningInsights {
            learning_progress,
            adaptation_history: self.get_recent_adaptations(),
            performance_trends,
            drift_detection,
            transfer_learning,
        })
    }

    /// Update performance baseline
    fn update_performance_baseline(
        &mut self,
        current_performance: &PerformanceMetrics,
    ) -> DeviceResult<()> {
        match &mut self.performance_baseline {
            Some(baseline) => {
                // Update baseline with exponential moving average
                let alpha = self.config.baseline_update_rate;
                baseline.measurement_success_rate = alpha
                    * current_performance.measurement_success_rate
                    + (1.0 - alpha) * baseline.measurement_success_rate;
                baseline.classical_efficiency = alpha * current_performance.classical_efficiency
                    + (1.0 - alpha) * baseline.classical_efficiency;
                baseline.circuit_fidelity = alpha * current_performance.circuit_fidelity
                    + (1.0 - alpha) * baseline.circuit_fidelity;
                baseline.measurement_error_rate = alpha
                    * current_performance.measurement_error_rate
                    + (1.0 - alpha) * baseline.measurement_error_rate;
                baseline.timing_overhead = alpha * current_performance.timing_overhead
                    + (1.0 - alpha) * baseline.timing_overhead;
            }
            None => {
                // Initialize baseline
                self.performance_baseline = Some(PerformanceBaseline {
                    measurement_success_rate: current_performance.measurement_success_rate,
                    classical_efficiency: current_performance.classical_efficiency,
                    circuit_fidelity: current_performance.circuit_fidelity,
                    measurement_error_rate: current_performance.measurement_error_rate,
                    timing_overhead: current_performance.timing_overhead,
                    established_at: std::time::SystemTime::now(),
                });
            }
        }

        Ok(())
    }

    /// Detect performance drift
    fn detect_performance_drift(
        &mut self,
        current_performance: &PerformanceMetrics,
    ) -> DeviceResult<DriftDetectionResults> {
        let baseline = match &self.performance_baseline {
            Some(baseline) => baseline,
            None => return Ok(DriftDetectionResults::default()),
        };

        // Calculate performance deviations
        let success_rate_drift = (current_performance.measurement_success_rate
            - baseline.measurement_success_rate)
            .abs();
        let efficiency_drift =
            (current_performance.classical_efficiency - baseline.classical_efficiency).abs();
        let fidelity_drift =
            (current_performance.circuit_fidelity - baseline.circuit_fidelity).abs();
        let error_rate_drift =
            (current_performance.measurement_error_rate - baseline.measurement_error_rate).abs();

        // Detect significant drift
        let drift_threshold = self.config.drift_threshold;
        let drift_detected = success_rate_drift > drift_threshold
            || efficiency_drift > drift_threshold
            || fidelity_drift > drift_threshold
            || error_rate_drift > drift_threshold;

        let drift_type = if drift_detected {
            if success_rate_drift > drift_threshold {
                Some(DriftType::PerformanceDegradation)
            } else if fidelity_drift > drift_threshold {
                Some(DriftType::QualityDrift)
            } else {
                Some(DriftType::ConceptDrift)
            }
        } else {
            None
        };

        let drift_magnitude =
            (success_rate_drift + efficiency_drift + fidelity_drift + error_rate_drift) / 4.0;

        // Generate recommendations
        let mut recommended_actions = Vec::new();
        if drift_detected {
            if success_rate_drift > drift_threshold {
                recommended_actions.push("Retrain measurement prediction models".to_string());
                recommended_actions.push("Adjust measurement parameters".to_string());
            }
            if efficiency_drift > drift_threshold {
                recommended_actions.push("Optimize classical processing pipeline".to_string());
            }
            if fidelity_drift > drift_threshold {
                recommended_actions.push("Recalibrate quantum measurement apparatus".to_string());
            }
        }

        // Update drift detector state
        self.drift_detector.update(current_performance)?;

        Ok(DriftDetectionResults {
            drift_detected,
            drift_type,
            drift_magnitude,
            detection_confidence: if drift_detected { 0.9 } else { 0.1 },
            recommended_actions,
        })
    }

    /// Update learning state
    async fn update_learning_state(
        &mut self,
        current_performance: &PerformanceMetrics,
        measurement_history: &[MeasurementEvent],
    ) -> DeviceResult<LearningProgress> {
        // Update learning metrics
        self.learning_state.iterations_completed += 1;

        // Calculate performance-based loss
        let target_success_rate = 0.99; // Target 99% success rate
        let current_loss =
            (target_success_rate - current_performance.measurement_success_rate).abs();
        self.learning_state.add_loss(current_loss);

        // Calculate accuracy based on recent performance
        let current_accuracy = current_performance.measurement_success_rate;
        self.learning_state.add_accuracy(current_accuracy);

        // Adaptive learning rate
        if self.learning_state.loss_history.len() > 5 {
            let recent_losses: Vec<f64> = self
                .learning_state
                .loss_history
                .iter()
                .rev()
                .take(5)
                .cloned()
                .collect();

            let loss_trend = self.calculate_trend(&recent_losses);

            if loss_trend > 0.0 {
                // Loss increasing - reduce learning rate
                self.learning_state.current_learning_rate *= 0.99; // Default decay rate
            } else if loss_trend < -0.01 {
                // Loss decreasing rapidly - increase learning rate
                self.learning_state.current_learning_rate *= 1.1;
            }

            // Clamp learning rate
            self.learning_state.current_learning_rate = self
                .learning_state
                .current_learning_rate
                .max(0.0001)
                .min(0.1);
        }

        // Update convergence status
        self.learning_state.convergence_status = self.assess_convergence_status()?;

        Ok(LearningProgress {
            iterations_completed: self.learning_state.iterations_completed,
            current_learning_rate: self.learning_state.current_learning_rate,
            loss_history: self.learning_state.get_loss_history(),
            accuracy_history: self.learning_state.get_accuracy_history(),
            convergence_status: self.learning_state.convergence_status.clone(),
        })
    }

    /// Analyze performance trends
    fn analyze_performance_trends(&self) -> DeviceResult<PerformanceTrends> {
        if self.adaptation_history.len() < 5 {
            return Ok(PerformanceTrends::default());
        }

        // Extract performance metrics from recent adaptations
        let recent_adaptations: Vec<&AdaptationEvent> =
            self.adaptation_history.iter().rev().take(10).collect();

        let success_rates: Vec<f64> = recent_adaptations
            .iter()
            .map(|event| event.performance_snapshot.measurement_success_rate)
            .collect();

        // Calculate trends
        let short_term_trend =
            self.calculate_trend_direction(&success_rates[..5.min(success_rates.len())]);
        let long_term_trend = self.calculate_trend_direction(&success_rates);
        let trend_strength = self.calculate_trend_strength(&success_rates);

        // Calculate volatility
        let volatility = if success_rates.len() > 1 {
            let mean = success_rates.iter().sum::<f64>() / success_rates.len() as f64;
            let variance = success_rates
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / success_rates.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Detect seasonal patterns (simplified)
        let seasonal_patterns = if success_rates.len() >= 8 {
            Some(self.detect_seasonal_patterns(&success_rates)?)
        } else {
            None
        };

        Ok(PerformanceTrends {
            short_term_trend,
            long_term_trend,
            trend_strength,
            seasonal_patterns: seasonal_patterns.map(|sp| {
                crate::mid_circuit_measurements::results::SeasonalityAnalysis {
                    periods: vec![sp.period_length],
                    seasonal_strength: sp.seasonal_strength,
                    seasonal_components: scirs2_core::ndarray::Array1::zeros(sp.period_length),
                    residual_components: scirs2_core::ndarray::Array1::zeros(sp.period_length),
                }
            }),
            volatility,
        })
    }

    /// Analyze transfer learning opportunities
    fn analyze_transfer_learning_opportunities(
        &mut self,
        measurement_history: &[MeasurementEvent],
    ) -> DeviceResult<TransferLearningInsights> {
        // Calculate transfer learning metrics
        let transfer_effectiveness = self
            .transfer_learning_engine
            .calculate_transfer_effectiveness(measurement_history)?;
        let domain_similarity = self
            .transfer_learning_engine
            .calculate_domain_similarity(measurement_history)?;
        let feature_transferability = self
            .transfer_learning_engine
            .analyze_feature_transferability(measurement_history)?;

        // Generate adaptation requirements
        let mut adaptation_requirements = Vec::new();
        if domain_similarity < 0.7 {
            adaptation_requirements.push("Significant domain adaptation required".to_string());
        }
        if transfer_effectiveness < 0.5 {
            adaptation_requirements.push("Limited transfer learning benefit expected".to_string());
        }

        // Generate recommendations
        let mut recommendations = Vec::new();
        if transfer_effectiveness > 0.8 {
            recommendations.push(
                "High transfer learning potential - proceed with knowledge transfer".to_string(),
            );
        } else if transfer_effectiveness > 0.5 {
            recommendations
                .push("Moderate transfer potential - use selective feature transfer".to_string());
        } else {
            recommendations
                .push("Low transfer potential - focus on domain-specific learning".to_string());
        }

        Ok(TransferLearningInsights {
            transfer_effectiveness,
            domain_similarity,
            feature_transferability,
            adaptation_requirements,
            recommendations,
        })
    }

    /// Record adaptation event
    fn record_adaptation_event(
        &mut self,
        current_performance: &PerformanceMetrics,
    ) -> DeviceResult<()> {
        let adaptation_event = AdaptationEvent {
            timestamp: std::time::SystemTime::now(),
            adaptation_type: AdaptationType::PerformanceOptimization,
            trigger: "Performance optimization trigger".to_string(),
            performance_before: self
                .performance_baseline
                .as_ref()
                .map(|b| b.measurement_success_rate)
                .unwrap_or(0.0),
            performance_after: current_performance.measurement_success_rate,
            performance_snapshot: current_performance.clone(),
            adaptation_magnitude: self.calculate_adaptation_magnitude(current_performance)?,
            success_indicator: if current_performance.measurement_success_rate > 0.95 {
                1.0
            } else {
                0.0
            },
            success: current_performance.measurement_success_rate
                > self
                    .performance_baseline
                    .as_ref()
                    .map(|b| b.measurement_success_rate)
                    .unwrap_or(0.0),
        };

        self.adaptation_history.push_back(adaptation_event);

        // Keep only recent adaptations
        if self.adaptation_history.len() > 1000 {
            self.adaptation_history.pop_front();
        }

        Ok(())
    }

    /// Get recent adaptations
    fn get_recent_adaptations(&self) -> Vec<AdaptationEvent> {
        self.adaptation_history
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect()
    }

    /// Calculate trend from data points
    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let x_sum = (0..values.len()).map(|i| i as f64).sum::<f64>();
        let y_sum = values.iter().sum::<f64>();
        let xy_sum = values
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum::<f64>();
        let x2_sum = (0..values.len()).map(|i| (i as f64).powi(2)).sum::<f64>();

        let denominator = n * x2_sum - x_sum * x_sum;
        if denominator.abs() > 1e-10 {
            (n * xy_sum - x_sum * y_sum) / denominator
        } else {
            0.0
        }
    }

    /// Calculate trend direction
    fn calculate_trend_direction(&self, values: &[f64]) -> TrendDirection {
        let trend_slope = self.calculate_trend(values);

        if trend_slope > 0.01 {
            TrendDirection::Increasing
        } else if trend_slope < -0.01 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }

    /// Calculate trend strength
    fn calculate_trend_strength(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let trend_slope = self.calculate_trend(values);
        let mean = values.iter().sum::<f64>() / values.len() as f64;

        // Normalize by mean to get relative trend strength
        if mean.abs() > 1e-10 {
            (trend_slope / mean).abs()
        } else {
            0.0
        }
    }

    /// Detect seasonal patterns
    fn detect_seasonal_patterns(&self, values: &[f64]) -> DeviceResult<SeasonalPatterns> {
        // Simple seasonal pattern detection
        let period_length = 4; // Look for patterns every 4 measurements

        if values.len() < period_length * 2 {
            return Ok(SeasonalPatterns::default());
        }

        let mut seasonal_strength = 0.0;
        let mut period_correlation = 0.0;

        // Check correlation between periods
        for offset in 1..=(values.len() / period_length - 1) {
            let start_idx = 0;
            let offset_idx = offset * period_length;
            let compare_length = (values.len() - offset_idx).min(period_length);

            if compare_length > 1 {
                let first_period = &values[start_idx..start_idx + compare_length];
                let offset_period = &values[offset_idx..offset_idx + compare_length];

                let correlation = self.calculate_correlation(first_period, offset_period);
                period_correlation += correlation;
            }
        }

        seasonal_strength = period_correlation / ((values.len() / period_length - 1) as f64);

        Ok(SeasonalPatterns {
            detected: seasonal_strength > 0.5,
            period_length,
            seasonal_strength,
            amplitude: seasonal_strength * 0.1, // Simplified amplitude
        })
    }

    /// Calculate correlation between two arrays
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }

    /// Assess convergence status
    fn assess_convergence_status(&self) -> DeviceResult<ConvergenceStatus> {
        if self.learning_state.loss_history.len() < 10 {
            return Ok(ConvergenceStatus::NotStarted);
        }

        let recent_losses: Vec<f64> = self
            .learning_state
            .loss_history
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();

        let loss_variance = {
            let mean = recent_losses.iter().sum::<f64>() / recent_losses.len() as f64;
            recent_losses
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / recent_losses.len() as f64
        };

        let convergence_threshold = 0.001;

        if loss_variance < convergence_threshold {
            if recent_losses.iter().all(|&loss| loss < 0.05) {
                Ok(ConvergenceStatus::Converged)
            } else {
                Ok(ConvergenceStatus::Stuck)
            }
        } else {
            let trend = self.calculate_trend(&recent_losses);
            if trend < -0.001 {
                Ok(ConvergenceStatus::Improving)
            } else if trend > 0.001 {
                Ok(ConvergenceStatus::Diverging)
            } else {
                Ok(ConvergenceStatus::Oscillating)
            }
        }
    }

    /// Calculate adaptation magnitude
    fn calculate_adaptation_magnitude(
        &self,
        current_performance: &PerformanceMetrics,
    ) -> DeviceResult<f64> {
        if let Some(ref baseline) = self.performance_baseline {
            let magnitude = (current_performance.measurement_success_rate
                - baseline.measurement_success_rate)
                .abs()
                + (current_performance.classical_efficiency - baseline.classical_efficiency).abs()
                + (current_performance.circuit_fidelity - baseline.circuit_fidelity).abs();
            Ok(magnitude / 3.0)
        } else {
            Ok(0.0)
        }
    }
}

/// Learning state tracker
#[derive(Debug, Clone)]
struct LearningState {
    iterations_completed: usize,
    current_learning_rate: f64,
    loss_history: VecDeque<f64>,
    accuracy_history: VecDeque<f64>,
    convergence_status: ConvergenceStatus,
}

impl LearningState {
    fn new() -> Self {
        Self {
            iterations_completed: 0,
            current_learning_rate: 0.001,
            loss_history: VecDeque::with_capacity(1000),
            accuracy_history: VecDeque::with_capacity(1000),
            convergence_status: ConvergenceStatus::NotStarted,
        }
    }

    fn add_loss(&mut self, loss: f64) {
        self.loss_history.push_back(loss);
        if self.loss_history.len() > 1000 {
            self.loss_history.pop_front();
        }
    }

    fn add_accuracy(&mut self, accuracy: f64) {
        self.accuracy_history.push_back(accuracy);
        if self.accuracy_history.len() > 1000 {
            self.accuracy_history.pop_front();
        }
    }

    fn get_loss_history(&self) -> scirs2_core::ndarray::Array1<f64> {
        scirs2_core::ndarray::Array1::from_vec(self.loss_history.iter().cloned().collect())
    }

    fn get_accuracy_history(&self) -> scirs2_core::ndarray::Array1<f64> {
        scirs2_core::ndarray::Array1::from_vec(self.accuracy_history.iter().cloned().collect())
    }
}

/// Drift detector
#[derive(Debug, Clone)]
struct DriftDetector {
    performance_window: VecDeque<PerformanceMetrics>,
    detection_sensitivity: f64,
}

impl DriftDetector {
    fn new() -> Self {
        Self {
            performance_window: VecDeque::with_capacity(50),
            detection_sensitivity: 0.1,
        }
    }

    fn update(&mut self, performance: &PerformanceMetrics) -> DeviceResult<()> {
        self.performance_window.push_back(performance.clone());
        if self.performance_window.len() > 50 {
            self.performance_window.pop_front();
        }
        Ok(())
    }
}

/// Transfer learning engine
#[derive(Debug, Clone)]
struct TransferLearningEngine {
    source_domain_knowledge: Option<DomainKnowledge>,
}

impl TransferLearningEngine {
    fn new() -> Self {
        Self {
            source_domain_knowledge: None,
        }
    }

    fn calculate_transfer_effectiveness(
        &self,
        measurement_history: &[MeasurementEvent],
    ) -> DeviceResult<f64> {
        // Simplified transfer effectiveness calculation
        if measurement_history.len() > 100 {
            Ok(0.8) // High effectiveness with sufficient data
        } else if measurement_history.len() > 50 {
            Ok(0.6) // Medium effectiveness
        } else {
            Ok(0.3) // Low effectiveness with limited data
        }
    }

    fn calculate_domain_similarity(
        &self,
        measurement_history: &[MeasurementEvent],
    ) -> DeviceResult<f64> {
        // Simplified domain similarity calculation
        if measurement_history.is_empty() {
            return Ok(0.0);
        }

        let avg_latency = measurement_history.iter().map(|m| m.latency).sum::<f64>()
            / measurement_history.len() as f64;
        let avg_confidence = measurement_history
            .iter()
            .map(|m| m.confidence)
            .sum::<f64>()
            / measurement_history.len() as f64;

        // Compare with expected ranges (simplified)
        let latency_similarity = if avg_latency > 0.0 && avg_latency < 2000.0 {
            0.9
        } else {
            0.5
        };
        let confidence_similarity = if avg_confidence > 0.8 { 0.9 } else { 0.5 };

        Ok((latency_similarity + confidence_similarity) / 2.0)
    }

    fn analyze_feature_transferability(
        &self,
        measurement_history: &[MeasurementEvent],
    ) -> DeviceResult<scirs2_core::ndarray::Array1<f64>> {
        // Simplified feature transferability analysis
        let n_features = 5; // latency, confidence, timestamp, success_rate, error_rate
        Ok(scirs2_core::ndarray::Array1::from_vec(vec![0.8, 0.7, 0.6, 0.9, 0.8]))
    }
}

/// Domain knowledge representation
#[derive(Debug, Clone)]
struct DomainKnowledge {
    feature_distributions: HashMap<String, (f64, f64)>, // mean, std
    performance_correlations: HashMap<String, f64>,
}

/// Performance baseline
#[derive(Debug, Clone)]
struct PerformanceBaseline {
    measurement_success_rate: f64,
    classical_efficiency: f64,
    circuit_fidelity: f64,
    measurement_error_rate: f64,
    timing_overhead: f64,
    established_at: std::time::SystemTime,
}

/// Seasonal patterns
#[derive(Debug, Clone)]
struct SeasonalPatterns {
    detected: bool,
    period_length: usize,
    seasonal_strength: f64,
    amplitude: f64,
}

impl Default for SeasonalPatterns {
    fn default() -> Self {
        Self {
            detected: false,
            period_length: 0,
            seasonal_strength: 0.0,
            amplitude: 0.0,
        }
    }
}
