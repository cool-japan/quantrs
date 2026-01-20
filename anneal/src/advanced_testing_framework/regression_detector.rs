//! Performance regression detection system

use super::{
    AlertThresholds, ApplicationError, ApplicationResult, Duration, HashMap, Instant,
    RegressionAlgorithmType, StatisticalModelType, TrendDirection, VecDeque,
};

/// Performance regression detector
#[derive(Debug)]
pub struct RegressionDetector {
    /// Performance history database
    pub performance_history: HashMap<String, VecDeque<PerformanceDataPoint>>,
    /// Regression detection algorithms
    pub detection_algorithms: Vec<RegressionAlgorithm>,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Statistical models for prediction
    pub statistical_models: HashMap<String, StatisticalModel>,
}

/// Performance data point
#[derive(Debug, Clone)]
pub struct PerformanceDataPoint {
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Performance value
    pub value: f64,
    /// Test configuration
    pub test_config: TestConfiguration,
    /// Environmental factors
    pub environment: EnvironmentalFactors,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Test configuration for reproducibility
#[derive(Debug, Clone)]
pub struct TestConfiguration {
    /// Test parameters
    pub parameters: HashMap<String, f64>,
    /// Hardware configuration
    pub hardware: HardwareConfiguration,
    /// Software configuration
    pub software: SoftwareConfiguration,
}

/// Hardware configuration
#[derive(Debug, Clone)]
pub struct HardwareConfiguration {
    /// CPU model
    pub cpu_model: String,
    /// Memory size (GB)
    pub memory_gb: usize,
    /// Number of cores
    pub num_cores: usize,
    /// GPU information
    pub gpu_info: Option<String>,
}

/// Software configuration
#[derive(Debug, Clone)]
pub struct SoftwareConfiguration {
    /// Operating system
    pub os: String,
    /// Compiler version
    pub compiler_version: String,
    /// Optimization flags
    pub optimization_flags: Vec<String>,
    /// Library versions
    pub dependencies: HashMap<String, String>,
}

/// Environmental factors affecting performance
#[derive(Debug, Clone)]
pub struct EnvironmentalFactors {
    /// System load
    pub system_load: f64,
    /// Temperature
    pub temperature: Option<f64>,
    /// Network conditions
    pub network_latency: Option<Duration>,
    /// Power mode
    pub power_mode: Option<String>,
}

/// Regression detection algorithm
#[derive(Debug)]
pub struct RegressionAlgorithm {
    /// Algorithm identifier
    pub id: String,
    /// Algorithm type
    pub algorithm_type: RegressionAlgorithmType,
    /// Algorithm parameters
    pub parameters: HashMap<String, f64>,
    /// Sensitivity level
    pub sensitivity: f64,
}

/// Statistical model for regression analysis
#[derive(Debug)]
pub struct StatisticalModel {
    /// Model type
    pub model_type: StatisticalModelType,
    /// Model parameters
    pub parameters: Vec<f64>,
    /// Model confidence
    pub confidence: f64,
    /// Last update time
    pub last_update: Instant,
}

impl RegressionDetector {
    #[must_use]
    pub fn new() -> Self {
        Self {
            performance_history: HashMap::new(),
            detection_algorithms: Self::create_default_algorithms(),
            alert_thresholds: AlertThresholds::default(),
            statistical_models: HashMap::new(),
        }
    }

    /// Create default regression detection algorithms
    fn create_default_algorithms() -> Vec<RegressionAlgorithm> {
        vec![
            RegressionAlgorithm {
                id: "statistical_process_control".to_string(),
                algorithm_type: RegressionAlgorithmType::StatisticalProcessControl,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("control_limit_factor".to_string(), 3.0);
                    params.insert("window_size".to_string(), 50.0);
                    params
                },
                sensitivity: 0.95,
            },
            RegressionAlgorithm {
                id: "change_point_detection".to_string(),
                algorithm_type: RegressionAlgorithmType::ChangePointDetection,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("penalty".to_string(), 1.0);
                    params.insert("min_segment_length".to_string(), 10.0);
                    params
                },
                sensitivity: 0.90,
            },
            RegressionAlgorithm {
                id: "time_series_analysis".to_string(),
                algorithm_type: RegressionAlgorithmType::TimeSeriesAnalysis,
                parameters: {
                    let mut params = HashMap::new();
                    params.insert("trend_threshold".to_string(), 0.05);
                    params.insert("seasonality_period".to_string(), 7.0);
                    params
                },
                sensitivity: 0.85,
            },
        ]
    }

    /// Add performance data point
    pub fn add_data_point(&mut self, test_id: String, data_point: PerformanceDataPoint) {
        let history = self
            .performance_history
            .entry(test_id)
            .or_insert_with(VecDeque::new);
        history.push_back(data_point);

        // Keep only recent data points
        while history.len() > 1000 {
            history.pop_front();
        }
    }

    /// Detect performance regression
    pub fn detect_regression(
        &self,
        test_id: &str,
    ) -> ApplicationResult<Vec<RegressionDetectionResult>> {
        let history = self.performance_history.get(test_id).ok_or_else(|| {
            ApplicationError::ConfigurationError(format!(
                "No performance history found for test: {test_id}"
            ))
        })?;

        if history.len() < self.alert_thresholds.min_sample_size {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();

        for algorithm in &self.detection_algorithms {
            let result = self.run_detection_algorithm(algorithm, history)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Run specific detection algorithm
    fn run_detection_algorithm(
        &self,
        algorithm: &RegressionAlgorithm,
        history: &VecDeque<PerformanceDataPoint>,
    ) -> ApplicationResult<RegressionDetectionResult> {
        match algorithm.algorithm_type {
            RegressionAlgorithmType::StatisticalProcessControl => {
                self.run_statistical_process_control(algorithm, history)
            }
            RegressionAlgorithmType::ChangePointDetection => {
                self.run_change_point_detection(algorithm, history)
            }
            RegressionAlgorithmType::TimeSeriesAnalysis => {
                self.run_time_series_analysis(algorithm, history)
            }
            _ => Ok(RegressionDetectionResult {
                algorithm_id: algorithm.id.clone(),
                regression_detected: false,
                confidence: 0.0,
                p_value: 1.0,
                change_point: None,
                trend_direction: TrendDirection::Stable,
                magnitude: 0.0,
                details: "Algorithm not implemented".to_string(),
            }),
        }
    }

    /// Run statistical process control algorithm
    fn run_statistical_process_control(
        &self,
        algorithm: &RegressionAlgorithm,
        history: &VecDeque<PerformanceDataPoint>,
    ) -> ApplicationResult<RegressionDetectionResult> {
        let window_size = *algorithm.parameters.get("window_size").unwrap_or(&50.0) as usize;
        let control_limit_factor = algorithm
            .parameters
            .get("control_limit_factor")
            .unwrap_or(&3.0);

        let values: Vec<f64> = history.iter().map(|dp| dp.value).collect();

        if values.len() < window_size {
            return Ok(RegressionDetectionResult {
                algorithm_id: algorithm.id.clone(),
                regression_detected: false,
                confidence: 0.0,
                p_value: 1.0,
                change_point: None,
                trend_direction: TrendDirection::Stable,
                magnitude: 0.0,
                details: "Insufficient data for SPC".to_string(),
            });
        }

        // Calculate control limits from baseline window
        let baseline = &values[..window_size];
        let mean = baseline.iter().sum::<f64>() / baseline.len() as f64;
        let variance =
            baseline.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (baseline.len() - 1) as f64;
        let std_dev = variance.sqrt();

        let upper_limit = mean + control_limit_factor * std_dev;
        let lower_limit = mean - control_limit_factor * std_dev;

        // Check recent values against control limits
        let recent_values = &values[window_size..];
        let violations: Vec<usize> = recent_values
            .iter()
            .enumerate()
            .filter(|(_, &value)| value > upper_limit || value < lower_limit)
            .map(|(i, _)| i + window_size)
            .collect();

        let regression_detected = !violations.is_empty();
        let confidence = if regression_detected {
            algorithm.sensitivity
        } else {
            1.0 - algorithm.sensitivity
        };

        let trend_direction = if recent_values.iter().any(|&v| v < lower_limit) {
            TrendDirection::Degrading
        } else if recent_values.iter().any(|&v| v > upper_limit) {
            TrendDirection::Improving
        } else {
            TrendDirection::Stable
        };

        Ok(RegressionDetectionResult {
            algorithm_id: algorithm.id.clone(),
            regression_detected,
            confidence,
            p_value: if regression_detected { 0.01 } else { 0.9 },
            change_point: violations.first().copied(),
            trend_direction,
            magnitude: if violations.is_empty() {
                0.0
            } else {
                let worst_violation = recent_values
                    .iter()
                    .map(|&v| (v - mean).abs() / std_dev)
                    .fold(0.0, f64::max);
                worst_violation
            },
            details: format!("SPC analysis: {} violations detected", violations.len()),
        })
    }

    /// Run change point detection algorithm
    fn run_change_point_detection(
        &self,
        algorithm: &RegressionAlgorithm,
        history: &VecDeque<PerformanceDataPoint>,
    ) -> ApplicationResult<RegressionDetectionResult> {
        let min_segment_length = *algorithm
            .parameters
            .get("min_segment_length")
            .unwrap_or(&10.0) as usize;
        let values: Vec<f64> = history.iter().map(|dp| dp.value).collect();

        if values.len() < min_segment_length * 2 {
            return Ok(RegressionDetectionResult {
                algorithm_id: algorithm.id.clone(),
                regression_detected: false,
                confidence: 0.0,
                p_value: 1.0,
                change_point: None,
                trend_direction: TrendDirection::Stable,
                magnitude: 0.0,
                details: "Insufficient data for change point detection".to_string(),
            });
        }

        // Simplified change point detection using variance changes
        let mut best_change_point = None;
        let mut best_score = 0.0;

        for i in min_segment_length..(values.len() - min_segment_length) {
            let before = &values[..i];
            let after = &values[i..];

            let mean_before = before.iter().sum::<f64>() / before.len() as f64;
            let mean_after = after.iter().sum::<f64>() / after.len() as f64;

            let score = (mean_before - mean_after).abs();

            if score > best_score {
                best_score = score;
                best_change_point = Some(i);
            }
        }

        let threshold = 0.1; // Simplified threshold
        let regression_detected = best_score > threshold;

        Ok(RegressionDetectionResult {
            algorithm_id: algorithm.id.clone(),
            regression_detected,
            confidence: if regression_detected {
                algorithm.sensitivity
            } else {
                1.0 - algorithm.sensitivity
            },
            p_value: if regression_detected { 0.05 } else { 0.8 },
            change_point: best_change_point,
            trend_direction: if regression_detected {
                if let Some(cp) = best_change_point {
                    let before_mean = values[..cp].iter().sum::<f64>() / cp as f64;
                    let after_mean = values[cp..].iter().sum::<f64>() / (values.len() - cp) as f64;
                    if after_mean < before_mean {
                        TrendDirection::Degrading
                    } else {
                        TrendDirection::Improving
                    }
                } else {
                    TrendDirection::Stable
                }
            } else {
                TrendDirection::Stable
            },
            magnitude: best_score,
            details: format!("Change point detection: score = {best_score:.4}"),
        })
    }

    /// Run time series analysis algorithm
    fn run_time_series_analysis(
        &self,
        algorithm: &RegressionAlgorithm,
        history: &VecDeque<PerformanceDataPoint>,
    ) -> ApplicationResult<RegressionDetectionResult> {
        let trend_threshold = algorithm.parameters.get("trend_threshold").unwrap_or(&0.05);
        let values: Vec<f64> = history.iter().map(|dp| dp.value).collect();

        if values.len() < 10 {
            return Ok(RegressionDetectionResult {
                algorithm_id: algorithm.id.clone(),
                regression_detected: false,
                confidence: 0.0,
                p_value: 1.0,
                change_point: None,
                trend_direction: TrendDirection::Stable,
                magnitude: 0.0,
                details: "Insufficient data for time series analysis".to_string(),
            });
        }

        // Simple linear trend calculation
        let n = values.len() as f64;
        let x_sum = (0..values.len()).map(|i| i as f64).sum::<f64>();
        let y_sum = values.iter().sum::<f64>();
        let xy_sum = values
            .iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum::<f64>();
        let x2_sum = (0..values.len()).map(|i| (i as f64).powi(2)).sum::<f64>();

        let slope = n.mul_add(xy_sum, -(x_sum * y_sum)) / x_sum.mul_add(-x_sum, n * x2_sum);
        let slope_abs = slope.abs();

        let regression_detected = slope_abs > *trend_threshold;

        let trend_direction = if slope > *trend_threshold {
            TrendDirection::Improving
        } else if slope < -*trend_threshold {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        };

        Ok(RegressionDetectionResult {
            algorithm_id: algorithm.id.clone(),
            regression_detected,
            confidence: if regression_detected {
                algorithm.sensitivity
            } else {
                1.0 - algorithm.sensitivity
            },
            p_value: if regression_detected { 0.02 } else { 0.7 },
            change_point: None,
            trend_direction,
            magnitude: slope_abs,
            details: format!("Time series analysis: slope = {slope:.6}"),
        })
    }

    /// Get performance summary for test
    #[must_use]
    pub fn get_performance_summary(&self, test_id: &str) -> Option<PerformanceSummary> {
        let history = self.performance_history.get(test_id)?;

        if history.is_empty() {
            return None;
        }

        let values: Vec<f64> = history.iter().map(|dp| dp.value).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        let std_dev = variance.sqrt();

        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Some(PerformanceSummary {
            test_id: test_id.to_string(),
            sample_count: values.len(),
            mean,
            std_dev,
            min: sorted_values[0],
            max: sorted_values[sorted_values.len() - 1],
            median: if sorted_values.len() % 2 == 0 {
                f64::midpoint(
                    sorted_values[sorted_values.len() / 2 - 1],
                    sorted_values[sorted_values.len() / 2],
                )
            } else {
                sorted_values[sorted_values.len() / 2]
            },
            recent_trend: self.calculate_recent_trend(&values),
        })
    }

    /// Calculate recent trend
    fn calculate_recent_trend(&self, values: &[f64]) -> TrendDirection {
        if values.len() < 10 {
            return TrendDirection::Stable;
        }

        let recent_size = (values.len() / 4).max(5).min(20);
        let recent = &values[values.len() - recent_size..];
        let earlier = &values[values.len() - 2 * recent_size..values.len() - recent_size];

        let recent_mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let earlier_mean = earlier.iter().sum::<f64>() / earlier.len() as f64;

        let change = (recent_mean - earlier_mean) / earlier_mean;

        if change > 0.05 {
            TrendDirection::Improving
        } else if change < -0.05 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        }
    }
}

/// Result from regression detection
#[derive(Debug, Clone)]
pub struct RegressionDetectionResult {
    /// Algorithm identifier
    pub algorithm_id: String,
    /// Whether regression was detected
    pub regression_detected: bool,
    /// Confidence level
    pub confidence: f64,
    /// Statistical p-value
    pub p_value: f64,
    /// Change point index (if detected)
    pub change_point: Option<usize>,
    /// Direction of trend
    pub trend_direction: TrendDirection,
    /// Magnitude of change
    pub magnitude: f64,
    /// Additional details
    pub details: String,
}

/// Performance summary statistics
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Test identifier
    pub test_id: String,
    /// Number of samples
    pub sample_count: usize,
    /// Mean performance
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Median value
    pub median: f64,
    /// Recent trend direction
    pub recent_trend: TrendDirection,
}
