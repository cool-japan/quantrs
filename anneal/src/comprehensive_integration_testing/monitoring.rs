//! Performance monitoring and alerting for tests

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Performance monitoring for tests
pub struct TestPerformanceMonitor {
    /// Performance metrics
    pub metrics: TestPerformanceMetrics,
    /// Benchmark comparisons
    pub benchmarks: HashMap<String, BenchmarkComparison>,
    /// Performance trends
    pub trends: PerformanceTrends,
    /// Alert system
    pub alert_system: PerformanceAlertSystem,
}

impl TestPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            metrics: TestPerformanceMetrics::default(),
            benchmarks: HashMap::new(),
            trends: PerformanceTrends::default(),
            alert_system: PerformanceAlertSystem::new(),
        }
    }

    /// Record a test execution time
    pub fn record_execution_time(&mut self, duration: Duration) {
        self.metrics.execution_time_distribution.push(duration);
        self.trends
            .execution_time_trend
            .push((SystemTime::now(), duration));

        // Update average
        if !self.metrics.execution_time_distribution.is_empty() {
            let sum: Duration = self.metrics.execution_time_distribution.iter().sum();
            self.metrics.avg_execution_time =
                sum / self.metrics.execution_time_distribution.len() as u32;
        }

        // Keep only last 1000 entries
        if self.metrics.execution_time_distribution.len() > 1000 {
            self.metrics.execution_time_distribution.drain(0..1);
        }
        if self.trends.execution_time_trend.len() > 1000 {
            self.trends.execution_time_trend.drain(0..1);
        }
    }

    /// Update success rate
    pub fn update_success_rate(&mut self, success: bool) {
        let current_count = self.metrics.execution_time_distribution.len();
        if current_count == 0 {
            self.metrics.success_rate = if success { 1.0 } else { 0.0 };
        } else {
            let current_successes = (self.metrics.success_rate * current_count as f64) as usize;
            let new_successes = if success {
                current_successes + 1
            } else {
                current_successes
            };
            self.metrics.success_rate = new_successes as f64 / (current_count + 1) as f64;
        }

        self.trends
            .success_rate_trend
            .push((SystemTime::now(), self.metrics.success_rate));

        // Keep only last 1000 entries
        if self.trends.success_rate_trend.len() > 1000 {
            self.trends.success_rate_trend.drain(0..1);
        }
    }

    /// Set a benchmark baseline
    pub fn set_benchmark(&mut self, name: String, baseline: PerformanceBaseline) {
        let comparison = BenchmarkComparison {
            baseline: baseline.clone(),
            current: self.metrics.clone(),
            delta: PerformanceDelta {
                execution_time_change: 0.0,
                success_rate_change: 0.0,
                resource_usage_change: 0.0,
                overall_change: 0.0,
            },
            timestamp: SystemTime::now(),
        };
        self.benchmarks.insert(name, comparison);
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> &TestPerformanceMetrics {
        &self.metrics
    }

    /// Get performance trends
    pub fn get_trends(&self) -> &PerformanceTrends {
        &self.trends
    }

    /// Analyze trends
    pub fn analyze_trends(&mut self) {
        // Simple trend analysis based on last N points
        let window_size = 10;

        // Analyze execution time trend
        if self.trends.execution_time_trend.len() >= window_size {
            let recent: Vec<_> = self
                .trends
                .execution_time_trend
                .iter()
                .rev()
                .take(window_size)
                .collect();

            let first_avg = recent
                .iter()
                .rev()
                .take(window_size / 2)
                .map(|(_, d)| d.as_secs_f64())
                .sum::<f64>()
                / (window_size / 2) as f64;

            let second_avg = recent
                .iter()
                .take(window_size / 2)
                .map(|(_, d)| d.as_secs_f64())
                .sum::<f64>()
                / (window_size / 2) as f64;

            let change = (second_avg - first_avg) / first_avg;

            self.trends.trend_analysis.execution_time_direction = if change < -0.1 {
                TrendDirection::Improving
            } else if change > 0.1 {
                TrendDirection::Degrading
            } else {
                TrendDirection::Stable
            };
        }
    }

    /// Check alert conditions
    pub fn check_alerts(&mut self) {
        self.alert_system.check_alerts(&self.metrics);
    }
}

/// Test performance metrics
#[derive(Debug, Clone)]
pub struct TestPerformanceMetrics {
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Execution time distribution
    pub execution_time_distribution: Vec<Duration>,
    /// Success rate
    pub success_rate: f64,
    /// Resource efficiency
    pub resource_efficiency: f64,
    /// Throughput rate
    pub throughput_rate: f64,
}

impl Default for TestPerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_execution_time: Duration::from_secs(0),
            execution_time_distribution: vec![],
            success_rate: 0.0,
            resource_efficiency: 0.0,
            throughput_rate: 0.0,
        }
    }
}

/// Benchmark comparison data
#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    /// Baseline performance
    pub baseline: PerformanceBaseline,
    /// Current performance
    pub current: TestPerformanceMetrics,
    /// Performance delta
    pub delta: PerformanceDelta,
    /// Comparison timestamp
    pub timestamp: SystemTime,
}

/// Performance baseline
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    /// Baseline execution time
    pub execution_time: Duration,
    /// Baseline success rate
    pub success_rate: f64,
    /// Baseline resource usage
    pub resource_usage: f64,
    /// Baseline timestamp
    pub timestamp: SystemTime,
}

/// Performance delta comparison
#[derive(Debug, Clone)]
pub struct PerformanceDelta {
    /// Execution time change
    pub execution_time_change: f64,
    /// Success rate change
    pub success_rate_change: f64,
    /// Resource usage change
    pub resource_usage_change: f64,
    /// Overall performance change
    pub overall_change: f64,
}

/// Performance trends tracking
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    /// Execution time trend
    pub execution_time_trend: Vec<(SystemTime, Duration)>,
    /// Success rate trend
    pub success_rate_trend: Vec<(SystemTime, f64)>,
    /// Resource usage trend
    pub resource_usage_trend: Vec<(SystemTime, f64)>,
    /// Trend analysis
    pub trend_analysis: TrendAnalysis,
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            execution_time_trend: vec![],
            success_rate_trend: vec![],
            resource_usage_trend: vec![],
            trend_analysis: TrendAnalysis::default(),
        }
    }
}

/// Trend analysis results
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Execution time trend direction
    pub execution_time_direction: TrendDirection,
    /// Success rate trend direction
    pub success_rate_direction: TrendDirection,
    /// Resource usage trend direction
    pub resource_usage_direction: TrendDirection,
    /// Trend confidence
    pub confidence: f64,
}

impl Default for TrendAnalysis {
    fn default() -> Self {
        Self {
            execution_time_direction: TrendDirection::Stable,
            success_rate_direction: TrendDirection::Stable,
            resource_usage_direction: TrendDirection::Stable,
            confidence: 0.0,
        }
    }
}

/// Trend directions
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
    Unknown,
}

/// Performance alert system
pub struct PerformanceAlertSystem {
    /// Alert rules
    pub alert_rules: Vec<AlertRule>,
    /// Active alerts
    pub active_alerts: HashMap<String, PerformanceAlert>,
    /// Alert history
    pub alert_history: Vec<PerformanceAlert>,
}

impl PerformanceAlertSystem {
    pub fn new() -> Self {
        Self {
            alert_rules: vec![],
            active_alerts: HashMap::new(),
            alert_history: vec![],
        }
    }

    /// Add an alert rule
    pub fn add_rule(&mut self, rule: AlertRule) {
        self.alert_rules.push(rule);
    }

    /// Remove an alert rule
    pub fn remove_rule(&mut self, rule_name: &str) {
        self.alert_rules.retain(|r| r.name != rule_name);
    }

    /// Check alert conditions against current metrics
    pub fn check_alerts(&mut self, metrics: &TestPerformanceMetrics) {
        // Collect rules that should trigger alerts
        let triggered_rules: Vec<String> = self
            .alert_rules
            .iter()
            .filter_map(|rule| {
                let should_alert = match &rule.condition {
                    AlertCondition::ThresholdExceeded(threshold) => {
                        // Check if execution time exceeds threshold
                        metrics.avg_execution_time.as_secs_f64() > *threshold
                    }
                    AlertCondition::ThresholdBelow(threshold) => {
                        // Check if success rate is below threshold
                        metrics.success_rate < *threshold
                    }
                    AlertCondition::PercentageChange(_percentage) => {
                        // Check for significant performance change
                        false // Placeholder
                    }
                    AlertCondition::Custom(_) => false,
                };

                if should_alert {
                    Some(rule.name.clone())
                } else {
                    None
                }
            })
            .collect();

        // Trigger alerts for collected rules
        for rule_name in triggered_rules {
            self.trigger_alert(&rule_name, metrics);
        }
    }

    /// Trigger an alert
    fn trigger_alert(&mut self, rule_name: &str, metrics: &TestPerformanceMetrics) {
        let rule = self.alert_rules.iter().find(|r| r.name == rule_name);
        if let Some(rule) = rule {
            let alert = PerformanceAlert {
                id: format!(
                    "alert_{}_{}",
                    rule_name,
                    SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                ),
                rule_name: rule_name.to_string(),
                message: format!("Performance alert triggered for rule: {}", rule_name),
                severity: rule.severity.clone(),
                timestamp: SystemTime::now(),
                metric_value: metrics.avg_execution_time.as_secs_f64(),
                threshold_value: 0.0, // Placeholder
                status: AlertStatus::Active,
            };

            self.active_alerts.insert(alert.id.clone(), alert.clone());
            self.alert_history.push(alert);

            // Keep only last 1000 alerts in history
            if self.alert_history.len() > 1000 {
                self.alert_history.drain(0..1);
            }
        }
    }

    /// Acknowledge an alert
    pub fn acknowledge_alert(&mut self, alert_id: &str) -> Result<(), String> {
        let alert = self
            .active_alerts
            .get_mut(alert_id)
            .ok_or_else(|| format!("Alert {} not found", alert_id))?;
        alert.status = AlertStatus::Acknowledged;
        Ok(())
    }

    /// Resolve an alert
    pub fn resolve_alert(&mut self, alert_id: &str) -> Result<(), String> {
        let alert = self
            .active_alerts
            .remove(alert_id)
            .ok_or_else(|| format!("Alert {} not found", alert_id))?;

        // Update in history
        for hist_alert in &mut self.alert_history {
            if hist_alert.id == alert_id {
                hist_alert.status = AlertStatus::Resolved;
            }
        }
        Ok(())
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<&PerformanceAlert> {
        self.active_alerts.values().collect()
    }

    /// Get alert history
    pub fn get_alert_history(&self) -> &[PerformanceAlert] {
        &self.alert_history
    }

    /// Clear all alerts
    pub fn clear_all_alerts(&mut self) {
        self.active_alerts.clear();
        self.alert_history.clear();
    }
}

/// Alert rule definition
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Metric to monitor
    pub metric: String,
    /// Alert condition
    pub condition: AlertCondition,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert actions
    pub actions: Vec<AlertAction>,
}

/// Alert conditions
#[derive(Debug, Clone)]
pub enum AlertCondition {
    /// Threshold exceeded
    ThresholdExceeded(f64),
    /// Threshold below
    ThresholdBelow(f64),
    /// Percentage change
    PercentageChange(f64),
    /// Custom condition
    Custom(String),
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Alert actions
#[derive(Debug, Clone)]
pub enum AlertAction {
    /// Log alert
    Log,
    /// Send email
    Email(String),
    /// Execute script
    ExecuteScript(String),
    /// Custom action
    Custom(String),
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    /// Alert ID
    pub id: String,
    /// Alert rule name
    pub rule_name: String,
    /// Alert message
    pub message: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert timestamp
    pub timestamp: SystemTime,
    /// Metric value
    pub metric_value: f64,
    /// Threshold value
    pub threshold_value: f64,
    /// Alert status
    pub status: AlertStatus,
}

/// Alert status
#[derive(Debug, Clone, PartialEq)]
pub enum AlertStatus {
    Active,
    Resolved,
    Acknowledged,
    Suppressed,
}
