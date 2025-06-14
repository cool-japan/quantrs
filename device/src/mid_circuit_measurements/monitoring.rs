//! Performance monitoring and optimization caching components

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};
use crate::DeviceResult;
use super::results::*;

/// Performance summary for aggregated metrics
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub total_executions: usize,
    pub average_execution_time: Duration,
    pub success_rate: f64,
    pub cache_hit_rate: f64,
    pub optimization_savings: f64,
    pub resource_utilization: f64,
    pub interval_summaries: Vec<IntervalSummary>,
    pub overall_stats: OverallStatistics,
    pub active_alerts: Vec<PerformanceAlert>,
    pub performance_trends: PerformanceTrend,
    pub monitoring_window: Duration,
    pub last_updated: SystemTime,
}

impl Default for PerformanceSummary {
    fn default() -> Self {
        Self {
            total_executions: 0,
            average_execution_time: Duration::from_secs(0),
            success_rate: 0.0,
            cache_hit_rate: 0.0,
            optimization_savings: 0.0,
            resource_utilization: 0.0,
            interval_summaries: vec![],
            overall_stats: OverallStatistics::default(),
            active_alerts: vec![],
            performance_trends: PerformanceTrend::default(),
            monitoring_window: Duration::from_secs(300),
            last_updated: SystemTime::now(),
        }
    }
}

/// Performance alert for threshold breaches
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    pub alert_type: String,
    pub message: String,
    pub severity: String,
    pub timestamp: SystemTime,
    pub metric_value: f64,
    pub threshold_value: f64,
}

/// Interval summary for time-based aggregations
#[derive(Debug, Clone, Default)]
pub struct IntervalSummary {
    pub interval: Duration,
    pub total_executions: usize,
    pub average_execution_time: Duration,
    pub success_rate: f64,
    pub throughput: f64,
}

/// Overall statistics across all intervals
#[derive(Debug, Clone, Default)]
pub struct OverallStatistics {
    pub total_executions: usize,
    pub overall_success_rate: f64,
    pub peak_throughput: f64,
    pub average_latency: Duration,
    pub resource_utilization: f64,
}

/// Performance trend analysis
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    pub trend_direction: String,
    pub trend_strength: f64,
    pub prediction_confidence: f64,
    pub time_window: Duration,
}

impl Default for PerformanceTrend {
    fn default() -> Self {
        Self {
            trend_direction: "Stable".to_string(),
            trend_strength: 0.0,
            prediction_confidence: 0.5,
            time_window: Duration::from_secs(300),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    pub total_entries: usize,
    pub total_access_count: usize,
    pub hit_rate: f64,
    pub cache_hit_rate: f64,
    pub miss_rate: f64,
    pub eviction_rate: f64,
    pub memory_usage: f64,
    pub avg_age: Duration,
}

/// Performance monitor for tracking measurement execution metrics
pub struct PerformanceMonitor {
    monitoring_enabled: bool,
    metrics_history: VecDeque<TimestampedMetrics>,
    alert_thresholds: AlertThresholds,
    monitoring_window: Duration,
    aggregation_intervals: Vec<Duration>,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            monitoring_enabled: true,
            metrics_history: VecDeque::with_capacity(10000),
            alert_thresholds: AlertThresholds::default(),
            monitoring_window: Duration::from_secs(3600), // 1 hour
            aggregation_intervals: vec![
                Duration::from_secs(60),    // 1 minute
                Duration::from_secs(300),   // 5 minutes
                Duration::from_secs(900),   // 15 minutes
                Duration::from_secs(3600),  // 1 hour
            ],
        }
    }

    /// Create performance monitor with custom configuration
    pub fn with_config(
        alert_thresholds: AlertThresholds,
        monitoring_window: Duration,
    ) -> Self {
        Self {
            monitoring_enabled: true,
            metrics_history: VecDeque::with_capacity(10000),
            alert_thresholds,
            monitoring_window,
            aggregation_intervals: vec![
                Duration::from_secs(60),
                Duration::from_secs(300),
                Duration::from_secs(900),
                Duration::from_secs(3600),
            ],
        }
    }

    /// Record performance metrics
    pub fn record_metrics(&mut self, metrics: &PerformanceMetrics) -> DeviceResult<()> {
        if !self.monitoring_enabled {
            return Ok(());
        }

        let timestamped_metrics = TimestampedMetrics {
            timestamp: SystemTime::now(),
            metrics: metrics.clone(),
        };

        self.metrics_history.push_back(timestamped_metrics);

        // Clean old metrics outside monitoring window
        self.cleanup_old_metrics()?;

        Ok(())
    }

    /// Get current performance summary
    pub fn get_performance_summary(&self) -> DeviceResult<PerformanceSummary> {
        if self.metrics_history.is_empty() {
            return Ok(PerformanceSummary::default());
        }

        // Calculate aggregated metrics for different time intervals
        let mut interval_summaries = HashMap::new();
        for &interval in &self.aggregation_intervals {
            let summary = self.calculate_interval_summary(interval)?;
            interval_summaries.insert(interval, summary);
        }

        // Calculate overall statistics
        let overall_stats = self.calculate_overall_statistics()?;

        // Detect performance alerts
        let active_alerts = self.detect_performance_alerts()?;

        // Calculate performance trends
        let performance_trends = self.calculate_performance_trends()?;

        Ok(PerformanceSummary {
            total_executions: overall_stats.total_executions,
            average_execution_time: overall_stats.average_latency,
            success_rate: overall_stats.overall_success_rate,
            cache_hit_rate: 0.8, // Placeholder
            optimization_savings: 0.1, // Placeholder
            resource_utilization: 0.7, // Placeholder
            interval_summaries: interval_summaries.into_values().collect(),
            overall_stats,
            active_alerts,
            performance_trends: performance_trends.into_iter().next().unwrap_or_default(),
            monitoring_window: self.monitoring_window,
            last_updated: SystemTime::now(),
        })
    }

    /// Check for performance alerts
    pub fn check_alerts(&self) -> DeviceResult<Vec<PerformanceAlert>> {
        self.detect_performance_alerts()
    }

    /// Get metrics for specific time range
    pub fn get_metrics_in_range(
        &self,
        start_time: SystemTime,
        end_time: SystemTime,
    ) -> DeviceResult<Vec<TimestampedMetrics>> {
        let metrics = self.metrics_history
            .iter()
            .filter(|m| m.timestamp >= start_time && m.timestamp <= end_time)
            .cloned()
            .collect();

        Ok(metrics)
    }

    /// Export metrics for analysis
    pub fn export_metrics(&self, format: ExportFormat) -> DeviceResult<String> {
        match format {
            ExportFormat::Json => self.export_as_json(),
            ExportFormat::Csv => self.export_as_csv(),
            ExportFormat::Metrics => self.export_as_prometheus_metrics(),
        }
    }

    /// Cleanup old metrics outside monitoring window
    fn cleanup_old_metrics(&mut self) -> DeviceResult<()> {
        let cutoff_time = SystemTime::now()
            .checked_sub(self.monitoring_window)
            .unwrap_or(SystemTime::UNIX_EPOCH);

        while let Some(front_metric) = self.metrics_history.front() {
            if front_metric.timestamp < cutoff_time {
                self.metrics_history.pop_front();
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Calculate summary for specific time interval
    fn calculate_interval_summary(&self, interval: Duration) -> DeviceResult<IntervalSummary> {
        let cutoff_time = SystemTime::now()
            .checked_sub(interval)
            .unwrap_or(SystemTime::UNIX_EPOCH);

        let recent_metrics: Vec<&PerformanceMetrics> = self.metrics_history
            .iter()
            .filter(|m| m.timestamp >= cutoff_time)
            .map(|m| &m.metrics)
            .collect();

        if recent_metrics.is_empty() {
            return Ok(IntervalSummary::default());
        }

        // Calculate aggregated statistics
        let count = recent_metrics.len();
        let avg_success_rate = recent_metrics.iter()
            .map(|m| m.measurement_success_rate)
            .sum::<f64>() / count as f64;
        
        let avg_efficiency = recent_metrics.iter()
            .map(|m| m.classical_efficiency)
            .sum::<f64>() / count as f64;
        
        let avg_fidelity = recent_metrics.iter()
            .map(|m| m.circuit_fidelity)
            .sum::<f64>() / count as f64;
        
        let avg_error_rate = recent_metrics.iter()
            .map(|m| m.measurement_error_rate)
            .sum::<f64>() / count as f64;
        
        let avg_timing_overhead = recent_metrics.iter()
            .map(|m| m.timing_overhead)
            .sum::<f64>() / count as f64;

        // Calculate min/max
        let min_success_rate = recent_metrics.iter()
            .map(|m| m.measurement_success_rate)
            .fold(f64::INFINITY, f64::min);
        
        let max_success_rate = recent_metrics.iter()
            .map(|m| m.measurement_success_rate)
            .fold(f64::NEG_INFINITY, f64::max);

        Ok(IntervalSummary {
            interval,
            total_executions: count,
            average_execution_time: Duration::from_millis((avg_efficiency * 1000.0) as u64),
            success_rate: avg_success_rate,
            throughput: count as f64 / interval.as_secs_f64(),
        })
    }

    /// Calculate overall statistics
    fn calculate_overall_statistics(&self) -> DeviceResult<OverallStatistics> {
        if self.metrics_history.is_empty() {
            return Ok(OverallStatistics::default());
        }

        let all_metrics: Vec<&PerformanceMetrics> = self.metrics_history
            .iter()
            .map(|m| &m.metrics)
            .collect();

        let total_samples = all_metrics.len();
        let uptime = self.calculate_uptime()?;
        let availability = self.calculate_availability(&all_metrics);
        let throughput = self.calculate_throughput();
        let reliability_score = self.calculate_reliability_score(&all_metrics);

        Ok(OverallStatistics {
            total_executions: total_samples,
            overall_success_rate: availability,
            peak_throughput: throughput,
            average_latency: uptime / total_samples.max(1) as u32,
            resource_utilization: reliability_score,
        })
    }

    /// Detect performance alerts
    fn detect_performance_alerts(&self) -> DeviceResult<Vec<PerformanceAlert>> {
        let mut alerts = Vec::new();

        if self.metrics_history.is_empty() {
            return Ok(alerts);
        }

        let latest_metrics = &self.metrics_history.back().unwrap().metrics;

        // Check success rate alert
        if latest_metrics.measurement_success_rate < self.alert_thresholds.min_success_rate {
            alerts.push(PerformanceAlert {
                alert_type: "LowSuccessRate".to_string(),
                message: format!(
                    "Measurement success rate ({:.2}%) below threshold ({:.2}%)",
                    latest_metrics.measurement_success_rate * 100.0,
                    self.alert_thresholds.min_success_rate * 100.0
                ),
                severity: if latest_metrics.measurement_success_rate < 0.8 {
                    "Critical".to_string()
                } else {
                    "Warning".to_string()
                },
                timestamp: SystemTime::now(),
                metric_value: latest_metrics.measurement_success_rate,
                threshold_value: self.alert_thresholds.min_success_rate,
            });
        }

        // Check error rate alert
        if latest_metrics.measurement_error_rate > self.alert_thresholds.max_error_rate {
            alerts.push(PerformanceAlert {
                alert_type: "HighErrorRate".to_string(),
                message: format!(
                    "Error rate ({:.2}%) above threshold ({:.2}%)",
                    latest_metrics.measurement_error_rate * 100.0,
                    self.alert_thresholds.max_error_rate * 100.0
                ),
                severity: if latest_metrics.measurement_error_rate > 0.1 {
                    "Critical".to_string()
                } else {
                    "Warning".to_string()
                },
                timestamp: SystemTime::now(),
                metric_value: latest_metrics.measurement_error_rate,
                threshold_value: self.alert_thresholds.max_error_rate,
            });
        }

        // Check timing overhead alert
        if latest_metrics.timing_overhead > self.alert_thresholds.max_timing_overhead {
            alerts.push(PerformanceAlert {
                alert_type: "HighLatency".to_string(),
                message: format!(
                    "Timing overhead ({:.2}) above threshold ({:.2})",
                    latest_metrics.timing_overhead,
                    self.alert_thresholds.max_timing_overhead
                ),
                severity: "Warning".to_string(),
                timestamp: SystemTime::now(),
                metric_value: latest_metrics.timing_overhead,
                threshold_value: self.alert_thresholds.max_timing_overhead,
            });
        }

        // Check efficiency alert
        if latest_metrics.classical_efficiency < self.alert_thresholds.min_efficiency {
            alerts.push(PerformanceAlert {
                alert_type: "LowEfficiency".to_string(),
                message: format!(
                    "Classical efficiency ({:.2}%) below threshold ({:.2}%)",
                    latest_metrics.classical_efficiency * 100.0,
                    self.alert_thresholds.min_efficiency * 100.0
                ),
                severity: "Info".to_string(),
                timestamp: SystemTime::now(),
                metric_value: latest_metrics.classical_efficiency,
                threshold_value: self.alert_thresholds.min_efficiency,
            });
        }

        Ok(alerts)
    }

    /// Calculate performance trends
    fn calculate_performance_trends(&self) -> DeviceResult<Vec<PerformanceTrend>> {
        let mut trends = Vec::new();

        if self.metrics_history.len() < 5 {
            return Ok(trends);
        }

        // Success rate trend
        let success_rates: Vec<f64> = self.metrics_history
            .iter()
            .map(|m| m.metrics.measurement_success_rate)
            .collect();
        
        let success_rate_trend = self.calculate_trend(&success_rates);
        trends.push(PerformanceTrend {
            trend_direction: format!("{:?}", self.classify_trend_direction(success_rate_trend)),
            trend_strength: success_rate_trend.abs(),
            prediction_confidence: 0.8,
            time_window: self.monitoring_window,
        });

        // Error rate trend
        let error_rates: Vec<f64> = self.metrics_history
            .iter()
            .map(|m| m.metrics.measurement_error_rate)
            .collect();
        
        let error_rate_trend = self.calculate_trend(&error_rates);
        trends.push(PerformanceTrend {
            trend_direction: format!("{:?}", self.classify_trend_direction(error_rate_trend)),
            trend_strength: error_rate_trend.abs(),
            prediction_confidence: 0.8,
            time_window: self.monitoring_window,
        });

        Ok(trends)
    }

    /// Export metrics as JSON
    fn export_as_json(&self) -> DeviceResult<String> {
        // Simplified JSON export
        let mut json_data = String::from("{\n");
        json_data.push_str(&format!("  \"total_samples\": {},\n", self.metrics_history.len()));
        json_data.push_str(&format!("  \"monitoring_window_secs\": {},\n", self.monitoring_window.as_secs()));
        json_data.push_str("  \"metrics\": [\n");
        
        for (i, metric) in self.metrics_history.iter().enumerate() {
            json_data.push_str(&format!(
                "    {{\"timestamp\": {}, \"success_rate\": {:.4}, \"error_rate\": {:.4}}}",
                metric.timestamp.duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or(Duration::ZERO).as_secs(),
                metric.metrics.measurement_success_rate,
                metric.metrics.measurement_error_rate
            ));
            if i < self.metrics_history.len() - 1 {
                json_data.push(',');
            }
            json_data.push('\n');
        }
        
        json_data.push_str("  ]\n}");
        Ok(json_data)
    }

    /// Export metrics as CSV
    fn export_as_csv(&self) -> DeviceResult<String> {
        let mut csv_data = String::from("timestamp,success_rate,error_rate,efficiency,fidelity,timing_overhead\n");
        
        for metric in &self.metrics_history {
            csv_data.push_str(&format!(
                "{},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                metric.timestamp.duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or(Duration::ZERO).as_secs(),
                metric.metrics.measurement_success_rate,
                metric.metrics.measurement_error_rate,
                metric.metrics.classical_efficiency,
                metric.metrics.circuit_fidelity,
                metric.metrics.timing_overhead
            ));
        }
        
        Ok(csv_data)
    }

    /// Export metrics in Prometheus format
    fn export_as_prometheus_metrics(&self) -> DeviceResult<String> {
        if self.metrics_history.is_empty() {
            return Ok(String::new());
        }

        let latest = &self.metrics_history.back().unwrap().metrics;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis();

        let mut prometheus_data = String::new();
        
        prometheus_data.push_str(&format!(
            "measurement_success_rate {} {}\n",
            latest.measurement_success_rate,
            timestamp
        ));
        
        prometheus_data.push_str(&format!(
            "measurement_error_rate {} {}\n",
            latest.measurement_error_rate,
            timestamp
        ));
        
        prometheus_data.push_str(&format!(
            "classical_efficiency {} {}\n",
            latest.classical_efficiency,
            timestamp
        ));
        
        prometheus_data.push_str(&format!(
            "circuit_fidelity {} {}\n",
            latest.circuit_fidelity,
            timestamp
        ));
        
        prometheus_data.push_str(&format!(
            "timing_overhead {} {}\n",
            latest.timing_overhead,
            timestamp
        ));

        Ok(prometheus_data)
    }

    // Helper methods
    fn calculate_standard_deviation(&self, values: &[f64]) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        
        variance.sqrt()
    }

    fn calculate_uptime(&self) -> DeviceResult<Duration> {
        if self.metrics_history.is_empty() {
            return Ok(Duration::ZERO);
        }

        let first_timestamp = self.metrics_history.front().unwrap().timestamp;
        let last_timestamp = self.metrics_history.back().unwrap().timestamp;
        
        Ok(last_timestamp.duration_since(first_timestamp).unwrap_or(Duration::ZERO))
    }

    fn calculate_availability(&self, metrics: &[&PerformanceMetrics]) -> f64 {
        if metrics.is_empty() {
            return 0.0;
        }

        let successful_measurements = metrics.iter()
            .filter(|m| m.measurement_success_rate > 0.9)
            .count();
        
        successful_measurements as f64 / metrics.len() as f64
    }

    fn calculate_throughput(&self) -> f64 {
        if self.metrics_history.len() < 2 {
            return 0.0;
        }

        let time_span = self.calculate_uptime().unwrap_or(Duration::ZERO);
        if time_span.as_secs() == 0 {
            return 0.0;
        }

        self.metrics_history.len() as f64 / time_span.as_secs() as f64
    }

    fn calculate_reliability_score(&self, metrics: &[&PerformanceMetrics]) -> f64 {
        if metrics.is_empty() {
            return 0.0;
        }

        let avg_success_rate = metrics.iter()
            .map(|m| m.measurement_success_rate)
            .sum::<f64>() / metrics.len() as f64;
        
        let avg_fidelity = metrics.iter()
            .map(|m| m.circuit_fidelity)
            .sum::<f64>() / metrics.len() as f64;

        (avg_success_rate + avg_fidelity) / 2.0
    }

    fn calculate_data_quality_score(&self, metrics: &[&PerformanceMetrics]) -> f64 {
        if metrics.is_empty() {
            return 0.0;
        }

        // Simple data quality based on completeness and consistency
        let complete_measurements = metrics.iter()
            .filter(|m| m.measurement_success_rate > 0.0 && m.circuit_fidelity > 0.0)
            .count();
        
        complete_measurements as f64 / metrics.len() as f64
    }

    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let x_sum = (0..values.len()).map(|i| i as f64).sum::<f64>();
        let y_sum = values.iter().sum::<f64>();
        let xy_sum = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum::<f64>();
        let x2_sum = (0..values.len()).map(|i| (i as f64).powi(2)).sum::<f64>();

        let denominator = n * x2_sum - x_sum * x_sum;
        if denominator.abs() > 1e-10 {
            (n * xy_sum - x_sum * y_sum) / denominator
        } else {
            0.0
        }
    }

    fn classify_trend_direction(&self, trend_slope: f64) -> crate::mid_circuit_measurements::results::TrendDirection {
        if trend_slope > 0.01 {
            crate::mid_circuit_measurements::results::TrendDirection::Increasing
        } else if trend_slope < -0.01 {
            crate::mid_circuit_measurements::results::TrendDirection::Decreasing
        } else {
            crate::mid_circuit_measurements::results::TrendDirection::Stable
        }
    }
}

/// Optimization cache for storing and retrieving optimization results
pub struct OptimizationCache {
    cache_enabled: bool,
    cache: HashMap<String, CachedOptimization>,
    max_cache_size: usize,
    cache_ttl: Duration,
}

impl OptimizationCache {
    /// Create new optimization cache
    pub fn new() -> Self {
        Self {
            cache_enabled: true,
            cache: HashMap::new(),
            max_cache_size: 1000,
            cache_ttl: Duration::from_secs(3600), // 1 hour
        }
    }

    /// Store optimization result in cache
    pub fn store_optimization(
        &mut self,
        key: String,
        optimization: OptimizationResult,
    ) -> DeviceResult<()> {
        if !self.cache_enabled {
            return Ok(());
        }

        let cached_optimization = CachedOptimization {
            result: optimization,
            cached_at: SystemTime::now(),
            access_count: 0,
        };

        self.cache.insert(key, cached_optimization);

        // Cleanup old entries if cache is full
        if self.cache.len() > self.max_cache_size {
            self.cleanup_cache()?;
        }

        Ok(())
    }

    /// Retrieve optimization result from cache
    pub fn get_optimization(&mut self, key: &str) -> Option<OptimizationResult> {
        if !self.cache_enabled {
            return None;
        }

        // Check if entry exists and is valid
        let is_valid = if let Some(cached) = self.cache.get(key) {
            SystemTime::now().duration_since(cached.cached_at).unwrap_or(Duration::MAX) < self.cache_ttl
        } else {
            false
        };

        if is_valid {
            if let Some(cached) = self.cache.get_mut(key) {
                cached.access_count += 1;
                return Some(cached.result.clone());
            }
        } else {
            // Remove expired entry
            self.cache.remove(key);
        }

        None
    }

    /// Check if optimization is cached
    pub fn contains_key(&self, key: &str) -> bool {
        if !self.cache_enabled {
            return false;
        }

        if let Some(cached) = self.cache.get(key) {
            SystemTime::now().duration_since(cached.cached_at).unwrap_or(Duration::MAX) < self.cache_ttl
        } else {
            false
        }
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStatistics {
        let total_entries = self.cache.len();
        let total_access_count = self.cache.values().map(|c| c.access_count as usize).sum();
        
        let avg_age = if total_entries > 0 {
            let total_age: Duration = self.cache.values()
                .map(|c| SystemTime::now().duration_since(c.cached_at).unwrap_or(Duration::ZERO))
                .sum();
            total_age / total_entries as u32
        } else {
            Duration::ZERO
        };

        CacheStatistics {
            total_entries,
            total_access_count,
            hit_rate: 0.8, // Placeholder - would need hit/miss tracking
            cache_hit_rate: 0.8, // Placeholder - would need hit/miss tracking
            miss_rate: 0.2,
            eviction_rate: 0.1,
            memory_usage: (total_entries * 100) as f64, // Estimated bytes
            avg_age,
        }
    }

    /// Cleanup old cache entries
    fn cleanup_cache(&mut self) -> DeviceResult<()> {
        let now = SystemTime::now();
        
        // Remove expired entries
        self.cache.retain(|_, cached| {
            now.duration_since(cached.cached_at).unwrap_or(Duration::MAX) < self.cache_ttl
        });

        // If still over limit, remove least recently used entries
        if self.cache.len() > self.max_cache_size {
            let mut entries: Vec<(String, u32)> = self.cache.iter()
                .map(|(k, v)| (k.clone(), v.access_count))
                .collect();
            
            entries.sort_by_key(|(_, count)| *count);
            
            let to_remove = self.cache.len() - self.max_cache_size;
            for (key, _) in entries.iter().take(to_remove) {
                self.cache.remove(key);
            }
        }

        Ok(())
    }
}

// Supporting types and default implementations

#[derive(Debug, Clone)]
pub struct TimestampedMetrics {
    pub timestamp: SystemTime,
    pub metrics: PerformanceMetrics,
}

#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub min_success_rate: f64,
    pub max_error_rate: f64,
    pub max_timing_overhead: f64,
    pub min_efficiency: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            min_success_rate: 0.9,
            max_error_rate: 0.05,
            max_timing_overhead: 2.0,
            min_efficiency: 0.8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CachedOptimization {
    pub result: OptimizationResult,
    pub cached_at: SystemTime,
    pub access_count: u32,
}

#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Metrics,
}

#[derive(Debug, Clone)]
pub enum AlertType {
    LowSuccessRate,
    HighErrorRate,
    HighLatency,
    LowEfficiency,
}

#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for OptimizationCache {
    fn default() -> Self {
        Self::new()
    }
}

