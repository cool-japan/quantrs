//! Analytics engine and analysis components

pub mod anomaly;
pub mod causal;
pub mod correlation;
pub mod distribution;
pub mod statistical;
pub mod time_series;

use super::config::AdvancedAnalyticsConfig;
use super::results::*;
use crate::DeviceResult;

/// Advanced analytics engine for mid-circuit measurements
pub struct AdvancedAnalyticsEngine {
    config: AdvancedAnalyticsConfig,
    statistical_analyzer: statistical::StatisticalAnalyzer,
    correlation_analyzer: correlation::CorrelationAnalyzer,
    time_series_analyzer: Option<time_series::TimeSeriesAnalyzer>,
    anomaly_detector: Option<anomaly::AnomalyDetector>,
    distribution_analyzer: distribution::DistributionAnalyzer,
    causal_analyzer: Option<causal::CausalAnalyzer>,
}

impl AdvancedAnalyticsEngine {
    /// Create new analytics engine
    pub fn new(config: &AdvancedAnalyticsConfig) -> Self {
        Self {
            config: config.clone(),
            statistical_analyzer: statistical::StatisticalAnalyzer::new(),
            correlation_analyzer: correlation::CorrelationAnalyzer::new(),
            time_series_analyzer: if config.enable_time_series {
                Some(time_series::TimeSeriesAnalyzer::new())
            } else {
                None
            },
            anomaly_detector: if config.enable_anomaly_detection {
                Some(anomaly::AnomalyDetector::new())
            } else {
                None
            },
            distribution_analyzer: distribution::DistributionAnalyzer::new(),
            causal_analyzer: if config.enable_causal_inference {
                Some(causal::CausalAnalyzer::new())
            } else {
                None
            },
        }
    }

    /// Perform comprehensive analytics
    pub async fn analyze(
        &self,
        measurement_history: &[MeasurementEvent],
        execution_stats: &ExecutionStats,
    ) -> DeviceResult<AdvancedAnalyticsResults> {
        // Extract measurement data for analysis
        let latencies: Vec<f64> = measurement_history.iter().map(|e| e.latency).collect();
        let confidences: Vec<f64> = measurement_history.iter().map(|e| e.confidence).collect();
        let timestamps: Vec<f64> = measurement_history.iter().map(|e| e.timestamp).collect();

        // Statistical analysis
        let statistical_analysis = self
            .statistical_analyzer
            .analyze(&latencies, &confidences)?;

        // Correlation analysis
        let correlation_analysis =
            self.correlation_analyzer
                .analyze(&latencies, &confidences, &timestamps)?;

        // Time series analysis (if enabled)
        let time_series_analysis = if let Some(ref analyzer) = self.time_series_analyzer {
            Some(analyzer.analyze(&latencies, &timestamps)?)
        } else {
            None
        };

        // Anomaly detection (if enabled)
        let anomaly_detection = if let Some(ref detector) = self.anomaly_detector {
            Some(detector.detect(&latencies, &confidences)?)
        } else {
            None
        };

        // Distribution analysis
        let distribution_analysis = self.distribution_analyzer.analyze(&latencies)?;

        // Causal analysis (if enabled)
        let causal_analysis = if let Some(ref analyzer) = self.causal_analyzer {
            Some(analyzer.analyze(&latencies, &confidences, &timestamps)?)
        } else {
            None
        };

        Ok(AdvancedAnalyticsResults {
            statistical_analysis,
            correlation_analysis,
            time_series_analysis,
            anomaly_detection,
            distribution_analysis,
            causal_analysis,
        })
    }
}
