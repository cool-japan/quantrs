//! Seasonal and trend decomposition with quantum enhancement

use super::config::*;
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Quantum seasonal decomposer for time series
#[derive(Debug, Clone)]
pub struct QuantumSeasonalDecomposer {
    /// Seasonality configuration
    config: SeasonalityConfig,

    /// Quantum circuits for seasonal extraction
    seasonal_circuits: HashMap<String, Vec<f64>>,

    /// Trend extractor
    trend_extractor: QuantumTrendExtractor,

    /// Residual analyzer
    residual_analyzer: QuantumResidualAnalyzer,

    /// Decomposition results cache
    decomposition_cache: DecompositionCache,
}

/// Quantum trend extractor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumTrendExtractor {
    /// Trend smoothing parameter
    smoothing_param: f64,

    /// Quantum circuit for trend extraction
    trend_circuit: Vec<f64>,

    /// Changepoint detector
    changepoint_detector: Option<QuantumChangepointDetector>,

    /// Trend model parameters
    trend_parameters: TrendParameters,
}

/// Quantum changepoint detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumChangepointDetector {
    /// Detection threshold
    threshold: f64,

    /// Quantum circuit for detection
    detection_circuit: Vec<f64>,

    /// Detected changepoints
    changepoints: Vec<ChangePoint>,

    /// Detection algorithm
    algorithm: ChangepointAlgorithm,
}

/// Changepoint information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePoint {
    /// Time index of changepoint
    pub time_index: usize,

    /// Confidence score
    pub confidence: f64,

    /// Type of change
    pub change_type: ChangeType,

    /// Magnitude of change
    pub magnitude: f64,
}

/// Types of changes detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    TrendChange,
    LevelShift,
    VarianceChange,
    SeasonalityChange,
    QuantumPhaseTransition,
}

/// Changepoint detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangepointAlgorithm {
    CUSUM,
    PELT,
    BinarySegmentation,
    QuantumDetection,
    HybridQuantumClassical,
}

/// Trend model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendParameters {
    /// Linear trend coefficient
    pub linear_coeff: f64,

    /// Quadratic trend coefficient
    pub quadratic_coeff: f64,

    /// Exponential growth rate
    pub exp_growth_rate: f64,

    /// Quantum enhancement parameters
    pub quantum_params: Array1<f64>,
}

/// Quantum residual analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResidualAnalyzer {
    /// Quantum circuit for residual analysis
    analysis_circuit: Vec<f64>,

    /// Anomaly detection threshold
    anomaly_threshold: f64,

    /// Detected anomalies
    anomalies: Vec<AnomalyPoint>,

    /// Residual statistics
    residual_stats: ResidualStatistics,
}

/// Anomaly point in residuals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyPoint {
    /// Time index
    pub timestamp: usize,

    /// Anomalous value
    pub value: f64,

    /// Anomaly score
    pub anomaly_score: f64,

    /// Type of anomaly
    pub anomaly_type: AnomalyType,
}

/// Residual statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualStatistics {
    /// Mean of residuals
    pub mean: f64,

    /// Standard deviation of residuals
    pub std: f64,

    /// Skewness
    pub skewness: f64,

    /// Kurtosis
    pub kurtosis: f64,

    /// Autocorrelation function
    pub autocorrelation: Array1<f64>,

    /// Quantum coherence measure
    pub quantum_coherence: f64,
}

/// Decomposition cache for performance
#[derive(Debug, Clone)]
pub struct DecompositionCache {
    /// Cached decompositions
    cache: HashMap<String, DecompositionResult>,

    /// Cache hit count
    hits: usize,

    /// Cache miss count
    misses: usize,

    /// Maximum cache size
    max_size: usize,
}

/// Decomposition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionResult {
    /// Original time series
    pub original: Array2<f64>,

    /// Trend component
    pub trend: Array1<f64>,

    /// Seasonal components
    pub seasonal: HashMap<String, Array1<f64>>,

    /// Residual component
    pub residual: Array1<f64>,

    /// Decomposition quality metrics
    pub quality_metrics: DecompositionQuality,
}

/// Quality metrics for decomposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionQuality {
    /// Explained variance ratio
    pub explained_variance: f64,

    /// Reconstruction error
    pub reconstruction_error: f64,

    /// Seasonal strength
    pub seasonal_strength: HashMap<String, f64>,

    /// Trend strength
    pub trend_strength: f64,

    /// Quantum enhancement effectiveness
    pub quantum_effectiveness: f64,
}

/// Seasonal component extractor
#[derive(Debug, Clone)]
pub struct SeasonalComponentExtractor {
    /// Period length
    period: usize,

    /// Component name
    name: String,

    /// Quantum circuit parameters
    quantum_circuit: Vec<f64>,

    /// Extraction method
    method: SeasonalExtractionMethod,
}

/// Methods for seasonal extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SeasonalExtractionMethod {
    MovingAverage,
    FourierDecomposition,
    STL,
    QuantumFourier,
    WaveletDecomposition,
    HybridQuantumClassical,
}

impl QuantumSeasonalDecomposer {
    /// Create new quantum seasonal decomposer
    pub fn new(config: SeasonalityConfig, num_qubits: usize) -> Result<Self> {
        let mut seasonal_circuits = HashMap::new();

        // Create circuits for each seasonality component
        if let Some(daily) = config.daily {
            seasonal_circuits.insert(
                "daily".to_string(),
                Self::create_seasonal_circuit(daily, num_qubits)?,
            );
        }

        if let Some(weekly) = config.weekly {
            seasonal_circuits.insert(
                "weekly".to_string(),
                Self::create_seasonal_circuit(weekly, num_qubits)?,
            );
        }

        if let Some(monthly) = config.monthly {
            seasonal_circuits.insert(
                "monthly".to_string(),
                Self::create_seasonal_circuit(monthly, num_qubits)?,
            );
        }

        if let Some(yearly) = config.yearly {
            seasonal_circuits.insert(
                "yearly".to_string(),
                Self::create_seasonal_circuit(yearly, num_qubits)?,
            );
        }

        // Create circuits for custom periods
        for (i, &period) in config.custom_periods.iter().enumerate() {
            let name = format!("custom_{}", i);
            seasonal_circuits.insert(name, Self::create_seasonal_circuit(period, num_qubits)?);
        }

        let trend_extractor = QuantumTrendExtractor::new(0.1, num_qubits)?;
        let residual_analyzer = QuantumResidualAnalyzer::new(num_qubits)?;
        let decomposition_cache = DecompositionCache::new(100);

        Ok(Self {
            config,
            seasonal_circuits,
            trend_extractor,
            residual_analyzer,
            decomposition_cache,
        })
    }

    /// Create quantum circuit for seasonal extraction
    fn create_seasonal_circuit(period: usize, num_qubits: usize) -> Result<Vec<f64>> {
        let mut circuit = Vec::new();

        // Frequency encoding for seasonal pattern
        for i in 0..num_qubits {
            let frequency = 2.0 * PI * i as f64 / period as f64;
            circuit.push(frequency);
        }

        // Phase adjustments for quantum enhancement
        for i in 0..num_qubits {
            let phase = PI * i as f64 / num_qubits as f64;
            circuit.push(phase);
        }

        Ok(circuit)
    }

    /// Decompose time series into components
    pub fn decompose(
        &mut self,
        data: &Array2<f64>,
    ) -> Result<(Array2<f64>, Option<Array1<f64>>, Option<Array1<f64>>)> {
        // Check cache first
        let cache_key = self.generate_cache_key(data);
        if let Some(cached_result) = self.decomposition_cache.get(&cache_key) {
            let cached_result = cached_result.clone();
            return self.convert_cached_result(&cached_result);
        }

        // Perform full decomposition
        let decomposition = self.full_decomposition(data)?;

        // Cache the result
        self.decomposition_cache
            .insert(cache_key, decomposition.clone());

        // Convert to expected format
        let detrended = &decomposition.original - &decomposition.trend.clone().insert_axis(Axis(1));
        let deseasonalized =
            self.remove_seasonal_components(&detrended, &decomposition.seasonal)?;

        Ok((
            deseasonalized,
            Some(decomposition.trend),
            self.combine_seasonal_components(&decomposition.seasonal),
        ))
    }

    /// Perform full decomposition analysis
    fn full_decomposition(&mut self, data: &Array2<f64>) -> Result<DecompositionResult> {
        // Extract trend component
        let trend = self.trend_extractor.extract_trend(data)?;

        // Remove trend
        let detrended = data - &trend.clone().insert_axis(Axis(1));

        // Extract seasonal components
        let seasonal = self.extract_all_seasonal_components(&detrended)?;

        // Calculate residual
        let mut residual_data = detrended.clone();
        for (_, seasonal_component) in &seasonal {
            residual_data = &residual_data - &seasonal_component.clone().insert_axis(Axis(1));
        }
        let residual = residual_data.column(0).to_owned();

        // Analyze residuals
        self.residual_analyzer.analyze_residuals(&residual)?;

        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(data, &trend, &seasonal, &residual)?;

        Ok(DecompositionResult {
            original: data.clone(),
            trend,
            seasonal,
            residual,
            quality_metrics,
        })
    }

    /// Extract all configured seasonal components
    fn extract_all_seasonal_components(
        &self,
        data: &Array2<f64>,
    ) -> Result<HashMap<String, Array1<f64>>> {
        let mut seasonal_components = HashMap::new();

        for (component_name, circuit) in &self.seasonal_circuits {
            let component = self.extract_seasonal_component(data, component_name, circuit)?;
            seasonal_components.insert(component_name.clone(), component);
        }

        Ok(seasonal_components)
    }

    /// Extract specific seasonal component
    fn extract_seasonal_component(
        &self,
        data: &Array2<f64>,
        name: &str,
        circuit: &[f64],
    ) -> Result<Array1<f64>> {
        let n_samples = data.nrows();
        let mut seasonal_component = Array1::zeros(n_samples);

        // Determine period from component name and config
        let period = self.get_period_for_component(name);

        if let Some(period) = period {
            // Apply quantum-enhanced seasonal extraction
            for t in 0..n_samples {
                let mut seasonal_value = 0.0;

                // Classical seasonal extraction
                let phase = 2.0 * PI * (t % period) as f64 / period as f64;
                seasonal_value += phase.sin();

                // Quantum enhancement
                if !circuit.is_empty() {
                    let circuit_idx = t % circuit.len();
                    let quantum_phase = circuit[circuit_idx] * phase;
                    seasonal_value += 0.1 * quantum_phase.cos(); // Quantum correction
                }

                seasonal_component[t] = seasonal_value;
            }

            // Normalize component
            self.normalize_seasonal_component(&mut seasonal_component);
        }

        Ok(seasonal_component)
    }

    /// Get period for component name
    fn get_period_for_component(&self, name: &str) -> Option<usize> {
        match name {
            "daily" => self.config.daily,
            "weekly" => self.config.weekly,
            "monthly" => self.config.monthly,
            "yearly" => self.config.yearly,
            custom_name if custom_name.starts_with("custom_") => {
                if let Ok(index) = custom_name[7..].parse::<usize>() {
                    self.config.custom_periods.get(index).copied()
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Normalize seasonal component
    fn normalize_seasonal_component(&self, component: &mut Array1<f64>) {
        let mean = component.mean().unwrap_or(0.0);
        for value in component.iter_mut() {
            *value -= mean; // Center around zero
        }
    }

    /// Remove seasonal components from data
    fn remove_seasonal_components(
        &self,
        data: &Array2<f64>,
        seasonal: &HashMap<String, Array1<f64>>,
    ) -> Result<Array2<f64>> {
        let mut result = data.clone();

        for (_, component) in seasonal {
            result = &result - &component.clone().insert_axis(Axis(1));
        }

        Ok(result)
    }

    /// Combine seasonal components
    fn combine_seasonal_components(
        &self,
        seasonal: &HashMap<String, Array1<f64>>,
    ) -> Option<Array1<f64>> {
        if seasonal.is_empty() {
            return None;
        }

        let first_component = seasonal.values().next()?;
        let mut combined = Array1::zeros(first_component.len());

        for component in seasonal.values() {
            combined = combined + component;
        }

        Some(combined)
    }

    /// Calculate decomposition quality metrics
    fn calculate_quality_metrics(
        &self,
        original: &Array2<f64>,
        trend: &Array1<f64>,
        seasonal: &HashMap<String, Array1<f64>>,
        residual: &Array1<f64>,
    ) -> Result<DecompositionQuality> {
        // Calculate explained variance
        let original_var = original.var(0.0);
        let residual_var = residual.var(0.0);
        let explained_variance = 1.0 - (residual_var / original_var.max(1e-10));

        // Calculate reconstruction error
        let mut reconstructed = trend.clone().insert_axis(Axis(1));
        for component in seasonal.values() {
            reconstructed = reconstructed + component.clone().insert_axis(Axis(1));
        }
        reconstructed = reconstructed + residual.clone().insert_axis(Axis(1));

        let reconstruction_error = (original - &reconstructed)
            .mapv(|x| x * x)
            .mean()
            .unwrap_or(0.0)
            .sqrt();

        // Calculate seasonal strengths
        let mut seasonal_strength = HashMap::new();
        for (name, component) in seasonal {
            let strength = component.var(0.0) / original_var.max(1e-10);
            seasonal_strength.insert(name.clone(), strength);
        }

        // Calculate trend strength
        let trend_strength = trend.var(0.0) / original_var.max(1e-10);

        // Simplified quantum effectiveness measure
        let quantum_effectiveness = 0.8; // Placeholder

        Ok(DecompositionQuality {
            explained_variance,
            reconstruction_error,
            seasonal_strength,
            trend_strength,
            quantum_effectiveness,
        })
    }

    /// Generate cache key for data
    fn generate_cache_key(&self, data: &Array2<f64>) -> String {
        // Simple hash-based key (in practice would use proper hashing)
        format!("{}x{}_{:.3}", data.nrows(), data.ncols(), data.sum())
    }

    /// Convert cached result to expected format
    fn convert_cached_result(
        &self,
        result: &DecompositionResult,
    ) -> Result<(Array2<f64>, Option<Array1<f64>>, Option<Array1<f64>>)> {
        let detrended = &result.original - &result.trend.clone().insert_axis(Axis(1));
        let deseasonalized = self.remove_seasonal_components(&detrended, &result.seasonal)?;
        let combined_seasonal = self.combine_seasonal_components(&result.seasonal);

        Ok((
            deseasonalized,
            Some(result.trend.clone()),
            combined_seasonal,
        ))
    }

    /// Get decomposition quality metrics
    pub fn get_quality_metrics(&self) -> Option<&DecompositionQuality> {
        // Return metrics from last decomposition (simplified)
        None
    }

    /// Get detected changepoints
    pub fn get_changepoints(&self) -> Vec<&ChangePoint> {
        if let Some(ref detector) = self.trend_extractor.changepoint_detector {
            detector.changepoints.iter().collect()
        } else {
            Vec::new()
        }
    }

    /// Get detected anomalies
    pub fn get_anomalies(&self) -> &[AnomalyPoint] {
        &self.residual_analyzer.anomalies
    }
}

impl QuantumTrendExtractor {
    /// Create new quantum trend extractor
    pub fn new(smoothing_param: f64, num_qubits: usize) -> Result<Self> {
        let mut trend_circuit = Vec::new();

        // Smoothing gates
        for i in 0..num_qubits {
            trend_circuit.push(smoothing_param * (i + 1) as f64);
        }

        // Phase rotation parameters
        for i in 0..num_qubits {
            trend_circuit.push(PI * i as f64 / num_qubits as f64);
        }

        let changepoint_detector = Some(QuantumChangepointDetector::new(0.05, num_qubits)?);

        let trend_parameters = TrendParameters {
            linear_coeff: 0.0,
            quadratic_coeff: 0.0,
            exp_growth_rate: 0.0,
            quantum_params: Array1::zeros(num_qubits),
        };

        Ok(Self {
            smoothing_param,
            trend_circuit,
            changepoint_detector,
            trend_parameters,
        })
    }

    /// Extract trend from time series data
    pub fn extract_trend(&mut self, data: &Array2<f64>) -> Result<Array1<f64>> {
        let n_samples = data.nrows();
        let mut trend = Array1::zeros(n_samples);

        // Apply quantum-enhanced trend extraction
        for t in 0..n_samples {
            let mut trend_value = 0.0;

            // Classical trend estimation (moving average)
            let window_size = (n_samples / 10).max(3).min(21); // Adaptive window
            let start = t.saturating_sub(window_size / 2);
            let end = (t + window_size / 2 + 1).min(n_samples);

            let window_sum: f64 = data.slice(s![start..end, 0]).sum();
            trend_value = window_sum / (end - start) as f64;

            // Quantum enhancement
            if !self.trend_circuit.is_empty() {
                let circuit_idx = t % self.trend_circuit.len();
                let quantum_factor = self.trend_circuit[circuit_idx];
                let quantum_phase = quantum_factor * trend_value * PI;
                trend_value += 0.05 * quantum_phase.sin(); // Small quantum correction
            }

            trend[t] = trend_value;
        }

        // Detect changepoints if detector is available
        if let Some(ref mut detector) = self.changepoint_detector {
            detector.detect_changepoints(&trend)?;
        }

        // Fit trend parameters
        self.fit_trend_parameters(&trend)?;

        Ok(trend)
    }

    /// Fit parametric trend model
    fn fit_trend_parameters(&mut self, trend: &Array1<f64>) -> Result<()> {
        let n = trend.len();
        if n < 3 {
            return Ok(()); // Not enough data for fitting
        }

        // Simple linear trend fitting
        let mut sum_t = 0.0;
        let mut sum_y = 0.0;
        let mut sum_ty = 0.0;
        let mut sum_t2 = 0.0;

        for (t, &y) in trend.iter().enumerate() {
            let t_val = t as f64;
            sum_t += t_val;
            sum_y += y;
            sum_ty += t_val * y;
            sum_t2 += t_val * t_val;
        }

        let n_f = n as f64;
        let denominator = n_f * sum_t2 - sum_t * sum_t;

        if denominator.abs() > 1e-10 {
            self.trend_parameters.linear_coeff = (n_f * sum_ty - sum_t * sum_y) / denominator;
            // Intercept would be (sum_y - linear_coeff * sum_t) / n_f
        }

        Ok(())
    }

    /// Get trend parameters
    pub fn get_trend_parameters(&self) -> &TrendParameters {
        &self.trend_parameters
    }
}

impl QuantumChangepointDetector {
    /// Create new changepoint detector
    pub fn new(threshold: f64, num_qubits: usize) -> Result<Self> {
        let mut detection_circuit = Vec::new();

        // Detection gates
        for i in 0..num_qubits {
            detection_circuit.push(1.0); // H gate marker
            detection_circuit.push(PI * i as f64 / num_qubits as f64); // Phase
        }

        Ok(Self {
            threshold,
            detection_circuit,
            changepoints: Vec::new(),
            algorithm: ChangepointAlgorithm::QuantumDetection,
        })
    }

    /// Detect changepoints in time series
    pub fn detect_changepoints(&mut self, data: &Array1<f64>) -> Result<()> {
        self.changepoints.clear();

        match self.algorithm {
            ChangepointAlgorithm::QuantumDetection => self.quantum_detection(data)?,
            ChangepointAlgorithm::CUSUM => self.cusum_detection(data)?,
            _ => {
                // Default to quantum detection
                self.quantum_detection(data)?
            }
        }

        Ok(())
    }

    /// Quantum-enhanced changepoint detection
    fn quantum_detection(&mut self, data: &Array1<f64>) -> Result<()> {
        let n = data.len();
        if n < 10 {
            return Ok(()); // Not enough data
        }

        let window_size = n / 10;

        for t in window_size..(n - window_size) {
            // Calculate local statistics
            let before_window = data.slice(s![t.saturating_sub(window_size)..t]);
            let after_window = data.slice(s![t..t + window_size]);

            let mean_before = before_window.mean().unwrap_or(0.0);
            let mean_after = after_window.mean().unwrap_or(0.0);
            let std_before = before_window.std(1.0);
            let std_after = after_window.std(1.0);

            // Classical change detection
            let mean_change = (mean_after - mean_before).abs() / (std_before + std_after + 1e-10);
            let var_change = (std_after - std_before).abs() / (std_before + 1e-10);

            // Quantum enhancement
            let quantum_factor = if !self.detection_circuit.is_empty() {
                let circuit_idx = t % self.detection_circuit.len();
                let phase = self.detection_circuit[circuit_idx] * mean_change;
                1.0 + 0.1 * phase.sin()
            } else {
                1.0
            };

            let change_score = (mean_change + var_change) * quantum_factor;

            if change_score > self.threshold {
                let changepoint = ChangePoint {
                    time_index: t,
                    confidence: change_score.min(1.0),
                    change_type: if mean_change > var_change {
                        ChangeType::LevelShift
                    } else {
                        ChangeType::VarianceChange
                    },
                    magnitude: change_score,
                };

                self.changepoints.push(changepoint);
            }
        }

        Ok(())
    }

    /// CUSUM-based changepoint detection
    fn cusum_detection(&mut self, data: &Array1<f64>) -> Result<()> {
        let n = data.len();
        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(1.0);

        let mut cusum_pos = 0.0;
        let mut cusum_neg = 0.0;
        let threshold = 4.0 * std; // CUSUM threshold

        for (t, &value) in data.iter().enumerate() {
            let deviation = value - mean;

            cusum_pos = (cusum_pos + deviation).max(0.0);
            cusum_neg = (cusum_neg - deviation).max(0.0);

            if cusum_pos > threshold || cusum_neg > threshold {
                let changepoint = ChangePoint {
                    time_index: t,
                    confidence: (cusum_pos.max(cusum_neg) / threshold).min(1.0),
                    change_type: ChangeType::TrendChange,
                    magnitude: cusum_pos.max(cusum_neg),
                };

                self.changepoints.push(changepoint);

                // Reset CUSUM
                cusum_pos = 0.0;
                cusum_neg = 0.0;
            }
        }

        Ok(())
    }

    /// Get detected changepoints
    pub fn get_changepoints(&self) -> &[ChangePoint] {
        &self.changepoints
    }
}

impl QuantumResidualAnalyzer {
    /// Create new residual analyzer
    pub fn new(num_qubits: usize) -> Result<Self> {
        let mut analysis_circuit = Vec::new();

        // Analysis gates
        for i in 0..num_qubits {
            analysis_circuit.push(1.0); // H gate marker
            analysis_circuit.push(PI * i as f64 / num_qubits as f64); // Phase
        }

        let residual_stats = ResidualStatistics {
            mean: 0.0,
            std: 1.0,
            skewness: 0.0,
            kurtosis: 3.0,
            autocorrelation: Array1::zeros(10),
            quantum_coherence: 0.0,
        };

        Ok(Self {
            analysis_circuit,
            anomaly_threshold: 2.0,
            anomalies: Vec::new(),
            residual_stats,
        })
    }

    /// Analyze residuals for anomalies and patterns
    pub fn analyze_residuals(&mut self, residuals: &Array1<f64>) -> Result<()> {
        // Update residual statistics
        self.update_residual_statistics(residuals)?;

        // Detect anomalies
        self.detect_anomalies(residuals)?;

        Ok(())
    }

    /// Update comprehensive residual statistics
    fn update_residual_statistics(&mut self, residuals: &Array1<f64>) -> Result<()> {
        let n = residuals.len() as f64;

        // Basic statistics
        self.residual_stats.mean = residuals.mean().unwrap_or(0.0);
        self.residual_stats.std = residuals.std(1.0);

        // Higher order moments
        let mean = self.residual_stats.mean;
        let mut skewness_sum = 0.0;
        let mut kurtosis_sum = 0.0;

        for &value in residuals.iter() {
            let deviation = value - mean;
            skewness_sum += deviation.powi(3);
            kurtosis_sum += deviation.powi(4);
        }

        let std_cubed = self.residual_stats.std.powi(3);
        let std_fourth = self.residual_stats.std.powi(4);

        if std_cubed > 1e-10 {
            self.residual_stats.skewness = skewness_sum / (n * std_cubed);
        }

        if std_fourth > 1e-10 {
            self.residual_stats.kurtosis = kurtosis_sum / (n * std_fourth);
        }

        // Autocorrelation function
        self.calculate_autocorrelation(residuals)?;

        // Quantum coherence measure
        self.residual_stats.quantum_coherence = self.calculate_quantum_coherence(residuals)?;

        Ok(())
    }

    /// Calculate autocorrelation function
    fn calculate_autocorrelation(&mut self, residuals: &Array1<f64>) -> Result<()> {
        let n = residuals.len();
        let max_lag = 10.min(n / 4);
        let mut autocorr = Array1::zeros(max_lag);

        let mean = residuals.mean().unwrap_or(0.0);
        let variance = residuals.var(1.0);

        for lag in 0..max_lag {
            let mut sum = 0.0;
            let mut count = 0;

            for t in lag..n {
                sum += (residuals[t] - mean) * (residuals[t - lag] - mean);
                count += 1;
            }

            if count > 0 && variance > 1e-10 {
                autocorr[lag] = sum / (count as f64 * variance);
            }
        }

        self.residual_stats.autocorrelation = autocorr;
        Ok(())
    }

    /// Calculate quantum coherence measure
    fn calculate_quantum_coherence(&self, residuals: &Array1<f64>) -> Result<f64> {
        // Simplified quantum coherence based on entropy
        let n = residuals.len();
        let n_bins = 10;

        let min_val = residuals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = residuals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        if range < 1e-10 {
            return Ok(0.0);
        }

        let mut bin_counts = vec![0; n_bins];
        for &value in residuals.iter() {
            let bin_idx = ((value - min_val) / range * (n_bins - 1) as f64) as usize;
            let bin_idx = bin_idx.min(n_bins - 1);
            bin_counts[bin_idx] += 1;
        }

        let mut entropy = 0.0;
        for &count in &bin_counts {
            if count > 0 {
                let prob = count as f64 / n as f64;
                entropy -= prob * prob.ln();
            }
        }

        // Normalize and convert to coherence measure
        let max_entropy = (n_bins as f64).ln();
        let normalized_entropy = entropy / max_entropy;

        Ok(1.0 - normalized_entropy) // High coherence = low entropy
    }

    /// Detect anomalies in residuals
    fn detect_anomalies(&mut self, residuals: &Array1<f64>) -> Result<()> {
        self.anomalies.clear();

        let mean = self.residual_stats.mean;
        let std = self.residual_stats.std;
        let threshold = self.anomaly_threshold * std;

        for (t, &value) in residuals.iter().enumerate() {
            let deviation = (value - mean).abs();

            if deviation > threshold {
                let anomaly_score = deviation / std;

                // Determine anomaly type based on context
                let anomaly_type = if anomaly_score > 4.0 {
                    AnomalyType::Point
                } else if self.is_contextual_anomaly(t, residuals) {
                    AnomalyType::Contextual
                } else {
                    AnomalyType::Point
                };

                let anomaly = AnomalyPoint {
                    timestamp: t,
                    value,
                    anomaly_score,
                    anomaly_type,
                };

                self.anomalies.push(anomaly);
            }
        }

        Ok(())
    }

    /// Check if anomaly is contextual
    fn is_contextual_anomaly(&self, index: usize, residuals: &Array1<f64>) -> bool {
        let window_size = 5;
        let start = index.saturating_sub(window_size);
        let end = (index + window_size + 1).min(residuals.len());

        if end - start < 3 {
            return false;
        }

        let window = residuals.slice(s![start..end]);
        let local_mean = window.mean().unwrap_or(0.0);
        let local_std = window.std(1.0);

        let global_mean = self.residual_stats.mean;
        let global_std = self.residual_stats.std;

        // Check if local statistics differ significantly from global
        let mean_diff = (local_mean - global_mean).abs() / global_std.max(1e-10);
        let std_ratio = local_std / global_std.max(1e-10);

        mean_diff > 1.0 || std_ratio > 2.0 || std_ratio < 0.5
    }

    /// Get residual statistics
    pub fn get_statistics(&self) -> &ResidualStatistics {
        &self.residual_stats
    }

    /// Get detected anomalies
    pub fn get_anomalies(&self) -> &[AnomalyPoint] {
        &self.anomalies
    }
}

impl DecompositionCache {
    /// Create new decomposition cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            hits: 0,
            misses: 0,
            max_size,
        }
    }

    /// Get cached decomposition result
    pub fn get(&mut self, key: &str) -> Option<&DecompositionResult> {
        if let Some(result) = self.cache.get(key) {
            self.hits += 1;
            Some(result)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert decomposition result into cache
    pub fn insert(&mut self, key: String, result: DecompositionResult) {
        if self.cache.len() >= self.max_size {
            // Simple eviction: remove a random entry
            if let Some(random_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&random_key);
            }
        }

        self.cache.insert(key, result);
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> (usize, usize, f64) {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
        (self.hits, self.misses, hit_rate)
    }
}
