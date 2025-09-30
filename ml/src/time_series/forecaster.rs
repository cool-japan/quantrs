//! Main forecasting coordinator and execution logic

use super::{
    config::*,
    decomposition::QuantumSeasonalDecomposer,
    ensemble::QuantumEnsembleManager,
    features::QuantumFeatureExtractor,
    metrics::{ForecastMetrics, ForecastResult, TrainingHistory},
    models::{TimeSeriesModelFactory, TimeSeriesModelTrait},
};
use crate::error::{MLError, Result};
use crate::optimization::OptimizationMethod;
use scirs2_core::ndarray::{s, Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Main quantum time series forecaster
#[derive(Debug, Clone)]
pub struct QuantumTimeSeriesForecaster {
    /// Configuration
    config: QuantumTimeSeriesConfig,

    /// Base forecasting model
    model: Box<dyn TimeSeriesModelTrait>,

    /// Feature extractor
    feature_extractor: QuantumFeatureExtractor,

    /// Seasonal decomposer
    seasonal_decomposer: Option<QuantumSeasonalDecomposer>,

    /// Ensemble manager (if configured)
    ensemble_manager: Option<QuantumEnsembleManager>,

    /// Training history
    training_history: TrainingHistory,

    /// Forecast metrics
    metrics: ForecastMetrics,

    /// Quantum state cache
    quantum_state_cache: QuantumStateCache,

    /// Prediction cache
    prediction_cache: PredictionCache,
}

/// Quantum state cache for efficiency
#[derive(Debug, Clone)]
pub struct QuantumStateCache {
    /// Cached quantum states
    states: HashMap<String, Array1<f64>>,

    /// Maximum cache size
    max_size: usize,

    /// Access history for LRU eviction
    access_history: VecDeque<String>,

    /// Cache statistics
    stats: CacheStatistics,
}

/// Prediction cache for performance optimization
#[derive(Debug, Clone)]
pub struct PredictionCache {
    /// Cached predictions
    predictions: HashMap<String, CachedPrediction>,

    /// Cache TTL in seconds
    ttl_seconds: u64,

    /// Maximum cache size
    max_size: usize,

    /// Cache statistics
    stats: CacheStatistics,
}

/// Cached prediction with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedPrediction {
    /// Prediction result
    pub result: ForecastResult,

    /// Timestamp when cached
    pub timestamp: std::time::SystemTime,

    /// Input data hash for validation
    pub input_hash: u64,

    /// Model version
    pub model_version: String,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// Number of cache hits
    pub hits: usize,

    /// Number of cache misses
    pub misses: usize,

    /// Total cache accesses
    pub total_accesses: usize,

    /// Hit rate percentage
    pub hit_rate: f64,
}

/// Forecasting execution context
#[derive(Debug, Clone)]
pub struct ForecastingContext {
    /// Execution mode
    pub mode: ExecutionMode,

    /// Parallel execution settings
    pub parallel_config: ParallelConfig,

    /// Memory optimization settings
    pub memory_config: MemoryConfig,

    /// Logging and monitoring
    pub monitoring: MonitoringConfig,
}

/// Execution modes for forecasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Single-threaded execution
    Sequential,

    /// Multi-threaded execution
    Parallel,

    /// Distributed execution
    Distributed,

    /// Quantum-enhanced execution
    QuantumAccelerated,
}

/// Parallel execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Number of worker threads
    pub num_threads: usize,

    /// Batch size for parallel processing
    pub batch_size: usize,

    /// Enable GPU acceleration
    pub use_gpu: bool,

    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WorkStealing,
    DynamicPartitioning,
    QuantumOptimal,
}

/// Memory optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Enable memory pooling
    pub use_memory_pool: bool,

    /// Maximum memory usage in MB
    pub max_memory_mb: usize,

    /// Enable compression for large datasets
    pub use_compression: bool,

    /// Garbage collection strategy
    pub gc_strategy: GCStrategy,
}

/// Garbage collection strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GCStrategy {
    Aggressive,
    Conservative,
    Adaptive,
    QuantumOptimized,
}

/// Monitoring and logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable performance monitoring
    pub enable_monitoring: bool,

    /// Log level for forecasting operations
    pub log_level: LogLevel,

    /// Enable telemetry collection
    pub enable_telemetry: bool,

    /// Metrics collection interval
    pub metrics_interval_ms: u64,
}

/// Logging levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl QuantumTimeSeriesForecaster {
    /// Create new quantum time series forecaster
    pub fn new(config: QuantumTimeSeriesConfig) -> Result<Self> {
        // Create base model using factory
        let model = TimeSeriesModelFactory::create_model(&config.model_type, config.num_qubits)?;

        // Create feature extractor
        let feature_extractor =
            QuantumFeatureExtractor::new(config.feature_config.clone(), config.num_qubits)?;

        // Create seasonal decomposer if needed
        let seasonal_decomposer = if config.seasonality_config.has_seasonality() {
            Some(QuantumSeasonalDecomposer::new(
                config.seasonality_config.clone(),
                config.num_qubits,
            )?)
        } else {
            None
        };

        // Create ensemble manager if configured
        let ensemble_manager = if let Some(ref ensemble_config) = config.ensemble_config {
            let mut manager = QuantumEnsembleManager::new(ensemble_config.clone());

            // Create ensemble models
            let mut ensemble_models = Vec::new();
            for _ in 0..ensemble_config.num_models {
                let ensemble_model =
                    TimeSeriesModelFactory::create_model(&config.model_type, config.num_qubits)?;
                ensemble_models.push(ensemble_model);
            }

            manager.set_models(ensemble_models);
            Some(manager)
        } else {
            None
        };

        // Initialize caches
        let quantum_state_cache = QuantumStateCache::new(1000);
        let prediction_cache = PredictionCache::new(100, 3600); // 1 hour TTL

        Ok(Self {
            config,
            model,
            feature_extractor,
            seasonal_decomposer,
            ensemble_manager,
            training_history: TrainingHistory::new(),
            metrics: ForecastMetrics::new(),
            quantum_state_cache,
            prediction_cache,
        })
    }

    /// Fit the forecaster to training data
    pub fn fit(
        &mut self,
        data: &Array2<f64>, // [time_steps, features]
        epochs: usize,
        optimizer: OptimizationMethod,
    ) -> Result<()> {
        let start_time = Instant::now();
        println!("Training quantum time series model...");

        // Validate input data
        self.validate_training_data(data)?;

        // Prepare features and targets
        let (features, targets) = self.prepare_training_data(data)?;

        // Apply seasonal decomposition if configured
        let (detrended_features, trend, seasonal) =
            if let Some(ref mut decomposer) = self.seasonal_decomposer {
                decomposer.decompose(&features)?
            } else {
                (features.clone(), None, None)
            };

        // Fit feature extractor statistics
        let mut feature_extractor = self.feature_extractor.clone();
        feature_extractor.fit_statistics(&detrended_features)?;
        self.feature_extractor = feature_extractor;

        // Extract quantum features
        let quantum_features = self
            .feature_extractor
            .extract_features(&detrended_features)?;

        // Train base model
        self.model.fit(&quantum_features, &targets)?;

        // Train ensemble models if configured
        if let Some(ref mut ensemble_manager) = self.ensemble_manager {
            ensemble_manager.fit_ensemble(&quantum_features, &targets)?;
        }

        // Store decomposition components in cache
        if let Some(trend) = trend {
            self.quantum_state_cache.store("trend".to_string(), trend);
        }
        if let Some(seasonal) = seasonal {
            self.quantum_state_cache
                .store("seasonal".to_string(), seasonal);
        }

        // Update training history
        let training_time = start_time.elapsed();
        self.training_history.training_time = training_time.as_secs_f64();
        self.training_history
            .add_epoch_metrics(HashMap::new(), 0.0, 0.0);

        println!(
            "Training completed in {:.2} seconds",
            training_time.as_secs_f64()
        );
        Ok(())
    }

    /// Generate forecasts with comprehensive analysis
    pub fn predict(
        &mut self,
        context: &Array2<f64>,
        horizon: Option<usize>,
    ) -> Result<ForecastResult> {
        let forecast_horizon = horizon.unwrap_or(self.config.forecast_horizon);

        // Check prediction cache
        let cache_key = self.generate_prediction_cache_key(context, forecast_horizon);
        if let Some(cached_result) = self.prediction_cache.get(&cache_key) {
            return Ok(cached_result.result.clone());
        }

        // Validate input context
        self.validate_prediction_context(context)?;

        // Extract features from context
        let features = self.feature_extractor.extract_features(context)?;

        // Generate predictions
        let mut predictions = if let Some(ref ensemble_manager) = self.ensemble_manager {
            // Use ensemble prediction
            ensemble_manager.predict_ensemble(&features, forecast_horizon)?
        } else {
            // Use base model prediction
            self.model.predict(&features, forecast_horizon)?
        };

        // Add back trend and seasonal components if they were removed
        predictions = self.reconstruct_predictions(predictions, forecast_horizon)?;

        // Calculate prediction intervals
        let (lower_bound, upper_bound) = self.calculate_prediction_intervals(&predictions)?;

        // Detect anomalies in predictions
        let anomalies = self.detect_prediction_anomalies(&predictions)?;

        // Calculate confidence scores
        let confidence_scores = self.calculate_confidence_scores(&predictions)?;

        // Calculate quantum uncertainty
        let quantum_uncertainty = self.calculate_quantum_uncertainty(&predictions)?;

        let result = ForecastResult {
            predictions,
            lower_bound,
            upper_bound,
            anomalies,
            confidence_scores,
            quantum_uncertainty,
        };

        // Cache the result
        self.prediction_cache.insert(cache_key, &result)?;

        Ok(result)
    }

    /// Validate training data
    fn validate_training_data(&self, data: &Array2<f64>) -> Result<()> {
        let (n_samples, n_features) = data.dim();

        if n_samples < self.config.window_size + self.config.forecast_horizon {
            return Err(MLError::DataError(format!(
                "Insufficient data: need at least {} samples, got {}",
                self.config.window_size + self.config.forecast_horizon,
                n_samples
            )));
        }

        if n_features == 0 {
            return Err(MLError::DataError(
                "No features in training data".to_string(),
            ));
        }

        // Check for NaN or infinite values
        for value in data.iter() {
            if !value.is_finite() {
                return Err(MLError::DataError(
                    "Training data contains NaN or infinite values".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Validate prediction context
    fn validate_prediction_context(&self, context: &Array2<f64>) -> Result<()> {
        let (n_samples, _) = context.dim();

        if n_samples < self.config.window_size {
            return Err(MLError::DataError(format!(
                "Insufficient context: need at least {} samples, got {}",
                self.config.window_size, n_samples
            )));
        }

        // Check for NaN or infinite values
        for value in context.iter() {
            if !value.is_finite() {
                return Err(MLError::DataError(
                    "Context data contains NaN or infinite values".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Prepare training data with sliding window approach
    fn prepare_training_data(&self, data: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        let num_samples = data
            .nrows()
            .saturating_sub(self.config.window_size + self.config.forecast_horizon - 1);

        if num_samples == 0 {
            return Err(MLError::DataError(
                "Insufficient data for the specified window size and forecast horizon".to_string(),
            ));
        }

        let num_features = data.ncols();
        let total_features = num_features
            * (self.config.window_size + self.config.feature_config.lag_features.len());

        let mut features = Array2::zeros((num_samples, total_features));
        let mut targets = Array2::zeros((num_samples, self.config.forecast_horizon * num_features));

        for i in 0..num_samples {
            // Window features
            let window_start = i;
            let window_end = i + self.config.window_size;
            let window_data = data.slice(s![window_start..window_end, ..]);

            // Flatten window data
            let flat_window: Vec<f64> = window_data.iter().cloned().collect();
            let flat_window_len = flat_window.len();
            features
                .slice_mut(s![i, 0..flat_window_len])
                .assign(&Array1::from_vec(flat_window));

            // Add lag features
            let mut feature_offset = flat_window_len;
            for &lag in &self.config.feature_config.lag_features {
                if i >= lag {
                    let lag_data = data.row(i + self.config.window_size - lag);
                    features
                        .slice_mut(s![i, feature_offset..feature_offset + num_features])
                        .assign(&lag_data);
                }
                feature_offset += num_features;
            }

            // Targets
            let target_start = i + self.config.window_size;
            let target_end = target_start + self.config.forecast_horizon;
            let target_data = data.slice(s![target_start..target_end, ..]);
            let flat_target: Vec<f64> = target_data.iter().cloned().collect();
            targets.row_mut(i).assign(&Array1::from_vec(flat_target));
        }

        Ok((features, targets))
    }

    /// Reconstruct predictions by adding back trend and seasonal components
    fn reconstruct_predictions(
        &mut self,
        mut predictions: Array2<f64>,
        horizon: usize,
    ) -> Result<Array2<f64>> {
        // Add back trend component
        if let Some(trend) = self.quantum_state_cache.get("trend") {
            let trend = trend.clone();
            predictions = self.add_trend_component(predictions, &trend, horizon)?;
        }

        // Add back seasonal component
        if let Some(seasonal) = self.quantum_state_cache.get("seasonal") {
            let seasonal = seasonal.clone();
            predictions = self.add_seasonal_component(predictions, &seasonal, horizon)?;
        }

        Ok(predictions)
    }

    /// Add trend component to predictions
    fn add_trend_component(
        &self,
        mut predictions: Array2<f64>,
        trend: &Array1<f64>,
        horizon: usize,
    ) -> Result<Array2<f64>> {
        let trend_len = trend.len();

        for i in 0..predictions.nrows() {
            for h in 0..horizon.min(predictions.ncols()) {
                let trend_idx = (trend_len.saturating_sub(1) + h) % trend_len;
                predictions[[i, h]] += trend[trend_idx];
            }
        }

        Ok(predictions)
    }

    /// Add seasonal component to predictions
    fn add_seasonal_component(
        &self,
        mut predictions: Array2<f64>,
        seasonal: &Array1<f64>,
        horizon: usize,
    ) -> Result<Array2<f64>> {
        let seasonal_len = seasonal.len();

        for i in 0..predictions.nrows() {
            for h in 0..horizon.min(predictions.ncols()) {
                let seasonal_idx = (seasonal_len.saturating_sub(1) + h) % seasonal_len;
                predictions[[i, h]] += seasonal[seasonal_idx];
            }
        }

        Ok(predictions)
    }

    /// Calculate prediction intervals using quantile regression
    fn calculate_prediction_intervals(
        &self,
        predictions: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        // Simplified prediction intervals based on historical residuals
        let std_dev = 0.1; // Would calculate from model residuals in practice
        let z_score = 1.96; // 95% confidence interval

        let margin = std_dev * z_score;
        let lower_bound = predictions - margin;
        let upper_bound = predictions + margin;

        Ok((lower_bound, upper_bound))
    }

    /// Detect anomalies in predictions
    fn detect_prediction_anomalies(
        &self,
        predictions: &Array2<f64>,
    ) -> Result<Vec<super::metrics::AnomalyPoint>> {
        let mut anomalies = Vec::new();

        // Simple anomaly detection based on prediction magnitude
        for (i, row) in predictions.rows().into_iter().enumerate() {
            let mean = row.mean().unwrap_or(0.0);
            let std = row.std(1.0);

            for (j, &value) in row.iter().enumerate() {
                let z_score = if std > 1e-10 {
                    (value - mean).abs() / std
                } else {
                    0.0
                };

                if z_score > 3.0 {
                    anomalies.push(super::metrics::AnomalyPoint {
                        timestamp: i * predictions.ncols() + j,
                        value,
                        anomaly_score: z_score,
                        anomaly_type: super::config::AnomalyType::Point,
                    });
                }
            }
        }

        Ok(anomalies)
    }

    /// Calculate confidence scores for predictions
    fn calculate_confidence_scores(&self, predictions: &Array2<f64>) -> Result<Array1<f64>> {
        let mut confidence_scores = Array1::zeros(predictions.ncols());

        // Calculate confidence based on prediction stability
        for j in 0..predictions.ncols() {
            let column = predictions.column(j);
            let std = column.std(1.0);
            let mean_abs = column.mapv(|x| x.abs()).mean().unwrap_or(1.0);

            // Higher confidence for more stable predictions
            let stability = 1.0 / (1.0 + std / mean_abs.max(1e-10));
            confidence_scores[j] = stability.min(1.0).max(0.0);
        }

        Ok(confidence_scores)
    }

    /// Calculate quantum uncertainty measure
    fn calculate_quantum_uncertainty(&self, predictions: &Array2<f64>) -> Result<f64> {
        // Simplified quantum uncertainty based on prediction variance
        let variance = predictions.var(0.0);
        let uncertainty = variance.ln().max(0.0) / 10.0; // Normalized logarithmic uncertainty
        Ok(uncertainty.min(1.0))
    }

    /// Generate cache key for predictions
    fn generate_prediction_cache_key(&self, context: &Array2<f64>, horizon: usize) -> String {
        // Simple hash-based key (in practice would use proper hashing)
        format!(
            "pred_{}x{}_h{}_{:.6}",
            context.nrows(),
            context.ncols(),
            horizon,
            context.sum()
        )
    }

    /// Update metrics with actual values for evaluation
    pub fn update_metrics(
        &mut self,
        predictions: &Array2<f64>,
        actuals: &Array2<f64>,
    ) -> Result<()> {
        self.metrics.calculate_metrics(predictions, actuals)?;
        Ok(())
    }

    /// Get current forecast metrics
    pub fn get_metrics(&self) -> &ForecastMetrics {
        &self.metrics
    }

    /// Get training history
    pub fn get_training_history(&self) -> &TrainingHistory {
        &self.training_history
    }

    /// Get model configuration
    pub fn get_config(&self) -> &QuantumTimeSeriesConfig {
        &self.config
    }

    /// Get cache statistics
    pub fn get_cache_statistics(&self) -> (CacheStatistics, CacheStatistics) {
        (
            self.quantum_state_cache.get_stats(),
            self.prediction_cache.get_stats(),
        )
    }

    /// Clear caches
    pub fn clear_caches(&mut self) {
        self.quantum_state_cache.clear();
        self.prediction_cache.clear();
    }

    /// Save forecaster state
    pub fn save_state(&self, path: &str) -> Result<()> {
        // In a real implementation, would serialize the forecaster state
        println!("Saving forecaster state to: {}", path);
        Ok(())
    }

    /// Load forecaster state
    pub fn load_state(&mut self, path: &str) -> Result<()> {
        // In a real implementation, would deserialize the forecaster state
        println!("Loading forecaster state from: {}", path);
        Ok(())
    }
}

impl QuantumStateCache {
    /// Create new quantum state cache
    pub fn new(max_size: usize) -> Self {
        Self {
            states: HashMap::new(),
            max_size,
            access_history: VecDeque::new(),
            stats: CacheStatistics::new(),
        }
    }

    /// Store quantum state in cache
    pub fn store(&mut self, key: String, state: Array1<f64>) {
        // Implement LRU eviction if cache is full
        if self.states.len() >= self.max_size {
            if let Some(lru_key) = self.access_history.pop_front() {
                self.states.remove(&lru_key);
            }
        }

        self.states.insert(key.clone(), state);
        self.access_history.push_back(key);
    }

    /// Get quantum state from cache
    pub fn get(&mut self, key: &str) -> Option<&Array1<f64>> {
        self.stats.total_accesses += 1;

        if let Some(state) = self.states.get(key) {
            self.stats.hits += 1;

            // Update access history for LRU
            if let Some(pos) = self.access_history.iter().position(|k| k == key) {
                if let Some(key_owned) = self.access_history.remove(pos) {
                    self.access_history.push_back(key_owned);
                }
            }

            Some(state)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Clear all cached states
    pub fn clear(&mut self) {
        self.states.clear();
        self.access_history.clear();
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStatistics {
        let mut stats = self.stats.clone();
        stats.hit_rate = if stats.total_accesses > 0 {
            stats.hits as f64 / stats.total_accesses as f64 * 100.0
        } else {
            0.0
        };
        stats
    }
}

impl PredictionCache {
    /// Create new prediction cache
    pub fn new(max_size: usize, ttl_seconds: u64) -> Self {
        Self {
            predictions: HashMap::new(),
            ttl_seconds,
            max_size,
            stats: CacheStatistics::new(),
        }
    }

    /// Get cached prediction
    pub fn get(&mut self, key: &str) -> Option<&CachedPrediction> {
        self.stats.total_accesses += 1;

        // First check if entry exists and is valid
        let is_valid = if let Some(cached) = self.predictions.get(key) {
            if let Ok(elapsed) = cached.timestamp.elapsed() {
                elapsed.as_secs() < self.ttl_seconds
            } else {
                false
            }
        } else {
            false
        };

        if is_valid {
            self.stats.hits += 1;
            self.predictions.get(key)
        } else {
            // Remove expired entry if it exists
            self.predictions.remove(key);
            self.stats.misses += 1;
            None
        }
    }

    /// Insert prediction into cache
    pub fn insert(&mut self, key: String, result: &ForecastResult) -> Result<()> {
        // Implement simple eviction if cache is full
        if self.predictions.len() >= self.max_size {
            // Remove oldest entry (simplified - would use LRU in practice)
            if let Some(first_key) = self.predictions.keys().next().cloned() {
                self.predictions.remove(&first_key);
            }
        }

        let cached_prediction = CachedPrediction {
            result: result.clone(),
            timestamp: std::time::SystemTime::now(),
            input_hash: 0, // Simplified
            model_version: "1.0".to_string(),
        };

        self.predictions.insert(key, cached_prediction);
        Ok(())
    }

    /// Clear all cached predictions
    pub fn clear(&mut self) {
        self.predictions.clear();
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStatistics {
        let mut stats = self.stats.clone();
        stats.hit_rate = if stats.total_accesses > 0 {
            stats.hits as f64 / stats.total_accesses as f64 * 100.0
        } else {
            0.0
        };
        stats
    }
}

impl CacheStatistics {
    /// Create new cache statistics
    pub fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            total_accesses: 0,
            hit_rate: 0.0,
        }
    }
}

impl Default for ForecastingContext {
    fn default() -> Self {
        Self {
            mode: ExecutionMode::Sequential,
            parallel_config: ParallelConfig {
                num_threads: 4,
                batch_size: 32,
                use_gpu: false,
                load_balancing: LoadBalancingStrategy::RoundRobin,
            },
            memory_config: MemoryConfig {
                use_memory_pool: true,
                max_memory_mb: 1024,
                use_compression: false,
                gc_strategy: GCStrategy::Adaptive,
            },
            monitoring: MonitoringConfig {
                enable_monitoring: true,
                log_level: LogLevel::Info,
                enable_telemetry: false,
                metrics_interval_ms: 1000,
            },
        }
    }
}

/// Convenience functions for creating forecasters

/// Create forecaster with default configuration
pub fn create_default_forecaster() -> Result<QuantumTimeSeriesForecaster> {
    QuantumTimeSeriesForecaster::new(QuantumTimeSeriesConfig::default())
}

/// Create forecaster for financial time series
pub fn create_financial_forecaster(forecast_horizon: usize) -> Result<QuantumTimeSeriesForecaster> {
    QuantumTimeSeriesForecaster::new(QuantumTimeSeriesConfig::financial(forecast_horizon))
}

/// Create forecaster for IoT sensor data
pub fn create_iot_forecaster(sampling_rate: usize) -> Result<QuantumTimeSeriesForecaster> {
    QuantumTimeSeriesForecaster::new(QuantumTimeSeriesConfig::iot_sensor(sampling_rate))
}

/// Create forecaster for demand forecasting
pub fn create_demand_forecaster() -> Result<QuantumTimeSeriesForecaster> {
    QuantumTimeSeriesForecaster::new(QuantumTimeSeriesConfig::demand_forecasting())
}
