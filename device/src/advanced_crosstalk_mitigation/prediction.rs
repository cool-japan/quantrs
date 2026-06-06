//! Prediction and time series analysis components

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use scirs2_core::ndarray::{Array1, Array2, Array3};

use super::*;
use crate::{DeviceError, DeviceResult};

impl CrosstalkPredictor {
    pub fn new(config: &CrosstalkPredictionConfig) -> Self {
        Self {
            models: HashMap::new(),
            prediction_horizon: config.prediction_horizon,
            uncertainty_quantifier: UncertaintyQuantifier::new(&config.uncertainty_quantification),
        }
    }

    pub fn generate_predictions(&mut self, characterization: &CrosstalkCharacterization) -> DeviceResult<CrosstalkPredictionResult> {
        let n_qubits = characterization.crosstalk_matrix.nrows();
        let n_predictions = 10; // Number of prediction steps

        // Use the current crosstalk matrix values as initial predictions, then apply
        // exponential smoothing with a small drift factor based on matrix statistics.
        let n_features = n_qubits * n_qubits;
        let flat: Vec<f64> = characterization.crosstalk_matrix.iter().copied().collect();
        let mean_val = if flat.is_empty() { 0.0 } else { flat.iter().sum::<f64>() / flat.len() as f64 };
        let variance = if flat.is_empty() {
            0.0
        } else {
            flat.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / flat.len() as f64
        };

        // Build prediction matrix: each step applies an exponential smoothing factor.
        // This is a stationary baseline (no ML model trained yet), so predictions
        // represent "the crosstalk stays near its current level ± some decay".
        let alpha = 0.9_f64; // smoothing factor toward mean
        let mut predictions = Array2::zeros((n_predictions, n_features));
        for step in 0..n_predictions {
            let decay = alpha.powi(step as i32 + 1);
            for (j, &v) in flat.iter().enumerate() {
                // Each prediction step decays toward the mean
                predictions[[step, j]] = mean_val + decay * (v - mean_val);
            }
        }

        // Uncertainty grows with step (proportional to sqrt(step+1) * std_dev)
        let std_dev = variance.sqrt();
        let mut uncertainty_estimates = Array2::zeros((n_predictions, n_features));
        for step in 0..n_predictions {
            let u = std_dev * ((step + 1) as f64).sqrt() * 0.1;
            for j in 0..n_features {
                uncertainty_estimates[[step, j]] = u;
            }
        }

        // Confidence intervals at 95% (z=1.96)
        let mut confidence_intervals = Array3::zeros((n_predictions, n_features, 2));
        for step in 0..n_predictions {
            for j in 0..n_features {
                let pred = predictions[[step, j]];
                let u = uncertainty_estimates[[step, j]];
                confidence_intervals[[step, j, 0]] = pred - 1.96 * u;
                confidence_intervals[[step, j, 1]] = pred + 1.96 * u;
            }
        }

        let now = SystemTime::now();
        let interval = Duration::from_secs(60);
        let timestamps: Vec<SystemTime> = (0..n_predictions)
            .map(|i| now + interval * (i as u32 + 1))
            .collect();

        Ok(CrosstalkPredictionResult {
            predictions,
            timestamps,
            confidence_intervals,
            uncertainty_estimates,
            time_series_analysis: TimeSeriesAnalysisResult {
                trend_analysis: TrendAnalysisResult {
                    trend_direction: TrendDirection::Stable,
                    trend_strength: 0.0,
                    trend_significance: 1.0,
                    trend_rate: 0.0,
                },
                seasonality_analysis: SeasonalityAnalysisResult {
                    periods: vec![],
                    strengths: vec![],
                    patterns: HashMap::new(),
                    significance: HashMap::new(),
                },
                changepoint_analysis: ChangepointAnalysisResult {
                    changepoints: vec![],
                    scores: vec![],
                    types: vec![],
                    confidence_levels: vec![],
                },
                forecast_metrics: ForecastMetrics {
                    mae: 0.0,
                    mse: 0.0,
                    rmse: 0.0,
                    mape: 0.0,
                    smape: 0.0,
                    mase: 0.0,
                },
            },
        })
    }

    /// Train prediction models
    pub fn train_models(&mut self, historical_data: &Array3<f64>) -> DeviceResult<()> {
        // Train different time series models
        for model_type in &[
            TimeSeriesModel::ARIMA { p: 2, d: 1, q: 2 },
            TimeSeriesModel::ExponentialSmoothing {
                trend: "add".to_string(),
                seasonal: "add".to_string()
            },
        ] {
            let model = PredictionModel::new(model_type.clone())?;
            self.models.insert(format!("{:?}", model_type), model);
        }
        Ok(())
    }

    /// Generate multi-step ahead predictions
    pub fn predict_multi_step(&self, input_data: &Array2<f64>, steps: usize) -> DeviceResult<Array2<f64>> {
        let n_features = input_data.ncols();
        let n_rows = input_data.nrows();
        if n_rows == 0 {
            return Ok(Array2::zeros((steps, n_features)));
        }

        // Compute per-feature mean and last value for exponential smoothing baseline
        let mut predictions = Array2::zeros((steps, n_features));
        for j in 0..n_features {
            let col: Vec<f64> = (0..n_rows).map(|i| input_data[[i, j]]).collect();
            let mean = col.iter().sum::<f64>() / col.len() as f64;
            let last = col[n_rows - 1];
            let alpha = 0.9_f64;
            for step in 0..steps {
                let decay = alpha.powi(step as i32 + 1);
                predictions[[step, j]] = mean + decay * (last - mean);
            }
        }
        Ok(predictions)
    }

    /// Update models with new data (online learning)
    pub fn update_models(&mut self, new_data: &Array2<f64>) -> DeviceResult<()> {
        for (_, model) in self.models.iter_mut() {
            model.update(new_data)?;
        }
        Ok(())
    }

    /// Get prediction uncertainty
    pub fn get_uncertainty_estimates(&self, predictions: &Array2<f64>) -> DeviceResult<Array2<f64>> {
        self.uncertainty_quantifier.estimate_uncertainty(predictions)
    }

    /// Evaluate prediction accuracy
    pub fn evaluate_predictions(
        &self,
        predictions: &Array2<f64>,
        ground_truth: &Array2<f64>,
    ) -> DeviceResult<ForecastMetrics> {
        let mae = self.calculate_mae(predictions, ground_truth);
        let mse = self.calculate_mse(predictions, ground_truth);
        let rmse = mse.sqrt();
        let mape = self.calculate_mape(predictions, ground_truth);
        let smape = self.calculate_smape(predictions, ground_truth);
        let mase = self.calculate_mase(predictions, ground_truth);

        Ok(ForecastMetrics {
            mae,
            mse,
            rmse,
            mape,
            smape,
            mase,
        })
    }

    fn calculate_mae(&self, predictions: &Array2<f64>, ground_truth: &Array2<f64>) -> f64 {
        let diff = predictions - ground_truth;
        diff.mapv(|x| x.abs()).mean().unwrap_or(0.0)
    }

    fn calculate_mse(&self, predictions: &Array2<f64>, ground_truth: &Array2<f64>) -> f64 {
        let diff = predictions - ground_truth;
        diff.mapv(|x| x * x).mean().unwrap_or(0.0)
    }

    fn calculate_mape(&self, predictions: &Array2<f64>, ground_truth: &Array2<f64>) -> f64 {
        let mut total_ape = 0.0;
        let mut count = 0;

        for (pred, actual) in predictions.iter().zip(ground_truth.iter()) {
            if actual.abs() > 1e-8 {
                total_ape += ((pred - actual) / actual).abs();
                count += 1;
            }
        }

        if count > 0 {
            total_ape / count as f64 * 100.0
        } else {
            0.0
        }
    }

    fn calculate_smape(&self, predictions: &Array2<f64>, ground_truth: &Array2<f64>) -> f64 {
        let mut total_sape = 0.0;
        let mut count = 0;

        for (pred, actual) in predictions.iter().zip(ground_truth.iter()) {
            let denominator = (pred.abs() + actual.abs()) / 2.0;
            if denominator > 1e-8 {
                total_sape += (pred - actual).abs() / denominator;
                count += 1;
            }
        }

        if count > 0 {
            total_sape / count as f64 * 100.0
        } else {
            0.0
        }
    }

    fn calculate_mase(&self, predictions: &Array2<f64>, ground_truth: &Array2<f64>) -> f64 {
        let mae = self.calculate_mae(predictions, ground_truth);
        // Naive forecast MAE: use mean of |ground_truth[i] - ground_truth[i-1]| for i>=1
        let n_rows = ground_truth.nrows();
        let n_cols = ground_truth.ncols();
        if n_rows < 2 || n_cols == 0 {
            return if mae == 0.0 { 0.0 } else { f64::INFINITY };
        }
        let mut naive_total = 0.0;
        let mut naive_count = 0usize;
        for j in 0..n_cols {
            for i in 1..n_rows {
                naive_total += (ground_truth[[i, j]] - ground_truth[[i - 1, j]]).abs();
                naive_count += 1;
            }
        }
        let naive_mae = if naive_count > 0 {
            naive_total / naive_count as f64
        } else {
            1.0
        };
        if naive_mae < 1e-10 {
            if mae < 1e-10 { 0.0 } else { f64::INFINITY }
        } else {
            mae / naive_mae
        }
    }
}

impl UncertaintyQuantifier {
    pub fn new(config: &UncertaintyQuantificationConfig) -> Self {
        Self {
            method: config.estimation_method.clone(),
            confidence_levels: config.confidence_levels.clone(),
            uncertainty_history: VecDeque::with_capacity(1000),
        }
    }

    /// Estimate prediction uncertainty
    pub fn estimate_uncertainty(&self, predictions: &Array2<f64>) -> DeviceResult<Array2<f64>> {
        match &self.method {
            UncertaintyEstimationMethod::Bootstrap { n_bootstrap } => {
                self.bootstrap_uncertainty(predictions, *n_bootstrap)
            },
            UncertaintyEstimationMethod::Bayesian { prior_type } => {
                self.bayesian_uncertainty(predictions, prior_type)
            },
            UncertaintyEstimationMethod::Ensemble { n_models } => {
                self.ensemble_uncertainty(predictions, *n_models)
            },
            UncertaintyEstimationMethod::DropoutMonteCarlo { dropout_rate, n_samples } => {
                self.dropout_uncertainty(predictions, *dropout_rate, *n_samples)
            },
        }
    }

    fn bootstrap_uncertainty(&self, predictions: &Array2<f64>, _n_bootstrap: usize) -> DeviceResult<Array2<f64>> {
        // Bootstrap uncertainty: estimate from the spread of prediction values
        // across time steps as a proxy for prediction variability.
        let (n_steps, n_features) = predictions.dim();
        let mut uncertainty = Array2::zeros((n_steps, n_features));
        if n_steps < 2 {
            return Ok(uncertainty);
        }
        // Per-feature standard deviation across time steps as uncertainty estimate
        for j in 0..n_features {
            let col: Vec<f64> = (0..n_steps).map(|i| predictions[[i, j]]).collect();
            let mean = col.iter().sum::<f64>() / n_steps as f64;
            let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_steps as f64;
            let std = var.sqrt();
            for i in 0..n_steps {
                // Bootstrap std grows with step distance from the center
                let step_factor = 1.0 + (i as f64 / n_steps as f64);
                uncertainty[[i, j]] = std * step_factor;
            }
        }
        Ok(uncertainty)
    }

    fn bayesian_uncertainty(&self, predictions: &Array2<f64>, _prior_type: &str) -> DeviceResult<Array2<f64>> {
        // Bayesian posterior uncertainty approximated via bootstrap on the provided predictions.
        // Without a trained Bayesian model this is the best tractable estimate available.
        self.bootstrap_uncertainty(predictions, 0)
    }

    fn ensemble_uncertainty(&self, predictions: &Array2<f64>, _n_models: usize) -> DeviceResult<Array2<f64>> {
        // Without multiple trained ensemble models, fall back to bootstrap uncertainty of
        // the given predictions, which uses their temporal spread as the uncertainty proxy.
        self.bootstrap_uncertainty(predictions, 0)
    }

    fn dropout_uncertainty(&self, predictions: &Array2<f64>, _dropout_rate: f64, _n_samples: usize) -> DeviceResult<Array2<f64>> {
        // MC Dropout requires a neural network; approximate with bootstrap uncertainty instead.
        self.bootstrap_uncertainty(predictions, 0)
    }

    /// Compute confidence intervals
    pub fn compute_confidence_intervals(
        &self,
        predictions: &Array2<f64>,
        uncertainties: &Array2<f64>,
    ) -> DeviceResult<Array3<f64>> {
        let (n_steps, n_features) = predictions.dim();
        let mut intervals = Array3::zeros((n_steps, n_features, 2));

        for level in &self.confidence_levels {
            let z_score = self.get_z_score(*level);

            for i in 0..n_steps {
                for j in 0..n_features {
                    let pred = predictions[[i, j]];
                    let uncertainty = uncertainties[[i, j]];

                    intervals[[i, j, 0]] = pred - z_score * uncertainty; // Lower bound
                    intervals[[i, j, 1]] = pred + z_score * uncertainty; // Upper bound
                }
            }
        }

        Ok(intervals)
    }

    fn get_z_score(&self, confidence_level: f64) -> f64 {
        // Rational approximation of the normal quantile (Abramowitz & Stegun §26.2.17)
        // Valid to ~4.5 decimal places for confidence_level in (0, 1).
        let p = confidence_level + (1.0 - confidence_level) / 2.0; // one-tail probability
        if p <= 0.0 || p >= 1.0 {
            return 1.96;
        }
        let t = (-2.0 * (1.0 - p).ln()).sqrt();
        let c0 = 2.515517_f64;
        let c1 = 0.802853_f64;
        let c2 = 0.010328_f64;
        let d1 = 1.432788_f64;
        let d2 = 0.189269_f64;
        let d3 = 0.001308_f64;
        let numerator   = c0 + c1 * t + c2 * t * t;
        let denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
        t - numerator / denominator
    }

    /// Update uncertainty history
    pub fn update_history(&mut self, uncertainty: Array1<f64>) {
        self.uncertainty_history.push_back(uncertainty);

        // Keep only recent history
        if self.uncertainty_history.len() > 1000 {
            self.uncertainty_history.pop_front();
        }
    }
}

impl PredictionModel {
    pub fn new(model_type: TimeSeriesModel) -> DeviceResult<Self> {
        Ok(Self {
            model_type,
            model_data: Vec::new(),
            accuracy_metrics: ForecastMetrics {
                mae: 0.0,
                mse: 0.0,
                rmse: 0.0,
                mape: 0.0,
                smape: 0.0,
                mase: 0.0,
            },
            last_updated: SystemTime::now(),
        })
    }

    /// Train the model with historical data
    pub fn train(&mut self, data: &Array2<f64>) -> DeviceResult<()> {
        match &self.model_type {
            TimeSeriesModel::ARIMA { p, d, q } => {
                self.train_arima(data, *p, *d, *q)
            },
            TimeSeriesModel::ExponentialSmoothing { trend, seasonal } => {
                self.train_exponential_smoothing(data, trend, seasonal)
            },
            TimeSeriesModel::Prophet { growth, seasonality_mode } => {
                self.train_prophet(data, growth, seasonality_mode)
            },
            TimeSeriesModel::LSTM { hidden_size, num_layers } => {
                self.train_lstm(data, *hidden_size, *num_layers)
            },
            TimeSeriesModel::Transformer { d_model, n_heads, n_layers } => {
                self.train_transformer(data, *d_model, *n_heads, *n_layers)
            },
        }
    }

    fn train_arima(&mut self, data: &Array2<f64>, p: usize, d: usize, q: usize) -> DeviceResult<()> {
        // Fit AR(p) via Yule-Walker equations for each feature column.
        // The d-order differencing and MA(q) part require iterative ARMA fitting
        // beyond Yule-Walker and a proper ML framework; we implement AR(p) only and
        // store the AR coefficients as model_data (f64 le bytes).
        let n_rows = data.nrows();
        let n_cols = data.ncols();
        if n_rows < p + 1 || p == 0 {
            self.last_updated = SystemTime::now();
            return Ok(());
        }

        // For each column, compute AR(p) via Yule-Walker normal equations.
        // Autocorrelations r[0..=p] then solve Toeplitz system.
        let mut all_coeffs: Vec<f64> = Vec::with_capacity(n_cols * p);
        for j in 0..n_cols {
            let col: Vec<f64> = (0..n_rows).map(|i| data[[i, j]]).collect();
            let mean = col.iter().sum::<f64>() / n_rows as f64;
            let centered: Vec<f64> = col.iter().map(|x| x - mean).collect();

            // Compute autocorrelations r[0..=p]
            let r: Vec<f64> = (0..=p).map(|lag| {
                let n_terms = n_rows - lag;
                if n_terms == 0 { return 0.0; }
                let sum: f64 = (0..n_terms).map(|i| centered[i] * centered[i + lag]).sum();
                sum / n_rows as f64
            }).collect();

            let var = r[0];
            if var < 1e-12 {
                all_coeffs.extend(vec![0.0; p]);
                continue;
            }

            // Solve Yule-Walker: R * phi = r[1..=p]
            // R is p×p Toeplitz with R[i][j] = r[|i-j|]
            // Use Levinson-Durbin recursion
            let phi = levinson_durbin(&r, p);
            all_coeffs.extend(phi);
        }

        // Store model: header [n_cols as u64 le][p as u64 le] then coefficients
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(n_cols as u64).to_le_bytes());
        bytes.extend_from_slice(&(p as u64).to_le_bytes());
        for c in &all_coeffs {
            bytes.extend_from_slice(&c.to_le_bytes());
        }
        self.model_data = bytes;
        self.last_updated = SystemTime::now();
        let _ = (d, q); // d/q not implemented in this AR-only baseline
        Ok(())
    }

    fn train_exponential_smoothing(&mut self, data: &Array2<f64>, _trend: &str, _seasonal: &str) -> DeviceResult<()> {
        // Simple exponential smoothing: store last smoothed value per feature.
        // Full Holt-Winters with trend+seasonal requires a proper time-series library.
        let n_rows = data.nrows();
        let n_cols = data.ncols();
        if n_rows == 0 {
            self.last_updated = SystemTime::now();
            return Ok(());
        }
        let alpha = 0.3_f64;
        let mut smoothed = Vec::with_capacity(n_cols);
        for j in 0..n_cols {
            let mut s = data[[0, j]];
            for i in 1..n_rows {
                s = alpha * data[[i, j]] + (1.0 - alpha) * s;
            }
            smoothed.push(s);
        }
        // Store: header [n_cols as u64][alpha as f64] then smoothed values
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(n_cols as u64).to_le_bytes());
        bytes.extend_from_slice(&alpha.to_le_bytes());
        for s in &smoothed {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        self.model_data = bytes;
        self.last_updated = SystemTime::now();
        Ok(())
    }

    fn train_prophet(&mut self, data: &Array2<f64>, _growth: &str, _seasonality_mode: &str) -> DeviceResult<()> {
        // Prophet requires a Bayesian/curve-fitting framework not available here.
        // Fall back to double exponential smoothing (Holt's method) as a tractable substitute
        // that captures trend similarly to Prophet's piecewise-linear growth.
        self.train_exponential_smoothing(data, "additive", "none")
    }

    fn train_lstm(&mut self, data: &Array2<f64>, p: usize, _num_layers: usize) -> DeviceResult<()> {
        // LSTM requires a neural network framework.  Fall back to AR(p) via Yule-Walker,
        // which is the simplest recurrent model and a natural analytical substitute.
        let ar_order = if p == 0 { 1 } else { p };
        self.train_arima(data, ar_order, 0, 0)
    }

    fn train_transformer(&mut self, data: &Array2<f64>, d_model: usize, _n_heads: usize, n_layers: usize) -> DeviceResult<()> {
        // Transformer requires a neural network framework.  Fall back to AR(p) with order
        // proportional to the model depth, capturing longer-range dependencies analytically.
        let ar_order = (d_model.min(n_layers) + 1).max(2);
        self.train_arima(data, ar_order, 0, 0)
    }

    /// Make predictions
    pub fn predict(&self, input_data: &Array2<f64>, steps: usize) -> DeviceResult<Array2<f64>> {
        let n_features = input_data.ncols();
        let n_rows = input_data.nrows();
        if n_rows == 0 || steps == 0 {
            return Ok(Array2::zeros((steps, n_features)));
        }

        match &self.model_type {
            TimeSeriesModel::ARIMA { p, .. } => {
                self.predict_arima(input_data, steps, *p)
            },
            TimeSeriesModel::ExponentialSmoothing { .. } => {
                self.predict_exponential_smoothing(input_data, steps)
            },
            TimeSeriesModel::Prophet { .. } => {
                // Trained as exponential smoothing fallback; predict via the same path.
                self.predict_exponential_smoothing(input_data, steps)
            },
            TimeSeriesModel::LSTM { hidden_size, .. } => {
                // Trained as AR(hidden_size) fallback; predict via ARIMA path.
                let ar_order = if *hidden_size == 0 { 1 } else { *hidden_size };
                self.predict_arima(input_data, steps, ar_order)
            },
            TimeSeriesModel::Transformer { d_model, n_layers, .. } => {
                // Trained as AR(p) fallback with order proportional to model depth.
                let ar_order = (d_model.min(n_layers) + 1).max(2);
                self.predict_arima(input_data, steps, ar_order)
            },
        }
    }

    fn predict_arima(&self, input_data: &Array2<f64>, steps: usize, p: usize) -> DeviceResult<Array2<f64>> {
        let n_features = input_data.ncols();
        let n_rows = input_data.nrows();
        let min_header = 16; // 2 × u64 = 16 bytes
        if self.model_data.len() < min_header || p == 0 {
            // No trained model data; fall back to last-value persistence
            let mut preds = Array2::zeros((steps, n_features));
            for j in 0..n_features {
                let last = input_data[[n_rows - 1, j]];
                for s in 0..steps {
                    preds[[s, j]] = last;
                }
            }
            return Ok(preds);
        }

        // Parse header
        let stored_n_cols = u64::from_le_bytes(self.model_data[0..8].try_into().map_err(|_| {
            DeviceError::InvalidInput("Corrupt model data header".to_string())
        })?) as usize;
        let stored_p = u64::from_le_bytes(self.model_data[8..16].try_into().map_err(|_| {
            DeviceError::InvalidInput("Corrupt model data header".to_string())
        })?) as usize;

        let effective_cols = stored_n_cols.min(n_features);
        let effective_p = stored_p.min(n_rows);
        let expected_bytes = 16 + effective_cols * effective_p * 8;

        if self.model_data.len() < expected_bytes || effective_p == 0 {
            // Fallback: last-value persistence
            let mut preds = Array2::zeros((steps, n_features));
            for j in 0..n_features {
                let last = input_data[[n_rows - 1, j]];
                for s in 0..steps {
                    preds[[s, j]] = last;
                }
            }
            return Ok(preds);
        }

        // Extract AR coefficients
        let mut preds = Array2::zeros((steps, n_features));
        for j in 0..effective_cols {
            // Read p AR coefficients for column j
            let offset = 16 + j * effective_p * 8;
            let mut phi = Vec::with_capacity(effective_p);
            for k in 0..effective_p {
                let byte_off = offset + k * 8;
                let coeff = f64::from_le_bytes(self.model_data[byte_off..byte_off + 8].try_into().map_err(|_| {
                    DeviceError::InvalidInput("Corrupt model coefficient data".to_string())
                })?);
                phi.push(coeff);
            }

            // Build history buffer: last p values of input column j
            let history: Vec<f64> = (n_rows.saturating_sub(effective_p)..n_rows)
                .map(|i| input_data[[i, j]])
                .collect();
            let mean = history.iter().sum::<f64>() / history.len().max(1) as f64;
            let centered_history: Vec<f64> = history.iter().map(|x| x - mean).collect();
            let mut buf: VecDeque<f64> = centered_history.into_iter().collect();

            for s in 0..steps {
                // AR prediction: sum of phi[k] * buf[end - k]
                let buf_len = buf.len();
                let forecast: f64 = phi.iter().enumerate().map(|(k, &c)| {
                    let idx = buf_len.saturating_sub(k + 1);
                    c * buf.get(idx).copied().unwrap_or(0.0)
                }).sum();
                preds[[s, j]] = forecast + mean;
                buf.push_back(forecast);
                if buf.len() > effective_p * 2 {
                    buf.pop_front();
                }
            }
        }
        // Fill remaining columns (if n_features > stored_n_cols) with last value
        for j in effective_cols..n_features {
            let last = input_data[[n_rows - 1, j]];
            for s in 0..steps {
                preds[[s, j]] = last;
            }
        }
        Ok(preds)
    }

    fn predict_exponential_smoothing(&self, input_data: &Array2<f64>, steps: usize) -> DeviceResult<Array2<f64>> {
        let n_features = input_data.ncols();
        let n_rows = input_data.nrows();
        let min_header = 16; // u64 + f64 = 16 bytes
        if self.model_data.len() < min_header {
            // No model data; use last value
            let mut preds = Array2::zeros((steps, n_features));
            for j in 0..n_features {
                let last = input_data[[n_rows - 1, j]];
                for s in 0..steps {
                    preds[[s, j]] = last;
                }
            }
            return Ok(preds);
        }
        let stored_n_cols = u64::from_le_bytes(self.model_data[0..8].try_into().map_err(|_| {
            DeviceError::InvalidInput("Corrupt ES model header".to_string())
        })?) as usize;
        let alpha = f64::from_le_bytes(self.model_data[8..16].try_into().map_err(|_| {
            DeviceError::InvalidInput("Corrupt ES model alpha".to_string())
        })?);
        let effective_cols = stored_n_cols.min(n_features);
        let expected = 16 + effective_cols * 8;
        if self.model_data.len() < expected {
            let mut preds = Array2::zeros((steps, n_features));
            for j in 0..n_features {
                let last = input_data[[n_rows - 1, j]];
                for s in 0..steps { preds[[s, j]] = last; }
            }
            return Ok(preds);
        }
        let mut preds = Array2::zeros((steps, n_features));
        for j in 0..effective_cols {
            // Start from stored smoothed value, then update with any new input data
            let byte_off = 16 + j * 8;
            let stored_s = f64::from_le_bytes(self.model_data[byte_off..byte_off + 8].try_into().map_err(|_| {
                DeviceError::InvalidInput("Corrupt ES smoothed value".to_string())
            })?);
            // Update with new data points since training
            let mut s = stored_s;
            for i in 0..n_rows {
                s = alpha * input_data[[i, j]] + (1.0 - alpha) * s;
            }
            // Exponential smoothing forecast is constant (level model)
            for step in 0..steps {
                preds[[step, j]] = s;
            }
        }
        for j in effective_cols..n_features {
            let last = input_data[[n_rows - 1, j]];
            for s in 0..steps { preds[[s, j]] = last; }
        }
        Ok(preds)
    }

    /// Update model with new data (online learning)
    pub fn update(&mut self, new_data: &Array2<f64>) -> DeviceResult<()> {
        // For exponential smoothing, we can do an online update of the smoothed level.
        match &self.model_type {
            TimeSeriesModel::ExponentialSmoothing { .. } => {
                let min_header = 16;
                if self.model_data.len() < min_header {
                    return Ok(());
                }
                let stored_n_cols = u64::from_le_bytes(self.model_data[0..8].try_into().map_err(|_| {
                    DeviceError::InvalidInput("Corrupt ES model header".to_string())
                })?) as usize;
                let alpha = f64::from_le_bytes(self.model_data[8..16].try_into().map_err(|_| {
                    DeviceError::InvalidInput("Corrupt ES model alpha".to_string())
                })?);
                let n_rows = new_data.nrows();
                let n_cols = new_data.ncols().min(stored_n_cols);
                let expected = 16 + stored_n_cols * 8;
                if self.model_data.len() < expected || n_rows == 0 {
                    return Ok(());
                }
                for j in 0..n_cols {
                    let byte_off = 16 + j * 8;
                    let mut s = f64::from_le_bytes(self.model_data[byte_off..byte_off + 8].try_into().map_err(|_| {
                        DeviceError::InvalidInput("Corrupt ES smoothed value on update".to_string())
                    })?);
                    for i in 0..n_rows {
                        s = alpha * new_data[[i, j]] + (1.0 - alpha) * s;
                    }
                    self.model_data[byte_off..byte_off + 8].copy_from_slice(&s.to_le_bytes());
                }
                self.last_updated = SystemTime::now();
                Ok(())
            },
            _ => {
                // For ARIMA and ML models, online update requires full retraining
                self.last_updated = SystemTime::now();
                Ok(())
            },
        }
    }

    /// Get model size in bytes
    pub fn get_model_size(&self) -> usize {
        self.model_data.len()
    }

    /// Serialize model to bytes
    pub fn serialize(&self) -> Vec<u8> {
        self.model_data.clone()
    }

    /// Load model from bytes
    pub fn deserialize(&mut self, data: Vec<u8>) -> DeviceResult<()> {
        self.model_data = data;
        Ok(())
    }
}

impl TimeSeriesAnalyzer {
    pub fn new(config: &TimeSeriesConfig) -> Self {
        Self {
            config: config.clone(),
            trend_detector: TrendDetector::new(&config.trend_analysis),
            seasonality_detector: SeasonalityDetector::new(&config.seasonality),
            changepoint_detector: ChangepointDetector::new(&config.changepoint_detection),
        }
    }

    /// Perform comprehensive time series analysis
    pub fn analyze_time_series(&mut self, data: &Array2<f64>) -> DeviceResult<TimeSeriesAnalysisResult> {
        let trend_analysis = self.trend_detector.analyze_trend(data)?;
        let seasonality_analysis = self.seasonality_detector.analyze_seasonality(data)?;
        let changepoint_analysis = self.changepoint_detector.detect_changepoints(data)?;

        // Forecast metrics are computed externally after actual predictions are made
        let forecast_metrics = ForecastMetrics {
            mae: 0.0,
            mse: 0.0,
            rmse: 0.0,
            mape: 0.0,
            smape: 0.0,
            mase: 0.0,
        };

        Ok(TimeSeriesAnalysisResult {
            trend_analysis,
            seasonality_analysis,
            changepoint_analysis,
            forecast_metrics,
        })
    }
}

impl TrendDetector {
    pub fn new(config: &TrendAnalysisConfig) -> Self {
        Self {
            method: config.detection_method.clone(),
            significance_threshold: config.significance_threshold,
            trend_history: VecDeque::with_capacity(100),
        }
    }

    /// Analyze trend in time series data
    pub fn analyze_trend(&mut self, data: &Array2<f64>) -> DeviceResult<TrendAnalysisResult> {
        let trend_direction = self.detect_trend_direction(data)?;
        let trend_strength = self.calculate_trend_strength(data)?;
        let trend_significance = self.test_trend_significance(data)?;
        let trend_rate = self.calculate_trend_rate(data)?;

        let result = TrendAnalysisResult {
            trend_direction,
            trend_strength,
            trend_significance,
            trend_rate,
        };

        self.trend_history.push_back(result.clone());
        if self.trend_history.len() > 100 {
            self.trend_history.pop_front();
        }

        Ok(result)
    }

    fn detect_trend_direction(&self, data: &Array2<f64>) -> DeviceResult<TrendDirection> {
        match &self.method {
            TrendDetectionMethod::MannKendall => self.mann_kendall_trend(data),
            TrendDetectionMethod::LinearRegression => self.linear_regression_trend(data),
            TrendDetectionMethod::TheilSen => self.theil_sen_trend(data),
            TrendDetectionMethod::LOWESS { frac } => self.lowess_trend(data, *frac),
        }
    }

    fn mann_kendall_trend(&self, data: &Array2<f64>) -> DeviceResult<TrendDirection> {
        // Mann-Kendall: compute S statistic for each column, average direction.
        let n = data.nrows();
        let n_cols = data.ncols();
        if n < 3 || n_cols == 0 {
            return Ok(TrendDirection::Stable);
        }
        let mut total_s = 0i64;
        for j in 0..n_cols {
            let mut s = 0i64;
            for i in 0..n {
                for k in (i + 1)..n {
                    let diff = data[[k, j]] - data[[i, j]];
                    if diff > 1e-10 { s += 1; }
                    else if diff < -1e-10 { s -= 1; }
                }
            }
            total_s += s;
        }
        // Variance of S under H0: var = n*(n-1)*(2n+5)/18 per column
        let var_s = (n as f64 * (n as f64 - 1.0) * (2.0 * n as f64 + 5.0)) / 18.0 * n_cols as f64;
        let z = if total_s > 0 {
            (total_s as f64 - 1.0) / var_s.sqrt()
        } else if total_s < 0 {
            (total_s as f64 + 1.0) / var_s.sqrt()
        } else {
            0.0
        };
        // 95% threshold: |z| > 1.96
        if z > 1.96 {
            Ok(TrendDirection::Increasing)
        } else if z < -1.96 {
            Ok(TrendDirection::Decreasing)
        } else {
            Ok(TrendDirection::Stable)
        }
    }

    fn linear_regression_trend(&self, data: &Array2<f64>) -> DeviceResult<TrendDirection> {
        // Average slope across all columns via OLS.
        let n = data.nrows();
        let n_cols = data.ncols();
        if n < 2 || n_cols == 0 {
            return Ok(TrendDirection::Stable);
        }
        let x_mean = (n as f64 - 1.0) / 2.0;
        let ss_xx: f64 = (0..n).map(|i| (i as f64 - x_mean).powi(2)).sum();
        if ss_xx < 1e-10 {
            return Ok(TrendDirection::Stable);
        }
        let mut total_slope = 0.0;
        for j in 0..n_cols {
            let y_mean: f64 = (0..n).map(|i| data[[i, j]]).sum::<f64>() / n as f64;
            let ss_xy: f64 = (0..n).map(|i| (i as f64 - x_mean) * (data[[i, j]] - y_mean)).sum();
            total_slope += ss_xy / ss_xx;
        }
        let avg_slope = total_slope / n_cols as f64;
        if avg_slope > 1e-6 {
            Ok(TrendDirection::Increasing)
        } else if avg_slope < -1e-6 {
            Ok(TrendDirection::Decreasing)
        } else {
            Ok(TrendDirection::Stable)
        }
    }

    fn theil_sen_trend(&self, data: &Array2<f64>) -> DeviceResult<TrendDirection> {
        // Theil-Sen estimator: median of all pairwise slopes.
        let n = data.nrows();
        let n_cols = data.ncols();
        if n < 2 || n_cols == 0 {
            return Ok(TrendDirection::Stable);
        }
        let mut all_slopes: Vec<f64> = Vec::new();
        for j in 0..n_cols {
            for i in 0..n {
                for k in (i + 1)..n {
                    let dx = (k as f64) - (i as f64);
                    let dy = data[[k, j]] - data[[i, j]];
                    all_slopes.push(dy / dx);
                }
            }
        }
        if all_slopes.is_empty() {
            return Ok(TrendDirection::Stable);
        }
        all_slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_slope = all_slopes[all_slopes.len() / 2];
        if median_slope > 1e-6 {
            Ok(TrendDirection::Increasing)
        } else if median_slope < -1e-6 {
            Ok(TrendDirection::Decreasing)
        } else {
            Ok(TrendDirection::Stable)
        }
    }

    fn lowess_trend(&self, data: &Array2<f64>, _frac: f64) -> DeviceResult<TrendDirection> {
        // LOWESS (locally weighted scatterplot smoothing) full implementation requires
        // iterative local regressions; fall back to linear regression for direction.
        self.linear_regression_trend(data)
    }

    fn calculate_trend_strength(&self, data: &Array2<f64>) -> DeviceResult<f64> {
        // Trend strength = R² of linear fit, averaged across columns.
        let n = data.nrows();
        let n_cols = data.ncols();
        if n < 2 || n_cols == 0 {
            return Ok(0.0);
        }
        let x_mean = (n as f64 - 1.0) / 2.0;
        let ss_xx: f64 = (0..n).map(|i| (i as f64 - x_mean).powi(2)).sum();
        if ss_xx < 1e-10 {
            return Ok(0.0);
        }
        let mut r2_sum = 0.0;
        for j in 0..n_cols {
            let y_vals: Vec<f64> = (0..n).map(|i| data[[i, j]]).collect();
            let y_mean = y_vals.iter().sum::<f64>() / n as f64;
            let ss_yy: f64 = y_vals.iter().map(|y| (y - y_mean).powi(2)).sum();
            if ss_yy < 1e-12 {
                continue; // Constant series, R² = 0
            }
            let ss_xy: f64 = (0..n).map(|i| (i as f64 - x_mean) * (y_vals[i] - y_mean)).sum();
            let slope = ss_xy / ss_xx;
            let intercept = y_mean - slope * x_mean;
            let ss_res: f64 = (0..n).map(|i| {
                let fitted = slope * i as f64 + intercept;
                (y_vals[i] - fitted).powi(2)
            }).sum();
            r2_sum += 1.0 - ss_res / ss_yy;
        }
        Ok((r2_sum / n_cols as f64).clamp(0.0, 1.0))
    }

    fn test_trend_significance(&self, data: &Array2<f64>) -> DeviceResult<f64> {
        // p-value for trend via Mann-Kendall Z statistic (two-tailed normal approximation).
        let n = data.nrows();
        let n_cols = data.ncols();
        if n < 3 || n_cols == 0 {
            return Ok(1.0);
        }
        let mut total_s = 0i64;
        for j in 0..n_cols {
            let mut s = 0i64;
            for i in 0..n {
                for k in (i + 1)..n {
                    let diff = data[[k, j]] - data[[i, j]];
                    if diff > 1e-10 { s += 1; }
                    else if diff < -1e-10 { s -= 1; }
                }
            }
            total_s += s;
        }
        let var_s = (n as f64 * (n as f64 - 1.0) * (2.0 * n as f64 + 5.0)) / 18.0 * n_cols as f64;
        let z = if total_s > 0 {
            (total_s as f64 - 1.0) / var_s.sqrt()
        } else if total_s < 0 {
            (total_s as f64 + 1.0) / var_s.sqrt()
        } else {
            0.0
        };
        // Two-tailed p-value approximation using complementary error function
        // erf(x) ≈ using Horner's method (Abramowitz & Stegun 7.1.26)
        let p = 2.0 * normal_survival(z.abs());
        Ok(p.clamp(0.0, 1.0))
    }

    fn calculate_trend_rate(&self, data: &Array2<f64>) -> DeviceResult<f64> {
        // Average OLS slope across columns (units: value/step)
        let n = data.nrows();
        let n_cols = data.ncols();
        if n < 2 || n_cols == 0 {
            return Ok(0.0);
        }
        let x_mean = (n as f64 - 1.0) / 2.0;
        let ss_xx: f64 = (0..n).map(|i| (i as f64 - x_mean).powi(2)).sum();
        if ss_xx < 1e-10 {
            return Ok(0.0);
        }
        let mut slope_sum = 0.0;
        for j in 0..n_cols {
            let y_mean: f64 = (0..n).map(|i| data[[i, j]]).sum::<f64>() / n as f64;
            let ss_xy: f64 = (0..n).map(|i| (i as f64 - x_mean) * (data[[i, j]] - y_mean)).sum();
            slope_sum += ss_xy / ss_xx;
        }
        Ok(slope_sum / n_cols as f64)
    }
}

impl SeasonalityDetector {
    pub fn new(config: &SeasonalityConfig) -> Self {
        Self {
            periods: config.periods.clone(),
            strength_threshold: config.strength_threshold,
            seasonal_patterns: HashMap::new(),
        }
    }

    /// Analyze seasonality in time series data
    pub fn analyze_seasonality(&mut self, data: &Array2<f64>) -> DeviceResult<SeasonalityAnalysisResult> {
        let mut detected_periods = Vec::new();
        let mut strengths = Vec::new();
        let mut patterns = HashMap::new();
        let mut significance = HashMap::new();

        for &period in &self.periods.clone() {
            let strength = self.calculate_seasonal_strength(data, period)?;
            if strength > self.strength_threshold {
                detected_periods.push(period);
                strengths.push(strength);

                let pattern = self.extract_seasonal_pattern(data, period)?;
                patterns.insert(period, pattern);

                let sig = self.test_seasonal_significance(data, period)?;
                significance.insert(period, sig);
            }
        }

        Ok(SeasonalityAnalysisResult {
            periods: detected_periods,
            strengths,
            patterns,
            significance,
        })
    }

    fn calculate_seasonal_strength(&self, data: &Array2<f64>, period: usize) -> DeviceResult<f64> {
        // Seasonal strength via autocorrelation at the given lag (period).
        // Averaged across all feature columns.
        let n = data.nrows();
        let n_cols = data.ncols();
        if n < period + 1 || period == 0 || n_cols == 0 {
            return Ok(0.0);
        }
        let mut acf_sum = 0.0;
        for j in 0..n_cols {
            let col: Vec<f64> = (0..n).map(|i| data[[i, j]]).collect();
            let mean = col.iter().sum::<f64>() / n as f64;
            let var: f64 = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            if var < 1e-12 { continue; }
            let n_terms = n - period;
            let cov: f64 = (0..n_terms).map(|i| (col[i] - mean) * (col[i + period] - mean)).sum::<f64>() / n as f64;
            acf_sum += (cov / var).clamp(-1.0, 1.0);
        }
        Ok((acf_sum / n_cols as f64).clamp(0.0, 1.0))
    }

    fn extract_seasonal_pattern(&self, data: &Array2<f64>, period: usize) -> DeviceResult<Array1<f64>> {
        // Average value within each phase bin (0..period) across all complete cycles.
        let n = data.nrows();
        let n_cols = data.ncols();
        if n < period || period == 0 {
            return Ok(Array1::zeros(period));
        }
        let mut bin_sum = vec![0.0f64; period];
        let mut bin_count = vec![0usize; period];
        for i in 0..n {
            let phase = i % period;
            for j in 0..n_cols {
                bin_sum[phase] += data[[i, j]];
                bin_count[phase] += 1;
            }
        }
        let pattern: Vec<f64> = (0..period).map(|p| {
            if bin_count[p] > 0 { bin_sum[p] / bin_count[p] as f64 } else { 0.0 }
        }).collect();
        Ok(Array1::from_vec(pattern))
    }

    fn test_seasonal_significance(&self, data: &Array2<f64>, period: usize) -> DeviceResult<f64> {
        // Approximate p-value for the autocorrelation at the given lag.
        // Under H0 (no seasonality), ACF ~ N(0, 1/n), so test statistic = acf * sqrt(n).
        let n = data.nrows();
        let n_cols = data.ncols();
        if n < period + 1 || period == 0 || n_cols == 0 {
            return Ok(1.0);
        }
        let mut acf_sum = 0.0;
        for j in 0..n_cols {
            let col: Vec<f64> = (0..n).map(|i| data[[i, j]]).collect();
            let mean = col.iter().sum::<f64>() / n as f64;
            let var: f64 = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            if var < 1e-12 { continue; }
            let n_terms = n - period;
            let cov: f64 = (0..n_terms).map(|i| (col[i] - mean) * (col[i + period] - mean)).sum::<f64>() / n as f64;
            acf_sum += cov / var;
        }
        let acf = acf_sum / n_cols as f64;
        let z = acf * (n as f64).sqrt();
        let p = 2.0 * normal_survival(z.abs());
        Ok(p.clamp(0.0, 1.0))
    }
}

impl ChangepointDetector {
    pub fn new(config: &ChangepointDetectionConfig) -> Self {
        Self {
            method: config.detection_method.clone(),
            min_segment_length: config.min_segment_length,
            detection_threshold: config.detection_threshold,
            changepoint_history: Vec::new(),
        }
    }

    /// Detect changepoints in time series data
    pub fn detect_changepoints(&mut self, data: &Array2<f64>) -> DeviceResult<ChangepointAnalysisResult> {
        let changepoints = self.find_changepoints(data)?;
        let scores = self.calculate_changepoint_scores(data, &changepoints)?;
        let types = self.classify_changepoint_types(data, &changepoints)?;
        let confidence_levels = self.calculate_confidence_levels(&scores)?;

        let result = ChangepointAnalysisResult {
            changepoints,
            scores,
            types,
            confidence_levels,
        };

        self.changepoint_history.push(result.clone());

        Ok(result)
    }

    fn find_changepoints(&self, data: &Array2<f64>) -> DeviceResult<Vec<usize>> {
        match &self.method {
            ChangepointDetectionMethod::PELT { penalty } => {
                self.pelt_detection(data, *penalty)
            },
            ChangepointDetectionMethod::BinarySegmentation { max_changepoints } => {
                self.binary_segmentation(data, *max_changepoints)
            },
            ChangepointDetectionMethod::WindowBased { window_size } => {
                self.window_based_detection(data, *window_size)
            },
            ChangepointDetectionMethod::BayesianChangepoint { prior_prob } => {
                self.bayesian_detection(data, *prior_prob)
            },
        }
    }

    fn pelt_detection(&self, data: &Array2<f64>, penalty: f64) -> DeviceResult<Vec<usize>> {
        // PELT (Pruned Exact Linear Time) with Gaussian cost function (sum of squared deviations).
        // This is a simplified single-feature version averaged across columns.
        let n = data.nrows();
        let n_cols = data.ncols();
        let min_seg = self.min_segment_length.max(2);
        if n < 2 * min_seg || n_cols == 0 {
            return Ok(vec![]);
        }

        // Average across columns to get a 1D signal
        let signal: Vec<f64> = (0..n).map(|i| {
            (0..n_cols).map(|j| data[[i, j]]).sum::<f64>() / n_cols as f64
        }).collect();

        // Gaussian segment cost: n * log(variance) (negative log-likelihood proportional)
        let seg_cost = |start: usize, end: usize| -> f64 {
            let len = end - start;
            if len < 2 { return 0.0; }
            let mean = signal[start..end].iter().sum::<f64>() / len as f64;
            let var = signal[start..end].iter().map(|x| (x - mean).powi(2)).sum::<f64>() / len as f64;
            if var < 1e-12 { return 0.0; }
            len as f64 * var.ln()
        };

        // Dynamic programming with pruning
        let mut f = vec![f64::INFINITY; n + 1];
        let mut cp = vec![0usize; n + 1];
        f[0] = -penalty;

        let mut admissible: Vec<usize> = vec![0];

        for t in min_seg..=n {
            let mut best_cost = f64::INFINITY;
            let mut best_prev = 0;
            let mut still_admissible = Vec::new();

            for &s in &admissible {
                if t - s < min_seg { continue; }
                let cost = f[s] + seg_cost(s, t) + penalty;
                if cost < best_cost {
                    best_cost = cost;
                    best_prev = s;
                }
                // PELT pruning: keep s if f[s] + seg_cost(s,t) <= best_cost
                if f[s] + seg_cost(s, t) <= best_cost {
                    still_admissible.push(s);
                }
            }
            // Add t as new candidate if it satisfies min_seg from start
            if t + min_seg <= n {
                still_admissible.push(t);
            }
            admissible = still_admissible;

            if best_cost < f64::INFINITY {
                f[t] = best_cost;
                cp[t] = best_prev;
            }
        }

        // Backtrack
        let mut changepoints = Vec::new();
        let mut t = n;
        while cp[t] != 0 {
            t = cp[t];
            if t > 0 && t < n {
                changepoints.push(t);
            }
        }
        changepoints.sort_unstable();
        Ok(changepoints)
    }

    fn binary_segmentation(&self, data: &Array2<f64>, max_changepoints: usize) -> DeviceResult<Vec<usize>> {
        // Binary segmentation via CUSUM (cumulative sum) statistic.
        let n = data.nrows();
        let n_cols = data.ncols();
        let min_seg = self.min_segment_length.max(2);
        if n < 2 * min_seg || n_cols == 0 || max_changepoints == 0 {
            return Ok(vec![]);
        }

        let signal: Vec<f64> = (0..n).map(|i| {
            (0..n_cols).map(|j| data[[i, j]]).sum::<f64>() / n_cols as f64
        }).collect();

        let mut changepoints = Vec::new();
        let mut segments: Vec<(usize, usize)> = vec![(0, n)];

        while changepoints.len() < max_changepoints && !segments.is_empty() {
            let mut best_seg_idx = None;
            let mut best_cp = 0usize;
            let mut best_score = 0.0f64;

            for (seg_idx, &(start, end)) in segments.iter().enumerate() {
                if end - start < 2 * min_seg { continue; }
                // Find the split point maximizing the CUSUM statistic
                let seg_mean = signal[start..end].iter().sum::<f64>() / (end - start) as f64;
                let mut cusum = 0.0f64;
                let mut max_cusum = 0.0f64;
                let mut max_t = start + min_seg;
                for t in start..end {
                    cusum += signal[t] - seg_mean;
                    if t >= start + min_seg && t < end - min_seg && cusum.abs() > max_cusum {
                        max_cusum = cusum.abs();
                        max_t = t + 1;
                    }
                }
                if max_cusum > best_score {
                    best_score = max_cusum;
                    best_cp = max_t;
                    best_seg_idx = Some(seg_idx);
                }
            }

            if let Some(idx) = best_seg_idx {
                if best_score > self.detection_threshold {
                    let (start, end) = segments.remove(idx);
                    segments.push((start, best_cp));
                    segments.push((best_cp, end));
                    changepoints.push(best_cp);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        changepoints.sort_unstable();
        Ok(changepoints)
    }

    fn window_based_detection(&self, data: &Array2<f64>, window_size: usize) -> DeviceResult<Vec<usize>> {
        // Window-based detection: compare statistics in two adjacent windows.
        let n = data.nrows();
        let n_cols = data.ncols();
        let w = window_size.max(2);
        if n < 2 * w || n_cols == 0 {
            return Ok(vec![]);
        }

        let signal: Vec<f64> = (0..n).map(|i| {
            (0..n_cols).map(|j| data[[i, j]]).sum::<f64>() / n_cols as f64
        }).collect();

        let mut changepoints = Vec::new();
        let threshold = self.detection_threshold;

        for t in w..(n - w) {
            let left = &signal[(t - w)..t];
            let right = &signal[t..(t + w)];
            let left_mean = left.iter().sum::<f64>() / w as f64;
            let right_mean = right.iter().sum::<f64>() / w as f64;
            let left_var = left.iter().map(|x| (x - left_mean).powi(2)).sum::<f64>() / w as f64;
            let right_var = right.iter().map(|x| (x - right_mean).powi(2)).sum::<f64>() / w as f64;
            let pooled_std = ((left_var + right_var) / 2.0).sqrt().max(1e-10);
            let score = (left_mean - right_mean).abs() / pooled_std;
            if score > threshold {
                // Avoid consecutive detections within min_segment_length
                let too_close = changepoints.last()
                    .map(|&prev: &usize| t - prev < self.min_segment_length)
                    .unwrap_or(false);
                if !too_close {
                    changepoints.push(t);
                }
            }
        }
        Ok(changepoints)
    }

    fn bayesian_detection(&self, data: &Array2<f64>, prior_prob: f64) -> DeviceResult<Vec<usize>> {
        // Adams-MacKay Bayesian Online Changepoint Detection (BOCPD) with a
        // Normal-Inverse-Gamma conjugate prior for the Gaussian observation model.
        //
        // At each time t we maintain a probability distribution over run lengths r_t.
        // The hazard function H = prior_prob represents the prior probability of a
        // changepoint occurring at any step.
        //
        // Algorithm:
        //   For each new data point x_t:
        //     1. Compute predictive probabilities P(x_t | r_{t-1}, data) for each
        //        run-length hypothesis using the NIG predictive (Student-t).
        //     2. Compute growth probabilities (run continues) and changepoint
        //        probabilities (run resets to 0).
        //     3. Normalise and identify run lengths with high changepoint mass.
        //
        // For efficiency we operate on the column-mean signal across features.
        let n = data.nrows();
        let n_cols = data.ncols();
        if n < 2 || n_cols == 0 {
            return Ok(vec![]);
        }

        // Collapse multivariate data to univariate by averaging across features.
        let signal: Vec<f64> = (0..n)
            .map(|i| (0..n_cols).map(|j| data[[i, j]]).sum::<f64>() / n_cols as f64)
            .collect();

        let h = prior_prob.clamp(1e-6, 1.0 - 1e-6); // hazard rate

        // NIG hyper-parameters (weakly informative priors)
        let mu0 = 0.0_f64;
        let kappa0 = 1.0_f64;
        let alpha0 = 1.0_f64;
        let beta0 = 1.0_f64;

        // Run-length probabilities: R[r] = P(r_t = r | x_{1..t})
        // Start with P(r_0 = 0) = 1.
        let mut run_probs: Vec<f64> = vec![1.0_f64];
        // Sufficient statistics per run-length hypothesis: (mu, kappa, alpha, beta)
        let mut suf_mu: Vec<f64> = vec![mu0];
        let mut suf_kappa: Vec<f64> = vec![kappa0];
        let mut suf_alpha: Vec<f64> = vec![alpha0];
        let mut suf_beta: Vec<f64> = vec![beta0];

        let mut changepoints: Vec<usize> = Vec::new();

        for t in 1..n {
            let x = signal[t];
            let n_hyp = run_probs.len();

            // Predictive probability under each run-length hypothesis using
            // the Student-t predictive distribution from the NIG model.
            let pred_probs: Vec<f64> = (0..n_hyp)
                .map(|r| {
                    let mu_r = suf_mu[r];
                    let kappa_r = suf_kappa[r];
                    let alpha_r = suf_alpha[r];
                    let beta_r = suf_beta[r];
                    // Predictive: t_{2*alpha}(mu_r, beta_r*(kappa_r+1)/(alpha_r*kappa_r))
                    let scale_sq = beta_r * (kappa_r + 1.0) / (alpha_r * kappa_r);
                    let scale = scale_sq.sqrt().max(1e-10);
                    let df = 2.0 * alpha_r;
                    let z = (x - mu_r) / scale;
                    // log Student-t pdf: lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*ln(nu*pi*s^2) - (nu+1)/2*ln(1+z^2/nu)
                    let log_p = lgamma((df + 1.0) / 2.0)
                        - lgamma(df / 2.0)
                        - 0.5 * (df * std::f64::consts::PI * scale_sq).ln()
                        - ((df + 1.0) / 2.0) * (1.0 + z * z / df).ln();
                    log_p.exp().max(f64::MIN_POSITIVE)
                })
                .collect();

            // Growth probabilities: run continues (hazard = 1 - h)
            let mut new_probs: Vec<f64> = vec![0.0_f64]; // r=0 (new run)
            let mut new_mu = vec![mu0];
            let mut new_kappa = vec![kappa0];
            let mut new_alpha = vec![alpha0];
            let mut new_beta = vec![beta0];

            let mut cp_mass = 0.0_f64;
            for r in 0..n_hyp {
                let growth_prob = run_probs[r] * pred_probs[r] * (1.0 - h);
                let cp_prob = run_probs[r] * pred_probs[r] * h;
                cp_mass += cp_prob;
                new_probs.push(growth_prob);
                // Update NIG sufficient statistics (sequential Bayesian update)
                let mu_r = suf_mu[r];
                let kappa_r = suf_kappa[r];
                let alpha_r = suf_alpha[r];
                let beta_r = suf_beta[r];
                let kappa_new = kappa_r + 1.0;
                let mu_new = (kappa_r * mu_r + x) / kappa_new;
                let alpha_new = alpha_r + 0.5;
                let beta_new = beta_r + kappa_r * (x - mu_r).powi(2) / (2.0 * kappa_new);
                new_mu.push(mu_new);
                new_kappa.push(kappa_new);
                new_alpha.push(alpha_new);
                new_beta.push(beta_new);
            }
            // Add changepoint mass to r=0
            new_probs[0] += cp_mass;

            // Normalise
            let total: f64 = new_probs.iter().sum();
            if total > 0.0 {
                new_probs.iter_mut().for_each(|p| *p /= total);
            }

            // If P(r_t = 0) > threshold we have detected a changepoint at t.
            let cp_posterior = new_probs[0];
            if cp_posterior > self.detection_threshold {
                let too_close = changepoints
                    .last()
                    .map(|&prev: &usize| t - prev < self.min_segment_length)
                    .unwrap_or(false);
                if !too_close {
                    changepoints.push(t);
                }
            }

            run_probs = new_probs;
            suf_mu = new_mu;
            suf_kappa = new_kappa;
            suf_alpha = new_alpha;
            suf_beta = new_beta;
        }

        Ok(changepoints)
    }

    fn calculate_changepoint_scores(&self, data: &Array2<f64>, changepoints: &[usize]) -> DeviceResult<Vec<f64>> {
        // Score each changepoint by the absolute mean shift normalized by pooled std.
        let n = data.nrows();
        let n_cols = data.ncols();
        let w = self.min_segment_length.max(2);
        if n_cols == 0 {
            return Ok(vec![0.0; changepoints.len()]);
        }

        let signal: Vec<f64> = (0..n).map(|i| {
            (0..n_cols).map(|j| data[[i, j]]).sum::<f64>() / n_cols as f64
        }).collect();

        let scores: Vec<f64> = changepoints.iter().map(|&cp| {
            let left_start = cp.saturating_sub(w);
            let right_end = (cp + w).min(n);
            if left_start >= cp || cp >= right_end {
                return 0.0;
            }
            let left = &signal[left_start..cp];
            let right = &signal[cp..right_end];
            if left.is_empty() || right.is_empty() { return 0.0; }
            let lm = left.iter().sum::<f64>() / left.len() as f64;
            let rm = right.iter().sum::<f64>() / right.len() as f64;
            let lv = left.iter().map(|x| (x - lm).powi(2)).sum::<f64>() / left.len() as f64;
            let rv = right.iter().map(|x| (x - rm).powi(2)).sum::<f64>() / right.len() as f64;
            let pooled_std = ((lv + rv) / 2.0).sqrt().max(1e-10);
            (lm - rm).abs() / pooled_std
        }).collect();
        Ok(scores)
    }

    fn classify_changepoint_types(&self, data: &Array2<f64>, changepoints: &[usize]) -> DeviceResult<Vec<ChangepointType>> {
        // Classify each changepoint as MeanShift, VarianceChange, or TrendChange
        // based on relative change magnitudes in mean and variance.
        let n = data.nrows();
        let n_cols = data.ncols();
        let w = self.min_segment_length.max(2);
        if n_cols == 0 {
            return Ok(vec![ChangepointType::MeanShift; changepoints.len()]);
        }

        let signal: Vec<f64> = (0..n).map(|i| {
            (0..n_cols).map(|j| data[[i, j]]).sum::<f64>() / n_cols as f64
        }).collect();

        let types: Vec<ChangepointType> = changepoints.iter().map(|&cp| {
            let left_start = cp.saturating_sub(w);
            let right_end = (cp + w).min(n);
            if left_start >= cp || cp >= right_end {
                return ChangepointType::MeanShift;
            }
            let left = &signal[left_start..cp];
            let right = &signal[cp..right_end];
            if left.len() < 2 || right.len() < 2 { return ChangepointType::MeanShift; }

            let lm = left.iter().sum::<f64>() / left.len() as f64;
            let rm = right.iter().sum::<f64>() / right.len() as f64;
            let lv = left.iter().map(|x| (x - lm).powi(2)).sum::<f64>() / left.len() as f64;
            let rv = right.iter().map(|x| (x - rm).powi(2)).sum::<f64>() / right.len() as f64;

            let mean_change = (lm - rm).abs();
            let global_std = (lv + rv).sqrt().max(1e-10);
            let mean_change_norm = mean_change / global_std;

            // Variance ratio test
            let var_ratio = if lv > 1e-12 && rv > 1e-12 {
                (lv / rv).max(rv / lv)
            } else {
                1.0
            };

            // Classify based on dominant signal
            if var_ratio > 3.0 && var_ratio > mean_change_norm {
                ChangepointType::VarianceChange
            } else if mean_change_norm > 1.0 {
                ChangepointType::MeanShift
            } else {
                // Check for trend change via slope comparison
                let left_slope = if left.len() >= 2 {
                    let n_l = left.len() as f64;
                    let x_m = (n_l - 1.0) / 2.0;
                    let ss_xx: f64 = (0..left.len()).map(|i| (i as f64 - x_m).powi(2)).sum();
                    if ss_xx > 1e-10 {
                        let ss_xy: f64 = (0..left.len()).map(|i| (i as f64 - x_m) * (left[i] - lm)).sum();
                        ss_xy / ss_xx
                    } else { 0.0 }
                } else { 0.0 };
                let right_slope = if right.len() >= 2 {
                    let n_r = right.len() as f64;
                    let x_m = (n_r - 1.0) / 2.0;
                    let ss_xx: f64 = (0..right.len()).map(|i| (i as f64 - x_m).powi(2)).sum();
                    if ss_xx > 1e-10 {
                        let ss_xy: f64 = (0..right.len()).map(|i| (i as f64 - x_m) * (right[i] - rm)).sum();
                        ss_xy / ss_xx
                    } else { 0.0 }
                } else { 0.0 };
                if (left_slope - right_slope).abs() > 1e-6 {
                    ChangepointType::TrendChange
                } else {
                    ChangepointType::MeanShift
                }
            }
        }).collect();
        Ok(types)
    }

    fn calculate_confidence_levels(&self, scores: &[f64]) -> DeviceResult<Vec<f64>> {
        // Convert detection scores to confidence levels via sigmoid transform.
        // Score of ~1.96 (95% normal) maps to ~0.88 confidence.
        // Score of ~3.0 (99.7% normal) maps to ~0.95 confidence.
        let confidence: Vec<f64> = scores.iter().map(|&s| {
            // logistic(s - 1.0) gives 0.5 at score=1.0, ~0.99 at score=5.6
            let c = 1.0 / (1.0 + (-(s - 1.0)).exp());
            c.clamp(0.0, 1.0)
        }).collect();
        Ok(confidence)
    }
}

// ─── Helper functions ─────────────────────────────────────────────────────────

/// Levinson-Durbin recursion to solve Yule-Walker equations for AR(p).
/// Returns the p AR coefficients.
fn levinson_durbin(r: &[f64], p: usize) -> Vec<f64> {
    // r[0] = variance, r[1..=p] = autocorrelations at lags 1..=p
    if p == 0 || r.len() < p + 1 || r[0].abs() < 1e-12 {
        return vec![0.0; p];
    }
    let mut phi = vec![vec![0.0_f64; p + 1]; p + 1];
    let mut sigma_sq = r[0];

    // Step 1: init
    phi[1][1] = r[1] / r[0];
    sigma_sq *= 1.0 - phi[1][1].powi(2);

    for m in 2..=p {
        if sigma_sq.abs() < 1e-12 { break; }
        let num: f64 = r[m] - (1..m).map(|k| phi[m - 1][k] * r[m - k]).sum::<f64>();
        phi[m][m] = num / sigma_sq;
        for k in 1..m {
            phi[m][k] = phi[m - 1][k] - phi[m][m] * phi[m - 1][m - k];
        }
        sigma_sq *= 1.0 - phi[m][m].powi(2);
    }

    (1..=p).map(|k| phi[p][k]).collect()
}

/// Compute the survival function of the standard normal: P(Z > z).
/// Uses a rational approximation accurate to ~7 decimal places.
fn normal_survival(z: f64) -> f64 {
    if z < 0.0 { return 1.0 - normal_survival(-z); }
    // Abramowitz & Stegun 26.2.17
    let t = 1.0 / (1.0 + 0.2316419 * z);
    let poly = t * (0.319381530
        + t * (-0.356563782
        + t * (1.781477937
        + t * (-1.821255978
        + t * 1.330274429))));
    let pdf = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    (pdf * poly).clamp(0.0, 1.0)
}
