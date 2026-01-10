//! Callbacks for Keras-like API

use super::TrainingHistory;
use crate::error::Result;
use scirs2_core::ndarray::ArrayD;
use std::collections::HashMap;

/// Callback trait for training
pub trait Callback: Send + Sync {
    /// Called at the end of each epoch
    fn on_epoch_end(&self, epoch: usize, history: &TrainingHistory) -> Result<()>;
}

/// Early stopping callback
pub struct EarlyStopping {
    /// Metric to monitor
    monitor: String,
    /// Minimum change to qualify as improvement
    min_delta: f64,
    /// Number of epochs with no improvement to wait
    patience: usize,
    /// Best value seen so far
    best: f64,
    /// Number of epochs without improvement
    wait: usize,
    /// Whether to stop training
    stopped: bool,
}

impl EarlyStopping {
    /// Create new early stopping callback
    pub fn new(monitor: String, min_delta: f64, patience: usize) -> Self {
        Self {
            monitor,
            min_delta,
            patience,
            best: f64::INFINITY,
            wait: 0,
            stopped: false,
        }
    }
}

impl Callback for EarlyStopping {
    fn on_epoch_end(&self, _epoch: usize, _history: &TrainingHistory) -> Result<()> {
        Ok(())
    }
}

/// Model checkpoint callback
pub struct ModelCheckpoint {
    /// File path pattern
    filepath: String,
    /// Metric to monitor
    monitor: String,
    /// Save best only
    save_best_only: bool,
    /// Mode (min or max)
    mode: String,
    /// Best value seen
    best: f64,
}

impl ModelCheckpoint {
    /// Create new model checkpoint callback
    pub fn new(filepath: impl Into<String>) -> Self {
        Self {
            filepath: filepath.into(),
            monitor: "val_loss".to_string(),
            save_best_only: false,
            mode: "min".to_string(),
            best: f64::INFINITY,
        }
    }

    /// Set metric to monitor
    pub fn monitor(mut self, monitor: impl Into<String>) -> Self {
        self.monitor = monitor.into();
        self
    }

    /// Save best only
    pub fn save_best_only(mut self, save_best_only: bool) -> Self {
        self.save_best_only = save_best_only;
        self
    }

    /// Set mode
    pub fn mode(mut self, mode: impl Into<String>) -> Self {
        self.mode = mode.into();
        if self.mode == "max" {
            self.best = f64::NEG_INFINITY;
        }
        self
    }
}

impl Callback for ModelCheckpoint {
    fn on_epoch_end(&self, _epoch: usize, history: &TrainingHistory) -> Result<()> {
        if let Some(&current) = history.loss.last() {
            let _ = current;
        }
        Ok(())
    }
}

/// CSV logger callback
pub struct CSVLogger {
    /// File path
    filename: String,
    /// Separator
    separator: String,
    /// Append mode
    append: bool,
    /// Logged data
    logged_data: Vec<HashMap<String, f64>>,
}

impl CSVLogger {
    /// Create new CSV logger
    pub fn new(filename: impl Into<String>) -> Self {
        Self {
            filename: filename.into(),
            separator: ",".to_string(),
            append: false,
            logged_data: Vec::new(),
        }
    }

    /// Set separator
    pub fn separator(mut self, separator: impl Into<String>) -> Self {
        self.separator = separator.into();
        self
    }

    /// Set append mode
    pub fn append(mut self, append: bool) -> Self {
        self.append = append;
        self
    }

    /// Get logged data
    pub fn get_logged_data(&self) -> &[HashMap<String, f64>] {
        &self.logged_data
    }
}

impl Callback for CSVLogger {
    fn on_epoch_end(&self, epoch: usize, history: &TrainingHistory) -> Result<()> {
        let _ = (epoch, history);
        Ok(())
    }
}

/// Reduce learning rate on plateau callback
pub struct ReduceLROnPlateau {
    /// Metric to monitor
    monitor: String,
    /// Factor to reduce LR by
    factor: f64,
    /// Patience epochs
    patience: usize,
    /// Minimum learning rate
    min_lr: f64,
    /// Mode (min or max)
    mode: String,
    /// Best value seen
    best: f64,
    /// Wait counter
    wait: usize,
    /// Current learning rate
    current_lr: f64,
}

impl ReduceLROnPlateau {
    /// Create new ReduceLROnPlateau callback
    pub fn new() -> Self {
        Self {
            monitor: "val_loss".to_string(),
            factor: 0.1,
            patience: 10,
            min_lr: 1e-7,
            mode: "min".to_string(),
            best: f64::INFINITY,
            wait: 0,
            current_lr: 0.001,
        }
    }

    /// Set metric to monitor
    pub fn monitor(mut self, monitor: impl Into<String>) -> Self {
        self.monitor = monitor.into();
        self
    }

    /// Set reduction factor
    pub fn factor(mut self, factor: f64) -> Self {
        self.factor = factor;
        self
    }

    /// Set patience
    pub fn patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }
}

impl Default for ReduceLROnPlateau {
    fn default() -> Self {
        Self::new()
    }
}

impl Callback for ReduceLROnPlateau {
    fn on_epoch_end(&self, _epoch: usize, history: &TrainingHistory) -> Result<()> {
        if let Some(&current) = history.loss.last() {
            let _ = current;
        }
        Ok(())
    }
}

/// Regularizer types
#[derive(Debug, Clone)]
pub enum RegularizerType {
    /// L1 regularization
    L1(f64),
    /// L2 regularization
    L2(f64),
    /// L1L2 regularization
    L1L2 { l1: f64, l2: f64 },
}

impl RegularizerType {
    /// Compute regularization loss
    pub fn compute(&self, weights: &ArrayD<f64>) -> f64 {
        match self {
            RegularizerType::L1(l1) => l1 * weights.iter().map(|w| w.abs()).sum::<f64>(),
            RegularizerType::L2(l2) => l2 * weights.iter().map(|w| w * w).sum::<f64>(),
            RegularizerType::L1L2 { l1, l2 } => {
                l1 * weights.iter().map(|w| w.abs()).sum::<f64>()
                    + l2 * weights.iter().map(|w| w * w).sum::<f64>()
            }
        }
    }
}
