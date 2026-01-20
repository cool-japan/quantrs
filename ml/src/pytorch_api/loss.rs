//! Loss functions for PyTorch-like API

use crate::error::{MLError, Result};
use crate::scirs2_integration::SciRS2Array;
use scirs2_core::ndarray::{ArrayD, IxDyn};

/// Loss functions for quantum ML
pub trait QuantumLoss: Send + Sync {
    /// Compute loss
    fn forward(&self, predictions: &SciRS2Array, targets: &SciRS2Array) -> Result<SciRS2Array>;

    /// Loss function name
    fn name(&self) -> &str;
}

/// Mean Squared Error loss
pub struct QuantumMSELoss;

impl QuantumLoss for QuantumMSELoss {
    fn forward(&self, predictions: &SciRS2Array, targets: &SciRS2Array) -> Result<SciRS2Array> {
        let diff = predictions.data.clone() - &targets.data;
        let squared_diff = &diff * &diff;
        let mse = squared_diff.mean().ok_or_else(|| {
            MLError::InvalidConfiguration("Cannot compute mean of empty array".to_string())
        })?;

        let loss_data = ArrayD::from_elem(IxDyn(&[]), mse);
        Ok(SciRS2Array::new(loss_data, predictions.requires_grad))
    }

    fn name(&self) -> &str {
        "MSELoss"
    }
}

/// Cross Entropy loss
pub struct QuantumCrossEntropyLoss;

impl QuantumLoss for QuantumCrossEntropyLoss {
    fn forward(&self, predictions: &SciRS2Array, targets: &SciRS2Array) -> Result<SciRS2Array> {
        let max_val = predictions
            .data
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_preds = predictions.data.mapv(|x| (x - max_val).exp());
        let sum_exp = exp_preds.sum();
        let softmax = exp_preds.mapv(|x| x / sum_exp);

        let log_softmax = softmax.mapv(|x| x.ln());
        let cross_entropy = -(&targets.data * &log_softmax).sum();

        let loss_data = ArrayD::from_elem(IxDyn(&[]), cross_entropy);
        Ok(SciRS2Array::new(loss_data, predictions.requires_grad))
    }

    fn name(&self) -> &str {
        "CrossEntropyLoss"
    }
}

/// Binary Cross Entropy Loss
pub struct QuantumBCELoss;

impl QuantumLoss for QuantumBCELoss {
    fn forward(&self, predictions: &SciRS2Array, targets: &SciRS2Array) -> Result<SciRS2Array> {
        let eps = 1e-7;
        let mut loss = 0.0;
        let n = predictions.data.len() as f64;

        for (pred, target) in predictions.data.iter().zip(targets.data.iter()) {
            let p = pred.clamp(eps, 1.0 - eps);
            loss -= target * p.ln() + (1.0 - target) * (1.0 - p).ln();
        }

        let output = ArrayD::from_elem(IxDyn(&[1]), loss / n);
        Ok(SciRS2Array::new(output, predictions.requires_grad))
    }

    fn name(&self) -> &str {
        "BCELoss"
    }
}

/// Binary Cross Entropy with Logits Loss
pub struct QuantumBCEWithLogitsLoss;

impl QuantumLoss for QuantumBCEWithLogitsLoss {
    fn forward(&self, predictions: &SciRS2Array, targets: &SciRS2Array) -> Result<SciRS2Array> {
        let mut loss = 0.0;
        let n = predictions.data.len() as f64;

        for (logit, target) in predictions.data.iter().zip(targets.data.iter()) {
            let max_val = logit.max(0.0);
            loss += max_val - logit * target + (1.0 + (-logit.abs()).exp()).ln();
        }

        let output = ArrayD::from_elem(IxDyn(&[1]), loss / n);
        Ok(SciRS2Array::new(output, predictions.requires_grad))
    }

    fn name(&self) -> &str {
        "BCEWithLogitsLoss"
    }
}

/// L1 Loss (Mean Absolute Error)
pub struct QuantumL1Loss;

impl QuantumLoss for QuantumL1Loss {
    fn forward(&self, predictions: &SciRS2Array, targets: &SciRS2Array) -> Result<SciRS2Array> {
        let mut loss = 0.0;
        let n = predictions.data.len() as f64;

        for (pred, target) in predictions.data.iter().zip(targets.data.iter()) {
            loss += (pred - target).abs();
        }

        let output = ArrayD::from_elem(IxDyn(&[1]), loss / n);
        Ok(SciRS2Array::new(output, predictions.requires_grad))
    }

    fn name(&self) -> &str {
        "L1Loss"
    }
}

/// Smooth L1 Loss (Huber Loss)
pub struct QuantumSmoothL1Loss {
    beta: f64,
}

impl QuantumSmoothL1Loss {
    /// Create new smooth L1 loss
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }
}

impl Default for QuantumSmoothL1Loss {
    fn default() -> Self {
        Self { beta: 1.0 }
    }
}

impl QuantumLoss for QuantumSmoothL1Loss {
    fn forward(&self, predictions: &SciRS2Array, targets: &SciRS2Array) -> Result<SciRS2Array> {
        let mut loss = 0.0;
        let n = predictions.data.len() as f64;

        for (pred, target) in predictions.data.iter().zip(targets.data.iter()) {
            let diff = (pred - target).abs();
            if diff < self.beta {
                loss += 0.5 * diff * diff / self.beta;
            } else {
                loss += diff - 0.5 * self.beta;
            }
        }

        let output = ArrayD::from_elem(IxDyn(&[1]), loss / n);
        Ok(SciRS2Array::new(output, predictions.requires_grad))
    }

    fn name(&self) -> &str {
        "SmoothL1Loss"
    }
}

/// Negative Log Likelihood Loss
pub struct QuantumNLLLoss;

impl QuantumLoss for QuantumNLLLoss {
    fn forward(&self, predictions: &SciRS2Array, targets: &SciRS2Array) -> Result<SciRS2Array> {
        let shape = predictions.data.shape();
        if shape.len() != 2 {
            return Err(MLError::InvalidConfiguration(
                "NLLLoss expects 2D predictions (batch_size, num_classes)".to_string(),
            ));
        }

        let batch_size = shape[0];
        let mut loss = 0.0;

        for b in 0..batch_size {
            let target_class = targets.data[[b]] as usize;
            loss -= predictions.data[[b, target_class]];
        }

        let output = ArrayD::from_elem(IxDyn(&[1]), loss / batch_size as f64);
        Ok(SciRS2Array::new(output, predictions.requires_grad))
    }

    fn name(&self) -> &str {
        "NLLLoss"
    }
}

/// Kullback-Leibler Divergence Loss
pub struct QuantumKLDivLoss {
    reduction: String,
}

impl QuantumKLDivLoss {
    /// Create new KL divergence loss
    pub fn new() -> Self {
        Self {
            reduction: "mean".to_string(),
        }
    }

    /// Set reduction type
    pub fn reduction(mut self, reduction: &str) -> Self {
        self.reduction = reduction.to_string();
        self
    }
}

impl Default for QuantumKLDivLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumLoss for QuantumKLDivLoss {
    fn forward(&self, predictions: &SciRS2Array, targets: &SciRS2Array) -> Result<SciRS2Array> {
        let mut loss = 0.0;

        for (log_q, p) in predictions.data.iter().zip(targets.data.iter()) {
            if *p > 0.0 {
                loss += p * (p.ln() - log_q);
            }
        }

        let output = match self.reduction.as_str() {
            "sum" => ArrayD::from_elem(IxDyn(&[1]), loss),
            "mean" => ArrayD::from_elem(IxDyn(&[1]), loss / predictions.data.len() as f64),
            _ => ArrayD::from_elem(IxDyn(&[1]), loss),
        };

        Ok(SciRS2Array::new(output, predictions.requires_grad))
    }

    fn name(&self) -> &str {
        "KLDivLoss"
    }
}
