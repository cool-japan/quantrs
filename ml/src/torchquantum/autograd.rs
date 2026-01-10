//! Autograd helpers for TorchQuantum-compatible quantum machine learning
//!
//! This module provides automatic differentiation utilities:
//! - **GradientAccumulator**: Accumulate gradients across multiple backward passes
//! - **ParameterRegistry**: Track and manage all parameters in a quantum model
//! - **GradientClipper**: Prevent exploding gradients with various clipping strategies
//! - **GradientChecker**: Numerical gradient verification for debugging
//! - **ParameterGroup**: Organize parameters with different optimization settings
//!
//! ## TorchQuantum Compatibility
//!
//! These utilities mirror PyTorch's autograd functionality adapted for quantum circuits:
//! - Parameter tracking similar to `torch.nn.parameter.Parameter`
//! - Gradient accumulation like PyTorch's backward pass
//! - Gradient clipping similar to `torch.nn.utils.clip_grad_*`

use super::{TQModule, TQParameter};
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, ArrayD, Axis, IxDyn};
use std::collections::HashMap;

// ============================================================================
// GradientAccumulator - Accumulate gradients across multiple passes
// ============================================================================

/// Gradient accumulator for mini-batch training
///
/// Accumulates gradients from multiple forward-backward passes before
/// applying parameter updates. Useful for:
/// - Simulating larger batch sizes with limited memory
/// - Gradient accumulation across multiple quantum circuit executions
/// - Variance reduction in parameter-shift rule calculations
#[derive(Debug, Clone)]
pub struct GradientAccumulator {
    /// Number of accumulation steps
    pub accumulation_steps: usize,
    /// Current step counter
    current_step: usize,
    /// Accumulated gradients for each parameter (keyed by parameter name)
    accumulated_grads: HashMap<String, ArrayD<f64>>,
    /// Whether to average gradients (vs sum)
    average: bool,
}

impl GradientAccumulator {
    /// Create new gradient accumulator
    pub fn new(accumulation_steps: usize) -> Self {
        Self {
            accumulation_steps,
            current_step: 0,
            accumulated_grads: HashMap::new(),
            average: true,
        }
    }

    /// Create accumulator with sum (no averaging)
    pub fn with_sum(accumulation_steps: usize) -> Self {
        Self {
            accumulation_steps,
            current_step: 0,
            accumulated_grads: HashMap::new(),
            average: false,
        }
    }

    /// Accumulate gradients from parameters
    pub fn accumulate(&mut self, params: &[TQParameter]) -> Result<()> {
        for param in params {
            if !param.requires_grad {
                continue;
            }

            if let Some(grad) = &param.grad {
                let entry = self
                    .accumulated_grads
                    .entry(param.name.clone())
                    .or_insert_with(|| ArrayD::zeros(grad.raw_dim()));

                *entry = &*entry + grad;
            }
        }

        self.current_step += 1;
        Ok(())
    }

    /// Check if ready to apply gradients
    pub fn is_ready(&self) -> bool {
        self.current_step >= self.accumulation_steps
    }

    /// Get accumulated gradients and reset
    pub fn get_and_reset(&mut self) -> HashMap<String, ArrayD<f64>> {
        let mut result = std::mem::take(&mut self.accumulated_grads);

        if self.average && self.accumulation_steps > 1 {
            let scale = 1.0 / self.accumulation_steps as f64;
            for grad in result.values_mut() {
                *grad = &*grad * scale;
            }
        }

        self.current_step = 0;
        result
    }

    /// Reset accumulator without returning gradients
    pub fn reset(&mut self) {
        self.accumulated_grads.clear();
        self.current_step = 0;
    }

    /// Get current step count
    pub fn step_count(&self) -> usize {
        self.current_step
    }
}

// ============================================================================
// ParameterRegistry - Track all parameters in a model
// ============================================================================

/// Parameter registry for tracking and managing quantum model parameters
///
/// Provides centralized parameter management:
/// - Track all parameters across multiple quantum modules
/// - Freeze/unfreeze specific parameters
/// - Get parameter statistics (count, memory usage)
/// - Named parameter access
#[derive(Debug)]
pub struct ParameterRegistry {
    /// Map of parameter name to parameter
    parameters: HashMap<String, TQParameter>,
    /// Frozen parameter names (not trainable)
    frozen: Vec<String>,
}

impl ParameterRegistry {
    /// Create new parameter registry
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            frozen: Vec::new(),
        }
    }

    /// Register parameters from a module
    pub fn register_module(&mut self, module: &dyn TQModule) -> Result<()> {
        let params = module.parameters();
        for param in params {
            self.parameters.insert(param.name.clone(), param);
        }
        Ok(())
    }

    /// Register a single parameter
    pub fn register(&mut self, param: TQParameter) {
        self.parameters.insert(param.name.clone(), param);
    }

    /// Get parameter by name
    pub fn get(&self, name: &str) -> Option<&TQParameter> {
        self.parameters.get(name)
    }

    /// Get mutable parameter by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut TQParameter> {
        self.parameters.get_mut(name)
    }

    /// Get all trainable parameters
    pub fn trainable_parameters(&self) -> Vec<&TQParameter> {
        self.parameters
            .values()
            .filter(|p| p.requires_grad && !self.frozen.contains(&p.name))
            .collect()
    }

    /// Get all parameter names
    pub fn parameter_names(&self) -> Vec<&str> {
        self.parameters.keys().map(|s| s.as_str()).collect()
    }

    /// Total number of parameters
    pub fn count(&self) -> usize {
        self.parameters.values().map(|p| p.numel()).sum()
    }

    /// Number of trainable parameters
    pub fn trainable_count(&self) -> usize {
        self.trainable_parameters().iter().map(|p| p.numel()).sum()
    }

    /// Freeze parameter (make non-trainable)
    pub fn freeze(&mut self, name: &str) -> Result<()> {
        if !self.parameters.contains_key(name) {
            return Err(MLError::InvalidConfiguration(format!(
                "Parameter '{}' not found",
                name
            )));
        }
        if !self.frozen.contains(&name.to_string()) {
            self.frozen.push(name.to_string());
        }
        Ok(())
    }

    /// Unfreeze parameter (make trainable)
    pub fn unfreeze(&mut self, name: &str) -> Result<()> {
        self.frozen.retain(|n| n != name);
        Ok(())
    }

    /// Freeze all parameters
    pub fn freeze_all(&mut self) {
        self.frozen = self.parameters.keys().cloned().collect();
    }

    /// Unfreeze all parameters
    pub fn unfreeze_all(&mut self) {
        self.frozen.clear();
    }

    /// Zero all gradients
    pub fn zero_grad(&mut self) {
        for param in self.parameters.values_mut() {
            param.zero_grad();
        }
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.parameters.values().map(|p| p.numel() * 8).sum() // 8 bytes per f64
    }

    /// Get parameter statistics
    pub fn statistics(&self) -> ParameterStatistics {
        let total_params = self.count();
        let trainable_params = self.trainable_count();
        let memory_mb = self.memory_bytes() as f64 / (1024.0 * 1024.0);

        ParameterStatistics {
            total_params,
            trainable_params,
            frozen_params: total_params - trainable_params,
            memory_mb,
        }
    }
}

impl Default for ParameterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Parameter statistics
#[derive(Debug, Clone)]
pub struct ParameterStatistics {
    pub total_params: usize,
    pub trainable_params: usize,
    pub frozen_params: usize,
    pub memory_mb: f64,
}

// ============================================================================
// GradientClipper - Prevent exploding gradients
// ============================================================================

/// Gradient clipping strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClippingStrategy {
    /// Clip by global norm (scale all gradients by same factor)
    Norm { max_norm: f64 },
    /// Clip each gradient individually by value
    Value { clip_value: f64 },
    /// Adaptive clipping based on parameter norm
    Adaptive { clip_factor: f64 },
}

/// Gradient clipper to prevent exploding gradients
///
/// Provides various clipping strategies:
/// - **Norm clipping**: Scales all gradients if total norm exceeds threshold
/// - **Value clipping**: Clips individual gradient values to [-clip_value, clip_value]
/// - **Adaptive clipping**: Clips based on parameter magnitude
pub struct GradientClipper {
    strategy: ClippingStrategy,
    /// Statistics about last clipping operation
    pub last_norm: Option<f64>,
    pub was_clipped: bool,
}

impl GradientClipper {
    /// Create clipper with norm-based strategy
    pub fn by_norm(max_norm: f64) -> Self {
        Self {
            strategy: ClippingStrategy::Norm { max_norm },
            last_norm: None,
            was_clipped: false,
        }
    }

    /// Create clipper with value-based strategy
    pub fn by_value(clip_value: f64) -> Self {
        Self {
            strategy: ClippingStrategy::Value { clip_value },
            last_norm: None,
            was_clipped: false,
        }
    }

    /// Create clipper with adaptive strategy
    pub fn adaptive(clip_factor: f64) -> Self {
        Self {
            strategy: ClippingStrategy::Adaptive { clip_factor },
            last_norm: None,
            was_clipped: false,
        }
    }

    /// Clip gradients in place
    pub fn clip(&mut self, params: &mut [TQParameter]) -> Result<()> {
        match self.strategy {
            ClippingStrategy::Norm { max_norm } => self.clip_by_norm(params, max_norm),
            ClippingStrategy::Value { clip_value } => self.clip_by_value(params, clip_value),
            ClippingStrategy::Adaptive { clip_factor } => self.clip_adaptive(params, clip_factor),
        }
    }

    fn clip_by_norm(&mut self, params: &mut [TQParameter], max_norm: f64) -> Result<()> {
        // Calculate total gradient norm
        let mut total_norm_sq = 0.0;
        for param in params.iter() {
            if let Some(grad) = &param.grad {
                for &val in grad.iter() {
                    total_norm_sq += val * val;
                }
            }
        }

        let total_norm = total_norm_sq.sqrt();
        self.last_norm = Some(total_norm);

        if total_norm > max_norm {
            let scale = max_norm / (total_norm + 1e-10);
            for param in params {
                if let Some(grad) = &mut param.grad {
                    *grad = &*grad * scale;
                }
            }
            self.was_clipped = true;
        } else {
            self.was_clipped = false;
        }

        Ok(())
    }

    fn clip_by_value(&mut self, params: &mut [TQParameter], clip_value: f64) -> Result<()> {
        self.was_clipped = false;

        for param in params {
            if let Some(grad) = &mut param.grad {
                for val in grad.iter_mut() {
                    if val.abs() > clip_value {
                        *val = val.signum() * clip_value;
                        self.was_clipped = true;
                    }
                }
            }
        }

        Ok(())
    }

    fn clip_adaptive(&mut self, params: &mut [TQParameter], clip_factor: f64) -> Result<()> {
        self.was_clipped = false;

        for param in params {
            if let Some(grad) = &mut param.grad {
                // Calculate parameter norm
                let param_norm: f64 = param.data.iter().map(|&v| v * v).sum::<f64>().sqrt();
                let max_grad = param_norm * clip_factor;

                // Calculate gradient norm
                let grad_norm: f64 = grad.iter().map(|&v| v * v).sum::<f64>().sqrt();

                if grad_norm > max_grad {
                    let scale = max_grad / (grad_norm + 1e-10);
                    *grad = &*grad * scale;
                    self.was_clipped = true;
                }
            }
        }

        Ok(())
    }

    /// Get clipping statistics
    pub fn statistics(&self) -> ClippingStatistics {
        ClippingStatistics {
            was_clipped: self.was_clipped,
            last_norm: self.last_norm,
            strategy: self.strategy,
        }
    }
}

/// Clipping statistics
#[derive(Debug, Clone)]
pub struct ClippingStatistics {
    pub was_clipped: bool,
    pub last_norm: Option<f64>,
    pub strategy: ClippingStrategy,
}

// ============================================================================
// GradientChecker - Numerical gradient verification
// ============================================================================

/// Gradient checker for numerical verification
///
/// Compares analytical gradients (from parameter-shift rule or adjoint method)
/// with numerical gradients (finite differences) to verify correctness.
pub struct GradientChecker {
    /// Epsilon for finite differences
    pub epsilon: f64,
    /// Relative tolerance for comparison
    pub rtol: f64,
    /// Absolute tolerance for comparison
    pub atol: f64,
}

impl GradientChecker {
    /// Create new gradient checker with default tolerances
    pub fn new() -> Self {
        Self {
            epsilon: 1e-5,
            rtol: 1e-3,
            atol: 1e-5,
        }
    }

    /// Create with custom epsilon
    pub fn with_epsilon(epsilon: f64) -> Self {
        Self {
            epsilon,
            rtol: 1e-3,
            atol: 1e-5,
        }
    }

    /// Create with custom tolerances
    pub fn with_tolerances(epsilon: f64, rtol: f64, atol: f64) -> Self {
        Self {
            epsilon,
            rtol,
            atol,
        }
    }

    /// Compute numerical gradient using finite differences
    ///
    /// For function f and parameter θ:
    /// ∂f/∂θ ≈ [f(θ + ε) - f(θ - ε)] / (2ε)
    pub fn numerical_gradient<F>(
        &self,
        param: &mut TQParameter,
        param_idx: usize,
        loss_fn: &mut F,
    ) -> Result<f64>
    where
        F: FnMut() -> Result<f64>,
    {
        // Get original value
        let flat_idx = self.flat_index(param_idx, param.shape());
        let original =
            param.data.as_slice_mut().ok_or_else(|| {
                MLError::InvalidConfiguration("Cannot get mutable slice".to_string())
            })?[flat_idx];

        // f(θ + ε)
        param.data.as_slice_mut().ok_or_else(|| {
            MLError::InvalidConfiguration("Cannot get mutable slice".to_string())
        })?[flat_idx] = original + self.epsilon;
        let loss_plus = loss_fn()?;

        // f(θ - ε)
        param.data.as_slice_mut().ok_or_else(|| {
            MLError::InvalidConfiguration("Cannot get mutable slice".to_string())
        })?[flat_idx] = original - self.epsilon;
        let loss_minus = loss_fn()?;

        // Restore original value
        param.data.as_slice_mut().ok_or_else(|| {
            MLError::InvalidConfiguration("Cannot get mutable slice".to_string())
        })?[flat_idx] = original;

        // Compute numerical gradient
        Ok((loss_plus - loss_minus) / (2.0 * self.epsilon))
    }

    /// Check if analytical and numerical gradients match
    pub fn check_gradient(&self, analytical: f64, numerical: f64) -> GradientCheckResult {
        let abs_diff = (analytical - numerical).abs();
        let rel_diff = if numerical.abs() > 1e-10 {
            abs_diff / numerical.abs()
        } else {
            abs_diff
        };

        let matches = abs_diff <= self.atol || rel_diff <= self.rtol;

        GradientCheckResult {
            analytical,
            numerical,
            abs_diff,
            rel_diff,
            matches,
        }
    }

    fn flat_index(&self, idx: usize, shape: &[usize]) -> usize {
        idx
    }
}

impl Default for GradientChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of gradient check
#[derive(Debug, Clone)]
pub struct GradientCheckResult {
    pub analytical: f64,
    pub numerical: f64,
    pub abs_diff: f64,
    pub rel_diff: f64,
    pub matches: bool,
}

// ============================================================================
// ParameterGroup - Group parameters with different settings
// ============================================================================

/// Parameter group for organizing parameters
///
/// Similar to PyTorch's parameter groups in optimizers.
/// Allows different learning rates, weight decay, etc. for different parameter sets.
#[derive(Debug, Clone)]
pub struct ParameterGroup {
    /// Group name
    pub name: String,
    /// Parameter names in this group
    pub param_names: Vec<String>,
    /// Learning rate multiplier for this group
    pub lr_multiplier: f64,
    /// Weight decay for this group
    pub weight_decay: f64,
    /// Whether gradients are enabled for this group
    pub requires_grad: bool,
}

impl ParameterGroup {
    /// Create new parameter group
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            param_names: Vec::new(),
            lr_multiplier: 1.0,
            weight_decay: 0.0,
            requires_grad: true,
        }
    }

    /// Add parameter to group
    pub fn add_param(&mut self, param_name: impl Into<String>) {
        self.param_names.push(param_name.into());
    }

    /// Set learning rate multiplier
    pub fn with_lr_multiplier(mut self, multiplier: f64) -> Self {
        self.lr_multiplier = multiplier;
        self
    }

    /// Set weight decay
    pub fn with_weight_decay(mut self, decay: f64) -> Self {
        self.weight_decay = decay;
        self
    }

    /// Set requires_grad
    pub fn with_requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    /// Check if parameter belongs to this group
    pub fn contains(&self, param_name: &str) -> bool {
        self.param_names.iter().any(|n| n == param_name)
    }
}

/// Manager for multiple parameter groups
#[derive(Debug)]
pub struct ParameterGroupManager {
    groups: Vec<ParameterGroup>,
}

impl ParameterGroupManager {
    /// Create new manager
    pub fn new() -> Self {
        Self { groups: Vec::new() }
    }

    /// Add a parameter group
    pub fn add_group(&mut self, group: ParameterGroup) {
        self.groups.push(group);
    }

    /// Get group for parameter
    pub fn get_group(&self, param_name: &str) -> Option<&ParameterGroup> {
        self.groups.iter().find(|g| g.contains(param_name))
    }

    /// Get all groups
    pub fn groups(&self) -> &[ParameterGroup] {
        &self.groups
    }

    /// Get learning rate multiplier for parameter
    pub fn lr_multiplier(&self, param_name: &str) -> f64 {
        self.get_group(param_name)
            .map(|g| g.lr_multiplier)
            .unwrap_or(1.0)
    }

    /// Get weight decay for parameter
    pub fn weight_decay(&self, param_name: &str) -> f64 {
        self.get_group(param_name)
            .map(|g| g.weight_decay)
            .unwrap_or(0.0)
    }

    /// Check if parameter requires grad
    pub fn requires_grad(&self, param_name: &str) -> bool {
        self.get_group(param_name)
            .map(|g| g.requires_grad)
            .unwrap_or(true)
    }
}

impl Default for ParameterGroupManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Gradient utilities
// ============================================================================

/// Compute gradient norm (L2 norm of all gradients)
pub fn gradient_norm(params: &[TQParameter]) -> f64 {
    let mut norm_sq = 0.0;
    for param in params {
        if let Some(grad) = &param.grad {
            for &val in grad.iter() {
                norm_sq += val * val;
            }
        }
    }
    norm_sq.sqrt()
}

/// Compute gradient statistics
pub fn gradient_statistics(params: &[TQParameter]) -> GradientStatistics {
    let mut all_grads = Vec::new();
    for param in params {
        if let Some(grad) = &param.grad {
            all_grads.extend(grad.iter().copied());
        }
    }

    if all_grads.is_empty() {
        return GradientStatistics::default();
    }

    let n = all_grads.len() as f64;
    let mean = all_grads.iter().sum::<f64>() / n;
    let variance = all_grads.iter().map(|&g| (g - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    let min = all_grads
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);
    let max = all_grads
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    let norm = gradient_norm(params);

    GradientStatistics {
        mean,
        std,
        min,
        max,
        norm,
    }
}

/// Gradient statistics
#[derive(Debug, Clone, Default)]
pub struct GradientStatistics {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub norm: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::ArrayD;

    #[test]
    fn test_gradient_accumulator() {
        let mut acc = GradientAccumulator::new(3);

        let mut param = TQParameter::new(ArrayD::zeros(IxDyn(&[2])), "test");
        param.grad = Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap());

        // Accumulate 3 times
        for _ in 0..3 {
            acc.accumulate(&[param.clone()]).unwrap();
        }

        assert!(acc.is_ready());

        let grads = acc.get_and_reset();
        let test_grad = &grads["test"];

        // Should be averaged: (1+1+1)/3 = 1, (2+2+2)/3 = 2
        assert!((test_grad[[0]] - 1.0).abs() < 1e-10);
        assert!((test_grad[[1]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_parameter_registry() {
        let mut registry = ParameterRegistry::new();

        let param1 = TQParameter::new(ArrayD::zeros(IxDyn(&[5])), "layer1");
        let param2 = TQParameter::new(ArrayD::zeros(IxDyn(&[10])), "layer2");

        registry.register(param1);
        registry.register(param2);

        assert_eq!(registry.count(), 15);
        assert_eq!(registry.trainable_count(), 15);

        registry.freeze("layer1").unwrap();
        assert_eq!(registry.trainable_count(), 10);

        let stats = registry.statistics();
        assert_eq!(stats.total_params, 15);
        assert_eq!(stats.trainable_params, 10);
        assert_eq!(stats.frozen_params, 5);
    }

    #[test]
    fn test_gradient_clipper_by_norm() {
        let mut clipper = GradientClipper::by_norm(1.0);

        let mut param = TQParameter::new(ArrayD::zeros(IxDyn(&[2])), "test");
        param.grad = Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 4.0]).unwrap());

        // Gradient norm is 5.0, should be clipped to 1.0
        clipper.clip(&mut [param]).unwrap();

        assert!(clipper.was_clipped);
        assert!((clipper.last_norm.unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_clipper_by_value() {
        let mut clipper = GradientClipper::by_value(2.0);

        let mut param = TQParameter::new(ArrayD::zeros(IxDyn(&[2])), "test");
        param.grad = Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, -4.0]).unwrap());

        clipper.clip(&mut [param]).unwrap();

        assert!(clipper.was_clipped);
    }

    #[test]
    fn test_parameter_group() {
        let mut manager = ParameterGroupManager::new();

        let mut group1 = ParameterGroup::new("backbone")
            .with_lr_multiplier(0.1)
            .with_weight_decay(0.01);
        group1.add_param("layer1");
        group1.add_param("layer2");

        let mut group2 = ParameterGroup::new("head")
            .with_lr_multiplier(1.0)
            .with_weight_decay(0.0);
        group2.add_param("output");

        manager.add_group(group1);
        manager.add_group(group2);

        assert_eq!(manager.lr_multiplier("layer1"), 0.1);
        assert_eq!(manager.lr_multiplier("output"), 1.0);
        assert_eq!(manager.weight_decay("layer1"), 0.01);
        assert_eq!(manager.weight_decay("output"), 0.0);
    }

    #[test]
    fn test_gradient_statistics() {
        let mut param1 = TQParameter::new(ArrayD::zeros(IxDyn(&[2])), "p1");
        param1.grad = Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap());

        let mut param2 = TQParameter::new(ArrayD::zeros(IxDyn(&[2])), "p2");
        param2.grad = Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 4.0]).unwrap());

        let stats = gradient_statistics(&[param1, param2]);

        assert!((stats.mean - 2.5).abs() < 1e-10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 4.0);
    }
}
