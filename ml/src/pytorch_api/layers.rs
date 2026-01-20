//! Basic layers for PyTorch-like API
//!
//! This module contains fundamental layers: Linear, Conv2d, Activations,
//! Normalization, Dropout, Pooling, and Embedding.

use super::{Parameter, QuantumModule};
use crate::circuit_integration::QuantumMLExecutor;
use crate::error::{MLError, Result};
use crate::scirs2_integration::SciRS2Array;
use scirs2_core::ndarray::{ArrayD, IxDyn};

// ============================================================================
// Linear Layer
// ============================================================================

/// Quantum linear layer
pub struct QuantumLinear {
    /// Weight parameters
    weights: Parameter,
    /// Bias parameters (optional)
    bias: Option<Parameter>,
    /// Input features
    pub in_features: usize,
    /// Output features
    pub out_features: usize,
    /// Training mode
    training: bool,
    /// Circuit executor
    executor: QuantumMLExecutor<8>,
}

impl QuantumLinear {
    /// Create new quantum linear layer
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        let weight_data = ArrayD::zeros(IxDyn(&[out_features, in_features]));
        let weights = Parameter::new(SciRS2Array::with_grad(weight_data), "weight");

        Ok(Self {
            weights,
            bias: None,
            in_features,
            out_features,
            training: true,
            executor: QuantumMLExecutor::new(),
        })
    }

    /// Create with bias
    pub fn with_bias(mut self) -> Result<Self> {
        let bias_data = ArrayD::zeros(IxDyn(&[self.out_features]));
        self.bias = Some(Parameter::new(SciRS2Array::with_grad(bias_data), "bias"));
        Ok(self)
    }

    /// Initialize weights using Xavier/Glorot uniform
    pub fn init_xavier_uniform(&mut self) -> Result<()> {
        let fan_in = self.in_features as f64;
        let fan_out = self.out_features as f64;
        let bound = (6.0 / (fan_in + fan_out)).sqrt();

        for elem in self.weights.data.data.iter_mut() {
            *elem = (fastrand::f64() * 2.0 - 1.0) * bound;
        }

        Ok(())
    }
}

impl QuantumModule for QuantumLinear {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        let output = input.matmul(&self.weights.data)?;

        if let Some(ref bias) = self.bias {
            output.add(&bias.data)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weights.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {
        self.weights.data.zero_grad();
        if let Some(ref mut bias) = self.bias {
            bias.data.zero_grad();
        }
    }

    fn name(&self) -> &str {
        "QuantumLinear"
    }
}

// ============================================================================
// Conv2d Layer
// ============================================================================

/// Quantum convolutional layer
pub struct QuantumConv2d {
    /// Convolution parameters
    weights: Parameter,
    /// Bias parameters
    bias: Option<Parameter>,
    /// Input channels
    pub in_channels: usize,
    /// Output channels
    pub out_channels: usize,
    /// Kernel size
    pub kernel_size: (usize, usize),
    /// Stride
    pub stride: (usize, usize),
    /// Padding
    pub padding: (usize, usize),
    /// Training mode
    training: bool,
}

impl QuantumConv2d {
    /// Create new quantum conv2d layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Result<Self> {
        let weight_shape = [out_channels, in_channels, kernel_size.0, kernel_size.1];
        let weight_data = ArrayD::zeros(IxDyn(&weight_shape));
        let weights = Parameter::new(SciRS2Array::with_grad(weight_data), "weight");

        Ok(Self {
            weights,
            bias: None,
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1),
            padding: (0, 0),
            training: true,
        })
    }

    /// Set stride
    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding
    pub fn padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Add bias
    pub fn with_bias(mut self) -> Result<Self> {
        let bias_data = ArrayD::zeros(IxDyn(&[self.out_channels]));
        self.bias = Some(Parameter::new(SciRS2Array::with_grad(bias_data), "bias"));
        Ok(self)
    }
}

impl QuantumModule for QuantumConv2d {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        let output_data = input.data.clone();
        let mut output = SciRS2Array::new(output_data, input.requires_grad);

        if let Some(ref bias) = self.bias {
            output = output.add(&bias.data)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weights.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {
        self.weights.data.zero_grad();
        if let Some(ref mut bias) = self.bias {
            bias.data.zero_grad();
        }
    }

    fn name(&self) -> &str {
        "QuantumConv2d"
    }
}

// ============================================================================
// Activation Functions
// ============================================================================

/// Activation function types
#[derive(Debug, Clone)]
pub enum ActivationType {
    /// Quantum ReLU (using rotation gates)
    QReLU,
    /// Quantum Sigmoid
    QSigmoid,
    /// Quantum Tanh
    QTanh,
    /// Quantum Softmax
    QSoftmax,
    /// Identity (no activation)
    Identity,
}

/// Quantum activation functions
pub struct QuantumActivation {
    /// Activation function type
    activation_type: ActivationType,
    /// Training mode
    training: bool,
}

impl QuantumActivation {
    /// Create new activation layer
    pub fn new(activation_type: ActivationType) -> Self {
        Self {
            activation_type,
            training: true,
        }
    }

    /// Create ReLU activation
    pub fn relu() -> Self {
        Self::new(ActivationType::QReLU)
    }

    /// Create Sigmoid activation
    pub fn sigmoid() -> Self {
        Self::new(ActivationType::QSigmoid)
    }

    /// Create Tanh activation
    pub fn tanh() -> Self {
        Self::new(ActivationType::QTanh)
    }

    /// Create Softmax activation
    pub fn softmax() -> Self {
        Self::new(ActivationType::QSoftmax)
    }
}

impl QuantumModule for QuantumActivation {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        match self.activation_type {
            ActivationType::QReLU => {
                let output_data = input.data.mapv(|x| x.max(0.0));
                Ok(SciRS2Array::new(output_data, input.requires_grad))
            }
            ActivationType::QSigmoid => {
                let output_data = input.data.mapv(|x| 1.0 / (1.0 + (-x).exp()));
                Ok(SciRS2Array::new(output_data, input.requires_grad))
            }
            ActivationType::QTanh => {
                let output_data = input.data.mapv(|x| x.tanh());
                Ok(SciRS2Array::new(output_data, input.requires_grad))
            }
            ActivationType::QSoftmax => {
                let max_val = input.data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let exp_data = input.data.mapv(|x| (x - max_val).exp());
                let sum_exp = exp_data.sum();
                let output_data = exp_data.mapv(|x| x / sum_exp);
                Ok(SciRS2Array::new(output_data, input.requires_grad))
            }
            ActivationType::Identity => {
                Ok(SciRS2Array::new(input.data.clone(), input.requires_grad))
            }
        }
    }

    fn parameters(&self) -> Vec<Parameter> {
        Vec::new()
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {}

    fn name(&self) -> &str {
        "QuantumActivation"
    }
}

// ============================================================================
// Extended Activation Functions
// ============================================================================

/// Parameter initialization types
#[derive(Debug, Clone, Copy)]
pub enum InitType {
    /// Xavier/Glorot initialization
    Xavier,
    /// He initialization
    He,
    /// Normal distribution
    Normal(f64, f64),
    /// Uniform distribution
    Uniform(f64, f64),
}

/// Extended activation function types
#[derive(Debug, Clone)]
pub enum ExtendedActivation {
    /// GELU activation
    GELU,
    /// ELU activation
    ELU { alpha: f64 },
    /// LeakyReLU activation
    LeakyReLU { negative_slope: f64 },
    /// SiLU/Swish activation
    SiLU,
    /// PReLU activation
    PReLU { num_parameters: usize },
    /// Softplus activation
    Softplus { beta: f64 },
    /// Mish activation
    Mish,
    /// Hardswish activation
    Hardswish,
    /// Hardsigmoid activation
    Hardsigmoid,
}

/// Extended activation layer
pub struct QuantumExtendedActivation {
    activation: ExtendedActivation,
    prelu_weights: Option<Parameter>,
    training: bool,
}

impl QuantumExtendedActivation {
    /// Create new extended activation
    pub fn new(activation: ExtendedActivation) -> Self {
        let prelu_weights = match &activation {
            ExtendedActivation::PReLU { num_parameters } => {
                let data = ArrayD::from_elem(IxDyn(&[*num_parameters]), 0.25);
                Some(Parameter::new(SciRS2Array::with_grad(data), "weight"))
            }
            _ => None,
        };

        Self {
            activation,
            prelu_weights,
            training: true,
        }
    }

    /// Create GELU activation
    pub fn gelu() -> Self {
        Self::new(ExtendedActivation::GELU)
    }

    /// Create ELU activation
    pub fn elu(alpha: f64) -> Self {
        Self::new(ExtendedActivation::ELU { alpha })
    }

    /// Create LeakyReLU activation
    pub fn leaky_relu(negative_slope: f64) -> Self {
        Self::new(ExtendedActivation::LeakyReLU { negative_slope })
    }

    /// Create SiLU/Swish activation
    pub fn silu() -> Self {
        Self::new(ExtendedActivation::SiLU)
    }

    /// Create Mish activation
    pub fn mish() -> Self {
        Self::new(ExtendedActivation::Mish)
    }
}

impl QuantumModule for QuantumExtendedActivation {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        let output_data = match &self.activation {
            ExtendedActivation::GELU => input.data.mapv(|x| {
                let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();
                0.5 * x * (1.0 + (sqrt_2_pi * (x + 0.044715 * x.powi(3))).tanh())
            }),
            ExtendedActivation::ELU { alpha } => {
                let a = *alpha;
                input
                    .data
                    .mapv(|x| if x >= 0.0 { x } else { a * (x.exp() - 1.0) })
            }
            ExtendedActivation::LeakyReLU { negative_slope } => {
                let slope = *negative_slope;
                input.data.mapv(|x| if x >= 0.0 { x } else { slope * x })
            }
            ExtendedActivation::SiLU => input.data.mapv(|x| x / (1.0 + (-x).exp())),
            ExtendedActivation::PReLU { .. } => {
                if let Some(ref weights) = self.prelu_weights {
                    let weight = weights.data.data[[0]];
                    input.data.mapv(|x| if x >= 0.0 { x } else { weight * x })
                } else {
                    input.data.mapv(|x| if x >= 0.0 { x } else { 0.25 * x })
                }
            }
            ExtendedActivation::Softplus { beta } => {
                let b = *beta;
                input.data.mapv(|x| (1.0 / b) * (1.0 + (b * x).exp()).ln())
            }
            ExtendedActivation::Mish => input.data.mapv(|x| x * ((1.0 + x.exp()).ln()).tanh()),
            ExtendedActivation::Hardswish => input.data.mapv(|x| {
                if x <= -3.0 {
                    0.0
                } else if x >= 3.0 {
                    x
                } else {
                    x * (x + 3.0) / 6.0
                }
            }),
            ExtendedActivation::Hardsigmoid => input.data.mapv(|x| {
                if x <= -3.0 {
                    0.0
                } else if x >= 3.0 {
                    1.0
                } else {
                    (x + 3.0) / 6.0
                }
            }),
        };
        Ok(SciRS2Array::new(output_data, input.requires_grad))
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.prelu_weights.iter().cloned().collect()
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut weights) = self.prelu_weights {
            weights.data.zero_grad();
        }
    }

    fn name(&self) -> &str {
        "ExtendedActivation"
    }
}

// ============================================================================
// Normalization Layers
// ============================================================================

/// Batch normalization layer
pub struct QuantumBatchNorm1d {
    num_features: usize,
    running_mean: Parameter,
    running_var: Parameter,
    weight: Parameter,
    bias: Parameter,
    eps: f64,
    momentum: f64,
    training: bool,
}

impl QuantumBatchNorm1d {
    /// Create new batch normalization layer
    pub fn new(num_features: usize) -> Self {
        let weight_data = ArrayD::ones(IxDyn(&[num_features]));
        let bias_data = ArrayD::zeros(IxDyn(&[num_features]));
        let mean_data = ArrayD::zeros(IxDyn(&[num_features]));
        let var_data = ArrayD::ones(IxDyn(&[num_features]));

        Self {
            num_features,
            running_mean: Parameter::no_grad(SciRS2Array::new(mean_data, false), "running_mean"),
            running_var: Parameter::no_grad(SciRS2Array::new(var_data, false), "running_var"),
            weight: Parameter::new(SciRS2Array::with_grad(weight_data), "weight"),
            bias: Parameter::new(SciRS2Array::with_grad(bias_data), "bias"),
            eps: 1e-5,
            momentum: 0.1,
            training: true,
        }
    }

    /// Set epsilon
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set momentum
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }
}

impl QuantumModule for QuantumBatchNorm1d {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        let shape = input.data.shape();
        if shape.len() < 2 || shape[1] != self.num_features {
            return Err(MLError::InvalidConfiguration(format!(
                "Expected {} features, got {:?}",
                self.num_features, shape
            )));
        }

        let batch_size = shape[0];
        let mut output = input.data.clone();

        if self.training {
            for f in 0..self.num_features {
                let mut sum = 0.0;
                for b in 0..batch_size {
                    sum += input.data[[b, f]];
                }
                let mean = sum / batch_size as f64;

                let mut var_sum = 0.0;
                for b in 0..batch_size {
                    let diff = input.data[[b, f]] - mean;
                    var_sum += diff * diff;
                }
                let var = var_sum / batch_size as f64;

                self.running_mean.data.data[[f]] =
                    (1.0 - self.momentum) * self.running_mean.data.data[[f]] + self.momentum * mean;
                self.running_var.data.data[[f]] =
                    (1.0 - self.momentum) * self.running_var.data.data[[f]] + self.momentum * var;

                let std = (var + self.eps).sqrt();
                for b in 0..batch_size {
                    output[[b, f]] = (input.data[[b, f]] - mean) / std;
                    output[[b, f]] =
                        output[[b, f]] * self.weight.data.data[[f]] + self.bias.data.data[[f]];
                }
            }
        } else {
            for f in 0..self.num_features {
                let mean = self.running_mean.data.data[[f]];
                let var = self.running_var.data.data[[f]];
                let std = (var + self.eps).sqrt();

                for b in 0..batch_size {
                    output[[b, f]] = (input.data[[b, f]] - mean) / std;
                    output[[b, f]] =
                        output[[b, f]] * self.weight.data.data[[f]] + self.bias.data.data[[f]];
                }
            }
        }

        Ok(SciRS2Array::new(output, input.requires_grad))
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {
        self.weight.data.zero_grad();
        self.bias.data.zero_grad();
    }

    fn name(&self) -> &str {
        "BatchNorm1d"
    }
}

/// Layer normalization
pub struct QuantumLayerNorm {
    normalized_shape: Vec<usize>,
    weight: Parameter,
    bias: Parameter,
    eps: f64,
    training: bool,
}

impl QuantumLayerNorm {
    /// Create new layer normalization
    pub fn new(normalized_shape: Vec<usize>) -> Self {
        let size: usize = normalized_shape.iter().product();
        let weight_data = ArrayD::ones(IxDyn(&[size]));
        let bias_data = ArrayD::zeros(IxDyn(&[size]));

        Self {
            normalized_shape,
            weight: Parameter::new(SciRS2Array::with_grad(weight_data), "weight"),
            bias: Parameter::new(SciRS2Array::with_grad(bias_data), "bias"),
            eps: 1e-5,
            training: true,
        }
    }
}

impl QuantumModule for QuantumLayerNorm {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        let mean: f64 = input.data.iter().sum::<f64>() / input.data.len() as f64;
        let var: f64 =
            input.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / input.data.len() as f64;
        let std = (var + self.eps).sqrt();

        let mut output = input.data.mapv(|x| (x - mean) / std);

        for (i, val) in output.iter_mut().enumerate() {
            let idx = i % self.weight.data.data.len();
            *val = *val * self.weight.data.data[[idx]] + self.bias.data.data[[idx]];
        }

        Ok(SciRS2Array::new(output, input.requires_grad))
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {
        self.weight.data.zero_grad();
        self.bias.data.zero_grad();
    }

    fn name(&self) -> &str {
        "LayerNorm"
    }
}

// ============================================================================
// Dropout Layers
// ============================================================================

/// Dropout layer
pub struct QuantumDropout {
    p: f64,
    training: bool,
}

impl QuantumDropout {
    /// Create new dropout layer
    pub fn new(p: f64) -> Self {
        Self { p, training: true }
    }
}

impl QuantumModule for QuantumDropout {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let scale = 1.0 / (1.0 - self.p);
        let output = input.data.mapv(|x| {
            if fastrand::f64() < self.p {
                0.0
            } else {
                x * scale
            }
        });

        Ok(SciRS2Array::new(output, input.requires_grad))
    }

    fn parameters(&self) -> Vec<Parameter> {
        Vec::new()
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {}

    fn name(&self) -> &str {
        "Dropout"
    }
}

/// Dropout2d for convolutional layers
pub struct QuantumDropout2d {
    p: f64,
    training: bool,
}

impl QuantumDropout2d {
    /// Create new dropout2d layer
    pub fn new(p: f64) -> Self {
        Self { p, training: true }
    }
}

impl QuantumModule for QuantumDropout2d {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let scale = 1.0 / (1.0 - self.p);
        let output = input.data.mapv(|x| {
            if fastrand::f64() < self.p {
                0.0
            } else {
                x * scale
            }
        });

        Ok(SciRS2Array::new(output, input.requires_grad))
    }

    fn parameters(&self) -> Vec<Parameter> {
        Vec::new()
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {}

    fn name(&self) -> &str {
        "Dropout2d"
    }
}

// ============================================================================
// Pooling Layers
// ============================================================================

/// Max pooling 2D
pub struct QuantumMaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    training: bool,
}

impl QuantumMaxPool2d {
    /// Create new max pooling layer
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: (0, 0),
            training: true,
        }
    }

    /// Set stride
    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding
    pub fn padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }
}

impl QuantumModule for QuantumMaxPool2d {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        let shape = input.data.shape();
        if shape.len() != 4 {
            return Err(MLError::InvalidConfiguration(
                "MaxPool2d expects 4D input (batch, channels, height, width)".to_string(),
            ));
        }

        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let out_height = (height + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let out_width = (width + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;

        let mut output = ArrayD::zeros(IxDyn(&[batch, channels, out_height, out_width]));

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let h_start = oh * self.stride.0;
                        let w_start = ow * self.stride.1;

                        let mut max_val = f64::NEG_INFINITY;
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let h = h_start + kh;
                                let w = w_start + kw;
                                if h < height && w < width {
                                    max_val = max_val.max(input.data[[b, c, h, w]]);
                                }
                            }
                        }
                        output[[b, c, oh, ow]] = max_val;
                    }
                }
            }
        }

        Ok(SciRS2Array::new(output, input.requires_grad))
    }

    fn parameters(&self) -> Vec<Parameter> {
        Vec::new()
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {}

    fn name(&self) -> &str {
        "MaxPool2d"
    }
}

/// Average pooling 2D
pub struct QuantumAvgPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    training: bool,
}

impl QuantumAvgPool2d {
    /// Create new average pooling layer
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: (0, 0),
            training: true,
        }
    }

    /// Set stride
    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding
    pub fn padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }
}

impl QuantumModule for QuantumAvgPool2d {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        let shape = input.data.shape();
        if shape.len() != 4 {
            return Err(MLError::InvalidConfiguration(
                "AvgPool2d expects 4D input".to_string(),
            ));
        }

        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let out_height = (height + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let out_width = (width + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;

        let mut output = ArrayD::zeros(IxDyn(&[batch, channels, out_height, out_width]));

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let h_start = oh * self.stride.0;
                        let w_start = ow * self.stride.1;

                        let mut sum = 0.0;
                        let mut count = 0;
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let h = h_start + kh;
                                let w = w_start + kw;
                                if h < height && w < width {
                                    sum += input.data[[b, c, h, w]];
                                    count += 1;
                                }
                            }
                        }
                        output[[b, c, oh, ow]] = if count > 0 { sum / count as f64 } else { 0.0 };
                    }
                }
            }
        }

        Ok(SciRS2Array::new(output, input.requires_grad))
    }

    fn parameters(&self) -> Vec<Parameter> {
        Vec::new()
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {}

    fn name(&self) -> &str {
        "AvgPool2d"
    }
}

/// Adaptive average pooling 2D
pub struct QuantumAdaptiveAvgPool2d {
    output_size: (usize, usize),
    training: bool,
}

impl QuantumAdaptiveAvgPool2d {
    /// Create new adaptive average pooling layer
    pub fn new(output_size: (usize, usize)) -> Self {
        Self {
            output_size,
            training: true,
        }
    }
}

impl QuantumModule for QuantumAdaptiveAvgPool2d {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        let shape = input.data.shape();
        if shape.len() != 4 {
            return Err(MLError::InvalidConfiguration(
                "AdaptiveAvgPool2d expects 4D input".to_string(),
            ));
        }

        let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let (out_h, out_w) = self.output_size;

        let mut output = ArrayD::zeros(IxDyn(&[batch, channels, out_h, out_w]));

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let h_start = (oh * height) / out_h;
                        let h_end = ((oh + 1) * height) / out_h;
                        let w_start = (ow * width) / out_w;
                        let w_end = ((ow + 1) * width) / out_w;

                        let mut sum = 0.0;
                        let mut count = 0;
                        for h in h_start..h_end {
                            for w in w_start..w_end {
                                sum += input.data[[b, c, h, w]];
                                count += 1;
                            }
                        }
                        output[[b, c, oh, ow]] = if count > 0 { sum / count as f64 } else { 0.0 };
                    }
                }
            }
        }

        Ok(SciRS2Array::new(output, input.requires_grad))
    }

    fn parameters(&self) -> Vec<Parameter> {
        Vec::new()
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {}

    fn name(&self) -> &str {
        "AdaptiveAvgPool2d"
    }
}

// ============================================================================
// Embedding Layer
// ============================================================================

/// Embedding layer for discrete inputs
pub struct QuantumEmbedding {
    num_embeddings: usize,
    embedding_dim: usize,
    weight: Parameter,
    padding_idx: Option<usize>,
    training: bool,
}

impl QuantumEmbedding {
    /// Create new embedding layer
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let weight_data = ArrayD::zeros(IxDyn(&[num_embeddings, embedding_dim]));
        let mut weight = Parameter::new(SciRS2Array::with_grad(weight_data), "weight");

        for val in weight.data.data.iter_mut() {
            *val = fastrand::f64() * 2.0 - 1.0;
        }

        Self {
            num_embeddings,
            embedding_dim,
            weight,
            padding_idx: None,
            training: true,
        }
    }

    /// Set padding index
    pub fn padding_idx(mut self, idx: usize) -> Self {
        self.padding_idx = Some(idx);
        for j in 0..self.embedding_dim {
            self.weight.data.data[[idx, j]] = 0.0;
        }
        self
    }

    /// Get embedding for indices
    pub fn get_embedding(&self, indices: &[usize]) -> Result<ArrayD<f64>> {
        let mut output = ArrayD::zeros(IxDyn(&[indices.len(), self.embedding_dim]));

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.num_embeddings {
                return Err(MLError::InvalidConfiguration(format!(
                    "Index {} out of range for {} embeddings",
                    idx, self.num_embeddings
                )));
            }
            for j in 0..self.embedding_dim {
                output[[i, j]] = self.weight.data.data[[idx, j]];
            }
        }

        Ok(output)
    }
}

impl QuantumModule for QuantumEmbedding {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        let indices: Vec<usize> = input.data.iter().map(|&x| x as usize).collect();
        let output = self.get_embedding(&indices)?;
        Ok(SciRS2Array::new(output, self.training))
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone()]
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {
        self.weight.data.zero_grad();
    }

    fn name(&self) -> &str {
        "Embedding"
    }
}

// ============================================================================
// Parameter Initialization
// ============================================================================

/// Initialize parameters with specified method
pub fn init_weights(param: &mut Parameter, init_type: InitType) -> Result<()> {
    let shape = param.data.data.shape().to_vec();
    let fan_in = if shape.len() >= 2 { shape[1] } else { shape[0] };
    let fan_out = shape[0];

    match init_type {
        InitType::Xavier => {
            let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
            for elem in param.data.data.iter_mut() {
                *elem = (fastrand::f64() * 2.0 - 1.0) * bound;
            }
        }
        InitType::He => {
            let std = (2.0 / fan_in as f64).sqrt();
            for elem in param.data.data.iter_mut() {
                let u1: f64 = fastrand::f64();
                let u2: f64 = fastrand::f64();
                let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                *elem = normal * std;
            }
        }
        InitType::Normal(mean, std) => {
            for elem in param.data.data.iter_mut() {
                let u1: f64 = fastrand::f64();
                let u2: f64 = fastrand::f64();
                let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                *elem = mean + normal * std;
            }
        }
        InitType::Uniform(low, high) => {
            for elem in param.data.data.iter_mut() {
                *elem = low + (high - low) * fastrand::f64();
            }
        }
    }
    Ok(())
}
