//! Basic layers for Keras-like API

use super::{ActivationFunction, InitializerType, KerasLayer};
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{ArrayD, Axis, IxDyn};

/// Dense (fully connected) layer
pub struct Dense {
    /// Number of units
    units: usize,
    /// Activation function
    activation: Option<ActivationFunction>,
    /// Use bias
    use_bias: bool,
    /// Kernel initializer
    kernel_initializer: InitializerType,
    /// Bias initializer
    bias_initializer: InitializerType,
    /// Layer name
    name: String,
    /// Built flag
    built: bool,
    /// Input shape
    input_shape: Option<Vec<usize>>,
    /// Weights (kernel and bias)
    weights: Vec<ArrayD<f64>>,
}

impl Dense {
    /// Create new dense layer
    pub fn new(units: usize) -> Self {
        Self {
            units,
            activation: None,
            use_bias: true,
            kernel_initializer: InitializerType::GlorotUniform,
            bias_initializer: InitializerType::Zeros,
            name: format!("dense_{}", fastrand::u32(..)),
            built: false,
            input_shape: None,
            weights: Vec::new(),
        }
    }

    /// Set activation function
    pub fn activation(mut self, activation: ActivationFunction) -> Self {
        self.activation = Some(activation);
        self
    }

    /// Set use bias
    pub fn use_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Set layer name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set kernel initializer
    pub fn kernel_initializer(mut self, initializer: InitializerType) -> Self {
        self.kernel_initializer = initializer;
        self
    }

    /// Initialize weights
    fn initialize_weights(
        &self,
        shape: &[usize],
        initializer: &InitializerType,
    ) -> Result<ArrayD<f64>> {
        match initializer {
            InitializerType::Zeros => Ok(ArrayD::zeros(shape)),
            InitializerType::Ones => Ok(ArrayD::ones(shape)),
            InitializerType::GlorotUniform => {
                let fan_in = if shape.len() >= 2 { shape[0] } else { 1 };
                let fan_out = if shape.len() >= 2 { shape[1] } else { shape[0] };
                let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();

                Ok(ArrayD::from_shape_fn(shape, |_| {
                    fastrand::f64() * 2.0 * limit - limit
                }))
            }
            InitializerType::GlorotNormal => {
                let fan_in = if shape.len() >= 2 { shape[0] } else { 1 };
                let fan_out = if shape.len() >= 2 { shape[1] } else { shape[0] };
                let std = (2.0 / (fan_in + fan_out) as f64).sqrt();

                Ok(ArrayD::from_shape_fn(shape, |_| {
                    let u1 = fastrand::f64();
                    let u2 = fastrand::f64();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    z * std
                }))
            }
            InitializerType::HeUniform => {
                let fan_in = if shape.len() >= 2 { shape[0] } else { 1 };
                let limit = (6.0 / fan_in as f64).sqrt();

                Ok(ArrayD::from_shape_fn(shape, |_| {
                    fastrand::f64() * 2.0 * limit - limit
                }))
            }
        }
    }

    /// Apply activation function
    fn apply_activation(
        &self,
        inputs: &ArrayD<f64>,
        activation: &ActivationFunction,
    ) -> Result<ArrayD<f64>> {
        Ok(match activation {
            ActivationFunction::Linear => inputs.clone(),
            ActivationFunction::ReLU => inputs.mapv(|x| x.max(0.0)),
            ActivationFunction::Sigmoid => inputs.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationFunction::Tanh => inputs.mapv(|x| x.tanh()),
            ActivationFunction::Softmax => {
                let mut outputs = inputs.clone();
                for mut row in outputs.axis_iter_mut(Axis(0)) {
                    let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    row.mapv_inplace(|x| (x - max_val).exp());
                    let sum = row.sum();
                    row /= sum;
                }
                outputs
            }
            ActivationFunction::LeakyReLU(alpha) => {
                inputs.mapv(|x| if x > 0.0 { x } else { alpha * x })
            }
            ActivationFunction::ELU(alpha) => {
                inputs.mapv(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            }
        })
    }
}

impl KerasLayer for Dense {
    fn build(&mut self, input_shape: &[usize]) -> Result<()> {
        if input_shape.is_empty() {
            return Err(MLError::InvalidConfiguration(
                "Dense layer requires input shape".to_string(),
            ));
        }

        let input_dim = input_shape[input_shape.len() - 1];
        self.input_shape = Some(input_shape.to_vec());

        let kernel = self.initialize_weights(&[input_dim, self.units], &self.kernel_initializer)?;
        self.weights.push(kernel);

        if self.use_bias {
            let bias = self.initialize_weights(&[self.units], &self.bias_initializer)?;
            self.weights.push(bias);
        }

        self.built = true;
        Ok(())
    }

    fn call(&self, inputs: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        if !self.built {
            return Err(MLError::InvalidConfiguration(
                "Layer must be built before calling".to_string(),
            ));
        }

        let kernel = &self.weights[0];
        let outputs = match (inputs.ndim(), kernel.ndim()) {
            (2, 2) => {
                let inputs_2d = inputs
                    .clone()
                    .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                    .map_err(|_| MLError::InvalidConfiguration("Input must be 2D".to_string()))?;
                let kernel_2d = kernel
                    .clone()
                    .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                    .map_err(|_| MLError::InvalidConfiguration("Kernel must be 2D".to_string()))?;
                inputs_2d.dot(&kernel_2d).into_dyn()
            }
            _ => {
                return Err(MLError::InvalidConfiguration(
                    "Unsupported array dimensions for matrix multiplication".to_string(),
                ));
            }
        };
        let mut outputs = outputs;

        if self.use_bias && self.weights.len() > 1 {
            let bias = &self.weights[1];
            outputs = outputs + bias;
        }

        if let Some(ref activation) = self.activation {
            outputs = self.apply_activation(&outputs, activation)?;
        }

        Ok(outputs)
    }

    fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let mut output_shape = input_shape.to_vec();
        let last_idx = output_shape.len() - 1;
        output_shape[last_idx] = self.units;
        output_shape
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn get_weights(&self) -> Vec<ArrayD<f64>> {
        self.weights.clone()
    }

    fn set_weights(&mut self, weights: Vec<ArrayD<f64>>) -> Result<()> {
        if weights.len() != self.weights.len() {
            return Err(MLError::InvalidConfiguration(
                "Number of weight arrays doesn't match layer structure".to_string(),
            ));
        }
        self.weights = weights;
        Ok(())
    }

    fn built(&self) -> bool {
        self.built
    }
}

/// Activation layer
pub struct Activation {
    /// Activation function
    function: ActivationFunction,
    /// Layer name
    name: String,
    /// Built flag
    built: bool,
}

impl Activation {
    /// Create new activation layer
    pub fn new(function: ActivationFunction) -> Self {
        Self {
            function,
            name: format!("activation_{}", fastrand::u32(..)),
            built: false,
        }
    }

    /// Set layer name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl KerasLayer for Activation {
    fn build(&mut self, _input_shape: &[usize]) -> Result<()> {
        self.built = true;
        Ok(())
    }

    fn call(&self, inputs: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        Ok(match &self.function {
            ActivationFunction::Linear => inputs.clone(),
            ActivationFunction::ReLU => inputs.mapv(|x| x.max(0.0)),
            ActivationFunction::Sigmoid => inputs.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationFunction::Tanh => inputs.mapv(|x| x.tanh()),
            ActivationFunction::Softmax => {
                let mut outputs = inputs.clone();
                for mut row in outputs.axis_iter_mut(Axis(0)) {
                    let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    row.mapv_inplace(|x| (x - max_val).exp());
                    let sum = row.sum();
                    row /= sum;
                }
                outputs
            }
            ActivationFunction::LeakyReLU(alpha) => {
                inputs.mapv(|x| if x > 0.0 { x } else { alpha * x })
            }
            ActivationFunction::ELU(alpha) => {
                inputs.mapv(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            }
        })
    }

    fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        input_shape.to_vec()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn get_weights(&self) -> Vec<ArrayD<f64>> {
        Vec::new()
    }

    fn set_weights(&mut self, _weights: Vec<ArrayD<f64>>) -> Result<()> {
        Ok(())
    }

    fn built(&self) -> bool {
        self.built
    }
}

/// Dropout layer for regularization
pub struct Dropout {
    /// Dropout rate (0 to 1)
    rate: f64,
    /// Layer name
    name: String,
    /// Built flag
    built: bool,
    /// Training mode
    training: bool,
}

impl Dropout {
    /// Create new dropout layer
    pub fn new(rate: f64) -> Self {
        Self {
            rate: rate.clamp(0.0, 1.0),
            name: format!("dropout_{}", fastrand::u32(..)),
            built: false,
            training: true,
        }
    }

    /// Set layer name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl KerasLayer for Dropout {
    fn build(&mut self, _input_shape: &[usize]) -> Result<()> {
        self.built = true;
        Ok(())
    }

    fn call(&self, inputs: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        if !self.training || self.rate == 0.0 {
            return Ok(inputs.clone());
        }

        let scale = 1.0 / (1.0 - self.rate);
        let output = inputs.mapv(|x| {
            if fastrand::f64() < self.rate {
                0.0
            } else {
                x * scale
            }
        });

        Ok(output)
    }

    fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        input_shape.to_vec()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn get_weights(&self) -> Vec<ArrayD<f64>> {
        Vec::new()
    }

    fn set_weights(&mut self, _weights: Vec<ArrayD<f64>>) -> Result<()> {
        Ok(())
    }

    fn built(&self) -> bool {
        self.built
    }
}

/// Batch normalization layer
pub struct BatchNormalization {
    /// Momentum for moving average
    momentum: f64,
    /// Epsilon for numerical stability
    epsilon: f64,
    /// Use center (beta)
    center: bool,
    /// Use scale (gamma)
    scale: bool,
    /// Layer name
    name: String,
    /// Built flag
    built: bool,
    /// Weights: [gamma, beta, moving_mean, moving_var]
    weights: Vec<ArrayD<f64>>,
    /// Training mode
    training: bool,
}

impl BatchNormalization {
    /// Create new batch normalization layer
    pub fn new() -> Self {
        Self {
            momentum: 0.99,
            epsilon: 1e-3,
            center: true,
            scale: true,
            name: format!("batch_norm_{}", fastrand::u32(..)),
            built: false,
            weights: Vec::new(),
            training: true,
        }
    }

    /// Set momentum
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set epsilon
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set layer name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl Default for BatchNormalization {
    fn default() -> Self {
        Self::new()
    }
}

impl KerasLayer for BatchNormalization {
    fn build(&mut self, input_shape: &[usize]) -> Result<()> {
        let features = input_shape[input_shape.len() - 1];

        let gamma = ArrayD::ones(IxDyn(&[features]));
        self.weights.push(gamma);

        let beta = ArrayD::zeros(IxDyn(&[features]));
        self.weights.push(beta);

        let moving_mean = ArrayD::zeros(IxDyn(&[features]));
        self.weights.push(moving_mean);

        let moving_var = ArrayD::ones(IxDyn(&[features]));
        self.weights.push(moving_var);

        self.built = true;
        Ok(())
    }

    fn call(&self, inputs: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        if !self.built {
            return Err(MLError::InvalidConfiguration("Layer not built".to_string()));
        }

        let gamma = &self.weights[0];
        let beta = &self.weights[1];
        let moving_mean = &self.weights[2];
        let moving_var = &self.weights[3];

        let shape = inputs.shape();
        let features = shape[shape.len() - 1];

        let mut output = inputs.clone();

        for (i, val) in output.iter_mut().enumerate() {
            let f = i % features;
            let mean = moving_mean[[f]];
            let var = moving_var[[f]];
            let std = (var + self.epsilon).sqrt();

            *val = (*val - mean) / std;

            if self.scale {
                *val *= gamma[[f]];
            }
            if self.center {
                *val += beta[[f]];
            }
        }

        Ok(output)
    }

    fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        input_shape.to_vec()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn get_weights(&self) -> Vec<ArrayD<f64>> {
        self.weights.clone()
    }

    fn set_weights(&mut self, weights: Vec<ArrayD<f64>>) -> Result<()> {
        if weights.len() != 4 {
            return Err(MLError::InvalidConfiguration(
                "BatchNormalization requires 4 weight arrays".to_string(),
            ));
        }
        self.weights = weights;
        Ok(())
    }

    fn built(&self) -> bool {
        self.built
    }
}

/// Flatten layer to reshape inputs
pub struct Flatten {
    /// Layer name
    name: String,
    /// Built flag
    built: bool,
    /// Input shape
    input_shape: Option<Vec<usize>>,
}

impl Flatten {
    /// Create new flatten layer
    pub fn new() -> Self {
        Self {
            name: format!("flatten_{}", fastrand::u32(..)),
            built: false,
            input_shape: None,
        }
    }

    /// Set layer name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl KerasLayer for Flatten {
    fn build(&mut self, input_shape: &[usize]) -> Result<()> {
        self.input_shape = Some(input_shape.to_vec());
        self.built = true;
        Ok(())
    }

    fn call(&self, inputs: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        let shape = inputs.shape();
        if shape.is_empty() {
            return Ok(inputs.clone());
        }

        let batch_size = shape[0];
        let flat_size: usize = shape[1..].iter().product();

        let output = inputs
            .clone()
            .into_shape(IxDyn(&[batch_size, flat_size]))
            .map_err(|e| MLError::InvalidConfiguration(format!("Reshape failed: {}", e)))?;

        Ok(output)
    }

    fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        if input_shape.is_empty() {
            return vec![];
        }
        let flat_size: usize = input_shape[1..].iter().product();
        vec![input_shape[0], flat_size]
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn get_weights(&self) -> Vec<ArrayD<f64>> {
        Vec::new()
    }

    fn set_weights(&mut self, _weights: Vec<ArrayD<f64>>) -> Result<()> {
        Ok(())
    }

    fn built(&self) -> bool {
        self.built
    }
}
