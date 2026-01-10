//! Convolutional layers for Keras-like API

use super::{ActivationFunction, KerasLayer};
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{ArrayD, IxDyn};

/// Conv2D layer (Keras-compatible)
pub struct Conv2D {
    /// Number of filters
    filters: usize,
    /// Kernel size
    kernel_size: (usize, usize),
    /// Stride
    strides: (usize, usize),
    /// Padding
    padding: String,
    /// Activation
    activation: Option<ActivationFunction>,
    /// Use bias
    use_bias: bool,
    /// Weights
    kernel: Option<ArrayD<f64>>,
    /// Bias
    bias: Option<ArrayD<f64>>,
    /// Built flag
    built: bool,
    /// Layer name
    layer_name: Option<String>,
}

impl Conv2D {
    /// Create new Conv2D layer
    pub fn new(filters: usize, kernel_size: (usize, usize)) -> Self {
        Self {
            filters,
            kernel_size,
            strides: (1, 1),
            padding: "valid".to_string(),
            activation: None,
            use_bias: true,
            kernel: None,
            bias: None,
            built: false,
            layer_name: None,
        }
    }

    /// Set strides
    pub fn strides(mut self, strides: (usize, usize)) -> Self {
        self.strides = strides;
        self
    }

    /// Set padding
    pub fn padding(mut self, padding: &str) -> Self {
        self.padding = padding.to_string();
        self
    }

    /// Set activation
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
    pub fn name(mut self, name: &str) -> Self {
        self.layer_name = Some(name.to_string());
        self
    }
}

impl KerasLayer for Conv2D {
    fn call(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        if !self.built {
            return Err(MLError::ModelNotTrained(
                "Layer not built. Call build() first.".to_string(),
            ));
        }

        let kernel = self
            .kernel
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Conv2D kernel not initialized".to_string()))?;

        let shape = input.shape();
        let (batch, height, width, _in_channels) = (shape[0], shape[1], shape[2], shape[3]);

        let (pad_h, pad_w) = if self.padding == "same" {
            (self.kernel_size.0 / 2, self.kernel_size.1 / 2)
        } else {
            (0, 0)
        };

        let out_h = (height + 2 * pad_h - self.kernel_size.0) / self.strides.0 + 1;
        let out_w = (width + 2 * pad_w - self.kernel_size.1) / self.strides.1 + 1;

        let mut output = ArrayD::zeros(IxDyn(&[batch, out_h, out_w, self.filters]));

        for b in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for f in 0..self.filters {
                        let mut sum = if self.use_bias {
                            self.bias.as_ref().map_or(0.0, |bias| bias[[f]])
                        } else {
                            0.0
                        };

                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let ih = oh * self.strides.0 + kh;
                                let iw = ow * self.strides.1 + kw;
                                if ih < height && iw < width {
                                    for ic in 0..shape[3] {
                                        sum += input[[b, ih, iw, ic]] * kernel[[kh, kw, ic, f]];
                                    }
                                }
                            }
                        }
                        output[[b, oh, ow, f]] = sum;
                    }
                }
            }
        }

        if let Some(ref activation) = self.activation {
            output = output.mapv(|x| match activation {
                ActivationFunction::ReLU => x.max(0.0),
                ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                ActivationFunction::Tanh => x.tanh(),
                ActivationFunction::Softmax => x,
                ActivationFunction::LeakyReLU(alpha) => {
                    if x > 0.0 {
                        x
                    } else {
                        alpha * x
                    }
                }
                ActivationFunction::ELU(alpha) => {
                    if x > 0.0 {
                        x
                    } else {
                        alpha * (x.exp() - 1.0)
                    }
                }
                ActivationFunction::Linear => x,
            });
        }

        Ok(output)
    }

    fn build(&mut self, input_shape: &[usize]) -> Result<()> {
        let in_channels = *input_shape
            .last()
            .ok_or_else(|| MLError::InvalidConfiguration("Invalid input shape".to_string()))?;

        let scale = (2.0 / ((self.kernel_size.0 * self.kernel_size.1 * in_channels) as f64)).sqrt();
        let kernel = ArrayD::from_shape_fn(
            IxDyn(&[
                self.kernel_size.0,
                self.kernel_size.1,
                in_channels,
                self.filters,
            ]),
            |_| fastrand::f64() * 2.0 * scale - scale,
        );

        self.kernel = Some(kernel);

        if self.use_bias {
            self.bias = Some(ArrayD::zeros(IxDyn(&[self.filters])));
        }

        self.built = true;
        Ok(())
    }

    fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let (height, width) = (input_shape[1], input_shape[2]);
        let (pad_h, pad_w) = if self.padding == "same" {
            (self.kernel_size.0 / 2, self.kernel_size.1 / 2)
        } else {
            (0, 0)
        };
        let out_h = (height + 2 * pad_h - self.kernel_size.0) / self.strides.0 + 1;
        let out_w = (width + 2 * pad_w - self.kernel_size.1) / self.strides.1 + 1;
        vec![input_shape[0], out_h, out_w, self.filters]
    }

    fn count_params(&self) -> usize {
        let kernel_params = self.kernel.as_ref().map_or(0, |k| k.len());
        let bias_params = self.bias.as_ref().map_or(0, |b| b.len());
        kernel_params + bias_params
    }

    fn get_weights(&self) -> Vec<ArrayD<f64>> {
        let mut weights = vec![];
        if let Some(ref k) = self.kernel {
            weights.push(k.clone());
        }
        if let Some(ref b) = self.bias {
            weights.push(b.clone());
        }
        weights
    }

    fn set_weights(&mut self, weights: Vec<ArrayD<f64>>) -> Result<()> {
        if !weights.is_empty() {
            self.kernel = Some(weights[0].clone());
        }
        if weights.len() > 1 {
            self.bias = Some(weights[1].clone());
        }
        Ok(())
    }

    fn built(&self) -> bool {
        self.built
    }

    fn name(&self) -> &str {
        self.layer_name.as_deref().unwrap_or("conv2d")
    }
}

/// MaxPooling2D layer
pub struct MaxPooling2D {
    /// Pool size
    pool_size: (usize, usize),
    /// Strides
    strides: (usize, usize),
    /// Padding
    padding: String,
    /// Built flag
    built: bool,
    /// Layer name
    layer_name: Option<String>,
}

impl MaxPooling2D {
    /// Create new MaxPooling2D layer
    pub fn new(pool_size: (usize, usize)) -> Self {
        Self {
            pool_size,
            strides: pool_size,
            padding: "valid".to_string(),
            built: false,
            layer_name: None,
        }
    }

    /// Set strides
    pub fn strides(mut self, strides: (usize, usize)) -> Self {
        self.strides = strides;
        self
    }

    /// Set padding
    pub fn padding(mut self, padding: &str) -> Self {
        self.padding = padding.to_string();
        self
    }

    /// Set layer name
    pub fn name(mut self, name: &str) -> Self {
        self.layer_name = Some(name.to_string());
        self
    }
}

impl KerasLayer for MaxPooling2D {
    fn call(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        let shape = input.shape();
        let (batch, height, width, channels) = (shape[0], shape[1], shape[2], shape[3]);

        let out_h = (height - self.pool_size.0) / self.strides.0 + 1;
        let out_w = (width - self.pool_size.1) / self.strides.1 + 1;

        let mut output = ArrayD::zeros(IxDyn(&[batch, out_h, out_w, channels]));

        for b in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    for c in 0..channels {
                        let mut max_val = f64::NEG_INFINITY;
                        for ph in 0..self.pool_size.0 {
                            for pw in 0..self.pool_size.1 {
                                let ih = oh * self.strides.0 + ph;
                                let iw = ow * self.strides.1 + pw;
                                if ih < height && iw < width {
                                    max_val = max_val.max(input[[b, ih, iw, c]]);
                                }
                            }
                        }
                        output[[b, oh, ow, c]] = max_val;
                    }
                }
            }
        }

        Ok(output)
    }

    fn build(&mut self, _input_shape: &[usize]) -> Result<()> {
        self.built = true;
        Ok(())
    }

    fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let out_h = (input_shape[1] - self.pool_size.0) / self.strides.0 + 1;
        let out_w = (input_shape[2] - self.pool_size.1) / self.strides.1 + 1;
        vec![input_shape[0], out_h, out_w, input_shape[3]]
    }

    fn count_params(&self) -> usize {
        0
    }

    fn get_weights(&self) -> Vec<ArrayD<f64>> {
        vec![]
    }

    fn set_weights(&mut self, _weights: Vec<ArrayD<f64>>) -> Result<()> {
        Ok(())
    }

    fn built(&self) -> bool {
        self.built
    }

    fn name(&self) -> &str {
        self.layer_name.as_deref().unwrap_or("max_pooling2d")
    }
}

/// GlobalAveragePooling2D layer
pub struct GlobalAveragePooling2D {
    /// Built flag
    built: bool,
    /// Layer name
    layer_name: Option<String>,
}

impl GlobalAveragePooling2D {
    /// Create new GlobalAveragePooling2D
    pub fn new() -> Self {
        Self {
            built: false,
            layer_name: None,
        }
    }

    /// Set layer name
    pub fn name(mut self, name: &str) -> Self {
        self.layer_name = Some(name.to_string());
        self
    }
}

impl Default for GlobalAveragePooling2D {
    fn default() -> Self {
        Self::new()
    }
}

impl KerasLayer for GlobalAveragePooling2D {
    fn call(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        let shape = input.shape();
        let (batch, height, width, channels) = (shape[0], shape[1], shape[2], shape[3]);

        let mut output = ArrayD::zeros(IxDyn(&[batch, channels]));
        let count = (height * width) as f64;

        for b in 0..batch {
            for c in 0..channels {
                let mut sum = 0.0;
                for h in 0..height {
                    for w in 0..width {
                        sum += input[[b, h, w, c]];
                    }
                }
                output[[b, c]] = sum / count;
            }
        }

        Ok(output)
    }

    fn build(&mut self, _input_shape: &[usize]) -> Result<()> {
        self.built = true;
        Ok(())
    }

    fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        vec![input_shape[0], input_shape[3]]
    }

    fn count_params(&self) -> usize {
        0
    }

    fn get_weights(&self) -> Vec<ArrayD<f64>> {
        vec![]
    }

    fn set_weights(&mut self, _weights: Vec<ArrayD<f64>>) -> Result<()> {
        Ok(())
    }

    fn built(&self) -> bool {
        self.built
    }

    fn name(&self) -> &str {
        self.layer_name
            .as_deref()
            .unwrap_or("global_average_pooling2d")
    }
}
