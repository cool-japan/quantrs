//! Convolutional layers for PyTorch-like API (Conv1d, Conv3d)

use super::{Parameter, QuantumModule};
use crate::error::{MLError, Result};
use crate::scirs2_integration::SciRS2Array;
use scirs2_core::ndarray::{ArrayD, IxDyn};

/// 1D Convolution layer
pub struct QuantumConv1d {
    weights: Parameter,
    bias: Option<Parameter>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    training: bool,
}

impl QuantumConv1d {
    /// Create new Conv1d
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Result<Self> {
        let weight_data =
            ArrayD::from_shape_fn(IxDyn(&[out_channels, in_channels, kernel_size]), |_| {
                fastrand::f64() * 0.1 - 0.05
            });

        Ok(Self {
            weights: Parameter::new(SciRS2Array::with_grad(weight_data), "weight"),
            bias: None,
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
            training: true,
        })
    }

    /// Set stride
    pub fn stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding
    pub fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Add bias
    pub fn with_bias(mut self) -> Self {
        let bias_data = ArrayD::zeros(IxDyn(&[self.out_channels]));
        self.bias = Some(Parameter::new(SciRS2Array::with_grad(bias_data), "bias"));
        self
    }
}

impl QuantumModule for QuantumConv1d {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        let shape = input.data.shape();
        if shape.len() != 3 {
            return Err(MLError::InvalidConfiguration(
                "Conv1d expects 3D input (batch, channels, length)".to_string(),
            ));
        }

        let (batch, _, length) = (shape[0], shape[1], shape[2]);
        let out_length = (length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
            / self.stride
            + 1;

        let mut output = ArrayD::zeros(IxDyn(&[batch, self.out_channels, out_length]));

        for b in 0..batch {
            for oc in 0..self.out_channels {
                for ol in 0..out_length {
                    let mut sum = 0.0;
                    for ic in 0..self.in_channels {
                        for k in 0..self.kernel_size {
                            let il = ol * self.stride + k * self.dilation;
                            if il < length + self.padding && il >= self.padding {
                                let input_idx = il - self.padding;
                                if input_idx < length {
                                    sum += input.data[[b, ic, input_idx]]
                                        * self.weights.data.data[[oc, ic, k]];
                                }
                            }
                        }
                    }
                    output[[b, oc, ol]] = sum;
                }
            }
        }

        if let Some(ref bias) = self.bias {
            for b in 0..batch {
                for oc in 0..self.out_channels {
                    for ol in 0..out_length {
                        output[[b, oc, ol]] += bias.data.data[[oc]];
                    }
                }
            }
        }

        Ok(SciRS2Array::new(output, input.requires_grad))
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
        "Conv1d"
    }
}

/// 3D Convolution layer
pub struct QuantumConv3d {
    weights: Parameter,
    bias: Option<Parameter>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    training: bool,
}

impl QuantumConv3d {
    /// Create new Conv3d
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
    ) -> Result<Self> {
        let weight_data = ArrayD::from_shape_fn(
            IxDyn(&[
                out_channels,
                in_channels,
                kernel_size.0,
                kernel_size.1,
                kernel_size.2,
            ]),
            |_| fastrand::f64() * 0.1 - 0.05,
        );

        Ok(Self {
            weights: Parameter::new(SciRS2Array::with_grad(weight_data), "weight"),
            bias: None,
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            training: true,
        })
    }

    /// Set stride
    pub fn stride(mut self, stride: (usize, usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding
    pub fn padding(mut self, padding: (usize, usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Add bias
    pub fn with_bias(mut self) -> Self {
        let bias_data = ArrayD::zeros(IxDyn(&[self.out_channels]));
        self.bias = Some(Parameter::new(SciRS2Array::with_grad(bias_data), "bias"));
        self
    }
}

impl QuantumModule for QuantumConv3d {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        let shape = input.data.shape();
        if shape.len() != 5 {
            return Err(MLError::InvalidConfiguration(
                "Conv3d expects 5D input".to_string(),
            ));
        }

        let (batch, _, depth, height, width) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let out_d = (depth + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let out_h = (height + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
        let out_w = (width + 2 * self.padding.2 - self.kernel_size.2) / self.stride.2 + 1;

        let mut output = ArrayD::zeros(IxDyn(&[batch, self.out_channels, out_d, out_h, out_w]));

        for b in 0..batch {
            for oc in 0..self.out_channels {
                for od in 0..out_d {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let mut sum = 0.0;
                            for ic in 0..self.in_channels {
                                for kd in 0..self.kernel_size.0 {
                                    for kh in 0..self.kernel_size.1 {
                                        for kw in 0..self.kernel_size.2 {
                                            let id = od * self.stride.0 + kd;
                                            let ih = oh * self.stride.1 + kh;
                                            let iw = ow * self.stride.2 + kw;
                                            if id < depth && ih < height && iw < width {
                                                sum += input.data[[b, ic, id, ih, iw]]
                                                    * self.weights.data.data[[oc, ic, kd, kh, kw]];
                                            }
                                        }
                                    }
                                }
                            }
                            output[[b, oc, od, oh, ow]] = sum;
                        }
                    }
                }
            }
        }

        if let Some(ref bias) = self.bias {
            for b in 0..batch {
                for oc in 0..self.out_channels {
                    for od in 0..out_d {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                output[[b, oc, od, oh, ow]] += bias.data.data[[oc]];
                            }
                        }
                    }
                }
            }
        }

        Ok(SciRS2Array::new(output, input.requires_grad))
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
        "Conv3d"
    }
}
