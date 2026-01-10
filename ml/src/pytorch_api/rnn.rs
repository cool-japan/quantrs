//! RNN layers for PyTorch-like API (LSTM, GRU)

use super::{Parameter, QuantumModule};
use crate::error::Result;
use crate::scirs2_integration::SciRS2Array;
use scirs2_core::ndarray::{s, ArrayD, IxDyn};

/// LSTM cell state
#[derive(Debug, Clone)]
pub struct LSTMState {
    /// Hidden state
    pub h: SciRS2Array,
    /// Cell state
    pub c: SciRS2Array,
}

/// LSTM layer
pub struct QuantumLSTM {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    dropout: f64,
    batch_first: bool,
    weights: Vec<Parameter>,
    training: bool,
}

impl QuantumLSTM {
    /// Create new LSTM
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let weight_ih = ArrayD::from_shape_fn(IxDyn(&[4 * hidden_size, input_size]), |_| {
            fastrand::f64() * 0.1 - 0.05
        });
        let weight_hh = ArrayD::from_shape_fn(IxDyn(&[4 * hidden_size, hidden_size]), |_| {
            fastrand::f64() * 0.1 - 0.05
        });
        let bias_ih = ArrayD::zeros(IxDyn(&[4 * hidden_size]));
        let bias_hh = ArrayD::zeros(IxDyn(&[4 * hidden_size]));

        Self {
            input_size,
            hidden_size,
            num_layers: 1,
            bidirectional: false,
            dropout: 0.0,
            batch_first: true,
            weights: vec![
                Parameter::new(SciRS2Array::with_grad(weight_ih), "weight_ih_l0"),
                Parameter::new(SciRS2Array::with_grad(weight_hh), "weight_hh_l0"),
                Parameter::new(SciRS2Array::with_grad(bias_ih), "bias_ih_l0"),
                Parameter::new(SciRS2Array::with_grad(bias_hh), "bias_hh_l0"),
            ],
            training: true,
        }
    }

    /// Set number of layers
    pub fn num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Set bidirectional
    pub fn bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }

    /// Set dropout
    pub fn dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set batch first
    pub fn batch_first(mut self, batch_first: bool) -> Self {
        self.batch_first = batch_first;
        self
    }

    /// Forward pass with optional initial state
    pub fn forward_with_state(
        &mut self,
        input: &SciRS2Array,
        initial_state: Option<LSTMState>,
    ) -> Result<(SciRS2Array, LSTMState)> {
        let shape = input.data.shape();
        let (batch_size, seq_len, _input_size) = if self.batch_first {
            (shape[0], shape[1], shape[2])
        } else {
            (shape[1], shape[0], shape[2])
        };

        let (mut h, mut c) = match initial_state {
            Some(state) => (state.h.data, state.c.data),
            None => (
                ArrayD::zeros(IxDyn(&[batch_size, self.hidden_size])),
                ArrayD::zeros(IxDyn(&[batch_size, self.hidden_size])),
            ),
        };

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = if self.batch_first {
                input.data.slice(s![.., t, ..]).to_owned()
            } else {
                input.data.slice(s![t, .., ..]).to_owned()
            };

            let weight_ih = &self.weights[0].data.data;
            let weight_hh = &self.weights[1].data.data;

            let mut gates = ArrayD::zeros(IxDyn(&[batch_size, 4 * self.hidden_size]));

            for b in 0..batch_size {
                for g in 0..4 * self.hidden_size {
                    let mut sum = 0.0;
                    for i in 0..self
                        .input_size
                        .min(x_t.shape().last().copied().unwrap_or(self.input_size))
                    {
                        sum += x_t[[b, i]] * weight_ih[[g, i]];
                    }
                    for j in 0..self.hidden_size {
                        sum += h[[b, j]] * weight_hh[[g, j]];
                    }
                    gates[[b, g]] = sum;
                }
            }

            let mut i_gate = ArrayD::zeros(IxDyn(&[batch_size, self.hidden_size]));
            let mut f_gate = ArrayD::zeros(IxDyn(&[batch_size, self.hidden_size]));
            let mut g_gate = ArrayD::zeros(IxDyn(&[batch_size, self.hidden_size]));
            let mut o_gate = ArrayD::zeros(IxDyn(&[batch_size, self.hidden_size]));

            for b in 0..batch_size {
                for j in 0..self.hidden_size {
                    i_gate[[b, j]] = 1.0 / (1.0 + (-gates[[b, j]]).exp());
                    f_gate[[b, j]] = 1.0 / (1.0 + (-gates[[b, self.hidden_size + j]]).exp());
                    g_gate[[b, j]] = gates[[b, 2 * self.hidden_size + j]].tanh();
                    o_gate[[b, j]] = 1.0 / (1.0 + (-gates[[b, 3 * self.hidden_size + j]]).exp());
                }
            }

            for b in 0..batch_size {
                for j in 0..self.hidden_size {
                    c[[b, j]] = f_gate[[b, j]] * c[[b, j]] + i_gate[[b, j]] * g_gate[[b, j]];
                    h[[b, j]] = o_gate[[b, j]] * c[[b, j]].tanh();
                }
            }

            outputs.push(h.clone());
        }

        let output_shape = if self.batch_first {
            IxDyn(&[batch_size, seq_len, self.hidden_size])
        } else {
            IxDyn(&[seq_len, batch_size, self.hidden_size])
        };
        let mut output = ArrayD::zeros(output_shape);

        for (t, h_t) in outputs.iter().enumerate() {
            for b in 0..batch_size {
                for j in 0..self.hidden_size {
                    if self.batch_first {
                        output[[b, t, j]] = h_t[[b, j]];
                    } else {
                        output[[t, b, j]] = h_t[[b, j]];
                    }
                }
            }
        }

        let final_state = LSTMState {
            h: SciRS2Array::new(h, input.requires_grad),
            c: SciRS2Array::new(c, input.requires_grad),
        };

        Ok((SciRS2Array::new(output, input.requires_grad), final_state))
    }
}

impl QuantumModule for QuantumLSTM {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        let (output, _) = self.forward_with_state(input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.weights.clone()
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {
        for w in &mut self.weights {
            w.data.zero_grad();
        }
    }

    fn name(&self) -> &str {
        "LSTM"
    }
}

/// GRU layer
pub struct QuantumGRU {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    dropout: f64,
    batch_first: bool,
    weights: Vec<Parameter>,
    training: bool,
}

impl QuantumGRU {
    /// Create new GRU
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let weight_ih = ArrayD::from_shape_fn(IxDyn(&[3 * hidden_size, input_size]), |_| {
            fastrand::f64() * 0.1 - 0.05
        });
        let weight_hh = ArrayD::from_shape_fn(IxDyn(&[3 * hidden_size, hidden_size]), |_| {
            fastrand::f64() * 0.1 - 0.05
        });
        let bias_ih = ArrayD::zeros(IxDyn(&[3 * hidden_size]));
        let bias_hh = ArrayD::zeros(IxDyn(&[3 * hidden_size]));

        Self {
            input_size,
            hidden_size,
            num_layers: 1,
            bidirectional: false,
            dropout: 0.0,
            batch_first: true,
            weights: vec![
                Parameter::new(SciRS2Array::with_grad(weight_ih), "weight_ih_l0"),
                Parameter::new(SciRS2Array::with_grad(weight_hh), "weight_hh_l0"),
                Parameter::new(SciRS2Array::with_grad(bias_ih), "bias_ih_l0"),
                Parameter::new(SciRS2Array::with_grad(bias_hh), "bias_hh_l0"),
            ],
            training: true,
        }
    }

    /// Set number of layers
    pub fn num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Set bidirectional
    pub fn bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }

    /// Set batch first
    pub fn batch_first(mut self, batch_first: bool) -> Self {
        self.batch_first = batch_first;
        self
    }

    /// Forward pass with optional initial hidden state
    pub fn forward_with_hidden(
        &mut self,
        input: &SciRS2Array,
        initial_hidden: Option<SciRS2Array>,
    ) -> Result<(SciRS2Array, SciRS2Array)> {
        let shape = input.data.shape();
        let (batch_size, seq_len, _) = if self.batch_first {
            (shape[0], shape[1], shape[2])
        } else {
            (shape[1], shape[0], shape[2])
        };

        let mut h = match initial_hidden {
            Some(state) => state.data,
            None => ArrayD::zeros(IxDyn(&[batch_size, self.hidden_size])),
        };

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = if self.batch_first {
                input.data.slice(s![.., t, ..]).to_owned()
            } else {
                input.data.slice(s![t, .., ..]).to_owned()
            };

            let weight_ih = &self.weights[0].data.data;
            let weight_hh = &self.weights[1].data.data;

            let mut gates = ArrayD::zeros(IxDyn(&[batch_size, 3 * self.hidden_size]));

            for b in 0..batch_size {
                for g in 0..3 * self.hidden_size {
                    let mut sum = 0.0;
                    for i in 0..self
                        .input_size
                        .min(x_t.shape().last().copied().unwrap_or(self.input_size))
                    {
                        sum += x_t[[b, i]] * weight_ih[[g, i]];
                    }
                    for j in 0..self.hidden_size {
                        sum += h[[b, j]] * weight_hh[[g, j]];
                    }
                    gates[[b, g]] = sum;
                }
            }

            let mut r_gate = ArrayD::zeros(IxDyn(&[batch_size, self.hidden_size]));
            let mut z_gate = ArrayD::zeros(IxDyn(&[batch_size, self.hidden_size]));
            let mut n_gate = ArrayD::zeros(IxDyn(&[batch_size, self.hidden_size]));

            for b in 0..batch_size {
                for j in 0..self.hidden_size {
                    r_gate[[b, j]] = 1.0 / (1.0 + (-gates[[b, j]]).exp());
                    z_gate[[b, j]] = 1.0 / (1.0 + (-gates[[b, self.hidden_size + j]]).exp());
                    n_gate[[b, j]] =
                        (gates[[b, 2 * self.hidden_size + j]] + r_gate[[b, j]] * h[[b, j]]).tanh();
                }
            }

            for b in 0..batch_size {
                for j in 0..self.hidden_size {
                    h[[b, j]] =
                        (1.0 - z_gate[[b, j]]) * n_gate[[b, j]] + z_gate[[b, j]] * h[[b, j]];
                }
            }

            outputs.push(h.clone());
        }

        let output_shape = if self.batch_first {
            IxDyn(&[batch_size, seq_len, self.hidden_size])
        } else {
            IxDyn(&[seq_len, batch_size, self.hidden_size])
        };
        let mut output = ArrayD::zeros(output_shape);

        for (t, h_t) in outputs.iter().enumerate() {
            for b in 0..batch_size {
                for j in 0..self.hidden_size {
                    if self.batch_first {
                        output[[b, t, j]] = h_t[[b, j]];
                    } else {
                        output[[t, b, j]] = h_t[[b, j]];
                    }
                }
            }
        }

        Ok((
            SciRS2Array::new(output, input.requires_grad),
            SciRS2Array::new(h, input.requires_grad),
        ))
    }
}

impl QuantumModule for QuantumGRU {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        let (output, _) = self.forward_with_hidden(input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.weights.clone()
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {
        for w in &mut self.weights {
            w.data.zero_grad();
        }
    }

    fn name(&self) -> &str {
        "GRU"
    }
}
