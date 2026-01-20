//! RNN layers for Keras-like API (LSTM, GRU, Bidirectional)

use super::{ActivationFunction, Dense, KerasLayer};
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{ArrayD, IxDyn};

/// LSTM layer (Keras-compatible)
pub struct LSTM {
    /// Number of units (hidden size)
    units: usize,
    /// Return sequences
    return_sequences: bool,
    /// Return state
    return_state: bool,
    /// Go backwards
    go_backwards: bool,
    /// Dropout rate
    dropout: f64,
    /// Recurrent dropout
    recurrent_dropout: f64,
    /// Activation function
    activation: ActivationFunction,
    /// Recurrent activation
    recurrent_activation: ActivationFunction,
    /// Weights
    weights: Option<(ArrayD<f64>, ArrayD<f64>, ArrayD<f64>)>,
    /// Built flag
    built: bool,
    /// Layer name
    layer_name: Option<String>,
}

impl LSTM {
    /// Create new LSTM layer
    pub fn new(units: usize) -> Self {
        Self {
            units,
            return_sequences: false,
            return_state: false,
            go_backwards: false,
            dropout: 0.0,
            recurrent_dropout: 0.0,
            activation: ActivationFunction::Tanh,
            recurrent_activation: ActivationFunction::Sigmoid,
            weights: None,
            built: false,
            layer_name: None,
        }
    }

    /// Set return sequences
    pub fn return_sequences(mut self, return_sequences: bool) -> Self {
        self.return_sequences = return_sequences;
        self
    }

    /// Set return state
    pub fn return_state(mut self, return_state: bool) -> Self {
        self.return_state = return_state;
        self
    }

    /// Set go backwards
    pub fn go_backwards(mut self, go_backwards: bool) -> Self {
        self.go_backwards = go_backwards;
        self
    }

    /// Set dropout
    pub fn dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set recurrent dropout
    pub fn recurrent_dropout(mut self, recurrent_dropout: f64) -> Self {
        self.recurrent_dropout = recurrent_dropout;
        self
    }

    /// Set layer name
    pub fn name(mut self, name: &str) -> Self {
        self.layer_name = Some(name.to_string());
        self
    }
}

impl KerasLayer for LSTM {
    fn call(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        if !self.built {
            return Err(MLError::ModelNotTrained(
                "Layer not built. Call build() first.".to_string(),
            ));
        }

        let (kernel, recurrent_kernel, bias) = self
            .weights
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("LSTM weights not initialized".to_string()))?;

        let shape = input.shape();
        let (batch_size, seq_len, features) = (shape[0], shape[1], shape[2]);

        let mut h: ArrayD<f64> = ArrayD::zeros(IxDyn(&[batch_size, self.units]));
        let mut c: ArrayD<f64> = ArrayD::zeros(IxDyn(&[batch_size, self.units]));

        let mut outputs = Vec::with_capacity(seq_len);

        let sequence: Vec<usize> = if self.go_backwards {
            (0..seq_len).rev().collect()
        } else {
            (0..seq_len).collect()
        };

        for t in sequence {
            let mut gates = ArrayD::zeros(IxDyn(&[batch_size, 4 * self.units]));

            for b in 0..batch_size {
                for g in 0..4 * self.units {
                    let mut sum = bias[[g]];
                    for f in 0..features.min(kernel.shape()[0]) {
                        sum += input[[b, t, f]] * kernel[[f, g]];
                    }
                    for j in 0..self.units {
                        sum += h[[b, j]] * recurrent_kernel[[j, g]];
                    }
                    gates[[b, g]] = sum;
                }
            }

            for b in 0..batch_size {
                for j in 0..self.units {
                    let i = 1.0 / (1.0 + (-gates[[b, j]]).exp());
                    let f = 1.0 / (1.0 + (-gates[[b, self.units + j]]).exp());
                    let g = gates[[b, 2 * self.units + j]].tanh();
                    let o = 1.0 / (1.0 + (-gates[[b, 3 * self.units + j]]).exp());

                    c[[b, j]] = f * c[[b, j]] + i * g;
                    h[[b, j]] = o * c[[b, j]].tanh();
                }
            }

            outputs.push(h.clone());
        }

        if self.go_backwards {
            outputs.reverse();
        }

        if self.return_sequences {
            let mut result = ArrayD::zeros(IxDyn(&[batch_size, seq_len, self.units]));
            for (t, h_t) in outputs.iter().enumerate() {
                for b in 0..batch_size {
                    for j in 0..self.units {
                        result[[b, t, j]] = h_t[[b, j]];
                    }
                }
            }
            Ok(result)
        } else {
            Ok(outputs.last().cloned().unwrap_or(h))
        }
    }

    fn build(&mut self, input_shape: &[usize]) -> Result<()> {
        let input_dim = *input_shape
            .last()
            .ok_or_else(|| MLError::InvalidConfiguration("Invalid input shape".to_string()))?;

        let scale = (6.0 / (input_dim + self.units) as f64).sqrt();
        let kernel = ArrayD::from_shape_fn(IxDyn(&[input_dim, 4 * self.units]), |_| {
            (fastrand::f64() * 2.0 - 1.0) * scale
        });
        let recurrent_kernel = ArrayD::from_shape_fn(IxDyn(&[self.units, 4 * self.units]), |_| {
            (fastrand::f64() * 2.0 - 1.0) * scale
        });
        let bias = ArrayD::zeros(IxDyn(&[4 * self.units]));

        self.weights = Some((kernel, recurrent_kernel, bias));
        self.built = true;

        Ok(())
    }

    fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        if self.return_sequences {
            vec![input_shape[0], input_shape[1], self.units]
        } else {
            vec![input_shape[0], self.units]
        }
    }

    fn count_params(&self) -> usize {
        if let Some((kernel, recurrent_kernel, bias)) = &self.weights {
            kernel.len() + recurrent_kernel.len() + bias.len()
        } else {
            0
        }
    }

    fn get_weights(&self) -> Vec<ArrayD<f64>> {
        if let Some((k, rk, b)) = &self.weights {
            vec![k.clone(), rk.clone(), b.clone()]
        } else {
            vec![]
        }
    }

    fn set_weights(&mut self, weights: Vec<ArrayD<f64>>) -> Result<()> {
        if weights.len() == 3 {
            self.weights = Some((weights[0].clone(), weights[1].clone(), weights[2].clone()));
            Ok(())
        } else {
            Err(MLError::InvalidConfiguration(
                "LSTM requires 3 weight arrays".to_string(),
            ))
        }
    }

    fn built(&self) -> bool {
        self.built
    }

    fn name(&self) -> &str {
        self.layer_name.as_deref().unwrap_or("lstm")
    }
}

/// GRU layer (Keras-compatible)
pub struct GRU {
    /// Number of units
    units: usize,
    /// Return sequences
    return_sequences: bool,
    /// Return state
    return_state: bool,
    /// Go backwards
    go_backwards: bool,
    /// Dropout
    dropout: f64,
    /// Recurrent dropout
    recurrent_dropout: f64,
    /// Weights
    weights: Option<(ArrayD<f64>, ArrayD<f64>, ArrayD<f64>)>,
    /// Built flag
    built: bool,
    /// Layer name
    layer_name: Option<String>,
}

impl GRU {
    /// Create new GRU layer
    pub fn new(units: usize) -> Self {
        Self {
            units,
            return_sequences: false,
            return_state: false,
            go_backwards: false,
            dropout: 0.0,
            recurrent_dropout: 0.0,
            weights: None,
            built: false,
            layer_name: None,
        }
    }

    /// Set return sequences
    pub fn return_sequences(mut self, return_sequences: bool) -> Self {
        self.return_sequences = return_sequences;
        self
    }

    /// Set return state
    pub fn return_state(mut self, return_state: bool) -> Self {
        self.return_state = return_state;
        self
    }

    /// Set go backwards
    pub fn go_backwards(mut self, go_backwards: bool) -> Self {
        self.go_backwards = go_backwards;
        self
    }

    /// Set dropout
    pub fn dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set layer name
    pub fn name(mut self, name: &str) -> Self {
        self.layer_name = Some(name.to_string());
        self
    }
}

impl KerasLayer for GRU {
    fn call(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        if !self.built {
            return Err(MLError::ModelNotTrained(
                "Layer not built. Call build() first.".to_string(),
            ));
        }

        let (kernel, recurrent_kernel, bias) = self
            .weights
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("GRU weights not initialized".to_string()))?;

        let shape = input.shape();
        let (batch_size, seq_len, features) = (shape[0], shape[1], shape[2]);

        let mut h: ArrayD<f64> = ArrayD::zeros(IxDyn(&[batch_size, self.units]));
        let mut outputs = Vec::with_capacity(seq_len);

        let sequence: Vec<usize> = if self.go_backwards {
            (0..seq_len).rev().collect()
        } else {
            (0..seq_len).collect()
        };

        for t in sequence {
            let mut gates: ArrayD<f64> = ArrayD::zeros(IxDyn(&[batch_size, 3 * self.units]));

            for b in 0..batch_size {
                for g in 0..3 * self.units {
                    let mut sum = bias[[g]];
                    for f in 0..features.min(kernel.shape()[0]) {
                        sum += input[[b, t, f]] * kernel[[f, g]];
                    }
                    for j in 0..self.units {
                        sum += h[[b, j]] * recurrent_kernel[[j, g]];
                    }
                    gates[[b, g]] = sum;
                }
            }

            for b in 0..batch_size {
                for j in 0..self.units {
                    let r = 1.0 / (1.0 + (-gates[[b, j]]).exp());
                    let z = 1.0 / (1.0 + (-gates[[b, self.units + j]]).exp());
                    let n_val: f64 = gates[[b, 2 * self.units + j]] + r * h[[b, j]];
                    let n = n_val.tanh();

                    h[[b, j]] = (1.0 - z) * n + z * h[[b, j]];
                }
            }

            outputs.push(h.clone());
        }

        if self.go_backwards {
            outputs.reverse();
        }

        if self.return_sequences {
            let mut result = ArrayD::zeros(IxDyn(&[batch_size, seq_len, self.units]));
            for (t, h_t) in outputs.iter().enumerate() {
                for b in 0..batch_size {
                    for j in 0..self.units {
                        result[[b, t, j]] = h_t[[b, j]];
                    }
                }
            }
            Ok(result)
        } else {
            Ok(outputs.last().cloned().unwrap_or(h))
        }
    }

    fn build(&mut self, input_shape: &[usize]) -> Result<()> {
        let input_dim = *input_shape
            .last()
            .ok_or_else(|| MLError::InvalidConfiguration("Invalid input shape".to_string()))?;

        let scale = (6.0 / (input_dim + self.units) as f64).sqrt();
        let kernel = ArrayD::from_shape_fn(IxDyn(&[input_dim, 3 * self.units]), |_| {
            (fastrand::f64() * 2.0 - 1.0) * scale
        });
        let recurrent_kernel = ArrayD::from_shape_fn(IxDyn(&[self.units, 3 * self.units]), |_| {
            (fastrand::f64() * 2.0 - 1.0) * scale
        });
        let bias = ArrayD::zeros(IxDyn(&[3 * self.units]));

        self.weights = Some((kernel, recurrent_kernel, bias));
        self.built = true;

        Ok(())
    }

    fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        if self.return_sequences {
            vec![input_shape[0], input_shape[1], self.units]
        } else {
            vec![input_shape[0], self.units]
        }
    }

    fn count_params(&self) -> usize {
        if let Some((kernel, recurrent_kernel, bias)) = &self.weights {
            kernel.len() + recurrent_kernel.len() + bias.len()
        } else {
            0
        }
    }

    fn get_weights(&self) -> Vec<ArrayD<f64>> {
        if let Some((k, rk, b)) = &self.weights {
            vec![k.clone(), rk.clone(), b.clone()]
        } else {
            vec![]
        }
    }

    fn set_weights(&mut self, weights: Vec<ArrayD<f64>>) -> Result<()> {
        if weights.len() == 3 {
            self.weights = Some((weights[0].clone(), weights[1].clone(), weights[2].clone()));
            Ok(())
        } else {
            Err(MLError::InvalidConfiguration(
                "GRU requires 3 weight arrays".to_string(),
            ))
        }
    }

    fn built(&self) -> bool {
        self.built
    }

    fn name(&self) -> &str {
        self.layer_name.as_deref().unwrap_or("gru")
    }
}

/// Bidirectional wrapper
pub struct Bidirectional {
    /// Forward layer
    forward_layer: Box<dyn KerasLayer>,
    /// Backward layer
    backward_layer: Box<dyn KerasLayer>,
    /// Merge mode
    merge_mode: String,
    /// Built flag
    built: bool,
    /// Layer name
    layer_name: Option<String>,
}

impl Bidirectional {
    /// Create new Bidirectional wrapper
    pub fn new(layer: Box<dyn KerasLayer>) -> Self {
        Self {
            forward_layer: layer,
            backward_layer: Box::new(Dense::new(1)),
            merge_mode: "concat".to_string(),
            built: false,
            layer_name: None,
        }
    }

    /// Set merge mode
    pub fn merge_mode(mut self, merge_mode: &str) -> Self {
        self.merge_mode = merge_mode.to_string();
        self
    }

    /// Set layer name
    pub fn name(mut self, name: &str) -> Self {
        self.layer_name = Some(name.to_string());
        self
    }
}

impl KerasLayer for Bidirectional {
    fn call(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        let forward_output = self.forward_layer.call(input)?;

        let shape = input.shape();
        let mut reversed = input.clone();
        let seq_len = shape[1];
        for b in 0..shape[0] {
            for t in 0..seq_len {
                for f in 0..shape[2] {
                    reversed[[b, t, f]] = input[[b, seq_len - 1 - t, f]];
                }
            }
        }

        let backward_output = self.backward_layer.call(&reversed)?;

        match self.merge_mode.as_str() {
            "sum" => Ok(&forward_output + &backward_output),
            "mul" => Ok(&forward_output * &backward_output),
            "ave" => Ok((&forward_output + &backward_output) / 2.0),
            _ => {
                let fwd_shape = forward_output.shape();
                let bwd_shape = backward_output.shape();
                let mut output = ArrayD::zeros(IxDyn(&[
                    fwd_shape[0],
                    fwd_shape.get(1).copied().unwrap_or(1),
                    fwd_shape.last().copied().unwrap_or(0) + bwd_shape.last().copied().unwrap_or(0),
                ]));

                let fwd_last = *fwd_shape.last().unwrap_or(&0);
                for b in 0..fwd_shape[0] {
                    for s in 0..fwd_shape.get(1).copied().unwrap_or(1) {
                        for f in 0..fwd_last {
                            output[[b, s, f]] = forward_output[[b, s, f]];
                        }
                        for f in 0..*bwd_shape.last().unwrap_or(&0) {
                            output[[b, s, fwd_last + f]] = backward_output[[b, s, f]];
                        }
                    }
                }
                Ok(output)
            }
        }
    }

    fn build(&mut self, input_shape: &[usize]) -> Result<()> {
        self.forward_layer.build(input_shape)?;
        self.backward_layer.build(input_shape)?;
        self.built = true;
        Ok(())
    }

    fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let fwd_shape = self.forward_layer.compute_output_shape(input_shape);
        match self.merge_mode.as_str() {
            "sum" | "mul" | "ave" => fwd_shape,
            _ => {
                let mut out = fwd_shape.clone();
                if let Some(last) = out.last_mut() {
                    *last *= 2;
                }
                out
            }
        }
    }

    fn count_params(&self) -> usize {
        self.forward_layer.count_params() + self.backward_layer.count_params()
    }

    fn get_weights(&self) -> Vec<ArrayD<f64>> {
        let mut weights = self.forward_layer.get_weights();
        weights.extend(self.backward_layer.get_weights());
        weights
    }

    fn set_weights(&mut self, _weights: Vec<ArrayD<f64>>) -> Result<()> {
        Ok(())
    }

    fn built(&self) -> bool {
        self.built
    }

    fn name(&self) -> &str {
        self.layer_name.as_deref().unwrap_or("bidirectional")
    }
}
