//! Attention layers for Keras-like API

use super::KerasLayer;
use crate::error::{MLError, Result};
use scirs2_core::ndarray::{ArrayD, IxDyn};

/// Multi-head attention layer (Keras-compatible)
pub struct MultiHeadAttention {
    /// Number of heads
    num_heads: usize,
    /// Key dimension
    key_dim: usize,
    /// Value dimension
    value_dim: usize,
    /// Dropout
    dropout: f64,
    /// Use bias
    use_bias: bool,
    /// Query projection weights
    query_weights: Option<ArrayD<f64>>,
    /// Key projection weights
    key_weights: Option<ArrayD<f64>>,
    /// Value projection weights
    value_weights: Option<ArrayD<f64>>,
    /// Output projection weights
    output_weights: Option<ArrayD<f64>>,
    /// Built flag
    built: bool,
    /// Layer name
    layer_name: Option<String>,
}

impl MultiHeadAttention {
    /// Create new MultiHeadAttention
    pub fn new(num_heads: usize, key_dim: usize) -> Self {
        Self {
            num_heads,
            key_dim,
            value_dim: key_dim,
            dropout: 0.0,
            use_bias: true,
            query_weights: None,
            key_weights: None,
            value_weights: None,
            output_weights: None,
            built: false,
            layer_name: None,
        }
    }

    /// Set value dimension
    pub fn value_dim(mut self, value_dim: usize) -> Self {
        self.value_dim = value_dim;
        self
    }

    /// Set dropout
    pub fn dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
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

    /// Forward with query, key, value
    pub fn call_with_qkv(
        &mut self,
        query: &ArrayD<f64>,
        key: &ArrayD<f64>,
        value: &ArrayD<f64>,
    ) -> Result<ArrayD<f64>> {
        if !self.built {
            return Err(MLError::ModelNotTrained(
                "Layer not built. Call build() first.".to_string(),
            ));
        }

        let q_weights = self
            .query_weights
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Query weights not initialized".to_string()))?;
        let k_weights = self
            .key_weights
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Key weights not initialized".to_string()))?;
        let v_weights = self
            .value_weights
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Value weights not initialized".to_string()))?;
        let out_weights = self.output_weights.as_ref().ok_or_else(|| {
            MLError::ModelNotTrained("Output weights not initialized".to_string())
        })?;

        let shape = query.shape();
        let (batch_size, seq_len, embed_dim) = (shape[0], shape[1], shape[2]);
        let head_dim = self.key_dim;
        let scale = (head_dim as f64).sqrt();

        let total_dim = self.num_heads * head_dim;
        let mut q: ArrayD<f64> = ArrayD::zeros(IxDyn(&[batch_size, seq_len, total_dim]));
        let mut k: ArrayD<f64> = ArrayD::zeros(IxDyn(&[batch_size, seq_len, total_dim]));
        let mut v: ArrayD<f64> = ArrayD::zeros(IxDyn(&[batch_size, seq_len, total_dim]));

        for b in 0..batch_size {
            for s in 0..seq_len {
                for o in 0..total_dim.min(q_weights.shape()[1]) {
                    let mut q_sum: f64 = 0.0;
                    let mut k_sum: f64 = 0.0;
                    let mut v_sum: f64 = 0.0;
                    for i in 0..embed_dim.min(q_weights.shape()[0]) {
                        q_sum += query[[b, s, i]] * q_weights[[i, o]];
                        k_sum += key[[b, s, i]] * k_weights[[i, o]];
                        v_sum += value[[b, s, i]] * v_weights[[i, o]];
                    }
                    q[[b, s, o]] = q_sum;
                    k[[b, s, o]] = k_sum;
                    v[[b, s, o]] = v_sum;
                }
            }
        }

        let mut attn: ArrayD<f64> =
            ArrayD::zeros(IxDyn(&[batch_size, self.num_heads, seq_len, seq_len]));

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut score: f64 = 0.0;
                        for d in 0..head_dim {
                            score += q[[b, i, h * head_dim + d]] * k[[b, j, h * head_dim + d]];
                        }
                        attn[[b, h, i, j]] = score / scale;
                    }
                }

                for i in 0..seq_len {
                    let max_score = (0..seq_len)
                        .map(|j| attn[[b, h, i, j]])
                        .fold(f64::NEG_INFINITY, f64::max);
                    let mut sum_exp: f64 = 0.0;
                    for j in 0..seq_len {
                        attn[[b, h, i, j]] = (attn[[b, h, i, j]] - max_score).exp();
                        sum_exp += attn[[b, h, i, j]];
                    }
                    for j in 0..seq_len {
                        attn[[b, h, i, j]] /= sum_exp;
                    }
                }
            }
        }

        let mut context: ArrayD<f64> = ArrayD::zeros(IxDyn(&[batch_size, seq_len, total_dim]));
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    for d in 0..head_dim {
                        let mut sum: f64 = 0.0;
                        for j in 0..seq_len {
                            sum += attn[[b, h, i, j]] * v[[b, j, h * head_dim + d]];
                        }
                        context[[b, i, h * head_dim + d]] = sum;
                    }
                }
            }
        }

        let mut output: ArrayD<f64> = ArrayD::zeros(IxDyn(&[batch_size, seq_len, embed_dim]));
        for b in 0..batch_size {
            for s in 0..seq_len {
                for o in 0..embed_dim.min(out_weights.shape()[1]) {
                    let mut out_sum: f64 = 0.0;
                    for i in 0..total_dim.min(out_weights.shape()[0]) {
                        out_sum += context[[b, s, i]] * out_weights[[i, o]];
                    }
                    output[[b, s, o]] = out_sum;
                }
            }
        }

        Ok(output)
    }
}

impl KerasLayer for MultiHeadAttention {
    fn call(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        if !self.built {
            return Err(MLError::ModelNotTrained(
                "Layer not built. Call build() first.".to_string(),
            ));
        }

        let q_weights = self
            .query_weights
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Query weights not initialized".to_string()))?;
        let k_weights = self
            .key_weights
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Key weights not initialized".to_string()))?;
        let v_weights = self
            .value_weights
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Value weights not initialized".to_string()))?;
        let out_weights = self.output_weights.as_ref().ok_or_else(|| {
            MLError::ModelNotTrained("Output weights not initialized".to_string())
        })?;

        let shape = input.shape();
        let (batch_size, seq_len, embed_dim) = (shape[0], shape[1], shape[2]);
        let head_dim = self.key_dim;
        let scale = (head_dim as f64).sqrt();

        let total_dim = self.num_heads * head_dim;
        let mut q: ArrayD<f64> = ArrayD::zeros(IxDyn(&[batch_size, seq_len, total_dim]));
        let mut k: ArrayD<f64> = ArrayD::zeros(IxDyn(&[batch_size, seq_len, total_dim]));
        let mut v: ArrayD<f64> = ArrayD::zeros(IxDyn(&[batch_size, seq_len, total_dim]));

        for b in 0..batch_size {
            for s in 0..seq_len {
                for o in 0..total_dim.min(q_weights.shape()[1]) {
                    let mut q_sum: f64 = 0.0;
                    let mut k_sum: f64 = 0.0;
                    let mut v_sum: f64 = 0.0;
                    for i in 0..embed_dim.min(q_weights.shape()[0]) {
                        q_sum += input[[b, s, i]] * q_weights[[i, o]];
                        k_sum += input[[b, s, i]] * k_weights[[i, o]];
                        v_sum += input[[b, s, i]] * v_weights[[i, o]];
                    }
                    q[[b, s, o]] = q_sum;
                    k[[b, s, o]] = k_sum;
                    v[[b, s, o]] = v_sum;
                }
            }
        }

        let mut attn: ArrayD<f64> =
            ArrayD::zeros(IxDyn(&[batch_size, self.num_heads, seq_len, seq_len]));

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut score: f64 = 0.0;
                        for d in 0..head_dim {
                            score += q[[b, i, h * head_dim + d]] * k[[b, j, h * head_dim + d]];
                        }
                        attn[[b, h, i, j]] = score / scale;
                    }
                }

                for i in 0..seq_len {
                    let max_score = (0..seq_len)
                        .map(|j| attn[[b, h, i, j]])
                        .fold(f64::NEG_INFINITY, f64::max);
                    let mut sum_exp: f64 = 0.0;
                    for j in 0..seq_len {
                        attn[[b, h, i, j]] = (attn[[b, h, i, j]] - max_score).exp();
                        sum_exp += attn[[b, h, i, j]];
                    }
                    for j in 0..seq_len {
                        attn[[b, h, i, j]] /= sum_exp;
                    }
                }
            }
        }

        let mut context: ArrayD<f64> = ArrayD::zeros(IxDyn(&[batch_size, seq_len, total_dim]));
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    for d in 0..head_dim {
                        let mut sum: f64 = 0.0;
                        for j in 0..seq_len {
                            sum += attn[[b, h, i, j]] * v[[b, j, h * head_dim + d]];
                        }
                        context[[b, i, h * head_dim + d]] = sum;
                    }
                }
            }
        }

        let mut output: ArrayD<f64> = ArrayD::zeros(IxDyn(&[batch_size, seq_len, embed_dim]));
        for b in 0..batch_size {
            for s in 0..seq_len {
                for o in 0..embed_dim.min(out_weights.shape()[1]) {
                    let mut out_sum: f64 = 0.0;
                    for i in 0..total_dim.min(out_weights.shape()[0]) {
                        out_sum += context[[b, s, i]] * out_weights[[i, o]];
                    }
                    output[[b, s, o]] = out_sum;
                }
            }
        }

        Ok(output)
    }

    fn build(&mut self, input_shape: &[usize]) -> Result<()> {
        let embed_dim = *input_shape
            .last()
            .ok_or_else(|| MLError::InvalidConfiguration("Invalid input shape".to_string()))?;

        let total_dim = self.num_heads * self.key_dim;
        let scale = (2.0 / (embed_dim + total_dim) as f64).sqrt();

        let query_weights = ArrayD::from_shape_fn(IxDyn(&[embed_dim, total_dim]), |_| {
            (fastrand::f64() * 2.0 - 1.0) * scale
        });
        let key_weights = ArrayD::from_shape_fn(IxDyn(&[embed_dim, total_dim]), |_| {
            (fastrand::f64() * 2.0 - 1.0) * scale
        });
        let value_weights = ArrayD::from_shape_fn(IxDyn(&[embed_dim, total_dim]), |_| {
            (fastrand::f64() * 2.0 - 1.0) * scale
        });
        let output_weights = ArrayD::from_shape_fn(IxDyn(&[total_dim, embed_dim]), |_| {
            (fastrand::f64() * 2.0 - 1.0) * scale
        });

        self.query_weights = Some(query_weights);
        self.key_weights = Some(key_weights);
        self.value_weights = Some(value_weights);
        self.output_weights = Some(output_weights);
        self.built = true;

        Ok(())
    }

    fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        input_shape.to_vec()
    }

    fn count_params(&self) -> usize {
        let q = self.query_weights.as_ref().map_or(0, |w| w.len());
        let k = self.key_weights.as_ref().map_or(0, |w| w.len());
        let v = self.value_weights.as_ref().map_or(0, |w| w.len());
        let o = self.output_weights.as_ref().map_or(0, |w| w.len());
        q + k + v + o
    }

    fn get_weights(&self) -> Vec<ArrayD<f64>> {
        let mut weights = vec![];
        if let Some(ref w) = self.query_weights {
            weights.push(w.clone());
        }
        if let Some(ref w) = self.key_weights {
            weights.push(w.clone());
        }
        if let Some(ref w) = self.value_weights {
            weights.push(w.clone());
        }
        if let Some(ref w) = self.output_weights {
            weights.push(w.clone());
        }
        weights
    }

    fn set_weights(&mut self, weights: Vec<ArrayD<f64>>) -> Result<()> {
        if weights.len() >= 4 {
            self.query_weights = Some(weights[0].clone());
            self.key_weights = Some(weights[1].clone());
            self.value_weights = Some(weights[2].clone());
            self.output_weights = Some(weights[3].clone());
        }
        Ok(())
    }

    fn built(&self) -> bool {
        self.built
    }

    fn name(&self) -> &str {
        self.layer_name.as_deref().unwrap_or("multi_head_attention")
    }
}

/// Embedding layer
pub struct Embedding {
    /// Input dimension (vocabulary size)
    input_dim: usize,
    /// Output dimension (embedding size)
    output_dim: usize,
    /// Embedding weights
    embeddings: Option<ArrayD<f64>>,
    /// Mask zero
    mask_zero: bool,
    /// Built flag
    built: bool,
    /// Layer name
    layer_name: Option<String>,
}

impl Embedding {
    /// Create new Embedding layer
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            embeddings: None,
            mask_zero: false,
            built: false,
            layer_name: None,
        }
    }

    /// Set mask zero
    pub fn mask_zero(mut self, mask_zero: bool) -> Self {
        self.mask_zero = mask_zero;
        self
    }

    /// Set layer name
    pub fn name(mut self, name: &str) -> Self {
        self.layer_name = Some(name.to_string());
        self
    }
}

impl KerasLayer for Embedding {
    fn call(&self, input: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        if !self.built {
            return Err(MLError::ModelNotTrained(
                "Layer not built. Call build() first.".to_string(),
            ));
        }

        let embeddings = self
            .embeddings
            .as_ref()
            .ok_or_else(|| MLError::ModelNotTrained("Embeddings not initialized".to_string()))?;

        let shape = input.shape();
        let batch_size = shape[0];
        let seq_len = *shape.get(1).unwrap_or(&1);

        let mut output = ArrayD::zeros(IxDyn(&[batch_size, seq_len, self.output_dim]));

        for b in 0..batch_size {
            for s in 0..seq_len {
                let idx = input[[b, s]] as usize;
                if idx < self.input_dim {
                    for d in 0..self.output_dim {
                        output[[b, s, d]] = embeddings[[idx, d]];
                    }
                }
            }
        }

        Ok(output)
    }

    fn build(&mut self, _input_shape: &[usize]) -> Result<()> {
        let scale = (1.0 / self.input_dim as f64).sqrt();
        let embeddings = ArrayD::from_shape_fn(IxDyn(&[self.input_dim, self.output_dim]), |_| {
            (fastrand::f64() * 2.0 - 1.0) * scale
        });

        self.embeddings = Some(embeddings);
        self.built = true;

        Ok(())
    }

    fn compute_output_shape(&self, input_shape: &[usize]) -> Vec<usize> {
        let mut out_shape = input_shape.to_vec();
        out_shape.push(self.output_dim);
        out_shape
    }

    fn count_params(&self) -> usize {
        self.input_dim * self.output_dim
    }

    fn get_weights(&self) -> Vec<ArrayD<f64>> {
        self.embeddings.iter().cloned().collect()
    }

    fn set_weights(&mut self, weights: Vec<ArrayD<f64>>) -> Result<()> {
        if !weights.is_empty() {
            self.embeddings = Some(weights[0].clone());
        }
        Ok(())
    }

    fn built(&self) -> bool {
        self.built
    }

    fn name(&self) -> &str {
        self.layer_name.as_deref().unwrap_or("embedding")
    }
}
