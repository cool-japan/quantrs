//! Transformer layers for PyTorch-like API

use super::layers::{QuantumLayerNorm, QuantumLinear};
use super::{Parameter, QuantumModule};
use crate::error::{MLError, Result};
use crate::scirs2_integration::SciRS2Array;
use scirs2_core::ndarray::{ArrayD, IxDyn};

/// Multi-head attention layer
pub struct QuantumMultiheadAttention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    q_proj: Parameter,
    k_proj: Parameter,
    v_proj: Parameter,
    out_proj: Parameter,
    dropout: f64,
    training: bool,
}

impl QuantumMultiheadAttention {
    /// Create new multi-head attention
    pub fn new(embed_dim: usize, num_heads: usize) -> Result<Self> {
        if embed_dim % num_heads != 0 {
            return Err(MLError::InvalidConfiguration(
                "embed_dim must be divisible by num_heads".to_string(),
            ));
        }

        let head_dim = embed_dim / num_heads;
        let scale = (1.0 / (embed_dim as f64)).sqrt();

        let q_proj = ArrayD::from_shape_fn(IxDyn(&[embed_dim, embed_dim]), |_| {
            (fastrand::f64() * 2.0 - 1.0) * scale
        });
        let k_proj = ArrayD::from_shape_fn(IxDyn(&[embed_dim, embed_dim]), |_| {
            (fastrand::f64() * 2.0 - 1.0) * scale
        });
        let v_proj = ArrayD::from_shape_fn(IxDyn(&[embed_dim, embed_dim]), |_| {
            (fastrand::f64() * 2.0 - 1.0) * scale
        });
        let out_proj = ArrayD::from_shape_fn(IxDyn(&[embed_dim, embed_dim]), |_| {
            (fastrand::f64() * 2.0 - 1.0) * scale
        });

        Ok(Self {
            embed_dim,
            num_heads,
            head_dim,
            q_proj: Parameter::new(SciRS2Array::with_grad(q_proj), "q_proj"),
            k_proj: Parameter::new(SciRS2Array::with_grad(k_proj), "k_proj"),
            v_proj: Parameter::new(SciRS2Array::with_grad(v_proj), "v_proj"),
            out_proj: Parameter::new(SciRS2Array::with_grad(out_proj), "out_proj"),
            dropout: 0.0,
            training: true,
        })
    }

    /// Set dropout
    pub fn dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Forward with query, key, value
    pub fn forward_qkv(
        &self,
        query: &SciRS2Array,
        key: &SciRS2Array,
        value: &SciRS2Array,
        attn_mask: Option<&ArrayD<f64>>,
    ) -> Result<(SciRS2Array, SciRS2Array)> {
        let shape = query.data.shape();
        let (batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);
        let scale = (self.head_dim as f64).sqrt();

        // Project Q, K, V
        let mut q = ArrayD::zeros(IxDyn(&[batch_size, seq_len, self.embed_dim]));
        let mut k = ArrayD::zeros(IxDyn(&[batch_size, seq_len, self.embed_dim]));
        let mut v = ArrayD::zeros(IxDyn(&[batch_size, seq_len, self.embed_dim]));

        // Simple matrix multiplication for projection
        for b in 0..batch_size {
            for s in 0..seq_len {
                for e_out in 0..self.embed_dim {
                    let mut q_sum = 0.0;
                    let mut k_sum = 0.0;
                    let mut v_sum = 0.0;
                    for e_in in 0..self.embed_dim {
                        q_sum += query.data[[b, s, e_in]] * self.q_proj.data.data[[e_out, e_in]];
                        k_sum += key.data[[b, s, e_in]] * self.k_proj.data.data[[e_out, e_in]];
                        v_sum += value.data[[b, s, e_in]] * self.v_proj.data.data[[e_out, e_in]];
                    }
                    q[[b, s, e_out]] = q_sum;
                    k[[b, s, e_out]] = k_sum;
                    v[[b, s, e_out]] = v_sum;
                }
            }
        }

        // Compute attention scores: Q @ K^T / sqrt(d_k)
        let mut attn_scores = ArrayD::zeros(IxDyn(&[batch_size, self.num_heads, seq_len, seq_len]));

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut score = 0.0;
                        for d in 0..self.head_dim {
                            let q_idx = h * self.head_dim + d;
                            let k_idx = h * self.head_dim + d;
                            score += q[[b, i, q_idx]] * k[[b, j, k_idx]];
                        }
                        attn_scores[[b, h, i, j]] = score / scale;
                    }
                }
            }
        }

        // Apply attention mask if provided
        if let Some(mask) = attn_mask {
            for b in 0..batch_size {
                for h in 0..self.num_heads {
                    for i in 0..seq_len {
                        for j in 0..seq_len {
                            if mask[[i, j]] == 0.0 {
                                attn_scores[[b, h, i, j]] = f64::NEG_INFINITY;
                            }
                        }
                    }
                }
            }
        }

        // Softmax
        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    let max_score = (0..seq_len)
                        .map(|j| attn_scores[[b, h, i, j]])
                        .fold(f64::NEG_INFINITY, f64::max);
                    let mut sum_exp = 0.0;
                    for j in 0..seq_len {
                        attn_scores[[b, h, i, j]] = (attn_scores[[b, h, i, j]] - max_score).exp();
                        sum_exp += attn_scores[[b, h, i, j]];
                    }
                    for j in 0..seq_len {
                        attn_scores[[b, h, i, j]] /= sum_exp;
                    }
                }
            }
        }

        // Attention output: attn_weights @ V
        let mut attn_output = ArrayD::zeros(IxDyn(&[batch_size, seq_len, self.embed_dim]));

        for b in 0..batch_size {
            for h in 0..self.num_heads {
                for i in 0..seq_len {
                    for d in 0..self.head_dim {
                        let mut sum = 0.0;
                        for j in 0..seq_len {
                            sum += attn_scores[[b, h, i, j]] * v[[b, j, h * self.head_dim + d]];
                        }
                        attn_output[[b, i, h * self.head_dim + d]] = sum;
                    }
                }
            }
        }

        // Output projection
        let mut output = ArrayD::zeros(IxDyn(&[batch_size, seq_len, self.embed_dim]));
        for b in 0..batch_size {
            for s in 0..seq_len {
                for e_out in 0..self.embed_dim {
                    let mut sum = 0.0;
                    for e_in in 0..self.embed_dim {
                        sum += attn_output[[b, s, e_in]] * self.out_proj.data.data[[e_out, e_in]];
                    }
                    output[[b, s, e_out]] = sum;
                }
            }
        }

        // Average attention weights across heads for output
        let mut avg_attn = ArrayD::zeros(IxDyn(&[batch_size, seq_len, seq_len]));
        for b in 0..batch_size {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let mut sum = 0.0;
                    for h in 0..self.num_heads {
                        sum += attn_scores[[b, h, i, j]];
                    }
                    avg_attn[[b, i, j]] = sum / self.num_heads as f64;
                }
            }
        }

        Ok((
            SciRS2Array::new(output, query.requires_grad),
            SciRS2Array::new(avg_attn, false),
        ))
    }
}

impl QuantumModule for QuantumMultiheadAttention {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        // Self-attention: query = key = value = input
        let (output, _) = self.forward_qkv(input, input, input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![
            self.q_proj.clone(),
            self.k_proj.clone(),
            self.v_proj.clone(),
            self.out_proj.clone(),
        ]
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {
        self.q_proj.data.zero_grad();
        self.k_proj.data.zero_grad();
        self.v_proj.data.zero_grad();
        self.out_proj.data.zero_grad();
    }

    fn name(&self) -> &str {
        "MultiheadAttention"
    }
}

/// Transformer encoder layer
pub struct QuantumTransformerEncoderLayer {
    self_attn: QuantumMultiheadAttention,
    linear1: QuantumLinear,
    linear2: QuantumLinear,
    norm1: QuantumLayerNorm,
    norm2: QuantumLayerNorm,
    dropout: f64,
    training: bool,
}

impl QuantumTransformerEncoderLayer {
    /// Create new transformer encoder layer
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize) -> Result<Self> {
        Ok(Self {
            self_attn: QuantumMultiheadAttention::new(d_model, nhead)?,
            linear1: QuantumLinear::new(d_model, dim_feedforward)?,
            linear2: QuantumLinear::new(dim_feedforward, d_model)?,
            norm1: QuantumLayerNorm::new(vec![d_model]),
            norm2: QuantumLayerNorm::new(vec![d_model]),
            dropout: 0.1,
            training: true,
        })
    }

    /// Set dropout
    pub fn dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }
}

impl QuantumModule for QuantumTransformerEncoderLayer {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        // Self attention
        let attn_output = self.self_attn.forward(input)?;

        // Add & Norm
        let residual1 = SciRS2Array::new(&input.data + &attn_output.data, input.requires_grad);
        let normed1 = self.norm1.forward(&residual1)?;

        // Feedforward
        let ff_output = self.linear1.forward(&normed1)?;
        let ff_activated =
            SciRS2Array::new(ff_output.data.mapv(|x| x.max(0.0)), ff_output.requires_grad);
        let ff_output2 = self.linear2.forward(&ff_activated)?;

        // Add & Norm
        let residual2 = SciRS2Array::new(&normed1.data + &ff_output2.data, input.requires_grad);
        self.norm2.forward(&residual2)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.self_attn.parameters();
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
        self.self_attn.train(mode);
        self.linear1.train(mode);
        self.linear2.train(mode);
        self.norm1.train(mode);
        self.norm2.train(mode);
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {
        self.self_attn.zero_grad();
        self.linear1.zero_grad();
        self.linear2.zero_grad();
        self.norm1.zero_grad();
        self.norm2.zero_grad();
    }

    fn name(&self) -> &str {
        "TransformerEncoderLayer"
    }
}

/// Positional encoding for transformers
pub struct PositionalEncoding {
    d_model: usize,
    max_len: usize,
    dropout: f64,
    encoding: ArrayD<f64>,
    training: bool,
}

impl PositionalEncoding {
    /// Create new positional encoding
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let mut encoding = ArrayD::zeros(IxDyn(&[max_len, d_model]));

        for pos in 0..max_len {
            for i in 0..d_model {
                let angle = pos as f64 / 10000.0_f64.powf(2.0 * (i / 2) as f64 / d_model as f64);
                encoding[[pos, i]] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            }
        }

        Self {
            d_model,
            max_len,
            dropout: 0.1,
            encoding,
            training: true,
        }
    }

    /// Set dropout
    pub fn dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }
}

impl QuantumModule for PositionalEncoding {
    fn forward(&mut self, input: &SciRS2Array) -> Result<SciRS2Array> {
        let shape = input.data.shape();
        let seq_len = shape[1];

        let mut output = input.data.clone();

        for b in 0..shape[0] {
            for s in 0..seq_len.min(self.max_len) {
                for d in 0..self.d_model.min(shape[2]) {
                    output[[b, s, d]] += self.encoding[[s, d]];
                }
            }
        }

        Ok(SciRS2Array::new(output, input.requires_grad))
    }

    fn parameters(&self) -> Vec<Parameter> {
        Vec::new() // Positional encoding is typically not learned
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn zero_grad(&mut self) {}

    fn name(&self) -> &str {
        "PositionalEncoding"
    }
}
