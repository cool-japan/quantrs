//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::gate::{multi::*, single::*, GateOp};
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4, Axis};

use super::types::{
    ActivationType, PositionEncodingType, QuantumAttentionType, QuantumFeedForward,
    QuantumMultiHeadAttention, QuantumPositionEncoding, QuantumTransformerConfig,
};

/// Helper function to create causal attention mask
pub fn create_causal_mask(batch_size: usize, seq_len: usize) -> Array3<bool> {
    let mut mask = Array3::from_elem((batch_size, seq_len, seq_len), false);
    for batch_idx in 0..batch_size {
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask[[batch_idx, i, j]] = true;
            }
        }
    }
    mask
}
/// Helper function to create padding mask
pub fn create_padding_mask(
    batch_size: usize,
    seq_len: usize,
    actual_lengths: &[usize],
) -> Array3<bool> {
    let mut mask = Array3::from_elem((batch_size, seq_len, seq_len), false);
    for (batch_idx, &actual_len) in actual_lengths.iter().enumerate() {
        if batch_idx < batch_size {
            for i in 0..seq_len {
                for j in actual_len..seq_len {
                    mask[[batch_idx, i, j]] = true;
                }
            }
        }
    }
    mask
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_quantum_transformer_config() {
        let config = QuantumTransformerConfig::default();
        assert_eq!(config.model_dim, 512);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.num_layers, 6);
        let large_config = QuantumTransformerConfig::large();
        assert_eq!(large_config.model_dim, 1024);
        assert_eq!(large_config.num_heads, 16);
    }
    #[test]
    fn test_quantum_multi_head_attention_creation() {
        let attention = QuantumMultiHeadAttention::new(
            8,
            512,
            QuantumAttentionType::HybridQuantumClassical,
            10,
        );
        assert!(attention.is_ok());
        let attn = attention.expect("Attention creation should succeed");
        assert_eq!(attn.num_heads, 8);
        assert_eq!(attn.model_dim, 512);
        assert_eq!(attn.head_dim, 64);
    }
    #[test]
    fn test_quantum_position_encoding() {
        let pos_enc = QuantumPositionEncoding::new(PositionEncodingType::Sinusoidal, 256, 512, 8);
        assert!(pos_enc.is_ok());
        let pe = pos_enc.expect("Position encoding creation should succeed");
        assert_eq!(pe.model_dim, 256);
        assert_eq!(pe.max_seq_len, 512);
    }
    #[test]
    fn test_quantum_feedforward() {
        let ff = QuantumFeedForward::new(256, 1024, 256, 8, ActivationType::QuantumGELU, 0.1);
        assert!(ff.is_ok());
        let feedforward = ff.expect("Feedforward creation should succeed");
        assert_eq!(feedforward.input_dim, 256);
        assert_eq!(feedforward.hidden_dim, 1024);
        assert_eq!(feedforward.output_dim, 256);
    }
    #[test]
    fn test_causal_mask_creation() {
        let mask = create_causal_mask(2, 4);
        assert_eq!(mask.dim(), (2, 4, 4));
        assert!(!mask[[0, 0, 0]]);
        assert!(!mask[[0, 1, 0]]);
        assert!(!mask[[0, 1, 1]]);
        assert!(mask[[0, 0, 1]]);
        assert!(mask[[0, 0, 2]]);
        assert!(mask[[0, 1, 2]]);
    }
    #[test]
    fn test_padding_mask_creation() {
        let actual_lengths = vec![3, 2];
        let mask = create_padding_mask(2, 4, &actual_lengths);
        assert!(!mask[[0, 0, 2]]);
        assert!(mask[[0, 0, 3]]);
        assert!(!mask[[1, 0, 1]]);
        assert!(mask[[1, 0, 2]]);
        assert!(mask[[1, 0, 3]]);
    }
}
