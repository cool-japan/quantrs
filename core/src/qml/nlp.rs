//! Quantum Machine Learning for Natural Language Processing
//!
//! This module provides specialized quantum machine learning layers and algorithms
//! optimized for natural language processing tasks such as text classification,
//! sentiment analysis, and language modeling.

use super::{Parameter, QMLLayer};
use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{multi::*, single::*, GateOp},
    parametric::{ParametricRotationX, ParametricRotationY, ParametricRotationZ},
    qubit::QubitId,
};
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Text embedding strategies for quantum NLP
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TextEmbeddingStrategy {
    /// Word-level embeddings: each word is encoded separately
    WordLevel,
    /// Character-level embeddings: each character is encoded
    CharLevel,
    /// N-gram embeddings: overlapping n-grams are encoded
    NGram(usize),
    /// Token embeddings with positional encoding
    TokenPositional,
    /// Hierarchical embeddings: words -> sentences -> documents
    Hierarchical,
}

/// Configuration for quantum NLP models
#[derive(Debug, Clone)]
pub struct QNLPConfig {
    /// Number of qubits for text representation
    pub text_qubits: usize,
    /// Number of qubits for feature extraction
    pub feature_qubits: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Text embedding strategy
    pub embedding_strategy: TextEmbeddingStrategy,
    /// Number of attention heads (for quantum attention)
    pub num_attention_heads: usize,
    /// Hidden dimension for feedforward layers
    pub hidden_dim: usize,
}

impl Default for QNLPConfig {
    fn default() -> Self {
        Self {
            text_qubits: 8,
            feature_qubits: 4,
            max_sequence_length: 32,
            vocab_size: 1000,
            embedding_dim: 64,
            embedding_strategy: TextEmbeddingStrategy::WordLevel,
            num_attention_heads: 4,
            hidden_dim: 128,
        }
    }
}

/// Quantum word embedding layer
pub struct QuantumWordEmbedding {
    /// Configuration
    config: QNLPConfig,
    /// Embedding parameters for each word in vocabulary
    embeddings: Vec<Vec<Parameter>>,
    /// Number of qubits
    num_qubits: usize,
}

impl QuantumWordEmbedding {
    /// Create a new quantum word embedding layer
    pub fn new(config: QNLPConfig) -> Self {
        let num_qubits = config.text_qubits;
        let mut embeddings = Vec::new();

        // Initialize embeddings for each word in vocabulary
        for word_id in 0..config.vocab_size {
            let mut word_embedding = Vec::new();
            for qubit in 0..num_qubits {
                // Initialize with random values
                let value = ((word_id * qubit) as f64 * 0.1).sin() * 0.5;
                word_embedding.push(Parameter {
                    name: format!("embed_{}_{}", word_id, qubit),
                    value,
                    bounds: None,
                });
            }
            embeddings.push(word_embedding);
        }

        Self {
            config,
            embeddings,
            num_qubits,
        }
    }

    /// Encode a sequence of word IDs into quantum gates
    pub fn encode_sequence(&self, word_ids: &[usize]) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut gates: Vec<Box<dyn GateOp>> = Vec::new();

        for (position, &word_id) in word_ids.iter().enumerate() {
            if word_id >= self.config.vocab_size {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Word ID {} exceeds vocabulary size {}",
                    word_id, self.config.vocab_size
                )));
            }

            if position >= self.config.max_sequence_length {
                break; // Truncate sequence if too long
            }

            // Encode word at this position
            let word_embedding = &self.embeddings[word_id];
            for (qubit_idx, param) in word_embedding.iter().enumerate() {
                let qubit = QubitId(qubit_idx as u32);

                // Use rotation gates to encode the embedding values
                gates.push(Box::new(ParametricRotationY {
                    target: qubit,
                    theta: crate::parametric::Parameter::Constant(param.value * PI), // Scale to appropriate range
                }));

                // Add positional encoding
                let positional_angle =
                    (position as f64) / (self.config.max_sequence_length as f64) * PI;
                gates.push(Box::new(ParametricRotationZ {
                    target: qubit,
                    theta: crate::parametric::Parameter::Constant(positional_angle * 0.1), // Smaller contribution
                }));
            }
        }

        Ok(gates)
    }
}

impl QMLLayer for QuantumWordEmbedding {
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn parameters(&self) -> &[Parameter] {
        // Flatten all embeddings into a single parameter vector
        // This is a simplified approach - in practice might want more efficient storage
        unimplemented!("Use flatten_parameters() method instead")
    }

    fn parameters_mut(&mut self) -> &mut [Parameter] {
        unimplemented!("Use flatten_parameters_mut() method instead")
    }

    fn gates(&self) -> Vec<Box<dyn GateOp>> {
        // Return empty - this layer provides encoding method
        Vec::new()
    }

    fn compute_gradients(
        &self,
        _state: &Array1<Complex64>,
        _loss_gradient: &Array1<Complex64>,
    ) -> QuantRS2Result<Vec<f64>> {
        // Placeholder for gradient computation
        let total_params = self.config.vocab_size * self.num_qubits;
        Ok(vec![0.0; total_params])
    }

    fn name(&self) -> &str {
        "QuantumWordEmbedding"
    }
}

/// Quantum attention mechanism for NLP
pub struct QuantumAttention {
    /// Number of qubits
    num_qubits: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Query parameters
    query_params: Vec<Parameter>,
    /// Key parameters
    key_params: Vec<Parameter>,
    /// Value parameters
    value_params: Vec<Parameter>,
    /// Output projection parameters
    output_params: Vec<Parameter>,
}

impl QuantumAttention {
    /// Create a new quantum attention layer
    pub fn new(num_qubits: usize, num_heads: usize) -> Self {
        let params_per_head = num_qubits / num_heads;

        let mut query_params = Vec::new();
        let mut key_params = Vec::new();
        let mut value_params = Vec::new();
        let mut output_params = Vec::new();

        // Initialize parameters for each head
        for head in 0..num_heads {
            for i in 0..params_per_head {
                // Query parameters
                query_params.push(Parameter {
                    name: format!("query_{}_{}", head, i),
                    value: ((head + i) as f64 * 0.1).sin() * 0.5,
                    bounds: None,
                });

                // Key parameters
                key_params.push(Parameter {
                    name: format!("key_{}_{}", head, i),
                    value: ((head + i + 1) as f64 * 0.1).cos() * 0.5,
                    bounds: None,
                });

                // Value parameters
                value_params.push(Parameter {
                    name: format!("value_{}_{}", head, i),
                    value: ((head + i + 2) as f64 * 0.1).sin() * 0.5,
                    bounds: None,
                });

                // Output parameters
                output_params.push(Parameter {
                    name: format!("output_{}_{}", head, i),
                    value: ((head + i + 3) as f64 * 0.1).cos() * 0.5,
                    bounds: None,
                });
            }
        }

        Self {
            num_qubits,
            num_heads,
            query_params,
            key_params,
            value_params,
            output_params,
        }
    }

    /// Generate attention gates for a sequence
    pub fn attention_gates(&self) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut gates: Vec<Box<dyn GateOp>> = Vec::new();
        let params_per_head = self.num_qubits / self.num_heads;

        // For each attention head
        for head in 0..self.num_heads {
            let head_offset = head * params_per_head;

            // Apply query transformations
            for i in 0..params_per_head {
                let qubit = QubitId((head_offset + i) as u32);
                let param_idx = head * params_per_head + i;

                gates.push(Box::new(ParametricRotationY {
                    target: qubit,
                    theta: crate::parametric::Parameter::Constant(
                        self.query_params[param_idx].value,
                    ),
                }));
            }

            // Apply key transformations
            for i in 0..params_per_head {
                let qubit = QubitId((head_offset + i) as u32);
                let param_idx = head * params_per_head + i;

                gates.push(Box::new(ParametricRotationZ {
                    target: qubit,
                    theta: crate::parametric::Parameter::Constant(self.key_params[param_idx].value),
                }));
            }

            // Add entanglement within head (for attention computation)
            for i in 0..params_per_head - 1 {
                let control = QubitId((head_offset + i) as u32);
                let target = QubitId((head_offset + i + 1) as u32);
                gates.push(Box::new(CNOT { control, target }));
            }

            // Apply value transformations
            for i in 0..params_per_head {
                let qubit = QubitId((head_offset + i) as u32);
                let param_idx = head * params_per_head + i;

                gates.push(Box::new(ParametricRotationX {
                    target: qubit,
                    theta: crate::parametric::Parameter::Constant(
                        self.value_params[param_idx].value,
                    ),
                }));
            }
        }

        // Add inter-head entanglement (for multi-head attention)
        for head in 0..self.num_heads - 1 {
            let control = QubitId((head * params_per_head) as u32);
            let target = QubitId(((head + 1) * params_per_head) as u32);
            gates.push(Box::new(CNOT { control, target }));
        }

        // Apply output projection
        for i in 0..self.output_params.len() {
            let qubit = QubitId(i as u32);
            gates.push(Box::new(ParametricRotationY {
                target: qubit,
                theta: crate::parametric::Parameter::Constant(self.output_params[i].value),
            }));
        }

        Ok(gates)
    }
}

impl QMLLayer for QuantumAttention {
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn parameters(&self) -> &[Parameter] {
        // This would need a flattened view of all parameters
        unimplemented!("Use all_parameters() method instead")
    }

    fn parameters_mut(&mut self) -> &mut [Parameter] {
        unimplemented!("Use all_parameters_mut() method instead")
    }

    fn gates(&self) -> Vec<Box<dyn GateOp>> {
        self.attention_gates().unwrap_or_default()
    }

    fn compute_gradients(
        &self,
        _state: &Array1<Complex64>,
        _loss_gradient: &Array1<Complex64>,
    ) -> QuantRS2Result<Vec<f64>> {
        let total_params = self.query_params.len()
            + self.key_params.len()
            + self.value_params.len()
            + self.output_params.len();
        Ok(vec![0.0; total_params])
    }

    fn name(&self) -> &str {
        "QuantumAttention"
    }
}

/// Quantum text classifier for sentiment analysis and text classification
pub struct QuantumTextClassifier {
    /// Configuration
    config: QNLPConfig,
    /// Word embedding layer
    embedding: QuantumWordEmbedding,
    /// Attention layers
    attention_layers: Vec<QuantumAttention>,
    /// Classification parameters
    classifier_params: Vec<Parameter>,
    /// Number of output classes
    num_classes: usize,
}

impl QuantumTextClassifier {
    /// Create a new quantum text classifier
    pub fn new(config: QNLPConfig, num_classes: usize) -> Self {
        let embedding = QuantumWordEmbedding::new(config.clone());

        // Create multiple attention layers for deeper models
        let mut attention_layers = Vec::new();
        for _layer_idx in 0..2 {
            // 2 attention layers
            attention_layers.push(QuantumAttention::new(
                config.text_qubits,
                config.num_attention_heads,
            ));
        }

        // Create classification parameters
        let mut classifier_params = Vec::new();
        for class in 0..num_classes {
            for qubit in 0..config.feature_qubits {
                classifier_params.push(Parameter {
                    name: format!("classifier_{}_{}", class, qubit),
                    value: ((class + qubit) as f64 * 0.2).sin() * 0.3,
                    bounds: None,
                });
            }
        }

        Self {
            config,
            embedding,
            attention_layers,
            classifier_params,
            num_classes,
        }
    }

    /// Classify a text sequence
    pub fn classify(&self, word_ids: &[usize]) -> QuantRS2Result<Vec<f64>> {
        // This would implement the full forward pass
        // For now, return dummy probabilities
        let mut probs = vec![1.0 / self.num_classes as f64; self.num_classes];

        // Add some variation based on input
        for (i, &word_id) in word_ids.iter().enumerate() {
            let variation = ((word_id + i) as f64 * 0.1).sin() * 0.1;
            probs[i % self.num_classes] += variation;
        }

        // Normalize probabilities
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for prob in &mut probs {
                *prob /= sum;
            }
        }

        Ok(probs)
    }

    /// Generate the full circuit for text classification
    pub fn build_circuit(&self, word_ids: &[usize]) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut gates = Vec::new();

        // 1. Word embedding
        gates.extend(self.embedding.encode_sequence(word_ids)?);

        // 2. Attention layers
        for attention in &self.attention_layers {
            gates.extend(attention.attention_gates()?);
        }

        // 3. Feature extraction and pooling (using measurement-like operations)
        // This would include global pooling operations
        for qubit in 0..self.config.text_qubits {
            gates.push(Box::new(Hadamard {
                target: QubitId(qubit as u32),
            }));
        }

        // 4. Classification layer
        for (_class, chunk) in self
            .classifier_params
            .chunks(self.config.feature_qubits)
            .enumerate()
        {
            for (i, param) in chunk.iter().enumerate() {
                let qubit = QubitId(i as u32);
                gates.push(Box::new(ParametricRotationY {
                    target: qubit,
                    theta: crate::parametric::Parameter::Constant(param.value),
                }));
            }
        }

        Ok(gates)
    }

    /// Train the classifier using a dataset
    pub fn train(
        &mut self,
        training_data: &[(Vec<usize>, usize)],
        learning_rate: f64,
        epochs: usize,
    ) -> QuantRS2Result<Vec<f64>> {
        let mut losses = Vec::new();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for (word_ids, true_label) in training_data {
                // Forward pass
                let predictions = self.classify(word_ids)?;

                // Compute loss (cross-entropy)
                let loss = -predictions[*true_label].ln();
                epoch_loss += loss;

                // Backward pass (simplified gradient computation)
                // In practice, this would use automatic differentiation
                self.update_parameters(predictions, *true_label, learning_rate)?;
            }

            epoch_loss /= training_data.len() as f64;
            losses.push(epoch_loss);

            if epoch % 10 == 0 {
                println!("Epoch {}: Loss = {:.4}", epoch, epoch_loss);
            }
        }

        Ok(losses)
    }

    /// Update parameters based on gradients (simplified)
    fn update_parameters(
        &mut self,
        predictions: Vec<f64>,
        true_label: usize,
        learning_rate: f64,
    ) -> QuantRS2Result<()> {
        // Simplified parameter update
        // In practice, would compute proper gradients using parameter shift rule

        for (i, param) in self.classifier_params.iter_mut().enumerate() {
            // All parameters are learnable in this simplified implementation
            {
                let class_idx = i / self.config.feature_qubits;
                let error = if class_idx == true_label {
                    predictions[class_idx] - 1.0
                } else {
                    predictions[class_idx]
                };

                // Simple gradient descent update
                param.value -= learning_rate * error * 0.1;
            }
        }

        Ok(())
    }
}

/// Quantum language model for text generation
pub struct QuantumLanguageModel {
    /// Configuration
    config: QNLPConfig,
    /// Embedding layer
    embedding: QuantumWordEmbedding,
    /// Transformer layers
    transformer_layers: Vec<QuantumAttention>,
    /// Output parameters
    output_params: Vec<Parameter>,
}

impl QuantumLanguageModel {
    /// Create a new quantum language model
    pub fn new(config: QNLPConfig) -> Self {
        let embedding = QuantumWordEmbedding::new(config.clone());

        // Create transformer layers
        let mut transformer_layers = Vec::new();
        for _layer in 0..3 {
            // 3 transformer layers
            transformer_layers.push(QuantumAttention::new(
                config.text_qubits,
                config.num_attention_heads,
            ));
        }

        // Create output parameters for next token prediction
        let mut output_params = Vec::new();
        for token in 0..config.vocab_size {
            output_params.push(Parameter {
                name: format!("output_{}", token),
                value: (token as f64 * 0.01).sin() * 0.1,
                bounds: None,
            });
        }

        Self {
            config,
            embedding,
            transformer_layers,
            output_params,
        }
    }

    /// Generate next token probabilities given a context
    pub fn predict_next_token(&self, context: &[usize]) -> QuantRS2Result<Vec<f64>> {
        // Build circuit for the context
        let _gates = self.build_circuit(context)?;

        // Simulate the circuit (placeholder)
        // In practice, would run the quantum circuit and measure

        // Return dummy probabilities for now
        let mut probs = vec![1.0 / self.config.vocab_size as f64; self.config.vocab_size];

        // Add some variation based on context
        for (i, &token) in context.iter().enumerate() {
            let variation = ((token + i) as f64 * 0.05).sin() * 0.01;
            probs[token % self.config.vocab_size] += variation;
        }

        // Normalize
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for prob in &mut probs {
                *prob /= sum;
            }
        }

        Ok(probs)
    }

    /// Generate text given a starting context
    pub fn generate_text(
        &self,
        start_context: &[usize],
        max_length: usize,
        temperature: f64,
    ) -> QuantRS2Result<Vec<usize>> {
        let mut generated = start_context.to_vec();

        for _step in 0..max_length {
            // Get context (last N tokens)
            let context_start = if generated.len() > self.config.max_sequence_length {
                generated.len() - self.config.max_sequence_length
            } else {
                0
            };
            let context = &generated[context_start..];

            // Predict next token
            let mut probs = self.predict_next_token(context)?;

            // Apply temperature scaling
            if temperature != 1.0 {
                for prob in &mut probs {
                    *prob = (*prob).powf(1.0 / temperature);
                }
                let sum: f64 = probs.iter().sum();
                for prob in &mut probs {
                    *prob /= sum;
                }
            }

            // Sample next token (using simple deterministic selection for now)
            let next_token = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            generated.push(next_token);
        }

        Ok(generated)
    }

    /// Build the full language model circuit
    fn build_circuit(&self, context: &[usize]) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut gates = Vec::new();

        // 1. Embedding
        gates.extend(self.embedding.encode_sequence(context)?);

        // 2. Transformer layers
        for transformer in &self.transformer_layers {
            gates.extend(transformer.attention_gates()?);
        }

        // 3. Output projection
        for (i, param) in self.output_params.iter().enumerate() {
            let qubit = QubitId((i % self.config.text_qubits) as u32);
            gates.push(Box::new(ParametricRotationZ {
                target: qubit,
                theta: crate::parametric::Parameter::Constant(param.value),
            }));
        }

        Ok(gates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_word_embedding() {
        let config = QNLPConfig {
            vocab_size: 100,
            text_qubits: 4,
            ..Default::default()
        };

        let embedding = QuantumWordEmbedding::new(config);
        assert_eq!(embedding.num_qubits(), 4);

        // Test encoding a simple sequence
        let word_ids = vec![1, 5, 10];
        let gates = embedding.encode_sequence(&word_ids).unwrap();
        assert!(!gates.is_empty());
    }

    #[test]
    fn test_quantum_attention() {
        let attention = QuantumAttention::new(8, 2);
        assert_eq!(attention.num_qubits(), 8);
        assert_eq!(attention.num_heads, 2);

        let gates = attention.attention_gates().unwrap();
        assert!(!gates.is_empty());
    }

    #[test]
    fn test_quantum_text_classifier() {
        let config = QNLPConfig {
            vocab_size: 50,
            text_qubits: 4,
            feature_qubits: 2,
            ..Default::default()
        };

        let classifier = QuantumTextClassifier::new(config, 3);

        // Test classification
        let word_ids = vec![1, 2, 3];
        let probs = classifier.classify(&word_ids).unwrap();
        assert_eq!(probs.len(), 3);

        // Check probabilities sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_language_model() {
        let config = QNLPConfig {
            vocab_size: 20,
            text_qubits: 4,
            max_sequence_length: 8,
            ..Default::default()
        };

        let lm = QuantumLanguageModel::new(config);

        // Test next token prediction
        let context = vec![1, 2, 3];
        let probs = lm.predict_next_token(&context).unwrap();
        assert_eq!(probs.len(), 20);

        // Test text generation
        let generated = lm.generate_text(&context, 5, 1.0).unwrap();
        assert_eq!(generated.len(), 8); // 3 context + 5 generated
    }

    #[test]
    fn test_text_classifier_training() {
        let config = QNLPConfig {
            vocab_size: 10,
            text_qubits: 3,
            feature_qubits: 2,
            ..Default::default()
        };

        let mut classifier = QuantumTextClassifier::new(config, 2);

        // Create dummy training data
        let training_data = vec![
            (vec![1, 2], 0), // Class 0
            (vec![3, 4], 1), // Class 1
            (vec![1, 3], 0), // Class 0
            (vec![2, 4], 1), // Class 1
        ];

        let losses = classifier.train(&training_data, 0.01, 5).unwrap();
        assert_eq!(losses.len(), 5);
    }
}

/// Advanced quantum NLP utilities and algorithms
pub mod advanced {
    use super::*;

    /// Quantum text preprocessing utilities
    pub struct QuantumTextPreprocessor {
        /// Vocabulary mapping
        vocab: HashMap<String, usize>,
        /// Reverse vocabulary mapping
        reverse_vocab: HashMap<usize, String>,
        /// Special tokens
        special_tokens: HashMap<String, usize>,
    }

    impl QuantumTextPreprocessor {
        /// Create a new preprocessor
        pub fn new() -> Self {
            let mut special_tokens = HashMap::new();
            special_tokens.insert("<PAD>".to_string(), 0);
            special_tokens.insert("<UNK>".to_string(), 1);
            special_tokens.insert("<START>".to_string(), 2);
            special_tokens.insert("<END>".to_string(), 3);

            Self {
                vocab: HashMap::new(),
                reverse_vocab: HashMap::new(),
                special_tokens,
            }
        }

        /// Build vocabulary from text corpus
        pub fn build_vocab(&mut self, texts: &[String], max_vocab_size: usize) {
            let mut word_counts: HashMap<String, usize> = HashMap::new();

            // Count word frequencies
            for text in texts {
                for word in text.split_whitespace() {
                    *word_counts.entry(word.to_lowercase()).or_insert(0) += 1;
                }
            }

            // Sort by frequency and take top words
            let mut word_freq: Vec<_> = word_counts.into_iter().collect();
            word_freq.sort_by(|a, b| b.1.cmp(&a.1));

            // Add special tokens first
            for (token, id) in &self.special_tokens {
                self.vocab.insert(token.clone(), *id);
                self.reverse_vocab.insert(*id, token.clone());
            }

            // Add most frequent words
            let mut vocab_id = self.special_tokens.len();
            for (word, _count) in word_freq
                .into_iter()
                .take(max_vocab_size - self.special_tokens.len())
            {
                self.vocab.insert(word.clone(), vocab_id);
                self.reverse_vocab.insert(vocab_id, word);
                vocab_id += 1;
            }
        }

        /// Tokenize text to word IDs
        pub fn tokenize(&self, text: &str) -> Vec<usize> {
            let mut tokens = vec![self.special_tokens["<START>"]];

            for word in text.split_whitespace() {
                let word = word.to_lowercase();
                let token_id = self
                    .vocab
                    .get(&word)
                    .copied()
                    .unwrap_or(self.special_tokens["<UNK>"]);
                tokens.push(token_id);
            }

            tokens.push(self.special_tokens["<END>"]);
            tokens
        }

        /// Convert token IDs back to text
        pub fn detokenize(&self, token_ids: &[usize]) -> String {
            token_ids
                .iter()
                .filter_map(|&id| self.reverse_vocab.get(&id))
                .filter(|&word| !["<PAD>", "<START>", "<END>"].contains(&word.as_str()))
                .cloned()
                .collect::<Vec<_>>()
                .join(" ")
        }

        /// Get vocabulary size
        pub fn vocab_size(&self) -> usize {
            self.vocab.len()
        }
    }

    /// Quantum semantic similarity computation
    pub struct QuantumSemanticSimilarity {
        /// Embedding dimension
        embedding_dim: usize,
        /// Number of qubits
        num_qubits: usize,
        /// Similarity computation parameters
        similarity_params: Vec<Parameter>,
    }

    impl QuantumSemanticSimilarity {
        /// Create a new quantum semantic similarity computer
        pub fn new(embedding_dim: usize, num_qubits: usize) -> Self {
            let mut similarity_params = Vec::new();

            // Parameters for similarity computation
            for i in 0..num_qubits * 2 {
                // For two text inputs
                similarity_params.push(Parameter {
                    name: format!("sim_{}", i),
                    value: (i as f64 * 0.1).sin() * 0.5,
                    bounds: None,
                });
            }

            Self {
                embedding_dim,
                num_qubits,
                similarity_params,
            }
        }

        /// Compute semantic similarity between two texts
        pub fn compute_similarity(
            &self,
            text1_tokens: &[usize],
            text2_tokens: &[usize],
        ) -> QuantRS2Result<f64> {
            // Create embeddings for both texts
            let config = QNLPConfig {
                text_qubits: self.num_qubits,
                vocab_size: 1000, // Default
                ..Default::default()
            };

            let embedding1 = QuantumWordEmbedding::new(config.clone());
            let embedding2 = QuantumWordEmbedding::new(config);

            // Generate quantum circuits for both texts
            let gates1 = embedding1.encode_sequence(text1_tokens)?;
            let gates2 = embedding2.encode_sequence(text2_tokens)?;

            // Compute similarity using quantum interference
            // This is a simplified version - full implementation would measure overlap
            let similarity = self.quantum_text_overlap(gates1, gates2)?;

            Ok(similarity)
        }

        /// Compute quantum overlap between two text representations
        fn quantum_text_overlap(
            &self,
            _gates1: Vec<Box<dyn GateOp>>,
            _gates2: Vec<Box<dyn GateOp>>,
        ) -> QuantRS2Result<f64> {
            // Placeholder for quantum overlap computation
            // In practice, would:
            // 1. Prepare states using gates1 and gates2
            // 2. Compute fidelity/overlap between states
            // 3. Return similarity score

            // Return dummy similarity for now
            Ok(0.7)
        }
    }

    /// Quantum text summarization model
    pub struct QuantumTextSummarizer {
        /// Configuration
        config: QNLPConfig,
        /// Encoder for input text
        encoder: QuantumWordEmbedding,
        /// Attention mechanism for importance scoring
        attention: QuantumAttention,
        /// Summary generation parameters
        summary_params: Vec<Parameter>,
    }

    impl QuantumTextSummarizer {
        /// Create a new quantum text summarizer
        pub fn new(config: QNLPConfig) -> Self {
            let encoder = QuantumWordEmbedding::new(config.clone());
            let attention = QuantumAttention::new(config.text_qubits, config.num_attention_heads);

            let mut summary_params = Vec::new();
            for i in 0..config.text_qubits {
                summary_params.push(Parameter {
                    name: format!("summary_{}", i),
                    value: (i as f64 * 0.15).sin() * 0.4,
                    bounds: None,
                });
            }

            Self {
                config,
                encoder,
                attention,
                summary_params,
            }
        }

        /// Generate extractive summary from input text
        pub fn extractive_summarize(
            &self,
            text_tokens: &[usize],
            summary_length: usize,
        ) -> QuantRS2Result<Vec<usize>> {
            // Encode input text
            let _encoding_gates = self.encoder.encode_sequence(text_tokens)?;

            // Apply attention to find important tokens
            let _attention_gates = self.attention.attention_gates()?;

            // Score tokens for importance (simplified)
            let mut token_scores = Vec::new();
            for (i, &token) in text_tokens.iter().enumerate() {
                // Simple scoring based on token frequency and position
                let position_weight = 1.0 - (i as f64 / text_tokens.len() as f64) * 0.5;
                let token_weight = (token as f64 * 0.1).sin().abs();
                let score = position_weight * token_weight;
                token_scores.push((i, token, score));
            }

            // Sort by score and select top tokens
            token_scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

            let mut summary_tokens = Vec::new();
            for (_, token, _) in token_scores.into_iter().take(summary_length) {
                summary_tokens.push(token);
            }

            Ok(summary_tokens)
        }

        /// Generate abstractive summary (placeholder)
        pub fn abstractive_summarize(
            &self,
            _text_tokens: &[usize],
            _summary_length: usize,
        ) -> QuantRS2Result<Vec<usize>> {
            // Placeholder for abstractive summarization
            // Would use sequence-to-sequence quantum model
            Ok(vec![1, 2, 3]) // Dummy summary
        }
    }

    /// Quantum named entity recognition
    pub struct QuantumNamedEntityRecognition {
        /// Configuration
        config: QNLPConfig,
        /// Token encoder
        encoder: QuantumWordEmbedding,
        /// Entity type classifiers
        entity_classifiers: HashMap<String, Vec<Parameter>>,
        /// Supported entity types
        entity_types: Vec<String>,
    }

    impl QuantumNamedEntityRecognition {
        /// Create a new quantum NER model
        pub fn new(config: QNLPConfig) -> Self {
            let encoder = QuantumWordEmbedding::new(config.clone());
            let entity_types = vec![
                "PERSON".to_string(),
                "ORGANIZATION".to_string(),
                "LOCATION".to_string(),
                "DATE".to_string(),
                "MONEY".to_string(),
            ];

            let mut entity_classifiers = HashMap::new();
            for entity_type in &entity_types {
                let mut classifier_params = Vec::new();
                for i in 0..config.text_qubits {
                    classifier_params.push(Parameter {
                        name: format!("{}_{}", entity_type, i),
                        value: (entity_type.len() as f64 + i as f64 * 0.1).sin() * 0.3,
                        bounds: None,
                    });
                }
                entity_classifiers.insert(entity_type.clone(), classifier_params);
            }

            Self {
                config,
                encoder,
                entity_classifiers,
                entity_types,
            }
        }

        /// Recognize named entities in text
        pub fn recognize_entities(
            &self,
            text_tokens: &[usize],
        ) -> QuantRS2Result<Vec<(usize, usize, String)>> {
            let mut entities = Vec::new();

            // Simple sliding window approach
            for start in 0..text_tokens.len() {
                for end in start + 1..=text_tokens.len().min(start + 5) {
                    // Max entity length 5
                    let entity_tokens = &text_tokens[start..end];

                    // Classify this span
                    if let Some(entity_type) = self.classify_span(entity_tokens)? {
                        entities.push((start, end, entity_type));
                    }
                }
            }

            // Remove overlapping entities (keep longer ones)
            entities.sort_by(|a, b| (b.1 - b.0).cmp(&(a.1 - a.0)));
            let mut final_entities = Vec::new();
            let mut used_positions = vec![false; text_tokens.len()];

            for (start, end, entity_type) in entities {
                if used_positions[start..end].iter().all(|&used| !used) {
                    for pos in start..end {
                        used_positions[pos] = true;
                    }
                    final_entities.push((start, end, entity_type));
                }
            }

            final_entities.sort_by_key(|&(start, _, _)| start);
            Ok(final_entities)
        }

        /// Classify a span of tokens as an entity type
        fn classify_span(&self, tokens: &[usize]) -> QuantRS2Result<Option<String>> {
            // Encode the span
            let _encoding_gates = self.encoder.encode_sequence(tokens)?;

            let mut best_score = 0.0;
            let mut best_type = None;

            // Score each entity type
            for entity_type in &self.entity_types {
                let score = self.compute_entity_score(tokens, entity_type)?;
                if score > best_score && score > 0.5 {
                    // Threshold
                    best_score = score;
                    best_type = Some(entity_type.clone());
                }
            }

            Ok(best_type)
        }

        /// Compute score for a specific entity type
        fn compute_entity_score(&self, tokens: &[usize], entity_type: &str) -> QuantRS2Result<f64> {
            // Simple scoring based on token patterns
            let mut score = 0.0;

            for &token in tokens {
                // Simple heuristics based on token ID patterns
                match entity_type {
                    "PERSON" => {
                        if token % 7 == 1 {
                            // Arbitrary pattern for person names
                            score += 0.3;
                        }
                    }
                    "LOCATION" => {
                        if token % 5 == 2 {
                            // Arbitrary pattern for locations
                            score += 0.3;
                        }
                    }
                    "ORGANIZATION" => {
                        if token % 11 == 3 {
                            // Arbitrary pattern for organizations
                            score += 0.3;
                        }
                    }
                    "DATE" => {
                        if token % 13 == 4 {
                            // Arbitrary pattern for dates
                            score += 0.3;
                        }
                    }
                    "MONEY" => {
                        if token % 17 == 5 {
                            // Arbitrary pattern for money
                            score += 0.3;
                        }
                    }
                    _ => {}
                }
            }

            score /= tokens.len() as f64; // Normalize by span length
            Ok(score)
        }
    }
}

// Re-export advanced utilities
pub use advanced::*;
