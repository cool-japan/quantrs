//! Quantum natural language processing (QNLP) models and utilities.
//!
//! Provides quantum circuit encodings for text data and [`QuantumNLPModel`]
//! for tasks such as classification, sequence labelling, and question
//! answering using quantum neural network backends.

use crate::error::{MLError, Result};
use crate::qnn::QuantumNeuralNetwork;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use std::fmt;

/// Type of NLP task
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NLPTaskType {
    /// Text classification
    Classification,

    /// Sequence labeling
    SequenceLabeling,

    /// Machine translation
    Translation,

    /// Language generation
    Generation,

    /// Sentiment analysis
    SentimentAnalysis,

    /// Text summarization
    Summarization,
}

/// Strategy for text embedding
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmbeddingStrategy {
    /// Bag of words
    BagOfWords,

    /// Term frequency-inverse document frequency
    TFIDF,

    /// Word2Vec
    Word2Vec,

    /// Custom embedding
    Custom,
}

impl From<usize> for EmbeddingStrategy {
    fn from(value: usize) -> Self {
        match value {
            0 => EmbeddingStrategy::BagOfWords,
            1 => EmbeddingStrategy::TFIDF,
            2 => EmbeddingStrategy::Word2Vec,
            _ => EmbeddingStrategy::Custom,
        }
    }
}

/// Text preprocessing for NLP
#[derive(Debug, Clone)]
pub struct TextPreprocessor {
    /// Whether to convert to lowercase
    pub lowercase: bool,

    /// Whether to remove stopwords
    pub remove_stopwords: bool,

    /// Whether to lemmatize
    pub lemmatize: bool,

    /// Whether to stem
    pub stem: bool,

    /// Custom stopwords
    pub stopwords: Vec<String>,
}

impl TextPreprocessor {
    /// Creates a new text preprocessor with default settings
    pub fn new() -> Self {
        TextPreprocessor {
            lowercase: true,
            remove_stopwords: true,
            lemmatize: false,
            stem: false,
            stopwords: Vec::new(),
        }
    }

    /// Sets whether to convert to lowercase
    pub fn with_lowercase(mut self, lowercase: bool) -> Self {
        self.lowercase = lowercase;
        self
    }

    /// Sets whether to remove stopwords
    pub fn with_remove_stopwords(mut self, remove_stopwords: bool) -> Self {
        self.remove_stopwords = remove_stopwords;
        self
    }

    /// Sets whether to lemmatize
    pub fn with_lemmatize(mut self, lemmatize: bool) -> Self {
        self.lemmatize = lemmatize;
        self
    }

    /// Sets whether to stem
    pub fn with_stem(mut self, stem: bool) -> Self {
        self.stem = stem;
        self
    }

    /// Sets custom stopwords
    pub fn with_stopwords(mut self, stopwords: Vec<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Preprocesses text
    pub fn preprocess(&self, text: &str) -> Result<String> {
        // This is a dummy implementation
        // In a real system, this would apply the specified preprocessing steps

        let mut processed = text.to_string();

        if self.lowercase {
            processed = processed.to_lowercase();
        }

        if self.remove_stopwords {
            for stopword in &self.stopwords {
                processed = processed.replace(stopword, "");
            }
        }

        Ok(processed)
    }

    /// Tokenizes text
    pub fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        // This is a dummy implementation
        // In a real system, this would use a proper tokenizer

        let processed = self.preprocess(text)?;
        let tokens = processed
            .split_whitespace()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();

        Ok(tokens)
    }
}

/// Word embedding for text representation
#[derive(Debug, Clone)]
pub struct WordEmbedding {
    /// Embedding strategy
    pub strategy: EmbeddingStrategy,

    /// Embedding dimension
    pub dimension: usize,

    /// Word-to-embedding mapping
    pub embeddings: HashMap<String, Array1<f64>>,

    /// Vocabulary
    pub vocabulary: Vec<String>,
}

impl WordEmbedding {
    /// Creates a new word embedding
    pub fn new(strategy: EmbeddingStrategy, dimension: usize) -> Self {
        WordEmbedding {
            strategy,
            dimension,
            embeddings: HashMap::new(),
            vocabulary: Vec::new(),
        }
    }

    /// Fits the embedding on a corpus using Random Indexing (Kanerva et al.;
    /// see also Sahlgren, 2005): each vocabulary word is assigned a fixed,
    /// sparse, near-orthogonal random "index vector"; a word's embedding is
    /// then the (L2-normalized) sum of the index vectors of every word that
    /// co-occurs with it within a sliding window across the corpus.
    ///
    /// This makes embeddings depend on real corpus co-occurrence statistics
    /// -- words that tend to appear in similar contexts end up with
    /// correlated embeddings -- unlike drawing an independent random vector
    /// per word regardless of context (which was the previous behavior for
    /// every [`EmbeddingStrategy`]). Random Indexing is used here as the one
    /// concrete real backend for all strategy variants; a full trained
    /// skip-gram/CBOW Word2Vec model and TF-IDF-weighted bag-of-words are
    /// not implemented separately in this release.
    pub fn fit(&mut self, corpus: &[&str]) -> Result<()> {
        const WINDOW_RADIUS: usize = 2;
        const INDEX_VECTOR_NONZEROS: usize = 4;

        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let tokenized_corpus: Vec<Vec<String>> = corpus
            .iter()
            .map(|text| {
                let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
                for token in &tokens {
                    *word_counts.entry(token.clone()).or_insert(0) += 1;
                }
                tokens
            })
            .collect();

        // Build the vocabulary, sorted by descending frequency.
        let mut vocab_items: Vec<(String, usize)> = word_counts
            .iter()
            .map(|(word, count)| (word.clone(), *count))
            .collect();
        vocab_items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        self.vocabulary = vocab_items
            .into_iter()
            .map(|(word, _)| word)
            .take(10000)
            .collect();

        self.embeddings.clear();
        if self.vocabulary.is_empty() {
            return Ok(());
        }

        let word_index: HashMap<&str, usize> = self
            .vocabulary
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();

        // Fixed, sparse, near-orthogonal index vector per vocabulary word.
        let mut rng = thread_rng();
        let index_vectors: Vec<Array1<f64>> = (0..self.vocabulary.len())
            .map(|_| {
                let mut vector = Array1::<f64>::zeros(self.dimension);
                let nonzeros = INDEX_VECTOR_NONZEROS.min(self.dimension);
                let mut placed = 0;
                let mut attempts = 0;
                while placed < nonzeros && attempts < nonzeros * 20 {
                    attempts += 1;
                    let raw_position = (rng.random::<f64>() * self.dimension as f64) as usize;
                    let position = raw_position.min(self.dimension.saturating_sub(1));
                    if vector[position] == 0.0 {
                        let sign = if rng.random::<f64>() < 0.5 { -1.0 } else { 1.0 };
                        vector[position] = sign;
                        placed += 1;
                    }
                }
                vector
            })
            .collect();

        // Accumulate context vectors from real corpus co-occurrence within a
        // sliding window, restricted to in-vocabulary words.
        let mut context_vectors: Vec<Array1<f64>> = (0..self.vocabulary.len())
            .map(|_| Array1::<f64>::zeros(self.dimension))
            .collect();

        for tokens in &tokenized_corpus {
            let indices: Vec<Option<usize>> = tokens
                .iter()
                .map(|token| word_index.get(token.as_str()).copied())
                .collect();

            for (position, target_idx_opt) in indices.iter().enumerate() {
                let target_idx = match target_idx_opt {
                    Some(idx) => *idx,
                    None => continue,
                };
                let window_start = position.saturating_sub(WINDOW_RADIUS);
                let window_end = (position + WINDOW_RADIUS + 1).min(indices.len());
                for context_position in window_start..window_end {
                    if context_position == position {
                        continue;
                    }
                    if let Some(context_idx) = indices[context_position] {
                        context_vectors[target_idx] =
                            &context_vectors[target_idx] + &index_vectors[context_idx];
                    }
                }
            }
        }

        for (i, word) in self.vocabulary.iter().enumerate() {
            let mut embedding = context_vectors[i].clone();
            let norm = embedding.dot(&embedding).sqrt();
            if norm > 1e-10 {
                embedding.mapv_inplace(|x| x / norm);
            } else {
                // Word never co-occurred with any in-vocabulary word within
                // the window (e.g. it only appears in single-word corpus
                // entries): fall back to its own deterministic index vector
                // rather than leaving an all-zero embedding.
                embedding = index_vectors[i].clone();
            }
            self.embeddings.insert(word.clone(), embedding);
        }

        Ok(())
    }

    /// Gets the embedding for a word
    pub fn get_embedding(&self, word: &str) -> Option<&Array1<f64>> {
        self.embeddings.get(word)
    }

    /// Gets the embedding for a sentence
    pub fn embed_text(&self, text: &str) -> Result<Array1<f64>> {
        // This is a simplified implementation
        // In a real system, this would properly combine word embeddings

        let words = text.split_whitespace().collect::<Vec<_>>();
        let mut embedding = Array1::zeros(self.dimension);
        let mut count = 0;

        for word in words {
            if let Some(word_embedding) = self.get_embedding(word) {
                embedding += word_embedding;
                count += 1;
            }
        }

        if count > 0 {
            embedding /= count as f64;
        }

        Ok(embedding)
    }
}

/// Quantum language model for NLP tasks
#[derive(Debug, Clone)]
pub struct QuantumLanguageModel {
    /// Number of qubits
    pub num_qubits: usize,

    /// Embedding strategy
    pub embedding_strategy: EmbeddingStrategy,

    /// Text preprocessor
    pub preprocessor: TextPreprocessor,

    /// Word embedding
    pub embedding: WordEmbedding,

    /// Quantum neural network
    pub qnn: QuantumNeuralNetwork,

    /// Type of NLP task
    pub task: NLPTaskType,

    /// Class labels (for classification tasks)
    pub labels: Vec<String>,
}

impl QuantumLanguageModel {
    /// Creates a new quantum language model
    pub fn new(
        num_qubits: usize,
        embedding_dimension: usize,
        strategy: EmbeddingStrategy,
        task: NLPTaskType,
        labels: Vec<String>,
    ) -> Result<Self> {
        let preprocessor = TextPreprocessor::new();
        let embedding = WordEmbedding::new(strategy, embedding_dimension);

        // Create a QNN architecture suitable for the task
        let layers = vec![
            crate::qnn::QNNLayerType::EncodingLayer {
                num_features: embedding_dimension,
            },
            crate::qnn::QNNLayerType::VariationalLayer {
                num_params: 2 * num_qubits,
            },
            crate::qnn::QNNLayerType::EntanglementLayer {
                connectivity: "full".to_string(),
            },
            crate::qnn::QNNLayerType::VariationalLayer {
                num_params: 2 * num_qubits,
            },
            crate::qnn::QNNLayerType::MeasurementLayer {
                measurement_basis: "computational".to_string(),
            },
        ];

        let output_dim = match task {
            NLPTaskType::Classification | NLPTaskType::SentimentAnalysis => labels.len(),
            NLPTaskType::SequenceLabeling => labels.len(),
            NLPTaskType::Translation => embedding_dimension,
            NLPTaskType::Generation => embedding_dimension,
            NLPTaskType::Summarization => embedding_dimension,
        };

        let qnn = QuantumNeuralNetwork::new(layers, num_qubits, embedding_dimension, output_dim)?;

        Ok(QuantumLanguageModel {
            num_qubits,
            embedding_strategy: strategy,
            preprocessor,
            embedding,
            qnn,
            task,
            labels,
        })
    }

    /// Fits the model on a corpus
    pub fn fit(&mut self, texts: &[&str], labels: &[usize]) -> Result<()> {
        // First, fit the embedding on the corpus
        self.embedding.fit(texts)?;

        // Convert texts to embeddings
        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let embedding = self.embedding.embed_text(text)?;
            embeddings.push(embedding);
        }

        // Convert to ndarray
        let x_train = Array2::from_shape_vec(
            (embeddings.len(), self.embedding.dimension),
            embeddings.iter().flat_map(|e| e.iter().cloned()).collect(),
        )
        .map_err(|e| MLError::DataError(format!("Failed to create training data: {}", e)))?;

        // Convert labels to one-hot encoding
        let y_train = Array1::from_vec(labels.iter().map(|&l| l as f64).collect());

        // Train the QNN
        self.qnn.train_1d(&x_train, &y_train, 100, 0.01)?;

        Ok(())
    }

    /// Predicts the label for a text
    pub fn predict(&self, text: &str) -> Result<(String, f64)> {
        // Embed the text
        let embedding = self.embedding.embed_text(text)?;

        // Run the QNN
        let output = self.qnn.forward(&embedding)?;

        // Find the label with the highest score
        let mut best_label = 0;
        let mut best_score = output[0];

        for i in 1..output.len() {
            if output[i] > best_score {
                best_score = output[i];
                best_label = i;
            }
        }

        if best_label < self.labels.len() {
            Ok((self.labels[best_label].clone(), best_score))
        } else {
            Err(MLError::MLOperationError(format!(
                "Invalid prediction index: {}",
                best_label
            )))
        }
    }
}

/// Sentiment analyzer using quantum language models
#[derive(Debug, Clone)]
pub struct SentimentAnalyzer {
    /// Quantum language model
    model: QuantumLanguageModel,
}

impl SentimentAnalyzer {
    /// Creates a new sentiment analyzer
    pub fn new(num_qubits: usize) -> Result<Self> {
        let model = QuantumLanguageModel::new(
            num_qubits,
            32, // embedding dimension
            EmbeddingStrategy::BagOfWords,
            NLPTaskType::SentimentAnalysis,
            vec![
                "negative".to_string(),
                "neutral".to_string(),
                "positive".to_string(),
            ],
        )?;

        Ok(SentimentAnalyzer { model })
    }

    /// Analyzes the sentiment of text
    pub fn analyze(&self, text: &str) -> Result<(String, f64)> {
        self.model.predict(text)
    }

    /// Trains the sentiment analyzer
    pub fn train(&mut self, texts: &[&str], labels: &[usize]) -> Result<()> {
        self.model.fit(texts, labels)
    }
}

/// Text summarizer using quantum language models
#[derive(Debug, Clone)]
pub struct TextSummarizer {
    /// Quantum language model
    model: QuantumLanguageModel,

    /// Maximum summary length
    max_length: usize,
}

impl TextSummarizer {
    /// Creates a new text summarizer
    pub fn new(num_qubits: usize) -> Result<Self> {
        let model = QuantumLanguageModel::new(
            num_qubits,
            64, // embedding dimension
            EmbeddingStrategy::BagOfWords,
            NLPTaskType::Summarization,
            Vec::new(), // No specific labels for summarization
        )?;

        Ok(TextSummarizer {
            model,
            max_length: 100,
        })
    }

    /// Sets the maximum summary length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Summarizes text
    pub fn summarize(&self, text: &str) -> Result<String> {
        // This is a dummy implementation
        // In a real system, this would use the quantum language model to generate a summary

        let sentences = text.split('.').collect::<Vec<_>>();
        let num_sentences = sentences.len();

        // Generate a summary by selecting key sentences
        let num_summary_sentences = (num_sentences / 4).max(1);
        let selected_indices = vec![0, num_sentences / 2, num_sentences - 1];

        let mut summary = String::new();

        for &index in selected_indices.iter().take(num_summary_sentences) {
            if index < sentences.len() {
                summary.push_str(sentences[index]);
                summary.push('.');
            }
        }

        // Truncate to max length if needed
        if summary.len() > self.max_length {
            let truncated = summary.chars().take(self.max_length).collect::<String>();
            let last_space = truncated.rfind(' ').unwrap_or(truncated.len());
            summary = truncated[..last_space].to_string();
            summary.push_str("...");
        }

        Ok(summary)
    }
}

impl fmt::Display for NLPTaskType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NLPTaskType::Classification => write!(f, "Classification"),
            NLPTaskType::SequenceLabeling => write!(f, "Sequence Labeling"),
            NLPTaskType::Translation => write!(f, "Translation"),
            NLPTaskType::Generation => write!(f, "Generation"),
            NLPTaskType::SentimentAnalysis => write!(f, "Sentiment Analysis"),
            NLPTaskType::Summarization => write!(f, "Summarization"),
        }
    }
}

impl fmt::Display for EmbeddingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbeddingStrategy::BagOfWords => write!(f, "Bag of Words"),
            EmbeddingStrategy::TFIDF => write!(f, "TF-IDF"),
            EmbeddingStrategy::Word2Vec => write!(f, "Word2Vec"),
            EmbeddingStrategy::Custom => write!(f, "Custom"),
        }
    }
}

/// Implementation of missing methods for QuantumLanguageModel
impl QuantumLanguageModel {
    /// Builds vocabulary from a set of texts
    pub fn build_vocabulary(&mut self, texts: &[String]) -> Result<usize> {
        // In a full implementation, this would analyze texts and build vocabulary
        // For now, just return a dummy vocabulary size
        let vocab_size = texts
            .iter()
            .flat_map(|text| text.split_whitespace())
            .collect::<std::collections::HashSet<_>>()
            .len();

        Ok(vocab_size)
    }

    /// Trains word embeddings
    pub fn train_embeddings(&mut self, texts: &[String]) -> Result<()> {
        // Dummy implementation that would train word embeddings
        // In reality, this would update the embedding matrix based on texts
        println!(
            "  Training embeddings for {} texts with strategy: {}",
            texts.len(),
            self.embedding_strategy
        );

        Ok(())
    }

    /// Trains the language model
    pub fn train(
        &mut self,
        texts: &[String],
        labels: &[usize],
        epochs: usize,
        learning_rate: f64,
    ) -> Result<()> {
        // Convert texts to feature vectors using the embedding
        let num_samples = texts.len();
        let mut features = Array2::zeros((num_samples, self.embedding.dimension));

        // Create dummy features
        for (i, text) in texts.iter().enumerate() {
            // Simple hash-based feature extraction
            let feature_vec = text
                .chars()
                .enumerate()
                .map(|(j, c)| (c as u32 % 8) as f64 / 8.0 + j as f64 * 0.001)
                .take(self.embedding.dimension)
                .collect::<Vec<_>>();

            for (j, &val) in feature_vec
                .iter()
                .enumerate()
                .take(self.embedding.dimension)
            {
                if j < features.ncols() {
                    features[[i, j]] = val;
                }
            }
        }

        // Convert labels to float array
        let y_train = Array1::from_vec(labels.iter().map(|&l| l as f64).collect());

        // Train the underlying QNN
        self.qnn
            .train_1d(&features, &y_train, epochs, learning_rate)?;

        Ok(())
    }

    /// Classifies a text
    pub fn classify(&self, text: &str) -> Result<(String, f64)> {
        // In a real implementation, this would encode the text and run it through the QNN

        // Simple hash-based classification for demonstration
        let hash = text.chars().map(|c| c as u32).sum::<u32>();
        let class_idx = (hash % self.labels.len() as u32) as usize;
        let confidence = 0.7 + 0.3 * (hash % 100) as f64 / 100.0;

        Ok((self.labels[class_idx].clone(), confidence))
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    /// Regression test for the "every word gets an independent random
    /// embedding" fabrication bug: words that share contexts across the
    /// corpus should end up with embeddings that are meaningfully more
    /// similar (higher cosine similarity) than words that never co-occur
    /// with anything, which is only possible if `fit` actually derives
    /// embeddings from real co-occurrence statistics.
    #[test]
    fn fit_produces_context_correlated_embeddings_not_pure_noise() {
        let corpus = [
            "king queen throne royal palace",
            "queen king throne royal crown",
            "throne king queen royal power",
            "banana apple fruit sweet tasty",
            "apple banana fruit juicy sweet",
            "fruit apple banana tasty juicy",
        ];

        let mut embedding = WordEmbedding::new(EmbeddingStrategy::Word2Vec, 64);
        embedding.fit(&corpus).expect("fit should succeed");

        assert!(!embedding.vocabulary.is_empty());
        assert!(embedding.get_embedding("king").is_some());
        assert!(embedding.get_embedding("apple").is_some());

        let cosine = |a: &Array1<f64>, b: &Array1<f64>| -> f64 {
            let dot = a.dot(b);
            let norm_a = a.dot(a).sqrt();
            let norm_b = b.dot(b).sqrt();
            if norm_a > 1e-12 && norm_b > 1e-12 {
                dot / (norm_a * norm_b)
            } else {
                0.0
            }
        };

        let king = embedding.get_embedding("king").expect("king embedded");
        let queen = embedding.get_embedding("queen").expect("queen embedded");
        let apple = embedding.get_embedding("apple").expect("apple embedded");

        // "king" and "queen" repeatedly co-occur with the same royalty
        // context words across the corpus, so their embeddings should be
        // noticeably more similar to each other than "king" is to the
        // unrelated "apple" -- a signal that cannot exist if embeddings are
        // independent random noise per word.
        let king_queen_similarity = cosine(king, queen);
        let king_apple_similarity = cosine(king, apple);
        assert!(
            king_queen_similarity > king_apple_similarity,
            "expected king~queen similarity ({king_queen_similarity}) to exceed \
             king~apple similarity ({king_apple_similarity})"
        );

        // Fitting twice on the same corpus must reproduce the same
        // vocabulary (a real, deterministic frequency-based selection).
        let mut embedding2 = WordEmbedding::new(EmbeddingStrategy::Word2Vec, 64);
        embedding2.fit(&corpus).expect("fit should succeed");
        assert_eq!(embedding.vocabulary, embedding2.vocabulary);
    }

    #[test]
    fn fit_on_empty_corpus_yields_empty_vocabulary_and_no_panic() {
        let mut embedding = WordEmbedding::new(EmbeddingStrategy::BagOfWords, 16);
        embedding
            .fit(&[])
            .expect("fit on empty corpus should succeed");
        assert!(embedding.vocabulary.is_empty());
        assert!(embedding.embeddings.is_empty());
    }
}
