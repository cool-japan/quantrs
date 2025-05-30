//! # Quantum Machine Learning
//!
//! This crate provides quantum machine learning capabilities for the QuantRS2 framework.
//! It includes quantum neural networks, variational algorithms, and specialized tools for
//! high-energy physics data analysis.
//!
//! ## Features
//!
//! - Quantum Neural Networks
//! - Variational Quantum Algorithms
//! - High-Energy Physics Data Analysis
//! - Quantum Reinforcement Learning
//! - Quantum Generative Models
//! - Quantum Kernels for Classification
//! - Quantum-Enhanced Cryptographic Protocols
//! - Quantum Blockchain and Distributed Ledger Technology
//! - Quantum-Enhanced Natural Language Processing

use fastrand;
use std::error::Error;
use thiserror::Error;

pub mod blockchain;
pub mod classification;
pub mod crypto;
pub mod gan;
pub mod hep;
pub mod kernels;
pub mod nlp;
pub mod optimization;
pub mod qnn;
pub mod qsvm;
pub mod reinforcement;
pub mod variational;

pub mod error;

/// Re-export error types for easier access
pub use error::MLError;
pub use error::Result;

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::blockchain::{ConsensusType, QuantumBlockchain, QuantumToken, SmartContract};
    pub use crate::classification::{ClassificationMetrics, Classifier};
    pub use crate::crypto::{
        ProtocolType, QuantumAuthentication, QuantumKeyDistribution, QuantumSignature,
    };
    pub use crate::error::{MLError, Result};
    pub use crate::gan::{Discriminator, GANEvaluationMetrics, Generator, QuantumGAN};
    pub use crate::hep::{
        AnomalyDetector, EventReconstructor, HEPQuantumClassifier, ParticleCollisionClassifier,
    };
    pub use crate::kernels::{KernelMethod, QuantumKernel};
    pub use crate::nlp::{NLPTaskType, QuantumLanguageModel, SentimentAnalyzer, TextSummarizer};
    pub use crate::optimization::{ObjectiveFunction, OptimizationMethod, Optimizer};
    pub use crate::qnn::{QNNBuilder, QNNLayer, QuantumNeuralNetwork};
    pub use crate::qsvm::{
        FeatureMapType, QSVMParams, QuantumKernel as QSVMKernel, QuantumKernelRidge, QSVM,
    };
    pub use crate::reinforcement::{Environment, QuantumAgent, ReinforcementLearning};
    pub use crate::variational::{VariationalAlgorithm, VariationalCircuit};
}
