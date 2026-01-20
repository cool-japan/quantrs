# QuantRS2-ML: Advanced Quantum Machine Learning Suite

[![Crates.io](https://img.shields.io/crates/v/quantrs2-ml.svg)](https://crates.io/crates/quantrs2-ml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

QuantRS2-ML is the comprehensive quantum machine learning library of the [QuantRS2](https://github.com/cool-japan/quantrs) quantum computing framework, providing cutting-edge quantum algorithms, hybrid architectures, and industry-specific applications for next-generation artificial intelligence and data science.

## Version 0.1.0

This release incorporates [SciRS2](https://github.com/cool-japan/scirs2) v0.1.0 with refined integration patterns for enhanced performance:
- Automatic differentiation leveraging SciRS2's linear algebra operations
- Parallel training with `scirs2_core::parallel_ops`
- SIMD-accelerated quantum kernel computations
- Memory-efficient handling of large quantum datasets

## Comprehensive Features

### Core Quantum Machine Learning
- **Quantum Neural Networks (QNN)**: Parameterized quantum circuits with automatic differentiation
- **Variational Quantum Algorithms**: VQE, QAOA, and hybrid optimization frameworks
- **Quantum Convolutional Networks (QCNN)**: Quantum feature maps with pooling operations
- **Quantum Support Vector Machines (QSVM)**: Kernel methods with quantum advantage
- **Quantum Autoencoders (QVAE)**: Dimensionality reduction and representation learning

### Advanced Deep Learning Architectures
- **Quantum Transformers**: Attention mechanisms with quantum features
- **Quantum LSTM Networks**: Recurrent architectures for sequence modeling
- **Quantum Graph Neural Networks**: Node and edge processing with quantum features
- **Quantum Diffusion Models**: Generative modeling with quantum denoising
- **Quantum Boltzmann Machines**: Energy-based models with quantum sampling

### Generative AI & Large Models
- **Quantum GANs**: Generative adversarial networks with Wasserstein loss
- **Quantum Large Language Models**: Transformer-based text generation with quantum layers
- **Quantum Computer Vision**: Image processing and recognition with quantum features
- **Quantum Recommender Systems**: Collaborative filtering with quantum kernels
- **Quantum Anomaly Detection**: Unsupervised learning for outlier identification

### Specialized Applications
- **High-Energy Physics**: Particle collision classification and analysis
- **Quantum Cryptography**: Post-quantum security and key distribution protocols
- **Blockchain Integration**: Quantum-secured distributed ledger technology
- **Federated Learning**: Privacy-preserving distributed quantum ML
- **Time Series Forecasting**: Financial and scientific data prediction

### Advanced Training & Optimization
- **Meta-Learning**: Few-shot and transfer learning with quantum adaptation
- **Neural Architecture Search**: Automated quantum circuit design
- **Adversarial Training**: Robustness against quantum attacks
- **Continual Learning**: Lifelong learning without catastrophic forgetting
- **AutoML**: Automated hyperparameter optimization and model selection

## Installation

The `quantrs2-ml` crate is included in the main QuantRS2 workspace. To use it in your project:

```toml
[dependencies]
quantrs2-ml = "0.1.0"
```

## Usage Examples

### Quantum Neural Network

```rust
use quantrs2_ml::prelude::*;
use quantrs2_ml::qnn::{QuantumNeuralNetwork, QNNLayer};

// Create a QNN with a custom architecture
let layers = vec![
    QNNLayer::EncodingLayer { num_features: 4 },
    QNNLayer::VariationalLayer { num_params: 18 },
    QNNLayer::EntanglementLayer { connectivity: "full".to_string() },
    QNNLayer::VariationalLayer { num_params: 18 },
    QNNLayer::MeasurementLayer { measurement_basis: "computational".to_string() },
];

let qnn = QuantumNeuralNetwork::new(
    layers, 
    6,     // 6 qubits
    4,     // 4 input features
    2,     // 2 output classes
)?;

// Train on data
let optimizer = Optimizer::Adam { learning_rate: 0.01 };
let result = qnn.train(&x_train, &y_train, optimizer, 100)?;
```

### High-Energy Physics Classification

```rust
use quantrs2_ml::prelude::*;
use quantrs2_ml::hep::{HEPQuantumClassifier, HEPEncodingMethod};

// Create a classifier for HEP data
let classifier = HEPQuantumClassifier::new(
    8,                              // 8 qubits
    10,                             // 10 features
    2,                              // binary classification
    HEPEncodingMethod::HybridEncoding,
    vec!["background".to_string(), "signal".to_string()],
)?;

// Train and evaluate
let training_result = classifier.train(&train_data, &train_labels, 100, 0.01)?;
let metrics = classifier.evaluate(&test_data, &test_labels)?;

println!("Test accuracy: {:.2}%", metrics.accuracy * 100.0);
```

### Quantum Generative Adversarial Network

```rust
use quantrs2_ml::prelude::*;
use quantrs2_ml::gan::{QuantumGAN, GeneratorType, DiscriminatorType};

// Create a quantum GAN
let qgan = QuantumGAN::new(
    6,                                      // 6 qubits for generator
    6,                                      // 6 qubits for discriminator
    4,                                      // 4D latent space
    8,                                      // 8D data space
    GeneratorType::HybridClassicalQuantum,
    DiscriminatorType::HybridQuantumFeatures,
)?;

// Train on data
let history = qgan.train(
    &real_data,
    50,    // epochs
    16,    // batch size
    0.01,  // generator learning rate
    0.01,  // discriminator learning rate
    1,     // discriminator steps per generator step
)?;

// Generate new samples
let generated_samples = qgan.generate(10)?;
```

### Quantum Cryptography

```rust
use quantrs2_ml::prelude::*;
use quantrs2_ml::crypto::{QuantumKeyDistribution, ProtocolType};

// Create a BB84 quantum key distribution protocol
let mut qkd = QuantumKeyDistribution::new(ProtocolType::BB84, 1000)
    .with_error_rate(0.03);

// Distribute a key
let key_length = qkd.distribute_key()?;
println!("Generated key of length: {} bits", key_length);

// Verify that Alice and Bob have the same key
if qkd.verify_keys() {
    println!("Key distribution successful!");
}
```

## GPU Acceleration

The `quantrs2-ml` crate supports GPU acceleration for quantum machine learning tasks through the `gpu` feature:

```toml
[dependencies]
quantrs2-ml = { version = "0.1.0", features = ["gpu"] }
```

## License

This project is licensed under either of:

- MIT license ([LICENSE-MIT](../LICENSE-MIT) or https://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)

at your option.