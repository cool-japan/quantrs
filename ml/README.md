# QuantRS2 Machine Learning Module

[![Crates.io](https://img.shields.io/crates/v/quantrs2-ml.svg)](https://crates.io/crates/quantrs2-ml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

The `quantrs2-ml` crate provides quantum machine learning capabilities for the QuantRS2 quantum computing framework.

## Features

- **Quantum Neural Networks**: Parameterized quantum circuits for machine learning tasks
- **Variational Quantum Algorithms**: Hybrid quantum-classical optimization for complex problems
- **Application-Specific Modules**:
  - **High-Energy Physics**: Quantum classification of particle collision data from accelerators
  - **Generative Models**: Quantum GANs for data generation and augmentation
  - **Cryptography**: Post-quantum cryptographic algorithms and protocols
  - **Natural Language Processing**: Quantum-enhanced text understanding and generation
  - **Blockchain**: Quantum-secured distributed ledger technology

## Installation

The `quantrs2-ml` crate is included in the main QuantRS2 workspace. To use it in your project:

```toml
[dependencies]
quantrs2-ml = "0.1.0-alpha.2"
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
quantrs2-ml = { version = "0.1.0-alpha.2", features = ["gpu"] }
```

## License

This project is licensed under either of:

- MIT license ([LICENSE-MIT](../LICENSE-MIT) or https://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)

at your option.