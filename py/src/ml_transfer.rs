//! Python bindings for quantum transfer learning functionality

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

#[cfg(feature = "ml")]
use quantrs2_ml::transfer::{PretrainedModel, QuantumTransferLearning, TransferStrategy};

#[cfg(feature = "ml")]
use quantrs2_ml::autodiff::optimizers::Optimizer;

/// Python wrapper for transfer learning strategies
#[pyclass]
#[derive(Clone)]
pub enum PyTransferStrategy {
    /// Fine-tune the last n layers
    FineTuning { num_trainable_layers: usize },
    /// Use as feature extractor (freeze all layers)
    FeatureExtraction {},
    /// Selective adaptation of specific layers
    SelectiveAdaptation {},
    /// Progressive unfreezing during training
    ProgressiveUnfreezing { unfreeze_rate: usize },
}

#[cfg(feature = "ml")]
impl From<PyTransferStrategy> for TransferStrategy {
    fn from(py_strategy: PyTransferStrategy) -> Self {
        match py_strategy {
            PyTransferStrategy::FineTuning {
                num_trainable_layers,
            } => Self::FineTuning {
                num_trainable_layers,
            },
            PyTransferStrategy::FeatureExtraction {} => Self::FeatureExtraction,
            PyTransferStrategy::SelectiveAdaptation {} => Self::SelectiveAdaptation,
            PyTransferStrategy::ProgressiveUnfreezing { unfreeze_rate } => {
                Self::ProgressiveUnfreezing { unfreeze_rate }
            }
        }
    }
}

/// Python wrapper for transfer learning configuration
#[pyclass]
pub struct PyTransferConfig {
    #[pyo3(get, set)]
    pub learning_rate_scaling: f64,
    #[pyo3(get, set)]
    pub regularization_strength: f64,
    #[pyo3(get, set)]
    pub warmup_epochs: usize,
    #[pyo3(get, set)]
    pub freeze_batch_norm: bool,
}

#[allow(clippy::missing_const_for_fn)]
#[pymethods]
impl PyTransferConfig {
    #[new]
    pub fn new() -> Self {
        Self {
            learning_rate_scaling: 0.1,
            regularization_strength: 0.01,
            warmup_epochs: 5,
            freeze_batch_norm: true,
        }
    }
}

impl Default for PyTransferConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Python wrapper for pretrained quantum model
#[pyclass]
pub struct PyPretrainedModel {
    #[cfg(feature = "ml")]
    inner: Option<PretrainedModel>,

    // Store model info for non-ML builds
    name: String,
    description: String,
}

#[allow(clippy::missing_const_for_fn)]
#[pymethods]
impl PyPretrainedModel {
    /// Get model name
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Get model description
    #[getter]
    fn description(&self) -> &str {
        &self.description
    }

    /// Get number of qubits
    fn n_qubits(&self) -> PyResult<usize> {
        #[cfg(feature = "ml")]
        {
            self.inner.as_ref().map_or_else(
                || Err(PyValueError::new_err("Model not initialized")),
                |model| Ok(model.qnn.num_qubits),
            )
        }

        #[cfg(not(feature = "ml"))]
        {
            Err(PyValueError::new_err(
                "ML features not enabled. Install with 'pip install quantrs2[ml]'",
            ))
        }
    }

    /// Get number of layers
    fn n_layers(&self) -> PyResult<usize> {
        #[cfg(feature = "ml")]
        {
            self.inner.as_ref().map_or_else(
                || Err(PyValueError::new_err("Model not initialized")),
                |model| Ok(model.qnn.layers.len()),
            )
        }

        #[cfg(not(feature = "ml"))]
        {
            Err(PyValueError::new_err("ML features not enabled"))
        }
    }
}

/// Python wrapper for quantum transfer learning
#[pyclass]
pub struct PyQuantumTransferLearning {
    #[cfg(feature = "ml")]
    inner: Option<QuantumTransferLearning>,

    strategy: PyTransferStrategy,
}

#[allow(clippy::missing_const_for_fn)]
#[pymethods]
impl PyQuantumTransferLearning {
    #[new]
    pub fn new(
        pretrained_model: &PyPretrainedModel,
        strategy: PyTransferStrategy,
    ) -> PyResult<Self> {
        #[cfg(feature = "ml")]
        {
            pretrained_model.inner.as_ref().map_or_else(
                || Err(PyValueError::new_err("Invalid pretrained model")),
                |_model| {
                    // Note: Using simplified constructor since TransferConfig is not available
                    // The actual API might need adjustment based on the Rust implementation
                    Ok(Self {
                        inner: None, // Set to None for now due to API incompatibility
                        strategy,
                    })
                },
            )
        }

        #[cfg(not(feature = "ml"))]
        {
            Ok(Self { strategy })
        }
    }

    /// Get current strategy
    #[getter]
    fn strategy(&self) -> PyTransferStrategy {
        self.strategy.clone()
    }

    /// Adapt model for new task
    fn adapt_for_task(&mut self, n_outputs: usize) -> PyResult<()> {
        #[cfg(feature = "ml")]
        {
            self.inner.as_mut().map_or_else(
                || Err(PyValueError::new_err("Transfer learning not initialized")),
                |_qtl| {
                    // Note: adapt_for_task method not yet implemented in the Rust side
                    // This would need to be implemented in the QuantumTransferLearning struct
                    Err(PyValueError::new_err(
                        "adapt_for_task method not yet implemented",
                    ))
                },
            )
        }

        #[cfg(not(feature = "ml"))]
        {
            Err(PyValueError::new_err("ML features not enabled"))
        }
    }

    /// Get trainable parameters count
    fn trainable_parameters(&self) -> PyResult<usize> {
        #[cfg(feature = "ml")]
        {
            self.inner.as_ref().map_or_else(
                || Err(PyValueError::new_err("Transfer learning not initialized")),
                |_qtl| {
                    // Note: trainable_parameters method not yet implemented in the Rust side
                    Err(PyValueError::new_err(
                        "trainable_parameters method not yet implemented",
                    ))
                },
            )
        }

        #[cfg(not(feature = "ml"))]
        {
            Err(PyValueError::new_err("ML features not enabled"))
        }
    }
}

/// Model zoo for pretrained quantum models
#[pyclass]
pub struct PyQuantumModelZoo;

#[allow(clippy::missing_const_for_fn)]
#[pymethods]
impl PyQuantumModelZoo {
    /// Get VQE feature extractor model
    #[staticmethod]
    fn vqe_feature_extractor(n_qubits: usize) -> PyResult<PyPretrainedModel> {
        #[cfg(feature = "ml")]
        {
            use quantrs2_ml::transfer::QuantumModelZoo;

            let model = QuantumModelZoo::vqe_feature_extractor(n_qubits)
                .map_err(|e| PyValueError::new_err(format!("Failed to create model: {e}")))?;

            Ok(PyPretrainedModel {
                inner: Some(model),
                name: "VQE Feature Extractor".to_string(),
                description: format!("Pretrained VQE model for {n_qubits} qubits"),
            })
        }

        #[cfg(not(feature = "ml"))]
        {
            Ok(PyPretrainedModel {
                name: "VQE Feature Extractor".to_string(),
                description: format!("Pretrained VQE model for {n_qubits} qubits"),
            })
        }
    }

    /// Get QAOA classifier model
    #[staticmethod]
    fn qaoa_classifier(n_qubits: usize, n_layers: usize) -> PyResult<PyPretrainedModel> {
        #[cfg(feature = "ml")]
        {
            use quantrs2_ml::transfer::QuantumModelZoo;

            let model = QuantumModelZoo::qaoa_classifier(n_qubits, n_layers)
                .map_err(|e| PyValueError::new_err(format!("Failed to create model: {e}")))?;

            Ok(PyPretrainedModel {
                inner: Some(model),
                name: "QAOA Classifier".to_string(),
                description: format!(
                    "Pretrained QAOA model with {n_qubits} qubits and {n_layers} layers"
                ),
            })
        }

        #[cfg(not(feature = "ml"))]
        {
            Ok(PyPretrainedModel {
                name: "QAOA Classifier".to_string(),
                description: format!(
                    "Pretrained QAOA model with {n_qubits} qubits and {n_layers} layers"
                ),
            })
        }
    }

    /// Get quantum autoencoder model
    #[staticmethod]
    fn quantum_autoencoder(n_qubits: usize, latent_dim: usize) -> PyResult<PyPretrainedModel> {
        #[cfg(feature = "ml")]
        {
            use quantrs2_ml::transfer::QuantumModelZoo;

            let model = QuantumModelZoo::quantum_autoencoder(n_qubits, latent_dim)
                .map_err(|e| PyValueError::new_err(format!("Failed to create model: {e}")))?;

            Ok(PyPretrainedModel {
                inner: Some(model),
                name: "Quantum Autoencoder".to_string(),
                description: format!(
                    "Pretrained autoencoder with {n_qubits} qubits and {latent_dim} latent dimensions"
                ),
            })
        }

        #[cfg(not(feature = "ml"))]
        {
            Ok(PyPretrainedModel {
                name: "Quantum Autoencoder".to_string(),
                description: format!(
                    "Pretrained autoencoder with {n_qubits} qubits and {latent_dim} latent dimensions"
                ),
            })
        }
    }

    /// List available models
    #[staticmethod]
    fn list_models() -> Vec<String> {
        vec![
            "vqe_feature_extractor".to_string(),
            "qaoa_classifier".to_string(),
            "quantum_autoencoder".to_string(),
        ]
    }
}

/// Register the ML transfer learning module
pub fn register_ml_transfer_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent_module.py(), "transfer")?;

    m.add_class::<PyTransferStrategy>()?;
    m.add_class::<PyTransferConfig>()?;
    m.add_class::<PyPretrainedModel>()?;
    m.add_class::<PyQuantumTransferLearning>()?;
    m.add_class::<PyQuantumModelZoo>()?;

    parent_module.add_submodule(&m)?;
    Ok(())
}
