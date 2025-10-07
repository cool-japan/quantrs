//! SciRS2 integration layer for quantum machine learning
//!
//! This module provides integration with the SciRS2 scientific computing framework,
//! enabling quantum ML models to leverage SciRS2's optimized tensor operations,
//! distributed training capabilities, and serialization formats.

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array, Array1, Array2, Array3, ArrayD, ArrayViewD, Dimension, IxDyn};
use std::collections::HashMap;

/// Trait for tensor operations compatible with SciRS2
pub trait SciRS2Tensor {
    /// Get tensor shape
    fn shape(&self) -> &[usize];

    /// Get tensor data as ArrayViewD
    fn view(&self) -> ArrayViewD<f64>;

    /// Convert to SciRS2 format (placeholder)
    fn to_scirs2(&self) -> Result<SciRS2Array>;

    /// Perform tensor operations using SciRS2 backend
    fn matmul(&self, other: &dyn SciRS2Tensor) -> Result<SciRS2Array>;

    /// Element-wise operations
    fn add(&self, other: &dyn SciRS2Tensor) -> Result<SciRS2Array>;
    fn mul(&self, other: &dyn SciRS2Tensor) -> Result<SciRS2Array>;
    fn sub(&self, other: &dyn SciRS2Tensor) -> Result<SciRS2Array>;

    /// Reduction operations
    fn sum(&self, axis: Option<usize>) -> Result<SciRS2Array>;
    fn mean(&self, axis: Option<usize>) -> Result<SciRS2Array>;
    fn max(&self, axis: Option<usize>) -> Result<SciRS2Array>;
    fn min(&self, axis: Option<usize>) -> Result<SciRS2Array>;
}

/// SciRS2 array wrapper for quantum ML operations
pub struct SciRS2Array {
    /// Array data
    pub data: ArrayD<f64>,
    /// Whether gradients are required
    pub requires_grad: bool,
    /// Gradient accumulator
    pub grad: Option<ArrayD<f64>>,
    /// Operation history for backpropagation
    pub grad_fn: Option<Box<dyn GradFunction>>,
}

impl std::fmt::Debug for SciRS2Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SciRS2Array")
            .field("data", &self.data)
            .field("requires_grad", &self.requires_grad)
            .field("grad", &self.grad)
            .field("grad_fn", &"<gradient_function>")
            .finish()
    }
}

impl Clone for SciRS2Array {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            requires_grad: self.requires_grad,
            grad: self.grad.clone(),
            grad_fn: None, // Cannot clone trait objects
        }
    }
}

impl SciRS2Array {
    /// Create a new SciRS2Array
    pub fn new(data: ArrayD<f64>, requires_grad: bool) -> Self {
        let grad = if requires_grad {
            Some(ArrayD::zeros(data.raw_dim()))
        } else {
            None
        };
        Self {
            data,
            requires_grad,
            grad,
            grad_fn: None,
        }
    }

    /// Create from ndarray
    pub fn from_array<D: Dimension>(arr: Array<f64, D>) -> Self {
        let data = arr.into_dyn();
        Self::new(data, false)
    }

    /// Create with gradient tracking
    pub fn with_grad<D: Dimension>(arr: Array<f64, D>) -> Self {
        let data = arr.into_dyn();
        Self::new(data, true)
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        if let Some(ref mut grad) = self.grad {
            grad.fill(0.0);
        }
    }

    /// Backward pass
    pub fn backward(&mut self) -> Result<()> {
        // Extract grad_fn to avoid borrow conflicts
        if let Some(grad_fn) = self.grad_fn.take() {
            grad_fn.backward(self)?;
            self.grad_fn = Some(grad_fn);
        }
        Ok(())
    }

    /// Matrix multiplication using SciRS2 backend
    pub fn matmul(&self, other: &SciRS2Array) -> Result<SciRS2Array> {
        // Placeholder - would use SciRS2 linalg operations
        let result_data = if self.data.ndim() == 2 && other.data.ndim() == 2 {
            let self_2d = self
                .data
                .view()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|e| MLError::ComputationError(format!("Shape error: {}", e)))?;
            let other_2d = other
                .data
                .view()
                .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                .map_err(|e| MLError::ComputationError(format!("Shape error: {}", e)))?;
            self_2d.dot(&other_2d).into_dyn()
        } else {
            return Err(MLError::InvalidConfiguration(
                "Matrix multiplication requires 2D arrays".to_string(),
            ));
        };

        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = SciRS2Array::new(result_data, requires_grad);

        if requires_grad {
            result.grad_fn = Some(Box::new(MatmulGradFn {
                left_shape: self.data.raw_dim(),
                right_shape: other.data.raw_dim(),
            }));
        }

        Ok(result)
    }

    /// Element-wise addition
    pub fn add(&self, other: &SciRS2Array) -> Result<SciRS2Array> {
        let result_data = &self.data + &other.data;
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = SciRS2Array::new(result_data, requires_grad);

        if requires_grad {
            result.grad_fn = Some(Box::new(AddGradFn));
        }

        Ok(result)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &SciRS2Array) -> Result<SciRS2Array> {
        let result_data = &self.data * &other.data;
        let requires_grad = self.requires_grad || other.requires_grad;
        let mut result = SciRS2Array::new(result_data, requires_grad);

        if requires_grad {
            result.grad_fn = Some(Box::new(MulGradFn {
                left_data: self.data.clone(),
                right_data: other.data.clone(),
            }));
        }

        Ok(result)
    }

    /// Reduction sum
    pub fn sum(&self, axis: Option<usize>) -> Result<SciRS2Array> {
        let result_data = match axis {
            Some(ax) => self
                .data
                .sum_axis(scirs2_core::ndarray::Axis(ax))
                .into_dyn(),
            None => {
                let sum_val = self.data.sum();
                ArrayD::from_elem(IxDyn(&[]), sum_val)
            }
        };

        let mut result = SciRS2Array::new(result_data, self.requires_grad);

        if self.requires_grad {
            result.grad_fn = Some(Box::new(SumGradFn { axis }));
        }

        Ok(result)
    }
}

impl SciRS2Tensor for SciRS2Array {
    fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    fn view(&self) -> ArrayViewD<f64> {
        self.data.view()
    }

    fn to_scirs2(&self) -> Result<SciRS2Array> {
        Ok(self.clone())
    }

    fn matmul(&self, other: &dyn SciRS2Tensor) -> Result<SciRS2Array> {
        // Convert other to SciRS2Array for computation
        let other_array = other.to_scirs2()?;
        self.matmul(&other_array)
    }

    fn add(&self, other: &dyn SciRS2Tensor) -> Result<SciRS2Array> {
        let other_array = other.to_scirs2()?;
        self.add(&other_array)
    }

    fn mul(&self, other: &dyn SciRS2Tensor) -> Result<SciRS2Array> {
        let other_array = other.to_scirs2()?;
        self.mul(&other_array)
    }

    fn sub(&self, other: &dyn SciRS2Tensor) -> Result<SciRS2Array> {
        let result_data = &self.data - &other.to_scirs2()?.data;
        let requires_grad = self.requires_grad || other.to_scirs2()?.requires_grad;
        Ok(SciRS2Array::new(result_data, requires_grad))
    }

    fn sum(&self, axis: Option<usize>) -> Result<SciRS2Array> {
        self.sum(axis)
    }

    fn mean(&self, axis: Option<usize>) -> Result<SciRS2Array> {
        let result_data = match axis {
            Some(ax) => self
                .data
                .mean_axis(scirs2_core::ndarray::Axis(ax))
                .unwrap()
                .into_dyn(),
            None => {
                let mean_val = self.data.mean().unwrap();
                ArrayD::from_elem(IxDyn(&[]), mean_val)
            }
        };
        Ok(SciRS2Array::new(result_data, self.requires_grad))
    }

    fn max(&self, axis: Option<usize>) -> Result<SciRS2Array> {
        let result_data = match axis {
            Some(ax) => self
                .data
                .map_axis(scirs2_core::ndarray::Axis(ax), |view| {
                    *view
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                })
                .into_dyn(),
            None => {
                let max_val = *self
                    .data
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                ArrayD::from_elem(IxDyn(&[]), max_val)
            }
        };
        Ok(SciRS2Array::new(result_data, self.requires_grad))
    }

    fn min(&self, axis: Option<usize>) -> Result<SciRS2Array> {
        let result_data = match axis {
            Some(ax) => self
                .data
                .map_axis(scirs2_core::ndarray::Axis(ax), |view| {
                    *view
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                })
                .into_dyn(),
            None => {
                let min_val = *self
                    .data
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                ArrayD::from_elem(IxDyn(&[]), min_val)
            }
        };
        Ok(SciRS2Array::new(result_data, self.requires_grad))
    }
}

/// Trait for gradient functions
pub trait GradFunction: Send + Sync {
    fn backward(&self, output: &mut SciRS2Array) -> Result<()>;
}

/// Gradient function for matrix multiplication
#[derive(Debug)]
struct MatmulGradFn {
    left_shape: IxDyn,
    right_shape: IxDyn,
}

impl GradFunction for MatmulGradFn {
    fn backward(&self, _output: &mut SciRS2Array) -> Result<()> {
        // Placeholder - would compute gradients for matmul inputs
        Ok(())
    }
}

/// Gradient function for addition
#[derive(Debug)]
struct AddGradFn;

impl GradFunction for AddGradFn {
    fn backward(&self, _output: &mut SciRS2Array) -> Result<()> {
        // Gradient flows through unchanged for addition
        Ok(())
    }
}

/// Gradient function for multiplication
#[derive(Debug)]
struct MulGradFn {
    left_data: ArrayD<f64>,
    right_data: ArrayD<f64>,
}

impl GradFunction for MulGradFn {
    fn backward(&self, _output: &mut SciRS2Array) -> Result<()> {
        // Placeholder - would compute gradients for element-wise multiplication
        Ok(())
    }
}

/// Gradient function for sum reduction
#[derive(Debug)]
struct SumGradFn {
    axis: Option<usize>,
}

impl GradFunction for SumGradFn {
    fn backward(&self, _output: &mut SciRS2Array) -> Result<()> {
        // Placeholder - would broadcast gradients for sum reduction
        Ok(())
    }
}

/// SciRS2 optimization interface
pub struct SciRS2Optimizer {
    /// Optimizer type
    pub optimizer_type: String,
    /// Configuration parameters
    pub config: HashMap<String, f64>,
    /// Parameter state (for stateful optimizers like Adam)
    pub state: HashMap<String, ArrayD<f64>>,
}

impl SciRS2Optimizer {
    /// Create a new SciRS2 optimizer
    pub fn new(optimizer_type: impl Into<String>) -> Self {
        Self {
            optimizer_type: optimizer_type.into(),
            config: HashMap::new(),
            state: HashMap::new(),
        }
    }

    /// Set optimizer configuration
    pub fn with_config(mut self, key: impl Into<String>, value: f64) -> Self {
        self.config.insert(key.into(), value);
        self
    }

    /// Update parameters using computed gradients
    pub fn step(&mut self, params: &mut HashMap<String, SciRS2Array>) -> Result<()> {
        match self.optimizer_type.as_str() {
            "adam" => self.adam_step(params),
            "sgd" => self.sgd_step(params),
            "lbfgs" => self.lbfgs_step(params),
            _ => Err(MLError::InvalidConfiguration(format!(
                "Unknown optimizer type: {}",
                self.optimizer_type
            ))),
        }
    }

    /// Adam optimizer step
    fn adam_step(&mut self, params: &mut HashMap<String, SciRS2Array>) -> Result<()> {
        let learning_rate = self.config.get("learning_rate").unwrap_or(&0.001);
        let beta1 = self.config.get("beta1").unwrap_or(&0.9);
        let beta2 = self.config.get("beta2").unwrap_or(&0.999);
        let epsilon = self.config.get("epsilon").unwrap_or(&1e-8);

        for (name, param) in params.iter_mut() {
            if let Some(ref grad) = param.grad {
                // Initialize momentum and velocity if not present
                let m_key = format!("{}_m", name);
                let v_key = format!("{}_v", name);

                if !self.state.contains_key(&m_key) {
                    self.state
                        .insert(m_key.clone(), ArrayD::zeros(grad.raw_dim()));
                    self.state
                        .insert(v_key.clone(), ArrayD::zeros(grad.raw_dim()));
                }

                // Update first moment estimate
                {
                    let m = self.state.get_mut(&m_key).unwrap();
                    *m = *beta1 * &*m + (1.0 - *beta1) * grad;
                }

                // Update second moment estimate
                {
                    let v = self.state.get_mut(&v_key).unwrap();
                    *v = *beta2 * &*v + (1.0 - *beta2) * grad * grad;
                }

                // Get references for bias correction
                let m_hat = self.state.get(&m_key).unwrap().clone();
                let v_hat = self.state.get(&v_key).unwrap().clone();

                // Update parameters
                param.data =
                    &param.data - *learning_rate * &m_hat / (v_hat.mapv(|x| x.sqrt()) + *epsilon);
            }
        }

        Ok(())
    }

    /// SGD optimizer step
    fn sgd_step(&mut self, params: &mut HashMap<String, SciRS2Array>) -> Result<()> {
        let learning_rate = self.config.get("learning_rate").unwrap_or(&0.01);
        let momentum = self.config.get("momentum").unwrap_or(&0.0);

        for (name, param) in params.iter_mut() {
            if let Some(ref grad) = param.grad {
                if *momentum > 0.0 {
                    let v_key = format!("{}_v", name);
                    if !self.state.contains_key(&v_key) {
                        self.state
                            .insert(v_key.clone(), ArrayD::zeros(grad.raw_dim()));
                    }

                    let v = self.state.get_mut(&v_key).unwrap();
                    *v = *momentum * &*v + *learning_rate * grad;
                    param.data = &param.data - &*v;
                } else {
                    param.data = &param.data - *learning_rate * grad;
                }
            }
        }

        Ok(())
    }

    /// L-BFGS optimizer step (placeholder)
    fn lbfgs_step(&mut self, _params: &mut HashMap<String, SciRS2Array>) -> Result<()> {
        // Placeholder - would implement L-BFGS using SciRS2
        Ok(())
    }
}

/// SciRS2 distributed training support
pub struct SciRS2DistributedTrainer {
    /// World size (number of processes)
    pub world_size: usize,
    /// Local rank
    pub rank: usize,
    /// Backend for communication
    pub backend: String,
}

impl SciRS2DistributedTrainer {
    /// Create a new distributed trainer
    pub fn new(world_size: usize, rank: usize) -> Self {
        Self {
            world_size,
            rank,
            backend: "nccl".to_string(),
        }
    }

    /// All-reduce operation for gradient synchronization
    pub fn all_reduce(&self, tensor: &mut SciRS2Array) -> Result<()> {
        // Placeholder - would use SciRS2 distributed operations
        Ok(())
    }

    /// All-reduce scalar operation for metrics synchronization
    pub fn all_reduce_scalar(&self, value: f64) -> Result<f64> {
        // Placeholder - would use SciRS2 distributed operations
        // For now, just return the value unchanged (single process behavior)
        Ok(value)
    }

    /// Broadcast operation
    pub fn broadcast(&self, tensor: &mut SciRS2Array, root: usize) -> Result<()> {
        // Placeholder - would use SciRS2 distributed operations
        Ok(())
    }

    /// All-gather operation
    pub fn all_gather(&self, tensor: &SciRS2Array) -> Result<Vec<SciRS2Array>> {
        // Placeholder - would use SciRS2 distributed operations
        Ok(vec![tensor.clone(); self.world_size])
    }

    /// Wrap a model for distributed training
    pub fn wrap_model<T>(&self, model: T) -> Result<T> {
        // Placeholder - would wrap the model with distributed training capabilities
        // For now, just return the model unchanged
        Ok(model)
    }
}

/// SciRS2 model serialization interface
pub struct SciRS2Serializer;

impl SciRS2Serializer {
    /// Serialize model parameters to SciRS2 format
    pub fn save_model(params: &HashMap<String, SciRS2Array>, path: &str) -> Result<()> {
        // Placeholder - would use SciRS2 serialization
        Ok(())
    }

    /// Load model parameters from SciRS2 format
    pub fn load_model(path: &str) -> Result<HashMap<String, SciRS2Array>> {
        // Placeholder - would use SciRS2 deserialization
        Ok(HashMap::new())
    }

    /// Save checkpoint with optimizer state
    pub fn save_checkpoint(
        params: &HashMap<String, SciRS2Array>,
        optimizer: &SciRS2Optimizer,
        epoch: usize,
        path: &str,
    ) -> Result<()> {
        // Placeholder - would use SciRS2 checkpoint format
        Ok(())
    }

    /// Load checkpoint with optimizer state
    pub fn load_checkpoint(
        path: &str,
    ) -> Result<(HashMap<String, SciRS2Array>, SciRS2Optimizer, usize)> {
        // Placeholder - would use SciRS2 checkpoint format
        Ok((HashMap::new(), SciRS2Optimizer::new("adam"), 0))
    }
}

/// SciRS2 Dataset wrapper for quantum ML
pub struct SciRS2Dataset {
    /// Training data
    pub data: ArrayD<f64>,
    /// Labels
    pub labels: ArrayD<f64>,
    /// Dataset size
    pub size: usize,
}

impl SciRS2Dataset {
    /// Create a new dataset
    pub fn new(data: ArrayD<f64>, labels: ArrayD<f64>) -> Result<Self> {
        let size = data.shape()[0];
        if labels.shape()[0] != size {
            return Err(MLError::InvalidConfiguration(
                "Data and labels must have same number of samples".to_string(),
            ));
        }

        Ok(Self { data, labels, size })
    }
}

/// SciRS2 DataLoader for batch processing
pub struct SciRS2DataLoader {
    /// Dataset reference
    pub dataset: SciRS2Dataset,
    /// Batch size
    pub batch_size: usize,
    /// Current index
    pub current_index: usize,
}

impl SciRS2DataLoader {
    /// Create a new data loader
    pub fn new(dataset: SciRS2Dataset, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
            current_index: 0,
        }
    }

    /// Iterator-like enumeration support
    pub fn enumerate(&mut self) -> DataLoaderIterator {
        DataLoaderIterator {
            loader: self,
            batch_idx: 0,
        }
    }
}

/// Iterator for DataLoader
pub struct DataLoaderIterator<'a> {
    loader: &'a mut SciRS2DataLoader,
    batch_idx: usize,
}

impl<'a> Iterator for DataLoaderIterator<'a> {
    type Item = (usize, (SciRS2Array, SciRS2Array));

    fn next(&mut self) -> Option<Self::Item> {
        if self.loader.current_index >= self.loader.dataset.size {
            return None;
        }

        let start = self.loader.current_index;
        let end = (start + self.loader.batch_size).min(self.loader.dataset.size);

        // Extract batch data and labels
        let batch_data = self
            .loader
            .dataset
            .data
            .slice(scirs2_core::ndarray::s![start..end, ..])
            .to_owned();
        let batch_labels = self
            .loader
            .dataset
            .labels
            .slice(scirs2_core::ndarray::s![start..end, ..])
            .to_owned();

        let data_array = SciRS2Array::from_array(batch_data);
        let label_array = SciRS2Array::from_array(batch_labels);

        self.loader.current_index = end;
        let batch_idx = self.batch_idx;
        self.batch_idx += 1;

        Some((batch_idx, (data_array, label_array)))
    }
}

/// SciRS2 Device enumeration
#[derive(Debug, Clone, Copy)]
pub enum SciRS2Device {
    CPU,
    GPU,
    Quantum,
}

/// Additional SciRS2Array methods for compatibility
impl SciRS2Array {
    /// Create array with specified device
    pub fn randn(shape: Vec<usize>, device: SciRS2Device) -> Result<Self> {
        use scirs2_core::random::prelude::*;
        let total_size = shape.iter().product();
        let mut rng = thread_rng();
        let data: Vec<f64> = (0..total_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| MLError::ComputationError(format!("Shape error: {}", e)))?;
        Ok(Self::new(array, false))
    }

    /// Create ones_like array
    pub fn ones_like(&self) -> Result<Self> {
        let ones = ArrayD::ones(self.data.raw_dim());
        Ok(Self::new(ones, false))
    }

    /// Create random integers
    pub fn randint(low: i32, high: i32, shape: Vec<usize>, device: SciRS2Device) -> Result<Self> {
        use scirs2_core::random::prelude::*;
        let total_size = shape.iter().product();
        let mut rng = thread_rng();
        let data: Vec<f64> = (0..total_size)
            .map(|_| rng.gen_range(low..high) as f64)
            .collect();
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| MLError::ComputationError(format!("Shape error: {}", e)))?;
        Ok(Self::new(array, false))
    }

    /// Create quantum observable
    pub fn quantum_observable(name: &str, num_qubits: usize) -> Result<Self> {
        match name {
            "pauli_z_all" => {
                let size = 1 << num_qubits;
                let mut data = ArrayD::zeros(IxDyn(&[size, size]));
                for i in 0..size {
                    let parity = i.count_ones() % 2;
                    data[[i, i]] = if parity == 0 { 1.0 } else { -1.0 };
                }
                Ok(Self::new(data, false))
            }
            _ => Err(MLError::InvalidConfiguration(format!(
                "Unknown observable: {}",
                name
            ))),
        }
    }
}

/// Integration helper functions
pub mod integration {
    use super::*;

    /// Convert ndarray to SciRS2Array
    pub fn from_ndarray<D: Dimension>(arr: Array<f64, D>) -> SciRS2Array {
        SciRS2Array::from_array(arr)
    }

    /// Convert SciRS2Array to ndarray
    pub fn to_ndarray<D: Dimension>(arr: &SciRS2Array) -> Result<Array<f64, D>> {
        arr.data
            .view()
            .into_dimensionality::<D>()
            .map(|v| v.to_owned())
            .map_err(|e| MLError::ComputationError(format!("Dimension error: {}", e)))
    }

    /// Create SciRS2 optimizer from configuration
    pub fn create_optimizer(optimizer_type: &str, config: HashMap<String, f64>) -> SciRS2Optimizer {
        let mut optimizer = SciRS2Optimizer::new(optimizer_type);
        for (key, value) in config {
            optimizer = optimizer.with_config(key, value);
        }
        optimizer
    }

    /// Setup distributed training
    pub fn setup_distributed(world_size: usize, rank: usize) -> SciRS2DistributedTrainer {
        SciRS2DistributedTrainer::new(world_size, rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_scirs2_array_creation() {
        let arr = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let scirs2_arr = SciRS2Array::from_array(arr);

        assert_eq!(scirs2_arr.data.shape(), &[2, 2]);
        assert!(!scirs2_arr.requires_grad);
    }

    #[test]
    fn test_scirs2_array_with_grad() {
        let arr = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let scirs2_arr = SciRS2Array::with_grad(arr);

        assert!(scirs2_arr.requires_grad);
        assert!(scirs2_arr.grad.is_some());
    }

    #[test]
    fn test_scirs2_matmul() {
        let arr1 = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let arr2 = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let scirs2_arr1 = SciRS2Array::from_array(arr1);
        let scirs2_arr2 = SciRS2Array::from_array(arr2);

        let result = scirs2_arr1.matmul(&scirs2_arr2).unwrap();
        assert_eq!(result.data.shape(), &[2, 2]);
    }

    #[test]
    fn test_scirs2_optimizer() {
        let mut optimizer = SciRS2Optimizer::new("adam")
            .with_config("learning_rate", 0.001)
            .with_config("beta1", 0.9);

        let mut params = HashMap::new();
        let param_arr = SciRS2Array::with_grad(Array1::from_vec(vec![1.0, 2.0, 3.0]));
        params.insert("weight".to_string(), param_arr);

        let result = optimizer.step(&mut params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_integration_helpers() {
        let arr = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let scirs2_arr = integration::from_ndarray(arr.clone());

        let back_to_ndarray: Array2<f64> = integration::to_ndarray(&scirs2_arr).unwrap();
        assert_eq!(arr, back_to_ndarray);
    }
}
