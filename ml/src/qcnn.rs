//! Quantum Convolutional Neural Networks (QCNN)
//!
//! This module implements quantum convolutional neural networks for
//! quantum data processing and feature extraction.

use crate::error::MLError;
use num_complex::Complex64 as Complex;
use quantrs2_circuit::prelude::*;
use std::f64::consts::PI;

// Simple matrix types for QCNN
type DMatrix = Vec<Vec<f64>>;
type DVector<T> = Vec<T>;

/// Quantum convolutional filter
#[derive(Debug, Clone)]
pub struct QuantumConvFilter {
    /// Number of qubits in the filter
    pub num_qubits: usize,
    /// Stride of the convolution
    pub stride: usize,
    /// Variational parameters
    pub params: Vec<f64>,
}

impl QuantumConvFilter {
    /// Create a new quantum convolutional filter
    pub fn new(num_qubits: usize, stride: usize) -> Self {
        // Parameters for rotation gates
        let num_params = num_qubits * 3; // RX, RY, RZ per qubit
        let params = vec![0.1; num_params];

        Self {
            num_qubits,
            stride,
            params,
        }
    }

    /// Apply the filter to a subset of qubits
    pub fn apply_filter<const N: usize>(
        &self,
        circuit: &mut Circuit<N>,
        start_qubit: usize,
    ) -> Result<(), MLError> {
        let end_qubit = (start_qubit + self.num_qubits).min(N);

        // Apply parameterized rotations
        let mut param_idx = 0;
        for i in start_qubit..end_qubit {
            if param_idx < self.params.len() {
                circuit.rx(i, self.params[param_idx])?;
                param_idx += 1;
            }
            if param_idx < self.params.len() {
                circuit.ry(i, self.params[param_idx])?;
                param_idx += 1;
            }
            if param_idx < self.params.len() {
                circuit.rz(i, self.params[param_idx])?;
                param_idx += 1;
            }
        }

        // Apply entangling gates
        for i in start_qubit..(end_qubit - 1) {
            circuit.cnot(i, i + 1)?;
        }

        Ok(())
    }
}

/// Quantum pooling layer
#[derive(Debug, Clone)]
pub struct QuantumPooling {
    /// Pooling size (number of qubits to pool)
    pub pool_size: usize,
    /// Pooling type
    pub pool_type: PoolingType,
}

#[derive(Debug, Clone, Copy)]
pub enum PoolingType {
    /// Trace out qubits (dimensionality reduction)
    TraceOut,
    /// Measure and reset qubits
    MeasureReset,
    /// Quantum pooling
    Quantum,
}

impl QuantumPooling {
    /// Create a new quantum pooling layer
    pub fn new(pool_size: usize, pool_type: PoolingType) -> Self {
        Self {
            pool_size,
            pool_type,
        }
    }

    /// Apply pooling to reduce the number of active qubits
    pub fn apply_pooling<const N: usize>(
        &self,
        circuit: &mut Circuit<N>,
        active_qubits: &mut Vec<usize>,
    ) -> Result<(), MLError> {
        match self.pool_type {
            PoolingType::TraceOut => {
                // Simply remove qubits from active set
                let new_size = active_qubits.len() / self.pool_size;
                active_qubits.truncate(new_size);
            }
            PoolingType::MeasureReset => {
                // Measure and reset every nth qubit
                let mut new_active = Vec::new();
                for (i, &qubit) in active_qubits.iter().enumerate() {
                    if i % self.pool_size == 0 {
                        new_active.push(qubit);
                    } else {
                        // In a real implementation, we'd measure and reset
                        // For now, we just exclude from active set
                    }
                }
                *active_qubits = new_active;
            }
            PoolingType::Quantum => {
                // Quantum pooling using unitary operations
                let pool_size = self.pool_size;
                let new_size = active_qubits.len() / pool_size;

                // Apply quantum pooling gates (simplified)
                for i in 0..new_size {
                    let start_idx = i * pool_size;
                    let end_idx = (start_idx + pool_size).min(active_qubits.len());

                    if end_idx > start_idx + 1 {
                        // Apply entangling gates between qubits in pool
                        for j in start_idx..end_idx - 1 {
                            circuit.cnot(active_qubits[j], active_qubits[j + 1]);
                        }
                    }
                }

                // Keep only the first qubit from each pool
                active_qubits.truncate(new_size);
            }
        }
        Ok(())
    }
}

/// Quantum Convolutional Neural Network
pub struct QCNN {
    /// Number of qubits
    pub num_qubits: usize,
    /// Convolutional layers
    pub conv_layers: Vec<(QuantumConvFilter, QuantumPooling)>,
    /// Final fully connected layer parameters
    pub fc_params: Vec<f64>,
}

impl QCNN {
    /// Create a new QCNN
    pub fn new(
        num_qubits: usize,
        conv_filters: Vec<(usize, usize)>, // (filter_size, stride)
        pool_sizes: Vec<usize>,
        fc_params: usize,
    ) -> Result<Self, MLError> {
        if conv_filters.len() != pool_sizes.len() {
            return Err(MLError::ModelCreationError(
                "Number of conv filters must match number of pooling layers".to_string(),
            ));
        }

        let mut conv_layers = Vec::new();
        for ((filter_size, stride), pool_size) in conv_filters.into_iter().zip(pool_sizes) {
            let filter = QuantumConvFilter::new(filter_size, stride);
            let pooling = QuantumPooling::new(pool_size, PoolingType::TraceOut);
            conv_layers.push((filter, pooling));
        }

        let fc_params = vec![0.1; fc_params];

        Ok(Self {
            num_qubits,
            conv_layers,
            fc_params,
        })
    }

    /// Forward pass through the QCNN
    pub fn forward(&self, input_state: &DVector<Complex>) -> Result<DVector<Complex>, MLError> {
        // For simulation, we'll use a fixed circuit size
        const MAX_QUBITS: usize = 20;

        if self.num_qubits > MAX_QUBITS {
            return Err(MLError::InvalidParameter(format!(
                "QCNN supports up to {} qubits",
                MAX_QUBITS
            )));
        }

        let mut circuit = Circuit::<MAX_QUBITS>::new();
        let mut active_qubits: Vec<usize> = (0..self.num_qubits).collect();

        // Initialize with input state (simplified)
        // In practice, we'd use amplitude encoding

        // Apply convolutional and pooling layers
        for (conv_filter, pooling) in &self.conv_layers {
            // Apply convolution with sliding window
            let mut pos = 0;
            while pos + conv_filter.num_qubits <= active_qubits.len() {
                let start_qubit = active_qubits[pos];
                conv_filter.apply_filter(&mut circuit, start_qubit)?;
                pos += conv_filter.stride;
            }

            // Apply pooling
            pooling.apply_pooling(&mut circuit, &mut active_qubits)?;
        }

        // Apply fully connected layer to remaining active qubits
        for (i, &qubit) in active_qubits.iter().enumerate() {
            if i < self.fc_params.len() {
                circuit.ry(qubit, self.fc_params[i])?;
            }
        }

        // For now, return a dummy output state
        // In a real implementation, we would simulate the circuit
        let output_size = 1 << active_qubits.len();
        let mut output = vec![Complex::new(0.0, 0.0); output_size];

        // Simple normalization
        let norm = 1.0 / (output_size as f64).sqrt();
        for i in 0..output_size {
            output[i] = Complex::new(norm, 0.0);
        }

        Ok(output)
    }

    /// Get all trainable parameters
    pub fn get_parameters(&self) -> Vec<f64> {
        let mut params = Vec::new();

        for (conv_filter, _) in &self.conv_layers {
            params.extend(&conv_filter.params);
        }
        params.extend(&self.fc_params);

        params
    }

    /// Set parameters from a flat vector
    pub fn set_parameters(&mut self, params: &[f64]) -> Result<(), MLError> {
        let mut idx = 0;

        for (conv_filter, _) in &mut self.conv_layers {
            let filter_params = conv_filter.params.len();
            if idx + filter_params > params.len() {
                return Err(MLError::InvalidParameter(
                    "Not enough parameters provided".to_string(),
                ));
            }
            conv_filter
                .params
                .copy_from_slice(&params[idx..idx + filter_params]);
            idx += filter_params;
        }

        let fc_params_len = self.fc_params.len();
        if idx + fc_params_len > params.len() {
            return Err(MLError::InvalidParameter(
                "Not enough parameters for FC layer".to_string(),
            ));
        }
        self.fc_params
            .copy_from_slice(&params[idx..idx + fc_params_len]);

        Ok(())
    }

    /// Compute gradients using parameter shift rule
    pub fn compute_gradients(
        &mut self,
        input_state: &DVector<Complex>,
        target: &DVector<Complex>,
        loss_fn: impl Fn(&DVector<Complex>, &DVector<Complex>) -> f64,
    ) -> Result<Vec<f64>, MLError> {
        let params = self.get_parameters();
        let mut gradients = vec![0.0; params.len()];
        let shift = PI / 2.0;

        for i in 0..params.len() {
            // Positive shift
            let mut params_plus = params.clone();
            params_plus[i] += shift;
            self.set_parameters(&params_plus)?;
            let output_plus = self.forward(input_state)?;
            let loss_plus = loss_fn(&output_plus, target);

            // Negative shift
            let mut params_minus = params.clone();
            params_minus[i] -= shift;
            self.set_parameters(&params_minus)?;
            let output_minus = self.forward(input_state)?;
            let loss_minus = loss_fn(&output_minus, target);

            // Parameter shift gradient
            gradients[i] = (loss_plus - loss_minus) / (2.0 * shift);
        }

        // Restore original parameters
        self.set_parameters(&params)?;

        Ok(gradients)
    }
}

/// Quantum image encoding for QCNN
pub struct QuantumImageEncoder {
    /// Image dimensions
    pub width: usize,
    pub height: usize,
    /// Number of qubits for encoding
    pub num_qubits: usize,
}

impl QuantumImageEncoder {
    /// Create a new quantum image encoder
    pub fn new(width: usize, height: usize, num_qubits: usize) -> Self {
        Self {
            width,
            height,
            num_qubits,
        }
    }

    /// Encode a classical image into quantum state
    pub fn encode(&self, image: &DMatrix) -> Result<DVector<Complex>, MLError> {
        if image.len() != self.height || image[0].len() != self.width {
            return Err(MLError::InvalidParameter(
                "Image dimensions don't match encoder settings".to_string(),
            ));
        }

        // Flatten and normalize image
        let pixels: Vec<f64> = image.iter().flat_map(|row| row.iter()).copied().collect();
        let norm = pixels.iter().map(|x| x * x).sum::<f64>().sqrt();

        // Create quantum state with amplitude encoding
        let state_size = 1 << self.num_qubits;
        let mut state = vec![Complex::new(0.0, 0.0); state_size];

        for (i, &pixel) in pixels.iter().enumerate() {
            if i < state_size {
                state[i] = Complex::new(pixel / norm, 0.0);
            }
        }

        Ok(state)
    }

    /// Decode quantum state back to classical image representation
    pub fn decode(&self, state: &DVector<Complex>) -> DMatrix {
        let mut image = vec![vec![0.0; self.width]; self.height];
        let mut idx = 0;

        for i in 0..self.height {
            for j in 0..self.width {
                if idx < state.len() {
                    image[i][j] = state[idx].norm();
                    idx += 1;
                }
            }
        }

        image
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qcnn_creation() {
        let qcnn = QCNN::new(
            8,                    // 8 qubits
            vec![(4, 2), (2, 1)], // Two conv layers
            vec![2, 2],           // Two pooling layers
            4,                    // FC layer params
        )
        .unwrap();

        assert_eq!(qcnn.num_qubits, 8);
        assert_eq!(qcnn.conv_layers.len(), 2);
    }

    #[test]
    fn test_quantum_filter() {
        let filter = QuantumConvFilter::new(3, 1);
        assert_eq!(filter.num_qubits, 3);
        assert_eq!(filter.params.len(), 9); // 3 qubits * 3 gates
    }

    #[test]
    fn test_filter_application() {
        let filter = QuantumConvFilter::new(3, 1);
        let mut circuit = Circuit::<8>::new();

        // Apply filter starting at qubit 0
        filter.apply_filter(&mut circuit, 0).unwrap();

        // Should have applied gates
        assert!(circuit.num_gates() > 0);
    }

    #[test]
    fn test_pooling_trace_out() {
        let pooling = QuantumPooling::new(2, PoolingType::TraceOut);
        let mut circuit = Circuit::<8>::new();
        let mut active_qubits = vec![0, 1, 2, 3, 4, 5, 6, 7];

        pooling
            .apply_pooling(&mut circuit, &mut active_qubits)
            .unwrap();

        // Should reduce active qubits by pool_size
        assert_eq!(active_qubits.len(), 4);
    }

    #[test]
    fn test_pooling_measure_reset() {
        let pooling = QuantumPooling::new(2, PoolingType::MeasureReset);
        let mut circuit = Circuit::<8>::new();
        let mut active_qubits = vec![0, 1, 2, 3, 4, 5, 6, 7];

        pooling
            .apply_pooling(&mut circuit, &mut active_qubits)
            .unwrap();

        // Should keep every 2nd qubit
        assert_eq!(active_qubits.len(), 4);
        assert_eq!(active_qubits, vec![0, 2, 4, 6]);
    }

    #[test]
    fn test_image_encoding() {
        let encoder = QuantumImageEncoder::new(2, 2, 2);
        let image = vec![vec![0.5, 0.5], vec![0.5, 0.5]];

        let encoded = encoder.encode(&image).unwrap();
        assert_eq!(encoded.len(), 4); // 2^2 = 4

        // Check normalization
        let norm: f64 = encoded.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_image_decode() {
        let encoder = QuantumImageEncoder::new(2, 2, 2);
        let state = vec![
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
        ];

        let decoded = encoder.decode(&state);
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].len(), 2);
    }

    #[test]
    fn test_qcnn_forward() {
        let qcnn = QCNN::new(
            4,            // 4 qubits
            vec![(2, 1)], // One conv layer
            vec![2],      // One pooling layer
            2,            // FC layer params
        )
        .unwrap();

        let input_state = vec![Complex::new(1.0, 0.0); 16]; // 2^4 = 16
        let output = qcnn.forward(&input_state).unwrap();

        // Output should be for reduced qubits after pooling
        assert!(output.len() > 0);
    }

    #[test]
    fn test_parameter_management() {
        let mut qcnn = QCNN::new(
            4,            // 4 qubits
            vec![(2, 1)], // One conv layer
            vec![2],      // One pooling layer
            2,            // FC layer params
        )
        .unwrap();

        let params = qcnn.get_parameters();
        let num_params = params.len();

        // Modify parameters
        let new_params: Vec<f64> = (0..num_params).map(|i| i as f64 * 0.1).collect();
        qcnn.set_parameters(&new_params).unwrap();

        let retrieved_params = qcnn.get_parameters();
        assert_eq!(retrieved_params, new_params);
    }

    #[test]
    fn test_gradient_computation() {
        let mut qcnn = QCNN::new(
            4,            // 4 qubits
            vec![(2, 1)], // One conv layer
            vec![2],      // One pooling layer
            2,            // FC layer params
        )
        .unwrap();

        let input_state = vec![Complex::new(0.5, 0.0); 16];
        let target_state = vec![Complex::new(0.707, 0.0); 2];

        // Simple MSE loss
        let loss_fn = |output: &DVector<Complex>, target: &DVector<Complex>| -> f64 {
            output
                .iter()
                .zip(target.iter())
                .map(|(o, t)| (o - t).norm_sqr())
                .sum::<f64>()
        };

        let gradients = qcnn
            .compute_gradients(&input_state, &target_state, loss_fn)
            .unwrap();

        // Should have gradients for all parameters
        assert_eq!(gradients.len(), qcnn.get_parameters().len());
    }

    #[test]
    fn test_invalid_layer_configuration() {
        // Mismatched conv and pool layers
        let result = QCNN::new(
            8,
            vec![(4, 2), (2, 1)], // Two conv layers
            vec![2],              // Only one pooling layer
            4,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_stride_behavior() {
        let filter = QuantumConvFilter::new(2, 2); // Filter size 2, stride 2
        assert_eq!(filter.stride, 2);

        let mut circuit = Circuit::<8>::new();

        // Apply with stride - should skip positions
        filter.apply_filter(&mut circuit, 0).unwrap();
        filter.apply_filter(&mut circuit, 2).unwrap(); // Next position based on stride
    }

    #[test]
    fn test_large_image_encoding() {
        let encoder = QuantumImageEncoder::new(4, 4, 4); // 4x4 image, 4 qubits
        let image = vec![vec![0.25; 4]; 4];

        let encoded = encoder.encode(&image).unwrap();
        assert_eq!(encoded.len(), 16); // 2^4 = 16

        // Verify partial encoding (16 pixels into 16 amplitudes)
        let decoded = encoder.decode(&encoded);
        assert_eq!(decoded.len(), 4);
        assert_eq!(decoded[0].len(), 4);
    }
}
