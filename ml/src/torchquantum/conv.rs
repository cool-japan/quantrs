//! Quantum convolutional layers for quantum machine learning
//!
//! This module provides quantum analogues of classical convolutional neural network layers,
//! enabling spatial feature extraction on quantum states through parameterized local unitaries.
//!
//! # Layers
//!
//! - **QConv1D**: 1D quantum convolution with sliding window kernels
//! - **QConv2D**: 2D quantum convolution for grid-arranged qubits
//!
//! # Features
//!
//! - Parameterized unitaries applied to local qubit neighborhoods
//! - Configurable kernel size and stride
//! - Compatible with TorchQuantum training framework
//!
//! # Example
//!
//! ```ignore
//! use quantrs2_ml::torchquantum::conv::QConv1D;
//!
//! // Create 1D conv layer: 8 qubits, kernel size 3, stride 1
//! let conv = QConv1D::new(8, 3, 1)?;
//! println!("Parameters: {}", conv.n_parameters());
//! ```

use crate::error::{MLError, Result as MLResult};

/// Quantum convolutional layer operating on 1D wire sequences
///
/// Applies a parameterized unitary to sliding windows of qubits,
/// similar to classical convolutional neural networks but operating
/// on quantum states.
#[derive(Debug, Clone)]
pub struct QConv1D {
    /// Number of input wires
    n_wires: usize,
    /// Kernel size (number of qubits per convolution window)
    kernel_size: usize,
    /// Stride (step size for sliding window)
    stride: usize,
    /// Number of parameters per kernel application
    n_params_per_kernel: usize,
    /// Total number of trainable parameters
    n_parameters: usize,
    /// Layer name for debugging
    name: String,
}

impl QConv1D {
    /// Create a new 1D quantum convolutional layer
    ///
    /// # Arguments
    /// * `n_wires` - Number of input qubits
    /// * `kernel_size` - Size of the convolutional kernel (number of qubits)
    /// * `stride` - Step size for the sliding window
    /// * `n_params_per_kernel` - Number of rotation parameters per kernel
    ///
    /// # Example
    /// ```ignore
    /// // 8-qubit input, 3-qubit kernel, stride 1, 6 parameters per kernel
    /// let conv = QConv1D::new(8, 3, 1, 6)?;
    /// // Will apply kernel at positions: (0,1,2), (1,2,3), ..., (5,6,7)
    /// // Total: 6 positions × 6 params = 36 parameters
    /// ```
    pub fn new(
        n_wires: usize,
        kernel_size: usize,
        stride: usize,
        n_params_per_kernel: usize,
    ) -> MLResult<Self> {
        if kernel_size > n_wires {
            return Err(MLError::InvalidConfiguration(format!(
                "Kernel size {} exceeds number of wires {}",
                kernel_size, n_wires
            )));
        }

        if stride == 0 {
            return Err(MLError::InvalidConfiguration(
                "Stride must be greater than 0".to_string(),
            ));
        }

        // Calculate number of kernel applications
        let n_kernels = (n_wires - kernel_size) / stride + 1;
        let n_parameters = n_kernels * n_params_per_kernel;

        Ok(Self {
            n_wires,
            kernel_size,
            stride,
            n_params_per_kernel,
            n_parameters,
            name: format!("QConv1D(kernel={}, stride={})", kernel_size, stride),
        })
    }

    /// Get the positions where kernels will be applied
    pub fn kernel_positions(&self) -> Vec<usize> {
        let mut positions = Vec::new();
        let mut pos = 0;

        while pos + self.kernel_size <= self.n_wires {
            positions.push(pos);
            pos += self.stride;
        }

        positions
    }

    /// Get the qubit indices for a specific kernel position
    pub fn kernel_qubits(&self, position: usize) -> Vec<usize> {
        (position..position + self.kernel_size).collect()
    }
}

impl QConv1D {
    /// Get the total number of trainable parameters
    pub fn n_parameters(&self) -> usize {
        self.n_parameters
    }
}

/// Quantum convolutional layer operating on 2D qubit grids
///
/// Extends QConv1D to operate on 2D arrangements of qubits,
/// applying kernels to rectangular patches of the qubit lattice.
#[derive(Debug, Clone)]
pub struct QConv2D {
    /// Grid width (number of qubits in x direction)
    width: usize,
    /// Grid height (number of qubits in y direction)
    height: usize,
    /// Kernel width
    kernel_width: usize,
    /// Kernel height
    kernel_height: usize,
    /// Stride in x direction
    stride_x: usize,
    /// Stride in y direction
    stride_y: usize,
    /// Number of parameters per kernel application
    n_params_per_kernel: usize,
    /// Total number of trainable parameters
    n_parameters: usize,
    /// Layer name for debugging
    name: String,
}

impl QConv2D {
    /// Create a new 2D quantum convolutional layer
    ///
    /// # Arguments
    /// * `width` - Grid width (qubits in x direction)
    /// * `height` - Grid height (qubits in y direction)
    /// * `kernel_width` - Kernel width
    /// * `kernel_height` - Kernel height
    /// * `stride_x` - Stride in x direction
    /// * `stride_y` - Stride in y direction
    /// * `n_params_per_kernel` - Number of rotation parameters per kernel
    ///
    /// # Example
    /// ```ignore
    /// // 4×4 qubit grid, 2×2 kernel, stride (1,1), 8 parameters per kernel
    /// let conv = QConv2D::new(4, 4, 2, 2, 1, 1, 8)?;
    /// // Will apply kernel at 9 positions (3×3 grid of positions)
    /// // Total: 9 × 8 = 72 parameters
    /// ```
    pub fn new(
        width: usize,
        height: usize,
        kernel_width: usize,
        kernel_height: usize,
        stride_x: usize,
        stride_y: usize,
        n_params_per_kernel: usize,
    ) -> MLResult<Self> {
        if kernel_width > width {
            return Err(MLError::InvalidConfiguration(format!(
                "Kernel width {} exceeds grid width {}",
                kernel_width, width
            )));
        }

        if kernel_height > height {
            return Err(MLError::InvalidConfiguration(format!(
                "Kernel height {} exceeds grid height {}",
                kernel_height, height
            )));
        }

        if stride_x == 0 || stride_y == 0 {
            return Err(MLError::InvalidConfiguration(
                "Strides must be greater than 0".to_string(),
            ));
        }

        // Calculate number of kernel applications
        let n_kernels_x = (width - kernel_width) / stride_x + 1;
        let n_kernels_y = (height - kernel_height) / stride_y + 1;
        let n_kernels = n_kernels_x * n_kernels_y;
        let n_parameters = n_kernels * n_params_per_kernel;

        Ok(Self {
            width,
            height,
            kernel_width,
            kernel_height,
            stride_x,
            stride_y,
            n_params_per_kernel,
            n_parameters,
            name: format!(
                "QConv2D(kernel={}×{}, stride=({},{}))",
                kernel_width, kernel_height, stride_x, stride_y
            ),
        })
    }

    /// Get the 2D positions where kernels will be applied
    pub fn kernel_positions(&self) -> Vec<(usize, usize)> {
        let mut positions = Vec::new();
        let mut y = 0;

        while y + self.kernel_height <= self.height {
            let mut x = 0;
            while x + self.kernel_width <= self.width {
                positions.push((x, y));
                x += self.stride_x;
            }
            y += self.stride_y;
        }

        positions
    }

    /// Get the qubit coordinates for a specific kernel position
    /// Returns (x, y) coordinates in the 2D grid
    pub fn kernel_qubits(&self, position: (usize, usize)) -> Vec<(usize, usize)> {
        let (x0, y0) = position;
        let mut qubits = Vec::new();

        for y in y0..y0 + self.kernel_height {
            for x in x0..x0 + self.kernel_width {
                qubits.push((x, y));
            }
        }

        qubits
    }

    /// Convert 2D coordinates to 1D qubit index
    pub fn coords_to_index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    /// Convert 1D qubit index to 2D coordinates
    pub fn index_to_coords(&self, index: usize) -> (usize, usize) {
        (index % self.width, index / self.width)
    }

    /// Total number of qubits in the grid
    pub fn n_wires(&self) -> usize {
        self.width * self.height
    }
}

impl QConv2D {
    /// Get the total number of trainable parameters
    pub fn n_parameters(&self) -> usize {
        self.n_parameters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qconv1d_creation() {
        let conv = QConv1D::new(8, 3, 1, 6).unwrap();
        assert_eq!(conv.n_wires, 8);
        assert_eq!(conv.kernel_size, 3);
        assert_eq!(conv.stride, 1);
        assert_eq!(conv.n_parameters(), 36); // 6 kernels × 6 params
    }

    #[test]
    fn test_qconv1d_kernel_positions() {
        let conv = QConv1D::new(8, 3, 2, 4).unwrap();
        let positions = conv.kernel_positions();
        assert_eq!(positions, vec![0, 2, 4]);
    }

    #[test]
    fn test_qconv1d_kernel_qubits() {
        let conv = QConv1D::new(8, 3, 1, 4).unwrap();
        let qubits = conv.kernel_qubits(2);
        assert_eq!(qubits, vec![2, 3, 4]);
    }

    #[test]
    fn test_qconv1d_invalid_kernel_size() {
        let result = QConv1D::new(4, 6, 1, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_qconv1d_zero_stride() {
        let result = QConv1D::new(8, 3, 0, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_qconv2d_creation() {
        let conv = QConv2D::new(4, 4, 2, 2, 1, 1, 8).unwrap();
        assert_eq!(conv.width, 4);
        assert_eq!(conv.height, 4);
        assert_eq!(conv.kernel_width, 2);
        assert_eq!(conv.kernel_height, 2);
        assert_eq!(conv.n_parameters(), 72); // 9 kernels × 8 params
    }

    #[test]
    fn test_qconv2d_kernel_positions() {
        let conv = QConv2D::new(4, 4, 2, 2, 1, 1, 8).unwrap();
        let positions = conv.kernel_positions();
        assert_eq!(positions.len(), 9); // 3×3 grid
        assert_eq!(positions[0], (0, 0));
        assert_eq!(positions[4], (1, 1));
        assert_eq!(positions[8], (2, 2));
    }

    #[test]
    fn test_qconv2d_kernel_qubits() {
        let conv = QConv2D::new(4, 4, 2, 2, 1, 1, 8).unwrap();
        let qubits = conv.kernel_qubits((1, 1));
        assert_eq!(qubits, vec![(1, 1), (2, 1), (1, 2), (2, 2)]);
    }

    #[test]
    fn test_qconv2d_coords_conversion() {
        let conv = QConv2D::new(4, 4, 2, 2, 1, 1, 8).unwrap();

        // Test forward conversion
        assert_eq!(conv.coords_to_index(0, 0), 0);
        assert_eq!(conv.coords_to_index(3, 0), 3);
        assert_eq!(conv.coords_to_index(0, 1), 4);
        assert_eq!(conv.coords_to_index(3, 3), 15);

        // Test reverse conversion
        assert_eq!(conv.index_to_coords(0), (0, 0));
        assert_eq!(conv.index_to_coords(5), (1, 1));
        assert_eq!(conv.index_to_coords(15), (3, 3));
    }

    #[test]
    fn test_qconv2d_invalid_kernel() {
        let result = QConv2D::new(4, 4, 5, 2, 1, 1, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_qconv2d_zero_stride() {
        let result = QConv2D::new(4, 4, 2, 2, 0, 1, 8);
        assert!(result.is_err());
    }
}
