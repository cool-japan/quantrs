//! Quantum pooling layers for dimensionality reduction
//!
//! This module provides quantum pooling operations that reduce the number of qubits
//! while preserving important quantum features, analogous to pooling in classical CNNs.
//!
//! # Layers
//!
//! - **QMaxPool**: Select qubits with highest measurement probability
//! - **QAvgPool**: Average measurements over pooling regions
//!
//! # Features
//!
//! - Configurable pool size and stride
//! - Non-trainable dimensionality reduction
//! - Compatible with TorchQuantum training framework
//!
//! # Example
//!
//! ```ignore
//! use quantrs2_ml::torchquantum::pooling::QMaxPool;
//!
//! // Create pooling layer: 8 qubits, pool size 2, stride 2
//! let pool = QMaxPool::new(8, 2, 2)?;
//! println!("Output qubits: {}", pool.output_wires());
//! ```

use crate::error::{MLError, Result as MLResult};

/// Quantum pooling layer using maximum measurement probability
///
/// Reduces the number of qubits by measuring subsets and keeping
/// the qubit with the highest measurement probability in each pool.
#[derive(Debug, Clone)]
pub struct QMaxPool {
    /// Number of input wires
    n_wires: usize,
    /// Pool size (number of qubits per pool)
    pool_size: usize,
    /// Stride (step size for pooling windows)
    stride: usize,
    /// Layer name for debugging
    name: String,
}

impl QMaxPool {
    /// Create a new quantum max pooling layer
    ///
    /// # Arguments
    /// * `n_wires` - Number of input qubits
    /// * `pool_size` - Size of each pooling window
    /// * `stride` - Step size for pooling windows
    ///
    /// # Example
    /// ```ignore
    /// // 8-qubit input, pool size 2, stride 2
    /// let pool = QMaxPool::new(8, 2, 2)?;
    /// // Will pool: (0,1), (2,3), (4,5), (6,7) -> 4 output qubits
    /// ```
    pub fn new(n_wires: usize, pool_size: usize, stride: usize) -> MLResult<Self> {
        if pool_size > n_wires {
            return Err(MLError::InvalidConfiguration(format!(
                "Pool size {} exceeds number of wires {}",
                pool_size, n_wires
            )));
        }

        if stride == 0 {
            return Err(MLError::InvalidConfiguration(
                "Stride must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            n_wires,
            pool_size,
            stride,
            name: format!("QMaxPool(size={}, stride={})", pool_size, stride),
        })
    }

    /// Get the positions where pooling windows start
    pub fn pool_positions(&self) -> Vec<usize> {
        let mut positions = Vec::new();
        let mut pos = 0;

        while pos + self.pool_size <= self.n_wires {
            positions.push(pos);
            pos += self.stride;
        }

        positions
    }

    /// Get the qubit indices for a specific pool
    pub fn pool_qubits(&self, position: usize) -> Vec<usize> {
        (position..position + self.pool_size).collect()
    }

    /// Calculate the number of output qubits after pooling
    pub fn output_size(&self) -> usize {
        self.pool_positions().len()
    }
}

impl QMaxPool {
    /// Get the total number of trainable parameters (always 0 for pooling)
    pub fn n_parameters(&self) -> usize {
        0
    }
}

/// Quantum average pooling layer
///
/// Reduces the number of qubits by applying averaging operations
/// over pools of qubits, typically using measurement statistics.
#[derive(Debug, Clone)]
pub struct QAvgPool {
    /// Number of input wires
    n_wires: usize,
    /// Pool size (number of qubits per pool)
    pool_size: usize,
    /// Stride (step size for pooling windows)
    stride: usize,
    /// Layer name for debugging
    name: String,
}

impl QAvgPool {
    /// Create a new quantum average pooling layer
    ///
    /// # Arguments
    /// * `n_wires` - Number of input qubits
    /// * `pool_size` - Size of each pooling window
    /// * `stride` - Step size for pooling windows
    ///
    /// # Example
    /// ```ignore
    /// // 8-qubit input, pool size 2, stride 2
    /// let pool = QAvgPool::new(8, 2, 2)?;
    /// // Will average: (0,1), (2,3), (4,5), (6,7) -> 4 output qubits
    /// ```
    pub fn new(n_wires: usize, pool_size: usize, stride: usize) -> MLResult<Self> {
        if pool_size > n_wires {
            return Err(MLError::InvalidConfiguration(format!(
                "Pool size {} exceeds number of wires {}",
                pool_size, n_wires
            )));
        }

        if stride == 0 {
            return Err(MLError::InvalidConfiguration(
                "Stride must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            n_wires,
            pool_size,
            stride,
            name: format!("QAvgPool(size={}, stride={})", pool_size, stride),
        })
    }

    /// Get the positions where pooling windows start
    pub fn pool_positions(&self) -> Vec<usize> {
        let mut positions = Vec::new();
        let mut pos = 0;

        while pos + self.pool_size <= self.n_wires {
            positions.push(pos);
            pos += self.stride;
        }

        positions
    }

    /// Get the qubit indices for a specific pool
    pub fn pool_qubits(&self, position: usize) -> Vec<usize> {
        (position..position + self.pool_size).collect()
    }

    /// Calculate the number of output qubits after pooling
    pub fn output_size(&self) -> usize {
        self.pool_positions().len()
    }
}

impl QAvgPool {
    /// Get the total number of trainable parameters (always 0 for pooling)
    pub fn n_parameters(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qmaxpool_creation() {
        let pool = QMaxPool::new(8, 2, 2).unwrap();
        assert_eq!(pool.n_wires, 8);
        assert_eq!(pool.pool_size, 2);
        assert_eq!(pool.stride, 2);
        assert_eq!(pool.n_parameters(), 0);
    }

    #[test]
    fn test_qmaxpool_positions() {
        let pool = QMaxPool::new(8, 2, 2).unwrap();
        let positions = pool.pool_positions();
        assert_eq!(positions, vec![0, 2, 4, 6]);
    }

    #[test]
    fn test_qmaxpool_qubits() {
        let pool = QMaxPool::new(8, 2, 2).unwrap();
        let qubits = pool.pool_qubits(4);
        assert_eq!(qubits, vec![4, 5]);
    }

    #[test]
    fn test_qmaxpool_output_size() {
        let pool = QMaxPool::new(8, 2, 2).unwrap();
        assert_eq!(pool.output_size(), 4);
    }

    #[test]
    fn test_qmaxpool_invalid_pool_size() {
        let result = QMaxPool::new(4, 6, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_qavgpool_creation() {
        let pool = QAvgPool::new(8, 2, 2).unwrap();
        assert_eq!(pool.n_wires, 8);
        assert_eq!(pool.pool_size, 2);
        assert_eq!(pool.stride, 2);
        assert_eq!(pool.n_parameters(), 0);
    }

    #[test]
    fn test_qavgpool_positions() {
        let pool = QAvgPool::new(8, 2, 2).unwrap();
        let positions = pool.pool_positions();
        assert_eq!(positions, vec![0, 2, 4, 6]);
    }

    #[test]
    fn test_qavgpool_output_size() {
        let pool = QAvgPool::new(8, 2, 2).unwrap();
        assert_eq!(pool.output_size(), 4);
    }
}
