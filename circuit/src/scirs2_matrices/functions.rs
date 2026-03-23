//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::builder::Circuit;
pub use scirs2_core::Complex64;
use scirs2_core::{
    parallel_ops::{IndexedParallelIterator, ParallelIterator},
    simd_ops::*,
};

use super::types::{
    CircuitToSparseMatrix, HardwareSpecification, SparseFormat, SparseGateLibrary, SparseMatrix,
    SparseOptimizer,
};

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::single::Hadamard;
    use quantrs2_core::qubit::QubitId;
    #[test]
    fn test_complex_arithmetic() {
        let c1 = Complex64::new(1.0, 2.0);
        let c2 = Complex64::new(3.0, 4.0);
        let sum = c1 + c2;
        assert_eq!(sum.re, 4.0);
        assert_eq!(sum.im, 6.0);
        let product = c1 * c2;
        assert_eq!(product.re, -5.0);
        assert_eq!(product.im, 10.0);
    }
    #[test]
    fn test_sparse_matrix_creation() {
        let matrix = SparseMatrix::identity(4);
        assert_eq!(matrix.shape, (4, 4));
        assert_eq!(matrix.nnz(), 4);
    }
    #[test]
    fn test_gate_library() {
        let mut library = SparseGateLibrary::new();
        let x_gate = library.get_gate("X");
        assert!(x_gate.is_some());
        let h_gate = library.get_gate("H");
        assert!(h_gate.is_some());
        let rz_gate = library.get_parameterized_gate("RZ", &[std::f64::consts::PI]);
        assert!(rz_gate.is_some());
    }
    #[test]
    fn test_matrix_operations() {
        let id = SparseMatrix::identity(2);
        let mut x_gate = SparseMatrix::new(2, 2, SparseFormat::COO);
        x_gate.insert(0, 1, Complex64::new(1.0, 0.0));
        x_gate.insert(1, 0, Complex64::new(1.0, 0.0));
        let result = x_gate
            .matmul(&x_gate)
            .expect("Failed to multiply X gate with itself");
        assert!(result.matrices_equal(&id, 1e-12));
    }
    #[test]
    fn test_unitary_check() {
        let library = SparseGateLibrary::new();
        let h_gate = library
            .get_gate("H")
            .expect("Hadamard gate should exist in library");
        assert!(h_gate.is_unitary(1e-10));
        assert_eq!(h_gate.shape, (2, 2));
    }
    #[test]
    fn test_circuit_conversion() {
        let converter = CircuitToSparseMatrix::new();
        let mut circuit = Circuit::<1>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("Failed to add Hadamard gate");
        let matrix = converter
            .convert(&circuit)
            .expect("Failed to convert circuit to sparse matrix");
        assert_eq!(matrix.shape, (2, 2));
    }
    #[test]
    fn test_enhanced_gate_properties_analysis() {
        let library = SparseGateLibrary::new();
        let x_gate = library
            .get_gate("X")
            .expect("X gate should exist in library");
        let optimizer = SparseOptimizer::new();
        let properties = optimizer.analyze_gate_properties(x_gate);
        assert!(properties.is_unitary);
        assert!(properties.is_hermitian);
        assert!(properties.sparsity < 1.0);
        assert!(properties.spectral_radius > 0.0);
        assert!(properties.matrix_norm > 0.0);
    }
    #[test]
    fn test_hardware_optimization() {
        let hardware_spec = HardwareSpecification {
            has_gpu: true,
            simd_width: 256,
            has_tensor_cores: true,
            ..Default::default()
        };
        let library = SparseGateLibrary::new_for_hardware(hardware_spec);
        let x_gate = library
            .get_gate("X")
            .expect("X gate should exist in hardware-optimized library");
        assert_eq!(x_gate.format, SparseFormat::GPUOptimized);
    }
    #[test]
    fn test_parameterized_gate_caching() {
        let mut library = SparseGateLibrary::new();
        let rz1 = library.get_parameterized_gate("RZ", &[std::f64::consts::PI]);
        assert!(rz1.is_some());
        assert_eq!(library.metrics.cache_misses, 1);
        let rz2 = library.get_parameterized_gate("RZ", &[std::f64::consts::PI]);
        assert!(rz2.is_some());
        assert_eq!(library.metrics.cache_hits, 1);
    }
    #[test]
    fn test_simd_matrix_operations() {
        let matrix1 = SparseMatrix::new(2, 2, SparseFormat::SIMDAligned);
        let matrix2 = SparseMatrix::new(2, 2, SparseFormat::SIMDAligned);
        let result = matrix1.matmul(&matrix2);
        assert!(result.is_ok());
        let result_matrix = result.expect("Failed to perform SIMD matrix multiplication");
        assert!(result_matrix.metrics.simd_utilization > 0.0);
    }
}
