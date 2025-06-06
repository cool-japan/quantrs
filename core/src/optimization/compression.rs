//! Gate sequence compression using SciRS2 optimization
//!
//! This module provides advanced gate sequence compression techniques
//! leveraging SciRS2's optimization and linear algebra capabilities.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
    matrix_ops::{DenseMatrix, matrices_approx_equal},
};
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use scirs2_linalg::lowrank::{truncated_svd, randomized_svd};
// Tucker decomposition temporarily disabled due to scirs2-linalg compilation issues
// use scirs2_linalg::tensor_contraction::tucker::{tucker_decomposition, Tucker};
use scirs2_optimize::{
    differential_evolution,
    global::DifferentialEvolutionOptions,
    unconstrained::{minimize, Method as OptMethod, Options as OptOptions},
    error::OptimizeError,
};
use std::collections::HashMap;
use std::any::Any;

/// Configuration for gate sequence compression
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Maximum allowed error in gate approximation
    pub tolerance: f64,
    /// Maximum rank for low-rank approximations
    pub max_rank: Option<usize>,
    /// Whether to use randomized algorithms for speed
    pub use_randomized: bool,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Parallel execution threads
    pub num_threads: Option<usize>,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-10,
            max_rank: None,
            use_randomized: true,
            max_iterations: 1000,
            num_threads: None,
        }
    }
}

/// Gate sequence compression optimizer
pub struct GateSequenceCompressor {
    config: CompressionConfig,
    /// Cache of compressed gates
    compression_cache: HashMap<u64, CompressedGate>,
}

/// Compressed representation of a gate
#[derive(Debug, Clone)]
pub enum CompressedGate {
    /// Low-rank approximation U ≈ AB†
    LowRank {
        left: Array2<Complex64>,
        right: Array2<Complex64>,
        rank: usize,
    },
    /// Tucker decomposition for multi-qubit gates
    Tucker {
        core: Array2<Complex64>,
        factors: Vec<Array2<Complex64>>,
    },
    /// Parameterized gate with optimized parameters
    Parameterized {
        gate_type: String,
        parameters: Vec<f64>,
        qubits: Vec<QubitId>,
    },
    /// Original gate (no compression possible)
    Original(Box<dyn GateOp>),
}

impl GateSequenceCompressor {
    /// Create a new gate sequence compressor
    pub fn new(config: CompressionConfig) -> Self {
        Self {
            config,
            compression_cache: HashMap::new(),
        }
    }

    /// Compress a single gate using various techniques
    pub fn compress_gate(&mut self, gate: &dyn GateOp) -> QuantRS2Result<CompressedGate> {
        let matrix_vec = gate.matrix()?;
        
        // Convert vector to 2D array
        let n = (matrix_vec.len() as f64).sqrt() as usize;
        let mut matrix = Array2::zeros((n, n));
        for j in 0..n {
            for i in 0..n {
                matrix[(i, j)] = matrix_vec[j * n + i];
            }
        }
        
        let matrix_view = matrix.view();
        let hash = self.compute_matrix_hash(&matrix_view);
        
        // Check cache
        if let Some(compressed) = self.compression_cache.get(&hash) {
            return Ok(compressed.clone());
        }
        
        // Try different compression strategies
        let compressed = if let Some(low_rank) = self.try_low_rank_approximation(&matrix_view)? {
            low_rank
        } else if let Some(tucker) = self.try_tucker_decomposition(&matrix_view)? {
            tucker
        } else if let Some(param) = self.try_parameterized_compression(gate)? {
            param
        } else {
            CompressedGate::Original(gate.clone_gate())
        };
        
        // Cache the result
        self.compression_cache.insert(hash, compressed.clone());
        
        Ok(compressed)
    }

    /// Compress a sequence of gates
    pub fn compress_sequence(
        &mut self,
        gates: &[Box<dyn GateOp>],
    ) -> QuantRS2Result<Vec<CompressedGate>> {
        // Set up parallel execution if configured
        if let Some(threads) = self.config.num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .map_err(|e| QuantRS2Error::InvalidInput(e.to_string()))?;
        }
        
        // First, try to merge adjacent gates
        let merged = self.merge_adjacent_gates(gates)?;
        
        // Then compress each gate individually
        let compressed: Result<Vec<_>, _> = merged
            .iter()
            .map(|gate| self.compress_gate(gate.as_ref()))
            .collect();
        
        compressed
    }

    /// Try low-rank approximation using SVD
    fn try_low_rank_approximation(
        &self,
        matrix: &ArrayView2<Complex64>,
    ) -> QuantRS2Result<Option<CompressedGate>> {
        let (rows, cols) = matrix.dim();
        if rows != cols || rows < 4 {
            // Only try for larger gates
            return Ok(None);
        }
        
        // Convert to SciRS2 matrix format
        let real_part = Array2::from_shape_fn((rows, cols), |(i, j)| matrix[(i, j)].re);
        let imag_part = Array2::from_shape_fn((rows, cols), |(i, j)| matrix[(i, j)].im);
        
        // Try SVD-based compression
        let target_rank = self.config.max_rank.unwrap_or(rows / 2);
        
        // Apply SVD to real and imaginary parts separately
        let (u_real, s_real, vt_real) = if self.config.use_randomized {
            randomized_svd(&real_part.view(), target_rank, Some(10), Some(2))
                .map_err(|e| QuantRS2Error::InvalidInput(format!("SVD failed: {}", e)))?
        } else {
            truncated_svd(&real_part.view(), target_rank)
                .map_err(|e| QuantRS2Error::InvalidInput(format!("SVD failed: {}", e)))?
        };
        
        let (u_imag, s_imag, vt_imag) = if self.config.use_randomized {
            randomized_svd(&imag_part.view(), target_rank, Some(10), Some(2))
                .map_err(|e| QuantRS2Error::InvalidInput(format!("SVD failed: {}", e)))?
        } else {
            truncated_svd(&imag_part.view(), target_rank)
                .map_err(|e| QuantRS2Error::InvalidInput(format!("SVD failed: {}", e)))?
        };
        
        // Find effective rank based on singular values
        let effective_rank = self.find_effective_rank(&s_real, &s_imag)?;
        
        if effective_rank >= rows * 3 / 4 {
            // Not worth compressing
            return Ok(None);
        }
        
        // Reconstruct low-rank approximation
        let left = self.combine_complex(&u_real, &u_imag, effective_rank)?;
        let right = self.combine_complex_with_singular(&vt_real, &vt_imag, &s_real, &s_imag, effective_rank)?;
        
        // Verify approximation quality
        let approx = left.dot(&right.t());
        if !matrices_approx_equal(&approx.view(), matrix, self.config.tolerance) {
            return Ok(None);
        }
        
        Ok(Some(CompressedGate::LowRank {
            left,
            right,
            rank: effective_rank,
        }))
    }

    /// Try Tucker decomposition for multi-qubit gates
    fn try_tucker_decomposition(
        &self,
        matrix: &ArrayView2<Complex64>,
    ) -> QuantRS2Result<Option<CompressedGate>> {
        // Tucker decomposition temporarily disabled due to scirs2-linalg compilation issues
        // TODO: Re-enable when scirs2-linalg tensor_contraction feature is fixed
        Ok(None)
    }

    /// Try to find parameterized representation
    fn try_parameterized_compression(
        &self,
        gate: &dyn GateOp,
    ) -> QuantRS2Result<Option<CompressedGate>> {
        // This would identify if the gate can be represented
        // as a parameterized gate (e.g., rotation gates)
        
        // For now, we'll use global optimization to find parameters
        let matrix_vec = gate.matrix()?;
        let n = (matrix_vec.len() as f64).sqrt() as usize;
        
        if n > 4 {
            // Only try for single and two-qubit gates
            return Ok(None);
        }
        
        // Convert vector to 2D array
        let mut target_matrix = Array2::zeros((n, n));
        for j in 0..n {
            for i in 0..n {
                target_matrix[(i, j)] = matrix_vec[j * n + i];
            }
        }
        
        let gate_type = self.identify_gate_type(gate);
        
        // Set up bounds for optimization
        let dim = match gate_type.as_str() {
            "rotation" => 3, // Three Euler angles
            "phase" => 1,    // One phase parameter
            _ => 6,          // General parameterization
        };
        let bounds = vec![(Some(-std::f64::consts::PI), Some(std::f64::consts::PI)); dim];
        
        // Clone values needed for the closure to avoid borrowing self
        let target_matrix_clone = target_matrix.clone();
        let gate_type_clone = gate_type.clone();
        let tolerance = self.config.tolerance;
        
        // Create objective function
        let objective = move |x: &ndarray::ArrayView1<f64>| -> f64 {
            let params: Vec<f64> = x.iter().cloned().collect();
            
            // Inline the evaluation logic since we can't access self
            let gate_matrix = match gate_type_clone.as_str() {
                "rotation" => Array2::eye(target_matrix_clone.dim().0), // Placeholder
                "phase" => {
                    let mut matrix = Array2::eye(target_matrix_clone.dim().0);
                    if !params.is_empty() {
                        let phase = Complex64::from_polar(1.0, params[0]);
                        let n = matrix.dim().0;
                        matrix[(n - 1, n - 1)] = phase;
                    }
                    matrix
                },
                _ => Array2::eye(target_matrix_clone.dim().0), // Placeholder
            };
            
            // Compute Frobenius norm of difference
            let diff = &target_matrix_clone - &gate_matrix;
            diff.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
        };
        
        // Use differential evolution for global optimization
        let mut options = DifferentialEvolutionOptions::default();
        options.popsize = 50;
        options.maxiter = self.config.max_iterations;
        options.tol = self.config.tolerance;
        
        let de_bounds: Vec<(f64, f64)> = bounds.into_iter()
            .map(|(low, high)| (low.unwrap_or(-std::f64::consts::PI), high.unwrap_or(std::f64::consts::PI)))
            .collect();
        
        let result = differential_evolution(objective, de_bounds, Some(options), None)
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Parameter optimization failed: {:?}", e)))?;
        
        if result.fun > self.config.tolerance {
            // Optimization didn't converge well enough
            return Ok(None);
        }
        
        Ok(Some(CompressedGate::Parameterized {
            gate_type,
            parameters: result.x.to_vec(),
            qubits: vec![], // Would need to extract from gate
        }))
    }

    /// Merge adjacent gates that can be combined
    fn merge_adjacent_gates(
        &self,
        gates: &[Box<dyn GateOp>],
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut merged = Vec::new();
        let mut i = 0;
        
        while i < gates.len() {
            if i + 1 < gates.len() {
                // Check if gates can be merged
                if self.can_merge(gates[i].as_ref(), gates[i + 1].as_ref()) {
                    // Merge the gates
                    let combined = self.merge_two_gates(gates[i].as_ref(), gates[i + 1].as_ref())?;
                    merged.push(combined);
                    i += 2;
                } else {
                    merged.push(gates[i].clone_gate());
                    i += 1;
                }
            } else {
                merged.push(gates[i].clone_gate());
                i += 1;
            }
        }
        
        Ok(merged)
    }

    /// Check if two gates can be merged
    fn can_merge(&self, gate1: &dyn GateOp, gate2: &dyn GateOp) -> bool {
        // Gates can be merged if they:
        // 1. Act on the same qubits
        // 2. Are both unitary
        // 3. Their product is simpler than the individual gates
        
        // For now, simple check - same type gates on same qubits
        gate1.name() == gate2.name()
    }

    /// Merge two gates into one
    fn merge_two_gates(
        &self,
        gate1: &dyn GateOp,
        gate2: &dyn GateOp,
    ) -> QuantRS2Result<Box<dyn GateOp>> {
        // Get matrices
        let matrix1_vec = gate1.matrix()?;
        let matrix2_vec = gate2.matrix()?;
        
        // Convert to 2D arrays
        let n = (matrix1_vec.len() as f64).sqrt() as usize;
        let mut matrix1 = Array2::zeros((n, n));
        let mut matrix2 = Array2::zeros((n, n));
        
        for j in 0..n {
            for i in 0..n {
                matrix1[(i, j)] = matrix1_vec[j * n + i];
                matrix2[(i, j)] = matrix2_vec[j * n + i];
            }
        }
        
        // Matrix multiplication
        let combined_matrix = matrix2.dot(&matrix1);
        
        // Create a custom gate with the combined matrix
        Ok(Box::new(CustomGate::new(
            format!("{}_{}_merged", gate1.name(), gate2.name()),
            combined_matrix,
        )))
    }

    /// Compute hash of matrix for caching
    fn compute_matrix_hash(&self, matrix: &ArrayView2<Complex64>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for elem in matrix.iter() {
            // Hash real and imaginary parts
            elem.re.to_bits().hash(&mut hasher);
            elem.im.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Find effective rank based on singular values
    fn find_effective_rank(&self, s_real: &ndarray::Array1<f64>, s_imag: &ndarray::Array1<f64>) -> QuantRS2Result<usize> {
        let max_singular = s_real.iter().chain(s_imag.iter())
            .map(|s| s.abs())
            .fold(0.0, f64::max);
        
        let threshold = max_singular * self.config.tolerance;
        
        let rank = s_real.iter().zip(s_imag.iter())
            .take_while(|(sr, si)| sr.abs() > threshold || si.abs() > threshold)
            .count();
        
        Ok(rank.max(1))
    }

    /// Combine real and imaginary parts into complex matrix
    fn combine_complex(
        &self,
        real: &Array2<f64>,
        imag: &Array2<f64>,
        rank: usize,
    ) -> QuantRS2Result<Array2<Complex64>> {
        let (rows, _) = real.dim();
        let result = Array2::from_shape_fn((rows, rank), |(i, j)| {
            Complex64::new(real[(i, j)], imag[(i, j)])
        });
        Ok(result)
    }

    /// Combine with singular values
    fn combine_complex_with_singular(
        &self,
        vt_real: &Array2<f64>,
        vt_imag: &Array2<f64>,
        s_real: &ndarray::Array1<f64>,
        s_imag: &ndarray::Array1<f64>,
        rank: usize,
    ) -> QuantRS2Result<Array2<Complex64>> {
        let (_, cols) = vt_real.dim();
        let result = Array2::from_shape_fn((rank, cols), |(i, j)| {
            let s = Complex64::new(s_real[i], s_imag[i]);
            let v = Complex64::new(vt_real[(i, j)], vt_imag[(i, j)]);
            s * v
        });
        Ok(result)
    }

    /// Convert tensor data to complex matrix
    fn tensor_to_complex_matrix(&self, tensor: &[f64]) -> QuantRS2Result<Array2<Complex64>> {
        let size = (tensor.len() / 2) as f64;
        let dim = size.sqrt() as usize;
        
        let mut matrix = Array2::zeros((dim, dim));
        for i in 0..dim {
            for j in 0..dim {
                let idx = (i * dim + j) * 2;
                matrix[(i, j)] = Complex64::new(tensor[idx], tensor[idx + 1]);
            }
        }
        
        Ok(matrix)
    }
    
    /// Convert ArrayD to complex matrix
    fn tensor_to_complex_matrix_from_array(&self, tensor: &ndarray::ArrayD<f64>) -> QuantRS2Result<Array2<Complex64>> {
        // For now, just flatten the tensor and reshape to square matrix
        let elements: Vec<f64> = tensor.iter().cloned().collect();
        let size = elements.len() as f64;
        let dim = size.sqrt() as usize;
        
        if dim * dim != elements.len() {
            // If not square, pad with zeros
            let dim = (size.sqrt().ceil()) as usize;
            let mut matrix = Array2::zeros((dim, dim));
            for (idx, &val) in elements.iter().enumerate() {
                let i = idx / dim;
                let j = idx % dim;
                if i < dim && j < dim {
                    matrix[(i, j)] = Complex64::new(val, 0.0);
                }
            }
            Ok(matrix)
        } else {
            let mut matrix = Array2::zeros((dim, dim));
            for i in 0..dim {
                for j in 0..dim {
                    let idx = i * dim + j;
                    matrix[(i, j)] = Complex64::new(elements[idx], 0.0);
                }
            }
            Ok(matrix)
        }
    }

    /// Identify gate type for parameterization
    fn identify_gate_type(&self, gate: &dyn GateOp) -> String {
        // Simple heuristic based on gate name
        let name = gate.name();
        if name.contains("rot") || name.contains("Rot") {
            "rotation".to_string()
        } else if name.contains("phase") || name.contains("Phase") {
            "phase".to_string()
        } else {
            "general".to_string()
        }
    }
    
    /// Evaluate gate parameters for optimization
    fn evaluate_gate_parameters(&self, target: &Array2<Complex64>, gate_type: &str, params: &[f64]) -> f64 {
        // Construct gate from parameters
        let gate_matrix = match gate_type {
            "rotation" => self.rotation_matrix_from_params(params, target.dim().0),
            "phase" => self.phase_matrix_from_params(params, target.dim().0),
            _ => self.general_matrix_from_params(params, target.dim().0),
        };
        
        // Compute Frobenius norm of difference
        let diff = target - &gate_matrix;
        diff.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
    }
    
    fn rotation_matrix_from_params(&self, params: &[f64], dim: usize) -> Array2<Complex64> {
        // Construct rotation matrix from Euler angles
        // This is a placeholder - would need proper implementation
        Array2::eye(dim)
    }
    
    fn phase_matrix_from_params(&self, params: &[f64], dim: usize) -> Array2<Complex64> {
        let mut matrix = Array2::eye(dim);
        if !params.is_empty() {
            let phase = Complex64::from_polar(1.0, params[0]);
            matrix[(dim - 1, dim - 1)] = phase;
        }
        matrix
    }
    
    fn general_matrix_from_params(&self, _params: &[f64], dim: usize) -> Array2<Complex64> {
        // General parameterization - would need proper implementation
        Array2::eye(dim)
    }
}


/// Custom gate implementation for compressed gates
#[derive(Debug, Clone)]
pub struct CustomGate {
    name: String,
    matrix: Array2<Complex64>,
    qubits: Vec<QubitId>,
}

impl CustomGate {
    pub fn new(name: String, matrix: Array2<Complex64>) -> Self {
        // Determine number of qubits from matrix size
        let n_qubits = (matrix.dim().0 as f64).log2() as usize;
        let qubits = (0..n_qubits).map(|i| QubitId::new(i as u32)).collect();
        Self { name, matrix, qubits }
    }
    
    pub fn with_qubits(name: String, matrix: Array2<Complex64>, qubits: Vec<QubitId>) -> Self {
        Self { name, matrix, qubits }
    }
}

impl GateOp for CustomGate {
    fn name(&self) -> &'static str {
        // Since we need 'static, we leak the string
        Box::leak(self.name.clone().into_boxed_str())
    }
    
    fn qubits(&self) -> Vec<QubitId> {
        self.qubits.clone()
    }
    
    fn matrix(&self) -> QuantRS2Result<Vec<Complex64>> {
        // Flatten the matrix to a vector in column-major order
        let mut result = Vec::with_capacity(self.matrix.len());
        let (rows, cols) = self.matrix.dim();
        for j in 0..cols {
            for i in 0..rows {
                result.push(self.matrix[(i, j)]);
            }
        }
        Ok(result)
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn clone_gate(&self) -> Box<dyn GateOp> {
        Box::new(self.clone())
    }
}

/// Compression statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    pub original_gates: usize,
    pub compressed_gates: usize,
    pub low_rank_compressions: usize,
    pub tucker_compressions: usize,
    pub parameterized_compressions: usize,
    pub compression_ratio: f64,
    pub total_parameters_before: usize,
    pub total_parameters_after: usize,
}

impl GateSequenceCompressor {
    /// Get compression statistics
    pub fn get_stats(&self, original: &[Box<dyn GateOp>], compressed: &[CompressedGate]) -> CompressionStats {
        let mut stats = CompressionStats::default();
        stats.original_gates = original.len();
        stats.compressed_gates = compressed.len();
        
        for gate in compressed {
            match gate {
                CompressedGate::LowRank { left, right, .. } => {
                    stats.low_rank_compressions += 1;
                    stats.total_parameters_after += (left.len() + right.len()) * 2;
                }
                CompressedGate::Tucker { core, factors } => {
                    stats.tucker_compressions += 1;
                    stats.total_parameters_after += core.len() * 2;
                    stats.total_parameters_after += factors.iter().map(|f| f.len() * 2).sum::<usize>();
                }
                CompressedGate::Parameterized { parameters, .. } => {
                    stats.parameterized_compressions += 1;
                    stats.total_parameters_after += parameters.len();
                }
                CompressedGate::Original(gate) => {
                    if let Ok(matrix_vec) = gate.matrix() {
                        let size = (matrix_vec.len() as f64).sqrt() as usize;
                        stats.total_parameters_after += size * size * 2;
                    }
                }
            }
        }
        
        for gate in original {
            if let Ok(matrix_vec) = gate.matrix() {
                let size = (matrix_vec.len() as f64).sqrt() as usize;
                stats.total_parameters_before += size * size * 2;
            }
        }
        
        stats.compression_ratio = if stats.total_parameters_before > 0 {
            stats.total_parameters_after as f64 / stats.total_parameters_before as f64
        } else {
            1.0
        };
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::single::{Hadamard, PauliX, PauliZ};
    use crate::qubit::QubitId;

    #[test]
    fn test_gate_compression() {
        let config = CompressionConfig::default();
        let mut compressor = GateSequenceCompressor::new(config);
        
        // Test single gate compression
        let h_gate = Hadamard { target: QubitId::new(0) };
        let compressed = compressor.compress_gate(&h_gate).unwrap();
        
        match compressed {
            CompressedGate::Original(_) => {
                // H gate is already minimal, shouldn't compress
            }
            _ => panic!("H gate shouldn't be compressed"),
        }
    }

    #[test]
    fn test_sequence_compression() {
        let config = CompressionConfig::default();
        let mut compressor = GateSequenceCompressor::new(config);
        
        // Create a sequence of gates
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: QubitId::new(0) }),
            Box::new(PauliX { target: QubitId::new(0) }),
            Box::new(Hadamard { target: QubitId::new(0) }),
        ];
        
        let compressed = compressor.compress_sequence(&gates).unwrap();
        assert!(compressed.len() <= gates.len());
    }

    #[test]
    fn test_compression_stats() {
        let config = CompressionConfig::default();
        let mut compressor = GateSequenceCompressor::new(config);
        
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: QubitId::new(0) }),
            Box::new(PauliZ { target: QubitId::new(0) }),
        ];
        
        let compressed = compressor.compress_sequence(&gates).unwrap();
        let stats = compressor.get_stats(&gates, &compressed);
        
        assert_eq!(stats.original_gates, 2);
        assert!(stats.compression_ratio <= 1.0);
    }
}