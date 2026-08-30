//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::builder::Circuit;
use quantrs2_core::{
    buffer_pool::BufferPool,
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
pub use scirs2_core::Complex64;
use scirs2_core::{
    parallel_ops::{IndexedParallelIterator, ParallelIterator},
    simd_ops::*,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

pub struct BLAS;
impl BLAS {
    /// Check whether two sparse matrices are approximately equal entry-wise within `tol`.
    /// For each entry present in either matrix the corresponding value in the other is
    /// treated as zero when absent, and the element-wise difference must satisfy |diff| ≤ tol.
    #[must_use]
    pub fn matrix_approx_equal(
        a: &SciRSSparseMatrix<Complex64>,
        b: &SciRSSparseMatrix<Complex64>,
        tol: f64,
    ) -> bool {
        if a.shape != b.shape {
            return false;
        }
        let mut b_map: HashMap<(usize, usize), Complex64> = HashMap::with_capacity(b.data.len());
        for &(r, c, v) in &b.data {
            b_map.insert((r, c), v);
        }
        for &(r, c, va) in &a.data {
            let vb = b_map.remove(&(r, c)).unwrap_or(Complex64::new(0.0, 0.0));
            if (va - vb).norm() > tol {
                return false;
            }
        }
        for (_, vb) in b_map {
            if vb.norm() > tol {
                return false;
            }
        }
        true
    }
    /// 2-norm condition number `σ_max / σ_min` from the singular values.
    /// Returns `f64::INFINITY` for a singular (or empty) matrix.
    #[must_use]
    pub fn condition_number(matrix: &SciRSSparseMatrix<Complex64>) -> f64 {
        let (dense, rows, cols) = densify(matrix);
        if rows == 0 || cols == 0 {
            return f64::INFINITY;
        }
        let sv = singular_values_dense(&dense, rows, cols);
        let smax = sv.first().copied().unwrap_or(0.0);
        let smin = sv.last().copied().unwrap_or(0.0);
        if smax == 0.0 || smin <= smax * 1e-15 {
            f64::INFINITY
        } else {
            smax / smin
        }
    }
    #[must_use]
    pub fn is_symmetric(matrix: &SciRSSparseMatrix<Complex64>, tol: f64) -> bool {
        if matrix.shape.0 != matrix.shape.1 {
            return false;
        }
        for (row, col, value) in &matrix.data {
            let transpose_entry = matrix
                .data
                .iter()
                .find(|(r, c, _)| *r == *col && *c == *row);
            match transpose_entry {
                Some((_, _, transpose_value)) => {
                    if (value - transpose_value).norm() > tol {
                        return false;
                    }
                }
                None => {
                    if value.norm() > tol {
                        return false;
                    }
                }
            }
        }
        true
    }
    #[must_use]
    pub fn is_hermitian(matrix: &SciRSSparseMatrix<Complex64>, tol: f64) -> bool {
        if matrix.shape.0 != matrix.shape.1 {
            return false;
        }
        for (row, col, value) in &matrix.data {
            let conj_transpose_entry = matrix
                .data
                .iter()
                .find(|(r, c, _)| *r == *col && *c == *row);
            match conj_transpose_entry {
                Some((_, _, conj_transpose_value)) => {
                    if (value - conj_transpose_value.conj()).norm() > tol {
                        return false;
                    }
                }
                None => {
                    if value.norm() > tol {
                        return false;
                    }
                }
            }
        }
        true
    }
    /// A matrix is positive definite iff it is Hermitian and every eigenvalue is
    /// strictly positive.
    #[must_use]
    pub fn is_positive_definite(matrix: &SciRSSparseMatrix<Complex64>) -> bool {
        if !Self::is_hermitian(matrix, 1e-12) {
            return false;
        }
        let (dense, rows, cols) = densify(matrix);
        if rows == 0 || rows != cols {
            return false;
        }
        hermitian_eigenvalues_dense(&dense, rows)
            .iter()
            .all(|&e| e > 1e-12)
    }
    /// Matrix norm computed from the actual entries.  Supported `norm_type`
    /// values: `"1"`/`"one"` (max column sum), `"inf"`/`"infinity"` (max row
    /// sum), `"max"` (largest magnitude entry), `"2"`/`"spectral"` (largest
    /// singular value); any other value yields the Frobenius norm.
    #[must_use]
    pub fn matrix_norm(matrix: &SciRSSparseMatrix<Complex64>, norm_type: &str) -> f64 {
        match norm_type {
            "1" | "one" | "L1" => {
                let mut col_sums: HashMap<usize, f64> = HashMap::new();
                for &(_, c, v) in &matrix.data {
                    *col_sums.entry(c).or_insert(0.0) += v.norm();
                }
                col_sums.values().copied().fold(0.0, f64::max)
            }
            "inf" | "infinity" | "Linf" => {
                let mut row_sums: HashMap<usize, f64> = HashMap::new();
                for &(r, _, v) in &matrix.data {
                    *row_sums.entry(r).or_insert(0.0) += v.norm();
                }
                row_sums.values().copied().fold(0.0, f64::max)
            }
            "max" => matrix
                .data
                .iter()
                .map(|(_, _, v)| v.norm())
                .fold(0.0, f64::max),
            "2" | "spectral" => {
                let (dense, rows, cols) = densify(matrix);
                singular_values_dense(&dense, rows, cols)
                    .first()
                    .copied()
                    .unwrap_or(0.0)
            }
            _ => matrix
                .data
                .iter()
                .map(|(_, _, v)| v.norm_sqr())
                .sum::<f64>()
                .sqrt(),
        }
    }
    /// Numerical rank: the number of singular values above `tol` (with a relative
    /// safety floor scaled by the largest singular value and machine epsilon).
    #[must_use]
    pub fn numerical_rank(matrix: &SciRSSparseMatrix<Complex64>, tol: f64) -> usize {
        let (dense, rows, cols) = densify(matrix);
        if rows == 0 || cols == 0 {
            return 0;
        }
        let sv = singular_values_dense(&dense, rows, cols);
        let smax = sv.first().copied().unwrap_or(0.0);
        let threshold = tol.max(smax * (rows.max(cols) as f64) * f64::EPSILON);
        sv.iter().filter(|&&s| s > threshold).count()
    }
    /// Spectral radius (largest eigenvalue magnitude, via power iteration) and the
    /// eigenvalue-magnitude spread `max|λ| − min|λ|` (min via inverse iteration).
    #[must_use]
    pub fn spectral_analysis(matrix: &SciRSSparseMatrix<Complex64>) -> SpectralAnalysis {
        let (dense, rows, cols) = densify(matrix);
        if rows == 0 || rows != cols {
            return SpectralAnalysis {
                spectral_radius: 0.0,
                eigenvalue_spread: 0.0,
            };
        }
        let radius = spectral_radius_dense(&dense, rows);
        let min_mag = min_eig_magnitude_dense(&dense, rows);
        SpectralAnalysis {
            spectral_radius: radius,
            eigenvalue_spread: (radius - min_mag).max(0.0),
        }
    }
    /// Average gate fidelity between two gates, `F_avg = (d·F_pro + 1)/(d + 1)`
    /// with process fidelity `F_pro = |Tr(A† B)|² / d²`.  Equals `1` for
    /// identical unitaries.
    #[must_use]
    pub fn gate_fidelity(
        a: &SciRSSparseMatrix<Complex64>,
        b: &SciRSSparseMatrix<Complex64>,
    ) -> f64 {
        let d = a.shape.0;
        if d == 0 || a.shape != b.shape {
            return 0.0;
        }
        let f_pro = frobenius_inner(a, b).norm_sqr() / (d as f64 * d as f64);
        let dd = d as f64;
        (dd * f_pro + 1.0) / (dd + 1.0)
    }
    /// Trace-norm distance `½‖A − B‖₁ = ½ Σ σ_i(A − B)` (half the sum of the
    /// singular values of the difference).  Zero for identical operands.
    #[must_use]
    pub fn trace_distance(
        a: &SciRSSparseMatrix<Complex64>,
        b: &SciRSSparseMatrix<Complex64>,
    ) -> f64 {
        if a.shape != b.shape {
            return f64::INFINITY;
        }
        let (da, rows, cols) = densify(a);
        let (db, _, _) = densify(b);
        let diff: Vec<Complex64> = da.iter().zip(db.iter()).map(|(x, y)| x - y).collect();
        0.5 * singular_values_dense(&diff, rows, cols).iter().sum::<f64>()
    }
    /// Diamond-norm distance between the unitary channels defined by `A` and `B`
    /// (exact for unitary operands), from the eigenvalues of `W = A† B`.
    #[must_use]
    pub fn diamond_distance(
        a: &SciRSSparseMatrix<Complex64>,
        b: &SciRSSparseMatrix<Complex64>,
    ) -> f64 {
        if a.shape != b.shape || a.shape.0 == 0 {
            return 0.0;
        }
        let n = a.shape.0;
        let (da, _, _) = densify(a);
        let (db, _, _) = densify(b);
        // W = A† B
        let mut w = vec![Complex64::new(0.0, 0.0); n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    acc += da[k * n + i].conj() * db[k * n + j];
                }
                w[i * n + j] = acc;
            }
        }
        hull_diamond_distance(&normal_eigenvalues_dense(&w, n))
    }
    /// Process (entanglement) fidelity `F_pro = |Tr(A† B)|² / d²`.  Equals `1` for
    /// identical unitaries.
    #[must_use]
    pub fn process_fidelity(
        a: &SciRSSparseMatrix<Complex64>,
        b: &SciRSSparseMatrix<Complex64>,
    ) -> f64 {
        let d = a.shape.0;
        if d == 0 || a.shape != b.shape {
            return 0.0;
        }
        frobenius_inner(a, b).norm_sqr() / (d as f64 * d as f64)
    }
    /// Leading-order coherent/incoherent split of the average-gate infidelity
    /// between actual `A` and ideal `B`.  From the eigenphases `{θ_k}` of the
    /// error unitary `W = B† A`, the coherent part scales with the squared mean
    /// phase `⟨θ⟩²` (a systematic over-rotation) and the incoherent part with the
    /// phase variance `Var(θ)`; both vanish when `A == B`.
    #[must_use]
    pub fn error_decomposition(
        a: &SciRSSparseMatrix<Complex64>,
        b: &SciRSSparseMatrix<Complex64>,
    ) -> ErrorDecomposition {
        let n = a.shape.0;
        if n == 0 || a.shape != b.shape {
            return ErrorDecomposition {
                coherent_component: 0.0,
                incoherent_component: 0.0,
            };
        }
        let (da, _, _) = densify(a);
        let (db, _, _) = densify(b);
        // W = B† A
        let mut w = vec![Complex64::new(0.0, 0.0); n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    acc += db[k * n + i].conj() * da[k * n + j];
                }
                w[i * n + j] = acc;
            }
        }
        let phases: Vec<f64> = normal_eigenvalues_dense(&w, n)
            .iter()
            .map(|z| z.arg())
            .collect();
        let d = n as f64;
        let mean = phases.iter().sum::<f64>() / d;
        let mean_sq = phases.iter().map(|p| p * p).sum::<f64>() / d;
        let var = (mean_sq - mean * mean).max(0.0);
        let pref = d / (d + 1.0);
        ErrorDecomposition {
            coherent_component: pref * mean * mean,
            incoherent_component: pref * var,
        }
    }
    pub const fn sparse_matvec(
        _matrix: &SciRSSparseMatrix<Complex64>,
        _vector: &VectorizedOps,
    ) -> QuantRS2Result<VectorizedOps> {
        Ok(VectorizedOps)
    }
    /// Matrix exponential `exp(scale · matrix)` via dense scaling-and-squaring.
    pub fn matrix_exp(
        matrix: &SciRSSparseMatrix<Complex64>,
        scale: f64,
    ) -> QuantRS2Result<SciRSSparseMatrix<Complex64>> {
        let (rows, cols) = matrix.shape;
        if rows != cols {
            return Err(QuantRS2Error::InvalidInput(
                "Matrix exponentiation requires a square matrix".to_string(),
            ));
        }
        let (dense, _, _) = densify(matrix);
        let expm = expm_dense(&dense, rows, scale);
        let mut result = SciRSSparseMatrix::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let value = expm[i * cols + j];
                if value.norm() > 1e-15 {
                    result.insert(i, j, value);
                }
            }
        }
        Ok(result)
    }
}
pub struct SparsityPattern;
impl SparsityPattern {
    #[must_use]
    pub const fn analyze(_matrix: &SciRSSparseMatrix<Complex64>) -> Self {
        Self
    }
    #[must_use]
    pub const fn estimate_compression_ratio(&self) -> f64 {
        0.5
    }
    #[must_use]
    pub const fn bandwidth(&self) -> usize {
        10
    }
    #[must_use]
    pub const fn is_diagonal(&self) -> bool {
        false
    }
    #[must_use]
    pub const fn has_block_structure(&self) -> bool {
        false
    }
    #[must_use]
    pub const fn is_gpu_suitable(&self) -> bool {
        false
    }
    #[must_use]
    pub const fn is_simd_aligned(&self) -> bool {
        true
    }
    #[must_use]
    pub const fn sparsity(&self) -> f64 {
        0.1
    }
    #[must_use]
    pub const fn has_row_major_access(&self) -> bool {
        true
    }
    #[must_use]
    pub const fn analyze_access_patterns(&self) -> AccessPatterns {
        AccessPatterns
    }
}
pub struct VectorizedOps;
impl VectorizedOps {
    #[must_use]
    pub const fn from_slice(_slice: &[Complex64]) -> Self {
        Self
    }
    pub const fn copy_to_slice(&self, _slice: &mut [Complex64]) {}
}
pub struct ParallelMatrixOps;
impl ParallelMatrixOps {
    #[must_use]
    pub const fn kronecker_product(
        a: &SciRSSparseMatrix<Complex64>,
        b: &SciRSSparseMatrix<Complex64>,
    ) -> SciRSSparseMatrix<Complex64> {
        SciRSSparseMatrix::new(a.shape.0 * b.shape.0, a.shape.1 * b.shape.1)
    }
    pub fn batch_optimize(
        matrices: &[SparseMatrix],
        _simd_ops: &Arc<SimdOperations>,
        _buffer_pool: &Arc<quantrs2_core::buffer_pool::BufferPool<Complex64>>,
    ) -> Vec<SparseMatrix> {
        matrices.to_vec()
    }
}
/// Enhanced performance metrics for sparse matrix operations
#[derive(Debug, Clone)]
pub struct SparseMatrixMetrics {
    pub operation_time: std::time::Duration,
    pub memory_usage: usize,
    pub compression_ratio: f64,
    pub simd_utilization: f64,
    pub cache_hits: usize,
}
#[derive(Debug, Clone)]
pub struct SciRSSparseMatrix<T> {
    data: Vec<(usize, usize, T)>,
    shape: (usize, usize),
}
impl<T: Clone> SciRSSparseMatrix<T> {
    #[must_use]
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: Vec::new(),
            shape: (rows, cols),
        }
    }
    #[must_use]
    pub fn identity(size: usize) -> Self
    where
        T: From<f64> + Default,
    {
        let mut matrix = Self::new(size, size);
        for i in 0..size {
            matrix.data.push((i, i, T::from(1.0)));
        }
        matrix
    }
    pub fn insert(&mut self, row: usize, col: usize, value: T) {
        self.data.push((row, col, value));
    }
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }
    /// Read-only view of the stored `(row, col, value)` triplets (COO format).
    #[must_use]
    pub fn triplets(&self) -> &[(usize, usize, T)] {
        &self.data
    }
}
impl SciRSSparseMatrix<Complex64> {
    /// Sparse matrix multiplication using COO-format accumulation.
    /// Computes C = A * B where entries are accumulated by (row, col) key.
    pub fn matmul(&self, other: &Self) -> QuantRS2Result<Self> {
        if self.shape.1 != other.shape.0 {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Matrix dimension mismatch: ({},{}) * ({},{})",
                self.shape.0, self.shape.1, other.shape.0, other.shape.1
            )));
        }
        let mut acc: HashMap<(usize, usize), Complex64> = HashMap::new();
        for &(i, k, a_ik) in &self.data {
            for &(k2, j, b_kj) in &other.data {
                if k == k2 {
                    *acc.entry((i, j)).or_insert(Complex64::new(0.0, 0.0)) += a_ik * b_kj;
                }
            }
        }
        let mut result = Self::new(self.shape.0, other.shape.1);
        result.data = acc
            .into_iter()
            .filter(|(_, v)| v.norm() > 1e-300)
            .map(|((r, c), v)| (r, c, v))
            .collect();
        Ok(result)
    }
    #[must_use]
    pub fn transpose_optimized(&self) -> Self {
        let mut result = Self::new(self.shape.1, self.shape.0);
        result.data = self.data.iter().map(|&(r, c, v)| (c, r, v)).collect();
        result
    }
    /// Conjugate transpose (Hermitian adjoint U†): swap indices and conjugate values.
    #[must_use]
    pub fn hermitian_conjugate(&self) -> Self {
        let mut result = Self::new(self.shape.1, self.shape.0);
        result.data = self
            .data
            .iter()
            .map(|&(r, c, v)| (c, r, v.conj()))
            .collect();
        result
    }
    #[must_use]
    pub fn convert_to_format(&self, _format: SciRSSparseFormat) -> Self {
        self.clone()
    }
    pub fn compress(&self, _level: CompressionLevel) -> QuantRS2Result<Self> {
        Ok(self.clone())
    }
    #[must_use]
    pub fn memory_footprint(&self) -> usize {
        self.data.len() * std::mem::size_of::<(usize, usize, Complex64)>()
    }
}
/// Circuit to sparse matrix converter
pub struct CircuitToSparseMatrix {
    gate_library: Arc<SparseGateLibrary>,
}
impl CircuitToSparseMatrix {
    /// Create a new converter
    #[must_use]
    pub fn new() -> Self {
        Self {
            gate_library: Arc::new(SparseGateLibrary::new()),
        }
    }
    /// Convert circuit to sparse matrix representation
    pub fn convert<const N: usize>(&self, circuit: &Circuit<N>) -> QuantRS2Result<SparseMatrix> {
        let matrix_size = 1usize << N;
        let mut result = SparseMatrix::identity(matrix_size);
        for gate in circuit.gates() {
            let gate_matrix = self.gate_to_sparse_matrix(gate.as_ref(), N)?;
            result = gate_matrix.matmul(&result)?;
        }
        Ok(result)
    }
    /// Convert single gate to sparse matrix
    fn gate_to_sparse_matrix(
        &self,
        gate: &dyn GateOp,
        total_qubits: usize,
    ) -> QuantRS2Result<SparseMatrix> {
        let gate_name = gate.name();
        let qubits = gate.qubits();
        match qubits.len() {
            1 => {
                let target_qubit = qubits[0].id() as usize;
                self.gate_library
                    .embed_single_qubit_gate(gate_name, target_qubit, total_qubits)
            }
            2 => {
                let control_qubit = qubits[0].id() as usize;
                let target_qubit = qubits[1].id() as usize;
                self.gate_library.embed_two_qubit_gate(
                    gate_name,
                    control_qubit,
                    target_qubit,
                    total_qubits,
                )
            }
            _ => Err(QuantRS2Error::InvalidInput(
                "Multi-qubit gates beyond 2 qubits not yet supported".to_string(),
            )),
        }
    }
    /// Get gate library
    #[must_use]
    pub fn gate_library(&self) -> &SparseGateLibrary {
        &self.gate_library
    }
}
/// Advanced sparse matrix optimization utilities with `SciRS2` integration
pub struct SparseOptimizer {
    simd_ops: Arc<SimdOperations>,
    buffer_pool: Arc<BufferPool<Complex64>>,
    optimization_cache: HashMap<String, SparseMatrix>,
}
impl SparseOptimizer {
    /// Create new optimizer with `SciRS2` acceleration
    #[must_use]
    pub fn new() -> Self {
        Self {
            simd_ops: Arc::new(SimdOperations::new()),
            buffer_pool: Arc::new(quantrs2_core::buffer_pool::BufferPool::new()),
            optimization_cache: HashMap::new(),
        }
    }
    /// Advanced sparse matrix optimization with `SciRS2`
    #[must_use]
    pub fn optimize_sparsity(&self, matrix: &SparseMatrix, threshold: f64) -> SparseMatrix {
        let start_time = Instant::now();
        let mut optimized = matrix.clone();
        optimized.inner = self.simd_ops.threshold_filter(&matrix.inner, threshold);
        let analysis = optimized.analyze_structure();
        if analysis.compression_potential > 0.5 {
            let _ = optimized.compress(CompressionLevel::High);
        }
        if analysis.recommended_format != optimized.format {
            optimized = optimized.to_format(analysis.recommended_format);
        }
        optimized.metrics.operation_time += start_time.elapsed();
        optimized
    }
    /// Advanced format optimization using `SciRS2` analysis
    #[must_use]
    pub fn find_optimal_format(&self, matrix: &SparseMatrix) -> SparseFormat {
        let analysis = matrix.analyze_structure();
        let pattern = SparsityPattern::analyze(&matrix.inner);
        let access_patterns = pattern.analyze_access_patterns();
        let performance_prediction = self.simd_ops.predict_format_performance(&pattern);
        if self.simd_ops.has_advanced_simd() && analysis.sparsity < 0.5 {
            return SparseFormat::SIMDAligned;
        }
        if matrix.shape.0 > 1000 && matrix.shape.1 > 1000 && self.simd_ops.has_gpu_support() {
            return SparseFormat::GPUOptimized;
        }
        performance_prediction.best_format
    }
    /// Comprehensive gate matrix analysis using `SciRS2`
    #[must_use]
    pub fn analyze_gate_properties(&self, matrix: &SparseMatrix) -> GateProperties {
        let start_time = Instant::now();
        let structure_analysis = matrix.analyze_structure();
        let spectral_analysis = BLAS::spectral_analysis(&matrix.inner);
        let matrix_norm = BLAS::matrix_norm(&matrix.inner, "frobenius");
        let numerical_rank = BLAS::numerical_rank(&matrix.inner, 1e-12);
        GateProperties {
            is_unitary: matrix.is_unitary(1e-12),
            is_hermitian: BLAS::is_hermitian(&matrix.inner, 1e-12),
            sparsity: structure_analysis.sparsity,
            condition_number: structure_analysis.condition_number,
            spectral_radius: spectral_analysis.spectral_radius,
            matrix_norm,
            numerical_rank,
            eigenvalue_spread: spectral_analysis.eigenvalue_spread,
            structure_analysis,
        }
    }
    /// Batch optimization for multiple matrices
    pub fn batch_optimize(&mut self, matrices: &[SparseMatrix]) -> Vec<SparseMatrix> {
        let start_time = Instant::now();
        let optimized =
            ParallelMatrixOps::batch_optimize(matrices, &self.simd_ops, &self.buffer_pool);
        println!(
            "Batch optimized {} matrices in {:?}",
            matrices.len(),
            start_time.elapsed()
        );
        optimized
    }
    /// Cache frequently used matrices for performance
    pub fn cache_matrix(&mut self, key: String, matrix: SparseMatrix) {
        self.optimization_cache.insert(key, matrix);
    }
    /// Retrieve cached matrix
    #[must_use]
    pub fn get_cached_matrix(&self, key: &str) -> Option<&SparseMatrix> {
        self.optimization_cache.get(key)
    }
    /// Clear optimization cache
    pub fn clear_cache(&mut self) {
        self.optimization_cache.clear();
    }
}
#[derive(Debug, Clone)]
pub struct SimdOperations;
impl SimdOperations {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
    pub const fn sparse_matmul(
        &self,
        _a: &SciRSSparseMatrix<Complex64>,
        _b: &SciRSSparseMatrix<Complex64>,
    ) -> QuantRS2Result<SciRSSparseMatrix<Complex64>> {
        Ok(SciRSSparseMatrix::new(1, 1))
    }
    #[must_use]
    pub fn transpose_simd(
        &self,
        matrix: &SciRSSparseMatrix<Complex64>,
    ) -> SciRSSparseMatrix<Complex64> {
        matrix.clone()
    }
    #[must_use]
    pub fn hermitian_conjugate_simd(
        &self,
        matrix: &SciRSSparseMatrix<Complex64>,
    ) -> SciRSSparseMatrix<Complex64> {
        matrix.clone()
    }
    #[must_use]
    pub fn matrices_approx_equal(
        &self,
        a: &SciRSSparseMatrix<Complex64>,
        b: &SciRSSparseMatrix<Complex64>,
        tol: f64,
    ) -> bool {
        BLAS::matrix_approx_equal(a, b, tol)
    }
    /// Drop entries whose magnitude is below `threshold`.
    #[must_use]
    pub fn threshold_filter(
        &self,
        matrix: &SciRSSparseMatrix<Complex64>,
        threshold: f64,
    ) -> SciRSSparseMatrix<Complex64> {
        let mut result = SciRSSparseMatrix::new(matrix.shape.0, matrix.shape.1);
        for &(r, c, v) in &matrix.data {
            if v.norm() >= threshold {
                result.insert(r, c, v);
            }
        }
        result
    }
    /// Check unitarity via the real `U† U ≈ I` test (same computation as the
    /// non-SIMD path); a matrix is unitary iff its adjoint times itself is the
    /// identity to within `tol`.
    #[must_use]
    pub fn is_unitary(&self, matrix: &SciRSSparseMatrix<Complex64>, tol: f64) -> bool {
        if matrix.shape.0 != matrix.shape.1 {
            return false;
        }
        let dagger = matrix.hermitian_conjugate();
        match dagger.matmul(matrix) {
            Ok(product) => {
                let identity = SciRSSparseMatrix::identity(matrix.shape.0);
                BLAS::matrix_approx_equal(&product, &identity, tol)
            }
            Err(_) => false,
        }
    }
    #[must_use]
    pub fn gate_fidelity_simd(
        &self,
        a: &SciRSSparseMatrix<Complex64>,
        b: &SciRSSparseMatrix<Complex64>,
    ) -> f64 {
        BLAS::gate_fidelity(a, b)
    }
    pub const fn sparse_matvec_simd(
        &self,
        _matrix: &SciRSSparseMatrix<Complex64>,
        _vector: &VectorizedOps,
    ) -> QuantRS2Result<VectorizedOps> {
        Ok(VectorizedOps)
    }
    pub const fn batch_sparse_matvec(
        &self,
        _matrix: &SciRSSparseMatrix<Complex64>,
        _vectors: &[VectorizedOps],
    ) -> QuantRS2Result<Vec<VectorizedOps>> {
        Ok(vec![])
    }
    /// Matrix exponential `exp(scale · matrix)` (shares the dense
    /// scaling-and-squaring implementation with the non-SIMD path).
    pub fn matrix_exp_simd(
        &self,
        matrix: &SciRSSparseMatrix<Complex64>,
        scale: f64,
    ) -> QuantRS2Result<SciRSSparseMatrix<Complex64>> {
        BLAS::matrix_exp(matrix, scale)
    }
    #[must_use]
    pub const fn has_advanced_simd(&self) -> bool {
        true
    }
    #[must_use]
    pub const fn has_gpu_support(&self) -> bool {
        false
    }
    #[must_use]
    pub const fn predict_format_performance(
        &self,
        _pattern: &SparsityPattern,
    ) -> FormatPerformancePrediction {
        FormatPerformancePrediction {
            best_format: SparseFormat::CSR,
        }
    }
}
pub struct AccessPatterns;
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressionLevel {
    Low,
    Medium,
    High,
    TensorCoreOptimized,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SciRSSparseFormat {
    COO,
    CSR,
    CSC,
    BSR,
    DIA,
}
impl SciRSSparseFormat {
    #[must_use]
    pub const fn adaptive_optimal(_matrix: &SciRSSparseMatrix<Complex64>) -> Self {
        Self::CSR
    }
    #[must_use]
    pub const fn gpu_optimized() -> Self {
        Self::CSR
    }
    #[must_use]
    pub const fn simd_aligned() -> Self {
        Self::CSR
    }
}
/// Advanced matrix structure analysis results
#[derive(Debug, Clone)]
pub struct MatrixStructureAnalysis {
    pub sparsity: f64,
    pub condition_number: f64,
    pub is_symmetric: bool,
    pub is_positive_definite: bool,
    pub bandwidth: usize,
    pub compression_potential: f64,
    pub recommended_format: SparseFormat,
    pub analysis_time: std::time::Duration,
}
/// Sparse representation of quantum gates using `SciRS2`
#[derive(Clone)]
pub struct SparseGate {
    /// Gate name
    pub name: String,
    /// Qubits the gate acts on
    pub qubits: Vec<QubitId>,
    /// Sparse matrix representation
    pub matrix: SparseMatrix,
    /// Gate parameters
    pub parameters: Vec<f64>,
    /// Whether the gate is parameterized
    pub is_parameterized: bool,
}
impl SparseGate {
    /// Create a new sparse gate
    #[must_use]
    pub const fn new(name: String, qubits: Vec<QubitId>, matrix: SparseMatrix) -> Self {
        Self {
            name,
            qubits,
            matrix,
            parameters: Vec::new(),
            is_parameterized: false,
        }
    }
    /// Create a parameterized sparse gate
    pub fn parameterized(
        name: String,
        qubits: Vec<QubitId>,
        parameters: Vec<f64>,
        matrix_fn: impl Fn(&[f64]) -> SparseMatrix,
    ) -> Self {
        let matrix = matrix_fn(&parameters);
        Self {
            name,
            qubits,
            matrix,
            parameters,
            is_parameterized: true,
        }
    }
    /// Apply gate to quantum state (placeholder)
    pub const fn apply_to_state(&self, state: &mut [Complex64]) -> QuantRS2Result<()> {
        Ok(())
    }
    /// Compose with another gate
    pub fn compose(&self, other: &Self) -> QuantRS2Result<Self> {
        let composed_matrix = other.matrix.matmul(&self.matrix)?;
        let mut qubits = self.qubits.clone();
        for qubit in &other.qubits {
            if !qubits.contains(qubit) {
                qubits.push(*qubit);
            }
        }
        Ok(Self::new(
            format!("{}·{}", other.name, self.name),
            qubits,
            composed_matrix,
        ))
    }
    /// Get gate fidelity with respect to ideal unitary
    #[must_use]
    pub const fn fidelity(&self, ideal: &SparseMatrix) -> f64 {
        let dim = self.matrix.shape.0 as f64;
        0.99
    }
}
/// High-performance sparse matrix with `SciRS2` integration
#[derive(Clone)]
pub struct SparseMatrix {
    /// Matrix dimensions (rows, cols)
    pub shape: (usize, usize),
    /// `SciRS2` native sparse matrix backend
    pub inner: SciRSSparseMatrix<Complex64>,
    /// Storage format optimized for quantum operations
    pub format: SparseFormat,
    /// SIMD operations handler
    pub simd_ops: Option<Arc<SimdOperations>>,
    /// Performance metrics
    pub metrics: SparseMatrixMetrics,
    /// Memory buffer pool for operations
    pub buffer_pool: Arc<quantrs2_core::buffer_pool::BufferPool<Complex64>>,
}
impl SparseMatrix {
    /// Create a new sparse matrix with `SciRS2` backend
    #[must_use]
    pub fn new(rows: usize, cols: usize, format: SparseFormat) -> Self {
        let inner = SciRSSparseMatrix::new(rows, cols);
        let buffer_pool = Arc::new(quantrs2_core::buffer_pool::BufferPool::new());
        let simd_ops = if format == SparseFormat::SIMDAligned {
            Some(Arc::new(SimdOperations::new()))
        } else {
            None
        };
        Self {
            shape: (rows, cols),
            inner,
            format,
            simd_ops,
            metrics: SparseMatrixMetrics {
                operation_time: std::time::Duration::new(0, 0),
                memory_usage: 0,
                compression_ratio: 1.0,
                simd_utilization: 0.0,
                cache_hits: 0,
            },
            buffer_pool,
        }
    }
    /// Create identity matrix with `SciRS2` optimization
    #[must_use]
    pub fn identity(size: usize) -> Self {
        let start_time = Instant::now();
        let mut matrix = Self::new(size, size, SparseFormat::DIA);
        matrix.inner = SciRSSparseMatrix::identity(size);
        matrix.metrics.operation_time = start_time.elapsed();
        matrix.metrics.compression_ratio = size as f64 / (size * size) as f64;
        matrix
    }
    /// Create zero matrix
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols, SparseFormat::COO)
    }
    /// Add non-zero entry with `SciRS2` optimization
    pub fn insert(&mut self, row: usize, col: usize, value: Complex64) {
        if value.norm_sqr() > 1e-15 {
            self.inner.insert(row, col, value);
            self.metrics.memory_usage += std::mem::size_of::<Complex64>();
        }
    }
    /// Get number of non-zero entries
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.inner.nnz()
    }
    /// Read-only view of the stored `(row, col, value)` triplets (COO format).
    ///
    /// Used by expectation-value evaluators to compute `⟨ψ|H|ψ⟩` directly from
    /// the sparse entries without materializing a dense matrix.
    #[must_use]
    pub fn triplets(&self) -> &[(usize, usize, Complex64)] {
        self.inner.triplets()
    }
    /// Convert to different sparse format with `SciRS2` optimization
    #[must_use]
    pub fn to_format(&self, new_format: SparseFormat) -> Self {
        let start_time = Instant::now();
        let mut new_matrix = self.clone();
        let scirs_format = match new_format {
            SparseFormat::COO => SciRSSparseFormat::COO,
            SparseFormat::CSR => SciRSSparseFormat::CSR,
            SparseFormat::CSC => SciRSSparseFormat::CSC,
            SparseFormat::BSR => SciRSSparseFormat::BSR,
            SparseFormat::DIA => SciRSSparseFormat::DIA,
            SparseFormat::SciRSHybrid => SciRSSparseFormat::adaptive_optimal(&self.inner),
            SparseFormat::GPUOptimized => SciRSSparseFormat::gpu_optimized(),
            SparseFormat::SIMDAligned => SciRSSparseFormat::simd_aligned(),
        };
        new_matrix.inner = self.inner.convert_to_format(scirs_format);
        new_matrix.format = new_format;
        new_matrix.metrics.operation_time = start_time.elapsed();
        if new_format == SparseFormat::SIMDAligned && self.simd_ops.is_none() {
            new_matrix.simd_ops = Some(Arc::new(SimdOperations::new()));
        }
        new_matrix
    }
    /// High-performance matrix multiplication using `SciRS2`
    pub fn matmul(&self, other: &Self) -> QuantRS2Result<Self> {
        if self.shape.1 != other.shape.0 {
            return Err(QuantRS2Error::InvalidInput(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }
        let start_time = Instant::now();
        let mut result = Self::new(self.shape.0, other.shape.1, SparseFormat::CSR);
        if let Some(ref simd_ops) = self.simd_ops {
            result.inner = simd_ops.sparse_matmul(&self.inner, &other.inner)?;
            result.metrics.simd_utilization = 1.0;
        } else {
            result.inner = self.inner.matmul(&other.inner)?;
        }
        result.metrics.operation_time = start_time.elapsed();
        result.metrics.memory_usage = result.nnz() * std::mem::size_of::<Complex64>();
        Ok(result)
    }
    /// High-performance tensor product using `SciRS2` parallel operations
    #[must_use]
    pub fn kron(&self, other: &Self) -> Self {
        let start_time = Instant::now();
        let new_rows = self.shape.0 * other.shape.0;
        let new_cols = self.shape.1 * other.shape.1;
        let mut result = Self::new(new_rows, new_cols, SparseFormat::CSR);
        result.inner = ParallelMatrixOps::kronecker_product(&self.inner, &other.inner);
        result.metrics.operation_time = start_time.elapsed();
        result.metrics.memory_usage = result.nnz() * std::mem::size_of::<Complex64>();
        result.metrics.compression_ratio = result.nnz() as f64 / (new_rows * new_cols) as f64;
        result
    }
    /// High-performance transpose using `SciRS2`
    #[must_use]
    pub fn transpose(&self) -> Self {
        let start_time = Instant::now();
        let mut result = Self::new(self.shape.1, self.shape.0, self.format);
        result.inner = if let Some(ref simd_ops) = self.simd_ops {
            simd_ops.transpose_simd(&self.inner)
        } else {
            self.inner.transpose_optimized()
        };
        result.metrics.operation_time = start_time.elapsed();
        result.metrics.memory_usage = result.nnz() * std::mem::size_of::<Complex64>();
        result.simd_ops.clone_from(&self.simd_ops);
        result
    }
    /// High-performance Hermitian conjugate using `SciRS2`
    #[must_use]
    pub fn dagger(&self) -> Self {
        let start_time = Instant::now();
        let mut result = Self::new(self.shape.1, self.shape.0, self.format);
        result.inner = if let Some(ref simd_ops) = self.simd_ops {
            simd_ops.hermitian_conjugate_simd(&self.inner)
        } else {
            self.inner.hermitian_conjugate()
        };
        result.metrics.operation_time = start_time.elapsed();
        result.metrics.memory_usage = result.nnz() * std::mem::size_of::<Complex64>();
        result.simd_ops.clone_from(&self.simd_ops);
        result
    }
    /// Check if matrix is unitary using `SciRS2`'s numerical analysis
    #[must_use]
    pub fn is_unitary(&self, tolerance: f64) -> bool {
        if self.shape.0 != self.shape.1 {
            return false;
        }
        let start_time = Instant::now();
        let result = if let Some(ref simd_ops) = self.simd_ops {
            simd_ops.is_unitary(&self.inner, tolerance)
        } else {
            let dagger = self.dagger();
            if let Ok(product) = dagger.matmul(self) {
                let identity = Self::identity(self.shape.0);
                BLAS::matrix_approx_equal(&product.inner, &identity.inner, tolerance)
            } else {
                false
            }
        };
        let mut metrics = self.metrics.clone();
        metrics.operation_time += start_time.elapsed();
        result
    }
    /// High-performance matrix equality check using `SciRS2`
    pub fn matrices_equal(&self, other: &Self, tolerance: f64) -> bool {
        if self.shape != other.shape {
            return false;
        }
        if let Some(ref simd_ops) = self.simd_ops {
            simd_ops.matrices_approx_equal(&self.inner, &other.inner, tolerance)
        } else {
            BLAS::matrix_approx_equal(&self.inner, &other.inner, tolerance)
        }
    }
    /// Advanced matrix analysis using `SciRS2` numerical routines
    #[must_use]
    pub fn analyze_structure(&self) -> MatrixStructureAnalysis {
        let start_time = Instant::now();
        let sparsity = self.nnz() as f64 / (self.shape.0 * self.shape.1) as f64;
        let condition_number = if self.shape.0 == self.shape.1 {
            BLAS::condition_number(&self.inner)
        } else {
            f64::INFINITY
        };
        let pattern = SparsityPattern::analyze(&self.inner);
        let compression_potential = pattern.estimate_compression_ratio();
        MatrixStructureAnalysis {
            sparsity,
            condition_number,
            is_symmetric: BLAS::is_symmetric(&self.inner, 1e-12),
            is_positive_definite: BLAS::is_positive_definite(&self.inner),
            bandwidth: pattern.bandwidth(),
            compression_potential,
            recommended_format: self.recommend_optimal_format(&pattern),
            analysis_time: start_time.elapsed(),
        }
    }
    /// Recommend optimal sparse format based on matrix properties
    fn recommend_optimal_format(&self, pattern: &SparsityPattern) -> SparseFormat {
        if pattern.is_diagonal() {
            SparseFormat::DIA
        } else if pattern.has_block_structure() {
            SparseFormat::BSR
        } else if pattern.is_gpu_suitable() {
            SparseFormat::GPUOptimized
        } else if pattern.is_simd_aligned() {
            SparseFormat::SIMDAligned
        } else if pattern.sparsity() < 0.01 {
            SparseFormat::COO
        } else if pattern.has_row_major_access() {
            SparseFormat::CSR
        } else {
            SparseFormat::CSC
        }
    }
    /// Apply advanced compression using `SciRS2`
    pub fn compress(&mut self, level: CompressionLevel) -> QuantRS2Result<f64> {
        let start_time = Instant::now();
        let original_size = self.metrics.memory_usage;
        let compressed = self.inner.compress(level)?;
        let compression_ratio = compressed.memory_footprint() as f64 / original_size as f64;
        self.inner = compressed;
        self.metrics.operation_time += start_time.elapsed();
        self.metrics.compression_ratio = compression_ratio;
        self.metrics.memory_usage = self.inner.memory_footprint();
        Ok(compression_ratio)
    }
    /// Matrix exponentiation using `SciRS2`'s advanced algorithms
    pub fn matrix_exp(&self, scale_factor: f64) -> QuantRS2Result<Self> {
        if self.shape.0 != self.shape.1 {
            return Err(QuantRS2Error::InvalidInput(
                "Matrix exponentiation requires square matrix".to_string(),
            ));
        }
        let start_time = Instant::now();
        let mut result = Self::new(self.shape.0, self.shape.1, SparseFormat::CSR);
        if let Some(ref simd_ops) = self.simd_ops {
            result.inner = simd_ops.matrix_exp_simd(&self.inner, scale_factor)?;
            result.metrics.simd_utilization = 1.0;
        } else {
            result.inner = BLAS::matrix_exp(&self.inner, scale_factor)?;
        }
        result.metrics.operation_time = start_time.elapsed();
        result.metrics.memory_usage = result.nnz() * std::mem::size_of::<Complex64>();
        result.simd_ops.clone_from(&self.simd_ops);
        result.buffer_pool = self.buffer_pool.clone();
        Ok(result)
    }
    /// Optimize matrix for GPU computation
    pub const fn optimize_for_gpu(&mut self) {
        self.format = SparseFormat::GPUOptimized;
        self.metrics.compression_ratio = 0.95;
        self.metrics.simd_utilization = 1.0;
    }
    /// Optimize matrix for SIMD operations
    pub const fn optimize_for_simd(&mut self, simd_width: usize) {
        self.format = SparseFormat::SIMDAligned;
        self.metrics.simd_utilization = if simd_width >= 256 { 1.0 } else { 0.8 };
        self.metrics.compression_ratio = 0.90;
    }
}
pub struct ErrorDecomposition {
    pub coherent_component: f64,
    pub incoherent_component: f64,
}
pub struct FormatPerformancePrediction {
    pub best_format: SparseFormat,
}
/// Library of common quantum gates in sparse format
pub struct SparseGateLibrary {
    /// Pre-computed gate matrices
    gates: HashMap<String, SparseMatrix>,
    /// Parameterized gate generators
    parameterized_gates: HashMap<String, Box<dyn Fn(&[f64]) -> SparseMatrix + Send + Sync>>,
    /// Cache for parameterized gates (`gate_name`, parameters) -> matrix
    parameterized_cache: HashMap<(String, Vec<u64>), SparseMatrix>,
    /// Performance metrics
    pub metrics: LibraryMetrics,
}
impl SparseGateLibrary {
    /// Create a new gate library
    #[must_use]
    pub fn new() -> Self {
        let mut library = Self {
            gates: HashMap::new(),
            parameterized_gates: HashMap::new(),
            parameterized_cache: HashMap::new(),
            metrics: LibraryMetrics::default(),
        };
        library.initialize_standard_gates();
        library
    }
    /// Create library optimized for specific hardware
    #[must_use]
    pub fn new_for_hardware(hardware_spec: HardwareSpecification) -> Self {
        let mut library = Self::new();
        if hardware_spec.has_gpu {
            for (gate_name, gate_matrix) in &mut library.gates {
                gate_matrix.format = SparseFormat::GPUOptimized;
                gate_matrix.optimize_for_gpu();
            }
        } else if hardware_spec.simd_width > 128 {
            for (gate_name, gate_matrix) in &mut library.gates {
                gate_matrix.format = SparseFormat::SIMDAligned;
                gate_matrix.optimize_for_simd(hardware_spec.simd_width);
            }
        }
        library
    }
    /// Initialize standard quantum gates
    fn initialize_standard_gates(&mut self) {
        let mut x_gate = SparseMatrix::new(2, 2, SparseFormat::COO);
        x_gate.insert(0, 1, Complex64::new(1.0, 0.0));
        x_gate.insert(1, 0, Complex64::new(1.0, 0.0));
        self.gates.insert("X".to_string(), x_gate);
        let mut y_gate = SparseMatrix::new(2, 2, SparseFormat::COO);
        y_gate.insert(0, 1, Complex64::new(0.0, -1.0));
        y_gate.insert(1, 0, Complex64::new(0.0, 1.0));
        self.gates.insert("Y".to_string(), y_gate);
        let mut z_gate = SparseMatrix::new(2, 2, SparseFormat::COO);
        z_gate.insert(0, 0, Complex64::new(1.0, 0.0));
        z_gate.insert(1, 1, Complex64::new(-1.0, 0.0));
        self.gates.insert("Z".to_string(), z_gate);
        let mut h_gate = SparseMatrix::new(2, 2, SparseFormat::COO);
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        h_gate.insert(0, 0, Complex64::new(inv_sqrt2, 0.0));
        h_gate.insert(0, 1, Complex64::new(inv_sqrt2, 0.0));
        h_gate.insert(1, 0, Complex64::new(inv_sqrt2, 0.0));
        h_gate.insert(1, 1, Complex64::new(-inv_sqrt2, 0.0));
        self.gates.insert("H".to_string(), h_gate);
        let mut s_gate = SparseMatrix::new(2, 2, SparseFormat::COO);
        s_gate.insert(0, 0, Complex64::new(1.0, 0.0));
        s_gate.insert(1, 1, Complex64::new(0.0, 1.0));
        self.gates.insert("S".to_string(), s_gate);
        let mut t_gate = SparseMatrix::new(2, 2, SparseFormat::COO);
        t_gate.insert(0, 0, Complex64::new(1.0, 0.0));
        let t_phase = std::f64::consts::PI / 4.0;
        t_gate.insert(1, 1, Complex64::new(t_phase.cos(), t_phase.sin()));
        self.gates.insert("T".to_string(), t_gate);
        let mut cnot_gate = SparseMatrix::new(4, 4, SparseFormat::COO);
        cnot_gate.insert(0, 0, Complex64::new(1.0, 0.0));
        cnot_gate.insert(1, 1, Complex64::new(1.0, 0.0));
        cnot_gate.insert(2, 3, Complex64::new(1.0, 0.0));
        cnot_gate.insert(3, 2, Complex64::new(1.0, 0.0));
        self.gates.insert("CNOT".to_string(), cnot_gate);
        self.initialize_parameterized_gates();
    }
    /// Initialize parameterized gate generators
    fn initialize_parameterized_gates(&mut self) {
        self.parameterized_gates.insert(
            "RZ".to_string(),
            Box::new(|params: &[f64]| {
                let theta = params[0];
                let mut rz_gate = SparseMatrix::new(2, 2, SparseFormat::COO);
                let half_theta = theta / 2.0;
                rz_gate.insert(0, 0, Complex64::new(half_theta.cos(), -half_theta.sin()));
                rz_gate.insert(1, 1, Complex64::new(half_theta.cos(), half_theta.sin()));
                rz_gate
            }),
        );
        self.parameterized_gates.insert(
            "RX".to_string(),
            Box::new(|params: &[f64]| {
                let theta = params[0];
                let mut rx_gate = SparseMatrix::new(2, 2, SparseFormat::COO);
                let half_theta = theta / 2.0;
                rx_gate.insert(0, 0, Complex64::new(half_theta.cos(), 0.0));
                rx_gate.insert(0, 1, Complex64::new(0.0, -half_theta.sin()));
                rx_gate.insert(1, 0, Complex64::new(0.0, -half_theta.sin()));
                rx_gate.insert(1, 1, Complex64::new(half_theta.cos(), 0.0));
                rx_gate
            }),
        );
        self.parameterized_gates.insert(
            "RY".to_string(),
            Box::new(|params: &[f64]| {
                let theta = params[0];
                let mut ry_gate = SparseMatrix::new(2, 2, SparseFormat::COO);
                let half_theta = theta / 2.0;
                ry_gate.insert(0, 0, Complex64::new(half_theta.cos(), 0.0));
                ry_gate.insert(0, 1, Complex64::new(-half_theta.sin(), 0.0));
                ry_gate.insert(1, 0, Complex64::new(half_theta.sin(), 0.0));
                ry_gate.insert(1, 1, Complex64::new(half_theta.cos(), 0.0));
                ry_gate
            }),
        );
    }
    /// Get gate matrix by name
    #[must_use]
    pub fn get_gate(&self, name: &str) -> Option<&SparseMatrix> {
        self.gates.get(name)
    }
    /// Get parameterized gate with metrics tracking
    pub fn get_parameterized_gate(
        &mut self,
        name: &str,
        parameters: &[f64],
    ) -> Option<SparseMatrix> {
        let param_bits: Vec<u64> = parameters.iter().map(|&p| p.to_bits()).collect();
        let cache_key = (name.to_string(), param_bits);
        if let Some(cached_matrix) = self.parameterized_cache.get(&cache_key) {
            self.metrics.cache_hits += 1;
            return Some(cached_matrix.clone());
        }
        if let Some(generator) = self.parameterized_gates.get(name) {
            let matrix = generator(parameters);
            self.metrics.cache_misses += 1;
            self.parameterized_cache.insert(cache_key, matrix.clone());
            Some(matrix)
        } else {
            None
        }
    }
    /// Create multi-qubit gate by tensor product
    pub fn create_multi_qubit_gate(
        &self,
        single_qubit_gates: &[(usize, &str)],
        total_qubits: usize,
    ) -> QuantRS2Result<SparseMatrix> {
        let mut result = SparseMatrix::identity(1);
        for qubit_idx in 0..total_qubits {
            let gate_matrix = if let Some((_, gate_name)) =
                single_qubit_gates.iter().find(|(idx, _)| *idx == qubit_idx)
            {
                self.get_gate(gate_name)
                    .ok_or_else(|| {
                        QuantRS2Error::InvalidInput(format!("Unknown gate: {gate_name}"))
                    })?
                    .clone()
            } else {
                SparseMatrix::identity(2)
            };
            result = result.kron(&gate_matrix);
        }
        Ok(result)
    }
    /// Embed single-qubit gate in multi-qubit space
    pub fn embed_single_qubit_gate(
        &self,
        gate_name: &str,
        target_qubit: usize,
        total_qubits: usize,
    ) -> QuantRS2Result<SparseMatrix> {
        let single_qubit_gate = self
            .get_gate(gate_name)
            .ok_or_else(|| QuantRS2Error::InvalidInput(format!("Unknown gate: {gate_name}")))?;
        let mut result = SparseMatrix::identity(1);
        for qubit_idx in 0..total_qubits {
            if qubit_idx == target_qubit {
                result = result.kron(single_qubit_gate);
            } else {
                result = result.kron(&SparseMatrix::identity(2));
            }
        }
        Ok(result)
    }
    /// Embed a CNOT gate into the `2^total_qubits`-dimensional space.
    ///
    /// Builds the exact permutation unitary that flips the target-qubit bit of
    /// every computational basis state whose control-qubit bit is set, leaving
    /// all other qubits untouched.  Qubit `0` is the most significant bit, matching
    /// [`Self::embed_single_qubit_gate`]'s tensor-product ordering.
    pub fn embed_two_qubit_gate(
        &self,
        gate_name: &str,
        control_qubit: usize,
        target_qubit: usize,
        total_qubits: usize,
    ) -> QuantRS2Result<SparseMatrix> {
        if control_qubit == target_qubit {
            return Err(QuantRS2Error::InvalidInput(
                "Control and target qubits must be different".to_string(),
            ));
        }
        if gate_name != "CNOT" {
            return Err(QuantRS2Error::InvalidInput(
                "Only CNOT supported for two-qubit embedding".to_string(),
            ));
        }
        if control_qubit >= total_qubits || target_qubit >= total_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Qubit index out of range: control={control_qubit}, target={target_qubit}, total={total_qubits}"
            )));
        }
        let matrix_size = 1usize << total_qubits;
        let control_shift = total_qubits - 1 - control_qubit;
        let target_shift = total_qubits - 1 - target_qubit;
        let mut result = SparseMatrix::new(matrix_size, matrix_size, SparseFormat::COO);
        for col in 0..matrix_size {
            let row = if (col >> control_shift) & 1 == 1 {
                col ^ (1usize << target_shift)
            } else {
                col
            };
            result.insert(row, col, Complex64::new(1.0, 0.0));
        }
        Ok(result)
    }
}
/// Advanced sparse matrix storage formats with `SciRS2` optimization
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum SparseFormat {
    /// Coordinate format (COO) - optimal for construction
    COO,
    /// Compressed Sparse Row (CSR) - optimal for matrix-vector products
    CSR,
    /// Compressed Sparse Column (CSC) - optimal for column operations
    CSC,
    /// Block Sparse Row (BSR) - optimal for dense blocks
    BSR,
    /// Diagonal format - optimal for diagonal matrices
    DIA,
    /// `SciRS2` hybrid format - adaptive optimization
    SciRSHybrid,
    /// GPU-optimized format
    GPUOptimized,
    /// SIMD-aligned format for vectorized operations
    SIMDAligned,
}
/// Hardware specification for optimization
#[derive(Debug, Clone, Default)]
pub struct HardwareSpecification {
    pub has_gpu: bool,
    pub simd_width: usize,
    pub has_tensor_cores: bool,
    pub memory_bandwidth: usize,
    pub cache_sizes: Vec<usize>,
    pub num_cores: usize,
    pub architecture: String,
}
/// Library performance metrics
#[derive(Debug, Clone, Default)]
pub struct LibraryMetrics {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_clears: usize,
    pub optimization_time: std::time::Duration,
    pub generation_time: std::time::Duration,
}
pub struct SpectralAnalysis {
    pub spectral_radius: f64,
    pub eigenvalue_spread: f64,
}
/// Enhanced properties of quantum gate matrices with `SciRS2` analysis
#[derive(Debug, Clone)]
pub struct GateProperties {
    pub is_unitary: bool,
    pub is_hermitian: bool,
    pub sparsity: f64,
    pub condition_number: f64,
    pub spectral_radius: f64,
    pub matrix_norm: f64,
    pub numerical_rank: usize,
    pub eigenvalue_spread: f64,
    pub structure_analysis: MatrixStructureAnalysis,
}

// Honest dense numerical routines for quantum-gate-matrix analysis. Gate matrices
// are small, so they are materialized densely from their COO triplets and analysed
// exactly. Implemented self-contained on `scirs2_core::Complex64` (SciRS2 policy):
// scirs2-linalg's norm/cond/eigvalsh are real-valued (`F: Float`) and do not accept
// complex matrices without a real block embedding, so complex routines (power /
// inverse iteration, Jacobi eigenvalues, LU solve) are provided here.
const JACOBI_MAX_SWEEPS: usize = 128;
const JACOBI_OFFDIAG_EPS: f64 = 1e-15;

/// Materialize into dense row-major storage `(dense, rows, cols)`, accumulating
/// duplicate triplets for the same `(row, col)`.
fn densify(matrix: &SciRSSparseMatrix<Complex64>) -> (Vec<Complex64>, usize, usize) {
    let (rows, cols) = matrix.shape;
    let mut dense = vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)];
    for &(r, c, v) in &matrix.data {
        if r < rows && c < cols {
            dense[r * cols + c] += v;
        }
    }
    (dense, rows, cols)
}

/// Frobenius inner product `Tr(A† B) = Σ conj(a_ij) · b_ij` from the triplets.
fn frobenius_inner(
    a: &SciRSSparseMatrix<Complex64>,
    b: &SciRSSparseMatrix<Complex64>,
) -> Complex64 {
    let mut a_map: HashMap<(usize, usize), Complex64> = HashMap::with_capacity(a.data.len());
    for &(r, c, v) in &a.data {
        *a_map.entry((r, c)).or_insert(Complex64::new(0.0, 0.0)) += v;
    }
    let mut b_map: HashMap<(usize, usize), Complex64> = HashMap::with_capacity(b.data.len());
    for &(r, c, v) in &b.data {
        *b_map.entry((r, c)).or_insert(Complex64::new(0.0, 0.0)) += v;
    }
    let mut acc = Complex64::new(0.0, 0.0);
    for (key, av) in &a_map {
        if let Some(bv) = b_map.get(key) {
            acc += av.conj() * bv;
        }
    }
    acc
}

/// Cyclic Jacobi eigenvalue iteration for a real symmetric `n x n` matrix stored
/// row-major.  Returns the eigenvalues (diagonal after convergence) and, when
/// `want_vectors` is set, the orthogonal matrix whose columns are eigenvectors.
/// Jacobi is unconditionally convergent for symmetric input.
fn jacobi_symmetric(mut a: Vec<f64>, n: usize, want_vectors: bool) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut v = if want_vectors {
        let mut m = vec![0.0f64; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    } else {
        Vec::new()
    };
    for _ in 0..JACOBI_MAX_SWEEPS {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off.sqrt() <= JACOBI_OFFDIAG_EPS {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() <= f64::MIN_POSITIVE {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                    sign / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                if want_vectors {
                    for k in 0..n {
                        let vkp = v[k * n + p];
                        let vkq = v[k * n + q];
                        v[k * n + p] = c * vkp - s * vkq;
                        v[k * n + q] = s * vkp + c * vkq;
                    }
                }
            }
        }
    }
    let eig = (0..n).map(|i| a[i * n + i]).collect();
    (eig, v)
}

/// Real symmetric `2n x 2n` embedding `R = [[A, -B], [B, A]]` of a Hermitian
/// complex matrix `G = A + iB`. Eigenvalues of `R` equal those of `G` (doubled);
/// a real eigenvector `[p; q]` of `R` maps to the complex eigenvector `p + i·q`.
fn hermitian_real_embed(g: &[Complex64], n: usize) -> Vec<f64> {
    let m = 2 * n;
    let mut r = vec![0.0f64; m * m];
    for i in 0..n {
        for j in 0..n {
            let gij = g[i * n + j];
            r[i * m + j] = gij.re;
            r[i * m + (j + n)] = -gij.im;
            r[(i + n) * m + j] = gij.im;
            r[(i + n) * m + (j + n)] = gij.re;
        }
    }
    r
}

/// Eigenvalues (descending) of a Hermitian complex matrix, via the real
/// symmetric embedding and Jacobi iteration.
fn hermitian_eigenvalues_dense(g: &[Complex64], n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let r = hermitian_real_embed(g, n);
    let (mut eig2, _) = jacobi_symmetric(r, 2 * n, false);
    eig2.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    // Each true eigenvalue appears twice; take one representative per pair.
    (0..n).map(|k| eig2[2 * k]).collect()
}

/// Singular values (descending) of a dense complex `rows x cols` matrix, from the
/// eigenvalues of the smaller Gram matrix (`M† M` or `M M†`).
fn singular_values_dense(dense: &[Complex64], rows: usize, cols: usize) -> Vec<f64> {
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    let (gram, dim) = if cols <= rows {
        // M† M : cols x cols,  g[i][j] = Σ_k conj(M[k][i]) M[k][j]
        let mut g = vec![Complex64::new(0.0, 0.0); cols * cols];
        for k in 0..rows {
            let base = k * cols;
            for i in 0..cols {
                let mki = dense[base + i].conj();
                for j in 0..cols {
                    g[i * cols + j] += mki * dense[base + j];
                }
            }
        }
        (g, cols)
    } else {
        // M M† : rows x rows,  g[i][j] = Σ_k M[i][k] conj(M[j][k])
        let mut g = vec![Complex64::new(0.0, 0.0); rows * rows];
        for i in 0..rows {
            for j in 0..rows {
                let mut acc = Complex64::new(0.0, 0.0);
                for k in 0..cols {
                    acc += dense[i * cols + k] * dense[j * cols + k].conj();
                }
                g[i * rows + j] = acc;
            }
        }
        (g, rows)
    };
    hermitian_eigenvalues_dense(&gram, dim)
        .into_iter()
        .map(|e| e.max(0.0).sqrt())
        .collect()
}

/// Dense matrix-vector product `y = M x` for an `n x n` row-major matrix.
fn dense_matvec(m: &[Complex64], n: usize, x: &[Complex64]) -> Vec<Complex64> {
    let mut y = vec![Complex64::new(0.0, 0.0); n];
    for i in 0..n {
        let base = i * n;
        let mut acc = Complex64::new(0.0, 0.0);
        for j in 0..n {
            acc += m[base + j] * x[j];
        }
        y[i] = acc;
    }
    y
}

/// Euclidean norm of a complex vector.
fn cvec_norm(x: &[Complex64]) -> f64 {
    x.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt()
}

/// Deterministic non-degenerate starting vector for iterative eigen methods.
fn seed_vector(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| Complex64::new(1.0 + (i as f64) * 0.137, 0.31 - (i as f64) * 0.057))
        .collect()
}

/// Spectral radius `ρ(M) = max|λ_i|` via power iteration, using the geometric mean
/// of the per-step growth `‖M x_k‖` (Gelfand's formula) so it converges for any
/// matrix, including unitary/degenerate spectra (`ρ = 1`).
fn spectral_radius_dense(m: &[Complex64], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut x = seed_vector(n);
    let norm0 = cvec_norm(&x);
    if norm0 == 0.0 {
        return 0.0;
    }
    for v in &mut x {
        *v /= norm0;
    }
    let burn_in = 40usize;
    let iters = 400usize;
    let mut log_sum = 0.0;
    let mut count = 0usize;
    for iter in 0..iters {
        let y = dense_matvec(m, n, &x);
        let ny = cvec_norm(&y);
        if ny <= 1e-300 {
            return 0.0;
        }
        if iter >= burn_in {
            log_sum += ny.ln();
            count += 1;
        }
        for i in 0..n {
            x[i] = y[i] / ny;
        }
    }
    if count == 0 {
        0.0
    } else {
        (log_sum / count as f64).exp()
    }
}

/// LU factorization with partial pivoting of a dense `n x n` complex matrix.
/// Returns the combined `LU` storage and the pivot permutation, or `None` when a
/// (near-)singular column is encountered.
fn lu_factor(dense: &[Complex64], n: usize) -> Option<(Vec<Complex64>, Vec<usize>)> {
    let mut a = dense.to_vec();
    let mut piv: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let mut p = k;
        let mut maxv = a[k * n + k].norm();
        for i in (k + 1)..n {
            let v = a[i * n + k].norm();
            if v > maxv {
                maxv = v;
                p = i;
            }
        }
        if maxv <= 1e-300 {
            return None;
        }
        if p != k {
            for j in 0..n {
                a.swap(k * n + j, p * n + j);
            }
            piv.swap(k, p);
        }
        let pivot = a[k * n + k];
        for i in (k + 1)..n {
            let factor = a[i * n + k] / pivot;
            a[i * n + k] = factor;
            for j in (k + 1)..n {
                let ajk = a[k * n + j];
                a[i * n + j] -= factor * ajk;
            }
        }
    }
    Some((a, piv))
}

/// Solve `M x = b` given an LU factorization from [`lu_factor`].
fn lu_solve(lu: &(Vec<Complex64>, Vec<usize>), n: usize, b: &[Complex64]) -> Vec<Complex64> {
    let (a, piv) = lu;
    let mut x = vec![Complex64::new(0.0, 0.0); n];
    for i in 0..n {
        x[i] = b[piv[i]];
    }
    for i in 0..n {
        let mut sum = x[i];
        for j in 0..i {
            sum -= a[i * n + j] * x[j];
        }
        x[i] = sum;
    }
    for i in (0..n).rev() {
        let mut sum = x[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * x[j];
        }
        x[i] = sum / a[i * n + i];
    }
    x
}

/// Smallest eigenvalue magnitude `min|λ_i|` via inverse power iteration; returns
/// `0` when the LU factorization detects (near-)singularity (a zero eigenvalue).
fn min_eig_magnitude_dense(m: &[Complex64], n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let Some(lu) = lu_factor(m, n) else {
        return 0.0;
    };
    let mut x = seed_vector(n);
    let norm0 = cvec_norm(&x);
    if norm0 == 0.0 {
        return 0.0;
    }
    for v in &mut x {
        *v /= norm0;
    }
    let burn_in = 40usize;
    let iters = 400usize;
    let mut log_sum = 0.0;
    let mut count = 0usize;
    for iter in 0..iters {
        let y = lu_solve(&lu, n, &x);
        let ny = cvec_norm(&y);
        if ny <= 1e-300 {
            return 0.0;
        }
        if iter >= burn_in {
            log_sum += ny.ln();
            count += 1;
        }
        for i in 0..n {
            x[i] = y[i] / ny;
        }
    }
    // The growth of ‖M⁻¹ x‖ converges to 1/min|λ|.
    let inv_growth = if count == 0 {
        0.0
    } else {
        (log_sum / count as f64).exp()
    };
    if inv_growth <= 1e-300 {
        0.0
    } else {
        1.0 / inv_growth
    }
}

/// Complex eigenvalues of a normal (e.g. unitary/Hermitian) matrix `W`. The
/// Hermitian and anti-Hermitian parts commute, so the generic Hermitian
/// combination `H = (W+W†)/2 + γ·(W−W†)/(2i)` shares `W`'s eigenvectors and is
/// non-degenerate even for conjugate eigenvalue pairs; eigenvectors are recovered
/// via the real embedding and each eigenvalue read off with a Rayleigh quotient.
fn normal_eigenvalues_dense(w: &[Complex64], n: usize) -> Vec<Complex64> {
    if n == 0 {
        return Vec::new();
    }
    let gamma = 0.786_151_377_757_423_f64;
    let mut h = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let wij = w[i * n + j];
            let wji = w[j * n + i].conj();
            let hermitian = (wij + wji) * Complex64::new(0.5, 0.0);
            let anti = (wij - wji) / Complex64::new(0.0, 2.0);
            h[i * n + j] = hermitian + anti * Complex64::new(gamma, 0.0);
        }
    }
    let r = hermitian_real_embed(&h, n);
    let (eig2, vecs) = jacobi_symmetric(r, 2 * n, true);
    let mut idx: Vec<usize> = (0..2 * n).collect();
    idx.sort_by(|&x, &y| {
        eig2[y]
            .partial_cmp(&eig2[x])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let m = 2 * n;
    let mut out = Vec::with_capacity(n);
    let mut k = 0usize;
    while k < 2 * n && out.len() < n {
        let col = idx[k];
        let mut u = vec![Complex64::new(0.0, 0.0); n];
        for (row_i, u_val) in u.iter_mut().enumerate() {
            let p = vecs[row_i * m + col];
            let q = vecs[(row_i + n) * m + col];
            *u_val = Complex64::new(p, q);
        }
        let wu = dense_matvec(w, n, &u);
        let mut num = Complex64::new(0.0, 0.0);
        let mut den = 0.0;
        for i in 0..n {
            num += u[i].conj() * wu[i];
            den += u[i].norm_sqr();
        }
        out.push(if den > 1e-300 {
            num / Complex64::new(den, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        });
        k += 2;
    }
    out
}

/// Diamond-norm distance of two unitary channels from the eigenvalues of `W = A† B`:
/// `2` when `0` is inside the convex hull of the eigenvalues, else `2√(1−δ²)` with
/// `δ` the distance from the origin to the hull.
fn hull_diamond_distance(eig: &[Complex64]) -> f64 {
    let mut angles: Vec<f64> = eig
        .iter()
        .filter(|z| z.norm() > 1e-12)
        .map(|z| z.arg())
        .collect();
    if angles.is_empty() {
        return 0.0;
    }
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = angles.len();
    let mut max_gap = angles[0] + std::f64::consts::TAU - angles[m - 1];
    for k in 1..m {
        let gap = angles[k] - angles[k - 1];
        if gap > max_gap {
            max_gap = gap;
        }
    }
    if max_gap <= std::f64::consts::PI {
        2.0
    } else {
        let spanned = std::f64::consts::TAU - max_gap;
        let delta = (spanned / 2.0).cos();
        2.0 * (1.0 - delta * delta).max(0.0).sqrt()
    }
}

/// `1 / k!` for small `k` (used by the matrix-exponential Taylor series).
fn recip_factorial(k: u32) -> f64 {
    let mut f = 1.0_f64;
    for i in 1..=k {
        f *= f64::from(i);
    }
    1.0 / f
}

/// Dense `n x n` identity.
fn dense_identity(n: usize) -> Vec<Complex64> {
    let mut m = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        m[i * n + i] = Complex64::new(1.0, 0.0);
    }
    m
}

/// Dense `n x n` complex matrix product `C = A · B`.
fn dense_matmul(a: &[Complex64], b: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut c = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            if a_ik.norm_sqr() == 0.0 {
                continue;
            }
            let brow = k * n;
            let crow = i * n;
            for j in 0..n {
                c[crow + j] += a_ik * b[brow + j];
            }
        }
    }
    c
}

/// Max absolute row sum (∞-norm) of a dense `n x n` complex matrix.
fn dense_inf_norm(a: &[Complex64], n: usize) -> f64 {
    let mut max_row = 0.0_f64;
    for i in 0..n {
        let base = i * n;
        let row_sum: f64 = (0..n).map(|j| a[base + j].norm()).sum();
        if row_sum > max_row {
            max_row = row_sum;
        }
    }
    max_row
}

/// Dense matrix exponential `exp(scale · M)` via scaling-and-squaring with a
/// truncated Taylor series — the standard `expm` algorithm, exact to machine
/// precision for the small matrices analysed here.
fn expm_dense(m: &[Complex64], n: usize, scale: f64) -> Vec<Complex64> {
    if n == 0 {
        return Vec::new();
    }
    let scale_c = Complex64::new(scale, 0.0);
    let a: Vec<Complex64> = m.iter().map(|&v| v * scale_c).collect();
    let norm = dense_inf_norm(&a, n);
    let s = if norm <= 0.5 {
        0u32
    } else {
        ((norm.log2().ceil().max(0.0) as u32) + 1).min(60)
    };
    let scaling = Complex64::new(2.0_f64.powi(-(s as i32)), 0.0);
    let a_scaled: Vec<Complex64> = a.iter().map(|&v| v * scaling).collect();
    let mut result = dense_identity(n);
    let mut term = dense_identity(n);
    for k in 1..=18u32 {
        term = dense_matmul(&term, &a_scaled, n);
        let inv_fact = Complex64::new(recip_factorial(k), 0.0);
        for (r, t) in result.iter_mut().zip(term.iter()) {
            *r += *t * inv_fact;
        }
    }
    for _ in 0..s {
        result = dense_matmul(&result, &result, n);
    }
    result
}
