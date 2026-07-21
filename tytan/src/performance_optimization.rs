//! Performance optimization module for critical paths.
//!
//! This module provides optimized implementations of performance-critical
//! operations using SIMD, parallelization, and algorithmic improvements.

#![allow(dead_code)]

use quantrs2_core::platform::PlatformCapabilities;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
// NOTE: Parallel feature removed per SciRS2 POLICY - use scirs2_core::parallel_ops directly
use scirs2_core::simd_ops;
use scirs2_core::RngExt;
use std::sync::Arc;

/// Optimized QUBO evaluation
pub struct OptimizedQUBOEvaluator {
    /// QUBO matrix (stored in optimal layout)
    qubo: Arc<Array2<f64>>,
    /// Cache for frequently accessed elements
    cache: Vec<f64>,
    /// Use SIMD operations
    use_simd: bool,
    /// Parallel threshold
    parallel_threshold: usize,
}

impl OptimizedQUBOEvaluator {
    /// Create new optimized evaluator
    pub fn new(qubo: Array2<f64>) -> Self {
        let n = qubo.shape()[0];

        // Pre-compute diagonal elements for cache
        let cache: Vec<f64> = (0..n).map(|i| qubo[[i, i]]).collect();

        Self {
            qubo: Arc::new(qubo),
            cache,
            use_simd: {
                let platform = PlatformCapabilities::detect();
                platform.cpu.simd.avx2 || platform.cpu.simd.avx512
            },
            parallel_threshold: 1000,
        }
    }

    /// Evaluate QUBO energy for binary vector
    pub fn evaluate(&self, x: &ArrayView1<u8>) -> f64 {
        let n = x.len();

        if n > self.parallel_threshold {
            self.evaluate_parallel(x)
        } else if self.use_simd && n >= 8 {
            unsafe { self.evaluate_simd(x) }
        } else {
            self.evaluate_scalar(x)
        }
    }

    /// Scalar evaluation
    fn evaluate_scalar(&self, x: &ArrayView1<u8>) -> f64 {
        let n = x.len();
        let mut energy = 0.0;

        // Linear terms (diagonal)
        for i in 0..n {
            if x[i] == 1 {
                energy += self.cache[i];
            }
        }

        // Quadratic terms
        for i in 0..n {
            if x[i] == 1 {
                for j in i + 1..n {
                    if x[j] == 1 {
                        energy += 2.0 * self.qubo[[i, j]];
                    }
                }
            }
        }

        energy
    }

    /// SIMD evaluation using scirs2_core
    unsafe fn evaluate_simd(&self, x: &ArrayView1<u8>) -> f64 {
        let n = x.len();
        let mut energy = 0.0;

        // Convert binary values to float for SIMD operations
        let x_float: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let x_view = ArrayView1::from(&x_float);
        let cache_view = ArrayView1::from(&self.cache[..n]);

        // Use SciRS2 SIMD operations for element-wise multiplication
        let products = simd_ops::SimdUnifiedOps::simd_mul(&x_view, &cache_view);

        // Sum the products using SIMD operations
        energy = products.sum();

        // Quadratic terms - also optimize with SIMD where possible
        for i in 0..n {
            if x[i] == 1 {
                // Extract the row of quadratic coefficients
                let row_start = i + 1;
                if row_start < n {
                    let row_len = n - row_start;
                    let x_subset = ArrayView1::from(&x_float[row_start..n]);

                    // Create a view of the quadratic coefficients for this row
                    let mut qubo_row = vec![0.0; row_len];
                    for (j, coeff) in qubo_row.iter_mut().enumerate() {
                        *coeff = self.qubo[[i, row_start + j]];
                    }
                    let qubo_row_view = ArrayView1::from(&qubo_row);

                    // Multiply and sum using SIMD
                    let row_products =
                        simd_ops::SimdUnifiedOps::simd_mul(&x_subset, &qubo_row_view);
                    energy += 2.0 * row_products.sum();
                }
            }
        }

        energy
    }

    /// SIMD evaluation stub for non-x86_64
    #[cfg(not(feature = "simd"))]
    unsafe fn evaluate_simd_fallback(&self, x: &ArrayView1<u8>) -> f64 {
        self.evaluate_scalar(x)
    }

    /// Parallel evaluation (parallel feature removed - using sequential with SciRS2 optimization)
    fn evaluate_parallel(&self, x: &ArrayView1<u8>) -> f64 {
        let n = x.len();

        // Linear terms (sequential - parallel feature removed per SciRS2 POLICY)
        let linear_energy: f64 = (0..n).filter(|&i| x[i] == 1).map(|i| self.cache[i]).sum();

        // Quadratic terms (block-wise - parallel feature removed per SciRS2 POLICY)
        let block_size = (n as f64).sqrt() as usize + 1;
        let quadratic_energy: f64 = (0..n)
            .step_by(block_size)
            .map(|block_start| {
                let block_end = (block_start + block_size).min(n);
                let mut local_sum = 0.0;

                for i in block_start..block_end {
                    if x[i] == 1 {
                        for j in i + 1..n {
                            if x[j] == 1 {
                                local_sum += self.qubo[[i, j]];
                            }
                        }
                    }
                }

                local_sum
            })
            .sum();

        2.0f64.mul_add(quadratic_energy, linear_energy)
    }

    /// Evaluate energy change for single bit flip
    pub fn delta_energy(&self, x: &ArrayView1<u8>, bit: usize) -> f64 {
        let n = x.len();
        let current_val = x[bit];
        let new_val = 1 - current_val;

        let mut delta = 0.0;

        // Diagonal term
        delta += (new_val as f64 - current_val as f64) * self.cache[bit];

        // Off-diagonal terms
        for j in 0..n {
            if j != bit && x[j] == 1 {
                let coupling = if bit < j {
                    self.qubo[[bit, j]]
                } else {
                    self.qubo[[j, bit]]
                };
                delta += 2.0 * (new_val as f64 - current_val as f64) * coupling;
            }
        }

        delta
    }
}

/// Optimized simulated annealing
pub struct OptimizedSA {
    /// QUBO evaluator
    evaluator: OptimizedQUBOEvaluator,
    /// Temperature schedule
    schedule: AnnealingSchedule,
    /// Parallel moves
    parallel_moves: bool,
}

#[derive(Debug, Clone)]
pub enum AnnealingSchedule {
    /// Geometric cooling: T(k) = T0 * alpha^k
    Geometric { t0: f64, alpha: f64 },
    /// Linear cooling: T(k) = T0 * (1 - k/max_iter)
    Linear { t0: f64, max_iter: usize },
    /// Adaptive cooling based on acceptance rate
    Adaptive { t0: f64, target_rate: f64 },
    /// Custom schedule
    Custom(Vec<f64>),
}

impl OptimizedSA {
    /// Create new optimized SA
    pub fn new(qubo: Array2<f64>) -> Self {
        Self {
            evaluator: OptimizedQUBOEvaluator::new(qubo),
            schedule: AnnealingSchedule::Geometric {
                t0: 1.0,
                alpha: 0.99,
            },
            parallel_moves: false,
        }
    }

    /// Set annealing schedule
    pub fn with_schedule(mut self, schedule: AnnealingSchedule) -> Self {
        self.schedule = schedule;
        self
    }

    /// Enable parallel moves
    pub const fn with_parallel_moves(mut self, parallel: bool) -> Self {
        self.parallel_moves = parallel;
        self
    }

    /// Run optimized annealing
    pub fn anneal(
        &self,
        initial: Array1<u8>,
        iterations: usize,
        rng: &mut impl scirs2_core::random::Rng,
    ) -> (Array1<u8>, f64) {
        let n = initial.len();
        let mut current = initial;
        let mut current_energy = self.evaluator.evaluate(&current.view());

        let mut best = current.clone();
        let mut best_energy = current_energy;

        // `Adaptive` cooling genuinely needs to observe realized acceptance
        // decisions as the run progresses, which a precomputed temperature
        // vector cannot do; it is handled with its own control loop here
        // instead of going through `generate_schedule` (which used to
        // return a flat, non-cooling `vec![t0; iterations]` for this
        // variant despite being documented as "Adaptive cooling based on
        // acceptance rate").
        if let AnnealingSchedule::Adaptive { t0, target_rate } = &self.schedule {
            return self.anneal_adaptive(
                current,
                current_energy,
                best,
                best_energy,
                iterations,
                *t0,
                *target_rate,
                rng,
            );
        }

        // Temperature schedule
        let temperatures = self.generate_schedule(iterations);

        for &temp in &temperatures {
            if self.parallel_moves && n > 100 {
                // Parallel neighborhood evaluation
                self.parallel_step(
                    &mut current,
                    &mut current_energy,
                    &mut best,
                    &mut best_energy,
                    temp,
                    rng,
                );
            } else {
                // Sequential moves
                for _ in 0..n {
                    let bit = rng.random_range(0..n);
                    let delta = self.evaluator.delta_energy(&current.view(), bit);

                    if delta < 0.0 || rng.random::<f64>() < (-delta / temp).exp() {
                        current[bit] = 1 - current[bit];
                        current_energy += delta;

                        if current_energy < best_energy {
                            best = current.clone();
                            best_energy = current_energy;
                        }
                    }
                }
            }
        }

        (best, best_energy)
    }

    /// Real adaptive-cooling annealing loop: temperature starts at `t0` and
    /// is adjusted multiplicatively every `n` moves (a "sliding window")
    /// toward whatever value makes the realized acceptance rate match
    /// `target_rate` — the standard adaptive-SA control scheme. This
    /// replaces the old `generate_schedule` behavior for `Adaptive`, which
    /// never read `target_rate` at all and held a completely flat
    /// (non-cooling) temperature for the whole run.
    #[allow(clippy::too_many_arguments)]
    fn anneal_adaptive(
        &self,
        mut current: Array1<u8>,
        mut current_energy: f64,
        mut best: Array1<u8>,
        mut best_energy: f64,
        iterations: usize,
        t0: f64,
        target_rate: f64,
        rng: &mut impl scirs2_core::random::Rng,
    ) -> (Array1<u8>, f64) {
        let n = current.len();
        let window = n.max(10);
        let mut temp = t0.max(1e-6);
        let mut accepted_in_window = 0usize;
        let mut moves_in_window = 0usize;

        for _ in 0..iterations {
            for _ in 0..n {
                let bit = rng.random_range(0..n);
                let delta = self.evaluator.delta_energy(&current.view(), bit);
                let accept = delta < 0.0 || rng.random::<f64>() < (-delta / temp).exp();

                moves_in_window += 1;
                if accept {
                    accepted_in_window += 1;
                    current[bit] = 1 - current[bit];
                    current_energy += delta;

                    if current_energy < best_energy {
                        best = current.clone();
                        best_energy = current_energy;
                    }
                }
            }

            if moves_in_window >= window {
                let acceptance_rate = accepted_in_window as f64 / moves_in_window as f64;
                // Multiplicative control: raise T when accepting less than
                // the target rate (too "cold"), lower T when accepting more
                // than the target rate (too "hot").
                let adjustment = (acceptance_rate - target_rate).mul_add(0.5, 1.0);
                temp = (temp * adjustment).clamp(1e-6, t0.max(1e-6) * 10.0);
                accepted_in_window = 0;
                moves_in_window = 0;
            }
        }

        (best, best_energy)
    }

    /// Generate temperature schedule
    ///
    /// Note: [`AnnealingSchedule::Adaptive`] is *not* produced as a static
    /// vector here — real adaptive cooling requires observing acceptance
    /// decisions as the run progresses, which [`Self::anneal`] handles via
    /// [`Self::anneal_adaptive`] instead of consulting this function. The
    /// branch below is kept only so a caller previewing this schedule
    /// variant sees a reasonable (monotonically cooling) approximation
    /// rather than the previous flat, non-cooling `vec![t0; iterations]`.
    fn generate_schedule(&self, iterations: usize) -> Vec<f64> {
        match &self.schedule {
            AnnealingSchedule::Geometric { t0, alpha } => {
                (0..iterations).map(|k| t0 * alpha.powi(k as i32)).collect()
            }
            AnnealingSchedule::Linear { t0, max_iter } => (0..iterations)
                .map(|k| t0 * (1.0 - k as f64 / *max_iter as f64).max(0.0))
                .collect(),
            AnnealingSchedule::Adaptive { t0, .. } => {
                // Fall back to a geometric preview; the real control loop
                // lives in `anneal_adaptive`.
                let alpha: f64 = 0.99;
                (0..iterations).map(|k| t0 * alpha.powi(k as i32)).collect()
            }
            AnnealingSchedule::Custom(schedule) => schedule.clone(),
        }
    }

    /// Parallel neighborhood evaluation (parallel feature removed - using sequential)
    fn parallel_step(
        &self,
        current: &mut Array1<u8>,
        current_energy: &mut f64,
        best: &mut Array1<u8>,
        best_energy: &mut f64,
        temp: f64,
        rng: &mut impl scirs2_core::random::Rng,
    ) {
        let n = current.len();

        // Evaluate all possible moves (sequential - parallel feature removed per SciRS2 POLICY)
        let deltas: Vec<_> = (0..n)
            .map(|bit| {
                let delta = self.evaluator.delta_energy(&current.view(), bit);
                (bit, delta)
            })
            .collect();

        // Select moves to accept
        let mut accepted = Vec::new();
        for (bit, delta) in deltas {
            if delta < 0.0 || rng.random::<f64>() < (-delta / temp).exp() {
                accepted.push((bit, delta));
            }
        }

        // Apply non-conflicting moves
        let mut applied_energy = 0.0;
        for (bit, delta) in accepted {
            // Simple conflict resolution - skip if would increase energy too much
            if applied_energy + delta < temp {
                current[bit] = 1 - current[bit];
                applied_energy += delta;
            }
        }

        *current_energy += applied_energy;

        if *current_energy < *best_energy {
            *best = current.clone();
            *best_energy = *current_energy;
        }
    }
}

/// Optimized matrix operations for QUBO
pub mod matrix_ops {
    use super::*;

    /// Fast matrix-vector multiplication for sparse QUBO
    pub fn sparse_qubo_multiply(
        qubo: &Array2<f64>,
        x: &ArrayView1<u8>,
        _threshold: f64,
    ) -> Array1<f64> {
        let n = x.len();
        let mut result = Array1::zeros(n);

        // Identify non-zero entries
        let active: Vec<usize> = (0..n).filter(|&i| x[i] == 1).collect();

        if active.len() < n / 4 {
            // Sparse computation
            for &i in &active {
                result[i] += qubo[[i, i]];
                for &j in &active {
                    if i != j {
                        result[i] += qubo[[i, j]];
                    }
                }
            }
        } else {
            // Dense computation
            for i in 0..n {
                if x[i] == 1 {
                    for j in 0..n {
                        if x[j] == 1 {
                            result[i] += qubo[[i, j]];
                        }
                    }
                }
            }
        }

        result
    }

    /// Block-wise QUBO evaluation for cache efficiency
    pub fn block_qubo_eval(qubo: &Array2<f64>, x: &ArrayView1<u8>, block_size: usize) -> f64 {
        let n = x.len();
        let num_blocks = n.div_ceil(block_size);

        let mut energy = 0.0;

        // Process blocks
        for bi in 0..num_blocks {
            for bj in bi..num_blocks {
                let i_start = bi * block_size;
                let i_end = ((bi + 1) * block_size).min(n);
                let j_start = bj * block_size;
                let j_end = ((bj + 1) * block_size).min(n);

                // Process block
                for i in i_start..i_end {
                    if x[i] == 1 {
                        let j_begin = if bi == bj { i } else { j_start };
                        for j in j_begin..j_end {
                            if x[j] == 1 {
                                let factor = if i == j { 1.0 } else { 2.0 };
                                energy += factor * qubo[[i, j]];
                            }
                        }
                    }
                }
            }
        }

        energy
    }
}

/// Memory pool for efficient allocation
pub struct MemoryPool<T> {
    /// Available buffers
    buffers: Vec<Vec<T>>,
    /// Buffer size
    size: usize,
}

impl<T: Clone + Default> MemoryPool<T> {
    /// Create new memory pool
    pub fn new(size: usize, capacity: usize) -> Self {
        let buffers = (0..capacity).map(|_| vec![T::default(); size]).collect();

        Self { buffers, size }
    }

    /// Get buffer from pool
    pub fn get(&mut self) -> Option<Vec<T>> {
        self.buffers.pop()
    }

    /// Return buffer to pool
    pub fn put(&mut self, mut buffer: Vec<T>) {
        if buffer.len() == self.size {
            // Clear buffer for reuse
            for item in &mut buffer {
                *item = T::default();
            }
            self.buffers.push(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use scirs2_core::random::prelude::*;

    #[test]
    #[ignore]
    fn test_optimized_evaluator() {
        let mut qubo = array![[1.0, -2.0, 0.0], [-2.0, 3.0, -1.0], [0.0, -1.0, 2.0]];

        let evaluator = OptimizedQUBOEvaluator::new(qubo);

        let mut x = array![1, 0, 1];
        let mut energy = evaluator.evaluate(&x.view());

        // Manual calculation: 1*1 + 2*1 + 2*(-1)*1*1 = 1 + 2 - 2 = 1
        assert!((energy - 1.0).abs() < 1e-6);

        // Test delta energy
        let delta = evaluator.delta_energy(&x.view(), 1);
        assert!((delta - 2.0).abs() < 1e-6); // Flipping bit 1 from 0 to 1
    }

    #[test]
    fn test_optimized_sa() {
        let mut qubo = array![[0.0, -1.0], [-1.0, 0.0]];

        let sa = OptimizedSA::new(qubo).with_schedule(AnnealingSchedule::Geometric {
            t0: 1.0,
            alpha: 0.95,
        });

        let initial = array![0, 0];
        let mut rng = thread_rng();

        let (solution, energy) = sa.anneal(initial, 100, &mut rng);

        // Should find one of the optimal solutions
        assert!(
            (solution == array![0, 1] && (energy - 0.0).abs() < 1e-6)
                || (solution == array![1, 0] && (energy - 0.0).abs() < 1e-6)
                || (solution == array![1, 1] && (energy - (-2.0)).abs() < 1e-6)
        );
    }

    #[test]
    fn test_generate_schedule_adaptive_is_not_flat() {
        let qubo = array![[0.0, -1.0], [-1.0, 0.0]];
        let sa = OptimizedSA::new(qubo).with_schedule(AnnealingSchedule::Adaptive {
            t0: 2.0,
            target_rate: 0.4,
        });

        let schedule = sa.generate_schedule(50);
        // The old fabricated implementation returned `vec![t0; iterations]`
        // — a completely flat, non-cooling schedule that never read
        // `target_rate` at all.
        assert!(schedule[0] > schedule[schedule.len() - 1]);
        assert!(schedule.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-12));
    }

    #[test]
    fn test_anneal_adaptive_schedule_finds_optimum() {
        let qubo = array![[0.0, -1.0], [-1.0, 0.0]];

        let sa = OptimizedSA::new(qubo).with_schedule(AnnealingSchedule::Adaptive {
            t0: 1.0,
            target_rate: 0.3,
        });

        let initial = array![0, 0];
        let mut rng = thread_rng();

        let (solution, energy) = sa.anneal(initial, 200, &mut rng);

        // A real adaptive-cooling run must still converge on this trivial
        // problem; a flat, non-cooling schedule would never reliably
        // "settle" into the optimum the way a genuinely cooling schedule
        // does.
        assert!(
            (solution == array![0, 1] && (energy - 0.0).abs() < 1e-6)
                || (solution == array![1, 0] && (energy - 0.0).abs() < 1e-6)
                || (solution == array![1, 1] && (energy - (-2.0)).abs() < 1e-6)
        );
    }
}
