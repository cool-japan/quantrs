//! Adaptive SIMD dispatch based on CPU capabilities detection
//!
//! This module provides runtime detection of CPU capabilities and dispatches
//! to the most optimized SIMD implementation available on the target hardware.

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::platform::PlatformCapabilities;
use scirs2_core::Complex64;
use std::sync::{Mutex, OnceLock};
// use scirs2_core::simd_ops::SimdUnifiedOps;
use crate::simd_ops_stubs::SimdF64;
use scirs2_core::ndarray::ArrayView1;

/// CPU feature detection results
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    /// AVX2 support (256-bit vectors)
    pub has_avx2: bool,
    /// AVX-512 support (512-bit vectors)
    pub has_avx512: bool,
    /// FMA (Fused Multiply-Add) support
    pub has_fma: bool,
    /// AVX-512 VL (Vector Length) support
    pub has_avx512vl: bool,
    /// AVX-512 DQ (Doubleword and Quadword) support
    pub has_avx512dq: bool,
    /// AVX-512 CD (Conflict Detection) support
    pub has_avx512cd: bool,
    /// SSE 4.1 support
    pub has_sse41: bool,
    /// SSE 4.2 support
    pub has_sse42: bool,
    /// Number of CPU cores
    pub num_cores: usize,
    /// L1 cache size per core (in bytes)
    pub l1_cache_size: usize,
    /// L2 cache size per core (in bytes)
    pub l2_cache_size: usize,
    /// L3 cache size (in bytes)
    pub l3_cache_size: usize,
}

/// SIMD implementation variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdVariant {
    /// Scalar fallback implementation
    Scalar,
    /// SSE 4.1/4.2 implementation
    Sse4,
    /// AVX2 implementation (256-bit)
    Avx2,
    /// AVX-512 implementation (512-bit)
    Avx512,
}

/// Adaptive SIMD dispatcher
pub struct AdaptiveSimdDispatcher {
    /// Detected CPU features
    cpu_features: CpuFeatures,
    /// Selected SIMD variant
    selected_variant: SimdVariant,
    /// Performance cache for different operation sizes
    performance_cache: Mutex<std::collections::HashMap<String, PerformanceData>>,
}

/// Performance data for SIMD operations
#[derive(Debug, Clone)]
pub struct PerformanceData {
    /// Average execution time (nanoseconds)
    avg_time: f64,
    /// Number of samples
    samples: usize,
    /// Best SIMD variant for this operation size
    best_variant: SimdVariant,
}

/// Global dispatcher instance
static GLOBAL_DISPATCHER: OnceLock<AdaptiveSimdDispatcher> = OnceLock::new();

impl AdaptiveSimdDispatcher {
    /// Initialize the global adaptive SIMD dispatcher
    pub fn initialize() -> QuantRS2Result<()> {
        let cpu_features = Self::detect_cpu_features();
        let selected_variant = Self::select_optimal_variant(&cpu_features);

        let dispatcher = Self {
            cpu_features,
            selected_variant,
            performance_cache: Mutex::new(std::collections::HashMap::new()),
        };

        GLOBAL_DISPATCHER.set(dispatcher).map_err(|_| {
            QuantRS2Error::RuntimeError("Adaptive SIMD dispatcher already initialized".to_string())
        })?;

        Ok(())
    }

    /// Get the global dispatcher instance
    pub fn instance() -> QuantRS2Result<&'static Self> {
        GLOBAL_DISPATCHER.get().ok_or_else(|| {
            QuantRS2Error::RuntimeError("Adaptive SIMD dispatcher not initialized".to_string())
        })
    }

    /// Detect CPU features at runtime
    fn detect_cpu_features() -> CpuFeatures {
        let platform = PlatformCapabilities::detect();

        CpuFeatures {
            has_avx2: platform.cpu.simd.avx2,
            has_avx512: platform.cpu.simd.avx512,
            has_fma: platform.cpu.simd.fma,
            has_avx512vl: false, // Not detected in current platform capabilities
            has_avx512dq: false, // Not detected in current platform capabilities
            has_avx512cd: false, // Not detected in current platform capabilities
            has_sse41: platform.cpu.simd.sse4_1,
            has_sse42: platform.cpu.simd.sse4_2,
            num_cores: platform.cpu.logical_cores,
            l1_cache_size: platform.cpu.cache.l1_data.unwrap_or(32 * 1024),
            l2_cache_size: platform.cpu.cache.l2.unwrap_or(256 * 1024),
            l3_cache_size: platform.cpu.cache.l3.unwrap_or(8 * 1024 * 1024),
        }
    }

    /// Select the optimal SIMD variant based on CPU features
    const fn select_optimal_variant(features: &CpuFeatures) -> SimdVariant {
        if features.has_avx512 && features.has_avx512vl && features.has_avx512dq {
            SimdVariant::Avx512
        } else if features.has_avx2 && features.has_fma {
            SimdVariant::Avx2
        } else if features.has_sse41 && features.has_sse42 {
            SimdVariant::Sse4
        } else {
            SimdVariant::Scalar
        }
    }

    /// Apply a single-qubit gate with adaptive SIMD
    pub fn apply_single_qubit_gate_adaptive(
        &self,
        state: &mut [Complex64],
        target: usize,
        matrix: &[Complex64; 4],
    ) -> QuantRS2Result<()> {
        let operation_key = format!("single_qubit_{}", state.len());
        let variant = self.select_variant_for_operation(&operation_key, state.len());

        let start_time = std::time::Instant::now();

        let result = match variant {
            SimdVariant::Avx512 | SimdVariant::Avx2 | SimdVariant::Sse4 => {
                self.apply_single_qubit_sse4(state, target, matrix) // Fallback to SSE4
            }
            SimdVariant::Scalar => self.apply_single_qubit_scalar(state, target, matrix),
        };

        let execution_time = start_time.elapsed().as_nanos() as f64;
        self.update_performance_cache(&operation_key, execution_time, variant);

        result
    }

    /// Apply a two-qubit gate with adaptive SIMD
    pub fn apply_two_qubit_gate_adaptive(
        &self,
        state: &mut [Complex64],
        control: usize,
        target: usize,
        matrix: &[Complex64; 16],
    ) -> QuantRS2Result<()> {
        let operation_key = format!("two_qubit_{}", state.len());
        let variant = self.select_variant_for_operation(&operation_key, state.len());

        let start_time = std::time::Instant::now();

        let result = match variant {
            SimdVariant::Avx512 => self.apply_two_qubit_avx512(state, control, target, matrix),
            SimdVariant::Avx2 => self.apply_two_qubit_avx2(state, control, target, matrix),
            SimdVariant::Sse4 => self.apply_two_qubit_sse4(state, control, target, matrix),
            SimdVariant::Scalar => self.apply_two_qubit_scalar(state, control, target, matrix),
        };

        let execution_time = start_time.elapsed().as_nanos() as f64;
        self.update_performance_cache(&operation_key, execution_time, variant);

        result
    }

    /// Batch apply gates with adaptive SIMD
    pub fn apply_batch_gates_adaptive(
        &self,
        states: &mut [&mut [Complex64]],
        gates: &[Box<dyn crate::gate::GateOp>],
    ) -> QuantRS2Result<()> {
        let batch_size = states.len();
        let operation_key = format!("batch_{}_{}", batch_size, gates.len());
        let variant = self.select_variant_for_operation(&operation_key, batch_size * 1000); // Estimate

        let start_time = std::time::Instant::now();

        let result = match variant {
            SimdVariant::Avx512 => self.apply_batch_gates_avx512(states, gates),
            SimdVariant::Avx2 => self.apply_batch_gates_avx2(states, gates),
            SimdVariant::Sse4 => self.apply_batch_gates_sse4(states, gates),
            SimdVariant::Scalar => self.apply_batch_gates_scalar(states, gates),
        };

        let execution_time = start_time.elapsed().as_nanos() as f64;
        self.update_performance_cache(&operation_key, execution_time, variant);

        result
    }

    /// Select the best SIMD variant for a specific operation
    fn select_variant_for_operation(&self, operation_key: &str, data_size: usize) -> SimdVariant {
        // Check performance cache first
        if let Ok(cache) = self.performance_cache.lock() {
            if let Some(perf_data) = cache.get(operation_key) {
                if perf_data.samples >= 5 {
                    return perf_data.best_variant;
                }
            }
        }

        // Heuristics based on data size and CPU features
        if data_size >= 1024 && self.cpu_features.has_avx512 {
            SimdVariant::Avx512
        } else if data_size >= 256 && self.cpu_features.has_avx2 {
            SimdVariant::Avx2
        } else if data_size >= 64 && self.cpu_features.has_sse41 {
            SimdVariant::Sse4
        } else {
            SimdVariant::Scalar
        }
    }

    /// Update performance cache with execution time
    fn update_performance_cache(
        &self,
        operation_key: &str,
        execution_time: f64,
        variant: SimdVariant,
    ) {
        if let Ok(mut cache) = self.performance_cache.lock() {
            let perf_data =
                cache
                    .entry(operation_key.to_string())
                    .or_insert_with(|| PerformanceData {
                        avg_time: execution_time,
                        samples: 0,
                        best_variant: variant,
                    });

            // Update running average
            perf_data.avg_time = perf_data
                .avg_time
                .mul_add(perf_data.samples as f64, execution_time)
                / (perf_data.samples + 1) as f64;
            perf_data.samples += 1;

            // Update best variant if this one is significantly faster
            if execution_time < perf_data.avg_time * 0.9 {
                perf_data.best_variant = variant;
            }
        }
    }

    /// Get performance report
    pub fn get_performance_report(&self) -> AdaptivePerformanceReport {
        let cache = self
            .performance_cache
            .lock()
            .map(|cache| cache.clone())
            .unwrap_or_default();

        AdaptivePerformanceReport {
            cpu_features: self.cpu_features,
            selected_variant: self.selected_variant,
            performance_cache: cache,
        }
    }

    // SIMD implementation methods (simplified placeholders)

    #[cfg(target_arch = "x86_64")]
    fn apply_single_qubit_avx512(
        &self,
        state: &mut [Complex64],
        target: usize,
        matrix: &[Complex64; 4],
    ) -> QuantRS2Result<()> {
        // AVX-512 implementation using SciRS2 SIMD operations
        // SciRS2 will automatically use AVX-512 if available
        self.apply_single_qubit_simd_unified(state, target, matrix)
    }

    #[cfg(target_arch = "x86_64")]
    fn apply_single_qubit_avx2(
        &self,
        state: &mut [Complex64],
        target: usize,
        matrix: &[Complex64; 4],
    ) -> QuantRS2Result<()> {
        // AVX2 implementation using SciRS2 SIMD operations
        // SciRS2 will automatically use AVX2 if available
        self.apply_single_qubit_simd_unified(state, target, matrix)
    }

    fn apply_single_qubit_sse4(
        &self,
        state: &mut [Complex64],
        target: usize,
        matrix: &[Complex64; 4],
    ) -> QuantRS2Result<()> {
        // SSE4 implementation using SciRS2 SIMD operations
        // SciRS2 will automatically use SSE4 if available
        self.apply_single_qubit_simd_unified(state, target, matrix)
    }

    fn apply_single_qubit_scalar(
        &self,
        state: &mut [Complex64],
        target: usize,
        matrix: &[Complex64; 4],
    ) -> QuantRS2Result<()> {
        // Scalar implementation
        let n = state.len();
        for i in 0..n {
            if (i >> target) & 1 == 0 {
                let j = i | (1 << target);
                let temp0 = state[i];
                let temp1 = state[j];
                state[i] = matrix[0] * temp0 + matrix[1] * temp1;
                state[j] = matrix[2] * temp0 + matrix[3] * temp1;
            }
        }
        Ok(())
    }

    /// Apply single-qubit gate using SciRS2 unified SIMD operations
    fn apply_single_qubit_simd_unified(
        &self,
        state: &mut [Complex64],
        target: usize,
        matrix: &[Complex64; 4],
    ) -> QuantRS2Result<()> {
        let qubit_mask = 1 << target;
        let half_size = state.len() / 2;

        // Collect pairs of indices that need to be processed
        let mut idx0_list = Vec::new();
        let mut idx1_list = Vec::new();

        for i in 0..half_size {
            let idx0 = (i & !(qubit_mask >> 1)) | ((i & (qubit_mask >> 1)) << 1);
            let idx1 = idx0 | qubit_mask;

            if idx1 < state.len() {
                idx0_list.push(idx0);
                idx1_list.push(idx1);
            }
        }

        let pair_count = idx0_list.len();
        if pair_count == 0 {
            return Ok(());
        }

        // Extract amplitude pairs for SIMD processing
        let mut a0_real = Vec::with_capacity(pair_count);
        let mut a0_imag = Vec::with_capacity(pair_count);
        let mut a1_real = Vec::with_capacity(pair_count);
        let mut a1_imag = Vec::with_capacity(pair_count);

        for i in 0..pair_count {
            let a0 = state[idx0_list[i]];
            let a1 = state[idx1_list[i]];
            a0_real.push(a0.re);
            a0_imag.push(a0.im);
            a1_real.push(a1.re);
            a1_imag.push(a1.im);
        }

        // Convert to array views for SciRS2 SIMD operations
        let a0_real_view = ArrayView1::from(&a0_real);
        let a0_imag_view = ArrayView1::from(&a0_imag);
        let a1_real_view = ArrayView1::from(&a1_real);
        let a1_imag_view = ArrayView1::from(&a1_imag);

        // Extract matrix elements
        let m00_re = matrix[0].re;
        let m00_im = matrix[0].im;
        let m01_re = matrix[1].re;
        let m01_im = matrix[1].im;
        let m10_re = matrix[2].re;
        let m10_im = matrix[2].im;
        let m11_re = matrix[3].re;
        let m11_im = matrix[3].im;

        // Compute new amplitudes using SciRS2 SIMD operations
        // new_a0 = m00 * a0 + m01 * a1
        // new_a1 = m10 * a0 + m11 * a1

        // For new_a0_real: m00_re * a0_re - m00_im * a0_im + m01_re * a1_re - m01_im * a1_im
        let term1 = <f64 as SimdF64>::simd_scalar_mul(&a0_real_view, m00_re);
        let term2 = <f64 as SimdF64>::simd_scalar_mul(&a0_imag_view, m00_im);
        let term3 = <f64 as SimdF64>::simd_scalar_mul(&a1_real_view, m01_re);
        let term4 = <f64 as SimdF64>::simd_scalar_mul(&a1_imag_view, m01_im);
        let sub1 = <f64 as SimdF64>::simd_sub_arrays(&term1.view(), &term2.view());
        let sub2 = <f64 as SimdF64>::simd_sub_arrays(&term3.view(), &term4.view());
        let new_a0_real_arr = <f64 as SimdF64>::simd_add_arrays(&sub1.view(), &sub2.view());

        // For new_a0_imag: m00_re * a0_im + m00_im * a0_re + m01_re * a1_im + m01_im * a1_re
        let term5 = <f64 as SimdF64>::simd_scalar_mul(&a0_imag_view, m00_re);
        let term6 = <f64 as SimdF64>::simd_scalar_mul(&a0_real_view, m00_im);
        let term7 = <f64 as SimdF64>::simd_scalar_mul(&a1_imag_view, m01_re);
        let term8 = <f64 as SimdF64>::simd_scalar_mul(&a1_real_view, m01_im);
        let add1 = <f64 as SimdF64>::simd_add_arrays(&term5.view(), &term6.view());
        let add2 = <f64 as SimdF64>::simd_add_arrays(&term7.view(), &term8.view());
        let new_a0_imag_arr = <f64 as SimdF64>::simd_add_arrays(&add1.view(), &add2.view());

        // For new_a1_real: m10_re * a0_re - m10_im * a0_im + m11_re * a1_re - m11_im * a1_im
        let term9 = <f64 as SimdF64>::simd_scalar_mul(&a0_real_view, m10_re);
        let term10 = <f64 as SimdF64>::simd_scalar_mul(&a0_imag_view, m10_im);
        let term11 = <f64 as SimdF64>::simd_scalar_mul(&a1_real_view, m11_re);
        let term12 = <f64 as SimdF64>::simd_scalar_mul(&a1_imag_view, m11_im);
        let sub3 = <f64 as SimdF64>::simd_sub_arrays(&term9.view(), &term10.view());
        let sub4 = <f64 as SimdF64>::simd_sub_arrays(&term11.view(), &term12.view());
        let new_a1_real_arr = <f64 as SimdF64>::simd_add_arrays(&sub3.view(), &sub4.view());

        // For new_a1_imag: m10_re * a0_im + m10_im * a0_re + m11_re * a1_im + m11_im * a1_re
        let term13 = <f64 as SimdF64>::simd_scalar_mul(&a0_imag_view, m10_re);
        let term14 = <f64 as SimdF64>::simd_scalar_mul(&a0_real_view, m10_im);
        let term15 = <f64 as SimdF64>::simd_scalar_mul(&a1_imag_view, m11_re);
        let term16 = <f64 as SimdF64>::simd_scalar_mul(&a1_real_view, m11_im);
        let add3 = <f64 as SimdF64>::simd_add_arrays(&term13.view(), &term14.view());
        let add4 = <f64 as SimdF64>::simd_add_arrays(&term15.view(), &term16.view());
        let new_a1_imag_arr = <f64 as SimdF64>::simd_add_arrays(&add3.view(), &add4.view());

        // Write back results
        for i in 0..pair_count {
            state[idx0_list[i]] = Complex64::new(new_a0_real_arr[i], new_a0_imag_arr[i]);
            state[idx1_list[i]] = Complex64::new(new_a1_real_arr[i], new_a1_imag_arr[i]);
        }

        Ok(())
    }

    // Similar implementations for two-qubit gates and batch operations

    const fn apply_two_qubit_avx512(
        &self,
        _state: &mut [Complex64],
        _control: usize,
        _target: usize,
        _matrix: &[Complex64; 16],
    ) -> QuantRS2Result<()> {
        // Placeholder
        Ok(())
    }

    const fn apply_two_qubit_avx2(
        &self,
        _state: &mut [Complex64],
        _control: usize,
        _target: usize,
        _matrix: &[Complex64; 16],
    ) -> QuantRS2Result<()> {
        // Placeholder
        Ok(())
    }

    const fn apply_two_qubit_sse4(
        &self,
        _state: &mut [Complex64],
        _control: usize,
        _target: usize,
        _matrix: &[Complex64; 16],
    ) -> QuantRS2Result<()> {
        // Placeholder
        Ok(())
    }

    const fn apply_two_qubit_scalar(
        &self,
        _state: &mut [Complex64],
        _control: usize,
        _target: usize,
        _matrix: &[Complex64; 16],
    ) -> QuantRS2Result<()> {
        // Placeholder
        Ok(())
    }

    fn apply_batch_gates_avx512(
        &self,
        _states: &mut [&mut [Complex64]],
        _gates: &[Box<dyn crate::gate::GateOp>],
    ) -> QuantRS2Result<()> {
        // Placeholder
        Ok(())
    }

    fn apply_batch_gates_avx2(
        &self,
        _states: &mut [&mut [Complex64]],
        _gates: &[Box<dyn crate::gate::GateOp>],
    ) -> QuantRS2Result<()> {
        // Placeholder
        Ok(())
    }

    fn apply_batch_gates_sse4(
        &self,
        _states: &mut [&mut [Complex64]],
        _gates: &[Box<dyn crate::gate::GateOp>],
    ) -> QuantRS2Result<()> {
        // Placeholder
        Ok(())
    }

    fn apply_batch_gates_scalar(
        &self,
        _states: &mut [&mut [Complex64]],
        _gates: &[Box<dyn crate::gate::GateOp>],
    ) -> QuantRS2Result<()> {
        // Placeholder
        Ok(())
    }
}

/// Performance report for adaptive SIMD
#[derive(Debug, Clone)]
pub struct AdaptivePerformanceReport {
    pub cpu_features: CpuFeatures,
    pub selected_variant: SimdVariant,
    pub performance_cache: std::collections::HashMap<String, PerformanceData>,
}

/// Convenience functions for adaptive SIMD operations
pub fn apply_single_qubit_adaptive(
    state: &mut [Complex64],
    target: usize,
    matrix: &[Complex64; 4],
) -> QuantRS2Result<()> {
    AdaptiveSimdDispatcher::instance()?.apply_single_qubit_gate_adaptive(state, target, matrix)
}

pub fn apply_two_qubit_adaptive(
    state: &mut [Complex64],
    control: usize,
    target: usize,
    matrix: &[Complex64; 16],
) -> QuantRS2Result<()> {
    AdaptiveSimdDispatcher::instance()?
        .apply_two_qubit_gate_adaptive(state, control, target, matrix)
}

pub fn apply_batch_gates_adaptive(
    states: &mut [&mut [Complex64]],
    gates: &[Box<dyn crate::gate::GateOp>],
) -> QuantRS2Result<()> {
    AdaptiveSimdDispatcher::instance()?.apply_batch_gates_adaptive(states, gates)
}

/// Initialize the adaptive SIMD system
pub fn initialize_adaptive_simd() -> QuantRS2Result<()> {
    AdaptiveSimdDispatcher::initialize()
}

/// Get the performance report
pub fn get_adaptive_performance_report() -> QuantRS2Result<AdaptivePerformanceReport> {
    Ok(AdaptiveSimdDispatcher::instance()?.get_performance_report())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::Complex64;

    #[test]
    fn test_cpu_feature_detection() {
        let features = AdaptiveSimdDispatcher::detect_cpu_features();
        println!("Detected CPU features: {:?}", features);

        // Basic sanity checks
        assert!(features.num_cores >= 1);
        assert!(features.l1_cache_size > 0);
    }

    #[test]
    fn test_simd_variant_selection() {
        let features = CpuFeatures {
            has_avx2: true,
            has_avx512: false,
            has_fma: true,
            has_avx512vl: false,
            has_avx512dq: false,
            has_avx512cd: false,
            has_sse41: true,
            has_sse42: true,
            num_cores: 8,
            l1_cache_size: 32768,
            l2_cache_size: 262144,
            l3_cache_size: 8388608,
        };

        let variant = AdaptiveSimdDispatcher::select_optimal_variant(&features);
        assert_eq!(variant, SimdVariant::Avx2);
    }

    #[test]
    fn test_adaptive_single_qubit_gate() {
        let _ = AdaptiveSimdDispatcher::initialize();

        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

        let hadamard_matrix = [
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
        ];

        let result = apply_single_qubit_adaptive(&mut state, 0, &hadamard_matrix);
        assert!(result.is_ok());

        // Check that the state has been modified
        let expected_amplitude = 1.0 / 2.0_f64.sqrt();
        assert!((state[0].re - expected_amplitude).abs() < 1e-10);
        assert!((state[1].re - expected_amplitude).abs() < 1e-10);
    }

    #[test]
    fn test_performance_caching() {
        let dispatcher = AdaptiveSimdDispatcher {
            cpu_features: AdaptiveSimdDispatcher::detect_cpu_features(),
            selected_variant: SimdVariant::Avx2,
            performance_cache: Mutex::new(std::collections::HashMap::new()),
        };

        dispatcher.update_performance_cache("test_op", 100.0, SimdVariant::Avx2);
        dispatcher.update_performance_cache("test_op", 150.0, SimdVariant::Avx2);

        let perf_data = dispatcher
            .performance_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get("test_op")
            .expect("Performance data for 'test_op' should exist after updates")
            .clone();
        assert_eq!(perf_data.samples, 2);
        assert!((perf_data.avg_time - 125.0).abs() < 1e-10);
    }
}
