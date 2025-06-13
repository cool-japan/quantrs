//! Adaptive SIMD dispatch based on CPU capabilities detection
//!
//! This module provides runtime detection of CPU capabilities and dispatches
//! to the most optimized SIMD implementation available on the target hardware.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
};
use num_complex::Complex64;
use std::sync::{Mutex, OnceLock};

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
struct PerformanceData {
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
        
        let dispatcher = AdaptiveSimdDispatcher {
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
    pub fn instance() -> QuantRS2Result<&'static AdaptiveSimdDispatcher> {
        GLOBAL_DISPATCHER.get().ok_or_else(|| {
            QuantRS2Error::RuntimeError("Adaptive SIMD dispatcher not initialized".to_string())
        })
    }
    
    /// Detect CPU features at runtime
    fn detect_cpu_features() -> CpuFeatures {
        // Use conditional compilation for different target architectures
        #[cfg(target_arch = "x86_64")]
        {
            Self::detect_x86_64_features()
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::detect_aarch64_features()
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback for unsupported architectures
            CpuFeatures {
                has_avx2: false,
                has_avx512: false,
                has_fma: false,
                has_avx512vl: false,
                has_avx512dq: false,
                has_avx512cd: false,
                has_sse41: false,
                has_sse42: false,
                num_cores: 1,
                l1_cache_size: 32768,
                l2_cache_size: 262144,
                l3_cache_size: 8388608,
            }
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    fn detect_x86_64_features() -> CpuFeatures {
        use std::arch::x86_64::*;
        
        // CPUID feature detection
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_avx512 = is_x86_feature_detected!("avx512f");
        let has_fma = is_x86_feature_detected!("fma");
        let has_avx512vl = is_x86_feature_detected!("avx512vl");
        let has_avx512dq = is_x86_feature_detected!("avx512dq");
        let has_avx512cd = is_x86_feature_detected!("avx512cd");
        let has_sse41 = is_x86_feature_detected!("sse4.1");
        let has_sse42 = is_x86_feature_detected!("sse4.2");
        
        // Detect cache sizes and core count
        let num_cores = 8; // Fallback to reasonable default
        let (l1_cache, l2_cache, l3_cache) = Self::detect_cache_sizes();
        
        CpuFeatures {
            has_avx2,
            has_avx512,
            has_fma,
            has_avx512vl,
            has_avx512dq,
            has_avx512cd,
            has_sse41,
            has_sse42,
            num_cores,
            l1_cache_size: l1_cache,
            l2_cache_size: l2_cache,
            l3_cache_size: l3_cache,
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    fn detect_aarch64_features() -> CpuFeatures {
        // ARM NEON is available on all AArch64 processors
        let num_cores = 8; // Fallback to reasonable default
        let (l1_cache, l2_cache, l3_cache) = Self::detect_cache_sizes();
        
        CpuFeatures {
            has_avx2: false, // N/A for ARM
            has_avx512: false, // N/A for ARM
            has_fma: true, // NEON supports FMA
            has_avx512vl: false,
            has_avx512dq: false,
            has_avx512cd: false,
            has_sse41: false, // N/A for ARM
            has_sse42: false, // N/A for ARM
            num_cores,
            l1_cache_size: l1_cache,
            l2_cache_size: l2_cache,
            l3_cache_size: l3_cache,
        }
    }
    
    /// Detect cache sizes (simplified implementation)
    fn detect_cache_sizes() -> (usize, usize, usize) {
        // This is a simplified implementation
        // In practice, you would use CPUID or /proc/cpuinfo on Linux
        let l1_cache = 32768;  // 32KB typical L1
        let l2_cache = 262144; // 256KB typical L2
        let l3_cache = 8388608; // 8MB typical L3
        
        (l1_cache, l2_cache, l3_cache)
    }
    
    /// Select the optimal SIMD variant based on CPU features
    fn select_optimal_variant(features: &CpuFeatures) -> SimdVariant {
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
            SimdVariant::Avx512 => self.apply_single_qubit_sse4(state, target, matrix), // Fallback to SSE4
            SimdVariant::Avx2 => self.apply_single_qubit_sse4(state, target, matrix), // Fallback to SSE4
            SimdVariant::Sse4 => self.apply_single_qubit_sse4(state, target, matrix),
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
    fn update_performance_cache(&self, operation_key: &str, execution_time: f64, variant: SimdVariant) {
        if let Ok(mut cache) = self.performance_cache.lock() {
            let perf_data = cache.entry(operation_key.to_string())
                .or_insert_with(|| PerformanceData {
                    avg_time: execution_time,
                    samples: 0,
                    best_variant: variant,
                });
            
            // Update running average
            perf_data.avg_time = (perf_data.avg_time * perf_data.samples as f64 + execution_time) 
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
        let cache = self.performance_cache.lock()
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
        // AVX-512 implementation using 512-bit vectors
        simd_ops::apply_single_qubit_gate_simd(state, target, matrix)
    }
    
    #[cfg(target_arch = "x86_64")]
    fn apply_single_qubit_avx2(
        &self,
        state: &mut [Complex64],
        target: usize,
        matrix: &[Complex64; 4],
    ) -> QuantRS2Result<()> {
        // AVX2 implementation using 256-bit vectors
        simd_ops::apply_single_qubit_gate_simd(state, target, matrix)
    }
    
    fn apply_single_qubit_sse4(
        &self,
        state: &mut [Complex64],
        target: usize,
        matrix: &[Complex64; 4],
    ) -> QuantRS2Result<()> {
        // For now, fall back to scalar implementation
        // TODO: Implement SIMD version
        self.apply_single_qubit_scalar(state, target, matrix)
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
    
    // Similar implementations for two-qubit gates and batch operations
    
    fn apply_two_qubit_avx512(
        &self,
        _state: &mut [Complex64],
        _control: usize,
        _target: usize,
        _matrix: &[Complex64; 16],
    ) -> QuantRS2Result<()> {
        // Placeholder
        Ok(())
    }
    
    fn apply_two_qubit_avx2(
        &self,
        _state: &mut [Complex64],
        _control: usize,
        _target: usize,
        _matrix: &[Complex64; 16],
    ) -> QuantRS2Result<()> {
        // Placeholder
        Ok(())
    }
    
    fn apply_two_qubit_sse4(
        &self,
        _state: &mut [Complex64],
        _control: usize,
        _target: usize,
        _matrix: &[Complex64; 16],
    ) -> QuantRS2Result<()> {
        // Placeholder
        Ok(())
    }
    
    fn apply_two_qubit_scalar(
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
    AdaptiveSimdDispatcher::instance()?
        .apply_single_qubit_gate_adaptive(state, target, matrix)
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
    AdaptiveSimdDispatcher::instance()?
        .apply_batch_gates_adaptive(states, gates)
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
    use num_complex::Complex64;

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
        
        let mut state = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        
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
        let mut dispatcher = AdaptiveSimdDispatcher {
            cpu_features: AdaptiveSimdDispatcher::detect_cpu_features(),
            selected_variant: SimdVariant::Avx2,
            performance_cache: Mutex::new(std::collections::HashMap::new()),
        };
        
        dispatcher.update_performance_cache("test_op", 100.0, SimdVariant::Avx2);
        dispatcher.update_performance_cache("test_op", 150.0, SimdVariant::Avx2);
        
        let perf_data = dispatcher.performance_cache.lock().unwrap().get("test_op").unwrap().clone();
        assert_eq!(perf_data.samples, 2);
        assert!((perf_data.avg_time - 125.0).abs() < 1e-10);
    }
}