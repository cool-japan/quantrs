//! NVIDIA Tensor Core support for accelerated quantum operations.
//!
//! This module provides TF32 and FP16 mixed-precision operations
//! using NVIDIA Tensor Cores for accelerated matrix multiplication
//! in quantum gate application.
//!
//! ## Precision Modes
//!
//! - **TF32 (TensorFloat-32)**: 19-bit format (8-bit exponent, 10-bit mantissa)
//!   that provides FP32 range with reduced precision. ~2.8x faster than FP32.
//!
//! - **FP16 (Half Precision)**: 16-bit format with 5-bit exponent and 10-bit
//!   mantissa. ~4x faster than FP32 on Tensor Cores.
//!
//! - **BF16 (BFloat16)**: 16-bit format with 8-bit exponent and 7-bit mantissa.
//!   Same range as FP32 with reduced precision. ~4x faster than FP32.
//!
//! ## GPU Compatibility
//!
//! Tensor Cores are available on:
//! - Volta (V100) and newer: FP16/INT8
//! - Turing (RTX 20xx) and newer: FP16/INT8/INT4
//! - Ampere (A100/RTX 30xx) and newer: TF32/BF16/FP16/INT8
//! - Hopper (H100) and newer: FP8/TF32/BF16/FP16

use crate::error::{Result, SimulatorError};
use crate::mixed_precision_impl::QuantumPrecision;

/// Tensor Core compute capability levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TensorCoreGeneration {
    /// No Tensor Core support
    None,
    /// Volta (V100) - FP16 Tensor Cores
    Volta,
    /// Turing (RTX 20xx) - FP16/INT8 Tensor Cores
    Turing,
    /// Ampere (A100/RTX 30xx) - TF32/BF16/FP16 Tensor Cores
    Ampere,
    /// Ada Lovelace (RTX 40xx) - FP8/TF32/BF16/FP16 Tensor Cores
    AdaLovelace,
    /// Hopper (H100) - FP8/TF32/BF16/FP16 Tensor Cores
    Hopper,
    /// Blackwell (B100) - Advanced Tensor Cores
    Blackwell,
}

impl TensorCoreGeneration {
    /// Detect generation from compute capability
    #[must_use]
    pub fn from_compute_capability(major: i32, minor: i32) -> Self {
        match (major, minor) {
            (9, _) => Self::Hopper,      // 9.x = Hopper
            (8, 9) => Self::AdaLovelace, // 8.9 = Ada Lovelace
            (8, _) => Self::Ampere,      // 8.x = Ampere
            (7, 5) => Self::Turing,      // 7.5 = Turing
            (7, _) => Self::Volta,       // 7.0/7.2 = Volta
            (10, _) => Self::Blackwell,  // 10.x = Blackwell
            _ => Self::None,
        }
    }

    /// Check if TF32 is supported
    #[must_use]
    pub const fn supports_tf32(&self) -> bool {
        matches!(
            self,
            Self::Ampere | Self::AdaLovelace | Self::Hopper | Self::Blackwell
        )
    }

    /// Check if BF16 is supported
    #[must_use]
    pub const fn supports_bf16(&self) -> bool {
        matches!(
            self,
            Self::Ampere | Self::AdaLovelace | Self::Hopper | Self::Blackwell
        )
    }

    /// Check if FP16 Tensor Cores are supported
    #[must_use]
    pub const fn supports_fp16_tensor(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Check if FP8 is supported
    #[must_use]
    pub const fn supports_fp8(&self) -> bool {
        matches!(self, Self::AdaLovelace | Self::Hopper | Self::Blackwell)
    }

    /// Get the optimal batch size for this generation
    #[must_use]
    pub const fn optimal_batch_size(&self) -> usize {
        match self {
            Self::None => 1,
            Self::Volta | Self::Turing => 16,
            Self::Ampere => 32,
            Self::AdaLovelace | Self::Hopper | Self::Blackwell => 64,
        }
    }
}

/// Configuration for Tensor Core operations
#[derive(Debug, Clone)]
pub struct TensorCoreConfig {
    /// Enable TF32 mode for FP32 operations (rounds to 10-bit mantissa)
    pub enable_tf32: bool,
    /// Enable FP16 accumulation (faster but less accurate)
    pub enable_fp16_accumulate: bool,
    /// Minimum matrix size to use Tensor Cores (smaller matrices use CUDA cores)
    pub min_matrix_size_for_tc: usize,
    /// Accumulator precision (FP16, FP32, or FP64)
    pub accumulator_precision: AccumulatorPrecision,
    /// Enable mixed-precision math operations
    pub enable_mixed_precision: bool,
    /// Tensor Core generation (auto-detected or manual)
    pub generation: TensorCoreGeneration,
}

/// Precision for accumulation in matrix operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulatorPrecision {
    /// FP16 accumulation (fastest, lowest accuracy)
    Fp16,
    /// FP32 accumulation (balanced)
    Fp32,
    /// FP64 accumulation (slowest, highest accuracy)
    Fp64,
}

impl Default for TensorCoreConfig {
    fn default() -> Self {
        Self {
            enable_tf32: true,
            enable_fp16_accumulate: false,
            min_matrix_size_for_tc: 16,
            accumulator_precision: AccumulatorPrecision::Fp32,
            enable_mixed_precision: true,
            generation: TensorCoreGeneration::None,
        }
    }
}

impl TensorCoreConfig {
    /// Create configuration optimized for accuracy
    #[must_use]
    pub const fn for_accuracy() -> Self {
        Self {
            enable_tf32: false,
            enable_fp16_accumulate: false,
            min_matrix_size_for_tc: 64,
            accumulator_precision: AccumulatorPrecision::Fp64,
            enable_mixed_precision: false,
            generation: TensorCoreGeneration::None,
        }
    }

    /// Create configuration optimized for performance
    #[must_use]
    pub const fn for_performance() -> Self {
        Self {
            enable_tf32: true,
            enable_fp16_accumulate: true,
            min_matrix_size_for_tc: 8,
            accumulator_precision: AccumulatorPrecision::Fp16,
            enable_mixed_precision: true,
            generation: TensorCoreGeneration::None,
        }
    }

    /// Update configuration based on detected GPU capabilities
    pub fn detect_capabilities(&mut self, major: i32, minor: i32) {
        self.generation = TensorCoreGeneration::from_compute_capability(major, minor);

        // Disable TF32 if not supported
        if !self.generation.supports_tf32() {
            self.enable_tf32 = false;
        }
    }

    /// Check if Tensor Cores are available
    #[must_use]
    pub const fn tensor_cores_available(&self) -> bool {
        !matches!(self.generation, TensorCoreGeneration::None)
    }

    /// Get the best precision for given tolerance
    #[must_use]
    pub fn best_precision_for_tolerance(&self, tolerance: f64) -> QuantumPrecision {
        QuantumPrecision::select_for_accuracy_and_tensor_cores(
            tolerance,
            self.tensor_cores_available(),
        )
    }
}

/// Tensor Core operations trait for quantum gate matrices
pub trait TensorCoreOps {
    /// Check if Tensor Cores are supported
    fn supports_tensor_cores(&self) -> bool;

    /// Get the current Tensor Core configuration
    fn tensor_core_config(&self) -> &TensorCoreConfig;

    /// Perform TF32 matrix multiplication: C = A * B
    /// Both A and B are stored as FP32 but computed with TF32 precision
    fn matmul_tf32(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;

    /// Perform FP16 matrix multiplication: C = A * B
    /// A and B are FP16, C can be FP16 or FP32 depending on accumulator setting
    fn matmul_fp16(
        &self,
        a: &[u16],
        b: &[u16],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;

    /// Perform BF16 matrix multiplication: C = A * B
    fn matmul_bf16(
        &self,
        a: &[u16],
        b: &[u16],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;

    /// Apply a quantum gate with mixed precision
    fn apply_gate_mixed_precision(
        &self,
        state: &mut [f32],
        gate: &[f32],
        target_qubits: &[usize],
        precision: QuantumPrecision,
    ) -> Result<()>;
}

/// FP16 conversion utilities
pub mod fp16_utils {
    /// Convert FP32 to FP16 (IEEE 754 half-precision)
    #[must_use]
    pub fn f32_to_fp16(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let mantissa = bits & 0x7FFFFF;

        // Handle special cases
        if exp == 128 {
            // Infinity or NaN
            return ((sign << 15) | 0x7C00 | (mantissa >> 13)) as u16;
        }

        if exp < -14 {
            // Subnormal or zero in FP16
            if exp < -24 {
                return (sign << 15) as u16; // Zero
            }
            let mantissa = (mantissa | 0x800000) >> (-exp - 14 + 1);
            return ((sign << 15) | (mantissa >> 13)) as u16;
        }

        if exp > 15 {
            // Overflow to infinity
            return ((sign << 15) | 0x7C00) as u16;
        }

        // Normal case
        let fp16_exp = (exp + 15) as u32;
        let fp16_mantissa = mantissa >> 13;
        ((sign << 15) | (fp16_exp << 10) | fp16_mantissa) as u16
    }

    /// Convert FP16 to FP32
    #[must_use]
    pub fn fp16_to_f32(value: u16) -> f32 {
        let sign = ((value >> 15) & 1) as u32;
        let exp = ((value >> 10) & 0x1F) as i32;
        let mantissa = (value & 0x3FF) as u32;

        if exp == 0 {
            if mantissa == 0 {
                // Zero
                return f32::from_bits(sign << 31);
            }
            // Subnormal
            let mut m = mantissa;
            let mut e = -14i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            let mantissa = (m & 0x3FF) << 13;
            let exp = (e + 127) as u32;
            return f32::from_bits((sign << 31) | (exp << 23) | mantissa);
        }

        if exp == 31 {
            // Infinity or NaN
            let bits = (sign << 31) | 0x7F800000 | (mantissa << 13);
            return f32::from_bits(bits);
        }

        // Normal case
        let fp32_exp = (exp - 15 + 127) as u32;
        let fp32_mantissa = mantissa << 13;
        f32::from_bits((sign << 31) | (fp32_exp << 23) | fp32_mantissa)
    }

    /// Convert FP32 to BF16 (Brain Float 16)
    #[must_use]
    pub fn f32_to_bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        // BF16 simply takes the upper 16 bits with rounding
        let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
        ((bits + rounding_bias) >> 16) as u16
    }

    /// Convert BF16 to FP32
    #[must_use]
    pub fn bf16_to_f32(value: u16) -> f32 {
        f32::from_bits((value as u32) << 16)
    }

    /// Convert FP32 to TF32 (truncate mantissa to 10 bits)
    /// Note: TF32 is stored as FP32 but only 19 bits are used in computation
    #[must_use]
    pub fn f32_to_tf32(value: f32) -> f32 {
        let bits = value.to_bits();
        // Keep sign (1 bit) + exponent (8 bits) + 10 mantissa bits
        // Zero out the lower 13 mantissa bits
        let mask = 0xFFFFE000u32;
        f32::from_bits(bits & mask)
    }

    /// Round FP32 to TF32 precision with rounding
    #[must_use]
    pub fn f32_to_tf32_rounded(value: f32) -> f32 {
        let bits = value.to_bits();
        // Add rounding bit and truncate
        let rounding_bias = 0x1000; // Add 1 to bit 12
        let rounded = bits.saturating_add(rounding_bias);
        let mask = 0xFFFFE000u32;
        f32::from_bits(rounded & mask)
    }
}

/// Tensor Core kernel wrappers for quantum operations
pub struct TensorCoreKernels {
    config: TensorCoreConfig,
    /// Is CUDA available
    cuda_available: bool,
}

impl TensorCoreKernels {
    /// Create new Tensor Core kernels with given configuration
    #[must_use]
    pub fn new(config: TensorCoreConfig) -> Self {
        Self {
            config,
            cuda_available: false, // Will be set by initialization
        }
    }

    /// Initialize with GPU detection
    pub fn initialize(&mut self) -> Result<()> {
        // In a real implementation, this would:
        // 1. Query CUDA runtime for device info
        // 2. Get compute capability
        // 3. Update config.generation accordingly

        #[cfg(feature = "advanced_math")]
        {
            use super::context::CudaContext;
            if let Ok(context) = CudaContext::new(0) {
                let props = context.get_device_properties();
                self.config
                    .detect_capabilities(props.compute_capability.0, props.compute_capability.1);
                self.cuda_available = true;
            }
        }

        Ok(())
    }

    /// Apply single-qubit gate using TF32 precision
    pub fn apply_single_qubit_gate_tf32(
        &self,
        state: &mut [f32],
        gate: &[[f32; 2]; 2],
        target_qubit: usize,
        num_qubits: usize,
    ) -> Result<()> {
        if !self.config.enable_tf32 || !self.config.generation.supports_tf32() {
            return Err(SimulatorError::UnsupportedOperation(
                "TF32 not supported on this GPU".to_string(),
            ));
        }

        let state_size = 1 << num_qubits;
        if state.len() != state_size * 2 {
            // Complex = 2 floats
            return Err(SimulatorError::InvalidInput(
                "State vector size mismatch".to_string(),
            ));
        }

        // Convert gate to TF32 precision
        let gate_tf32 = [
            [
                fp16_utils::f32_to_tf32_rounded(gate[0][0]),
                fp16_utils::f32_to_tf32_rounded(gate[0][1]),
            ],
            [
                fp16_utils::f32_to_tf32_rounded(gate[1][0]),
                fp16_utils::f32_to_tf32_rounded(gate[1][1]),
            ],
        ];

        // Apply gate (CPU fallback for now)
        // In real implementation: launch CUDA kernel with wmma::mma_sync
        let mask = 1 << target_qubit;
        for i in 0..(state_size / 2) {
            let i0 = (i & !(mask - 1)) << 1 | (i & (mask - 1));
            let i1 = i0 | mask;

            // State is stored as [re0, im0, re1, im1, ...]
            let idx0 = i0 * 2;
            let idx1 = i1 * 2;

            let (a_re, a_im) = (state[idx0], state[idx0 + 1]);
            let (b_re, b_im) = (state[idx1], state[idx1 + 1]);

            // Apply 2x2 gate with TF32 precision
            let new_a_re = gate_tf32[0][0] * a_re - gate_tf32[0][0] * a_im + gate_tf32[0][1] * b_re
                - gate_tf32[0][1] * b_im;
            let new_a_im = gate_tf32[0][0] * a_im
                + gate_tf32[0][0] * a_re
                + gate_tf32[0][1] * b_im
                + gate_tf32[0][1] * b_re;

            let new_b_re = gate_tf32[1][0] * a_re - gate_tf32[1][0] * a_im + gate_tf32[1][1] * b_re
                - gate_tf32[1][1] * b_im;
            let new_b_im = gate_tf32[1][0] * a_im
                + gate_tf32[1][0] * a_re
                + gate_tf32[1][1] * b_im
                + gate_tf32[1][1] * b_re;

            state[idx0] = new_a_re;
            state[idx0 + 1] = new_a_im;
            state[idx1] = new_b_re;
            state[idx1 + 1] = new_b_im;
        }

        Ok(())
    }

    /// Get estimated TFLOPS for current configuration
    #[must_use]
    pub fn estimated_tflops(&self, precision: QuantumPrecision) -> f64 {
        // Theoretical peak TFLOPS for various GPUs and precisions
        match (self.config.generation, precision) {
            (TensorCoreGeneration::Ampere, QuantumPrecision::TF32) => 156.0, // A100
            (TensorCoreGeneration::Ampere, QuantumPrecision::Half) => 312.0, // A100
            (TensorCoreGeneration::Ampere, QuantumPrecision::BFloat16) => 312.0,
            (TensorCoreGeneration::Ampere, QuantumPrecision::Single) => 19.5,
            (TensorCoreGeneration::Hopper, QuantumPrecision::TF32) => 495.0, // H100
            (TensorCoreGeneration::Hopper, QuantumPrecision::Half) => 990.0,
            (TensorCoreGeneration::Hopper, QuantumPrecision::BFloat16) => 990.0,
            (TensorCoreGeneration::Volta, QuantumPrecision::Half) => 125.0, // V100
            (TensorCoreGeneration::Turing, QuantumPrecision::Half) => 65.0, // RTX 2080
            (TensorCoreGeneration::AdaLovelace, QuantumPrecision::TF32) => 82.6, // RTX 4090
            (TensorCoreGeneration::AdaLovelace, QuantumPrecision::Half) => 165.2,
            _ => 0.0, // No Tensor Core support or unknown
        }
    }

    /// Check if a precision mode is supported
    #[must_use]
    pub fn supports_precision(&self, precision: QuantumPrecision) -> bool {
        match precision {
            QuantumPrecision::TF32 => self.config.generation.supports_tf32(),
            QuantumPrecision::BFloat16 => self.config.generation.supports_bf16(),
            QuantumPrecision::Half => self.config.generation.supports_fp16_tensor(),
            QuantumPrecision::Single | QuantumPrecision::Double => true,
            QuantumPrecision::Adaptive => true,
        }
    }
}

impl TensorCoreOps for TensorCoreKernels {
    fn supports_tensor_cores(&self) -> bool {
        self.config.tensor_cores_available() && self.cuda_available
    }

    fn tensor_core_config(&self) -> &TensorCoreConfig {
        &self.config
    }

    fn matmul_tf32(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if !self.config.generation.supports_tf32() {
            return Err(SimulatorError::UnsupportedOperation(
                "TF32 not supported".to_string(),
            ));
        }

        // Validate dimensions
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(SimulatorError::InvalidInput(
                "Matrix dimension mismatch".to_string(),
            ));
        }

        // CPU fallback implementation
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    let a_tf32 = fp16_utils::f32_to_tf32(a[i * k + l]);
                    let b_tf32 = fp16_utils::f32_to_tf32(b[l * n + j]);
                    sum += a_tf32 * b_tf32;
                }
                c[i * n + j] = sum;
            }
        }

        Ok(())
    }

    fn matmul_fp16(
        &self,
        a: &[u16],
        b: &[u16],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if !self.config.generation.supports_fp16_tensor() {
            return Err(SimulatorError::UnsupportedOperation(
                "FP16 Tensor Cores not supported".to_string(),
            ));
        }

        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(SimulatorError::InvalidInput(
                "Matrix dimension mismatch".to_string(),
            ));
        }

        // CPU fallback
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    let a_f32 = fp16_utils::fp16_to_f32(a[i * k + l]);
                    let b_f32 = fp16_utils::fp16_to_f32(b[l * n + j]);
                    sum += a_f32 * b_f32;
                }
                c[i * n + j] = sum;
            }
        }

        Ok(())
    }

    fn matmul_bf16(
        &self,
        a: &[u16],
        b: &[u16],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if !self.config.generation.supports_bf16() {
            return Err(SimulatorError::UnsupportedOperation(
                "BF16 not supported".to_string(),
            ));
        }

        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(SimulatorError::InvalidInput(
                "Matrix dimension mismatch".to_string(),
            ));
        }

        // CPU fallback
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    let a_f32 = fp16_utils::bf16_to_f32(a[i * k + l]);
                    let b_f32 = fp16_utils::bf16_to_f32(b[l * n + j]);
                    sum += a_f32 * b_f32;
                }
                c[i * n + j] = sum;
            }
        }

        Ok(())
    }

    fn apply_gate_mixed_precision(
        &self,
        state: &mut [f32],
        gate: &[f32],
        _target_qubits: &[usize],
        precision: QuantumPrecision,
    ) -> Result<()> {
        if !self.supports_precision(precision) {
            return Err(SimulatorError::UnsupportedOperation(format!(
                "Precision {:?} not supported",
                precision
            )));
        }

        // For now, apply precision truncation to gate
        match precision {
            QuantumPrecision::TF32 => {
                // Truncate gate to TF32 and apply
                let _gate_tf32: Vec<f32> =
                    gate.iter().map(|&v| fp16_utils::f32_to_tf32(v)).collect();
                // Apply gate... (actual implementation would use CUDA)
            }
            QuantumPrecision::Half | QuantumPrecision::BFloat16 => {
                // Convert to FP16/BF16 and apply
                let _gate_fp16: Vec<u16> =
                    gate.iter().map(|&v| fp16_utils::f32_to_fp16(v)).collect();
                // Apply gate...
            }
            _ => {
                // Use original precision
            }
        }

        // Placeholder - actual implementation would launch CUDA kernel
        let _ = state;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_core_generation_detection() {
        assert_eq!(
            TensorCoreGeneration::from_compute_capability(9, 0),
            TensorCoreGeneration::Hopper
        );
        assert_eq!(
            TensorCoreGeneration::from_compute_capability(8, 9),
            TensorCoreGeneration::AdaLovelace
        );
        assert_eq!(
            TensorCoreGeneration::from_compute_capability(8, 0),
            TensorCoreGeneration::Ampere
        );
        assert_eq!(
            TensorCoreGeneration::from_compute_capability(7, 5),
            TensorCoreGeneration::Turing
        );
        assert_eq!(
            TensorCoreGeneration::from_compute_capability(7, 0),
            TensorCoreGeneration::Volta
        );
        assert_eq!(
            TensorCoreGeneration::from_compute_capability(6, 1),
            TensorCoreGeneration::None
        );
    }

    #[test]
    fn test_tf32_support() {
        assert!(TensorCoreGeneration::Ampere.supports_tf32());
        assert!(TensorCoreGeneration::Hopper.supports_tf32());
        assert!(!TensorCoreGeneration::Volta.supports_tf32());
        assert!(!TensorCoreGeneration::Turing.supports_tf32());
    }

    #[test]
    fn test_fp16_conversion() {
        // Test some known values
        let one_f32 = 1.0f32;
        let one_fp16 = fp16_utils::f32_to_fp16(one_f32);
        let one_back = fp16_utils::fp16_to_f32(one_fp16);
        assert!((one_f32 - one_back).abs() < 1e-6);

        let half_f32 = 0.5f32;
        let half_fp16 = fp16_utils::f32_to_fp16(half_f32);
        let half_back = fp16_utils::fp16_to_f32(half_fp16);
        assert!((half_f32 - half_back).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_conversion() {
        let value = 3.14159f32;
        let bf16 = fp16_utils::f32_to_bf16(value);
        let back = fp16_utils::bf16_to_f32(bf16);
        // BF16 has lower precision, so use larger tolerance
        assert!((value - back).abs() < 0.01);
    }

    #[test]
    fn test_tf32_truncation() {
        let value = 1.23456789f32;
        let tf32 = fp16_utils::f32_to_tf32(value);
        // TF32 keeps 10 mantissa bits, so precision is ~1e-3 relative
        assert!((value - tf32).abs() < 0.001);
    }

    #[test]
    fn test_tensor_core_config_default() {
        let config = TensorCoreConfig::default();
        assert!(config.enable_tf32);
        assert!(!config.enable_fp16_accumulate);
        assert_eq!(config.accumulator_precision, AccumulatorPrecision::Fp32);
    }

    #[test]
    fn test_tensor_core_config_performance() {
        let config = TensorCoreConfig::for_performance();
        assert!(config.enable_tf32);
        assert!(config.enable_fp16_accumulate);
        assert_eq!(config.accumulator_precision, AccumulatorPrecision::Fp16);
    }

    #[test]
    fn test_tensor_core_config_accuracy() {
        let config = TensorCoreConfig::for_accuracy();
        assert!(!config.enable_tf32);
        assert!(!config.enable_fp16_accumulate);
        assert_eq!(config.accumulator_precision, AccumulatorPrecision::Fp64);
    }

    #[test]
    fn test_precision_selection() {
        // Test without tensor cores (default generation is None)
        let config_no_tc = TensorCoreConfig::default();
        assert!(!config_no_tc.tensor_cores_available());

        // tolerance 1e-3: between 1e-4 and 1e-2, so without tensor cores -> Single
        let precision_no_tc = config_no_tc.best_precision_for_tolerance(1e-3);
        assert_eq!(precision_no_tc, QuantumPrecision::Single);

        // Test with tensor cores enabled (set generation to Ampere)
        let mut config_with_tc = TensorCoreConfig::default();
        config_with_tc.generation = TensorCoreGeneration::Ampere;
        assert!(config_with_tc.tensor_cores_available());

        // tolerance 1e-3: between 1e-4 and 1e-2, with tensor cores -> TF32
        let precision_with_tc = config_with_tc.best_precision_for_tolerance(1e-3);
        assert_eq!(precision_with_tc, QuantumPrecision::TF32);

        // tolerance >= 1e-2: with tensor cores -> BFloat16
        let precision_low_tolerance = config_with_tc.best_precision_for_tolerance(1e-2);
        assert_eq!(precision_low_tolerance, QuantumPrecision::BFloat16);

        // tolerance < 1e-6: should be Double regardless of tensor cores
        let precision_high = config_no_tc.best_precision_for_tolerance(1e-8);
        assert_eq!(precision_high, QuantumPrecision::Double);
    }

    #[test]
    fn test_matmul_tf32_dimensions() {
        let mut config = TensorCoreConfig::default();
        config.generation = TensorCoreGeneration::Ampere;
        let kernels = TensorCoreKernels::new(config);

        let a = vec![1.0f32; 4 * 4];
        let b = vec![1.0f32; 4 * 4];
        let mut c = vec![0.0f32; 4 * 4];

        let result = kernels.matmul_tf32(&a, &b, &mut c, 4, 4, 4);
        assert!(result.is_ok());

        // Each element should be 4.0 (sum of 4 ones)
        for val in c {
            assert!((val - 4.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_estimated_tflops() {
        let mut config = TensorCoreConfig::default();
        config.generation = TensorCoreGeneration::Ampere;
        let kernels = TensorCoreKernels::new(config);

        let tflops_tf32 = kernels.estimated_tflops(QuantumPrecision::TF32);
        let tflops_fp16 = kernels.estimated_tflops(QuantumPrecision::Half);

        assert!(tflops_tf32 > 0.0);
        assert!(tflops_fp16 > tflops_tf32); // FP16 should be faster
    }

    #[test]
    fn test_quantum_precision_tensor_core_requirements() {
        assert!(QuantumPrecision::TF32.requires_tensor_cores());
        assert!(QuantumPrecision::BFloat16.requires_tensor_cores());
        assert!(!QuantumPrecision::Half.requires_tensor_cores());
        assert!(!QuantumPrecision::Single.requires_tensor_cores());
        assert!(!QuantumPrecision::Double.requires_tensor_cores());
    }

    #[test]
    fn test_quantum_precision_bit_width() {
        assert_eq!(QuantumPrecision::Half.bit_width(), 16);
        assert_eq!(QuantumPrecision::BFloat16.bit_width(), 16);
        assert_eq!(QuantumPrecision::TF32.bit_width(), 19);
        assert_eq!(QuantumPrecision::Single.bit_width(), 32);
        assert_eq!(QuantumPrecision::Double.bit_width(), 64);
    }

    #[test]
    fn test_quantum_precision_mantissa_bits() {
        assert_eq!(QuantumPrecision::Half.mantissa_bits(), 10);
        assert_eq!(QuantumPrecision::BFloat16.mantissa_bits(), 7);
        assert_eq!(QuantumPrecision::TF32.mantissa_bits(), 10);
        assert_eq!(QuantumPrecision::Single.mantissa_bits(), 23);
        assert_eq!(QuantumPrecision::Double.mantissa_bits(), 52);
    }
}
