//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{
    MixerType, ParameterInitialization, QaoaConfig, QaoaError, QaoaOptimizer, QaoaVariant,
    QuantumState,
};

/// Result type for QAOA operations
pub type QaoaResult<T> = Result<T, QaoaError>;
/// Complex number type alias for quantum state amplitudes
pub mod complex {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Complex64 {
        pub re: f64,
        pub im: f64,
    }
    impl Complex64 {
        #[must_use]
        pub const fn new(re: f64, im: f64) -> Self {
            Self { re, im }
        }
        #[must_use]
        pub fn norm_squared(&self) -> f64 {
            self.re.mul_add(self.re, self.im * self.im)
        }
        #[must_use]
        pub fn abs(&self) -> f64 {
            self.re.hypot(self.im)
        }
        #[must_use]
        pub fn conj(&self) -> Self {
            Self {
                re: self.re,
                im: -self.im,
            }
        }
    }
    impl std::ops::Add for Complex64 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self {
                re: self.re + rhs.re,
                im: self.im + rhs.im,
            }
        }
    }
    impl std::ops::Mul for Complex64 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self {
                re: self.re.mul_add(rhs.re, -(self.im * rhs.im)),
                im: self.re.mul_add(rhs.im, self.im * rhs.re),
            }
        }
    }
    impl std::ops::Mul<f64> for Complex64 {
        type Output = Self;
        fn mul(self, rhs: f64) -> Self {
            Self {
                re: self.re * rhs,
                im: self.im * rhs,
            }
        }
    }
}
/// Helper functions for creating common QAOA configurations
/// Create a standard QAOA configuration
#[must_use]
pub fn create_standard_qaoa_config(layers: usize, shots: usize) -> QaoaConfig {
    QaoaConfig {
        variant: QaoaVariant::Standard { layers },
        num_shots: shots,
        ..Default::default()
    }
}
/// Create a QAOA+ configuration with multi-angle mixers
#[must_use]
pub fn create_qaoa_plus_config(layers: usize, shots: usize) -> QaoaConfig {
    QaoaConfig {
        variant: QaoaVariant::QaoaPlus {
            layers,
            multi_angle: true,
        },
        mixer_type: MixerType::XMixer,
        num_shots: shots,
        ..Default::default()
    }
}
/// Create a warm-start QAOA configuration
#[must_use]
pub fn create_warm_start_qaoa_config(
    layers: usize,
    initial_solution: Vec<i8>,
    shots: usize,
) -> QaoaConfig {
    QaoaConfig {
        variant: QaoaVariant::WarmStart {
            layers,
            initial_solution: initial_solution.clone(),
        },
        parameter_init: ParameterInitialization::WarmStart {
            solution: initial_solution,
        },
        num_shots: shots,
        ..Default::default()
    }
}
/// Create a QAOA configuration with XY mixer for constrained problems
#[must_use]
pub fn create_constrained_qaoa_config(layers: usize, shots: usize) -> QaoaConfig {
    QaoaConfig {
        variant: QaoaVariant::Standard { layers },
        mixer_type: MixerType::XYMixer,
        num_shots: shots,
        ..Default::default()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_qaoa_config_creation() {
        let config = create_standard_qaoa_config(3, 1000);
        match config.variant {
            QaoaVariant::Standard { layers } => {
                assert_eq!(layers, 3);
            }
            _ => panic!("Expected Standard QAOA variant"),
        }
        assert_eq!(config.num_shots, 1000);
    }
    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::new(3);
        assert_eq!(state.num_qubits, 3);
        assert_eq!(state.amplitudes.len(), 8);
        assert_eq!(state.get_probability(0), 1.0);
        for i in 1..8 {
            assert_eq!(state.get_probability(i), 0.0);
        }
    }
    #[test]
    fn test_uniform_superposition() {
        let state = QuantumState::uniform_superposition(2);
        assert_eq!(state.num_qubits, 2);
        for i in 0..4 {
            assert!((state.get_probability(i) - 0.25).abs() < 1e-10);
        }
    }
    #[test]
    fn test_bitstring_to_spins() {
        let state = QuantumState::new(3);
        let spins = state.bitstring_to_spins(0b101);
        assert_eq!(spins, vec![1, -1, 1]);
    }
    #[test]
    fn test_qaoa_optimizer_creation() {
        let config = create_standard_qaoa_config(2, 100);
        let optimizer = QaoaOptimizer::new(config);
        assert!(optimizer.is_ok());
    }
}
