//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::super::{
    CType, NParamsEnum, OpHistoryEntry, TQDevice, TQModule, TQOperator, TQParameter, WiresEnum,
};
use scirs2_core::ndarray::{Array1, Array2, ArrayD, IxDyn};

/// Rotation-X gate
#[derive(Debug, Clone)]
pub struct TQRx {
    pub(super) params: Option<TQParameter>,
    pub(super) has_params: bool,
    pub(super) trainable: bool,
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQRx {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(ArrayD::zeros(IxDyn(&[1, 1])), "rx_params"))
        } else {
            None
        };
        Self {
            params,
            has_params,
            trainable,
            inverse: false,
            static_mode: false,
        }
    }
    pub fn with_init_params(mut self, init_params: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.init_constant(init_params);
        }
        self
    }
}
/// Rotation-Z gate
#[derive(Debug, Clone)]
pub struct TQRz {
    pub(super) params: Option<TQParameter>,
    pub(super) has_params: bool,
    pub(super) trainable: bool,
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQRz {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(ArrayD::zeros(IxDyn(&[1, 1])), "rz_params"))
        } else {
            None
        };
        Self {
            params,
            has_params,
            trainable,
            inverse: false,
            static_mode: false,
        }
    }
    pub fn with_init_params(mut self, init_params: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.init_constant(init_params);
        }
        self
    }
}
/// Rotation-Y gate
#[derive(Debug, Clone)]
pub struct TQRy {
    pub(super) params: Option<TQParameter>,
    pub(super) has_params: bool,
    pub(super) trainable: bool,
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQRy {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(ArrayD::zeros(IxDyn(&[1, 1])), "ry_params"))
        } else {
            None
        };
        Self {
            params,
            has_params,
            trainable,
            inverse: false,
            static_mode: false,
        }
    }
    pub fn with_init_params(mut self, init_params: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.init_constant(init_params);
        }
        self
    }
}
/// Pauli-X gate
#[derive(Debug, Clone)]
pub struct TQPauliX {
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQPauliX {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}
/// SX gate (sqrt(X))
#[derive(Debug, Clone)]
pub struct TQSX {
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQSX {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}
/// U1 gate (phase gate) - U1(λ) = diag(1, e^(iλ))
#[derive(Debug, Clone)]
pub struct TQU1 {
    pub(super) params: Option<TQParameter>,
    pub(super) has_params: bool,
    pub(super) trainable: bool,
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQU1 {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(ArrayD::zeros(IxDyn(&[1, 1])), "u1_params"))
        } else {
            None
        };
        Self {
            params,
            has_params,
            trainable,
            inverse: false,
            static_mode: false,
        }
    }
    pub fn with_init_params(mut self, lambda: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.init_constant(lambda);
        }
        self
    }
}
/// U3 gate - Universal single-qubit gate
/// U3(θ, φ, λ) = [[cos(θ/2), -e^(iλ)sin(θ/2)], [e^(iφ)sin(θ/2), e^(i(φ+λ))cos(θ/2)]]
#[derive(Debug, Clone)]
pub struct TQU3 {
    pub(super) params: Option<TQParameter>,
    pub(super) has_params: bool,
    pub(super) trainable: bool,
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQU3 {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(ArrayD::zeros(IxDyn(&[1, 3])), "u3_params"))
        } else {
            None
        };
        Self {
            params,
            has_params,
            trainable,
            inverse: false,
            static_mode: false,
        }
    }
    pub fn with_init_params(mut self, theta: f64, phi: f64, lambda: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = theta;
            p.data[[0, 1]] = phi;
            p.data[[0, 2]] = lambda;
        }
        self
    }
}
/// Hadamard gate
#[derive(Debug, Clone)]
pub struct TQHadamard {
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQHadamard {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}
/// T gate (fourth root of Z)
#[derive(Debug, Clone)]
pub struct TQT {
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQT {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}
/// U2 gate - U2(φ, λ) = (1/√2) * [[1, -e^(iλ)], [e^(iφ), e^(i(φ+λ))]]
#[derive(Debug, Clone)]
pub struct TQU2 {
    pub(super) params: Option<TQParameter>,
    pub(super) has_params: bool,
    pub(super) trainable: bool,
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQU2 {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(ArrayD::zeros(IxDyn(&[1, 2])), "u2_params"))
        } else {
            None
        };
        Self {
            params,
            has_params,
            trainable,
            inverse: false,
            static_mode: false,
        }
    }
    pub fn with_init_params(mut self, phi: f64, lambda: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = phi;
            p.data[[0, 1]] = lambda;
        }
        self
    }
}
/// S gate (sqrt(Z))
#[derive(Debug, Clone)]
pub struct TQS {
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQS {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}
/// Identity gate - does nothing but is useful for circuit structure
#[derive(Debug, Clone)]
pub struct TQI {
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQI {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}
/// Global phase gate - applies a global phase to the state
#[derive(Debug, Clone)]
pub struct TQGlobalPhase {
    pub(super) params: Option<TQParameter>,
    pub(super) has_params: bool,
    pub(super) trainable: bool,
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQGlobalPhase {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(
                ArrayD::zeros(IxDyn(&[1, 1])),
                "global_phase_params",
            ))
        } else {
            None
        };
        Self {
            params,
            has_params,
            trainable,
            inverse: false,
            static_mode: false,
        }
    }
}
/// Pauli-Z gate
#[derive(Debug, Clone)]
pub struct TQPauliZ {
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQPauliZ {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}
/// Pauli-Y gate
#[derive(Debug, Clone)]
pub struct TQPauliY {
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQPauliY {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}
