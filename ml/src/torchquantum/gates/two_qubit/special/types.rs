//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::torchquantum::{
    CType, NParamsEnum, OpHistoryEntry, TQDevice, TQModule, TQOperator, TQParameter, WiresEnum,
};
use scirs2_core::ndarray::{Array2, ArrayD, IxDyn};

/// CY gate - Controlled Y gate
#[derive(Debug, Clone)]
pub struct TQCY {
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQCY {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}
/// iSWAP gate - swaps qubits and applies i phase
#[derive(Debug, Clone)]
pub struct TQiSWAP {
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQiSWAP {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}
/// DCX gate - Double CNOT gate
#[derive(Debug, Clone)]
pub struct TQDCX {
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQDCX {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}
/// XXMinusYY gate - parameterized (XX - YY) interaction
#[derive(Debug, Clone)]
pub struct TQXXMinusYY {
    pub(super) params: Option<TQParameter>,
    pub(super) has_params: bool,
    pub(super) trainable: bool,
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQXXMinusYY {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(
                ArrayD::zeros(IxDyn(&[1, 2])),
                "xxmyy_params",
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
    pub fn with_init_params(mut self, theta: f64, beta: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = theta;
            p.data[[0, 1]] = beta;
        }
        self
    }
}
/// ECR gate - Echoed Cross-Resonance gate
#[derive(Debug, Clone)]
pub struct TQECR {
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQECR {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}
/// Phase shift gate (also known as P gate)
///
/// This is a parameterized version of the phase gate that applies
/// a phase shift to the |1⟩ state.
///
/// Matrix representation:
/// ```text
/// [[1,   0,        0,           0],
///  [0,   1,        0,           0],
///  [0,   0,        1,           0],
///  [0,   0,        0,   e^{iφ}]]
/// ```
///
/// Note: This is the two-qubit controlled version. For single-qubit
/// phase shift, use the P gate in single_qubit module.
#[derive(Debug, Clone)]
pub struct TQPhaseShift2 {
    pub(super) params: Option<TQParameter>,
    pub(super) has_params: bool,
    pub(super) trainable: bool,
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQPhaseShift2 {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(
                ArrayD::zeros(IxDyn(&[1, 1])),
                "phaseshift2_phi",
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
    /// Create with initial phase parameter
    pub fn with_init_params(mut self, phi: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = phi;
        }
        self
    }
}
/// CPhase gate - Controlled phase gate (also known as CU1)
#[derive(Debug, Clone)]
pub struct TQCPhase {
    pub(super) params: Option<TQParameter>,
    pub(super) has_params: bool,
    pub(super) trainable: bool,
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQCPhase {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(
                ArrayD::zeros(IxDyn(&[1, 1])),
                "cphase_theta",
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
    pub fn with_init_params(mut self, theta: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = theta;
        }
        self
    }
}
/// SSWAP gate - Square root SWAP gate
#[derive(Debug, Clone)]
pub struct TQSSWAP {
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQSSWAP {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}
/// CH gate - Controlled Hadamard gate
#[derive(Debug, Clone)]
pub struct TQCH {
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQCH {
    pub fn new() -> Self {
        Self {
            inverse: false,
            static_mode: false,
        }
    }
}
/// fSim gate - Google's fermionic simulation gate
///
/// This gate is used in quantum chemistry simulations and is native
/// to Google's Sycamore processor. It combines an iSWAP-like interaction
/// with a controlled phase.
///
/// Matrix representation:
/// ```text
/// [[1,           0,                0,           0        ],
///  [0,           cos(θ),           -i·sin(θ),   0        ],
///  [0,           -i·sin(θ),        cos(θ),      0        ],
///  [0,           0,                0,           e^{-iφ}  ]]
/// ```
#[derive(Debug, Clone)]
pub struct TQFSimGate {
    pub(super) params: Option<TQParameter>,
    pub(super) has_params: bool,
    pub(super) trainable: bool,
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQFSimGate {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(
                ArrayD::zeros(IxDyn(&[1, 2])),
                "fsim_params",
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
    /// Create with initial parameters (theta, phi)
    pub fn with_init_params(mut self, theta: f64, phi: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = theta;
            p.data[[0, 1]] = phi;
        }
        self
    }
    /// Create a full iSWAP (theta=π/2, phi=0)
    pub fn iswap_like() -> Self {
        Self::new(true, false).with_init_params(std::f64::consts::FRAC_PI_2, 0.0)
    }
    /// Create a sqrt-iSWAP (theta=π/4, phi=0)
    pub fn sqrt_iswap() -> Self {
        Self::new(true, false).with_init_params(std::f64::consts::FRAC_PI_4, 0.0)
    }
    /// Create Sycamore gate (theta≈π/2, phi≈π/6) - Google's native gate
    pub fn sycamore() -> Self {
        Self::new(true, false)
            .with_init_params(std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_6)
    }
}
/// Givens rotation gate - fundamental for quantum chemistry
///
/// The Givens rotation performs a rotation in the (i,j) subspace of the Hilbert space.
/// It is widely used in molecular orbital transformations and VQE circuits for chemistry.
///
/// G(θ, φ) = exp(-i·θ/2·(e^{iφ}·|01⟩⟨10| + e^{-iφ}·|10⟩⟨01|))
///
/// Matrix representation:
/// ```text
/// [[1,   0,                   0,                   0],
///  [0,   cos(θ/2),            -e^{iφ}·sin(θ/2),   0],
///  [0,   e^{-iφ}·sin(θ/2),    cos(θ/2),           0],
///  [0,   0,                   0,                   1]]
/// ```
#[derive(Debug, Clone)]
pub struct TQGivensRotation {
    pub(super) params: Option<TQParameter>,
    pub(super) has_params: bool,
    pub(super) trainable: bool,
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQGivensRotation {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(
                ArrayD::zeros(IxDyn(&[1, 2])),
                "givens_params",
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
    /// Create with initial parameters (theta, phi)
    pub fn with_init_params(mut self, theta: f64, phi: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = theta;
            p.data[[0, 1]] = phi;
        }
        self
    }
    /// Create a real Givens rotation (phi=0)
    pub fn real(theta: f64) -> Self {
        Self::new(true, false).with_init_params(theta, 0.0)
    }
    /// Create a complex Givens rotation
    pub fn complex(theta: f64, phi: f64) -> Self {
        Self::new(true, false).with_init_params(theta, phi)
    }
}
/// General controlled rotation gate
///
/// Applies a controlled rotation around an arbitrary axis.
/// CRot(theta, phi, omega) = CR_z(omega) @ CR_y(phi) @ CR_z(theta)
///
/// This is the controlled version of the general U3 rotation.
#[derive(Debug, Clone)]
pub struct TQControlledRot {
    pub(super) params: Option<TQParameter>,
    pub(super) has_params: bool,
    pub(super) trainable: bool,
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQControlledRot {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(
                ArrayD::zeros(IxDyn(&[1, 3])),
                "crot_params",
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
    /// Create with initial parameters (theta, phi, omega)
    pub fn with_init_params(mut self, theta: f64, phi: f64, omega: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = theta;
            p.data[[0, 1]] = phi;
            p.data[[0, 2]] = omega;
        }
        self
    }
}
/// XXPlusYY gate - parameterized (XX + YY) interaction
#[derive(Debug, Clone)]
pub struct TQXXPlusYY {
    pub(super) params: Option<TQParameter>,
    pub(super) has_params: bool,
    pub(super) trainable: bool,
    pub(super) inverse: bool,
    pub(super) static_mode: bool,
}
impl TQXXPlusYY {
    pub fn new(has_params: bool, trainable: bool) -> Self {
        let params = if has_params {
            Some(TQParameter::new(
                ArrayD::zeros(IxDyn(&[1, 2])),
                "xxpyy_params",
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
    pub fn with_init_params(mut self, theta: f64, beta: f64) -> Self {
        if let Some(ref mut p) = self.params {
            p.data[[0, 0]] = theta;
            p.data[[0, 1]] = beta;
        }
        self
    }
}
