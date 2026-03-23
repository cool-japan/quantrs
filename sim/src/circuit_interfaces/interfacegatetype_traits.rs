//! # InterfaceGateType - Trait Implementations
//!
//! This module contains trait implementations for `InterfaceGateType`.
//!
//! ## Implemented Traits
//!
//! - `Hash`
//! - `Eq`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::hash::{Hash, Hasher};

use super::types::InterfaceGateType;

impl std::hash::Hash for InterfaceGateType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        use std::mem;
        match self {
            Self::Identity => 0u8.hash(state),
            Self::PauliX => 1u8.hash(state),
            Self::X => 2u8.hash(state),
            Self::PauliY => 3u8.hash(state),
            Self::PauliZ => 4u8.hash(state),
            Self::Hadamard => 5u8.hash(state),
            Self::H => 6u8.hash(state),
            Self::S => 7u8.hash(state),
            Self::T => 8u8.hash(state),
            Self::Phase(angle) => {
                9u8.hash(state);
                angle.to_bits().hash(state);
            }
            Self::RX(angle) => {
                10u8.hash(state);
                angle.to_bits().hash(state);
            }
            Self::RY(angle) => {
                11u8.hash(state);
                angle.to_bits().hash(state);
            }
            Self::RZ(angle) => {
                12u8.hash(state);
                angle.to_bits().hash(state);
            }
            Self::U1(angle) => {
                13u8.hash(state);
                angle.to_bits().hash(state);
            }
            Self::U2(theta, phi) => {
                14u8.hash(state);
                theta.to_bits().hash(state);
                phi.to_bits().hash(state);
            }
            Self::U3(theta, phi, lambda) => {
                15u8.hash(state);
                theta.to_bits().hash(state);
                phi.to_bits().hash(state);
                lambda.to_bits().hash(state);
            }
            Self::CNOT => 16u8.hash(state),
            Self::CZ => 17u8.hash(state),
            Self::CY => 18u8.hash(state),
            Self::SWAP => 19u8.hash(state),
            Self::ISwap => 20u8.hash(state),
            Self::CRX(angle) => {
                21u8.hash(state);
                angle.to_bits().hash(state);
            }
            Self::CRY(angle) => {
                22u8.hash(state);
                angle.to_bits().hash(state);
            }
            Self::CRZ(angle) => {
                23u8.hash(state);
                angle.to_bits().hash(state);
            }
            Self::CPhase(angle) => {
                24u8.hash(state);
                angle.to_bits().hash(state);
            }
            Self::Toffoli => 25u8.hash(state),
            Self::Fredkin => 26u8.hash(state),
            Self::MultiControlledX(n) => {
                27u8.hash(state);
                n.hash(state);
            }
            Self::MultiControlledZ(n) => {
                28u8.hash(state);
                n.hash(state);
            }
            Self::Custom(name, matrix) => {
                29u8.hash(state);
                name.hash(state);
                matrix.shape().hash(state);
            }
            Self::Measure => 30u8.hash(state),
            Self::Reset => 31u8.hash(state),
        }
    }
}

impl Eq for InterfaceGateType {}
