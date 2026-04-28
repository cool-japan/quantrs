//! Quantum communication protocols.
//!
//! Implements three canonical quantum networking protocols:
//! - **BB84 QKD**: Bennett-Brassard 1984 quantum key distribution with eavesdropping detection
//! - **E91 QKD**: Ekert 1991 entanglement-based QKD with CHSH Bell test
//! - **Quantum teleportation + entanglement swapping**: single-hop and n-hop chain

pub mod bb84;
pub mod channel;
pub mod e91;
pub mod teleportation;

pub use bb84::{Bb84Protocol, Bb84Result};
pub use channel::{
    AmplitudeDampingChannel, DephazingChannel, DepolarizingChannel, NoiseChannel,
};
pub use e91::{E91Protocol, E91Result};
pub use teleportation::{EntanglementSwapping, SwappingResult, TeleportationProtocol, TeleportationResult};
