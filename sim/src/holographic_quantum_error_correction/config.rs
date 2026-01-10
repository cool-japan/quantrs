//! Configuration types for holographic quantum error correction.
//!
//! This module contains configuration structures and enums for the holographic
//! quantum error correction framework.

use crate::quantum_gravity_simulation::AdSCFTConfig;
use serde::{Deserialize, Serialize};

/// Holographic quantum error correction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolographicQECConfig {
    /// AdS/CFT configuration for holographic duality
    pub ads_cft_config: AdSCFTConfig,
    /// Number of boundary qubits
    pub boundary_qubits: usize,
    /// Number of bulk qubits (typically exponentially larger)
    pub bulk_qubits: usize,
    /// `AdS` radius for geometry
    pub ads_radius: f64,
    /// Central charge of boundary CFT
    pub central_charge: f64,
    /// Holographic error correction code type
    pub error_correction_code: HolographicCodeType,
    /// Bulk reconstruction method
    pub reconstruction_method: BulkReconstructionMethod,
    /// Error correction threshold
    pub error_threshold: f64,
    /// Enable geometric protection
    pub geometric_protection: bool,
    /// Entanglement entropy threshold
    pub entanglement_threshold: f64,
    /// Number of Ryu-Takayanagi surfaces
    pub rt_surfaces: usize,
    /// Enable quantum error correction
    pub enable_qec: bool,
    /// Operator reconstruction accuracy
    pub reconstruction_accuracy: f64,
}

impl Default for HolographicQECConfig {
    fn default() -> Self {
        Self {
            ads_cft_config: AdSCFTConfig::default(),
            boundary_qubits: 8,
            bulk_qubits: 20,
            ads_radius: 1.0,
            central_charge: 12.0,
            error_correction_code: HolographicCodeType::AdSRindler,
            reconstruction_method: BulkReconstructionMethod::HKLL,
            error_threshold: 0.01,
            geometric_protection: true,
            entanglement_threshold: 0.1,
            rt_surfaces: 10,
            enable_qec: true,
            reconstruction_accuracy: 1e-6,
        }
    }
}

/// Types of holographic quantum error correction codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HolographicCodeType {
    /// AdS-Rindler code
    AdSRindler,
    /// Holographic stabilizer code
    HolographicStabilizer,
    /// Quantum error correction with bulk geometry
    BulkGeometry,
    /// Tensor network error correction
    TensorNetwork,
    /// Holographic surface code
    HolographicSurface,
    /// Perfect tensor network code
    PerfectTensor,
    /// Holographic entanglement entropy code
    EntanglementEntropy,
    /// AdS/CFT quantum error correction
    AdSCFTCode,
}

/// Methods for bulk reconstruction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BulkReconstructionMethod {
    /// Hamilton-Kabat-Lifschytz-Lowe (HKLL) reconstruction
    HKLL,
    /// Entanglement wedge reconstruction
    EntanglementWedge,
    /// Quantum error correction reconstruction
    QECReconstruction,
    /// Tensor network reconstruction
    TensorNetwork,
    /// Holographic tensor network
    HolographicTensorNetwork,
    /// Bulk boundary dictionary
    BulkBoundaryDictionary,
    /// Minimal surface reconstruction
    MinimalSurface,
}
