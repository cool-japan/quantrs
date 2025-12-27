//! Data encoding strategies for quantum machine learning
//!
//! This module provides various methods to encode classical data
//! into quantum states for processing by quantum circuits.

use super::EncodingStrategy;
use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{single::Hadamard, GateOp},
    parametric::{ParametricRotationY, ParametricRotationZ},
    qubit::QubitId,
};
use scirs2_core::Complex64;
use std::f64::consts::PI;

// Type aliases for convenience
#[allow(dead_code)]
type RYGate = ParametricRotationY;
#[allow(dead_code)]
type RZGate = ParametricRotationZ;

// Simple CNOT gate for encoding usage
#[derive(Debug, Clone, Copy)]
struct CNOT {
    control: QubitId,
    target: QubitId,
}

impl GateOp for CNOT {
    fn name(&self) -> &'static str {
        "CNOT"
    }

    fn qubits(&self) -> Vec<QubitId> {
        vec![self.control, self.target]
    }

    fn matrix(&self) -> crate::error::QuantRS2Result<Vec<Complex64>> {
        Ok(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ])
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_gate(&self) -> Box<dyn GateOp> {
        Box::new(*self)
    }
}

/// Data encoder for quantum circuits
pub struct DataEncoder {
    /// Encoding strategy
    strategy: EncodingStrategy,
    /// Number of qubits
    num_qubits: usize,
    /// Number of features that can be encoded
    num_features: usize,
}

impl DataEncoder {
    /// Create a new data encoder
    pub const fn new(strategy: EncodingStrategy, num_qubits: usize) -> Self {
        let num_features = match strategy {
            EncodingStrategy::Amplitude => 1 << num_qubits, // 2^n amplitudes
            EncodingStrategy::Angle | EncodingStrategy::Basis => num_qubits, // One per qubit
            EncodingStrategy::IQP => num_qubits * (num_qubits + 1) / 2, // All pairs
        };

        Self {
            strategy,
            num_qubits,
            num_features,
        }
    }

    /// Get the number of features this encoder can handle
    pub const fn num_features(&self) -> usize {
        self.num_features
    }

    /// Encode classical data into quantum gates
    pub fn encode(&self, data: &[f64]) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        if data.len() != self.num_features {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Expected {} features, got {}",
                self.num_features,
                data.len()
            )));
        }

        match self.strategy {
            EncodingStrategy::Amplitude => self.amplitude_encoding(data),
            EncodingStrategy::Angle => self.angle_encoding(data),
            EncodingStrategy::IQP => self.iqp_encoding(data),
            EncodingStrategy::Basis => self.basis_encoding(data),
        }
    }

    /// Amplitude encoding: encode data in state amplitudes
    fn amplitude_encoding(&self, data: &[f64]) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        // Normalize data
        let norm: f64 = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            return Err(QuantRS2Error::InvalidInput(
                "Cannot encode zero vector".to_string(),
            ));
        }

        let _normalized: Vec<f64> = data.iter().map(|x| x / norm).collect();

        // For amplitude encoding, we need to prepare a state with given amplitudes
        // This is complex and typically requires decomposition into gates
        // For now, return a placeholder

        let mut gates: Vec<Box<dyn GateOp>> = vec![];

        // Start with uniform superposition
        for i in 0..self.num_qubits {
            gates.push(Box::new(Hadamard {
                target: QubitId(i as u32),
            }));
        }

        // Would need to implement state preparation algorithm here
        // This is a non-trivial operation requiring careful decomposition

        Ok(gates)
    }

    /// Angle encoding: encode data as rotation angles
    fn angle_encoding(&self, data: &[f64]) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut gates: Vec<Box<dyn GateOp>> = vec![];

        // Apply Hadamard gates first for superposition
        for i in 0..self.num_qubits {
            gates.push(Box::new(Hadamard {
                target: QubitId(i as u32),
            }));
        }

        // Encode each feature as a rotation angle
        for (i, &value) in data.iter().enumerate() {
            let qubit = QubitId(i as u32);
            // Scale data to [0, 2π]
            let angle = value * PI;
            gates.push(Box::new(ParametricRotationY::new(qubit, angle)));
        }

        Ok(gates)
    }

    /// IQP (Instantaneous Quantum Polynomial) encoding
    fn iqp_encoding(&self, data: &[f64]) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut gates: Vec<Box<dyn GateOp>> = vec![];

        // Apply Hadamard gates
        for i in 0..self.num_qubits {
            gates.push(Box::new(Hadamard {
                target: QubitId(i as u32),
            }));
        }

        // Single-qubit rotations
        for i in 0..self.num_qubits {
            let angle = data[i] * PI;
            gates.push(Box::new(ParametricRotationZ::new(QubitId(i as u32), angle)));
        }

        // Two-qubit interactions
        let mut idx = self.num_qubits;
        for i in 0..self.num_qubits {
            for j in i + 1..self.num_qubits {
                if idx < data.len() {
                    let angle = data[idx] * PI;
                    // Would implement RZZ gate here
                    // For now, use two RZ gates as placeholder
                    gates.push(Box::new(ParametricRotationZ::new(
                        QubitId(i as u32),
                        angle / 2.0,
                    )));
                    gates.push(Box::new(ParametricRotationZ::new(
                        QubitId(j as u32),
                        angle / 2.0,
                    )));
                    idx += 1;
                }
            }
        }

        Ok(gates)
    }

    /// Basis encoding: encode binary data in computational basis
    fn basis_encoding(&self, data: &[f64]) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        use crate::gate::single::PauliX;

        let mut gates: Vec<Box<dyn GateOp>> = vec![];

        // Encode each bit
        for (i, &value) in data.iter().enumerate() {
            if value.abs() > 0.5 {
                // Threshold at 0.5
                gates.push(Box::new(PauliX {
                    target: QubitId(i as u32),
                }));
            }
        }

        Ok(gates)
    }
}

/// Feature map for kernel methods
pub struct FeatureMap {
    /// Number of qubits
    num_qubits: usize,
    /// Number of features
    num_features: usize,
    /// Feature map type
    map_type: FeatureMapType,
    /// Number of repetitions
    reps: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum FeatureMapType {
    /// Pauli feature map
    Pauli,
    /// Z feature map
    ZFeature,
    /// ZZ feature map
    ZZFeature,
    /// Custom feature map
    Custom,
}

impl FeatureMap {
    /// Create a new feature map
    pub const fn new(num_qubits: usize, map_type: FeatureMapType, reps: usize) -> Self {
        let num_features = num_qubits; // All feature map types use one feature per qubit

        Self {
            num_qubits,
            num_features,
            map_type,
            reps,
        }
    }

    /// Create gates for the feature map
    pub fn create_gates(&self, features: &[f64]) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        if features.len() != self.num_features {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Expected {} features, got {}",
                self.num_features,
                features.len()
            )));
        }

        let mut gates = vec![];

        for _ in 0..self.reps {
            match self.map_type {
                FeatureMapType::Pauli => {
                    gates.extend(self.pauli_feature_map(features)?);
                }
                FeatureMapType::ZFeature => {
                    gates.extend(self.z_feature_map(features)?);
                }
                FeatureMapType::ZZFeature => {
                    gates.extend(self.zz_feature_map(features)?);
                }
                FeatureMapType::Custom => {
                    // Custom implementation would go here
                }
            }
        }

        Ok(gates)
    }

    /// Pauli feature map
    fn pauli_feature_map(&self, features: &[f64]) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut gates: Vec<Box<dyn GateOp>> = vec![];

        // Apply Hadamard gates
        for i in 0..self.num_qubits {
            gates.push(Box::new(Hadamard {
                target: QubitId(i as u32),
            }));
        }

        // Apply rotations based on features
        for (i, &feature) in features.iter().enumerate() {
            gates.push(Box::new(ParametricRotationZ::new(
                QubitId(i as u32),
                2.0 * feature,
            )));
        }

        Ok(gates)
    }

    /// Z feature map
    fn z_feature_map(&self, features: &[f64]) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut gates: Vec<Box<dyn GateOp>> = vec![];

        // First layer: Hadamard gates
        for i in 0..self.num_qubits {
            gates.push(Box::new(Hadamard {
                target: QubitId(i as u32),
            }));
        }

        // Second layer: RZ rotations
        for (i, &feature) in features.iter().enumerate() {
            gates.push(Box::new(ParametricRotationZ::new(
                QubitId(i as u32),
                2.0 * feature,
            )));
        }

        Ok(gates)
    }

    /// ZZ feature map
    fn zz_feature_map(&self, features: &[f64]) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut gates = self.z_feature_map(features)?;

        // Add entangling gates
        for i in 0..self.num_qubits - 1 {
            gates.push(Box::new(CNOT {
                control: QubitId(i as u32),
                target: QubitId((i + 1) as u32),
            }));

            // Two-qubit rotation
            let angle = (PI - features[i]) * (PI - features[i + 1]);
            gates.push(Box::new(ParametricRotationZ::new(
                QubitId((i + 1) as u32),
                angle,
            )));

            gates.push(Box::new(CNOT {
                control: QubitId(i as u32),
                target: QubitId((i + 1) as u32),
            }));
        }

        Ok(gates)
    }
}

/// Data re-uploading strategy
pub struct DataReuploader {
    /// Base encoder
    encoder: DataEncoder,
    /// Number of layers to repeat encoding
    num_layers: usize,
    /// Whether to use different parameters per layer
    trainable_scaling: bool,
}

impl DataReuploader {
    /// Create a new data re-uploader
    pub const fn new(encoder: DataEncoder, num_layers: usize, trainable_scaling: bool) -> Self {
        Self {
            encoder,
            num_layers,
            trainable_scaling,
        }
    }

    /// Create gates with data re-uploading
    pub fn create_gates(
        &self,
        data: &[f64],
        scaling_params: Option<&[f64]>,
    ) -> QuantRS2Result<Vec<Vec<Box<dyn GateOp>>>> {
        let mut layers = vec![];

        for layer in 0..self.num_layers {
            let scaled_data = if self.trainable_scaling {
                if let Some(params) = scaling_params {
                    let offset = layer * data.len();
                    if offset + data.len() > params.len() {
                        return Err(QuantRS2Error::InvalidInput(
                            "Not enough scaling parameters".to_string(),
                        ));
                    }

                    data.iter()
                        .zip(&params[offset..offset + data.len()])
                        .map(|(d, p)| d * p)
                        .collect()
                } else {
                    data.to_vec()
                }
            } else {
                data.to_vec()
            };

            layers.push(self.encoder.encode(&scaled_data)?);
        }

        Ok(layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angle_encoding() {
        let encoder = DataEncoder::new(EncodingStrategy::Angle, 3);
        assert_eq!(encoder.num_features(), 3);

        let data = vec![0.5, 1.0, 0.0];
        let gates = encoder.encode(&data).expect("Failed to encode angle data");

        // Should have 3 Hadamards + 3 RY gates
        assert_eq!(gates.len(), 6);
    }

    #[test]
    fn test_basis_encoding() {
        let encoder = DataEncoder::new(EncodingStrategy::Basis, 4);
        assert_eq!(encoder.num_features(), 4);

        let data = vec![1.0, 0.0, 1.0, 0.0];
        let gates = encoder.encode(&data).expect("Failed to encode basis data");

        // Should have 2 X gates (for the 1.0 values)
        assert_eq!(gates.len(), 2);
    }

    #[test]
    fn test_feature_map() {
        let feature_map = FeatureMap::new(2, FeatureMapType::ZFeature, 1);
        let features = vec![0.5, 0.7];

        let gates = feature_map
            .create_gates(&features)
            .expect("Failed to create feature map gates");

        // Should have 2 Hadamards + 2 RZ gates
        assert_eq!(gates.len(), 4);
    }

    #[test]
    fn test_data_reuploader() {
        let encoder = DataEncoder::new(EncodingStrategy::Angle, 2);
        let reuploader = DataReuploader::new(encoder, 3, false);

        let data = vec![0.5, 0.5];
        let layers = reuploader
            .create_gates(&data, None)
            .expect("Failed to create reuploader gates");

        assert_eq!(layers.len(), 3); // 3 layers
        assert_eq!(layers[0].len(), 4); // Each layer has 2 H + 2 RY
    }
}
