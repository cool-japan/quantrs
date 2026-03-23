//! Gate and quantum system characterization
//!
//! This module provides comprehensive tools for analyzing and characterizing quantum gates
//! and quantum systems using their eigenstructure and other advanced techniques. This is useful for:
//! - Gate synthesis and decomposition
//! - Identifying gate types and parameters
//! - Optimizing gate sequences
//! - Verifying gate implementations
//! - Quantum volume measurement
//! - Quantum process tomography
//! - Noise characterization and mitigation

pub mod gate_characterizer;
pub mod noise_characterizer;
pub mod noise_model;
pub mod quantum_volume;
pub mod tomography;

pub use gate_characterizer::{GateCharacterizer, GateEigenstructure, GateType};
pub use noise_characterizer::{
    MitigationResult, MitigationTechnique, NoiseCharacterizationResult, NoiseCharacterizer,
    NoiseMitigator,
};
pub use noise_model::NoiseModel;
pub use quantum_volume::{
    QuantumVolumeConfig, QuantumVolumeMeasurement, QuantumVolumeResult, RandomGate,
    RandomQuantumCircuit,
};
pub use tomography::{
    ProcessBasis, ProcessTomography, ProcessTomographyConfig, ProcessTomographyResult,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::{single::*, GateOp};
    use crate::qubit::QubitId;
    use scirs2_core::Complex64 as Complex;
    use std::f64::consts::PI;

    #[test]
    fn test_pauli_identification() {
        let characterizer = GateCharacterizer::new(1e-10);

        assert_eq!(
            characterizer
                .identify_gate_type(&PauliX { target: QubitId(0) })
                .expect("identify PauliX failed"),
            GateType::PauliX
        );
        assert_eq!(
            characterizer
                .identify_gate_type(&PauliY { target: QubitId(0) })
                .expect("identify PauliY failed"),
            GateType::PauliY
        );
        assert_eq!(
            characterizer
                .identify_gate_type(&PauliZ { target: QubitId(0) })
                .expect("identify PauliZ failed"),
            GateType::PauliZ
        );
    }

    #[test]
    fn test_rotation_decomposition() {
        let characterizer = GateCharacterizer::new(1e-10);
        let rx = RotationX {
            target: QubitId(0),
            theta: PI / 4.0,
        };

        let decomposition = characterizer
            .decompose_to_rotations(&rx)
            .expect("decompose to rotations failed");
        assert_eq!(decomposition.len(), 3); // Rz-Ry-Rz decomposition
    }

    #[test]
    fn test_eigenphases() {
        let characterizer = GateCharacterizer::new(1e-10);
        let rz = RotationZ {
            target: QubitId(0),
            theta: PI / 2.0,
        };

        let eigen = characterizer
            .eigenstructure(&rz)
            .expect("eigenstructure failed");
        let phases = eigen.eigenphases();

        assert_eq!(phases.len(), 2);
        assert!((phases[0] + phases[1]).abs() < 1e-10); // Opposite phases
    }

    #[test]
    fn test_closest_clifford() {
        let characterizer = GateCharacterizer::new(1e-10);

        let t_like = RotationZ {
            target: QubitId(0),
            theta: PI / 4.0,
        };
        let closest = characterizer
            .find_closest_clifford(&t_like)
            .expect("find closest clifford failed");

        let s_distance = characterizer
            .gate_distance(&t_like, &Phase { target: QubitId(0) })
            .expect("gate distance to S failed");
        let actual_distance = characterizer
            .gate_distance(&t_like, closest.as_ref())
            .expect("gate distance to closest failed");

        assert!(actual_distance <= s_distance + 1e-10);
    }

    #[test]
    fn test_identity_check() {
        let characterizer = GateCharacterizer::new(1e-10);

        let identity_gate = RotationZ {
            target: QubitId(0),
            theta: 0.0,
        };
        assert!(characterizer.is_identity(&identity_gate, 1e-10));
        assert!(!characterizer.is_identity(&PauliX { target: QubitId(0) }, 1e-10));

        let x_squared_vec = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

        #[derive(Debug)]
        struct CustomGate(Vec<Complex>);
        impl GateOp for CustomGate {
            fn name(&self) -> &'static str {
                "X²"
            }
            fn qubits(&self) -> Vec<QubitId> {
                vec![QubitId(0)]
            }
            fn matrix(&self) -> crate::error::QuantRS2Result<Vec<Complex>> {
                Ok(self.0.clone())
            }
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn clone_gate(&self) -> Box<dyn GateOp> {
                Box::new(CustomGate(self.0.clone()))
            }
        }

        let x_squared_gate = CustomGate(x_squared_vec);
        assert!(characterizer.is_identity(&x_squared_gate, 1e-10));
    }

    #[test]
    fn test_global_phase() {
        let characterizer = GateCharacterizer::new(1e-10);

        let z_phase = characterizer
            .global_phase(&PauliZ { target: QubitId(0) })
            .expect("global phase of Z failed");
        assert!((z_phase - PI / 2.0).abs() < 1e-10 || (z_phase + PI / 2.0).abs() < 1e-10);

        let phase_gate = Phase { target: QubitId(0) };
        let global_phase = characterizer
            .global_phase(&phase_gate)
            .expect("global phase of S failed");
        assert!((global_phase - PI / 4.0).abs() < 1e-10);
    }
}
