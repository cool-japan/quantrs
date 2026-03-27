//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;
use scirs2_core::{Complex32, Complex64};
use std::f64::consts::PI;

use super::quantumnerf_type::QuantumNeRF;
use super::types::{CameraMatrix, QuantumMLPState, QuantumNeRFConfig, Ray};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_quantum_nerf_creation() {
        let config = QuantumNeRFConfig::default();
        let nerf = QuantumNeRF::new(config);
        assert!(nerf.is_ok());
    }
    #[test]
    fn test_quantum_positional_encoding() {
        let config = QuantumNeRFConfig::default();
        let nerf = QuantumNeRF::new(config).expect("Failed to create QuantumNeRF");
        let position = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let encoding = nerf.quantum_positional_encoding(&position);
        assert!(encoding.is_ok());
        let output = encoding.expect("Positional encoding should succeed");
        assert!(output.features.len() > 3);
        assert!(output.entanglement_measure >= 0.0);
        assert!(output.entanglement_measure <= 1.0);
    }
    #[test]
    fn test_quantum_ray_sampling() {
        let config = QuantumNeRFConfig::default();
        let nerf = QuantumNeRF::new(config).expect("Failed to create QuantumNeRF");
        let ray = Ray {
            origin: Array1::from_vec(vec![0.0, 0.0, 0.0]),
            direction: Array1::from_vec(vec![0.0, 0.0, 1.0]),
            near: 0.1,
            far: 5.0,
        };
        let sampling = nerf.quantum_ray_sampling(&ray);
        assert!(sampling.is_ok());
        let output = sampling.expect("Ray sampling should succeed");
        assert!(!output.points.is_empty());
        assert!(!output.distances.is_empty());
        assert_eq!(output.points.len(), output.distances.len());
    }
    #[test]
    fn test_quantum_mlp_query() {
        let config = QuantumNeRFConfig::default();
        let nerf = QuantumNeRF::new(config).expect("Failed to create QuantumNeRF");
        let input_features = Array1::ones(64);
        let result = nerf.query_quantum_mlp(&nerf.quantum_mlp_coarse, &input_features);
        assert!(result.is_ok());
        let output = result.expect("MLP query should succeed");
        assert_eq!(output.color.len(), 3);
        assert!(output.density >= 0.0);
        assert!(output.quantum_state.entanglement_measure >= 0.0);
    }
    #[test]
    fn test_quantum_volume_rendering() {
        let config = QuantumNeRFConfig::default();
        let nerf = QuantumNeRF::new(config).expect("Failed to create QuantumNeRF");
        let colors = vec![
            Array1::from_vec(vec![1.0, 0.0, 0.0]),
            Array1::from_vec(vec![0.0, 1.0, 0.0]),
            Array1::from_vec(vec![0.0, 0.0, 1.0]),
        ];
        let densities = vec![0.5, 0.3, 0.2];
        let quantum_states = vec![
            QuantumMLPState {
                quantum_amplitudes: Array1::zeros(8).mapv(|_: f64| Complex64::new(0.0, 0.0)),
                entanglement_measure: 0.5,
                quantum_fidelity: 0.9,
            };
            3
        ];
        let distances = vec![1.0, 2.0, 3.0];
        let result =
            nerf.quantum_volume_rendering(&colors, &densities, &quantum_states, &distances);
        assert!(result.is_ok());
        let output = result.expect("Volume rendering should succeed");
        assert_eq!(output.final_color.len(), 3);
        assert!(output.depth >= 0.0);
    }
    #[test]
    fn test_quantum_spherical_harmonics() {
        let config = QuantumNeRFConfig::default();
        let nerf = QuantumNeRF::new(config).expect("Failed to create QuantumNeRF");
        let view_direction = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let encoding = nerf.quantum_spherical_harmonics_encoding(&view_direction);
        assert!(encoding.is_ok());
        let output = encoding.expect("Spherical harmonics encoding should succeed");
        assert!(!output.features.is_empty());
        assert!(output.entanglement_measure > 0.0);
    }
    #[test]
    fn test_camera_ray_generation() {
        let config = QuantumNeRFConfig::default();
        let nerf = QuantumNeRF::new(config).expect("Failed to create QuantumNeRF");
        let camera = CameraMatrix {
            position: Array1::from_vec(vec![0.0, 0.0, 0.0]),
            forward: Array1::from_vec(vec![0.0, 0.0, 1.0]),
            right: Array1::from_vec(vec![1.0, 0.0, 0.0]),
            up: Array1::from_vec(vec![0.0, 1.0, 0.0]),
            fov: PI / 4.0,
        };
        let ray = nerf.generate_camera_ray(&camera, 100, 100, 200, 200, PI / 4.0);
        assert!(ray.is_ok());
        let ray_output = ray.expect("Camera ray generation should succeed");
        assert_eq!(ray_output.origin.len(), 3);
        assert_eq!(ray_output.direction.len(), 3);
        assert!(ray_output.near > 0.0);
        assert!(ray_output.far > ray_output.near);
    }
    #[test]
    fn test_entanglement_based_encoding() {
        let config = QuantumNeRFConfig {
            quantum_enhancement_level: 0.8,
            ..Default::default()
        };
        let nerf = QuantumNeRF::new(config).expect("Failed to create QuantumNeRF");
        let position = Array1::from_vec(vec![0.5, 0.3, 0.7]);
        let encoding = nerf.entanglement_based_encoding(&position);
        assert!(encoding.is_ok());
        let output = encoding.expect("Entanglement encoding should succeed");
        assert!(output.entanglement_measure > 0.8);
        assert!(!output
            .quantum_amplitudes
            .iter()
            .all(|amp| amp.norm() < 1e-10));
    }
}
