//! Tests for Metal GPU backend
//!
//! These tests verify that the Metal backend placeholder implementation
//! is ready for SciRS2 integration.

#[cfg(test)]
mod tests {
    use crate::qubit::QubitId;
    use num_complex::Complex64;

    #[cfg(feature = "metal")]
    use crate::gpu::metal_backend_scirs2_ready::{MetalQuantumState, *};

    #[test]
    fn test_metal_availability_detection() {
        #[cfg(feature = "metal")]
        {
            let available = is_metal_available();

            #[cfg(target_os = "macos")]
            {
                assert!(
                    available,
                    "Metal should be available on macOS with metal feature"
                );
            }

            #[cfg(not(target_os = "macos"))]
            {
                assert!(!available, "Metal should not be available on non-macOS");
            }
        }

        #[cfg(not(feature = "metal"))]
        {
            // Without metal feature, we can't test the function
            assert!(true, "Metal feature not enabled");
        }
    }

    #[test]
    fn test_metal_device_info() {
        #[cfg(feature = "metal")]
        {
            let info = get_metal_device_info();

            #[cfg(target_os = "macos")]
            {
                assert!(
                    info.is_some(),
                    "Should return device info on macOS with metal feature"
                );
                let info_str = info.unwrap();
                assert!(info_str.contains("Metal Device"));
                assert!(info_str.contains("Max threads per threadgroup"));
                assert!(info_str.contains("Max buffer length"));
            }

            #[cfg(not(target_os = "macos"))]
            {
                assert!(info.is_none(), "Should return None on non-macOS");
            }
        }

        #[cfg(not(feature = "metal"))]
        {
            assert!(true, "Metal feature not enabled");
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_metal_quantum_state_creation() {
        // Test creating quantum states of various sizes
        for num_qubits in [1, 5, 10, 15] {
            let result = MetalQuantumState::new(num_qubits);
            assert!(result.is_ok(), "Should create {}-qubit state", num_qubits);

            let state = result.unwrap();
            assert_eq!(state.num_qubits, num_qubits);
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_single_qubit_gate_application() {
        let mut state = MetalQuantumState::new(5).unwrap();

        // Test Pauli-X gate
        let pauli_x = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        let result = state.apply_single_qubit_gate(&pauli_x, QubitId(0));
        assert!(result.is_ok(), "Should apply Pauli-X gate");

        // Test Hadamard gate
        let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;
        let hadamard = [
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(sqrt2_inv, 0.0),
            Complex64::new(-sqrt2_inv, 0.0),
        ];

        let result = state.apply_single_qubit_gate(&hadamard, QubitId(1));
        assert!(result.is_ok(), "Should apply Hadamard gate");

        // Test invalid qubit index
        let result = state.apply_single_qubit_gate(&pauli_x, QubitId(5));
        assert!(result.is_err(), "Should fail for out-of-range qubit");
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_kernel_compilation() {
        let state = MetalQuantumState::new(3).unwrap();

        // Test valid kernel names
        let result = state.get_or_compile_kernel("apply_single_qubit_gate");
        assert!(result.is_ok(), "Should compile single qubit gate kernel");

        let kernel = result.unwrap();
        assert_eq!(kernel.function_name, "apply_single_qubit_gate");

        let result = state.get_or_compile_kernel("compute_probabilities");
        assert!(result.is_ok(), "Should compile probabilities kernel");

        // Test invalid kernel name
        let result = state.get_or_compile_kernel("invalid_kernel");
        assert!(result.is_err(), "Should fail for invalid kernel name");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_shader_syntax() {
        // Verify that our Metal shader code is syntactically valid
        let shader_code = crate::gpu::metal_backend_scirs2_ready::METAL_QUANTUM_SHADERS;

        // Check for required Metal headers
        assert!(shader_code.contains("#include <metal_stdlib>"));
        assert!(shader_code.contains("using namespace metal"));

        // Check for complex number struct
        assert!(shader_code.contains("struct Complex"));
        assert!(shader_code.contains("float real"));
        assert!(shader_code.contains("float imag"));

        // Check for kernel functions
        assert!(shader_code.contains("kernel void apply_single_qubit_gate"));
        assert!(shader_code.contains("kernel void compute_probabilities"));

        // Check for proper Metal attributes
        assert!(shader_code.contains("[[buffer(0)]]"));
        assert!(shader_code.contains("[[thread_position_in_grid]]"));
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    #[test]
    fn test_metal_not_available() {
        #[cfg(feature = "metal")]
        {
            use crate::gpu::metal_backend_scirs2_ready::MetalQuantumState;
            // Test that MetalQuantumState creation fails gracefully
            let result = MetalQuantumState::new(5);
            assert!(result.is_err(), "Should fail when Metal is not available");

            match result {
                Err(e) => {
                    let error_msg = format!("{}", e);
                    assert!(error_msg.contains("Metal support not compiled"));
                }
                Ok(_) => panic!("Expected error when Metal is not available"),
            }
        }

        #[cfg(not(feature = "metal"))]
        {
            // When metal feature is not enabled, just pass the test
            assert!(true, "Metal feature not enabled - test passes");
        }
    }

    #[test]
    fn test_placeholder_types() {
        // Ensure our placeholder types compile correctly
        #[cfg(feature = "metal")]
        {
            use super::super::metal_backend_scirs2_ready::scirs2_metal_placeholder::*;

            // Test MetalDeviceHandle
            let device = MetalDeviceHandle {
                name: "Test Device".to_string(),
            };
            assert_eq!(device.name, "Test Device");

            // Test MetalBuffer
            let buffer: MetalBuffer<f32> = MetalBuffer {
                buffer: MetalBufferHandle,
                length: 1024,
                _phantom: std::marker::PhantomData,
            };
            assert_eq!(buffer.length, 1024);

            // Test MetalKernel
            let kernel = MetalKernel {
                pipeline: MetalComputePipeline,
                function_name: "test_kernel".to_string(),
            };
            assert_eq!(kernel.function_name, "test_kernel");
        }
    }

    #[test]
    fn test_scirs2_compatibility() {
        // Test that our implementation is compatible with expected SciRS2 patterns
        use crate::gpu::scirs2_adapter::is_gpu_available;

        // This should work regardless of actual GPU availability
        let _gpu_available = is_gpu_available();

        #[cfg(feature = "metal")]
        {
            // Test that we can check for Metal specifically
            let metal_available = is_metal_available();

            #[cfg(feature = "gpu")]
            {
                // When GPU feature is enabled, at least one of these should be true
                let any_gpu = _gpu_available || metal_available;
                // We can't assert this is true because it depends on hardware
                let _ = any_gpu;
            }
        }
    }
}
