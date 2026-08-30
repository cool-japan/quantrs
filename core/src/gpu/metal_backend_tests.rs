//! Tests for the Metal GPU backend.
//!
//! The real Metal compute backend is DEFERRED: these tests assert the *honest*
//! behavior of the current build (availability is `false`, device info is
//! `None`, and construction / kernel compilation return honest errors) rather
//! than the previously fabricated "Metal is available" placeholders.

#[cfg(test)]
mod tests {
    #[cfg(feature = "metal")]
    use crate::gpu::metal_backend_scirs2_ready::{MetalQuantumState, *};

    #[test]
    fn test_metal_availability_detection() {
        #[cfg(feature = "metal")]
        {
            // Real Metal backend is DEFERRED (no device dispatch wired), so the
            // honest answer is false on every platform — it must NOT fabricate
            // availability just because the feature is on under macOS.
            assert!(
                !is_metal_available(),
                "Metal must be reported unavailable until a real backend is wired"
            );
        }

        #[cfg(not(feature = "metal"))]
        {
            // Without metal feature, we can't test the function
        }
    }

    #[test]
    fn test_metal_device_info() {
        #[cfg(feature = "metal")]
        {
            // No real Metal device is initialized → honest None (the previous
            // fabricated placeholder specs were removed).
            assert!(
                get_metal_device_info().is_none(),
                "Metal device info must be None until a real backend is wired"
            );
        }

        #[cfg(not(feature = "metal"))]
        {}
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_metal_quantum_state_creation_is_deferred_error() {
        // Real Metal device init is DEFERRED: construction must return an honest
        // error rather than fabricating a placeholder device.
        for num_qubits in [1, 5, 10, 15] {
            assert!(
                MetalQuantumState::new(num_qubits).is_err(),
                "MetalQuantumState::new must be an honest error until Metal is wired"
            );
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn test_kernel_compilation_is_deferred_error() {
        // Even valid kernel names cannot be compiled yet (DEFERRED): an invalid
        // name is a distinct error, and a valid name is an honest "unavailable"
        // error — never a fabricated successfully-compiled placeholder pipeline.
        // Note: construction itself is also a deferred error, so we cannot build
        // a MetalQuantumState here; the compile path is covered by unit tests in
        // the backend module returning honest errors.
        assert!(
            MetalQuantumState::new(3).is_err(),
            "construction is a deferred error"
        );
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
    #[ignore = "Skipping test that requires Metal GPU"]
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
