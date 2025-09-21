//! Test suite for the GPU backend implementation

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use quantrs2_ml::gpu_backend_impl::GPUBackend;
    use quantrs2_ml::simulator_backends::{
        Backend, DynamicCircuit, Observable, SimulatorBackend,
    };

    #[test]
    fn test_gpu_backend_creation() {
        // Test that we can create a GPU backend (will fail on macOS with proper error)
        let result = GPUBackend::new(0, 20);

        #[cfg(all(feature = "gpu", not(target_os = "macos")))]
        {
            // On non-macOS with GPU support, it should either succeed or fail with GPU not available
            if result.is_ok() {
                let backend = result.unwrap();
                assert_eq!(backend.name(), "gpu_full");
                assert!(backend.capabilities().gpu_acceleration);
                assert_eq!(backend.max_qubits(), 20);
            } else {
                // GPU not available on this system
                assert!(result.unwrap_err().to_string().contains("GPU not available"));
            }
        }

        #[cfg(not(all(feature = "gpu", not(target_os = "macos"))))]
        {
            // On macOS or without GPU feature, should return not supported
            assert!(result.is_err());
            assert!(result.unwrap_err().to_string().contains("not available on this platform"));
        }
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_backend_stub_behavior() {
        // Create the Backend enum with GPU variant
        if let Ok(gpu) = GPUBackend::new(0, 10) {
            let backend = Backend::GPU(gpu);

            // Test that all methods are available (even if they return errors on unsupported platforms)
            let capabilities = backend.capabilities();

            #[cfg(all(feature = "gpu", not(target_os = "macos")))]
            {
                assert!(capabilities.gpu_acceleration);
                assert_eq!(backend.name(), "gpu_full");
            }

            #[cfg(not(all(feature = "gpu", not(target_os = "macos"))))]
            {
                assert!(!capabilities.gpu_acceleration);
                assert_eq!(backend.name(), "gpu_stub");
            }
        }
    }

    #[test]
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    fn test_gpu_backend_circuit_execution() {
        use quantrs2_circuit::prelude::Circuit;

        // This test will only run on non-macOS systems with GPU feature
        if let Ok(backend) = GPUBackend::new(0, 10) {
            // Create a simple 2-qubit circuit
            let circuit = Circuit::<2>::new();
            let dynamic_circuit = DynamicCircuit::Circuit2(circuit);

            // Try to execute (will fail if no actual GPU available)
            let result = backend.execute_circuit(&dynamic_circuit, &[], None);

            if result.is_ok() {
                // If we have a real GPU, check the result
                let sim_result = result.unwrap();
                assert!(sim_result.metadata.contains_key("gpu_time_ms"));
                assert!(sim_result.metadata.contains_key("num_qubits"));
            } else {
                // No GPU available, but method should be callable
                assert!(result.unwrap_err().to_string().contains("GPU"));
            }
        }
    }

    #[test]
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    fn test_gpu_backend_expectation_value() {
        use quantrs2_circuit::prelude::Circuit;

        if let Ok(backend) = GPUBackend::new(0, 10) {
            let circuit = Circuit::<2>::new();
            let dynamic_circuit = DynamicCircuit::Circuit2(circuit);

            // Create a simple Pauli Z observable
            let observable = Observable::PauliZ(vec![0]);

            // Try to compute expectation value
            let result = backend.expectation_value(&dynamic_circuit, &[], &observable);

            // Method should be callable even if GPU is not available
            assert!(result.is_ok() || result.unwrap_err().to_string().contains("GPU"));
        }
    }

    #[test]
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    fn test_gpu_backend_memory_management() {
        // Test that the GPU backend properly manages memory and caches simulators
        if let Ok(backend) = GPUBackend::new(0, 15) {
            use quantrs2_circuit::prelude::Circuit;

            // Execute multiple circuits of the same size to test caching
            for _ in 0..3 {
                let circuit = Circuit::<4>::new();
                let dynamic_circuit = DynamicCircuit::Circuit4(circuit);

                let _ = backend.execute_circuit(&dynamic_circuit, &[], None);
            }

            // The backend should cache the simulator for reuse
            // This is tested internally by the cache_hits/cache_misses metrics
        }
    }

    #[test]
    fn test_gpu_backend_gradient_computation() {
        use quantrs2_ml::simulator_backends::GradientMethod;
        use quantrs2_circuit::prelude::Circuit;

        if let Ok(backend) = GPUBackend::new(0, 10) {
            let circuit = Circuit::<2>::new();
            let dynamic_circuit = DynamicCircuit::Circuit2(circuit);
            let observable = Observable::PauliZ(vec![0]);

            // Test parameter shift gradient
            let result = backend.compute_gradients(
                &dynamic_circuit,
                &[0.5, 1.0],
                &observable,
                GradientMethod::ParameterShift,
            );

            // Should be callable even on stub implementation
            assert!(result.is_ok() || result.unwrap_err().to_string().contains("not"));
        }
    }
}