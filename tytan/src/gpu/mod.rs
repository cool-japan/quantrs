//! GPU acceleration for QUBO/HOBO optimization.
//!
//! This module provides GPU-accelerated implementations for
//! solving QUBO and HOBO problems, using SciRS2 when available.

use ndarray::{Array, ArrayD, Ix2};
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use thiserror::Error;

use crate::sampler::SampleResult;
use quantrs2_anneal::is_available as anneal_gpu_available;

/// Errors that can occur during GPU operations
#[derive(Error, Debug)]
pub enum GpuError {
    /// GPU is not available
    #[error("GPU not available: {0}")]
    NotAvailable(String),

    /// Error during memory transfer
    #[error("Memory transfer error: {0}")]
    MemoryTransfer(String),

    /// Error during kernel execution
    #[error("Kernel execution error: {0}")]
    KernelExecution(String),

    /// Error in tensor operations
    #[error("Tensor operation error: {0}")]
    TensorOperation(String),
}

/// Result type for GPU operations
pub type GpuResult<T> = Result<T, GpuError>;

/// Check if GPU acceleration is available
pub fn is_available() -> bool {
    #[cfg(feature = "gpu_accelerated")]
    {
        // For SciRS2 GPU integration
        #[cfg(feature = "scirs")]
        {
            return anneal_gpu_available();
        }

        // For plain OCL
        #[cfg(not(feature = "scirs"))]
        {
            match ocl::Platform::list().first() {
                Some(_) => true,
                None => false,
            }
        }
    }

    #[cfg(not(feature = "gpu_accelerated"))]
    {
        false
    }
}

/// GPU-accelerated QUBO solver
#[cfg(feature = "gpu_accelerated")]
pub fn gpu_solve_qubo(
    matrix: &Array<f64, Ix2>,
    var_map: &HashMap<String, usize>,
    shots: usize,
    temperature_steps: usize,
) -> GpuResult<Vec<SampleResult>> {
    let n_vars = var_map.len();

    // Map from indices back to variable names
    let idx_to_var: HashMap<usize, String> = var_map
        .iter()
        .map(|(var, &idx)| (idx, var.clone()))
        .collect();

    // With SciRS2 integration
    #[cfg(feature = "scirs")]
    {
        use crate::scirs_stub::scirs2_core::gpu::{GpuArray, GpuDevice};

        // Initialize GPU device
        let device = GpuDevice::new(0).map_err(|e| GpuError::NotAvailable(e.to_string()))?;

        // Transfer QUBO matrix to GPU
        let gpu_matrix = GpuArray::from_ndarray(device.clone(), matrix)
            .map_err(|e| GpuError::MemoryTransfer(e.to_string()))?;

        // Initialize random states on GPU
        let gpu_states = device
            .random_array::<f32>((shots, n_vars))
            .map_err(|e| GpuError::MemoryTransfer(e.to_string()))?;

        // Binarize states (threshold at 0.5)
        let gpu_binary = device
            .binarize(&gpu_states, 0.5)
            .map_err(|e| GpuError::KernelExecution(e.to_string()))?;

        // Execute parallel annealing
        // (In a real implementation, we'd run the full annealing logic here)
        // For now, we'll just compute the energies of the initial states

        // Compute energies
        let gpu_energies = device
            .qubo_energy(&gpu_binary, &gpu_matrix)
            .map_err(|e| GpuError::KernelExecution(e.to_string()))?;

        // Transfer results back to CPU
        let binary_states: Array<bool, Ix2> = gpu_binary
            .to_ndarray()
            .map_err(|e| GpuError::MemoryTransfer(e.to_string()))?;

        let energies: Array<f64, Ix2> = gpu_energies
            .to_ndarray()
            .map_err(|e| GpuError::MemoryTransfer(e.to_string()))?;

        // Convert to SampleResults
        let mut results = Vec::new();

        for i in 0..shots {
            let mut state = binary_states.slice(ndarray::s![i, ..]);
            let mut energy = energies[[i, 0]];

            // Create variable assignment dictionary
            let assignments: HashMap<String, bool> = state
                .iter()
                .enumerate()
                .map(|(idx, &value)| {
                    let var_name = idx_to_var.get(&idx).unwrap().clone();
                    (var_name, value)
                })
                .collect();

            // Create sample result
            let mut result = SampleResult {
                assignments,
                energy,
                occurrences: 1, // For now, each result has one occurrence
            };

            results.push(result);
        }

        // Sort by energy (best solutions first)
        results.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

        // Combine identical solutions
        let mut consolidated = HashMap::new();
        for result in results {
            // Convert assignments to a sortable, hashable representation
            let mut sorted_assignments: Vec<(String, bool)> = result
                .assignments
                .iter()
                .map(|(k, &v)| (k.clone(), v))
                .collect();
            sorted_assignments.sort_by(|a, b| a.0.cmp(&b.0));

            let entry = consolidated.entry(sorted_assignments).or_insert((
                result.assignments.clone(),
                result.energy,
                0,
            ));
            entry.2 += 1;
        }

        // Convert back to SampleResults
        let mut final_results: Vec<SampleResult> = consolidated
            .into_iter()
            .map(|(_, (assignments, energy, occurrences))| SampleResult {
                assignments,
                energy,
                occurrences,
            })
            .collect();

        // Sort again
        final_results.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

        Ok(final_results)
    }

    // With plain OCL
    #[cfg(not(feature = "scirs"))]
    {
        use ocl::{Buffer, MemFlags, ProQue};

        // Build OCL context and queue
        let ocl_pq = ProQue::builder()
            .src(
                r#"
                __kernel void qubo_energy(__global const uchar* binary,
                                         __global const double* matrix,
                                         __global double* energies,
                                         const int n_vars) {
                    int gid = get_global_id(0);
                    int offset = gid * n_vars;

                    double energy = 0.0;

                    // Linear terms
                    for (int i = 0; i < n_vars; i++) {
                        if (binary[offset + i]) {
                            energy += matrix[i * n_vars + i];
                        }
                    }

                    // Quadratic terms
                    for (int i = 0; i < n_vars; i++) {
                        if (binary[offset + i]) {
                            for (int j = i + 1; j < n_vars; j++) {
                                if (binary[offset + j]) {
                                    energy += matrix[i * n_vars + j];
                                }
                            }
                        }
                    }

                    energies[gid] = energy;
                }
            "#,
            )
            .dims(shots)
            .build()
            .map_err(|e| GpuError::NotAvailable(e.to_string()))?;

        // Flatten the matrix for OCL
        let flat_matrix: Vec<f64> = matrix.iter().cloned().collect();

        // Create random binary states
        let mut rng = thread_rng();
        let binary_states: Vec<u8> = (0..shots * n_vars)
            .map(|_| if rng.gen::<bool>() { 1u8 } else { 0u8 })
            .collect();

        // Create OCL buffers
        let binary_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::READ_ONLY)
            .len(shots * n_vars)
            .copy_host_slice(&binary_states)
            .build()
            .map_err(|e| GpuError::MemoryTransfer(e.to_string()))?;

        let matrix_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::READ_ONLY)
            .len(n_vars * n_vars)
            .copy_host_slice(&flat_matrix)
            .build()
            .map_err(|e| GpuError::MemoryTransfer(e.to_string()))?;

        let energies_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::WRITE_ONLY)
            .len(shots)
            .build()
            .map_err(|e| GpuError::MemoryTransfer(e.to_string()))?;

        // Execute kernel
        let mut kernel = ocl_pq
            .kernel_builder("qubo_energy")
            .arg(&binary_buffer)
            .arg(&matrix_buffer)
            .arg(&energies_buffer)
            .arg(n_vars as i32)
            .build()
            .map_err(|e| GpuError::KernelExecution(e.to_string()))?;

        unsafe {
            kernel
                .enq()
                .map_err(|e| GpuError::KernelExecution(e.to_string()))?;
        }

        // Read results
        let mut energies = vec![0.0f64; shots];
        energies_buffer
            .read(&mut energies)
            .enq()
            .map_err(|e| GpuError::MemoryTransfer(e.to_string()))?;

        // Convert to SampleResults
        let mut results = Vec::new();

        for i in 0..shots {
            // Extract binary state
            let state: Vec<bool> = binary_states[i * n_vars..(i + 1) * n_vars]
                .iter()
                .map(|&b| b == 1)
                .collect();

            // Create variable assignment dictionary
            let assignments: HashMap<String, bool> = state
                .iter()
                .enumerate()
                .map(|(idx, &value)| {
                    let var_name = idx_to_var.get(&idx).unwrap().clone();
                    (var_name, value)
                })
                .collect();

            // Create sample result
            let mut result = SampleResult {
                assignments,
                energy: energies[i],
                occurrences: 1, // For now, each result has one occurrence
            };

            results.push(result);
        }

        // Sort by energy (best solutions first)
        results.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

        Ok(results)
    }
}

/// Fallback implementation when GPU is not available
#[cfg(not(feature = "gpu_accelerated"))]
pub fn gpu_solve_qubo(
    _matrix: &Array<f64, Ix2>,
    _var_map: &HashMap<String, usize>,
    _shots: usize,
    _temperature_steps: usize,
) -> GpuResult<Vec<SampleResult>> {
    Err(GpuError::NotAvailable(
        "GPU acceleration not enabled. Rebuild with '--features gpu_accelerated'".to_string(),
    ))
}

/// GPU-accelerated HOBO solver using tensor methods
#[cfg(all(feature = "gpu_accelerated", feature = "scirs"))]
pub fn gpu_solve_hobo(
    tensor: &ArrayD<f64>,
    var_map: &HashMap<String, usize>,
    shots: usize,
    temperature_steps: usize,
) -> GpuResult<Vec<SampleResult>> {
    use crate::scirs_stub::scirs2_core::gpu::{GpuArray, GpuDevice};

    let n_vars = var_map.len();
    let dim = tensor.ndim();

    // Map from indices back to variable names
    let idx_to_var: HashMap<usize, String> = var_map
        .iter()
        .map(|(var, &idx)| (idx, var.clone()))
        .collect();

    // Initialize GPU device
    let device = GpuDevice::new(0).map_err(|e| GpuError::NotAvailable(e.to_string()))?;

    // For high-dimensional tensors, use CP decomposition on CPU first
    let rank = std::cmp::min(n_vars, 50); // Truncate to reasonable rank

    // Initialize placeholder results
    // (In a real implementation, would perform tensor operations)
    let mut results = Vec::new();

    for _ in 0..shots {
        // Generate random solution for placeholder
        let mut rng = thread_rng();
        let assignments: HashMap<String, bool> = var_map
            .iter()
            .map(|(name, _)| (name.clone(), rng.gen::<bool>()))
            .collect();

        // Create sample result
        let mut result = SampleResult {
            assignments,
            energy: 0.0, // This would be an actual energy in real implementation
            occurrences: 1,
        };

        results.push(result);
    }

    Ok(results)
}

/// Fallback HOBO solver when GPU acceleration is not available
#[cfg(not(all(feature = "gpu_accelerated", feature = "scirs")))]
pub fn gpu_solve_hobo(
    _tensor: &ArrayD<f64>,
    _var_map: &HashMap<String, usize>,
    _shots: usize,
    _temperature_steps: usize,
) -> GpuResult<Vec<SampleResult>> {
    Err(GpuError::NotAvailable("GPU acceleration for HOBO not available. Requires both 'gpu_accelerated' and 'scirs' features.".to_string()))
}
