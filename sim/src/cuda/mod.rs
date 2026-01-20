//! CUDA acceleration module for quantum simulations.
//!
//! This module provides GPU acceleration for quantum circuit simulation
//! using CUDA kernels, with memory management, stream handling, and
//! device context management.

pub mod context;
pub mod graph;
pub mod kernels;
pub mod memory;
pub mod streams;
pub mod tensor_core;

// Re-export commonly used types and structs
#[cfg(feature = "advanced_math")]
pub use context::{CudaContext, CudaDeviceProperties, CudaProfiler};
pub use graph::{
    CudaGraph, CudaGraphBuilder, CudaGraphExec, GraphCaptureMode, GraphExecUpdateResult,
    GraphInstantiationFlags, GraphNode, GraphNodeType, GraphStats, GraphUpdateResult,
    HostNodeParams, KernelNodeParams, MemCopyKind, MemCopyNodeParams, MemSetNodeParams,
    QuantumGraphScheduler,
};
#[cfg(feature = "advanced_math")]
pub use kernels::CudaKernel;
pub use kernels::{
    CudaKernelConfig, CudaKernelStats, CudaQuantumKernels, GateType, OptimizationLevel,
};
pub use memory::GpuMemoryType;
#[cfg(feature = "advanced_math")]
pub use memory::{GpuMemory, GpuMemoryBlock, GpuMemoryPool};
#[cfg(feature = "advanced_math")]
pub use streams::CudaStream;
pub use streams::{StreamFlags, StreamPriority};
pub use tensor_core::{
    fp16_utils, AccumulatorPrecision, TensorCoreConfig, TensorCoreGeneration, TensorCoreKernels,
    TensorCoreOps,
};

use crate::error::Result;

/// Initialize CUDA subsystem
pub fn initialize() -> Result<()> {
    #[cfg(feature = "advanced_math")]
    {
        // Check if CUDA is available
        let device_count = CudaContext::get_device_count()?;
        if device_count == 0 {
            return Err(crate::error::SimulatorError::ResourceExhausted(
                "No CUDA devices available".to_string(),
            ));
        }

        // Initialize default context
        let _context = CudaContext::new(0)?;
    }

    #[cfg(not(feature = "advanced_math"))]
    {
        // CUDA not available, return success anyway
    }

    Ok(())
}

/// Check if CUDA is available and working
pub fn is_available() -> bool {
    #[cfg(feature = "advanced_math")]
    {
        CudaContext::get_device_count().unwrap_or(0) > 0
    }

    #[cfg(not(feature = "advanced_math"))]
    {
        false
    }
}

/// Get number of available CUDA devices
pub fn get_device_count() -> Result<i32> {
    #[cfg(feature = "advanced_math")]
    {
        CudaContext::get_device_count()
    }

    #[cfg(not(feature = "advanced_math"))]
    {
        Ok(0)
    }
}

/// Get device properties for a specific device
#[cfg(feature = "advanced_math")]
pub fn get_device_properties(device_id: i32) -> Result<CudaDeviceProperties> {
    let context = CudaContext::new(device_id)?;
    Ok(context.get_device_properties().clone())
}

#[cfg(not(feature = "advanced_math"))]
pub fn get_device_properties(_device_id: i32) -> Result<()> {
    Err(crate::error::SimulatorError::UnsupportedOperation(
        "CUDA not available".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        // This test will pass whether CUDA is available or not
        let _is_available = is_available();
        let _device_count = get_device_count().unwrap_or(0);
    }

    #[test]
    fn test_cuda_initialization() {
        let result = initialize();
        // Should succeed whether CUDA is available or not
        assert!(result.is_ok() || !is_available());
    }

    #[cfg(feature = "advanced_math")]
    #[test]
    fn test_context_creation() {
        if is_available() {
            let context = CudaContext::new(0);
            assert!(context.is_ok());
        }
    }

    #[cfg(feature = "advanced_math")]
    #[test]
    fn test_stream_creation() {
        if is_available() {
            let stream = CudaStream::new();
            assert!(stream.is_ok());
        }
    }

    #[cfg(feature = "advanced_math")]
    #[test]
    fn test_memory_allocation() {
        let mut memory = GpuMemory::new();
        let result = memory.allocate_pool(1024);
        // Should work with or without CUDA
        assert!(result.is_ok());
    }

    #[test]
    fn test_kernel_config_default() {
        let config = CudaKernelConfig::default();
        assert_eq!(config.device_id, 0);
        assert_eq!(config.num_streams, 4);
        assert_eq!(config.block_size, 256);
    }

    #[test]
    fn test_cuda_graph_creation() {
        let mut graph = CudaGraph::new();
        assert!(graph.is_empty());

        let params = KernelNodeParams::default();
        let node_id = graph.add_kernel_node(params, &[]).expect("add kernel");
        assert_eq!(node_id, 0);
        assert_eq!(graph.node_count(), 1);
    }

    #[test]
    fn test_cuda_graph_execution() {
        let mut graph = CudaGraph::new();

        // Add some nodes
        let n1 = graph
            .add_kernel_node(KernelNodeParams::default(), &[])
            .expect("add");
        let n2 = graph
            .add_kernel_node(KernelNodeParams::default(), &[n1])
            .expect("add");
        let _n3 = graph
            .add_memcpy_node(
                MemCopyNodeParams {
                    src: 0,
                    dst: 1,
                    size: 1024,
                    kind: MemCopyKind::DeviceToDevice,
                },
                &[n2],
            )
            .expect("add memcpy");

        graph.finalize().expect("finalize");

        let stats = graph.get_stats();
        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.kernel_count, 2);
        assert_eq!(stats.mem_op_count, 1);

        let exec = graph.instantiate().expect("instantiate");
        assert_eq!(exec.execution_count(), 0);
    }

    #[test]
    fn test_quantum_graph_scheduler() {
        let mut scheduler = QuantumGraphScheduler::new(10);

        let _exec = scheduler
            .get_or_create("bell_circuit", || {
                let mut graph = CudaGraph::new();
                // Hadamard
                graph.add_kernel_node(
                    KernelNodeParams {
                        function: 1, // hadamard
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        ..Default::default()
                    },
                    &[],
                )?;
                // CNOT
                graph.add_kernel_node(
                    KernelNodeParams {
                        function: 2, // cnot
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        ..Default::default()
                    },
                    &[0],
                )?;
                graph.finalize()?;
                Ok(graph)
            })
            .expect("create graph");

        let (hits, misses) = scheduler.cache_stats();
        assert_eq!(misses, 1);
        assert_eq!(hits, 0);
    }
}
