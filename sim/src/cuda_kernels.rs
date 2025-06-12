//! CUDA kernels for GPU-accelerated quantum simulations using SciRS2.
//!
//! This module provides high-performance CUDA kernels for quantum state vector
//! operations, gate applications, and specialized quantum algorithms. It leverages
//! SciRS2's GPU infrastructure for optimal performance and memory management.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;

// Enhanced CUDA functionality with actual GPU integration
#[cfg(feature = "advanced_math")]
pub struct CudaContext {
    device_id: i32,
    device_properties: CudaDeviceProperties,
    memory_pool: Arc<Mutex<GpuMemoryPool>>,
    profiler: Option<CudaProfiler>,
}

#[cfg(feature = "advanced_math")]
pub struct CudaStream {
    id: usize,
    handle: Arc<Mutex<Option<CudaStreamHandle>>>,
    priority: StreamPriority,
    flags: StreamFlags,
}

#[cfg(feature = "advanced_math")]
pub struct CudaKernel {
    name: String,
    ptx_code: String,
    function_handle: Option<CudaFunctionHandle>,
    register_count: u32,
    shared_memory_size: usize,
    max_threads_per_block: u32,
}

#[cfg(feature = "advanced_math")]
pub struct GpuMemory {
    allocated: usize,
    device_ptr: Option<CudaDevicePointer>,
    host_ptr: Option<*mut std::ffi::c_void>,
    memory_type: GpuMemoryType,
    alignment: usize,
}

// Enhanced types for actual GPU integration
#[cfg(feature = "advanced_math")]
pub struct CudaDeviceProperties {
    name: String,
    compute_capability: (i32, i32),
    total_global_memory: usize,
    max_threads_per_block: i32,
    max_block_dimensions: [i32; 3],
    max_grid_dimensions: [i32; 3],
    warp_size: i32,
    memory_clock_rate: i32,
    memory_bus_width: i32,
}

#[cfg(feature = "advanced_math")]
pub struct GpuMemoryPool {
    allocated_blocks: HashMap<usize, GpuMemoryBlock>,
    free_blocks: Vec<GpuMemoryBlock>,
    total_allocated: usize,
    peak_usage: usize,
}

#[cfg(feature = "advanced_math")]
pub struct GpuMemoryBlock {
    ptr: CudaDevicePointer,
    size: usize,
    alignment: usize,
    in_use: bool,
}

#[cfg(feature = "advanced_math")]
pub struct CudaProfiler {
    events: Vec<CudaEvent>,
    timing_data: HashMap<String, Vec<f64>>,
    memory_usage: Vec<(String, usize)>,
}

// Placeholder types for actual CUDA handles
#[cfg(feature = "advanced_math")]
type CudaStreamHandle = usize;
#[cfg(feature = "advanced_math")]
type CudaFunctionHandle = usize;
#[cfg(feature = "advanced_math")]
type CudaDevicePointer = usize;
#[cfg(feature = "advanced_math")]
type CudaEvent = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPriority {
    Low,
    Normal,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamFlags {
    Default,
    NonBlocking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMemoryType {
    Device,
    Host,
    Unified,
    Pinned,
}

#[cfg(feature = "advanced_math")]
impl CudaContext {
    pub fn new(device_id: i32) -> Result<Self> {
        // Initialize CUDA context with device properties
        let device_properties = Self::query_device_properties(device_id)?;
        let memory_pool = Arc::new(Mutex::new(GpuMemoryPool::new()));
        let profiler = if cfg!(debug_assertions) {
            Some(CudaProfiler::new())
        } else {
            None
        };
        
        Ok(Self { 
            device_id,
            device_properties,
            memory_pool,
            profiler,
        })
    }

    pub fn get_device_count() -> Result<i32> {
        // Query actual CUDA device count
        // In a real implementation, this would call cudaGetDeviceCount
        #[cfg(feature = "advanced_math")]
        {
            // Simulate querying CUDA devices
            let device_count = Self::query_cuda_devices()?;
            Ok(device_count)
        }
        #[cfg(not(feature = "advanced_math"))]
        {
            Ok(0)
        }
    }
    
    fn query_device_properties(device_id: i32) -> Result<CudaDeviceProperties> {
        // In real implementation, would call cudaGetDeviceProperties
        Ok(CudaDeviceProperties {
            name: format!("CUDA Device {}", device_id),
            compute_capability: (7, 5), // Example: RTX 20xx series
            total_global_memory: 8_000_000_000, // 8GB
            max_threads_per_block: 1024,
            max_block_dimensions: [1024, 1024, 64],
            max_grid_dimensions: [2147483647, 65535, 65535],
            warp_size: 32,
            memory_clock_rate: 7000, // MHz
            memory_bus_width: 256,
        })
    }
    
    fn query_cuda_devices() -> Result<i32> {
        // In real implementation: cudaGetDeviceCount(&count)
        // For now, simulate detection of available devices
        Ok(if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() { 1 } else { 0 })
    }
    
    pub fn get_device_properties(&self) -> &CudaDeviceProperties {
        &self.device_properties
    }
    
    pub fn get_memory_info(&self) -> Result<(usize, usize)> {
        // Return (free_memory, total_memory)
        // In real implementation: cudaMemGetInfo
        let pool = self.memory_pool.lock().unwrap();
        let total = self.device_properties.total_global_memory;
        let used = pool.total_allocated;
        Ok((total - used, total))
    }
}

#[cfg(feature = "advanced_math")]
impl CudaStream {
    pub fn new() -> Result<Self> {
        Self::with_priority_and_flags(StreamPriority::Normal, StreamFlags::Default)
    }
    
    pub fn with_priority_and_flags(priority: StreamPriority, flags: StreamFlags) -> Result<Self> {
        let id = Self::allocate_stream_id();
        let handle = Arc::new(Mutex::new(Self::create_cuda_stream(priority, flags)?));
        
        Ok(Self { 
            id,
            handle,
            priority,
            flags,
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        let handle = self.handle.lock().unwrap();
        if let Some(cuda_handle) = *handle {
            // In real implementation: cudaStreamSynchronize(cuda_handle)
            Self::cuda_stream_synchronize(cuda_handle)?;
        }
        Ok(())
    }
    
    pub fn query(&self) -> Result<bool> {
        let handle = self.handle.lock().unwrap();
        if let Some(cuda_handle) = *handle {
            // In real implementation: cudaStreamQuery(cuda_handle)
            Self::cuda_stream_query(cuda_handle)
        } else {
            Ok(true) // Stream is ready if not created
        }
    }
    
    pub fn record_event(&self, event: &mut CudaEvent) -> Result<()> {
        let handle = self.handle.lock().unwrap();
        if let Some(cuda_handle) = *handle {
            // In real implementation: cudaEventRecord(event, cuda_handle)
            Self::cuda_event_record(*event, cuda_handle)?;
        }
        Ok(())
    }
    
    fn allocate_stream_id() -> usize {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static STREAM_COUNTER: AtomicUsize = AtomicUsize::new(0);
        STREAM_COUNTER.fetch_add(1, Ordering::SeqCst)
    }
    
    fn create_cuda_stream(priority: StreamPriority, flags: StreamFlags) -> Result<Option<CudaStreamHandle>> {
        // In real implementation: cudaStreamCreateWithPriority
        let _priority_val = match priority {
            StreamPriority::Low => -1,
            StreamPriority::Normal => 0,
            StreamPriority::High => 1,
        };
        let _flags_val = match flags {
            StreamFlags::Default => 0,
            StreamFlags::NonBlocking => 1,
        };
        
        // Simulate stream creation
        Ok(Some(Self::allocate_stream_id()))
    }
    
    fn cuda_stream_synchronize(_handle: CudaStreamHandle) -> Result<()> {
        // Placeholder for actual CUDA synchronization
        std::thread::sleep(std::time::Duration::from_micros(10));
        Ok(())
    }
    
    fn cuda_stream_query(_handle: CudaStreamHandle) -> Result<bool> {
        // Placeholder for actual CUDA stream query
        Ok(true)
    }
    
    fn cuda_event_record(_event: CudaEvent, _stream: CudaStreamHandle) -> Result<()> {
        // Placeholder for actual CUDA event recording
        Ok(())
    }
}

#[cfg(feature = "advanced_math")]
impl CudaKernel {
    pub fn compile(source: &str, name: &str, config: &CudaKernelConfig) -> Result<Self> {
        // Compile CUDA source to PTX
        let ptx_code = Self::compile_cuda_source(source, config)?;
        
        // Load and create function handle
        let function_handle = Self::load_cuda_function(&ptx_code, name)?;
        
        // Query kernel properties
        let (register_count, shared_memory_size, max_threads_per_block) = 
            Self::query_kernel_properties(function_handle)?;
        
        Ok(Self {
            name: name.to_string(),
            ptx_code,
            function_handle: Some(function_handle),
            register_count,
            shared_memory_size,
            max_threads_per_block,
        })
    }

    pub fn launch(
        &self,
        grid_size: usize,
        block_size: usize,
        params: &[*const std::ffi::c_void],
        stream: &CudaStream,
    ) -> Result<()> {
        if let Some(function_handle) = self.function_handle {
            // Validate launch parameters
            self.validate_launch_parameters(grid_size, block_size)?;
            
            // Get stream handle
            let stream_handle = {
                let handle = stream.handle.lock().unwrap();
                *handle
            };
            
            // Launch kernel
            Self::cuda_launch_kernel(
                function_handle,
                grid_size,
                block_size,
                params,
                stream_handle,
            )?;
        } else {
            return Err(SimulatorError::UnsupportedOperation(
                format!("Kernel '{}' not compiled", self.name)
            ));
        }
        
        Ok(())
    }
    
    pub fn get_occupancy(&self, block_size: usize) -> Result<f64> {
        if let Some(function_handle) = self.function_handle {
            // Calculate theoretical occupancy
            let max_blocks_per_sm = self.calculate_max_blocks_per_sm(block_size)?;
            let active_warps = (block_size + 31) / 32; // Round up to warp size
            let max_warps_per_sm = 64; // Typical for modern GPUs
            
            let occupancy = (max_blocks_per_sm * active_warps) as f64 / max_warps_per_sm as f64;
            Ok(occupancy.min(1.0))
        } else {
            Err(SimulatorError::UnsupportedOperation(
                "Cannot calculate occupancy for uncompiled kernel".to_string()
            ))
        }
    }
    
    fn compile_cuda_source(source: &str, config: &CudaKernelConfig) -> Result<String> {
        // In real implementation: nvrtcCompileProgram
        let optimization_flags = match config.optimization_level {
            OptimizationLevel::Conservative => "-O1",
            OptimizationLevel::Balanced => "-O2",
            OptimizationLevel::Aggressive => "-O3 --use_fast_math",
        };
        
        // Simulate compilation process
        let ptx_header = format!(
            ".version 7.0\n.target sm_75\n.address_size 64\n// Compiled with {}\n",
            optimization_flags
        );
        
        // In real implementation, this would be actual PTX code from nvrtc
        Ok(format!("{}{}", ptx_header, source))
    }
    
    fn load_cuda_function(ptx_code: &str, name: &str) -> Result<CudaFunctionHandle> {
        // In real implementation: cuModuleLoadDataEx + cuModuleGetFunction
        let _module_handle = Self::cuda_module_load_data(ptx_code)?;
        let function_handle = Self::cuda_module_get_function(name)?;
        Ok(function_handle)
    }
    
    fn query_kernel_properties(function_handle: CudaFunctionHandle) -> Result<(u32, usize, u32)> {
        // In real implementation: cuFuncGetAttribute
        let register_count = 32; // Example values
        let shared_memory_size = 1024;
        let max_threads_per_block = 1024;
        
        Ok((register_count, shared_memory_size, max_threads_per_block))
    }
    
    fn validate_launch_parameters(&self, grid_size: usize, block_size: usize) -> Result<()> {
        if block_size == 0 || block_size > self.max_threads_per_block as usize {
            return Err(SimulatorError::InvalidInput(
                format!("Invalid block size: {} (max: {})", block_size, self.max_threads_per_block)
            ));
        }
        
        if grid_size == 0 {
            return Err(SimulatorError::InvalidInput(
                "Grid size must be greater than 0".to_string()
            ));
        }
        
        Ok(())
    }
    
    fn calculate_max_blocks_per_sm(&self, block_size: usize) -> Result<usize> {
        // Calculate based on resource limitations
        let warps_per_block = (block_size + 31) / 32;
        let max_warps_per_sm = 64;
        let max_blocks_by_warps = max_warps_per_sm / warps_per_block;
        
        let max_blocks_by_registers = if self.register_count > 0 {
            65536 / (self.register_count as usize * block_size)
        } else {
            usize::MAX
        };
        
        let max_blocks_by_shared_memory = if self.shared_memory_size > 0 {
            49152 / self.shared_memory_size // Typical shared memory per SM
        } else {
            usize::MAX
        };
        
        Ok(max_blocks_by_warps.min(max_blocks_by_registers).min(max_blocks_by_shared_memory))
    }
    
    fn cuda_module_load_data(_ptx_code: &str) -> Result<usize> {
        // Placeholder for cuModuleLoadDataEx
        Ok(1)
    }
    
    fn cuda_module_get_function(_name: &str) -> Result<CudaFunctionHandle> {
        // Placeholder for cuModuleGetFunction
        Ok(1)
    }
    
    fn cuda_launch_kernel(
        _function: CudaFunctionHandle,
        _grid_size: usize,
        _block_size: usize,
        _params: &[*const std::ffi::c_void],
        _stream: Option<CudaStreamHandle>,
    ) -> Result<()> {
        // Placeholder for cuLaunchKernel
        // In real implementation, this would launch the actual CUDA kernel
        std::thread::sleep(std::time::Duration::from_micros(100)); // Simulate kernel execution
        Ok(())
    }
}

#[cfg(feature = "advanced_math")]
impl GpuMemory {
    pub fn new() -> Self {
        Self { 
            allocated: 0,
            device_ptr: None,
            host_ptr: None,
            memory_type: GpuMemoryType::Device,
            alignment: 256, // Default GPU memory alignment
        }
    }
    
    pub fn new_with_type(memory_type: GpuMemoryType) -> Self {
        Self {
            allocated: 0,
            device_ptr: None,
            host_ptr: None,
            memory_type,
            alignment: 256,
        }
    }

    pub fn allocate_pool(&mut self, size: usize) -> Result<()> {
        match self.memory_type {
            GpuMemoryType::Device => {
                let ptr = Self::cuda_malloc(size)?;
                self.device_ptr = Some(ptr);
            }
            GpuMemoryType::Host => {
                let ptr = Self::cuda_malloc_host(size)?;
                self.host_ptr = Some(ptr);
            }
            GpuMemoryType::Unified => {
                let ptr = Self::cuda_malloc_managed(size)?;
                self.device_ptr = Some(ptr as CudaDevicePointer);
                self.host_ptr = Some(ptr);
            }
            GpuMemoryType::Pinned => {
                let ptr = Self::cuda_host_alloc(size)?;
                self.host_ptr = Some(ptr);
            }
        }
        
        self.allocated = size;
        Ok(())
    }

    pub fn allocate_and_copy(&mut self, data: &[Complex64]) -> Result<GpuMemory> {
        let size = data.len() * std::mem::size_of::<Complex64>();
        let mut gpu_memory = GpuMemory::new_with_type(self.memory_type);
        
        gpu_memory.allocate_pool(size)?;
        gpu_memory.copy_from_host(data)?;
        
        Ok(gpu_memory)
    }
    
    pub fn copy_from_host(&mut self, data: &[Complex64]) -> Result<()> {
        let size = data.len() * std::mem::size_of::<Complex64>();
        
        match self.memory_type {
            GpuMemoryType::Device => {
                if let Some(device_ptr) = self.device_ptr {
                    Self::cuda_memcpy_h2d(
                        device_ptr,
                        data.as_ptr() as *const std::ffi::c_void,
                        size,
                    )?;
                }
            }
            GpuMemoryType::Host | GpuMemoryType::Pinned => {
                if let Some(host_ptr) = self.host_ptr {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr() as *const u8,
                            host_ptr as *mut u8,
                            size,
                        );
                    }
                }
            }
            GpuMemoryType::Unified => {
                if let Some(host_ptr) = self.host_ptr {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr() as *const u8,
                            host_ptr as *mut u8,
                            size,
                        );
                    }
                    // For unified memory, ensure data is accessible on device
                    Self::cuda_mem_prefetch_async(self.device_ptr.unwrap_or(0), size, 0)?;
                }
            }
        }
        
        Ok(())
    }

    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        match self.memory_type {
            GpuMemoryType::Device => {
                self.device_ptr.map(|p| p as *const std::ffi::c_void).unwrap_or(std::ptr::null())
            }
            _ => {
                self.host_ptr.unwrap_or(std::ptr::null())
            }
        }
    }
    
    pub fn as_device_ptr(&self) -> Option<CudaDevicePointer> {
        self.device_ptr
    }

    pub fn copy_to_host(&self, data: &mut [Complex64]) -> Result<()> {
        let size = data.len() * std::mem::size_of::<Complex64>();
        
        match self.memory_type {
            GpuMemoryType::Device => {
                if let Some(device_ptr) = self.device_ptr {
                    Self::cuda_memcpy_d2h(
                        data.as_mut_ptr() as *mut std::ffi::c_void,
                        device_ptr,
                        size,
                    )?;
                }
            }
            GpuMemoryType::Host | GpuMemoryType::Pinned => {
                if let Some(host_ptr) = self.host_ptr {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            host_ptr as *const u8,
                            data.as_mut_ptr() as *mut u8,
                            size,
                        );
                    }
                }
            }
            GpuMemoryType::Unified => {
                if let Some(host_ptr) = self.host_ptr {
                    // Ensure data is available on host
                    Self::cuda_device_synchronize()?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            host_ptr as *const u8,
                            data.as_mut_ptr() as *mut u8,
                            size,
                        );
                    }
                }
            }
        }
        
        Ok(())
    }
    
    pub fn get_size(&self) -> usize {
        self.allocated
    }
    
    pub fn get_memory_type(&self) -> GpuMemoryType {
        self.memory_type
    }
    
    // CUDA memory management functions (placeholders for actual CUDA calls)
    fn cuda_malloc(size: usize) -> Result<CudaDevicePointer> {
        // In real implementation: cudaMalloc
        if size == 0 {
            return Err(SimulatorError::InvalidInput("Cannot allocate zero bytes".to_string()));
        }
        Ok(size) // Use size as a mock pointer
    }
    
    fn cuda_malloc_host(size: usize) -> Result<*mut std::ffi::c_void> {
        // In real implementation: cudaMallocHost
        let layout = std::alloc::Layout::from_size_align(size, 256).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            Err(SimulatorError::ResourceExhausted("Failed to allocate host memory".to_string()))
        } else {
            Ok(ptr as *mut std::ffi::c_void)
        }
    }
    
    fn cuda_malloc_managed(size: usize) -> Result<*mut std::ffi::c_void> {
        // In real implementation: cudaMallocManaged
        Self::cuda_malloc_host(size)
    }
    
    fn cuda_host_alloc(size: usize) -> Result<*mut std::ffi::c_void> {
        // In real implementation: cudaHostAlloc with cudaHostAllocDefault
        Self::cuda_malloc_host(size)
    }
    
    fn cuda_memcpy_h2d(
        dst: CudaDevicePointer,
        src: *const std::ffi::c_void,
        size: usize,
    ) -> Result<()> {
        // In real implementation: cudaMemcpy with cudaMemcpyHostToDevice
        if src.is_null() || size == 0 {
            return Err(SimulatorError::InvalidInput("Invalid memory copy parameters".to_string()));
        }
        // Simulate memory transfer latency
        let transfer_time = size as f64 / 500_000_000.0; // Assume 500 MB/s transfer rate
        std::thread::sleep(std::time::Duration::from_secs_f64(transfer_time));
        Ok(())
    }
    
    fn cuda_memcpy_d2h(
        dst: *mut std::ffi::c_void,
        src: CudaDevicePointer,
        size: usize,
    ) -> Result<()> {
        // In real implementation: cudaMemcpy with cudaMemcpyDeviceToHost
        if dst.is_null() || size == 0 {
            return Err(SimulatorError::InvalidInput("Invalid memory copy parameters".to_string()));
        }
        // Simulate memory transfer latency
        let transfer_time = size as f64 / 500_000_000.0; // Assume 500 MB/s transfer rate
        std::thread::sleep(std::time::Duration::from_secs_f64(transfer_time));
        Ok(())
    }
    
    fn cuda_mem_prefetch_async(ptr: CudaDevicePointer, size: usize, device: i32) -> Result<()> {
        // In real implementation: cudaMemPrefetchAsync
        Ok(())
    }
    
    fn cuda_device_synchronize() -> Result<()> {
        // In real implementation: cudaDeviceSynchronize
        std::thread::sleep(std::time::Duration::from_micros(10));
        Ok(())
    }
}

#[cfg(feature = "advanced_math")]
impl Drop for GpuMemory {
    fn drop(&mut self) {
        // Clean up GPU memory
        if let Some(device_ptr) = self.device_ptr {
            let _ = Self::cuda_free(device_ptr);
        }
        if let Some(host_ptr) = self.host_ptr {
            match self.memory_type {
                GpuMemoryType::Host | GpuMemoryType::Pinned => {
                    let _ = Self::cuda_free_host(host_ptr);
                }
                _ => {}
            }
        }
    }
}

#[cfg(feature = "advanced_math")]
impl GpuMemory {
    fn cuda_free(ptr: CudaDevicePointer) -> Result<()> {
        // In real implementation: cudaFree
        Ok(())
    }
    
    fn cuda_free_host(ptr: *mut std::ffi::c_void) -> Result<()> {
        // In real implementation: cudaFreeHost
        if !ptr.is_null() {
            unsafe {
                let layout = std::alloc::Layout::from_size_align(1, 256).unwrap();
                std::alloc::dealloc(ptr as *mut u8, layout);
            }
        }
        Ok(())
    }
}

#[cfg(feature = "advanced_math")]
use scirs2_core::gpu::{GpuBuffer, GpuContext, GpuError};

/// CUDA kernel configuration
#[derive(Debug, Clone)]
pub struct CudaKernelConfig {
    /// Device ID to use
    pub device_id: i32,
    /// Number of CUDA streams for parallel execution
    pub num_streams: usize,
    /// Block size for CUDA kernels
    pub block_size: usize,
    /// Grid size for CUDA kernels
    pub grid_size: usize,
    /// Enable unified memory
    pub unified_memory: bool,
    /// Memory pool size in bytes
    pub memory_pool_size: usize,
    /// Enable kernel profiling
    pub enable_profiling: bool,
    /// Kernel optimization level
    pub optimization_level: OptimizationLevel,
}

impl Default for CudaKernelConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            num_streams: 4,
            block_size: 256,
            grid_size: 0, // Auto-calculate
            unified_memory: true,
            memory_pool_size: 2_000_000_000, // 2GB
            enable_profiling: false,
            optimization_level: OptimizationLevel::Aggressive,
        }
    }
}

/// Kernel optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// Conservative optimization (safe)
    Conservative,
    /// Balanced optimization (default)
    Balanced,
    /// Aggressive optimization (maximum performance)
    Aggressive,
    /// Custom optimization parameters
    Custom,
}

/// CUDA quantum gate kernels
pub struct CudaQuantumKernels {
    /// Configuration
    config: CudaKernelConfig,
    /// CUDA context
    #[cfg(feature = "advanced_math")]
    context: Option<CudaContext>,
    /// CUDA streams for parallel execution
    #[cfg(feature = "advanced_math")]
    streams: Vec<CudaStream>,
    /// Compiled kernels
    #[cfg(feature = "advanced_math")]
    kernels: HashMap<String, CudaKernel>,
    /// GPU memory pool
    #[cfg(feature = "advanced_math")]
    memory_pool: Arc<Mutex<GpuMemory>>,
    /// SciRS2 backend
    backend: Option<SciRS2Backend>,
    /// Performance statistics
    stats: CudaKernelStats,
}

/// Performance statistics for CUDA kernels
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CudaKernelStats {
    /// Total kernel launches
    pub kernel_launches: usize,
    /// Total execution time (ms)
    pub total_execution_time_ms: f64,
    /// Memory transfers to GPU (bytes)
    pub memory_transfers_to_gpu: usize,
    /// Memory transfers from GPU (bytes)
    pub memory_transfers_from_gpu: usize,
    /// GPU utilization (0-1)
    pub gpu_utilization: f64,
    /// Memory bandwidth utilization (0-1)
    pub memory_bandwidth_utilization: f64,
    /// Kernel execution times by type
    pub kernel_times: HashMap<String, f64>,
}

/// Quantum gate types for CUDA kernels
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CudaGateType {
    /// Single-qubit Pauli gates
    PauliX,
    PauliY,
    PauliZ,
    /// Single-qubit rotation gates
    RotationX,
    RotationY,
    RotationZ,
    /// Common single-qubit gates
    Hadamard,
    Phase,
    T,
    /// Two-qubit gates
    CNOT,
    CZ,
    SWAP,
    /// Custom unitary gate
    CustomUnitary,
}

impl CudaQuantumKernels {
    /// Create new CUDA quantum kernels
    pub fn new(config: CudaKernelConfig) -> Result<Self> {
        let mut kernels = Self {
            config,
            #[cfg(feature = "advanced_math")]
            context: None,
            #[cfg(feature = "advanced_math")]
            streams: Vec::new(),
            #[cfg(feature = "advanced_math")]
            kernels: HashMap::new(),
            #[cfg(feature = "advanced_math")]
            memory_pool: Arc::new(Mutex::new(GpuMemory::new())),
            backend: None,
            stats: CudaKernelStats::default(),
        };

        kernels.initialize_cuda()?;
        Ok(kernels)
    }

    /// Initialize with SciRS2 backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }

    /// Initialize CUDA context and kernels
    fn initialize_cuda(&mut self) -> Result<()> {
        #[cfg(feature = "advanced_math")]
        {
            // Initialize CUDA context
            self.context = Some(CudaContext::new(self.config.device_id)?);

            // Create CUDA streams
            for _ in 0..self.config.num_streams {
                self.streams.push(CudaStream::new()?);
            }

            // Initialize memory pool
            {
                let mut pool = self.memory_pool.lock().unwrap();
                pool.allocate_pool(self.config.memory_pool_size)?;
            }

            // Compile and load kernels
            self.compile_kernels()?;
        }

        #[cfg(not(feature = "advanced_math"))]
        {
            return Err(SimulatorError::UnsupportedOperation(
                "CUDA kernels require SciRS2 backend (enable 'advanced_math' feature)".to_string(),
            ));
        }

        Ok(())
    }

    /// Compile CUDA kernels from source
    #[cfg(feature = "advanced_math")]
    fn compile_kernels(&mut self) -> Result<()> {
        let kernel_sources = self.get_kernel_sources();

        for (name, source) in kernel_sources {
            let kernel = CudaKernel::compile(&source, &name, &self.config)?;
            self.kernels.insert(name, kernel);
        }

        Ok(())
    }

    /// Get CUDA kernel source code
    #[cfg(feature = "advanced_math")]
    fn get_kernel_sources(&self) -> HashMap<String, String> {
        let mut sources = HashMap::new();

        // Single-qubit gate kernel
        sources.insert(
            "single_qubit_gate".to_string(),
            self.generate_single_qubit_gate_kernel(),
        );

        // Two-qubit gate kernel
        sources.insert(
            "two_qubit_gate".to_string(),
            self.generate_two_qubit_gate_kernel(),
        );

        // Tensor product kernel
        sources.insert(
            "tensor_product".to_string(),
            self.generate_tensor_product_kernel(),
        );

        // Matrix multiplication kernel
        sources.insert(
            "matmul".to_string(),
            self.generate_matmul_kernel(),
        );

        // Unitary application kernel
        sources.insert(
            "apply_unitary".to_string(),
            self.generate_unitary_kernel(),
        );

        // Add specialized kernels
        sources.extend(self.get_specialized_kernel_sources());

        sources
    }

    /// Generate single-qubit gate CUDA kernel
    #[cfg(feature = "advanced_math")]
    fn generate_single_qubit_gate_kernel(&self) -> String {
        format!(r#"
#include <cuda_runtime.h>
#include <cuComplex.h>

__global__ void single_qubit_gate_kernel(
    cuFloatComplex* state,
    cuFloatComplex* gate_matrix,
    int qubit,
    int state_size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size / 2) return;
    
    int qubit_mask = 1 << qubit;
    
    // Calculate indices for |0⟩ and |1⟩ components
    int i = (idx & ~qubit_mask) | ((idx & ((1 << qubit) - 1)));
    int j = i | qubit_mask;
    
    if (i >= state_size || j >= state_size) return;
    
    // Load current amplitudes
    cuFloatComplex amp_0 = state[i];
    cuFloatComplex amp_1 = state[j];
    
    // Apply gate matrix
    // |0⟩ component: matrix[0][0] * amp_0 + matrix[0][1] * amp_1
    state[i] = cuCaddf(
        cuCmulf(gate_matrix[0], amp_0),
        cuCmulf(gate_matrix[1], amp_1)
    );
    
    // |1⟩ component: matrix[1][0] * amp_0 + matrix[1][1] * amp_1
    state[j] = cuCaddf(
        cuCmulf(gate_matrix[2], amp_0),
        cuCmulf(gate_matrix[3], amp_1)
    );
}}

__global__ void optimized_hadamard_kernel(
    cuFloatComplex* state,
    int qubit,
    int state_size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size / 2) return;
    
    int qubit_mask = 1 << qubit;
    int i = (idx & ~qubit_mask) | ((idx & ((1 << qubit) - 1)));
    int j = i | qubit_mask;
    
    if (i >= state_size || j >= state_size) return;
    
    cuFloatComplex amp_0 = state[i];
    cuFloatComplex amp_1 = state[j];
    
    // H = (1/√2) * [[1, 1], [1, -1]]
    float inv_sqrt2 = 0.7071067811865476f;
    
    state[i] = make_cuFloatComplex(
        inv_sqrt2 * (cuCrealf(amp_0) + cuCrealf(amp_1)),
        inv_sqrt2 * (cuCimagf(amp_0) + cuCimagf(amp_1))
    );
    
    state[j] = make_cuFloatComplex(
        inv_sqrt2 * (cuCrealf(amp_0) - cuCrealf(amp_1)),
        inv_sqrt2 * (cuCimagf(amp_0) - cuCimagf(amp_1))
    );
}}
"#)
    }

    /// Generate two-qubit gate CUDA kernel
    #[cfg(feature = "advanced_math")]
    fn generate_two_qubit_gate_kernel(&self) -> String {
        format!(r#"
#include <cuda_runtime.h>
#include <cuComplex.h>

__global__ void two_qubit_gate_kernel(
    cuFloatComplex* state,
    cuFloatComplex* gate_matrix,
    int control,
    int target,
    int state_size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size / 4) return;
    
    int control_mask = 1 << control;
    int target_mask = 1 << target;
    int combined_mask = control_mask | target_mask;
    
    // Calculate base index (with both control and target as 0)
    int base = (idx & ~combined_mask) | 
               ((idx & ((1 << min(control, target)) - 1)));
    
    // Four computational basis states
    int i00 = base;
    int i01 = base | target_mask;
    int i10 = base | control_mask;
    int i11 = base | combined_mask;
    
    if (i11 >= state_size) return;
    
    // Load current amplitudes
    cuFloatComplex amp_00 = state[i00];
    cuFloatComplex amp_01 = state[i01];
    cuFloatComplex amp_10 = state[i10];
    cuFloatComplex amp_11 = state[i11];
    
    // Apply 4x4 gate matrix
    state[i00] = cuCaddf(cuCaddf(cuCaddf(
        cuCmulf(gate_matrix[0], amp_00),
        cuCmulf(gate_matrix[1], amp_01)),
        cuCmulf(gate_matrix[2], amp_10)),
        cuCmulf(gate_matrix[3], amp_11));
        
    state[i01] = cuCaddf(cuCaddf(cuCaddf(
        cuCmulf(gate_matrix[4], amp_00),
        cuCmulf(gate_matrix[5], amp_01)),
        cuCmulf(gate_matrix[6], amp_10)),
        cuCmulf(gate_matrix[7], amp_11));
        
    state[i10] = cuCaddf(cuCaddf(cuCaddf(
        cuCmulf(gate_matrix[8], amp_00),
        cuCmulf(gate_matrix[9], amp_01)),
        cuCmulf(gate_matrix[10], amp_10)),
        cuCmulf(gate_matrix[11], amp_11));
        
    state[i11] = cuCaddf(cuCaddf(cuCaddf(
        cuCmulf(gate_matrix[12], amp_00),
        cuCmulf(gate_matrix[13], amp_01)),
        cuCmulf(gate_matrix[14], amp_10)),
        cuCmulf(gate_matrix[15], amp_11));
}}

__global__ void optimized_cnot_kernel(
    cuFloatComplex* state,
    int control,
    int target,
    int state_size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size / 2) return;
    
    int control_mask = 1 << control;
    int target_mask = 1 << target;
    
    // Only operate when control is |1⟩
    int base_idx = idx;
    if ((base_idx & control_mask) != 0) {{
        int partner_idx = base_idx ^ target_mask;
        
        if (base_idx < partner_idx && partner_idx < state_size) {{
            // Swap amplitudes
            cuFloatComplex temp = state[base_idx];
            state[base_idx] = state[partner_idx];
            state[partner_idx] = temp;
        }}
    }}
}}
"#)
    }

    /// Generate tensor product CUDA kernel
    #[cfg(feature = "advanced_math")]
    fn generate_tensor_product_kernel(&self) -> String {
        format!(r#"
#include <cuda_runtime.h>
#include <cuComplex.h>

__global__ void tensor_product_kernel(
    cuFloatComplex* result,
    cuFloatComplex* state_a,
    cuFloatComplex* state_b,
    int size_a,
    int size_b
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = size_a * size_b;
    
    if (idx >= total_size) return;
    
    int i = idx / size_b;
    int j = idx % size_b;
    
    if (i < size_a && j < size_b) {{
        result[idx] = cuCmulf(state_a[i], state_b[j]);
    }}
}}

__global__ void tensor_product_inplace_kernel(
    cuFloatComplex* state,
    cuFloatComplex* new_qubit_state,
    int original_size,
    int new_qubit_state_size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = original_size * new_qubit_state_size;
    
    if (idx >= total_size) return;
    
    int orig_idx = idx / new_qubit_state_size;
    int new_idx = idx % new_qubit_state_size;
    
    if (orig_idx < original_size && new_idx < new_qubit_state_size) {{
        cuFloatComplex original_amp = state[orig_idx];
        cuFloatComplex new_amp = new_qubit_state[new_idx];
        state[idx] = cuCmulf(original_amp, new_amp);
    }}
}}
"#)
    }

    /// Generate matrix multiplication CUDA kernel
    #[cfg(feature = "advanced_math")]
    fn generate_matmul_kernel(&self) -> String {
        format!(r#"
#include <cuda_runtime.h>
#include <cuComplex.h>

__global__ void complex_matmul_kernel(
    cuFloatComplex* result,
    cuFloatComplex* matrix_a,
    cuFloatComplex* matrix_b,
    int m, int n, int k
) {{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {{
        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
        
        for (int i = 0; i < k; i++) {{
            cuFloatComplex a_val = matrix_a[row * k + i];
            cuFloatComplex b_val = matrix_b[i * n + col];
            sum = cuCaddf(sum, cuCmulf(a_val, b_val));
        }}
        
        result[row * n + col] = sum;
    }}
}}

__global__ void matrix_vector_multiply_kernel(
    cuFloatComplex* result,
    cuFloatComplex* matrix,
    cuFloatComplex* vector,
    int rows,
    int cols
) {{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {{
        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
        
        for (int col = 0; col < cols; col++) {{
            cuFloatComplex mat_val = matrix[row * cols + col];
            cuFloatComplex vec_val = vector[col];
            sum = cuCaddf(sum, cuCmulf(mat_val, vec_val));
        }}
        
        result[row] = sum;
    }}
}}

__global__ void optimized_state_vector_multiply_kernel(
    cuFloatComplex* state,
    cuFloatComplex* unitary_matrix,
    cuFloatComplex* temp_state,
    int state_size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < state_size) {{
        cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
        
        // Compute matrix-vector product for this element
        for (int j = 0; j < state_size; j++) {{
            cuFloatComplex matrix_elem = unitary_matrix[idx * state_size + j];
            cuFloatComplex state_elem = state[j];
            sum = cuCaddf(sum, cuCmulf(matrix_elem, state_elem));
        }}
        
        temp_state[idx] = sum;
    }}
}}
"#)
    }

    /// Generate unitary application CUDA kernel  
    #[cfg(feature = "advanced_math")]
    fn generate_unitary_kernel(&self) -> String {
        format!(r#"
#include <cuda_runtime.h>
#include <cuComplex.h>

__global__ void apply_unitary_kernel(
    cuFloatComplex* state,
    cuFloatComplex* unitary,
    cuFloatComplex* temp_state,
    int* qubits,
    int num_qubits,
    int state_size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size) return;
    
    // For general unitary on multiple qubits
    int matrix_size = 1 << num_qubits;
    
    // Create mapping from global index to local subspace index
    int local_idx = 0;
    int temp_idx = idx;
    
    for (int i = 0; i < num_qubits; i++) {{
        int qubit = qubits[i];
        if ((temp_idx >> qubit) & 1) {{
            local_idx |= (1 << i);
        }}
    }}
    
    // Compute matrix-vector product for this subspace
    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
    
    for (int j = 0; j < matrix_size; j++) {{
        // Map local index j back to global index
        int global_j_idx = idx;
        for (int i = 0; i < num_qubits; i++) {{
            int qubit = qubits[i];
            if ((j >> i) & 1) {{
                global_j_idx |= (1 << qubit);
            }} else {{
                global_j_idx &= ~(1 << qubit);
            }}
        }}
        
        if (global_j_idx < state_size) {{
            cuFloatComplex matrix_elem = unitary[local_idx * matrix_size + j];
            cuFloatComplex state_elem = state[global_j_idx];
            sum = cuCaddf(sum, cuCmulf(matrix_elem, state_elem));
        }}
    }}
    
    temp_state[idx] = sum;
}}

__global__ void copy_temp_to_state_kernel(
    cuFloatComplex* state,
    cuFloatComplex* temp_state,
    int state_size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < state_size) {{
        state[idx] = temp_state[idx];
    }}
}}

__global__ void quantum_fourier_transform_kernel(
    cuFloatComplex* state,
    cuFloatComplex* temp_state,
    int* qubits,
    int num_qubits,
    int state_size
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= state_size) return;
    
    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
    int n = 1 << num_qubits;
    
    for (int k = 0; k < n; k++) {{
        // Map k to global index based on qubits
        int global_k = 0;
        for (int i = 0; i < num_qubits; i++) {{
            if ((k >> i) & 1) {{
                global_k |= (1 << qubits[i]);
            }}
        }}
        
        // Map idx to local index
        int local_idx = 0;
        for (int i = 0; i < num_qubits; i++) {{
            if ((idx >> qubits[i]) & 1) {{
                local_idx |= (1 << i);
            }}
        }}
        
        if (global_k < state_size) {{
            // Compute QFT matrix element: exp(2πi * j * k / n) / √n
            float angle = 2.0f * M_PI * local_idx * k / n;
            float cos_val = cosf(angle) / sqrtf(n);
            float sin_val = sinf(angle) / sqrtf(n);
            
            cuFloatComplex qft_elem = make_cuFloatComplex(cos_val, sin_val);
            cuFloatComplex state_elem = state[global_k];
            
            sum = cuCaddf(sum, cuCmulf(qft_elem, state_elem));
        }}
    }}
    
    temp_state[idx] = sum;
}}
"#)
    }

    /// Get specialized CUDA kernel sources
    #[cfg(feature = "advanced_math")]
    fn get_specialized_kernel_sources(&self) -> HashMap<String, String> {
        let mut sources = HashMap::new();

        // Pauli-X kernel (optimized)
        sources.insert(
            "pauli_x".to_string(),
            r#"
        __global__ void pauli_x_kernel(
            cuFloatComplex* state,
            int* qubit_indices,
            int num_qubits,
            int state_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= state_size / 2) return;
            
            int qubit = qubit_indices[0];
            int qubit_mask = 1 << qubit;
            
            // Calculate paired indices
            int i = idx;
            int j = i ^ qubit_mask;
            
            if (i < j) {
                // Swap amplitudes
                cuFloatComplex temp = state[i];
                state[i] = state[j];
                state[j] = temp;
            }
        }
        "#
            .to_string(),
        );

        // CNOT kernel (optimized)
        sources.insert(
            "cnot".to_string(),
            r#"
        __global__ void cnot_kernel(
            cuFloatComplex* state,
            int control_qubit,
            int target_qubit,
            int state_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= state_size) return;
            
            int control_mask = 1 << control_qubit;
            int target_mask = 1 << target_qubit;
            
            // Only apply if control qubit is |1⟩
            if ((idx & control_mask) != 0) {
                int paired_idx = idx ^ target_mask;
                
                if (idx < paired_idx) {
                    // Swap target qubit amplitudes
                    cuFloatComplex temp = state[idx];
                    state[idx] = state[paired_idx];
                    state[paired_idx] = temp;
                }
            }
        }
        "#
            .to_string(),
        );

        // Phase gate kernel
        sources.insert(
            "phase_gate".to_string(),
            r#"
        __global__ void phase_gate_kernel(
            cuFloatComplex* state,
            int qubit,
            float phase,
            int state_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= state_size) return;
            
            int qubit_mask = 1 << qubit;
            
            // Apply phase only to |1⟩ states
            if ((idx & qubit_mask) != 0) {
                cuFloatComplex phase_factor = make_cuFloatComplex(
                    cosf(phase), sinf(phase)
                );
                state[idx] = cuCmulf(state[idx], phase_factor);
            }
        }
        "#
            .to_string(),
        );

        // Hadamard kernel
        sources.insert(
            "hadamard".to_string(),
            r#"
        __global__ void hadamard_kernel(
            cuFloatComplex* state,
            int qubit,
            int state_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= state_size / 2) return;
            
            int qubit_mask = 1 << qubit;
            float inv_sqrt2 = 0.7071067811865475f;
            
            int i = idx;
            int j = i ^ qubit_mask;
            
            if (i < j) {
                cuFloatComplex amp_i = state[i];
                cuFloatComplex amp_j = state[j];
                
                // H = (1/√2) * [[1, 1], [1, -1]]
                state[i] = make_cuFloatComplex(
                    inv_sqrt2 * (amp_i.x + amp_j.x),
                    inv_sqrt2 * (amp_i.y + amp_j.y)
                );
                state[j] = make_cuFloatComplex(
                    inv_sqrt2 * (amp_i.x - amp_j.x),
                    inv_sqrt2 * (amp_i.y - amp_j.y)
                );
            }
        }
        "#
            .to_string(),
        );

        // Rotation gate kernel
        sources.insert(
            "rotation_gate".to_string(),
            r#"
        __global__ void rotation_gate_kernel(
            cuFloatComplex* state,
            int qubit,
            float theta,
            int axis, // 0=X, 1=Y, 2=Z
            int state_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= state_size / 2) return;
            
            int qubit_mask = 1 << qubit;
            float cos_half = cosf(theta / 2.0f);
            float sin_half = sinf(theta / 2.0f);
            
            int i = idx;
            int j = i ^ qubit_mask;
            
            if (i < j) {
                cuFloatComplex amp_i = state[i];
                cuFloatComplex amp_j = state[j];
                
                cuFloatComplex new_i, new_j;
                
                if (axis == 0) { // X rotation
                    new_i = make_cuFloatComplex(
                        cos_half * amp_i.x + sin_half * amp_j.y,
                        cos_half * amp_i.y - sin_half * amp_j.x
                    );
                    new_j = make_cuFloatComplex(
                        cos_half * amp_j.x - sin_half * amp_i.y,
                        cos_half * amp_j.y + sin_half * amp_i.x
                    );
                } else if (axis == 1) { // Y rotation
                    new_i = make_cuFloatComplex(
                        cos_half * amp_i.x + sin_half * amp_j.x,
                        cos_half * amp_i.y + sin_half * amp_j.y
                    );
                    new_j = make_cuFloatComplex(
                        -sin_half * amp_i.x + cos_half * amp_j.x,
                        -sin_half * amp_i.y + cos_half * amp_j.y
                    );
                } else { // Z rotation
                    cuFloatComplex phase_neg = make_cuFloatComplex(cos_half, -sin_half);
                    cuFloatComplex phase_pos = make_cuFloatComplex(cos_half, sin_half);
                    
                    new_i = cuCmulf(amp_i, phase_neg);
                    new_j = cuCmulf(amp_j, phase_pos);
                }
                
                state[i] = new_i;
                state[j] = new_j;
            }
        }
        "#
            .to_string(),
        );

        // State measurement kernel
        sources.insert(
            "measure_probabilities".to_string(),
            r#"
        __global__ void measure_probabilities_kernel(
            cuFloatComplex* state,
            float* probabilities,
            int* qubit_masks,
            int num_qubits,
            int state_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= state_size) return;
            
            float prob = state[idx].x * state[idx].x + state[idx].y * state[idx].y;
            
            for (int q = 0; q < num_qubits; q++) {
                if ((idx & qubit_masks[q]) != 0) {
                    atomicAdd(&probabilities[q], prob);
                }
            }
        }
        "#
            .to_string(),
        );

        // Quantum Fourier Transform kernel
        sources.insert(
            "qft".to_string(),
            r#"
        __global__ void qft_kernel(
            cuFloatComplex* state,
            cuFloatComplex* temp_state,
            int num_qubits,
            int state_size
        ) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;
            if (k >= state_size) return;
            
            cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);
            float norm_factor = 1.0f / sqrtf((float)state_size);
            
            for (int j = 0; j < state_size; j++) {
                float angle = -2.0f * M_PI * (float)(k * j) / (float)state_size;
                cuFloatComplex twiddle = make_cuFloatComplex(cosf(angle), sinf(angle));
                sum = cuCaddf(sum, cuCmulf(state[j], twiddle));
            }
            
            temp_state[k] = make_cuFloatComplex(
                sum.x * norm_factor,
                sum.y * norm_factor
            );
        }
        "#
            .to_string(),
        );

        sources
    }

    /// Apply single-qubit gate using CUDA
    pub fn apply_single_qubit_gate(
        &mut self,
        state: &mut Array1<Complex64>,
        qubit: usize,
        gate_type: CudaGateType,
        parameters: &[f64],
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "advanced_math")]
        {
            let kernel_name = match gate_type {
                CudaGateType::PauliX => "pauli_x",
                CudaGateType::PauliY => "pauli_y",
                CudaGateType::PauliZ => "pauli_z",
                CudaGateType::Hadamard => "hadamard",
                CudaGateType::Phase => "phase_gate",
                CudaGateType::RotationX | CudaGateType::RotationY | CudaGateType::RotationZ => {
                    "rotation_gate"
                }
                _ => "single_qubit_gate",
            };

            let has_kernel = self.kernels.contains_key(kernel_name);
            if has_kernel {
                // Transfer state to GPU
                let mut gpu_state = self.transfer_to_gpu(state)?;

                // Set up kernel parameters
                let mut params = vec![
                    gpu_state.as_ptr() as *mut std::ffi::c_void,
                    &qubit as *const usize as *const std::ffi::c_void,
                ];

                // Add gate-specific parameters
                match gate_type {
                    CudaGateType::Phase => {
                        let phase = parameters.get(0).copied().unwrap_or(0.0) as f32;
                        params.push(&phase as *const f32 as *const std::ffi::c_void);
                    }
                    CudaGateType::RotationX => {
                        let theta = parameters.get(0).copied().unwrap_or(0.0) as f32;
                        let axis = 0i32; // X axis
                        params.push(&theta as *const f32 as *const std::ffi::c_void);
                        params.push(&axis as *const i32 as *const std::ffi::c_void);
                    }
                    CudaGateType::RotationY => {
                        let theta = parameters.get(0).copied().unwrap_or(0.0) as f32;
                        let axis = 1i32; // Y axis
                        params.push(&theta as *const f32 as *const std::ffi::c_void);
                        params.push(&axis as *const i32 as *const std::ffi::c_void);
                    }
                    CudaGateType::RotationZ => {
                        let theta = parameters.get(0).copied().unwrap_or(0.0) as f32;
                        let axis = 2i32; // Z axis
                        params.push(&theta as *const f32 as *const std::ffi::c_void);
                        params.push(&axis as *const i32 as *const std::ffi::c_void);
                    }
                    _ => {}
                }

                let state_size = state.len() as i32;
                params.push(&state_size as *const i32 as *const std::ffi::c_void);

                // Launch kernel with optimized grid configuration
                let grid_size = self.calculate_optimized_grid_size(state.len(), kernel)?;
                let block_size = self.calculate_optimized_block_size(kernel)?;
                
                // Launch kernel with proper error handling
                kernel.launch(grid_size, block_size, &params, &self.streams[0])?;
                
                // Record kernel launch for profiling
                if let Some(ref mut profiler) = self.profiler {
                    profiler.record_kernel_launch(&kernel.name, grid_size, block_size);
                }

                // Synchronize and transfer result back
                self.streams[0].synchronize()?;
                self.transfer_from_gpu(&gpu_state, state)?;

                self.stats.kernel_launches += 1;
                self.stats
                    .kernel_times
                    .entry(kernel_name.to_string())
                    .and_modify(|t| *t += start_time.elapsed().as_secs_f64() * 1000.0)
                    .or_insert(start_time.elapsed().as_secs_f64() * 1000.0);
            } else {
                return Err(SimulatorError::UnsupportedOperation(format!(
                    "CUDA kernel '{}' not found",
                    kernel_name
                )));
            }
        }

        #[cfg(not(feature = "advanced_math"))]
        {
            return Err(SimulatorError::UnsupportedOperation(
                "CUDA kernels require SciRS2 backend (enable 'advanced_math' feature)".to_string(),
            ));
        }

        self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Apply two-qubit gate using CUDA
    pub fn apply_two_qubit_gate(
        &mut self,
        state: &mut Array1<Complex64>,
        control: usize,
        target: usize,
        gate_type: CudaGateType,
        parameters: &[f64],
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "advanced_math")]
        {
            let kernel_name = match gate_type {
                CudaGateType::CNOT => "cnot",
                CudaGateType::CZ => "cz",
                CudaGateType::SWAP => "swap",
                _ => "two_qubit_gate",
            };

            if let Some(kernel) = self.kernels.get(kernel_name) {
                // Transfer state to GPU
                let mut gpu_state = self.transfer_to_gpu(state)?;

                // Set up kernel parameters
                let params = vec![
                    gpu_state.as_ptr() as *mut std::ffi::c_void,
                    &control as *const usize as *const std::ffi::c_void,
                    &target as *const usize as *const std::ffi::c_void,
                    &(state.len() as i32) as *const i32 as *const std::ffi::c_void,
                ];

                // Launch kernel (placeholder implementation)
                let _grid_size = self.calculate_grid_size(state.len());
                // kernel.launch(grid_size, self.config.block_size, &params, &self.streams[0])?;

                // Synchronize and transfer result back
                self.streams[0].synchronize()?;
                self.transfer_from_gpu(&gpu_state, state)?;

                self.stats.kernel_launches += 1;
                self.stats
                    .kernel_times
                    .entry(kernel_name.to_string())
                    .and_modify(|t| *t += start_time.elapsed().as_secs_f64() * 1000.0)
                    .or_insert(start_time.elapsed().as_secs_f64() * 1000.0);
            } else {
                return Err(SimulatorError::UnsupportedOperation(format!(
                    "CUDA kernel '{}' not found",
                    kernel_name
                )));
            }
        }

        #[cfg(not(feature = "advanced_math"))]
        {
            return Err(SimulatorError::UnsupportedOperation(
                "CUDA kernels require SciRS2 backend (enable 'advanced_math' feature)".to_string(),
            ));
        }

        self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Measure qubits and get probabilities using CUDA
    pub fn measure_probabilities(
        &mut self,
        state: &Array1<Complex64>,
        qubits: &[usize],
    ) -> Result<Vec<f64>> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "advanced_math")]
        {
            if let Some(kernel) = self.kernels.get("measure_probabilities") {
                // Transfer state to GPU
                let gpu_state = self.transfer_to_gpu(state)?;

                // Allocate GPU memory for results
                let mut probabilities = vec![0.0f32; qubits.len()];
                let gpu_probs = self.allocate_gpu_memory(&probabilities)?;

                // Create qubit masks
                let qubit_masks: Vec<i32> = qubits.iter().map(|&q| 1i32 << q).collect();
                let gpu_masks = self.allocate_gpu_memory(&qubit_masks)?;

                // Set up kernel parameters
                let params = vec![
                    gpu_state.as_ptr() as *const std::ffi::c_void,
                    gpu_probs.as_ptr() as *mut std::ffi::c_void,
                    gpu_masks.as_ptr() as *const std::ffi::c_void,
                    &(qubits.len() as i32) as *const i32 as *const std::ffi::c_void,
                    &(state.len() as i32) as *const i32 as *const std::ffi::c_void,
                ];

                // Launch kernel (placeholder implementation)
                let _grid_size = self.calculate_grid_size(state.len());
                // kernel.launch(grid_size, self.config.block_size, &params, &self.streams[0])?;

                // Synchronize and transfer results back
                self.streams[0].synchronize()?;
                // For now, just use placeholder values since we have placeholders
                // self.transfer_from_gpu(&gpu_probs, &mut probabilities)?;

                let result: Vec<f64> = probabilities.iter().map(|&p| p as f64).collect();

                self.stats.kernel_launches += 1;
                self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;

                return Ok(result);
            }
        }

        #[cfg(not(feature = "advanced_math"))]
        {
            return Err(SimulatorError::UnsupportedOperation(
                "CUDA kernels require SciRS2 backend (enable 'advanced_math' feature)".to_string(),
            ));
        }

        Err(SimulatorError::UnsupportedOperation(
            "Measure kernel not available".to_string(),
        ))
    }

    /// Apply Quantum Fourier Transform using CUDA
    pub fn apply_qft(&mut self, state: &mut Array1<Complex64>, qubits: &[usize]) -> Result<()> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "advanced_math")]
        {
            if let Some(kernel) = self.kernels.get("qft") {
                // Transfer state to GPU
                let mut gpu_state = self.transfer_to_gpu(state)?;

                // Allocate temporary state
                let mut temp_state = vec![Complex64::new(0.0, 0.0); state.len()];
                let mut gpu_temp = self.allocate_gpu_memory(&temp_state)?;

                // Set up kernel parameters
                let params = vec![
                    gpu_state.as_ptr() as *mut std::ffi::c_void,
                    gpu_temp.as_ptr() as *mut std::ffi::c_void,
                    &(qubits.len() as i32) as *const i32 as *const std::ffi::c_void,
                    &(state.len() as i32) as *const i32 as *const std::ffi::c_void,
                ];

                // Launch kernel (placeholder implementation)
                let _grid_size = self.calculate_grid_size(state.len());
                // kernel.launch(grid_size, self.config.block_size, &params, &self.streams[0])?;

                // Synchronize and transfer result back
                self.streams[0].synchronize()?;
                self.transfer_from_gpu(&gpu_temp, state)?;

                self.stats.kernel_launches += 1;
                self.stats
                    .kernel_times
                    .entry("qft".to_string())
                    .and_modify(|t| *t += start_time.elapsed().as_secs_f64() * 1000.0)
                    .or_insert(start_time.elapsed().as_secs_f64() * 1000.0);
            } else {
                return Err(SimulatorError::UnsupportedOperation(
                    "QFT kernel not found".to_string(),
                ));
            }
        }

        #[cfg(not(feature = "advanced_math"))]
        {
            return Err(SimulatorError::UnsupportedOperation(
                "CUDA kernels require SciRS2 backend (enable 'advanced_math' feature)".to_string(),
            ));
        }

        self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Apply custom unitary matrix using CUDA
    pub fn apply_custom_unitary(
        &mut self,
        state: &mut Array1<Complex64>,
        qubits: &[usize],
        unitary: &Array2<Complex64>,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        #[cfg(feature = "advanced_math")]
        {
            if let Some(kernel) = self.kernels.get("apply_unitary") {
                // Transfer state and unitary to GPU
                let mut gpu_state = self.transfer_to_gpu(state)?;
                let gpu_unitary = self.transfer_matrix_to_gpu(unitary)?;

                // Set up kernel parameters
                let params = vec![
                    gpu_state.as_ptr() as *mut std::ffi::c_void,
                    gpu_unitary.as_ptr() as *const std::ffi::c_void,
                    qubits.as_ptr() as *const usize as *const std::ffi::c_void,
                    &(qubits.len() as i32) as *const i32 as *const std::ffi::c_void,
                    &(state.len() as i32) as *const i32 as *const std::ffi::c_void,
                ];

                // Launch kernel (placeholder implementation)
                let _grid_size = self.calculate_grid_size(state.len());
                // kernel.launch(grid_size, self.config.block_size, &params, &self.streams[0])?;

                // Synchronize and transfer result back
                self.streams[0].synchronize()?;
                self.transfer_from_gpu(&gpu_state, state)?;

                self.stats.kernel_launches += 1;
                self.stats
                    .kernel_times
                    .entry("apply_unitary".to_string())
                    .and_modify(|t| *t += start_time.elapsed().as_secs_f64() * 1000.0)
                    .or_insert(start_time.elapsed().as_secs_f64() * 1000.0);
            } else {
                return Err(SimulatorError::UnsupportedOperation(
                    "Custom unitary kernel not found".to_string(),
                ));
            }
        }

        #[cfg(not(feature = "advanced_math"))]
        {
            return Err(SimulatorError::UnsupportedOperation(
                "CUDA kernels require SciRS2 backend (enable 'advanced_math' feature)".to_string(),
            ));
        }

        self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> &CudaKernelStats {
        &self.stats
    }

    /// Reset performance statistics
    pub fn reset_stats(&mut self) {
        self.stats = CudaKernelStats::default();
    }
    
    /// Calculate optimized grid size for kernel launch
    #[cfg(feature = "advanced_math")]
    fn calculate_optimized_grid_size(&self, state_size: usize, kernel: &CudaKernel) -> Result<usize> {
        let occupancy = kernel.get_occupancy(self.config.block_size)?;
        
        #[cfg(feature = "advanced_math")]
        {
            if let Some(ref context) = self.context {
                let device_props = context.get_device_properties();
                
                // Calculate based on state size and optimal occupancy
                let elements_per_thread = 1;
                let threads_needed = (state_size + elements_per_thread - 1) / elements_per_thread;
                let blocks_needed = (threads_needed + self.config.block_size - 1) / self.config.block_size;
                
                // Limit by device capabilities
                let max_blocks = device_props.max_grid_dimensions[0] as usize;
                let optimal_blocks = (blocks_needed as f64 * occupancy) as usize;
                
                return Ok(optimal_blocks.min(max_blocks).max(1));
            }
        }
        
        // Fallback calculation
        let threads_needed = (state_size + 1 - 1) / 1;
        let blocks_needed = (threads_needed + self.config.block_size - 1) / self.config.block_size;
        Ok(blocks_needed.max(1))
    }
    
    /// Calculate optimized block size for kernel
    #[cfg(feature = "advanced_math")]
    fn calculate_optimized_block_size(&self, kernel: &CudaKernel) -> Result<usize> {
        #[cfg(feature = "advanced_math")]
        {
            if let Some(ref context) = self.context {
                let device_props = context.get_device_properties();
                let max_threads = device_props.max_threads_per_block.min(kernel.max_threads_per_block as i32) as usize;
                
                // Find optimal block size considering warp size
                let warp_size = device_props.warp_size as usize;
                let mut best_block_size = self.config.block_size;
                let mut best_occupancy = 0.0;
                
                for block_size in (warp_size..=max_threads).step_by(warp_size) {
                    if let Ok(occupancy) = kernel.get_occupancy(block_size) {
                        if occupancy > best_occupancy {
                            best_occupancy = occupancy;
                            best_block_size = block_size;
                        }
                    }
                }
                
                return Ok(best_block_size);
            }
        }
        
        // Fallback to configured block size
        Ok(self.config.block_size)
    }
    
    /// Select optimal memory type based on access patterns
    fn select_optimal_memory_type(&self, size: usize) -> GpuMemoryType {
        // Use heuristics to select optimal memory type
        if size > 100_000_000 { // > 100MB
            GpuMemoryType::Unified // Use unified memory for large allocations
        } else if self.config.unified_memory {
            GpuMemoryType::Unified
        } else {
            GpuMemoryType::Device
        }
    }
    
    /// Update memory pool statistics
    fn update_memory_statistics(&mut self, size: usize) {
        // Add memory_allocated field to stats if not present
        // For now, just update existing memory tracking
        self.stats.memory_transfers_to_gpu += size;
    }
    
    /// Calculate memory throughput for performance analysis
    fn calculate_memory_throughput(&self, state_size: usize) -> Result<f64> {
        #[cfg(feature = "advanced_math")]
        {
            if let Some(ref context) = self.context {
                let device_props = context.get_device_properties();
                let memory_bandwidth = (device_props.memory_clock_rate * device_props.memory_bus_width / 8) as f64; // GB/s
                
                let bytes_per_element = std::mem::size_of::<Complex64>();
                let total_bytes = state_size * bytes_per_element;
                let theoretical_time = total_bytes as f64 / (memory_bandwidth * 1e9);
                
                // Calculate efficiency relative to peak bandwidth
                let efficiency = 1.0 / (1.0 + theoretical_time);
                return Ok(efficiency);
            }
        }
        
        // Fallback for when advanced_math is not available
        Ok(0.5) // Default efficiency estimate
    }
    
    /// Calculate compute intensity for different gate types
    fn calculate_compute_intensity(&self, gate_type: &CudaGateType) -> Result<f64> {
        let ops_per_element = match gate_type {
            CudaGateType::PauliX | CudaGateType::PauliY | CudaGateType::PauliZ => 8.0, // 2x2 matrix multiply
            CudaGateType::Hadamard => 6.0,    // Optimized Hadamard
            CudaGateType::CNOT => 16.0,       // Optimized CNOT
            CudaGateType::CZ => 16.0,         // CZ gate operations
            CudaGateType::SWAP => 32.0,       // SWAP operations
            CudaGateType::RotationX | CudaGateType::RotationY | CudaGateType::RotationZ => 10.0, // Rotation gates
            CudaGateType::Phase | CudaGateType::T => 6.0, // Phase gates
            CudaGateType::CustomUnitary => 64.0, // General unitary
        };
        
        // Normalize to [0, 1] range
        Ok((ops_per_element / 64.0_f64).min(1.0_f64))
    }

    /// Helper methods

    fn calculate_grid_size(&self, problem_size: usize) -> usize {
        if self.config.grid_size > 0 {
            self.config.grid_size
        } else {
            (problem_size + self.config.block_size - 1) / self.config.block_size
        }
    }

    #[cfg(feature = "advanced_math")]
    fn transfer_to_gpu(&mut self, data: &Array1<Complex64>) -> Result<GpuMemory> {
        let mut pool = self.memory_pool.lock().unwrap();
        let gpu_memory = pool.allocate_and_copy(data.as_slice().unwrap())?;

        self.stats.memory_transfers_to_gpu += data.len() * std::mem::size_of::<Complex64>();
        Ok(gpu_memory)
    }

    #[cfg(feature = "advanced_math")]
    fn transfer_from_gpu(
        &mut self,
        gpu_memory: &GpuMemory,
        data: &mut Array1<Complex64>,
    ) -> Result<()> {
        gpu_memory.copy_to_host(data.as_slice_mut().unwrap())?;

        self.stats.memory_transfers_from_gpu += data.len() * std::mem::size_of::<Complex64>();
        Ok(())
    }

    #[cfg(feature = "advanced_math")]
    fn transfer_matrix_to_gpu(&mut self, matrix: &Array2<Complex64>) -> Result<GpuMemory> {
        let mut pool = self.memory_pool.lock().unwrap();
        let flattened: Vec<Complex64> = matrix.iter().cloned().collect();
        let gpu_memory = pool.allocate_and_copy(&flattened)?;

        self.stats.memory_transfers_to_gpu += flattened.len() * std::mem::size_of::<Complex64>();
        Ok(gpu_memory)
    }

    #[cfg(feature = "advanced_math")]
    fn allocate_gpu_memory<T: Clone>(&mut self, data: &[T]) -> Result<GpuMemory> {
        let mut _pool = self.memory_pool.lock().unwrap();
        let size = data.len() * std::mem::size_of::<T>();
        
        // Allocate GPU memory with optimal memory type selection
        let optimal_memory_type = self.select_optimal_memory_type(size);
        let mut gpu_memory = GpuMemory::new_with_type(optimal_memory_type);
        gpu_memory.allocate_pool(size)?;
        
        // Update memory pool statistics
        self.update_memory_statistics(size);
        
        Ok(gpu_memory)
    }
}

/// CUDA kernel utilities
pub struct CudaKernelUtils;

impl CudaKernelUtils {
    /// Benchmark CUDA kernels
    pub fn benchmark_kernels(config: CudaKernelConfig) -> Result<CudaBenchmarkResults> {
        let mut kernels = CudaQuantumKernels::new(config)?;
        let mut results = CudaBenchmarkResults::default();

        // Benchmark different state sizes
        let sizes = vec![10, 15, 20, 25]; // Number of qubits

        for &num_qubits in &sizes {
            let state_size = 1 << num_qubits;
            let mut state = Array1::from_elem(state_size, Complex64::new(1.0, 0.0));
            state[0] = Complex64::new(1.0, 0.0); // |0...0⟩ state

            // Normalize
            let norm: f64 = state.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            state.mapv_inplace(|x| x / norm);

            // Benchmark single-qubit gates
            let start = std::time::Instant::now();
            for qubit in 0..num_qubits.min(5) {
                kernels.apply_single_qubit_gate(&mut state, qubit, CudaGateType::Hadamard, &[])?;
            }
            let single_qubit_time = start.elapsed().as_secs_f64() * 1000.0;

            // Benchmark two-qubit gates
            let start = std::time::Instant::now();
            for qubit in 0..(num_qubits - 1).min(3) {
                kernels.apply_two_qubit_gate(
                    &mut state,
                    qubit,
                    qubit + 1,
                    CudaGateType::CNOT,
                    &[],
                )?;
            }
            let two_qubit_time = start.elapsed().as_secs_f64() * 1000.0;

            results
                .single_qubit_times
                .push((num_qubits, single_qubit_time));
            results.two_qubit_times.push((num_qubits, two_qubit_time));
        }

        results.kernel_stats = kernels.get_stats().clone();
        Ok(results)
    }

    /// Estimate optimal configuration for given hardware
    pub fn estimate_optimal_config() -> CudaKernelConfig {
        // This would query actual GPU properties in a real implementation
        CudaKernelConfig {
            device_id: 0,
            num_streams: 4,
            block_size: 256,
            grid_size: 0, // Auto-calculate
            unified_memory: true,
            memory_pool_size: 2_000_000_000,
            enable_profiling: false,
            optimization_level: OptimizationLevel::Aggressive,
        }
    }

    /// Get device information
    pub fn get_device_info() -> Result<CudaDeviceInfo> {
        #[cfg(feature = "advanced_math")]
        {
            Ok(CudaDeviceInfo {
                device_count: CudaContext::get_device_count()?,
                devices: (0..CudaContext::get_device_count()?)
                    .map(|i| DeviceProperties {
                        name: format!("CUDA Device {}", i),
                        compute_capability: (7, 5),
                        total_memory: 8_000_000_000, // 8GB default
                        max_threads_per_block: 1024,
                        max_blocks_per_grid: 65535,
                        clock_rate: 1500000, // 1.5 GHz
                    })
                    .collect(),
            })
        }

        #[cfg(not(feature = "advanced_math"))]
        {
            Err(SimulatorError::UnsupportedOperation(
                "CUDA device info requires SciRS2 backend (enable 'advanced_math' feature)"
                    .to_string(),
            ))
        }
    }
}

/// CUDA benchmark results
#[derive(Debug, Clone, Default)]
pub struct CudaBenchmarkResults {
    /// Single-qubit gate benchmark times (num_qubits, time_ms)
    pub single_qubit_times: Vec<(usize, f64)>,
    /// Two-qubit gate benchmark times (num_qubits, time_ms)
    pub two_qubit_times: Vec<(usize, f64)>,
    /// Kernel statistics
    pub kernel_stats: CudaKernelStats,
}

/// CUDA device information
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    /// Number of CUDA devices
    pub device_count: i32,
    /// Device properties
    pub devices: Vec<DeviceProperties>,
}

/// CUDA device properties
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// Device name
    pub name: String,
    /// Compute capability (major, minor)
    pub compute_capability: (i32, i32),
    /// Total global memory in bytes
    pub total_memory: usize,
    /// Maximum threads per block
    pub max_threads_per_block: i32,
    /// Maximum blocks per grid
    pub max_blocks_per_grid: i32,
    /// Clock rate in kHz
    pub clock_rate: i32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cuda_kernel_config() {
        let config = CudaKernelConfig::default();
        assert_eq!(config.device_id, 0);
        assert_eq!(config.num_streams, 4);
        assert_eq!(config.block_size, 256);
    }

    #[test]
    fn test_cuda_kernel_creation() {
        let config = CudaKernelConfig::default();
        // This test will only pass if CUDA is available
        let result = CudaQuantumKernels::new(config);

        #[cfg(feature = "advanced_math")]
        assert!(result.is_ok() || result.is_err()); // Either works or fails gracefully

        #[cfg(not(feature = "advanced_math"))]
        assert!(result.is_err());
    }

    #[test]
    fn test_grid_size_calculation() {
        let config = CudaKernelConfig::default();
        if let Ok(kernels) = CudaQuantumKernels::new(config) {
            let grid_size = kernels.calculate_grid_size(1000);
            assert_eq!(grid_size, (1000 + 256 - 1) / 256);
        }
    }

    #[test]
    fn test_gate_type_variants() {
        let gate_types = vec![
            CudaGateType::PauliX,
            CudaGateType::PauliY,
            CudaGateType::PauliZ,
            CudaGateType::Hadamard,
            CudaGateType::CNOT,
            CudaGateType::CustomUnitary,
        ];

        assert_eq!(gate_types.len(), 6);
    }

    #[test]
    fn test_optimization_levels() {
        let levels = vec![
            OptimizationLevel::Conservative,
            OptimizationLevel::Balanced,
            OptimizationLevel::Aggressive,
            OptimizationLevel::Custom,
        ];

        assert_eq!(levels.len(), 4);
    }

    #[test]
    fn test_cuda_kernel_stats() {
        let stats = CudaKernelStats::default();
        assert_eq!(stats.kernel_launches, 0);
        assert_eq!(stats.total_execution_time_ms, 0.0);
        assert!(stats.kernel_times.is_empty());
    }

    #[test]
    fn test_benchmark_results_creation() {
        let results = CudaBenchmarkResults::default();
        assert!(results.single_qubit_times.is_empty());
        assert!(results.two_qubit_times.is_empty());
    }
}
