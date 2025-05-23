//! GPU-accelerated linear algebra operations for quantum simulation
//! 
//! This module provides GPU-accelerated implementations of common linear algebra
//! operations used in quantum simulation, leveraging both WGPU and SciRS2.

use crate::linalg_ops;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num_complex::Complex64;
use quantrs2_core::prelude::*;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

/// GPU buffer for complex numbers
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuComplex {
    real: f32,
    imag: f32,
}

impl From<Complex64> for GpuComplex {
    fn from(c: Complex64) -> Self {
        Self {
            real: c.re as f32,
            imag: c.im as f32,
        }
    }
}

impl From<GpuComplex> for Complex64 {
    fn from(gc: GpuComplex) -> Self {
        Complex64::new(gc.real as f64, gc.imag as f64)
    }
}

/// GPU-accelerated linear algebra operations
pub struct GpuLinearAlgebra {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    matmul_pipeline: wgpu::ComputePipeline,
    tensor_product_pipeline: wgpu::ComputePipeline,
    apply_unitary_pipeline: wgpu::ComputePipeline,
}

impl GpuLinearAlgebra {
    /// Create a new GPU linear algebra instance
    pub async fn new() -> Result<Self, QuantRS2Error> {
        // Initialize WGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| QuantRS2Error::BackendExecutionFailed("No GPU adapter found".to_string()))?;
        
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Quantum GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| QuantRS2Error::BackendExecutionFailed(e.to_string()))?;
        
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        
        // Create compute pipelines
        let matmul_pipeline = Self::create_matmul_pipeline(&device);
        let tensor_product_pipeline = Self::create_tensor_product_pipeline(&device);
        let apply_unitary_pipeline = Self::create_apply_unitary_pipeline(&device);
        
        Ok(Self {
            device,
            queue,
            matmul_pipeline,
            tensor_product_pipeline,
            apply_unitary_pipeline,
        })
    }
    
    /// Matrix multiplication on GPU
    pub async fn matmul(
        &self,
        a: &Array2<Complex64>,
        b: &Array2<Complex64>,
    ) -> Result<Array2<Complex64>, QuantRS2Error> {
        let (m, k1) = a.dim();
        let (k2, n) = b.dim();
        
        if k1 != k2 {
            return Err(QuantRS2Error::InvalidInput(
                format!("Matrix dimensions don't match for multiplication: ({}, {}) x ({}, {})", m, k1, k2, n)
            ));
        }
        
        // Convert to GPU format
        let a_gpu: Vec<GpuComplex> = a.iter().map(|&c| c.into()).collect();
        let b_gpu: Vec<GpuComplex> = b.iter().map(|&c| c.into()).collect();
        
        // Create GPU buffers
        let a_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix A"),
            contents: bytemuck::cast_slice(&a_gpu),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let b_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matrix B"),
            contents: bytemuck::cast_slice(&b_gpu),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let result_size = (m * n * std::mem::size_of::<GpuComplex>()) as u64;
        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Matrix"),
            size: result_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Create uniform buffer for dimensions
        let dimensions = [m as u32, n as u32, k1 as u32, 0u32];
        let dimensions_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dimensions"),
            contents: bytemuck::cast_slice(&dimensions),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MatMul Bind Group"),
            layout: &self.matmul_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dimensions_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Execute compute shader
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MatMul Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MatMul Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.matmul_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch with appropriate workgroup size
            let workgroup_size = 16;
            let workgroups_x = (n + workgroup_size - 1) / workgroup_size;
            let workgroups_y = (m + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
        }
        
        // Copy result to staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: result_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, result_size);
        
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        self.device.poll(wgpu::MaintainBase::Wait);
        rx.await.unwrap().map_err(|e| QuantRS2Error::BackendExecutionFailed(e.to_string()))?;
        
        let data = buffer_slice.get_mapped_range();
        let gpu_result: &[GpuComplex] = bytemuck::cast_slice(&data);
        
        // Convert back to Complex64
        let mut result = Array2::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                result[[i, j]] = gpu_result[i * n + j].into();
            }
        }
        
        drop(data);
        staging_buffer.unmap();
        
        Ok(result)
    }
    
    /// Tensor product on GPU
    pub async fn tensor_product(
        &self,
        a: &Array2<Complex64>,
        b: &Array2<Complex64>,
    ) -> Result<Array2<Complex64>, QuantRS2Error> {
        let (m1, n1) = a.dim();
        let (m2, n2) = b.dim();
        
        // For now, fall back to CPU implementation
        // TODO: Implement GPU tensor product
        Ok(linalg_ops::tensor_product(a, b))
    }
    
    /// Apply unitary matrix to state vector on GPU
    pub async fn apply_unitary(
        &self,
        state: &mut Vec<Complex64>,
        unitary: &Array2<Complex64>,
        target_qubits: &[usize],
    ) -> Result<(), QuantRS2Error> {
        // For now, fall back to CPU implementation
        // TODO: Implement GPU unitary application
        linalg_ops::apply_unitary(state, unitary, target_qubits)
    }
    
    /// Create matrix multiplication compute pipeline
    fn create_matmul_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MatMul Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/matmul.wgsl"))),
        });
        
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MatMul Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("matmul"),
            cache: None,
        })
    }
    
    /// Create tensor product compute pipeline
    fn create_tensor_product_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tensor Product Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/tensor_product.wgsl"))),
        });
        
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tensor Product Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("tensor_product"),
            cache: None,
        })
    }
    
    /// Create apply unitary compute pipeline
    fn create_apply_unitary_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Apply Unitary Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shaders/apply_unitary.wgsl"))),
        });
        
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Apply Unitary Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("apply_unitary"),
            cache: None,
        })
    }
}

/// Benchmark GPU vs CPU linear algebra operations
pub async fn benchmark_gpu_linalg() -> Result<(), QuantRS2Error> {
    use std::time::Instant;
    
    println!("=== GPU Linear Algebra Benchmark ===\n");
    
    let gpu_linalg = GpuLinearAlgebra::new().await?;
    
    // Test different matrix sizes
    for size in [4, 8, 16, 32, 64, 128] {
        println!("Matrix size: {}x{}", size, size);
        
        // Create random matrices
        let a = Array2::from_shape_fn((size, size), |_| {
            Complex64::new(fastrand::f64() - 0.5, fastrand::f64() - 0.5)
        });
        let b = Array2::from_shape_fn((size, size), |_| {
            Complex64::new(fastrand::f64() - 0.5, fastrand::f64() - 0.5)
        });
        
        // CPU benchmark
        let cpu_start = Instant::now();
        let _cpu_result = a.dot(&b);
        let cpu_time = cpu_start.elapsed();
        
        // GPU benchmark
        let gpu_start = Instant::now();
        let _gpu_result = gpu_linalg.matmul(&a, &b).await?;
        let gpu_time = gpu_start.elapsed();
        
        println!("  CPU time: {:?}", cpu_time);
        println!("  GPU time: {:?}", gpu_time);
        println!("  Speedup: {:.2}x", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
        println!();
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_gpu_matmul() {
        let gpu_linalg = match GpuLinearAlgebra::new().await {
            Ok(gpu) => gpu,
            Err(_) => {
                println!("GPU not available, skipping test");
                return;
            }
        };
        
        // Simple 2x2 matrix multiplication
        let a = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0),
        ]).unwrap();
        
        let b = Array2::from_shape_vec((2, 2), vec![
            Complex64::new(5.0, 0.0), Complex64::new(6.0, 0.0),
            Complex64::new(7.0, 0.0), Complex64::new(8.0, 0.0),
        ]).unwrap();
        
        let result = gpu_linalg.matmul(&a, &b).await.unwrap();
        
        // Expected: [[19, 22], [43, 50]]
        assert!((result[[0, 0]].re - 19.0).abs() < 1e-6);
        assert!((result[[0, 1]].re - 22.0).abs() < 1e-6);
        assert!((result[[1, 0]].re - 43.0).abs() < 1e-6);
        assert!((result[[1, 1]].re - 50.0).abs() < 1e-6);
    }
}