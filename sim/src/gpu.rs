//! GPU-accelerated quantum simulation module
//!
//! This module provides GPU-accelerated implementations of quantum simulators
//! using WGPU (WebGPU). This implementation is optimized for simulating
//! quantum circuits on GPUs, which significantly speeds up simulations
//! for large qubit counts.

use bytemuck::{Pod, Zeroable};
use num_complex::Complex64;
use quantrs_circuit::prelude::{Circuit, GateType};
use quantrs_core::prelude::QubitId;
use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::simulator::{Simulator, SimulatorResult};

/// The alignment used for buffers
const BUFFER_ALIGNMENT: u64 = 256;

/// GPU-accelerated state vector simulator
#[derive(Debug)]
pub struct GpuStateVectorSimulator {
    /// The WGPU device
    device: Arc<wgpu::Device>,
    /// The WGPU queue
    queue: Arc<wgpu::Queue>,
    /// The compute pipeline for applying single-qubit gates
    single_qubit_pipeline: wgpu::ComputePipeline,
    /// The compute pipeline for applying two-qubit gates
    two_qubit_pipeline: wgpu::ComputePipeline,
}

/// Complex number for GPU computation
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuComplex {
    /// Real part
    real: f32,
    /// Imaginary part
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
    fn from(c: GpuComplex) -> Self {
        Complex64::new(c.real as f64, c.imag as f64)
    }
}

/// Uniform buffer for single-qubit gate operations
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct SingleQubitGateParams {
    /// Target qubit index
    target_qubit: u32,
    /// Number of qubits
    n_qubits: u32,
    /// Matrix elements (row-major order)
    matrix: [GpuComplex; 4],
}

/// Uniform buffer for two-qubit gate operations
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct TwoQubitGateParams {
    /// Control qubit index
    control_qubit: u32,
    /// Target qubit index
    target_qubit: u32,
    /// Number of qubits
    n_qubits: u32,
    /// Matrix elements (row-major order)
    matrix: [GpuComplex; 16],
}

impl GpuStateVectorSimulator {
    /// Create a new GPU-accelerated state vector simulator
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Create WGPU instance
        let instance = wgpu::Instance::default();

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or("Failed to find GPU adapter")?;

        // Create device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Quantrs GPU Simulator"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Shader for single-qubit gates
        let single_qubit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Single Qubit Gate Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "shaders/single_qubit_gate.wgsl"
            ))),
        });

        // Shader for two-qubit gates
        let two_qubit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Two Qubit Gate Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "shaders/two_qubit_gate.wgsl"
            ))),
        });

        // Create compute pipeline layouts
        let single_qubit_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Single Qubit Pipeline Layout"),
                bind_group_layouts: &[&device.create_bind_group_layout(
                    &wgpu::BindGroupLayoutDescriptor {
                        label: Some("Single Qubit Bind Group Layout"),
                        entries: &[
                            // State vector
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // Gate parameters
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    },
                )],
                push_constant_ranges: &[],
            });

        let two_qubit_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Two Qubit Pipeline Layout"),
                bind_group_layouts: &[&device.create_bind_group_layout(
                    &wgpu::BindGroupLayoutDescriptor {
                        label: Some("Two Qubit Bind Group Layout"),
                        entries: &[
                            // State vector
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // Gate parameters
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    },
                )],
                push_constant_ranges: &[],
            });

        // Create compute pipelines
        let single_qubit_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Single Qubit Pipeline"),
                layout: Some(&single_qubit_pipeline_layout),
                module: &single_qubit_shader,
                entry_point: "main",
            });

        let two_qubit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Two Qubit Pipeline"),
            layout: Some(&two_qubit_pipeline_layout),
            module: &two_qubit_shader,
            entry_point: "main",
        });

        Ok(Self {
            device,
            queue,
            single_qubit_pipeline,
            two_qubit_pipeline,
        })
    }

    /// Create a new GPU-accelerated state vector simulator synchronously
    pub fn new_blocking() -> Result<Self, Box<dyn std::error::Error>> {
        // Create a runtime for async operations
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        // Run the async initialization in the runtime
        rt.block_on(Self::new())
    }

    /// Check if GPU acceleration is available on this system
    pub fn is_available() -> bool {
        // Try to create the simulator
        match std::panic::catch_unwind(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            rt.block_on(async {
                let instance = wgpu::Instance::default();
                instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        force_fallback_adapter: false,
                        compatible_surface: None,
                    })
                    .await
                    .is_some()
            })
        }) {
            Ok(result) => result,
            Err(_) => false,
        }
    }
}

impl Simulator for GpuStateVectorSimulator {
    fn run<const N: usize>(&self, circuit: &Circuit<N>) -> crate::simulator::SimulatorResult<N> {
        use quantrs_core::gate::GateMatrix;

        // Skip GPU simulation for small circuits (less than 4 qubits)
        // CPU is often faster for these small circuits due to overhead
        if N < 4 {
            let cpu_sim = crate::statevector::StateVectorSimulator::new();
            return cpu_sim.run(circuit);
        }

        // Calculate state vector size
        let state_size = 1 << N;
        let buffer_size = (state_size * std::mem::size_of::<GpuComplex>()) as u64;

        // Create initial state |0...0⟩
        let mut initial_state = vec![
            GpuComplex {
                real: 0.0,
                imag: 0.0
            };
            state_size
        ];
        initial_state[0].real = 1.0; // Set |0...0⟩ amplitude to 1

        // Create GPU buffer for state vector
        let state_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("State Vector Buffer"),
                contents: bytemuck::cast_slice(&initial_state),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        // Create a buffer to read back the results from the GPU
        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Process each gate in the circuit
        for gate in circuit.gates() {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Gate Execution Encoder"),
                });

            match gate.gate_type {
                GateType::SingleQubit { target, matrix } => {
                    // Convert matrix to GPU format
                    let gpu_matrix = [
                        GpuComplex::from(matrix[(0, 0)]),
                        GpuComplex::from(matrix[(0, 1)]),
                        GpuComplex::from(matrix[(1, 0)]),
                        GpuComplex::from(matrix[(1, 1)]),
                    ];

                    // Prepare gate parameters
                    let params = SingleQubitGateParams {
                        target_qubit: target.id() as u32,
                        n_qubits: N as u32,
                        matrix: gpu_matrix,
                    };

                    // Create parameter buffer
                    let param_buffer =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Single Qubit Gate Params Buffer"),
                                contents: bytemuck::bytes_of(&params),
                                usage: wgpu::BufferUsages::UNIFORM,
                            });

                    // Create bind group
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Single Qubit Gate Bind Group"),
                        layout: &self.single_qubit_pipeline.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: state_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: param_buffer.as_entire_binding(),
                            },
                        ],
                    });

                    // Compute workgroup count (1 per 256 state vector elements)
                    let workgroup_count = ((state_size + 255) / 256) as u32;

                    // Dispatch compute shader
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Single Qubit Gate Compute Pass"),
                            timestamp_writes: None,
                        });
                    compute_pass.set_pipeline(&self.single_qubit_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                    drop(compute_pass);
                }
                GateType::TwoQubit {
                    control,
                    target,
                    matrix,
                } => {
                    // Convert matrix to GPU format (assuming a 4x4 matrix)
                    let mut gpu_matrix = [GpuComplex {
                        real: 0.0,
                        imag: 0.0,
                    }; 16];
                    for i in 0..4 {
                        for j in 0..4 {
                            gpu_matrix[i * 4 + j] = GpuComplex::from(matrix[(i, j)]);
                        }
                    }

                    // Prepare gate parameters
                    let params = TwoQubitGateParams {
                        control_qubit: control.id() as u32,
                        target_qubit: target.id() as u32,
                        n_qubits: N as u32,
                        matrix: gpu_matrix,
                    };

                    // Create parameter buffer
                    let param_buffer =
                        self.device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("Two Qubit Gate Params Buffer"),
                                contents: bytemuck::bytes_of(&params),
                                usage: wgpu::BufferUsages::UNIFORM,
                            });

                    // Create bind group
                    let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Two Qubit Gate Bind Group"),
                        layout: &self.two_qubit_pipeline.get_bind_group_layout(0),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: state_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: param_buffer.as_entire_binding(),
                            },
                        ],
                    });

                    // Compute workgroup count (1 per 256 state vector elements)
                    let workgroup_count = ((state_size + 255) / 256) as u32;

                    // Dispatch compute shader
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Two Qubit Gate Compute Pass"),
                            timestamp_writes: None,
                        });
                    compute_pass.set_pipeline(&self.two_qubit_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
                    drop(compute_pass);
                }
            }

            // Submit command encoder to GPU
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        // After all gates, copy the state vector back from the GPU
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Result Copy Encoder"),
            });

        encoder.copy_buffer_to_buffer(&state_buffer, 0, &result_buffer, 0, buffer_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map the buffer to read the results
        let buffer_slice = result_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        // Wait for the buffer to be mapped
        self.device.poll(wgpu::Maintain::Wait);
        if rx.recv().unwrap().is_err() {
            panic!("Failed to map buffer for reading");
        }

        // Read the data
        let data = buffer_slice.get_mapped_range();
        let result_data: Vec<GpuComplex> = bytemuck::cast_slice(&data).to_vec();
        drop(data); // Unmap the buffer

        // Convert GPU results to complex amplitudes
        let amplitudes: Vec<Complex64> = result_data.into_iter().map(|c| c.into()).collect();

        // Return simulation result
        crate::simulator::SimulatorResult {
            amplitudes,
            num_qubits: N,
        }
    }
}

// This module now uses the WGSL shaders in the "shaders" directory:
// - shaders/single_qubit_gate.wgsl: Handles single-qubit gate operations
// - shaders/two_qubit_gate.wgsl: Handles two-qubit gate operations
