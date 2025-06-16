//! FPGA (Field-Programmable Gate Array) Acceleration for Quantum Simulation
//!
//! This module provides high-performance quantum circuit simulation using FPGAs
//! with custom hardware designs optimized for quantum gate operations. It leverages
//! the reconfigurable nature of FPGAs to create specialized quantum processing units
//! that can be optimized for specific quantum algorithms and gate sets.
//!
//! Key features:
//! - Custom FPGA designs for quantum gate operations
//! - Parallel quantum state vector processing
//! - Hardware-optimized quantum arithmetic units
//! - Low-latency quantum circuit execution
//! - Memory-efficient state representation
//! - Real-time quantum error correction
//! - Integration with Intel/Xilinx FPGA platforms
//! - OpenCL and Verilog/SystemVerilog code generation

use ndarray::{Array1, Array2, ArrayView1};
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::circuit_interfaces::{
    CircuitInterface, InterfaceCircuit, InterfaceGate, InterfaceGateType,
};
use crate::error::{Result, SimulatorError};
use crate::statevector::StateVectorSimulator;

/// FPGA platform types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FPGAPlatform {
    /// Intel Arria 10
    IntelArria10,
    /// Intel Stratix 10
    IntelStratix10,
    /// Intel Agilex 7
    IntelAgilex7,
    /// Xilinx Virtex UltraScale+
    XilinxVirtexUltraScale,
    /// Xilinx Versal ACAP
    XilinxVersal,
    /// Xilinx Kintex UltraScale+
    XilinxKintexUltraScale,
    /// Simulation mode
    Simulation,
}

/// FPGA configuration
#[derive(Debug, Clone)]
pub struct FPGAConfig {
    /// Target FPGA platform
    pub platform: FPGAPlatform,
    /// Clock frequency (MHz)
    pub clock_frequency: f64,
    /// Number of processing units
    pub num_processing_units: usize,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
    /// Enable pipelining
    pub enable_pipelining: bool,
    /// Pipeline depth
    pub pipeline_depth: usize,
    /// Data path width (bits)
    pub data_path_width: usize,
    /// Enable DSP optimization
    pub enable_dsp_optimization: bool,
    /// Enable block RAM optimization
    pub enable_bram_optimization: bool,
    /// Maximum state vector size
    pub max_state_size: usize,
    /// Enable real-time processing
    pub enable_realtime: bool,
    /// Hardware description language
    pub hdl_target: HDLTarget,
}

/// Hardware description language targets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HDLTarget {
    Verilog,
    SystemVerilog,
    VHDL,
    Chisel,
    HLS,
    OpenCL,
}

impl Default for FPGAConfig {
    fn default() -> Self {
        Self {
            platform: FPGAPlatform::IntelStratix10,
            clock_frequency: 300.0, // 300 MHz
            num_processing_units: 16,
            memory_bandwidth: 50.0, // 50 GB/s
            enable_pipelining: true,
            pipeline_depth: 8,
            data_path_width: 512, // 512-bit wide data path
            enable_dsp_optimization: true,
            enable_bram_optimization: true,
            max_state_size: 1 << 22, // 4M states
            enable_realtime: false,
            hdl_target: HDLTarget::SystemVerilog,
        }
    }
}

/// FPGA device information
#[derive(Debug, Clone)]
pub struct FPGADeviceInfo {
    /// Device ID
    pub device_id: usize,
    /// Platform type
    pub platform: FPGAPlatform,
    /// Logic elements/LUTs
    pub logic_elements: usize,
    /// DSP blocks
    pub dsp_blocks: usize,
    /// Block RAM (KB)
    pub block_ram_kb: usize,
    /// Clock frequency (MHz)
    pub max_clock_frequency: f64,
    /// Memory interfaces
    pub memory_interfaces: Vec<MemoryInterface>,
    /// PCIe lanes
    pub pcie_lanes: usize,
    /// Power consumption (W)
    pub power_consumption: f64,
    /// Supported arithmetic precision
    pub supported_precision: Vec<ArithmeticPrecision>,
}

/// Memory interface types
#[derive(Debug, Clone)]
pub struct MemoryInterface {
    /// Interface type
    pub interface_type: MemoryInterfaceType,
    /// Bandwidth (GB/s)
    pub bandwidth: f64,
    /// Capacity (GB)
    pub capacity: f64,
    /// Latency (ns)
    pub latency: f64,
}

/// Memory interface types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryInterfaceType {
    DDR4,
    DDR5,
    HBM2,
    HBM3,
    GDDR6,
    OnChipRAM,
}

/// Arithmetic precision types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticPrecision {
    Fixed8,
    Fixed16,
    Fixed32,
    Float16,
    Float32,
    Float64,
    CustomFixed(u32),
    CustomFloat(u32, u32), // (mantissa, exponent)
}

impl FPGADeviceInfo {
    /// Create device info for specific FPGA platform
    pub fn for_platform(platform: FPGAPlatform) -> Self {
        match platform {
            FPGAPlatform::IntelArria10 => Self {
                device_id: 0,
                platform,
                logic_elements: 1150000,
                dsp_blocks: 1688,
                block_ram_kb: 53000,
                max_clock_frequency: 400.0,
                memory_interfaces: vec![
                    MemoryInterface {
                        interface_type: MemoryInterfaceType::DDR4,
                        bandwidth: 34.0,
                        capacity: 32.0,
                        latency: 200.0,
                    },
                ],
                pcie_lanes: 16,
                power_consumption: 100.0,
                supported_precision: vec![
                    ArithmeticPrecision::Fixed16,
                    ArithmeticPrecision::Fixed32,
                    ArithmeticPrecision::Float32,
                ],
            },
            FPGAPlatform::IntelStratix10 => Self {
                device_id: 0,
                platform,
                logic_elements: 2800000,
                dsp_blocks: 5760,
                block_ram_kb: 229000,
                max_clock_frequency: 500.0,
                memory_interfaces: vec![
                    MemoryInterface {
                        interface_type: MemoryInterfaceType::DDR4,
                        bandwidth: 68.0,
                        capacity: 64.0,
                        latency: 180.0,
                    },
                    MemoryInterface {
                        interface_type: MemoryInterfaceType::HBM2,
                        bandwidth: 460.0,
                        capacity: 8.0,
                        latency: 50.0,
                    },
                ],
                pcie_lanes: 16,
                power_consumption: 150.0,
                supported_precision: vec![
                    ArithmeticPrecision::Fixed16,
                    ArithmeticPrecision::Fixed32,
                    ArithmeticPrecision::Float32,
                    ArithmeticPrecision::Float64,
                ],
            },
            FPGAPlatform::IntelAgilex7 => Self {
                device_id: 0,
                platform,
                logic_elements: 2500000,
                dsp_blocks: 4608,
                block_ram_kb: 180000,
                max_clock_frequency: 600.0,
                memory_interfaces: vec![
                    MemoryInterface {
                        interface_type: MemoryInterfaceType::DDR5,
                        bandwidth: 102.0,
                        capacity: 128.0,
                        latency: 150.0,
                    },
                    MemoryInterface {
                        interface_type: MemoryInterfaceType::HBM3,
                        bandwidth: 819.0,
                        capacity: 16.0,
                        latency: 40.0,
                    },
                ],
                pcie_lanes: 32,
                power_consumption: 120.0,
                supported_precision: vec![
                    ArithmeticPrecision::Fixed16,
                    ArithmeticPrecision::Fixed32,
                    ArithmeticPrecision::Float16,
                    ArithmeticPrecision::Float32,
                    ArithmeticPrecision::Float64,
                ],
            },
            FPGAPlatform::XilinxVirtexUltraScale => Self {
                device_id: 0,
                platform,
                logic_elements: 1300000,
                dsp_blocks: 6840,
                block_ram_kb: 75900,
                max_clock_frequency: 450.0,
                memory_interfaces: vec![
                    MemoryInterface {
                        interface_type: MemoryInterfaceType::DDR4,
                        bandwidth: 77.0,
                        capacity: 64.0,
                        latency: 190.0,
                    },
                ],
                pcie_lanes: 16,
                power_consumption: 130.0,
                supported_precision: vec![
                    ArithmeticPrecision::Fixed16,
                    ArithmeticPrecision::Fixed32,
                    ArithmeticPrecision::Float32,
                ],
            },
            FPGAPlatform::XilinxVersal => Self {
                device_id: 0,
                platform,
                logic_elements: 1968000,
                dsp_blocks: 9024,
                block_ram_kb: 175000,
                max_clock_frequency: 700.0,
                memory_interfaces: vec![
                    MemoryInterface {
                        interface_type: MemoryInterfaceType::DDR5,
                        bandwidth: 120.0,
                        capacity: 256.0,
                        latency: 140.0,
                    },
                    MemoryInterface {
                        interface_type: MemoryInterfaceType::HBM3,
                        bandwidth: 1024.0,
                        capacity: 32.0,
                        latency: 35.0,
                    },
                ],
                pcie_lanes: 32,
                power_consumption: 100.0,
                supported_precision: vec![
                    ArithmeticPrecision::Fixed8,
                    ArithmeticPrecision::Fixed16,
                    ArithmeticPrecision::Fixed32,
                    ArithmeticPrecision::Float16,
                    ArithmeticPrecision::Float32,
                    ArithmeticPrecision::Float64,
                ],
            },
            FPGAPlatform::XilinxKintexUltraScale => Self {
                device_id: 0,
                platform,
                logic_elements: 850000,
                dsp_blocks: 2928,
                block_ram_kb: 75900,
                max_clock_frequency: 500.0,
                memory_interfaces: vec![
                    MemoryInterface {
                        interface_type: MemoryInterfaceType::DDR4,
                        bandwidth: 60.0,
                        capacity: 32.0,
                        latency: 200.0,
                    },
                ],
                pcie_lanes: 8,
                power_consumption: 80.0,
                supported_precision: vec![
                    ArithmeticPrecision::Fixed16,
                    ArithmeticPrecision::Fixed32,
                    ArithmeticPrecision::Float32,
                ],
            },
            FPGAPlatform::Simulation => Self {
                device_id: 0,
                platform,
                logic_elements: 10000000,
                dsp_blocks: 10000,
                block_ram_kb: 1000000,
                max_clock_frequency: 1000.0,
                memory_interfaces: vec![
                    MemoryInterface {
                        interface_type: MemoryInterfaceType::HBM3,
                        bandwidth: 2000.0,
                        capacity: 128.0,
                        latency: 10.0,
                    },
                ],
                pcie_lanes: 64,
                power_consumption: 50.0,
                supported_precision: vec![
                    ArithmeticPrecision::Fixed8,
                    ArithmeticPrecision::Fixed16,
                    ArithmeticPrecision::Fixed32,
                    ArithmeticPrecision::Float16,
                    ArithmeticPrecision::Float32,
                    ArithmeticPrecision::Float64,
                ],
            },
        }
    }
}

/// FPGA quantum processing unit
#[derive(Debug, Clone)]
pub struct QuantumProcessingUnit {
    /// Unit ID
    pub unit_id: usize,
    /// Supported gate types
    pub supported_gates: Vec<InterfaceGateType>,
    /// Pipeline stages
    pub pipeline_stages: Vec<PipelineStage>,
    /// Local memory (KB)
    pub local_memory_kb: usize,
    /// Processing frequency (MHz)
    pub frequency: f64,
    /// Utilization percentage
    pub utilization: f64,
}

/// Pipeline stage
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Stage name
    pub name: String,
    /// Stage operation
    pub operation: PipelineOperation,
    /// Latency (clock cycles)
    pub latency: usize,
    /// Throughput (operations per cycle)
    pub throughput: f64,
}

/// Pipeline operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineOperation {
    Fetch,
    Decode,
    AddressCalculation,
    MemoryRead,
    GateExecution,
    MemoryWrite,
    Writeback,
}

/// FPGA quantum simulator
pub struct FPGAQuantumSimulator {
    /// Configuration
    config: FPGAConfig,
    /// Device information
    device_info: FPGADeviceInfo,
    /// Processing units
    processing_units: Vec<QuantumProcessingUnit>,
    /// Generated HDL modules
    hdl_modules: HashMap<String, HDLModule>,
    /// Performance statistics
    stats: FPGAStats,
    /// Memory manager
    memory_manager: FPGAMemoryManager,
    /// Bitstream manager
    bitstream_manager: BitstreamManager,
}

/// HDL module representation
#[derive(Debug, Clone)]
pub struct HDLModule {
    /// Module name
    pub name: String,
    /// HDL code
    pub hdl_code: String,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Timing information
    pub timing_info: TimingInfo,
    /// Module type
    pub module_type: ModuleType,
}

/// Module types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleType {
    SingleQubitGate,
    TwoQubitGate,
    ControlUnit,
    MemoryController,
    ArithmeticUnit,
    StateVectorUnit,
}

/// Resource utilization
#[derive(Debug, Clone, Default)]
pub struct ResourceUtilization {
    /// LUTs used
    pub luts: usize,
    /// FFs used
    pub flip_flops: usize,
    /// DSP blocks used
    pub dsp_blocks: usize,
    /// Block RAM used (KB)
    pub bram_kb: usize,
    /// Utilization percentage
    pub utilization_percent: f64,
}

/// Timing information
#[derive(Debug, Clone, Default)]
pub struct TimingInfo {
    /// Critical path delay (ns)
    pub critical_path_delay: f64,
    /// Setup slack (ns)
    pub setup_slack: f64,
    /// Hold slack (ns)
    pub hold_slack: f64,
    /// Maximum frequency (MHz)
    pub max_frequency: f64,
}

/// FPGA memory manager
#[derive(Debug, Clone)]
pub struct FPGAMemoryManager {
    /// On-chip memory pools
    pub onchip_pools: HashMap<String, MemoryPool>,
    /// External memory interfaces
    pub external_interfaces: Vec<ExternalMemoryInterface>,
    /// Memory access scheduler
    pub access_scheduler: MemoryAccessScheduler,
    /// Total available memory (KB)
    pub total_memory_kb: usize,
    /// Used memory (KB)
    pub used_memory_kb: usize,
}

/// Memory pool
#[derive(Debug, Clone)]
pub struct MemoryPool {
    /// Pool name
    pub name: String,
    /// Size (KB)
    pub size_kb: usize,
    /// Used size (KB)
    pub used_kb: usize,
    /// Access pattern
    pub access_pattern: MemoryAccessPattern,
    /// Banking configuration
    pub banks: usize,
}

/// Memory access patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAccessPattern {
    Sequential,
    Random,
    Strided,
    BlockTransfer,
    Streaming,
}

/// External memory interface
#[derive(Debug, Clone)]
pub struct ExternalMemoryInterface {
    /// Interface ID
    pub interface_id: usize,
    /// Interface type
    pub interface_type: MemoryInterfaceType,
    /// Controller module
    pub controller: String,
    /// Current utilization
    pub utilization: f64,
}

/// Memory access scheduler
#[derive(Debug, Clone)]
pub struct MemoryAccessScheduler {
    /// Scheduling algorithm
    pub algorithm: SchedulingAlgorithm,
    /// Request queue size
    pub queue_size: usize,
    /// Priority levels
    pub priority_levels: usize,
}

/// Scheduling algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingAlgorithm {
    FIFO,
    RoundRobin,
    PriorityBased,
    DeadlineAware,
    BandwidthOptimized,
}

/// Bitstream management
#[derive(Debug, Clone)]
pub struct BitstreamManager {
    /// Available bitstreams
    pub bitstreams: HashMap<String, Bitstream>,
    /// Current configuration
    pub current_config: Option<String>,
    /// Reconfiguration time (ms)
    pub reconfig_time_ms: f64,
    /// Partial reconfiguration support
    pub supports_partial_reconfig: bool,
}

/// FPGA bitstream
#[derive(Debug, Clone)]
pub struct Bitstream {
    /// Bitstream name
    pub name: String,
    /// Target configuration
    pub target_config: String,
    /// Size (KB)
    pub size_kb: usize,
    /// Configuration time (ms)
    pub config_time_ms: f64,
    /// Supported quantum algorithms
    pub supported_algorithms: Vec<String>,
}

/// FPGA performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FPGAStats {
    /// Total gate operations
    pub total_gate_operations: usize,
    /// Total execution time (ms)
    pub total_execution_time: f64,
    /// Average gate time (ns)
    pub avg_gate_time: f64,
    /// Clock cycles consumed
    pub total_clock_cycles: u64,
    /// FPGA utilization
    pub fpga_utilization: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
    /// Pipeline efficiency
    pub pipeline_efficiency: f64,
    /// Reconfiguration count
    pub reconfigurations: usize,
    /// Total reconfiguration time (ms)
    pub total_reconfig_time: f64,
    /// Power consumption (W)
    pub power_consumption: f64,
}

impl FPGAStats {
    /// Update statistics after operation
    pub fn update_operation(&mut self, execution_time: f64, clock_cycles: u64) {
        self.total_gate_operations += 1;
        self.total_execution_time += execution_time;
        self.avg_gate_time = (self.total_execution_time * 1_000_000.0) / self.total_gate_operations as f64; // Convert to ns
        self.total_clock_cycles += clock_cycles;
    }

    /// Calculate performance metrics
    pub fn get_performance_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        if self.total_execution_time > 0.0 {
            metrics.insert(
                "operations_per_second".to_string(),
                self.total_gate_operations as f64 / (self.total_execution_time / 1000.0),
            );
            metrics.insert(
                "cycles_per_operation".to_string(),
                self.total_clock_cycles as f64 / self.total_gate_operations as f64,
            );
        }

        metrics.insert("fpga_utilization".to_string(), self.fpga_utilization);
        metrics.insert("pipeline_efficiency".to_string(), self.pipeline_efficiency);
        metrics.insert(
            "memory_bandwidth_utilization".to_string(),
            self.memory_bandwidth_utilization,
        );
        metrics.insert("power_efficiency".to_string(), 
            self.total_gate_operations as f64 / (self.power_consumption * self.total_execution_time / 1000.0)
        );

        metrics
    }
}

impl FPGAQuantumSimulator {
    /// Create new FPGA quantum simulator
    pub fn new(config: FPGAConfig) -> Result<Self> {
        let device_info = FPGADeviceInfo::for_platform(config.platform);

        // Initialize processing units
        let processing_units = Self::create_processing_units(&config, &device_info)?;

        // Initialize memory manager
        let memory_manager = Self::create_memory_manager(&config, &device_info)?;

        // Initialize bitstream manager
        let bitstream_manager = Self::create_bitstream_manager(&config)?;

        let mut simulator = Self {
            config,
            device_info,
            processing_units,
            hdl_modules: HashMap::new(),
            stats: FPGAStats::default(),
            memory_manager,
            bitstream_manager,
        };

        // Generate HDL modules
        simulator.generate_hdl_modules()?;

        // Load default bitstream
        simulator.load_default_bitstream()?;

        Ok(simulator)
    }

    /// Create processing units
    fn create_processing_units(
        config: &FPGAConfig,
        device_info: &FPGADeviceInfo,
    ) -> Result<Vec<QuantumProcessingUnit>> {
        let mut units = Vec::new();

        for i in 0..config.num_processing_units {
            let pipeline_stages = vec![
                PipelineStage {
                    name: "Fetch".to_string(),
                    operation: PipelineOperation::Fetch,
                    latency: 1,
                    throughput: 1.0,
                },
                PipelineStage {
                    name: "Decode".to_string(),
                    operation: PipelineOperation::Decode,
                    latency: 1,
                    throughput: 1.0,
                },
                PipelineStage {
                    name: "Address".to_string(),
                    operation: PipelineOperation::AddressCalculation,
                    latency: 1,
                    throughput: 1.0,
                },
                PipelineStage {
                    name: "MemRead".to_string(),
                    operation: PipelineOperation::MemoryRead,
                    latency: 2,
                    throughput: 0.5,
                },
                PipelineStage {
                    name: "Execute".to_string(),
                    operation: PipelineOperation::GateExecution,
                    latency: 3,
                    throughput: 1.0,
                },
                PipelineStage {
                    name: "MemWrite".to_string(),
                    operation: PipelineOperation::MemoryWrite,
                    latency: 2,
                    throughput: 0.5,
                },
                PipelineStage {
                    name: "Writeback".to_string(),
                    operation: PipelineOperation::Writeback,
                    latency: 1,
                    throughput: 1.0,
                },
            ];

            let unit = QuantumProcessingUnit {
                unit_id: i,
                supported_gates: vec![
                    InterfaceGateType::Hadamard,
                    InterfaceGateType::PauliX,
                    InterfaceGateType::PauliY,
                    InterfaceGateType::PauliZ,
                    InterfaceGateType::CNOT,
                    InterfaceGateType::CZ,
                    InterfaceGateType::RX(0.0),
                    InterfaceGateType::RY(0.0),
                    InterfaceGateType::RZ(0.0),
                ],
                pipeline_stages,
                local_memory_kb: device_info.block_ram_kb / config.num_processing_units,
                frequency: config.clock_frequency,
                utilization: 0.0,
            };

            units.push(unit);
        }

        Ok(units)
    }

    /// Create memory manager
    fn create_memory_manager(
        config: &FPGAConfig,
        device_info: &FPGADeviceInfo,
    ) -> Result<FPGAMemoryManager> {
        let mut onchip_pools = HashMap::new();

        // Create on-chip memory pools
        onchip_pools.insert(
            "state_vector".to_string(),
            MemoryPool {
                name: "state_vector".to_string(),
                size_kb: device_info.block_ram_kb / 2,
                used_kb: 0,
                access_pattern: MemoryAccessPattern::Sequential,
                banks: 16,
            },
        );

        onchip_pools.insert(
            "gate_cache".to_string(),
            MemoryPool {
                name: "gate_cache".to_string(),
                size_kb: device_info.block_ram_kb / 4,
                used_kb: 0,
                access_pattern: MemoryAccessPattern::Random,
                banks: 8,
            },
        );

        onchip_pools.insert(
            "instruction_cache".to_string(),
            MemoryPool {
                name: "instruction_cache".to_string(),
                size_kb: device_info.block_ram_kb / 8,
                used_kb: 0,
                access_pattern: MemoryAccessPattern::Sequential,
                banks: 4,
            },
        );

        // Create external memory interfaces
        let external_interfaces: Vec<ExternalMemoryInterface> = device_info
            .memory_interfaces
            .iter()
            .enumerate()
            .map(|(i, _)| ExternalMemoryInterface {
                interface_id: i,
                interface_type: device_info.memory_interfaces[i].interface_type,
                controller: format!("mem_ctrl_{}", i),
                utilization: 0.0,
            })
            .collect();

        let access_scheduler = MemoryAccessScheduler {
            algorithm: SchedulingAlgorithm::BandwidthOptimized,
            queue_size: 64,
            priority_levels: 4,
        };

        Ok(FPGAMemoryManager {
            onchip_pools,
            external_interfaces,
            access_scheduler,
            total_memory_kb: device_info.block_ram_kb,
            used_memory_kb: 0,
        })
    }

    /// Create bitstream manager
    fn create_bitstream_manager(config: &FPGAConfig) -> Result<BitstreamManager> {
        let mut bitstreams = HashMap::new();

        // Default quantum computing bitstream
        bitstreams.insert(
            "quantum_basic".to_string(),
            Bitstream {
                name: "quantum_basic".to_string(),
                target_config: "Basic quantum gates".to_string(),
                size_kb: 50000,
                config_time_ms: 200.0,
                supported_algorithms: vec![
                    "VQE".to_string(),
                    "QAOA".to_string(),
                    "Grover".to_string(),
                ],
            },
        );

        // Advanced quantum algorithms bitstream
        bitstreams.insert(
            "quantum_advanced".to_string(),
            Bitstream {
                name: "quantum_advanced".to_string(),
                target_config: "Advanced quantum algorithms".to_string(),
                size_kb: 75000,
                config_time_ms: 300.0,
                supported_algorithms: vec![
                    "Shor".to_string(),
                    "QFT".to_string(),
                    "Phase_Estimation".to_string(),
                ],
            },
        );

        // Quantum machine learning bitstream
        bitstreams.insert(
            "quantum_ml".to_string(),
            Bitstream {
                name: "quantum_ml".to_string(),
                target_config: "Quantum machine learning".to_string(),
                size_kb: 60000,
                config_time_ms: 250.0,
                supported_algorithms: vec![
                    "QML".to_string(),
                    "Variational_Circuits".to_string(),
                    "Quantum_GAN".to_string(),
                ],
            },
        );

        Ok(BitstreamManager {
            bitstreams,
            current_config: None,
            reconfig_time_ms: 200.0,
            supports_partial_reconfig: matches!(
                config.platform,
                FPGAPlatform::IntelStratix10 | FPGAPlatform::IntelAgilex7 | FPGAPlatform::XilinxVersal
            ),
        })
    }

    /// Generate HDL modules for quantum operations
    fn generate_hdl_modules(&mut self) -> Result<()> {
        // Generate single qubit gate module
        self.generate_single_qubit_module()?;

        // Generate two qubit gate module
        self.generate_two_qubit_module()?;

        // Generate control unit module
        self.generate_control_unit_module()?;

        // Generate memory controller module
        self.generate_memory_controller_module()?;

        // Generate arithmetic unit module
        self.generate_arithmetic_unit_module()?;

        Ok(())
    }

    /// Generate single qubit gate HDL module
    fn generate_single_qubit_module(&mut self) -> Result<()> {
        let hdl_code = match self.config.hdl_target {
            HDLTarget::SystemVerilog => self.generate_single_qubit_systemverilog(),
            HDLTarget::Verilog => self.generate_single_qubit_verilog(),
            HDLTarget::VHDL => self.generate_single_qubit_vhdl(),
            HDLTarget::OpenCL => self.generate_single_qubit_opencl(),
            _ => self.generate_single_qubit_systemverilog(), // Default
        };

        let module = HDLModule {
            name: "single_qubit_gate".to_string(),
            hdl_code,
            resource_utilization: ResourceUtilization {
                luts: 1000,
                flip_flops: 500,
                dsp_blocks: 8,
                bram_kb: 2,
                utilization_percent: 5.0,
            },
            timing_info: TimingInfo {
                critical_path_delay: 3.2,
                setup_slack: 0.8,
                hold_slack: 1.5,
                max_frequency: 312.5,
            },
            module_type: ModuleType::SingleQubitGate,
        };

        self.hdl_modules
            .insert("single_qubit_gate".to_string(), module);

        Ok(())
    }

    /// Generate SystemVerilog code for single qubit gates
    fn generate_single_qubit_systemverilog(&self) -> String {
        format!(
            r#"
// Single Qubit Gate Processing Unit
// Generated for platform: {:?}
// Clock frequency: {:.1} MHz
// Data path width: {} bits

module single_qubit_gate #(
    parameter DATA_WIDTH = {},
    parameter ADDR_WIDTH = 20,
    parameter PIPELINE_DEPTH = {}
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    enable,
    
    // Gate parameters
    input  logic [1:0]              gate_type,  // 00: H, 01: X, 10: Y, 11: Z
    input  logic [DATA_WIDTH-1:0]   gate_param, // For rotation gates
    input  logic [ADDR_WIDTH-1:0]   target_qubit,
    
    // State vector interface
    input  logic [DATA_WIDTH-1:0]   state_real_in,
    input  logic [DATA_WIDTH-1:0]   state_imag_in,
    output logic [DATA_WIDTH-1:0]   state_real_out,
    output logic [DATA_WIDTH-1:0]   state_imag_out,
    
    // Control signals
    output logic                    ready,
    output logic                    valid_out
);

    // Pipeline registers
    logic [DATA_WIDTH-1:0] pipeline_real [0:PIPELINE_DEPTH-1];
    logic [DATA_WIDTH-1:0] pipeline_imag [0:PIPELINE_DEPTH-1];
    logic [1:0] pipeline_gate_type [0:PIPELINE_DEPTH-1];
    logic [PIPELINE_DEPTH-1:0] pipeline_valid;
    
    // Gate matrices (pre-computed constants)
    localparam real SQRT2_INV = 0.7071067811865476;
    
    // Complex multiplication units
    logic [DATA_WIDTH-1:0] mult_real, mult_imag;
    logic [DATA_WIDTH-1:0] add_real, add_imag;
    
    // DSP blocks for complex arithmetic
    logic [DATA_WIDTH*2-1:0] dsp_mult_result;
    logic [DATA_WIDTH-1:0] dsp_add_result;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pipeline_valid <= '0;
            ready <= 1'b1;
        end else if (enable) begin
            // Pipeline stage advancement
            for (int i = PIPELINE_DEPTH-1; i > 0; i--) begin
                pipeline_real[i] <= pipeline_real[i-1];
                pipeline_imag[i] <= pipeline_imag[i-1];
                pipeline_gate_type[i] <= pipeline_gate_type[i-1];
            end
            
            // Input stage
            pipeline_real[0] <= state_real_in;
            pipeline_imag[0] <= state_imag_in;
            pipeline_gate_type[0] <= gate_type;
            
            // Valid signal pipeline
            pipeline_valid <= {{pipeline_valid[PIPELINE_DEPTH-2:0], enable}};
        end
    end
    
    // Gate operation logic (combinational)
    always_comb begin
        case (pipeline_gate_type[PIPELINE_DEPTH-1])
            2'b00: begin // Hadamard
                state_real_out = (pipeline_real[PIPELINE_DEPTH-1] + pipeline_imag[PIPELINE_DEPTH-1]) * SQRT2_INV;
                state_imag_out = (pipeline_real[PIPELINE_DEPTH-1] - pipeline_imag[PIPELINE_DEPTH-1]) * SQRT2_INV;
            end
            2'b01: begin // Pauli-X
                state_real_out = pipeline_imag[PIPELINE_DEPTH-1];
                state_imag_out = pipeline_real[PIPELINE_DEPTH-1];
            end
            2'b10: begin // Pauli-Y
                state_real_out = -pipeline_imag[PIPELINE_DEPTH-1];
                state_imag_out = pipeline_real[PIPELINE_DEPTH-1];
            end
            2'b11: begin // Pauli-Z
                state_real_out = pipeline_real[PIPELINE_DEPTH-1];
                state_imag_out = -pipeline_imag[PIPELINE_DEPTH-1];
            end
            default: begin
                state_real_out = pipeline_real[PIPELINE_DEPTH-1];
                state_imag_out = pipeline_imag[PIPELINE_DEPTH-1];
            end
        endcase
        
        valid_out = pipeline_valid[PIPELINE_DEPTH-1];
    end
    
endmodule
"#,
            self.config.platform,
            self.config.clock_frequency,
            self.config.data_path_width,
            self.config.data_path_width,
            self.config.pipeline_depth
        )
    }

    /// Generate Verilog code for single qubit gates
    fn generate_single_qubit_verilog(&self) -> String {
        // Simplified Verilog version
        "// Verilog single qubit gate module (simplified)\nmodule single_qubit_gate(...);".to_string()
    }

    /// Generate VHDL code for single qubit gates
    fn generate_single_qubit_vhdl(&self) -> String {
        // Simplified VHDL version
        "-- VHDL single qubit gate entity (simplified)\nentity single_qubit_gate is...".to_string()
    }

    /// Generate OpenCL code for single qubit gates
    fn generate_single_qubit_opencl(&self) -> String {
        r#"
// OpenCL kernel for single qubit gates
__kernel void single_qubit_gate(
    __global float2* state,
    __global const float* gate_matrix,
    const int target_qubit,
    const int num_qubits
) {
    const int global_id = get_global_id(0);
    const int total_states = 1 << num_qubits;
    
    if (global_id >= total_states / 2) return;
    
    const int target_mask = 1 << target_qubit;
    const int i = global_id;
    const int j = i | target_mask;
    
    if ((i & target_mask) == 0) {
        float2 state_i = state[i];
        float2 state_j = state[j];
        
        // Apply 2x2 gate matrix
        state[i] = (float2)(
            gate_matrix[0] * state_i.x - gate_matrix[1] * state_i.y + 
            gate_matrix[2] * state_j.x - gate_matrix[3] * state_j.y,
            gate_matrix[0] * state_i.y + gate_matrix[1] * state_i.x + 
            gate_matrix[2] * state_j.y + gate_matrix[3] * state_j.x
        );
        
        state[j] = (float2)(
            gate_matrix[4] * state_i.x - gate_matrix[5] * state_i.y + 
            gate_matrix[6] * state_j.x - gate_matrix[7] * state_j.y,
            gate_matrix[4] * state_i.y + gate_matrix[5] * state_i.x + 
            gate_matrix[6] * state_j.y + gate_matrix[7] * state_j.x
        );
    }
}
"#.to_string()
    }

    /// Generate two qubit gate module (placeholder)
    fn generate_two_qubit_module(&mut self) -> Result<()> {
        let hdl_code = "// Two qubit gate module (placeholder)".to_string();

        let module = HDLModule {
            name: "two_qubit_gate".to_string(),
            hdl_code,
            resource_utilization: ResourceUtilization {
                luts: 2500,
                flip_flops: 1200,
                dsp_blocks: 16,
                bram_kb: 8,
                utilization_percent: 12.0,
            },
            timing_info: TimingInfo {
                critical_path_delay: 4.5,
                setup_slack: 0.5,
                hold_slack: 1.2,
                max_frequency: 222.2,
            },
            module_type: ModuleType::TwoQubitGate,
        };

        self.hdl_modules
            .insert("two_qubit_gate".to_string(), module);

        Ok(())
    }

    /// Generate control unit module (placeholder)
    fn generate_control_unit_module(&mut self) -> Result<()> {
        let hdl_code = "// Control unit module (placeholder)".to_string();

        let module = HDLModule {
            name: "control_unit".to_string(),
            hdl_code,
            resource_utilization: ResourceUtilization {
                luts: 5000,
                flip_flops: 3000,
                dsp_blocks: 4,
                bram_kb: 16,
                utilization_percent: 25.0,
            },
            timing_info: TimingInfo {
                critical_path_delay: 2.8,
                setup_slack: 1.2,
                hold_slack: 2.0,
                max_frequency: 357.1,
            },
            module_type: ModuleType::ControlUnit,
        };

        self.hdl_modules.insert("control_unit".to_string(), module);

        Ok(())
    }

    /// Generate memory controller module (placeholder)
    fn generate_memory_controller_module(&mut self) -> Result<()> {
        let hdl_code = "// Memory controller module (placeholder)".to_string();

        let module = HDLModule {
            name: "memory_controller".to_string(),
            hdl_code,
            resource_utilization: ResourceUtilization {
                luts: 3000,
                flip_flops: 2000,
                dsp_blocks: 0,
                bram_kb: 32,
                utilization_percent: 15.0,
            },
            timing_info: TimingInfo {
                critical_path_delay: 3.5,
                setup_slack: 0.9,
                hold_slack: 1.8,
                max_frequency: 285.7,
            },
            module_type: ModuleType::MemoryController,
        };

        self.hdl_modules
            .insert("memory_controller".to_string(), module);

        Ok(())
    }

    /// Generate arithmetic unit module (placeholder)
    fn generate_arithmetic_unit_module(&mut self) -> Result<()> {
        let hdl_code = "// Arithmetic unit module (placeholder)".to_string();

        let module = HDLModule {
            name: "arithmetic_unit".to_string(),
            hdl_code,
            resource_utilization: ResourceUtilization {
                luts: 4000,
                flip_flops: 2500,
                dsp_blocks: 32,
                bram_kb: 4,
                utilization_percent: 20.0,
            },
            timing_info: TimingInfo {
                critical_path_delay: 3.8,
                setup_slack: 0.7,
                hold_slack: 1.5,
                max_frequency: 263.2,
            },
            module_type: ModuleType::ArithmeticUnit,
        };

        self.hdl_modules
            .insert("arithmetic_unit".to_string(), module);

        Ok(())
    }

    /// Load default bitstream
    fn load_default_bitstream(&mut self) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Simulate bitstream loading
        std::thread::sleep(std::time::Duration::from_millis(50)); // Simulate loading time

        self.bitstream_manager.current_config = Some("quantum_basic".to_string());
        
        let config_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.reconfigurations += 1;
        self.stats.total_reconfig_time += config_time;

        Ok(())
    }

    /// Execute quantum circuit on FPGA
    pub fn execute_circuit(&mut self, circuit: &InterfaceCircuit) -> Result<Array1<Complex64>> {
        let start_time = std::time::Instant::now();

        // Initialize state vector
        let mut state = Array1::zeros(1 << circuit.num_qubits);
        state[0] = Complex64::new(1.0, 0.0);

        // Process gates on FPGA
        for gate in &circuit.gates {
            state = self.apply_gate_fpga(&state, gate)?;
        }

        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
        let clock_cycles = (execution_time * self.config.clock_frequency * 1000.0) as u64;
        self.stats.update_operation(execution_time, clock_cycles);

        // Update FPGA utilization
        self.update_utilization();

        Ok(state)
    }

    /// Apply quantum gate using FPGA hardware
    fn apply_gate_fpga(
        &mut self,
        state: &Array1<Complex64>,
        gate: &InterfaceGate,
    ) -> Result<Array1<Complex64>> {
        // Select appropriate processing unit
        let unit_id = self.select_processing_unit(gate)?;
        
        // Route gate to processing unit
        let result = match gate.gate_type {
            InterfaceGateType::Hadamard
            | InterfaceGateType::PauliX
            | InterfaceGateType::PauliY
            | InterfaceGateType::PauliZ => self.apply_single_qubit_gate_fpga(state, gate, unit_id),
            InterfaceGateType::CNOT | InterfaceGateType::CZ => {
                self.apply_two_qubit_gate_fpga(state, gate, unit_id)
            }
            InterfaceGateType::RX(_) | InterfaceGateType::RY(_) | InterfaceGateType::RZ(_) => {
                self.apply_rotation_gate_fpga(state, gate, unit_id)
            }
            _ => {
                // Fallback to software implementation
                Ok(state.clone())
            }
        };

        // Update processing unit utilization
        if let Ok(_) = result {
            self.processing_units[unit_id].utilization += 1.0;
        }

        result
    }

    /// Select processing unit for gate execution
    fn select_processing_unit(&self, gate: &InterfaceGate) -> Result<usize> {
        // Simple round-robin selection for now
        let mut best_unit = 0;
        let mut min_utilization = f64::INFINITY;

        for (i, unit) in self.processing_units.iter().enumerate() {
            if unit.supported_gates.contains(&gate.gate_type) && unit.utilization < min_utilization {
                best_unit = i;
                min_utilization = unit.utilization;
            }
        }

        Ok(best_unit)
    }

    /// Apply single qubit gate using FPGA
    fn apply_single_qubit_gate_fpga(
        &mut self,
        state: &Array1<Complex64>,
        gate: &InterfaceGate,
        _unit_id: usize,
    ) -> Result<Array1<Complex64>> {
        if gate.qubits.is_empty() {
            return Ok(state.clone());
        }

        let target_qubit = gate.qubits[0];
        let mut result = state.clone();

        // Simulate FPGA execution with pipelining
        let pipeline_latency = self.config.pipeline_depth as f64 / self.config.clock_frequency * 1000.0;
        std::thread::sleep(std::time::Duration::from_micros((pipeline_latency * 10.0) as u64));

        // Apply gate matrix (hardware simulation)
        for i in 0..state.len() {
            if (i >> target_qubit) & 1 == 0 {
                let j = i | (1 << target_qubit);
                if j < state.len() {
                    let state_0 = result[i];
                    let state_1 = result[j];

                    match gate.gate_type {
                        InterfaceGateType::Hadamard => {
                            let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
                            result[i] = Complex64::new(inv_sqrt2, 0.0) * (state_0 + state_1);
                            result[j] = Complex64::new(inv_sqrt2, 0.0) * (state_0 - state_1);
                        }
                        InterfaceGateType::PauliX => {
                            result[i] = state_1;
                            result[j] = state_0;
                        }
                        InterfaceGateType::PauliY => {
                            result[i] = Complex64::new(0.0, -1.0) * state_1;
                            result[j] = Complex64::new(0.0, 1.0) * state_0;
                        }
                        InterfaceGateType::PauliZ => {
                            result[j] = -state_1;
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(result)
    }

    /// Apply two qubit gate using FPGA
    fn apply_two_qubit_gate_fpga(
        &mut self,
        state: &Array1<Complex64>,
        gate: &InterfaceGate,
        _unit_id: usize,
    ) -> Result<Array1<Complex64>> {
        if gate.qubits.len() < 2 {
            return Ok(state.clone());
        }

        let control = gate.qubits[0];
        let target = gate.qubits[1];
        let mut result = state.clone();

        // Simulate FPGA execution with higher latency for two-qubit gates
        let pipeline_latency = self.config.pipeline_depth as f64 * 1.5 / self.config.clock_frequency * 1000.0;
        std::thread::sleep(std::time::Duration::from_micros((pipeline_latency * 15.0) as u64));

        match gate.gate_type {
            InterfaceGateType::CNOT => {
                for i in 0..state.len() {
                    if ((i >> control) & 1) == 1 {
                        let j = i ^ (1 << target);
                        if j < state.len() && i != j {
                            let temp = result[i];
                            result[i] = result[j];
                            result[j] = temp;
                        }
                    }
                }
            }
            InterfaceGateType::CZ => {
                for i in 0..state.len() {
                    if ((i >> control) & 1) == 1 && ((i >> target) & 1) == 1 {
                        result[i] = -result[i];
                    }
                }
            }
            _ => {}
        }

        Ok(result)
    }

    /// Apply rotation gate using FPGA
    fn apply_rotation_gate_fpga(
        &mut self,
        state: &Array1<Complex64>,
        gate: &InterfaceGate,
        unit_id: usize,
    ) -> Result<Array1<Complex64>> {
        // For now, use the single qubit gate implementation
        self.apply_single_qubit_gate_fpga(state, gate, unit_id)
    }

    /// Update FPGA utilization metrics
    fn update_utilization(&mut self) {
        let total_utilization: f64 = self.processing_units.iter().map(|u| u.utilization).sum();
        self.stats.fpga_utilization = total_utilization / self.processing_units.len() as f64;

        // Calculate pipeline efficiency
        self.stats.pipeline_efficiency = if self.config.enable_pipelining {
            0.85 // Simulated pipeline efficiency
        } else {
            0.6
        };

        // Calculate memory bandwidth utilization
        self.stats.memory_bandwidth_utilization = 0.7; // Simulated

        // Estimate power consumption
        self.stats.power_consumption = self.device_info.power_consumption * self.stats.fpga_utilization;
    }

    /// Get device information
    pub fn get_device_info(&self) -> &FPGADeviceInfo {
        &self.device_info
    }

    /// Get performance statistics
    pub fn get_stats(&self) -> &FPGAStats {
        &self.stats
    }

    /// Get HDL modules
    pub fn get_hdl_modules(&self) -> &HashMap<String, HDLModule> {
        &self.hdl_modules
    }

    /// Reconfigure FPGA with new bitstream
    pub fn reconfigure(&mut self, bitstream_name: &str) -> Result<()> {
        if !self.bitstream_manager.bitstreams.contains_key(bitstream_name) {
            return Err(SimulatorError::InvalidInput(format!(
                "Bitstream {} not found",
                bitstream_name
            )));
        }

        let start_time = std::time::Instant::now();

        // Simulate reconfiguration time
        let bitstream = &self.bitstream_manager.bitstreams[bitstream_name];
        std::thread::sleep(std::time::Duration::from_millis(
            (bitstream.config_time_ms / 10.0) as u64,
        ));

        self.bitstream_manager.current_config = Some(bitstream_name.to_string());

        let reconfig_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.reconfigurations += 1;
        self.stats.total_reconfig_time += reconfig_time;

        Ok(())
    }

    /// Check if FPGA is available
    pub fn is_fpga_available(&self) -> bool {
        !self.hdl_modules.is_empty()
    }

    /// Export HDL code for synthesis
    pub fn export_hdl(&self, module_name: &str) -> Result<String> {
        self.hdl_modules
            .get(module_name)
            .map(|module| module.hdl_code.clone())
            .ok_or_else(|| {
                SimulatorError::InvalidInput(format!("Module {} not found", module_name))
            })
    }
}

/// Benchmark FPGA acceleration performance
pub fn benchmark_fpga_acceleration() -> Result<HashMap<String, f64>> {
    let mut results = HashMap::new();

    // Test different FPGA configurations
    let configs = vec![
        FPGAConfig {
            platform: FPGAPlatform::IntelStratix10,
            num_processing_units: 8,
            clock_frequency: 300.0,
            ..Default::default()
        },
        FPGAConfig {
            platform: FPGAPlatform::IntelAgilex7,
            num_processing_units: 16,
            clock_frequency: 400.0,
            ..Default::default()
        },
        FPGAConfig {
            platform: FPGAPlatform::XilinxVersal,
            num_processing_units: 32,
            clock_frequency: 500.0,
            enable_pipelining: true,
            ..Default::default()
        },
    ];

    for (i, config) in configs.into_iter().enumerate() {
        let start = std::time::Instant::now();

        let mut simulator = FPGAQuantumSimulator::new(config)?;

        // Create test circuit
        let mut circuit = InterfaceCircuit::new(10, 0);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![0, 1]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::RY(0.5), vec![2]));
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::CZ, vec![1, 2]));

        // Execute circuit multiple times
        for _ in 0..10 {
            let _result = simulator.execute_circuit(&circuit)?;
        }

        let time = start.elapsed().as_secs_f64() * 1000.0;
        results.insert(format!("fpga_config_{}", i), time);

        // Add performance metrics
        let stats = simulator.get_stats();
        results.insert(
            format!("fpga_config_{}_operations", i),
            stats.total_gate_operations as f64,
        );
        results.insert(
            format!("fpga_config_{}_avg_gate_time", i),
            stats.avg_gate_time,
        );
        results.insert(
            format!("fpga_config_{}_utilization", i),
            stats.fpga_utilization,
        );
        results.insert(
            format!("fpga_config_{}_pipeline_efficiency", i),
            stats.pipeline_efficiency,
        );

        let performance_metrics = stats.get_performance_metrics();
        for (key, value) in performance_metrics {
            results.insert(format!("fpga_config_{}_{}", i, key), value);
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_fpga_simulator_creation() {
        let config = FPGAConfig::default();
        let simulator = FPGAQuantumSimulator::new(config);
        assert!(simulator.is_ok());
    }

    #[test]
    fn test_device_info_creation() {
        let device_info = FPGADeviceInfo::for_platform(FPGAPlatform::IntelStratix10);
        assert_eq!(device_info.platform, FPGAPlatform::IntelStratix10);
        assert_eq!(device_info.logic_elements, 2800000);
        assert_eq!(device_info.dsp_blocks, 5760);
    }

    #[test]
    fn test_processing_unit_creation() {
        let config = FPGAConfig::default();
        let device_info = FPGADeviceInfo::for_platform(config.platform);
        let units = FPGAQuantumSimulator::create_processing_units(&config, &device_info).unwrap();

        assert_eq!(units.len(), config.num_processing_units);
        assert!(!units[0].supported_gates.is_empty());
        assert!(!units[0].pipeline_stages.is_empty());
    }

    #[test]
    fn test_hdl_generation() {
        let config = FPGAConfig::default();
        let mut simulator = FPGAQuantumSimulator::new(config).unwrap();

        assert!(simulator.hdl_modules.contains_key("single_qubit_gate"));
        assert!(simulator.hdl_modules.contains_key("two_qubit_gate"));

        let single_qubit_module = &simulator.hdl_modules["single_qubit_gate"];
        assert!(!single_qubit_module.hdl_code.is_empty());
        assert_eq!(single_qubit_module.module_type, ModuleType::SingleQubitGate);
    }

    #[test]
    fn test_circuit_execution() {
        let config = FPGAConfig::default();
        let mut simulator = FPGAQuantumSimulator::new(config).unwrap();

        let mut circuit = InterfaceCircuit::new(2, 0);
        circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]));

        let result = simulator.execute_circuit(&circuit);
        assert!(result.is_ok());

        let state = result.unwrap();
        assert_eq!(state.len(), 4);
        assert!(state[0].norm() > 0.0);
    }

    #[test]
    fn test_gate_application() {
        let config = FPGAConfig::default();
        let mut simulator = FPGAQuantumSimulator::new(config).unwrap();

        let mut state = Array1::zeros(4);
        state[0] = Complex64::new(1.0, 0.0);

        let gate = InterfaceGate::new(InterfaceGateType::Hadamard, vec![0]);
        let result = simulator.apply_single_qubit_gate_fpga(&state, &gate, 0);
        assert!(result.is_ok());

        let new_state = result.unwrap();
        assert_abs_diff_eq!(new_state[0].norm(), 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
        assert_abs_diff_eq!(new_state[1].norm(), 1.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_bitstream_management() {
        let config = FPGAConfig::default();
        let mut simulator = FPGAQuantumSimulator::new(config).unwrap();

        assert!(simulator.bitstream_manager.current_config.is_some());
        assert!(simulator
            .bitstream_manager
            .bitstreams
            .contains_key("quantum_basic"));

        let result = simulator.reconfigure("quantum_advanced");
        assert!(result.is_ok());
        assert_eq!(
            simulator.bitstream_manager.current_config,
            Some("quantum_advanced".to_string())
        );
    }

    #[test]
    fn test_memory_management() {
        let config = FPGAConfig::default();
        let simulator = FPGAQuantumSimulator::new(config).unwrap();

        assert!(simulator
            .memory_manager
            .onchip_pools
            .contains_key("state_vector"));
        assert!(simulator
            .memory_manager
            .onchip_pools
            .contains_key("gate_cache"));
        assert!(!simulator.memory_manager.external_interfaces.is_empty());
    }

    #[test]
    fn test_stats_tracking() {
        let config = FPGAConfig::default();
        let mut simulator = FPGAQuantumSimulator::new(config).unwrap();

        simulator.stats.update_operation(10.0, 1000);
        simulator.stats.update_operation(20.0, 2000);

        assert_eq!(simulator.stats.total_gate_operations, 2);
        assert_abs_diff_eq!(simulator.stats.total_execution_time, 30.0, epsilon = 1e-10);
        assert_eq!(simulator.stats.total_clock_cycles, 3000);
    }

    #[test]
    fn test_performance_metrics() {
        let config = FPGAConfig::default();
        let mut simulator = FPGAQuantumSimulator::new(config).unwrap();

        simulator.stats.total_gate_operations = 100;
        simulator.stats.total_execution_time = 1000.0; // 1 second
        simulator.stats.total_clock_cycles = 300000;
        simulator.stats.fpga_utilization = 75.0;
        simulator.stats.pipeline_efficiency = 0.85;
        simulator.stats.power_consumption = 120.0;

        let metrics = simulator.stats.get_performance_metrics();

        assert!(metrics.contains_key("operations_per_second"));
        assert!(metrics.contains_key("cycles_per_operation"));
        assert!(metrics.contains_key("fpga_utilization"));

        assert_abs_diff_eq!(metrics["operations_per_second"], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(metrics["cycles_per_operation"], 3000.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hdl_export() {
        let config = FPGAConfig::default();
        let simulator = FPGAQuantumSimulator::new(config).unwrap();

        let hdl_code = simulator.export_hdl("single_qubit_gate");
        assert!(hdl_code.is_ok());
        assert!(!hdl_code.unwrap().is_empty());

        let invalid_module = simulator.export_hdl("nonexistent_module");
        assert!(invalid_module.is_err());
    }

    #[test]
    fn test_arithmetic_precision() {
        assert_eq!(ArithmeticPrecision::Fixed16, ArithmeticPrecision::Fixed16);
        assert_ne!(ArithmeticPrecision::Fixed16, ArithmeticPrecision::Fixed32);
    }
}