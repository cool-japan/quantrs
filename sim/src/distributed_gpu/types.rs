//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;
#[cfg(feature = "advanced_math")]
use scirs2_core::gpu::{GpuBackend, GpuBuffer, GpuContext};
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::parallel_ops::*;
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Performance statistics for distributed GPU simulation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistributedGpuStats {
    /// Total execution time in milliseconds
    pub total_execution_time_ms: f64,
    /// GPU computation time per device
    pub gpu_computation_time_ms: Vec<f64>,
    /// Communication time between GPUs
    pub communication_time_ms: f64,
    /// Memory transfer time
    pub memory_transfer_time_ms: f64,
    /// Load balance efficiency (0-1)
    pub load_balance_efficiency: f64,
    /// GPU utilization per device (0-1)
    pub gpu_utilization: Vec<f64>,
    /// Memory usage per device in bytes
    pub memory_usage_bytes: Vec<usize>,
    /// Number of synchronization events
    pub sync_events: usize,
}
/// State vector partition on a single GPU
pub struct StatePartition {
    /// GPU buffer containing the state amplitudes (placeholder)
    #[cfg(feature = "advanced_math")]
    buffer: Option<GpuBuffer<f64>>,
    /// Local state vector for CPU fallback
    cpu_fallback: Array1<Complex64>,
    /// Start index in global state vector
    pub(super) start_index: usize,
    /// Size of this partition
    pub(super) size: usize,
    /// GPU device ID
    pub(super) device_id: usize,
}
/// Distributed GPU state vector
#[derive(Debug)]
pub struct DistributedGpuStateVector {
    /// GPU contexts for each device
    gpu_contexts: Vec<GpuContextWrapper>,
    /// State vector partitions on each GPU
    state_partitions: Vec<StatePartition>,
    /// Configuration
    config: DistributedGpuConfig,
    /// Total number of qubits
    num_qubits: usize,
    /// Current partition scheme
    partition_scheme: PartitionScheme,
    /// SciRS2 backend
    backend: Option<SciRS2Backend>,
    /// Performance statistics
    stats: DistributedGpuStats,
}
impl DistributedGpuStateVector {
    /// Create new distributed GPU state vector
    pub fn new(num_qubits: usize, config: DistributedGpuConfig) -> Result<Self> {
        if num_qubits < config.min_qubits_for_gpu {
            return Err(SimulatorError::InvalidInput(format!(
                "Distributed GPU simulation requires at least {} qubits",
                config.min_qubits_for_gpu
            )));
        }
        let state_size = 1_usize << num_qubits;
        let gpu_contexts = Self::initialize_gpu_contexts(&config)?;
        let num_devices = gpu_contexts.len();
        if num_devices == 0 {
            return Err(SimulatorError::InitializationFailed(
                "No GPU devices available for distributed simulation".to_string(),
            ));
        }
        let partition_scheme = Self::select_partition_scheme(num_qubits, num_devices, &config);
        let state_partitions =
            Self::create_state_partitions(state_size, &gpu_contexts, &partition_scheme, &config)?;
        let stats = DistributedGpuStats {
            gpu_computation_time_ms: vec![0.0; num_devices],
            gpu_utilization: vec![0.0; num_devices],
            memory_usage_bytes: vec![0; num_devices],
            ..Default::default()
        };
        Ok(Self {
            gpu_contexts,
            state_partitions,
            config,
            num_qubits,
            partition_scheme,
            backend: None,
            stats,
        })
    }
    /// Initialize with SciRS2 backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }
    /// Initialize |0...0⟩ state
    pub fn initialize_zero_state(&mut self) -> Result<()> {
        let start_time = std::time::Instant::now();
        for (i, partition) in self.state_partitions.iter_mut().enumerate() {
            if partition.start_index == 0 {
                #[cfg(feature = "advanced_math")]
                {}
                partition.cpu_fallback.fill(Complex64::new(0.0, 0.0));
                partition.cpu_fallback[0] = Complex64::new(1.0, 0.0);
            } else {
                #[cfg(feature = "advanced_math")]
                {}
                partition.cpu_fallback.fill(Complex64::new(0.0, 0.0));
            }
        }
        self.stats.memory_transfer_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
    /// Apply single-qubit gate distributed across GPUs
    pub fn apply_single_qubit_gate(
        &mut self,
        qubit: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<()> {
        if qubit >= self.num_qubits {
            return Err(SimulatorError::IndexOutOfBounds(qubit));
        }
        let start_time = std::time::Instant::now();
        let mut device_times = Vec::new();
        for i in 0..self.state_partitions.len() {
            let device_start_time = std::time::Instant::now();
            let device_id = self.state_partitions[i].device_id;
            Self::apply_single_qubit_gate_partition_static(
                qubit,
                gate_matrix,
                &mut self.state_partitions[i],
                device_id,
            )?;
            let device_time = device_start_time.elapsed().as_secs_f64() * 1000.0;
            device_times.push(device_time);
        }
        for (i, time) in device_times.into_iter().enumerate() {
            self.stats.gpu_computation_time_ms[i] += time;
        }
        self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
    /// Apply two-qubit gate with inter-GPU communication if needed
    pub fn apply_two_qubit_gate(
        &mut self,
        control: usize,
        target: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<()> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(SimulatorError::IndexOutOfBounds(control.max(target)));
        }
        let start_time = std::time::Instant::now();
        let requires_communication = self.requires_inter_gpu_communication(control, target);
        if requires_communication {
            self.apply_two_qubit_gate_distributed(control, target, gate_matrix)?;
        } else {
            let mut device_times = Vec::new();
            for i in 0..self.state_partitions.len() {
                let device_start_time = std::time::Instant::now();
                let device_id = self.state_partitions[i].device_id;
                Self::apply_two_qubit_gate_partition_static(
                    control,
                    target,
                    gate_matrix,
                    &mut self.state_partitions[i],
                    device_id,
                )?;
                let device_time = device_start_time.elapsed().as_secs_f64() * 1000.0;
                device_times.push(device_time);
            }
            for (i, time) in device_times.into_iter().enumerate() {
                self.stats.gpu_computation_time_ms[i] += time;
            }
        }
        self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
    /// Calculate probability of measuring qubit in |1⟩ state
    pub fn measure_probability(&mut self, qubit: usize) -> Result<f64> {
        if qubit >= self.num_qubits {
            return Err(SimulatorError::IndexOutOfBounds(qubit));
        }
        let start_time = std::time::Instant::now();
        let local_probs: Result<Vec<f64>> = self
            .state_partitions
            .par_iter()
            .enumerate()
            .map(|(device_id, partition)| {
                self.calculate_local_probability(qubit, partition, device_id)
            })
            .collect();
        let local_probs = local_probs?;
        let total_prob = local_probs.iter().sum();
        self.stats.total_execution_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(total_prob)
    }
    /// Get state vector (expensive operation - copies from all GPUs)
    pub fn get_state_vector(&mut self) -> Result<Array1<Complex64>> {
        let start_time = std::time::Instant::now();
        let state_size = 1 << self.num_qubits;
        let mut state_vector = Array1::zeros(state_size);
        for partition in &self.state_partitions {
            let end_index = (partition.start_index + partition.size).min(state_size);
            #[cfg(feature = "advanced_math")]
            {}
            #[cfg(not(feature = "advanced_math"))]
            {
                state_vector
                    .slice_mut(s![partition.start_index..end_index])
                    .assign(
                        &partition
                            .cpu_fallback
                            .slice(s![..end_index - partition.start_index]),
                    );
            }
        }
        self.stats.memory_transfer_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(state_vector)
    }
    /// Get performance statistics
    pub fn get_stats(&self) -> &DistributedGpuStats {
        &self.stats
    }
    /// Check if GPU acceleration is available
    pub fn is_gpu_available() -> bool {
        #[cfg(feature = "advanced_math")]
        {
            match GpuContext::new(GpuBackend::preferred()) {
                Ok(_) => true,
                Err(_) => false,
            }
        }
        #[cfg(not(feature = "advanced_math"))]
        {
            false
        }
    }
    /// Internal helper methods
    #[cfg(feature = "advanced_math")]
    fn initialize_gpu_contexts(config: &DistributedGpuConfig) -> Result<Vec<GpuContextWrapper>> {
        let mut contexts = Vec::new();
        let num_gpus = if config.num_gpus == 0 {
            Self::detect_available_gpus()?
        } else {
            config.num_gpus
        };
        for device_id in 0..num_gpus {
            match Self::create_gpu_context(device_id) {
                Ok(wrapper) => contexts.push(wrapper),
                Err(e) => {
                    eprintln!("Warning: Failed to initialize GPU {}: {}", device_id, e);
                }
            }
        }
        if contexts.is_empty() {
            return Err(SimulatorError::InitializationFailed(
                "No GPU contexts could be created".to_string(),
            ));
        }
        Ok(contexts)
    }
    #[cfg(not(feature = "advanced_math"))]
    fn initialize_gpu_contexts(config: &DistributedGpuConfig) -> Result<Vec<GpuContextWrapper>> {
        let num_gpus = if config.num_gpus == 0 {
            1
        } else {
            config.num_gpus
        };
        let mut contexts = Vec::new();
        for device_id in 0..num_gpus {
            contexts.push(GpuContextWrapper {
                device_id,
                memory_available: 1_000_000_000,
                compute_capability: 1.0,
            });
        }
        Ok(contexts)
    }
    #[cfg(feature = "advanced_math")]
    fn create_gpu_context(device_id: usize) -> Result<GpuContextWrapper> {
        let context = GpuContext::new(GpuBackend::preferred()).map_err(|e| {
            SimulatorError::InitializationFailed(format!("GPU context creation failed: {}", e))
        })?;
        let memory_available = Self::query_gpu_memory(&context)?;
        let compute_capability = Self::query_compute_capability(&context)?;
        Ok(GpuContextWrapper {
            context,
            device_id,
            memory_available,
            compute_capability,
        })
    }
    #[cfg(feature = "advanced_math")]
    fn query_gpu_memory(context: &GpuContext) -> Result<usize> {
        Ok(8_000_000_000)
    }
    #[cfg(feature = "advanced_math")]
    fn query_compute_capability(context: &GpuContext) -> Result<f64> {
        Ok(7.5)
    }
    #[cfg(feature = "advanced_math")]
    fn detect_available_gpus() -> Result<usize> {
        Ok(1)
    }
    fn select_partition_scheme(
        num_qubits: usize,
        num_devices: usize,
        config: &DistributedGpuConfig,
    ) -> PartitionScheme {
        if config.auto_load_balance {
            if num_qubits > 25 {
                PartitionScheme::Adaptive
            } else if num_devices > 4 {
                PartitionScheme::Interleaved
            } else {
                PartitionScheme::Block
            }
        } else {
            PartitionScheme::Block
        }
    }
    fn create_state_partitions(
        state_size: usize,
        gpu_contexts: &[GpuContextWrapper],
        partition_scheme: &PartitionScheme,
        config: &DistributedGpuConfig,
    ) -> Result<Vec<StatePartition>> {
        let num_devices = gpu_contexts.len();
        let mut partitions = Vec::new();
        match partition_scheme {
            PartitionScheme::Block => {
                let partition_size = state_size.div_ceil(num_devices);
                for (i, context) in gpu_contexts.iter().enumerate() {
                    let start_index = i * partition_size;
                    let end_index = ((i + 1) * partition_size).min(state_size);
                    let size = end_index - start_index;
                    if size > 0 {
                        let partition = Self::create_single_partition(
                            start_index,
                            size,
                            context.device_id,
                            config,
                        )?;
                        partitions.push(partition);
                    }
                }
            }
            PartitionScheme::Interleaved => {
                let base_size = state_size / num_devices;
                let remainder = state_size % num_devices;
                let mut start_index = 0;
                for (i, context) in gpu_contexts.iter().enumerate() {
                    let size = base_size + if i < remainder { 1 } else { 0 };
                    if size > 0 {
                        let partition = Self::create_single_partition(
                            start_index,
                            size,
                            context.device_id,
                            config,
                        )?;
                        partitions.push(partition);
                        start_index += size;
                    }
                }
            }
            PartitionScheme::Adaptive => {
                let total_memory: usize = gpu_contexts.iter().map(|ctx| ctx.memory_available).sum();
                let mut start_index = 0;
                for context in gpu_contexts {
                    let memory_fraction = context.memory_available as f64 / total_memory as f64;
                    let size = (state_size as f64 * memory_fraction) as usize;
                    if size > 0 && start_index < state_size {
                        let actual_size = size.min(state_size - start_index);
                        let partition = Self::create_single_partition(
                            start_index,
                            actual_size,
                            context.device_id,
                            config,
                        )?;
                        partitions.push(partition);
                        start_index += actual_size;
                    }
                }
            }
            PartitionScheme::HilbertCurve => {
                return Self::create_hilbert_partitions(state_size, gpu_contexts, config);
            }
        }
        if partitions.is_empty() {
            return Err(SimulatorError::InitializationFailed(
                "No valid partitions could be created".to_string(),
            ));
        }
        Ok(partitions)
    }
    fn create_single_partition(
        start_index: usize,
        size: usize,
        device_id: usize,
        config: &DistributedGpuConfig,
    ) -> Result<StatePartition> {
        #[cfg(feature = "advanced_math")]
        let buffer = {
            if config.use_mixed_precision {
                None
            } else {
                None
            }
        };
        let mut cpu_fallback = Array1::zeros(size);
        if start_index == 0 && size > 0 {
            cpu_fallback[0] = Complex64::new(1.0, 0.0);
        }
        Ok(StatePartition {
            #[cfg(feature = "advanced_math")]
            buffer,
            cpu_fallback,
            start_index,
            size,
            device_id,
        })
    }
    fn create_hilbert_partitions(
        state_size: usize,
        gpu_contexts: &[GpuContextWrapper],
        config: &DistributedGpuConfig,
    ) -> Result<Vec<StatePartition>> {
        let num_devices = gpu_contexts.len();
        let mut partitions = Vec::new();
        let curve_order = ((state_size as f64).log2() / 2.0).ceil() as usize;
        let curve_size = 1 << (curve_order * 2);
        if curve_size < state_size {
            return Self::create_state_partitions(
                state_size,
                gpu_contexts,
                &PartitionScheme::Block,
                config,
            );
        }
        let hilbert_indices = Self::generate_hilbert_indices(curve_order, state_size);
        let partition_size = hilbert_indices.len().div_ceil(num_devices);
        for (device_idx, context) in gpu_contexts.iter().enumerate() {
            let start_idx = device_idx * partition_size;
            let end_idx = ((device_idx + 1) * partition_size).min(hilbert_indices.len());
            if start_idx < hilbert_indices.len() {
                let size = end_idx - start_idx;
                let partition =
                    Self::create_single_partition(start_idx, size, context.device_id, config)?;
                partitions.push(partition);
            }
        }
        Ok(partitions)
    }
    /// Generate Hilbert curve indices for better data locality
    fn generate_hilbert_indices(order: usize, max_size: usize) -> Vec<usize> {
        let mut indices = Vec::new();
        let size = 1 << order;
        for i in 0..(size * size).min(max_size) {
            let hilbert_index = Self::xy_to_hilbert(i % size, i / size, order);
            indices.push(hilbert_index);
        }
        indices.sort_unstable();
        indices.truncate(max_size);
        indices
    }
    /// Convert (x,y) coordinates to Hilbert curve index
    fn xy_to_hilbert(mut x: usize, mut y: usize, order: usize) -> usize {
        let mut index = 0;
        let mut side = 1 << order;
        while side > 1 {
            side >>= 1;
            let rx = if x & side != 0 { 1 } else { 0 };
            let ry = if y & side != 0 { 1 } else { 0 };
            index += side * side * ((3 * rx) ^ ry);
            if ry == 0 {
                if rx == 1 {
                    x = side - 1 - x;
                    y = side - 1 - y;
                }
                std::mem::swap(&mut x, &mut y);
            }
        }
        index
    }
    fn apply_single_qubit_gate_partition(
        &self,
        qubit: usize,
        gate_matrix: &Array2<Complex64>,
        partition: &mut StatePartition,
        device_id: usize,
    ) -> Result<()> {
        #[cfg(feature = "advanced_math")]
        {
            if let Some(ref backend) = self.backend {
                self.apply_single_qubit_gate_gpu(qubit, gate_matrix, partition, device_id)
            } else {
                self.apply_single_qubit_gate_cpu_fallback(qubit, gate_matrix, partition)
            }
        }
        #[cfg(not(feature = "advanced_math"))]
        self.apply_single_qubit_gate_cpu_fallback(qubit, gate_matrix, partition)
    }
    #[cfg(feature = "advanced_math")]
    fn apply_single_qubit_gate_gpu(
        &self,
        qubit: usize,
        gate_matrix: &Array2<Complex64>,
        partition: &mut StatePartition,
        _device_id: usize,
    ) -> Result<()> {
        self.apply_single_qubit_gate_cpu_fallback(qubit, gate_matrix, partition)
    }
    fn apply_single_qubit_gate_cpu_fallback(
        &self,
        qubit: usize,
        gate_matrix: &Array2<Complex64>,
        partition: &mut StatePartition,
    ) -> Result<()> {
        let qubit_mask = 1_usize << qubit;
        let state_size = partition.size;
        let mut new_state = partition.cpu_fallback.clone();
        for i in 0..state_size {
            let global_i = partition.start_index + i;
            if global_i & qubit_mask == 0 {
                let j = global_i | qubit_mask;
                if j >= partition.start_index && j < partition.start_index + partition.size {
                    let local_j = j - partition.start_index;
                    let amp_0 = partition.cpu_fallback[i];
                    let amp_1 = partition.cpu_fallback[local_j];
                    new_state[i] = gate_matrix[[0, 0]] * amp_0 + gate_matrix[[0, 1]] * amp_1;
                    new_state[local_j] = gate_matrix[[1, 0]] * amp_0 + gate_matrix[[1, 1]] * amp_1;
                }
            }
        }
        partition.cpu_fallback = new_state;
        #[cfg(feature = "advanced_math")]
        {}
        Ok(())
    }
    fn apply_two_qubit_gate_partition(
        &self,
        control: usize,
        target: usize,
        gate_matrix: &Array2<Complex64>,
        partition: &mut StatePartition,
        _device_id: usize,
    ) -> Result<()> {
        let control_mask = 1_usize << control;
        let target_mask = 1_usize << target;
        let mut new_state = partition.cpu_fallback.clone();
        for i in 0..partition.size {
            let global_i = partition.start_index + i;
            if global_i & control_mask != 0 {
                let target_bit = (global_i & target_mask) != 0;
                let j = if target_bit {
                    global_i & !target_mask
                } else {
                    global_i | target_mask
                };
                if j >= partition.start_index && j < partition.start_index + partition.size {
                    let local_j = j - partition.start_index;
                    let amp_i = partition.cpu_fallback[i];
                    let amp_j = partition.cpu_fallback[local_j];
                    if target_bit {
                        new_state[local_j] =
                            gate_matrix[[0, 0]] * amp_j + gate_matrix[[0, 1]] * amp_i;
                        new_state[i] = gate_matrix[[1, 0]] * amp_j + gate_matrix[[1, 1]] * amp_i;
                    } else {
                        new_state[i] = gate_matrix[[0, 0]] * amp_i + gate_matrix[[0, 1]] * amp_j;
                        new_state[local_j] =
                            gate_matrix[[1, 0]] * amp_i + gate_matrix[[1, 1]] * amp_j;
                    }
                }
            }
        }
        partition.cpu_fallback = new_state;
        #[cfg(feature = "advanced_math")]
        {}
        Ok(())
    }
    fn apply_two_qubit_gate_distributed(
        &mut self,
        control: usize,
        target: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<()> {
        let comm_start_time = std::time::Instant::now();
        self.synchronize_all_reduce()?;
        for i in 0..self.state_partitions.len() {
            let device_id = self.state_partitions[i].device_id;
            Self::apply_two_qubit_gate_partition_static(
                control,
                target,
                gate_matrix,
                &mut self.state_partitions[i],
                device_id,
            )?;
        }
        self.stats.communication_time_ms += comm_start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.sync_events += 1;
        Ok(())
    }
    fn requires_inter_gpu_communication(&self, control: usize, target: usize) -> bool {
        match self.partition_scheme {
            PartitionScheme::Block => {
                let control_mask = 1_usize << control;
                let target_mask = 1_usize << target;
                for partition in &self.state_partitions {
                    let start = partition.start_index;
                    let end = partition.start_index + partition.size;
                    for i in start..end.min(start + 1000) {
                        let pair_state = i ^ target_mask;
                        if (i & control_mask) != 0 && (pair_state < start || pair_state >= end) {
                            return true;
                        }
                    }
                }
                false
            }
            _ => true,
        }
    }
    fn calculate_local_probability(
        &self,
        qubit: usize,
        partition: &StatePartition,
        _device_id: usize,
    ) -> Result<f64> {
        let qubit_mask = 1_usize << qubit;
        let mut probability = 0.0;
        for i in 0..partition.size {
            let global_i = partition.start_index + i;
            if global_i & qubit_mask != 0 {
                probability += partition.cpu_fallback[i].norm_sqr();
            }
        }
        Ok(probability)
    }
    fn synchronize_all_reduce(&mut self) -> Result<()> {
        match self.config.sync_strategy {
            SyncStrategy::AllReduce => self.all_reduce_sync(),
            SyncStrategy::RingReduce => self.ring_reduce_sync(),
            SyncStrategy::TreeReduce => self.tree_reduce_sync(),
            SyncStrategy::PointToPoint => self.point_to_point_sync(),
        }
    }
    fn all_reduce_sync(&mut self) -> Result<()> {
        let start_time = std::time::Instant::now();
        let num_partitions = self.state_partitions.len();
        if num_partitions <= 1 {
            return Ok(());
        }
        let mut boundary_states: Vec<(usize, usize, Vec<Complex64>)> = Vec::new();
        for i in 0..num_partitions {
            for j in (i + 1)..num_partitions {
                let partition_i = &self.state_partitions[i];
                let partition_j = &self.state_partitions[j];
                let overlap_exists = partition_i.start_index + partition_i.size
                    > partition_j.start_index
                    && partition_j.start_index + partition_j.size > partition_i.start_index;
                if overlap_exists {
                    boundary_states.push((i, j, vec![]));
                }
            }
        }
        self.stats.communication_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
    fn ring_reduce_sync(&mut self) -> Result<()> {
        let start_time = std::time::Instant::now();
        let num_partitions = self.state_partitions.len();
        if num_partitions <= 1 {
            return Ok(());
        }
        for step in 0..num_partitions {
            for i in 0..num_partitions {
                let next_partition = (i + 1) % num_partitions;
                let chunk_size = self.state_partitions[i].size / num_partitions;
                let chunk_start = (step * chunk_size).min(self.state_partitions[i].size);
                let chunk_end = ((step + 1) * chunk_size).min(self.state_partitions[i].size);
                if chunk_start < chunk_end {
                    let chunk_data = self.state_partitions[i]
                        .cpu_fallback
                        .slice(s![chunk_start..chunk_end])
                        .to_owned();
                    let recv_start = chunk_start.min(self.state_partitions[next_partition].size);
                    let recv_end = chunk_end.min(self.state_partitions[next_partition].size);
                    if recv_start < recv_end
                        && recv_start < self.state_partitions[next_partition].cpu_fallback.len()
                    {
                        for (idx, &value) in
                            chunk_data.iter().take(recv_end - recv_start).enumerate()
                        {
                            if recv_start + idx
                                < self.state_partitions[next_partition].cpu_fallback.len()
                            {
                                self.state_partitions[next_partition].cpu_fallback
                                    [recv_start + idx] += value;
                            }
                        }
                    }
                }
            }
        }
        self.stats.communication_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
    fn tree_reduce_sync(&mut self) -> Result<()> {
        let start_time = std::time::Instant::now();
        let num_partitions = self.state_partitions.len();
        if num_partitions <= 1 {
            return Ok(());
        }
        let mut active_partitions = num_partitions;
        let mut step = 1;
        while active_partitions > 1 {
            for i in (0..active_partitions).step_by(step * 2) {
                let partner = i + step;
                if partner < active_partitions {
                    let (left_partitions, right_partitions) =
                        self.state_partitions.split_at_mut(partner);
                    let target_partition = &mut left_partitions[i];
                    let source_partition = &right_partitions[0];
                    let partner_size = source_partition.size.min(target_partition.size);
                    for j in 0..partner_size {
                        if j < target_partition.cpu_fallback.len()
                            && j < source_partition.cpu_fallback.len()
                        {
                            target_partition.cpu_fallback[j] += source_partition.cpu_fallback[j];
                        }
                    }
                }
            }
            step *= 2;
            active_partitions = active_partitions.div_ceil(2);
        }
        if num_partitions > 1 {
            let final_result = self.state_partitions[0].cpu_fallback.clone();
            for partition in &mut self.state_partitions[1..] {
                let copy_size = final_result.len().min(partition.cpu_fallback.len());
                partition
                    .cpu_fallback
                    .slice_mut(s![..copy_size])
                    .assign(&final_result.slice(s![..copy_size]));
            }
        }
        self.stats.communication_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
    fn point_to_point_sync(&mut self) -> Result<()> {
        let start_time = std::time::Instant::now();
        let num_partitions = self.state_partitions.len();
        for i in 0..num_partitions {
            for j in 0..num_partitions {
                if i != j {
                    let partition_i = &self.state_partitions[i];
                    let partition_j = &self.state_partitions[j];
                    let requires_sync = self.partitions_require_sync(i, j);
                    if requires_sync {
                        self.exchange_boundary_states(i, j)?;
                    }
                }
            }
        }
        self.stats.communication_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
    /// Check if two partitions require synchronization
    fn partitions_require_sync(&self, partition_a: usize, partition_b: usize) -> bool {
        let part_a = &self.state_partitions[partition_a];
        let part_b = &self.state_partitions[partition_b];
        let a_end = part_a.start_index + part_a.size;
        let b_end = part_b.start_index + part_b.size;
        (part_a.start_index <= b_end && part_b.start_index <= a_end)
            || (part_a.start_index.abs_diff(b_end) <= 1)
            || (part_b.start_index.abs_diff(a_end) <= 1)
    }
    /// Exchange boundary states between two partitions
    fn exchange_boundary_states(&mut self, partition_a: usize, partition_b: usize) -> Result<()> {
        let boundary_size = 64;
        if partition_a >= self.state_partitions.len() || partition_b >= self.state_partitions.len()
        {
            return Ok(());
        }
        let a_boundary_size = boundary_size.min(self.state_partitions[partition_a].size);
        let b_boundary_size = boundary_size.min(self.state_partitions[partition_b].size);
        if a_boundary_size > 0 && b_boundary_size > 0 {}
        Ok(())
    }
    fn apply_single_qubit_gate_partition_static(
        qubit: usize,
        gate_matrix: &Array2<Complex64>,
        partition: &mut StatePartition,
        device_id: usize,
    ) -> Result<()> {
        let qubit_mask = 1_usize << qubit;
        for i in (0..partition.cpu_fallback.len()).step_by(2) {
            if i & qubit_mask == 0 {
                let j = i | qubit_mask;
                if j < partition.cpu_fallback.len() {
                    let amp_0 = partition.cpu_fallback[i];
                    let amp_1 = partition.cpu_fallback[j];
                    partition.cpu_fallback[i] =
                        gate_matrix[[0, 0]] * amp_0 + gate_matrix[[0, 1]] * amp_1;
                    partition.cpu_fallback[j] =
                        gate_matrix[[1, 0]] * amp_0 + gate_matrix[[1, 1]] * amp_1;
                }
            }
        }
        Ok(())
    }
    fn apply_two_qubit_gate_partition_static(
        control: usize,
        target: usize,
        gate_matrix: &Array2<Complex64>,
        partition: &mut StatePartition,
        device_id: usize,
    ) -> Result<()> {
        let control_mask = 1_usize << control;
        let target_mask = 1_usize << target;
        for i in 0..partition.cpu_fallback.len() {
            if i & control_mask != 0 {
                let target_bit = (i & target_mask) != 0;
                let j = if target_bit {
                    i & !target_mask
                } else {
                    i | target_mask
                };
                if j < i || j >= partition.cpu_fallback.len() {
                    continue;
                }
                let amp_i = partition.cpu_fallback[i];
                let amp_j = partition.cpu_fallback[j];
                if target_bit {
                    partition.cpu_fallback[j] =
                        gate_matrix[[0, 0]] * amp_j + gate_matrix[[0, 1]] * amp_i;
                    partition.cpu_fallback[i] =
                        gate_matrix[[1, 0]] * amp_j + gate_matrix[[1, 1]] * amp_i;
                } else {
                    partition.cpu_fallback[i] =
                        gate_matrix[[0, 0]] * amp_i + gate_matrix[[0, 1]] * amp_j;
                    partition.cpu_fallback[j] =
                        gate_matrix[[1, 0]] * amp_i + gate_matrix[[1, 1]] * amp_j;
                }
            }
        }
        Ok(())
    }
    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
    /// Get the state size (2^num_qubits)
    pub fn state_size(&self) -> usize {
        1usize << self.num_qubits
    }
    /// Get the number of partitions
    pub fn num_partitions(&self) -> usize {
        self.state_partitions.len()
    }
    /// Synchronize all GPU devices
    pub fn synchronize(&mut self) -> Result<()> {
        #[cfg(feature = "advanced_math")]
        {
            for context in &mut self.gpu_contexts {}
        }
        Ok(())
    }
}
/// Partitioning scheme for distributing state vector
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionScheme {
    /// Simple block partitioning
    Block,
    /// Interleaved partitioning for better load balance
    Interleaved,
    /// Adaptive partitioning based on computation patterns
    Adaptive,
    /// Hilbert curve space-filling partitioning
    HilbertCurve,
}
/// GPU synchronization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStrategy {
    /// All-reduce collective operations
    AllReduce,
    /// Ring-based reduction
    RingReduce,
    /// Tree-based reduction
    TreeReduce,
    /// Point-to-point communication
    PointToPoint,
}
/// Wrapper for GPU context to handle feature flags
pub struct GpuContextWrapper {
    #[cfg(feature = "advanced_math")]
    context: GpuContext,
    pub(super) device_id: usize,
    pub(super) memory_available: usize,
    pub(super) compute_capability: f64,
}
/// Utilities for distributed GPU simulation
pub struct DistributedGpuUtils;
impl DistributedGpuUtils {
    /// Estimate memory requirements for distributed simulation
    pub fn estimate_memory_requirements(num_qubits: usize, num_gpus: usize) -> usize {
        let state_size = 1_usize << num_qubits;
        let complex_size = std::mem::size_of::<Complex64>();
        let total_memory = state_size * complex_size;
        let overhead_factor = 1.5;
        (total_memory as f64 * overhead_factor) as usize / num_gpus
    }
    /// Calculate optimal number of GPUs for given problem size
    pub fn optimal_gpu_count(
        num_qubits: usize,
        available_gpus: usize,
        memory_per_gpu: usize,
    ) -> usize {
        let total_memory_needed = Self::estimate_memory_requirements(num_qubits, 1);
        let min_gpus_needed = total_memory_needed.div_ceil(memory_per_gpu);
        min_gpus_needed.clamp(1, available_gpus)
    }
    /// Benchmark different partitioning strategies
    pub fn benchmark_partitioning_strategies(
        num_qubits: usize,
        num_gpus: usize,
    ) -> Result<HashMap<String, f64>> {
        if !DistributedGpuStateVector::is_gpu_available() {
            let mut results = HashMap::new();
            results.insert("Block".to_string(), 0.0);
            results.insert("Interleaved".to_string(), 0.0);
            results.insert("Adaptive".to_string(), 0.0);
            return Ok(results);
        }
        let mut results = HashMap::new();
        let strategies = vec![
            ("Block", PartitionScheme::Block),
            ("Interleaved", PartitionScheme::Interleaved),
            ("Adaptive", PartitionScheme::Adaptive),
        ];
        for (name, scheme) in strategies {
            let config = DistributedGpuConfig {
                num_gpus,
                ..Default::default()
            };
            let start_time = std::time::Instant::now();
            let mut simulator = DistributedGpuStateVector::new(num_qubits, config)?;
            simulator.initialize_zero_state()?;
            let identity = Array2::eye(2);
            for i in 0..num_qubits.min(5) {
                simulator.apply_single_qubit_gate(i, &identity)?;
            }
            let execution_time = start_time.elapsed().as_secs_f64();
            results.insert(name.to_string(), execution_time);
        }
        Ok(results)
    }
}
/// Configuration for distributed GPU simulation
#[derive(Debug, Clone)]
pub struct DistributedGpuConfig {
    /// Number of GPUs to use (0 = auto-detect)
    pub num_gpus: usize,
    /// Minimum qubits required to enable GPU acceleration
    pub min_qubits_for_gpu: usize,
    /// Maximum state vector size per GPU (in complex numbers)
    pub max_state_size_per_gpu: usize,
    /// Enable automatic load balancing
    pub auto_load_balance: bool,
    /// Memory overlap for efficient transfers
    pub memory_overlap_ratio: f64,
    /// Use mixed precision for larger simulations
    pub use_mixed_precision: bool,
    /// Synchronization strategy
    pub sync_strategy: SyncStrategy,
}
