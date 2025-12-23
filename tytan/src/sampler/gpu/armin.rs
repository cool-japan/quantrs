//! GPU-accelerated Armin Sampler Implementation

use scirs2_core::ndarray::{Array, Ix2};
#[cfg(all(feature = "gpu", feature = "dwave"))]
use scirs2_core::random::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use std::collections::HashMap;

#[cfg(all(feature = "gpu", feature = "dwave"))]
use super::super::evaluate_qubo_energy;
use super::super::{SampleResult, Sampler, SamplerError, SamplerResult};

#[cfg(all(feature = "gpu", feature = "dwave"))]
use ocl::{
    self,
    enums::{DeviceInfo, DeviceInfoResult},
    Context, DeviceType, Program,
};

/// GPU-accelerated Sampler (Armin)
///
/// This sampler uses GPU acceleration to find solutions to
/// QUBO/HOBO problems. It is based on parallel tempering and
/// is optimized for large problems.
#[cfg(feature = "gpu")]
pub struct ArminSampler {
    /// Random number generator seed
    seed: Option<u64>,
    /// Whether to use GPU ("GPU") or CPU ("CPU")
    mode: String,
    /// Device to use (e.g., "cuda:0")
    device: String,
    /// Whether to show verbose output
    verbose: bool,
}

#[cfg(feature = "gpu")]
impl ArminSampler {
    /// Create a new GPU-accelerated sampler
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional random seed for reproducibility
    #[must_use]
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            seed,
            mode: "GPU".to_string(),
            device: "cuda:0".to_string(),
            verbose: true,
        }
    }

    /// Create a new GPU-accelerated sampler with custom parameters
    ///
    /// # Arguments
    ///
    /// * `seed` - An optional random seed for reproducibility
    /// * `mode` - Whether to use GPU ("GPU") or CPU ("CPU")
    /// * `device` - Device to use (e.g., "cuda:0")
    /// * `verbose` - Whether to show verbose output
    #[must_use]
    pub fn with_params(seed: Option<u64>, mode: &str, device: &str, verbose: bool) -> Self {
        Self {
            seed,
            mode: mode.to_string(),
            device: device.to_string(),
            verbose,
        }
    }

    /// Run GPU-accelerated annealing using OpenCL
    #[cfg(all(feature = "gpu", feature = "dwave"))]
    fn run_gpu_annealing(
        &self,
        n_vars: usize,
        h_vector: &[f64],
        j_matrix: &[f64],
        num_shots: usize,
    ) -> Result<Vec<Vec<bool>>, ocl::Error> {
        use ocl::flags;
        use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue};

        // Check problem size
        if n_vars > 2048 {
            if self.verbose {
                println!(
                    "Problem size too large for standard OpenCL kernel. Using chunked approach."
                );
            }
            return self.run_gpu_annealing_chunked(n_vars, h_vector, j_matrix, num_shots);
        }

        // Display progress if verbose
        if self.verbose {
            println!("Initializing GPU with {n_vars} variables and {num_shots} shots");
        }

        // Set up OpenCL environment
        let platform = if self.device.contains("cpu") {
            // Find CPU platform
            Platform::list()
                .into_iter()
                .find(|p| p.name().unwrap_or_default().to_lowercase().contains("cpu"))
                .unwrap_or_else(Platform::default)
        } else {
            // Default platform (typically GPU)
            Platform::default()
        };

        if self.verbose {
            println!("Using platform: {}", platform.name().unwrap_or_default());
        }

        // Find appropriate device
        let device = if self.device.contains("cpu") {
            // CPU device
            Device::list_all(platform)
                .unwrap_or_default()
                .into_iter()
                .find(|d| {
                    matches!(d.info(DeviceInfo::Type).ok(), Some(DeviceInfoResult::Type(dt)) if dt == DeviceType::default().cpu())
                })
                .unwrap_or_else(|| Device::first(platform).unwrap())
        } else {
            // GPU device
            Device::list_all(platform)
                .unwrap_or_default()
                .into_iter()
                .find(|d| {
                    matches!(d.info(DeviceInfo::Type).ok(), Some(DeviceInfoResult::Type(dt)) if dt == DeviceType::default().gpu())
                })
                .unwrap_or_else(|| Device::first(platform).unwrap())
        };

        if self.verbose {
            println!("Using device: {}", device.name().unwrap_or_default());
        }

        // Build context and queue
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;
        let queue = Queue::new(&context, device, None)?;

        // Use GPU kernel from kernels module
        let src = super::kernels::SIMULATED_ANNEALING_KERNEL;

        // Compile the program
        let context = Context::builder().devices(device).build()?;
        let program = Program::builder()
            .devices(device)
            .src(src)
            .build(&context)?;

        // Set up buffers
        let h_buffer = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .len(n_vars)
            .build()?;

        let j_buffer = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .len(n_vars * n_vars)
            .build()?;

        let solutions_buffer = Buffer::<u8>::builder()
            .queue(queue)
            .flags(flags::MEM_WRITE_ONLY)
            .len(num_shots * n_vars)
            .build()?;

        // Convert h_vector and j_matrix to f32
        let h_vec_f32: Vec<f32> = h_vector.iter().map(|&x| x as f32).collect();
        let j_mat_f32: Vec<f32> = j_matrix.iter().map(|&x| x as f32).collect();

        // Transfer data to GPU
        h_buffer.write(&h_vec_f32).enq()?;
        j_buffer.write(&j_mat_f32).enq()?;

        // Set up kernel parameters
        let init_temp = 10.0f32;
        let mut final_temp = 0.1f32;
        let sweeps = if n_vars < 100 {
            1000
        } else if n_vars < 500 {
            2000
        } else {
            5000
        };

        if self.verbose {
            println!("Running {sweeps} sweeps with temperature range [{final_temp}, {init_temp}]");
        }

        // Create a seed based on input seed or random value
        let seed_val = self.seed.unwrap_or_else(|| thread_rng().gen());

        // Set up and run standard simulated annealing kernel
        let mut kernel = Kernel::builder()
            .program(&program)
            .name("simulated_annealing")
            .global_work_size(num_shots)
            .arg(n_vars as i32)
            .arg(&h_buffer)
            .arg(&j_buffer)
            .arg(&solutions_buffer)
            .arg(num_shots as i32)
            .arg(init_temp)
            .arg(final_temp)
            .arg(sweeps)
            .arg(seed_val)
            .build()?;

        // Execute kernel
        unsafe {
            kernel.enq()?;
        }

        // Read results
        let mut solutions_data = vec![0u8; num_shots * n_vars];
        solutions_buffer.read(&mut solutions_data).enq()?;

        // Convert to Vec<Vec<bool>>
        let mut results = Vec::with_capacity(num_shots);
        for i in 0..num_shots {
            let mut solution = Vec::with_capacity(n_vars);
            for j in 0..n_vars {
                solution.push(solutions_data[i * n_vars + j] != 0);
            }
            results.push(solution);
        }

        Ok(results)
    }

    /// Chunked GPU annealing for very large problems
    #[cfg(all(feature = "gpu", feature = "dwave"))]
    fn run_gpu_annealing_chunked(
        &self,
        n_vars: usize,
        h_vector: &[f64],
        j_matrix: &[f64],
        num_shots: usize,
    ) -> Result<Vec<Vec<bool>>, ocl::Error> {
        // For problems too large to fit in a single kernel, we chunk the problem
        // into smaller subproblems and solve iteratively

        if self.verbose {
            println!("Using chunked approach for large problem: {n_vars} variables");
        }

        // Maximum number of variables to process in a single chunk
        const MAX_CHUNK_SIZE: usize = 1024;

        // Calculate number of chunks needed
        let num_chunks = n_vars.div_ceil(MAX_CHUNK_SIZE);

        if self.verbose {
            println!(
                "Processing in {num_chunks} chunks of at most {MAX_CHUNK_SIZE} variables each"
            );
        }

        // Initialize random number generator
        let mut rng = if let Some(seed) = self.seed {
            StdRng::seed_from_u64(seed)
        } else {
            let seed: u64 = thread_rng().gen();
            StdRng::seed_from_u64(seed)
        };

        // Initialize random solutions for all shots
        let mut solutions: Vec<Vec<bool>> = Vec::with_capacity(num_shots);
        for _ in 0..num_shots {
            let mut solution = Vec::with_capacity(n_vars);
            for _ in 0..n_vars {
                solution.push(rng.gen_bool(0.5));
            }
            solutions.push(solution);
        }

        // Track energies for each solution
        let mut energies = vec![0.0; num_shots];

        // Initialize energies
        for (i, solution) in solutions.iter().enumerate() {
            energies[i] = evaluate_qubo_energy(solution, h_vector, j_matrix, n_vars);
        }

        // Process each chunk iteratively
        for chunk_idx in 0..num_chunks {
            // Calculate start and end indices for this chunk
            let start_var = chunk_idx * MAX_CHUNK_SIZE;
            let end_var = std::cmp::min((chunk_idx + 1) * MAX_CHUNK_SIZE, n_vars);
            let chunk_size = end_var - start_var;

            if self.verbose {
                println!(
                    "Processing chunk {}/{}: variables {}..{}",
                    chunk_idx + 1,
                    num_chunks,
                    start_var,
                    end_var - 1
                );
            }

            // Extract subproblem
            let mut chunk_h = Vec::with_capacity(chunk_size);
            let mut chunk_j = Vec::with_capacity(chunk_size * chunk_size);

            // Extract linear terms for this chunk
            for i in start_var..end_var {
                chunk_h.push(h_vector[i]);
            }

            // Extract quadratic terms for this chunk
            for i in start_var..end_var {
                for j in start_var..end_var {
                    chunk_j.push(j_matrix[i * n_vars + j]);
                }
            }

            // Adjust linear terms based on fixed variables outside this chunk
            for sol_idx in 0..solutions.len() {
                let mut adjusted_h = chunk_h.clone();

                // Add contributions from fixed variables
                for i in start_var..end_var {
                    for j in 0..n_vars {
                        if (j < start_var || j >= end_var) && solutions[sol_idx][j] {
                            adjusted_h[i - start_var] += j_matrix[i * n_vars + j];
                        }
                    }
                }

                // Process this specific solution's subproblem
                let mut chunk_solution = Vec::with_capacity(chunk_size);
                for i in start_var..end_var {
                    chunk_solution.push(solutions[sol_idx][i]);
                }

                // Optimize just this chunk using GPU
                let optimized_chunk = self.optimize_chunk(
                    &chunk_solution,
                    &adjusted_h,
                    &chunk_j,
                    chunk_size,
                    self.seed.map(|s| s + sol_idx as u64),
                )?;

                // Update the original solution with optimized chunk
                for (i, &val) in optimized_chunk.iter().enumerate() {
                    solutions[sol_idx][start_var + i] = val;
                }

                // Update energy
                energies[sol_idx] =
                    evaluate_qubo_energy(&solutions[sol_idx], h_vector, j_matrix, n_vars);
            }
        }

        // Sort solutions by energy
        let mut solution_pairs: Vec<(Vec<bool>, f64)> =
            solutions.into_iter().zip(energies).collect();
        solution_pairs.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());

        // Return sorted solutions
        Ok(solution_pairs.into_iter().map(|(sol, _)| sol).collect())
    }

    /// Optimize a single chunk of variables
    #[cfg(all(feature = "gpu", feature = "dwave"))]
    fn optimize_chunk(
        &self,
        initial_state: &[bool],
        h_vector: &[f64],
        j_matrix: &[f64],
        n_vars: usize,
        seed: Option<u64>,
    ) -> Result<Vec<bool>, ocl::Error> {
        use ocl::flags;
        use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue};

        // Set up OpenCL environment (same as in run_gpu_annealing)
        let platform = Platform::default();
        let device = Device::first(platform)?;
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()?;
        let queue = Queue::new(&context, device, None)?;

        // Use chunk optimization kernel from kernels module
        let src = super::kernels::OPTIMIZE_CHUNK_KERNEL;

        // Compile the program
        let program = Program::builder()
            .devices(device)
            .src(src)
            .build(&context)?;

        // Set up buffers
        let h_buffer = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .len(n_vars)
            .build()?;

        let j_buffer = Buffer::<f32>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .len(n_vars * n_vars)
            .build()?;

        let initial_buffer = Buffer::<u8>::builder()
            .queue(queue.clone())
            .flags(flags::MEM_READ_ONLY)
            .len(n_vars)
            .build()?;

        let result_buffer = Buffer::<u8>::builder()
            .queue(queue)
            .flags(flags::MEM_WRITE_ONLY)
            .len(n_vars)
            .build()?;

        // Convert data types
        let h_vec_f32: Vec<f32> = h_vector.iter().map(|&x| x as f32).collect();
        let j_mat_f32: Vec<f32> = j_matrix.iter().map(|&x| x as f32).collect();
        let initial_u8: Vec<u8> = initial_state.iter().map(|&b| u8::from(b)).collect();

        // Transfer data to device
        h_buffer.write(&h_vec_f32).enq()?;
        j_buffer.write(&j_mat_f32).enq()?;
        initial_buffer.write(&initial_u8).enq()?;

        // Set kernel parameters
        let mut kernel = Kernel::builder()
            .program(&program)
            .name("optimize_chunk")
            .global_work_size(1) // Only one optimization task
            .arg(n_vars as i32)
            .arg(&h_buffer)
            .arg(&j_buffer)
            .arg(&initial_buffer)
            .arg(&result_buffer)
            .arg(5000i32) // More sweeps for thorough optimization of a chunk
            .arg(5.0f32)  // Higher initial temperature
            .arg(0.01f32) // Lower final temperature
            .arg(seed.unwrap_or_else(|| thread_rng().gen()))
            .build()?;

        // Execute kernel
        unsafe {
            kernel.enq()?;
        }

        // Read result
        let mut result_u8 = vec![0u8; n_vars];
        result_buffer.read(&mut result_u8).enq()?;

        // Convert back to bool
        let mut result = result_u8.iter().map(|&b| b != 0).collect();

        Ok(result)
    }

    #[cfg(not(all(feature = "gpu", feature = "dwave")))]
    fn run_gpu_annealing(
        &self,
        _n_vars: usize,
        _h_vector: &[f64],
        _j_matrix: &[f64],
        _num_shots: usize,
    ) -> Result<Vec<Vec<bool>>, String> {
        Err("GPU support not enabled. Rebuild with '--features gpu,dwave'".to_string())
    }

    #[cfg(not(all(feature = "gpu", feature = "dwave")))]
    fn run_gpu_annealing_chunked(
        &self,
        _n_vars: usize,
        _h_vector: &[f64],
        _j_matrix: &[f64],
        _num_shots: usize,
    ) -> Result<Vec<Vec<bool>>, String> {
        Err("GPU support not enabled. Rebuild with '--features gpu,dwave'".to_string())
    }

    #[cfg(not(all(feature = "gpu", feature = "dwave")))]
    fn optimize_chunk(
        &self,
        _initial_state: &[bool],
        _h_vector: &[f64],
        _j_matrix: &[f64],
        _n_vars: usize,
        _seed: Option<u64>,
    ) -> Result<Vec<bool>, String> {
        Err("GPU support not enabled. Rebuild with '--features gpu,dwave'".to_string())
    }
}

#[cfg(feature = "gpu")]
impl Sampler for ArminSampler {
    fn run_qubo(
        &self,
        qubo: &(Array<f64, Ix2>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Extract matrix and variable mapping
        let (matrix, var_map) = qubo;

        // Get the problem dimension
        let n_vars = var_map.len();

        // Map from indices back to variable names
        let idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

        // Determine compute resources based on mode
        let is_gpu = self.mode.to_uppercase() == "GPU";
        let device_info = if is_gpu {
            format!("Using GPU device: {}", self.device)
        } else {
            "Using CPU acceleration".to_string()
        };

        if self.verbose {
            println!("{device_info}");
            println!("Problem size: {n_vars} variables");
        }

        // Convert QUBO matrix to appropriate format for OpenCL
        let mut h_vector = Vec::with_capacity(n_vars);
        let mut j_matrix = Vec::with_capacity(n_vars * n_vars);

        // Extract diagonal (linear) terms
        for i in 0..n_vars {
            h_vector.push(matrix[[i, i]]);
        }

        // Extract off-diagonal (quadratic) terms
        for i in 0..n_vars {
            for j in 0..n_vars {
                if i == j {
                    j_matrix.push(0.0); // Zero on diagonal in J matrix
                } else {
                    j_matrix.push(matrix[[i, j]]);
                }
            }
        }

        // Set up OpenCL and run GPU annealing
        #[cfg(all(feature = "gpu", feature = "dwave"))]
        let ocl_result = self.run_gpu_annealing(n_vars, &h_vector, &j_matrix, shots);

        #[cfg(not(all(feature = "gpu", feature = "dwave")))]
        let ocl_result: Result<Vec<Vec<i32>>, SamplerError> = Err(SamplerError::GpuError(
            "GPU support not enabled".to_string(),
        ));

        #[cfg(all(feature = "gpu", feature = "dwave"))]
        match ocl_result {
            Ok(binary_solutions) => {
                // Process results
                let mut solution_counts: HashMap<Vec<bool>, (f64, usize)> = HashMap::new();

                for solution in binary_solutions {
                    // Calculate energy using our helper function
                    let mut energy = evaluate_qubo_energy(&solution, &h_vector, &j_matrix, n_vars);

                    // Update solution counts
                    let entry = solution_counts.entry(solution).or_insert((energy, 0));
                    entry.1 += 1;
                }

                // Convert to SampleResult format
                let mut results: Vec<SampleResult> = solution_counts
                    .into_iter()
                    .map(|(bin_solution, (energy, count))| {
                        // Convert to variable assignments
                        let assignments: HashMap<String, bool> = bin_solution
                            .iter()
                            .enumerate()
                            .map(|(idx, &value)| {
                                let var_name = idx_to_var.get(&idx).unwrap().clone();
                                (var_name, value)
                            })
                            .collect();

                        SampleResult {
                            assignments,
                            energy,
                            occurrences: count,
                        }
                    })
                    .collect();

                // Sort by energy
                results.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

                // Limit to requested number of shots
                if results.len() > shots {
                    results.truncate(shots);
                }

                Ok(results)
            }
            Err(e) => Err(SamplerError::GpuError(e.to_string())),
        }

        #[cfg(not(all(feature = "gpu", feature = "dwave")))]
        match ocl_result {
            Ok(_) => unreachable!("GPU support not enabled"),
            Err(e) => Err(e),
        }
    }

    fn run_hobo(
        &self,
        hobo: &(
            Array<f64, scirs2_core::ndarray::IxDyn>,
            HashMap<String, usize>,
        ),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        // Handle QUBO case directly
        if hobo.0.ndim() == 2 {
            let mut qubo = (
                hobo.0
                    .clone()
                    .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                    .unwrap(),
                hobo.1.clone(),
            );
            return self.run_qubo(&qubo, shots);
        }

        // For higher-order problems, we could implement specialized kernels
        // For now, return an error suggesting quadratization
        Err(SamplerError::InvalidParameter(
            "GPU acceleration for HOBO problems not yet implemented. Consider quadratization to QUBO format.".to_string()
        ))
    }
}

// Fallback implementation when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
pub struct ArminSampler {
    _seed: Option<u64>,
}

#[cfg(not(feature = "gpu"))]
impl ArminSampler {
    #[must_use]
    pub fn new(_seed: Option<u64>) -> Self {
        Self { _seed }
    }

    #[must_use]
    pub fn with_params(_seed: Option<u64>, _mode: &str, _device: &str, _verbose: bool) -> Self {
        Self { _seed }
    }
}

#[cfg(not(feature = "gpu"))]
impl Sampler for ArminSampler {
    fn run_qubo(
        &self,
        _qubo: &(Array<f64, Ix2>, HashMap<String, usize>),
        _shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        Err(SamplerError::GpuError(
            "GPU support not enabled. Rebuild with '--features gpu'".to_string(),
        ))
    }

    fn run_hobo(
        &self,
        _hobo: &(
            Array<f64, scirs2_core::ndarray::IxDyn>,
            HashMap<String, usize>,
        ),
        _shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        Err(SamplerError::GpuError(
            "GPU support not enabled. Rebuild with '--features gpu'".to_string(),
        ))
    }
}
