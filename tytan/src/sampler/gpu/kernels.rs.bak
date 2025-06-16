//! OpenCL kernels for GPU-accelerated sampling

/// OpenCL kernel for simulated annealing
pub const SIMULATED_ANNEALING_KERNEL: &str = r#"
// Helper function for xorshift RNG
inline ulong xorshift64(ulong *state) {
    ulong x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

// Fast random float generation
inline float random_float(ulong *state) {
    ulong x = xorshift64(state);
    return (float)(x & 0xFFFFFFFF) / (float)0xFFFFFFFF;
}

// Simulated annealing kernel
__kernel void simulated_annealing(
    const int n_vars,
    __global const float* h_vector,
    __global const float* j_matrix,
    __global uchar* solutions,
    const int num_runs,
    const float init_temp,
    const float final_temp,
    const int sweeps,
    const ulong seed
) {
    // Get global ID
    int gid = get_global_id(0);
    if (gid >= num_runs) return;

    // Initialize RNG for this thread
    ulong rng_state = seed + gid;

    // Initialize spin state with local storage for better performance
    uchar state[2048]; // Max vars supported
    float best_energy = 0.0f;
    float current_energy = 0.0f;
    uchar best_state[2048];

    // Random initialization
    for (int i = 0; i < n_vars; i++) {
        state[i] = (xorshift64(&rng_state) & 1) ? 1 : 0;
    }

    // Calculate initial energy
    for (int i = 0; i < n_vars; i++) {
        if (state[i]) {
            current_energy += h_vector[i];

            for (int j = 0; j < n_vars; j++) {
                if (j != i && state[j]) {
                    current_energy += j_matrix[i * n_vars + j];
                }
            }
        }
    }

    // Initialize best solution
    best_energy = current_energy;
    for (int i = 0; i < n_vars; i++) {
        best_state[i] = state[i];
    }

    // Annealing process
    for (int sweep = 0; sweep < sweeps; sweep++) {
        // Calculate temperature for this sweep using exponential schedule
        float t_ratio = (float)sweep / (float)sweeps;
        float temp = init_temp * pow(final_temp / init_temp, t_ratio);

        // Perform n_vars spin flips per sweep
        for (int flip = 0; flip < n_vars; flip++) {
            // Choose random spin to flip using efficient sampling
            int idx = xorshift64(&rng_state) % n_vars;

            // Calculate energy change efficiently
            float delta_e = 0.0f;

            // Contribution from h (linear terms)
            delta_e += h_vector[idx] * (state[idx] ? -2.0f : 2.0f);

            // Contribution from J (quadratic terms)
            for (int j = 0; j < n_vars; j++) {
                if (state[j]) {
                    delta_e += j_matrix[idx * n_vars + j] * (state[idx] ? -2.0f : 2.0f);
                }
            }

            // Metropolis acceptance criterion with optimized branching
            bool accept = (delta_e <= 0.0f) || (random_float(&rng_state) < exp(-delta_e / temp));

            // Update state and energy if accepted
            if (accept) {
                state[idx] = !state[idx];
                current_energy += delta_e;

                // Update best solution if improved
                if (current_energy < best_energy) {
                    best_energy = current_energy;
                    for (int i = 0; i < n_vars; i++) {
                        best_state[i] = state[i];
                    }
                }
            }
        }

        // Periodic check if we've converged to a stable solution
        if (sweep % 50 == 0 && sweep > 0) {
            // If temperature is very low and we've reached a good solution, we can terminate early
            if (temp < 0.01f * final_temp && sweep > (sweeps / 2)) {
                break;
            }
        }
    }

    // Write best solution to global memory
    for (int i = 0; i < n_vars; i++) {
        solutions[gid * n_vars + i] = best_state[i];
    }
}
"#;

/// OpenCL kernel for chunk optimization
pub const OPTIMIZE_CHUNK_KERNEL: &str = r#"
__kernel void optimize_chunk(
    const int n_vars,
    __global const float* h_vector,
    __global const float* j_matrix,
    __global uchar* initial_state,
    __global uchar* result_state,
    const int sweeps,
    const float init_temp,
    const float final_temp,
    const ulong seed
) {
    // Initialize RNG
    ulong rng_state = seed;

    // Copy initial state to local array
    uchar state[1024]; // Max chunk size
    for (int i = 0; i < n_vars; i++) {
        state[i] = initial_state[i];
    }

    // Calculate initial energy
    float energy = 0.0f;
    for (int i = 0; i < n_vars; i++) {
        if (state[i]) {
            energy += h_vector[i];

            for (int j = 0; j < n_vars; j++) {
                if (j != i && state[j]) {
                    energy += j_matrix[i * n_vars + j];
                }
            }
        }
    }

    // Track best solution
    float best_energy = energy;
    uchar best_state[1024];
    for (int i = 0; i < n_vars; i++) {
        best_state[i] = state[i];
    }

    // Annealing process
    for (int sweep = 0; sweep < sweeps; sweep++) {
        // Calculate temperature
        float t_ratio = (float)sweep / (float)sweeps;
        float temp = init_temp * pow(final_temp / init_temp, t_ratio);

        // Monte Carlo steps
        for (int i = 0; i < n_vars; i++) {
            // Choose variable to flip
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            int idx = rng_state % n_vars;

            // Calculate energy change
            float delta_e = 0.0f;
            delta_e += h_vector[idx] * (state[idx] ? -2.0f : 2.0f);

            for (int j = 0; j < n_vars; j++) {
                if (state[j]) {
                    delta_e += j_matrix[idx * n_vars + j] * (state[idx] ? -2.0f : 2.0f);
                }
            }

            // Metropolis acceptance criterion
            bool accept = false;
            if (delta_e <= 0.0f) {
                accept = true;
            } else {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                float rand_val = (float)(rng_state & 0xFFFFFFFF) / (float)0xFFFFFFFF;
                accept = rand_val < exp(-delta_e / temp);
            }

            // Apply change if accepted
            if (accept) {
                state[idx] = !state[idx];
                energy += delta_e;

                // Update best solution if improved
                if (energy < best_energy) {
                    best_energy = energy;
                    for (int j = 0; j < n_vars; j++) {
                        best_state[j] = state[j];
                    }
                }
            }
        }
    }

    // Write best solution back
    for (int i = 0; i < n_vars; i++) {
        result_state[i] = best_state[i];
    }
}
"#;
