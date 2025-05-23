// Apply unitary matrix to quantum state vector

struct Complex {
    real: f32,
    imag: f32,
}

struct UniformData {
    num_qubits: u32,
    target_qubit: u32,
    matrix_size: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read_write> state: array<Complex>;
@group(0) @binding(1) var<storage, read> unitary: array<Complex>;
@group(0) @binding(2) var<uniform> params: UniformData;

// Complex multiplication
fn complex_mul(a: Complex, b: Complex) -> Complex {
    return Complex(
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    );
}

// Complex addition
fn complex_add(a: Complex, b: Complex) -> Complex {
    return Complex(a.real + b.real, a.imag + b.imag);
}

@compute @workgroup_size(256)
fn apply_unitary(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let state_idx = global_id.x;
    let state_size = 1u << params.num_qubits;
    
    if (state_idx >= state_size) {
        return;
    }
    
    // This is a simplified version for single-qubit gates
    // Full implementation would handle multi-qubit unitaries
    
    let qubit_mask = 1u << params.target_qubit;
    let is_one = (state_idx & qubit_mask) != 0u;
    
    if (!is_one) {
        // This state index has target qubit = 0
        let partner_idx = state_idx | qubit_mask;
        
        if (partner_idx < state_size) {
            // Apply 2x2 unitary
            let s0 = state[state_idx];
            let s1 = state[partner_idx];
            
            // New amplitudes after applying unitary
            let new_s0 = complex_add(
                complex_mul(unitary[0], s0),
                complex_mul(unitary[1], s1)
            );
            let new_s1 = complex_add(
                complex_mul(unitary[2], s0),
                complex_mul(unitary[3], s1)
            );
            
            state[state_idx] = new_s0;
            state[partner_idx] = new_s1;
        }
    }
}