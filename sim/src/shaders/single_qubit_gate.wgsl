// Single-qubit gate shader for quantum simulation

// Complex number structure
struct Complex {
    real: f32,
    imag: f32,
}

// Gate parameters
struct Params {
    target_qubit: u32,
    n_qubits: u32,
    matrix: array<Complex, 4>, // 2x2 matrix in row-major format
}

// State vector storage buffer
@group(0) @binding(0)
var<storage, read_write> state_vector: array<Complex>;

// Uniform buffer for gate parameters
@group(0) @binding(1)
var<uniform> params: Params;

// Complex number operations
fn complex_add(a: Complex, b: Complex) -> Complex {
    return Complex(a.real + b.real, a.imag + b.imag);
}

fn complex_mul(a: Complex, b: Complex) -> Complex {
    return Complex(
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    );
}

// Apply a single-qubit gate to the state vector
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let dim = 1u << params.n_qubits;
    
    // Check if we are within bounds
    if (idx >= dim) {
        return;
    }
    
    // Determine if this index corresponds to target qubit = 0 or 1
    let mask = 1u << params.target_qubit;
    let paired_idx = idx ^ mask; // Flip the target qubit bit
    
    // Skip half of the indices to avoid double computation
    if (idx > paired_idx) {
        return;
    }
    
    // Get the current state values
    let val0 = state_vector[idx];
    let val1 = state_vector[paired_idx];
    
    // Apply the 2x2 matrix
    // [ matrix[0,0] matrix[0,1] ] [ val0 ]
    // [ matrix[1,0] matrix[1,1] ] [ val1 ]
    
    // New values after matrix multiplication
    let new_val0 = complex_add(
        complex_mul(params.matrix[0], val0),
        complex_mul(params.matrix[1], val1)
    );
    
    let new_val1 = complex_add(
        complex_mul(params.matrix[2], val0),
        complex_mul(params.matrix[3], val1)
    );
    
    // Update the state vector
    state_vector[idx] = new_val0;
    state_vector[paired_idx] = new_val1;
}
