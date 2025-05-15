// Two-qubit gate shader for quantum simulation

// Complex number structure
struct Complex {
    real: f32,
    imag: f32,
}

// Gate parameters
struct Params {
    control_qubit: u32,
    target_qubit: u32,
    n_qubits: u32,
    matrix: array<Complex, 16>, // 4x4 matrix in row-major format
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

// Apply a two-qubit gate to the state vector
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let dim = 1u << params.n_qubits;
    
    // Check if we are within bounds
    if (idx >= dim) {
        return;
    }
    
    // Get the masks for control and target qubits
    let control_mask = 1u << params.control_qubit;
    let target_mask = 1u << params.target_qubit;
    
    // Get the bit values for this index
    let control_bit = (idx & control_mask) >> params.control_qubit;
    let target_bit = (idx & target_mask) >> params.target_qubit;
    
    // Find the indices for the four states we need to consider (00, 01, 10, 11)
    let idx_mask = control_mask | target_mask;
    let base_idx = idx & ~idx_mask; // Clear control and target bits
    
    let idx00 = base_idx;
    let idx01 = base_idx | target_mask;
    let idx10 = base_idx | control_mask;
    let idx11 = base_idx | control_mask | target_mask;
    
    // Only process once for each set of indices
    if (idx \!= idx00) {
        return;
    }
    
    // Get the four amplitudes from the state vector
    let val00 = state_vector[idx00];
    let val01 = state_vector[idx01];
    let val10 = state_vector[idx10];
    let val11 = state_vector[idx11];
    
    // Apply the 4x4 matrix
    // [ m00 m01 m02 m03 ] [ val00 ]
    // [ m10 m11 m12 m13 ] [ val01 ]
    // [ m20 m21 m22 m23 ] [ val10 ]
    // [ m30 m31 m32 m33 ] [ val11 ]
    
    let matrix = params.matrix;
    
    // Calculate new values
    let new_val00 = complex_add(
        complex_add(
            complex_mul(matrix[0], val00),
            complex_mul(matrix[1], val01)
        ),
        complex_add(
            complex_mul(matrix[2], val10),
            complex_mul(matrix[3], val11)
        )
    );
    
    let new_val01 = complex_add(
        complex_add(
            complex_mul(matrix[4], val00),
            complex_mul(matrix[5], val01)
        ),
        complex_add(
            complex_mul(matrix[6], val10),
            complex_mul(matrix[7], val11)
        )
    );
    
    let new_val10 = complex_add(
        complex_add(
            complex_mul(matrix[8], val00),
            complex_mul(matrix[9], val01)
        ),
        complex_add(
            complex_mul(matrix[10], val10),
            complex_mul(matrix[11], val11)
        )
    );
    
    let new_val11 = complex_add(
        complex_add(
            complex_mul(matrix[12], val00),
            complex_mul(matrix[13], val01)
        ),
        complex_add(
            complex_mul(matrix[14], val10),
            complex_mul(matrix[15], val11)
        )
    );
    
    // Update the state vector
    state_vector[idx00] = new_val00;
    state_vector[idx01] = new_val01;
    state_vector[idx10] = new_val10;
    state_vector[idx11] = new_val11;
}
