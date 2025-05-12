// Single qubit gate shader for quantum simulation
// Computes the effect of a single-qubit gate on the state vector

// Define the buffer structures
struct StateVector {
    data: array<vec2<f32>>, // Array of complex numbers (real, imag)
}

struct GateParams {
    target_qubit: u32,
    n_qubits: u32,
    matrix: array<vec2<f32>, 4>, // 2x2 complex matrix as flattened array
    padding: vec2<u32>,         // Padding for alignment
}

// Binding group
@group(0) @binding(0) var<storage, read_write> state_vector: StateVector;
@group(0) @binding(1) var<uniform> params: GateParams;

// Helper function to multiply complex numbers
fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

// The main compute shader function
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let state_size = 1u << params.n_qubits;
    let idx = global_id.x;
    
    // Return if the index is out of bounds
    if (idx >= state_size) {
        return;
    }
    
    let target = params.target_qubit;
    let mask = 1u << target;
    
    // Determine if this index has the target qubit set to 0 or 1
    let bit_is_set = (idx & mask) != 0u;
    
    // Compute the paired index (flipping the target bit)
    let paired_idx = idx ^ mask;
    
    // Only process the lower indices to avoid race conditions
    if (idx < paired_idx) {
        // Get the current amplitudes
        let amp0 = state_vector.data[idx];
        let amp1 = state_vector.data[paired_idx];
        
        // Matrix operation: [a b; c d] * [amp0; amp1]
        let a = params.matrix[0]; // [0, 0]
        let b = params.matrix[1]; // [0, 1]
        let c = params.matrix[2]; // [1, 0]
        let d = params.matrix[3]; // [1, 1]
        
        // Calculate new amplitudes
        let new_amp0 = complex_mul(a, amp0) + complex_mul(b, amp1);
        let new_amp1 = complex_mul(c, amp0) + complex_mul(d, amp1);
        
        // Update the state vector
        state_vector.data[idx] = new_amp0;
        state_vector.data[paired_idx] = new_amp1;
    }
}