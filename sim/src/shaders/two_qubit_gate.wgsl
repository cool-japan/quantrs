// Two qubit gate shader for quantum simulation
// Computes the effect of a two-qubit gate on the state vector

// Define the buffer structures
struct StateVector {
    data: array<vec2<f32>>, // Array of complex numbers (real, imag)
}

struct GateParams {
    control_qubit: u32,
    target_qubit: u32,
    n_qubits: u32,
    matrix: array<vec2<f32>, 16>, // 4x4 complex matrix as flattened array
    padding: u32,
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
    
    let control = params.control_qubit;
    let target = params.target_qubit;
    
    let control_mask = 1u << control;
    let target_mask = 1u << target;
    
    // We only process if the control qubit mask is set (for controlled gates)
    // For non-controlled gates, we would remove this check
    let control_bit_set = (idx & control_mask) != 0u;
    if (!control_bit_set) {
        return;
    }
    
    // Determine the four indices that form our 2-qubit basis
    let idx_00 = idx & ~(control_mask | target_mask); // both bits cleared
    let idx_01 = idx_00 | target_mask;                // only target bit set
    let idx_10 = idx_00 | control_mask;               // only control bit set
    let idx_11 = idx_00 | control_mask | target_mask; // both bits set
    
    // Only process one of the four indices to avoid race conditions
    if (idx == idx_00) {
        // Get the current amplitudes
        let amp00 = state_vector.data[idx_00];
        let amp01 = state_vector.data[idx_01];
        let amp10 = state_vector.data[idx_10];
        let amp11 = state_vector.data[idx_11];
        
        // Create a vector of the amplitudes [amp00, amp01, amp10, amp11]
        let amps = array<vec2<f32>, 4>(amp00, amp01, amp10, amp11);
        
        // Apply 4x4 matrix: Result = M * [amp00, amp01, amp10, amp11]^T
        var new_amps: array<vec2<f32>, 4>;
        
        for (var i = 0u; i < 4u; i = i + 1u) {
            new_amps[i] = vec2<f32>(0.0, 0.0);
            for (var j = 0u; j < 4u; j = j + 1u) {
                new_amps[i] = new_amps[i] + complex_mul(params.matrix[i * 4u + j], amps[j]);
            }
        }
        
        // Update the state vector
        state_vector.data[idx_00] = new_amps[0];
        state_vector.data[idx_01] = new_amps[1];
        state_vector.data[idx_10] = new_amps[2];
        state_vector.data[idx_11] = new_amps[3];
    }
}