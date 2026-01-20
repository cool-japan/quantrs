// Tensor product shader for complex matrices

struct Complex {
    real: f32,
    imag: f32,
}

struct Dimensions {
    m1: u32,  // rows of first matrix
    n1: u32,  // cols of first matrix
    m2: u32,  // rows of second matrix
    n2: u32,  // cols of second matrix
}

@group(0) @binding(0) var<storage, read> matrix_a: array<Complex>;
@group(0) @binding(1) var<storage, read> matrix_b: array<Complex>;
@group(0) @binding(2) var<storage, read_write> result: array<Complex>;
@group(0) @binding(3) var<uniform> dims: Dimensions;

// Complex multiplication
fn complex_mul(a: Complex, b: Complex) -> Complex {
    return Complex(
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    );
}

@compute @workgroup_size(16, 16)
fn tensor_product(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    
    let total_rows = dims.m1 * dims.m2;
    let total_cols = dims.n1 * dims.n2;
    
    // Check bounds
    if (row >= total_rows || col >= total_cols) {
        return;
    }
    
    // Decompose indices
    let i1 = row / dims.m2;
    let i2 = row % dims.m2;
    let j1 = col / dims.n2;
    let j2 = col % dims.n2;
    
    // Get values from matrices
    let a_idx = i1 * dims.n1 + j1;
    let b_idx = i2 * dims.n2 + j2;
    
    let a_val = matrix_a[a_idx];
    let b_val = matrix_b[b_idx];
    
    // Compute tensor product element
    let result_idx = row * total_cols + col;
    result[result_idx] = complex_mul(a_val, b_val);
}