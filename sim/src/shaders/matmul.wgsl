// Matrix multiplication shader for complex numbers

struct Complex {
    real: f32,
    imag: f32,
}

struct Dimensions {
    m: u32,  // rows of A, rows of result
    n: u32,  // cols of B, cols of result
    k: u32,  // cols of A, rows of B
    pad: u32,
}

@group(0) @binding(0) var<storage, read> matrix_a: array<Complex>;
@group(0) @binding(1) var<storage, read> matrix_b: array<Complex>;
@group(0) @binding(2) var<storage, read_write> result: array<Complex>;
@group(0) @binding(3) var<uniform> dims: Dimensions;

// Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
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

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    
    // Check bounds
    if (row >= dims.m || col >= dims.n) {
        return;
    }
    
    // Compute dot product of row from A and column from B
    var sum = Complex(0.0, 0.0);
    
    for (var i = 0u; i < dims.k; i = i + 1u) {
        let a_idx = row * dims.k + i;
        let b_idx = i * dims.n + col;
        
        let a_val = matrix_a[a_idx];
        let b_val = matrix_b[b_idx];
        
        sum = complex_add(sum, complex_mul(a_val, b_val));
    }
    
    // Store result
    let result_idx = row * dims.n + col;
    result[result_idx] = sum;
}