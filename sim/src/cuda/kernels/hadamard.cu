// Hadamard gate CUDA kernel
__global__ void hadamard_kernel(cuFloatComplex* state, int qubit, int n_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_states) return;
    
    int mask = 1 << qubit;
    if ((idx & mask) == 0) {
        int partner = idx | mask;
        if (partner < n_states) {
            cuFloatComplex a = state[idx];
            cuFloatComplex b = state[partner];
            
            // H|0⟩ = (|0⟩ + |1⟩)/√2, H|1⟩ = (|0⟩ - |1⟩)/√2
            float sqrt2_inv = 0.7071067811865475f; // 1/√2
            state[idx] = make_cuFloatComplex(
                sqrt2_inv * (cuCrealf(a) + cuCrealf(b)),
                sqrt2_inv * (cuCimagf(a) + cuCimagf(b))
            );
            state[partner] = make_cuFloatComplex(
                sqrt2_inv * (cuCrealf(a) - cuCrealf(b)),
                sqrt2_inv * (cuCimagf(a) - cuCimagf(b))
            );
        }
    }
}