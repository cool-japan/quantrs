// Pauli-X gate CUDA kernel
__global__ void pauli_x_kernel(cuFloatComplex* state, int qubit, int n_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_states) return;
    
    int mask = 1 << qubit;
    if ((idx & mask) == 0) {
        int partner = idx | mask;
        if (partner < n_states) {
            // Swap amplitudes: X|0⟩ = |1⟩, X|1⟩ = |0⟩
            cuFloatComplex temp = state[idx];
            state[idx] = state[partner];
            state[partner] = temp;
        }
    }
}