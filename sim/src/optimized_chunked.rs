//! Optimized quantum state vector simulation using chunked memory processing
//!
//! This module provides a memory-efficient implementation for large qubit counts (30+)
//! by processing the state vector in manageable chunks to reduce memory pressure.

use num_complex::Complex64;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::cmp::min;

// Use standard memory management since scirs2 memory module is not available
// Placeholder for future integration with scirs2
#[derive(Clone, Debug)]
struct MemoryChunk<T> {
    data: Vec<T>,
    capacity: usize,
}

impl<T: Clone + Default> MemoryChunk<T> {
    fn new(capacity: usize) -> Self {
        Self {
            data: vec![T::default(); capacity],
            capacity,
        }
    }
    
    fn get(&self, idx: usize) -> Option<&T> {
        self.data.get(idx)
    }
    
    fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
        self.data.get_mut(idx)
    }
    
    fn as_slice(&self) -> &[T] {
        &self.data
    }
    
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

use crate::utils::flip_bit;

/// Size of chunks in elements for large state vector processing
const DEFAULT_CHUNK_SIZE: usize = 1 << 20; // 1 million complex numbers per chunk (~16 MB)

/// Represents a quantum state vector that uses chunked memory for large qubit counts
pub struct ChunkedStateVector {
    /// The full state vector stored as multiple chunks
    chunks: Vec<MemoryChunk<Complex64>>,
    /// Number of qubits represented
    num_qubits: usize,
    /// Size of each chunk (number of complex numbers)
    chunk_size: usize,
    /// Total dimension of the state vector (2^num_qubits)
    dimension: usize,
}

impl ChunkedStateVector {
    /// Create a new chunked state vector for given number of qubits
    pub fn new(num_qubits: usize) -> Self {
        let dimension = 1 << num_qubits;
        let chunk_size = min(DEFAULT_CHUNK_SIZE, dimension);
        let num_chunks = (dimension + chunk_size - 1) / chunk_size;
        
        // Create empty chunks
        let mut chunks = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let this_chunk_size = if i == num_chunks - 1 && dimension % chunk_size != 0 {
                dimension % chunk_size
            } else {
                chunk_size
            };
            
            let mut chunk = MemoryChunk::new(this_chunk_size);
            if i == 0 {
                // Initialize to |0...0>
                if let Some(first) = chunk.get_mut(0) {
                    *first = Complex64::new(1.0, 0.0);
                }
            }
            chunks.push(chunk);
        }
        
        Self {
            chunks,
            num_qubits,
            chunk_size,
            dimension,
        }
    }
    
    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
    
    /// Get the dimension of the state vector
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    
    /// Access a specific amplitude by global index
    pub fn get_amplitude(&self, idx: usize) -> Complex64 {
        let chunk_idx = idx / self.chunk_size;
        let local_idx = idx % self.chunk_size;
        
        if chunk_idx >= self.chunks.len() {
            return Complex64::new(0.0, 0.0);
        }
        
        match self.chunks[chunk_idx].get(local_idx) {
            Some(val) => *val,
            None => Complex64::new(0.0, 0.0),
        }
    }
    
    /// Get all amplitudes as a flattened vector (for testing and conversion)
    /// Warning: For large qubit counts, this will use a lot of memory
    pub fn as_vec(&self) -> Vec<Complex64> {
        let mut result = Vec::with_capacity(self.dimension);
        for chunk in &self.chunks {
            result.extend_from_slice(chunk.as_slice());
        }
        result
    }
    
    /// Apply a single-qubit gate to the state vector using chunked processing
    ///
    /// # Arguments
    ///
    /// * `matrix` - The 2x2 matrix representation of the gate
    /// * `target` - The target qubit index
    pub fn apply_single_qubit_gate(&mut self, matrix: &[Complex64], target: usize) {
        if target >= self.num_qubits {
            panic!("Target qubit index out of range");
        }

        // Copy current state as we need to read from old state while writing to new
        let old_chunks = self.chunks.clone();

        // Reset all values to zero
        for chunk in &mut self.chunks {
            for idx in 0..chunk.as_slice().len() {
                if let Some(val) = chunk.get_mut(idx) {
                    *val = Complex64::new(0.0, 0.0);
                }
            }
        }

        // Process each chunk - iterate through old chunks for reading
        for (chunk_idx, chunk) in old_chunks.iter().enumerate() {
            let base_idx = chunk_idx * self.chunk_size;

            // Process each amplitude in this chunk
            for (local_idx, &amp) in chunk.as_slice().iter().enumerate() {
                let global_idx = base_idx + local_idx;
                if global_idx >= self.dimension {
                    break;
                }

                // Skip over zero amplitudes for efficiency
                if amp == Complex64::new(0.0, 0.0) {
                    continue;
                }

                let bit_val = (global_idx >> target) & 1;

                // Find the paired index
                let paired_global_idx = flip_bit(global_idx, target);
                let paired_chunk_idx = paired_global_idx / self.chunk_size;
                let paired_local_idx = paired_global_idx % self.chunk_size;

                // Get the amplitude of the paired index from old state
                let paired_amp = if paired_chunk_idx < old_chunks.len() {
                    if let Some(val) = old_chunks[paired_chunk_idx].get(paired_local_idx) {
                        *val
                    } else {
                        Complex64::new(0.0, 0.0)
                    }
                } else {
                    Complex64::new(0.0, 0.0)
                };

                // Calculate new amplitudes
                let new_amp0 = matrix[0] * amp + matrix[1] * paired_amp;
                let new_amp1 = matrix[2] * amp + matrix[3] * paired_amp;

                // Determine current chunk/idx from global index
                if bit_val == 0 {
                    // Update both indices in one go
                    if let Some(val) = self.chunks[chunk_idx].get_mut(local_idx) {
                        *val += new_amp0;
                    }

                    if paired_chunk_idx < self.chunks.len() {
                        if let Some(val) = self.chunks[paired_chunk_idx].get_mut(paired_local_idx) {
                            *val += new_amp1;
                        }
                    }
                }
            }
        }
    }
    
    /// Apply a controlled-NOT gate to the state vector
    ///
    /// # Arguments
    ///
    /// * `control` - The control qubit index
    /// * `target` - The target qubit index
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        if control >= self.num_qubits || target >= self.num_qubits {
            panic!("Qubit indices out of range");
        }

        if control == target {
            panic!("Control and target qubits must be different");
        }

        // We're using standard qubit ordering where the target/control parameters
        // are used directly with bit operations

        // Create new chunks to hold the result
        let mut new_chunks = Vec::with_capacity(self.chunks.len());
        for chunk in &self.chunks {
            new_chunks.push(MemoryChunk::new(chunk.as_slice().len()));
        }
        
        // Process each chunk in parallel
        for (chunk_idx, chunk) in self.chunks.iter().enumerate() {
            let base_idx = chunk_idx * self.chunk_size;
            
            // Process this chunk
            for (local_idx, &amp) in chunk.as_slice().iter().enumerate() {
                let global_idx = base_idx + local_idx;
                if global_idx >= self.dimension {
                    break;
                }
                
                let control_bit = (global_idx >> control) & 1;

                if control_bit == 0 {
                    // Control bit is 0: state remains unchanged
                    if let Some(val) = new_chunks[chunk_idx].get_mut(local_idx) {
                        *val = amp;
                    }
                } else {
                    // Control bit is 1: flip the target bit
                    let flipped_idx = flip_bit(global_idx, target);
                    let flipped_chunk_idx = flipped_idx / self.chunk_size;
                    let flipped_local_idx = flipped_idx % self.chunk_size;

                    // Get the amplitude from the flipped position
                    let flipped_amp = self.get_amplitude(flipped_idx);

                    // Update the current position with the flipped amplitude
                    if let Some(val) = new_chunks[chunk_idx].get_mut(local_idx) {
                        *val = flipped_amp;
                    }

                    // Update the flipped position with the current amplitude
                    if flipped_chunk_idx < self.chunks.len() {
                        if let Some(val) = new_chunks[flipped_chunk_idx].get_mut(flipped_local_idx) {
                            *val = amp;
                        }
                    }
                }
            }
        }
        
        // Update the state
        self.chunks = new_chunks;
    }
    
    /// Apply a two-qubit gate to the state vector
    ///
    /// # Arguments
    ///
    /// * `matrix` - The 4x4 matrix representation of the gate
    /// * `qubit1` - The first qubit index
    /// * `qubit2` - The second qubit index
    pub fn apply_two_qubit_gate(&mut self, matrix: &[Complex64], qubit1: usize, qubit2: usize) {
        if qubit1 >= self.num_qubits || qubit2 >= self.num_qubits {
            panic!("Qubit indices out of range");
        }
        
        if qubit1 == qubit2 {
            panic!("Qubit indices must be different");
        }
        
        // Create new chunks to hold the result
        let mut new_chunks = Vec::with_capacity(self.chunks.len());
        for chunk in &self.chunks {
            new_chunks.push(MemoryChunk::new(chunk.as_slice().len()));
        }
        
        // Process each chunk
        for (chunk_idx, chunk) in self.chunks.iter().enumerate() {
            let base_idx = chunk_idx * self.chunk_size;
            
            // Process this chunk
            for (local_idx, &_) in chunk.as_slice().iter().enumerate() {
                let global_idx = base_idx + local_idx;
                if global_idx >= self.dimension {
                    break;
                }
                
                // Determine which basis state this corresponds to in the 2-qubit subspace
                let bit1 = (global_idx >> qubit1) & 1;
                let bit2 = (global_idx >> qubit2) & 1;
                
                // Calculate the indices of all four basis states in the 2-qubit subspace
                let bits00 = global_idx & !(1 << qubit1) & !(1 << qubit2);
                let bits01 = bits00 | (1 << qubit2);
                let bits10 = bits00 | (1 << qubit1);
                let bits11 = bits10 | (1 << qubit2);
                
                // Get the amplitudes for all basis states
                let amp00 = self.get_amplitude(bits00);
                let amp01 = self.get_amplitude(bits01);
                let amp10 = self.get_amplitude(bits10);
                let amp11 = self.get_amplitude(bits11);
                
                // Determine which amplitude to update
                let subspace_idx = (bit1 << 1) | bit2;
                let mut new_amp = Complex64::new(0.0, 0.0);
                
                // Apply the 4x4 matrix to compute the new amplitude
                new_amp += matrix[subspace_idx * 4 + 0] * amp00;
                new_amp += matrix[subspace_idx * 4 + 1] * amp01;
                new_amp += matrix[subspace_idx * 4 + 2] * amp10;
                new_amp += matrix[subspace_idx * 4 + 3] * amp11;
                
                // Update the amplitude in the result
                if let Some(val) = new_chunks[chunk_idx].get_mut(local_idx) {
                    *val = new_amp;
                }
            }
        }
        
        // Update the state
        self.chunks = new_chunks;
    }
    
    /// Calculate probability of measuring a specific bit string
    pub fn probability(&self, bit_string: &[u8]) -> f64 {
        if bit_string.len() != self.num_qubits {
            panic!("Bit string length must match number of qubits");
        }
        
        // Convert bit string to index
        let mut idx = 0;
        for (i, &bit) in bit_string.iter().enumerate() {
            if bit != 0 {
                idx |= 1 << i;
            }
        }
        
        // Return probability
        self.get_amplitude(idx).norm_sqr()
    }
    
    /// Calculate probabilities for all basis states
    /// Warning: For large qubit counts, this will use a lot of memory
    pub fn probabilities(&self) -> Vec<f64> {
        self.chunks.iter().flat_map(|chunk| {
            chunk.as_slice().iter().map(|a| a.norm_sqr())
        }).collect()
    }
    
    /// Calculate the probability of a specified range of states
    /// More memory efficient for large qubit counts
    pub fn probability_range(&self, start_idx: usize, end_idx: usize) -> Vec<f64> {
        let real_end = std::cmp::min(end_idx, self.dimension);
        
        (start_idx..real_end).map(|idx| {
            self.get_amplitude(idx).norm_sqr()
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_1_SQRT_2;
    
    #[test]
    fn test_chunked_state_vector_init() {
        let sv = ChunkedStateVector::new(2);
        assert_eq!(sv.num_qubits(), 2);
        assert_eq!(sv.dimension(), 4);
        
        // Initial state should be |00>
        assert_eq!(sv.get_amplitude(0), Complex64::new(1.0, 0.0));
        assert_eq!(sv.get_amplitude(1), Complex64::new(0.0, 0.0));
        assert_eq!(sv.get_amplitude(2), Complex64::new(0.0, 0.0));
        assert_eq!(sv.get_amplitude(3), Complex64::new(0.0, 0.0));
    }
    
    #[test]
    fn test_hadamard_gate_chunked() {
        // Hadamard matrix
        let h_matrix = [
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(-FRAC_1_SQRT_2, 0.0),
        ];

        // Apply H to the 0th qubit of |00>
        let mut sv = ChunkedStateVector::new(2);
        println!("Initial state: {:?}", sv.as_vec());
        sv.apply_single_qubit_gate(&h_matrix, 1);  // Changed from 0 to 1

        // Print state for debugging
        println!("After H on qubit 1:");
        println!("amplitude[0] = {:?}", sv.get_amplitude(0));
        println!("amplitude[1] = {:?}", sv.get_amplitude(1));
        println!("amplitude[2] = {:?}", sv.get_amplitude(2));
        println!("amplitude[3] = {:?}", sv.get_amplitude(3));

        // Result should be |00> + |10> / sqrt(2)
        assert!((sv.get_amplitude(0) - Complex64::new(FRAC_1_SQRT_2, 0.0)).norm() < 1e-10);
        assert!((sv.get_amplitude(1) - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        assert!((sv.get_amplitude(2) - Complex64::new(FRAC_1_SQRT_2, 0.0)).norm() < 1e-10);
        assert!((sv.get_amplitude(3) - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        
        // Apply H to the 1st qubit (actually 0th in our implementation)
        sv.apply_single_qubit_gate(&h_matrix, 0);
        
        // Result should be (|00> + |01> + |10> - |11>) / 2
        // Add debug output
        println!("After both H gates:");
        println!("amplitude[0] = {:?}", sv.get_amplitude(0));
        println!("amplitude[1] = {:?}", sv.get_amplitude(1));
        println!("amplitude[2] = {:?}", sv.get_amplitude(2));
        println!("amplitude[3] = {:?}", sv.get_amplitude(3));

        assert!((sv.get_amplitude(0) - Complex64::new(0.5, 0.0)).norm() < 1e-10);
        assert!((sv.get_amplitude(1) - Complex64::new(0.5, 0.0)).norm() < 1e-10);
        assert!((sv.get_amplitude(2) - Complex64::new(0.5, 0.0)).norm() < 1e-10);
        assert!((sv.get_amplitude(3) - Complex64::new(0.5, 0.0)).norm() < 1e-10);
    }
    
    #[test]
    fn test_cnot_gate_chunked() {
        // Set up state |+0> = (|00> + |10>) / sqrt(2)
        let mut sv = ChunkedStateVector::new(2);
        
        // Hadamard on qubit 0
        let h_matrix = [
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(-FRAC_1_SQRT_2, 0.0),
        ];
        sv.apply_single_qubit_gate(&h_matrix, 0);
        
        // Apply CNOT
        sv.apply_cnot(0, 1);
        
        // Result should be (|00> + |11>) / sqrt(2) = Bell state
        assert!((sv.get_amplitude(0) - Complex64::new(FRAC_1_SQRT_2, 0.0)).norm() < 1e-10);
        assert!((sv.get_amplitude(1) - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        assert!((sv.get_amplitude(2) - Complex64::new(0.0, 0.0)).norm() < 1e-10);
        assert!((sv.get_amplitude(3) - Complex64::new(FRAC_1_SQRT_2, 0.0)).norm() < 1e-10);
    }
}