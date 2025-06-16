// Simple test to verify the quantum gate fix
use ndarray::Array1;
use std::f64::consts::PI;

// Mock structure for testing
struct MockQNN {
    num_qubits: usize,
}

impl MockQNN {
    fn new(num_qubits: usize) -> Self {
        Self { num_qubits }
    }

    fn apply_rx_gate(&self, state: &Array1<f64>, qubit: usize, angle: f64) -> Array1<f64> {
        let num_qubits = self.num_qubits;
        let new_state = state.clone();
        let state_dim = state.len();

        for i in 0..state_dim {
            // Map qubit index to bit position (qubit 0 is the most significant bit)
            let bit_pos = num_qubits - 1 - qubit;
            if (i & (1 << bit_pos)) == 0 {
                let j = i | (1 << bit_pos);
                if j < state_dim {
                    let cos_half = (angle / 2.0).cos();
                    let sin_half = (angle / 2.0).sin();

                    let state_i = state[i];
                    let state_j = state[j];

                    new_state[i] = cos_half * state_i + sin_half * state_j;
                    new_state[j] = sin_half * state_i + cos_half * state_j;
                }
            }
        }

        new_state
    }
}

fn main() {
    let qnn = MockQNN::new(2);
    let state = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]); // |00⟩

    let new_state = qnn.apply_rx_gate(&state, 0, PI / 2.0);

    println!("Original state: {:?}", state);
    println!("New state: {:?}", new_state);
    println!("Expected: [1/√2, 0, 1/√2, 0] = [{}, 0, {}, 0]", 1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt());
    println!("new_state[0] = {}, expected = {}", new_state[0], 1.0 / 2.0_f64.sqrt());
    println!("new_state[2] = {}, expected = {}", new_state[2], 1.0 / 2.0_f64.sqrt());

    // Check if our fix works
    let tolerance = 1e-10;
    let expected = 1.0 / 2.0_f64.sqrt();
    
    if (new_state[0] - expected).abs() < tolerance && (new_state[2] - expected).abs() < tolerance {
        println!("✅ Test PASSED! RX gate fix is working correctly.");
    } else {
        println!("❌ Test FAILED! RX gate fix needs more work.");
        println!("Difference in [0]: {}", (new_state[0] - expected).abs());
        println!("Difference in [2]: {}", (new_state[2] - expected).abs());
    }
}