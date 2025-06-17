use crate::continuous_variable::{CVGateSequence, GaussianState, Complex};

fn main() {
    let mut sequence = CVGateSequence::new(2);
    sequence.squeezing(1, 0.5, 0.0).unwrap();

    let mut state = GaussianState::vacuum_state(2);
    println\!("Initial covariance matrix:");
    for i in 0..4 {
        for j in 0..4 {
            print\!("{:.3} ", state.covariance_matrix[i][j]);
        }
        println\!();
    }
    
    sequence.execute_on_state(&mut state).unwrap();
    
    println\!("After squeezing:");
    for i in 0..4 {
        for j in 0..4 {
            print\!("{:.3} ", state.covariance_matrix[i][j]);
        }
        println\!();
    }
    
    println\!("state.covariance_matrix[2][2] = {}", state.covariance_matrix[2][2]);
    println\!("Is [2][2] < 0.5? {}", state.covariance_matrix[2][2] < 0.5);
}
EOF < /dev/null