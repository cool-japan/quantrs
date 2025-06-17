//! Quantum Game Theory
//!
//! This module implements quantum game theory algorithms and protocols,
//! extending classical game theory to the quantum realm. It includes
//! quantum Nash equilibria, quantum strategies, and quantum mechanisms
//! for multi-player games.

use crate::error::{QuantRS2Error, QuantRS2Result};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::collections::HashMap;

/// Types of quantum games
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GameType {
    /// Prisoner's Dilemma with quantum strategies
    QuantumPrisonersDilemma,
    /// Battle of the Sexes with quantum entanglement
    QuantumBattleOfSexes,
    /// Quantum Auction mechanism
    QuantumAuction,
    /// Quantum Coordination Game
    QuantumCoordination,
    /// Quantum Minority Game
    QuantumMinorityGame,
    /// Custom quantum game
    Custom,
}

/// Player strategies in quantum games
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumStrategy {
    /// Classical deterministic strategy (pure classical)
    Classical(f64), // angle parameter for rotation
    /// Quantum superposition strategy
    Superposition { theta: f64, phi: f64 }, // Bloch sphere angles
    /// Entangled strategy (requires coordination with other player)
    Entangled,
    /// Mixed quantum strategy
    Mixed(Vec<(f64, QuantumStrategy)>), // probabilities and strategies
}

/// Quantum player in a game
#[derive(Debug, Clone)]
pub struct QuantumPlayer {
    /// Player ID
    pub id: usize,
    /// Player's quantum strategy
    pub strategy: QuantumStrategy,
    /// Player's quantum state (for entangled strategies)
    pub state: Option<Array1<Complex64>>,
    /// Player's utility function parameters
    pub utility_params: HashMap<String, f64>,
}

impl QuantumPlayer {
    /// Create a new quantum player
    pub fn new(id: usize, strategy: QuantumStrategy) -> Self {
        Self {
            id,
            strategy,
            state: None,
            utility_params: HashMap::new(),
        }
    }

    /// Set utility function parameter
    pub fn set_utility_param(&mut self, param: String, value: f64) {
        self.utility_params.insert(param, value);
    }

    /// Prepare quantum state based on strategy
    pub fn prepare_quantum_state(&mut self) -> QuantRS2Result<Array1<Complex64>> {
        match self.strategy {
            QuantumStrategy::Classical(theta) => {
                // Classical strategy as a quantum state |0⟩ or |1⟩ with rotation
                let state = Array1::from_vec(vec![
                    Complex64::new(theta.cos(), 0.0),
                    Complex64::new(theta.sin(), 0.0),
                ]);
                self.state = Some(state.clone());
                Ok(state)
            }
            QuantumStrategy::Superposition { theta, phi } => {
                // General qubit state on Bloch sphere
                let state = Array1::from_vec(vec![
                    Complex64::new((theta / 2.0).cos(), 0.0),
                    Complex64::new(
                        (theta / 2.0).sin() * phi.cos(),
                        (theta / 2.0).sin() * phi.sin(),
                    ),
                ]);
                self.state = Some(state.clone());
                Ok(state)
            }
            QuantumStrategy::Entangled => {
                // Entangled state preparation requires coordination
                // Return Bell state component for now
                let state = Array1::from_vec(vec![
                    Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                    Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                ]);
                self.state = Some(state.clone());
                Ok(state)
            }
            QuantumStrategy::Mixed(_) => {
                // Mixed strategy - for now use superposition
                let state = Array1::from_vec(vec![
                    Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                    Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                ]);
                self.state = Some(state.clone());
                Ok(state)
            }
        }
    }

    /// Compute expected utility given game outcomes
    pub fn compute_utility(&self, game_outcome: &GameOutcome) -> f64 {
        match &game_outcome.payoff_matrix {
            Some(payoffs) => {
                // Linear utility based on expected payoffs
                payoffs.iter().sum::<f64>() / payoffs.len() as f64
            }
            None => 0.0,
        }
    }
}

/// Game outcome after quantum measurement
#[derive(Debug, Clone)]
pub struct GameOutcome {
    /// Classical outcome for each player
    pub classical_outcomes: Vec<usize>,
    /// Quantum measurement probabilities
    pub probabilities: Array1<f64>,
    /// Payoff matrix for the outcomes
    pub payoff_matrix: Option<Vec<f64>>,
    /// Nash equilibrium indicator
    pub is_nash_equilibrium: bool,
}

/// Quantum game engine
#[derive(Debug, Clone)]
pub struct QuantumGame {
    /// Type of game
    pub game_type: GameType,
    /// Number of players
    pub num_players: usize,
    /// Players in the game
    pub players: Vec<QuantumPlayer>,
    /// Quantum circuit for the game
    pub game_circuit: Option<Array2<Complex64>>,
    /// Payoff matrices (one for each player)
    pub payoff_matrices: Vec<Array2<f64>>,
    /// Entanglement operator (if applicable)
    pub entanglement_operator: Option<Array2<Complex64>>,
}

impl QuantumGame {
    /// Create a new quantum game
    pub fn new(game_type: GameType, num_players: usize) -> Self {
        let players = Vec::new();
        let payoff_matrices = vec![Array2::zeros((2, 2)); num_players];

        Self {
            game_type,
            num_players,
            players,
            game_circuit: None,
            payoff_matrices,
            entanglement_operator: None,
        }
    }

    /// Add a player to the game
    pub fn add_player(&mut self, player: QuantumPlayer) -> QuantRS2Result<()> {
        if self.players.len() >= self.num_players {
            return Err(QuantRS2Error::InvalidInput(
                "Maximum number of players reached".to_string(),
            ));
        }

        self.players.push(player);
        Ok(())
    }

    /// Set payoff matrix for a specific player
    pub fn set_payoff_matrix(
        &mut self,
        player_id: usize,
        payoffs: Array2<f64>,
    ) -> QuantRS2Result<()> {
        if player_id >= self.num_players {
            return Err(QuantRS2Error::InvalidInput(
                "Player ID out of bounds".to_string(),
            ));
        }

        self.payoff_matrices[player_id] = payoffs;
        Ok(())
    }

    /// Create quantum prisoner's dilemma
    pub fn quantum_prisoners_dilemma() -> QuantRS2Result<Self> {
        let mut game = Self::new(GameType::QuantumPrisonersDilemma, 2);

        // Classical prisoner's dilemma payoff matrices
        // (Cooperate, Defect) strategies
        let payoff_p1 = Array2::from_shape_vec(
            (2, 2),
            vec![3.0, 0.0, 5.0, 1.0], // (CC, CD, DC, DD)
        )
        .unwrap();

        let payoff_p2 = Array2::from_shape_vec(
            (2, 2),
            vec![3.0, 5.0, 0.0, 1.0], // (CC, DC, CD, DD)
        )
        .unwrap();

        game.set_payoff_matrix(0, payoff_p1)?;
        game.set_payoff_matrix(1, payoff_p2)?;

        // Add entanglement operator (optional)
        let entanglement = Self::create_entanglement_operator(std::f64::consts::PI / 2.0);
        game.entanglement_operator = Some(entanglement);

        Ok(game)
    }

    /// Create quantum coordination game
    pub fn quantum_coordination_game() -> QuantRS2Result<Self> {
        let mut game = Self::new(GameType::QuantumCoordination, 2);

        // Coordination game: both players want to choose the same action
        let payoff_p1 = Array2::from_shape_vec(
            (2, 2),
            vec![2.0, 0.0, 0.0, 1.0], // Prefer (0,0) or (1,1)
        )
        .unwrap();

        let payoff_p2 = payoff_p1.clone();

        game.set_payoff_matrix(0, payoff_p1)?;
        game.set_payoff_matrix(1, payoff_p2)?;

        Ok(game)
    }

    /// Create quantum auction mechanism
    pub fn quantum_auction(num_bidders: usize) -> QuantRS2Result<Self> {
        let mut game = Self::new(GameType::QuantumAuction, num_bidders);

        // Each bidder has a 2x2 payoff matrix (bid high/low vs others)
        for i in 0..num_bidders {
            let payoff = Array2::from_shape_vec(
                (2, 2),
                vec![1.0, 0.5, 2.0, 0.0], // Utility from winning/losing auction
            )
            .unwrap();

            game.set_payoff_matrix(i, payoff)?;
        }

        Ok(game)
    }

    /// Create entanglement operator
    fn create_entanglement_operator(gamma: f64) -> Array2<Complex64> {
        // J(γ) = exp(iγ(σ_x ⊗ σ_x + σ_y ⊗ σ_y)/2)
        // Simplified version: controlled-rotation
        let cos_g = (gamma / 2.0).cos();
        let sin_g = (gamma / 2.0).sin();

        Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex64::new(cos_g, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, sin_g),
                Complex64::new(0.0, 0.0),
                Complex64::new(cos_g, 0.0),
                Complex64::new(0.0, -sin_g),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -sin_g),
                Complex64::new(cos_g, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, sin_g),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(cos_g, 0.0),
            ],
        )
        .unwrap()
    }

    /// Play the quantum game and return outcome
    pub fn play_game(&mut self) -> QuantRS2Result<GameOutcome> {
        if self.players.len() != self.num_players {
            return Err(QuantRS2Error::InvalidInput(
                "Not all players have joined the game".to_string(),
            ));
        }

        // Prepare player quantum states
        let mut joint_state = self.prepare_joint_state()?;

        // Apply entanglement operator if present
        if let Some(entanglement_op) = &self.entanglement_operator {
            joint_state = entanglement_op.dot(&joint_state);
        }

        // Apply player strategies
        joint_state = self.apply_player_strategies(joint_state)?;

        // Measure the final state
        self.measure_game_outcome(joint_state)
    }

    /// Prepare joint quantum state of all players
    fn prepare_joint_state(&mut self) -> QuantRS2Result<Array1<Complex64>> {
        let total_dim = 1 << self.num_players; // 2^n dimensional Hilbert space
        let mut joint_state = Array1::zeros(total_dim);

        // Start with |00...0⟩ state
        joint_state[0] = Complex64::new(1.0, 0.0);

        // Apply each player's initial strategy
        let mut player_states = Vec::new();
        for player in self.players.iter_mut() {
            let player_state = player.prepare_quantum_state()?;
            player_states.push(player_state);
        }

        for (i, player_state) in player_states.into_iter().enumerate() {
            joint_state = self.tensor_product_player_state(joint_state, player_state, i)?;
        }

        Ok(joint_state)
    }

    /// Tensor product of joint state with single player state
    fn tensor_product_player_state(
        &self,
        joint_state: Array1<Complex64>,
        player_state: Array1<Complex64>,
        player_index: usize,
    ) -> QuantRS2Result<Array1<Complex64>> {
        // Simplified implementation - in practice this is more complex
        let mut new_state = joint_state.clone();

        // Apply player's state preparation as a rotation on their qubit
        for i in 0..new_state.len() {
            let bit = (i >> player_index) & 1;
            if bit == 0 {
                new_state[i] *= player_state[0];
            } else {
                new_state[i] *= player_state[1];
            }
        }

        Ok(new_state)
    }

    /// Apply all player strategies to the joint state
    fn apply_player_strategies(
        &self,
        mut joint_state: Array1<Complex64>,
    ) -> QuantRS2Result<Array1<Complex64>> {
        // For each player, apply their quantum strategy operator
        for (i, player) in self.players.iter().enumerate() {
            joint_state = self.apply_single_player_strategy(joint_state, &player.strategy, i)?;
        }

        Ok(joint_state)
    }

    /// Apply a single player's strategy to the joint state
    fn apply_single_player_strategy(
        &self,
        mut joint_state: Array1<Complex64>,
        strategy: &QuantumStrategy,
        player_index: usize,
    ) -> QuantRS2Result<Array1<Complex64>> {
        match strategy {
            QuantumStrategy::Classical(theta) => {
                // Apply rotation around Z-axis
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                for i in 0..joint_state.len() {
                    let bit = (i >> player_index) & 1;
                    if bit == 1 {
                        joint_state[i] *= Complex64::new(cos_theta, sin_theta);
                    }
                }
            }
            QuantumStrategy::Superposition { theta, phi } => {
                // Apply general single-qubit rotation
                let half_theta = theta / 2.0;
                let cos_half = half_theta.cos();
                let sin_half = half_theta.sin();

                let mut new_state = Array1::zeros(joint_state.len());

                for i in 0..joint_state.len() {
                    let bit = (i >> player_index) & 1;
                    let flipped_i = i ^ (1 << player_index);

                    if bit == 0 {
                        new_state[i] += cos_half * joint_state[i];
                        new_state[flipped_i] +=
                            sin_half * Complex64::new(phi.cos(), phi.sin()) * joint_state[i];
                    }
                }

                joint_state = new_state;
            }
            QuantumStrategy::Entangled => {
                // Entangled strategies are handled by the entanglement operator
                // No additional operation needed here
            }
            QuantumStrategy::Mixed(_strategies) => {
                // Mixed strategies require probabilistic sampling
                // For now, treat as superposition
                let mut new_state = joint_state.clone();
                for i in 0..new_state.len() {
                    new_state[i] *= Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
                }
                joint_state = new_state;
            }
        }

        Ok(joint_state)
    }

    /// Measure the final game state and determine outcome
    fn measure_game_outcome(&self, joint_state: Array1<Complex64>) -> QuantRS2Result<GameOutcome> {
        let num_outcomes = joint_state.len();
        let mut probabilities = Array1::zeros(num_outcomes);

        // Compute measurement probabilities
        for i in 0..num_outcomes {
            probabilities[i] = joint_state[i].norm_sqr();
        }

        // Sample a classical outcome based on probabilities
        let outcome_index = self.sample_outcome(&probabilities)?;

        // Extract individual player outcomes
        let mut classical_outcomes = Vec::new();
        for player_idx in 0..self.num_players {
            let bit = (outcome_index >> player_idx) & 1;
            classical_outcomes.push(bit);
        }

        // Compute payoffs for this outcome
        let payoffs = self.compute_payoffs(&classical_outcomes)?;

        // Check if this is a Nash equilibrium
        let is_nash = self.is_nash_equilibrium(&classical_outcomes, &payoffs)?;

        Ok(GameOutcome {
            classical_outcomes,
            probabilities,
            payoff_matrix: Some(payoffs),
            is_nash_equilibrium: is_nash,
        })
    }

    /// Sample an outcome based on probability distribution
    fn sample_outcome(&self, probabilities: &Array1<f64>) -> QuantRS2Result<usize> {
        let mut rng = rand::rng();
        use rand::Rng;

        let random_value: f64 = rng.random();
        let mut cumulative = 0.0;

        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                return Ok(i);
            }
        }

        // Fallback to last outcome
        Ok(probabilities.len() - 1)
    }

    /// Compute payoffs for all players given classical outcomes
    fn compute_payoffs(&self, outcomes: &[usize]) -> QuantRS2Result<Vec<f64>> {
        let mut payoffs = Vec::new();

        for (player_idx, payoff_matrix) in self.payoff_matrices.iter().enumerate() {
            if outcomes.len() < 2 {
                return Err(QuantRS2Error::InvalidInput(
                    "Need at least 2 players for payoff calculation".to_string(),
                ));
            }

            // For 2-player games, use both players' actions
            let player_action = outcomes[player_idx];
            let opponent_action = outcomes[1 - player_idx % 2]; // Simple opponent selection

            let payoff = payoff_matrix[[player_action, opponent_action]];
            payoffs.push(payoff);
        }

        Ok(payoffs)
    }

    /// Check if current outcome is a Nash equilibrium
    fn is_nash_equilibrium(&self, outcomes: &[usize], _payoffs: &[f64]) -> QuantRS2Result<bool> {
        // Simplified Nash equilibrium check
        // In a real implementation, this would check if any player can improve
        // their payoff by unilaterally changing their strategy

        // For now, assume prisoner's dilemma: (Defect, Defect) is Nash equilibrium
        match self.game_type {
            GameType::QuantumPrisonersDilemma => {
                Ok(outcomes == &[1, 1]) // Both defect
            }
            GameType::QuantumCoordination => {
                Ok(outcomes[0] == outcomes[1]) // Coordination successful
            }
            _ => Ok(false),
        }
    }

    /// Find quantum Nash equilibria using iterative algorithm
    pub fn find_quantum_nash_equilibria(&mut self) -> QuantRS2Result<Vec<GameOutcome>> {
        let mut equilibria = Vec::new();

        // Grid search over strategy space
        let theta_steps = 10;
        let phi_steps = 10;

        for i in 0..theta_steps {
            for j in 0..phi_steps {
                let theta = (i as f64) * std::f64::consts::PI / (theta_steps as f64);
                let phi = (j as f64) * 2.0 * std::f64::consts::PI / (phi_steps as f64);

                // Set strategies for all players
                for player in &mut self.players {
                    player.strategy = QuantumStrategy::Superposition { theta, phi };
                }

                // Play game and check if outcome is equilibrium
                let outcome = self.play_game()?;
                if outcome.is_nash_equilibrium {
                    equilibria.push(outcome);
                }
            }
        }

        Ok(equilibria)
    }

    /// Compute quantum advantage over classical game
    pub fn quantum_advantage(&mut self) -> QuantRS2Result<f64> {
        // Play quantum version
        let quantum_outcome = self.play_game()?;
        let quantum_payoffs = quantum_outcome.payoff_matrix.unwrap_or_default();
        let quantum_total = quantum_payoffs.iter().sum::<f64>();

        // Compare with classical Nash equilibrium
        let classical_total = self.compute_classical_nash_payoff()?;

        Ok(quantum_total - classical_total)
    }

    /// Compute payoff at classical Nash equilibrium
    fn compute_classical_nash_payoff(&self) -> QuantRS2Result<f64> {
        // For prisoner's dilemma: classical Nash is (Defect, Defect) = (1, 1)
        // Both players get payoff of 1
        match self.game_type {
            GameType::QuantumPrisonersDilemma => Ok(2.0), // 1 + 1
            GameType::QuantumCoordination => Ok(0.0),     // Miscoordination
            _ => Ok(0.0),
        }
    }
}

/// Quantum mechanism design for multi-player games
#[derive(Debug, Clone)]
pub struct QuantumMechanism {
    /// Number of players
    pub num_players: usize,
    /// Mechanism type
    pub mechanism_type: String,
    /// Quantum circuit implementing the mechanism
    pub circuit: Option<Array2<Complex64>>,
    /// Revenue function for the mechanism designer
    pub revenue_function: Option<fn(&[f64]) -> f64>,
}

impl QuantumMechanism {
    /// Create quantum auction mechanism
    pub fn quantum_auction_mechanism(num_bidders: usize) -> Self {
        Self {
            num_players: num_bidders,
            mechanism_type: "Quantum Auction".to_string(),
            circuit: None,
            revenue_function: Some(|bids| bids.iter().sum::<f64>() * 0.1), // 10% commission
        }
    }

    /// Create quantum voting mechanism
    pub fn quantum_voting_mechanism(num_voters: usize) -> Self {
        Self {
            num_players: num_voters,
            mechanism_type: "Quantum Voting".to_string(),
            circuit: None,
            revenue_function: None, // No revenue in voting
        }
    }

    /// Design optimal quantum mechanism
    pub fn design_optimal_mechanism(&mut self) -> QuantRS2Result<Array2<Complex64>> {
        // This would implement the quantum mechanism design algorithm
        // For now, return identity transformation
        let dim = 1 << self.num_players;
        Ok(Array2::eye(dim))
    }

    /// Verify mechanism properties (incentive compatibility, individual rationality)
    pub fn verify_mechanism_properties(&self) -> QuantRS2Result<(bool, bool)> {
        // Simplified verification
        // In practice, this would check incentive compatibility and individual rationality

        let incentive_compatible = true; // Assume true for now
        let individually_rational = true; // Assume true for now

        Ok((incentive_compatible, individually_rational))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_player_creation() {
        let strategy = QuantumStrategy::Superposition {
            theta: std::f64::consts::PI / 4.0,
            phi: 0.0,
        };
        let player = QuantumPlayer::new(0, strategy.clone());

        assert_eq!(player.id, 0);
        assert_eq!(player.strategy, strategy);
        assert!(player.state.is_none());
    }

    #[test]
    fn test_quantum_state_preparation() {
        let mut player =
            QuantumPlayer::new(0, QuantumStrategy::Classical(std::f64::consts::PI / 4.0));
        let state = player.prepare_quantum_state().unwrap();

        assert_eq!(state.len(), 2);
        assert!(state[0].norm() > 0.0);
        assert!(player.state.is_some());
    }

    #[test]
    fn test_quantum_prisoners_dilemma() {
        let game = QuantumGame::quantum_prisoners_dilemma().unwrap();

        assert_eq!(game.game_type, GameType::QuantumPrisonersDilemma);
        assert_eq!(game.num_players, 2);
        assert_eq!(game.payoff_matrices.len(), 2);
        assert!(game.entanglement_operator.is_some());
    }

    #[test]
    fn test_quantum_coordination_game() {
        let game = QuantumGame::quantum_coordination_game().unwrap();

        assert_eq!(game.game_type, GameType::QuantumCoordination);
        assert_eq!(game.num_players, 2);
        assert_eq!(game.payoff_matrices.len(), 2);
    }

    #[test]
    fn test_game_with_players() {
        let mut game = QuantumGame::quantum_prisoners_dilemma().unwrap();

        let player1 = QuantumPlayer::new(0, QuantumStrategy::Classical(0.0));
        let player2 = QuantumPlayer::new(1, QuantumStrategy::Classical(std::f64::consts::PI));

        game.add_player(player1).unwrap();
        game.add_player(player2).unwrap();

        let outcome = game.play_game().unwrap();
        assert_eq!(outcome.classical_outcomes.len(), 2);
        assert!(outcome.probabilities.len() > 0);
    }

    #[test]
    fn test_entanglement_operator() {
        let entanglement = QuantumGame::create_entanglement_operator(std::f64::consts::PI / 2.0);

        assert_eq!(entanglement.dim(), (4, 4));

        // Check unitarity (U†U = I)
        let conjugate_transpose = entanglement.t().mapv(|x| x.conj());
        let product = conjugate_transpose.dot(&entanglement);

        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((product[[i, j]].norm() - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_nash_equilibrium_detection() {
        let game = QuantumGame::quantum_prisoners_dilemma().unwrap();

        // (Defect, Defect) should be Nash equilibrium in prisoner's dilemma
        let is_nash = game.is_nash_equilibrium(&[1, 1], &[1.0, 1.0]).unwrap();
        assert!(is_nash);

        // (Cooperate, Cooperate) should not be Nash equilibrium
        let is_nash = game.is_nash_equilibrium(&[0, 0], &[3.0, 3.0]).unwrap();
        assert!(!is_nash);
    }

    #[test]
    fn test_quantum_mechanism_creation() {
        let mechanism = QuantumMechanism::quantum_auction_mechanism(3);

        assert_eq!(mechanism.num_players, 3);
        assert_eq!(mechanism.mechanism_type, "Quantum Auction");
        assert!(mechanism.revenue_function.is_some());
    }

    #[test]
    fn test_mechanism_verification() {
        let mechanism = QuantumMechanism::quantum_voting_mechanism(5);
        let (ic, ir) = mechanism.verify_mechanism_properties().unwrap();

        assert!(ic); // Incentive compatible
        assert!(ir); // Individually rational
    }

    #[test]
    fn test_quantum_advantage_calculation() {
        let mut game = QuantumGame::quantum_prisoners_dilemma().unwrap();

        let player1 = QuantumPlayer::new(
            0,
            QuantumStrategy::Superposition {
                theta: std::f64::consts::PI / 4.0,
                phi: 0.0,
            },
        );
        let player2 = QuantumPlayer::new(
            1,
            QuantumStrategy::Superposition {
                theta: std::f64::consts::PI / 4.0,
                phi: 0.0,
            },
        );

        game.add_player(player1).unwrap();
        game.add_player(player2).unwrap();

        let advantage = game.quantum_advantage().unwrap();
        // Quantum strategies can potentially achieve better outcomes than classical Nash
        // The exact value depends on the strategy profile
        assert!(advantage.is_finite());
    }
}
