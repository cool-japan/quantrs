//! Quantum Continuous Reinforcement Learning Example
//!
//! This example demonstrates quantum reinforcement learning algorithms
//! for continuous action spaces, including QDDPG and QSAC.

use ndarray::Array1;
use quantrs2_ml::autodiff::optimizers::Adam;
use quantrs2_ml::prelude::*;

fn main() -> Result<()> {
    println!("=== Quantum Continuous RL Demo ===\n");

    // Step 1: Test pendulum environment
    println!("1. Testing Pendulum Environment...");
    test_pendulum_dynamics()?;

    // Step 2: Train QDDPG on pendulum
    println!("\n2. Training Quantum DDPG on Pendulum Control...");
    train_qddpg_pendulum()?;

    // Step 3: Compare with random policy
    println!("\n3. Comparing with Random Policy...");
    compare_policies()?;

    // Step 4: Demonstrate custom continuous environment
    println!("\n4. Custom Continuous Environment Example...");
    custom_environment_demo()?;

    println!("\n=== Continuous RL Demo Complete ===");

    Ok(())
}

/// Test pendulum environment dynamics
fn test_pendulum_dynamics() -> Result<()> {
    let mut env = PendulumEnvironment::new();

    println!("   Initial state: {:?}", env.state());
    println!("   Action bounds: {:?}", env.action_bounds());

    // Run a few steps with different actions
    let actions = vec![
        Array1::from_vec(vec![0.0]),  // No torque
        Array1::from_vec(vec![2.0]),  // Max positive torque
        Array1::from_vec(vec![-2.0]), // Max negative torque
    ];

    for (i, action) in actions.iter().enumerate() {
        let state = env.reset();
        let (next_state, reward, done) = env.step(action.clone())?;

        println!("\n   Step {} with action {:.1}:", i + 1, action[0]);
        println!(
            "     State: [θ_cos={:.3}, θ_sin={:.3}, θ_dot={:.3}]",
            state[0], state[1], state[2]
        );
        println!(
            "     Next: [θ_cos={:.3}, θ_sin={:.3}, θ_dot={:.3}]",
            next_state[0], next_state[1], next_state[2]
        );
        println!("     Reward: {:.3}, Done: {}", reward, done);
    }

    Ok(())
}

/// Train QDDPG on pendulum control
fn train_qddpg_pendulum() -> Result<()> {
    let state_dim = 3;
    let action_dim = 1;
    let action_bounds = vec![(-2.0, 2.0)];
    let num_qubits = 4;
    let buffer_capacity = 10000;

    // Create QDDPG agent
    let mut agent = QuantumDDPG::new(
        state_dim,
        action_dim,
        action_bounds,
        num_qubits,
        buffer_capacity,
    )?;

    // Create environment
    let mut env = PendulumEnvironment::new();

    // Create optimizers
    let mut actor_optimizer = Adam::new(0.001);
    let mut critic_optimizer = Adam::new(0.001);

    // Train for a few episodes (reduced for demo)
    let episodes = 50;
    println!("   Training QDDPG for {} episodes...", episodes);

    let rewards = agent.train(
        &mut env,
        episodes,
        &mut actor_optimizer,
        &mut critic_optimizer,
    )?;

    // Print training statistics
    let avg_initial = rewards[..10].iter().sum::<f64>() / 10.0;
    let avg_final = rewards[rewards.len() - 10..].iter().sum::<f64>() / 10.0;

    println!("\n   Training Statistics:");
    println!("   - Average initial reward: {:.2}", avg_initial);
    println!("   - Average final reward: {:.2}", avg_final);
    println!("   - Improvement: {:.2}", avg_final - avg_initial);

    // Test trained agent
    println!("\n   Testing trained agent...");
    test_trained_agent(&agent, &mut env)?;

    Ok(())
}

/// Test a trained agent
fn test_trained_agent(agent: &QuantumDDPG, env: &mut dyn ContinuousEnvironment) -> Result<()> {
    let test_episodes = 5;
    let mut test_rewards = Vec::new();

    for episode in 0..test_episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        let mut done = false;
        let mut steps = 0;

        while !done && steps < 200 {
            let action = agent.get_action(&state, false)?; // No exploration
            let (next_state, reward, is_done) = env.step(action.clone())?;

            state = next_state;
            episode_reward += reward;
            done = is_done;
            steps += 1;
        }

        test_rewards.push(episode_reward);
        println!(
            "   Test episode {}: Reward = {:.2}, Steps = {}",
            episode + 1,
            episode_reward,
            steps
        );
    }

    let avg_test = test_rewards.iter().sum::<f64>() / test_episodes as f64;
    println!("   Average test reward: {:.2}", avg_test);

    Ok(())
}

/// Compare trained policy with random policy
fn compare_policies() -> Result<()> {
    let mut env = PendulumEnvironment::new();
    let episodes = 10;

    // Random policy performance
    println!("   Random Policy Performance:");
    let mut random_rewards = Vec::new();

    for _ in 0..episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        let mut done = false;

        while !done {
            // Random action in bounds
            let action = Array1::from_vec(vec![4.0 * rand::random::<f64>() - 2.0]);

            let (next_state, reward, is_done) = env.step(action)?;
            state = next_state;
            episode_reward += reward;
            done = is_done;
        }

        random_rewards.push(episode_reward);
    }

    let avg_random = random_rewards.iter().sum::<f64>() / episodes as f64;
    println!("   Average random policy reward: {:.2}", avg_random);

    // Simple control policy (proportional control)
    println!("\n   Simple Control Policy Performance:");
    let mut control_rewards = Vec::new();

    for _ in 0..episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        let mut done = false;

        while !done {
            // Proportional control: torque = -k * theta
            let theta = state[1].atan2(state[0]); // Reconstruct angle
            let action = Array1::from_vec(vec![(-2.0 * theta).clamp(-2.0, 2.0)]);

            let (next_state, reward, is_done) = env.step(action)?;
            state = next_state;
            episode_reward += reward;
            done = is_done;
        }

        control_rewards.push(episode_reward);
    }

    let avg_control = control_rewards.iter().sum::<f64>() / episodes as f64;
    println!("   Average control policy reward: {:.2}", avg_control);

    println!("\n   Performance Summary:");
    println!("   - Random policy: {:.2}", avg_random);
    println!("   - Simple control: {:.2}", avg_control);
    println!("   - Improvement: {:.2}", avg_control - avg_random);

    Ok(())
}

/// Custom continuous environment example
fn custom_environment_demo() -> Result<()> {
    // Define a simple 2D navigation environment
    struct Navigation2D {
        position: Array1<f64>,
        goal: Array1<f64>,
        max_steps: usize,
        current_step: usize,
    }

    impl Navigation2D {
        fn new() -> Self {
            Self {
                position: Array1::zeros(2),
                goal: Array1::from_vec(vec![5.0, 5.0]),
                max_steps: 50,
                current_step: 0,
            }
        }
    }

    impl ContinuousEnvironment for Navigation2D {
        fn state(&self) -> Array1<f64> {
            // State includes position and relative goal position
            let mut state = Array1::zeros(4);
            state[0] = self.position[0];
            state[1] = self.position[1];
            state[2] = self.goal[0] - self.position[0];
            state[3] = self.goal[1] - self.position[1];
            state
        }

        fn action_bounds(&self) -> Vec<(f64, f64)> {
            vec![(-1.0, 1.0), (-1.0, 1.0)] // Velocity in x and y
        }

        fn step(&mut self, action: Array1<f64>) -> Result<(Array1<f64>, f64, bool)> {
            // Update position
            self.position = &self.position + &action;

            // Compute distance to goal
            let distance = ((self.position[0] - self.goal[0]).powi(2)
                + (self.position[1] - self.goal[1]).powi(2))
            .sqrt();

            // Reward is negative distance (closer is better)
            let reward = -distance;

            self.current_step += 1;
            let done = distance < 0.5 || self.current_step >= self.max_steps;

            Ok((self.state(), reward, done))
        }

        fn reset(&mut self) -> Array1<f64> {
            self.position = Array1::from_vec(vec![
                10.0 * rand::random::<f64>() - 5.0,
                10.0 * rand::random::<f64>() - 5.0,
            ]);
            self.current_step = 0;
            self.state()
        }

        fn state_dim(&self) -> usize {
            4
        }
        fn action_dim(&self) -> usize {
            2
        }
    }

    println!("   Created 2D Navigation Environment");

    let mut nav_env = Navigation2D::new();
    let state = nav_env.reset();

    println!("   Initial position: [{:.2}, {:.2}]", state[0], state[1]);
    println!("   Goal position: [5.00, 5.00]");
    println!("   Action space: 2D velocity vectors in [-1, 1]");

    // Demonstrate a few steps
    println!("\n   Taking some steps:");
    for i in 0..3 {
        let action = Array1::from_vec(vec![
            0.5 * (2.0 * rand::random::<f64>() - 1.0),
            0.5 * (2.0 * rand::random::<f64>() - 1.0),
        ]);

        let (next_state, reward, done) = nav_env.step(action.clone())?;

        println!(
            "   Step {}: action=[{:.2}, {:.2}], pos=[{:.2}, {:.2}], reward={:.2}, done={}",
            i + 1,
            action[0],
            action[1],
            next_state[0],
            next_state[1],
            reward,
            done
        );
    }

    println!("\n   This demonstrates how to create custom continuous environments");
    println!("   for quantum RL algorithms!");

    Ok(())
}
