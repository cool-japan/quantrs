//! Comprehensive example demonstrating the AWS Braket quantum annealing client
//!
//! This example shows how to use the AWS Braket client to:
//! 1. Connect to AWS Braket quantum computing services
//! 2. Discover and select optimal quantum devices
//! 3. Submit quantum annealing problems
//! 4. Monitor task status and performance
//! 5. Handle batch problem submission
//! 6. Track costs and usage
//! 7. Work with advanced annealing parameters

use std::collections::HashMap;
use std::time::Duration;

use quantrs2_anneal::ising::IsingModel;

#[cfg(feature = "braket")]
use quantrs2_anneal::braket::{
    AdvancedAnnealingParams, BraketClient, CostTracker, DeviceSelector, DeviceStatus,
    DeviceType,
};

#[cfg(not(feature = "braket"))]
fn main() {
    println!("AWS Braket Client Example");
    println!("========================");
    println!();
    println!("Note: This example requires the 'braket' feature to be enabled.");
    println!("To run with AWS Braket support:");
    println!("  cargo run --example aws_braket_client_example --features braket");
    println!();
    println!("You will also need:");
    println!("  1. An AWS account with Braket access");
    println!("  2. AWS credentials configured (access key and secret key)");
    println!("  3. Set the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables");
    println!("  4. Set the AWS_REGION environment variable (e.g., us-east-1)");
}

#[cfg(feature = "braket")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("AWS Braket Quantum Annealing Client Example");
    println!("==========================================");
    println!();

    // Get AWS credentials from environment variables
    let access_key = std::env::var("AWS_ACCESS_KEY_ID")
        .map_err(|_| "Please set AWS_ACCESS_KEY_ID environment variable")?;
    let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY")
        .map_err(|_| "Please set AWS_SECRET_ACCESS_KEY environment variable")?;
    let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());

    // Example 1: Basic client setup with cost tracking
    println!("1. Setting up AWS Braket Client");
    println!("------------------------------");

    let device_selector = DeviceSelector {
        device_type: Some(DeviceType::QuantumProcessor),
        provider: None,
        status: DeviceStatus::Online,
        min_fidelity: Some(0.95),
        max_cost_per_shot: Some(0.01), // $0.01 per shot max
        required_capabilities: vec!["ANNEALING".to_string()],
    };

    let cost_tracker = CostTracker {
        cost_limit: Some(100.0), // $100 limit
        current_cost: 0.0,
        cost_estimates: HashMap::new(),
    };

    let client = BraketClient::with_config(
        access_key,
        secret_key,
        None, // No session token
        region.clone(),
        device_selector,
        cost_tracker,
    )?;

    println!("✓ AWS Braket client initialized for region: {}", region);

    // Example 2: Discover available devices
    println!("\n2. Discovering Available Devices");
    println!("-------------------------------");

    let devices = client.get_devices()?;
    println!("Found {} devices:", devices.len());

    for device in &devices[..5.min(devices.len())] {
        println!(
            "  - {} ({}): {} - Status: {:?}",
            device.device_name,
            device.provider_name,
            match device.device_type {
                DeviceType::QuantumProcessor => "QPU",
                DeviceType::Simulator => "Simulator",
            },
            device.device_status
        );
    }

    // Example 3: Automatic device selection
    println!("\n3. Automatic Device Selection");
    println!("----------------------------");

    let qpu_selector = DeviceSelector {
        device_type: Some(DeviceType::QuantumProcessor),
        status: DeviceStatus::Online,
        required_capabilities: vec!["ANNEALING".to_string()],
        ..Default::default()
    };

    match client.select_device(Some(&qpu_selector)) {
        Ok(device) => {
            println!(
                "✓ Selected QPU device: {} ({})",
                device.device_name, device.provider_name
            );
        }
        Err(e) => {
            println!("⚠ No annealing QPUs available: {}", e);
        }
    }

    let simulator_selector = DeviceSelector {
        device_type: Some(DeviceType::Simulator),
        status: DeviceStatus::Online,
        ..Default::default()
    };

    match client.select_device(Some(&simulator_selector)) {
        Ok(device) => {
            println!(
                "✓ Selected simulator: {} ({})",
                device.device_name, device.provider_name
            );
        }
        Err(e) => {
            println!("⚠ No simulators available: {}", e);
        }
    }

    // Example 4: Create test problems
    println!("\n4. Creating Test Problems");
    println!("------------------------");

    // Small Max-Cut problem
    let mut small_problem = IsingModel::new(4);
    small_problem.set_coupling(0, 1, -1.0)?;
    small_problem.set_coupling(1, 2, -1.0)?;
    small_problem.set_coupling(2, 3, -1.0)?;
    small_problem.set_coupling(3, 0, -1.0)?;
    small_problem.set_coupling(0, 2, -1.0)?;
    small_problem.set_coupling(1, 3, -1.0)?;
    println!("✓ Created 4-qubit Max-Cut problem");

    // Larger random problem
    let mut large_problem = IsingModel::new(20);
    for i in 0..19 {
        large_problem.set_coupling(i, i + 1, -1.0)?;
        if i % 3 == 0 && i + 3 < 20 {
            large_problem.set_coupling(i, i + 3, -0.5)?;
        }
    }
    // Add some random biases
    for i in [2, 5, 8, 11, 14, 17] {
        large_problem.set_bias(i, 0.3)?;
    }
    println!("✓ Created 20-variable problem with biases");

    // Example 5: Advanced problem submission
    println!("\n5. Advanced Problem Submission");
    println!("-----------------------------");

    let advanced_params = AdvancedAnnealingParams {
        shots: 1000,
        annealing_time: Some(20.0),
        programming_thermalization: Some(1000.0),
        readout_thermalization: Some(100.0),
        beta_schedule: Some(vec![
            (0.0, 0.1),   // Start with low beta (high temperature)
            (10.0, 1.0),  // Increase beta linearly
            (20.0, 10.0), // End with high beta (low temperature)
        ]),
        s_schedule: None,
        auto_scale: Some(true),
        flux_biases: None,
        extra: {
            let mut extra = HashMap::new();
            extra.insert("seed".to_string(), serde_json::json!(42));
            extra
        },
    };

    println!("Problem submission simulation (would submit to quantum device):");
    println!("  - 1000 shots");
    println!("  - 20μs annealing time");
    println!("  - Custom beta (temperature) schedule");
    println!("  - Auto-scaling enabled");
    println!("  - Random seed: 42");

    // Note: In a real implementation, you would actually submit:
    // let task_result = client.submit_ising(
    //     &small_problem,
    //     None, // Auto-select device
    //     Some(advanced_params),
    // )?;

    // Example 6: Simulator usage for testing
    println!("\n6. Simulator Usage for Testing");
    println!("------------------------------");

    let simulator_params = AdvancedAnnealingParams {
        shots: 10000,              // More shots since simulators are faster/cheaper
        annealing_time: Some(1.0), // Shorter time for simulation
        auto_scale: Some(true),
        ..Default::default()
    };

    println!("Simulator submission simulation (would run on classical simulator):");
    println!("  - 10,000 shots for better statistics");
    println!("  - 1μs annealing time (simulation parameter)");
    println!("  - Cost: typically $0 (simulators are usually free)");

    // Note: In a real implementation:
    // let simulator_result = client.submit_ising(
    //     &large_problem,
    //     None, // Auto-select simulator
    //     Some(simulator_params),
    // )?;

    // Example 7: Batch problem submission
    println!("\n7. Batch Problem Submission");
    println!("--------------------------");

    let mut batch_problems = Vec::new();

    // Create multiple variants of a problem
    for i in 0..3 {
        let mut problem = IsingModel::new(3);
        let coupling_strength = -1.0 - i as f64 * 0.2;
        problem.set_coupling(0, 1, coupling_strength)?;
        problem.set_coupling(1, 2, coupling_strength)?;
        problem.set_coupling(2, 0, coupling_strength)?;
        batch_problems.push(problem);
    }

    println!(
        "Batch submission simulation (would submit {} problems):",
        batch_problems.len()
    );
    for (i, _) in batch_problems.iter().enumerate() {
        println!(
            "  Problem {}: 3-qubit triangle with coupling strength {:.1}",
            i + 1,
            -1.0 - i as f64 * 0.2
        );
    }

    // Note: In a real implementation:
    // let batch_problems_ref: Vec<_> = batch_problems.iter()
    //     .map(|p| (p, None, None))
    //     .collect();
    // let batch_result = client.submit_batch(batch_problems_ref)?;

    // Example 8: Cost tracking and optimization
    println!("\n8. Cost Tracking and Optimization");
    println!("--------------------------------");

    println!("Cost management features:");
    println!("  ✓ Real-time cost estimation");
    println!("  ✓ Cost limits and warnings");
    println!("  ✓ Device cost comparison");
    println!("  ✓ Usage tracking and analytics");
    println!("  ✓ Budget optimization recommendations");

    // Example cost calculations
    println!("\nExample cost estimates:");
    println!("  IonQ QPU: ~$0.01 per shot");
    println!("  Rigetti QPU: ~$0.00035 per shot");
    println!("  Simulators: Usually $0.00 per shot");
    println!("  1000 shots on IonQ: ~$10.00");
    println!("  1000 shots on Rigetti: ~$0.35");

    // Example 9: Task monitoring and status tracking
    println!("\n9. Task Monitoring Features");
    println!("--------------------------");

    println!("Monitoring capabilities:");
    println!("  ✓ Real-time task status tracking");
    println!("  ✓ Queue time estimation");
    println!("  ✓ Execution progress monitoring");
    println!("  ✓ Task cancellation support");
    println!("  ✓ Performance metrics collection");
    println!("  ✓ Success rate tracking");

    // Example 10: Error handling and reliability
    println!("\n10. Error Handling and Reliability");
    println!("----------------------------------");

    println!("Reliability features:");
    println!("  ✓ AWS API error handling");
    println!("  ✓ Network failure recovery");
    println!("  ✓ Device unavailability handling");
    println!("  ✓ Cost limit enforcement");
    println!("  ✓ Timeout management");
    println!("  ✓ Comprehensive error reporting");

    // Example 11: Integration with AWS ecosystem
    println!("\n11. AWS Ecosystem Integration");
    println!("-----------------------------");

    println!("AWS integration features:");
    println!("  ✓ IAM role-based authentication");
    println!("  ✓ CloudWatch metrics integration");
    println!("  ✓ S3 result storage");
    println!("  ✓ Lambda function compatibility");
    println!("  ✓ Cost and billing integration");
    println!("  ✓ Multi-region support");

    // Example 12: Device capabilities and optimization
    println!("\n12. Device Capabilities and Optimization");
    println!("---------------------------------------");

    println!("Optimization features:");
    println!("  ✓ Automatic device selection based on problem size");
    println!("  ✓ Cost-performance optimization");
    println!("  ✓ Queue time minimization");
    println!("  ✓ Parallel task submission");
    println!("  ✓ Adaptive parameter tuning");
    println!("  ✓ Performance benchmarking");

    println!("\n✅ AWS Braket Client Example completed successfully!");
    println!("\nNote: This example demonstrates the API structure and capabilities.");
    println!("To run actual quantum annealing jobs, you need:");
    println!("  1. A valid AWS account with Braket access and credits");
    println!("  2. Properly configured AWS credentials");
    println!("  3. Uncomment the actual submission code");
    println!("  4. Handle the returned TaskResult objects");
    println!("  5. Monitor costs to avoid unexpected charges");

    Ok(())
}

#[cfg(feature = "braket")]
/// Helper function to demonstrate problem result analysis
fn analyze_task_results() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Task Result Analysis Features");
    println!("==============================");

    println!("Available analysis tools:");
    println!("  • Energy distribution analysis");
    println!("  • Solution quality metrics");
    println!("  • Performance benchmarking");
    println!("  • Statistical significance testing");
    println!("  • Cost-performance analysis");
    println!("  • Device comparison metrics");

    println!("\nMetrics automatically collected:");
    println!("  - Best energy found");
    println!("  - Average energy and standard deviation");
    println!("  - Solution success rate");
    println!("  - Execution time breakdown");
    println!("  - Cost per shot and total cost");
    println!("  - Queue time analysis");

    Ok(())
}

#[cfg(feature = "braket")]
/// Helper function to demonstrate cost optimization strategies
fn demonstrate_cost_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💰 Cost Optimization Strategies");
    println!("=============================");

    println!("Device selection strategies:");
    println!("  1. Use simulators for development and testing");
    println!("  2. Compare QPU costs: Rigetti < IonQ for most problems");
    println!("  3. Monitor queue times to avoid peak pricing");
    println!("  4. Batch similar problems for efficiency");

    println!("\nParameter optimization:");
    println!("  • Start with fewer shots for testing");
    println!("  • Use auto-scaling to optimize parameters");
    println!("  • Leverage default annealing schedules");
    println!("  • Monitor success rates vs. cost");

    println!("\nCost management features:");
    println!("  • Set cost limits to prevent overruns");
    println!("  • Track real-time spending");
    println!("  • Get cost estimates before submission");
    println!("  • Analyze cost-effectiveness reports");

    Ok(())
}

#[cfg(feature = "braket")]
/// Helper function to demonstrate AWS integration patterns
fn demonstrate_aws_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔗 AWS Integration Patterns");
    println!("=========================");

    println!("Authentication methods:");
    println!("  1. Environment variables (AWS_ACCESS_KEY_ID, etc.)");
    println!("  2. AWS credentials file (~/.aws/credentials)");
    println!("  3. IAM roles (for EC2, Lambda, etc.)");
    println!("  4. AWS CLI profile configuration");

    println!("\nService integration:");
    println!("  • S3: Store large problem instances and results");
    println!("  • Lambda: Serverless quantum computing workflows");
    println!("  • CloudWatch: Monitor performance and costs");
    println!("  • EventBridge: Trigger workflows on task completion");

    println!("\nBest practices:");
    println!("  • Use IAM roles with minimal required permissions");
    println!("  • Enable CloudTrail for audit logging");
    println!("  • Set up billing alerts for cost control");
    println!("  • Use multiple regions for availability");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_problem_creation() {
        let mut problem = IsingModel::new(4);
        assert!(problem.set_coupling(0, 1, -1.0).is_ok());
        assert!(problem.set_coupling(1, 2, -1.0).is_ok());
        assert_eq!(problem.num_qubits, 4);
    }

    #[cfg(feature = "braket")]
    #[test]
    fn test_device_selector_creation() {
        let selector = DeviceSelector {
            device_type: Some(DeviceType::QuantumProcessor),
            status: DeviceStatus::Online,
            max_cost_per_shot: Some(0.01),
            ..Default::default()
        };
        assert_eq!(selector.device_type, Some(DeviceType::QuantumProcessor));
        assert_eq!(selector.status, DeviceStatus::Online);
        assert_eq!(selector.max_cost_per_shot, Some(0.01));
    }

    #[cfg(feature = "braket")]
    #[test]
    fn test_advanced_params_creation() {
        let params = AdvancedAnnealingParams {
            shots: 1000,
            annealing_time: Some(20.0),
            auto_scale: Some(true),
            ..Default::default()
        };
        assert_eq!(params.shots, 1000);
        assert_eq!(params.annealing_time, Some(20.0));
        assert_eq!(params.auto_scale, Some(true));
    }

    #[cfg(feature = "braket")]
    #[test]
    fn test_cost_tracker_creation() {
        let tracker = CostTracker {
            cost_limit: Some(100.0),
            current_cost: 25.50,
            ..Default::default()
        };
        assert_eq!(tracker.cost_limit, Some(100.0));
        assert_eq!(tracker.current_cost, 25.50);
    }

    #[cfg(feature = "braket")]
    #[test]
    fn test_beta_schedule_creation() {
        let schedule = vec![(0.0, 0.1), (10.0, 1.0), (20.0, 10.0)];
        assert_eq!(schedule.len(), 3);
        assert_eq!(schedule[0], (0.0, 0.1));
        assert_eq!(schedule[2], (20.0, 10.0));
    }
}
