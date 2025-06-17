use quantrs2_circuit::prelude::Circuit;
use quantrs2_device::{aws_device::AWSDeviceConfig, create_aws_client, create_aws_device, CircuitExecutor};
use std::env;

/// This example demonstrates how to create and use an AWS Braket quantum device.
///
/// To run this example, you need to have AWS credentials set up.
/// You can set them using environment variables or the AWS credentials file.
///
/// Required environment variables:
/// - AWS_ACCESS_KEY_ID
/// - AWS_SECRET_ACCESS_KEY
/// - AWS_S3_BUCKET - S3 bucket for storing results
///
/// Optional environment variables:
/// - AWS_REGION - AWS region to use (default: us-east-1)
/// - AWS_S3_KEY_PREFIX - S3 key prefix for results (default: quantrs2)
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("QuantRS2 AWS Example");
    println!("====================");

    // Get AWS credentials from environment variables
    let aws_access_key = env::var("AWS_ACCESS_KEY_ID").expect("AWS_ACCESS_KEY_ID not set");
    let aws_secret_key = env::var("AWS_SECRET_ACCESS_KEY").expect("AWS_SECRET_ACCESS_KEY not set");
    let aws_s3_bucket = env::var("AWS_S3_BUCKET").expect("AWS_S3_BUCKET not set");
    let aws_region = env::var("AWS_REGION").ok();
    let aws_s3_key_prefix = env::var("AWS_S3_KEY_PREFIX").ok();

    // Create AWS client
    println!("Creating AWS Braket client...");
    let client = create_aws_client(
        &aws_access_key,
        &aws_secret_key,
        aws_region.as_deref(),
        &aws_s3_bucket,
        aws_s3_key_prefix.as_deref(),
    )?;

    // Create device configuration
    let config = AWSDeviceConfig {
        default_shots: 1000,
        ir_type: "BRAKET".to_string(),
        device_parameters: None,
        timeout_secs: Some(180),
    };

    // List available simulators
    println!("Listing available simulators...");
    #[cfg(feature = "aws")]
    {
        let devices = client.list_devices().await?;
        println!("Available simulators:");
        for device in devices.iter().filter(|d| d.device_type == "SIMULATOR") {
            println!("- {} ({})", device.name, device.device_arn);
        }
        println!();

        // Select a simulator from the list
        let simulator = devices
            .iter()
            .find(|d| d.device_type == "SIMULATOR" && d.name.contains("SV1"))
            .expect("No SV1 simulator found");

        println!(
            "Using simulator: {} ({})",
            simulator.name, simulator.device_arn
        );

        // Create AWS device
        println!("Creating AWS device...");
        let device = create_aws_device(client, &simulator.device_arn, Some(config)).await?;

        // Create a simple circuit
        println!("Creating a Bell state circuit...");
        let mut circuit = Circuit::<2>::new();
        circuit.h(0)?;
        circuit.cnot(0, 1)?;

        // Execute circuit on the device
        println!("Executing circuit on AWS Braket...");
        let result = device.execute_circuit(&circuit, 1000).await?;

        // Print results
        println!("Results:");
        for (state, count) in result.counts.iter() {
            println!(
                "- |{}⟩: {} ({:.1}%)",
                state,
                count,
                (*count as f64 / 1000.0) * 100.0
            );
        }
    }

    #[cfg(not(feature = "aws"))]
    {
        println!("This example requires the 'aws' feature to be enabled.");
        println!("Recompile with: cargo build --example aws_example --features aws");
    }

    Ok(())
}
