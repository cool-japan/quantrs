# AWS Braket Quantum Annealing Client

This document describes the comprehensive AWS Braket quantum annealing client implementation in the QuantRS2 anneal module.

## Overview

The AWS Braket client provides a complete interface to Amazon's quantum computing services, including:

- **Quantum Processing Units (QPUs)**: Access to quantum annealing hardware from various providers
- **Quantum Simulators**: Classical simulation of quantum annealing for development and testing
- **Advanced Device Management**: Intelligent device selection and cost optimization
- **Task Management**: Status tracking, batch submission, and monitoring
- **Cost Tracking**: Real-time cost estimation and budget management
- **AWS Integration**: Seamless integration with AWS ecosystem and services

## Key Features

### 1. Enhanced Device Management

```rust
use quantrs2_anneal::braket::{BraketClient, DeviceSelector, DeviceType, DeviceStatus};

let selector = DeviceSelector {
    device_type: Some(DeviceType::QuantumProcessor),
    provider: Some("IonQ".to_string()),
    status: DeviceStatus::Online,
    min_fidelity: Some(0.95),
    max_cost_per_shot: Some(0.01),
    required_capabilities: vec!["ANNEALING".to_string()],
};

let client = BraketClient::new(access_key, secret_key, "us-east-1")?;
let best_device = client.select_device(Some(&selector))?;
```

### 2. Cost Tracking and Management

```rust
use quantrs2_anneal::braket::CostTracker;

let cost_tracker = CostTracker {
    cost_limit: Some(100.0), // $100 budget limit
    current_cost: 0.0,
    cost_estimates: HashMap::new(),
};

let client = BraketClient::with_config(
    access_key,
    secret_key,
    None,
    "us-east-1",
    DeviceSelector::default(),
    cost_tracker,
)?;
```

### 3. Advanced Annealing Parameters

```rust
use quantrs2_anneal::braket::AdvancedAnnealingParams;

let params = AdvancedAnnealingParams {
    shots: 1000,
    annealing_time: Some(20.0),
    programming_thermalization: Some(1000.0),
    readout_thermalization: Some(100.0),
    beta_schedule: Some(vec![
        (0.0, 0.1),   // Start with high temperature
        (10.0, 1.0),  // Linear cooling
        (20.0, 10.0), // End with low temperature
    ]),
    auto_scale: Some(true),
    flux_biases: Some(flux_bias_map),
    extra: extra_params,
};
```

### 4. Quantum Task Submission

```rust
// Submit Ising model
let task_result = client.submit_ising(
    &ising_model,
    None, // Auto-select device
    Some(params),
)?;

// Submit QUBO model
let task_result = client.submit_qubo(
    &qubo_model,
    Some("arn:aws:braket::device/qpu/ionq/ionQdevice"),
    Some(params),
)?;
```

### 5. Task Status Monitoring

```rust
// Submit task asynchronously
let task_result = client.submit_ising(&model, None, None)?;
let task_arn = &task_result.task_arn;

// Monitor status
loop {
    let status = client.get_task_status(task_arn)?;
    match status.task_status {
        TaskStatus::Completed => break,
        TaskStatus::Failed => return Err("Task failed"),
        TaskStatus::Cancelled => return Err("Task cancelled"),
        _ => std::thread::sleep(Duration::from_secs(5)),
    }
}

// Get detailed metrics
let metrics = client.get_task_metrics(task_arn)?;
println!("Total cost: ${:.4}", metrics.cost);
println!("Best energy: {:.4}", metrics.best_energy.unwrap_or(0.0));
println!("Execution time: {:?}", metrics.execution_time);
```

### 6. Batch Task Submission

```rust
let tasks = vec![
    (&model1, None, None),
    (&model2, Some("device_arn"), Some(params)),
    (&model3, None, None),
];

let batch_result = client.submit_batch(tasks)?;
println!("Total estimated cost: ${:.4}", batch_result.estimated_cost);

for (i, status) in batch_result.statuses.iter().enumerate() {
    match status {
        Ok(task_arn) => println!("Task {}: {}", i, task_arn),
        Err(e) => println!("Task {} failed: {}", i, e),
    }
}
```

## API Reference

### Core Types

#### `BraketClient`
The main client for interacting with AWS Braket services.

**Methods:**
- `new(access_key, secret_key, region)` - Create basic client
- `with_config(...)` - Create with custom configuration
- `get_devices()` - Get all available devices
- `select_device(selector)` - Select optimal device based on criteria
- `submit_ising(model, device_arn, params)` - Submit Ising model
- `submit_qubo(model, device_arn, params)` - Submit QUBO model
- `get_task_status(task_arn)` - Get task status
- `get_task_metrics(task_arn)` - Get performance and cost metrics
- `submit_batch(tasks)` - Submit multiple tasks
- `cancel_task(task_arn)` - Cancel running task
- `list_tasks(limit)` - List recent tasks
- `get_cost_summary()` - Get cost and usage information

#### `DeviceSelector`
Criteria for automatic device selection.

**Fields:**
- `device_type: Option<DeviceType>` - Filter by device type
- `provider: Option<String>` - Filter by provider (IonQ, Rigetti, etc.)
- `status: DeviceStatus` - Only consider devices with this status
- `min_fidelity: Option<f64>` - Minimum gate fidelity requirement
- `max_cost_per_shot: Option<f64>` - Maximum acceptable cost per shot
- `required_capabilities: Vec<String>` - Required device capabilities

#### `AdvancedAnnealingParams`
Advanced parameters for quantum annealing tasks.

**Fields:**
- `shots: usize` - Number of measurements
- `annealing_time: Option<f64>` - Annealing time in microseconds
- `programming_thermalization: Option<f64>` - Programming thermalization time
- `readout_thermalization: Option<f64>` - Readout thermalization time
- `beta_schedule: Option<Vec<(f64, f64)>>` - Custom temperature schedule
- `s_schedule: Option<Vec<(f64, f64)>>` - Custom transverse field schedule
- `auto_scale: Option<bool>` - Enable automatic parameter scaling
- `flux_biases: Option<HashMap<String, f64>>` - Per-qubit flux biases

#### `TaskMetrics`
Performance and cost metrics for completed tasks.

**Fields:**
- `total_time: Duration` - Total execution time
- `queue_time: Duration` - Time spent in queue
- `execution_time: Duration` - Time spent executing on device
- `cost: f64` - Total cost in USD
- `success_rate: f64` - Task success rate
- `average_energy: Option<f64>` - Average energy of solutions
- `best_energy: Option<f64>` - Best energy found
- `energy_std: Option<f64>` - Standard deviation of energies

#### `CostTracker`
Cost tracking and budget management.

**Fields:**
- `cost_limit: Option<f64>` - Maximum spending limit
- `current_cost: f64` - Current accumulated cost
- `cost_estimates: HashMap<String, f64>` - Per-device cost estimates

### Device Types

#### `DeviceType`
Types of AWS Braket devices.

**Variants:**
- `QuantumProcessor` - Quantum processing units (real quantum hardware)
- `Simulator` - Classical simulators

#### `DeviceStatus`
Status of AWS Braket devices.

**Variants:**
- `Online` - Device is available for use
- `Offline` - Device is temporarily unavailable
- `Retired` - Device is permanently unavailable

#### `TaskStatus`
Status of quantum tasks.

**Variants:**
- `Created` - Task has been created
- `Queued` - Task is waiting in queue
- `Running` - Task is currently executing
- `Completed` - Task completed successfully
- `Failed` - Task execution failed
- `Cancelled` - Task was cancelled

### Error Handling

The client provides comprehensive error handling through the `BraketError` enum:

```rust
use quantrs2_anneal::braket::BraketError;

match client.submit_ising(&model, None, None) {
    Ok(task_result) => {
        // Process task result
    }
    Err(BraketError::CostLimitError(msg)) => {
        // Handle cost limit exceeded
    }
    Err(BraketError::DeviceConfigError(msg)) => {
        // Handle device configuration issues
    }
    Err(BraketError::TaskError(msg)) => {
        // Handle task execution errors
    }
    Err(e) => {
        // Handle other errors
    }
}
```

## Usage Patterns

### 1. Simple QPU Access

```rust
use quantrs2_anneal::{IsingModel, braket::BraketClient};

let mut model = IsingModel::new(4);
model.set_coupling(0, 1, -1.0)?;
model.set_coupling(1, 2, -1.0)?;
model.set_coupling(2, 3, -1.0)?;
model.set_coupling(3, 0, -1.0)?;

let client = BraketClient::new(access_key, secret_key, "us-east-1")?;
let task_result = client.submit_ising(&model, None, None)?;

let completed_task = client.get_task_result(&task_result.task_arn)?;
println!("Task completed with {} measurements", 
         completed_task.measurements.as_ref().map(|m| m.len()).unwrap_or(0));
```

### 2. Cost-Optimized Simulator Testing

```rust
let simulator_selector = DeviceSelector {
    device_type: Some(DeviceType::Simulator),
    status: DeviceStatus::Online,
    ..Default::default()
};

let test_params = AdvancedAnnealingParams {
    shots: 10000, // More shots since simulators are cheap
    annealing_time: Some(1.0), // Fast simulation
    auto_scale: Some(true),
    ..Default::default()
};

let client = BraketClient::new(access_key, secret_key, "us-east-1")?;
let task_result = client.submit_ising(&large_model, None, Some(test_params))?;
```

### 3. Cost-Controlled QPU Usage

```rust
let cost_tracker = CostTracker {
    cost_limit: Some(50.0), // $50 budget
    ..Default::default()
};

let qpu_selector = DeviceSelector {
    device_type: Some(DeviceType::QuantumProcessor),
    max_cost_per_shot: Some(0.001), // Prefer cheaper devices
    status: DeviceStatus::Online,
    required_capabilities: vec!["ANNEALING".to_string()],
};

let client = BraketClient::with_config(
    access_key,
    secret_key,
    None,
    "us-east-1",
    qpu_selector,
    cost_tracker,
)?;

let optimized_params = AdvancedAnnealingParams {
    shots: 100, // Start with fewer shots
    auto_scale: Some(true),
    ..Default::default()
};

let task_result = client.submit_ising(&model, None, Some(optimized_params))?;
```

### 4. Advanced Temperature Scheduling

```rust
let custom_beta_schedule = vec![
    (0.0, 0.01),    // Start very hot
    (5.0, 0.1),     // Rapid initial cooling
    (15.0, 1.0),    // Moderate cooling
    (18.0, 5.0),    // Slow final cooling
    (20.0, 10.0),   // Final low temperature
];

let advanced_params = AdvancedAnnealingParams {
    shots: 1000,
    annealing_time: Some(20.0),
    beta_schedule: Some(custom_beta_schedule),
    programming_thermalization: Some(2000.0),
    readout_thermalization: Some(200.0),
    auto_scale: Some(false), // Manual scaling
    ..Default::default()
};

let task_result = client.submit_ising(&complex_model, None, Some(advanced_params))?;
```

### 5. Comprehensive Monitoring and Analysis

```rust
// Submit task
let task_result = client.submit_ising(&model, None, None)?;
let task_arn = &task_result.task_arn;

// Wait for completion with timeout
let start = std::time::Instant::now();
let timeout = Duration::from_secs(600); // 10 minutes

loop {
    let status = client.get_task_status(task_arn)?;
    
    match status.task_status {
        TaskStatus::Completed => {
            let metrics = client.get_task_metrics(task_arn)?;
            
            println!("Task completed successfully!");
            println!("Total cost: ${:.4}", metrics.cost);
            println!("Queue time: {:?}", metrics.queue_time);
            println!("Execution time: {:?}", metrics.execution_time);
            
            if let Some(best_energy) = metrics.best_energy {
                println!("Best energy: {:.6}", best_energy);
            }
            if let Some(avg_energy) = metrics.average_energy {
                println!("Average energy: {:.6}", avg_energy);
            }
            if let Some(std_energy) = metrics.energy_std {
                println!("Energy std dev: {:.6}", std_energy);
            }
            
            println!("Success rate: {:.2}%", metrics.success_rate * 100.0);
            break;
        }
        TaskStatus::Failed => {
            return Err("Task execution failed".into());
        }
        TaskStatus::Cancelled => {
            return Err("Task was cancelled".into());
        }
        _ => {
            if start.elapsed() > timeout {
                client.cancel_task(task_arn)?;
                return Err("Task timed out".into());
            }
            println!("Task status: {:?}, waiting...", status.task_status);
            std::thread::sleep(Duration::from_secs(10));
        }
    }
}
```

## Best Practices

### 1. Cost Management

- Set reasonable cost limits to prevent unexpected charges
- Use simulators for development and testing
- Compare device costs: Rigetti QPUs are typically cheaper than IonQ
- Monitor queue times as they affect effective cost per result
- Start with fewer shots and scale up based on results

### 2. Device Selection

- Use `DeviceSelector` to automatically choose optimal devices
- Consider both cost and performance requirements
- Check device availability and queue times before submission
- Prefer newer device generations for better performance

### 3. Parameter Optimization

- Start with default parameters and adjust based on problem characteristics
- Use auto-scaling for most problems unless you have specific requirements
- Custom temperature schedules can improve solution quality for complex problems
- Balance number of shots with cost constraints

### 4. Error Handling

- Implement retry logic for network failures
- Have fallback strategies for device unavailability
- Monitor cost limits to prevent overruns
- Handle task failures gracefully with alternative approaches

### 5. Performance Monitoring

- Track task metrics to optimize submission parameters
- Monitor success rates and adjust parameters accordingly
- Analyze cost-effectiveness of different devices and parameter settings
- Use batch submission for multiple similar problems

## Integration Examples

### With AWS Lambda

```rust
use quantrs2_anneal::{
    ising::IsingModel,
    braket::BraketClient,
};

// Lambda function for quantum annealing
pub async fn lambda_handler(event: LambdaEvent<Value>) -> Result<Value, Error> {
    let client = BraketClient::new(
        std::env::var("AWS_ACCESS_KEY_ID")?,
        std::env::var("AWS_SECRET_ACCESS_KEY")?,
        std::env::var("AWS_REGION")?,
    )?;
    
    // Parse problem from event
    let model = parse_ising_from_event(&event.payload)?;
    
    // Submit to Braket
    let task_result = client.submit_ising(&model, None, None)?;
    
    // Return task ARN for async monitoring
    Ok(serde_json::json!({
        "taskArn": task_result.task_arn,
        "status": "submitted"
    }))
}
```

### With S3 Storage

```rust
// Store large problems and results in S3
let problem_key = format!("problems/{}.json", problem_id);
let result_key = format!("results/{}.json", task_arn);

// Submit task with S3 integration
let task_result = client.submit_ising(&model, None, None)?;

// Store task metadata
let metadata = serde_json::json!({
    "task_arn": task_result.task_arn,
    "problem_id": problem_id,
    "submitted_at": chrono::Utc::now(),
    "device_arn": device_arn,
});
```

### With CloudWatch Monitoring

```rust
// Publish custom metrics to CloudWatch
let metrics = client.get_task_metrics(&task_arn)?;

publish_metric("QuantumAnnealing/Cost", metrics.cost)?;
publish_metric("QuantumAnnealing/ExecutionTime", metrics.execution_time.as_secs_f64())?;
publish_metric("QuantumAnnealing/SuccessRate", metrics.success_rate)?;

if let Some(best_energy) = metrics.best_energy {
    publish_metric("QuantumAnnealing/BestEnergy", best_energy)?;
}
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify AWS credentials are correctly configured
   - Check IAM permissions for Braket service access
   - Ensure region is supported for Braket

2. **Cost Limit Errors**
   - Review and adjust cost limits
   - Use simulators for testing to reduce costs
   - Monitor spending in AWS billing dashboard

3. **Device Unavailability**
   - Use `DeviceSelector` with fallback options
   - Check device status before submission
   - Consider using simulators when QPUs are unavailable

4. **Task Failures**
   - Verify problem formulation is correct
   - Check device capabilities match problem requirements
   - Review task logs in AWS console for specific error messages

### Performance Optimization

1. **For Small Problems (< 50 variables)**
   - Use simulators for development
   - Consider both IonQ and Rigetti QPUs for production
   - Start with default parameters

2. **For Large Problems (> 100 variables)**
   - Use simulators for initial testing
   - May require problem decomposition for current QPUs
   - Consider hybrid classical-quantum approaches

3. **For Cost-Sensitive Applications**
   - Prefer Rigetti QPUs over IonQ for cost optimization
   - Use batch submission for multiple similar problems
   - Implement adaptive shot count based on problem complexity

## AWS Setup and Configuration

### Required AWS Permissions

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "braket:GetDevice",
                "braket:SearchDevices",
                "braket:CreateQuantumTask",
                "braket:GetQuantumTask",
                "braket:CancelQuantumTask",
                "braket:SearchQuantumTasks"
            ],
            "Resource": "*"
        }
    ]
}
```

### Environment Variables

```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_REGION="us-east-1"  # or other Braket-supported region
```

### AWS CLI Configuration

```bash
aws configure set aws_access_key_id your-access-key-id
aws configure set aws_secret_access_key your-secret-access-key
aws configure set region us-east-1
```

## License and Usage

The AWS Braket client requires:
- A valid AWS account with Braket service access
- Appropriate IAM permissions
- The `braket` feature enabled in Cargo.toml
- Compliance with AWS terms of service and pricing

For more information, visit [AWS Braket Documentation](https://docs.aws.amazon.com/braket/).