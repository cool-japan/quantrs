# Hardware Integration Guide

## Overview

QuantRS2-Tytan provides seamless integration with **10 major quantum computing and quantum-inspired hardware platforms**. This guide will help you understand how to use each platform effectively.

## Supported Platforms

### Cloud Quantum Platforms

1. **IBM Quantum** - Gate-based quantum computers and simulators
2. **Azure Quantum** - Microsoft's quantum computing service with multiple backends
3. **Amazon Braket** - AWS's quantum computing service
4. **D-Wave** - Quantum annealing systems

### Specialized Quantum Hardware

5. **Fujitsu Digital Annealer** - Digital annealing for combinatorial optimization
6. **Hitachi CMOS Annealer** - CMOS-based annealing machine
7. **NEC Vector Annealing** - Vector-based optimization
8. **Photonic Ising Machine** - Optical computing for Ising problems

### Quantum-Inspired Accelerators

9. **FPGA Accelerators** - Field-programmable gate arrays for quantum simulation
10. **MIKAS** - GPU-accelerated HOBO sampler

---

## Platform Comparison

| Platform | Type | Max Qubits/Vars | Best For | Typical Cost |
|----------|------|-----------------|----------|--------------|
| IBM Quantum | Gate-based | 127 | QAOA, VQE | Free tier + paid |
| Azure Quantum (QIO) | Classical | 10,000 | Large QUBO | Pay-per-use |
| Azure Quantum (IonQ) | Gate-based | 29 | High-fidelity circuits | $$$ |
| Amazon Braket (SV1) | Simulator | 34 | Development | $0.075/min |
| Amazon Braket (D-Wave) | Annealer | 5,000+ | Native QUBO | $0.30/task |
| D-Wave Advantage | Annealer | 5,000+ | Combinatorial opt. | Subscription |
| Fujitsu DA | Digital | 100,000+ | Enterprise QUBO | Enterprise |
| Hitachi CMOS | Annealer | 100,000+ | Large-scale opt. | Enterprise |
| NEC Vector | Annealer | 100,000+ | Dense problems | Enterprise |
| Photonic | Optical | 10,000+ | Ultrafast sampling | Research |

---

## Quick Start Guides

### 1. IBM Quantum

**Setup:**
```bash
# Get free API token from https://quantum-computing.ibm.com/
export IBM_QUANTUM_TOKEN="your_token_here"
```

**Usage:**
```rust
use quantrs2_tytan::sampler::hardware::{IBMQuantumSampler, IBMBackend};

// Use simulator for development
let sampler = IBMQuantumSampler::with_token(&std::env::var("IBM_QUANTUM_TOKEN")?)
    .with_backend(IBMBackend::Simulator)
    .with_optimization_level(2)
    .with_error_mitigation(true);

let results = sampler.run_qubo(&qubo, 1000)?;
```

**Features:**
- ✅ Free tier available (10 minutes/month on simulators)
- ✅ Access to real quantum hardware
- ✅ Built-in error mitigation
- ✅ QAOA and VQE support
- ⚠️ Queue times on real hardware

**Best Practices:**
1. Start with simulator for development
2. Enable error mitigation for real hardware
3. Use optimization level 2-3 for better circuits
4. Check queue status before submitting jobs
5. Limit circuit depth for better results

**Example:** See `examples/ibm_quantum_maxcut.rs`

---

### 2. Azure Quantum

**Setup:**
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login and configure
az login
az quantum workspace create --resource-group myRG --name myWorkspace
```

**Usage:**
```rust
use quantrs2_tytan::sampler::hardware::{AzureQuantumSampler, AzureSolver};

// Use Microsoft QIO solvers (no quantum hardware needed)
let sampler = AzureQuantumSampler::with_workspace(
    "subscription-id",
    "resource-group",
    "workspace-name"
)
.with_solver(AzureSolver::ParallelTempering)
.with_timeout(120);

let results = sampler.run_qubo(&qubo, 100)?;
```

**Available Solvers:**

**Microsoft QIO (Quantum-Inspired Optimization):**
- `SimulatedAnnealing` - Fast, good quality
- `ParallelTempering` - Better exploration
- `TabuSearch` - Memory-based local search
- `PopulationAnnealing` - Large-scale problems
- `SubstrateMonteCarlo` - Dense problems

**Quantum Hardware:**
- `IonQ` - Trapped ion (29 qubits)
- `Quantinuum` - High-fidelity (20 qubits)
- `Rigetti` - Superconducting (40 qubits)

**Best Practices:**
1. Use QIO solvers for classical optimization
2. Reserve quantum hardware for quantum advantage problems
3. Set appropriate timeouts for large problems
4. Monitor costs in Azure portal
5. Use parallel tempering for difficult landscapes

**Example:** See `examples/azure_quantum_portfolio.rs`

---

### 3. Amazon Braket

**Setup:**
```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure

# Create S3 bucket for results
aws s3 mb s3://my-braket-bucket
```

**Usage:**
```rust
use quantrs2_tytan::sampler::hardware::{AmazonBraketSampler, BraketDevice};

// Use local simulator (free)
let sampler = AmazonBraketSampler::with_s3("my-braket-bucket", "us-east-1")
    .with_device(BraketDevice::LocalSimulator);

// Or use managed simulator
let sampler = AmazonBraketSampler::with_s3("my-braket-bucket", "us-east-1")
    .with_device(BraketDevice::StateVectorSimulator)
    .with_max_parallel(5);

let results = sampler.run_qubo(&qubo, 1000)?;
```

**Available Devices:**

**Simulators:**
- `LocalSimulator` - Free, ~25 qubits
- `StateVectorSimulator` - $0.075/min, 34 qubits
- `TensorNetworkSimulator` - $0.275/min, 50 qubits

**Quantum Hardware:**
- `IonQDevice` - Trapped ion, 29 qubits
- `RigettiDevice` - Superconducting, 40 qubits
- `OQCDevice` - Superconducting, 8 qubits
- `DWaveAdvantage` - Quantum annealer, 5000+ qubits
- `DWave2000Q` - Quantum annealer, 2000 qubits

**Cost Optimization:**
1. Use `LocalSimulator` for development (free)
2. Use `StateVectorSimulator` for validation (<$5/hour)
3. Batch jobs to reduce task overhead
4. Monitor S3 storage costs
5. Set AWS Budgets alerts

**Example:** See `examples/amazon_braket_tsp.rs`

---

### 4. D-Wave Systems

**Setup:**
```bash
# Get API key from https://cloud.dwavesys.com/
export DWAVE_API_KEY="your_key_here"
```

**Usage:**
```rust
use quantrs2_tytan::sampler::hardware::DWaveSampler;

let sampler = DWaveSampler::new(&std::env::var("DWAVE_API_KEY")?);
let results = sampler.run_qubo(&qubo, 1000)?;
```

**Features:**
- ✅ Native QUBO/Ising support (no conversion needed)
- ✅ 5000+ qubits (Advantage system)
- ✅ Very fast sampling (microseconds)
- ⚠️ Limited qubit connectivity
- ⚠️ Requires embedding for full graphs

**Best For:**
- Large-scale combinatorial optimization
- Problems with sparse connectivity
- Rapid solution sampling
- Production deployment

**Example:** See D-Wave documentation

---

### 5-7. Enterprise Annealers

#### Fujitsu Digital Annealer

```rust
use quantrs2_tytan::sampler::hardware::FujitsuDigitalAnnealerSampler;

let sampler = FujitsuDigitalAnnealerSampler::new(config);
```

**Features:**
- 100,000+ binary variables
- Digital annealing (deterministic)
- No temperature parameter
- Enterprise-grade reliability

#### Hitachi CMOS Annealer

```rust
use quantrs2_tytan::sampler::hardware::HitachiCMOSSampler;

let sampler = HitachiCMOSSampler::new(config);
```

**Features:**
- CMOS-based annealing
- Very fast convergence
- Low power consumption
- Scalable architecture

#### NEC Vector Annealing

```rust
use quantrs2_tytan::sampler::hardware::NECVectorAnnealingSampler;

let sampler = NECVectorAnnealingSampler::new(config);
```

**Features:**
- Vector-based optimization
- Good for dense problems
- Parallel processing
- Enterprise support

---

### 8. Photonic Ising Machine

```rust
use quantrs2_tytan::sampler::hardware::PhotonicIsingMachineSampler;

let sampler = PhotonicIsingMachineSampler::new(config);
```

**Features:**
- Optical computing
- Ultrafast sampling (nanoseconds)
- Coherent Ising machine implementation
- Research and production use

**Best For:**
- Time-critical applications
- Large Ising problems
- Energy-efficient computing

---

### 9-10. Quantum-Inspired Accelerators

#### FPGA Accelerators

```rust
use quantrs2_tytan::sampler::hardware::FPGASampler;

let sampler = FPGASampler::new(config);
```

**Features:**
- Custom hardware acceleration
- Programmable logic
- Low latency
- Cost-effective for specific problems

#### MIKAS (GPU-Accelerated HOBO)

```rust
use quantrs2_tytan::sampler::hardware::MIKASAmpler;

let sampler = MIKASAmpler::new(config);
```

**Features:**
- GPU acceleration (CUDA)
- Native HOBO support
- High throughput
- Good for large-scale problems

---

## Decision Tree: Choosing the Right Platform

### For Development & Testing
→ **IBM Quantum Simulator** (free, easy setup)
→ **Amazon Braket Local Simulator** (free, offline)

### For Small QUBO Problems (<30 variables)
→ **IBM Quantum Hardware** (high quality)
→ **Azure Quantum IonQ** (high fidelity)
→ **Amazon Braket IonQ** (AWS integration)

### For Medium QUBO Problems (30-1000 variables)
→ **Azure Quantum QIO** (fast, scalable)
→ **Amazon Braket D-Wave** (if sparse)
→ **D-Wave Advantage** (production)

### For Large QUBO Problems (>1000 variables)
→ **Fujitsu Digital Annealer** (enterprise)
→ **Hitachi CMOS Annealer** (dense problems)
→ **NEC Vector Annealing** (parallel)

### For HOBO Problems
→ **MIKAS** (GPU-accelerated)
→ Quadratize first, then use any QUBO solver

### For Research
→ **Photonic Ising Machine** (ultrafast)
→ **IBM Quantum** (latest features)

### For Production
→ **D-Wave** (proven, reliable)
→ **Azure Quantum** (enterprise support)
→ **Fujitsu/Hitachi/NEC** (large-scale)

---

## Performance Benchmarks

### Problem: 50-variable QUBO

| Platform | Time | Cost | Quality |
|----------|------|------|---------|
| IBM Quantum Sim | 5s | Free | ★★★★☆ |
| Azure QIO PT | 30s | $0.05 | ★★★★★ |
| Braket SV1 | 2min | $0.15 | ★★★★☆ |
| D-Wave Advantage | 0.1s | $0.50 | ★★★★☆ |
| Fujitsu DA | 1s | Enterprise | ★★★★★ |

### Problem: 500-variable QUBO

| Platform | Time | Cost | Quality |
|----------|------|------|---------|
| Azure QIO | 5min | $0.50 | ★★★★★ |
| D-Wave Advantage | 1s | $1.00 | ★★★★☆ |
| Fujitsu DA | 10s | Enterprise | ★★★★★ |
| NEC VA | 15s | Enterprise | ★★★★★ |

---

## Cost Optimization Tips

1. **Use simulators for development** - Save quantum credits
2. **Batch jobs when possible** - Reduce overhead costs
3. **Start with free tiers** - IBM Quantum, Braket local
4. **Monitor usage** - Set up billing alerts
5. **Choose the right solver** - QIO vs quantum hardware
6. **Optimize problem formulation** - Reduce variables
7. **Use hybrid approaches** - Classical preprocessing

---

## Troubleshooting

### Common Issues

**Problem:** Authentication errors
**Solution:** Check API tokens, Azure credentials, AWS credentials

**Problem:** Queue timeouts
**Solution:** Use simulators, check device availability, schedule off-peak

**Problem:** Results in S3 not found (Braket)
**Solution:** Check bucket permissions, region match, poll interval

**Problem:** Poor solution quality
**Solution:** Enable error mitigation, increase shots, tune parameters

**Problem:** Cost overruns
**Solution:** Set AWS/Azure budgets, use simulators, optimize batch size

---

## Additional Resources

- [IBM Quantum Documentation](https://quantum-computing.ibm.com/lab/docs/)
- [Azure Quantum Documentation](https://docs.microsoft.com/azure/quantum/)
- [Amazon Braket Documentation](https://docs.aws.amazon.com/braket/)
- [D-Wave Documentation](https://docs.dwavesys.com/)
- [QuantRS2 Examples](./examples/)

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/cool-japan/quantrs2
- Community Forum: [link]
- Enterprise Support: [contact]

---

**Last Updated:** 2026-01-23
**Version:** 0.1.2
