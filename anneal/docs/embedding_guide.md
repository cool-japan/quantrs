# Graph Embedding Guide for Quantum Annealing

This comprehensive guide covers graph embedding techniques in QuantRS2-Anneal, including minor graph embedding, layout optimization, and hardware-aware mapping.

## Overview

Graph embedding is the process of mapping a logical problem graph onto a physical quantum annealing hardware topology. This is crucial for:

- **Hardware Constraints**: Quantum annealers have limited connectivity
- **Chain Formation**: Logical variables are represented by chains of physical qubits
- **Performance Optimization**: Good embeddings reduce chain breaks and improve solution quality

## Embedding Modules

### 1. Core Embedding (`embedding.rs`)

The core embedding module provides fundamental embedding algorithms and data structures.

#### Key Types

```rust
use quantrs2_anneal::embedding::{
    Embedding, HardwareGraph, MinorMiner, EmbeddingResult, EmbeddingConfig
};

// Define the problem graph
let logical_edges = vec![
    (0, 1), (1, 2), (2, 3), (3, 0), // Square connectivity
    (0, 2), (1, 3), // Cross connections
];

// Create hardware graph (Chimera topology)
let hardware = HardwareGraph::new_chimera(4, 4, 4)?;

// Configure embedding parameters
let config = EmbeddingConfig {
    max_tries: 100,
    timeout_ms: 30000,
    chain_strength_range: (0.1, 10.0),
    minimize_chains: true,
    random_seed: Some(42),
};

// Find embedding
let embedder = MinorMiner::new(config);
let embedding = embedder.find_embedding(&logical_edges, 4, &hardware)?;

println!("Embedding chains:");
for (logical_var, physical_chain) in &embedding.chains {
    println!("  Variable {}: {:?}", logical_var, physical_chain);
}
```

#### Hardware Topologies

```rust
// D-Wave topologies
let chimera = HardwareGraph::new_chimera(16, 16, 4)?;      // 16x16 Chimera
let pegasus = HardwareGraph::new_pegasus(6)?;             // Pegasus P6
let zephyr = HardwareGraph::new_zephyr(4)?;               // Zephyr Z4

// Custom topology
let custom_edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)]; // Ring
let custom = HardwareGraph::new_custom(4, custom_edges);

// Analyze topology properties
println!("Chimera connectivity: {}", chimera.average_degree());
println!("Pegasus max degree: {}", pegasus.max_degree());
println!("Zephyr diameter: {}", zephyr.diameter());
```

#### Embedding Quality Metrics

```rust
use quantrs2_anneal::embedding::EmbeddingMetrics;

let metrics = EmbeddingMetrics::analyze(&embedding, &hardware);

println!("Embedding Quality Report:");
println!("  Total physical qubits used: {}", metrics.qubits_used);
println!("  Average chain length: {:.2}", metrics.avg_chain_length);
println!("  Maximum chain length: {}", metrics.max_chain_length);
println!("  Chain length variance: {:.2}", metrics.chain_length_variance);
println!("  Hardware utilization: {:.1}%", metrics.utilization_percent);
println!("  Expected chain break rate: {:.3}", metrics.expected_chain_breaks);
```

### 2. Layout Embedding (`layout_embedding.rs`)

Layout embedding optimizes embeddings based on hardware-specific layout considerations.

#### Layout-Aware Embedding

```rust
use quantrs2_anneal::layout_embedding::{
    LayoutEmbedder, LayoutConstraints, TopologyLayout, PlacementStrategy
};

// Define layout constraints
let constraints = LayoutConstraints {
    preferred_regions: vec![
        (0..100, 0..100),   // Top-left quadrant
        (400..500, 400..500), // Bottom-right quadrant
    ],
    forbidden_qubits: vec![50, 51, 52], // Defective qubits
    chain_length_penalty: 2.0,
    crossing_penalty: 1.5,
    placement_strategy: PlacementStrategy::MinimizeChainLength,
};

// Create layout-aware embedder
let layout_embedder = LayoutEmbedder::new(
    &hardware,
    constraints,
    TopologyLayout::Chimera { m: 16, n: 16, t: 4 },
)?;

// Find optimized embedding
let embedding = layout_embedder.embed_with_layout(&logical_edges, num_vars)?;

// Analyze layout quality
let layout_metrics = layout_embedder.analyze_layout(&embedding);
println!("Layout efficiency: {:.2}", layout_metrics.efficiency_score);
println!("Region utilization: {:?}", layout_metrics.region_usage);
```

#### Placement Strategies

```rust
// Different placement strategies for different objectives
let strategies = vec![
    PlacementStrategy::MinimizeChainLength,     // Shortest chains
    PlacementStrategy::MinimizeSpread,          // Compact placement
    PlacementStrategy::BalanceLoad,            // Even qubit usage
    PlacementStrategy::AvoidDefects,           // Work around defective qubits
    PlacementStrategy::MinimizeCrossings,      // Reduce chain crossings
];

for strategy in strategies {
    let constraints = LayoutConstraints {
        placement_strategy: strategy,
        ..Default::default()
    };
    
    let embedder = LayoutEmbedder::new(&hardware, constraints, layout)?;
    let embedding = embedder.embed_with_layout(&logical_edges, num_vars)?;
    
    println!("Strategy {:?}: efficiency = {:.3}", 
             strategy, 
             embedder.analyze_layout(&embedding).efficiency_score);
}
```

### 3. Advanced Embedding Techniques

#### Multi-Objective Embedding

```rust
use quantrs2_anneal::embedding::{MultiObjectiveEmbedder, EmbeddingObjective};

let objectives = vec![
    EmbeddingObjective::MinimizeChainLength { weight: 0.4 },
    EmbeddingObjective::MinimizeQubitsUsed { weight: 0.3 },
    EmbeddingObjective::BalanceChainLengths { weight: 0.2 },
    EmbeddingObjective::MinimizeCrossings { weight: 0.1 },
];

let mo_embedder = MultiObjectiveEmbedder::new(objectives);
let pareto_embeddings = mo_embedder.find_pareto_optimal_embeddings(
    &logical_edges, 
    num_vars, 
    &hardware,
    50 // number of solutions
)?;

// Select best embedding based on problem characteristics
let best_embedding = mo_embedder.select_embedding(
    &pareto_embeddings,
    &problem_characteristics,
)?;
```

#### Hierarchical Embedding

```rust
use quantrs2_anneal::embedding::HierarchicalEmbedder;

// For large problems, use hierarchical decomposition
let hierarchical = HierarchicalEmbedder::new(
    &hardware,
    max_subproblem_size: 100,
    overlap_size: 10,
)?;

// Decompose and embed
let hierarchical_embedding = hierarchical.embed_hierarchically(
    &large_logical_graph,
    num_vars,
)?;

// Analyze decomposition quality
println!("Number of subproblems: {}", hierarchical_embedding.subproblems.len());
println!("Total overlap qubits: {}", hierarchical_embedding.overlap_qubits);
```

## Chain Strength Optimization

Chain strength determines how strongly physical qubits in a chain are coupled together.

### Automatic Chain Strength Calculation

```rust
use quantrs2_anneal::embedding::ChainStrengthCalculator;

let calculator = ChainStrengthCalculator::new();

// Method 1: Based on problem coupling strengths
let coupling_strengths: Vec<f64> = ising_model.couplings()
    .iter()
    .map(|c| c.strength.abs())
    .collect();

let chain_strength = calculator.calculate_from_couplings(
    &coupling_strengths,
    method: ChainStrengthMethod::MaxCoupling { multiplier: 2.0 },
)?;

// Method 2: Based on chain structure
let chain_strength = calculator.calculate_from_chains(
    &embedding.chains,
    &hardware,
    method: ChainStrengthMethod::Adaptive,
)?;

// Method 3: Problem-adaptive calculation
let chain_strength = calculator.calculate_adaptive(
    &ising_model,
    &embedding,
    target_chain_break_rate: 0.05,
)?;

println!("Optimal chain strength: {:.3}", chain_strength);
```

### Chain Strength Methods

```rust
use quantrs2_anneal::embedding::ChainStrengthMethod;

let methods = vec![
    ChainStrengthMethod::Fixed(2.0),                    // Constant value
    ChainStrengthMethod::MaxCoupling { multiplier: 2.0 }, // Based on max coupling
    ChainStrengthMethod::AvgCoupling { multiplier: 3.0 }, // Based on average coupling
    ChainStrengthMethod::Adaptive,                      // Adaptive calculation
    ChainStrengthMethod::PerChain,                      // Individual per chain
];

for method in methods {
    let strength = calculator.calculate_from_couplings(&coupling_strengths, method)?;
    println!("Method {:?}: chain strength = {:.3}", method, strength);
}
```

## Embedding Validation and Repair

### Validation

```rust
use quantrs2_anneal::embedding::{EmbeddingValidator, ValidationError};

let validator = EmbeddingValidator::new(&hardware);

// Comprehensive validation
match validator.validate(&embedding) {
    Ok(report) => {
        println!("Embedding validation passed");
        println!("  Connected components: {}", report.components);
        println!("  Chain connectivity: OK");
        println!("  Hardware constraints: satisfied");
    }
    Err(ValidationError::DisconnectedChain { variable, chain }) => {
        println!("Variable {} has disconnected chain: {:?}", variable, chain);
    }
    Err(ValidationError::InvalidQubit { qubit }) => {
        println!("Invalid qubit used: {}", qubit);
    }
    Err(e) => {
        println!("Validation failed: {:?}", e);
    }
}
```

### Embedding Repair

```rust
use quantrs2_anneal::embedding::EmbeddingRepairer;

let repairer = EmbeddingRepairer::new(&hardware);

// Attempt to repair broken embedding
match repairer.repair_embedding(&broken_embedding) {
    Ok(repaired) => {
        println!("Embedding successfully repaired");
        let metrics = EmbeddingMetrics::analyze(&repaired, &hardware);
        println!("  Repair efficiency: {:.2}", metrics.repair_efficiency);
    }
    Err(e) => {
        println!("Could not repair embedding: {:?}", e);
        // Fall back to finding new embedding
    }
}
```

## Integration with Cloud Services

### D-Wave Integration

```rust
use quantrs2_anneal::dwave::{DWaveClient, EmbeddingConfig as DWaveEmbeddingConfig};

// Advanced D-Wave embedding configuration
let dwave_config = DWaveEmbeddingConfig {
    auto_embed: true,
    embedding_timeout: Duration::from_secs(60),
    chain_strength_method: ChainStrengthMethod::Adaptive,
    find_embedding_max_tries: 1000,
    embedding_parameters: HashMap::from([
        ("max_no_improvement".to_string(), 100.into()),
        ("random_seed".to_string(), 42.into()),
        ("tries".to_string(), 10.into()),
    ]),
    post_processing_config: Some(PostProcessingConfig {
        remove_weak_chains: true,
        optimize_chain_strength: true,
        validate_embedding: true,
    }),
};

// Submit with advanced embedding
let solution = client.submit_ising_with_embedding(
    &ising_model,
    None, // Auto-select solver
    Some(problem_params),
    Some(&dwave_config),
)?;
```

### AWS Braket Integration

```rust
use quantrs2_anneal::braket::{BraketClient, EmbeddingStrategy};

// Braket doesn't require explicit embedding for most devices,
// but we can pre-optimize the problem structure
let embedding_optimizer = EmbeddingStrategy::PreOptimize {
    target_connectivity: ConnectivityType::AllToAll, // For simulators
    problem_transformation: ProblemTransformation::MinimizeVars,
};

let task_result = client.submit_ising_with_strategy(
    &ising_model,
    None, // Auto-select device
    Some(annealing_params),
    Some(embedding_optimizer),
)?;
```

## Performance Optimization

### Embedding Caching

```rust
use quantrs2_anneal::embedding::{EmbeddingCache, CacheKey};

// Cache embeddings for similar problems
let cache = EmbeddingCache::new(max_size: 1000);

// Generate cache key from problem structure
let cache_key = CacheKey::from_edges(&logical_edges, &hardware.topology_id());

// Try to get cached embedding
if let Some(cached_embedding) = cache.get(&cache_key) {
    println!("Using cached embedding");
    return Ok(cached_embedding);
}

// Find new embedding and cache it
let embedding = embedder.find_embedding(&logical_edges, num_vars, &hardware)?;
cache.insert(cache_key, embedding.clone());
```

### Parallel Embedding Search

```rust
use quantrs2_anneal::embedding::ParallelEmbedder;

// Use multiple threads for embedding search
let parallel_embedder = ParallelEmbedder::new(
    num_threads: 8,
    timeout_per_thread: Duration::from_secs(30),
);

// Run multiple embedding attempts in parallel
let embeddings = parallel_embedder.find_multiple_embeddings(
    &logical_edges,
    num_vars,
    &hardware,
    num_attempts: 50,
)?;

// Select best embedding
let best_embedding = embeddings.into_iter()
    .min_by_key(|e| e.total_chain_length())
    .unwrap();
```

## Best Practices

### 1. Problem Preprocessing

```rust
// Simplify problem graph before embedding
use quantrs2_anneal::embedding::GraphPreprocessor;

let preprocessor = GraphPreprocessor::new();

// Remove redundant edges and nodes
let simplified_graph = preprocessor.simplify_graph(&logical_edges)?;

// Identify cliques and handle specially
let cliques = preprocessor.find_cliques(&logical_edges);
for clique in cliques {
    println!("Found clique of size {}: {:?}", clique.len(), clique);
}

// Apply graph reduction techniques
let reduced_graph = preprocessor.apply_reductions(&logical_edges)?;
```

### 2. Hardware-Specific Optimization

```rust
// Optimize for specific hardware characteristics
match hardware.topology_type() {
    TopologyType::Chimera => {
        // Use Chimera-specific optimization
        let chimera_optimizer = ChimeraEmbeddingOptimizer::new();
        embedding = chimera_optimizer.optimize_for_chimera(&embedding)?;
    }
    TopologyType::Pegasus => {
        // Use Pegasus-specific optimization
        let pegasus_optimizer = PegasusEmbeddingOptimizer::new();
        embedding = pegasus_optimizer.optimize_for_pegasus(&embedding)?;
    }
    TopologyType::Zephyr => {
        // Use Zephyr-specific optimization
        let zephyr_optimizer = ZephyrEmbeddingOptimizer::new();
        embedding = zephyr_optimizer.optimize_for_zephyr(&embedding)?;
    }
}
```

### 3. Iterative Improvement

```rust
use quantrs2_anneal::embedding::IterativeImprover;

let improver = IterativeImprover::new(
    max_iterations: 100,
    improvement_threshold: 0.01,
);

// Iteratively improve embedding quality
let improved_embedding = improver.improve_embedding(
    &initial_embedding,
    &hardware,
    &optimization_objectives,
)?;

println!("Improvement: {:.2}% reduction in chain length", 
         improver.improvement_percentage());
```

## Troubleshooting

### Common Issues

1. **Embedding Not Found**
   ```rust
   // Increase search parameters
   let config = EmbeddingConfig {
       max_tries: 1000,      // Increase attempts
       timeout_ms: 120000,   // Increase timeout
       chain_strength_range: (0.01, 100.0), // Wider range
       ..Default::default()
   };
   ```

2. **Poor Chain Quality**
   ```rust
   // Optimize for chain quality
   let config = EmbeddingConfig {
       minimize_chains: true,
       balance_chain_lengths: true,
       avoid_long_chains: true,
       max_chain_length: Some(8),
       ..Default::default()
   };
   ```

3. **High Chain Break Rate**
   ```rust
   // Adjust chain strength
   let adaptive_strength = calculator.calculate_adaptive(
       &ising_model,
       &embedding,
       target_chain_break_rate: 0.02, // Lower target
   )?;
   ```

### Performance Tips

1. **For Small Problems (< 50 variables)**
   - Use simple MinorMiner with default settings
   - Focus on minimizing chain length
   - Cache embeddings for repeated similar problems

2. **For Medium Problems (50-500 variables)**
   - Use layout-aware embedding
   - Consider multi-objective optimization
   - Validate embeddings before use

3. **For Large Problems (> 500 variables)**
   - Use hierarchical embedding
   - Consider problem decomposition
   - Use parallel embedding search

## Examples

See the [`examples/`](../examples/) directory for complete examples:

- `advanced_embedding.rs` - Comprehensive embedding example
- `layout_optimization.rs` - Layout-aware embedding optimization
- `chain_strength_tuning.rs` - Chain strength optimization
- `embedding_benchmarks.rs` - Performance comparison of embedding methods

## References

1. [D-Wave Embedding Guide](https://docs.dwavesys.com/docs/latest/c_gs_workflow.html)
2. [Minor Graph Embedding Algorithms](https://arxiv.org/abs/1406.2741)
3. [Chain Strength Optimization](https://arxiv.org/abs/1807.09806)
4. [Hardware Topology Analysis](https://arxiv.org/abs/1901.07636)