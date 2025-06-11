# Grover's Search Algorithm

**Level:** 🟡 Intermediate  
**Runtime:** < 5 seconds  
**Topics:** Quantum search, Amplitude amplification, Oracle functions  
**Speedup:** O(√N) vs O(N) classical

Discover how to search unsorted databases quadratically faster than any classical algorithm using Grover's quantum search.

## What is Grover's Algorithm?

Grover's algorithm, invented by Lov Grover in 1996, provides a quadratic speedup for searching unsorted databases. While classical algorithms need O(N) queries to find a target item, Grover's algorithm needs only O(√N) queries.

**Real Impact:**
- Search 1 million items: Classical ~500,000 queries, Quantum ~1,000 queries
- Search 1 billion items: Classical ~500 million queries, Quantum ~31,623 queries

## Algorithm Overview

Grover's algorithm works through **amplitude amplification**:

1. **Initialize**: Create uniform superposition of all states
2. **Oracle**: Mark the target item with a phase flip
3. **Diffusion**: Amplify the marked amplitude
4. **Repeat**: Iterate ~√N times
5. **Measure**: High probability of measuring the target

## Implementation

### Basic 4-Item Search

```python
import quantrs2
import numpy as np

def grovers_4_item_search(target_item=3):
    """
    Search for one item among 4 using Grover's algorithm.
    
    Args:
        target_item: Which item to search for (0, 1, 2, or 3)
    """
    
    print(f"🔍 Grover's Search: Finding item {target_item} among 4 items")
    print("=" * 55)
    
    # 4 items require 2 qubits (2² = 4)
    num_qubits = 2
    circuit = quantrs2.Circuit(num_qubits)
    
    print("Step 1: Initialize uniform superposition")
    # Create uniform superposition: |00⟩ + |01⟩ + |10⟩ + |11⟩
    for qubit in range(num_qubits):
        circuit.h(qubit)
    
    print("  All 4 items have equal probability: 25% each")
    
    # For 4 items, optimal number of iterations is 1
    num_iterations = 1
    print(f"\nStep 2: Apply Grover operator {num_iterations} time(s)")
    
    for iteration in range(num_iterations):
        print(f"\n  Iteration {iteration + 1}:")
        
        # Oracle: Mark the target item
        print(f"    🎯 Oracle: Mark item {target_item}")
        apply_oracle(circuit, target_item, num_qubits)
        
        # Diffusion operator: Amplitude amplification
        print("    📡 Diffusion: Amplify marked amplitude")
        apply_diffusion_operator(circuit, num_qubits)
    
    print("\nStep 3: Measure result")
    circuit.measure_all()
    result = circuit.run()
    
    # Analyze results
    probabilities = result.state_probabilities()
    
    print(f"\n📊 Search Results:")
    for state, prob in probabilities.items():
        item = int(state, 2)
        marker = "🎯" if item == target_item else "  "
        print(f"  {marker} Item {item} (|{state}⟩): {prob:.3f} ({prob*100:.1f}%)")
    
    # Find most likely result
    most_likely_state = max(probabilities, key=probabilities.get)
    found_item = int(most_likely_state, 2)
    success_probability = probabilities[most_likely_state]
    
    print(f"\n🎉 Result: Found item {found_item}")
    print(f"   Success probability: {success_probability:.3f}")
    
    if found_item == target_item:
        print("   ✅ Correct item found!")
    else:
        print("   ❌ Wrong item found")
    
    return circuit, result, found_item

def apply_oracle(circuit, target_item, num_qubits):
    """Apply oracle that marks the target item with a phase flip."""
    
    # Convert target to binary representation
    target_binary = format(target_item, f'0{num_qubits}b')
    
    # Apply X gates to qubits that should be |0⟩ in target state
    for i, bit in enumerate(target_binary):
        if bit == '0':
            circuit.x(i)
    
    # Apply multi-controlled Z gate (phase flip when all qubits are |1⟩)
    if num_qubits == 1:
        circuit.z(0)
    elif num_qubits == 2:
        # Controlled-Z for 2 qubits
        circuit.h(1)
        circuit.cx(0, 1)
        circuit.h(1)
    else:
        # For more qubits, use multi-controlled Z
        multi_controlled_z(circuit, list(range(num_qubits)))
    
    # Restore qubits to original state
    for i, bit in enumerate(target_binary):
        if bit == '0':
            circuit.x(i)

def apply_diffusion_operator(circuit, num_qubits):
    """Apply Grover diffusion operator (inversion about average)."""
    
    # Step 1: Transform to |+⟩^⊗n basis
    for qubit in range(num_qubits):
        circuit.h(qubit)
    
    # Step 2: Apply oracle for |00...0⟩ state
    for qubit in range(num_qubits):
        circuit.x(qubit)
    
    # Multi-controlled Z gate
    if num_qubits == 1:
        circuit.z(0)
    elif num_qubits == 2:
        circuit.h(1)
        circuit.cx(0, 1)
        circuit.h(1)
    else:
        multi_controlled_z(circuit, list(range(num_qubits)))
    
    for qubit in range(num_qubits):
        circuit.x(qubit)
    
    # Step 3: Transform back to computational basis
    for qubit in range(num_qubits):
        circuit.h(qubit)

def multi_controlled_z(circuit, qubits):
    """Apply multi-controlled Z gate (simplified implementation)."""
    if len(qubits) <= 2:
        if len(qubits) == 2:
            circuit.h(qubits[1])
            circuit.cx(qubits[0], qubits[1])
            circuit.h(qubits[1])
    else:
        # Simplified multi-controlled Z for demonstration
        # Real implementation would use ancilla qubits
        circuit.h(qubits[-1])
        for i in range(len(qubits) - 1):
            circuit.cx(qubits[i], qubits[-1])
        circuit.h(qubits[-1])

# Test Grover's algorithm on all 4 items
print("Testing Grover's algorithm on 4-item database:")
for target in range(4):
    circuit, result, found = grovers_4_item_search(target)
    print()
```

### Scalable Grover's Algorithm

```python
def grovers_general_search(num_items, target_item, max_iterations=None):
    """
    General Grover's search for any number of items.
    
    Args:
        num_items: Total number of items to search
        target_item: Which item to find
        max_iterations: Maximum iterations (auto-calculated if None)
    """
    
    print(f"🔍 Grover's Search: {num_items} items, finding item {target_item}")
    print("=" * 60)
    
    # Calculate required qubits
    num_qubits = int(np.ceil(np.log2(num_items)))
    
    # Calculate optimal number of iterations
    if max_iterations is None:
        optimal_iterations = int(np.pi * np.sqrt(num_items) / 4)
    else:
        optimal_iterations = max_iterations
    
    print(f"Database size: {num_items} items")
    print(f"Qubits required: {num_qubits}")
    print(f"Optimal iterations: {optimal_iterations}")
    
    # Classical comparison
    classical_queries = num_items // 2  # Average case
    quantum_queries = optimal_iterations * 2  # Oracle + diffusion per iteration
    speedup = classical_queries / quantum_queries if quantum_queries > 0 else float('inf')
    
    print(f"\nComplexity comparison:")
    print(f"  Classical average: ~{classical_queries} queries")
    print(f"  Quantum queries: ~{quantum_queries} queries")
    print(f"  Speedup: ~{speedup:.1f}x")
    
    # Build circuit
    circuit = quantrs2.Circuit(num_qubits)
    
    # Initialize superposition
    print(f"\nStep 1: Initialize uniform superposition")
    for qubit in range(num_qubits):
        circuit.h(qubit)
    
    # Apply Grover iterations
    print(f"Step 2: Apply {optimal_iterations} Grover iterations")
    for iteration in range(optimal_iterations):
        # Oracle
        apply_oracle(circuit, target_item, num_qubits)
        
        # Diffusion
        apply_diffusion_operator(circuit, num_qubits)
        
        if iteration < 3:  # Show first few iterations
            print(f"  Completed iteration {iteration + 1}")
    
    if optimal_iterations > 3:
        print(f"  ... (completed remaining {optimal_iterations - 3} iterations)")
    
    # Measure
    print(f"Step 3: Measure result")
    circuit.measure_all()
    result = circuit.run()
    
    # Analyze results
    probabilities = result.state_probabilities()
    
    print(f"\n📊 Top search results:")
    sorted_results = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    for i, (state, prob) in enumerate(sorted_results[:5]):  # Top 5 results
        item = int(state, 2)
        if item < num_items:  # Only show valid items
            marker = "🎯" if item == target_item else f"{i+1:2d}."
            print(f"  {marker} Item {item:3d}: {prob:.3f} ({prob*100:.1f}%)")
    
    # Success analysis
    target_state = format(target_item, f'0{num_qubits}b')
    success_probability = probabilities.get(target_state, 0)
    
    print(f"\nSuccess analysis:")
    print(f"  Target probability: {success_probability:.3f}")
    
    if success_probability > 0.8:
        print("  ✅ Excellent! High success probability")
    elif success_probability > 0.5:
        print("  ✅ Good success probability")
    elif success_probability > 0.3:
        print("  ⚠ Moderate success - might need more iterations")
    else:
        print("  ❌ Low success probability - check implementation")
    
    return circuit, result, success_probability

# Test on different database sizes
test_cases = [
    (4, 3),    # 4 items, find item 3
    (8, 5),    # 8 items, find item 5
    (16, 10),  # 16 items, find item 10
]

for num_items, target in test_cases:
    circuit, result, success = grovers_general_search(num_items, target)
    print()
```

### Multiple Target Search

```python
def grovers_multiple_targets(num_items, target_items):
    """
    Grover's search for multiple target items.
    
    Args:
        num_items: Total number of items
        target_items: List of target items to find
    """
    
    print(f"🎯 Grover's Multi-Target Search")
    print(f"Database: {num_items} items")
    print(f"Targets: {target_items}")
    print("=" * 40)
    
    num_qubits = int(np.ceil(np.log2(num_items)))
    
    # Adjust iterations for multiple targets
    # With M targets out of N items, we need π√(N/M)/4 iterations
    M = len(target_items)
    N = num_items
    optimal_iterations = int(np.pi * np.sqrt(N/M) / 4) if M > 0 else 0
    
    print(f"Number of targets: {M}")
    print(f"Optimal iterations: {optimal_iterations}")
    
    circuit = quantrs2.Circuit(num_qubits)
    
    # Initialize superposition
    for qubit in range(num_qubits):
        circuit.h(qubit)
    
    # Apply Grover iterations
    for iteration in range(optimal_iterations):
        # Multi-target oracle
        apply_multi_target_oracle(circuit, target_items, num_qubits)
        
        # Diffusion operator
        apply_diffusion_operator(circuit, num_qubits)
    
    circuit.measure_all()
    result = circuit.run()
    
    # Analyze results
    probabilities = result.state_probabilities()
    
    print(f"\n📊 Search Results:")
    total_target_probability = 0
    
    for state, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        item = int(state, 2)
        if item < num_items and prob > 0.01:  # Only show significant probabilities
            is_target = item in target_items
            marker = "🎯" if is_target else "  "
            print(f"  {marker} Item {item:3d}: {prob:.3f} ({prob*100:.1f}%)")
            
            if is_target:
                total_target_probability += prob
    
    print(f"\nTotal target probability: {total_target_probability:.3f}")
    print(f"Average per target: {total_target_probability/M:.3f}")
    
    return circuit, result, total_target_probability

def apply_multi_target_oracle(circuit, target_items, num_qubits):
    """Oracle that marks multiple target items."""
    
    # Apply oracle for each target (superposition of phase flips)
    for target in target_items:
        target_binary = format(target, f'0{num_qubits}b')
        
        # Flip qubits for target preparation
        for i, bit in enumerate(target_binary):
            if bit == '0':
                circuit.x(i)
        
        # Apply controlled Z
        if num_qubits == 2:
            circuit.h(1)
            circuit.cx(0, 1)
            circuit.h(1)
        
        # Restore qubits
        for i, bit in enumerate(target_binary):
            if bit == '0':
                circuit.x(i)

# Test multi-target search
multi_targets = [2, 5, 7]
circuit, result, prob = grovers_multiple_targets(8, multi_targets)
print()
```

## Quantum Amplitude Analysis

```python
def analyze_grover_amplitudes():
    """Analyze how amplitudes evolve during Grover's algorithm."""
    
    print("📈 Grover's Amplitude Evolution Analysis")
    print("=" * 45)
    
    num_qubits = 2
    target_item = 3
    
    # Track amplitude evolution
    amplitude_history = []
    
    circuit = quantrs2.Circuit(num_qubits)
    
    # Initial superposition
    for qubit in range(num_qubits):
        circuit.h(qubit)
    
    # Record initial amplitudes
    result = circuit.run()
    amplitude_history.append(result.state_probabilities())
    
    print("Amplitude evolution:")
    print("Iteration | Item 0 | Item 1 | Item 2 | Item 3*")
    print("-" * 45)
    
    # Show initial state
    probs = amplitude_history[0]
    print(f"   Initial |  {probs.get('00', 0):.3f} |  {probs.get('01', 0):.3f} |  {probs.get('10', 0):.3f} |  {probs.get('11', 0):.3f}")
    
    # Apply iterations and track amplitudes
    max_iterations = 3
    for iteration in range(max_iterations):
        # Create fresh circuit for this iteration
        iter_circuit = quantrs2.Circuit(num_qubits)
        
        # Initialize
        for qubit in range(num_qubits):
            iter_circuit.h(qubit)
        
        # Apply Grover operators up to current iteration
        for i in range(iteration + 1):
            apply_oracle(iter_circuit, target_item, num_qubits)
            apply_diffusion_operator(iter_circuit, num_qubits)
        
        iter_circuit.measure_all()
        result = iter_circuit.run()
        probs = result.state_probabilities()
        amplitude_history.append(probs)
        
        print(f"        {iteration+1} |  {probs.get('00', 0):.3f} |  {probs.get('01', 0):.3f} |  {probs.get('10', 0):.3f} |  {probs.get('11', 0):.3f}")
    
    # Theoretical analysis
    print(f"\n🧮 Theoretical Analysis:")
    N = 4  # 4 items
    optimal_iterations = int(np.pi * np.sqrt(N) / 4)
    print(f"  Optimal iterations: {optimal_iterations}")
    
    # Success probability after k iterations
    for k in range(max_iterations + 1):
        angle = (2*k + 1) * np.arcsin(1/np.sqrt(N))
        success_prob = np.sin(angle)**2
        measured_prob = amplitude_history[k].get('11', 0)
        
        print(f"  After {k} iterations:")
        print(f"    Theoretical: {success_prob:.3f}")
        print(f"    Measured: {measured_prob:.3f}")
        print(f"    Difference: {abs(success_prob - measured_prob):.3f}")

analyze_grover_amplitudes()
```

## Performance Benchmarking

```python
def benchmark_grover_performance():
    """Benchmark Grover's algorithm performance."""
    
    print("⚡ Grover's Algorithm Performance Benchmark")
    print("=" * 50)
    
    import time
    
    # Test different database sizes
    test_sizes = [4, 8, 16, 32, 64]
    
    print(f"{'Size':<6} {'Qubits':<7} {'Iterations':<11} {'Time (ms)':<10} {'Success %'}")
    print("-" * 50)
    
    for size in test_sizes:
        num_qubits = int(np.ceil(np.log2(size)))
        optimal_iterations = int(np.pi * np.sqrt(size) / 4)
        target_item = size - 1  # Search for last item
        
        # Time the algorithm
        start_time = time.time()
        
        circuit = quantrs2.Circuit(num_qubits)
        
        # Initialize
        for qubit in range(num_qubits):
            circuit.h(qubit)
        
        # Apply Grover iterations
        for _ in range(optimal_iterations):
            apply_oracle(circuit, target_item, num_qubits)
            apply_diffusion_operator(circuit, num_qubits)
        
        circuit.measure_all()
        result = circuit.run()
        
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Calculate success probability
        target_state = format(target_item, f'0{num_qubits}b')
        success_prob = result.state_probabilities().get(target_state, 0)
        
        print(f"{size:<6} {num_qubits:<7} {optimal_iterations:<11} {execution_time:<10.2f} {success_prob*100:<8.1f}")
    
    print(f"\nScaling Analysis:")
    print(f"  Classical complexity: O(N)")
    print(f"  Quantum complexity: O(√N)")
    print(f"  For 1M items: Classical ~500K, Quantum ~785 queries")

benchmark_grover_performance()
```

## Real-World Applications

```python
def grover_applications():
    """Demonstrate real-world applications of Grover's algorithm."""
    
    print("🌍 Real-World Applications of Grover's Algorithm")
    print("=" * 55)
    
    applications = [
        {
            "domain": "Database Search",
            "problem": "Find customer records without indexing",
            "classical": "O(N) linear search",
            "quantum": "O(√N) Grover search",
            "example_size": "1 million records",
            "classical_queries": 500000,
            "quantum_queries": 1000,
            "speedup": 500
        },
        {
            "domain": "Cryptography",
            "problem": "Brute force key search",
            "classical": "O(2^n) exhaustive search", 
            "quantum": "O(2^(n/2)) Grover + QRAM",
            "example_size": "128-bit keys",
            "classical_queries": "2^128",
            "quantum_queries": "2^64",
            "speedup": "2^64"
        },
        {
            "domain": "Optimization",
            "problem": "Find minimum in unstructured space",
            "classical": "Random sampling O(N)",
            "quantum": "Quantum amplitude amplification O(√N)",
            "example_size": "10^6 solutions",
            "classical_queries": 500000,
            "quantum_queries": 1000,
            "speedup": 500
        },
        {
            "domain": "Machine Learning",
            "problem": "Nearest neighbor search",
            "classical": "O(N) distance calculations",
            "quantum": "O(√N) quantum search",
            "example_size": "1M data points",
            "classical_queries": 1000000,
            "quantum_queries": 1000,
            "speedup": 1000
        }
    ]
    
    for app in applications:
        print(f"\n{app['domain']}: {app['problem']}")
        print(f"  Classical: {app['classical']}")
        print(f"  Quantum: {app['quantum']}")
        print(f"  Example: {app['example_size']}")
        print(f"  Speedup: {app['speedup']}x")
    
    print(f"\n🚀 Future Impact:")
    print(f"  • Quantum search in quantum databases")
    print(f"  • Breaking symmetric cryptography")
    print(f"  • Accelerating AI/ML algorithms")
    print(f"  • Solving NP problems with square-root speedup")

grover_applications()
```

## Exercises and Extensions

### Exercise 1: Custom Oracle
```python
def exercise_custom_oracle():
    """Exercise: Create custom oracle functions."""
    
    print("🎯 Exercise: Custom Oracle Functions")
    print("=" * 35)
    
    # TODO: Create oracle that finds even numbers
    # TODO: Create oracle that finds prime numbers  
    # TODO: Create oracle that finds numbers divisible by 3
    
    print("Your turn! Implement custom oracle functions:")
    print("1. Find all even numbers in database")
    print("2. Find all prime numbers")
    print("3. Find numbers with specific properties")

exercise_custom_oracle()
```

### Exercise 2: Amplitude Amplification
```python
def exercise_amplitude_amplification():
    """Exercise: Generalize to amplitude amplification."""
    
    print("🎯 Exercise: Quantum Amplitude Amplification")
    print("=" * 45)
    
    # TODO: Implement general amplitude amplification
    # TODO: Use different success probabilities
    # TODO: Compare with Grover's special case
    
    print("Extend Grover's algorithm to general amplitude amplification!")

exercise_amplitude_amplification()
```

### Exercise 3: Noisy Grover
```python
def exercise_noisy_grover():
    """Exercise: Grover's algorithm with noise."""
    
    print("🎯 Exercise: Noisy Grover's Algorithm")
    print("=" * 35)
    
    # TODO: Add noise to oracle and diffusion operations
    # TODO: Study how noise affects success probability
    # TODO: Implement error mitigation strategies
    
    print("Analyze how noise affects Grover's algorithm!")

exercise_noisy_grover()
```

## Common Mistakes and Troubleshooting

### Mistake 1: Wrong Number of Iterations
```python
# ❌ Wrong: Too many iterations causes oscillation
iterations = num_items  # Way too many!

# ✅ Correct: Use optimal number
iterations = int(np.pi * np.sqrt(num_items) / 4)
```

### Mistake 2: Incorrect Oracle Implementation
```python
# ❌ Wrong: Oracle flips amplitude instead of phase
circuit.x(target_qubit)  # This is not a phase flip

# ✅ Correct: Oracle applies phase flip
circuit.z(target_qubit)  # Phase flip for single qubit
```

### Mistake 3: Missing Diffusion Operator
```python
# ❌ Wrong: Only applying oracle
apply_oracle(circuit, target)
# Missing diffusion operator!

# ✅ Correct: Oracle + Diffusion = Grover operator
apply_oracle(circuit, target)
apply_diffusion_operator(circuit)
```

## Algorithm Variants

### Amplitude Amplification
- Generalization of Grover's algorithm
- Works with arbitrary success probabilities
- Applications in quantum counting

### Fixed-Point Search
- Modified Grover that doesn't require knowing the number of solutions
- More robust to implementation errors
- Useful for practical applications

### Quantum Counting
- Estimate number of solutions without finding them
- Uses quantum phase estimation
- Important for algorithm complexity analysis

## Summary

🎉 **Congratulations!** You've learned:
- How Grover's algorithm provides quadratic quantum speedup
- The roles of oracle and diffusion operators
- Implementation for different database sizes
- Multi-target search capabilities
- Real-world applications and impact
- Performance analysis and benchmarking

Grover's algorithm is one of the most practical quantum algorithms, with applications from database search to cryptography. Master it, and you're ready for advanced quantum algorithms!

**Next Steps:**
- Try [Quantum Fourier Transform](qft.md)
- Explore [Shor's Algorithm](shor.md)
- Learn about [QAOA](../optimization/qaoa_maxcut.md)

## References

### Original Papers
- Grover, L.K. (1996). "A fast quantum mechanical algorithm for database search"
- Boyer et al. (1998). "Tight bounds on quantum searching"

### Modern Developments
- Amplitude amplification framework
- Fixed-point quantum search
- Quantum counting algorithms

### Applications
- Quantum machine learning
- Cryptanalysis
- Optimization problems

---

*"The quantum world is not just stranger than we imagine, it is stranger than we can imagine. But with Grover's algorithm, we can search it efficiently!" - Adapted from J.B.S. Haldane*

🚀 **Ready to search the quantum realm?** Explore more [Quantum Algorithms](../index.md#quantum-algorithms)!