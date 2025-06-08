#!/usr/bin/env python3
"""
Advanced Features Demo

Demonstrates the new features in QuantRS2:
- Quantum Transfer Learning
- Quantum Annealing with Graph Embedding
- Advanced Visualization for Quantum Annealing
"""

import numpy as np
import quantrs2
from quantrs2.transfer_learning import (
    QuantumModelZoo, TransferLearningHelper, create_transfer_strategy
)
from quantrs2.anneal import (
    QUBOBuilder, GraphEmbeddingHelper, ChimeraGraph,
    create_max_cut_qubo, create_tsp_qubo
)
from quantrs2.tytan_viz import (
    SampleResult, VisualizationHelper, ProblemSpecificVisualizer,
    analyze_convergence
)


def demo_transfer_learning():
    """Demonstrate quantum transfer learning."""
    print("\n=== Quantum Transfer Learning Demo ===\n")
    
    try:
        # 1. Load a pretrained model from the model zoo
        print("1. Loading pretrained VQE feature extractor...")
        pretrained_model = QuantumModelZoo.vqe_feature_extractor(n_qubits=4)
        print(f"   Model: {pretrained_model.name}")
        print(f"   Description: {pretrained_model.description}")
        
        # 2. Create transfer learning helper with fine-tuning strategy
        print("\n2. Setting up transfer learning with fine-tuning...")
        helper = TransferLearningHelper(
            pretrained_model,
            strategy="fine_tuning"  # Only last 2 layers trainable
        )
        
        # 3. Adapt for a classification task
        print("\n3. Adapting model for 3-class classification...")
        helper.adapt_for_classification(n_classes=3)
        
        # 4. Display model information
        model_info = helper.get_model_info()
        print("\n4. Model Information:")
        for key, value in model_info.items():
            print(f"   {key}: {value}")
        
        # 5. Try progressive unfreezing
        print("\n5. Progressive unfreezing example...")
        qaoa_model = QuantumModelZoo.qaoa_classifier(n_qubits=6, n_layers=4)
        
        strategy = create_transfer_strategy(
            "progressive_unfreezing",
            unfreeze_rate=1  # Unfreeze 1 layer per epoch
        )
        
        helper2 = TransferLearningHelper(qaoa_model, strategy)
        helper2.adapt_for_classification(n_classes=4)
        
        print("   Progressive unfreezing configured successfully!")
        
    except ImportError as e:
        print(f"   Transfer learning not available: {e}")
        print("   Install with: pip install quantrs2[ml]")


def demo_quantum_annealing():
    """Demonstrate quantum annealing with graph embedding."""
    print("\n\n=== Quantum Annealing & Graph Embedding Demo ===\n")
    
    try:
        # 1. Create a simple Max Cut problem
        print("1. Creating Max Cut QUBO...")
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        weights = [1.0, 1.0, 1.0, 1.0, 2.0]  # Diagonal edge has higher weight
        
        qubo_builder = create_max_cut_qubo(edges, weights)
        print(f"   Created QUBO for graph with {len(edges)} edges")
        
        # 2. Convert to Ising model
        print("\n2. Converting to Ising model...")
        ising_model, offset = qubo_builder.to_ising()
        print(f"   Ising model created with offset: {offset}")
        
        # 3. Create graph embedding helper
        print("\n3. Setting up graph embedding...")
        embedder = GraphEmbeddingHelper(
            target_topology="chimera",
            chain_strength_factor=1.5,
            metric="euclidean"
        )
        
        # 4. Embed on small Chimera topology
        print("\n4. Finding embedding on 2x2 Chimera...")
        target_graph = ChimeraGraph.generate_edges(2, 2, 4)
        embedding = embedder.embed_graph(edges, target_graph)
        
        print("   Embedding found:")
        for logical, chain in embedding.items():
            print(f"     Logical qubit {logical} -> Physical qubits {chain}")
        
        # 5. Get embedding metrics
        metrics = embedder.get_embedding_metrics()
        print("\n5. Embedding metrics:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.3f}")
        
        # 6. Create TSP problem
        print("\n6. Creating TSP QUBO...")
        cities = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        
        distances = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                distances[i, j] = np.linalg.norm(cities[i] - cities[j])
        
        tsp_builder = create_tsp_qubo(distances, penalty=10.0)
        print("   TSP QUBO created for 4 cities")
        
    except ImportError as e:
        print(f"   Annealing features not available: {e}")
        print("   Install with: pip install quantrs2[anneal]")


def demo_visualization():
    """Demonstrate advanced visualization for quantum annealing."""
    print("\n\n=== Advanced Visualization Demo ===\n")
    
    try:
        # 1. Create sample results (simulating quantum annealing output)
        print("1. Creating sample quantum annealing results...")
        results = []
        
        # Generate diverse solutions
        for i in range(200):
            # Create correlated binary variables
            x0 = i % 2 == 0
            x1 = i % 3 == 0
            x2 = x0 and (i % 5 == 0)  # Correlated with x0
            x3 = not x1  # Anti-correlated with x1
            
            assignments = {
                'x0': x0,
                'x1': x1,
                'x2': x2,
                'x3': x3,
            }
            
            # Energy favors certain patterns
            energy = 0.0
            if x0 and x2:
                energy -= 2.0  # Good pattern
            if x1 and not x3:
                energy += 1.0  # Bad pattern
            energy += np.random.normal(0, 0.5)  # Add noise
            
            results.append(SampleResult(assignments, energy, 1))
        
        print(f"   Generated {len(results)} sample results")
        
        # 2. Create visualization helper
        print("\n2. Analyzing results...")
        viz = VisualizationHelper(results)
        
        # 3. Prepare energy landscape
        energy_data = viz.prepare_energy_landscape(
            num_bins=30,
            compute_kde=True,
            kde_points=100
        )
        print("   Energy landscape analysis complete")
        
        # 4. Analyze solution distribution
        solution_data = viz.analyze_solutions(
            compute_correlations=True,
            compute_pca=True,
            n_components=2
        )
        print("   Solution distribution analysis complete")
        
        # 5. Get variable statistics
        stats = viz.get_variable_statistics()
        print("\n3. Variable Statistics:")
        for var, var_stats in sorted(stats.items()):
            print(f"   {var}:")
            print(f"     Frequency of 1: {var_stats['frequency']:.3f}")
            if 'correlations' in var_stats and var_stats['correlations']:
                print("     Correlations:")
                for other_var, corr in var_stats['correlations'].items():
                    if abs(corr) > 0.1:
                        print(f"       with {other_var}: {corr:.3f}")
        
        # 6. Simulate convergence over iterations
        print("\n4. Simulating convergence analysis...")
        iteration_results = []
        best_energy = 0.0
        
        for iter in range(20):
            iter_results = []
            for i in range(50):
                # Energy improves over iterations
                energy = best_energy - iter * 0.5 + np.random.normal(0, 1.0)
                assignments = {'x0': True, 'x1': False}  # Dummy assignments
                iter_results.append(SampleResult(assignments, energy, 1))
            
            iteration_results.append(iter_results)
            best_energy = min(r.energy for r in iter_results)
        
        conv_data = analyze_convergence(iteration_results, ma_window=5, show=False)
        print(f"   Initial best energy: {conv_data['best_energies'][0]:.2f}")
        print(f"   Final best energy: {conv_data['best_energies'][-1]:.2f}")
        print(f"   Improvement: {conv_data['best_energies'][0] - conv_data['best_energies'][-1]:.2f}")
        
        # 7. TSP visualization example
        print("\n5. TSP problem visualization...")
        # Create TSP solution
        tsp_assignments = {
            'x_0_1': True,   # City 0 -> City 1
            'x_1_2': True,   # City 1 -> City 2
            'x_2_3': True,   # City 2 -> City 3
            'x_3_0': True,   # City 3 -> City 0
            # Other edges are False
            'x_0_2': False,
            'x_0_3': False,
            'x_1_0': False,
            'x_1_3': False,
            'x_2_0': False,
            'x_2_1': False,
            'x_3_1': False,
            'x_3_2': False,
        }
        
        tsp_result = SampleResult(tsp_assignments, -4.0, 1)
        
        cities = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        
        # Note: In a real scenario, you would call:
        # tour_length = ProblemSpecificVisualizer.visualize_tsp_solution(
        #     tsp_result, cities, save_path='tsp_solution.png', show=True
        # )
        
        print("   TSP visualization ready (requires matplotlib for plotting)")
        
        print("\n6. Visualization demo complete!")
        print("   Note: Install matplotlib for actual plotting functionality")
        
    except ImportError as e:
        print(f"   Visualization features not available: {e}")
        print("   Install with: pip install quantrs2[tytan]")


def main():
    """Run all demos."""
    print("=" * 60)
    print("QuantRS2 Advanced Features Demo")
    print("=" * 60)
    
    # Check available features
    print("\nChecking available features...")
    
    features = {
        "ML Transfer Learning": hasattr(quantrs2, 'transfer_learning'),
        "Quantum Annealing": hasattr(quantrs2, 'anneal'),
        "Advanced Visualization": hasattr(quantrs2, 'tytan_viz'),
    }
    
    for feature, available in features.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {feature}: {status}")
    
    # Run demos
    demo_transfer_learning()
    demo_quantum_annealing()
    demo_visualization()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()