use super::*;
// use approx::assert_relative_eq;
use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;

#[test]
fn test_graph_creation() {
    let graph = Graph::new(GraphType::Cycle, 4);
    assert_eq!(graph.num_vertices, 4);
    assert_eq!(graph.degree(0), 2);

    let complete = Graph::new(GraphType::Complete, 5);
    assert_eq!(complete.degree(0), 4);
}

#[test]
fn test_discrete_walk_initialization() {
    let graph = Graph::new(GraphType::Line, 5);
    let mut walk = DiscreteQuantumWalk::new(graph, CoinOperator::Hadamard);

    walk.initialize_position(2);
    let probs = walk.position_probabilities();

    assert!((probs[2] - 1.0).abs() < 1e-10);
}

#[test]
fn test_continuous_walk() {
    let graph = Graph::new(GraphType::Cycle, 4);
    let mut walk = ContinuousQuantumWalk::new(graph);

    walk.initialize_vertex(0);
    walk.evolve(1.0);

    let probs = walk.vertex_probabilities();
    let total: f64 = probs.iter().sum();
    assert!((total - 1.0).abs() < 1e-10);
}

#[test]
fn test_weighted_graph() {
    let mut graph = Graph::new_empty(3);
    graph.add_weighted_edge(0, 1, 2.0);
    graph.add_weighted_edge(1, 2, 3.0);

    let adj_matrix = graph.adjacency_matrix();
    assert_eq!(adj_matrix[[0, 1]], 2.0);
    assert_eq!(adj_matrix[[1, 2]], 3.0);
    assert_eq!(adj_matrix[[0, 2]], 0.0);
}

#[test]
fn test_graph_from_adjacency_matrix() {
    let mut matrix = Array2::zeros((3, 3));
    matrix[[0, 1]] = 1.0;
    matrix[[1, 0]] = 1.0;
    matrix[[1, 2]] = 2.0;
    matrix[[2, 1]] = 2.0;

    let graph = Graph::from_adjacency_matrix(&matrix)
        .expect("Failed to create graph from adjacency matrix in test_graph_from_adjacency_matrix");
    assert_eq!(graph.num_vertices, 3);
    assert_eq!(graph.degree(0), 1);
    assert_eq!(graph.degree(1), 2);
    assert_eq!(graph.degree(2), 1);
}

#[test]
fn test_laplacian_matrix() {
    let graph = Graph::new(GraphType::Cycle, 3);
    let laplacian = graph.laplacian_matrix();

    // Each vertex in a 3-cycle has degree 2
    assert_eq!(laplacian[[0, 0]], 2.0);
    assert_eq!(laplacian[[1, 1]], 2.0);
    assert_eq!(laplacian[[2, 2]], 2.0);

    // Adjacent vertices have -1
    assert_eq!(laplacian[[0, 1]], -1.0);
    assert_eq!(laplacian[[1, 2]], -1.0);
    assert_eq!(laplacian[[2, 0]], -1.0);
}

#[test]
fn test_bipartite_detection() {
    let bipartite = Graph::new(GraphType::Cycle, 4); // Even cycle is bipartite
    assert!(bipartite.is_bipartite());

    let non_bipartite = Graph::new(GraphType::Cycle, 3); // Odd cycle is not bipartite
    assert!(!non_bipartite.is_bipartite());

    let complete = Graph::new(GraphType::Complete, 3); // Complete graph with >2 vertices is not bipartite
    assert!(!complete.is_bipartite());
}

#[test]
fn test_shortest_paths() {
    let graph = Graph::new(GraphType::Line, 4); // 0-1-2-3
    let distances = graph.all_pairs_shortest_paths();

    assert_eq!(distances[[0, 0]], 0.0);
    assert_eq!(distances[[0, 1]], 1.0);
    assert_eq!(distances[[0, 2]], 2.0);
    assert_eq!(distances[[0, 3]], 3.0);
    assert_eq!(distances[[1, 3]], 2.0);
}

#[test]
fn test_szegedy_walk() {
    let graph = Graph::new(GraphType::Cycle, 4);
    let mut szegedy = SzegedyQuantumWalk::new(graph);

    szegedy.initialize_uniform();
    let initial_probs = szegedy.vertex_probabilities();

    // Should have some probability on each vertex
    for &prob in &initial_probs {
        assert!(prob > 0.0);
    }

    // Take a few steps
    for _ in 0..5 {
        szegedy.step();
    }

    let final_probs = szegedy.vertex_probabilities();
    let total: f64 = final_probs.iter().sum();
    assert!((total - 1.0).abs() < 1e-10);
}

#[test]
fn test_szegedy_edge_initialization() {
    let mut graph = Graph::new_empty(3);
    graph.add_edge(0, 1);
    graph.add_edge(1, 2);

    let mut szegedy = SzegedyQuantumWalk::new(graph);
    szegedy.initialize_edge(0, 1);

    let edge_probs = szegedy.edge_probabilities();
    assert_eq!(edge_probs.len(), 1);
    assert_eq!(edge_probs[0].0, (0, 1));
    assert!((edge_probs[0].1 - 1.0).abs() < 1e-10);
}

#[test]
fn test_multi_walker_quantum_walk() {
    let graph = Graph::new(GraphType::Cycle, 3);
    let mut multi_walk = MultiWalkerQuantumWalk::new(graph, 2);

    // Initialize two walkers at positions 0 and 1
    multi_walk
        .initialize_positions(&[0, 1])
        .expect("Failed to initialize positions in test_multi_walker_quantum_walk");

    let marginal_0 = multi_walk.marginal_probabilities(0);
    let marginal_1 = multi_walk.marginal_probabilities(1);

    assert!((marginal_0[0] - 1.0).abs() < 1e-10);
    assert!((marginal_1[1] - 1.0).abs() < 1e-10);

    // Take a step
    multi_walk.step_independent();

    // Probabilities should have spread
    let new_marginal_0 = multi_walk.marginal_probabilities(0);
    let new_marginal_1 = multi_walk.marginal_probabilities(1);

    assert!(new_marginal_0[0] < 1.0);
    assert!(new_marginal_1[1] < 1.0);
}

#[test]
fn test_multi_walker_bell_state() {
    let graph = Graph::new(GraphType::Cycle, 4);
    let mut multi_walk = MultiWalkerQuantumWalk::new(graph, 2);

    multi_walk
        .initialize_entangled_bell_state(0, 1)
        .expect("Failed to initialize entangled Bell state in test_multi_walker_bell_state");

    let marginal_0 = multi_walk.marginal_probabilities(0);
    let marginal_1 = multi_walk.marginal_probabilities(1);

    // Each walker should have 50% probability at each of their initial positions
    assert!((marginal_0[0] - 0.5).abs() < 1e-10);
    assert!((marginal_0[1] - 0.5).abs() < 1e-10);
    assert!((marginal_1[0] - 0.5).abs() < 1e-10);
    assert!((marginal_1[1] - 0.5).abs() < 1e-10);
}

#[test]
fn test_multi_walker_error_handling() {
    let graph = Graph::new(GraphType::Line, 3);
    let mut multi_walk = MultiWalkerQuantumWalk::new(graph.clone(), 2);

    // Wrong number of positions
    assert!(multi_walk.initialize_positions(&[0]).is_err());

    // Position out of bounds
    assert!(multi_walk.initialize_positions(&[0, 5]).is_err());

    // Bell state with wrong number of walkers
    let mut single_walk = MultiWalkerQuantumWalk::new(graph, 1);
    assert!(single_walk.initialize_entangled_bell_state(0, 1).is_err());
}

#[test]
fn test_decoherent_quantum_walk() {
    let graph = Graph::new(GraphType::Line, 5);
    let mut decoherent = DecoherentQuantumWalk::new(graph, CoinOperator::Hadamard, 0.1);

    decoherent.initialize_position(2);
    let initial_probs = decoherent.position_probabilities();
    assert!((initial_probs[2] - 1.0).abs() < 1e-10);

    // Take steps with decoherence
    for _ in 0..10 {
        decoherent.step();
    }

    let final_probs = decoherent.position_probabilities();
    let total: f64 = final_probs.iter().sum();
    assert!((total - 1.0).abs() < 1e-10);

    // Should have spread from initial position
    assert!(final_probs[2] < 1.0);
}

#[test]
fn test_decoherence_rate_bounds() {
    let graph = Graph::new(GraphType::Cycle, 4);
    let mut decoherent = DecoherentQuantumWalk::new(graph, CoinOperator::Grover, 0.5);

    // Test clamping
    decoherent.set_decoherence_rate(-0.1);
    decoherent.initialize_position(0);
    decoherent.step(); // Should work without panicking

    decoherent.set_decoherence_rate(1.5);
    decoherent.step(); // Should work without panicking
}

#[test]
fn test_transition_matrix() {
    let graph = Graph::new(GraphType::Cycle, 3);
    let transition = graph.transition_matrix();

    // Each vertex has degree 2, so each transition probability is 1/2
    for i in 0..3 {
        let mut row_sum = 0.0;
        for j in 0..3 {
            row_sum += transition[[i, j]];
        }
        assert!((row_sum - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_normalized_laplacian() {
    let graph = Graph::new(GraphType::Complete, 3);
    let norm_laplacian = graph.normalized_laplacian_matrix();

    // Diagonal entries should be 1
    for i in 0..3 {
        assert!((norm_laplacian[[i, i]] - 1.0).abs() < 1e-10);
    }

    // Off-diagonal entries for complete graph K_3
    let expected_off_diag = -1.0 / 2.0; // -1/sqrt(2*2)
    assert!((norm_laplacian[[0, 1]] - expected_off_diag).abs() < 1e-10);
    assert!((norm_laplacian[[1, 2]] - expected_off_diag).abs() < 1e-10);
    assert!((norm_laplacian[[0, 2]] - expected_off_diag).abs() < 1e-10);
}

#[test]
fn test_algebraic_connectivity() {
    let complete_3 = Graph::new(GraphType::Complete, 3);
    let connectivity = complete_3.algebraic_connectivity();
    assert!(connectivity > 0.0); // Complete graphs have positive algebraic connectivity

    let line_5 = Graph::new(GraphType::Line, 5);
    let line_connectivity = line_5.algebraic_connectivity();
    assert!(line_connectivity > 0.0);
}

#[test]
fn test_mixing_time_estimation() {
    let graph = Graph::new(GraphType::Complete, 4);
    let mut szegedy = SzegedyQuantumWalk::new(graph);

    let mixing_time = szegedy.estimate_mixing_time(0.1);
    assert!(mixing_time > 0);
    assert!(mixing_time <= 1000); // Should converge within max steps
}

#[test]
fn test_quantum_walk_search_on_custom_graph() {
    // Create a star graph: central vertex connected to all others
    let mut graph = Graph::new_empty(5);
    for i in 1..5 {
        graph.add_edge(0, i);
    }

    let oracle = SearchOracle::new(vec![3]); // Mark vertex 3
    let mut search = QuantumWalkSearch::new(graph, oracle);

    let (found_vertex, prob, steps) = search.run(20);
    assert_eq!(found_vertex, 3);
    assert!(prob > 0.0);
    assert!(steps <= 20);
}

#[test]
fn test_custom_coin_operator() {
    let graph = Graph::new(GraphType::Line, 3);

    // Create a custom 2x2 coin (Pauli-X)
    let mut coin_matrix = Array2::zeros((2, 2));
    coin_matrix[[0, 1]] = Complex64::new(1.0, 0.0);
    coin_matrix[[1, 0]] = Complex64::new(1.0, 0.0);

    let custom_coin = CoinOperator::Custom(coin_matrix);
    let mut walk = DiscreteQuantumWalk::new(graph, custom_coin);

    walk.initialize_position(1);
    walk.step();

    let probs = walk.position_probabilities();
    let total: f64 = probs.iter().sum();
    assert!((total - 1.0).abs() < 1e-10);
}

#[test]
fn test_empty_graph_edge_cases() {
    let empty_graph = Graph::new_empty(3);
    let mut szegedy = SzegedyQuantumWalk::new(empty_graph);

    szegedy.initialize_uniform();
    let probs = szegedy.vertex_probabilities();

    // No edges means no probability distribution
    for &prob in &probs {
        assert_eq!(prob, 0.0);
    }
}

#[test]
fn test_hypercube_graph() {
    let hypercube = Graph::new(GraphType::Hypercube, 3); // 2^3 = 8 vertices
    assert_eq!(hypercube.num_vertices, 8);

    // Each vertex in a 3D hypercube has degree 3
    for i in 0..8 {
        assert_eq!(hypercube.degree(i), 3);
    }
}

#[test]
fn test_grid_2d_graph() {
    let grid = Graph::new(GraphType::Grid2D, 3); // 3x3 grid
    assert_eq!(grid.num_vertices, 9);

    // Corner vertices have degree 2
    assert_eq!(grid.degree(0), 2); // Top-left
    assert_eq!(grid.degree(2), 2); // Top-right
    assert_eq!(grid.degree(6), 2); // Bottom-left
    assert_eq!(grid.degree(8), 2); // Bottom-right

    // Center vertex has degree 4
    assert_eq!(grid.degree(4), 4);
}

/// Path graph P4 has Laplacian eigenvalues 0, 2-√2, 2, 2+√2.
#[test]
fn test_path_graph_eigenvalues() {
    // Build 4-vertex path: 0-1-2-3
    let graph = Graph::new(GraphType::Line, 4);
    let laplacian = graph.laplacian_matrix();

    let eigenvalues = compute_laplacian_eigenvalues_impl(&laplacian)
        .expect("eigenvalue computation must succeed for P4");

    assert_eq!(eigenvalues.len(), 4, "P4 must have 4 eigenvalues");

    // Expected (sorted): 0, 2-√2 ≈ 0.5858, 2, 2+√2 ≈ 3.4142
    let expected = [
        0.0_f64,
        2.0 - std::f64::consts::SQRT_2,
        2.0,
        2.0 + std::f64::consts::SQRT_2,
    ];

    for (got, &exp) in eigenvalues.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < 1e-8,
            "eigenvalue mismatch: got {got:.10}, expected {exp:.10}"
        );
    }
}

/// Complete graph K4 has Laplacian eigenvalues 0 (once) and 4 (three times).
#[test]
fn test_fiedler_value_complete_graph() {
    let graph = Graph::new(GraphType::Complete, 4); // K4
    let laplacian = graph.laplacian_matrix();

    // Verify full eigenvalue spectrum
    let eigenvalues = compute_laplacian_eigenvalues_impl(&laplacian)
        .expect("eigenvalue computation must succeed for K4");

    assert_eq!(eigenvalues.len(), 4, "K4 must have 4 eigenvalues");
    assert!(
        eigenvalues[0].abs() < 1e-8,
        "smallest eigenvalue of K4 must be 0, got {}",
        eigenvalues[0]
    );
    for ev in &eigenvalues[1..] {
        assert!(
            (ev - 4.0).abs() < 1e-8,
            "non-zero eigenvalue of K4 must be 4, got {ev}"
        );
    }

    // Fiedler value via power iteration must also be close to 4
    let fiedler = estimate_fiedler_value_impl(&laplacian);
    assert!(
        (fiedler - 4.0).abs() < 0.1,
        "Fiedler estimate for K4 should be ~4, got {fiedler}"
    );
}
