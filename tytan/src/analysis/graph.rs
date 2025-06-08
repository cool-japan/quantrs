//! Graph generation utilities for testing and examples
//!
//! This module provides tools for generating various types of graphs
//! for testing optimization algorithms.

use std::collections::HashSet;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Generate a random graph with specified edge probability
pub fn generate_graph(
    n_nodes: usize,
    edge_probability: f64,
) -> Result<Vec<(usize, usize)>, Box<dyn std::error::Error>> {
    if edge_probability < 0.0 || edge_probability > 1.0 {
        return Err("Edge probability must be between 0 and 1".into());
    }
    
    let mut rng = StdRng::seed_from_u64(42);
    let mut edges = Vec::new();
    
    // Generate edges with given probability
    for i in 0..n_nodes {
        for j in (i + 1)..n_nodes {
            if rng.gen::<f64>() < edge_probability {
                edges.push((i, j));
            }
        }
    }
    
    Ok(edges)
}

/// Generate a complete graph
pub fn generate_complete_graph(n_nodes: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    
    for i in 0..n_nodes {
        for j in (i + 1)..n_nodes {
            edges.push((i, j));
        }
    }
    
    edges
}

/// Generate a cycle graph
pub fn generate_cycle_graph(n_nodes: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    
    if n_nodes < 3 {
        return edges;
    }
    
    // Connect nodes in a cycle
    for i in 0..n_nodes {
        edges.push((i, (i + 1) % n_nodes));
    }
    
    edges
}

/// Generate a planar graph using grid structure
pub fn generate_grid_graph(rows: usize, cols: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    
    // Helper to convert 2D coordinates to node index
    let node_index = |r, c| r * cols + c;
    
    // Add horizontal edges
    for r in 0..rows {
        for c in 0..(cols - 1) {
            edges.push((node_index(r, c), node_index(r, c + 1)));
        }
    }
    
    // Add vertical edges
    for r in 0..(rows - 1) {
        for c in 0..cols {
            edges.push((node_index(r, c), node_index(r + 1, c)));
        }
    }
    
    edges
}

/// Generate a bipartite graph
pub fn generate_bipartite_graph(
    left_nodes: usize,
    right_nodes: usize,
    edge_probability: f64,
) -> Result<Vec<(usize, usize)>, Box<dyn std::error::Error>> {
    if edge_probability < 0.0 || edge_probability > 1.0 {
        return Err("Edge probability must be between 0 and 1".into());
    }
    
    let mut rng = StdRng::seed_from_u64(42);
    let mut edges = Vec::new();
    
    // Connect left nodes to right nodes with given probability
    for i in 0..left_nodes {
        for j in 0..right_nodes {
            if rng.gen::<f64>() < edge_probability {
                edges.push((i, left_nodes + j));
            }
        }
    }
    
    Ok(edges)
}

/// Generate a regular graph (all nodes have same degree)
pub fn generate_regular_graph(
    n_nodes: usize,
    degree: usize,
) -> Result<Vec<(usize, usize)>, Box<dyn std::error::Error>> {
    if degree >= n_nodes {
        return Err("Degree must be less than number of nodes".into());
    }
    
    if (n_nodes * degree) % 2 != 0 {
        return Err("n_nodes * degree must be even for regular graph".into());
    }
    
    let mut rng = StdRng::seed_from_u64(42);
    let mut edges = HashSet::new();
    let mut node_degrees = vec![0; n_nodes];
    
    // Simple algorithm: randomly connect nodes until each has the desired degree
    let max_attempts = n_nodes * degree * 10;
    let mut attempts = 0;
    
    while node_degrees.iter().any(|&d| d < degree) && attempts < max_attempts {
        attempts += 1;
        
        // Find nodes that need more connections
        let available: Vec<usize> = (0..n_nodes)
            .filter(|&i| node_degrees[i] < degree)
            .collect();
        
        if available.len() < 2 {
            continue;
        }
        
        // Pick two random nodes that need connections
        let i = available[rng.gen_range(0..available.len())];
        let j = available[rng.gen_range(0..available.len())];
        
        if i != j && !edges.contains(&(i.min(j), i.max(j))) {
            edges.insert((i.min(j), i.max(j)));
            node_degrees[i] += 1;
            node_degrees[j] += 1;
        }
    }
    
    if node_degrees.iter().any(|&d| d < degree) {
        return Err("Failed to generate regular graph with given parameters".into());
    }
    
    Ok(edges.into_iter().collect())
}

/// Generate a tree graph (connected acyclic graph)
pub fn generate_tree_graph(n_nodes: usize) -> Vec<(usize, usize)> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut edges = Vec::new();
    
    if n_nodes == 0 {
        return edges;
    }
    
    // Use Prüfer sequence to generate random tree
    // For now, simple approach: connect each new node to a random existing node
    for i in 1..n_nodes {
        let parent = rng.gen_range(0..i);
        edges.push((parent, i));
    }
    
    edges
}

/// Calculate graph properties
pub struct GraphProperties {
    pub n_nodes: usize,
    pub n_edges: usize,
    pub avg_degree: f64,
    pub max_degree: usize,
    pub min_degree: usize,
    pub is_connected: bool,
    pub density: f64,
}

/// Analyze graph properties
pub fn analyze_graph(n_nodes: usize, edges: &[(usize, usize)]) -> GraphProperties {
    let mut degrees = vec![0; n_nodes];
    
    for (i, j) in edges {
        degrees[*i] += 1;
        degrees[*j] += 1;
    }
    
    let avg_degree = degrees.iter().sum::<usize>() as f64 / n_nodes as f64;
    let max_degree = *degrees.iter().max().unwrap_or(&0);
    let min_degree = *degrees.iter().min().unwrap_or(&0);
    
    // Simple connectivity check (BFS from node 0)
    let is_connected = if n_nodes > 0 {
        let mut visited = vec![false; n_nodes];
        let mut queue = vec![0];
        visited[0] = true;
        
        while let Some(node) = queue.pop() {
            for (i, j) in edges {
                if *i == node && !visited[*j] {
                    visited[*j] = true;
                    queue.push(*j);
                } else if *j == node && !visited[*i] {
                    visited[*i] = true;
                    queue.push(*i);
                }
            }
        }
        
        visited.iter().all(|&v| v)
    } else {
        true
    };
    
    let max_edges = n_nodes * (n_nodes - 1) / 2;
    let density = if max_edges > 0 {
        edges.len() as f64 / max_edges as f64
    } else {
        0.0
    };
    
    GraphProperties {
        n_nodes,
        n_edges: edges.len(),
        avg_degree,
        max_degree,
        min_degree,
        is_connected,
        density,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_complete_graph() {
        let edges = generate_complete_graph(5);
        assert_eq!(edges.len(), 10); // 5 * 4 / 2
    }
    
    #[test]
    fn test_cycle_graph() {
        let edges = generate_cycle_graph(5);
        assert_eq!(edges.len(), 5);
    }
    
    #[test]
    fn test_grid_graph() {
        let edges = generate_grid_graph(3, 3);
        // 3x3 grid has 12 edges: 6 horizontal + 6 vertical
        assert_eq!(edges.len(), 12);
    }
    
    #[test]
    fn test_tree_graph() {
        let edges = generate_tree_graph(10);
        assert_eq!(edges.len(), 9); // n-1 edges for n nodes
    }
}