//! Constraint Satisfaction Problem decomposition methods

use super::types::*;
use std::collections::{HashMap, HashSet, VecDeque};

/// Constraint satisfaction problem decomposer
pub struct ConstraintSatisfactionDecomposer {
    /// Decomposition strategy
    strategy: CSPDecompositionStrategy,
    /// Variable ordering heuristic
    variable_ordering: VariableOrderingHeuristic,
    /// Constraint propagation level
    propagation_level: PropagationLevel,
    /// Maximum cluster size
    max_cluster_size: usize,
}

impl ConstraintSatisfactionDecomposer {
    /// Create new CSP decomposer
    pub fn new() -> Self {
        Self {
            strategy: CSPDecompositionStrategy::TreeDecomposition,
            variable_ordering: VariableOrderingHeuristic::MinWidth,
            propagation_level: PropagationLevel::ArcConsistency,
            max_cluster_size: 10,
        }
    }

    /// Set decomposition strategy
    pub fn with_strategy(mut self, strategy: CSPDecompositionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set variable ordering heuristic
    pub fn with_variable_ordering(mut self, ordering: VariableOrderingHeuristic) -> Self {
        self.variable_ordering = ordering;
        self
    }

    /// Decompose CSP into subproblems
    pub fn decompose(&self, csp: &CSPProblem) -> Result<CSPDecomposition, String> {
        match self.strategy {
            CSPDecompositionStrategy::TreeDecomposition => {
                self.tree_decomposition(csp)
            }
            CSPDecompositionStrategy::ConstraintClustering => {
                self.constraint_clustering_decomposition(csp)
            }
            CSPDecompositionStrategy::CycleCutset => {
                self.cycle_cutset_decomposition(csp)
            }
            _ => {
                // Default to tree decomposition
                self.tree_decomposition(csp)
            }
        }
    }

    /// Tree decomposition of CSP
    fn tree_decomposition(&self, csp: &CSPProblem) -> Result<CSPDecomposition, String> {
        // Build primal graph
        let (primal_graph, var_names) = self.build_primal_graph(csp);
        
        // Find tree decomposition
        let tree_dec = self.find_tree_decomposition(&primal_graph, &var_names)?;
        
        // Extract clusters
        let clusters = self.extract_clusters_from_tree(&tree_dec, csp)?;
        
        // Build cluster tree
        let cluster_tree = self.build_cluster_tree(&clusters, &tree_dec)?;
        
        let separator_sets = self.compute_separator_sets(&cluster_tree);
        Ok(CSPDecomposition {
            clusters,
            cluster_tree,
            separator_sets,
            width: tree_dec.width,
        })
    }

    /// Build primal graph from CSP
    fn build_primal_graph(&self, csp: &CSPProblem) -> (Graph, Vec<String>) {
        let mut edges = Vec::new();
        let var_names: Vec<String> = csp.variables.keys().cloned().collect();
        let var_to_idx: HashMap<&str, usize> = var_names
            .iter()
            .enumerate()
            .map(|(i, v)| (v.as_str(), i))
            .collect();
        
        // Add edge between variables that appear in same constraint
        for constraint in &csp.constraints {
            let vars_in_constraint: Vec<_> = constraint.scope
                .iter()
                .filter_map(|v| var_to_idx.get(v.as_str()))
                .collect();
            
            // Add all pairs as edges
            for i in 0..vars_in_constraint.len() {
                for j in i+1..vars_in_constraint.len() {
                    edges.push(Edge {
                        from: *vars_in_constraint[i],
                        to: *vars_in_constraint[j],
                        weight: 1.0,
                    });
                }
            }
        }
        
        (Graph {
            num_nodes: csp.variables.len(),
            edges,
            node_weights: vec![1.0; csp.variables.len()],
        }, var_names)
    }

    /// Find tree decomposition using elimination ordering
    fn find_tree_decomposition(&self, graph: &Graph, var_names: &[String]) -> Result<TreeDecomposition, String> {
        // Get variable elimination ordering
        let ordering = self.get_elimination_ordering(graph)?;
        
        // Build tree decomposition from elimination ordering
        let mut bags = Vec::new();
        let mut tree_edges = Vec::new();
        let mut max_bag_size = 0;
        
        let mut remaining_vars: HashSet<usize> = (0..graph.num_nodes).collect();
        let mut adjacency = self.build_adjacency_list(graph);
        
        for &var in &ordering {
            if !remaining_vars.contains(&var) {
                continue;
            }
            
            // Create bag with variable and its remaining neighbors
            let mut bag_indices = HashSet::new();
            bag_indices.insert(var);
            
            // Add neighbors
            if let Some(neighbors) = adjacency.get(&var) {
                for &neighbor in neighbors {
                    if remaining_vars.contains(&neighbor) {
                        bag_indices.insert(neighbor);
                    }
                }
            }
            
            // Connect neighbors (make clique)
            let bag_vars: Vec<_> = bag_indices.iter().cloned().collect();
            for i in 0..bag_vars.len() {
                for j in i+1..bag_vars.len() {
                    let var_i = bag_vars[i];
                    let var_j = bag_vars[j];
                    adjacency.entry(var_i).or_insert_with(HashSet::new).insert(var_j);
                    adjacency.entry(var_j).or_insert_with(HashSet::new).insert(var_i);
                }
            }
            
            // Convert indices to variable names for the bag
            let bag: HashSet<String> = bag_indices.iter()
                .filter_map(|&idx| var_names.get(idx).cloned())
                .collect();
            bags.push(bag);
            max_bag_size = max_bag_size.max(bag_indices.len());
            
            // Remove variable
            remaining_vars.remove(&var);
            if let Some(neighbors) = adjacency.get_mut(&var) {
                neighbors.clear();
            }
            for adj_list in adjacency.values_mut() {
                adj_list.remove(&var);
            }
        }
        
        // Build tree structure (simplified - just linear chain)
        for i in 1..bags.len() {
            tree_edges.push((i-1, i));
        }
        
        Ok(TreeDecomposition {
            bags,
            tree_edges,
            width: max_bag_size.saturating_sub(1),
        })
    }

    /// Get variable elimination ordering
    fn get_elimination_ordering(&self, graph: &Graph) -> Result<Vec<usize>, String> {
        match self.variable_ordering {
            VariableOrderingHeuristic::MinWidth => {
                self.min_width_ordering(graph)
            }
            VariableOrderingHeuristic::MaxCardinality => {
                self.max_cardinality_ordering(graph)
            }
            VariableOrderingHeuristic::MinFillIn => {
                self.min_fill_in_ordering(graph)
            }
            _ => {
                // Default to min width
                self.min_width_ordering(graph)
            }
        }
    }

    /// Minimum width ordering heuristic
    fn min_width_ordering(&self, graph: &Graph) -> Result<Vec<usize>, String> {
        let mut ordering = Vec::new();
        let mut remaining: HashSet<usize> = (0..graph.num_nodes).collect();
        let mut adjacency = self.build_adjacency_list(graph);
        
        while !remaining.is_empty() {
            // Find variable with minimum degree
            let mut min_degree = usize::MAX;
            let mut best_var = 0;
            
            for &var in &remaining {
                let degree = adjacency.get(&var)
                    .map(|adj| adj.iter().filter(|&&v| remaining.contains(&v)).count())
                    .unwrap_or(0);
                
                if degree < min_degree {
                    min_degree = degree;
                    best_var = var;
                }
            }
            
            ordering.push(best_var);
            remaining.remove(&best_var);
            
            // Make neighbors adjacent (fill-in)
            if let Some(neighbors) = adjacency.get(&best_var) {
                let active_neighbors: Vec<_> = neighbors.iter()
                    .filter(|&&v| remaining.contains(&v))
                    .cloned()
                    .collect();
                
                for i in 0..active_neighbors.len() {
                    for j in i+1..active_neighbors.len() {
                        let var_i = active_neighbors[i];
                        let var_j = active_neighbors[j];
                        adjacency.entry(var_i).or_insert_with(HashSet::new).insert(var_j);
                        adjacency.entry(var_j).or_insert_with(HashSet::new).insert(var_i);
                    }
                }
            }
        }
        
        Ok(ordering)
    }

    /// Maximum cardinality ordering heuristic
    fn max_cardinality_ordering(&self, graph: &Graph) -> Result<Vec<usize>, String> {
        let mut ordering = Vec::new();
        let mut numbered = HashSet::new();
        let mut cardinality = vec![0; graph.num_nodes];
        let adjacency = self.build_adjacency_list(graph);
        
        for _ in 0..graph.num_nodes {
            // Find unnumbered vertex with maximum cardinality
            let mut max_cardinality = 0;
            let mut best_var = 0;
            
            for var in 0..graph.num_nodes {
                if !numbered.contains(&var) && cardinality[var] >= max_cardinality {
                    max_cardinality = cardinality[var];
                    best_var = var;
                }
            }
            
            ordering.push(best_var);
            numbered.insert(best_var);
            
            // Update cardinality of neighbors
            if let Some(neighbors) = adjacency.get(&best_var) {
                for &neighbor in neighbors {
                    if !numbered.contains(&neighbor) {
                        cardinality[neighbor] += 1;
                    }
                }
            }
        }
        
        ordering.reverse();
        Ok(ordering)
    }

    /// Minimum fill-in ordering heuristic
    fn min_fill_in_ordering(&self, graph: &Graph) -> Result<Vec<usize>, String> {
        let mut ordering = Vec::new();
        let mut remaining: HashSet<usize> = (0..graph.num_nodes).collect();
        let mut adjacency = self.build_adjacency_list(graph);
        
        while !remaining.is_empty() {
            let mut min_fill_in = usize::MAX;
            let mut best_var = 0;
            
            for &var in &remaining {
                let fill_in = self.compute_fill_in(&adjacency, var, &remaining);
                if fill_in < min_fill_in {
                    min_fill_in = fill_in;
                    best_var = var;
                }
            }
            
            ordering.push(best_var);
            remaining.remove(&best_var);
            
            // Add fill-in edges
            if let Some(neighbors) = adjacency.get(&best_var) {
                let active_neighbors: Vec<_> = neighbors.iter()
                    .filter(|&&v| remaining.contains(&v))
                    .cloned()
                    .collect();
                
                for i in 0..active_neighbors.len() {
                    for j in i+1..active_neighbors.len() {
                        let var_i = active_neighbors[i];
                        let var_j = active_neighbors[j];
                        adjacency.entry(var_i).or_insert_with(HashSet::new).insert(var_j);
                        adjacency.entry(var_j).or_insert_with(HashSet::new).insert(var_i);
                    }
                }
            }
        }
        
        Ok(ordering)
    }

    /// Compute fill-in edges if variable is eliminated
    fn compute_fill_in(&self, adjacency: &HashMap<usize, HashSet<usize>>, var: usize, remaining: &HashSet<usize>) -> usize {
        let neighbors = adjacency.get(&var).map(|adj| {
            adj.iter().filter(|&&v| remaining.contains(&v)).cloned().collect::<Vec<_>>()
        }).unwrap_or_default();
        
        let mut fill_in_count = 0;
        for i in 0..neighbors.len() {
            for j in i+1..neighbors.len() {
                let var_i = neighbors[i];
                let var_j = neighbors[j];
                
                // Check if edge already exists
                if !adjacency.get(&var_i).map(|adj| adj.contains(&var_j)).unwrap_or(false) {
                    fill_in_count += 1;
                }
            }
        }
        
        fill_in_count
    }

    /// Build adjacency list from graph
    fn build_adjacency_list(&self, graph: &Graph) -> HashMap<usize, HashSet<usize>> {
        let mut adjacency = HashMap::new();
        
        for edge in &graph.edges {
            adjacency.entry(edge.from).or_insert_with(HashSet::new).insert(edge.to);
            adjacency.entry(edge.to).or_insert_with(HashSet::new).insert(edge.from);
        }
        
        adjacency
    }

    /// Extract clusters from tree decomposition
    fn extract_clusters_from_tree(&self, tree_dec: &TreeDecomposition, csp: &CSPProblem) -> Result<Vec<CSPCluster>, String> {
        let mut clusters = Vec::new();
        let var_names: Vec<_> = csp.variables.keys().collect();
        
        for (i, bag) in tree_dec.bags.iter().enumerate() {
            let variables: HashSet<String> = bag.clone();
            
            // Find constraints that involve only variables in this bag
            let mut cluster_constraints = Vec::new();
            for (constraint_id, constraint) in csp.constraints.iter().enumerate() {
                let constraint_vars: HashSet<_> = constraint.scope.iter().collect();
                let bag_vars: HashSet<_> = variables.iter().collect();
                
                if constraint_vars.is_subset(&bag_vars) {
                    cluster_constraints.push(constraint_id);
                }
            }
            
            clusters.push(CSPCluster {
                id: i,
                variables,
                constraints: cluster_constraints,
                subproblem: None, // Would be filled in later
            });
        }
        
        Ok(clusters)
    }

    /// Build cluster tree
    fn build_cluster_tree(&self, clusters: &[CSPCluster], tree_dec: &TreeDecomposition) -> Result<ClusterTree, String> {
        let mut nodes = Vec::new();
        
        for (i, cluster) in clusters.iter().enumerate() {
            let separator = if i > 0 {
                // Compute separator with parent (simplified)
                clusters[i-1].variables.intersection(&cluster.variables).cloned().collect()
            } else {
                HashSet::new()
            };
            
            nodes.push(TreeNode {
                id: i,
                cluster_id: i,
                separator,
                children: if i < clusters.len() - 1 { vec![i + 1] } else { vec![] },
                parent: if i > 0 { Some(i - 1) } else { None },
            });
        }
        
        Ok(ClusterTree {
            nodes,
            edges: tree_dec.tree_edges.clone(),
            root: 0,
        })
    }

    /// Compute separator sets
    fn compute_separator_sets(&self, cluster_tree: &ClusterTree) -> Vec<HashSet<String>> {
        cluster_tree.nodes.iter().map(|node| node.separator.clone()).collect()
    }

    /// Constraint clustering decomposition
    fn constraint_clustering_decomposition(&self, csp: &CSPProblem) -> Result<CSPDecomposition, String> {
        // Group constraints by shared variables
        let mut constraint_clusters = Vec::new();
        let mut assigned_constraints = HashSet::new();
        
        for (i, constraint) in csp.constraints.iter().enumerate() {
            if assigned_constraints.contains(&i) {
                continue;
            }
            
            let mut cluster_vars = HashSet::new();
            let mut cluster_constraints = vec![i];
            assigned_constraints.insert(i);
            
            // Add variables from this constraint
            for var in &constraint.scope {
                cluster_vars.insert(var.clone());
            }
            
            // Find other constraints sharing variables
            for (j, other_constraint) in csp.constraints.iter().enumerate() {
                if i != j && !assigned_constraints.contains(&j) {
                    let shared_vars = constraint.scope.iter()
                        .filter(|var| other_constraint.scope.contains(var))
                        .count();
                    
                    if shared_vars > 0 {
                        cluster_constraints.push(j);
                        assigned_constraints.insert(j);
                        
                        for var in &other_constraint.scope {
                            cluster_vars.insert(var.clone());
                        }
                    }
                }
            }
            
            constraint_clusters.push(CSPCluster {
                id: constraint_clusters.len(),
                variables: cluster_vars,
                constraints: cluster_constraints,
                subproblem: None,
            });
        }
        
        // Build simple cluster tree
        let mut tree_nodes = Vec::new();
        let mut tree_edges = Vec::new();
        
        for (i, cluster) in constraint_clusters.iter().enumerate() {
            tree_nodes.push(TreeNode {
                id: i,
                cluster_id: i,
                separator: HashSet::new(),
                children: if i < constraint_clusters.len() - 1 { vec![i + 1] } else { vec![] },
                parent: if i > 0 { Some(i - 1) } else { None },
            });
            
            if i > 0 {
                tree_edges.push((i - 1, i));
            }
        }
        
        let cluster_tree = ClusterTree {
            nodes: tree_nodes,
            edges: tree_edges,
            root: 0,
        };
        
        let separator_sets = self.compute_separator_sets(&cluster_tree);
        
        Ok(CSPDecomposition {
            clusters: constraint_clusters,
            cluster_tree,
            separator_sets,
            width: 0, // Would compute properly
        })
    }

    /// Cycle cutset decomposition
    fn cycle_cutset_decomposition(&self, csp: &CSPProblem) -> Result<CSPDecomposition, String> {
        // Find cycle cutset (variables whose removal makes graph acyclic)
        let cutset = self.find_cycle_cutset(csp)?;
        
        // Create clusters based on remaining tree structure
        // This is simplified - a full implementation would be more complex
        self.constraint_clustering_decomposition(csp)
    }

    /// Find cycle cutset using greedy heuristic
    fn find_cycle_cutset(&self, csp: &CSPProblem) -> Result<Vec<String>, String> {
        let (primal_graph, var_names) = self.build_primal_graph(csp);
        let mut cutset = Vec::new();
        let mut remaining_edges = primal_graph.edges.clone();
        
        // Greedy removal of high-degree vertices
        while self.has_cycles(&remaining_edges, primal_graph.num_nodes) {
            let degrees = self.compute_degrees(&remaining_edges, primal_graph.num_nodes);
            
            if let Some((max_var, _)) = degrees.iter().enumerate().max_by_key(|(_, &deg)| deg) {
                // Remove this variable from graph
                remaining_edges.retain(|edge| edge.from != max_var && edge.to != max_var);
                
                // Add to cutset
                if let Some(var_name) = var_names.get(max_var) {
                    cutset.push(var_name.clone());
                }
            } else {
                break;
            }
        }
        
        Ok(cutset)
    }

    /// Check if graph has cycles using DFS
    fn has_cycles(&self, edges: &[Edge], num_nodes: usize) -> bool {
        let mut visited = vec![false; num_nodes];
        let mut rec_stack = vec![false; num_nodes];
        let mut adjacency = HashMap::new();
        
        // Build adjacency list
        for edge in edges {
            adjacency.entry(edge.from).or_insert_with(Vec::new).push(edge.to);
        }
        
        // DFS from each unvisited node
        for node in 0..num_nodes {
            if !visited[node] {
                if self.dfs_has_cycle(node, &adjacency, &mut visited, &mut rec_stack) {
                    return true;
                }
            }
        }
        
        false
    }

    /// DFS cycle detection helper
    fn dfs_has_cycle(
        &self,
        node: usize,
        adjacency: &HashMap<usize, Vec<usize>>,
        visited: &mut [bool],
        rec_stack: &mut [bool],
    ) -> bool {
        visited[node] = true;
        rec_stack[node] = true;
        
        if let Some(neighbors) = adjacency.get(&node) {
            for &neighbor in neighbors {
                if !visited[neighbor] {
                    if self.dfs_has_cycle(neighbor, adjacency, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack[neighbor] {
                    return true;
                }
            }
        }
        
        rec_stack[node] = false;
        false
    }

    /// Compute vertex degrees
    fn compute_degrees(&self, edges: &[Edge], num_nodes: usize) -> Vec<usize> {
        let mut degrees = vec![0; num_nodes];
        
        for edge in edges {
            degrees[edge.from] += 1;
            degrees[edge.to] += 1;
        }
        
        degrees
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csp_decomposer_creation() {
        let decomposer = ConstraintSatisfactionDecomposer::new();
        assert_eq!(decomposer.max_cluster_size, 10);
    }

    #[test]
    fn test_primal_graph_building() {
        let decomposer = ConstraintSatisfactionDecomposer::new();
        
        // Create simple CSP
        let mut variables = HashMap::new();
        variables.insert("x0".to_string(), Domain_CSP { values: vec![0, 1] });
        variables.insert("x1".to_string(), Domain_CSP { values: vec![0, 1] });
        variables.insert("x2".to_string(), Domain_CSP { values: vec![0, 1] });
        
        let constraints = vec![
            CSPConstraint {
                id: 0,
                scope: vec!["x0".to_string(), "x1".to_string()],
                constraint_type: ConstraintType::AllDifferent,
                tuples: None,
            },
            CSPConstraint {
                id: 1,
                scope: vec!["x1".to_string(), "x2".to_string()],
                constraint_type: ConstraintType::AllDifferent,
                tuples: None,
            },
        ];
        
        let csp = CSPProblem {
            variables,
            constraints,
            constraint_graph: ConstraintGraph {
                adjacency: HashMap::new(),
                hyperedges: Vec::new(),
            },
        };
        
        let (graph, var_names) = decomposer.build_primal_graph(&csp);
        assert_eq!(graph.num_nodes, 3);
        assert_eq!(graph.edges.len(), 2); // (x0,x1) and (x1,x2)
        assert_eq!(var_names.len(), 3);
    }

    #[test]
    fn test_constraint_clustering() {
        let decomposer = ConstraintSatisfactionDecomposer::new()
            .with_strategy(CSPDecompositionStrategy::ConstraintClustering);
        
        // Create CSP with disconnected constraint groups
        let mut variables = HashMap::new();
        for i in 0..5 {
            variables.insert(format!("x{}", i), Domain_CSP { values: vec![0, 1] });
        }
        
        let constraints = vec![
            CSPConstraint {
                id: 0,
                scope: vec!["x0".to_string(), "x1".to_string()],
                constraint_type: ConstraintType::AllDifferent,
                tuples: None,
            },
            CSPConstraint {
                id: 1,
                scope: vec!["x1".to_string(), "x2".to_string()],
                constraint_type: ConstraintType::AllDifferent,
                tuples: None,
            },
            CSPConstraint {
                id: 2,
                scope: vec!["x3".to_string(), "x4".to_string()],
                constraint_type: ConstraintType::AllDifferent,
                tuples: None,
            },
        ];
        
        let csp = CSPProblem {
            variables,
            constraints,
            constraint_graph: ConstraintGraph {
                adjacency: HashMap::new(),
                hyperedges: Vec::new(),
            },
        };
        
        let result = decomposer.decompose(&csp);
        assert!(result.is_ok());
        
        let decomposition = result.unwrap();
        // Should create at least 2 clusters due to disconnected constraints
        assert!(decomposition.clusters.len() >= 2);
    }
}