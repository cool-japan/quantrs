//! Example problems for the problem DSL.

/// Example problems
pub struct Examples;

impl Examples {
    /// Simple binary problem
    pub const SIMPLE_BINARY: &str = r#"
        var x binary;
        var y binary;
        
        minimize x + y;
        
        subject to
            x + y >= 1;
    "#;

    /// Traveling salesman problem
    pub const TSP: &str = r#"
        param n = 4;
        param distances = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ];
        
        var x[n, n] binary;
        
        minimize sum(i in 0..n, j in 0..n: distances[i][j] * x[i,j]);
        
        subject to
            // Each city visited exactly once
            forall(i in 0..n): sum(j in 0..n: x[i,j]) == 1;
            forall(j in 0..n): sum(i in 0..n: x[i,j]) == 1;
    "#;

    /// Graph coloring
    pub const GRAPH_COLORING: &str = r#"
        param n_vertices = 5;
        param n_colors = 3;
        param edges = [(0,1), (1,2), (2,3), (3,4), (4,0)];
        
        var color[n_vertices, n_colors] binary;
        
        minimize sum(v in 0..n_vertices, c in 0..n_colors: c * color[v,c]);
        
        subject to
            // Each vertex has exactly one color
            forall(v in 0..n_vertices): sum(c in 0..n_colors: color[v,c]) == 1;
            
            // Adjacent vertices have different colors
            forall((u,v) in edges, c in 0..n_colors):
                color[u,c] + color[v,c] <= 1;
    "#;
}

/// Get example by name
pub fn get_example(name: &str) -> Option<&str> {
    match name {
        "simple_binary" => Some(Examples::SIMPLE_BINARY),
        "tsp" => Some(Examples::TSP),
        "graph_coloring" => Some(Examples::GRAPH_COLORING),
        _ => None,
    }
}
