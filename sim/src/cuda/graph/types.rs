//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "advanced_math")]
use crate::cuda::streams::{CudaStream, CudaStreamHandle};
use crate::error::{Result, SimulatorError};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use super::functions::{CudaGraphExecHandle, CudaGraphHandle, CudaGraphNodeHandle};

/// Host function callback parameters
#[derive(Debug, Clone)]
pub struct HostNodeParams {
    /// Callback function index (we use index because function pointers aren't Clone)
    pub callback_id: usize,
    /// User data pointer
    pub user_data: usize,
}
/// CUDA Graph representation
pub struct CudaGraph {
    /// Graph handle
    handle: CudaGraphHandle,
    /// Graph nodes indexed by ID
    nodes: HashMap<usize, GraphNode>,
    /// Next node ID
    next_node_id: AtomicUsize,
    /// Graph name for debugging
    pub(crate) name: Option<String>,
    /// Whether the graph is finalized (no more nodes can be added)
    finalized: bool,
    /// Graph creation timestamp
    created_at: std::time::Instant,
    /// Total number of kernel nodes
    kernel_count: usize,
    /// Total number of memory operations
    mem_op_count: usize,
}
impl CudaGraph {
    /// Create a new CUDA graph
    pub fn new() -> Self {
        static GRAPH_COUNTER: AtomicUsize = AtomicUsize::new(0);
        let handle = GRAPH_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self {
            handle,
            nodes: HashMap::new(),
            next_node_id: AtomicUsize::new(0),
            name: None,
            finalized: false,
            created_at: std::time::Instant::now(),
            kernel_count: 0,
            mem_op_count: 0,
        }
    }
    /// Create a new graph with a name
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
    /// Get the graph handle
    pub fn handle(&self) -> CudaGraphHandle {
        self.handle
    }
    /// Get the number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    /// Get the number of kernel nodes
    pub fn kernel_count(&self) -> usize {
        self.kernel_count
    }
    /// Get the number of memory operation nodes
    pub fn mem_op_count(&self) -> usize {
        self.mem_op_count
    }
    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    /// Check if the graph is finalized
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }
    /// Add a kernel node to the graph
    pub fn add_kernel_node(
        &mut self,
        params: KernelNodeParams,
        dependencies: &[usize],
    ) -> Result<usize> {
        if self.finalized {
            return Err(SimulatorError::InvalidOperation(
                "Cannot add nodes to finalized graph".to_string(),
            ));
        }
        let node_id = self.next_node_id.fetch_add(1, Ordering::SeqCst);
        let mut node = GraphNode::new(node_id, GraphNodeType::Kernel);
        node.kernel_params = Some(params);
        for &dep_id in dependencies {
            if !self.nodes.contains_key(&dep_id) {
                return Err(SimulatorError::InvalidInput(format!(
                    "Dependency node {} does not exist",
                    dep_id
                )));
            }
            node.add_dependency(dep_id);
        }
        self.nodes.insert(node_id, node);
        self.kernel_count += 1;
        Ok(node_id)
    }
    /// Add a memory copy node to the graph
    pub fn add_memcpy_node(
        &mut self,
        params: MemCopyNodeParams,
        dependencies: &[usize],
    ) -> Result<usize> {
        if self.finalized {
            return Err(SimulatorError::InvalidOperation(
                "Cannot add nodes to finalized graph".to_string(),
            ));
        }
        let node_id = self.next_node_id.fetch_add(1, Ordering::SeqCst);
        let mut node = GraphNode::new(node_id, GraphNodeType::MemCopy);
        node.memcopy_params = Some(params);
        for &dep_id in dependencies {
            if !self.nodes.contains_key(&dep_id) {
                return Err(SimulatorError::InvalidInput(format!(
                    "Dependency node {} does not exist",
                    dep_id
                )));
            }
            node.add_dependency(dep_id);
        }
        self.nodes.insert(node_id, node);
        self.mem_op_count += 1;
        Ok(node_id)
    }
    /// Add a memory set node to the graph
    pub fn add_memset_node(
        &mut self,
        params: MemSetNodeParams,
        dependencies: &[usize],
    ) -> Result<usize> {
        if self.finalized {
            return Err(SimulatorError::InvalidOperation(
                "Cannot add nodes to finalized graph".to_string(),
            ));
        }
        let node_id = self.next_node_id.fetch_add(1, Ordering::SeqCst);
        let mut node = GraphNode::new(node_id, GraphNodeType::MemSet);
        node.memset_params = Some(params);
        for &dep_id in dependencies {
            if !self.nodes.contains_key(&dep_id) {
                return Err(SimulatorError::InvalidInput(format!(
                    "Dependency node {} does not exist",
                    dep_id
                )));
            }
            node.add_dependency(dep_id);
        }
        self.nodes.insert(node_id, node);
        self.mem_op_count += 1;
        Ok(node_id)
    }
    /// Add an empty node (for synchronization)
    pub fn add_empty_node(&mut self, dependencies: &[usize]) -> Result<usize> {
        if self.finalized {
            return Err(SimulatorError::InvalidOperation(
                "Cannot add nodes to finalized graph".to_string(),
            ));
        }
        let node_id = self.next_node_id.fetch_add(1, Ordering::SeqCst);
        let mut node = GraphNode::new(node_id, GraphNodeType::Empty);
        for &dep_id in dependencies {
            if !self.nodes.contains_key(&dep_id) {
                return Err(SimulatorError::InvalidInput(format!(
                    "Dependency node {} does not exist",
                    dep_id
                )));
            }
            node.add_dependency(dep_id);
        }
        self.nodes.insert(node_id, node);
        Ok(node_id)
    }
    /// Add a child graph node
    pub fn add_child_graph(
        &mut self,
        child_graph: &CudaGraph,
        dependencies: &[usize],
    ) -> Result<usize> {
        if self.finalized {
            return Err(SimulatorError::InvalidOperation(
                "Cannot add nodes to finalized graph".to_string(),
            ));
        }
        if !child_graph.is_finalized() {
            return Err(SimulatorError::InvalidOperation(
                "Child graph must be finalized".to_string(),
            ));
        }
        let node_id = self.next_node_id.fetch_add(1, Ordering::SeqCst);
        let mut node = GraphNode::new(node_id, GraphNodeType::Graph);
        for &dep_id in dependencies {
            if !self.nodes.contains_key(&dep_id) {
                return Err(SimulatorError::InvalidInput(format!(
                    "Dependency node {} does not exist",
                    dep_id
                )));
            }
            node.add_dependency(dep_id);
        }
        self.nodes.insert(node_id, node);
        self.kernel_count += child_graph.kernel_count;
        self.mem_op_count += child_graph.mem_op_count;
        Ok(node_id)
    }
    /// Get a node by ID
    pub fn get_node(&self, node_id: usize) -> Option<&GraphNode> {
        self.nodes.get(&node_id)
    }
    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, node_id: usize) -> Option<&mut GraphNode> {
        if self.finalized {
            return None;
        }
        self.nodes.get_mut(&node_id)
    }
    /// Update kernel node parameters
    pub fn update_kernel_params(&mut self, node_id: usize, params: KernelNodeParams) -> Result<()> {
        let node = self.nodes.get_mut(&node_id).ok_or_else(|| {
            SimulatorError::InvalidInput(format!("Node {} does not exist", node_id))
        })?;
        if node.node_type != GraphNodeType::Kernel {
            return Err(SimulatorError::InvalidOperation(format!(
                "Node {} is not a kernel node",
                node_id
            )));
        }
        node.kernel_params = Some(params);
        Ok(())
    }
    /// Finalize the graph (no more nodes can be added)
    pub fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        self.validate()?;
        self.finalized = true;
        Ok(())
    }
    /// Validate the graph structure
    pub fn validate(&self) -> Result<()> {
        let mut visited = HashMap::new();
        let mut rec_stack = HashMap::new();
        for &node_id in self.nodes.keys() {
            if self.has_cycle_dfs(node_id, &mut visited, &mut rec_stack)? {
                return Err(SimulatorError::InvalidOperation(
                    "Graph contains a cycle".to_string(),
                ));
            }
        }
        Ok(())
    }
    /// DFS helper for cycle detection
    fn has_cycle_dfs(
        &self,
        node_id: usize,
        visited: &mut HashMap<usize, bool>,
        rec_stack: &mut HashMap<usize, bool>,
    ) -> Result<bool> {
        if let Some(&in_stack) = rec_stack.get(&node_id) {
            if in_stack {
                return Ok(true);
            }
        }
        if let Some(&is_visited) = visited.get(&node_id) {
            if is_visited {
                return Ok(false);
            }
        }
        visited.insert(node_id, true);
        rec_stack.insert(node_id, true);
        if let Some(node) = self.nodes.get(&node_id) {
            for &dep_id in &node.dependencies {
                if self.has_cycle_dfs(dep_id, visited, rec_stack)? {
                    return Ok(true);
                }
            }
        }
        rec_stack.insert(node_id, false);
        Ok(false)
    }
    /// Get root nodes (no dependencies)
    pub fn get_root_nodes(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.is_root())
            .map(|(&id, _)| id)
            .collect()
    }
    /// Get leaf nodes (no dependents)
    pub fn get_leaf_nodes(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|(_, node)| node.is_leaf(self))
            .map(|(&id, _)| id)
            .collect()
    }
    /// Get nodes in topological order
    pub fn topological_order(&self) -> Result<Vec<usize>> {
        let mut result = Vec::new();
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        for &id in self.nodes.keys() {
            in_degree.entry(id).or_insert(0);
        }
        for node in self.nodes.values() {
            for &dep_id in &node.dependencies {
                *in_degree.entry(dep_id).or_insert(0) += 0;
            }
        }
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        for &id in self.nodes.keys() {
            in_degree.insert(id, 0);
        }
        for _node in self.nodes.values() {}
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        for (&id, node) in &self.nodes {
            in_degree.insert(id, node.dependencies.len());
        }
        let mut queue: Vec<usize> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut dependents: HashMap<usize, Vec<usize>> = HashMap::new();
        for (&id, node) in &self.nodes {
            for &dep_id in &node.dependencies {
                dependents.entry(dep_id).or_default().push(id);
            }
        }
        while let Some(node_id) = queue.pop() {
            result.push(node_id);
            if let Some(deps) = dependents.get(&node_id) {
                for &dep_id in deps {
                    if let Some(deg) = in_degree.get_mut(&dep_id) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push(dep_id);
                        }
                    }
                }
            }
        }
        if result.len() != self.nodes.len() {
            return Err(SimulatorError::InvalidOperation(
                "Graph contains a cycle - topological sort failed".to_string(),
            ));
        }
        Ok(result)
    }
    /// Instantiate the graph for execution
    pub fn instantiate(&self) -> Result<CudaGraphExec> {
        self.instantiate_with_flags(GraphInstantiationFlags::Default)
    }
    /// Instantiate with flags
    pub fn instantiate_with_flags(&self, flags: GraphInstantiationFlags) -> Result<CudaGraphExec> {
        if !self.finalized {
            return Err(SimulatorError::InvalidOperation(
                "Graph must be finalized before instantiation".to_string(),
            ));
        }
        let execution_order = self.topological_order()?;
        CudaGraphExec::new(self, execution_order, flags)
    }
    /// Clone the graph structure (deep clone)
    pub fn clone_graph(&self) -> Result<Self> {
        let mut new_graph = Self::new();
        new_graph.name = self.name.clone();
        for (&id, node) in &self.nodes {
            let mut new_node = GraphNode::new(id, node.node_type);
            new_node.dependencies = node.dependencies.clone();
            new_node.name = node.name.clone();
            new_node.kernel_params = node.kernel_params.clone();
            new_node.memcopy_params = node.memcopy_params.clone();
            new_node.memset_params = node.memset_params.clone();
            new_node.host_params = node.host_params.clone();
            new_graph.nodes.insert(id, new_node);
        }
        new_graph.next_node_id = AtomicUsize::new(self.next_node_id.load(Ordering::SeqCst));
        new_graph.kernel_count = self.kernel_count;
        new_graph.mem_op_count = self.mem_op_count;
        Ok(new_graph)
    }
    /// Get graph statistics
    pub fn get_stats(&self) -> GraphStats {
        let mut edge_count = 0;
        let mut max_depth = 0;
        let mut max_fan_out = 0;
        let mut max_fan_in = 0;
        let mut dependents: HashMap<usize, Vec<usize>> = HashMap::new();
        for (&id, node) in &self.nodes {
            edge_count += node.dependencies.len();
            max_fan_in = max_fan_in.max(node.dependencies.len());
            for &dep_id in &node.dependencies {
                dependents.entry(dep_id).or_default().push(id);
            }
        }
        for deps in dependents.values() {
            max_fan_out = max_fan_out.max(deps.len());
        }
        let roots = self.get_root_nodes();
        let mut depths: HashMap<usize, usize> = HashMap::new();
        for root in roots {
            depths.insert(root, 0);
        }
        if let Ok(order) = self.topological_order() {
            for node_id in order {
                if let Some(node) = self.nodes.get(&node_id) {
                    let depth = node
                        .dependencies
                        .iter()
                        .filter_map(|&dep_id| depths.get(&dep_id))
                        .max()
                        .map(|d| d + 1)
                        .unwrap_or(0);
                    depths.insert(node_id, depth);
                    max_depth = max_depth.max(depth);
                }
            }
        }
        GraphStats {
            node_count: self.nodes.len(),
            edge_count,
            kernel_count: self.kernel_count,
            mem_op_count: self.mem_op_count,
            max_depth,
            max_fan_out,
            max_fan_in,
            root_count: self.get_root_nodes().len(),
            leaf_count: self.get_leaf_nodes().len(),
        }
    }
}
/// Graph execution scheduler for quantum circuits
pub struct QuantumGraphScheduler {
    /// Pre-built graphs for common circuit patterns
    pub(crate) cached_graphs: HashMap<String, CudaGraphExec>,
    /// Maximum cache size
    max_cache_size: usize,
    /// Cache hits
    cache_hits: AtomicUsize,
    /// Cache misses
    cache_misses: AtomicUsize,
}
impl QuantumGraphScheduler {
    /// Create a new scheduler
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            cached_graphs: HashMap::new(),
            max_cache_size,
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
        }
    }
    /// Get or create a graph for a circuit pattern
    pub fn get_or_create<F>(&mut self, pattern_key: &str, create_fn: F) -> Result<&CudaGraphExec>
    where
        F: FnOnce() -> Result<CudaGraph>,
    {
        if self.cached_graphs.contains_key(pattern_key) {
            self.cache_hits.fetch_add(1, Ordering::SeqCst);
            return Ok(self.cached_graphs.get(pattern_key).expect("key exists"));
        }
        self.cache_misses.fetch_add(1, Ordering::SeqCst);
        if self.cached_graphs.len() >= self.max_cache_size {
            if let Some(key) = self.cached_graphs.keys().next().cloned() {
                self.cached_graphs.remove(&key);
            }
        }
        let graph = create_fn()?;
        let exec = graph.instantiate()?;
        self.cached_graphs.insert(pattern_key.to_string(), exec);
        Ok(self.cached_graphs.get(pattern_key).expect("just inserted"))
    }
    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cached_graphs.clear();
    }
    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (
            self.cache_hits.load(Ordering::SeqCst),
            self.cache_misses.load(Ordering::SeqCst),
        )
    }
    /// Get cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::SeqCst);
        let misses = self.cache_misses.load(Ordering::SeqCst);
        let total = hits + misses;
        if total == 0 {
            return 0.0;
        }
        hits as f64 / total as f64
    }
}
/// Graph instantiation flags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphInstantiationFlags {
    /// Default flags
    Default,
    /// Device launch - allow child graph nodes
    DeviceLaunch,
    /// Use node priority
    UsePriority,
    /// Auto free on launch
    AutoFreeOnLaunch,
    /// Upload the graph to the device
    Upload,
}
/// Memory copy node parameters
#[derive(Debug, Clone)]
pub struct MemCopyNodeParams {
    /// Source pointer
    pub src: usize,
    /// Destination pointer
    pub dst: usize,
    /// Size in bytes
    pub size: usize,
    /// Copy kind (host-to-device, device-to-host, device-to-device)
    pub kind: MemCopyKind,
}
/// Memory set node parameters
#[derive(Debug, Clone)]
pub struct MemSetNodeParams {
    /// Destination pointer
    pub dst: usize,
    /// Value to set (as u8)
    pub value: u8,
    /// Size in bytes
    pub size: usize,
}
/// Statistics about a CUDA graph
#[derive(Debug, Clone)]
pub struct GraphStats {
    /// Total number of nodes
    pub node_count: usize,
    /// Total number of edges (dependencies)
    pub edge_count: usize,
    /// Number of kernel nodes
    pub kernel_count: usize,
    /// Number of memory operation nodes
    pub mem_op_count: usize,
    /// Maximum depth (longest path from root to leaf)
    pub max_depth: usize,
    /// Maximum fan-out (most dependents for any node)
    pub max_fan_out: usize,
    /// Maximum fan-in (most dependencies for any node)
    pub max_fan_in: usize,
    /// Number of root nodes
    pub root_count: usize,
    /// Number of leaf nodes
    pub leaf_count: usize,
}
/// Result of graph update operation
#[derive(Debug, Clone)]
pub struct GraphUpdateResult {
    /// Whether the update was successful
    pub success: bool,
    /// Node that caused the error (if any)
    pub error_node: Option<usize>,
    /// Detailed update result
    pub update_result: GraphExecUpdateResult,
}
/// Graph capture mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphCaptureMode {
    /// Thread-local capture mode (default)
    ThreadLocal,
    /// Global capture mode across all threads
    Global,
    /// Relaxed capture mode (allows some non-captured operations)
    Relaxed,
}
/// Graph node type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphNodeType {
    /// Kernel launch node
    Kernel,
    /// Memory copy node
    MemCopy,
    /// Memory set node
    MemSet,
    /// Host function callback node
    Host,
    /// Child graph node
    Graph,
    /// Empty node (for dependencies only)
    Empty,
    /// Wait for external event
    EventWait,
    /// Record an event
    EventRecord,
    /// External semaphore signal
    ExternalSemaphoreSignal,
    /// External semaphore wait
    ExternalSemaphoreWait,
    /// Memory allocation node
    MemAlloc,
    /// Memory free node
    MemFree,
    /// Batch memory operations node
    BatchMemOp,
}
/// Kernel node parameters
#[derive(Debug, Clone)]
pub struct KernelNodeParams {
    /// Kernel function pointer (handle)
    pub function: usize,
    /// Grid dimensions (x, y, z)
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (x, y, z)
    pub block_dim: (u32, u32, u32),
    /// Shared memory size in bytes
    pub shared_mem_bytes: u32,
    /// Kernel parameters (as raw pointers)
    pub params: Vec<usize>,
    /// Extra options
    pub extra: Option<Vec<u8>>,
}
/// A node in the CUDA graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique node ID
    pub id: usize,
    /// Node handle
    pub handle: CudaGraphNodeHandle,
    /// Node type
    pub node_type: GraphNodeType,
    /// Dependencies (node IDs that must complete before this node)
    pub dependencies: Vec<usize>,
    /// Node name for debugging
    pub name: Option<String>,
    /// Kernel parameters (if kernel node)
    pub kernel_params: Option<KernelNodeParams>,
    /// Memory copy parameters (if memcpy node)
    pub memcopy_params: Option<MemCopyNodeParams>,
    /// Memory set parameters (if memset node)
    pub memset_params: Option<MemSetNodeParams>,
    /// Host callback parameters (if host node)
    pub host_params: Option<HostNodeParams>,
    /// Execution time in microseconds (after profiling)
    pub execution_time_us: Option<f64>,
}
impl GraphNode {
    /// Create a new graph node
    fn new(id: usize, node_type: GraphNodeType) -> Self {
        Self {
            id,
            handle: id,
            node_type,
            dependencies: Vec::new(),
            name: None,
            kernel_params: None,
            memcopy_params: None,
            memset_params: None,
            host_params: None,
            execution_time_us: None,
        }
    }
    /// Add a dependency
    pub fn add_dependency(&mut self, dep_id: usize) {
        if !self.dependencies.contains(&dep_id) {
            self.dependencies.push(dep_id);
        }
    }
    /// Set node name
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
    /// Check if this node has no dependencies
    pub fn is_root(&self) -> bool {
        self.dependencies.is_empty()
    }
    /// Check if this node is a leaf (no dependents)
    pub fn is_leaf(&self, graph: &CudaGraph) -> bool {
        !graph
            .nodes
            .iter()
            .any(|(_, n)| n.dependencies.contains(&self.id))
    }
}
/// Detailed graph update result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphExecUpdateResult {
    /// Update succeeded
    Success,
    /// Graph structure changed - need full reinstantiation
    TopologyChanged,
    /// Node type changed
    NodeTypeChanged,
    /// Parameters changed but can be updated
    ParametersChanged,
    /// Function changed - need reinstantiation
    FunctionChanged,
    /// Not supported - need reinstantiation
    NotSupported,
    /// Unknown error
    Error,
}
/// Memory copy direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemCopyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    HostToHost,
    Default,
}
/// Builder for constructing CUDA graphs with stream capture
pub struct CudaGraphBuilder {
    /// Current graph being built
    graph: Option<CudaGraph>,
    /// Whether we're currently capturing
    is_capturing: bool,
    /// Capture mode
    capture_mode: GraphCaptureMode,
    /// Captured operations count
    captured_ops: usize,
    /// Last added node ID (for automatic dependency chaining)
    last_node_id: Option<usize>,
    /// Enable automatic dependency chaining
    auto_chain: bool,
}
impl CudaGraphBuilder {
    /// Create a new graph builder
    pub fn new() -> Self {
        Self {
            graph: None,
            is_capturing: false,
            capture_mode: GraphCaptureMode::ThreadLocal,
            captured_ops: 0,
            last_node_id: None,
            auto_chain: true,
        }
    }
    /// Enable or disable automatic dependency chaining
    pub fn with_auto_chain(mut self, auto_chain: bool) -> Self {
        self.auto_chain = auto_chain;
        self
    }
    /// Set capture mode
    pub fn with_capture_mode(mut self, mode: GraphCaptureMode) -> Self {
        self.capture_mode = mode;
        self
    }
    /// Begin capturing operations
    #[cfg(feature = "advanced_math")]
    pub fn begin_capture(&mut self, stream: &CudaStream) -> Result<()> {
        if self.is_capturing {
            return Err(SimulatorError::InvalidOperation(
                "Already capturing".to_string(),
            ));
        }
        Self::cuda_stream_begin_capture(stream, self.capture_mode)?;
        self.graph = Some(CudaGraph::new());
        self.is_capturing = true;
        self.captured_ops = 0;
        self.last_node_id = None;
        Ok(())
    }
    #[cfg(not(feature = "advanced_math"))]
    pub fn begin_capture(&mut self) -> Result<()> {
        if self.is_capturing {
            return Err(SimulatorError::InvalidOperation(
                "Already capturing".to_string(),
            ));
        }
        self.graph = Some(CudaGraph::new());
        self.is_capturing = true;
        self.captured_ops = 0;
        self.last_node_id = None;
        Ok(())
    }
    /// End capturing and return the graph
    #[cfg(feature = "advanced_math")]
    pub fn end_capture(&mut self, stream: &CudaStream) -> Result<CudaGraph> {
        if !self.is_capturing {
            return Err(SimulatorError::InvalidOperation(
                "Not currently capturing".to_string(),
            ));
        }
        let _graph_handle = Self::cuda_stream_end_capture(stream)?;
        self.is_capturing = false;
        let mut graph = self
            .graph
            .take()
            .ok_or_else(|| SimulatorError::InvalidState("No graph available".to_string()))?;
        graph.finalize()?;
        Ok(graph)
    }
    #[cfg(not(feature = "advanced_math"))]
    pub fn end_capture(&mut self) -> Result<CudaGraph> {
        if !self.is_capturing {
            return Err(SimulatorError::InvalidOperation(
                "Not currently capturing".to_string(),
            ));
        }
        self.is_capturing = false;
        let mut graph = self
            .graph
            .take()
            .ok_or_else(|| SimulatorError::InvalidState("No graph available".to_string()))?;
        graph.finalize()?;
        Ok(graph)
    }
    /// Check if currently capturing
    pub fn is_capturing(&self) -> bool {
        self.is_capturing
    }
    /// Get the number of captured operations
    pub fn captured_ops_count(&self) -> usize {
        self.captured_ops
    }
    /// Manually add a kernel node during capture
    pub fn capture_kernel(&mut self, params: KernelNodeParams) -> Result<usize> {
        if !self.is_capturing {
            return Err(SimulatorError::InvalidOperation(
                "Not currently capturing".to_string(),
            ));
        }
        let graph = self
            .graph
            .as_mut()
            .ok_or_else(|| SimulatorError::InvalidState("No graph available".to_string()))?;
        let deps = if self.auto_chain {
            self.last_node_id.map(|id| vec![id]).unwrap_or_default()
        } else {
            Vec::new()
        };
        let node_id = graph.add_kernel_node(params, &deps)?;
        self.last_node_id = Some(node_id);
        self.captured_ops += 1;
        Ok(node_id)
    }
    /// Manually add a memory copy node during capture
    pub fn capture_memcpy(&mut self, params: MemCopyNodeParams) -> Result<usize> {
        if !self.is_capturing {
            return Err(SimulatorError::InvalidOperation(
                "Not currently capturing".to_string(),
            ));
        }
        let graph = self
            .graph
            .as_mut()
            .ok_or_else(|| SimulatorError::InvalidState("No graph available".to_string()))?;
        let deps = if self.auto_chain {
            self.last_node_id.map(|id| vec![id]).unwrap_or_default()
        } else {
            Vec::new()
        };
        let node_id = graph.add_memcpy_node(params, &deps)?;
        self.last_node_id = Some(node_id);
        self.captured_ops += 1;
        Ok(node_id)
    }
    /// Add a synchronization point
    pub fn add_sync_point(&mut self, dependencies: &[usize]) -> Result<usize> {
        if !self.is_capturing {
            return Err(SimulatorError::InvalidOperation(
                "Not currently capturing".to_string(),
            ));
        }
        let graph = self
            .graph
            .as_mut()
            .ok_or_else(|| SimulatorError::InvalidState("No graph available".to_string()))?;
        let node_id = graph.add_empty_node(dependencies)?;
        self.last_node_id = Some(node_id);
        Ok(node_id)
    }
    /// Reset the last node ID (break the auto-chain)
    pub fn reset_chain(&mut self) {
        self.last_node_id = None;
    }
    /// Set the last node ID for dependency chaining
    pub fn set_chain_point(&mut self, node_id: usize) {
        self.last_node_id = Some(node_id);
    }
    #[cfg(feature = "advanced_math")]
    fn cuda_stream_begin_capture(_stream: &CudaStream, _mode: GraphCaptureMode) -> Result<()> {
        Ok(())
    }
    #[cfg(feature = "advanced_math")]
    fn cuda_stream_end_capture(_stream: &CudaStream) -> Result<CudaGraphHandle> {
        Ok(0)
    }
}
/// Executable CUDA graph
pub struct CudaGraphExec {
    /// Execution handle
    handle: CudaGraphExecHandle,
    /// Reference to the source graph (for updates)
    source_graph_handle: CudaGraphHandle,
    /// Execution order (node IDs in topological order)
    execution_order: Vec<usize>,
    /// Instantiation flags
    flags: GraphInstantiationFlags,
    /// Execution count
    execution_count: AtomicUsize,
    /// Total execution time in microseconds
    total_execution_time_us: Arc<RwLock<f64>>,
    /// Last execution time in microseconds
    last_execution_time_us: Arc<RwLock<f64>>,
    /// Is the executable up to date
    is_up_to_date: Arc<RwLock<bool>>,
}
impl CudaGraphExec {
    /// Create a new executable graph
    fn new(
        graph: &CudaGraph,
        execution_order: Vec<usize>,
        flags: GraphInstantiationFlags,
    ) -> Result<Self> {
        static EXEC_COUNTER: AtomicUsize = AtomicUsize::new(0);
        let handle = EXEC_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self::cuda_graph_instantiate(graph.handle, &flags)?;
        Ok(Self {
            handle,
            source_graph_handle: graph.handle,
            execution_order,
            flags,
            execution_count: AtomicUsize::new(0),
            total_execution_time_us: Arc::new(RwLock::new(0.0)),
            last_execution_time_us: Arc::new(RwLock::new(0.0)),
            is_up_to_date: Arc::new(RwLock::new(true)),
        })
    }
    /// Get the execution handle
    pub fn handle(&self) -> CudaGraphExecHandle {
        self.handle
    }
    /// Get execution count
    pub fn execution_count(&self) -> usize {
        self.execution_count.load(Ordering::SeqCst)
    }
    /// Get average execution time in microseconds
    pub fn average_execution_time_us(&self) -> f64 {
        let count = self.execution_count.load(Ordering::SeqCst);
        if count == 0 {
            return 0.0;
        }
        let total = *self
            .total_execution_time_us
            .read()
            .unwrap_or_else(|e| e.into_inner());
        total / count as f64
    }
    /// Get last execution time in microseconds
    pub fn last_execution_time_us(&self) -> f64 {
        *self
            .last_execution_time_us
            .read()
            .unwrap_or_else(|e| e.into_inner())
    }
    /// Launch the graph on the default stream
    pub fn launch(&self) -> Result<()> {
        self.launch_on_stream(None)
    }
    /// Launch the graph on a specific stream
    #[cfg(feature = "advanced_math")]
    pub fn launch_on_stream(&self, stream: Option<&CudaStream>) -> Result<()> {
        let start = std::time::Instant::now();
        let stream_handle = stream.and_then(|s: &CudaStream| s.get_handle_value());
        Self::cuda_graph_launch(self.handle, stream_handle)?;
        let elapsed_us = start.elapsed().as_micros() as f64;
        self.execution_count.fetch_add(1, Ordering::SeqCst);
        if let Ok(mut total) = self.total_execution_time_us.write() {
            *total += elapsed_us;
        }
        if let Ok(mut last) = self.last_execution_time_us.write() {
            *last = elapsed_us;
        }
        Ok(())
    }
    #[cfg(not(feature = "advanced_math"))]
    pub fn launch_on_stream(&self, _stream: Option<()>) -> Result<()> {
        let start = std::time::Instant::now();
        std::thread::sleep(std::time::Duration::from_micros(10));
        let elapsed_us = start.elapsed().as_micros() as f64;
        self.execution_count.fetch_add(1, Ordering::SeqCst);
        if let Ok(mut total) = self.total_execution_time_us.write() {
            *total += elapsed_us;
        }
        if let Ok(mut last) = self.last_execution_time_us.write() {
            *last = elapsed_us;
        }
        Ok(())
    }
    /// Update the executable graph from the source graph
    pub fn update(&self, graph: &CudaGraph) -> Result<GraphUpdateResult> {
        if graph.handle != self.source_graph_handle {
            return Err(SimulatorError::InvalidOperation(
                "Cannot update from a different source graph".to_string(),
            ));
        }
        let result = Self::cuda_graph_exec_update(self.handle, graph.handle)?;
        if result.success {
            if let Ok(mut up_to_date) = self.is_up_to_date.write() {
                *up_to_date = true;
            }
        }
        Ok(result)
    }
    /// Check if the executable is up to date
    pub fn is_up_to_date(&self) -> bool {
        *self.is_up_to_date.read().unwrap_or_else(|e| e.into_inner())
    }
    /// Mark the executable as needing update
    pub fn mark_stale(&self) {
        if let Ok(mut up_to_date) = self.is_up_to_date.write() {
            *up_to_date = false;
        }
    }
    fn cuda_graph_instantiate(
        _graph_handle: CudaGraphHandle,
        _flags: &GraphInstantiationFlags,
    ) -> Result<()> {
        Ok(())
    }
    #[cfg(feature = "advanced_math")]
    fn cuda_graph_launch(
        _exec_handle: CudaGraphExecHandle,
        _stream: Option<CudaStreamHandle>,
    ) -> Result<()> {
        std::thread::sleep(std::time::Duration::from_micros(10));
        Ok(())
    }
    fn cuda_graph_exec_update(
        _exec_handle: CudaGraphExecHandle,
        _graph_handle: CudaGraphHandle,
    ) -> Result<GraphUpdateResult> {
        Ok(GraphUpdateResult {
            success: true,
            error_node: None,
            update_result: GraphExecUpdateResult::Success,
        })
    }
}
