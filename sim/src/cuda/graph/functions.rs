//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    CudaGraph, CudaGraphBuilder, GraphNodeType, KernelNodeParams, MemCopyKind, MemCopyNodeParams,
    QuantumGraphScheduler,
};

pub type CudaGraphHandle = usize;
pub type CudaGraphNodeHandle = usize;
pub type CudaGraphExecHandle = usize;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_graph_creation() {
        let graph = CudaGraph::new();
        assert!(graph.is_empty());
        assert!(!graph.is_finalized());
    }
    #[test]
    fn test_add_kernel_node() {
        let mut graph = CudaGraph::new();
        let params = KernelNodeParams {
            function: 1,
            grid_dim: (16, 1, 1),
            block_dim: (256, 1, 1),
            ..Default::default()
        };
        let node_id = graph
            .add_kernel_node(params, &[])
            .expect("should add kernel node");
        assert_eq!(node_id, 0);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.kernel_count(), 1);
    }
    #[test]
    fn test_add_nodes_with_dependencies() {
        let mut graph = CudaGraph::new();
        let params1 = KernelNodeParams::default();
        let node1 = graph.add_kernel_node(params1, &[]).expect("add node 1");
        let params2 = KernelNodeParams::default();
        let node2 = graph
            .add_kernel_node(params2, &[node1])
            .expect("add node 2");
        let params3 = KernelNodeParams::default();
        let _node3 = graph
            .add_kernel_node(params3, &[node1, node2])
            .expect("add node 3");
        assert_eq!(graph.node_count(), 3);
        let node = graph.get_node(node2).expect("node should exist");
        assert!(node.dependencies.contains(&node1));
    }
    #[test]
    fn test_graph_finalization() {
        let mut graph = CudaGraph::new();
        let params = KernelNodeParams::default();
        graph.add_kernel_node(params, &[]).expect("add kernel");
        assert!(!graph.is_finalized());
        graph.finalize().expect("finalization should succeed");
        assert!(graph.is_finalized());
        let params2 = KernelNodeParams::default();
        let result = graph.add_kernel_node(params2, &[]);
        assert!(result.is_err());
    }
    #[test]
    fn test_topological_order() {
        let mut graph = CudaGraph::new();
        let node0 = graph
            .add_kernel_node(KernelNodeParams::default(), &[])
            .expect("add");
        let node1 = graph
            .add_kernel_node(KernelNodeParams::default(), &[node0])
            .expect("add");
        let node2 = graph
            .add_kernel_node(KernelNodeParams::default(), &[node0])
            .expect("add");
        let _node3 = graph
            .add_kernel_node(KernelNodeParams::default(), &[node1, node2])
            .expect("add");
        let order = graph.topological_order().expect("should succeed");
        assert_eq!(order.len(), 4);
        let pos0 = order.iter().position(|&x| x == node0).expect("find 0");
        let pos1 = order.iter().position(|&x| x == node1).expect("find 1");
        let pos2 = order.iter().position(|&x| x == node2).expect("find 2");
        assert!(pos0 < pos1);
        assert!(pos0 < pos2);
    }
    #[test]
    fn test_cycle_detection() {
        let mut graph = CudaGraph::new();
        let node0 = graph
            .add_kernel_node(KernelNodeParams::default(), &[])
            .expect("add");
        let node1 = graph
            .add_kernel_node(KernelNodeParams::default(), &[node0])
            .expect("add");
        if let Some(node) = graph.get_node_mut(node0) {
            node.dependencies.push(node1);
        }
        let result = graph.validate();
        assert!(result.is_err());
    }
    #[test]
    fn test_graph_instantiation() {
        let mut graph = CudaGraph::new();
        graph
            .add_kernel_node(KernelNodeParams::default(), &[])
            .expect("add");
        graph.finalize().expect("finalize");
        let exec = graph.instantiate().expect("instantiate");
        assert_eq!(exec.execution_count(), 0);
    }
    #[test]
    fn test_graph_execution() {
        let mut graph = CudaGraph::new();
        graph
            .add_kernel_node(KernelNodeParams::default(), &[])
            .expect("add");
        graph.finalize().expect("finalize");
        let exec = graph.instantiate().expect("instantiate");
        #[cfg(not(feature = "advanced_math"))]
        {
            exec.launch_on_stream(None).expect("launch");
            exec.launch_on_stream(None).expect("launch again");
            assert_eq!(exec.execution_count(), 2);
            assert!(exec.average_execution_time_us() > 0.0);
        }
        #[cfg(feature = "advanced_math")]
        {
            assert_eq!(exec.execution_count(), 0);
        }
    }
    #[test]
    fn test_graph_stats() {
        let mut graph = CudaGraph::new();
        let node0 = graph
            .add_kernel_node(KernelNodeParams::default(), &[])
            .expect("add");
        let node1 = graph
            .add_kernel_node(KernelNodeParams::default(), &[node0])
            .expect("add");
        let _node2 = graph
            .add_memcpy_node(
                MemCopyNodeParams {
                    src: 0,
                    dst: 1,
                    size: 1024,
                    kind: MemCopyKind::DeviceToDevice,
                },
                &[node1],
            )
            .expect("add");
        let stats = graph.get_stats();
        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.kernel_count, 2);
        assert_eq!(stats.mem_op_count, 1);
        assert_eq!(stats.edge_count, 2);
        assert_eq!(stats.root_count, 1);
        assert_eq!(stats.leaf_count, 1);
    }
    #[test]
    fn test_graph_builder() {
        let mut builder = CudaGraphBuilder::new();
        #[cfg(not(feature = "advanced_math"))]
        {
            builder.begin_capture().expect("begin capture");
            assert!(builder.is_capturing());
            builder
                .capture_kernel(KernelNodeParams::default())
                .expect("capture kernel");
            builder
                .capture_kernel(KernelNodeParams::default())
                .expect("capture kernel");
            let graph = builder.end_capture().expect("end capture");
            assert!(!builder.is_capturing());
            assert_eq!(graph.node_count(), 2);
            assert!(graph.is_finalized());
        }
    }
    #[test]
    fn test_scheduler_caching() {
        let mut scheduler = QuantumGraphScheduler::new(10);
        let _exec1 = scheduler
            .get_or_create("pattern1", || {
                let mut graph = CudaGraph::new();
                graph.add_kernel_node(KernelNodeParams::default(), &[])?;
                graph.finalize()?;
                Ok(graph)
            })
            .expect("create");
        let _exec2 = scheduler
            .get_or_create("pattern1", || {
                panic!("Should not be called - cache hit expected");
            })
            .expect("cached");
        let (hits, misses) = scheduler.cache_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert!((scheduler.cache_hit_rate() - 0.5).abs() < 0.01);
    }
    #[test]
    fn test_memcpy_node() {
        let mut graph = CudaGraph::new();
        let params = MemCopyNodeParams {
            src: 0x1000,
            dst: 0x2000,
            size: 4096,
            kind: MemCopyKind::HostToDevice,
        };
        let node_id = graph.add_memcpy_node(params, &[]).expect("add memcpy");
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.mem_op_count(), 1);
        let node = graph.get_node(node_id).expect("get node");
        assert_eq!(node.node_type, GraphNodeType::MemCopy);
    }
    #[test]
    fn test_empty_node() {
        let mut graph = CudaGraph::new();
        let k1 = graph
            .add_kernel_node(KernelNodeParams::default(), &[])
            .expect("add");
        let k2 = graph
            .add_kernel_node(KernelNodeParams::default(), &[])
            .expect("add");
        let sync = graph.add_empty_node(&[k1, k2]).expect("add sync");
        let node = graph.get_node(sync).expect("get node");
        assert_eq!(node.node_type, GraphNodeType::Empty);
        assert_eq!(node.dependencies.len(), 2);
    }
    #[test]
    fn test_root_and_leaf_nodes() {
        let mut graph = CudaGraph::new();
        let n0 = graph
            .add_kernel_node(KernelNodeParams::default(), &[])
            .expect("add");
        let n1 = graph
            .add_kernel_node(KernelNodeParams::default(), &[])
            .expect("add");
        let n2 = graph
            .add_kernel_node(KernelNodeParams::default(), &[n0, n1])
            .expect("add");
        let _n3 = graph
            .add_kernel_node(KernelNodeParams::default(), &[n2])
            .expect("add");
        let roots = graph.get_root_nodes();
        assert_eq!(roots.len(), 2);
        assert!(roots.contains(&n0));
        assert!(roots.contains(&n1));
        let leaves = graph.get_leaf_nodes();
        assert_eq!(leaves.len(), 1);
    }
    #[test]
    fn test_graph_clone() {
        let mut graph = CudaGraph::new();
        graph
            .add_kernel_node(KernelNodeParams::default(), &[])
            .expect("add");
        graph
            .add_kernel_node(KernelNodeParams::default(), &[0])
            .expect("add");
        let cloned = graph.clone_graph().expect("clone");
        assert_eq!(cloned.node_count(), graph.node_count());
        assert_eq!(cloned.kernel_count(), graph.kernel_count());
    }
    #[test]
    fn test_update_kernel_params() {
        let mut graph = CudaGraph::new();
        let node_id = graph
            .add_kernel_node(KernelNodeParams::default(), &[])
            .expect("add");
        let new_params = KernelNodeParams {
            function: 42,
            grid_dim: (32, 1, 1),
            block_dim: (512, 1, 1),
            ..Default::default()
        };
        graph
            .update_kernel_params(node_id, new_params)
            .expect("update");
        let node = graph.get_node(node_id).expect("get node");
        let params = node.kernel_params.as_ref().expect("has params");
        assert_eq!(params.function, 42);
        assert_eq!(params.grid_dim, (32, 1, 1));
    }
    #[test]
    fn test_graph_with_name() {
        let graph = CudaGraph::new().with_name("test_circuit");
        assert_eq!(graph.name, Some("test_circuit".to_string()));
    }
    #[test]
    fn test_invalid_dependency() {
        let mut graph = CudaGraph::new();
        let result = graph.add_kernel_node(KernelNodeParams::default(), &[999]);
        assert!(result.is_err());
    }
    #[test]
    fn test_scheduler_eviction() {
        let mut scheduler = QuantumGraphScheduler::new(2);
        for i in 0..3 {
            let key = format!("pattern{i}");
            let _ = scheduler.get_or_create(&key, || {
                let mut graph = CudaGraph::new();
                graph.add_kernel_node(KernelNodeParams::default(), &[])?;
                graph.finalize()?;
                Ok(graph)
            });
        }
        assert!(scheduler.cached_graphs.len() <= 2);
    }
}
