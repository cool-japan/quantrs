//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{single::*, GateOp},
    qubit::QubitId,
};
use std::f64::consts::PI;

use super::types::{CircuitToZX, EdgeType, Spider, SpiderType, ZXDiagram, ZXOptimizer};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::multi::CNOT;
    #[test]
    fn test_spider_creation() {
        let spider = Spider::new(0, SpiderType::Z, PI / 2.0);
        assert_eq!(spider.id, 0);
        assert_eq!(spider.spider_type, SpiderType::Z);
        assert!((spider.phase - PI / 2.0).abs() < 1e-10);
        assert!(spider.is_clifford(1e-10));
        assert!(!spider.is_pauli(1e-10));
    }
    #[test]
    fn test_diagram_creation() {
        let mut diagram = ZXDiagram::new();
        let z_id = diagram.add_spider(SpiderType::Z, 0.0);
        let x_id = diagram.add_spider(SpiderType::X, PI);
        diagram.add_edge(z_id, x_id, EdgeType::Regular);
        assert_eq!(diagram.degree(z_id), 1);
        assert_eq!(diagram.degree(x_id), 1);
    }
    #[test]
    fn test_spider_fusion() {
        let mut diagram = ZXDiagram::new();
        let z1 = diagram.add_spider(SpiderType::Z, PI / 4.0);
        let z2 = diagram.add_spider(SpiderType::Z, PI / 4.0);
        let boundary = diagram.add_boundary(QubitId(0), true);
        diagram.add_edge(z1, z2, EdgeType::Regular);
        diagram.add_edge(z2, boundary, EdgeType::Regular);
        assert!(diagram.spider_fusion(z1, z2).is_ok());
        assert!(!diagram.spiders.contains_key(&z2));
        assert_eq!(diagram.spiders[&z1].phase, PI / 2.0);
        assert_eq!(diagram.degree(z1), 1);
    }
    #[test]
    fn test_identity_removal() {
        let mut diagram = ZXDiagram::new();
        let b1 = diagram.add_boundary(QubitId(0), true);
        let id_spider = diagram.add_spider(SpiderType::Z, 0.0);
        let b2 = diagram.add_boundary(QubitId(0), false);
        diagram.add_edge(b1, id_spider, EdgeType::Regular);
        diagram.add_edge(id_spider, b2, EdgeType::Regular);
        let removed = diagram.remove_identities();
        assert_eq!(removed, 1);
        assert!(!diagram.spiders.contains_key(&id_spider));
        assert!(diagram.neighbors(b1).iter().any(|(id, _)| *id == b2));
    }
    #[test]
    fn test_circuit_to_zx_hadamard() {
        let mut converter = CircuitToZX::new(1);
        let h_gate = Hadamard { target: QubitId(0) };
        assert!(converter.add_gate(&h_gate).is_ok());
        let diagram = converter.into_diagram();
        let has_hadamard = diagram.adjacency.values().any(|neighbors| {
            neighbors
                .iter()
                .any(|(_, edge_type)| *edge_type == EdgeType::Hadamard)
        });
        assert!(has_hadamard);
    }
    #[test]
    fn test_circuit_to_zx_cnot() {
        let mut converter = CircuitToZX::new(2);
        let cnot = CNOT {
            control: QubitId(0),
            target: QubitId(1),
        };
        assert!(converter.add_gate(&cnot).is_ok());
        let diagram = converter.into_diagram();
        assert_eq!(diagram.spiders.len(), 6);
        let z_count = diagram
            .spiders
            .values()
            .filter(|s| s.spider_type == SpiderType::Z && s.phase.abs() < 1e-10)
            .count();
        let x_count = diagram
            .spiders
            .values()
            .filter(|s| s.spider_type == SpiderType::X && s.phase.abs() < 1e-10)
            .count();
        assert_eq!(z_count, 1);
        assert_eq!(x_count, 1);
    }
    #[test]
    fn test_zx_optimizer() {
        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: QubitId(0) }),
            Box::new(PauliZ { target: QubitId(0) }),
            Box::new(Hadamard { target: QubitId(0) }),
        ];
        let optimizer = ZXOptimizer::new();
        let result = optimizer.optimize_circuit(&gates);
        assert!(result.is_ok());
    }
    #[test]
    fn test_dot_generation() {
        let mut diagram = ZXDiagram::new();
        let input = diagram.add_boundary(QubitId(0), true);
        let z = diagram.add_spider(SpiderType::Z, PI / 2.0);
        let x = diagram.add_spider(SpiderType::X, 0.0);
        let output = diagram.add_boundary(QubitId(0), false);
        diagram.add_edge(input, z, EdgeType::Regular);
        diagram.add_edge(z, x, EdgeType::Hadamard);
        diagram.add_edge(x, output, EdgeType::Regular);
        let dot = diagram.to_dot();
        assert!(dot.contains("graph ZX"));
        assert!(dot.contains("color=green"));
        assert!(dot.contains("color=red"));
        assert!(dot.contains("style=dashed"));
    }
    #[test]
    fn test_spider_fusion_reduces_node_count() {
        let mut diagram = ZXDiagram::new();
        let b_in = diagram.add_boundary(QubitId(0), true);
        let z1 = diagram.add_spider(SpiderType::Z, PI / 4.0);
        let z2 = diagram.add_spider(SpiderType::Z, PI / 4.0);
        let b_out = diagram.add_boundary(QubitId(0), false);
        diagram.add_edge(b_in, z1, EdgeType::Regular);
        diagram.add_edge(z1, z2, EdgeType::Regular);
        diagram.add_edge(z2, b_out, EdgeType::Regular);
        let fired = diagram
            .apply_spider_fusion_in_component(&[z1, z2])
            .expect("no error");
        assert!(fired);
        let z_count = diagram
            .spiders
            .values()
            .filter(|s| s.spider_type == SpiderType::Z)
            .count();
        assert_eq!(z_count, 1, "two spiders should merge into one");
        let fused = diagram
            .spiders
            .values()
            .find(|s| s.spider_type == SpiderType::Z)
            .expect("spider");
        assert!((fused.phase - PI / 2.0).abs() < 1e-9, "fused phase = π/2");
    }
    #[test]
    fn test_spider_fusion_different_colour_no_change() {
        let mut diagram = ZXDiagram::new();
        let z = diagram.add_spider(SpiderType::Z, PI / 4.0);
        let x = diagram.add_spider(SpiderType::X, PI / 4.0);
        diagram.add_edge(z, x, EdgeType::Regular);
        let changed = diagram
            .apply_spider_fusion_in_component(&[z, x])
            .expect("no error");
        assert!(!changed);
        assert_eq!(diagram.spiders.len(), 2);
    }
    #[test]
    fn test_spider_fusion_hadamard_edge_no_change() {
        let mut diagram = ZXDiagram::new();
        let z1 = diagram.add_spider(SpiderType::Z, PI / 4.0);
        let z2 = diagram.add_spider(SpiderType::Z, PI / 4.0);
        diagram.add_edge(z1, z2, EdgeType::Hadamard);
        let changed = diagram
            .apply_spider_fusion_in_component(&[z1, z2])
            .expect("no error");
        assert!(!changed);
    }
    #[test]
    fn test_hadamard_cancellation() {
        let mut diagram = ZXDiagram::new();
        let b_in = diagram.add_boundary(QubitId(0), true);
        let z = diagram.add_spider(SpiderType::Z, 0.0);
        let b_out = diagram.add_boundary(QubitId(0), false);
        diagram.add_edge(b_in, z, EdgeType::Hadamard);
        diagram.add_edge(z, b_out, EdgeType::Hadamard);
        let changed = diagram
            .apply_hadamard_cancellation_in_component(&[z])
            .expect("no error");
        assert!(changed);
        assert!(!diagram.spiders.contains_key(&z));
        let joined = diagram
            .neighbors(b_in)
            .into_iter()
            .any(|(nb, et)| nb == b_out && et == EdgeType::Regular);
        assert!(joined, "b_in and b_out should be joined by a Regular edge");
    }
    #[test]
    fn test_identity_chain_elimination() {
        let mut diagram = ZXDiagram::new();
        let b_in = diagram.add_boundary(QubitId(0), true);
        let z0 = diagram.add_spider(SpiderType::Z, 0.0);
        let z1 = diagram.add_spider(SpiderType::Z, 0.0);
        let z2 = diagram.add_spider(SpiderType::Z, 0.0);
        let b_out = diagram.add_boundary(QubitId(0), false);
        diagram.add_edge(b_in, z0, EdgeType::Regular);
        diagram.add_edge(z0, z1, EdgeType::Regular);
        diagram.add_edge(z1, z2, EdgeType::Regular);
        diagram.add_edge(z2, b_out, EdgeType::Regular);
        let component = [z0, z1, z2];
        let mut rounds = 0usize;
        loop {
            match diagram.apply_identity_removal_in_component(&component) {
                Ok(true) => rounds += 1,
                Ok(false) => break,
                Err(e) => panic!("error: {e:?}"),
            }
        }
        assert!(rounds >= 1);
        let remaining_z = diagram
            .spiders
            .values()
            .filter(|s| s.spider_type == SpiderType::Z)
            .count();
        assert_eq!(remaining_z, 0, "all Z(0) identities eliminated");
        assert!(diagram.neighbors(b_in).iter().any(|(nb, _)| *nb == b_out));
    }
    #[test]
    fn test_decompose_clifford_component_fires() {
        let mut diagram = ZXDiagram::new();
        let b_in = diagram.add_boundary(QubitId(0), true);
        let z1 = diagram.add_spider(SpiderType::Z, PI / 4.0);
        let z2 = diagram.add_spider(SpiderType::Z, PI / 4.0);
        let b_out = diagram.add_boundary(QubitId(0), false);
        diagram.add_edge(b_in, z1, EdgeType::Regular);
        diagram.add_edge(z1, z2, EdgeType::Regular);
        diagram.add_edge(z2, b_out, EdgeType::Regular);
        let changed = diagram
            .decompose_clifford_component(&[z1, z2])
            .expect("no error");
        assert!(changed);
    }
    #[test]
    fn test_apply_tableau_reduction_returns_nonzero() {
        let mut diagram = ZXDiagram::new();
        let b_in = diagram.add_boundary(QubitId(0), true);
        let z1 = diagram.add_spider(SpiderType::Z, PI / 2.0);
        let z2 = diagram.add_spider(SpiderType::Z, PI / 2.0);
        let b_out = diagram.add_boundary(QubitId(0), false);
        diagram.add_edge(b_in, z1, EdgeType::Regular);
        diagram.add_edge(z1, z2, EdgeType::Regular);
        diagram.add_edge(z2, b_out, EdgeType::Regular);
        let reductions = ZXOptimizer::new().apply_tableau_reduction(&mut diagram);
        assert!(reductions > 0, "expected ≥1 reduction, got {reductions}");
    }
    #[test]
    fn test_full_optimizer_clifford_chain() {
        let mut diagram = ZXDiagram::new();
        let b_in = diagram.add_boundary(QubitId(0), true);
        let z1 = diagram.add_spider(SpiderType::Z, PI / 2.0);
        let z2 = diagram.add_spider(SpiderType::Z, PI / 2.0);
        let z3 = diagram.add_spider(SpiderType::Z, 0.0);
        let b_out = diagram.add_boundary(QubitId(0), false);
        diagram.add_edge(b_in, z1, EdgeType::Regular);
        diagram.add_edge(z1, z2, EdgeType::Regular);
        diagram.add_edge(z2, z3, EdgeType::Regular);
        diagram.add_edge(z3, b_out, EdgeType::Regular);
        let initial = diagram.spiders.len();
        let rewrites = ZXOptimizer::new().optimize(&mut diagram);
        let final_sz = diagram.spiders.len();
        assert!(
            final_sz < initial || rewrites > 0,
            "optimizer should reduce; initial={initial}, final={final_sz}, rewrites={rewrites}"
        );
    }
}
