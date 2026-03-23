use super::*;

#[test]
fn test_h_all() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<5>::new();
    circuit.h_all(&[0, 1, 2])?;

    assert_eq!(circuit.gates().len(), 3);
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "H");
    }
    Ok(())
}

#[test]
fn test_x_all() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<5>::new();
    circuit.x_all(&[0, 2, 4])?;

    assert_eq!(circuit.gates().len(), 3);
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "X");
    }
    Ok(())
}

#[test]
fn test_y_all() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<3>::new();
    circuit.y_all(&[0, 1, 2])?;

    assert_eq!(circuit.gates().len(), 3);
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "Y");
    }
    Ok(())
}

#[test]
fn test_z_all() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<4>::new();
    circuit.z_all(&[1, 3])?;

    assert_eq!(circuit.gates().len(), 2);
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "Z");
    }
    Ok(())
}

#[test]
fn test_h_range() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<5>::new();
    circuit.h_range(0..3)?;

    assert_eq!(circuit.gates().len(), 3);
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "H");
    }
    Ok(())
}

#[test]
fn test_x_range() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<5>::new();
    circuit.x_range(1..4)?;

    assert_eq!(circuit.gates().len(), 3);
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "X");
    }
    Ok(())
}

#[test]
fn test_bell_state() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<2>::new();
    circuit.bell_state(0, 1)?;

    assert_eq!(circuit.gates().len(), 2);
    assert_eq!(circuit.gates()[0].name(), "H");
    assert_eq!(circuit.gates()[1].name(), "CNOT");
    Ok(())
}

#[test]
fn test_ghz_state() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<4>::new();
    circuit.ghz_state(&[0, 1, 2, 3])?;

    // Should have 1 H + 3 CNOTs
    assert_eq!(circuit.gates().len(), 4);
    assert_eq!(circuit.gates()[0].name(), "H");
    for i in 1..4 {
        assert_eq!(circuit.gates()[i].name(), "CNOT");
    }
    Ok(())
}

#[test]
fn test_ghz_state_empty() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<4>::new();
    circuit.ghz_state(&[])?;

    assert_eq!(circuit.gates().len(), 0);
    Ok(())
}

#[test]
fn test_w_state() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<3>::new();
    circuit.w_state(&[0, 1, 2])?;

    // W state requires RY + CRY + CNOT gates
    assert!(!circuit.gates().is_empty());
    // At least one rotation gate
    assert!(circuit
        .gates()
        .iter()
        .any(|g| g.name() == "RY" || g.name() == "CRY"));
    Ok(())
}

#[test]
fn test_w_state_empty() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<3>::new();
    circuit.w_state(&[])?;

    assert_eq!(circuit.gates().len(), 0);
    Ok(())
}

#[test]
fn test_plus_state_all() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<4>::new();
    circuit.plus_state_all()?;

    assert_eq!(circuit.gates().len(), 4);
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "H");
    }
    Ok(())
}

#[test]
fn test_rx_all() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<4>::new();
    let theta = std::f64::consts::PI / 4.0;
    circuit.rx_all(&[0, 1, 2], theta)?;

    assert_eq!(circuit.gates().len(), 3);
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "RX");
    }
    Ok(())
}

#[test]
fn test_ry_all() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<4>::new();
    let theta = std::f64::consts::PI / 3.0;
    circuit.ry_all(&[0, 2], theta)?;

    assert_eq!(circuit.gates().len(), 2);
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "RY");
    }
    Ok(())
}

#[test]
fn test_rz_all() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<5>::new();
    let theta = std::f64::consts::PI / 2.0;
    circuit.rz_all(&[1, 2, 3], theta)?;

    assert_eq!(circuit.gates().len(), 3);
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "RZ");
    }
    Ok(())
}

#[test]
fn test_cnot_ladder() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<5>::new();
    circuit.cnot_ladder(&[0, 1, 2, 3])?;

    assert_eq!(circuit.gates().len(), 3);
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "CNOT");
    }
    Ok(())
}

#[test]
fn test_cnot_ladder_too_small() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<5>::new();
    circuit.cnot_ladder(&[0])?;

    assert_eq!(circuit.gates().len(), 0);
    Ok(())
}

#[test]
fn test_cnot_ring() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<4>::new();
    circuit.cnot_ring(&[0, 1, 2, 3])?;

    // Should have 4 CNOTs (3 for ladder + 1 to close ring)
    assert_eq!(circuit.gates().len(), 4);
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "CNOT");
    }
    Ok(())
}

#[test]
fn test_cnot_ring_too_small() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<4>::new();
    circuit.cnot_ring(&[0])?;

    assert_eq!(circuit.gates().len(), 0);
    Ok(())
}

#[test]
fn test_combined_patterns() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<5>::new();

    // Initialize all qubits to |+⟩
    circuit.plus_state_all()?;

    // Create entanglement with CNOT ladder
    circuit.cnot_ladder(&[0, 1, 2, 3, 4])?;

    // Apply phase to some qubits
    circuit.z_all(&[0, 2, 4])?;

    let stats = circuit.get_stats();
    assert_eq!(stats.total_gates, 5 + 4 + 3); // 5 H + 4 CNOT + 3 Z
    assert_eq!(stats.total_qubits, 5);
    Ok(())
}

#[test]
fn test_swap_ladder() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<5>::new();
    circuit.swap_ladder(&[0, 1, 2, 3])?;

    assert_eq!(circuit.gates().len(), 3); // SWAP(0,1), SWAP(1,2), SWAP(2,3)
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "SWAP");
    }
    Ok(())
}

#[test]
fn test_swap_ladder_empty() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<5>::new();
    circuit.swap_ladder(&[])?;

    assert_eq!(circuit.gates().len(), 0);
    Ok(())
}

#[test]
fn test_swap_ladder_single() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<5>::new();
    circuit.swap_ladder(&[0])?;

    assert_eq!(circuit.gates().len(), 0);
    Ok(())
}

#[test]
fn test_cz_ladder() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<4>::new();
    circuit.cz_ladder(&[0, 1, 2, 3])?;

    assert_eq!(circuit.gates().len(), 3); // CZ(0,1), CZ(1,2), CZ(2,3)
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "CZ");
    }
    Ok(())
}

#[test]
fn test_cz_ladder_empty() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<4>::new();
    circuit.cz_ladder(&[])?;

    assert_eq!(circuit.gates().len(), 0);
    Ok(())
}

#[test]
fn test_swap_all() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<6>::new();
    circuit.swap_all(&[(0, 1), (2, 3), (4, 5)])?;

    assert_eq!(circuit.gates().len(), 3);
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "SWAP");
    }
    Ok(())
}

#[test]
fn test_swap_all_empty() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<6>::new();
    circuit.swap_all(&[])?;

    assert_eq!(circuit.gates().len(), 0);
    Ok(())
}

#[test]
fn test_cz_all() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<6>::new();
    circuit.cz_all(&[(0, 1), (2, 3), (4, 5)])?;

    assert_eq!(circuit.gates().len(), 3);
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "CZ");
    }
    Ok(())
}

#[test]
fn test_cz_all_empty() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<6>::new();
    circuit.cz_all(&[])?;

    assert_eq!(circuit.gates().len(), 0);
    Ok(())
}

#[test]
fn test_cnot_all() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<6>::new();
    circuit.cnot_all(&[(0, 1), (2, 3), (4, 5)])?;

    assert_eq!(circuit.gates().len(), 3);
    for gate in circuit.gates() {
        assert_eq!(gate.name(), "CNOT");
    }
    Ok(())
}

#[test]
fn test_cnot_all_empty() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<6>::new();
    circuit.cnot_all(&[])?;

    assert_eq!(circuit.gates().len(), 0);
    Ok(())
}

#[test]
fn test_barrier_all() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<5>::new();
    circuit.h_all(&[0, 1, 2])?;
    circuit.barrier_all(&[0, 1, 2])?;
    circuit.cnot_ladder(&[0, 1, 2])?;

    // Barriers don't currently add gates (they're implicit in the optimization framework)
    // Should have 3 H + 2 CNOT
    assert_eq!(circuit.gates().len(), 5);
    Ok(())
}

#[test]
fn test_barrier_all_empty() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<5>::new();
    circuit.barrier_all(&[])?;

    assert_eq!(circuit.gates().len(), 0);
    Ok(())
}

#[test]
fn test_advanced_entanglement_patterns() -> QuantRS2Result<()> {
    let mut circuit = Circuit::<6>::new();

    // Create superposition
    circuit.h_all(&[0, 1, 2, 3, 4, 5])?;

    // Add barrier to prevent optimization (implicit, doesn't add gates)
    circuit.barrier_all(&[0, 1, 2, 3, 4, 5])?;

    // Create entanglement with CZ ladder
    circuit.cz_ladder(&[0, 1, 2, 3, 4, 5])?;

    // Add more entanglement with CNOT pairs
    circuit.cnot_all(&[(0, 3), (1, 4), (2, 5)])?;

    let stats = circuit.get_stats();
    // 6 H + 5 CZ + 3 CNOT = 14 gates (barriers are implicit)
    assert_eq!(stats.total_gates, 14);
    assert_eq!(stats.total_qubits, 6);
    Ok(())
}
