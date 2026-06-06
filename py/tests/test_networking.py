# Smoke tests for the quantrs2.networking Python module.
#
# These tests verify that the networking classes are importable and produce
# sensible output. They are skipped if the extension module is not available
# (e.g. when running without maturin-built binaries).

import math
import pytest

try:
    from quantrs2.networking import (
        BB84Protocol,
        E91Protocol,
        TeleportationProtocol,
        EntanglementSwapping,
    )
    _HAS_NETWORKING = True
except ImportError:
    _HAS_NETWORKING = False

skip_if_unavailable = pytest.mark.skipif(
    not _HAS_NETWORKING,
    reason="quantrs2.networking not available (not built with maturin)",
)


# ---------------------------------------------------------------------------
# BB84 QKD
# ---------------------------------------------------------------------------


@skip_if_unavailable
def test_bb84_creates_and_runs():
    proto = BB84Protocol(n_bits=500, error_rate=0.0, eavesdrop_rate=0.0, seed=42)
    result = proto.run()
    assert isinstance(result, dict)
    assert "sifted_key" in result
    assert "qber" in result
    assert "secret_key" in result
    assert "detected_eavesdrop" in result
    assert result["qber"] >= 0.0
    assert result["qber"] <= 1.0


@skip_if_unavailable
def test_bb84_no_noise_low_qber():
    proto = BB84Protocol(n_bits=2000, error_rate=0.0, eavesdrop_rate=0.0, seed=7)
    result = proto.run()
    assert result["qber"] < 0.10, f"Expected low QBER, got {result['qber']}"
    assert not result["detected_eavesdrop"]


@skip_if_unavailable
def test_bb84_full_eavesdrop_detected():
    proto = BB84Protocol(n_bits=3000, error_rate=0.0, eavesdrop_rate=1.0, seed=13)
    result = proto.run()
    assert result["detected_eavesdrop"], "Full eavesdropping should be detected"
    assert result["qber"] > 0.10, f"Expected high QBER, got {result['qber']}"


@skip_if_unavailable
def test_bb84_secret_key_half_sifted():
    proto = BB84Protocol(n_bits=2000, seed=55)
    result = proto.run()
    sifted = result["sifted_key"]
    secret = result["secret_key"]
    if sifted:
        assert len(secret) == len(sifted) // 2, (
            f"Secret key ({len(secret)}) should be half sifted ({len(sifted)})"
        )


# ---------------------------------------------------------------------------
# E91 QKD
# ---------------------------------------------------------------------------


@skip_if_unavailable
def test_e91_creates_and_runs():
    proto = E91Protocol(n_pairs=200, noise=0.0, seed=42)
    result = proto.run()
    assert isinstance(result, dict)
    assert "key" in result
    assert "chsh_value" in result
    assert "passed_bell_test" in result
    assert "key_rate" in result


@skip_if_unavailable
def test_e91_ideal_chsh():
    proto = E91Protocol(n_pairs=300, noise=0.0, seed=42)
    result = proto.run()
    assert result["chsh_value"] > 2.0, (
        f"Ideal CHSH should be > 2 (quantum), got {result['chsh_value']}"
    )
    assert result["passed_bell_test"]


@skip_if_unavailable
def test_e91_high_noise_fails_bell_test():
    proto = E91Protocol(n_pairs=200, noise=0.9, seed=42)
    result = proto.run()
    assert not result["passed_bell_test"], (
        f"High-noise E91 should fail Bell test, got S={result['chsh_value']}"
    )


# ---------------------------------------------------------------------------
# Quantum Teleportation
# ---------------------------------------------------------------------------


@skip_if_unavailable
def test_teleportation_creates_and_runs():
    proto = TeleportationProtocol(noise=0.0, seed=42)
    result = proto.teleport(alpha_re=1.0)  # |0⟩ state
    assert isinstance(result, dict)
    assert "fidelity" in result
    assert "correction_bits" in result
    assert 0.0 <= result["fidelity"] <= 1.0


@skip_if_unavailable
def test_teleportation_no_noise_fidelity_one():
    proto = TeleportationProtocol(noise=0.0, seed=42)
    result = proto.teleport(alpha_re=1.0, beta_re=0.0)
    assert result["fidelity"] > 0.95, (
        f"No-noise fidelity should be ≈ 1, got {result['fidelity']}"
    )


@skip_if_unavailable
def test_teleportation_plus_state():
    inv_sqrt2 = 1.0 / math.sqrt(2)
    proto = TeleportationProtocol(noise=0.0, seed=42)
    result = proto.teleport(alpha_re=inv_sqrt2, beta_re=inv_sqrt2)
    assert result["fidelity"] > 0.95, (
        f"|+⟩ teleportation fidelity should be ≈ 1, got {result['fidelity']}"
    )


@skip_if_unavailable
def test_teleportation_noisy_lower_fidelity():
    proto_clean = TeleportationProtocol(noise=0.0, seed=42)
    proto_noisy = TeleportationProtocol(noise=0.3, seed=42)
    f_clean = proto_clean.teleport(alpha_re=1.0).get("fidelity")
    f_noisy = proto_noisy.teleport(alpha_re=1.0).get("fidelity")
    assert f_clean > f_noisy, (
        f"Clean fidelity ({f_clean:.3f}) should exceed noisy ({f_noisy:.3f})"
    )


# ---------------------------------------------------------------------------
# Entanglement Swapping
# ---------------------------------------------------------------------------


@skip_if_unavailable
def test_entanglement_swapping_creates_and_runs():
    swap = EntanglementSwapping(n_hops=1, noise_per_link=0.0, seed=42)
    result = swap.run()
    assert isinstance(result, dict)
    assert "end_to_end_fidelity" in result
    assert 0.0 <= result["end_to_end_fidelity"] <= 1.0


@skip_if_unavailable
def test_entanglement_swapping_no_noise_high_fidelity():
    swap = EntanglementSwapping(n_hops=1, noise_per_link=0.0, seed=42)
    result = swap.run()
    assert result["end_to_end_fidelity"] > 0.90, (
        f"1-hop no-noise fidelity should be > 0.90, got {result['end_to_end_fidelity']}"
    )


@skip_if_unavailable
def test_entanglement_swapping_fidelity_degrades():
    f1 = EntanglementSwapping(n_hops=1, noise_per_link=0.05, seed=42).run()["end_to_end_fidelity"]
    f3 = EntanglementSwapping(n_hops=3, noise_per_link=0.05, seed=42).run()["end_to_end_fidelity"]
    assert f1 > f3, (
        f"1-hop fidelity ({f1:.3f}) should exceed 3-hop ({f3:.3f}) with noise"
    )


@skip_if_unavailable
def test_entanglement_swapping_repr():
    swap = EntanglementSwapping(n_hops=2, noise_per_link=0.1, seed=42)
    r = repr(swap)
    assert "EntanglementSwapping" in r
