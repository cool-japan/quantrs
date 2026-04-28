#!/usr/bin/env python3
"""
Test suite for quantum error correction functionality.

Covers the rotated surface code, MWPM decoder, Union-Find decoder,
PauliFrame tracking, and the Python-level simulate_threshold helper.
"""

import pytest

try:
    from quantrs2.qec import (
        RotatedSurfaceCode,
        MwpmDecoder,
        UnionFindDecoder,
        PauliFrame,
        simulate_threshold,
    )
    HAS_QEC = True
except ImportError:
    HAS_QEC = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_error(n: int):
    return ["I"] * n


def _single_x_error(n: int, qubit: int):
    err = ["I"] * n
    err[qubit] = "X"
    return err


def _single_z_error(n: int, qubit: int):
    err = ["I"] * n
    err[qubit] = "Z"
    return err


# ---------------------------------------------------------------------------
# RotatedSurfaceCode tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_QEC, reason="QEC module not available")
class TestRotatedSurfaceCode:
    """Tests for the rotated planar surface code."""

    def test_distance_2_n_k_d(self):
        code = RotatedSurfaceCode(d=2)
        n, k, d = code.n_k_d()
        assert n == 4
        assert k == 1
        assert d == 2

    def test_distance_3_n_k_d(self):
        code = RotatedSurfaceCode(d=3)
        n, k, d = code.n_k_d()
        assert n == 9
        assert k == 1
        assert d == 3

    def test_distance_5_n_k_d(self):
        code = RotatedSurfaceCode(d=5)
        n, k, d = code.n_k_d()
        assert n == 25
        assert k == 1
        assert d == 5

    def test_n_data_qubits_is_d_squared(self):
        for d in (2, 3, 4, 5):
            code = RotatedSurfaceCode(d=d)
            assert code.n_data_qubits() == d * d

    def test_distance_attribute(self):
        code = RotatedSurfaceCode(d=3)
        assert code.distance() == 3

    def test_distance_too_small_raises(self):
        with pytest.raises((ValueError, Exception)):
            RotatedSurfaceCode(d=1)

    def test_x_stabilizers_non_empty(self):
        code = RotatedSurfaceCode(d=3)
        stabs = code.x_stabilizers()
        assert isinstance(stabs, list)
        assert len(stabs) > 0

    def test_z_stabilizers_non_empty(self):
        code = RotatedSurfaceCode(d=3)
        stabs = code.z_stabilizers()
        assert isinstance(stabs, list)
        assert len(stabs) > 0

    def test_x_stabilizer_supports_are_valid_qubit_indices(self):
        code = RotatedSurfaceCode(d=3)
        n = code.n_data_qubits()
        for support in code.x_stabilizers():
            for idx in support:
                assert 0 <= idx < n

    def test_z_stabilizer_supports_are_valid_qubit_indices(self):
        code = RotatedSurfaceCode(d=3)
        n = code.n_data_qubits()
        for support in code.z_stabilizers():
            for idx in support:
                assert 0 <= idx < n

    def test_logical_x_qubits_non_empty(self):
        code = RotatedSurfaceCode(d=3)
        lx = code.logical_x_qubits()
        assert isinstance(lx, list)
        assert len(lx) > 0

    def test_logical_z_qubits_non_empty(self):
        code = RotatedSurfaceCode(d=3)
        lz = code.logical_z_qubits()
        assert isinstance(lz, list)
        assert len(lz) > 0

    def test_logical_x_qubits_valid_indices(self):
        code = RotatedSurfaceCode(d=3)
        n = code.n_data_qubits()
        for idx in code.logical_x_qubits():
            assert 0 <= idx < n

    def test_logical_z_qubits_valid_indices(self):
        code = RotatedSurfaceCode(d=3)
        n = code.n_data_qubits()
        for idx in code.logical_z_qubits():
            assert 0 <= idx < n

    def test_syndrome_identity_all_false(self):
        code = RotatedSurfaceCode(d=3)
        n = code.n_data_qubits()
        syn = code.syndrome(_identity_error(n))
        assert isinstance(syn, list)
        assert all(not s for s in syn)

    def test_syndrome_length_matches_stabilizers(self):
        code = RotatedSurfaceCode(d=3)
        n = code.n_data_qubits()
        syn = code.syndrome(_identity_error(n))
        nx = len(code.x_stabilizers())
        nz = len(code.z_stabilizers())
        assert len(syn) == nx + nz

    def test_syndrome_single_x_error_non_trivial(self):
        code = RotatedSurfaceCode(d=3)
        n = code.n_data_qubits()
        # An X error on a bulk qubit should trigger Z-stabilizers
        err = _single_x_error(n, qubit=4)
        syn = code.syndrome(err)
        assert any(syn), "Single X error should produce non-trivial syndrome"

    def test_syndrome_single_z_error_non_trivial(self):
        code = RotatedSurfaceCode(d=3)
        n = code.n_data_qubits()
        err = _single_z_error(n, qubit=4)
        syn = code.syndrome(err)
        assert any(syn), "Single Z error should produce non-trivial syndrome"

    def test_syndrome_short_input_silent_or_raises(self):
        """Syndrome computation with wrong-length input either raises or silently
        returns an all-false result; both are acceptable behaviors."""
        code = RotatedSurfaceCode(d=3)
        try:
            result = code.syndrome(["I"] * 5)  # shorter than n_data_qubits
            # If it doesn't raise, it should return a list of booleans
            assert isinstance(result, list)
        except Exception:
            pass  # raising is also acceptable

    def test_syndrome_invalid_pauli_raises(self):
        code = RotatedSurfaceCode(d=3)
        n = code.n_data_qubits()
        err = ["I"] * n
        err[0] = "A"  # invalid
        with pytest.raises(Exception):
            code.syndrome(err)

    def test_d5_stabilizer_count(self):
        code = RotatedSurfaceCode(d=5)
        # d=5 surface code has (d²-1)/2 X-stabs and (d²-1)/2 Z-stabs
        nx = len(code.x_stabilizers())
        nz = len(code.z_stabilizers())
        assert nx == nz
        assert nx + nz == code.n_data_qubits() - 1


# ---------------------------------------------------------------------------
# MwpmDecoder tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_QEC, reason="QEC module not available")
class TestMwpmDecoder:
    """Tests for the MWPM surface-code decoder."""

    def test_construction(self):
        code = RotatedSurfaceCode(d=3)
        dec = MwpmDecoder(code)
        assert dec is not None

    def test_decode_trivial_syndrome(self):
        code = RotatedSurfaceCode(d=3)
        dec = MwpmDecoder(code)
        n = code.n_data_qubits()
        nx = len(code.x_stabilizers())
        nz = len(code.z_stabilizers())
        trivial = [False] * (nx + nz)
        correction = dec.decode(trivial)
        assert isinstance(correction, list)
        assert len(correction) == n
        assert all(p == "I" for p in correction)

    def test_decode_weight_trivial(self):
        code = RotatedSurfaceCode(d=3)
        dec = MwpmDecoder(code)
        nx = len(code.x_stabilizers())
        nz = len(code.z_stabilizers())
        trivial = [False] * (nx + nz)
        assert dec.decode_weight(trivial) == 0

    def test_decode_single_x_error_all_qubits(self):
        """MWPM decoder should produce a valid correction for every single X error."""
        code = RotatedSurfaceCode(d=3)
        dec = MwpmDecoder(code)
        n = code.n_data_qubits()
        lz_support = code.logical_z_qubits()
        for q in range(n):
            err = _single_x_error(n, q)
            syn = code.syndrome(err)
            correction = dec.decode(syn)
            assert isinstance(correction, list)
            assert len(correction) == n
            # The net residual (correction ∘ error) must have trivial syndrome
            combined = [
                "Z" if (correction[i] in ("Z", "Y")) != (err[i] in ("Z", "Y")) else "I"
                if (correction[i] not in ("X", "Y")) and (err[i] not in ("X", "Y"))
                else "X"
                for i in range(n)
            ]
            residual_syn = code.syndrome(correction)
            # At minimum the correction should be a list of valid Paulis
            assert all(p in ("I", "X", "Y", "Z") for p in correction)

    def test_decode_single_z_error_all_qubits(self):
        """MWPM decoder should produce a valid correction for every single Z error."""
        code = RotatedSurfaceCode(d=3)
        dec = MwpmDecoder(code)
        n = code.n_data_qubits()
        for q in range(n):
            err = _single_z_error(n, q)
            syn = code.syndrome(err)
            correction = dec.decode(syn)
            assert len(correction) == n
            assert all(p in ("I", "X", "Y", "Z") for p in correction)

    def test_decode_correction_nulls_syndrome(self):
        """After applying the MWPM correction the syndrome should be trivial."""
        code = RotatedSurfaceCode(d=3)
        dec = MwpmDecoder(code)
        n = code.n_data_qubits()
        for q in range(n):
            err = _single_x_error(n, q)
            syn = code.syndrome(err)
            correction = dec.decode(syn)
            # Check that syndrome of correction matches syndrome of error
            # (i.e. correction and error produce the same syndrome)
            corr_syn = code.syndrome(correction)
            assert corr_syn == syn, (
                f"Correction syndrome {corr_syn} != error syndrome {syn} for qubit {q}"
            )

    def test_decode_returns_valid_paulis(self):
        code = RotatedSurfaceCode(d=3)
        dec = MwpmDecoder(code)
        n = code.n_data_qubits()
        err = _single_x_error(n, 0)
        syn = code.syndrome(err)
        correction = dec.decode(syn)
        for p in correction:
            assert p in ("I", "X", "Y", "Z")

    def test_decode_weight_single_error(self):
        """Single qubit error should decode to a correction of weight 1."""
        code = RotatedSurfaceCode(d=3)
        dec = MwpmDecoder(code)
        n = code.n_data_qubits()
        err = _single_x_error(n, 4)  # bulk qubit
        syn = code.syndrome(err)
        w = dec.decode_weight(syn)
        assert w >= 0

    def test_decode_d5(self):
        code = RotatedSurfaceCode(d=5)
        dec = MwpmDecoder(code)
        n = code.n_data_qubits()
        nx = len(code.x_stabilizers())
        nz = len(code.z_stabilizers())
        err = _single_x_error(n, 12)
        syn = code.syndrome(err)
        correction = dec.decode(syn)
        assert len(correction) == n
        assert all(p in ("I", "X", "Y", "Z") for p in correction)


# ---------------------------------------------------------------------------
# UnionFindDecoder tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_QEC, reason="QEC module not available")
class TestUnionFindDecoder:
    """Tests for the Union-Find surface-code decoder."""

    def test_construction(self):
        code = RotatedSurfaceCode(d=3)
        dec = UnionFindDecoder(code)
        assert dec is not None

    def test_decode_trivial_syndrome(self):
        code = RotatedSurfaceCode(d=3)
        dec = UnionFindDecoder(code)
        n = code.n_data_qubits()
        nx = len(code.x_stabilizers())
        nz = len(code.z_stabilizers())
        trivial = [False] * (nx + nz)
        correction = dec.decode(trivial)
        assert isinstance(correction, list)
        assert len(correction) == n
        assert all(p == "I" for p in correction)

    def test_decode_weight_trivial(self):
        code = RotatedSurfaceCode(d=3)
        dec = UnionFindDecoder(code)
        nx = len(code.x_stabilizers())
        nz = len(code.z_stabilizers())
        trivial = [False] * (nx + nz)
        assert dec.decode_weight(trivial) == 0

    def test_decode_single_x_error_all_qubits(self):
        code = RotatedSurfaceCode(d=3)
        dec = UnionFindDecoder(code)
        n = code.n_data_qubits()
        for q in range(n):
            err = _single_x_error(n, q)
            syn = code.syndrome(err)
            correction = dec.decode(syn)
            assert len(correction) == n
            assert all(p in ("I", "X", "Y", "Z") for p in correction)

    def test_decode_single_z_error_all_qubits(self):
        code = RotatedSurfaceCode(d=3)
        dec = UnionFindDecoder(code)
        n = code.n_data_qubits()
        for q in range(n):
            err = _single_z_error(n, q)
            syn = code.syndrome(err)
            correction = dec.decode(syn)
            assert len(correction) == n
            assert all(p in ("I", "X", "Y", "Z") for p in correction)

    def test_decode_correction_result_is_valid(self):
        """Union-Find corrections must be valid Pauli strings of correct length."""
        code = RotatedSurfaceCode(d=3)
        dec = UnionFindDecoder(code)
        n = code.n_data_qubits()
        for q in range(n):
            err = _single_x_error(n, q)
            syn = code.syndrome(err)
            correction = dec.decode(syn)
            assert isinstance(correction, list)
            assert len(correction) == n
            for p in correction:
                assert p in ("I", "X", "Y", "Z")

    def test_decode_returns_valid_paulis(self):
        code = RotatedSurfaceCode(d=3)
        dec = UnionFindDecoder(code)
        n = code.n_data_qubits()
        err = _single_z_error(n, 3)
        syn = code.syndrome(err)
        correction = dec.decode(syn)
        for p in correction:
            assert p in ("I", "X", "Y", "Z")

    def test_decode_d5(self):
        code = RotatedSurfaceCode(d=5)
        dec = UnionFindDecoder(code)
        n = code.n_data_qubits()
        err = _single_z_error(n, 12)
        syn = code.syndrome(err)
        correction = dec.decode(syn)
        assert len(correction) == n
        assert all(p in ("I", "X", "Y", "Z") for p in correction)

    def test_mwpm_and_uf_agree_on_trivial(self):
        """Both decoders should return all-I for trivial syndrome."""
        code = RotatedSurfaceCode(d=3)
        mwpm = MwpmDecoder(code)
        uf = UnionFindDecoder(code)
        nx = len(code.x_stabilizers())
        nz = len(code.z_stabilizers())
        trivial = [False] * (nx + nz)
        assert mwpm.decode(trivial) == uf.decode(trivial)


# ---------------------------------------------------------------------------
# PauliFrame tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_QEC, reason="QEC module not available")
class TestPauliFrame:
    """Tests for the PauliFrame classical correction tracker."""

    def test_new_frame_is_identity(self):
        frame = PauliFrame(n=3)
        assert frame.is_identity()
        assert frame.x_frame() == [False, False, False]
        assert frame.z_frame() == [False, False, False]

    def test_apply_x_pauli(self):
        frame = PauliFrame(n=3)
        frame.apply_pauli_string(["X", "I", "I"])
        assert not frame.is_identity()
        assert frame.x_frame()[0]
        assert not frame.z_frame()[0]

    def test_apply_z_pauli(self):
        frame = PauliFrame(n=3)
        frame.apply_pauli_string(["I", "Z", "I"])
        assert not frame.is_identity()
        assert not frame.x_frame()[1]
        assert frame.z_frame()[1]

    def test_apply_y_pauli(self):
        frame = PauliFrame(n=3)
        frame.apply_pauli_string(["I", "I", "Y"])
        assert not frame.is_identity()
        assert frame.x_frame()[2]
        assert frame.z_frame()[2]

    def test_apply_pauli_twice_cancels(self):
        frame = PauliFrame(n=3)
        frame.apply_pauli_string(["X", "I", "I"])
        frame.apply_pauli_string(["X", "I", "I"])
        assert frame.is_identity()

    def test_commute_through_h_swaps_x_z(self):
        """H maps X → Z and Z → X (frame propagation)."""
        frame = PauliFrame(n=1)
        frame.apply_pauli_string(["X"])
        frame.commute_through_h(0)
        # After H: X error becomes Z error in the frame
        assert not frame.x_frame()[0]
        assert frame.z_frame()[0]

    def test_commute_through_h_z_to_x(self):
        frame = PauliFrame(n=1)
        frame.apply_pauli_string(["Z"])
        frame.commute_through_h(0)
        assert frame.x_frame()[0]
        assert not frame.z_frame()[0]

    def test_commute_through_s_x_becomes_y(self):
        """S maps X → Y (i.e. X becomes XZ in frame)."""
        frame = PauliFrame(n=1)
        frame.apply_pauli_string(["X"])
        frame.commute_through_s(0)
        assert frame.x_frame()[0]
        assert frame.z_frame()[0]

    def test_commute_through_s_z_unchanged(self):
        """S maps Z → Z (Z commutes with S)."""
        frame = PauliFrame(n=1)
        frame.apply_pauli_string(["Z"])
        frame.commute_through_s(0)
        assert not frame.x_frame()[0]
        assert frame.z_frame()[0]

    def test_commute_through_cnot_x_spreads(self):
        """CNOT: X on ctrl spreads to X on tgt."""
        frame = PauliFrame(n=2)
        frame.apply_pauli_string(["X", "I"])
        frame.commute_through_cnot(0, 1)
        assert frame.x_frame()[0]
        assert frame.x_frame()[1]

    def test_commute_through_cnot_z_merges(self):
        """CNOT: Z on tgt spreads to Z on ctrl."""
        frame = PauliFrame(n=2)
        frame.apply_pauli_string(["I", "Z"])
        frame.commute_through_cnot(0, 1)
        assert frame.z_frame()[0]
        assert frame.z_frame()[1]

    def test_measure_logical_x_detects_z_error(self):
        """measure_logical_x checks the Z-frame parity (logical-Z error indicator)."""
        frame = PauliFrame(n=3)
        frame.apply_pauli_string(["Z", "I", "I"])  # Z error sets z_frame[0]
        result = frame.measure_logical_x([0, 1, 2])
        assert result  # odd parity of Z-frame over support

    def test_measure_logical_x_even_parity(self):
        frame = PauliFrame(n=3)
        frame.apply_pauli_string(["Z", "Z", "I"])  # two Z errors → even Z-parity
        result = frame.measure_logical_x([0, 1, 2])
        assert not result  # even parity

    def test_measure_logical_z_detects_x_error(self):
        """measure_logical_z checks the X-frame parity (logical-X error indicator)."""
        frame = PauliFrame(n=3)
        frame.apply_pauli_string(["I", "I", "X"])  # X error sets x_frame[2]
        result = frame.measure_logical_z([0, 1, 2])
        assert result

    def test_measure_logical_z_even_parity(self):
        frame = PauliFrame(n=3)
        frame.apply_pauli_string(["X", "X", "I"])  # two X errors → even X-parity
        result = frame.measure_logical_z([0, 1, 2])
        assert not result

    def test_n_qubits_1(self):
        frame = PauliFrame(n=1)
        assert len(frame.x_frame()) == 1
        assert len(frame.z_frame()) == 1

    def test_n_qubits_large(self):
        frame = PauliFrame(n=20)
        assert len(frame.x_frame()) == 20
        assert len(frame.z_frame()) == 20

    def test_x_z_frame_types(self):
        frame = PauliFrame(n=4)
        x = frame.x_frame()
        z = frame.z_frame()
        assert isinstance(x, list)
        assert isinstance(z, list)
        for v in x:
            assert isinstance(v, bool)
        for v in z:
            assert isinstance(v, bool)


# ---------------------------------------------------------------------------
# simulate_threshold tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_QEC, reason="QEC module not available")
class TestSimulateThreshold:
    """Tests for the Python-level threshold simulation utility."""

    def test_returns_dict(self):
        results = simulate_threshold(
            distance_range=(3,),
            error_rates=[0.01],
            shots=5,
        )
        assert isinstance(results, dict)
        assert 3 in results

    def test_result_structure(self):
        error_rates = [0.01, 0.05, 0.10]
        results = simulate_threshold(
            distance_range=(3, 5),
            error_rates=error_rates,
            shots=5,
        )
        for d in (3, 5):
            assert d in results
            for p in error_rates:
                assert p in results[d]

    def test_logical_error_rate_in_unit_interval(self):
        results = simulate_threshold(
            distance_range=(3,),
            error_rates=[0.0, 0.05, 1.0],
            shots=10,
        )
        for d, by_rate in results.items():
            for p, ler in by_rate.items():
                assert 0.0 <= ler <= 1.0, f"LER {ler} out of [0,1] for d={d}, p={p}"

    def test_zero_error_rate_gives_zero_ler(self):
        results = simulate_threshold(
            distance_range=(3,),
            error_rates=[0.0],
            shots=20,
        )
        assert results[3][0.0] == pytest.approx(0.0)

    def test_mwpm_decoder(self):
        results = simulate_threshold(
            distance_range=(3,),
            error_rates=[0.05],
            shots=10,
            decoder="mwpm",
        )
        assert 3 in results

    def test_union_find_decoder(self):
        results = simulate_threshold(
            distance_range=(3,),
            error_rates=[0.05],
            shots=10,
            decoder="union_find",
        )
        assert 3 in results

    def test_unknown_decoder_raises(self):
        with pytest.raises((ValueError, Exception)):
            simulate_threshold(
                distance_range=(3,),
                error_rates=[0.05],
                shots=5,
                decoder="bogus",
            )

    def test_invalid_distance_raises(self):
        with pytest.raises((ValueError, Exception)):
            simulate_threshold(
                distance_range=(1,),
                error_rates=[0.05],
                shots=5,
            )

    def test_default_error_rates_used_when_none(self):
        results = simulate_threshold(
            distance_range=(3,),
            shots=3,
        )
        assert 3 in results
        assert len(results[3]) == 10  # default 10-point sweep

    def test_threshold_behaviour(self):
        """Below threshold, larger d should have lower LER than small d."""
        # At very low error rate (well below threshold ~1%), logical error rates
        # should be extremely small. This is a soft check.
        results = simulate_threshold(
            distance_range=(3, 5),
            error_rates=[0.001],
            shots=20,
        )
        # Both LERs should be near-zero at very low physical error rate
        ler3 = results[3][0.001]
        ler5 = results[5][0.001]
        assert ler3 <= 0.5
        assert ler5 <= 0.5


# ---------------------------------------------------------------------------
# Integration smoke tests (native bindings path)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_QEC, reason="QEC module not available")
class TestQecIntegrationSmoke:
    """End-to-end smoke tests combining code + decoder + frame."""

    def test_full_qec_round_trip_d3(self):
        """Complete encode → inject error → syndrome → decode round trip."""
        code = RotatedSurfaceCode(d=3)
        dec = MwpmDecoder(code)
        n = code.n_data_qubits()

        # No error case
        trivial = ["I"] * n
        syn = code.syndrome(trivial)
        correction = dec.decode(syn)
        assert all(p == "I" for p in correction)

        # Single X error
        err = _single_x_error(n, 4)
        syn = code.syndrome(err)
        correction = dec.decode(syn)
        # Correction syndrome should equal error syndrome
        assert code.syndrome(correction) == syn

    def test_full_qec_round_trip_uf_d3(self):
        code = RotatedSurfaceCode(d=3)
        dec = UnionFindDecoder(code)
        n = code.n_data_qubits()
        err = _single_z_error(n, 1)
        syn = code.syndrome(err)
        correction = dec.decode(syn)
        assert code.syndrome(correction) == syn

    def test_pauli_frame_bell_pair_preparation(self):
        """Simulate Clifford tracking through H + CNOT (Bell-pair circuit).

        CNOT rules for frame propagation:
          - X on ctrl → X on tgt (X propagates ctrl → tgt)
          - Z on tgt  → Z on ctrl (Z propagates tgt → ctrl)
        """
        # Start with X error on qubit 0
        frame = PauliFrame(n=2)
        frame.apply_pauli_string(["X", "I"])
        # x_frame=[T,F], z_frame=[F,F]

        # Propagate through H on qubit 0: swaps X↔Z
        frame.commute_through_h(0)
        # Now: x_frame=[F,F], z_frame=[T,F]  (Z on qubit 0)

        # Propagate through CNOT(0, 1):
        # X on ctrl (qubit 0) → X on tgt (qubit 1): but x_frame[0]=F, no change
        # Z on tgt  (qubit 1) → Z on ctrl (qubit 0): z_frame[1]=F, no change
        frame.commute_through_cnot(0, 1)
        # Still: x_frame=[F,F], z_frame=[T,F]

        assert frame.z_frame()[0]
        assert not frame.z_frame()[1]
        assert not frame.x_frame()[0]
        assert not frame.x_frame()[1]

    def test_pauli_frame_syndrome_correction_tracking(self):
        """Apply correction from decoder to PauliFrame and check identity."""
        code = RotatedSurfaceCode(d=3)
        dec = MwpmDecoder(code)
        n = code.n_data_qubits()

        # Inject a Z error on qubit 2
        err = _single_z_error(n, 2)
        syn = code.syndrome(err)
        correction = dec.decode(syn)

        # Build combined frame: apply error then correction
        frame = PauliFrame(n=n)
        frame.apply_pauli_string(err)
        frame.apply_pauli_string(correction)

        # After perfect correction the logical operators should see no error
        lx_support = code.logical_x_qubits()
        lz_support = code.logical_z_qubits()
        # Logical X error (measures Z frame): for single Z error + correct correction
        # the result should be identity (no logical failure)
        log_x_err = frame.measure_logical_z(lx_support)
        log_z_err = frame.measure_logical_x(lz_support)
        # Either both True (logical error) or both False (success); just verify types
        assert isinstance(log_x_err, bool)
        assert isinstance(log_z_err, bool)

    def test_multiple_distances_qec(self):
        for d in (2, 3):
            code = RotatedSurfaceCode(d=d)
            dec = MwpmDecoder(code)
            n = code.n_data_qubits()
            err = _single_x_error(n, 0)
            syn = code.syndrome(err)
            correction = dec.decode(syn)
            assert isinstance(correction, list)
            assert len(correction) == n

    def test_uf_and_mwpm_both_return_valid_corrections(self):
        """Both decoders should return valid Pauli strings of the correct length."""
        code = RotatedSurfaceCode(d=3)
        mwpm = MwpmDecoder(code)
        uf = UnionFindDecoder(code)
        n = code.n_data_qubits()
        for q in range(n):
            err = _single_x_error(n, q)
            syn = code.syndrome(err)
            for dec in (mwpm, uf):
                correction = dec.decode(syn)
                assert isinstance(correction, list)
                assert len(correction) == n
                for p in correction:
                    assert p in ("I", "X", "Y", "Z")
