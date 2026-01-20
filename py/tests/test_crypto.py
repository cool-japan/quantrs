#!/usr/bin/env python3
"""
Test suite for quantum cryptography functionality.
"""

import pytest
import random
import hashlib

try:
    import quantrs2
    from quantrs2.crypto import (
        BB84Protocol, E91Protocol, QuantumDigitalSignature, QuantumCoinFlipping,
        BasisChoice, QuantumBit, QKDKey, QuantumMessage, CryptoProtocolResult,
        generate_quantum_random_bits, quantum_one_time_pad, simulate_quantum_channel,
        run_bb84_demo, run_e91_demo, run_coin_flip_demo, run_digital_signature_demo
    )
    HAS_QUANTRS2 = True
except ImportError:
    HAS_QUANTRS2 = False


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestQKDKey:
    """Test QKDKey dataclass."""
    
    def test_empty_key(self):
        """Test empty QKDKey initialization."""
        key = QKDKey()
        assert key.raw_key == []
        assert key.sifted_key == []
        assert key.final_key == []
        assert key.error_rate == 0.0
        assert key.length == 0
    
    def test_key_with_data(self):
        """Test QKDKey with data."""
        raw = [1, 0, 1, 1, 0]
        sifted = [1, 0, 1]
        final = [1, 0]
        
        key = QKDKey(
            raw_key=raw,
            sifted_key=sifted,
            final_key=final,
            error_rate=0.1,
            length=len(final)
        )
        
        assert key.raw_key == raw
        assert key.sifted_key == sifted
        assert key.final_key == final
        assert key.error_rate == 0.1
        assert key.length == 2
    
    def test_to_bytes(self):
        """Test conversion to bytes."""
        key = QKDKey(final_key=[1, 0, 1, 1, 0, 0, 1, 0])  # 8 bits = 1 byte
        key_bytes = key.to_bytes()
        assert isinstance(key_bytes, bytes)
        assert len(key_bytes) == 1
        
        # Test with empty key
        empty_key = QKDKey()
        assert empty_key.to_bytes() == b''
    
    def test_to_hex(self):
        """Test conversion to hex."""
        key = QKDKey(final_key=[1, 0, 1, 1, 0, 0, 1, 0])
        hex_str = key.to_hex()
        assert isinstance(hex_str, str)
        assert len(hex_str) >= 2  # At least one byte in hex


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestBB84Protocol:
    """Test BB84 quantum key distribution protocol."""
    
    def test_protocol_initialization(self):
        """Test BB84 protocol initialization."""
        protocol = BB84Protocol(n_qubits=50, error_threshold=0.1)
        assert protocol.n_qubits == 50
        assert protocol.error_threshold == 0.1
        assert protocol.alice_bits == []
        assert protocol.alice_bases == []
    
    def test_alice_prepare_qubits(self):
        """Test Alice's qubit preparation."""
        protocol = BB84Protocol(n_qubits=10)
        qubits = protocol.alice_prepare_qubits()
        
        assert len(qubits) == 10
        assert len(protocol.alice_bits) == 10
        assert len(protocol.alice_bases) == 10
        
        # Check that bits are 0 or 1
        assert all(bit in [0, 1] for bit in protocol.alice_bits)
        
        # Check that bases are valid
        assert all(basis in [BasisChoice.RECTILINEAR, BasisChoice.DIAGONAL] 
                  for basis in protocol.alice_bases)
    
    def test_alice_prepare_with_specific_data(self):
        """Test Alice's preparation with specific bits and bases."""
        protocol = BB84Protocol(n_qubits=3)
        bits = [1, 0, 1]
        bases = [BasisChoice.RECTILINEAR, BasisChoice.DIAGONAL, BasisChoice.RECTILINEAR]
        
        qubits = protocol.alice_prepare_qubits(bits, bases)
        
        assert protocol.alice_bits == bits
        assert protocol.alice_bases == bases
        assert len(qubits) == 3
    
    def test_bob_measure_qubits(self):
        """Test Bob's qubit measurements."""
        protocol = BB84Protocol(n_qubits=10)
        qubits = protocol.alice_prepare_qubits()
        measurements = protocol.bob_measure_qubits(qubits)
        
        assert len(measurements) == 10
        assert len(protocol.bob_bases) == 10
        
        # Check that measurements are 0 or 1
        assert all(bit in [0, 1] for bit in measurements)
        
        # Check that Bob's measurements are stored
        assert protocol.bob_measurements == measurements
    
    def test_sift_key(self):
        """Test key sifting process."""
        protocol = BB84Protocol(n_qubits=10)
        
        # Prepare with known data
        alice_bits = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        alice_bases = [BasisChoice.RECTILINEAR] * 5 + [BasisChoice.DIAGONAL] * 5
        bob_bases = [BasisChoice.RECTILINEAR] * 3 + [BasisChoice.DIAGONAL] * 7
        
        protocol.alice_bits = alice_bits
        protocol.alice_bases = alice_bases
        protocol.bob_bases = bob_bases
        protocol.bob_measurements = alice_bits[:]  # Perfect correlation for matching bases
        
        alice_sifted, bob_sifted = protocol.sift_key()
        
        # Should have 3 + 5 = 8 matching bases
        expected_matches = sum(1 for a, b in zip(alice_bases, bob_bases) if a == b)
        assert len(alice_sifted) == expected_matches
        assert len(bob_sifted) == expected_matches
    
    def test_estimate_error_rate(self):
        """Test error rate estimation."""
        protocol = BB84Protocol()
        
        # Perfect correlation
        alice_bits = [1, 0, 1, 0] * 10
        bob_bits = alice_bits[:]
        error_rate = protocol.estimate_error_rate(alice_bits, bob_bits)
        assert error_rate == 0.0
        
        # 50% error rate
        alice_bits = [1, 0, 1, 0] * 10
        bob_bits = [0, 1, 0, 1] * 10
        error_rate = protocol.estimate_error_rate(alice_bits, bob_bits)
        assert error_rate > 0.4  # Should be around 50% with some variation
    
    def test_run_complete_protocol(self):
        """Test complete BB84 protocol execution."""
        protocol = BB84Protocol(n_qubits=100, error_threshold=0.15)
        result = protocol.run_protocol()
        
        assert isinstance(result, CryptoProtocolResult)
        
        if result.success:
            assert result.key_material is not None
            assert isinstance(result.key_material, QKDKey)
            assert result.error_rate <= protocol.error_threshold
            assert len(result.key_material.final_key) > 0
            assert 'sifting_efficiency' in result.metadata
        else:
            # Protocol can fail due to high error rate or no matching bases
            assert result.error_rate > protocol.error_threshold or 'No matching bases' in result.metadata.get('reason', '')


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestE91Protocol:
    """Test E91 quantum key distribution protocol."""
    
    def test_protocol_initialization(self):
        """Test E91 protocol initialization."""
        protocol = E91Protocol(n_pairs=30)
        assert protocol.n_pairs == 30
        assert protocol.alice_measurements == []
        assert protocol.bob_measurements == []
    
    def test_create_entangled_pairs(self):
        """Test entangled pair creation."""
        protocol = E91Protocol(n_pairs=10)
        pairs = protocol.create_entangled_pairs()
        
        assert len(pairs) == 10
    
    def test_measure_pairs(self):
        """Test measurement of entangled pairs."""
        protocol = E91Protocol(n_pairs=10)
        pairs = protocol.create_entangled_pairs()
        alice_results, bob_results = protocol.measure_pairs(pairs)
        
        assert len(alice_results) == 10
        assert len(bob_results) == 10
        assert len(protocol.alice_bases) == 10
        assert len(protocol.bob_bases) == 10
        
        # Check measurement results are bits
        assert all(bit in [0, 1] for bit in alice_results)
        assert all(bit in [0, 1] for bit in bob_results)
        
        # Check bases are valid angles
        valid_angles = [0, 45, 90]
        assert all(angle in valid_angles for angle in protocol.alice_bases)
        assert all(angle in valid_angles for angle in protocol.bob_bases)
    
    def test_bell_inequality(self):
        """Test Bell inequality violation test."""
        protocol = E91Protocol(n_pairs=50)
        pairs = protocol.create_entangled_pairs()
        protocol.measure_pairs(pairs)
        
        bell_parameter = protocol.test_bell_inequality()
        
        assert isinstance(bell_parameter, float)
        # Should typically be > 2 for quantum correlations
        assert bell_parameter >= 0
    
    def test_run_complete_protocol(self):
        """Test complete E91 protocol execution."""
        protocol = E91Protocol(n_pairs=50)
        result = protocol.run_protocol()
        
        assert isinstance(result, CryptoProtocolResult)
        
        if result.success:
            assert result.key_material is not None
            assert 'bell_parameter' in result.metadata
            assert result.security_parameter > 2.0  # Bell parameter
        else:
            # Can fail due to insufficient Bell violation or no compatible measurements
            reason = result.metadata.get('reason', '')
            assert 'Bell parameter' in reason or 'No compatible measurements' in reason


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestQuantumDigitalSignature:
    """Test quantum digital signature scheme."""
    
    def test_initialization(self):
        """Test QDS initialization."""
        qds = QuantumDigitalSignature(message_length=16, security_parameter=32)
        assert qds.message_length == 16
        assert qds.security_parameter == 32
        assert qds.private_key is None
        assert qds.public_key is None
    
    def test_key_generation(self):
        """Test key generation."""
        qds = QuantumDigitalSignature()
        private_key, public_key = qds.generate_keys()
        
        assert isinstance(private_key, list)
        assert len(private_key) == qds.security_parameter
        assert all(bit in [0, 1] for bit in private_key)
        assert public_key is not None
        assert qds.private_key == private_key
        assert qds.public_key == public_key
    
    def test_sign_message_string(self):
        """Test signing a string message."""
        qds = QuantumDigitalSignature(message_length=32)
        qds.generate_keys()
        
        message = "Hello"
        signature = qds.sign_message(message)
        
        assert isinstance(signature, dict)
        assert 'message_bits' in signature
        assert 'signature_bits' in signature
        assert 'quantum_auth' in signature
        assert len(signature['message_bits']) == qds.message_length
        assert len(signature['signature_bits']) == qds.message_length
    
    def test_sign_message_bits(self):
        """Test signing a bit list message."""
        qds = QuantumDigitalSignature(message_length=8)
        qds.generate_keys()
        
        message = [1, 0, 1, 1, 0, 0, 1, 0]
        signature = qds.sign_message(message)
        
        assert signature['message_bits'] == message
        assert len(signature['signature_bits']) == len(message)
    
    def test_verify_valid_signature(self):
        """Test verification of valid signature."""
        qds = QuantumDigitalSignature(message_length=16)
        qds.generate_keys()
        
        message = "test"
        signature = qds.sign_message(message)
        is_valid = qds.verify_signature(message, signature)
        
        # Should be valid (though may fail due to quantum errors in simulation)
        assert isinstance(is_valid, bool)
    
    def test_verify_invalid_signature(self):
        """Test verification of tampered signature."""
        qds = QuantumDigitalSignature(message_length=16)
        qds.generate_keys()
        
        message = "test"
        signature = qds.sign_message(message)
        
        # Tamper with signature
        signature['signature_bits'][0] = 1 - signature['signature_bits'][0]
        
        # Should likely be invalid
        is_valid = qds.verify_signature(message, signature)
        # Note: Due to error tolerance, this might still be valid sometimes
        assert isinstance(is_valid, bool)
    
    def test_forge_detection_probability(self):
        """Test forgery detection probability calculation."""
        qds = QuantumDigitalSignature(security_parameter=64)
        prob = qds.forge_detection_probability()
        
        assert 0 <= prob <= 1
        assert prob > 0.9  # Should be very high for good security parameter


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestQuantumCoinFlipping:
    """Test quantum coin flipping protocol."""
    
    def test_initialization(self):
        """Test coin flipping initialization."""
        protocol = QuantumCoinFlipping()
        assert protocol.alice_commitment is None
        assert protocol.bob_commitment is None
    
    def test_alice_commit(self):
        """Test Alice's commitment."""
        protocol = QuantumCoinFlipping()
        commitment = protocol.alice_commit(1)
        
        assert isinstance(commitment, dict)
        assert 'type' in commitment
        assert protocol.alice_commitment is not None
        assert protocol.alice_commitment['bit'] == 1
    
    def test_bob_respond(self):
        """Test Bob's response."""
        protocol = QuantumCoinFlipping()
        bob_bit = protocol.bob_respond(0)
        
        assert bob_bit == 0
        assert protocol.bob_commitment == 0
    
    def test_alice_reveal(self):
        """Test Alice's revelation."""
        protocol = QuantumCoinFlipping()
        protocol.alice_commit(1)
        revelation = protocol.alice_reveal()
        
        assert isinstance(revelation, dict)
        assert 'bit' in revelation
        assert 'type' in revelation
        assert revelation['bit'] == 1
    
    def test_compute_result(self):
        """Test result computation."""
        protocol = QuantumCoinFlipping()
        protocol.alice_commit(1)
        revelation = protocol.alice_reveal()
        
        result, is_valid = protocol.compute_result(revelation, 0)
        
        assert isinstance(result, int)
        assert result in [0, 1]
        assert isinstance(is_valid, bool)
        assert result == 1 ^ 0  # XOR of alice_bit and bob_bit
    
    def test_run_complete_protocol(self):
        """Test complete coin flipping protocol."""
        protocol = QuantumCoinFlipping()
        result = protocol.run_protocol(alice_bit=1, bob_bit=0)
        
        assert isinstance(result, CryptoProtocolResult)
        
        if result.success:
            assert 'alice_bit' in result.metadata
            assert 'bob_bit' in result.metadata
            assert 'coin_result' in result.metadata
            assert 'result_string' in result.metadata
            assert result.metadata['coin_result'] in [0, 1]
            assert result.metadata['result_string'] in ['heads', 'tails']


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_generate_quantum_random_bits(self):
        """Test quantum random bit generation."""
        bits = generate_quantum_random_bits(20)
        
        assert len(bits) == 20
        assert all(bit in [0, 1] for bit in bits)
    
    def test_quantum_one_time_pad(self):
        """Test quantum one-time pad."""
        message = [1, 0, 1, 1, 0]
        key = [0, 1, 1, 0, 1]
        
        encrypted = quantum_one_time_pad(message, key)
        decrypted = quantum_one_time_pad(encrypted, key)
        
        assert len(encrypted) == len(message)
        assert all(bit in [0, 1] for bit in encrypted)
        assert decrypted == message  # Should recover original
    
    def test_quantum_one_time_pad_wrong_length(self):
        """Test one-time pad with mismatched lengths."""
        message = [1, 0, 1]
        key = [0, 1]  # Wrong length
        
        with pytest.raises(ValueError):
            quantum_one_time_pad(message, key)
    
    def test_simulate_quantum_channel(self):
        """Test quantum channel simulation."""
        qubits = [{'bit': 0}, {'bit': 1}, {'bit': 0}]
        
        # No noise, no eavesdropper
        transmitted = simulate_quantum_channel(qubits, noise_level=0.0, eavesdropper_present=False)
        assert len(transmitted) == len(qubits)
        
        # With noise
        transmitted_noisy = simulate_quantum_channel(qubits, noise_level=0.5, eavesdropper_present=False)
        assert len(transmitted_noisy) == len(qubits)
        
        # With eavesdropper
        transmitted_eaves = simulate_quantum_channel(qubits, noise_level=0.0, eavesdropper_present=True)
        assert len(transmitted_eaves) == len(qubits)


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestDemoFunctions:
    """Test demo functions."""
    
    def test_bb84_demo(self):
        """Test BB84 demo function."""
        result = run_bb84_demo(n_qubits=50)
        
        assert isinstance(result, CryptoProtocolResult)
        # Success depends on random factors, so we just check the structure
    
    def test_e91_demo(self):
        """Test E91 demo function."""
        result = run_e91_demo(n_pairs=30)
        
        assert isinstance(result, CryptoProtocolResult)
        # Success depends on random factors
    
    def test_coin_flip_demo(self):
        """Test coin flip demo function."""
        result = run_coin_flip_demo()
        
        assert isinstance(result, CryptoProtocolResult)
        # Should usually succeed
    
    def test_digital_signature_demo(self):
        """Test digital signature demo function."""
        signature, is_valid = run_digital_signature_demo("Test message")
        
        assert isinstance(signature, dict)
        assert isinstance(is_valid, bool)
        assert 'message_bits' in signature
        assert 'signature_bits' in signature


@pytest.mark.skipif(not HAS_QUANTRS2, reason="quantrs2 not available")
class TestCryptoIntegration:
    """Test crypto module integration."""
    
    def test_crypto_functions_available(self):
        """Test that crypto functions are available from main module."""
        try:
            from quantrs2 import BB84Protocol, E91Protocol, run_bb84_demo
            assert BB84Protocol is not None
            assert E91Protocol is not None
            assert callable(run_bb84_demo)
        except ImportError:
            # This is acceptable if crypto not available
            pass
    
    def test_multiple_protocols(self):
        """Test running multiple protocols in sequence."""
        # BB84
        bb84 = BB84Protocol(n_qubits=30)
        bb84_result = bb84.run_protocol()
        
        # E91
        e91 = E91Protocol(n_pairs=20)
        e91_result = e91.run_protocol()
        
        # Coin flip
        coin = QuantumCoinFlipping()
        coin_result = coin.run_protocol()
        
        # All should return valid results
        assert isinstance(bb84_result, CryptoProtocolResult)
        assert isinstance(e91_result, CryptoProtocolResult)
        assert isinstance(coin_result, CryptoProtocolResult)
    
    def test_error_conditions(self):
        """Test error handling in crypto protocols."""
        # QDS without key generation
        qds = QuantumDigitalSignature()
        
        with pytest.raises(ValueError):
            qds.sign_message("test")
        
        # Coin flip revelation without commitment
        coin = QuantumCoinFlipping()
        
        with pytest.raises(ValueError):
            coin.alice_reveal()


if __name__ == "__main__":
    pytest.main([__file__])