#!/usr/bin/env python3
"""
Quantum Cryptography Toolkit Demo

This example demonstrates various quantum cryptographic protocols
including quantum key distribution, digital signatures, and other
quantum security applications using QuantRS2.
"""

import numpy as np
import random
from typing import List, Dict, Any

try:
    import quantrs2
    from quantrs2 import Circuit
    from quantrs2.crypto import (
        BB84Protocol, E91Protocol, QuantumDigitalSignature, QuantumCoinFlipping,
        BasisChoice, generate_quantum_random_bits, quantum_one_time_pad,
        simulate_quantum_channel, run_bb84_demo, run_e91_demo
    )
    HAS_QUANTRS2 = True
except ImportError:
    print("QuantRS2 not available. Please install QuantRS2 first.")
    HAS_QUANTRS2 = False
    exit(1)


def demo_bb84_protocol():
    """Demonstrate BB84 quantum key distribution protocol."""
    print("="*60)
    print("BB84 Quantum Key Distribution Protocol")
    print("="*60)
    
    print("The BB84 protocol allows Alice and Bob to generate a shared secret key")
    print("using quantum mechanics, with security guaranteed by quantum physics.")
    print()
    
    # Initialize protocol
    n_qubits = 100
    error_threshold = 0.11
    protocol = BB84Protocol(n_qubits=n_qubits, error_threshold=error_threshold)
    
    print(f"Protocol parameters:")
    print(f"  Number of qubits: {n_qubits}")
    print(f"  Error threshold: {error_threshold}")
    print()
    
    # Run the protocol step by step
    print("Step 1: Alice prepares qubits...")
    qubits = protocol.alice_prepare_qubits()
    print(f"  Alice prepared {len(qubits)} qubits")
    print(f"  Alice's bits (first 10): {protocol.alice_bits[:10]}")
    print(f"  Alice's bases (first 10): {[b.value for b in protocol.alice_bases[:10]]}")
    print()
    
    print("Step 2: Bob measures qubits...")
    measurements = protocol.bob_measure_qubits(qubits)
    print(f"  Bob measured {len(measurements)} qubits")
    print(f"  Bob's results (first 10): {measurements[:10]}")
    print(f"  Bob's bases (first 10): {[b.value for b in protocol.bob_bases[:10]]}")
    print()
    
    print("Step 3: Basis sifting...")
    alice_sifted, bob_sifted = protocol.sift_key()
    print(f"  Matching bases found: {len(alice_sifted)}")
    print(f"  Sifting efficiency: {len(alice_sifted)/len(qubits)*100:.1f}%")
    print(f"  Alice's sifted key (first 10): {alice_sifted[:10]}")
    print(f"  Bob's sifted key (first 10): {bob_sifted[:10]}")
    print()
    
    print("Step 4: Error rate estimation...")
    error_rate = protocol.estimate_error_rate(alice_sifted, bob_sifted, test_fraction=0.3)
    print(f"  Estimated error rate: {error_rate:.3f}")
    print(f"  Security threshold: {error_threshold}")
    
    if error_rate <= error_threshold:
        print("  ✓ Error rate acceptable - protocol can continue")
    else:
        print("  ✗ Error rate too high - possible eavesdropping detected!")
    print()
    
    print("Step 5: Complete protocol execution...")
    result = protocol.run_protocol()
    
    if result.success:
        print("  ✓ Protocol completed successfully!")
        print(f"  Final key length: {result.key_material.length} bits")
        print(f"  Key generation rate: {result.metadata['key_generation_rate']*100:.1f}%")
        print(f"  Final key (hex): {result.key_material.to_hex()}")
        print(f"  Security parameter: {result.security_parameter:.3f}")
    else:
        print(f"  ✗ Protocol failed: {result.metadata.get('reason', 'Unknown error')}")
        print(f"  Error rate: {result.error_rate:.3f}")


def demo_e91_protocol():
    """Demonstrate E91 quantum key distribution protocol."""
    print("\n" + "="*60)
    print("E91 (Ekert) Quantum Key Distribution Protocol")
    print("="*60)
    
    print("The E91 protocol uses entangled quantum pairs and Bell inequality")
    print("tests to detect eavesdropping and generate secure keys.")
    print()
    
    # Initialize protocol
    n_pairs = 50
    protocol = E91Protocol(n_pairs=n_pairs)
    
    print(f"Protocol parameters:")
    print(f"  Number of entangled pairs: {n_pairs}")
    print()
    
    print("Step 1: Creating entangled pairs...")
    pairs = protocol.create_entangled_pairs()
    print(f"  Created {len(pairs)} EPR pairs")
    print("  Each pair is in state |Φ+⟩ = (|00⟩ + |11⟩)/√2")
    print()
    
    print("Step 2: Alice and Bob measure their qubits...")
    alice_results, bob_results = protocol.measure_pairs(pairs)
    print(f"  Alice's measurements: {alice_results[:10]}...")
    print(f"  Bob's measurements: {bob_results[:10]}...")
    print(f"  Alice's angles: {protocol.alice_bases[:10]}...")
    print(f"  Bob's angles: {protocol.bob_bases[:10]}...")
    print()
    
    print("Step 3: Testing Bell inequality...")
    bell_parameter = protocol.test_bell_inequality()
    print(f"  Bell parameter S: {bell_parameter:.3f}")
    print(f"  Classical limit: S ≤ 2.0")
    print(f"  Quantum limit: S ≤ 2√2 ≈ 2.83")
    
    if bell_parameter > 2.0:
        print("  ✓ Bell inequality violated - quantum correlations confirmed!")
        if bell_parameter > 2.5:
            print("  ✓ Strong quantum violation - high security")
        else:
            print("  ⚠ Moderate quantum violation - acceptable security")
    else:
        print("  ✗ No Bell violation - possible eavesdropping or classical correlations")
    print()
    
    print("Step 4: Complete protocol execution...")
    result = protocol.run_protocol()
    
    if result.success:
        print("  ✓ E91 protocol completed successfully!")
        print(f"  Bell parameter: {result.metadata['bell_parameter']:.3f}")
        print(f"  Final key length: {result.key_material.length} bits")
        print(f"  Error rate: {result.error_rate:.3f}")
        print(f"  Key generation rate: {result.metadata['key_generation_rate']*100:.1f}%")
        print(f"  Final key (hex): {result.key_material.to_hex()}")
    else:
        print(f"  ✗ Protocol failed: {result.metadata.get('reason', 'Unknown error')}")


def demo_quantum_digital_signatures():
    """Demonstrate quantum digital signatures."""
    print("\n" + "="*60)
    print("Quantum Digital Signatures")
    print("="*60)
    
    print("Quantum digital signatures provide information-theoretic security")
    print("for message authentication using quantum mechanics.")
    print()
    
    # Initialize signature scheme
    message_length = 64
    security_parameter = 128
    qds = QuantumDigitalSignature(
        message_length=message_length,
        security_parameter=security_parameter
    )
    
    print(f"Signature scheme parameters:")
    print(f"  Message length: {message_length} bits")
    print(f"  Security parameter: {security_parameter} bits")
    print()
    
    print("Step 1: Key generation...")
    private_key, public_key = qds.generate_keys()
    print(f"  Private key length: {len(private_key)} bits")
    print(f"  Private key (first 16 bits): {private_key[:16]}")
    print(f"  Public key type: {public_key['verification_data']}")
    
    forge_detection = qds.forge_detection_probability()
    print(f"  Forgery detection probability: {forge_detection:.6f}")
    print(f"  Security level: {-np.log2(1-forge_detection):.1f} bits")
    print()
    
    print("Step 2: Message signing...")
    message = "Quantum cryptography is the future of secure communication!"
    print(f"  Message: '{message}'")
    
    signature = qds.sign_message(message)
    print(f"  Signature generated successfully")
    print(f"  Message bits (first 16): {signature['message_bits'][:16]}")
    print(f"  Signature bits (first 16): {signature['signature_bits'][:16]}")
    print(f"  Quantum authentication: {signature['quantum_auth']['type']}")
    print(f"  Timestamp: {signature['timestamp']}")
    print()
    
    print("Step 3: Signature verification...")
    is_valid = qds.verify_signature(message, signature)
    
    if is_valid:
        print("  ✓ Signature verification successful!")
        print("  ✓ Message authenticity confirmed")
        print("  ✓ Sender identity verified")
    else:
        print("  ✗ Signature verification failed!")
        print("  ✗ Message may have been tampered with")
    print()
    
    print("Step 4: Testing signature tampering...")
    tampered_signature = signature.copy()
    tampered_signature['signature_bits'] = signature['signature_bits'][:]
    tampered_signature['signature_bits'][0] = 1 - tampered_signature['signature_bits'][0]
    
    is_tampered_valid = qds.verify_signature(message, tampered_signature)
    print(f"  Tampered signature verification: {'Valid' if is_tampered_valid else 'Invalid'}")
    
    if not is_tampered_valid:
        print("  ✓ Tampering detected successfully!")
    else:
        print("  ⚠ Tampering not detected (may occur due to error tolerance)")


def demo_quantum_coin_flipping():
    """Demonstrate quantum coin flipping protocol."""
    print("\n" + "="*60)
    print("Quantum Coin Flipping Protocol")
    print("="*60)
    
    print("Quantum coin flipping allows two parties to fairly flip a coin")
    print("even when they don't trust each other, using quantum commitment.")
    print()
    
    # Initialize protocol
    protocol = QuantumCoinFlipping()
    
    print("Participants: Alice and Bob")
    print("Goal: Generate a fair random bit")
    print()
    
    print("Step 1: Alice commits to her bit...")
    alice_bit = random.randint(0, 1)
    commitment = protocol.alice_commit(alice_bit)
    print(f"  Alice chose bit: {alice_bit}")
    print(f"  Commitment type: {commitment['type']}")
    print("  Alice sends commitment to Bob (without revealing her bit)")
    print()
    
    print("Step 2: Bob responds with his bit...")
    bob_bit = random.randint(0, 1)
    bob_response = protocol.bob_respond(bob_bit)
    print(f"  Bob chose bit: {bob_response}")
    print("  Bob sends his bit to Alice")
    print()
    
    print("Step 3: Alice reveals her commitment...")
    revelation = protocol.alice_reveal()
    print(f"  Alice reveals her bit: {revelation['bit']}")
    print(f"  Revelation type: {revelation['type']}")
    print()
    
    print("Step 4: Computing coin flip result...")
    result_bit, is_valid = protocol.compute_result(revelation, bob_bit)
    
    if is_valid:
        result_string = "heads" if result_bit == 0 else "tails"
        print(f"  ✓ Commitment verification successful")
        print(f"  Alice's bit: {revelation['bit']}")
        print(f"  Bob's bit: {bob_bit}")
        print(f"  Result bit (XOR): {result_bit}")
        print(f"  Coin flip result: {result_string}")
        
        # Check fairness
        print(f"\n  Protocol fairness analysis:")
        print(f"  - Neither party can control the outcome alone")
        print(f"  - Both parties contribute to the randomness")
        print(f"  - Quantum commitment prevents cheating")
    else:
        print("  ✗ Commitment verification failed!")
        print("  ✗ Possible cheating attempt detected")
    
    print()
    print("Step 5: Running complete protocol...")
    full_result = protocol.run_protocol()
    
    if full_result.success:
        print("  ✓ Complete protocol executed successfully!")
        print(f"  Final result: {full_result.metadata['result_string']}")
        print(f"  Alice contributed: {full_result.metadata['alice_bit']}")
        print(f"  Bob contributed: {full_result.metadata['bob_bit']}")
    else:
        print(f"  ✗ Protocol failed: {full_result.metadata.get('reason', 'Unknown error')}")


def demo_quantum_random_generation():
    """Demonstrate quantum random number generation."""
    print("\n" + "="*60)
    print("Quantum Random Number Generation")
    print("="*60)
    
    print("Quantum mechanics provides true randomness through quantum measurements.")
    print("This is superior to classical pseudorandom number generators.")
    print()
    
    # Generate quantum random bits
    n_bits = 100
    print(f"Generating {n_bits} quantum random bits...")
    
    quantum_bits = generate_quantum_random_bits(n_bits)
    
    print(f"Generated bits (first 32): {quantum_bits[:32]}")
    print(f"Generated bits (hex): {quantum_bits_to_hex(quantum_bits[:32])}")
    print()
    
    # Analyze randomness
    zeros = quantum_bits.count(0)
    ones = quantum_bits.count(1)
    bias = abs(zeros - ones) / len(quantum_bits)
    
    print("Randomness analysis:")
    print(f"  Total bits: {len(quantum_bits)}")
    print(f"  Zeros: {zeros} ({zeros/len(quantum_bits)*100:.1f}%)")
    print(f"  Ones: {ones} ({ones/len(quantum_bits)*100:.1f}%)")
    print(f"  Bias: {bias:.3f}")
    
    if bias < 0.1:
        print("  ✓ Good randomness quality")
    else:
        print("  ⚠ High bias detected (may be due to simulation)")
    
    # Demonstrate one-time pad encryption
    print("\nQuantum One-Time Pad Demo:")
    message_bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    key_bits = quantum_bits[:len(message_bits)]
    
    print(f"  Message:    {message_bits}")
    print(f"  Random key: {key_bits}")
    
    encrypted = quantum_one_time_pad(message_bits, key_bits)
    decrypted = quantum_one_time_pad(encrypted, key_bits)
    
    print(f"  Encrypted:  {encrypted}")
    print(f"  Decrypted:  {decrypted}")
    
    if decrypted == message_bits:
        print("  ✓ Perfect encryption/decryption!")
    else:
        print("  ✗ Encryption error!")


def demo_eavesdropping_detection():
    """Demonstrate eavesdropping detection in QKD."""
    print("\n" + "="*60)
    print("Eavesdropping Detection in QKD")
    print("="*60)
    
    print("Quantum key distribution can detect eavesdropping attempts")
    print("due to the fundamental properties of quantum mechanics.")
    print()
    
    print("Scenario 1: Secure channel (no eavesdropping)")
    print("-" * 50)
    
    # Run BB84 without eavesdropping
    protocol1 = BB84Protocol(n_qubits=100, error_threshold=0.11)
    
    # Simulate clean channel
    qubits1 = protocol1.alice_prepare_qubits()
    clean_qubits = simulate_quantum_channel(qubits1, noise_level=0.01, eavesdropper_present=False)
    measurements1 = protocol1.bob_measure_qubits(clean_qubits)
    
    alice_sifted1, bob_sifted1 = protocol1.sift_key()
    error_rate1 = protocol1.estimate_error_rate(alice_sifted1, bob_sifted1)
    
    print(f"  Sifted key length: {len(alice_sifted1)} bits")
    print(f"  Error rate: {error_rate1:.3f}")
    print(f"  Status: {'Secure' if error_rate1 <= 0.11 else 'Insecure'}")
    print()
    
    print("Scenario 2: Channel with eavesdropping")
    print("-" * 50)
    
    # Run BB84 with eavesdropping
    protocol2 = BB84Protocol(n_qubits=100, error_threshold=0.11)
    
    # Simulate eavesdropped channel
    qubits2 = protocol2.alice_prepare_qubits()
    tapped_qubits = simulate_quantum_channel(qubits2, noise_level=0.01, eavesdropper_present=True)
    measurements2 = protocol2.bob_measure_qubits(tapped_qubits)
    
    alice_sifted2, bob_sifted2 = protocol2.sift_key()
    error_rate2 = protocol2.estimate_error_rate(alice_sifted2, bob_sifted2)
    
    print(f"  Sifted key length: {len(alice_sifted2)} bits")
    print(f"  Error rate: {error_rate2:.3f}")
    print(f"  Status: {'Secure' if error_rate2 <= 0.11 else 'Eavesdropping detected!'}")
    print()
    
    print("Comparison:")
    print(f"  Clean channel error rate: {error_rate1:.3f}")
    print(f"  Tapped channel error rate: {error_rate2:.3f}")
    print(f"  Error rate increase: {error_rate2 - error_rate1:.3f}")
    
    if error_rate2 > error_rate1 + 0.05:
        print("  ✓ Eavesdropping successfully detected!")
    else:
        print("  ⚠ Eavesdropping not clearly detected (simulation limitation)")


def demo_protocol_comparison():
    """Compare different quantum cryptographic protocols."""
    print("\n" + "="*60)
    print("Quantum Cryptographic Protocol Comparison")
    print("="*60)
    
    protocols = []
    
    print("Running multiple protocols for comparison...")
    print()
    
    # BB84 Protocol
    print("1. BB84 Protocol:")
    bb84_result = run_bb84_demo(n_qubits=80)
    protocols.append(("BB84", bb84_result))
    
    if bb84_result.success:
        print(f"   ✓ Success | Key: {bb84_result.key_material.length} bits | Error: {bb84_result.error_rate:.3f}")
    else:
        print(f"   ✗ Failed | Error: {bb84_result.error_rate:.3f}")
    
    # E91 Protocol
    print("2. E91 Protocol:")
    e91_result = run_e91_demo(n_pairs=40)
    protocols.append(("E91", e91_result))
    
    if e91_result.success:
        bell_param = e91_result.metadata.get('bell_parameter', 0)
        print(f"   ✓ Success | Key: {e91_result.key_material.length} bits | Bell: {bell_param:.2f}")
    else:
        print(f"   ✗ Failed | Reason: {e91_result.metadata.get('reason', 'Unknown')}")
    
    print()
    print("Protocol Comparison Summary:")
    print("=" * 50)
    
    successful_protocols = [p for p in protocols if p[1].success]
    
    if successful_protocols:
        print(f"Successful protocols: {len(successful_protocols)}/{len(protocols)}")
        print()
        
        for name, result in successful_protocols:
            key_length = result.key_material.length if result.key_material else 0
            error_rate = result.error_rate
            security = result.security_parameter
            
            print(f"{name}:")
            print(f"  Key length: {key_length} bits")
            print(f"  Error rate: {error_rate:.3f}")
            print(f"  Security parameter: {security:.3f}")
            
            if name == "BB84":
                efficiency = result.metadata.get('key_generation_rate', 0)
                print(f"  Key generation rate: {efficiency*100:.1f}%")
            elif name == "E91":
                bell_param = result.metadata.get('bell_parameter', 0)
                print(f"  Bell parameter: {bell_param:.3f}")
            print()
        
        # Find best performer
        if len(successful_protocols) > 1:
            best_key_length = max(successful_protocols, key=lambda x: x[1].key_material.length if x[1].key_material else 0)
            best_security = max(successful_protocols, key=lambda x: x[1].security_parameter)
            
            print("Best performers:")
            print(f"  Longest key: {best_key_length[0]} ({best_key_length[1].key_material.length} bits)")
            print(f"  Highest security: {best_security[0]} ({best_security[1].security_parameter:.3f})")
    else:
        print("No protocols succeeded in this run.")
        print("This can happen due to the randomness in quantum simulations.")


def quantum_bits_to_hex(bits: List[int]) -> str:
    """Convert quantum bits to hex representation."""
    # Pad to byte boundary
    padded_bits = bits[:]
    while len(padded_bits) % 8 != 0:
        padded_bits.append(0)
    
    # Convert to bytes
    hex_chars = []
    for i in range(0, len(padded_bits), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(padded_bits):
                byte_val |= padded_bits[i + j] << (7 - j)
        hex_chars.append(f"{byte_val:02x}")
    
    return ''.join(hex_chars)


def main():
    """Run all quantum cryptography demos."""
    print("QuantRS2 Quantum Cryptography Toolkit Demo")
    print("This demo showcases various quantum cryptographic protocols")
    print("demonstrating the unique security properties of quantum mechanics")
    
    try:
        # Run all demos
        demo_bb84_protocol()
        demo_e91_protocol()
        demo_quantum_digital_signatures()
        demo_quantum_coin_flipping()
        demo_quantum_random_generation()
        demo_eavesdropping_detection()
        demo_protocol_comparison()
        
        print("\n" + "="*60)
        print("Demo Complete!")
        print("="*60)
        print("\nKey takeaways:")
        print("• Quantum cryptography provides information-theoretic security")
        print("• QKD protocols can detect eavesdropping attempts")
        print("• Quantum digital signatures ensure message authenticity")
        print("• Quantum coin flipping enables fair random bit generation")
        print("• Quantum random numbers are truly random, not pseudorandom")
        print("• Bell inequality tests verify quantum correlations")
        
        print("\nSecurity advantages:")
        print("• Unconditional security based on quantum physics")
        print("• Eavesdropping detection through quantum mechanics")
        print("• No computational assumptions required")
        print("• Future-proof against quantum computers")
        
        print("\nNext steps:")
        print("• Implement quantum key distribution networks")
        print("• Develop quantum authentication protocols")
        print("• Create quantum-safe communication systems")
        print("• Explore post-quantum cryptography integration")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("This may be due to missing dependencies or system limitations.")
        print("The crypto module provides classical simulations when quantum hardware is unavailable.")


if __name__ == "__main__":
    if HAS_QUANTRS2:
        main()
    else:
        print("Please install QuantRS2 to run this demo.")