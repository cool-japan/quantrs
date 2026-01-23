//! Quantum Cryptography and Key Distribution Example
//!
//! This example demonstrates advanced quantum cryptography protocols including:
//! - BB84 Quantum Key Distribution (QKD)
//! - E91 Entanglement-Based QKD
//! - Quantum Digital Signatures
//! - Post-Quantum Cryptographic Primitives
//!
//! These protocols provide information-theoretic security guaranteed by
//! the laws of quantum mechanics, not computational complexity.
//!
//! Run with: cargo run --example quantum_cryptography

use quantrs2_core::{
    gate::{multi, single, GateOp},
    qubit::QubitId,
};
use scirs2_core::random::{thread_rng, RandBeta};
use scirs2_core::Complex64;

fn main() {
    println!("=================================================================");
    println!("   QuantRS2-Core: Quantum Cryptography & Key Distribution");
    println!("=================================================================\n");

    // Demonstrate BB84 protocol
    demonstrate_bb84_protocol();
    println!();

    // Demonstrate E91 protocol
    demonstrate_e91_protocol();
    println!();

    // Demonstrate quantum digital signatures
    demonstrate_quantum_digital_signatures();
    println!();

    // Demonstrate post-quantum cryptography
    demonstrate_post_quantum_crypto();
    println!();

    // Demonstrate security analysis
    demonstrate_security_analysis();
    println!();

    println!("=================================================================");
    println!("   Example Complete!");
    println!("=================================================================");
}

/// Demonstrate BB84 Quantum Key Distribution Protocol
fn demonstrate_bb84_protocol() {
    println!("BB84 QUANTUM KEY DISTRIBUTION PROTOCOL");
    println!("-----------------------------------------------------------------");

    println!("BB84 (Bennett & Brassard, 1984) is the first QKD protocol.");
    println!("It uses two non-orthogonal bases to encode classical bits.");
    println!();

    println!("Protocol Overview:");
    println!("  Participants: Alice (sender) and Bob (receiver)");
    println!("  Security: Information-theoretic (unconditionally secure)");
    println!("  Threat Model: Protects against quantum attacks including:");
    println!("    • Intercept-resend attacks");
    println!("    • Entanglement attacks");
    println!("    • Coherent attacks");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("STEP 1: Quantum Transmission");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("Encoding Bases:");
    println!("  Rectilinear basis (⊕): {{|0⟩, |1⟩}}");
    println!("    |0⟩ represents bit 0");
    println!("    |1⟩ represents bit 1");
    println!("    Preparation: None (|0⟩) or X gate (|1⟩)");
    println!();
    println!("  Diagonal basis (⊗): {{|+⟩, |−⟩}}");
    println!("    |+⟩ = (|0⟩ + |1⟩)/√2 represents bit 0");
    println!("    |−⟩ = (|0⟩ − |1⟩)/√2 represents bit 1");
    println!("    Preparation: H (|+⟩) or X·H (|−⟩)");
    println!();

    println!("Example Transmission (4 qubits):");
    println!();
    println!("  Alice's random bits:       [1, 0, 1, 0]");
    println!("  Alice's random bases:      [⊕, ⊗, ⊕, ⊗]");
    println!("  Alice's quantum states:    [|1⟩, |+⟩, |1⟩, |+⟩]");
    println!();
    println!("  Quantum Channel");
    println!("       ↓↓↓↓");
    println!();
    println!("  Bob's random bases:        [⊕, ⊕, ⊕, ⊗]");
    println!("  Bob's measurement results: [1, ?, 1, 0]");
    println!("                                 ↑");
    println!("                          Wrong basis!");
    println!();

    println!("State Preparation Circuits:");
    println!("  Bit 0, Basis ⊕:  |0⟩ ─────────");
    println!("  Bit 1, Basis ⊕:  |0⟩ ───X──── → |1⟩");
    println!("  Bit 0, Basis ⊗:  |0⟩ ───H──── → |+⟩");
    println!("  Bit 1, Basis ⊗:  |0⟩ ─X─H──── → |−⟩");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("STEP 2: Basis Reconciliation (Classical Communication)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("Alice publicly announces her basis choices: [⊕, ⊗, ⊕, ⊗]");
    println!("Bob publicly announces his basis choices:   [⊕, ⊕, ⊕, ⊗]");
    println!();
    println!("Matching bases (sifted key positions):     [✓, ✗, ✓, ✓]");
    println!("                                            pos 0  pos 2  pos 3");
    println!();
    println!("Sifted key:");
    println!("  Alice's sifted bits: [1, 1, 0]  (positions 0, 2, 3)");
    println!("  Bob's sifted bits:   [1, 1, 0]  (should match!)");
    println!();
    println!("Expected sift rate: ~50% (half the qubits discarded)");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("STEP 3: Error Detection & Privacy Amplification");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("Error Checking (randomly sample ~50% of sifted key):");
    println!("  Alice announces: position 0 → bit 1");
    println!("  Bob confirms:    position 0 → bit 1  ✓");
    println!("  Error rate: 0% (in this example)");
    println!();
    println!("  These revealed bits are discarded from final key.");
    println!();

    println!("Quantum Bit Error Rate (QBER) Analysis:");
    println!("  QBER = (# disagreements) / (# compared bits)");
    println!();
    println!("  QBER < 11%:  Secure (Eve's information < Alice's)");
    println!("  QBER > 15%:  Abort (too much noise or eavesdropping)");
    println!("  11% < QBER < 15%: Gray zone (advanced analysis needed)");
    println!();

    println!("Privacy Amplification (if QBER acceptable):");
    println!("  Apply universal hash functions to compress key");
    println!("  Reduces Eve's information exponentially");
    println!("  Final key: Information-theoretically secure");
    println!();
    println!("  Example: 1000 sifted bits → 512 secure key bits");
    println!("           (depends on QBER and security parameter)");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("Security Analysis");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("Why BB84 is Secure:");
    println!();
    println!("  1. No-Cloning Theorem");
    println!("     Eve cannot perfectly copy unknown quantum states");
    println!("     Any cloning attempt introduces detectable errors");
    println!();
    println!("  2. Measurement Disturbance");
    println!("     Measuring in wrong basis creates ~25% error rate");
    println!("     Eve's interference increases QBER → detected");
    println!();
    println!("  3. Information-Theoretic Security");
    println!("     After privacy amplification: Eve's info → 0");
    println!("     Security proven against any attack (including quantum)");
    println!();

    println!("Attack Examples:");
    println!();
    println!("  Intercept-Resend Attack:");
    println!("    1. Eve intercepts qubit from Alice");
    println!("    2. Eve measures in random basis");
    println!("    3. Eve resends state to Bob");
    println!("    Result: ~25% QBER → Alice & Bob abort");
    println!();
    println!("  Optimal Individual Attack:");
    println!("    Eve uses optimal measurement strategy");
    println!("    Still introduces minimum 12.5% QBER");
    println!();

    println!("Practical Considerations:");
    println!("  • Typical key rates: 1-10 kbps over 10-100 km");
    println!("  • Distance limited by photon loss (no amplification)");
    println!("  • Quantum repeaters needed for long distances");
    println!("  • Secure against future quantum computers");
    println!();

    println!("  ✓ BB84 protocol demonstrated");
    println!("  ✓ First QKD protocol, deployed in commercial systems");
}

/// Demonstrate E91 Entanglement-Based QKD Protocol
fn demonstrate_e91_protocol() {
    println!("E91 ENTANGLEMENT-BASED QKD PROTOCOL");
    println!("-----------------------------------------------------------------");

    println!("E91 (Ekert, 1991) uses quantum entanglement for key distribution.");
    println!("Security verified by Bell inequality violations.");
    println!();

    println!("Protocol Overview:");
    println!("  Source: Generates entangled pairs (EPR pairs)");
    println!("  State: |Ψ⁻⟩ = (|01⟩ − |10⟩)/√2 (Bell singlet state)");
    println!("  Security: Based on quantum non-locality");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("STEP 1: Entanglement Distribution");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("Entangled Pair Generation:");
    println!("  |0⟩─────────H────●──── → Alice's qubit");
    println!("                   │");
    println!("  |0⟩─────────────X──── → Bob's qubit");
    println!();
    println!("  Result: |Ψ⁻⟩ = (|01⟩ − |10⟩)/√2");
    println!("  Properties:");
    println!("    • Maximally entangled");
    println!("    • Anti-correlated in any basis");
    println!("    • Perfect (negative) correlations");
    println!();

    println!("Distribution:");
    println!("  Trusted Source");
    println!("       / \\");
    println!("      /   \\");
    println!("     ↓     ↓");
    println!("  Alice   Bob");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("STEP 2: Measurement in Random Bases");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("Measurement Bases (3 bases each):");
    println!();
    println!("  Alice's bases:");
    println!("    θ_A1 = 0°    (Z basis)");
    println!("    θ_A2 = 45°   (intermediate)");
    println!("    θ_A3 = 90°   (X basis)");
    println!();
    println!("  Bob's bases:");
    println!("    θ_B1 = 45°   (intermediate)");
    println!("    θ_B2 = 90°   (X basis)");
    println!("    θ_B3 = 135°  (intermediate)");
    println!();

    println!("For each entangled pair:");
    println!("  1. Alice randomly chooses basis θ_Ai");
    println!("  2. Bob randomly chooses basis θ_Bj");
    println!("  3. Both measure their qubits");
    println!("  4. Record results");
    println!();

    println!("Example (4 pairs):");
    println!();
    println!("  Pair  Alice basis  Bob basis  Alice bit  Bob bit");
    println!("  ───────────────────────────────────────────────────");
    println!("   1      θ_A1(0°)   θ_B1(45°)     0         1");
    println!("   2      θ_A2(45°)  θ_B2(90°)     1         0");
    println!("   3      θ_A1(0°)   θ_B3(135°)    1         1");
    println!("   4      θ_A3(90°)  θ_B2(90°)     0         1  ← Match!");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("STEP 3: Bell Inequality Test (Security Verification)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("CHSH Inequality:");
    println!("  S = |E(θ_A1,θ_B1) + E(θ_A1,θ_B2) + E(θ_A2,θ_B1) − E(θ_A2,θ_B2)|");
    println!();
    println!("  Classical bound: S ≤ 2");
    println!("  Quantum maximum: S = 2√2 ≈ 2.828");
    println!();
    println!("  where E(θ_i,θ_j) = correlation coefficient");
    println!();

    println!("Security Guarantee:");
    println!("  IF S > 2: Entanglement verified");
    println!("           → No local hidden variable explanation");
    println!("           → Eve cannot have copied the states");
    println!("           → Key is secure!");
    println!();
    println!("  IF S ≤ 2: Abort protocol (possible eavesdropping)");
    println!();

    println!("Typical experimental values:");
    println!("  Clean quantum channel: S ≈ 2.7-2.82");
    println!("  With noise: S decreases");
    println!("  Threshold: S > 2.0 (security proven)");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("STEP 4: Key Extraction");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("Basis Selection for Key:");
    println!("  Use measurements where Alice and Bob chose SAME basis");
    println!("  (e.g., both at 90° in example above)");
    println!();
    println!("  Due to anti-correlation: flip one party's bits");
    println!("  Example: Alice (0), Bob (1) → both get key bit '0'");
    println!();

    println!("Post-processing:");
    println!("  1. Error correction (similar to BB84)");
    println!("  2. Privacy amplification");
    println!("  3. Final secure key");
    println!();

    println!("Advantages over BB84:");
    println!("  ✓ Device-independent security possible");
    println!("  ✓ Detects eavesdropping via Bell violations");
    println!("  ✓ No need to trust quantum devices");
    println!("  ✓ Stronger security proof");
    println!();

    println!("Challenges:");
    println!("  ✗ Requires entanglement source");
    println!("  ✗ More complex implementation");
    println!("  ✗ Lower key rates in practice");
    println!();

    println!("  ✓ E91 protocol demonstrated");
    println!("  ✓ First entanglement-based QKD, enables device-independent security");
}

/// Demonstrate Quantum Digital Signatures
fn demonstrate_quantum_digital_signatures() {
    println!("QUANTUM DIGITAL SIGNATURES");
    println!("-----------------------------------------------------------------");

    println!("Quantum digital signatures provide authentication and");
    println!("non-repudiation with information-theoretic security.");
    println!();

    println!("Classical vs Quantum Signatures:");
    println!();
    println!("  Classical (RSA, ECDSA):");
    println!("    • Based on computational hardness");
    println!("    • Vulnerable to quantum computers (Shor's algorithm)");
    println!("    • Security assumptions may fail");
    println!();
    println!("  Quantum:");
    println!("    • Based on laws of quantum mechanics");
    println!("    • Information-theoretically secure");
    println!("    • No computational assumptions");
    println!("    • Future-proof against quantum attacks");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("Protocol: Quantum Digital Signature Scheme");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("Participants:");
    println!("  • Signer (Alice): Creates signatures");
    println!("  • Recipients (Bob, Charlie, ...): Verify signatures");
    println!();

    println!("PHASE 1: Key Distribution");
    println!("-----------------------------------------------------------------");
    println!();
    println!("  1. Alice generates quantum keys using QKD:");
    println!("     k_AB: shared key with Bob");
    println!("     k_AC: shared key with Charlie");
    println!("     ...");
    println!();
    println!("  2. Each key has sufficient length:");
    println!("     |k| ≥ L × (# messages to sign)");
    println!("     where L is security parameter");
    println!();

    println!("PHASE 2: Signature Generation");
    println!("-----------------------------------------------------------------");
    println!();
    println!("  To sign message m:");
    println!();
    println!("  1. Compute classical hash: h = H(m)");
    println!("     H is universal hash function");
    println!();
    println!("  2. For each recipient i:");
    println!("     sig_i = Authenticate(h, k_Ai)");
    println!("     where Authenticate uses quantum key k_Ai");
    println!();
    println!("  3. Signature package:");
    println!("     S = (m, {{sig_B, sig_C, ...}})");
    println!();

    println!("  Quantum Authentication:");
    println!("     Uses quantum states as authentication tags");
    println!("     |ψ_auth⟩ = Encode(h, k)");
    println!("     Properties:");
    println!("       • Unclonable (no-cloning theorem)");
    println!("       • Tampering detectable");
    println!();

    println!("PHASE 3: Signature Distribution & Verification");
    println!("-----------------------------------------------------------------");
    println!();
    println!("  Alice sends to Bob:");
    println!("     Classical: (m, sig_C, sig_D, ...)");
    println!("     Quantum:   sig_B (quantum signature for Bob)");
    println!();
    println!("  Bob's Verification:");
    println!();
    println!("    1. Verify own quantum signature sig_B:");
    println!("       Measure quantum states using key k_AB");
    println!("       Check: Verify(H(m), k_AB, sig_B) = ✓");
    println!();
    println!("    2. Store other signatures (sig_C, sig_D) classically");
    println!();

    println!("PHASE 4: Transfer & Non-Repudiation");
    println!("-----------------------------------------------------------------");
    println!();
    println!("  Bob forwards signature to Charlie:");
    println!("     Send: (m, sig_C, sig_B_measured)");
    println!();
    println!("  Charlie's Verification:");
    println!();
    println!("    1. Verify own quantum signature sig_C:");
    println!("       Verify(H(m), k_AC, sig_C) = ✓");
    println!();
    println!("    2. Verify Bob's measured signature:");
    println!("       Check Bob correctly verified: sig_B_measured");
    println!();
    println!("    3. If both pass: Accept signature as valid from Alice");
    println!();

    println!("Security Properties:");
    println!();
    println!("  1. Unforgeability:");
    println!("     No one (even Bob) can forge Alice's signature");
    println!("     Guaranteed by quantum no-cloning theorem");
    println!();
    println!("  2. Non-Repudiation:");
    println!("     Alice cannot deny signing message m");
    println!("     All recipients can verify independently");
    println!();
    println!("  3. Transferability:");
    println!("     Bob can prove to Charlie that Alice signed");
    println!("     Without revealing his quantum signature");
    println!();
    println!("  4. Information-Theoretic Security:");
    println!("     Even adversary with unlimited computing power");
    println!("     Cannot forge or repudiate signatures");
    println!();

    println!("Practical Applications:");
    println!("  • Quantum blockchain and distributed ledgers");
    println!("  • Secure government communications");
    println!("  • Financial transactions requiring long-term security");
    println!("  • Legal documents with quantum certification");
    println!("  • Supply chain verification");
    println!();

    println!("Implementation Considerations:");
    println!("  • Requires quantum key distribution infrastructure");
    println!("  • Limited signature lifetime (quantum memory)");
    println!("  • Trade-off: security vs. signature length");
    println!("  • Hybrid schemes combine classical and quantum");
    println!();

    println!("  ✓ Quantum digital signatures demonstrated");
    println!("  ✓ Information-theoretically secure authentication");
}

/// Demonstrate Post-Quantum Cryptographic Primitives
fn demonstrate_post_quantum_crypto() {
    println!("POST-QUANTUM CRYPTOGRAPHIC PRIMITIVES");
    println!("-----------------------------------------------------------------");

    println!("Post-quantum cryptography (PQC) protects against attacks by");
    println!("quantum computers while running on classical computers.");
    println!();

    println!("Quantum Threat to Classical Cryptography:");
    println!();
    println!("  Broken by Quantum Computers:");
    println!("    ✗ RSA (Shor's algorithm)");
    println!("    ✗ Elliptic Curve Cryptography (Shor's algorithm)");
    println!("    ✗ Diffie-Hellman (Shor's algorithm)");
    println!("    ✗ DSA, ECDSA signatures (Shor's algorithm)");
    println!();
    println!("  Still Secure (with larger parameters):");
    println!("    ✓ AES-256 (Grover's gives √n speedup → use 256 bits)");
    println!("    ✓ SHA-3 (collision resistance reduced but acceptable)");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("1. LATTICE-BASED CRYPTOGRAPHY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("Learning With Errors (LWE) Problem:");
    println!("  Given: (A, b = As + e) where");
    println!("    A: random matrix");
    println!("    s: secret vector");
    println!("    e: small error vector");
    println!("  Find: secret s");
    println!();
    println!("  Security: No known quantum algorithm");
    println!("  Basis: Lattice shortest vector problem (SVP)");
    println!();

    println!("CRYSTALS-Kyber (NIST PQC Standard for Encryption):");
    println!("  Key Generation:");
    println!("    1. Sample secret s, error e from χ");
    println!("    2. Compute A·s + e = public key");
    println!("    3. Private key = s");
    println!();
    println!("  Encryption (message m → ciphertext c):");
    println!("    c = (u, v) where");
    println!("    u = A^T·r + e_1");
    println!("    v = pk^T·r + e_2 + Encode(m)");
    println!();
    println!("  Decryption:");
    println!("    m = Decode(v - s^T·u)");
    println!();
    println!("  Parameters (Kyber-768, security level 3):");
    println!("    Public key: 1,184 bytes");
    println!("    Ciphertext: 1,088 bytes");
    println!("    Shared secret: 32 bytes");
    println!();

    println!("CRYSTALS-Dilithium (NIST PQC Standard for Signatures):");
    println!("  Based on: Fiat-Shamir with rejection sampling");
    println!("  Signature size: ~2,420 bytes");
    println!("  Public key: ~1,312 bytes");
    println!("  Security: 128-bit post-quantum");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("2. CODE-BASED CRYPTOGRAPHY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("McEliece Cryptosystem (1978, still secure!):");
    println!("  Based on: Decoding random linear codes (NP-hard)");
    println!();
    println!("  Key Generation:");
    println!("    1. Choose random Goppa code G");
    println!("    2. Generate S (scrambler), P (permutation)");
    println!("    3. Public key: G' = S·G·P (looks random)");
    println!("    4. Private key: (S, G, P)");
    println!();
    println!("  Encryption:");
    println!("    c = m·G' + e");
    println!("    where e is random error vector (weight t)");
    println!();
    println!("  Decryption:");
    println!("    1. Compute c·P^{{-1}} = m·S·G + e·P^{{-1}}");
    println!("    2. Use Goppa decoder to find m·S");
    println!("    3. Recover m using S^{{-1}}");
    println!();
    println!("  Disadvantage: Large public keys (~1 MB)");
    println!("  Advantage: Very fast encryption, 40+ years of analysis");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("3. HASH-BASED SIGNATURES");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("SPHINCS+ (NIST PQC Standard):");
    println!("  Based on: Hash functions (e.g., SHA-256)");
    println!("  Security: Relies only on hash function security");
    println!();
    println!("  Properties:");
    println!("    ✓ Stateless (no state synchronization needed)");
    println!("    ✓ Simple security proof");
    println!("    ✓ Well-understood cryptographic primitive");
    println!("    ✗ Larger signatures (~17-50 KB)");
    println!("    ✗ Slower signing/verification");
    println!();
    println!("  Signature size: ~17,088 bytes (SPHINCS+-128s)");
    println!("  Public key: 32 bytes");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("4. MULTIVARIATE CRYPTOGRAPHY");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("Based on: Solving systems of multivariate polynomial equations");
    println!();
    println!("Example System (simplified):");
    println!("  f₁(x₁,...,xₙ) = a₁₁x₁² + a₁₂x₁x₂ + ... = y₁");
    println!("  f₂(x₁,...,xₙ) = a₂₁x₂² + a₂₂x₂x₃ + ... = y₂");
    println!("  ...");
    println!();
    println!("  Hard Problem: Given y, find x such that f(x) = y");
    println!();
    println!("  Advantages:");
    println!("    ✓ Fast verification");
    println!("    ✓ Short signatures");
    println!();
    println!("  Challenges:");
    println!("    ✗ Some schemes broken in past");
    println!("    ✗ Large public keys");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("NIST Post-Quantum Cryptography Standardization");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("Selected Algorithms:");
    println!();
    println!("  Public Key Encryption / KEMs:");
    println!("    • CRYSTALS-Kyber (primary)");
    println!();
    println!("  Digital Signatures:");
    println!("    • CRYSTALS-Dilithium (primary)");
    println!("    • FALCON (alternative)");
    println!("    • SPHINCS+ (stateless alternative)");
    println!();
    println!("  Under Consideration:");
    println!("    • BIKE, HQC (code-based KEMs)");
    println!("    • Additional signature schemes");
    println!();

    println!("Migration Strategy:");
    println!();
    println!("  Phase 1 (Now): Hybrid approach");
    println!("    Classical + PQC together");
    println!("    Example: TLS with both RSA and Kyber");
    println!();
    println!("  Phase 2 (2026-2030): Gradual transition");
    println!("    Deploy PQC in critical infrastructure");
    println!("    Maintain backward compatibility");
    println!();
    println!("  Phase 3 (2030+): Full PQC deployment");
    println!("    Classical algorithms deprecated");
    println!("    All systems quantum-safe");
    println!();

    println!("Quantum Computing in PQC:");
    println!();
    println!("  Role of Quantum Computers:");
    println!("    • QKD: Uses quantum mechanics for key distribution");
    println!("    • PQC: Classical crypto resistant to quantum attacks");
    println!("    • Complementary approaches for quantum-safe future");
    println!();

    println!("  ✓ Post-quantum cryptographic primitives demonstrated");
    println!("  ✓ Essential for protecting against future quantum computers");
}

/// Demonstrate Security Analysis and Threat Models
fn demonstrate_security_analysis() {
    println!("SECURITY ANALYSIS & THREAT MODELS");
    println!("-----------------------------------------------------------------");

    println!("Comprehensive security analysis of quantum cryptographic protocols.");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("Attack Strategies in Quantum Cryptography");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("1. PHOTON NUMBER SPLITTING (PNS) ATTACK");
    println!("-----------------------------------------------------------------");
    println!();
    println!("  Scenario: Practical QKD uses weak coherent pulses");
    println!("    Some pulses contain >1 photon");
    println!();
    println!("  Attack:");
    println!("    1. Eve blocks all single-photon pulses");
    println!("    2. For multi-photon pulses, Eve keeps one photon");
    println!("    3. Eve forwards remaining photons to Bob");
    println!("    4. Eve measures her photon after basis announcement");
    println!();
    println!("  Result: Eve gains full key information without detection");
    println!();
    println!("  Countermeasures:");
    println!("    • Decoy state protocol (attenuated decoys)");
    println!("    • Monitor channel loss carefully");
    println!("    • Use single-photon sources");
    println!();

    println!("2. TROJAN HORSE ATTACK");
    println!("-----------------------------------------------------------------");
    println!();
    println!("  Attack:");
    println!("    1. Eve sends bright light into Alice/Bob's device");
    println!("    2. Light reflects off internal components");
    println!("    3. Eve analyzes reflected light");
    println!("    4. Gains information about device settings");
    println!();
    println!("  Countermeasures:");
    println!("    • Optical isolators");
    println!("    • Spectral filtering");
    println!("    • Active monitoring for unexpected light");
    println!();

    println!("3. DETECTOR BLINDING ATTACK");
    println!("-----------------------------------------------------------------");
    println!();
    println!("  Attack on Single-Photon Detectors:");
    println!("    1. Eve sends bright pulse to blind detector");
    println!("    2. Detector operates in linear mode (not single-photon)");
    println!("    3. Eve controls detector clicks");
    println!("    4. Eve gains full key without error increase");
    println!();
    println!("  Countermeasures:");
    println!("    • Monitor detector parameters");
    println!("    • Randomize detector efficiency");
    println!("    • Use multiple detector types");
    println!();

    println!("4. TIME-SHIFT ATTACK");
    println!("-----------------------------------------------------------------");
    println!();
    println!("  Attack:");
    println!("    1. Eve delays certain pulses");
    println!("    2. Causes timing correlations");
    println!("    3. Exploits if timing affects basis choice");
    println!();
    println!("  Countermeasures:");
    println!("    • Strict timing windows");
    println!("    • Random delays in preparation");
    println!("    • Independent basis choice from timing");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("Device-Independent QKD (DI-QKD)");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("Motivation: Don't trust quantum devices!");
    println!();
    println!("  Standard QKD assumes:");
    println!("    • Devices work as specified");
    println!("    • No hidden channels or backdoors");
    println!("    • Proper implementation");
    println!();
    println!("  DI-QKD only assumes:");
    println!("    • Laws of quantum mechanics");
    println!("    • No information leakage from lab");
    println!();

    println!("Security via Bell Inequalities:");
    println!();
    println!("  1. Use entangled states (like E91)");
    println!("  2. Measure Bell inequality violation");
    println!("  3. Violation → quantum correlations → secure");
    println!("  4. No need to trust device internals");
    println!();
    println!("  S > 2 proves:");
    println!("    • Eve cannot have full knowledge");
    println!("    • Devices not completely compromised");
    println!("    • Key is provably secure");
    println!();

    println!("Challenges:");
    println!("  ✗ Very low key rates (~10 bits/sec)");
    println!("  ✗ Requires very low noise");
    println!("  ✗ Complex experimental setup");
    println!("  ✓ Ultimate security guarantee");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("Quantum Hacking: Real-World Attacks");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("Historical Attacks on Commercial QKD Systems:");
    println!();
    println!("  2010 - Detector Blinding (University of Toronto):");
    println!("    • Successfully attacked commercial systems");
    println!("    • Gained full key without detection");
    println!("    • Led to improved security standards");
    println!();
    println!("  2015 - Phase Remapping Attack:");
    println!("    • Exploited phase modulator imperfections");
    println!("    • Affected phase-encoded QKD");
    println!();
    println!("  2016 - Wavelength Attack:");
    println!("    • Used different wavelengths to exploit detectors");
    println!();

    println!("Lessons Learned:");
    println!("  • Security proofs assume ideal devices");
    println!("  • Implementation vulnerabilities are real");
    println!("  • Continuous security evaluation needed");
    println!("  • Quantum hacking drives better systems");
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("Security Parameters & Standards");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    println!("Key Security Metrics:");
    println!();
    println!("  Quantum Bit Error Rate (QBER):");
    println!("    QBER < 11%:  Provably secure");
    println!("    QBER < 15%:  Potentially secure (context-dependent)");
    println!("    QBER > 15%:  Abort protocol");
    println!();
    println!("  Security Parameter (ε):");
    println!("    ε = 10⁻¹⁰: Probability of security failure");
    println!("    Lower ε → longer key needed for same security");
    println!();
    println!("  Key Rate:");
    println!("    R ≈ q × [1 - h(QBER) - leak_EC]");
    println!("    where:");
    println!("      q: sifting rate");
    println!("      h: binary entropy");
    println!("      leak_EC: error correction leakage");
    println!();

    println!("Composable Security:");
    println!("  Modern QKD security proofs use composability");
    println!("  Ensures security when:");
    println!("    • Running multiple protocol instances");
    println!("    • Composing with other crypto protocols");
    println!("    • Using in real-world applications");
    println!();

    println!("Standards & Certifications:");
    println!("  • ETSI GS QKD: European standards");
    println!("  • ITU-T: International telecom standards");
    println!("  • NIST: PQC standardization");
    println!("  • Common Criteria: Security evaluation");
    println!();

    println!("  ✓ Security analysis demonstrated");
    println!("  ✓ Comprehensive threat model and countermeasures");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example_runs() {
        main();
    }

    #[test]
    fn test_bb84_protocol() {
        demonstrate_bb84_protocol();
    }

    #[test]
    fn test_e91_protocol() {
        demonstrate_e91_protocol();
    }

    #[test]
    fn test_quantum_signatures() {
        demonstrate_quantum_digital_signatures();
    }

    #[test]
    fn test_post_quantum_crypto() {
        demonstrate_post_quantum_crypto();
    }

    #[test]
    fn test_security_analysis() {
        demonstrate_security_analysis();
    }
}
