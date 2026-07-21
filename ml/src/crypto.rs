//! Quantum cryptography: QKD protocols and post-quantum key exchange.
//!
//! Provides simulations of BB84, E91, and B92 quantum key distribution
//! protocols, plus lattice-based post-quantum key encapsulation suitable
//! for integration with quantum-secured network protocols.

use crate::error::{MLError, Result};
use quantrs2_circuit::prelude::Circuit;
use quantrs2_sim::statevector::StateVectorSimulator;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use std::fmt;

/// Types of quantum key distribution protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProtocolType {
    /// BB84 protocol (Bennett and Brassard, 1984)
    BB84,

    /// E91 protocol (Ekert, 1991)
    E91,

    /// B92 protocol (Bennett, 1992)
    B92,

    /// BBM92 protocol (Bennett, Brassard, and Mermin, 1992)
    BBM92,

    /// SARG04 protocol (Scarani, Acin, Ribordy, and Gisin, 2004)
    SARG04,
}

/// Represents a party in a quantum cryptographic protocol
#[derive(Debug, Clone)]
pub struct Party {
    /// Party's name
    pub name: String,

    /// Party's key (if generated)
    pub key: Option<Vec<u8>>,

    /// Party's chosen bases (for BB84-like protocols)
    pub bases: Option<Vec<usize>>,

    /// Party's quantum state (if applicable)
    pub state: Option<Vec<f64>>,
}

/// Quantum key distribution protocol
#[derive(Debug, Clone)]
pub struct QuantumKeyDistribution {
    /// Type of QKD protocol
    pub protocol: ProtocolType,

    /// Number of qubits to use in the protocol
    pub num_qubits: usize,

    /// Alice party
    pub alice: Party,

    /// Bob party
    pub bob: Party,

    /// Error rate for the quantum channel
    pub error_rate: f64,

    /// Security parameter (number of bits to use for security checks)
    pub security_bits: usize,
}

impl QuantumKeyDistribution {
    /// Creates a new QKD protocol instance
    pub fn new(protocol: ProtocolType, num_qubits: usize) -> Self {
        QuantumKeyDistribution {
            protocol,
            num_qubits,
            alice: Party {
                name: "Alice".to_string(),
                key: None,
                bases: None,
                state: None,
            },
            bob: Party {
                name: "Bob".to_string(),
                key: None,
                bases: None,
                state: None,
            },
            error_rate: 0.0,
            security_bits: num_qubits / 10,
        }
    }

    /// Sets the error rate for the quantum channel
    pub fn with_error_rate(mut self, error_rate: f64) -> Self {
        self.error_rate = error_rate;
        self
    }

    /// Sets the security parameter
    pub fn with_security_bits(mut self, security_bits: usize) -> Self {
        self.security_bits = security_bits;
        self
    }

    /// Distributes a key using the specified QKD protocol
    pub fn distribute_key(&mut self) -> Result<usize> {
        match self.protocol {
            ProtocolType::BB84 => self.bb84_protocol(),
            ProtocolType::E91 => self.e91_protocol(),
            ProtocolType::B92 => self.b92_protocol(),
            ProtocolType::BBM92 => self.bbm92_protocol(),
            ProtocolType::SARG04 => self.sarg04_protocol(),
        }
    }

    /// Implements the BB84 protocol
    fn bb84_protocol(&mut self) -> Result<usize> {
        // This is a dummy implementation
        // In a real implementation, this would simulate the BB84 protocol

        // Generate random bits for Alice
        let alice_bits = (0..self.num_qubits)
            .map(|_| {
                if thread_rng().random::<f64>() > 0.5 {
                    1u8
                } else {
                    0u8
                }
            })
            .collect::<Vec<_>>();

        // Generate random bases for Alice and Bob
        let alice_bases = (0..self.num_qubits)
            .map(|_| {
                if thread_rng().random::<f64>() > 0.5 {
                    1usize
                } else {
                    0usize
                }
            })
            .collect::<Vec<_>>();

        let bob_bases = (0..self.num_qubits)
            .map(|_| {
                if thread_rng().random::<f64>() > 0.5 {
                    1usize
                } else {
                    0usize
                }
            })
            .collect::<Vec<_>>();

        // Determine where Alice and Bob used the same basis
        let matching_bases = alice_bases
            .iter()
            .zip(bob_bases.iter())
            .enumerate()
            .filter_map(|(i, (a, b))| if a == b { Some(i) } else { None })
            .collect::<Vec<_>>();

        // Get the key bits from matching bases positions
        let mut key_bits = Vec::new();
        for &i in &matching_bases {
            // Apply error rate
            if thread_rng().random::<f64>() > self.error_rate {
                key_bits.push(alice_bits[i]);
            } else {
                // Flip the bit to simulate an error
                key_bits.push(alice_bits[i] ^ 1);
            }
        }

        // Convert bits to bytes
        let mut key_bytes = Vec::new();
        for chunk in key_bits.chunks(8) {
            let byte = chunk
                .iter()
                .enumerate()
                .fold(0u8, |acc, (i, &bit)| acc | (bit << i));
            key_bytes.push(byte);
        }

        // Store keys
        self.alice.key = Some(key_bytes.clone());
        self.bob.key = Some(key_bytes);

        // Store bases
        self.alice.bases = Some(alice_bases);
        self.bob.bases = Some(bob_bases);

        Ok(matching_bases.len())
    }

    /// Implements the E91 protocol
    fn e91_protocol(&mut self) -> Result<usize> {
        // This is a dummy implementation
        // In a real implementation, this would simulate the E91 protocol
        let key_length = self.num_qubits / 3; // Roughly 1/3 of qubits become key bits

        // Generate random key bytes
        let key_bytes = (0..key_length / 8 + 1)
            .map(|_| thread_rng().random::<u8>())
            .collect::<Vec<_>>();

        // Store keys
        self.alice.key = Some(key_bytes.clone());
        self.bob.key = Some(key_bytes);

        Ok(key_length)
    }

    /// Implements the B92 protocol
    fn b92_protocol(&mut self) -> Result<usize> {
        // This is a dummy implementation
        // In a real implementation, this would simulate the B92 protocol
        let key_length = self.num_qubits / 4; // Roughly 1/4 of qubits become key bits

        // Generate random key bytes
        let key_bytes = (0..key_length / 8 + 1)
            .map(|_| thread_rng().random::<u8>())
            .collect::<Vec<_>>();

        // Store keys
        self.alice.key = Some(key_bytes.clone());
        self.bob.key = Some(key_bytes);

        Ok(key_length)
    }

    /// BBM92 protocol (Bennett-Brassard-Mermin 1992) — entanglement-based QKD.
    ///
    /// Alice and Bob share EPR pairs; each measures independently in a randomly
    /// chosen basis (Z or X). Bases are compared classically; matching positions
    /// yield perfectly anti-correlated raw key bits (Alice flips hers). Retention
    /// rate ≈ 50% (same as BB84) because each measurement in the Z/X basis is
    /// equally likely to match the other party's choice.
    fn bbm92_protocol(&mut self) -> Result<usize> {
        let mut rng = thread_rng();

        // Simulate entangled pair measurements: both choose basis 0 (Z) or 1 (X).
        let alice_bases: Vec<usize> = (0..self.num_qubits)
            .map(|_| if rng.random::<f64>() > 0.5 { 1 } else { 0 })
            .collect();
        let bob_bases: Vec<usize> = (0..self.num_qubits)
            .map(|_| if rng.random::<f64>() > 0.5 { 1 } else { 0 })
            .collect();

        // Alice measures her qubit; bob's result is anti-correlated in matching basis.
        let alice_bits: Vec<u8> = (0..self.num_qubits)
            .map(|_| if rng.random::<f64>() > 0.5 { 1 } else { 0 })
            .collect();

        // Sifting: keep positions where bases agree.
        let sifted_indices: Vec<usize> = (0..self.num_qubits)
            .filter(|&i| alice_bases[i] == bob_bases[i])
            .collect();
        let key_length = sifted_indices.len();

        // Build raw key bytes from sifted bits.
        let key_bytes: Vec<u8> = sifted_indices
            .chunks(8)
            .map(|chunk| {
                chunk.iter().enumerate().fold(0u8, |acc, (bit_pos, &idx)| {
                    acc | (alice_bits[idx] << bit_pos)
                })
            })
            .collect();

        self.alice.key = Some(key_bytes.clone());
        // Bob's key is identical after anti-correlation flip (Alice pre-flips hers).
        self.bob.key = Some(key_bytes);
        Ok(key_length)
    }

    /// SARG04 protocol (Scarani-Acin-Ribordy-Gisin 2004).
    ///
    /// SARG04 is a BB84 variant with modified sifting: Alice announces one of two
    /// non-orthogonal state pairs to reveal her bit, making photon-number-splitting
    /// attacks harder. Retention rate ≈ 25% (half that of BB84) because Bob's
    /// conclusive unambiguous-state-discrimination succeeds only when his measurement
    /// basis matches the natural eigenbasis of the announced pair.
    fn sarg04_protocol(&mut self) -> Result<usize> {
        let mut rng = thread_rng();

        // Alice chooses random bits and random bases.
        let alice_bits: Vec<u8> = (0..self.num_qubits)
            .map(|_| if rng.random::<f64>() > 0.5 { 1 } else { 0 })
            .collect();
        let alice_bases: Vec<usize> = (0..self.num_qubits)
            .map(|_| if rng.random::<f64>() > 0.5 { 1 } else { 0 })
            .collect();

        // Bob measures in a random basis; he succeeds (gets conclusive result)
        // with probability 1/2 (USD strategy on a 2-state ensemble).
        let bob_conclusive: Vec<bool> = (0..self.num_qubits)
            .map(|_| rng.random::<f64>() > 0.5)
            .collect();

        // Only conclusive Bob measurements where bases align produce key bits.
        let bob_bases: Vec<usize> = (0..self.num_qubits)
            .map(|_| if rng.random::<f64>() > 0.5 { 1 } else { 0 })
            .collect();
        let sifted_indices: Vec<usize> = (0..self.num_qubits)
            .filter(|&i| bob_conclusive[i] && alice_bases[i] == bob_bases[i])
            .collect();
        let key_length = sifted_indices.len();

        let key_bytes: Vec<u8> = sifted_indices
            .chunks(8)
            .map(|chunk| {
                chunk.iter().enumerate().fold(0u8, |acc, (bit_pos, &idx)| {
                    acc | (alice_bits[idx] << bit_pos)
                })
            })
            .collect();

        self.alice.key = Some(key_bytes.clone());
        self.bob.key = Some(key_bytes);
        Ok(key_length)
    }

    /// Verifies that Alice and Bob have identical keys
    pub fn verify_keys(&self) -> bool {
        match (&self.alice.key, &self.bob.key) {
            (Some(alice_key), Some(bob_key)) => alice_key == bob_key,
            _ => false,
        }
    }

    /// Gets Alice's key (if generated)
    pub fn get_alice_key(&self) -> Option<Vec<u8>> {
        self.alice.key.clone()
    }

    /// Gets Bob's key (if generated)
    pub fn get_bob_key(&self) -> Option<Vec<u8>> {
        self.bob.key.clone()
    }
}

/// A from-scratch, dependency-free implementation of SHA-256 (FIPS 180-4).
///
/// This workspace's policy is to reuse `scirs2-core` rather than pull in new
/// crates for array/RNG/complex-number needs, and this bundle's fix must not
/// touch `Cargo.toml`; there is no hashing crate already in the dependency
/// graph, so this hand-written implementation provides a real, standard,
/// cryptographically diffusing hash function (verified against the official
/// SHA-256 test vectors in this module's tests) in place of the previous
/// plain byte concatenation.
pub(crate) mod sha256 {
    const ROUND_CONSTANTS: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    const INITIAL_HASH: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    /// Compute the 32-byte SHA-256 digest of `message`.
    pub fn digest(message: &[u8]) -> [u8; 32] {
        let bit_len = (message.len() as u64).wrapping_mul(8);
        let mut padded = message.to_vec();
        padded.push(0x80);
        while padded.len() % 64 != 56 {
            padded.push(0);
        }
        padded.extend_from_slice(&bit_len.to_be_bytes());

        let mut hash_state = INITIAL_HASH;
        for chunk in padded.chunks_exact(64) {
            let mut schedule = [0u32; 64];
            for i in 0..16 {
                schedule[i] = u32::from_be_bytes([
                    chunk[i * 4],
                    chunk[i * 4 + 1],
                    chunk[i * 4 + 2],
                    chunk[i * 4 + 3],
                ]);
            }
            for i in 16..64 {
                let s0 = schedule[i - 15].rotate_right(7)
                    ^ schedule[i - 15].rotate_right(18)
                    ^ (schedule[i - 15] >> 3);
                let s1 = schedule[i - 2].rotate_right(17)
                    ^ schedule[i - 2].rotate_right(19)
                    ^ (schedule[i - 2] >> 10);
                schedule[i] = schedule[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(schedule[i - 7])
                    .wrapping_add(s1);
            }

            let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h) = (
                hash_state[0],
                hash_state[1],
                hash_state[2],
                hash_state[3],
                hash_state[4],
                hash_state[5],
                hash_state[6],
                hash_state[7],
            );

            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let temp1 = h
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(ROUND_CONSTANTS[i])
                    .wrapping_add(schedule[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let temp2 = s0.wrapping_add(maj);

                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }

            hash_state[0] = hash_state[0].wrapping_add(a);
            hash_state[1] = hash_state[1].wrapping_add(b);
            hash_state[2] = hash_state[2].wrapping_add(c);
            hash_state[3] = hash_state[3].wrapping_add(d);
            hash_state[4] = hash_state[4].wrapping_add(e);
            hash_state[5] = hash_state[5].wrapping_add(f);
            hash_state[6] = hash_state[6].wrapping_add(g);
            hash_state[7] = hash_state[7].wrapping_add(h);
        }

        let mut result = [0u8; 32];
        for (i, word) in hash_state.iter().enumerate() {
            result[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        result
    }

    #[cfg(test)]
    mod tests {
        use super::digest;

        fn to_hex(bytes: &[u8]) -> String {
            bytes.iter().map(|b| format!("{b:02x}")).collect()
        }

        #[test]
        fn matches_official_test_vectors() {
            assert_eq!(
                to_hex(&digest(b"")),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            );
            assert_eq!(
                to_hex(&digest(b"abc")),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            );
            assert_eq!(
                to_hex(&digest(
                    b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
                )),
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
            );
        }
    }
}

/// Returns bit `bit_index` (0 = least significant bit of the first byte) of
/// a 32-byte digest.
fn digest_bit(digest: &[u8; 32], bit_index: usize) -> u8 {
    let byte = digest[bit_index / 8];
    (byte >> (bit_index % 8)) & 1
}

/// Number of message-digest bits committed to by a Lamport signature key
/// pair, derived from the caller's requested `security_bits` and capped at
/// 256 (the digest size of the SHA-256 hash used internally).
fn lamport_bit_count(security_bits: usize) -> usize {
    security_bits.clamp(8, 256)
}

/// Public-key-only half of a [`QuantumSignature`].
///
/// A legitimate verifier only ever needs -- and only ever has -- the public
/// key, never the private key. Splitting this out as its own type (rather
/// than verifying via a `QuantumSignature` that also stores the private key)
/// makes that structurally explicit.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantumSignatureVerifyingKey {
    bit_count: usize,
    public_key: Vec<[u8; 32]>,
}

impl QuantumSignatureVerifyingKey {
    /// Verifies `signature` against `message` using only this public key.
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> Result<bool> {
        QuantumSignature::verify_with_public_key(
            message,
            signature,
            &self.public_key,
            self.bit_count,
        )
    }

    /// Serializes this verifying key to bytes (a big-endian bit-count prefix
    /// followed by each 32-byte public-key entry), suitable for embedding in,
    /// e.g., a blockchain transaction's `sender` field.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(8 + self.public_key.len() * 32);
        bytes.extend_from_slice(&(self.bit_count as u64).to_be_bytes());
        for entry in &self.public_key {
            bytes.extend_from_slice(entry);
        }
        bytes
    }

    /// Deserializes a verifying key previously produced by [`Self::to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(MLError::InvalidParameter(
                "Verifying key bytes too short".to_string(),
            ));
        }
        let bit_count_bytes: [u8; 8] = bytes[0..8]
            .try_into()
            .map_err(|_| MLError::InvalidParameter("Malformed bit-count prefix".to_string()))?;
        let bit_count = u64::from_be_bytes(bit_count_bytes) as usize;
        let expected_len = 8 + bit_count * 2 * 32;
        if bytes.len() != expected_len {
            return Err(MLError::InvalidParameter(format!(
                "Verifying key length mismatch: expected {expected_len} bytes, got {}",
                bytes.len()
            )));
        }
        let public_key = bytes[8..]
            .chunks_exact(32)
            .map(|chunk| {
                let mut entry = [0u8; 32];
                entry.copy_from_slice(chunk);
                entry
            })
            .collect();
        Ok(Self {
            bit_count,
            public_key,
        })
    }
}

/// Quantum-safe one-time digital signature key pair.
///
/// Implements a Lamport one-time signature (Lamport, 1979): a hash-based
/// scheme whose security rests only on the one-wayness of a hash function,
/// not on factoring/discrete-log assumptions that Shor's algorithm breaks --
/// making it "quantum-safe" in the sense this module's name advertises,
/// unlike the previous XOR-based placeholder. Verification (see
/// [`QuantumSignatureVerifyingKey::verify`]) uses *only* the public key,
/// structurally correcting the previous bug where `verify` could only be
/// called by whoever held the private key.
///
/// **This is genuinely a *one-time* signature scheme**: signing two
/// different messages with the same key pair reveals enough of the private
/// key to forge further signatures, exactly as with real Lamport signatures.
/// A new [`QuantumSignature::new`] key pair should be generated per message.
#[derive(Debug, Clone)]
pub struct QuantumSignature {
    /// Number of committed message-digest bits.
    bit_count: usize,

    /// Signature algorithm label.
    algorithm: String,

    /// Public key: `public_key[2*i]`/`public_key[2*i+1]` are the hashes of
    /// the "digest bit `i` is 0" / "digest bit `i` is 1" private-key secrets.
    public_key: Vec<[u8; 32]>,

    /// Private key: `private_key[2*i]`/`private_key[2*i+1]` are the two
    /// preimages for digest bit `i`.
    private_key: Vec<[u8; 32]>,
}

impl QuantumSignature {
    /// Creates a new quantum signature key pair.
    pub fn new(security_bits: usize, algorithm: &str) -> Result<Self> {
        let bit_count = lamport_bit_count(security_bits);
        let mut rng = thread_rng();
        let private_key: Vec<[u8; 32]> = (0..bit_count * 2)
            .map(|_| {
                let mut secret = [0u8; 32];
                for byte in secret.iter_mut() {
                    *byte = rng.random::<u8>();
                }
                secret
            })
            .collect();
        let public_key: Vec<[u8; 32]> = private_key
            .iter()
            .map(|secret| sha256::digest(secret))
            .collect();

        Ok(QuantumSignature {
            bit_count,
            algorithm: algorithm.to_string(),
            public_key,
            private_key,
        })
    }

    /// Signs a message: hashes it with SHA-256, then reveals one of the two
    /// private-key preimages per digest bit (selected by that bit's value).
    pub fn sign(&self, message: &[u8]) -> Result<Vec<u8>> {
        let message_digest = sha256::digest(message);
        let mut signature = Vec::with_capacity(self.bit_count * 32);
        for bit_index in 0..self.bit_count {
            let bit = digest_bit(&message_digest, bit_index);
            let secret = &self.private_key[2 * bit_index + bit as usize];
            signature.extend_from_slice(secret);
        }
        Ok(signature)
    }

    /// Verifies a signature using this key pair's public key. Structurally
    /// identical to [`QuantumSignatureVerifyingKey::verify`] -- exposed here
    /// too so existing callers that only have a full `QuantumSignature`
    /// (e.g. the signer itself, checking its own work) do not need to call
    /// [`Self::verifying_key`] first.
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> Result<bool> {
        Self::verify_with_public_key(message, signature, &self.public_key, self.bit_count)
    }

    /// Extracts a standalone [`QuantumSignatureVerifyingKey`] containing only
    /// the public key, for distribution to verifiers.
    pub fn verifying_key(&self) -> QuantumSignatureVerifyingKey {
        QuantumSignatureVerifyingKey {
            bit_count: self.bit_count,
            public_key: self.public_key.clone(),
        }
    }

    /// Serializes this key pair's public key; see
    /// [`QuantumSignatureVerifyingKey::to_bytes`].
    pub fn public_key_bytes(&self) -> Vec<u8> {
        self.verifying_key().to_bytes()
    }

    fn verify_with_public_key(
        message: &[u8],
        signature: &[u8],
        public_key: &[[u8; 32]],
        bit_count: usize,
    ) -> Result<bool> {
        if signature.len() != bit_count * 32 || public_key.len() != bit_count * 2 {
            return Ok(false);
        }
        let message_digest = sha256::digest(message);
        for bit_index in 0..bit_count {
            let bit = digest_bit(&message_digest, bit_index);
            let revealed_preimage = &signature[bit_index * 32..(bit_index + 1) * 32];
            let expected_public_entry = public_key[2 * bit_index + bit as usize];
            if sha256::digest(revealed_preimage) != expected_public_entry {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Quantum authentication
#[derive(Debug, Clone)]
pub struct QuantumAuthentication {
    /// Protocol type
    protocol: String,

    /// Security parameter
    security_bits: usize,

    /// Authentication keys
    keys: HashMap<String, Vec<u8>>,
}

impl QuantumAuthentication {
    /// Creates a new quantum authentication protocol
    pub fn new(protocol: &str, security_bits: usize) -> Self {
        QuantumAuthentication {
            protocol: protocol.to_string(),
            security_bits,
            keys: HashMap::new(),
        }
    }

    /// Adds a party to the authentication system
    pub fn add_party(&mut self, party_name: &str) -> Result<()> {
        // Generate a random key
        let key = (0..self.security_bits / 8 + 1)
            .map(|_| thread_rng().random::<u8>())
            .collect::<Vec<_>>();

        self.keys.insert(party_name.to_string(), key);

        Ok(())
    }

    /// Authenticates a message from a party
    pub fn authenticate(&self, party_name: &str, message: &[u8]) -> Result<Vec<u8>> {
        // Get the party's key
        let key = self
            .keys
            .get(party_name)
            .ok_or_else(|| MLError::InvalidParameter(format!("Party {} not found", party_name)))?;

        // Generate a random authentication tag
        let mut tag = key.clone();

        // XOR with the message (simplified)
        for (i, &byte) in message.iter().enumerate() {
            if i < tag.len() {
                tag[i] ^= byte;
            }
        }

        Ok(tag)
    }

    /// Verifies an authentication tag
    pub fn verify(&self, party_name: &str, message: &[u8], tag: &[u8]) -> Result<bool> {
        // Generate the expected tag
        let expected_tag = self.authenticate(party_name, message)?;

        // Compare tags
        let is_valid = tag.len() == expected_tag.len()
            && tag.iter().zip(expected_tag.iter()).all(|(a, b)| a == b);

        Ok(is_valid)
    }
}

/// Quantum Secure Direct Communication protocol
#[derive(Debug, Clone)]
pub struct QSDC {
    /// Number of qubits to use
    pub num_qubits: usize,

    /// Error rate for the quantum channel
    pub error_rate: f64,
}

impl QSDC {
    /// Creates a new QSDC protocol instance
    pub fn new(num_qubits: usize) -> Self {
        QSDC {
            num_qubits,
            error_rate: 0.01, // Default 1% error rate
        }
    }

    /// Sets the error rate for the quantum channel
    pub fn with_error_rate(mut self, error_rate: f64) -> Self {
        self.error_rate = error_rate;
        self
    }

    /// Transmits a message directly using the quantum channel
    pub fn transmit_message(&self, message: &[u8]) -> Result<Vec<u8>> {
        // This is a dummy implementation
        // In a real implementation, this would use quantum entanglement
        // to directly transmit the message

        // Create a copy of the message
        let mut received = message.to_vec();

        // Apply the error rate to simulate channel noise
        for byte in &mut received {
            for bit_pos in 0..8 {
                if thread_rng().random::<f64>() < self.error_rate {
                    // Flip the bit
                    *byte ^= 1 << bit_pos;
                }
            }
        }

        Ok(received)
    }
}

/// Encrypts a message using a quantum key
pub fn encrypt_with_qkd(message: &[u8], key: Vec<u8>) -> Vec<u8> {
    // Simple XOR encryption
    message
        .iter()
        .enumerate()
        .map(|(i, &byte)| byte ^ key[i % key.len()])
        .collect()
}

/// Decrypts a message using a quantum key
pub fn decrypt_with_qkd(encrypted: &[u8], key: Vec<u8>) -> Vec<u8> {
    // XOR is symmetric, so encryption and decryption are the same
    encrypt_with_qkd(encrypted, key)
}

impl fmt::Display for ProtocolType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProtocolType::BB84 => write!(f, "BB84"),
            ProtocolType::E91 => write!(f, "E91"),
            ProtocolType::B92 => write!(f, "B92"),
            ProtocolType::BBM92 => write!(f, "BBM92"),
            ProtocolType::SARG04 => write!(f, "SARG04"),
        }
    }
}

#[cfg(test)]
mod signature_regression_tests {
    use super::*;

    /// Regression test for the "verify() requires the private key" bug: a
    /// verifier holding *only* the public verifying key (never the private
    /// key) must be able to check a signature produced by the signer.
    #[test]
    fn verify_succeeds_with_only_the_public_verifying_key() {
        let signer = QuantumSignature::new(64, "lamport-test").expect("key generation");
        let message = b"transfer 10 QBTC to bob";
        let signature = signer.sign(message).expect("signing should succeed");

        // The verifier only ever sees this -- it structurally cannot access
        // `signer.private_key` (a private field of a different value it
        // never receives).
        let verifying_key = signer.verifying_key();
        assert!(verifying_key
            .verify(message, &signature)
            .expect("verification should succeed"));

        // Round-trip through serialization, as a transaction's `sender`
        // field would carry it.
        let bytes = verifying_key.to_bytes();
        let restored = QuantumSignatureVerifyingKey::from_bytes(&bytes).expect("deserialize");
        assert!(restored
            .verify(message, &signature)
            .expect("verification should succeed after round-trip"));
    }

    #[test]
    fn verify_rejects_tampered_message_or_signature() {
        let signer = QuantumSignature::new(64, "lamport-test").expect("key generation");
        let message = b"transfer 10 QBTC to bob";
        let signature = signer.sign(message).expect("signing should succeed");
        let verifying_key = signer.verifying_key();

        let tampered_message = b"transfer 99 QBTC to mallory";
        assert!(!verifying_key
            .verify(tampered_message, &signature)
            .expect("verification should not error"));

        let mut tampered_signature = signature.clone();
        tampered_signature[0] ^= 0xFF;
        assert!(!verifying_key
            .verify(message, &tampered_signature)
            .expect("verification should not error"));

        // A signature produced by a *different* key pair must not verify
        // against this signer's public key.
        let other_signer = QuantumSignature::new(64, "lamport-test").expect("key generation");
        let other_signature = other_signer.sign(message).expect("signing should succeed");
        assert!(!verifying_key
            .verify(message, &other_signature)
            .expect("verification should not error"));
    }
}
