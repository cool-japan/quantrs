//! Quantum-secured blockchain primitives.
//!
//! Implements quantum-safe consensus mechanisms, quantum-signed transactions,
//! and a simulated quantum blockchain ledger using post-quantum and
//! QKD-based cryptographic primitives from the `crypto` module.

use crate::crypto::{sha256, QuantumSignatureVerifyingKey};
use crate::error::{MLError, Result};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Type of consensus algorithm for quantum blockchains
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsensusType {
    /// Quantum-secured Proof of Work
    QuantumProofOfWork,

    /// Quantum-secured Proof of Stake
    QuantumProofOfStake,

    /// Quantum Byzantine Agreement
    QuantumByzantineAgreement,

    /// Quantum Federated Consensus
    QuantumFederated,
}

/// Represents a transaction in a quantum blockchain
#[derive(Debug, Clone)]
pub struct Transaction {
    /// Sender's serialized [`QuantumSignatureVerifyingKey`]
    /// (see [`QuantumSignatureVerifyingKey::to_bytes`]), used to verify
    /// `signature` against this transaction's signing message. Not merely a
    /// hash of the key: signature verification needs the actual public key.
    pub sender: Vec<u8>,

    /// Recipient's public key bytes
    pub recipient: Vec<u8>,

    /// Amount to transfer
    pub amount: f64,

    /// Additional data (can be used for smart contracts)
    pub data: Vec<u8>,

    /// Transaction timestamp
    timestamp: u64,

    /// Transaction signature (a Lamport one-time signature produced by
    /// [`crate::crypto::QuantumSignature::sign`] over `signing_message()`)
    signature: Option<Vec<u8>>,
}

impl Transaction {
    /// Creates a new transaction
    pub fn new(sender: Vec<u8>, recipient: Vec<u8>, amount: f64, data: Vec<u8>) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_secs();

        Transaction {
            sender,
            recipient,
            amount,
            data,
            timestamp,
            signature: None,
        }
    }

    /// Signs the transaction by attaching a pre-computed signature (produced
    /// by, e.g., `QuantumSignature::sign(&transaction.signing_message())`).
    pub fn sign(&mut self, signature: Vec<u8>) -> Result<()> {
        self.signature = Some(signature);
        Ok(())
    }

    /// The canonical byte sequence this transaction's signature is computed
    /// (and verified) over: every field except the signature itself.
    fn signing_message(&self) -> Vec<u8> {
        let mut message = Vec::new();
        message.extend_from_slice(&self.sender);
        message.extend_from_slice(&self.recipient);
        message.extend_from_slice(&self.amount.to_be_bytes());
        message.extend_from_slice(&self.timestamp.to_be_bytes());
        message.extend_from_slice(&self.data);
        message
    }

    /// Verifies the transaction signature against `sender`'s public key.
    ///
    /// Returns `Ok(false)` (rather than an error) both when there is no
    /// signature to check and when `sender` cannot be parsed as a
    /// [`QuantumSignatureVerifyingKey`] -- either way the transaction is not
    /// validly signed.
    pub fn verify(&self) -> Result<bool> {
        let signature = match &self.signature {
            Some(signature) => signature,
            None => return Ok(false),
        };
        let verifying_key = match QuantumSignatureVerifyingKey::from_bytes(&self.sender) {
            Ok(key) => key,
            Err(_) => return Ok(false),
        };
        verifying_key.verify(&self.signing_message(), signature)
    }

    /// Gets the transaction hash: the real SHA-256 digest (see
    /// [`crate::crypto::sha256`]) of every transaction field, replacing the
    /// previous plain byte concatenation (which had no cryptographic
    /// diffusion and was trivially invertible/collidable).
    pub fn hash(&self) -> Vec<u8> {
        let mut preimage = self.signing_message();
        // The signing message intentionally excludes the signature (it is
        // what gets signed), but the transaction *hash* should still be
        // sensitive to it so that a re-signed/re-attached signature changes
        // the hash.
        if let Some(signature) = &self.signature {
            preimage.extend_from_slice(signature);
        }
        sha256::digest(&preimage).to_vec()
    }
}

/// Represents a block in a quantum blockchain
#[derive(Debug, Clone)]
pub struct Block {
    /// Block index
    pub index: usize,

    /// Previous block hash
    pub previous_hash: Vec<u8>,

    /// Block timestamp
    pub timestamp: u64,

    /// Transactions in the block
    pub transactions: Vec<Transaction>,

    /// Nonce for proof of work
    pub nonce: u64,

    /// Block hash
    pub hash: Vec<u8>,
}

impl Block {
    /// Creates a new block
    pub fn new(index: usize, previous_hash: Vec<u8>, transactions: Vec<Transaction>) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_secs();

        let mut block = Block {
            index,
            previous_hash,
            timestamp,
            transactions,
            nonce: 0,
            hash: Vec::new(),
        };

        block.hash = block.calculate_hash();

        block
    }

    /// Calculates the block hash: the real SHA-256 digest (see
    /// [`crate::crypto::sha256`]) of the block's index, previous hash,
    /// timestamp, transaction hashes, and nonce, replacing the previous
    /// plain byte concatenation (which had no cryptographic diffusion, so
    /// mining by nonce search or tampering with a transaction was not
    /// meaningfully harder than editing the "hash" bytes directly).
    pub fn calculate_hash(&self) -> Vec<u8> {
        let mut preimage = Vec::new();

        preimage.extend_from_slice(&(self.index as u64).to_be_bytes());
        preimage.extend_from_slice(&self.previous_hash);
        preimage.extend_from_slice(&self.timestamp.to_be_bytes());

        for transaction in &self.transactions {
            preimage.extend_from_slice(&transaction.hash());
        }

        preimage.extend_from_slice(&self.nonce.to_be_bytes());

        sha256::digest(&preimage).to_vec()
    }

    /// Mines the block with proof of work
    pub fn mine(&mut self, difficulty: usize) -> Result<()> {
        // `hash` is now always a fixed 32-byte SHA-256 digest (previously it
        // was a variable-length, unhashed byte concatenation whose length
        // happened to grow with the number of fields/transactions); clamp
        // the number of leading zero bytes we demand so an overly large
        // `difficulty` cannot slice past the end of the digest.
        let target_len = (difficulty / 8 + 1).min(self.hash.len());
        let target = vec![0u8; target_len];

        while self.hash[0..target_len] != target[..] {
            self.nonce += 1;
            self.hash = self.calculate_hash();

            // Optional: add a check to prevent infinite loops
            if self.nonce > 1_000_000 {
                return Err(MLError::MLOperationError(
                    "Mining took too long. Consider reducing difficulty.".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Verifies the block
    pub fn verify(&self, previous_hash: &[u8]) -> Result<bool> {
        // This is a dummy implementation
        // In a real system, this would verify the block

        if self.previous_hash != previous_hash {
            return Ok(false);
        }

        let calculated_hash = self.calculate_hash();
        if self.hash != calculated_hash {
            return Ok(false);
        }

        for transaction in &self.transactions {
            if !transaction.verify()? {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

/// Smart contract for quantum blockchains
#[derive(Debug, Clone)]
pub struct SmartContract {
    /// Contract bytecode
    pub bytecode: Vec<u8>,

    /// Contract owner
    pub owner: Vec<u8>,

    /// Contract state
    pub state: HashMap<Vec<u8>, Vec<u8>>,
}

impl SmartContract {
    /// Creates a new smart contract
    pub fn new(bytecode: Vec<u8>, owner: Vec<u8>) -> Self {
        SmartContract {
            bytecode,
            owner,
            state: HashMap::new(),
        }
    }

    /// Executes the contract
    pub fn execute(&mut self, input: &[u8]) -> Result<Vec<u8>> {
        // This is a dummy implementation
        // In a real system, this would execute the contract bytecode

        if input.is_empty() {
            return Err(MLError::InvalidParameter("Input is empty".to_string()));
        }

        let operation = input[0];

        match operation {
            0 => {
                // Store operation
                if input.len() < 3 {
                    return Err(MLError::InvalidParameter("Invalid store input".to_string()));
                }

                let key = vec![input[1]];
                let value = vec![input[2]];

                self.state.insert(key, value.clone());

                Ok(value)
            }
            1 => {
                // Load operation
                if input.len() < 2 {
                    return Err(MLError::InvalidParameter("Invalid load input".to_string()));
                }

                let key = vec![input[1]];

                let value = self.state.get(&key).ok_or_else(|| {
                    MLError::MLOperationError(format!("Key not found: {:?}", key))
                })?;

                Ok(value.clone())
            }
            _ => Err(MLError::InvalidParameter(format!(
                "Invalid operation: {}",
                operation
            ))),
        }
    }
}

/// Quantum token for digital assets
#[derive(Debug, Clone)]
pub struct QuantumToken {
    /// Token name
    pub name: String,

    /// Token symbol
    pub symbol: String,

    /// Total supply
    pub total_supply: u64,

    /// Balances for addresses
    pub balances: HashMap<Vec<u8>, u64>,
}

impl QuantumToken {
    /// Creates a new quantum token
    pub fn new(name: &str, symbol: &str, total_supply: u64, owner: Vec<u8>) -> Self {
        let mut balances = HashMap::new();
        balances.insert(owner, total_supply);

        QuantumToken {
            name: name.to_string(),
            symbol: symbol.to_string(),
            total_supply,
            balances,
        }
    }

    /// Transfers tokens from one address to another
    pub fn transfer(&mut self, from: &[u8], to: &[u8], amount: u64) -> Result<()> {
        // Get the from balance first and copy it
        let from_balance = *self.balances.get(from).ok_or_else(|| {
            MLError::MLOperationError(format!("From address not found: {:?}", from))
        })?;

        if from_balance < amount {
            return Err(MLError::MLOperationError(format!(
                "Insufficient balance: {} < {}",
                from_balance, amount
            )));
        }

        // Update from balance
        self.balances.insert(from.to_vec(), from_balance - amount);

        // Update to balance
        let to_balance = self.balances.entry(to.to_vec()).or_insert(0);
        *to_balance += amount;

        Ok(())
    }

    /// Gets the balance for an address
    pub fn balance_of(&self, address: &[u8]) -> u64 {
        self.balances.get(address).cloned().unwrap_or(0)
    }
}

/// Quantum blockchain with distributed ledger
#[derive(Debug, Clone)]
pub struct QuantumBlockchain {
    /// Chain of blocks
    pub chain: Vec<Block>,

    /// Pending transactions
    pub pending_transactions: Vec<Transaction>,

    /// Mining difficulty
    pub difficulty: usize,

    /// Consensus algorithm
    pub consensus: ConsensusType,

    /// Network nodes
    pub nodes: Vec<String>,
}

impl QuantumBlockchain {
    /// Creates a new quantum blockchain
    pub fn new(consensus: ConsensusType, difficulty: usize) -> Self {
        // Create genesis block
        let genesis_block = Block::new(0, vec![0u8; 32], Vec::new());

        QuantumBlockchain {
            chain: vec![genesis_block],
            pending_transactions: Vec::new(),
            difficulty,
            consensus,
            nodes: Vec::new(),
        }
    }

    /// Adds a transaction to the pending transactions
    pub fn add_transaction(&mut self, transaction: Transaction) -> Result<()> {
        // Verify transaction
        if !transaction.verify()? {
            return Err(MLError::MLOperationError(
                "Transaction verification failed".to_string(),
            ));
        }

        self.pending_transactions.push(transaction);

        Ok(())
    }

    /// Mines a new block
    pub fn mine_block(&mut self) -> Result<Block> {
        if self.pending_transactions.is_empty() {
            return Err(MLError::MLOperationError(
                "No pending transactions to mine".to_string(),
            ));
        }

        let transactions = self.pending_transactions.clone();
        self.pending_transactions.clear();

        let previous_block = self
            .chain
            .last()
            .ok_or_else(|| MLError::MLOperationError("Blockchain is empty".to_string()))?;

        let mut block = Block::new(self.chain.len(), previous_block.hash.clone(), transactions);

        // Mine the block based on consensus algorithm
        match self.consensus {
            ConsensusType::QuantumProofOfWork => {
                block.mine(self.difficulty)?;
            }
            _ => {
                // Other consensus algorithms (simplified for example)
                block.hash = block.calculate_hash();
            }
        }

        self.chain.push(block.clone());

        Ok(block)
    }

    /// Verifies the blockchain
    pub fn verify(&self) -> Result<bool> {
        for i in 1..self.chain.len() {
            let current_block = &self.chain[i];
            let previous_block = &self.chain[i - 1];

            if !current_block.verify(&previous_block.hash)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Alias for verify() - to match the example call
    pub fn verify_chain(&self) -> Result<bool> {
        self.verify()
    }

    /// Gets a blockchain with a tampered block for testing
    pub fn tamper_with_block(
        &self,
        block_index: usize,
        sender: &str,
        amount: f64,
    ) -> Result<QuantumBlockchain> {
        if block_index >= self.chain.len() {
            return Err(MLError::MLOperationError(format!(
                "Block index out of range: {}",
                block_index
            )));
        }

        let mut tampered = self.clone();

        // Create a tampered transaction
        let tampered_transaction = Transaction::new(
            sender.as_bytes().to_vec(),
            vec![1, 2, 3, 4],
            amount,
            Vec::new(),
        );

        // Replace the first transaction in the block
        if !tampered.chain[block_index].transactions.is_empty() {
            tampered.chain[block_index].transactions[0] = tampered_transaction;
        } else {
            tampered.chain[block_index]
                .transactions
                .push(tampered_transaction);
        }

        // Recalculate the hash (but don't fix it)
        let hash = tampered.chain[block_index].calculate_hash();
        tampered.chain[block_index].hash = hash;

        Ok(tampered)
    }
}

impl fmt::Display for ConsensusType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConsensusType::QuantumProofOfWork => write!(f, "Quantum Proof of Work"),
            ConsensusType::QuantumProofOfStake => write!(f, "Quantum Proof of Stake"),
            ConsensusType::QuantumByzantineAgreement => write!(f, "Quantum Byzantine Agreement"),
            ConsensusType::QuantumFederated => write!(f, "Quantum Federated Consensus"),
        }
    }
}

#[cfg(test)]
mod integrity_regression_tests {
    use super::*;
    use crate::crypto::QuantumSignature;

    fn signed_transaction(
        signer: &QuantumSignature,
        recipient: Vec<u8>,
        amount: f64,
    ) -> Transaction {
        let mut tx = Transaction::new(signer.public_key_bytes(), recipient, amount, Vec::new());
        let signature = signer.sign(&tx.signing_message()).expect("sign");
        tx.sign(signature).expect("attach signature");
        tx
    }

    /// Regression test for the "verify() only checks Option::is_some()" bug:
    /// a properly signed transaction must verify true, and tampering with
    /// any signed field afterward must make it verify false.
    #[test]
    fn transaction_verify_checks_the_real_signature() {
        let signer = QuantumSignature::new(64, "lamport-test").expect("key generation");
        let mut tx = signed_transaction(&signer, vec![9, 9, 9], 42.0);
        assert!(tx.verify().expect("verify should not error"));

        // Tampering with the amount after signing must invalidate it.
        tx.amount = 999.0;
        assert!(!tx.verify().expect("verify should not error"));
    }

    #[test]
    fn transaction_without_signature_never_verifies() {
        let tx = Transaction::new(vec![1, 2, 3], vec![4, 5, 6], 1.0, Vec::new());
        assert!(!tx.verify().expect("verify should not error"));
    }

    /// Regression test for the "hash is plain byte concatenation" bug: the
    /// transaction/block hash must behave like a real cryptographic hash --
    /// fixed-size and radically different for even a tiny input change
    /// (avalanche effect) -- rather than an easily-inspected concatenation.
    #[test]
    fn transaction_hash_is_fixed_size_and_diffuses_small_changes() {
        let signer = QuantumSignature::new(64, "lamport-test").expect("key generation");
        let tx_a = signed_transaction(&signer, vec![9, 9, 9], 42.0);
        let mut tx_b = tx_a.clone();
        tx_b.amount = 42.000001;

        let hash_a = tx_a.hash();
        let hash_b = tx_b.hash();
        assert_eq!(hash_a.len(), 32, "expected a 32-byte SHA-256 digest");
        assert_eq!(hash_b.len(), 32, "expected a 32-byte SHA-256 digest");
        assert_ne!(hash_a, hash_b);

        // Avalanche effect: roughly half the bits should differ for a
        // one-bit-ish change in the input, not just a few trailing bytes as
        // plain concatenation would produce.
        let differing_bits: u32 = hash_a
            .iter()
            .zip(hash_b.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        assert!(
            differing_bits > 32,
            "expected substantial bit diffusion, got {differing_bits} differing bits"
        );
    }

    /// End-to-end regression test: mining and verifying a small chain with
    /// genuinely signed transactions.
    #[test]
    fn mined_chain_with_signed_transactions_verifies() {
        let signer = QuantumSignature::new(48, "lamport-test").expect("key generation");
        let mut blockchain = QuantumBlockchain::new(ConsensusType::QuantumProofOfWork, 8);

        blockchain
            .add_transaction(signed_transaction(&signer, vec![1, 2, 3], 5.0))
            .expect("adding a validly signed transaction should succeed");
        blockchain.mine_block().expect("mining should succeed");

        assert!(blockchain.verify().expect("chain should verify"));
    }

    /// Adding a transaction with an invalid/missing signature must be
    /// rejected up front, not silently accepted.
    #[test]
    fn add_transaction_rejects_unsigned_transaction() {
        let mut blockchain = QuantumBlockchain::new(ConsensusType::QuantumProofOfWork, 8);
        let unsigned = Transaction::new(vec![1, 2, 3], vec![4, 5, 6], 1.0, Vec::new());
        assert!(blockchain.add_transaction(unsigned).is_err());
    }
}
