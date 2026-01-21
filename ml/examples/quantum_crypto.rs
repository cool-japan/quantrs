#![allow(
    clippy::pedantic,
    clippy::unnecessary_wraps,
    clippy::needless_range_loop,
    clippy::useless_vec,
    clippy::needless_collect,
    clippy::too_many_arguments
)]
use quantrs2_ml::crypto::{ProtocolType, QuantumKeyDistribution, QuantumSignature, QSDC};
use quantrs2_ml::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("Quantum Cryptography Examples");
    println!("============================");

    // BB84 Quantum Key Distribution
    run_bb84_example()?;

    // E91 Protocol
    run_e91_example()?;

    // Quantum Digital Signatures
    run_signature_example()?;

    // Quantum Secure Direct Communication
    run_qsdc_example()?;

    // Quantum Blockchain Example
    run_blockchain_example()?;

    Ok(())
}

fn run_bb84_example() -> Result<()> {
    println!("\nBB84 Quantum Key Distribution");
    println!("----------------------------");

    // Create BB84 QKD with 1000 qubits
    let num_qubits = 1000;
    println!("Creating BB84 protocol with {num_qubits} qubits");
    let mut qkd = QuantumKeyDistribution::new(ProtocolType::BB84, num_qubits);

    // Optional: set error rate
    qkd = qkd.with_error_rate(0.03);
    println!("Simulated error rate: {:.1}%", qkd.error_rate * 100.0);

    // Distribute key
    println!("Performing quantum key distribution...");
    let start = Instant::now();
    let key_length = qkd.distribute_key()?;
    println!("Key distribution completed in {:.2?}", start.elapsed());
    println!("Final key length: {key_length} bits");

    // Verify keys match
    println!("Verifying Alice and Bob have identical keys...");
    if qkd.verify_keys() {
        println!("✓ Key verification successful!");

        // Display part of the key (first 8 bytes)
        if let Some(key) = qkd.get_alice_key() {
            println!("First 8 bytes of key: {:?}", &key[0..8.min(key.len())]);
        }
    } else {
        println!("✗ Key verification failed!");
    }

    // Use the key for encryption
    if let Some(key) = qkd.get_alice_key() {
        let message = b"Hello, quantum world!";

        println!(
            "Encrypting message: '{}'",
            std::str::from_utf8(message).unwrap()
        );
        let encrypted = quantrs2_ml::crypto::encrypt_with_qkd(message, key);

        println!("Encrypted data: {:?}", &encrypted);

        // Decrypt with Bob's key
        if let Some(bob_key) = qkd.get_bob_key() {
            let decrypted = quantrs2_ml::crypto::decrypt_with_qkd(&encrypted, bob_key);
            println!(
                "Decrypted message: '{}'",
                std::str::from_utf8(&decrypted).unwrap()
            );
        }
    }

    Ok(())
}

fn run_e91_example() -> Result<()> {
    println!("\nE91 Entanglement-Based Protocol");
    println!("------------------------------");

    // Create E91 QKD with 800 qubits
    let num_qubits = 800;
    println!("Creating E91 protocol with {num_qubits} qubits");
    let mut qkd = QuantumKeyDistribution::new(ProtocolType::E91, num_qubits);

    // Set error rate
    qkd = qkd.with_error_rate(0.02);
    println!("Simulated error rate: {:.1}%", qkd.error_rate * 100.0);

    // Distribute key
    println!("Performing quantum key distribution with entangled pairs...");
    let start = Instant::now();
    let key_length = qkd.distribute_key()?;
    println!("Key distribution completed in {:.2?}", start.elapsed());
    println!("Final key length: {key_length} bits");

    // Verify keys match
    println!("Verifying Alice and Bob have identical keys...");
    if qkd.verify_keys() {
        println!("✓ Key verification successful!");

        // Display part of the key
        if let Some(key) = qkd.get_alice_key() {
            println!("First 8 bytes of key: {:?}", &key[0..8.min(key.len())]);
        }
    } else {
        println!("✗ Key verification failed!");
    }

    Ok(())
}

fn run_signature_example() -> Result<()> {
    println!("\nQuantum Digital Signatures");
    println!("-------------------------");

    // Create quantum signature with 256 qubits
    let num_qubits = 256;
    println!("Creating quantum signature scheme with {num_qubits} qubits");

    // Choose a quantum-resistant algorithm
    let algorithm = "Dilithium";
    println!("Using algorithm: {algorithm}");

    let signature = QuantumSignature::new(num_qubits, algorithm)?;
    println!("Quantum signature scheme created");

    // Sign a message
    let message = b"This message is quantum-signed";
    println!(
        "Signing message: '{}'",
        std::str::from_utf8(message).unwrap()
    );

    let start = Instant::now();
    let sig = signature.sign(message)?;
    println!("Signature generated in {:.2?}", start.elapsed());
    println!("Signature size: {} bytes", sig.len());

    // Verify signature
    println!("Verifying signature...");
    let start = Instant::now();
    let is_valid = signature.verify(message, &sig)?;
    println!("Verification completed in {:.2?}", start.elapsed());

    if is_valid {
        println!("✓ Signature verification successful!");
    } else {
        println!("✗ Signature verification failed!");
    }

    // Try with tampered message
    let tampered = b"This message has been modified";
    println!("Verifying signature with tampered message...");
    let is_valid = signature.verify(tampered, &sig)?;

    if is_valid {
        println!("✗ Signature incorrectly verified on tampered message!");
    } else {
        println!("✓ Signature correctly rejected tampered message!");
    }

    Ok(())
}

fn run_qsdc_example() -> Result<()> {
    println!("\nQuantum Secure Direct Communication");
    println!("---------------------------------");

    // Create QSDC protocol with 1000 qubits
    let num_qubits = 1000;
    println!("Creating QSDC protocol with {num_qubits} qubits");
    let qsdc = QSDC::new(num_qubits);

    // Transmit message directly
    let message = b"This message is sent directly using quantum channel";
    println!(
        "Message to transmit: '{}'",
        std::str::from_utf8(message).unwrap()
    );

    let start = Instant::now();
    let received = qsdc.transmit_message(message)?;
    println!("Transmission completed in {:.2?}", start.elapsed());

    println!(
        "Received message: '{}'",
        std::str::from_utf8(&received).unwrap()
    );

    // Check for errors
    let errors = message
        .iter()
        .zip(received.iter())
        .filter(|(&a, &b)| a != b)
        .count();

    println!(
        "Bit error rate: {:.2}%",
        (errors as f64) / (message.len() as f64) * 100.0
    );

    Ok(())
}

fn run_blockchain_example() -> Result<()> {
    println!("\nQuantum Blockchain Example");
    println!("-------------------------");

    use quantrs2_ml::blockchain::{ConsensusType, QuantumBlockchain, Transaction};

    // Create a quantum blockchain
    let difficulty = 2; // 2 leading zeros required for mining
    println!("Creating quantum blockchain with difficulty {difficulty}");
    let mut blockchain = QuantumBlockchain::new(ConsensusType::QuantumProofOfWork, difficulty);

    // Create a transaction
    let sender = vec![1, 2, 3, 4];
    let recipient = vec![5, 6, 7, 8];
    let amount = 100.0;

    println!(
        "Creating transaction: {} sends {} units to recipient",
        sender.iter().map(|&b| b.to_string()).collect::<String>(),
        amount
    );

    let transaction = Transaction::new(sender, recipient, amount, Vec::new());

    // Add transaction
    println!("Adding transaction to blockchain...");
    blockchain.add_transaction(transaction)?;

    // Mine a block
    println!("Mining new block...");
    let start = Instant::now();
    let block = blockchain.mine_block()?;
    println!("Block mined in {:.2?}", start.elapsed());

    println!(
        "Block hash: {:02x?}",
        &block.hash[0..8.min(block.hash.len())]
    );
    println!("Blockchain length: {}", blockchain.chain.len());

    // Verify blockchain
    println!("Verifying blockchain integrity...");
    let is_valid = blockchain.verify_chain()?;

    if is_valid {
        println!("✓ Blockchain verification successful!");
    } else {
        println!("✗ Blockchain verification failed!");
    }

    Ok(())
}
