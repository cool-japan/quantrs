use ndarray::{Array1, Array2};
use quantrs2_ml::prelude::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("QuantRS2 Machine Learning Showcase");
    println!("=================================");

    // Run different examples based on command line args
    let args: Vec<String> = std::env::args().collect();
    let example = args.get(1).map(|s| s.as_str()).unwrap_or("all");

    match example {
        "hep" => run_hep_example()?,
        "gan" => run_gan_example()?,
        "crypto" => run_crypto_example()?,
        "nlp" => run_nlp_example()?,
        "blockchain" => run_blockchain_example()?,
        "all" => {
            run_hep_example()?;
            run_gan_example()?;
            run_crypto_example()?;
            run_nlp_example()?;
            run_blockchain_example()?;
        }
        _ => {
            println!("Unknown example: {}", example);
            println!("Available examples: hep, gan, crypto, nlp, blockchain, all");
        }
    }

    Ok(())
}

fn run_hep_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nHigh-Energy Physics Example");
    println!("==========================");

    // Import HEP-specific types
    use quantrs2_ml::hep::{HEPQuantumClassifier, ParticleFeatures, ParticleType, CollisionEvent, HEPEncodingMethod};

    // Create a quantum classifier for high-energy physics data
    let num_qubits = 8;
    let feature_dim = 8;
    let num_classes = 2;

    println!("Creating HEP quantum classifier with {} qubits...", num_qubits);
    let mut classifier = HEPQuantumClassifier::new(
        num_qubits,
        feature_dim,
        num_classes,
        HEPEncodingMethod::HybridEncoding,
        vec!["background".to_string(), "higgs".to_string()],
    )?;

    // Create example particle data
    let electron = ParticleFeatures {
        particle_type: ParticleType::Electron,
        four_momentum: [50.5, 10.2, -15.7, 45.9],
        additional_features: vec![0.8, 0.2, 0.3, 0.1],
    };

    let photon = ParticleFeatures {
        particle_type: ParticleType::Photon,
        four_momentum: [62.8, 25.4, 30.1, 41.2],
        additional_features: vec![0.9, 0.1, 0.4, 0.2],
    };

    let higgs = ParticleFeatures {
        particle_type: ParticleType::Higgs,
        four_momentum: [125.3, 5.2, -7.1, 12.5],
        additional_features: vec![0.7, 0.5, 0.6, 0.3],
    };

    // Create a collision event
    let event = CollisionEvent {
        particles: vec![electron.clone(), photon.clone(), higgs.clone()],
        global_features: vec![250.0], // Total energy
        event_type: Some("higgs_candidate".to_string()),
    };

    // Classify a particle
    println!("\nClassifying particle (electron):");
    let (class, confidence) = classifier.classify_particle(&electron)?;
    println!("Classification: {} (confidence: {:.2})", class, confidence);

    // Classify a Higgs particle
    println!("\nClassifying particle (Higgs candidate):");
    let (class, confidence) = classifier.classify_particle(&higgs)?;
    println!("Classification: {} (confidence: {:.2})", class, confidence);

    // Create a Higgs detector
    println!("\nCreating Higgs detector and analyzing event...");
    let higgs_detector = quantrs2_ml::hep::HiggsDetector::new(num_qubits)?;
    let higgs_detections = higgs_detector.detect_higgs(&event)?;

    println!("Higgs detection results:");
    for (i, &is_higgs) in higgs_detections.iter().enumerate() {
        println!("  Particle {}: {}", i, if is_higgs { "Higgs candidate" } else { "Not Higgs" });
    }

    println!("\nHEP example completed successfully!");
    Ok(())
}

fn run_gan_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nQuantum GAN Example");
    println!("==================");

    // Import GAN-specific types
    use quantrs2_ml::gan::{QuantumGAN, GeneratorType, DiscriminatorType};

    // GAN parameters
    let num_qubits_gen = 6;
    let num_qubits_disc = 6;
    let latent_dim = 4;
    let data_dim = 8;

    println!("Creating Quantum GAN...");
    println!("  Generator: {} qubits", num_qubits_gen);
    println!("  Discriminator: {} qubits", num_qubits_disc);
    println!("  Latent dimension: {}", latent_dim);
    println!("  Data dimension: {}", data_dim);

    // Create quantum GAN
    let mut qgan = QuantumGAN::new(
        num_qubits_gen,
        num_qubits_disc,
        latent_dim,
        data_dim,
        GeneratorType::HybridClassicalQuantum,
        DiscriminatorType::HybridQuantumFeatures,
    )?;

    // Generate synthetic data (sine wave)
    println!("Generating sine wave training data...");
    let mut real_data = Array2::zeros((100, data_dim));

    for i in 0..100 {
        let x = (i as f64) / 100.0 * 2.0 * std::f64::consts::PI;

        for j in 0..data_dim {
            let freq = (j as f64 + 1.0) * 0.5;
            real_data[[i, j]] = (x * freq).sin() + 0.1 * rand::random::<f64>();
        }
    }

    // Train GAN (minimal training for demo)
    println!("Training GAN for a few iterations...");
    let start = Instant::now();
    let history = qgan.train(
        &real_data,
        5,    // epochs
        10,   // batch size
        0.01, // generator learning rate
        0.01, // discriminator learning rate
        1,    // discriminator steps
    )?;

    println!("Training completed in {:.2?}", start.elapsed());
    println!("Final losses:");
    println!("  Generator: {:.4}", history.gen_losses.last().unwrap_or(&0.0));
    println!("  Discriminator: {:.4}", history.disc_losses.last().unwrap_or(&0.0));

    // Generate samples
    println!("\nGenerating samples from trained GAN...");
    let num_samples = 3;
    let generated_samples = qgan.generate(num_samples)?;

    println!("Generated {} samples:", num_samples);
    for i in 0..num_samples {
        let sample = generated_samples.slice(ndarray::s![i, ..]);

        print!("  Sample {}: [", i);
        for (j, &val) in sample.iter().enumerate() {
            if j > 0 {
                print!(", ");
            }
            print!("{:.2}", val);

            // Only show first few values
            if j >= 5 {
                print!(", ...");
                break;
            }
        }
        println!("]");
    }

    println!("\nGAN example completed successfully!");
    Ok(())
}

fn run_crypto_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nQuantum Cryptography Example");
    println!("===========================");

    // Import crypto-specific types
    use quantrs2_ml::crypto::{QuantumKeyDistribution, ProtocolType, QuantumSignature};

    // Create BB84 QKD with 1000 qubits
    let num_qubits = 1000;
    println!("Creating BB84 protocol with {} qubits", num_qubits);
    let mut qkd = QuantumKeyDistribution::new(ProtocolType::BB84, num_qubits);

    // Set error rate
    qkd = qkd.with_error_rate(0.03);
    println!("Simulated error rate: {:.1}%", qkd.error_rate * 100.0);

    // Distribute key
    println!("Performing quantum key distribution...");
    let start = Instant::now();
    let key_length = qkd.distribute_key()?;
    println!("Key distribution completed in {:.2?}", start.elapsed());
    println!("Final key length: {} bits", key_length);

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

    // Quantum signatures
    println!("\nDemonstrating quantum digital signatures...");
    let signature = QuantumSignature::new(256, "Dilithium")?;

    // Sign a message
    let message = b"This message is quantum-signed";
    println!("Signing message: '{}'", std::str::from_utf8(message)?);

    let sig = signature.sign(message)?;
    println!("Signature generated (size: {} bytes)", sig.len());

    // Verify signature
    println!("Verifying signature...");
    let is_valid = signature.verify(message, &sig)?;
    println!("{} Signature verification {}!",
             if is_valid { "✓" } else { "✗" },
             if is_valid { "successful" } else { "failed" });

    println!("\nCrypto example completed successfully!");
    Ok(())
}

fn run_nlp_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nQuantum NLP Example");
    println!("==================");

    // Import NLP-specific types
    use quantrs2_ml::nlp::{SentimentAnalyzer, TextSummarizer};

    // Create sentiment analyzer
    println!("Creating quantum sentiment analyzer...");
    let analyzer = SentimentAnalyzer::new(6)?;

    // Test sentiment analysis
    let texts = [
        "I really enjoyed this product, it works perfectly!",
        "The service was terrible and the staff was rude",
    ];

    println!("\nAnalyzing sentiment of test texts:");
    for text in &texts {
        let (sentiment, confidence) = analyzer.analyze(text)?;

        println!("Text: \"{}\"", text);
        println!("Sentiment: {} (confidence: {:.2})\n", sentiment, confidence);
    }

    // Text summarization
    println!("\nDemonstrating quantum text summarization...");
    let summarizer = TextSummarizer::new(8)?;

    // Text to summarize
    let long_text = "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers. While traditional computers use bits as the smallest unit of data, quantum computers use quantum bits or qubits. Qubits can represent numerous possible combinations of 1 and 0 at the same time through a property called superposition. This allows quantum computers to consider and manipulate many combinations of information simultaneously, making them well suited to specific types of complex calculations. Major technology companies including IBM, Google, Microsoft, Amazon, and several startups are racing to build practical quantum computers. In 2019, Google claimed to have achieved quantum supremacy, performing a calculation that would be practically impossible for a classical computer.";

    println!("\nOriginal text ({} characters):", long_text.len());
    println!("{}\n", long_text);

    // Generate summary
    let summary = summarizer.summarize(long_text)?;

    println!("\nSummary ({} characters):", summary.len());
    println!("{}", summary);

    // Calculate compression ratio
    let compression = 100.0 * (1.0 - (summary.len() as f64) / (long_text.len() as f64));
    println!("\nCompression ratio: {:.1}%", compression);

    println!("\nNLP example completed successfully!");
    Ok(())
}

fn run_blockchain_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\nQuantum Blockchain Example");
    println!("=========================");

    // Import blockchain-specific types
    use quantrs2_ml::blockchain::{QuantumBlockchain, ConsensusType, Transaction, SmartContract};

    // Create a quantum blockchain
    println!("Creating quantum blockchain...");
    let mut blockchain = QuantumBlockchain::new(ConsensusType::QuantumProofOfWork, 2);

    // Create a transaction
    let sender = vec![1, 2, 3, 4];
    let recipient = vec![5, 6, 7, 8];
    let amount = 100.0;

    println!("Creating transaction: {} sends {} units to recipient",
             sender.iter().map(|&b| b.to_string()).collect::<Vec<_>>().join(""),
             amount);

    let transaction = Transaction::new(
        sender.clone(),
        recipient.clone(),
        amount,
        Vec::new(),
    );

    // Add transaction
    println!("Adding transaction to blockchain...");
    blockchain.add_transaction(transaction)?;

    // Mine a block
    println!("Mining new block...");
    let start = Instant::now();
    let block = blockchain.mine_block()?;
    println!("Block mined in {:.2?}", start.elapsed());

    println!("Block hash: {:02x?}", &block.hash[0..8.min(block.hash.len())]);
    println!("Blockchain length: {}", blockchain.chain.len());

    // Create a smart contract
    println!("\nCreating and deploying smart contract...");
    let owner = vec![9, 8, 7, 6];
    let bytecode = vec![0, 1, 2, 3, 4, 5]; // Simplified bytecode

    let contract = SmartContract::new(bytecode, owner.clone());

    // Execute contract (store data)
    println!("Executing smart contract store operation...");
    let store_input = vec![0, 1, 42]; // Operation 0 (store), key 1, value 42
    let mut contract_mut = contract.clone();
    let store_result = contract_mut.execute(&store_input)?;

    println!("Contract execution result: {:?}", store_result);

    // Execute contract (load data)
    println!("Executing smart contract load operation...");
    let load_input = vec![1, 1]; // Operation 1 (load), key 1
    let load_result = contract_mut.execute(&load_input)?;

    println!("Contract load result: {:?}", load_result);

    println!("\nBlockchain example completed successfully!");
    Ok(())
}