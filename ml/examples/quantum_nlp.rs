use quantrs2_ml::nlp::{
    EmbeddingStrategy, NLPTaskType, QuantumLanguageModel, TextPreprocessor, WordEmbedding,
};
use quantrs2_ml::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("Quantum Natural Language Processing Examples");
    println!("==========================================");

    // Classification example
    run_text_classification()?;

    // Sentiment analysis
    run_sentiment_analysis()?;

    // Text summarization
    run_text_summarization()?;

    Ok(())
}

fn run_text_classification() -> Result<()> {
    println!("\nText Classification Example");
    println!("--------------------------");

    // Create quantum language model for classification
    let num_qubits = 6;
    let embedding_dim = 16;
    let embedding_strategy = EmbeddingStrategy::from(64); // Was max_seq_length before

    println!("Creating quantum language model with {} qubits", num_qubits);
    let mut model = QuantumLanguageModel::new(
        num_qubits,
        embedding_dim,
        embedding_strategy,
        NLPTaskType::Classification,
        vec![
            "technology".to_string(),
            "sports".to_string(),
            "politics".to_string(),
            "entertainment".to_string(),
        ],
    )?;

    // Create training data
    println!("Preparing training data...");
    let training_texts = vec![
        "Latest smartphone features advanced AI capabilities".to_string(),
        "The football team won the championship yesterday".to_string(),
        "New legislation passed regarding climate change".to_string(),
        "The movie premiere attracted numerous celebrities".to_string(),
        "Software engineers developed a new programming language".to_string(),
        "Athletes compete in the international tournament next week".to_string(),
        "Senator announces campaign for presidential election".to_string(),
        "Actor receives award for outstanding performance".to_string(),
    ];

    let training_labels = vec![0, 1, 2, 3, 0, 1, 2, 3];

    // Build vocabulary
    println!("Building vocabulary from training texts...");
    let vocab_size = model.build_vocabulary(&training_texts)?;
    println!("Vocabulary size: {}", vocab_size);

    // Train embeddings
    println!("Training word embeddings...");
    model.train_embeddings(&training_texts)?;

    // Train model
    println!("Training quantum language model...");
    let start = Instant::now();
    model.train(&training_texts, &training_labels, 10, 0.05)?;
    println!("Training completed in {:.2?}", start.elapsed());

    // Test classification
    let test_texts = [
        "New computer processor breaks performance records",
        "Basketball player scores winning point in final seconds",
        "Government announces new tax policy",
        "New series premieres with record viewership",
    ];

    println!("\nClassifying test texts:");
    for text in &test_texts {
        let start = Instant::now();
        let (category, confidence) = model.classify(text)?;

        println!("Text: \"{}\"", text);
        println!(
            "Classification: {} (confidence: {:.2})",
            category, confidence
        );
        println!("Classification time: {:.2?}\n", start.elapsed());
    }

    Ok(())
}

fn run_sentiment_analysis() -> Result<()> {
    println!("\nSentiment Analysis Example");
    println!("-------------------------");

    // Create sentiment analyzer
    let num_qubits = 6;
    println!(
        "Creating quantum sentiment analyzer with {} qubits",
        num_qubits
    );
    let analyzer = quantrs2_ml::nlp::SentimentAnalyzer::new(num_qubits)?;

    // Test sentiment analysis
    let test_texts = [
        "I really enjoyed this product, it works perfectly!",
        "The service was terrible and the staff was rude",
        "The movie was okay, nothing special but not bad either",
        "The experience exceeded all my expectations!",
    ];

    println!("\nAnalyzing sentiment of test texts:");
    for text in &test_texts {
        let start = Instant::now();
        let (sentiment, confidence) = analyzer.analyze(text)?;

        println!("Text: \"{}\"", text);
        println!("Sentiment: {} (confidence: {:.2})", sentiment, confidence);
        println!("Analysis time: {:.2?}\n", start.elapsed());
    }

    Ok(())
}

fn run_text_summarization() -> Result<()> {
    println!("\nText Summarization Example");
    println!("-------------------------");

    // Create text summarizer
    let num_qubits = 8;
    println!(
        "Creating quantum text summarizer with {} qubits",
        num_qubits
    );
    let summarizer = quantrs2_ml::nlp::TextSummarizer::new(num_qubits)?;

    // Text to summarize
    let long_text = "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers. While traditional computers use bits as the smallest unit of data, quantum computers use quantum bits or qubits. Qubits can represent numerous possible combinations of 1 and 0 at the same time through a property called superposition. This allows quantum computers to consider and manipulate many combinations of information simultaneously, making them well suited to specific types of complex calculations. Another key property of quantum computing is entanglement, which allows qubits that are separated by great distances to still be connected. Changing the state of one entangled qubit will instantaneously change the state of its partner regardless of how far apart they are. Quantum computers excel at solving certain types of problems, such as factoring very large numbers, searching unsorted databases, and simulating quantum systems like molecules for drug development. However, they are not expected to replace classical computers for most everyday tasks. Major technology companies including IBM, Google, Microsoft, Amazon, and several startups are racing to build practical quantum computers. In 2019, Google claimed to have achieved quantum supremacy, performing a calculation that would be practically impossible for a classical computer. While current quantum computers are still limited by high error rates and the need for extreme cooling, they represent one of the most promising frontier technologies of the 21st century.";

    println!("\nOriginal text ({} characters):", long_text.len());
    println!("{}\n", long_text);

    // Generate summary
    println!("Generating quantum summary...");
    let start = Instant::now();
    let summary = summarizer.summarize(long_text)?;
    println!("Summarization completed in {:.2?}", start.elapsed());

    println!("\nSummary ({} characters):", summary.len());
    println!("{}", summary);

    // Calculate compression ratio
    let compression = 100.0 * (1.0 - (summary.len() as f64) / (long_text.len() as f64));
    println!("\nCompression ratio: {:.1}%", compression);

    Ok(())
}
