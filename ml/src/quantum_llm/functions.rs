//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::gate::{multi::*, single::*, GateOp};

use super::types::{
    GenerationConfig, QuantumLLMConfig, QuantumMemoryConfig, QuantumMemorySystem,
    QuantumReasoningConfig, QuantumReasoningModule, Vocabulary,
};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_qllm_config_creation() {
        let config = QuantumLLMConfig::small(10000);
        assert_eq!(config.vocab_size, 10000);
        assert_eq!(config.transformer_config.model_dim, 768);
        let large_config = QuantumLLMConfig::large(50000);
        assert_eq!(large_config.vocab_size, 50000);
        assert_eq!(large_config.transformer_config.model_dim, 1536);
    }
    #[test]
    fn test_vocabulary_creation() {
        let vocab = Vocabulary::new(1000).expect("Failed to create vocabulary");
        assert_eq!(vocab.quantum_embeddings.nrows(), 1000);
        assert!(vocab.special_tokens.contains_key("<eos>"));
    }
    #[test]
    fn test_generation_config() {
        let config = GenerationConfig::default();
        assert_eq!(config.max_length, 100);
        assert_eq!(config.temperature, 1.0);
        let creative_config = GenerationConfig::creative();
        assert!(creative_config.temperature > 1.0);
        assert!(creative_config.chain_of_thought);
    }
    #[test]
    fn test_quantum_memory_system() {
        let config = QuantumMemoryConfig::default();
        let memory_system = QuantumMemorySystem::new(config);
        assert!(memory_system.is_ok());
        let memory = memory_system.expect("QuantumMemorySystem::new should succeed");
        assert!(!memory.associative_banks.is_empty());
    }
    #[test]
    fn test_quantum_reasoning_module() {
        let config = QuantumReasoningConfig::default();
        let reasoning_module = QuantumReasoningModule::new(config);
        assert!(reasoning_module.is_ok());
        let reasoning = reasoning_module.expect("QuantumReasoningModule::new should succeed");
        assert!(!reasoning.logical_circuits.is_empty());
    }
}
