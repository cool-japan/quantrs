//! Hardware integration samplers

pub mod amazon_braket;
pub mod azure_quantum;
pub mod dwave;
pub mod fpga;
pub mod fujitsu;
pub mod hitachi;
pub mod ibm_quantum;
pub mod mikas;
pub mod nec;
pub mod photonic;

pub use amazon_braket::{AmazonBraketConfig, AmazonBraketSampler, BraketDevice};
pub use azure_quantum::{AzureQuantumConfig, AzureQuantumSampler, AzureSolver};
pub use dwave::DWaveSampler;
pub use fpga::FPGASampler;
pub use fujitsu::FujitsuDigitalAnnealerSampler;
pub use hitachi::HitachiCMOSSampler;
pub use ibm_quantum::{IBMBackend, IBMQuantumConfig, IBMQuantumSampler};
pub use mikas::MIKASAmpler;
pub use nec::NECVectorAnnealingSampler;
pub use photonic::PhotonicIsingMachineSampler;
