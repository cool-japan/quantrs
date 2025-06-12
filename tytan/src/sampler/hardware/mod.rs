//! Hardware integration samplers

pub mod dwave;
pub mod mikas;
pub mod fujitsu;
pub mod hitachi;
pub mod nec;
pub mod fpga;
pub mod photonic;

pub use dwave::DWaveSampler;
pub use mikas::MIKASAmpler;
pub use fujitsu::FujitsuDigitalAnnealerSampler;
pub use hitachi::HitachiCMOSSampler;
pub use nec::NECVectorAnnealingSampler;
pub use fpga::FPGASampler;
pub use photonic::PhotonicIsingMachineSampler;
