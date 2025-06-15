//! Hardware integration samplers

pub mod dwave;
pub mod fpga;
pub mod fujitsu;
pub mod hitachi;
pub mod mikas;
pub mod nec;
pub mod photonic;

pub use dwave::DWaveSampler;
pub use fpga::FPGASampler;
pub use fujitsu::FujitsuDigitalAnnealerSampler;
pub use hitachi::HitachiCMOSSampler;
pub use mikas::MIKASAmpler;
pub use nec::NECVectorAnnealingSampler;
pub use photonic::PhotonicIsingMachineSampler;
