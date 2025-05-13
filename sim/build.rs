fn main() {
    // Check if running on macOS
    if cfg!(target_os = "macos") {
        // Link to the Accelerate framework on macOS
        println!("cargo:rustc-link-lib=framework=Accelerate");

        // Force use of system BLAS
        println!("cargo:rustc-env=OPENBLAS_SYSTEM=1");
        println!("cargo:rustc-env=OPENBLAS64_SYSTEM=1");

        // Print debug info
        println!("cargo:warning=Building on macOS with Accelerate framework");
    }
}