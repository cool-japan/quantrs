fn main() {
    // Check if running on macOS
    if cfg!(target_os = "macos") {
        // Link to the Accelerate framework on macOS
        println!("cargo:rustc-link-lib=framework=Accelerate");

        // Force use of system BLAS
        println!("cargo:rustc-env=OPENBLAS_SYSTEM=1");
        println!("cargo:rustc-env=OPENBLAS64_SYSTEM=1");

        // Fix C++ standard library linking on macOS
        // Use libc++ instead of libstdc++
        println!("cargo:rustc-link-lib=c++");

        // Set environment variables to influence symengine compilation
        println!("cargo:rustc-env=CXXFLAGS=-stdlib=libc++");
        println!("cargo:rustc-env=LDFLAGS=-lc++");

        // Accelerate framework and C++ linking fix applied for macOS
    }
}
