fn main() {
    // Check if running on macOS
    if cfg!(target_os = "macos") {
        // Fix C++ standard library linking on macOS
        // Use libc++ instead of libstdc++
        println!("cargo:rustc-link-lib=c++");

        // Set environment variables to influence symengine compilation
        println!("cargo:rustc-env=CXXFLAGS=-stdlib=libc++");
        println!("cargo:rustc-env=LDFLAGS=-lc++");

        // Print debug info
        println!("cargo:warning=Building on macOS with C++ linking fix");
    }
}
