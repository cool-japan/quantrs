fn main() {
    // Check if running on macOS
    if cfg!(target_os = "macos") {
        // Fix C++ standard library linking on macOS
        // Use libc++ instead of libstdc++
        println!("cargo:rustc-link-lib=c++");

        // Set environment variables to influence symengine compilation
        println!("cargo:rustc-env=CXXFLAGS=-stdlib=libc++");
        println!("cargo:rustc-env=LDFLAGS=-lc++");

        // C++ linking fix applied for macOS
    }
}
