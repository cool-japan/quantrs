use std::{env, path::PathBuf};

fn main() {
    // Rerun build script if wrapper.h changes
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=SYMENGINE_DIR");
    println!("cargo:rerun-if-env-changed=GMP_DIR");
    println!("cargo:rerun-if-env-changed=MPFR_DIR");

    // Try to find SymEngine using pkg-config first
    #[cfg(feature = "system-deps")]
    {
        if let Ok(library) = pkg_config::Config::new()
            .atleast_version("0.11.0")
            .probe("symengine")
        {
            println!("Found SymEngine via pkg-config");
            for path in &library.include_paths {
                println!("cargo:include={}", path.display());
            }
            // pkg-config handles linking automatically
        } else {
            eprintln!(
                "Warning: pkg-config failed to find SymEngine, falling back to manual detection"
            );
            setup_manual_linking();
        }
    }

    #[cfg(not(feature = "system-deps"))]
    {
        setup_manual_linking();
    }

    // Generate bindings
    generate_bindings();
}

#[cfg(target_os = "macos")]
fn check_static_lib_exists() -> bool {
    // Check common Homebrew paths for static symengine library
    let paths = [
        "/opt/homebrew/opt/symengine/lib/libsymengine.a",
        "/opt/homebrew/lib/libsymengine.a",
        "/usr/local/opt/symengine/lib/libsymengine.a",
        "/usr/local/lib/libsymengine.a",
    ];

    for path in &paths {
        if std::path::Path::new(path).exists() {
            return true;
        }
    }

    // Also check if SYMENGINE_DIR is set and has static library
    if let Ok(dir) = env::var("SYMENGINE_DIR") {
        let static_lib_path = format!("{dir}/lib/libsymengine.a");
        if std::path::Path::new(&static_lib_path).exists() {
            return true;
        }
    }

    false
}

fn setup_manual_linking() {
    // Link to symengine
    if cfg!(feature = "static") {
        // Check if static library exists on any platform
        let mut static_lib_exists = false;

        #[cfg(target_os = "macos")]
        {
            static_lib_exists = check_static_lib_exists();
        }

        #[cfg(target_os = "linux")]
        {
            // Check common Linux paths for static symengine library
            let paths = [
                "/usr/local/lib/libsymengine.a",
                "/usr/lib/libsymengine.a",
                "/usr/lib/x86_64-linux-gnu/libsymengine.a",
            ];

            for path in &paths {
                if std::path::Path::new(path).exists() {
                    static_lib_exists = true;
                    break;
                }
            }

            // Also check if SYMENGINE_DIR is set and has static library
            if !static_lib_exists {
                if let Ok(dir) = env::var("SYMENGINE_DIR") {
                    let static_lib_path = format!("{}/lib/libsymengine.a", dir);
                    if std::path::Path::new(&static_lib_path).exists() {
                        static_lib_exists = true;
                    }
                }
            }
        }

        if static_lib_exists {
            println!("cargo:rustc-link-lib=static=symengine");
        } else {
            eprintln!("Warning: Static library not found, using dynamic linking instead");
            println!("cargo:rustc-link-lib=symengine");
        }
    } else {
        println!("cargo:rustc-link-lib=symengine");
    }

    // Also link required dependencies
    println!("cargo:rustc-link-lib=gmp");
    println!("cargo:rustc-link-lib=mpfr");

    // Use appropriate C++ standard library
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=c++");
    #[cfg(not(target_os = "macos"))]
    println!("cargo:rustc-link-lib=stdc++");

    // Platform-specific setup
    setup_platform_specific();

    // Custom symengine directory if specified
    if let Ok(dir) = env::var("SYMENGINE_DIR") {
        println!("cargo:rustc-link-search=native={dir}/lib");
        println!("cargo:include={dir}/include");
    }

    // Custom GMP directory
    if let Ok(dir) = env::var("GMP_DIR") {
        println!("cargo:rustc-link-search=native={dir}/lib");
        println!("cargo:include={dir}/include");
    }

    // Custom MPFR directory
    if let Ok(dir) = env::var("MPFR_DIR") {
        println!("cargo:rustc-link-search=native={dir}/lib");
        println!("cargo:include={dir}/include");
    }
}

fn setup_platform_specific() {
    #[cfg(target_os = "macos")]
    {
        // Try common homebrew paths
        let homebrew_paths = [
            "/opt/homebrew", // Apple Silicon
            "/usr/local",    // Intel
        ];

        for base_path in &homebrew_paths {
            let lib_path = format!("{base_path}/lib");
            let include_path = format!("{base_path}/include");

            if std::path::Path::new(&lib_path).exists() {
                println!("cargo:rustc-link-search=native={lib_path}");
                println!("cargo:include={include_path}");

                // Add specific paths for SymEngine
                let symengine_opt_path = format!("{base_path}/opt/symengine");
                if std::path::Path::new(&symengine_opt_path).exists() {
                    let symengine_lib = format!("{symengine_opt_path}/lib");
                    let symengine_include = format!("{symengine_opt_path}/include");
                    if std::path::Path::new(&symengine_lib).exists() {
                        println!("cargo:rustc-link-search=native={symengine_lib}");
                        println!("cargo:include={symengine_include}");
                        println!(
                            "cargo:rerun-if-changed={symengine_opt_path}/include/symengine/cwrapper.h"
                        );
                    }
                }

                // Add specific paths for dependencies
                for dep in &["gmp", "mpfr", "symengine"] {
                    let dep_lib = format!("{base_path}/lib/{dep}");
                    let dep_include = format!("{base_path}/include/{dep}");
                    if std::path::Path::new(&dep_lib).exists() {
                        println!("cargo:rustc-link-search=native={dep_lib}");
                    }
                    if std::path::Path::new(&dep_include).exists() {
                        println!("cargo:include={dep_include}");
                    }
                }
                break;
            }
        }

        // macOS frameworks
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    #[cfg(target_os = "linux")]
    {
        // Common Linux paths - prioritize /usr/local/lib for custom installations
        let common_paths = ["/usr/local/lib", "/usr/lib", "/usr/lib/x86_64-linux-gnu"];
        for path in &common_paths {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-search=native={}", path);
            }
        }

        // Include paths
        let include_paths = ["/usr/local/include", "/usr/include"];
        for path in &include_paths {
            if std::path::Path::new(path).exists() {
                println!("cargo:include={}", path);
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Windows-specific setup
        if let Ok(vcpkg_root) = env::var("VCPKG_ROOT") {
            let target_triplet = env::var("TARGET").unwrap_or_else(|_| "x64-windows".to_string());
            let lib_path = format!("{}/installed/{}/lib", vcpkg_root, target_triplet);
            let include_path = format!("{}/installed/{}/include", vcpkg_root, target_triplet);

            if std::path::Path::new(&lib_path).exists() {
                println!("cargo:rustc-link-search=native={}", lib_path);
                println!("cargo:include={}", include_path);
            }
        }
    }
}

fn generate_bindings() {
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Function allowlist
        .allowlist_function("basic_.*")
        .allowlist_function("symengine_.*")
        .allowlist_function("integer_.*")
        .allowlist_function("real_double_.*")
        .allowlist_function("complex_double_.*")
        .allowlist_function("rational_.*")
        .allowlist_function("symbol_.*")
        // Type allowlist
        .allowlist_type("basic_struct")
        .allowlist_type("CVecBasic")
        .allowlist_type("CSetBasic")
        .allowlist_type("MapBasicBasic")
        // Block problematic types
        .blocklist_type("max_align_t")
        .blocklist_type("__darwin_.*")
        // Configuration
        .prepend_enum_name(false)
        .generate_comments(true)
        .layout_tests(false)  // Disable layout tests to avoid issues on different platforms
        .default_enum_style(bindgen::EnumVariation::Rust { non_exhaustive: false });

    // Add include paths from environment
    let mut clang_args = Vec::new();

    // Add custom include paths
    if let Ok(paths) = env::var("BINDGEN_EXTRA_CLANG_ARGS") {
        clang_args.extend(paths.split_whitespace().map(String::from));
    }

    // Platform-specific clang args
    #[cfg(target_os = "macos")]
    {
        clang_args.push("-I/opt/homebrew/include".to_string());
        clang_args.push("-I/usr/local/include".to_string());
        clang_args.push("-I/opt/homebrew/opt/symengine/include".to_string());
        // Use libc++ on macOS
        clang_args.push("-stdlib=libc++".to_string());
    }

    #[cfg(target_os = "linux")]
    {
        clang_args.push("-I/usr/include".to_string());
        clang_args.push("-I/usr/local/include".to_string());
        // On Linux, use libstdc++ (default, no need to specify)
    }

    // Apply clang args
    for arg in clang_args {
        builder = builder.clang_arg(arg);
    }

    let bindings = builder.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");
}
