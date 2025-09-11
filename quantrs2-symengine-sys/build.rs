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

fn setup_manual_linking() {
    // Link to symengine
    if cfg!(feature = "static") {
        println!("cargo:rustc-link-lib=static=symengine");
    } else {
        println!("cargo:rustc-link-lib=symengine");
    }

    // Also link required dependencies
    println!("cargo:rustc-link-lib=gmp");
    println!("cargo:rustc-link-lib=mpfr");
    println!("cargo:rustc-link-lib=stdc++");

    // Platform-specific setup
    setup_platform_specific();

    // Custom symengine directory if specified
    if let Ok(dir) = env::var("SYMENGINE_DIR") {
        println!("cargo:rustc-link-search=native={}/lib", dir);
        println!("cargo:include={}/include", dir);
    }

    // Custom GMP directory
    if let Ok(dir) = env::var("GMP_DIR") {
        println!("cargo:rustc-link-search=native={}/lib", dir);
        println!("cargo:include={}/include", dir);
    }

    // Custom MPFR directory
    if let Ok(dir) = env::var("MPFR_DIR") {
        println!("cargo:rustc-link-search=native={}/lib", dir);
        println!("cargo:include={}/include", dir);
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
            let lib_path = format!("{}/lib", base_path);
            let include_path = format!("{}/include", base_path);

            if std::path::Path::new(&lib_path).exists() {
                println!("cargo:rustc-link-search=native={}", lib_path);
                println!("cargo:include={}", include_path);

                // Add specific paths for SymEngine
                let symengine_opt_path = format!("{}/opt/symengine", base_path);
                if std::path::Path::new(&symengine_opt_path).exists() {
                    let symengine_lib = format!("{}/lib", symengine_opt_path);
                    let symengine_include = format!("{}/include", symengine_opt_path);
                    if std::path::Path::new(&symengine_lib).exists() {
                        println!("cargo:rustc-link-search=native={}", symengine_lib);
                        println!("cargo:include={}", symengine_include);
                        println!(
                            "cargo:rerun-if-changed={}/include/symengine/cwrapper.h",
                            symengine_opt_path
                        );
                    }
                }

                // Add specific paths for dependencies
                for dep in &["gmp", "mpfr", "symengine"] {
                    let dep_lib = format!("{}/lib/{}", base_path, dep);
                    let dep_include = format!("{}/include/{}", base_path, dep);
                    if std::path::Path::new(&dep_lib).exists() {
                        println!("cargo:rustc-link-search=native={}", dep_lib);
                    }
                    if std::path::Path::new(&dep_include).exists() {
                        println!("cargo:include={}", dep_include);
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
        // Common Linux paths
        let common_paths = ["/usr/lib", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu"];
        for path in &common_paths {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-search=native={}", path);
            }
        }

        // Include paths
        let include_paths = ["/usr/include", "/usr/local/include"];
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
    }

    // Apply clang args
    for arg in clang_args {
        builder = builder.clang_arg(arg);
    }

    let bindings = builder.generate().expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings");
}
