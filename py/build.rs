fn main() {
    // Tell Cargo that if the given file changes, rerun this build script
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");

    // We won't try to link directly with Python framework
    // as maturin will handle this for us
}
