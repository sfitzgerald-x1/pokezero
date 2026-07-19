//! Build script for the `model` feature: embed the venv torch/lib rpath.
//!
//! The supported libtorch source is the shared venv's own PyTorch
//! (LIBTORCH_USE_PYTORCH=1 — never a vendored libtorch), whose dylibs carry
//! `@rpath/...` install names. torch-sys emits the link-search path but no
//! rpath, so without this the built extension only loads if `import torch`
//! ran first in the same process. Embedding the rpath makes the module
//! self-sufficient regardless of import order.
//!
//! Resolution order mirrors how the build is invoked: PYO3_PYTHON (set by
//! maturin), then PYTHON, then `python3` on PATH — the build recipe
//! (scripts/build_search_crate_model.sh) puts the venv's bin first.

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON");
    if std::env::var_os("CARGO_FEATURE_MODEL").is_none() {
        return;
    }
    let python = std::env::var("PYO3_PYTHON")
        .or_else(|_| std::env::var("PYTHON"))
        .unwrap_or_else(|_| "python3".to_string());
    let output = std::process::Command::new(&python)
        .args([
            "-c",
            "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))",
        ])
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if path.is_empty() {
                println!("cargo:warning=torch lib dir query returned empty; no rpath embedded");
            } else {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{path}");
            }
        }
        _ => println!(
            "cargo:warning=could not query torch lib dir via {python}; \
             no rpath embedded (import torch before pokezero_search at runtime)"
        ),
    }
}
