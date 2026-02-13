pub mod registry;
pub mod signal;
pub mod state;
pub mod types;

// --- MOBILE (iOS/Android) ---
// Only compile this for non-wasm targets
#[cfg(not(target_arch = "wasm32"))]
pub mod mobile;

// --- UNIFFI SCAFFOLDING ---
// We only enable it for non-Wasm builds (iOS, Python, etc.)
#[cfg(not(target_arch = "wasm32"))]
uniffi::setup_scaffolding!();

// --- PYTHON ---
#[cfg(feature = "python")]
use pyo3::prelude::*;

mod bindings;

#[cfg(feature = "python")]
#[pymodule]
fn vitallens_core(_py: Python, m: &PyModule) -> PyResult<()> {
    bindings::python::register_functions(m)?;
    Ok(())
}