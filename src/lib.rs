#![allow(non_local_definitions)]

pub mod registry;
pub mod signal;
pub mod state;
pub mod types;
mod bindings;

#[cfg(not(target_arch = "wasm32"))]
pub mod mobile;

// --- TIER 1: STATEFUL SESSION (iOS, JS, Python Apps) ---
// Re-export these so they appear at the top level for UniFFI/Wasm
pub use state::session::Session;
pub use types::{ModelConfig, InputChunk, WaveformMode, SessionResult};

// --- UNIFFI SETUP (iOS) ---
#[cfg(not(target_arch = "wasm32"))]
uniffi::setup_scaffolding!();

// --- TIER 2: PYTHON MODULE (Research/Backend) ---
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn vitallens_core(_py: Python, m: &PyModule) -> PyResult<()> {
    // 1. Export the Session class
    m.add_class::<Session>()?;

    // 2. Export the Data Types (CRITICAL FIX)
    m.add_class::<types::ModelConfig>()?;
    m.add_class::<types::InputChunk>()?;
    m.add_class::<types::FaceInput>()?;
    m.add_class::<types::SessionResult>()?;
    m.add_class::<types::SignalResult>()?;
    m.add_class::<types::FaceResult>()?;

    // 3. Export the Stateless Math functions
    bindings::python::register_functions(m)?;
    
    Ok(())
}