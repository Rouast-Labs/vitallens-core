#![allow(non_local_definitions)]

pub mod registry;
pub mod signal;
pub mod state;
pub mod types;
pub mod geometry;
mod bindings;

// --- TIER 1: STATEFUL SESSION (iOS, JS, Python Apps) ---
// Re-export these so they appear at the top level for UniFFI/Wasm
pub use state::session::Session;
pub use types::{SessionConfig, InputChunk, WaveformMode, SessionResult};

// --- UNIFFI SETUP (iOS) ---
#[cfg(not(target_arch = "wasm32"))]
uniffi::setup_scaffolding!();

// --- TIER 2: PYTHON MODULE (Research/Backend) ---
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn vitallens_core(m: &pyo3::Bound<'_, PyModule>) -> PyResult<()> {
    // 1. Export the Session class
    m.add_class::<Session>()?;

    // 2. Export the Data types
    m.add_class::<types::SessionConfig>()?;
    m.add_class::<types::InputChunk>()?;
    m.add_class::<types::FaceInput>()?;
    m.add_class::<types::SessionResult>()?;
    m.add_class::<types::SignalResult>()?;
    m.add_class::<types::FaceResult>()?;
    m.add_class::<types::Rect>()?;
    m.add_class::<types::InferenceMode>()?;
    m.add_class::<types::BufferConfig>()?;
    m.add_class::<types::BufferActionType>()?;
    m.add_class::<types::BufferAction>()?;
    m.add_class::<types::InferenceCommand>()?;
    m.add_class::<types::ExecutionPlan>()?;

    // 3. Export the Stateless Math functions
    // This calls code inside src/bindings/python.rs
    bindings::python::register_functions(m)?;
    
    Ok(())
}