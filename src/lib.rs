#![allow(non_local_definitions)]

pub mod registry;
pub mod signal;
pub mod state;
pub mod types;
pub mod geometry;
mod bindings;

pub use state::session::Session;
pub use types::{SessionConfig, InputChunk, WaveformMode, SessionResult};

#[cfg(not(target_arch = "wasm32"))]
uniffi::setup_scaffolding!();

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn vitallens_core(m: &pyo3::Bound<'_, PyModule>) -> PyResult<()> {
    
    m.add_class::<Session>()?;

    m.add_class::<types::SessionConfig>()?;
    m.add_class::<types::InputChunk>()?;
    m.add_class::<types::SignalInput>()?;
    m.add_class::<types::FaceInput>()?;
    m.add_class::<types::SessionResult>()?;
    m.add_class::<types::WaveformResult>()?;
    m.add_class::<types::VitalResult>()?;
    m.add_class::<types::FaceResult>()?;
    m.add_class::<types::Rect>()?;
    m.add_class::<types::InferenceMode>()?;
    m.add_class::<types::BufferConfig>()?;
    m.add_class::<types::BufferActionType>()?;
    m.add_class::<types::BufferAction>()?;
    m.add_class::<types::BufferMetadata>()?;
    m.add_class::<types::InferenceCommand>()?;
    m.add_class::<types::ExecutionPlan>()?;

    bindings::python::register_functions(m)?;
    
    Ok(())
}