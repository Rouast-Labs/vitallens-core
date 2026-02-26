pub mod registry;
pub mod signal;
pub mod state;
pub mod types;
pub mod geometry;
mod bindings;

pub use state::session::Session;
pub use types::{SessionConfig, SessionInput, WaveformMode, SessionResult, VitalDisplayMeta};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg_attr(not(target_arch = "wasm32"), uniffi::export)]
pub fn get_vital_info(vital_id: String) -> Option<VitalDisplayMeta> {
    registry::get_vital_meta(&vital_id).map(|meta| VitalDisplayMeta {
        id: meta.id,
        display_name: meta.display_name,
        short_name: meta.short_name,
        unit: meta.unit,
        color: meta.color,
        emoji: meta.emoji,
    })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = getVitalInfo)]
pub fn get_vital_info_js(vital_id: &str) -> JsValue {
    let info = get_vital_info(vital_id.to_string());
    serde_wasm_bindgen::to_value(&info).unwrap()
}

#[cfg(not(target_arch = "wasm32"))]
uniffi::setup_scaffolding!();

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn vitallens_core(m: &pyo3::Bound<'_, PyModule>) -> PyResult<()> {
    
    m.add_class::<Session>()?;

    m.add_class::<types::SessionConfig>()?;
    m.add_class::<types::SessionInput>()?;
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
    m.add_class::<types::VitalDisplayMeta>()?;

    bindings::python::register_functions(m)?;
    
    Ok(())
}