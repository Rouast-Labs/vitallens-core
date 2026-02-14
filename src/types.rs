// FILE: src/types.rs
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[cfg(feature = "python")]
use pyo3::prelude::*;

// --- CONFIG ---
// Input/Config types get getters AND setters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))] 
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct ModelConfig {
    pub name: String,
    pub supported_vitals: Vec<String>,
    pub fps_target: f32,
    pub input_size: u64,
    pub roi_method: String,
}

// --- INPUTS ---
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct FaceInput {
    pub coordinates: Vec<f32>, 
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct InputChunk {
    pub timestamp: Vec<f64>,
    pub signals: HashMap<String, Vec<f32>>,
    pub confidences: HashMap<String, Vec<f32>>,
    pub face: Option<FaceInput>, 
}

// --- ENUMS ---
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Enum))]
pub enum WaveformMode {
    Incremental,
    Windowed { seconds: f32 },
    Global,
}

#[cfg(feature = "python")]
impl<'source> FromPyObject<'source> for WaveformMode {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        if let Ok(s) = ob.extract::<String>() {
            match s.as_str() {
                "Incremental" => Ok(WaveformMode::Incremental),
                "Global" => Ok(WaveformMode::Global),
                _ => Err(pyo3::exceptions::PyValueError::new_err("Invalid WaveformMode string")),
            }
        } 
        else if let Ok((name, val)) = ob.extract::<(String, f32)>() {
            if name == "Windowed" {
                Ok(WaveformMode::Windowed { seconds: val })
            } else {
                Err(pyo3::exceptions::PyValueError::new_err("Invalid WaveformMode tuple"))
            }
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("WaveformMode must be a string or ('Windowed', seconds)"))
        }
    }
}

// --- OUTPUTS ---
// Output types get getters only (read-only for Python users)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct SignalResult {
    pub value: Option<f32>,
    pub data: Vec<f32>,
    pub confidence: Vec<f32>,
    pub unit: String,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct FaceResult {
    pub coordinates: Vec<Vec<f32>>,
    pub confidence: Vec<f32>,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct SessionResult {
    pub timestamp: Vec<f64>,
    pub face: Option<FaceResult>,
    pub signals: HashMap<String, SignalResult>,
    pub fps: f32,
    pub message: String,
    pub model_used: String,
}

// --- PYTHON CONSTRUCTORS ---
#[cfg(feature = "python")]
#[pymethods]
impl ModelConfig {
    #[new]
    fn new(name: String, supported_vitals: Vec<String>, fps_target: f32, input_size: u64, roi_method: String) -> Self {
        Self { name, supported_vitals, fps_target, input_size, roi_method }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl FaceInput {
    #[new]
    fn new(coordinates: Vec<f32>, confidence: f32) -> Self {
        Self { coordinates, confidence }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl InputChunk {
    #[new]
    fn new(
        timestamp: Vec<f64>,
        signals: HashMap<String, Vec<f32>>,
        confidences: HashMap<String, Vec<f32>>,
        face: Option<FaceInput>
    ) -> Self {
        Self { timestamp, signals, confidences, face }
    }
}