use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[cfg(feature = "python")]
use pyo3::prelude::*;

// --- CONFIG ---

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))] 
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct SessionConfig {
    pub supported_vitals: Vec<String>,
    pub fps_target: f32,
    pub input_size: u64,
    pub n_inputs: u64,
    pub roi_method: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }
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

// --- ROI CONFIGURATION ---

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Enum))]
pub enum RoiMethod {
    Face,
    Forehead,
    UpperBody,
    UpperBodyCropped,
    Custom { 
        left: f32, top: f32, right: f32, bottom: f32 
    },
}

#[cfg(feature = "python")]
impl<'source> FromPyObject<'source> for RoiMethod {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        if let Ok(s) = ob.extract::<String>() {
            match s.as_str() {
                "face" => Ok(RoiMethod::Face),
                "forehead" => Ok(RoiMethod::Forehead),
                "upper_body" => Ok(RoiMethod::UpperBody),
                "upper_body_cropped" => Ok(RoiMethod::UpperBodyCropped),
                _ => Err(pyo3::exceptions::PyValueError::new_err("Invalid RoiMethod string")),
            }
        } else if let Ok((l, t, r, b)) = ob.extract::<(f32, f32, f32, f32)>() {
             Ok(RoiMethod::Custom { left: l, top: t, right: r, bottom: b })
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("RoiMethod must be a string or tuple(l,t,r,b)"))
        }
    }
}

// --- BUFFER MANAGEMENT TYPES ---

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Enum))]
pub enum InferenceMode {
    Stream,
    File,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct BufferConfig {
    pub min_no_state: u32,
    pub min_with_state: u32,
    pub stream_max: u32,
    pub file_max: u32,
    pub overlap: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Enum))]
pub enum BufferActionType {
    Create,
    KeepAlive,
    Ignore,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct BufferAction {
    pub action: BufferActionType,
    pub matched_id: Option<String>,
    pub roi: Option<Rect>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct BufferMetadata {
    pub id: String,
    pub roi: Rect,
    pub count: u32,
    pub created_at: f64,
    pub last_seen: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct InferenceCommand {
    pub buffer_id: String,
    pub take_count: u32,
    pub keep_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct ExecutionPlan {
    pub command: Option<InferenceCommand>,
    pub buffers_to_drop: Vec<String>,
}

// --- OUTPUTS ---

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
}

// --- PYTHON CONSTRUCTORS ---

#[cfg(feature = "python")]
#[pymethods]
impl SessionConfig {
    #[new]
    fn new(supported_vitals: Vec<String>, fps_target: f32, input_size: u64, n_inputs: u64, roi_method: String) -> Self {
        Self { supported_vitals, fps_target, input_size, n_inputs, roi_method }
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

#[cfg(feature = "python")]
#[pymethods]
impl BufferMetadata {
    #[new]
    fn py_new(id: String, roi: Rect, count: u32, created_at: f64, last_seen: f64) -> Self {
        Self { id, roi, count, created_at, last_seen }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Rect {
    #[new]
    fn py_new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    fn __repr__(&self) -> String {
        format!("Rect(x={:.2}, y={:.2}, w={:.2}, h={:.2})", self.x, self.y, self.width, self.height)
    }
}