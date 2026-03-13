use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Configuration for initializing a VitalLens session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))] 
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct SessionConfig {
    pub model_name: String,
    pub supported_vitals: Vec<String>,
    pub return_waveforms: Option<Vec<String>>,
    pub fps_target: f32,
    pub input_size: u64,
    pub n_inputs: u64,
    pub roi_method: String,
    #[serde(default)]
    #[cfg_attr(not(target_arch = "wasm32"), uniffi(default = None))]
    pub estimate_rolling_vitals: Option<bool>,
}

/// A generic wrapper for a single physical signal and its corresponding confidence array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct SignalInput {
    pub data: Vec<f32>,
    pub confidence: Vec<f32>,
}

/// Input payload representing tracked face bounding boxes over a sequence of frames.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct FaceInput {
    pub coordinates: Vec<Vec<f32>>, 
    pub confidence: Vec<f32>,
}

/// The core batch input object submitted to the session engine for processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct SessionInput {
    pub face: Option<FaceInput>,
    pub signals: HashMap<String, SignalInput>,
    pub timestamp: Vec<f64>,
}

impl SessionInput {
    /// Validates that all provided signals and face inputs share the exact same length as the timestamps.
    pub fn validate_lengths(&self) -> Result<usize, String> {
        let expected_len = self.timestamp.len();

        if let Some(face) = &self.face {
            if face.coordinates.len() != expected_len {
                return Err(format!("Length mismatch: face.coordinates has length {}, expected {}", face.coordinates.len(), expected_len));
            }
            if face.confidence.len() != expected_len {
                return Err(format!("Length mismatch: face.confidence has length {}, expected {}", face.confidence.len(), expected_len));
            }
        }

        for (name, sig) in &self.signals {
            if sig.data.len() != expected_len {
                return Err(format!("Length mismatch: signals[{}].data has length {}, expected {}", name, sig.data.len(), expected_len));
            }
            if sig.confidence.len() != expected_len {
                return Err(format!("Length mismatch: signals[{}].confidence has length {}, expected {}", name, sig.confidence.len(), expected_len));
            }
        }

        Ok(expected_len)
    }
}

/// Dictates how the session should prune state and yield output waveforms.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Enum))]
pub enum WaveformMode {
    /// Returns only newly added frames, discarding older history.
    Incremental,
    /// Returns a fixed trailing window of data (in seconds).
    Windowed { seconds: f32 },
    /// Returns the entire history and disables state pruning.
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

/// A simple bounding box definition.
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

/// Pre-defined and custom methods for extracting physical ROIs from a base face bounding box.
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

/// The origin detector format of the incoming bounding box.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Enum))]
pub enum FaceDetector {
    Default,
    AppleVision,
}

#[cfg(feature = "python")]
impl<'source> FromPyObject<'source> for FaceDetector {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        if let Ok(s) = ob.extract::<String>() {
            match s.to_lowercase().as_str() {
                "default" => Ok(FaceDetector::Default),
                "apple_vision" | "applevision" | "vision" => Ok(FaceDetector::AppleVision),
                _ => Err(pyo3::exceptions::PyValueError::new_err("Invalid FaceDetector string")),
            }
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("FaceDetector must be a string"))
        }
    }
}

// --- BUFFER MANAGEMENT TYPES ---

/// Operational paradigm dictating buffer fill limits.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Enum))]
pub enum InferenceMode {
    Stream,
    File,
}

/// Capacity constraints and threshold configuration for buffers.
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

/// Action to be taken on a target ROI after evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Enum))]
pub enum BufferActionType {
    Create,
    KeepAlive,
    Ignore,
}

/// The concrete instruction resulting from evaluating a target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct BufferAction {
    pub action: BufferActionType,
    pub matched_id: Option<String>,
    pub roi: Option<Rect>,
}

/// Identifiable state metadata attached to a specific processing buffer.
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

/// Command returned by the BufferPlanner indicating how many frames to extract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct InferenceCommand {
    pub buffer_id: String,
    pub take_count: u32,
    pub keep_count: u32,
}

/// Aggregated output from the BufferPlanner dictating next steps for all buffers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct ExecutionPlan {
    pub command: Option<InferenceCommand>,
    pub buffers_to_drop: Vec<String>,
}

// --- OUTPUTS ---

/// Derived continuous waveform data alongside unit metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct WaveformResult {
    pub data: Vec<f32>,
    pub confidence: Vec<f32>,
    pub unit: String,
    pub note: String,
}

/// Derived scalar physiological value.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct VitalResult {
    pub value: f32,
    pub confidence: f32,
    pub unit: String,
    pub note: String,
}

/// Filtered/smoothed output face coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct FaceResult {
    pub coordinates: Vec<Vec<f32>>,
    pub confidence: Vec<f32>,
    pub note: Option<String>,
}

/// The final payload aggregated from a session processing tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct SessionResult {
    pub timestamp: Vec<f64>,
    pub face: Option<FaceResult>,
    pub waveforms: HashMap<String, WaveformResult>,
    pub vitals: HashMap<String, VitalResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rolling_vitals: Option<HashMap<String, WaveformResult>>,
    pub fps: f32,
    pub message: String,
}

/// Extracted UI/display properties for a specific vital sign.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all))]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Record))]
pub struct VitalDisplayMeta {
    pub id: String,
    pub display_name: String,
    pub short_name: String,
    pub unit: String,
    pub color: String,
    pub emoji: String,
}

// --- PYTHON CONSTRUCTORS ---

#[cfg(feature = "python")]
#[pymethods]
impl SessionConfig {
    #[new]
    #[pyo3(signature = (model_name, supported_vitals, fps_target, input_size, n_inputs, roi_method, return_waveforms=None, estimate_rolling_vitals=None))]
    fn new(
        model_name: String,
        supported_vitals: Vec<String>, 
        fps_target: f32, 
        input_size: u64, 
        n_inputs: u64, 
        roi_method: String,
        return_waveforms: Option<Vec<String>>,
        estimate_rolling_vitals: Option<bool>
    ) -> Self {
        Self {
            model_name, supported_vitals, return_waveforms,
            fps_target, input_size, n_inputs, roi_method,
            estimate_rolling_vitals
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl SignalInput {
    #[new]
    fn new(data: Vec<f32>, confidence: Vec<f32>) -> Self {
        Self { data, confidence }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl FaceInput {
    #[new]
    fn new(coordinates: Vec<Vec<f32>>, confidence: Vec<f32>) -> Self {
        Self { coordinates, confidence }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl SessionInput {
    #[new]
    #[pyo3(signature = (face, signals, timestamp))]
    fn new(
        face: Option<FaceInput>,
        signals: HashMap<String, SignalInput>,
        timestamp: Vec<f64>
    ) -> Self {
        Self { face, signals, timestamp }
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

#[cfg(feature = "python")]
#[pymethods]
impl VitalDisplayMeta {
    fn __repr__(&self) -> String {
        format!("VitalDisplayMeta(id='{}', name='{}')", self.id, self.display_name)
    }
}