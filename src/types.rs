use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum WaveformMode {
    Incremental,
    Windowed(f32),
    Complete,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub supported_vitals: Vec<String>,
    pub fps_target: f32,
    pub input_size: usize,
    pub roi_method: String,
}

#[derive(Debug, Clone)]
pub struct FaceInput {
    pub coordinates: [f32; 4], // x, y, w, h
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct FaceResult {
    pub coordinates: Vec<[f32; 4]>,
    pub confidence: Vec<f32>,
    pub note: Option<String>,
}

#[derive(Debug, Clone)]
pub struct InputChunk {
    pub timestamp: Vec<f64>,
    pub signals: HashMap<String, Vec<f32>>,
    pub confidences: HashMap<String, Vec<f32>>,
    pub face: Option<FaceInput>, 
}

#[derive(Debug, Clone)]
pub struct SessionResult {
    pub timestamp: Vec<f64>,
    pub face: Option<FaceResult>,
    pub signals: HashMap<String, SignalResult>,
    pub fps: f32,
    pub message: String,
    pub model_used: String,
}

#[derive(Debug, Clone)]
pub struct SignalResult {
    pub value: Option<f32>,
    pub data: Vec<f32>,
    pub confidence: Vec<f32>,
    pub unit: String,
    pub note: Option<String>,
}