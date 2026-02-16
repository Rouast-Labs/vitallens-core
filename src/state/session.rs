use std::collections::HashMap;
use std::sync::Mutex;
use crate::types::{InputChunk, ModelConfig, SessionResult, WaveformMode, SignalResult, FaceResult, FaceInput};
use crate::state::buffers::SignalBuffer;
use crate::registry::{self, VitalType, CalculationMethod, VitalMeta};
use crate::signal::fft::FftScratch;
use crate::signal::peaks::SignalBounds;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// ========================================================================
// THE LOGIC CORE (Private, no FFI attributes, uses &mut self freely)
// ========================================================================

#[derive(Debug)]
struct SessionCore {
    config: ModelConfig,
    timestamps: Vec<f64>,
    signal_data: HashMap<String, SignalBuffer>,
    signal_confs: HashMap<String, SignalBuffer>,
    face_coords: Vec<[f32; 4]>,
    face_conf: Vec<f32>,
    face_note: Option<String>,
    last_emitted_timestamp: f64,
    max_history: usize,
    active_vitals: Vec<String>,
    vital_metas: HashMap<String, VitalMeta>,
    fft_scratch: FftScratch,
}

impl SessionCore {
    fn new(config: ModelConfig) -> Self {
        let max_history = (config.fps_target * 60.0) as usize;
        
        let mut resolved_metas = Vec::new();
        let mut meta_map = HashMap::new();

        for vital_id in &config.supported_vitals {
            if let Some(meta) = registry::get_vital_meta(vital_id) {
                meta_map.insert(meta.id.clone(), meta.clone());
                resolved_metas.push(meta);
            }
        }
        
        resolved_metas.sort_by_key(|m| {
            m.derivations.first().map(|d| d.order).unwrap_or(0)
        });
        let active_vitals = resolved_metas.into_iter().map(|m| m.id).collect();
        
        Self {
            config,
            timestamps: Vec::new(),
            signal_data: HashMap::new(),
            signal_confs: HashMap::new(),
            face_coords: Vec::new(),
            face_conf: Vec::new(),
            face_note: None,
            last_emitted_timestamp: -1.0,
            max_history,
            active_vitals,
            vital_metas: meta_map,
            fft_scratch: FftScratch::new(),
        }
    }

    fn process_chunk(&mut self, chunk: InputChunk, mode: WaveformMode) -> SessionResult {
        let overlap = self.merge_timestamps(&chunk.timestamp);
        self.merge_signals(chunk.signals, chunk.confidences, overlap);
        self.merge_face(chunk.face, overlap, chunk.timestamp.len());
        if !matches!(mode, WaveformMode::Global) {
            // Only prune history for streaming modes
            self.prune_state();
        }

        let derived_results = self.perform_derivations(&mode);

        self.construct_output(mode, derived_results)
    }

    // --- Private Helpers (Moved here to avoid FFI type checks) ---

    fn merge_timestamps(&mut self, new_times: &[f64]) -> usize {
        if new_times.is_empty() { return 0; }
        
        if self.timestamps.is_empty() {
            self.timestamps.extend_from_slice(new_times);
            return 0;
        }

        let last_time = *self.timestamps.last().unwrap();
        let epsilon = 0.005;

        match new_times.iter().position(|&t| t > last_time + epsilon) {
            Some(idx) => {
                let overlap = idx;
                self.timestamps.extend_from_slice(&new_times[idx..]);
                overlap
            }
            None => new_times.len()  
        }
    }

    fn merge_signals(&mut self, signals: HashMap<String, Vec<f32>>, confidences: HashMap<String, Vec<f32>>, overlap: usize) {
        for (key, data) in signals {
            let data_buf = self.signal_data.entry(key.clone()).or_insert_with(SignalBuffer::new);
            data_buf.merge(&data, overlap, Some("unitless".to_string()));

            if let Some(conf_data) = confidences.get(&key) {
                let conf_buf = self.signal_confs.entry(key).or_insert_with(SignalBuffer::new);
                conf_buf.merge(conf_data, overlap, None);
            }
        }
    }

    fn merge_face(&mut self, face: Option<FaceInput>, overlap: usize, chunk_len: usize) {
        let (coords_vec, conf) = match face {
            Some(f) => (f.coordinates, f.confidence),
            None => (vec![0.0; 4], 0.0)
        };
        
        let coords: [f32; 4] = if coords_vec.len() == 4 {
            [coords_vec[0], coords_vec[1], coords_vec[2], coords_vec[3]]
        } else {
            [0.0; 4]
        };

        let new_frames_count = if chunk_len > overlap { chunk_len - overlap } else { 0 };

        for _ in 0..new_frames_count {
            self.face_coords.push(coords);
            self.face_conf.push(conf);
        }
    }

    fn prune_state(&mut self) {
        if self.timestamps.len() > self.max_history {
            let remove_count = self.timestamps.len() - self.max_history;
            
            self.timestamps.drain(0..remove_count);
            self.face_coords.drain(0..remove_count);
            self.face_conf.drain(0..remove_count);

            for buf in self.signal_data.values_mut() {
                buf.prune(self.max_history);
            }
            for buf in self.signal_confs.values_mut() {
                buf.prune(self.max_history);
            }
        }
    }

    fn perform_derivations(&mut self, mode: &WaveformMode) -> HashMap<String, (f32, f32)> {
        let mut results = HashMap::new();

        let vital_ids: Vec<String> = self.active_vitals.clone();

        for vital_id in vital_ids {
            if let Some(meta) = self.vital_metas.get(&vital_id).cloned() { 
                for cfg in &meta.derivations {
                    if let Some(source_buf) = self.signal_data.get(&cfg.source_signal) {
                        let full_data = source_buf.compute_average();
                        let full_conf = self.signal_confs.get(&cfg.source_signal)
                            .map(|b| b.compute_average())
                            .unwrap_or_else(|| vec![1.0; full_data.len()]);
                        let available_frames = full_data.len();
                        let target_fs = self.config.fps_target;
                        let min_frames = (cfg.min_window_seconds * target_fs) as usize;
                        if available_frames >= min_frames {
                            let take_frames = match mode {
                                WaveformMode::Global => available_frames,
                                _ => {
                                    let preferred_frames = (cfg.preferred_window_seconds * target_fs) as usize;
                                    preferred_frames.min(available_frames)
                                }
                            };

                            let start_idx = available_frames - take_frames;
                            let data_slice = &full_data[start_idx..];
                            let conf_slice = &full_conf[start_idx..];

                            let ts_len = self.timestamps.len();
                            let slice_len = data_slice.len();
                            let ts_start = ts_len.saturating_sub(slice_len);

                            let actual_fs = if ts_len >= 2 && slice_len >= 2 {
                                let relevant_timestamps = &self.timestamps[ts_start..];
                                let duration = relevant_timestamps.last().unwrap() - relevant_timestamps.first().unwrap();
                                if duration > 0.0 {
                                    (relevant_timestamps.len() - 1) as f32 / duration as f32
                                } else {
                                    target_fs
                                }
                            } else {
                                target_fs
                            };

                            let slice_avg_conf = if !conf_slice.is_empty() {
                                conf_slice.iter().sum::<f32>() / conf_slice.len() as f32
                            } else {
                                0.0
                            };

                            let (val, conf) = match &cfg.method {
                                CalculationMethod::Rate(strategy) => {
                                    let bounds = crate::signal::rate::RateBounds { min: cfg.min_value, max: cfg.max_value };
                                    let res = crate::signal::rate::estimate_rate(
                                        data_slice, actual_fs, bounds, *strategy, None, Some(&mut self.fft_scratch)
                                    );                                    
                                    (res.value, slice_avg_conf)
                                },
                                CalculationMethod::HrvFromPeaks(metric) => {
                                    let ts_slice = &self.timestamps[start_idx..];
                                    let rate_hint = results.get("heart_rate").map(|(v, _)| *v);
                                    let bounds = SignalBounds { min_rate: 40.0, max_rate: 220.0 };
                                    crate::signal::hrv::estimate_hrv(
                                        data_slice,
                                        actual_fs,
                                        *metric,
                                        ts_slice,
                                        conf_slice,
                                        bounds,
                                        rate_hint
                                    )
                                },
                                CalculationMethod::Average => {
                                    let (avg_val, _) = crate::signal::calculate_average(data_slice);
                                    (avg_val, slice_avg_conf)
                                },
                                CalculationMethod::BpSystolic => {
                                    crate::signal::bp::extract_systolic_pressure(data_slice, actual_fs, conf_slice)
                                },
                                CalculationMethod::BpDiastolic => {
                                    crate::signal::bp::extract_diastolic_pressure(data_slice, actual_fs, conf_slice)
                                },
                                CalculationMethod::PulsePressure => {
                                    crate::signal::bp::extract_pulse_pressure(data_slice, actual_fs, conf_slice)
                                },
                                CalculationMethod::IeRatio => {
                                    crate::signal::resp::calculate_ie_ratio(data_slice, actual_fs, conf_slice)
                                },
                            };

                            if val >= cfg.min_value && val <= cfg.max_value {
                                results.insert(vital_id.clone(), (val, conf));
                                break;
                            }
                        }
                    }
                }
            }
        }
        results
    }

    fn construct_output(&mut self, mode: WaveformMode, scalar_results: HashMap<String, (f32, f32)>) -> SessionResult {
         
        let start_index = match mode {
            WaveformMode::Global => 0,
            WaveformMode::Windowed { seconds } => {
                let window_frames = (seconds * self.config.fps_target) as usize;
                self.timestamps.len().saturating_sub(window_frames)
            },
            WaveformMode::Incremental => {
                self.timestamps.iter().position(|&t| t > self.last_emitted_timestamp)
                    .unwrap_or(self.timestamps.len())
            }
        };

        if let Some(&last) = self.timestamps.last() {
            self.last_emitted_timestamp = last;
        }

        let slice_vec_f32 = |v: &[f32]| -> Vec<f32> {
            if start_index < v.len() { v[start_index..].to_vec() } else { Vec::new() }
        };
        
        let slice_vec_f64 = |v: &[f64]| -> Vec<f64> {
            if start_index < v.len() { v[start_index..].to_vec() } else { Vec::new() }
        };

        let slice_vec_coords = |v: &[[f32; 4]]| -> Vec<Vec<f32>> {
            if start_index < v.len() { 
                v[start_index..].iter().map(|arr| arr.to_vec()).collect()
            } else { 
                Vec::new() 
            }
        };

        let timestamp_slice = slice_vec_f64(&self.timestamps);
        let effective_fps = if timestamp_slice.len() > 1 {
            let duration = timestamp_slice.last().unwrap() - timestamp_slice.first().unwrap();
            if duration > 0.0 {
                (timestamp_slice.len() - 1) as f32 / duration as f32
            } else {
                self.config.fps_target
            }
        } else {
            self.config.fps_target
        };

        let mut signals_out = HashMap::new();
        let fs = self.config.fps_target;  

        for vital_id in &self.active_vitals {
            if let Some(meta) = self.vital_metas.get(vital_id) {
                
                let mut waveform_data = Vec::new();
                let mut waveform_conf = Vec::new();
                
                if let VitalType::Provided = meta.vital_type {
                    if let Some(buf) = self.signal_data.get(vital_id) {
                        let full_data = buf.compute_average();
                        
                        let processed_data = if let Some(proc_cfg) = &meta.processing {
                            if full_data.len() >= (proc_cfg.min_window_seconds * fs) as usize {
                                crate::signal::filters::apply_processing(&full_data, proc_cfg.operation, fs)
                            } else {
                                full_data
                            }
                        } else {
                            full_data
                        };
                        
                        waveform_data = slice_vec_f32(&processed_data);
                        
                        if let Some(c_buf) = self.signal_confs.get(vital_id) {
                            let full_conf = c_buf.compute_average();
                            waveform_conf = slice_vec_f32(&full_conf);
                        } else {
                            waveform_conf = vec![1.0; waveform_data.len()];
                        }
                    }
                }

                let (val, conf_scalar) = match scalar_results.get(vital_id) {
                    Some(&(v, c)) => (Some(v), vec![c]),
                    None => (None, Vec::new()),
                };
                
                if !waveform_data.is_empty() || val.is_some() {
                    signals_out.insert(vital_id.clone(), SignalResult {
                        value: val,
                        data: waveform_data,
                        confidence: if !waveform_conf.is_empty() { waveform_conf } else { conf_scalar },
                        unit: meta.unit.clone(),
                        note: None,
                    });
                }
            }
        }

        let face_result = if !self.face_coords.is_empty() {
            Some(FaceResult {
                coordinates: slice_vec_coords(&self.face_coords),
                confidence: slice_vec_f32(&self.face_conf),
                note: self.face_note.clone(),
            })
        } else {
            None
        };
        
        SessionResult {
            timestamp: slice_vec_f64(&self.timestamps),
            face: face_result,
            signals: signals_out,
            fps: effective_fps,
            message: "OK".to_string(),
            model_used: self.config.name.clone(),
        }
    }
}

// ========================================================================
// THE PUBLIC SHELL (Exported, FFI-safe types only, Interior Mutability)
// ========================================================================

#[derive(Debug)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Object))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct Session {
    core: Mutex<SessionCore>, 
}

// --- NATIVE & PYTHON IMPLEMENTATION ---
#[cfg_attr(not(target_arch = "wasm32"), uniffi::export)]
impl Session {
    #[cfg_attr(not(target_arch = "wasm32"), uniffi::constructor)]
    pub fn new(config: ModelConfig) -> Self {
        Self {
            core: Mutex::new(SessionCore::new(config))
        }
    }

    pub fn process_chunk(&self, chunk: InputChunk, mode: WaveformMode) -> SessionResult {
        let mut guard = self.core.lock().expect("Failed to lock Session core");
        guard.process_chunk(chunk, mode)
    }
}

// --- PYTHON SPECIFIC IMPLEMENTATION ---
#[cfg(feature = "python")]
#[pymethods]
impl Session {
    // Maps to __init__ in Python
    // Takes &ModelConfig because PyO3 can provide references to Python objects
    #[new]
    pub fn py_new(config: &ModelConfig) -> Self {
        Self {
            core: Mutex::new(SessionCore::new(config.clone()))
        }
    }

    // Re-expose process_chunk for Python (signature is the same, PyO3 handles conversion)
    #[pyo3(name = "process_chunk")]
    pub fn py_process_chunk(&self, chunk: InputChunk, mode: WaveformMode) -> SessionResult {
        let mut guard = self.core.lock().expect("Failed to lock Session core");
        guard.process_chunk(chunk, mode)
    }
}

// --- WASM SPECIFIC IMPLEMENTATION ---
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Session {
    #[wasm_bindgen(constructor)]
    pub fn new_js(config_val: JsValue) -> Result<Session, JsError> {
        let config: ModelConfig = serde_wasm_bindgen::from_value(config_val)?;
        Ok(Self {
            core: Mutex::new(SessionCore::new(config))
        })
    }

    #[wasm_bindgen(js_name = processChunkJs)]
    pub fn process_chunk_js(&self, chunk_val: JsValue, mode_val: JsValue) -> Result<JsValue, JsError> {
        let chunk: InputChunk = serde_wasm_bindgen::from_value(chunk_val)?;
        let mode: WaveformMode = serde_wasm_bindgen::from_value(mode_val)?;
        
        let result = self.process_chunk(chunk, mode); // Uses the method from the uniffi impl block
        
        Ok(serde_wasm_bindgen::to_value(&result)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::types::{FaceInput, WaveformMode}; // Ensure other types are imported if not already available in super

    // --- Mocks & Helpers ---

    fn mock_config(vitals: Vec<&str>) -> ModelConfig {
        ModelConfig {
            name: "test_model".to_string(),
            supported_vitals: vitals.iter().map(|s| s.to_string()).collect(),
            fps_target: 30.0,
            input_size: 30,
            roi_method: "face".to_string(),
        }
    }

    fn mock_chunk(
        times: Vec<f64>, 
        signals: Vec<(&str, Vec<f32>)>, 
        face: Option<FaceInput>
    ) -> InputChunk {
        let mut sig_map = HashMap::new();
        let mut conf_map = HashMap::new();
        
        for (key, data) in signals {
            let len = data.len();
            sig_map.insert(key.to_string(), data);
            conf_map.insert(key.to_string(), vec![1.0; len]);  
        }

        InputChunk {
            timestamp: times,
            signals: sig_map,
            confidences: conf_map,
            face,
        }
    }

    fn mock_sine(len: usize, fs: f32, freq_hz: f32) -> Vec<f32> {
        (0..len).map(|i| {
            let t = i as f32 / fs;
            (t * 2.0 * std::f32::consts::PI * freq_hz).sin()
        }).collect()
    }

    // --- Tests ---

    #[test]
    fn st_01_soft_stitching_averages_overlap() {
        let config = mock_config(vec!["ppg_waveform"]);
        let session = Session::new(config);

        let chunk1 = mock_chunk(
            vec![1.0, 2.0], 
            vec![("ppg_waveform", vec![10.0, 20.0])], 
            None
        );
        let _ = session.process_chunk(chunk1, WaveformMode::Global);

        let chunk2 = mock_chunk(
            vec![2.0, 3.0], 
            vec![("ppg_waveform", vec![22.0, 30.0])], 
            None
        );
        let result = session.process_chunk(chunk2, WaveformMode::Global);

        let ppg = result.signals.get("ppg_waveform").unwrap();
        
        assert_eq!(result.timestamp, vec![1.0, 2.0, 3.0]);
        // Average of 20.0 and 22.0 is 21.0
        assert!((ppg.data[1] - 21.0).abs() < 0.001, "Expected 21.0, got {}", ppg.data[1]);
    }

    #[test]
    fn st_02_disjoint_chunks_handle_gaps() {
        let config = mock_config(vec!["ppg_waveform"]);
        let session = Session::new(config);

        let chunk1 = mock_chunk(vec![1.0], vec![("ppg_waveform", vec![10.0])], None);
        let _ = session.process_chunk(chunk1, WaveformMode::Global);

        let chunk2 = mock_chunk(vec![5.0], vec![("ppg_waveform", vec![50.0])], None);
        let result = session.process_chunk(chunk2, WaveformMode::Global);

        assert_eq!(result.timestamp, vec![1.0, 5.0]);
        assert_eq!(result.signals["ppg_waveform"].data, vec![10.0, 50.0]);
    }

    #[test]
    fn st_03_exact_duplicate_chunks_ignored() {
        let config = mock_config(vec!["ppg_waveform"]);
        let session = Session::new(config);

        let chunk1 = mock_chunk(vec![1.0, 2.0], vec![("ppg_waveform", vec![10.0, 20.0])], None);
        let _ = session.process_chunk(chunk1, WaveformMode::Global);

        let chunk2 = mock_chunk(vec![1.0, 2.0], vec![("ppg_waveform", vec![10.0, 20.0])], None);
        let result = session.process_chunk(chunk2, WaveformMode::Global);

        assert_eq!(result.timestamp, vec![1.0, 2.0]);
        
        let ppg = &result.signals["ppg_waveform"].data;
        assert_eq!(ppg[0], 10.0);
        assert_eq!(ppg[1], 20.0);
    }

    #[test]
    fn st_04_nan_handling_in_signal() {
        let config = mock_config(vec!["ppg_waveform"]);
        let session = Session::new(config);

        let c1 = mock_chunk(vec![1.0], vec![("ppg_waveform", vec![10.0])], None);
        session.process_chunk(c1, WaveformMode::Global);

        let c2 = mock_chunk(vec![1.0], vec![("ppg_waveform", vec![f32::NAN])], None);
        let result = session.process_chunk(c2, WaveformMode::Global);
    
        let val = result.signals["ppg_waveform"].data[0];
        assert!(!val.is_nan());
        assert!((val - 10.0).abs() < 0.001);
    }

    #[test]
    fn st_05_pruning_limits_history() {
        let mut config = mock_config(vec!["ppg_waveform"]);
        config.fps_target = 1.0; 
        
        let session = Session::new(config);

        let times: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let ppg: Vec<f32> = vec![1.0; 100];
        
        let chunk = mock_chunk(times, vec![("ppg_waveform", ppg)], None);
        let result = session.process_chunk(chunk, WaveformMode::Incremental);

        // Expect truncation to 60s (default logic)
        assert_eq!(result.timestamp.len(), 60);
        assert_eq!(result.timestamp.first(), Some(&40.0));  
        assert_eq!(result.timestamp.last(), Some(&99.0));
    }

    #[test]
    fn reg_01_dependency_ordering() {
        // Needs heart_rate to calculate SDNN
        let config = mock_config(vec!["ppg_waveform", "hrv_sdnn", "heart_rate"]);
        let session = Session::new(config);

        let fs: f32 = 30.0;  
        let total_samples = 660;  
        let times: Vec<f64> = (0..total_samples).map(|i| i as f64 / fs as f64).collect();
        
        // Generate a signal that changes frequency to ensure HR detection works
        let mut ppg = Vec::new();
        let mut phase = 0.0;
        for i in 0..total_samples {
            let t = i as f32 / fs;  
            let current_freq = if t < 10.0 { 1.0 } else { 1.3 };
            phase += 2.0 * std::f32::consts::PI * current_freq / fs;
            let val = phase.sin().max(0.0).powf(4.0);
            ppg.push(val);
        }

        let chunk = mock_chunk(times, vec![("ppg_waveform", ppg)], None);
        let result = session.process_chunk(chunk, WaveformMode::Global);

        if !result.signals.contains_key("hrv_sdnn") {
            println!("Available signals: {:?}", result.signals.keys());
        }

        assert!(result.signals.contains_key("heart_rate"), "Heart Rate missing");
        assert!(result.signals.contains_key("hrv_sdnn"), "SDNN missing");
    }

    #[test]
    fn reg_02_minimum_data_gating() {
        let config = mock_config(vec!["ppg_waveform", "heart_rate"]);
        let session = Session::new(config);

        // 2 seconds of data (too short for HR)
        let times: Vec<f64> = (0..60).map(|i| i as f64 / 30.0).collect();
        let ppg = mock_sine(60, 30.0, 1.2);
        
        let chunk = mock_chunk(times, vec![("ppg_waveform", ppg)], None);
        let result = session.process_chunk(chunk, WaveformMode::Global);
        
        assert!(result.signals.contains_key("ppg_waveform"));
        assert!(!result.signals.contains_key("heart_rate"));
    }

    #[test]
    fn reg_03_alias_resolution() {
        // Request "pulse" (alias), expect "heart_rate" (canonical)
        let config = mock_config(vec!["ppg_waveform", "pulse"]);
        let session = Session::new(config);

        let times: Vec<f64> = (0..150).map(|i| i as f64 / 30.0).collect();
        let ppg = mock_sine(150, 30.0, 1.2);

        let chunk = mock_chunk(times, vec![("ppg_waveform", ppg)], None);
        let result = session.process_chunk(chunk, WaveformMode::Global);

        assert!(result.signals.contains_key("heart_rate"));
        assert!(!result.signals.contains_key("pulse"));
    }

    #[test]
    fn reg_04_provided_scalar_spo2() {
        // If "spo2" is provided as input, it should pass through
        let mut config = mock_config(vec!["spo2"]);
        config.fps_target = 1.0;  
        
        let session = Session::new(config);

        let chunk = mock_chunk(
            vec![1.0, 2.0, 3.0], 
            vec![("spo2", vec![98.0, 99.0, 98.0])], 
            None
        );
        let result = session.process_chunk(chunk, WaveformMode::Global);

        let sig = result.signals.get("spo2").unwrap();
        
        assert_eq!(sig.data, vec![98.0, 99.0, 98.0]);
        
        assert!(sig.value.is_some(), "Scalar value missing");
        let val = sig.value.unwrap();
        assert!((val - 98.333).abs() < 0.01);
    }

    #[test]
    fn reg_05_unsupported_vital() {
        let config = mock_config(vec!["blood_pressure"]);  
        let session = Session::new(config);

        let chunk = mock_chunk(vec![1.0], vec![], None);
        let result = session.process_chunk(chunk, WaveformMode::Global);

        assert!(!result.signals.contains_key("blood_pressure"));
    }

    #[test]
    fn out_01_incremental_mode() {
        let config = mock_config(vec!["ppg_waveform"]);
        let session = Session::new(config);

        let c1 = mock_chunk(vec![1.0], vec![("ppg_waveform", vec![10.0])], None);
        let r1 = session.process_chunk(c1, WaveformMode::Incremental);
        assert_eq!(r1.timestamp, vec![1.0]);

        let c2 = mock_chunk(vec![2.0], vec![("ppg_waveform", vec![20.0])], None);
        let r2 = session.process_chunk(c2, WaveformMode::Incremental);
    
        assert_eq!(r2.timestamp, vec![2.0]);
        assert_eq!(r2.signals["ppg_waveform"].data, vec![20.0]);
    }

    #[test]
    fn out_02_windowed_mode() {
        let config = mock_config(vec!["ppg_waveform"]);
        let session = Session::new(config);

        let times: Vec<f64> = (0..300).map(|i| i as f64 / 30.0).collect();
        let ppg = vec![0.0; 300];
        let chunk = mock_chunk(times, vec![("ppg_waveform", ppg)], None);

        let result = session.process_chunk(chunk, WaveformMode::Windowed { seconds: 2.0 });

        // Windowed mode should output the requested seconds (2.0s = 60 samples @ 30Hz)
        assert_eq!(result.timestamp.len(), 60);
        assert!(result.timestamp.last().unwrap() > &9.9);
    }

    #[test]
    fn out_03_global_mode() {
        let config = mock_config(vec!["spo2"]);
        let session = Session::new(config);

        let times1: Vec<f64> = (0..300).map(|i| i as f64 / 30.0).collect();
        let ppg1 = vec![1.0; 300];
        let chunk1 = mock_chunk(times1, vec![("spo2", ppg1)], None);
        let _ = session.process_chunk(chunk1, WaveformMode::Global);

        let times2: Vec<f64> = (300..360).map(|i| i as f64 / 30.0).collect();
        let ppg2 = vec![2.0; 60];
        let chunk2 = mock_chunk(times2, vec![("spo2", ppg2)], None);
        
        let result = session.process_chunk(chunk2, WaveformMode::Global);

        assert_eq!(result.timestamp.len(), 360);
        assert_eq!(result.timestamp.first(), Some(&0.0));
        
        let data = &result.signals["spo2"].data;
        assert_eq!(data[0], 1.0);     
        assert_eq!(data[359], 2.0);   
    }

    #[test]
    fn face_01_sync_with_signal() {
        let config = mock_config(vec!["ppg_waveform"]);
        let session = Session::new(config);

        let face = FaceInput {
            coordinates: vec![10.0, 10.0, 50.0, 50.0],
            confidence: 0.9,
        };

        let chunk = mock_chunk(
            vec![1.0, 2.0], 
            vec![("ppg_waveform", vec![10.0, 20.0])], 
            Some(face)
        );

        let result = session.process_chunk(chunk, WaveformMode::Global);

        assert!(result.face.is_some());
        let f = result.face.unwrap();
        assert_eq!(f.coordinates.len(), 2);  
        assert_eq!(f.confidence.len(), 2);
        assert_eq!(f.coordinates[0], vec![10.0, 10.0, 50.0, 50.0]);
    }

    #[test]
    fn face_02_missing_face_data_pads_zeros() {
        let config = mock_config(vec!["ppg_waveform"]);
        let session = Session::new(config);

        let chunk = mock_chunk(
            vec![1.0, 2.0], 
            vec![("ppg_waveform", vec![10.0, 20.0])], 
            None
        );

        let result = session.process_chunk(chunk, WaveformMode::Global);

        assert!(result.face.is_some());
        let f = result.face.unwrap();
        assert_eq!(f.coordinates.len(), 2);
        assert_eq!(f.coordinates[0], vec![0.0, 0.0, 0.0, 0.0]);
    }
}