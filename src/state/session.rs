use std::collections::HashMap;
use std::sync::Mutex;
use crate::types::{InputChunk, SessionConfig, SessionResult, WaveformMode, SignalResult, FaceResult, FaceInput};
use crate::state::series::SignalBuffer;
use crate::registry::{self, VitalType, CalculationMethod, VitalMeta};
use crate::signal::fft::FftScratch;
use crate::signal::peaks::SignalBounds;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[derive(Debug)]
struct SessionCore {
    config: SessionConfig,
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
    fn new(config: SessionConfig) -> Self {
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
            self.prune_state();
        }

        let derived_results = self.perform_derivations(&mode);

        self.construct_output(mode, derived_results)
    }

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
        // Clone vital IDs to avoid borrowing issues while iterating
        let vital_ids: Vec<String> = self.active_vitals.clone();

        for vital_id in vital_ids {
            if let Some(meta) = self.vital_metas.get(&vital_id).cloned() { 
                for cfg in &meta.derivations {
                    
                    // Gather sources
                    let mut source_data_bufs = Vec::new();
                    let mut source_conf_bufs = Vec::new();
                    let mut missing_source = false;

                    for source_id in &cfg.sources {
                        if let Some(buf) = self.signal_data.get(source_id) {
                            source_data_bufs.push(buf.compute_average());
                            
                            if let Some(c_buf) = self.signal_confs.get(source_id) {
                                source_conf_bufs.push(c_buf.compute_average());
                            } else {
                                source_conf_bufs.push(Vec::new()); 
                            }
                        } else {
                            missing_source = true;
                            break;
                        }
                    }

                    if missing_source || source_data_bufs.is_empty() {
                        continue;
                    }

                    // Align & window
                    let available_len = source_data_bufs.iter().map(|v| v.len()).min().unwrap_or(0);
                    let target_fs = self.config.fps_target;
                    let min_frames = (cfg.min_window_seconds * target_fs) as usize;

                    if available_len < min_frames {
                        continue;
                    }

                    let take_frames = match mode {
                        WaveformMode::Global => available_len,
                        _ => {
                            let pref = (cfg.preferred_window_seconds * target_fs) as usize;
                            pref.min(available_len)
                        }
                    };

                    // Slice the inputs to the latest `take_frames`
                    let mut inputs: Vec<&[f32]> = Vec::with_capacity(source_data_bufs.len());
                    let mut confs: Vec<&[f32]> = Vec::with_capacity(source_conf_bufs.len());

                    let fallback_conf = vec![1.0; take_frames];

                    for i in 0..source_data_bufs.len() {
                        let data_vec = &source_data_bufs[i];
                        let start = data_vec.len() - take_frames;
                        inputs.push(&data_vec[start..]);

                        if !source_conf_bufs[i].is_empty() {
                            let conf_vec = &source_conf_bufs[i];
                            let c_start = conf_vec.len().saturating_sub(take_frames);
                            confs.push(&conf_vec[c_start..]);
                        } else {
                            confs.push(&fallback_conf);
                        }
                    }

                    // Calculate actual FS from timestamps (using the slice duration)
                    let ts_start_idx = self.timestamps.len().saturating_sub(take_frames);
                    let ts_slice = if ts_start_idx < self.timestamps.len() {
                        &self.timestamps[ts_start_idx..]
                    } else { &[] };

                    let actual_fs = if ts_slice.len() > 1 {
                        let duration = ts_slice.last().unwrap() - ts_slice.first().unwrap();
                        if duration > 0.0 { (ts_slice.len() - 1) as f32 / duration as f32 } else { target_fs }
                    } else {
                        target_fs
                    };

                    // Calculate average confidence of the primary source
                    let slice_avg_conf = if !confs.is_empty() {
                        confs[0].iter().sum::<f32>() / confs[0].len() as f32
                    } else { 0.0 };

                    // Execute calculation
                    let (val, conf) = match &cfg.method {
                        CalculationMethod::Rate(strategy) => {
                            let bounds = crate::signal::rate::RateBounds { min: cfg.min_value, max: cfg.max_value };
                            // Use rate hint from previously calculated Heart Rate if available
                            let hint = results.get("heart_rate").map(|(v, _)| *v);
                            
                            let res = crate::signal::rate::estimate_rate(
                                inputs[0], actual_fs, bounds, *strategy, hint, Some(&mut self.fft_scratch)
                            );                                    
                            (res.value, slice_avg_conf)
                        },
                        CalculationMethod::HrvFromPeaks(metric) => {
                            let rate_hint = results.get("heart_rate").map(|(v, _)| *v);
                            // Get HR bounds from registry to ensure consistency
                            let bounds = if let Some(hr_meta) = registry::get_vital_meta("heart_rate") {
                                let d = &hr_meta.derivations[0];
                                SignalBounds { min_rate: d.min_value, max_rate: d.max_value }
                            } else {
                                SignalBounds { min_rate: 40.0, max_rate: 220.0 }
                            };

                            crate::signal::hrv::estimate_hrv(
                                inputs[0], actual_fs, *metric, ts_slice, confs[0], bounds, rate_hint
                            )
                        },
                        CalculationMethod::Average => {
                            let (avg_val, _) = crate::signal::calculate_average(inputs[0]);
                            (avg_val, slice_avg_conf)
                        },
                        CalculationMethod::BpSystolic => {
                            crate::signal::bp::extract_systolic_pressure(inputs[0], actual_fs, confs[0])
                        },
                        CalculationMethod::BpDiastolic => {
                            crate::signal::bp::extract_diastolic_pressure(inputs[0], actual_fs, confs[0])
                        },
                        CalculationMethod::PulsePressureFromSignal => {
                            crate::signal::bp::extract_pulse_pressure(inputs[0], actual_fs, confs[0])
                        },
                        CalculationMethod::PulsePressureFromScalars => {
                            if inputs.len() >= 2 {
                                crate::signal::bp::calculate_pp_from_signals(inputs[0], inputs[1])
                            } else {
                                (0.0, 0.0)
                            }
                        },
                        CalculationMethod::MapFromScalars => {
                            if inputs.len() >= 2 {
                                crate::signal::bp::calculate_map_from_signals(inputs[0], inputs[1])
                            } else {
                                (0.0, 0.0)
                            }
                        },
                        CalculationMethod::IeRatio => {
                            let rate_hint = results.get("respiratory_rate").map(|(v, _)| *v);
                            let rr_meta = registry::get_vital_meta("respiratory_rate").unwrap();
                            let rr_deriv = &rr_meta.derivations[0];
                            let bounds = SignalBounds { 
                                min_rate: rr_deriv.min_value, 
                                max_rate: rr_deriv.max_value 
                            };
                            crate::signal::resp::calculate_ie_ratio(inputs[0], actual_fs, confs[0], bounds, rate_hint)
                        }
                    };

                    if val >= cfg.min_value && val <= cfg.max_value {
                        results.insert(vital_id.clone(), (val, conf));
                        // Break derivation loop for this vital (successful)
                        break; 
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
                                crate::signal::filters::apply_processing(
                                    &full_data, 
                                    proc_cfg.operation, 
                                    fs,
                                    proc_cfg.min_freq,
                                    proc_cfg.max_freq
                                )
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

// --- FFI WRAPPERS ---

#[derive(Debug)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Object))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct Session {
    core: Mutex<SessionCore>, 
}

#[cfg_attr(not(target_arch = "wasm32"), uniffi::export)]
impl Session {
    #[cfg_attr(not(target_arch = "wasm32"), uniffi::constructor)]
    pub fn new(config: SessionConfig) -> Self {
        Self {
            core: Mutex::new(SessionCore::new(config))
        }
    }

    pub fn process_chunk(&self, chunk: InputChunk, mode: WaveformMode) -> SessionResult {
        let mut guard = self.core.lock().expect("Failed to lock Session core");
        guard.process_chunk(chunk, mode)
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Session {
    #[new]
    pub fn py_new(config: &SessionConfig) -> Self {
        Self {
            core: Mutex::new(SessionCore::new(config.clone()))
        }
    }

    #[pyo3(name = "process_chunk")]
    pub fn py_process_chunk(&self, chunk: InputChunk, mode: WaveformMode) -> SessionResult {
        let mut guard = self.core.lock().expect("Failed to lock Session core");
        guard.process_chunk(chunk, mode)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Session {
    #[wasm_bindgen(constructor)]
    pub fn new_js(config_val: JsValue) -> Result<Session, JsError> {
        let config: SessionConfig = serde_wasm_bindgen::from_value(config_val)?;
        Ok(Self {
            core: Mutex::new(SessionCore::new(config))
        })
    }

    #[wasm_bindgen(js_name = processChunkJs)]
    pub fn process_chunk_js(&self, chunk_val: JsValue, mode_val: JsValue) -> Result<JsValue, JsError> {
        let chunk: InputChunk = serde_wasm_bindgen::from_value(chunk_val)?;
        let mode: WaveformMode = serde_wasm_bindgen::from_value(mode_val)?;
        
        let result = self.process_chunk(chunk, mode);  
        
        Ok(serde_wasm_bindgen::to_value(&result)?)
    }
}