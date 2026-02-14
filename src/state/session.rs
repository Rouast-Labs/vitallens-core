use std::collections::HashMap;
use crate::types::{InputChunk, ModelConfig, SessionResult, WaveformMode, SignalResult, FaceResult, FaceInput};
use crate::state::buffers::SignalBuffer;
use crate::registry::{self, VitalType, CalculationMethod};

pub struct Session {
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
}

impl Session {
    pub fn new(config: ModelConfig) -> Self {
        let max_history = (config.fps_target * 60.0) as usize;
        // Resolve aliases and identify supported vitals
        let mut resolved_metas = Vec::new();
        for vital_id in &config.supported_vitals {
            if let Some(meta) = registry::get_vital_meta(vital_id) {
                resolved_metas.push(meta);
            }
        }
        // Sort by order
        resolved_metas.sort_by_key(|m| {
            m.derivation.as_ref().map(|d| d.order).unwrap_or(0)
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
        }
    }

    pub fn process_chunk(&mut self, chunk: InputChunk, mode: WaveformMode) -> SessionResult {
        let overlap = self.merge_timestamps(&chunk.timestamp);
        self.merge_signals(chunk.signals, chunk.confidences, overlap);
        self.merge_face(chunk.face, overlap, chunk.timestamp.len());
        self.prune_state();

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
            None => new_times.len() // Fully overlapping
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
        let (coords, conf) = match face {
            Some(f) => (f.coordinates, f.confidence),
            None => ([0.0; 4], 0.0)
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

    fn perform_derivations(&self, mode: &WaveformMode) -> HashMap<String, (f32, f32)> {
        let mut results = HashMap::new();
        let fs = self.config.fps_target;

        for vital_id in &self.active_vitals {
            if let Some(meta) = registry::get_vital_meta(vital_id) {
                if let Some(cfg) = &meta.derivation {
                    
                    if let Some(source_buf) = self.signal_data.get(&cfg.source_signal) {
                        let full_data = source_buf.compute_average();
                        
                        let window_seconds = match mode {
                            WaveformMode::Complete => full_data.len() as f32 / fs,
                            _ => cfg.optimal_window_seconds,
                        };

                        let available_frames = full_data.len();
                        let min_frames = (cfg.min_required_seconds * fs) as usize;

                        if available_frames >= min_frames {
                            let take_frames = ((window_seconds * fs) as usize).min(available_frames);
                            let start_idx = available_frames - take_frames;
                            let slice = &full_data[start_idx..];

                            let (val, conf) = match &cfg.method {
                                CalculationMethod::Rate(strategy) => {
                                    // 1. Construct bounds from registry config
                                    let bounds = crate::signal::rate::RateBounds {
                                        min: cfg.min_value,
                                        max: cfg.max_value,
                                    };

                                    // 2. Call generic estimator
                                    let res = crate::signal::rate::estimate_rate(
                                        slice, fs, bounds, *strategy, None
                                    );
                                    
                                    (res.value, res.confidence)
                                },
                                CalculationMethod::HrvFromPeaks(metric) => {
                                    let current_hr = results.get("heart_rate").map(|(v, _)| *v);
                                    crate::signal::stubs::estimate_hrv(
                                        slice, fs, *metric, current_hr
                                    )
                                },
                                CalculationMethod::Average => {
                                    crate::signal::stubs::calculate_average(slice)
                                }
                            };

                            if val >= cfg.min_value && val <= cfg.max_value {
                                results.insert(vital_id.clone(), (val, conf));
                            }
                        }
                    }
                }
            }
        }
        results
    }

    fn construct_output(&mut self, mode: WaveformMode, scalar_results: HashMap<String, (f32, f32)>) -> SessionResult {
        // Determine start index based on mode
        let start_index = match mode {
            WaveformMode::Complete => 0,
            WaveformMode::Windowed(seconds) => {
                let window_frames = (seconds * self.config.fps_target) as usize;
                if self.timestamps.len() > window_frames {
                    self.timestamps.len() - window_frames
                } else {
                    0
                }
            },
            WaveformMode::Incremental => {
                match self.timestamps.iter().position(|&t| t > self.last_emitted_timestamp) {
                    Some(idx) => idx,
                    None => self.timestamps.len() 
                }
            }
        };

        if let Some(&last) = self.timestamps.last() {
            self.last_emitted_timestamp = last;
        }

        // 2. Slice data
        let slice_vec_f32 = |v: &[f32]| -> Vec<f32> {
            if start_index < v.len() { v[start_index..].to_vec() } else { Vec::new() }
        };
        
        let slice_vec_f64 = |v: &[f64]| -> Vec<f64> {
            if start_index < v.len() { v[start_index..].to_vec() } else { Vec::new() }
        };

        let slice_vec_coords = |v: &[[f32; 4]]| -> Vec<[f32; 4]> {
            if start_index < v.len() { v[start_index..].to_vec() } else { Vec::new() }
        };
      
        let mut signals_out = HashMap::new();

        for vital_id in &self.active_vitals {
            if let Some(meta) = registry::get_vital_meta(vital_id) {
                
                // 1. Handle waveform data
                let mut waveform_data = Vec::new();
                let mut waveform_conf = Vec::new();
                
                if let VitalType::Provided = meta.vital_type {
                    if let Some(buf) = self.signal_data.get(vital_id) {
                        let full_data = buf.compute_average();
                        // Post-Processing?
                        let processed_data = if let Some(proc_cfg) = &meta.processing {
                             if full_data.len() >= (proc_cfg.min_window_seconds * self.config.fps_target) as usize {
                                 crate::signal::stubs::apply_processing(&full_data, proc_cfg.operation)
                             } else {
                                 full_data
                             }
                        } else {
                            full_data
                        };
                        
                        waveform_data = slice_vec_f32(&processed_data); // Slice for output
                        
                        // Handle Confidence
                        if let Some(c_buf) = self.signal_confs.get(vital_id) {
                             waveform_conf = slice_vec_f32(&c_buf.compute_average());
                        } else {
                             waveform_conf = vec![1.0; waveform_data.len()];
                        }
                    }
                }

                // 2. Handle scalar value
                let (val, conf_scalar) = match scalar_results.get(vital_id) {
                    Some(&(v, c)) => (Some(v), vec![c]),
                    None => (None, Vec::new()),
                };
                
                // Only insert if we have *something* (waveform OR value)
                if !waveform_data.is_empty() || val.is_some() {
                    signals_out.insert(vital_id.clone(), SignalResult {
                        value: val,
                        data: waveform_data,
                        confidence: if !waveform_conf.is_empty() { waveform_conf } else { conf_scalar },
                        unit: meta.unit,
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
            fps: self.config.fps_target, // TODO: Compute actual fps
            message: "OK".to_string(),
            model_used: self.config.name.clone(),
        }
    }
}