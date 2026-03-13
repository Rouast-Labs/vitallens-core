use std::collections::HashMap;
use std::sync::Mutex;
use crate::types::{SessionInput, SessionConfig, SessionResult, WaveformMode, WaveformResult, VitalResult, FaceResult, FaceInput, SignalInput};
use crate::state::series::SignalBuffer;
use crate::registry::{self, VitalType, CalculationMethod, VitalMeta, DerivationConfig};
use crate::signal::fft::FftScratch;
use crate::signal::peaks::SignalBounds;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// Internal state engine for a continuous tracking session.
/// Manages historical buffers, timestamps, and calculates vitals iteratively.
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
    /// Initializes a new internal session core.
    ///
    /// # Arguments
    /// * `config` - The configuration parameters for the session.
    ///
    /// # Returns
    /// A new `SessionCore` instance.
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

    /// Processes a new batch of incoming data, updates historical buffers, and derives vitals.
    ///
    /// # Arguments
    /// * `input` - The latest batch of input data including signals, face coordinates, and timestamps.
    /// * `mode` - The operational mode determining how much output history is returned.
    ///
    /// # Returns
    /// A `SessionResult` containing the calculated vitals, waveforms, and payload metadata.
    fn process(&mut self, input: SessionInput, mode: WaveformMode) -> SessionResult {
        if let Err(msg) = input.validate_lengths() {
            log::error!("[Session] Data mismatch: {}", msg);
            return SessionResult {
                timestamp: Vec::new(),
                face: None,
                waveforms: HashMap::new(),
                vitals: HashMap::new(),
                rolling_vitals: None,
                fps: self.config.fps_target,
                message: msg,
            };
        }

        let input_len = input.timestamp.len();
        let overlap = self.merge_timestamps(&input.timestamp);
        self.merge_signals(input.signals, overlap);
        self.merge_face(input.face, overlap, input_len);
        
        if !matches!(mode, WaveformMode::Global) {
            self.prune_state();
        }

        let derived_results = self.perform_derivations(&mode);
        let rolling_results = self.perform_rolling_derivations(&mode, &derived_results);

        self.construct_output(mode, derived_results, rolling_results)
    }

    /// Merges new timestamps into the state, determining how many frames overlap with existing history.
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

    /// Appends new signal vectors into their respective running buffers using the calculated overlap.
    fn merge_signals(&mut self, signals: HashMap<String, SignalInput>, overlap: usize) {
        for (key, input) in signals {
            let data_buf = self.signal_data.entry(key.clone()).or_insert_with(SignalBuffer::new);
            data_buf.merge(&input.data, overlap, Some("unitless".to_string()));

            let conf_buf = self.signal_confs.entry(key).or_insert_with(SignalBuffer::new);
            conf_buf.merge(&input.confidence, overlap, None);
        }
    }

    /// Merges bounding box coordinates into the historical tracking array.
    fn merge_face(&mut self, face: Option<FaceInput>, overlap: usize, input_len: usize) {
        let new_frames_count = if input_len > overlap { input_len - overlap } else { 0 };
        if new_frames_count == 0 { return; }

        let start_idx = overlap;

        match face {
            Some(f) => {
                for i in start_idx..input_len {
                    let coords_vec = &f.coordinates[i];
                    let coords: [f32; 4] = if coords_vec.len() >= 4 {
                        [coords_vec[0], coords_vec[1], coords_vec[2], coords_vec[3]]
                    } else {
                        [0.0; 4]
                    };
                    self.face_coords.push(coords);
                    self.face_conf.push(f.confidence[i]);
                }
            },
            None => {
                for _ in 0..new_frames_count {
                    self.face_coords.push([0.0; 4]);
                    self.face_conf.push(0.0);
                }
            }
        }
    }

    /// Trims internal buffers to the maximum configured history limit.
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

    /// Evaluates a specific vital sign derivation logic over a provided data window.
    ///
    /// # Arguments
    /// * `cfg` - The derivation configuration defining the method and bounds.
    /// * `inputs` - A slice of input signal data slices.
    /// * `confs` - A slice of confidence score slices corresponding to the inputs.
    /// * `ts_slice` - The timestamp slice for the current window.
    /// * `actual_fs` - The calculated sampling frequency for this window.
    /// * `slice_avg_conf` - The average confidence score across the slice.
    /// * `global_results` - A map of already computed global vitals for cross-referencing.
    /// * `fft_scratch` - A mutable reference to an FFT scratchpad for memory reuse.
    ///
    /// # Returns
    /// A tuple of `(calculated_value, confidence_score)`.
    fn evaluate_derivation(
        cfg: &DerivationConfig,
        inputs: &[&[f32]],
        confs: &[&[f32]],
        ts_slice: &[f64],
        actual_fs: f32,
        slice_avg_conf: f32,
        global_results: &HashMap<String, (f32, f32)>,
        fft_scratch: &mut FftScratch,
    ) -> (f32, f32) {
        match &cfg.method {
            CalculationMethod::Rate(strategy) => {
                let bounds = crate::signal::rate::RateBounds { min: cfg.min_value, max: cfg.max_value };
                let hint = global_results.get("heart_rate").map(|(v, _)| *v);
                let res = crate::signal::rate::estimate_rate(
                    inputs[0], actual_fs, bounds, strategy.clone(), hint, Some(fft_scratch)
                );
                (res.value, slice_avg_conf)
            },
            CalculationMethod::HrvFromPeaks(metric) => {
                let rate_hint = global_results.get("heart_rate").map(|(v, _)| *v);
                let bounds = if let Some(hr_meta) = registry::get_vital_meta("heart_rate") {
                    let d = &hr_meta.derivations[0];
                    SignalBounds { min_rate: d.min_value, max_rate: d.max_value }
                } else {
                    SignalBounds { min_rate: 40.0, max_rate: 220.0 }
                };

                crate::signal::hrv::estimate_hrv(
                    inputs[0], actual_fs, metric.clone(), ts_slice, confs[0], bounds, rate_hint
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
                let rate_hint = global_results.get("respiratory_rate").map(|(v, _)| *v);
                let rr_meta = registry::get_vital_meta("respiratory_rate").unwrap();
                let rr_deriv = &rr_meta.derivations[0];
                let bounds = SignalBounds {
                    min_rate: rr_deriv.min_value,
                    max_rate: rr_deriv.max_value
                };
                crate::signal::resp::calculate_ie_ratio(inputs[0], actual_fs, confs[0], bounds, rate_hint)
            }
        }
    }

    /// Executes the required mathematical derivations for all active vitals based on current buffers.
    ///
    /// # Arguments
    /// * `mode` - The extraction mode, dictating how much data to feed into the derivation algorithms.
    ///
    /// # Returns
    /// A `HashMap` mapping vital IDs to their `(value, confidence)` scalar tuples.
    fn perform_derivations(&mut self, mode: &WaveformMode) -> HashMap<String, (f32, f32)> {
        let mut results = HashMap::new();
        let vital_ids: Vec<String> = self.active_vitals.clone();

        for vital_id in vital_ids {
            if let Some(meta) = self.vital_metas.get(&vital_id).cloned() { 
                for cfg in &meta.derivations {
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

                    let slice_avg_conf = if !confs.is_empty() {
                        confs[0].iter().sum::<f32>() / confs[0].len() as f32
                    } else { 0.0 };

                    let (val, conf) = SessionCore::evaluate_derivation(
                        cfg, &inputs, &confs, ts_slice, actual_fs, slice_avg_conf, &results, &mut self.fft_scratch
                    );

                    if val >= cfg.min_value && val <= cfg.max_value {
                        results.insert(vital_id.clone(), (val, conf));
                        break; 
                    }
                }
            }
        }
        results
    }

    /// Computes vital signs using a sliding window across the entire session history.
    ///
    /// # Arguments
    /// * `mode` - The extraction mode (only executed if Global).
    /// * `global_results` - Pre-computed scalar vitals used as hints for detection.
    ///
    /// # Returns
    /// An optional map of `WaveformResult` containing the interpolated rolling vitals.
    fn perform_rolling_derivations(
        &mut self,
        mode: &WaveformMode,
        global_results: &HashMap<String, (f32, f32)>
    ) -> Option<HashMap<String, WaveformResult>> {

        if !self.config.estimate_rolling_vitals.unwrap_or(false) || !matches!(mode, WaveformMode::Global) {
            return None;
        }

        let mut results = HashMap::new();
        let n_frames = self.timestamps.len();
        if n_frames == 0 { return None; }

        let vital_ids: Vec<String> = self.active_vitals.clone();

        for vital_id in vital_ids {
            if let Some(meta) = self.vital_metas.get(&vital_id).cloned() {
                for cfg in &meta.derivations {

                    let mut source_data_bufs = Vec::new();
                    let mut source_conf_bufs = Vec::new();
                    let mut missing_source = false;

                    for source_id in &cfg.sources {
                        if let Some(buf) = self.signal_data.get(source_id) {
                            source_data_bufs.push(buf.compute_average());
                            if let Some(c_buf) = self.signal_confs.get(source_id) {
                                source_conf_bufs.push(c_buf.compute_average());
                            } else {
                                source_conf_bufs.push(vec![1.0; n_frames]); 
                            }
                        } else {
                            missing_source = true;
                            break;
                        }
                    }

                    if missing_source || source_data_bufs.is_empty() || source_data_bufs[0].len() != n_frames {
                        continue;
                    }

                    let target_fs = self.config.fps_target;
                    let max_window_samples = (cfg.preferred_window_seconds * target_fs) as usize;
                    let min_window_samples = (cfg.min_window_seconds * target_fs) as usize;
                    let stride_samples = (cfg.rolling_stride_seconds * target_fs).round() as usize;
                    let stride_samples = stride_samples.max(1);

                    if n_frames < min_window_samples {
                        continue;
                    }

                    let mut out_data = vec![f32::NAN; n_frames];
                    let mut out_conf = vec![f32::NAN; n_frames];

                    // We will keep track of calculated points for interpolation
                    let mut calc_indices = Vec::new();
                    let mut calc_vals = Vec::new();
                    let mut calc_confs = Vec::new();

                    // Step through the array using the defined stride
                    let mut i = min_window_samples - 1;
                    while i < n_frames {
                        let start_idx = i.saturating_sub(max_window_samples - 1);
                        let end_idx = i + 1;

                        let ts_slice = &self.timestamps[start_idx..end_idx];
                        let actual_fs = if ts_slice.len() > 1 {
                            let duration = ts_slice.last().unwrap() - ts_slice.first().unwrap();
                            if duration > 0.0 { (ts_slice.len() - 1) as f32 / duration as f32 } else { target_fs }
                        } else {
                            target_fs
                        };

                        let mut inputs: Vec<&[f32]> = Vec::with_capacity(source_data_bufs.len());
                        let mut confs: Vec<&[f32]> = Vec::with_capacity(source_conf_bufs.len());

                        for j in 0..source_data_bufs.len() {
                            inputs.push(&source_data_bufs[j][start_idx..end_idx]);
                            confs.push(&source_conf_bufs[j][start_idx..end_idx]);
                        }

                        let slice_avg_conf = confs[0].iter().sum::<f32>() / confs[0].len() as f32;

                        let (val, conf) = SessionCore::evaluate_derivation(
                            cfg, &inputs, &confs, ts_slice, actual_fs, slice_avg_conf, global_results, &mut self.fft_scratch
                        );

                        if val >= cfg.min_value && val <= cfg.max_value {
                            calc_indices.push(i as f32);
                            calc_vals.push(val);
                            calc_confs.push(conf);

                            // Also drop it directly into the output arrays
                            out_data[i] = val;
                            out_conf[i] = conf;
                        }

                        // Always calculate the very last frame, otherwise jump by stride
                        if i == n_frames - 1 {
                            break;
                        }
                        i += stride_samples;
                        if i >= n_frames {
                            i = n_frames - 1;
                        }
                    }

                    // Linearly interpolate the gaps between calculated strides
                    if calc_indices.len() > 1 {
                        let all_indices: Vec<f32> = (0..n_frames).map(|x| x as f32).collect();

                        // We only interpolate from the first calculated point to the end.
                        // Everything before min_window_samples remains NaN.
                        let first_calc_idx = calc_indices[0] as usize;
                        let target_indices = &all_indices[first_calc_idx..];

                        // Interpolate data and confidence
                        let interp_data = crate::signal::interp_linear_1d(&calc_indices, &calc_vals, target_indices);
                        let interp_conf = crate::signal::interp_linear_1d(&calc_indices, &calc_confs, target_indices);

                        out_data[first_calc_idx..].copy_from_slice(&interp_data);
                        out_conf[first_calc_idx..].copy_from_slice(&interp_conf);
                    }

                    results.insert(vital_id.clone(), WaveformResult {
                        data: out_data,
                        confidence: out_conf,
                        unit: meta.unit.clone(),
                        note: format!("Rolling estimate of {} with frame-wise confidence levels.", meta.display_name),
                    });
                    break;
                }
            }
        }

        if results.is_empty() { None } else { Some(results) }
    }

    /// Bundles the evaluated results and requested waveforms into the final returned struct.
    ///
    /// # Arguments
    /// * `mode` - The operational mode determining the span of the returned output.
    /// * `scalar_results` - Pre-computed vital sign values.
    /// * `rolling_results` - Pre-computed rolling vitals results (optional).
    ///
    /// # Returns
    /// The formatted `SessionResult`.
    fn construct_output(
        &mut self, 
        mode: WaveformMode, 
        scalar_results: HashMap<String, (f32, f32)>,
        rolling_results: Option<HashMap<String, WaveformResult>>
    ) -> SessionResult {
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

        let mut waveforms_out = HashMap::new();
        let mut vitals_out = HashMap::new();

        let is_global = matches!(mode, WaveformMode::Global);
        let return_waveforms = self.config.return_waveforms.clone().unwrap_or_default();

        let actual_global_fs = if self.timestamps.len() > 1 {
            let duration = self.timestamps.last().unwrap() - self.timestamps.first().unwrap();
            if duration > 0.0 {
                (self.timestamps.len() - 1) as f32 / duration as f32
            } else {
                self.config.fps_target
            }
        } else {
            self.config.fps_target
        };

        for vital_id in return_waveforms {
            if let Some(meta) = registry::get_vital_meta(&vital_id) {
                if let VitalType::Provided = meta.vital_type {
                    if let Some(buf) = self.signal_data.get(&vital_id) {
                        let full_data = buf.compute_average();
                        
                        let processed_data = if let Some(proc_cfg) = &meta.processing {
                            if full_data.len() >= (proc_cfg.min_window_seconds * actual_global_fs) as usize {
                                crate::signal::filters::apply_processing(
                                    &full_data, 
                                    proc_cfg.operation, 
                                    actual_global_fs,  
                                    proc_cfg.min_freq,
                                    proc_cfg.max_freq
                                )
                            } else {
                                full_data
                            }
                        } else {
                            full_data
                        };
                        
                        let waveform_data = slice_vec_f32(&processed_data);
                        
                        let waveform_conf = if let Some(c_buf) = self.signal_confs.get(&vital_id) {
                            let full_conf = c_buf.compute_average();
                            slice_vec_f32(&full_conf)
                        } else {
                            vec![1.0; waveform_data.len()]
                        };

                        if !waveform_data.is_empty() {
                            let note = if is_global {
                                format!("Global estimate of {} with frame-wise confidence scores using {}.", meta.display_name, self.config.model_name)
                            } else {
                                format!("Latest estimate of {} with frame-wise confidence scores using {}.", meta.display_name, self.config.model_name)
                            };

                            waveforms_out.insert(vital_id.clone(), WaveformResult {
                                data: waveform_data,
                                confidence: waveform_conf,
                                unit: meta.unit.clone(),
                                note,
                            });
                        }
                    }
                }
            }
        }

        for vital_id in &self.active_vitals {
            if let Some(meta) = self.vital_metas.get(vital_id) {
                if let Some(&(val, conf_scalar)) = scalar_results.get(vital_id) {
                    let note = if is_global {
                        format!("Global estimate of {} with confidence score using {}.", meta.display_name, self.config.model_name)
                    } else {
                        format!("Latest estimate of {} with confidence score using {}.", meta.display_name, self.config.model_name)
                    };

                    vitals_out.insert(vital_id.clone(), VitalResult {
                        value: val,
                        confidence: conf_scalar,
                        unit: meta.unit.clone(),
                        note,
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
            waveforms: waveforms_out,
            vitals: vitals_out,
            rolling_vitals: rolling_results,
            fps: effective_fps,
            message: "The provided values are estimates and should be interpreted according to the provided confidence scores. The VitalLens API is not a medical device and its estimates are not intended for any medical purposes.".to_string(),
        }
    }

    /// Hard resets the core state.
    fn reset(&mut self) {
        self.timestamps.clear();
        self.face_coords.clear();
        self.face_conf.clear();
        self.face_note = None;
        self.last_emitted_timestamp = -1.0;
        
        for buf in self.signal_data.values_mut() {
            buf.clear();
        }
        for buf in self.signal_confs.values_mut() {
            buf.clear();
        }
    }
}

/// A thread-safe, stateful session object for processing continuous frame data natively across FFI and Wasm.
#[derive(Debug)]
#[cfg_attr(feature = "python", pyclass)]
#[cfg_attr(not(target_arch = "wasm32"), derive(uniffi::Object))]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct Session {
    core: Mutex<SessionCore>, 
}

#[cfg_attr(not(target_arch = "wasm32"), uniffi::export)]
impl Session {
    /// Creates a new stateful session using the provided configuration.
    ///
    /// # Arguments
    /// * `config` - The configuration parameters dictating target frame rate and model metadata.
    ///
    /// # Returns
    /// A new instance of `Session`.
    #[cfg_attr(not(target_arch = "wasm32"), uniffi::constructor)]
    pub fn new(config: SessionConfig) -> Self {
        Self {
            core: Mutex::new(SessionCore::new(config))
        }
    }

    /// Processes a new batch of inputs, updating the session's internal state
    /// and returning the calculated vital signs and waveforms.
    ///
    /// # Arguments
    /// * `input` - The latest batch of input data (signals, face coordinates, timestamps).
    /// * `mode` - The extraction mode dictating the payload size (Incremental, Windowed, or Global).
    ///
    /// # Returns
    /// A `SessionResult` containing the estimated vitals, waveforms, and metadata.
    pub fn process(&self, input: SessionInput, mode: WaveformMode) -> SessionResult {
        let mut guard = self.core.lock().expect("Failed to lock Session core");
        guard.process(input, mode)
    }

    /// Resets the internal state of the session, clearing all historical buffers.
    pub fn reset(&self) {
        let mut guard = self.core.lock().expect("Failed to lock Session core");
        guard.reset();
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

    #[pyo3(name = "process")]
    pub fn py_process(&self, input: SessionInput, mode: WaveformMode) -> SessionResult {
        let mut guard = self.core.lock().expect("Failed to lock Session core");
        guard.process(input, mode)
    }

    #[pyo3(name = "reset")]
    pub fn py_reset(&self) {
        self.reset();
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

    #[wasm_bindgen(js_name = processJs)]
    pub fn process_js(&self, input_val: JsValue, mode_val: JsValue) -> Result<JsValue, JsError> {
        let input: SessionInput = serde_wasm_bindgen::from_value(input_val)?;
        let mode: WaveformMode = serde_wasm_bindgen::from_value(mode_val)?;
        
        let result = self.process(input, mode);  
        
        Ok(serde_wasm_bindgen::to_value(&result)?)
    }

    #[wasm_bindgen(js_name = reset)]
    pub fn reset_js(&self) {
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn mock_config(fps: f32) -> SessionConfig {
        SessionConfig {
            model_name: "test-model".to_string(),
            supported_vitals: vec!["heart_rate".to_string()],
            return_waveforms: Some(vec!["abp_waveform".to_string()]),
            fps_target: fps,
            input_size: 30,
            n_inputs: 4,
            roi_method: "face".to_string(),
            estimate_rolling_vitals: None,
        }
    }

    fn mock_input(timestamps: Vec<f64>, signal_data: Vec<f32>) -> SessionInput {
        let mut signals = HashMap::new();
        signals.insert(
            "abp_waveform".to_string(),
            SignalInput {
                data: signal_data.clone(),
                confidence: vec![1.0; signal_data.len()],
            },
        );

        SessionInput {
            timestamp: timestamps,
            signals,
            face: None,
        }
    }

    #[test]
    fn test_session_validation_mismatch() {
        let session = Session::new(mock_config(30.0));
        
        let input = mock_input(vec![0.0, 1.0, 2.0], vec![1.0, 2.0]);
        let result = session.process(input, WaveformMode::Global);

        assert!(result.timestamp.is_empty(), "Result should be empty on failure");
        assert!(result.message.contains("Length mismatch"), "Should return length mismatch error");
    }

    #[test]
    fn test_session_incremental_mode() {
        let session = Session::new(mock_config(1.0));

        let input1 = mock_input(vec![0.0, 1.0, 2.0], vec![10.0, 11.0, 12.0]);
        let res1 = session.process(input1, WaveformMode::Incremental);
        
        assert_eq!(res1.timestamp.len(), 3, "Initial incremental should return all 3 frames");

        let input2 = mock_input(vec![1.0, 2.0, 3.0], vec![11.0, 12.0, 13.0]);
        let res2 = session.process(input2, WaveformMode::Incremental);

        assert_eq!(res2.timestamp.len(), 1, "Incremental should only return the 1 strictly new frame");
        assert_eq!(res2.timestamp[0], 3.0);
        assert_eq!(res2.waveforms["abp_waveform"].data[0], 13.0);
    }

    #[test]
    fn test_session_windowed_mode() {
        let session = Session::new(mock_config(1.0));  

        let input = mock_input(
            vec![0.0, 1.0, 2.0, 3.0, 4.0], 
            vec![10.0, 20.0, 30.0, 40.0, 50.0]
        );
        
        let result = session.process(input, WaveformMode::Windowed { seconds: 3.0 });

        assert_eq!(result.timestamp.len(), 3, "Windowed mode should restrict output to exactly 3 frames");
        assert_eq!(result.timestamp, vec![2.0, 3.0, 4.0], "Should return the most recent timestamps");
        assert_eq!(result.waveforms["abp_waveform"].data, vec![30.0, 40.0, 50.0]);
    }

    #[test]
    fn test_session_history_pruning() {
        let session = Session::new(mock_config(1.0));

        let timestamps: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let data: Vec<f32> = (0..100).map(|x| x as f32).collect();
        
        let input = mock_input(timestamps, data);
        
        let result = session.process(input, WaveformMode::Incremental);

        assert_eq!(
            result.timestamp.len(), 
            60, 
            "Session should aggressively prune data exceeding the 60-second max history"
        );
        assert_eq!(
            result.timestamp.last().unwrap(), 
            &99.0, 
            "The kept history should be the most recent data"
        );
    }

    #[test]
    fn test_session_rolling_vitals_flag_logic() {
        let mut config = mock_config(30.0);

        config.estimate_rolling_vitals = Some(false);
        let session = Session::new(config.clone());
        let input = mock_input(vec![0.0, 0.1, 0.2], vec![1.0, 2.0, 3.0]);

        let res = session.process(input.clone(), WaveformMode::Global);
        assert!(res.rolling_vitals.is_none());

        config.estimate_rolling_vitals = Some(true);
        let session_stateful = Session::new(config);
        let res = session_stateful.process(input, WaveformMode::Incremental);
        assert!(res.rolling_vitals.is_none());
    }

    #[test]
    fn test_session_rolling_vitals_output() {
        let mut config = mock_config(10.0);
        config.estimate_rolling_vitals = Some(true);
        config.supported_vitals = vec!["heart_rate".to_string()];
        let session = Session::new(config);

        let fs = 10.0;
        let seconds = 10;
        let n_samples = seconds * fs as usize;
        let timestamps: Vec<f64> = (0..n_samples).map(|i| i as f64 / fs).collect();
        let ppg_data: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 1.0 * (i as f32 / fs as f32)).sin())
            .collect();

        let mut signals = HashMap::new();
        signals.insert(
            "ppg_waveform".to_string(),
            SignalInput {
                data: ppg_data,
                confidence: vec![1.0; n_samples]
            }
        );

        let input = SessionInput {
            timestamp: timestamps,
            signals,
            face: None,
        };

        let res = session.process(input, WaveformMode::Global);

        let rolling = res.rolling_vitals.expect("Rolling vitals should be present");
        let hr_rolling = rolling.get("heart_rate").expect("Heart rate should be in rolling results");

        assert_eq!(hr_rolling.data.len(), n_samples);

        assert!(hr_rolling.data[0].is_nan());

        let last_val = hr_rolling.data[n_samples - 1];
        assert!((last_val - 60.0).abs() < 1.0, "Expected last rolling HR ~60, got {}", last_val);

        assert!(hr_rolling.confidence[n_samples - 1] > 0.8);
    }

    #[test]
    fn test_session_reset() {
        let session = Session::new(mock_config(30.0));

        let input1 = mock_input(vec![0.0, 0.1, 0.2], vec![1.0, 2.0, 3.0]);
        let res1 = session.process(input1, WaveformMode::Global);
        assert_eq!(res1.timestamp.len(), 3);

        session.reset();

        let input2 = mock_input(vec![5.0, 5.1], vec![4.0, 5.0]);
        let res2 = session.process(input2, WaveformMode::Global);

        assert_eq!(res2.timestamp.len(), 2, "Session should treat input2 as a brand new session after reset");
        assert_eq!(res2.waveforms["abp_waveform"].data, vec![4.0, 5.0], "Should not contain old signal data");
    }
}