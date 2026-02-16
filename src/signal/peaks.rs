use crate::signal::filters;

#[derive(Debug, Clone, Copy)]
pub struct Peak {
    pub index: usize,
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct SignalBounds {
    pub min_rate: f32,
    pub max_rate: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct PeakOptions {
    pub fs: f32,
    pub avg_rate_hint: Option<f32>, 
    pub bounds: SignalBounds,
    pub threshold: f32,
    pub window_cycles: f32,
    pub max_rate_change_per_sec: f32,
    pub interval_buffer: f32,
    pub refine: bool,
    pub smooth_input: bool,
}

impl Default for PeakOptions {
    fn default() -> Self {
        Self {
            fs: 30.0,
            avg_rate_hint: None,
            bounds: SignalBounds { min_rate: 40.0, max_rate: 220.0 },
            threshold: 1.0,
            window_cycles: 2.5,
            max_rate_change_per_sec: 3.0,
            interval_buffer: 0.25,
            refine: true,
            smooth_input: false,
        }
    }
}

/// Detects peaks using a Centered Adaptive Z-Score.
/// Returns a list of valid segments.
pub fn find_peaks(signal: &[f32], options: PeakOptions) -> Vec<Vec<Peak>> {
    if signal.len() < 3 {
        return Vec::new();
    }

    // 1. Optional pre-processing
    let processed_data_storage;
    let mut search_radius = 0;
    let working_signal: &[f32] = if options.smooth_input {
        let cutoff_hz = options.bounds.max_rate / 60.0;
        let window = filters::estimate_moving_average_window(options.fs, cutoff_hz, true);
        search_radius = window / 2;
        processed_data_storage = filters::moving_average(signal, window);
        &processed_data_storage
    } else {
        signal
    };

    // 2. Auto-tune parameters
    
    let reference_rate = options.avg_rate_hint.unwrap_or(options.bounds.min_rate);
    let window_seconds = (60.0 / reference_rate) * options.window_cycles;
    
    let radius = ((window_seconds * options.fs) / 2.0).round() as usize;
    let radius = radius.max(2).min(working_signal.len() / 2);

    let max_possible_rate = if let Some(avg) = options.avg_rate_hint {
        let duration = working_signal.len() as f32 / options.fs;
        let drift = options.max_rate_change_per_sec * (duration / 2.0);
        (avg + drift).min(options.bounds.max_rate)
    } else {
        options.bounds.max_rate
    };
    
    let min_dist_seconds = (60.0 / max_possible_rate) * (1.0 - options.interval_buffer);
    let min_dist_samples = (min_dist_seconds * options.fs) as usize;

    let slowest_period = 60.0 / options.bounds.min_rate;
    let max_gap_samples = (slowest_period * 2.5 * options.fs) as usize;
    
    // 3. Compute centered rolling stats (z-score algorithm)
    
    let mut means = vec![0.0; working_signal.len()];
    let mut stds = vec![0.0; working_signal.len()];
    
    let mut start = 0;
    let mut end = radius.min(working_signal.len() - 1);
    
    let mut sum: f32 = working_signal[start..=end].iter().sum();
    let mut sq_sum: f32 = working_signal[start..=end].iter().map(|x| x*x).sum();
    
    for i in 0..working_signal.len() {
        let new_end = (i + radius).min(working_signal.len() - 1);
        if new_end > end {
            sum += working_signal[new_end];
            sq_sum += working_signal[new_end] * working_signal[new_end];
            end = new_end;
        }
        
        let new_start = i.saturating_sub(radius);
        if new_start > start {
            sum -= working_signal[start];
            sq_sum -= working_signal[start] * working_signal[start];
            start = new_start;
        }
        
        let count = (end - start + 1) as f32;
        let mean = sum / count;
        let variance = (sq_sum / count) - (mean * mean);
        
        means[i] = mean;
        stds[i] = variance.max(0.0).sqrt();
    }

    // 4. Peak detection logic
    
    let mut peaks: Vec<Peak> = Vec::new();
    let mut last_peak_idx: Option<usize> = None;

    for i in 1..working_signal.len()-1 {
        let val = working_signal[i];
        let mean = means[i];
        let std = stds[i];

        let is_candidate = if std > 1e-6 {
             (val - mean) > options.threshold * std
        } else {
             false
        };

        if is_candidate {
            if val > working_signal[i-1] && val >= working_signal[i+1] {
                
                let in_refractory_window = match last_peak_idx {
                    Some(last) => (i - last) < min_dist_samples,
                    None => false
                };

                let search_start = i.saturating_sub(search_radius);
                let search_end = (i + search_radius + 1).min(signal.len());
                
                let mut best_idx = i;
                let mut best_val = signal[i];

                if search_radius > 0 {
                    for j in search_start..search_end {
                        if signal[j] > best_val {
                            best_val = signal[j];
                            best_idx = j;
                        }
                    }
                }

                let mut final_peak = Peak {
                    index: best_idx,
                    x: best_idx as f32,
                    y: best_val,
                };

                if options.refine && best_idx > 0 && best_idx < signal.len() - 1 {
                    let y_l = signal[best_idx - 1];
                    let y_c = signal[best_idx];
                    let y_r = signal[best_idx + 1];

                    let denom: f32 = 2.0 * (y_l - 2.0 * y_c + y_r);
                    if denom.abs() > 1e-6 {
                        let delta = (y_l - y_r) / denom;
                        if delta.abs() <= 0.5 {
                            final_peak.x = best_idx as f32 + delta;
                        }
                    }
                }

                if in_refractory_window {
                    if let Some(last_peak) = peaks.last() {
                        let left_neighbor_higher = signal[i-1] > last_peak.y; 
                        let right_neighbor_higher = signal[i+1] > last_peak.y;
                        
                        if final_peak.y > last_peak.y && (left_neighbor_higher || right_neighbor_higher) {
                            peaks.pop();
                            peaks.push(final_peak);
                            last_peak_idx = Some(i);
                        }
                    }
                } else {
                    peaks.push(final_peak);
                    last_peak_idx = Some(i);
                }
            }
        }
    }

    // 5. Segmentation
    
    if peaks.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut current_segment = Vec::new();
    current_segment.push(peaks[0]);

    for i in 1..peaks.len() {
        let prev = &peaks[i-1];
        let curr = &peaks[i];
        
        if (curr.x - prev.x) > max_gap_samples as f32 {
            segments.push(current_segment);
            current_segment = Vec::new();
        }
        current_segment.push(*curr);
    }
    segments.push(current_segment);

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Helpers ---

    fn mock_sine(fs: f32, freq: f32, duration: f32) -> Vec<f32> {
        (0..(fs * duration) as usize).map(|i| {
            (i as f32 / fs * 2.0 * std::f32::consts::PI * freq).sin()
        }).collect()
    }

    fn mock_step(fs: f32, duration: f32, jump_at: f32, height: f32) -> Vec<f32> {
        (0..(fs * duration) as usize).map(|i| {
            if (i as f32 / fs) >= jump_at { height } else { 0.0 }
        }).collect()
    }

    // --- 1. Physiological Constraints ---

    #[test]
    fn peak_01_refractory_hr() {
        let fs = 30.0;
        let mut sig = mock_sine(fs, 1.0, 2.0);
        sig[14] = 2.0; // The noise spike

        let opts = PeakOptions {
            fs,
            avg_rate_hint: Some(60.0),
            threshold: 0.5,
            ..Default::default()
        };

        let segments = find_peaks(&sig, opts);
        let peaks = &segments[0];

        let found_sine = peaks.iter().any(|p| (p.index as i32 - 8).abs() <= 1);
        let found_noise = peaks.iter().any(|p| p.index == 14);

        assert!(found_sine, "Real peak missed");
        assert!(!found_noise, "Noise spike should be rejected by refractory period");
    }

    #[test]
    fn peak_02_refractory_resp() {
        let fs = 30.0;
        let mut sig = mock_sine(fs, 0.2, 10.0);
        sig[68] = 2.0; 

        let opts = PeakOptions {
            fs,
            avg_rate_hint: Some(12.0),
            bounds: SignalBounds { min_rate: 6.0, max_rate: 40.0 },
            threshold: 0.5,
            ..Default::default()
        };

        let segments = find_peaks(&sig, opts);
        let peaks = &segments[0];

        let found_noise = peaks.iter().any(|p| p.index == 68);
        assert!(!found_noise, "Respiration noise should be rejected");
    }

    #[test]
    fn peak_03_max_rate_gating() {
        // Max HR is 220 BPM (3.66 Hz).
        // 6Hz is clearly faster than 3.66Hz, so beats should be skipped.
        let fs = 30.0;
        let sig = mock_sine(fs, 6.0, 2.0); 
        
        let opts = PeakOptions {
            fs,
            avg_rate_hint: None,
            threshold: 0.5,
            ..Default::default()
        };

        let segments = find_peaks(&sig, opts);
        
        if !segments.is_empty() {
            let peaks = &segments[0];
            // 6Hz * 2s = 12 potential peaks.
            // Max rate 220 BPM allows ~7 peaks in 2s.
            // We expect significantly fewer than 12.
            assert!(peaks.len() < 10, "Detected too many peaks ({}) for physical limits", peaks.len());
        }
    }

    #[test]
    fn peak_04_hrv_buffer_allows_arrhythmia() {
        let fs = 10.0;
        let mut sig = vec![0.0; 50]; 
        sig[10] = 1.0; 
        sig[20] = 1.0; 
        sig[27] = 1.0; // Premature (Gap = 0.7s)
        sig[37] = 1.0;

        let opts = PeakOptions {
            fs,
            avg_rate_hint: Some(60.0),
            threshold: 0.5,
            interval_buffer: 0.25,
            ..Default::default()
        };

        let segments = find_peaks(&sig, opts);
        let peaks = &segments[0];

        let found_premature = peaks.iter().any(|p| p.index == 27);
        assert!(found_premature, "Premature valid beat should be detected");
    }

    // --- 2. Robustness ---

    #[test]
    fn peak_05_amplitude_flutter() {
        let fs = 30.0;
        let mut sig = mock_sine(fs, 1.0, 4.0);
        for i in 60..120 { sig[i] *= 2.0; }

        let opts = PeakOptions {
            fs,
            avg_rate_hint: Some(60.0),
            threshold: 0.5,
            ..Default::default()
        };

        let segments = find_peaks(&sig, opts);
        let peaks = &segments[0];

        let peaks_first = peaks.iter().filter(|p| p.index < 60).count();
        let peaks_second = peaks.iter().filter(|p| p.index >= 60).count();

        assert!(peaks_first >= 1);
        assert!(peaks_second >= 1);
    }

    #[test]
    fn peak_06_baseline_wander() {
        // If drift is too massive, its variance hides the pulse variance.
        let fs = 30.0;
        let pulse = mock_sine(fs, 1.0, 5.0);
        let drift = mock_sine(fs, 0.1, 5.0);
        
        let sig: Vec<f32> = pulse.iter().zip(drift.iter())
            .map(|(p, d)| p + d * 2.0) // 2x drift is still significant but handleable
            .collect();

        let opts = PeakOptions {
            fs,
            avg_rate_hint: Some(60.0),
            threshold: 0.5,
            ..Default::default()
        };

        let segments = find_peaks(&sig, opts);
        assert!(!segments.is_empty());
        assert!(segments[0].len() >= 4);
    }

    #[test]
    fn peak_07_step_function_stability() {
        // [FIXED] Used helper function
        let fs = 10.0;
        let mut sig = mock_step(fs, 4.0, 2.0, 100.0);
        
        // Peak at index 20 (The step edge)
        sig[20] = 110.0; 
        sig[19] = 50.0;  
        sig[21] = 105.0; 

        let opts = PeakOptions {
            fs,
            threshold: 1.0, 
            ..Default::default()
        };

        let segments = find_peaks(&sig, opts);
        assert!(!segments.is_empty());
        
        let p = segments[0][0];
        assert_eq!(p.index, 20);
    }

    #[test]
    fn peak_08_flatline() {
        let sig = vec![0.0; 100];
        let opts = PeakOptions::default();
        let segments = find_peaks(&sig, opts);
        assert!(segments.is_empty());
    }

    // --- 3. Precision ---

    #[test]
    fn peak_09_sub_sample_precision() {
        let mut sig = vec![0.0; 20];
        sig[10] = 0.75;
        sig[11] = 0.75;
        sig[9] = 0.0;
        sig[12] = 0.0;

        let opts = PeakOptions {
            fs: 10.0,
            threshold: 0.1,
            refine: true,
            ..Default::default()
        };

        let segments = find_peaks(&sig, opts);
        let p = segments[0][0];

        assert!((p.x - 10.5).abs() < 1e-4);
    }

    #[test]
    fn peak_10_jagged_peak_fallback() {
        let mut sig = vec![0.0; 10];
        sig[4] = 0.5;
        sig[5] = 1.0;
        sig[6] = 0.1;

        let opts = PeakOptions {
            fs: 10.0,
            threshold: 0.1,
            refine: true,
            ..Default::default()
        };

        let segments = find_peaks(&sig, opts);
        let p = segments[0][0];
        
        assert!(p.x.is_finite());
        assert!((p.x - 5.0).abs() < 0.6);
    }

    // --- 4. Segmentation ---

    #[test]
    fn peak_11_signal_gap() {
        let fs = 10.0;
        let pulse_train = mock_sine(fs, 1.0, 5.0);
        let silence = vec![0.0; 50]; 
        
        let sig = [pulse_train.clone(), silence, pulse_train].concat();

        let opts = PeakOptions {
            fs,
            avg_rate_hint: Some(60.0),
            threshold: 0.5,
            ..Default::default()
        };

        let segments = find_peaks(&sig, opts);
        
        assert_eq!(segments.len(), 2);
        assert!(segments[0].len() >= 4);
        assert!(segments[1].len() >= 4);
    }

    #[test]
    fn peak_12_single_segment_continuity() {
        let fs = 10.0;
        let sig = mock_sine(fs, 1.0, 10.0); 

        let opts = PeakOptions {
            fs,
            avg_rate_hint: Some(60.0),
            threshold: 0.5,
            ..Default::default()
        };

        let segments = find_peaks(&sig, opts);
        
        assert_eq!(segments.len(), 1);
        assert!(segments[0].len() >= 9);
    }
}