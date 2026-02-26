use crate::signal::filters;

/// Represents a detected peak in a signal.
#[derive(Debug, Clone, Copy)]
pub struct Peak {
    pub index: usize,
    pub x: f32,
    pub y: f32,
}

/// Represents a full physiological cycle (e.g., a breath), bounded by valleys.
#[derive(Debug, Clone, Copy)]
pub struct Cycle {
    pub start_valley: Peak, 
    pub peak: Peak,
    pub end_valley: Peak,
}

/// Defines the operational boundaries for rate detection in BPM.
#[derive(Debug, Clone, Copy)]
pub struct SignalBounds {
    pub min_rate: f32,
    pub max_rate: f32,
}

/// Configuration options for the peak detection algorithm.
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

struct StatsResult {
    working_signal: Vec<f32>,
    means: Vec<f32>,
    stds: Vec<f32>,
    search_radius: usize,
}

impl Default for PeakOptions {
    fn default() -> Self {
        Self {
            fs: 30.0,
            avg_rate_hint: None,
            bounds: SignalBounds { min_rate: 40.0, max_rate: 220.0 },
            threshold: 1.0,
            window_cycles: 2.5,
            max_rate_change_per_sec: 1.0,
            interval_buffer: 0.25,
            refine: true,
            smooth_input: false,
        }
    }
}

/// Computes local rolling statistics (mean and standard deviation) for adaptive thresholding.
/// Optionally applies low-pass smoothing to the input signal prior to calculation.
///
/// # Arguments
/// * `signal` - The input time-domain signal.
/// * `options` - Configuration options defining smoothing and window parameters.
///
/// # Returns
/// A `StatsResult` containing the working signal and arrays of local means and standard deviations.
fn compute_statistics(signal: &[f32], options: &PeakOptions) -> StatsResult {
    let smoothed_storage;
    let mut search_radius = 0;
    
    let working_signal: &[f32] = if options.smooth_input {
        let cutoff_hz = options.bounds.max_rate / 60.0;
        let window = filters::estimate_moving_average_window(options.fs, cutoff_hz, true);
        search_radius = window / 2;
        
        smoothed_storage = filters::moving_average(signal, window);
        &smoothed_storage
    } else {
        signal
    };

    let reference_rate = options.avg_rate_hint.unwrap_or(options.bounds.min_rate);
    let window_seconds = (60.0 / reference_rate) * options.window_cycles;
    
    let radius = ((window_seconds * options.fs) / 2.0).round() as usize;
    let radius = radius.max(2).min(working_signal.len() / 2);

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

    StatsResult {
        working_signal: working_signal.to_vec(),
        means,
        stds,
        search_radius,
    }
}

/// Refines a peak's integer index into a sub-sample continuous float position
/// using quadratic interpolation.
///
/// # Arguments
/// * `signal` - The raw signal array.
/// * `idx` - The integer index of the detected peak.
///
/// # Returns
/// The interpolated sub-sample location as an `f32`.
fn refine_location(signal: &[f32], idx: usize) -> f32 {
    if idx > 0 && idx < signal.len() - 1 {
        let y_l = signal[idx - 1];
        let y_c = signal[idx];
        let y_r = signal[idx + 1];

        let denom = 2.0 * (y_l - 2.0 * y_c + y_r);
        if denom.abs() > 1e-6 {
            let delta = (y_l - y_r) / denom;
            if delta.abs() <= 0.5 {
                return idx as f32 + delta;
            }
        }
    }
    idx as f32
}

/// Detects peaks in a physiological signal using adaptive thresholding and refractory periods.
/// Groups detected peaks into continuous sequences if gaps are present.
///
/// # Arguments
/// * `signal` - The input time-domain signal.
/// * `options` - Configuration options defining bounds, thresholds, and smoothing.
///
/// # Returns
/// A vector of peak sequences (`Vec<Vec<Peak>>`). Each sub-vector represents a continuous train of peaks.
pub fn find_peaks(signal: &[f32], options: PeakOptions) -> Vec<Vec<Peak>> {
    if signal.len() < 3 {
        return Vec::new();
    }

    let stats = compute_statistics(signal, &options);
    let working_signal = &stats.working_signal;
    let means = &stats.means;
    let stds = &stats.stds;
    let search_radius = stats.search_radius;

    let max_possible_rate = if let Some(avg) = options.avg_rate_hint {
        let duration = working_signal.len() as f32 / options.fs;
        let drift = options.max_rate_change_per_sec * (duration / 2.0);
        (avg + drift).min(options.bounds.max_rate)
    } else {
        options.bounds.max_rate
    };
    
    let min_dist_seconds = (60.0 / max_possible_rate) * (1.0 - options.interval_buffer);
    let min_dist_samples = (min_dist_seconds * options.fs).ceil() as usize;

    let slowest_period = 60.0 / options.bounds.min_rate;
    let max_gap_samples = (slowest_period * 2.5 * options.fs) as usize;
    
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

                let mut best_idx = i;
                let mut best_val = signal[i];

                if search_radius > 0 {
                    let search_start = i.saturating_sub(search_radius);
                    let search_end = (i + search_radius + 1).min(signal.len());
                    
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

                if options.refine {
                    final_peak.x = refine_location(signal, best_idx);
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

const EDGE_SEARCH_FRACTION: f32 = 0.6;

/// Detects full physiological cycles (valley-to-valley) based on detected peaks.
///
/// # Arguments
/// * `signal` - The input time-domain signal.
/// * `options` - Configuration options for peak detection.
///
/// # Returns
/// A vector of `Cycle` structs representing complete physiological waveforms.
pub fn find_cycles(signal: &[f32], options: PeakOptions) -> Vec<Cycle> {
    if signal.len() < 3 {
        return Vec::new();
    }

    let peak_segments = find_peaks(signal, options);
    let mut all_cycles = Vec::new();

    for segment in peak_segments {
        if segment.is_empty() { continue; }

        let avg_cycle_samples = if let Some(rate) = options.avg_rate_hint {
            if rate > 1e-5 {
                (options.fs * 60.0 / rate).round() as usize
            } else {
                (options.fs * 60.0 / options.bounds.min_rate) as usize
            }
        } else if segment.len() > 1 {
            let total_dist: f32 = segment.windows(2)
                .map(|w| w[1].x - w[0].x)
                .sum();
            (total_dist / (segment.len() - 1) as f32).round() as usize
        } else {
            let slowest_period = 60.0 / options.bounds.min_rate;
            (slowest_period * options.fs) as usize
        };

        let edge_search_window = (avg_cycle_samples as f32 * EDGE_SEARCH_FRACTION) as usize;

        let mut valleys = Vec::new();
        for i in 0..segment.len() - 1 {
            let p_curr = segment[i];
            let p_next = segment[i+1];

            let start = (p_curr.index + 1).min(signal.len());
            let end = p_next.index.min(signal.len());

            if start < end {
                let (min_idx, min_val) = find_min_in_range(signal, start, end);
                
                let mut v = Peak {
                    index: min_idx,
                    x: min_idx as f32,
                    y: min_val
                };
                if options.refine { v.x = refine_location(signal, min_idx); }
                valleys.push(v);
            } else {
                 valleys.push(Peak { index: start, x: start as f32, y: signal.get(start).copied().unwrap_or(0.0) });
            }
        }

        let p_first = segment[0];
        let v_start_opt = if p_first.index >= edge_search_window {
            let start_limit = p_first.index - edge_search_window;
            let (idx, val) = find_min_in_range(signal, start_limit, p_first.index);
            let mut v = Peak { index: idx, x: idx as f32, y: val };
            if options.refine { v.x = refine_location(signal, idx); }
            Some(v)
        } else {
            None
        };

        let p_last = segment[segment.len()-1];
        let v_end_opt = if p_last.index + edge_search_window < signal.len() {
            let end_limit = p_last.index + edge_search_window;
            let (idx, val) = find_min_in_range(signal, p_last.index + 1, end_limit);
            let mut v = Peak { index: idx, x: idx as f32, y: val };
            if options.refine { v.x = refine_location(signal, idx); }
            Some(v)
        } else {
            None
        };
        
        let mut sequence_valleys: Vec<Option<Peak>> = Vec::with_capacity(valleys.len() + 2);
        sequence_valleys.push(v_start_opt);
        for v in valleys { sequence_valleys.push(Some(v)); }
        sequence_valleys.push(v_end_opt);

        for i in 0..segment.len() {
            if let (Some(v_prev), Some(v_next)) = (sequence_valleys[i], sequence_valleys[i+1]) {
                let peak = segment[i];
                if v_prev.x < peak.x && peak.x < v_next.x {
                    all_cycles.push(Cycle {
                        start_valley: v_prev,
                        peak,
                        end_valley: v_next,
                    });
                }
            }
        }
    }

    all_cycles
}

/// Finds the minimum value and its corresponding index within a sub-slice of a signal.
///
/// # Arguments
/// * `signal` - The full signal array.
/// * `start` - The inclusive starting index.
/// * `end` - The exclusive ending index.
///
/// # Returns
/// A tuple of `(absolute_index, minimum_value)`.
fn find_min_in_range(signal: &[f32], start: usize, end: usize) -> (usize, f32) {
    let slice = &signal[start..end];
    if slice.is_empty() {
        return (start, 0.0);
    }

    let (min_idx_local, min_val) = slice.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    (start + min_idx_local, *min_val)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn peak_01_refractory_hr() {
        let fs = 30.0;
        let mut sig = mock_sine(fs, 1.0, 2.0);
        sig[14] = 2.0; 

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
            assert!(peaks.len() < 10, "Detected too many peaks ({}) for physical limits", peaks.len());
        }
    }

    #[test]
    fn peak_04_hrv_buffer_allows_arrhythmia() {
        let fs = 10.0;
        let mut sig = vec![0.0; 50]; 
        sig[10] = 1.0; 
        sig[20] = 1.0; 
        sig[28] = 1.0;  
        sig[38] = 1.0;

        let opts = PeakOptions {
            fs,
            avg_rate_hint: Some(60.0),
            threshold: 0.5,
            interval_buffer: 0.25,
            ..Default::default()
        };

        let segments = find_peaks(&sig, opts);
        let peaks = &segments[0];

        let found_premature = peaks.iter().any(|p| p.index == 28);
        assert!(found_premature, "Premature valid beat should be detected");
    }

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
        let fs = 30.0;
        let pulse = mock_sine(fs, 1.0, 5.0);
        let drift = mock_sine(fs, 0.1, 5.0);
        
        let sig: Vec<f32> = pulse.iter().zip(drift.iter())
            .map(|(p, d)| p + d * 2.0)  
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
        let fs = 10.0;
        let mut sig = mock_step(fs, 4.0, 2.0, 100.0);
        
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