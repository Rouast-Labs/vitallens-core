use crate::signal::{fft, peaks};
use crate::signal::peaks::Peak;

/// Defines the minimum and maximum physiological bounds for rate estimation in BPM.
#[derive(Debug, Clone, Copy)]
pub struct RateBounds {
    pub min: f32,
    pub max: f32,
}

/// Defines the algorithmic strategy used to estimate a rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RateStrategy {
    /// Uses a Fast Fourier Transform (FFT) periodogram to find the dominant frequency.
    Periodogram {
        target_res_hz: f32,
    },
    /// Uses time-domain peak detection and calculates rate from median interval distances.
    PeakDetection {
        refine: bool,
        interval_buffer: f32,
    },
}

/// Represents the result of a rate estimation operation.
#[derive(Debug, Clone)]
pub struct RateResult {
    pub value: f32,
    pub confidence: f32,
    pub method: String,
}

/// Helper to calculate the effective sampling rate using the mean time difference
fn calculate_effective_fs(timestamps: &[f64], fallback_fs: f32) -> f32 {
    if timestamps.len() < 2 {
        return fallback_fs;
    }
    
    let duration = timestamps.last().unwrap() - timestamps.first().unwrap();
    if duration > 0.0 {
        ((timestamps.len() - 1) as f64 / duration) as f32
    } else {
        fallback_fs
    }
}

/// Estimates the dominant rate (e.g., Heart Rate, Respiratory Rate) of a signal.
///
/// # Arguments
/// * `signal` - The input time-domain signal.
/// * `fs` - The sampling frequency in Hz.
/// * `timestamps` - Optional array of frame timestamps in seconds.
/// * `bounds` - The allowed physiological bounds for the rate in BPM.
/// * `strategy` - The calculation method to use (`Periodogram` or `PeakDetection`).
/// * `rate_hint` - An optional prior rate in BPM to guide detection algorithms.
/// * `scratch` - An optional mutable reference to an `FftScratch` buffer to avoid reallocations.
///
/// # Returns
/// A `RateResult` containing the estimated rate, a confidence score (0.0 to 1.0), and the method used.
pub fn estimate_rate(
    signal: &[f32],
    fs: f32,
    timestamps: Option<&[f64]>,
    bounds: RateBounds,
    strategy: RateStrategy,
    rate_hint: Option<f32>,
    scratch: Option<&mut fft::FftScratch>,
) -> RateResult {
    let effective_fs = match timestamps {
        Some(ts) => calculate_effective_fs(ts, fs),
        None => fs,
    };
    match strategy {
        RateStrategy::Periodogram { target_res_hz } => {
            let working_signal: Vec<f32>;
            let signal_to_use = if let Some(ts) = timestamps {
                working_signal = resample_to_uniform(signal, ts, effective_fs);
                &working_signal
            } else {
                signal
            };

            let (val, conf) = fft::estimate_rate_periodogram(
                signal_to_use, effective_fs, bounds.min, bounds.max, target_res_hz, scratch
            );
            
            RateResult {
                value: val,
                confidence: conf,
                method: "Periodogram".to_string(),
            }
        },
        RateStrategy::PeakDetection { refine, interval_buffer } => {
            let options = peaks::PeakOptions {
                fs: effective_fs,
                avg_rate_hint: rate_hint,
                bounds: peaks::SignalBounds { 
                    min_rate: bounds.min, 
                    max_rate: bounds.max 
                },
                interval_buffer,
                refine,
                ..Default::default()
            };

            let segments = peaks::find_peaks(signal, options);
            calculate_from_peaks(&segments, effective_fs, timestamps, None)
        }
    }
}

/// Estimates the rate from pre-computed peak detection sequences.
///
/// # Arguments
/// * `segments` - A slice of contiguous peak sequences.
/// * `fs_fallback` - The sampling frequency in Hz to use if timestamps are missing or insufficient.
/// * `timestamps` - Optional array of frame timestamps in seconds.
/// * `confidence` - Optional array of confidence scores corresponding to the original signal.
///
/// # Returns
/// A `RateResult` containing the estimated rate, a confidence score, and the method used.
pub fn estimate_rate_from_detections(
    segments: &[Vec<Peak>],
    fs_fallback: f32,
    timestamps: Option<&[f64]>,
    confidence: Option<&[f32]>,
) -> RateResult {
    calculate_from_peaks(segments, fs_fallback, timestamps, confidence)
}

/// Estimates rolling rate from pre-computed peak detection sequences using a time-based window.
///
/// # Arguments
/// * `segments` - A slice of contiguous peak sequences.
/// * `fs_fallback` - The sampling frequency in Hz.
/// * `timestamps` - Optional array of frame timestamps in seconds.
/// * `confidence` - Optional array of confidence scores.
/// * `min_window_seconds` - Minimum elapsed time required to calculate a valid rate.
/// * `preferred_window_seconds` - Maximum backward-looking window size in seconds.
/// * `rolling_stride_seconds` - The stride for the rolling window.
/// * `total_frames` - The total length of the target output arrays.
///
/// # Returns
/// A tuple containing `(rates, confidences)` aligned to the total frames.
pub fn estimate_rolling_rate_from_detections(
    segments: &[Vec<Peak>],
    fs_fallback: f32,
    timestamps: Option<&[f64]>,
    confidence: Option<&[f32]>,
    min_window_seconds: f32,
    preferred_window_seconds: f32,
    rolling_stride_seconds: f32,
    total_frames: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut out_rates = vec![f32::NAN; total_frames];
    let mut out_confs = vec![0.0; total_frames];
    let ts_slice = timestamps.unwrap_or(&[]);

    for segment in segments {
        if segment.is_empty() {
            continue;
        }

        let times: Vec<f64> = segment.iter()
            .map(|p| crate::signal::peaks::resolve_time(p, ts_slice, fs_fallback))
            .collect();

        let mut det_rates = Vec::with_capacity(segment.len());
        let mut det_confs = Vec::with_capacity(segment.len());
        
        let mut last_calc_time = -1.0;
        let mut last_rate = f32::NAN;
        let mut last_conf = 0.0;

        for i in 0..segment.len() {
            let t_curr = times[i];
            let time_in_seq = t_curr - times[0];

            if time_in_seq < min_window_seconds as f64 {
                det_rates.push(f32::NAN);
                det_confs.push(0.0);
                continue;
            }

            if last_calc_time >= 0.0 && (t_curr - last_calc_time) < rolling_stride_seconds as f64 {
                det_rates.push(last_rate);
                det_confs.push(last_conf);
                continue;
            }

            let window_start_time = t_curr - preferred_window_seconds as f64;
            let mut start_idx = 0;
            while start_idx < i && times[start_idx] < window_start_time {
                start_idx += 1;
            }

            let window_slice = &segment[start_idx..=i];

            if window_slice.len() < 2 {
                det_rates.push(f32::NAN);
                det_confs.push(0.0);
                continue;
            }

            let res = estimate_rate_from_detections(&[window_slice.to_vec()], fs_fallback, timestamps, confidence);
            
            last_rate = if res.value > 0.0 { res.value } else { f32::NAN };
            last_conf = res.confidence;
            last_calc_time = t_curr;
            
            det_rates.push(last_rate);
            det_confs.push(last_conf);
        }

        for i in 0..segment.len() {
            let start_frame = segment[i].index;
            let end_frame = if i + 1 < segment.len() {
                segment[i+1].index
            } else {
                if segment.len() >= 2 {
                    let diff = segment[i].index - segment[i-1].index;
                    (segment[i].index + diff).min(total_frames)
                } else {
                    total_frames
                }
            };

            let rate = det_rates[i];
            let conf = det_confs[i];

            for f in start_frame..end_frame {
                if f < total_frames {
                    out_rates[f] = rate;
                    out_confs[f] = conf;
                }
            }
        }
    }

    (out_rates, out_confs)
}

/// Calculates the average rate from detected sequences of peaks based on interval distances.
///
/// # Arguments
/// * `segments` - A vector of contiguous peak sequences.
/// * `fs_fallback` - The sampling frequency in Hz.
/// * `timestamps` - Optional timestamps.
/// * `confidence` - Optional confidences.
///
/// # Returns
/// A `RateResult` containing the estimated rate and confidence derived from the coefficient of variation.
fn calculate_from_peaks(segments: &[Vec<Peak>], fs_fallback: f32, timestamps: Option<&[f64]>, confidence: Option<&[f32]>) -> RateResult {
    if segments.is_empty() {
        return RateResult { value: 0.0, confidence: 0.0, method: "PeakDetection".to_string() };
    }

    let ts_slice = timestamps.unwrap_or(&[]);
    let conf_slice = confidence.unwrap_or(&[]);
    let mut all_intervals = Vec::new();
    let mut used_confidences = Vec::new();
    
    for segment in segments {
        if segment.len() < 2 { continue; }
        for i in 0..segment.len()-1 {
            let p1 = &segment[i];
            let p2 = &segment[i+1];
            
            let t1 = peaks::resolve_time(p1, ts_slice, fs_fallback);
            let t2 = peaks::resolve_time(p2, ts_slice, fs_fallback);
            
            let diff_secs = (t2 - t1) as f32;
            if diff_secs > 0.0 {
                all_intervals.push(diff_secs);
                
                if !conf_slice.is_empty() && p1.index < conf_slice.len() && p2.index < conf_slice.len() {
                    used_confidences.push(conf_slice[p1.index]);
                    used_confidences.push(conf_slice[p2.index]);
                }
            }
        }
    }
    
    if all_intervals.is_empty() {
        return RateResult { value: 0.0, confidence: 0.0, method: "PeakDetection".to_string() };
    }

    let sum: f32 = all_intervals.iter().sum();
    let mean_interval = sum / all_intervals.len() as f32;
    
    let variance: f32 = all_intervals.iter()
        .map(|val| (val - mean_interval).powi(2))
        .sum::<f32>() / all_intervals.len() as f32;
    let std_dev = variance.sqrt();
    
    let cv = if mean_interval > 0.0 { std_dev / mean_interval } else { 0.0 };
    let cv_confidence = (1.0 - (cv / 0.3)).max(0.0);
    
    // Multiply rhythm confidence by signal confidence (if provided)
    let final_confidence = if used_confidences.is_empty() {
        cv_confidence
    } else {
        let avg_conf: f32 = used_confidences.iter().sum::<f32>() / used_confidences.len() as f32;
        cv_confidence * avg_conf
    };
    
    let rate = if mean_interval > 0.0 { 60.0 / mean_interval } else { 0.0 };

    RateResult {
        value: rate,
        confidence: final_confidence,
        method: "PeakDetection".to_string(),
    }
}

/// Resamples an unevenly sampled signal onto a uniform time grid using linear interpolation.
fn resample_to_uniform(signal: &[f32], timestamps: &[f64], target_fs: f32) -> Vec<f32> {
    if timestamps.len() < 2 || signal.len() != timestamps.len() {
        return signal.to_vec();
    }

    let t_start = timestamps[0];
    let t_end = *timestamps.last().unwrap();
    let duration = t_end - t_start;
    
    // Calculate how many samples we need for a uniform grid
    let num_samples = (duration * target_fs as f64).round() as usize;
    if num_samples < 2 { return signal.to_vec(); }

    let step = duration / (num_samples - 1) as f64;
    
    let x_orig: Vec<f32> = timestamps.iter().map(|&t| (t - t_start) as f32).collect();
    let x_new: Vec<f32> = (0..num_samples).map(|i| (i as f64 * step) as f32).collect();

    crate::signal::interp_linear_1d(&x_orig, signal, &x_new)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f32) -> Peak {
        Peak { index: x as usize, x, y: 1.0 }
    }

    #[test]
    fn test_logic_perfect_rhythm_returns_high_confidence() {
        let fs = 30.0;
        
        let segments = vec![
            vec![p(0.0), p(30.0), p(60.0)]
        ];

        let result = calculate_from_peaks(&segments, fs, None, None);

        assert_eq!(result.value, 60.0);
        assert_eq!(result.confidence, 1.0, "Perfect rhythm (CV=0) must have confidence 1.0");
    }

    #[test]
    fn test_logic_irregular_rhythm_returns_low_confidence() {
        let fs = 10.0;
        
        let segments = vec![
            vec![p(0.0), p(15.0), p(25.0)] 
        ];

        let result = calculate_from_peaks(&segments, fs, None, None);

        assert!((result.value - 48.0).abs() < 0.1);        
        assert!(result.confidence < 0.8, "High CV should lower confidence. Got {}", result.confidence);
        assert!(result.confidence > 0.0, "Confidence should not be zero for moderate irregularity");
    }

    #[test]
    fn test_logic_handles_fragmented_segments() {
        let fs = 1.0;

        let segments = vec![
            vec![p(0.0), p(2.0)],
            vec![p(10.0), p(12.0)]
        ];

        let result = calculate_from_peaks(&segments, fs, None, None);

        assert_eq!(result.value, 30.0);
        assert_eq!(result.confidence, 1.0, "Logic should ignore gaps between segments");
    }

    #[test]
    fn test_logic_empty_input() {
        let result = calculate_from_peaks(&[], 30.0, None, None);
        assert_eq!(result.value, 0.0);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_logic_single_peak_segment_is_ignored() {
        let fs = 30.0;
        let segments = vec![vec![p(10.0)]]; 

        let result = calculate_from_peaks(&segments, fs, None, None);
        assert_eq!(result.value, 0.0);
    }

    #[test]
    fn test_estimate_rate_from_detections_with_timestamps() {
        let fs_fallback = 30.0;
        let segments = vec![
            vec![p(0.0), p(1.0), p(2.0)]
        ];
        
        let timestamps = vec![0.0, 1.0, 2.0];
        
        let result = estimate_rate_from_detections(&segments, fs_fallback, Some(&timestamps), None);
        
        assert!((result.value - 60.0).abs() < 0.1);
        assert_eq!(result.confidence, 1.0);
    }

    #[test]
    fn test_estimate_rate_from_detections_with_confidence() {
        let fs = 1.0;
        let segments = vec![
            vec![p(0.0), p(1.0), p(2.0)]
        ];
        
        let confidences = vec![0.5, 0.5, 0.5];
        
        let result = estimate_rate_from_detections(&segments, fs, None, Some(&confidences));
        
        assert_eq!(result.value, 60.0);
        assert_eq!(result.confidence, 0.5); 
    }

    #[test]
    fn test_estimate_rolling_rate_basic_time_window() {
        let fs = 10.0;
        let total_frames = 100;
        
        let segment: Vec<Peak> = (0..10).map(|i| p((i * 10) as f32)).collect();
        
        let (rates, confs) = estimate_rolling_rate_from_detections(
            &[segment], fs, None, None, 
            3.0,
            5.0,
            1.0,
            total_frames
        );

        assert_eq!(rates.len(), total_frames);
        assert_eq!(confs.len(), total_frames);

        assert!(rates[29].is_nan());
        
        assert!((rates[30] - 60.0).abs() < 0.1);
        
        assert!((rates[99] - 60.0).abs() < 0.1);
    }

    #[test]
    fn test_estimate_rolling_rate_stride_logic() {
        let fs = 10.0;
        let total_frames = 100;
        
        let segment = vec![
            p(0.0), p(10.0), p(25.0), p(35.0), p(50.0), p(60.0)
        ];

        let (rates, _) = estimate_rolling_rate_from_detections(
            &[segment], fs, None, None,
            2.0,
            4.0,
            3.0,
            total_frames
        );

        assert!(rates[24].is_nan());
        assert!(!rates[25].is_nan());
        
        let first_calc_rate = rates[25];
        
        assert_eq!(rates[35], first_calc_rate);
        
        assert!(rates[60] != first_calc_rate && !rates[60].is_nan());
    }

    #[test]
    fn test_estimate_rolling_rate_multiple_segments() {
        let fs = 10.0;
        let total_frames = 100;
        
        let seg1: Vec<Peak> = (0..4).map(|i| p((i * 10) as f32)).collect();
        
        let seg2: Vec<Peak> = (0..7).map(|i| p((60 + i * 5) as f32)).collect();

        let (rates, _) = estimate_rolling_rate_from_detections(
            &[seg1, seg2], fs, None, None,
            2.0, 5.0, 1.0, total_frames
        );

        assert!((rates[20] - 60.0).abs() < 0.1);

        assert!((rates[39] - 60.0).abs() < 0.1);
        assert!(rates[40].is_nan());
        assert!(rates[50].is_nan());
        
        assert!(rates[79].is_nan());
        assert!((rates[80] - 120.0).abs() < 0.1);

        assert!((rates[94] - 120.0).abs() < 0.1);
        assert!(rates[95].is_nan());
        assert!(rates[99].is_nan());
    }
}