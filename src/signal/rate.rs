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
}