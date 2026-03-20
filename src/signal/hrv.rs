use crate::signal::peaks::{Peak, PeakOptions, SignalBounds, find_peaks, resolve_time};
use crate::signal::filters; 

/// Configuration options for extracting Normal-to-Normal (NN) intervals.
#[derive(Debug, Clone, Copy)]
pub struct HrvOptions {
    pub confidence_threshold: f32,
    pub fs_fallback: f32,
    pub outlier_threshold: f32,
}

impl Default for HrvOptions {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            fs_fallback: 30.0,
            outlier_threshold: 0.3,
        }
    }
}

/// High-level coordinator for Heart Rate Variability (HRV) metrics.
/// Detects peaks, extracts gated intervals, and computes the requested metric.
///
/// # Arguments
/// * `signal` - The input time-domain signal (e.g., PPG).
/// * `fs` - The sampling frequency in Hz.
/// * `metric` - The specific HRV metric to calculate.
/// * `timestamps` - Array of timestamps corresponding to the signal.
/// * `confidence` - Array of confidence scores for the signal.
/// * `bounds` - Allowed bounding rates for peak detection.
/// * `rate_hint` - Optional prior heart rate hint in BPM.
///
/// # Returns
/// A tuple of `(calculated_value, confidence_score)`.
pub fn estimate_hrv(
    signal: &[f32], 
    fs: f32, 
    metric: crate::registry::HrvMetric, 
    timestamps: &[f64],
    confidence: &[f32],
    bounds: SignalBounds,
    rate_hint: Option<f32>
) -> (f32, f32) {
    let options = PeakOptions {
        fs,
        bounds,
        avg_rate_hint: rate_hint,
        threshold: 0.45,
        window_cycles: 2.5,
        max_rate_change_per_sec: 1.0,
        refine: true,
        smooth_input: true,
        ..Default::default()
    };

    let sequences = find_peaks(signal, options);

    let hrv_opts = HrvOptions {
        fs_fallback: fs,
        ..Default::default()
    };
    
    let (intervals, min_conf_peaks) = extract_nn_intervals(&sequences, Some(timestamps), Some(confidence), hrv_opts);

    if intervals.is_empty() {
        return (0.0, 0.0);
    }

    let value = match metric {
        crate::registry::HrvMetric::Sdnn => calculate_sdnn(&intervals),
        crate::registry::HrvMetric::Rmssd => calculate_rmssd(&intervals),
        crate::registry::HrvMetric::LfHf => calculate_lfhf(&intervals),
        crate::registry::HrvMetric::StressIndex => calculate_stress_index(&intervals),
        crate::registry::HrvMetric::Pnn50 => calculate_pnn50(&intervals),
        crate::registry::HrvMetric::Sd1Sd2 => calculate_sd1sd2(&intervals),
    };

    let avg_conf = if confidence.is_empty() {
        0.0
    } else {
        confidence.iter().sum::<f32>() / confidence.len() as f32
    };

    let final_conf = avg_conf.min(min_conf_peaks);

    (value, final_conf)
}

/// Estimates a Heart Rate Variability (HRV) metric from pre-detected peak sequences.
///
/// # Arguments
/// * `sequences` - A slice of vectors, where each vector contains a continuous sequence of peaks.
/// * `fs_fallback` - The sampling frequency in Hz to use for time resolution if timestamps are absent.
/// * `metric` - The specific HRV metric to calculate (e.g., SDNN, RMSSD).
/// * `timestamps` - Optional array of reference timestamps in seconds.
/// * `confidence` - Optional array of confidence scores for the signal frames.
///
/// # Returns
/// A tuple of `(calculated_value, confidence_score)`.
pub fn estimate_hrv_from_detections(
    sequences: &[Vec<Peak>],
    fs_fallback: f32,
    metric: crate::registry::HrvMetric,
    timestamps: Option<&[f64]>,
    confidence: Option<&[f32]>,
) -> (f32, f32) {
    let hrv_opts = HrvOptions {
        fs_fallback,
        ..Default::default()
    };
    
    let (intervals, min_conf_peaks) = extract_nn_intervals(sequences, timestamps, confidence, hrv_opts);

    if intervals.is_empty() {
        return (0.0, 0.0);
    }

    let value = match metric {
        crate::registry::HrvMetric::Sdnn => calculate_sdnn(&intervals),
        crate::registry::HrvMetric::Rmssd => calculate_rmssd(&intervals),
        crate::registry::HrvMetric::LfHf => calculate_lfhf(&intervals),
        crate::registry::HrvMetric::StressIndex => calculate_stress_index(&intervals),
        crate::registry::HrvMetric::Pnn50 => calculate_pnn50(&intervals),
        crate::registry::HrvMetric::Sd1Sd2 => calculate_sd1sd2(&intervals),
    };

    (value, min_conf_peaks)
}

/// Estimates rolling HRV metric from pre-computed peak detection sequences using a time-based window.
///
/// # Arguments
/// * `sequences` - A slice of contiguous peak sequences.
/// * `fs_fallback` - The sampling frequency in Hz.
/// * `metric` - The specific HRV metric to calculate.
/// * `timestamps` - Optional array of frame timestamps in seconds.
/// * `confidence` - Optional array of confidence scores.
/// * `min_window_seconds` - Minimum elapsed time required to calculate a valid metric.
/// * `preferred_window_seconds` - Maximum backward-looking window size in seconds.
/// * `rolling_stride_seconds` - The stride for the rolling window.
/// * `total_frames` - The total length of the target output arrays.
///
/// # Returns
/// A tuple containing `(hrv_values, confidences)` aligned to the total frames.
pub fn estimate_rolling_hrv_from_detections(
    sequences: &[Vec<Peak>],
    fs_fallback: f32,
    metric: crate::registry::HrvMetric,
    timestamps: Option<&[f64]>,
    confidence: Option<&[f32]>,
    min_window_seconds: f32,
    preferred_window_seconds: f32,
    rolling_stride_seconds: f32,
    total_frames: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut out_vals = vec![f32::NAN; total_frames];
    let mut out_confs = vec![0.0; total_frames];
    let ts_slice = timestamps.unwrap_or(&[]);

    for sequence in sequences {
        if sequence.is_empty() {
            continue;
        }

        let times: Vec<f64> = sequence.iter()
            .map(|p| crate::signal::peaks::resolve_time(p, ts_slice, fs_fallback))
            .collect();

        let mut det_vals = Vec::with_capacity(sequence.len());
        let mut det_confs = Vec::with_capacity(sequence.len());
        
        let mut last_calc_time = -1.0;
        let mut last_val = f32::NAN;
        let mut last_conf = 0.0;

        for i in 0..sequence.len() {
            let t_curr = times[i];
            let time_in_seq = t_curr - times[0];

            if time_in_seq < min_window_seconds as f64 {
                det_vals.push(f32::NAN);
                det_confs.push(0.0);
                continue;
            }

            // Stride logic: skip expensive calc if we haven't crossed the stride threshold
            if last_calc_time >= 0.0 && (t_curr - last_calc_time) < rolling_stride_seconds as f64 {
                det_vals.push(last_val);
                det_confs.push(last_conf);
                continue;
            }

            let window_start_time = t_curr - preferred_window_seconds as f64;
            let mut start_idx = 0;
            while start_idx < i && times[start_idx] < window_start_time {
                start_idx += 1;
            }

            let window_slice = &sequence[start_idx..=i];

            if window_slice.len() < 3 {
                det_vals.push(f32::NAN);
                det_confs.push(0.0);
                continue;
            }

            let (val, conf) = estimate_hrv_from_detections(
                &[window_slice.to_vec()],
                fs_fallback,
                metric,
                timestamps,
                confidence
            );

            last_val = if val > 0.0 || (val == 0.0 && conf > 0.0) { val } else { f32::NAN };
            last_conf = conf;
            last_calc_time = t_curr;
            
            det_vals.push(last_val);
            det_confs.push(last_conf);
        }

        for i in 0..sequence.len() {
            let start_frame = sequence[i].index;
            let end_frame = if i + 1 < sequence.len() {
                sequence[i+1].index
            } else {
                if sequence.len() >= 2 {
                    let diff = sequence[i].index - sequence[i-1].index;
                    (sequence[i].index + diff).min(total_frames)
                } else {
                    total_frames
                }
            };

            let val = det_vals[i];
            let conf = det_confs[i];

            for f in start_frame..end_frame {
                if f < total_frames {
                    out_vals[f] = val;
                    out_confs[f] = conf;
                }
            }
        }
    }

    (out_vals, out_confs)
}

/// Extracts valid Normal-to-Normal (NN) intervals from multiple sequences of peaks.
/// 
/// This function prevents "phantom intervals" by only calculating differences within
/// the provided continuous sequences. It interpolates timestamps for sub-sample precision.
/// 
/// # Arguments
/// * `sequences` - A slice of peak sequences detected from the signal.
/// * `timestamps` - Optional array of reference timestamps.
/// * `confidence` - Optional array of confidence scores.
/// * `options` - Configuration options for HRV extraction.
///
/// # Returns
/// A tuple containing `(intervals_in_ms, minimum_confidence_used)`.
pub fn extract_nn_intervals(
    sequences: &[Vec<Peak>],
    timestamps: Option<&[f64]>,
    confidence: Option<&[f32]>,
    options: HrvOptions
) -> (Vec<f32>, f32) {
    let mut all_intervals = Vec::new();
    let mut all_used_confidences = Vec::new();
    
    let ts_slice = timestamps.unwrap_or(&[]);
    let conf_slice = confidence.unwrap_or(&[]);

    for sequence in sequences {
        if sequence.len() < 2 { continue; }

        for i in 0..sequence.len() - 1 {
            let p1 = &sequence[i];
            let p2 = &sequence[i+1];

            let c1 = if !conf_slice.is_empty() && p1.index < conf_slice.len() { conf_slice[p1.index] } else { 1.0 };
            let c2 = if !conf_slice.is_empty() && p2.index < conf_slice.len() { conf_slice[p2.index] } else { 1.0 };

            if c1 >= options.confidence_threshold && c2 >= options.confidence_threshold {
                
                let t1 = resolve_time(p1, ts_slice, options.fs_fallback);
                let t2 = resolve_time(p2, ts_slice, options.fs_fallback);

                let diff_sec = t2 - t1;
                
                if diff_sec > 0.0 {
                    all_intervals.push((diff_sec * 1000.0) as f32);
                    all_used_confidences.push(c1);
                    all_used_confidences.push(c2);
                }
            }
        }
    }

    if all_intervals.is_empty() {
        return (Vec::new(), 0.0);
    }

    let filtered_intervals = filter_outliers(&all_intervals, options.outlier_threshold);

    if filtered_intervals.is_empty() {
        return (Vec::new(), 0.0);
    }

    let min_conf = all_used_confidences.iter().fold(1.0f32, |a, &b| a.min(b));

    (filtered_intervals, min_conf)
}

/// Calculates the Standard Deviation of NN intervals (SDNN).
///
/// # Arguments
/// * `nn_intervals` - Slice of valid NN intervals in milliseconds.
///
/// # Returns
/// The SDNN value in milliseconds as an `f32`.
pub fn calculate_sdnn(nn_intervals: &[f32]) -> f32 {
    if nn_intervals.len() < 2 {
        return 0.0;
    }

    let mean: f32 = nn_intervals.iter().sum::<f32>() / nn_intervals.len() as f32;
    let variance: f32 = nn_intervals.iter()
        .map(|v| {
            let diff = v - mean;
            diff * diff
        })
        .sum::<f32>() / (nn_intervals.len() - 1) as f32; 

    variance.sqrt()
}

/// Calculates the Root Mean Square of Successive Differences (RMSSD).
///
/// # Arguments
/// * `nn_intervals` - Slice of valid NN intervals in milliseconds.
///
/// # Returns
/// The RMSSD value in milliseconds as an `f32`.
pub fn calculate_rmssd(nn_intervals: &[f32]) -> f32 {
    if nn_intervals.len() < 2 {
        return 0.0;
    }

    let mut sum_sq_diff = 0.0;
    for i in 0..nn_intervals.len() - 1 {
        let diff = nn_intervals[i+1] - nn_intervals[i];
        sum_sq_diff += diff * diff;
    }

    (sum_sq_diff / (nn_intervals.len() - 1) as f32).sqrt()
}

/// Calculates the ratio of Low Frequency (LF) to High Frequency (HF) power.
///
/// # Arguments
/// * `nn_intervals` - Slice of valid NN intervals in milliseconds.
///
/// # Returns
/// The LF/HF power ratio as an `f32`.
pub fn calculate_lfhf(nn_intervals: &[f32]) -> f32 {
    const FS_R: f32 = 4.0;
    
    let resampled = resample_nn_intervals(nn_intervals, FS_R);
    let detrended = filters::detrend(&resampled, FS_R, 0.04);
    let mut scratch = crate::signal::fft::FftScratch::new();
    
    crate::signal::fft::compute_periodogram(&detrended, FS_R, 0.001, &mut scratch, false);
    
    let mut lf_power = 0.0;
    let mut hf_power = 0.0;

    for (i, &f) in scratch.frequencies.iter().enumerate() {
        if f >= 0.04 && f < 0.15 {
            lf_power += scratch.power[i];
        } else if f >= 0.15 && f <= 0.40 {
            hf_power += scratch.power[i];
        }
    }

    if hf_power > 0.0 {
        lf_power / hf_power
    } else {
        0.0
    }
}

/// Calculates the Baevsky Stress Index (SI).
///
/// # Arguments
/// * `nn_intervals_ms` - Slice of valid NN intervals in milliseconds.
///
/// # Returns
/// The Stress Index value as an `f32`.
pub fn calculate_stress_index(nn_intervals_ms: &[f32]) -> f32 {
    if nn_intervals_ms.len() < 5 { return 0.0; }

    let bin_size = 50.0;
    let min_val = nn_intervals_ms.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = nn_intervals_ms.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let range = max_val - min_val;
    if range < f32::EPSILON { return 0.0; }

    let max_bin_idx = (max_val / bin_size).floor() as usize;
    let mut histogram = vec![0; max_bin_idx + 1];

    for &interval in nn_intervals_ms {
        let bin_idx = (interval / bin_size).floor() as usize;
        if bin_idx < histogram.len() {
            histogram[bin_idx] += 1;
        }
    }

    let (max_count_idx, &max_count) = histogram.iter().enumerate()
        .max_by_key(|&(_, count)| count)
        .unwrap_or((0, &0));

    let amo = (max_count as f32 / nn_intervals_ms.len() as f32) * 100.0;
    let mode_ms = (max_count_idx as f32 * bin_size) + (bin_size / 2.0);
    
    let mo_sec = mode_ms / 1000.0;
    let mxdmn_sec = range / 1000.0;
    
    if mo_sec <= 0.0 || mxdmn_sec <= 0.0 { return 0.0; }

    amo / (2.0 * mo_sec * mxdmn_sec)
}

/// Calculates the percentage of successive NN intervals that differ by more than 50 ms (pNN50).
///
/// # Arguments
/// * `nn_intervals` - Slice of valid NN intervals in milliseconds.
///
/// # Returns
/// The pNN50 value as a percentage between 0.0 and 100.0 as an `f32`.
pub fn calculate_pnn50(nn_intervals: &[f32]) -> f32 {
    if nn_intervals.len() < 2 {
        return 0.0;
    }

    let mut count_above_50 = 0;
    let total_comparisons = nn_intervals.len() - 1;

    for i in 0..total_comparisons {
        let diff = (nn_intervals[i+1] - nn_intervals[i]).abs();
        if diff > 50.0 {
            count_above_50 += 1;
        }
    }

    if total_comparisons > 0 {
        (count_above_50 as f32 / total_comparisons as f32) * 100.0
    } else {
        0.0
    }
}

/// Calculates the ratio of SD1 (short-term) to SD2 (long-term) variability derived from a Poincaré plot.
///
/// # Arguments
/// * `nn_intervals` - Slice of valid NN intervals in milliseconds.
///
/// # Returns
/// The SD1/SD2 ratio as an `f32`.
pub fn calculate_sd1sd2(nn_intervals: &[f32]) -> f32 {
    if nn_intervals.len() < 3 {
        return 0.0;
    }

    let mut diff_sq_sum = 0.0;
    let mut sum_sq_sum = 0.0;
    let mut diff_sum = 0.0;
    let mut sum_sum = 0.0;
    
    let n = (nn_intervals.len() - 1) as f32;

    for i in 0..nn_intervals.len() - 1 {
        let x_i = nn_intervals[i];
        let x_next = nn_intervals[i+1];

        let v_perp = (x_next - x_i) / std::f32::consts::SQRT_2;
        diff_sum += v_perp;
        diff_sq_sum += v_perp * v_perp;

        let v_along = (x_next + x_i) / std::f32::consts::SQRT_2;
        sum_sum += v_along;
        sum_sq_sum += v_along * v_along;
    }
    
    let var_sd1 = (diff_sq_sum / n) - (diff_sum / n).powi(2);
    let var_sd2 = (sum_sq_sum / n) - (sum_sum / n).powi(2);

    let sd1 = var_sd1.sqrt();
    let sd2 = var_sd2.sqrt();

    if sd2 > 1e-6 {
        sd1 / sd2
    } else {
        0.0
    }
}

/// Removes extreme outliers from a sequence of NN intervals using a median-seeded bounds check.
fn filter_outliers(intervals: &[f32], _threshold: f32) -> Vec<f32> {
    if intervals.len() < 2 {
        return intervals.to_vec();
    }

    const MIN_RR: f32 = 300.0;
    const MAX_RR: f32 = 2000.0;
    const MAX_REL_CHANGE: f32 = 0.25;

    let mut valid_range_intervals: Vec<f32> = intervals.iter()
        .cloned()
        .filter(|&x| x >= MIN_RR && x <= MAX_RR)
        .collect();

    if valid_range_intervals.is_empty() {
        return Vec::new();
    }

    valid_range_intervals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = valid_range_intervals.len() / 2;
    let median = valid_range_intervals[mid];

    let mut filtered = Vec::with_capacity(intervals.len());
    let mut last_accepted = median; 

    for &curr in intervals {
        if curr < MIN_RR || curr > MAX_RR {
            continue;
        }

        let diff = (curr - last_accepted).abs();
        if diff > last_accepted * MAX_REL_CHANGE {
            continue;
        }

        filtered.push(curr);
        last_accepted = curr;
    }

    filtered
}

/// Resamples unevenly spaced NN intervals to a constant time-series frequency (e.g., 4Hz for LF/HF).
fn resample_nn_intervals(nn_intervals: &[f32], target_fs: f32) -> Vec<f32> {
    if nn_intervals.is_empty() { return Vec::new(); }

    let mut t = Vec::with_capacity(nn_intervals.len());
    let mut current_t = 0.0;
    for &interval_ms in nn_intervals {
        current_t += interval_ms / 1000.0;
        t.push(current_t);
    }

    let t_max = t[t.len() - 1];
    let step = 1.0 / target_fs;
    let num_samples = (t_max / step).floor() as usize;
    let mut resampled = Vec::with_capacity(num_samples);
    
    let mut cursor = 0;
    for i in 0..num_samples {
        let t_u = i as f32 * step;
        
        while cursor < t.len() - 1 && t[cursor + 1] < t_u {
            cursor += 1;
        }
        
        if cursor >= t.len() - 1 { break; }
        
        let x0 = t[cursor];
        let x1 = t[cursor + 1];
        let y0 = nn_intervals[cursor];
        let y1 = nn_intervals[cursor + 1];
        
        let val = y0 + (y1 - y0) * (t_u - x0) / (x1 - x0);
        resampled.push(val);
    }
    
    resampled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_intervals_interpolates_time() {
        let seq1 = vec![
            Peak { index: 0, x: 0.5, y: 1.0 }, 
            Peak { index: 1, x: 1.5, y: 1.0 }, 
        ];
        let sequences = vec![seq1];
        
        let timestamps = vec![0.0, 1.2, 2.2]; 
        let confidence = vec![1.0, 1.0, 1.0];

        let (intervals, _) = extract_nn_intervals(&sequences, Some(&timestamps), Some(&confidence), HrvOptions::default());
        
        assert_eq!(intervals.len(), 1);
        assert!((intervals[0] - 1100.0).abs() < 0.1);
    }

    #[test]
    fn test_extract_intervals_gates_low_confidence() {
        let seq1 = vec![
            Peak { index: 0, x: 0.0, y: 1.0 },
            Peak { index: 1, x: 1.0, y: 1.0 }, 
            Peak { index: 2, x: 2.0, y: 1.0 },
        ];
        let sequences = vec![seq1];
        let timestamps = vec![0.0, 1.0, 2.0];
        let confidence = vec![1.0, 0.1, 1.0]; 

        let (intervals, _) = extract_nn_intervals(&sequences, Some(&timestamps), Some(&confidence), HrvOptions::default());
        
        assert_eq!(intervals.len(), 0);
    }

    #[test]
    fn test_multiple_sequences_no_phantom_intervals() {
        let seq1 = vec![
            Peak { index: 0, x: 0.0, y: 1.0 },
            Peak { index: 1, x: 1.0, y: 1.0 },
        ];
        let seq2 = vec![
            Peak { index: 5, x: 5.0, y: 1.0 },
            Peak { index: 6, x: 6.0, y: 1.0 },
        ];
        let sequences = vec![seq1, seq2];
        
        let timestamps: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let confidence = vec![1.0; 8];

        let (intervals, _) = extract_nn_intervals(&sequences, Some(&timestamps), Some(&confidence), HrvOptions::default());
        
        assert_eq!(intervals.len(), 2);
        assert!((intervals[0] - 1000.0).abs() < 0.1);
        assert!((intervals[1] - 1000.0).abs() < 0.1);
    }

    #[test]
    fn test_filter_outliers_removes_extreme_values() {
        let raw = vec![800.0, 810.0, 805.0, 1500.0, 800.0];
        let filtered = filter_outliers(&raw, 0.3);
        assert_eq!(filtered.len(), 4);
        assert!(!filtered.contains(&1500.0));
    }

    #[test]
    fn test_fallback_to_fs() {
        let seq = vec![
            Peak { index: 0, x: 0.0, y: 1.0 },
            Peak { index: 10, x: 10.0, y: 1.0 },
        ];
        let sequences = vec![seq];
        let timestamps = vec![]; 
        let confidence = vec![1.0; 11]; 

        let mut opts = HrvOptions::default();
        opts.fs_fallback = 10.0;

        let (intervals, _) = extract_nn_intervals(&sequences, Some(&timestamps), Some(&confidence), opts);

        assert_eq!(intervals.len(), 1);
        assert!((intervals[0] - 1000.0).abs() < 0.1);
    }

    #[test]
    fn test_sub_sample_precision_fallback() {
        let seq = vec![
            Peak { index: 0, x: 0.7, y: 1.0 }, 
            Peak { index: 10, x: 10.5, y: 1.0 },
        ];
        let sequences = vec![seq];
        let timestamps = vec![];
        let confidence = vec![1.0; 11];

        let mut opts = HrvOptions::default();
        opts.fs_fallback = 10.0;

        let (intervals, _) = extract_nn_intervals(&sequences, Some(&timestamps), Some(&confidence), opts);

        assert_eq!(intervals.len(), 1);
        assert!((intervals[0] - 980.0).abs() < 0.1);
    }

    #[test]
    fn test_empty_and_single_sequences() {
        let (intervals, _) = extract_nn_intervals(&[], None, None, HrvOptions::default());
        assert!(intervals.is_empty());

        let seq = vec![Peak { index: 0, x: 0.0, y: 1.0 }];
        let sequences = vec![seq];
        let timestamps = vec![0.0];
        let confidence = vec![1.0];

        let (intervals, _) = extract_nn_intervals(&sequences, Some(&timestamps), Some(&confidence), HrvOptions::default());
        assert!(intervals.is_empty());
    }

    #[test]
    fn test_zero_length_intervals_dropped() {
        let seq = vec![
            Peak { index: 0, x: 0.0, y: 1.0 },
            Peak { index: 1, x: 1.0, y: 1.0 }, 
        ];
        let sequences = vec![seq];
        let timestamps = vec![1.0, 1.0]; 
        let confidence = vec![1.0, 1.0];

        let (intervals, _) = extract_nn_intervals(&sequences, Some(&timestamps), Some(&confidence), HrvOptions::default());

        assert!(intervals.is_empty());
    }

    #[test]
    fn test_bounds_safety() {
        let seq = vec![
            Peak { index: 0, x: 0.0, y: 1.0 },
            Peak { index: 5, x: 5.0, y: 1.0 }, 
        ];
        let sequences = vec![seq];
        let timestamps = vec![0.0]; 
        let confidence = vec![1.0; 6]; 

        let mut opts = HrvOptions::default();
        opts.fs_fallback = 1.0; 

        let (intervals, _) = extract_nn_intervals(&sequences, Some(&timestamps), Some(&confidence), opts);

        assert_eq!(intervals.len(), 1);
        assert!((intervals[0] - 5000.0).abs() < 0.1);
    }

    #[test]
    fn test_output_confidence_accuracy() {
        let seq = vec![
            Peak { index: 0, x: 0.0, y: 1.0 },
            Peak { index: 1, x: 1.0, y: 1.0 },
            Peak { index: 2, x: 2.0, y: 1.0 },
            Peak { index: 3, x: 3.0, y: 1.0 },
        ];
        let sequences = vec![seq];
        let timestamps = vec![0.0, 1.0, 2.0, 3.0];
        let confidence = vec![1.0, 0.6, 1.0, 0.4];

        let mut opts = HrvOptions::default();
        opts.confidence_threshold = 0.5;

        let (intervals, min_conf) = extract_nn_intervals(&sequences, Some(&timestamps), Some(&confidence), opts);

        assert_eq!(intervals.len(), 2); 
        assert!((min_conf - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_calculate_sdnn() {
        let intervals = vec![800.0, 900.0, 1000.0];
        let sdnn = calculate_sdnn(&intervals);
        assert!((sdnn - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_calculate_rmssd() {
        let intervals = vec![800.0, 900.0, 850.0];
        let rmssd = calculate_rmssd(&intervals);
        assert!((rmssd - 79.0569).abs() < 0.01);
    }

    #[test]
    fn test_resample_logic_accuracy() {
        let intervals = vec![1000.0; 10]; 
        let resampled = resample_nn_intervals(&intervals, 4.0);
        
        for &val in &resampled {
            assert!((val - 1000.0).abs() < 0.1);
        }
        assert!(resampled.len() >= 39 && resampled.len() <= 41);
    }

    #[test]
    fn test_lfhf_high_vagal_tone_simulation() {
        let mut intervals = Vec::new();
        for i in 0..100 {
            let t = i as f32;
            let val = 1000.0 + 200.0 * (2.0 * std::f32::consts::PI * 0.25 * t).sin();
            intervals.push(val);
        }

        let ratio = calculate_lfhf(&intervals);
        
        assert!(ratio < 0.5, "Expected low ratio for high-frequency oscillation, got {}", ratio);
    }

    #[test]
    fn test_calculate_stress_index_steady_rhythm() {
        let mut intervals = vec![1000.0; 10];
        intervals.extend_from_slice(&[950.0, 950.0, 1050.0, 1050.0]);
        
        let si = calculate_stress_index(&intervals);
        
        assert!(si > 300.0 && si < 400.0, "Expected SI ~357, got {}", si);
    }

    #[test]
    fn test_calculate_stress_index_high_variability() {
        let intervals = vec![600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0];
        
        let si = calculate_stress_index(&intervals);
        
        assert!(si < 50.0, "Expected low SI for high variability, got {}", si);
    }

    #[test]
    fn test_calculate_stress_index_minimum_data() {
        let short_intervals = vec![1000.0, 1010.0, 990.0, 1000.0];
        let si = calculate_stress_index(&short_intervals);
        assert_eq!(si, 0.0);
    }

    #[test]
    fn test_calculate_stress_index_identical_intervals() {
        let intervals = vec![1000.0; 10];
        let si = calculate_stress_index(&intervals);
        assert_eq!(si, 0.0);
    }

    #[test]
    fn test_calculate_stress_index_binning_boundary() {
        let intervals = vec![1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 1200.0];
        let si = calculate_stress_index(&intervals);
        
        assert!(si > 0.0);
        assert!(si > 190.0 && si < 220.0, "Expected SI ~203, got {}", si);
    }

    #[test]
    fn test_calculate_pnn50() {
        let stable = vec![800.0, 810.0, 805.0, 820.0]; 
        assert_eq!(calculate_pnn50(&stable), 0.0);

        let unstable = vec![800.0, 900.0, 800.0, 900.0];
        assert_eq!(calculate_pnn50(&unstable), 100.0);

        let mixed = vec![800.0, 810.0, 900.0, 910.0];
        let pnn50 = calculate_pnn50(&mixed);
        assert!((pnn50 - 33.333).abs() < 0.01, "Expected ~33.33, got {}", pnn50);
    }

    #[test]
    fn test_calculate_sd1sd2() {
        let linear = vec![1000.0, 1000.0, 1000.0, 1000.0];
        assert_eq!(calculate_sd1sd2(&linear), 0.0);

        let alternating = vec![1000.0, 800.0, 1000.0, 800.0];
        assert_eq!(calculate_sd1sd2(&alternating), 0.0); 

        let data = vec![800.0, 810.0, 820.0, 810.0, 800.0];
        let ratio = calculate_sd1sd2(&data);
        
        assert!(ratio > 0.0);
        assert!(ratio < 1.0, "Expected SD1 < SD2 for smooth rhythm, got ratio {}", ratio);
    }

    #[test]
    fn test_estimate_hrv_confidence_logic() {
        let fs = 30.0;
        let duration = 5.0;
        let total_samples = (fs * duration) as usize;
        
        let signal: Vec<f32> = (0..total_samples)
            .map(|i| (i as f32 / fs * 2.0 * std::f32::consts::PI * 1.0).sin())
            .collect();
            
        let mut confidence = vec![0.2; total_samples];
        for i in 0..total_samples {
            if signal[i] > 0.9 {
                confidence[i] = 1.0;
            }
        }

        let timestamps: Vec<f64> = (0..total_samples).map(|i| i as f64 / fs as f64).collect();
        let bounds = SignalBounds { min_rate: 40.0, max_rate: 220.0 };

        let (_, final_conf) = estimate_hrv(&signal, fs, crate::registry::HrvMetric::Sdnn, &timestamps, &confidence, bounds, None);

        let avg_conf = confidence.iter().sum::<f32>() / total_samples as f32;
        
        assert!(final_conf < 1.0);
        assert!((final_conf - avg_conf).abs() < 1e-4);
    }

    #[test]
    fn test_estimate_hrv_from_detections_minimal_none() {
        let seq = vec![
            Peak { index: 0, x: 0.0, y: 1.0 },
            Peak { index: 1, x: 1.0, y: 1.0 },
            Peak { index: 2, x: 2.0, y: 1.0 },
        ];
        
        let (val, conf) = estimate_hrv_from_detections(
            &[seq], 
            1.0,
            crate::registry::HrvMetric::Sdnn, 
            None, 
            None
        );

        assert_eq!(val, 0.0);
        assert_eq!(conf, 1.0);
    }

    #[test]
    fn test_estimate_hrv_from_detections_with_provided_confidence() {
        let seq = vec![
            Peak { index: 0, x: 0.0, y: 1.0 },
            Peak { index: 1, x: 1.0, y: 1.0 },
        ];
        
        // Confidence slice provided
        let confidence = vec![1.0, 0.7];
        
        let (_, conf) = estimate_hrv_from_detections(
            &[seq], 
            1.0, 
            crate::registry::HrvMetric::Sdnn, 
            None, 
            Some(&confidence)
        );

        assert!((conf - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_hrv_from_detections_with_provided_timestamps() {
        let seq = vec![
            Peak { index: 0, x: 0.0, y: 1.0 },
            Peak { index: 1, x: 1.0, y: 1.0 },
            Peak { index: 2, x: 2.0, y: 1.0 },
        ];
        
        let timestamps = vec![0.0, 1.0, 2.1]; 
        
        let (val, _) = estimate_hrv_from_detections(
            &[seq], 
            10.0, 
            crate::registry::HrvMetric::Rmssd, 
            Some(&timestamps), 
            None
        );

        assert!((val - 100.0).abs() < 0.1, "Expected RMSSD 100.0, got {}", val);
    }

    // Helper for concise peak creation in tests
    fn p(x: f32) -> Peak {
        Peak { index: x as usize, x, y: 1.0 }
    }

    #[test]
    fn test_estimate_rolling_hrv_basic_time_window() {
        let fs = 10.0;
        let total_frames = 100;
        
        // Peaks with alternating intervals: 8 frames (800ms) and 10 frames (1000ms)
        // Times: 0.0s, 0.8s, 1.8s, 2.6s, 3.6s, 4.4s, 5.4s, 6.2s
        let peak_indices = vec![0, 8, 18, 26, 36, 44, 54, 62];
        let segment: Vec<Peak> = peak_indices.into_iter().map(|i| p(i as f32)).collect();
        
        let (vals, confs) = estimate_rolling_hrv_from_detections(
            &[segment], fs, crate::registry::HrvMetric::Rmssd, None, None,
            2.0, // min_window_seconds
            5.0, // preferred_window_seconds
            1.0, // rolling_stride_seconds
            total_frames
        );

        assert_eq!(vals.len(), total_frames);
        assert_eq!(confs.len(), total_frames);

        // Before 2.0 seconds (e.g., peak at 1.8s -> index 18), HRV should be NaN
        assert!(vals[18].is_nan());
        assert!(vals[25].is_nan());
        
        // At 2.6s (index 26), window is [0, 8, 18, 26]
        // Intervals: 800, 1000, 800. RMSSD of alternating 800/1000 is exactly 200.0 ms.
        assert!((vals[26] - 200.0).abs() < 0.1);
        
        // Value carries forward to the next peak (index 36)
        assert!((vals[35] - 200.0).abs() < 0.1);
        
        // Last peak is at 62. The last interval was 54 to 62 (8 frames).
        // So it holds until 62 + 8 = 70. At 70 it drops to NaN.
        assert!((vals[69] - 200.0).abs() < 0.1);
        assert!(vals[70].is_nan());
        assert!(vals[99].is_nan());
    }

    #[test]
    fn test_estimate_rolling_hrv_stride_logic() {
        let fs = 10.0;
        let total_frames = 100;
        
        // Steady rhythm: 0s, 1s, 2s, 3s, 4s, 5s
        let peak_indices = vec![0, 10, 20, 30, 40, 50];
        let segment: Vec<Peak> = peak_indices.into_iter().map(|i| p(i as f32)).collect();

        let (vals, _) = estimate_rolling_hrv_from_detections(
            &[segment], fs, crate::registry::HrvMetric::Sdnn, None, None,
            2.0, // min_window_seconds
            5.0, // preferred_window_seconds
            3.0, // rolling_stride_seconds (large stride)
            total_frames
        );

        // First calculation triggers at 2.0s (index 20)
        assert!(vals[19].is_nan());
        assert!(!vals[20].is_nan());
        
        let first_calc_val = vals[20];
        
        // Because the stride is 3.0s, the peaks at 3.0s (idx 30) and 4.0s (idx 40) 
        // will just hold the exact same value without recalculating.
        assert_eq!(vals[30], first_calc_val);
        assert_eq!(vals[40], first_calc_val);
        
        // At 5.0s (idx 50), 3.0s have passed since the 2.0s calculation, 
        // so it recalculates.
        assert!(!vals[50].is_nan());
    }

    #[test]
    fn test_estimate_rolling_hrv_multiple_segments() {
        let fs = 10.0;
        let total_frames = 120;
        
        // Segment 1: 0s to 3s, steady 10 frames (1s) -> SDNN = 0.0
        let seg1: Vec<Peak> = (0..4).map(|i| p((i * 10) as f32)).collect();
        
        // Segment 2: 6s to 10s, steady 10 frames (1s) -> SDNN = 0.0
        let seg2: Vec<Peak> = (0..5).map(|i| p((60 + i * 10) as f32)).collect();

        let (vals, _) = estimate_rolling_hrv_from_detections(
            &[seg1, seg2], fs, crate::registry::HrvMetric::Sdnn, None, None,
            2.0, 5.0, 1.0, total_frames
        );

        // Segment 1 becomes valid at 2.0s (index 20)
        assert!((vals[20] - 0.0).abs() < 0.1);
        
        // Segment 1's last peak is at 30, interval is 10. Holds until 40, then NaNs.
        assert!((vals[39] - 0.0).abs() < 0.1);
        assert!(vals[40].is_nan());
        assert!(vals[59].is_nan());
        
        // Segment 2 starts at 60, becomes valid at 80 (after 2s).
        assert!(vals[79].is_nan());
        assert!((vals[80] - 0.0).abs() < 0.1);
        
        // Segment 2's last peak is at 100, interval is 10. Holds until 110, then NaNs.
        assert!((vals[109] - 0.0).abs() < 0.1);
        assert!(vals[110].is_nan());
        assert!(vals[119].is_nan());
    }
}