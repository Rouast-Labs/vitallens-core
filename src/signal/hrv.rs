use crate::signal::peaks::{Peak, PeakOptions, SignalBounds, find_peaks};
use crate::registry::HrvMetric;

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

/// High-level coordinator for HRV metrics.
/// Detects peaks, extracts gated intervals, and computes the requested metric.
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
        threshold: 0.5,
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
    
    let (intervals, min_conf) = extract_nn_intervals(&sequences, timestamps, confidence, hrv_opts);

    if intervals.is_empty() {
        return (0.0, 0.0);
    }

    let value = match metric {
        crate::registry::HrvMetric::Sdnn => calculate_sdnn(&intervals),
        crate::registry::HrvMetric::Rmssd => calculate_rmssd(&intervals),
        crate::registry::HrvMetric::LfHf => calculate_lfhf(&intervals),
        crate::registry::HrvMetric::StressIndex => calculate_stress_index(&intervals),
    };

    (value, min_conf)
}

/// Extracts valid Normal-to-Normal (NN) intervals from multiple sequences of peaks.
/// 
/// This function prevents "phantom intervals" by only calculating differences within
/// the provided continuous sequences. It interpolates timestamps for sub-sample precision.
/// 
/// Returns a tuple: `(intervals_ms, min_confidence_used)`
pub fn extract_nn_intervals(
    sequences: &[Vec<Peak>],
    timestamps: &[f64],
    confidence: &[f32],
    options: HrvOptions
) -> (Vec<f32>, f32) {
    let mut all_intervals = Vec::new();
    let mut all_used_confidences = Vec::new();

    for sequence in sequences {
        if sequence.len() < 2 { continue; }

        // Iterate over adjacent pairs in the original sequence
        for i in 0..sequence.len() - 1 {
            let p1 = &sequence[i];
            let p2 = &sequence[i+1];

            // 1. Check Bounds
            if p1.index >= confidence.len() || p2.index >= confidence.len() {
                continue;
            }

            let c1 = confidence[p1.index];
            let c2 = confidence[p2.index];

            // 2. Confidence Gating
            // If EITHER peak in the pair is low confidence, we drop this specific interval.
            // We do NOT bridge the gap.
            if c1 >= options.confidence_threshold && c2 >= options.confidence_threshold {
                
                // 3. Resolve Timestamps
                let t1 = resolve_time(p1, timestamps, options.fs_fallback);
                let t2 = resolve_time(p2, timestamps, options.fs_fallback);

                let diff_sec = t2 - t1;
                
                if diff_sec > 0.0 {
                    all_intervals.push((diff_sec * 1000.0) as f32);
                    
                    // Track confidence
                    all_used_confidences.push(c1);
                    all_used_confidences.push(c2);
                }
            }
        }
    }

    if all_intervals.is_empty() {
        return (Vec::new(), 0.0);
    }

    // 4. Filter Outliers (Global Median Filter)
    let filtered_intervals = filter_outliers(&all_intervals, options.outlier_threshold);

    if filtered_intervals.is_empty() {
        return (Vec::new(), 0.0);
    }

    // 5. Calculate Aggregate Confidence
    let min_conf = all_used_confidences.iter().fold(1.0f32, |a, &b| a.min(b));

    (filtered_intervals, min_conf)
}

/// Standard Deviation of NN intervals (SDNN).
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
        .sum::<f32>() / (nn_intervals.len() - 1) as f32; // Bessel's correction

    variance.sqrt()
}

/// Root Mean Square of Successive Differences (RMSSD).
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

/// Calculates the LF/HF ratio (Autonomic Balance).
/// Uses a high-resolution Periodogram on 4Hz resampled data.
pub fn calculate_lfhf(nn_intervals: &[f32]) -> f32 {
    const FS_R: f32 = 4.0;
    let resampled = resample_nn_intervals(nn_intervals, FS_R);
    
    let mut scratch = crate::signal::fft::FftScratch::new();
    
    crate::signal::fft::compute_periodogram(&resampled, FS_R, 0.001, &mut scratch);
    
    let mut lf_power = 0.0;
    let mut hf_power = 0.0;

    for (i, &f) in scratch.frequencies.iter().enumerate() {
        
        if f >= 0.04 && f < 0.15 {
            lf_power += scratch.power[i];
        } 
        
        else if f >= 0.15 && f <= 0.40 {
            hf_power += scratch.power[i];
        }
    }

    if hf_power > 0.0 {
        lf_power / hf_power
    } else {
        0.0
    }
}

/// Calculates Baevsky Stress Index (SI).
/// SI = AMo / (2 * Mo * MxDMn)
/// - AMo: Amplitude of Mode (percent of intervals in the mode bin)
/// - Mo: Mode (most frequent interval value in seconds)
/// - MxDMn: Variational range (max - min interval in seconds)
pub fn calculate_stress_index(nn_intervals_ms: &[f32]) -> f32 {
    if nn_intervals_ms.len() < 5 { return 0.0; }

    let bin_size = 50.0; // Standard 50ms bins
    let min_val = nn_intervals_ms.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = nn_intervals_ms.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    if (max_val - min_val).abs() < f32::EPSILON { return 0.0; }

    let num_bins = ((max_val - min_val) / bin_size).ceil() as usize + 1;
    let mut histogram = vec![0; num_bins];

    for &interval in nn_intervals_ms {
        let bin_idx = ((interval - min_val) / bin_size).floor() as usize;
        if bin_idx < histogram.len() {
            histogram[bin_idx] += 1;
        }
    }

    let (max_count_idx, &max_count) = histogram.iter().enumerate()
        .max_by_key(|&(_, count)| count)
        .unwrap_or((0, &0));

    if max_count == 0 { return 0.0; }

    // AMo: Amplitude of Mode (%)
    let amo = (max_count as f32 / nn_intervals_ms.len() as f32) * 100.0;

    // Mo: Mode (seconds) - center of the bin
    let mode_ms = min_val + (max_count_idx as f32 * bin_size) + (bin_size / 2.0);
    let mo_sec = mode_ms / 1000.0;

    // MxDMn: Range (seconds)
    let mxdmn_sec = (max_val - min_val) / 1000.0;
    
    // Prevent division by zero
    if mo_sec <= 0.0 || mxdmn_sec <= 0.0 { return 0.0; }

    // Formula: SI = AMo / (2 * Mo * MxDMn)
    amo / (2.0 * mo_sec * mxdmn_sec)
}

// --- Helpers ---

fn resolve_time(p: &Peak, timestamps: &[f64], fs_fallback: f32) -> f64 {
    if timestamps.len() > p.index + 1 {
        // Interpolate
        let t_floor = timestamps[p.index];
        let t_ceil = timestamps[p.index + 1];
        let fraction = p.x - p.index as f32;
        t_floor + (t_ceil - t_floor) * fraction as f64
    } else if !timestamps.is_empty() && p.index < timestamps.len() {
        // Fallback: Exact timestamp
        timestamps[p.index]
    } else {
        // Fallback: FS
        p.x as f64 / fs_fallback as f64
    }
}

fn filter_outliers(intervals: &[f32], threshold: f32) -> Vec<f32> {
    if intervals.len() < 3 {
        return intervals.to_vec();
    }

    let mut sorted = intervals.to_vec();
    // sort_by handles NaNs safely by pushing them to the end or panicking (we assume no NaNs here)
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mid = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    };

    let lower_bound = median * (1.0 - threshold);
    let upper_bound = median * (1.0 + threshold);

    intervals.iter()
        .filter(|&&x| x >= lower_bound && x <= upper_bound)
        .cloned()
        .collect()
}

/// Resamples unevenly spaced NN intervals to a constant 4Hz time-series.
fn resample_nn_intervals(nn_intervals: &[f32], target_fs: f32) -> Vec<f32> {
    if nn_intervals.is_empty() { return Vec::new(); }

    // 1. Create cumulative time axis (in seconds)
    // We place each interval at its ending time.
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
        
        // Find interval in t that surrounds t_u
        while cursor < t.len() - 1 && t[cursor + 1] < t_u {
            cursor += 1;
        }
        
        if cursor >= t.len() - 1 { break; }
        
        // Linear Interpolation: y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
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
            Peak { index: 0, x: 0.5, y: 1.0 }, // t=0.6
            Peak { index: 1, x: 1.5, y: 1.0 }, // t=1.7
        ];
        let sequences = vec![seq1];
        
        let timestamps = vec![0.0, 1.2, 2.2]; 
        let confidence = vec![1.0, 1.0, 1.0];

        let (intervals, _) = extract_nn_intervals(&sequences, &timestamps, &confidence, HrvOptions::default());
        
        assert_eq!(intervals.len(), 1);
        assert!((intervals[0] - 1100.0).abs() < 0.1);
    }

    #[test]
    fn test_extract_intervals_gates_low_confidence() {
        let seq1 = vec![
            Peak { index: 0, x: 0.0, y: 1.0 },
            Peak { index: 1, x: 1.0, y: 1.0 }, // Low confidence -> Dropped
            Peak { index: 2, x: 2.0, y: 1.0 },
        ];
        let sequences = vec![seq1];
        let timestamps = vec![0.0, 1.0, 2.0];
        let confidence = vec![1.0, 0.1, 1.0]; 

        let (intervals, _) = extract_nn_intervals(&sequences, &timestamps, &confidence, HrvOptions::default());
        
        // There should be no valid intervals
        assert_eq!(intervals.len(), 0);
    }

    #[test]
    fn test_multiple_sequences_no_phantom_intervals() {
        // Sequence 1: Ends at t=1.0
        let seq1 = vec![
            Peak { index: 0, x: 0.0, y: 1.0 },
            Peak { index: 1, x: 1.0, y: 1.0 },
        ];
        // Sequence 2: Starts at t=5.0 (Gap of 4s)
        let seq2 = vec![
            Peak { index: 5, x: 5.0, y: 1.0 },
            Peak { index: 6, x: 6.0, y: 1.0 },
        ];
        let sequences = vec![seq1, seq2];
        
        // Mock timestamps for indices 0..7
        let timestamps: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let confidence = vec![1.0; 8];

        let (intervals, _) = extract_nn_intervals(&sequences, &timestamps, &confidence, HrvOptions::default());
        
        // Expecting:
        // Seq 1: 1 interval (1000ms)
        // Seq 2: 1 interval (1000ms)
        // Total: 2 intervals
        // We MUST NOT see a 4000ms interval
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
        // Scenario: No timestamps provided (empty vector).
        // Expectation: Use p.x / fs_fallback.
        // FS = 10Hz. Peak at 0 (0.0s) and 10 (1.0s). Interval = 1.0s = 1000ms.
        let seq = vec![
            Peak { index: 0, x: 0.0, y: 1.0 },
            Peak { index: 10, x: 10.0, y: 1.0 },
        ];
        let sequences = vec![seq];
        let timestamps = vec![]; 
        // Need confidence array large enough for indices 0..10
        let confidence = vec![1.0; 11]; 

        let mut opts = HrvOptions::default();
        opts.fs_fallback = 10.0;

        let (intervals, _) = extract_nn_intervals(&sequences, &timestamps, &confidence, opts);

        assert_eq!(intervals.len(), 1);
        assert!((intervals[0] - 1000.0).abs() < 0.1);
    }

    #[test]
    fn test_sub_sample_precision_fallback() {
        // Scenario: No timestamps, but peaks have sub-sample precision.
        // FS = 10Hz. 
        // P1 at 0.7 (0.07s)
        // P2 at 10.5 (1.05s)
        // Diff = 980ms.
        let seq = vec![
            Peak { index: 0, x: 0.7, y: 1.0 }, 
            Peak { index: 10, x: 10.5, y: 1.0 },
        ];
        let sequences = vec![seq];
        let timestamps = vec![];
        let confidence = vec![1.0; 11];

        let mut opts = HrvOptions::default();
        opts.fs_fallback = 10.0;

        let (intervals, _) = extract_nn_intervals(&sequences, &timestamps, &confidence, opts);

        assert_eq!(intervals.len(), 1);
        assert!((intervals[0] - 980.0).abs() < 0.1);
    }

    #[test]
    fn test_empty_and_single_sequences() {
        // Scenario 1: Empty input
        let (intervals, _) = extract_nn_intervals(&[], &[], &[], HrvOptions::default());
        assert!(intervals.is_empty());

        // Scenario 2: Sequence with single peak (cannot form interval)
        let seq = vec![Peak { index: 0, x: 0.0, y: 1.0 }];
        let sequences = vec![seq];
        let timestamps = vec![0.0];
        let confidence = vec![1.0];

        let (intervals, _) = extract_nn_intervals(&sequences, &timestamps, &confidence, HrvOptions::default());
        assert!(intervals.is_empty());
    }

    #[test]
    fn test_zero_length_intervals_dropped() {
        // Scenario: Bad data where two distinct peaks map to same timestamp.
        let seq = vec![
            Peak { index: 0, x: 0.0, y: 1.0 },
            Peak { index: 1, x: 1.0, y: 1.0 }, 
        ];
        let sequences = vec![seq];
        // Timestamps are identical!
        let timestamps = vec![1.0, 1.0]; 
        let confidence = vec![1.0, 1.0];

        let (intervals, _) = extract_nn_intervals(&sequences, &timestamps, &confidence, HrvOptions::default());

        // Should drop the 0ms interval
        assert!(intervals.is_empty());
    }

    #[test]
    fn test_bounds_safety() {
        // Scenario: Peak index exceeds timestamp length.
        // Should fall back to FS instead of panicking.
        let seq = vec![
            Peak { index: 0, x: 0.0, y: 1.0 },
            Peak { index: 5, x: 5.0, y: 1.0 }, // Index 5 exists in confidence, but not timestamps
        ];
        let sequences = vec![seq];
        let timestamps = vec![0.0]; // Only 1 timestamp provided
        let confidence = vec![1.0; 6]; // Confidence exists

        let mut opts = HrvOptions::default();
        opts.fs_fallback = 1.0; // 1Hz fallback

        let (intervals, _) = extract_nn_intervals(&sequences, &timestamps, &confidence, opts);

        // P1(0) -> T=0.0 (from timestamp)
        // P2(5) -> T=5.0 (from fallback 5.0/1.0)
        // Interval = 5000ms
        assert_eq!(intervals.len(), 1);
        assert!((intervals[0] - 5000.0).abs() < 0.1);
    }

    #[test]
    fn test_output_confidence_accuracy() {
        // Scenario: 
        // P1 (1.0) --[Valid]--> P2 (0.6) --[Valid]--> P3 (1.0)
        // P3 (1.0) --[Invalid]--> P4 (0.4) 
        //
        // Valid Intervals: P1-P2 (uses 1.0, 0.6), P2-P3 (uses 0.6, 1.0).
        // Minimum used confidence should be 0.6.
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

        let (intervals, min_conf) = extract_nn_intervals(&sequences, &timestamps, &confidence, opts);

        assert_eq!(intervals.len(), 2); // P1-P2, P2-P3. (P3-P4 dropped).
        assert!((min_conf - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_calculate_sdnn() {
        // Data: [800, 900, 1000]
        // Mean: 900
        // Variance: ((800-900)^2 + (900-900)^2 + (1000-900)^2) / (3-1)
        // Variance: (10000 + 0 + 10000) / 2 = 10000
        // SD: sqrt(10000) = 100
        let intervals = vec![800.0, 900.0, 1000.0];
        let sdnn = calculate_sdnn(&intervals);
        assert!((sdnn - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_calculate_rmssd() {
        // Intervals: [800, 900, 850]
        // Diffs: [100, -50]
        // Sq Diffs: [10000, 2500]
        // Mean Sq: (10000 + 2500) / 2 = 6250
        // RMSSD: sqrt(6250) = 79.0569
        let intervals = vec![800.0, 900.0, 850.0];
        let rmssd = calculate_rmssd(&intervals);
        assert!((rmssd - 79.0569).abs() < 0.01);
    }

    #[test]
    fn test_resample_logic_accuracy() {
        // Constant HR of 60 BPM = 1000ms intervals
        let intervals = vec![1000.0; 10]; 
        let resampled = resample_nn_intervals(&intervals, 4.0);
        
        // All resampled points should be exactly 1000ms
        for &val in &resampled {
            assert!((val - 1000.0).abs() < 0.1);
        }
        // Total time 10s @ 4Hz = ~40 samples
        assert!(resampled.len() >= 39 && resampled.len() <= 41);
    }

    #[test]
    fn test_lfhf_high_vagal_tone_simulation() {
        // Simulate Deep Breathing (0.25 Hz / 4s period)
        // This should put massive power in the HF band (0.15-0.4Hz)
        let mut intervals = Vec::new();
        for i in 0..100 {
            let t = i as f32;
            // HR oscillates between 800ms and 1200ms every 4 seconds
            let val = 1000.0 + 200.0 * (2.0 * std::f32::consts::PI * 0.25 * t).sin();
            intervals.push(val);
        }

        let ratio = calculate_lfhf(&intervals);
        
        // High HF power means LF/HF ratio should be very low (< 1.0)
        assert!(ratio < 0.5, "Expected low ratio for high-frequency oscillation, got {}", ratio);
    }

    #[test]
    fn test_calculate_stress_index_steady_rhythm() {
        // High stress simulation: Most intervals are very similar
        // Intervals: 10 of 1000ms, 2 of 950ms, 2 of 1050ms
        let mut intervals = vec![1000.0; 10];
        intervals.extend_from_slice(&[950.0, 950.0, 1050.0, 1050.0]);
        
        let si = calculate_stress_index(&intervals);
        
        // AMo = 10/14 = ~71.4%
        // Mo = 1.0s
        // MxDMn = (1050 - 950) / 1000 = 0.1s
        // SI = 71.4 / (2 * 1.0 * 0.1) = ~357
        assert!(si > 300.0 && si < 400.0, "Expected SI ~357, got {}", si);
    }

    #[test]
    fn test_calculate_stress_index_high_variability() {
        // Low stress simulation: Intervals spread across many bins
        let intervals = vec![600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0];
        
        let si = calculate_stress_index(&intervals);
        
        // Range is large (0.6s), and intervals are spread (AMo is low)
        // SI should be significantly lower than the steady case
        assert!(si < 50.0, "Expected low SI for high variability, got {}", si);
    }

    #[test]
    fn test_calculate_stress_index_minimum_data() {
        // Implementation requires at least 5 intervals
        let short_intervals = vec![1000.0, 1010.0, 990.0, 1000.0];
        let si = calculate_stress_index(&short_intervals);
        assert_eq!(si, 0.0);
    }

    #[test]
    fn test_calculate_stress_index_identical_intervals() {
        // Prevent division by zero if MxDMn is 0
        let intervals = vec![1000.0; 10];
        let si = calculate_stress_index(&intervals);
        assert_eq!(si, 0.0);
    }

    #[test]
    fn test_calculate_stress_index_binning_boundary() {
        // Ensure bins (50ms) are handled correctly.
        // 1000, 1010, 1020, 1030, 1040 should all fall into the same bin 
        // if the min_val starts at 1000.
        let intervals = vec![1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 1200.0];
        let si = calculate_stress_index(&intervals);
        
        assert!(si > 0.0);
        // AMo should be 5/6 (83.3%)
        // MxDMn should be 0.2s
        // Mo should be ~1.025s
        // SI = 83.3 / (2 * 1.025 * 0.2) = ~203
        assert!(si > 190.0 && si < 220.0, "Expected SI ~203, got {}", si);
    }
}