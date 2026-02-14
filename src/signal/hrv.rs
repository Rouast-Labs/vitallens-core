use crate::signal::peaks::Peak;

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
    
    // Compute PSD using the refactored engine in fft.rs
    // High resolution (0.001 Hz) ensures clean band separation
    let psd = crate::signal::fft::compute_periodogram(&resampled, FS_R, 0.001);
    
    let mut lf_power = 0.0;
    let mut hf_power = 0.0;

    for (i, &f) in psd.frequencies.iter().enumerate() {
        // LF: 0.04 - 0.15 Hz
        if f >= 0.04 && f < 0.15 {
            lf_power += psd.power[i];
        } 
        // HF: 0.15 - 0.40 Hz
        else if f >= 0.15 && f <= 0.40 {
            hf_power += psd.power[i];
        }
    }

    if hf_power > 0.0 {
        lf_power / hf_power
    } else {
        0.0
    }
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
        
        // Should calculate interval between Peak 0 and Peak 2 directly
        // TODO: Shouldn't there be no valid intervals here?
        assert_eq!(intervals.len(), 0);
        // assert!((intervals[0] - 2000.0).abs() < 0.1);
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
}