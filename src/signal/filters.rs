use std::f32;

pub fn apply_processing(
    signal: &[f32], 
    op: crate::registry::PostProcessOp,
    fs: f32,
) -> Vec<f32> {
    match op {
        crate::registry::PostProcessOp::Detrend => detrend(signal, fs),
        crate::registry::PostProcessOp::Standardize => standardize(signal),
        crate::registry::PostProcessOp::None => signal.to_vec(),
    }
}

/// Calculates the centered moving average of a signal.
/// 
/// This implementation uses a shrinking window at the boundaries to avoid padding artifacts.
/// Complexity: O(N * W) currently (simple slice sum).
///
/// # Arguments
/// * `signal` - Input data
/// * `window_size` - Number of frames to average (should be odd for perfect centering)
pub fn moving_average(signal: &[f32], window_size: usize) -> Vec<f32> {
    let len = signal.len();
    if len == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(len);
    let half_window = window_size / 2;

    for i in 0..len {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(len);
        
        let slice = &signal[start..end];
        let sum: f32 = slice.iter().sum();
        result.push(sum / slice.len() as f32);
    }

    result
}

/// Estimates the required moving average window size to achieve a specific
/// low-pass cutoff frequency.
///
/// # Arguments
/// * `fs` - Sampling frequency
/// * `cutoff_hz` - Desired cutoff frequency
/// * `force_odd` - If true, ensures the result is odd (floors even numbers, e.g., 4 -> 3)
pub fn estimate_moving_average_window(fs: f32, cutoff_hz: f32, force_odd: bool) -> usize {
    if fs <= 0.0 || cutoff_hz <= 0.0 {
        return 1;
    }

    let f = cutoff_hz / fs;
    if f.abs() < 1e-6 {
        return usize::MAX; 
    }

    // Approximation for SMA cutoff
    let size = (0.196202 + f * f).sqrt() / f;
    let mut size_int = size as usize;
    
    if force_odd && size_int % 2 == 0 {
        // Prefer slightly smaller odd window (higher cutoff) than larger (lower cutoff)
        size_int = size_int.saturating_sub(1);
    }
    
    size_int.max(1)
}

/// Removes low-frequency trends by subtracting a moving average.
/// This is a high-pass filter equivalent suitable for rPPG.
// TODO: Proper implementation (from prpy)
// TODO: Test (from prpy)
pub fn detrend(signal: &[f32], fs: f32) -> Vec<f32> {
    // Standard window for rPPG detrending is roughly 1.0 second (fs frames)
    let window_size = fs.ceil() as usize; 
    
    let trend = moving_average(signal, window_size);
    
    signal.iter()
        .zip(trend.iter())
        .map(|(raw, tr)| raw - tr)
        .collect()
}

/// Z-Score normalization (Zero Mean, Unit Variance).
/// Robust to outliers? No, standard Z-score. 
/// Vital for FFT to prevent spectral leakage from offsets.
// TODO: Check implementation (from prpy)
// TODO: Test (from prpy)
pub fn standardize(signal: &[f32]) -> Vec<f32> {
    if signal.is_empty() {
        return Vec::new();
    }

    let len = signal.len() as f32;
    let mean = signal.iter().sum::<f32>() / len;
    
    let variance = signal.iter()
        .map(|v| {
            let diff = v - mean;
            diff * diff
        })
        .sum::<f32>() / len;

    let std_dev = variance.sqrt();

    // Prevent division by zero if signal is flat
    if std_dev.abs() < 1e-6 {
        return vec![0.0; signal.len()];
    }

    signal.iter()
        .map(|v| (v - mean) / std_dev)
        .collect()
}

// --- UNIT TESTS ---
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ma_impulse_response() {
        // Input: [0, 0, 10, 0, 0]
        // Window 3:
        // i=0 (win [0,0]): 0
        // i=1 (win [0,0,10]): 3.33
        // i=2 (win [0,10,0]): 3.33
        // i=3 (win [10,0,0]): 3.33
        // i=4 (win [0,0]): 0
        let data = vec![0.0, 0.0, 10.0, 0.0, 0.0];
        let result = moving_average(&data, 3);
        
        assert_eq!(result.len(), 5);
        assert!((result[2] - 3.333).abs() < 0.01); 
        // Peak should be centered
        assert!(result[2] >= result[1] && result[2] >= result[3]);
    }

    #[test]
    fn test_ma_step_response_smoothing() {
        // Step from 0 to 1
        let data = vec![0.0, 0.0, 1.0, 1.0, 1.0];
        let result = moving_average(&data, 3);
        
        // Should transition smoothly
        assert!(result[1] > 0.0);       // Starts rising before step (lookahead/centering)
        assert!(result[2] > result[1]); // Rising
        assert!((result[4] - 1.0).abs() < 0.01); // Settles at 1.0
    }

    #[test]
    fn test_ma_boundary_handling() {
        let data = vec![10.0, 20.0, 30.0];
        let result = moving_average(&data, 3);
        
        // i=0: window is [10, 20] (size 2) -> 15.0
        assert_eq!(result[0], 15.0);
        
        // i=1: window is [10, 20, 30] (size 3) -> 20.0
        assert_eq!(result[1], 20.0);

        // i=2: window is [20, 30] (size 2) -> 25.0
        assert_eq!(result[2], 25.0);
    }

    #[test]
    fn test_ma_window_larger_than_signal() {
        let data = vec![1.0, 2.0, 3.0];
        // Window 10 > Len 3
        // Should return average of whole signal for all points
        let result = moving_average(&data, 10);
        
        assert_eq!(result.len(), 3);
        for x in result {
            assert_eq!(x, 2.0); // Mean of 1,2,3 is 2
        }
    }

    #[test]
    fn test_ma_empty_input() {
        let data: Vec<f32> = vec![];
        let result = moving_average(&data, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_estimate_window_inverse_relationship() {
        let fs = 30.0;
        // Higher cutoff = Smaller window
        let w_high_cutoff = estimate_moving_average_window(fs, 5.0, true);
        // Lower cutoff = Larger window
        let w_low_cutoff = estimate_moving_average_window(fs, 1.0, true);
        
        assert!(w_low_cutoff > w_high_cutoff, 
            "Lower cutoff (1Hz) should require larger window than high cutoff (5Hz)");
    }

    #[test]
    fn test_estimate_window_fs_relationship() {
        let cutoff = 2.0;
        // Higher FS = More samples needed for same duration
        let w_high_fs = estimate_moving_average_window(60.0, cutoff, true);
        let w_low_fs = estimate_moving_average_window(30.0, cutoff, true);

        assert!(w_high_fs > w_low_fs,
            "Higher sampling rate should require more samples for same cutoff");
    }

    #[test]
    fn test_estimate_window_force_odd() {
        // With these params, raw calculation might be even.
        // We ensure it returns odd.
        let w = estimate_moving_average_window(30.0, 2.5, true);
        assert_eq!(w % 2, 1, "Window size must be odd");
    }

    #[test]
    fn test_estimate_window_zero_cutoff() {
        let w = estimate_moving_average_window(30.0, 0.0, false);
        assert_eq!(w, 1); // Fallback
    }

    #[test]
    fn test_moving_average_centering() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = moving_average(&data, 3);
        assert!((result[1] - 2.0).abs() < 1e-5);
        assert!((result[2] - 3.0).abs() < 1e-5);
        assert!((result[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_detrend_removes_drift() {
        let mut signal = Vec::new();
        for i in 0..100 {
            let t = i as f32 / 30.0;
            let sine = (2.0 * std::f32::consts::PI * 1.0 * t).sin();
            let drift = t * 2.0; 
            signal.push(sine + drift);
        }
        let clean = detrend(&signal, 30.0);
        let mean = clean.iter().sum::<f32>() / clean.len() as f32;
        assert!(mean.abs() < 0.1, "Mean was {}", mean);
    }

    #[test]
    fn test_standardize_calculates_zscore() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let res = standardize(&data);
        let mean = res.iter().sum::<f32>() / res.len() as f32;
        let variance = res.iter().map(|x| x * x).sum::<f32>() / res.len() as f32;
        assert!(mean.abs() < 1e-5);
        assert!((variance - 1.0).abs() < 1e-5);
    }
}