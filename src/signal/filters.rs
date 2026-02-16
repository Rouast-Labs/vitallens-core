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
/// # Arguments
/// * `signal` - Input data
/// * `window_size` - Number of frames to average (should be odd for perfect centering)
pub fn moving_average(signal: &[f32], window_size: usize) -> Vec<f32> {
    let len = signal.len();
    if len == 0 {
        return Vec::new();
    }
    if window_size >= len {
        let sum: f32 = signal.iter().sum();
        return vec![sum / len as f32; len];
    }

    let mut result = Vec::with_capacity(len);
    let half_window = window_size / 2;

    for i in 0..len {
        // Calculate the window bounds centered at i
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
// TODO: Test (from prpy)
pub fn estimate_moving_average_window(fs: f32, cutoff_hz: f32, force_odd: bool) -> usize {
    if fs <= 0.0 || cutoff_hz <= 0.0 {
        return 1;
    }

    let f = cutoff_hz / fs;
    if f.abs() < 1e-6 {
        return usize::MAX; 
    }

    let size = (0.196202 + f * f).sqrt() / f;
    let mut size_int = size as usize;
    
    if force_odd && size_int % 2 == 0 {
        // Conservative approach: 4 -> 3 rather than 5, to avoid over-smoothing
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
    fn test_moving_average_centering() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Window 3. 
        // Idx 0 (Window [1,2]) -> 1.5
        // Idx 1 (Window [1,2,3]) -> 2.0
        // Idx 2 (Window [2,3,4]) -> 3.0 (Center)
        // Idx 3 (Window [3,4,5]) -> 4.0
        // Idx 4 (Window [4,5]) -> 4.5
        let result = moving_average(&data, 3);
        
        assert!((result[1] - 2.0).abs() < 1e-5);
        assert!((result[2] - 3.0).abs() < 1e-5);
        assert!((result[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_window_estimation_oddness() {
        // Fs=30, Cutoff=2.5 => Raw ~5.4 => 5
        let w = estimate_moving_average_window(30.0, 2.5, true);
        assert_eq!(w, 5);

        // Fs=30, Cutoff=3.5 => Raw ~3.8 => 3
        // If it were even (e.g. 4), we'd expect it to drop to 3
        let w2 = estimate_moving_average_window(30.0, 3.5, true);
        assert!(w2 % 2 != 0);
    }

    #[test]
    fn test_detrend_removes_drift() {
        // Create a signal: Sin wave + Linear Ramp
        let mut signal = Vec::new();
        for i in 0..100 {
            let t = i as f32 / 30.0;
            let sine = (2.0 * std::f32::consts::PI * 1.0 * t).sin();
            let drift = t * 2.0; // Strong drift
            signal.push(sine + drift);
        }

        let clean = detrend(&signal, 30.0);

        // The mean of the clean signal should be close to 0
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