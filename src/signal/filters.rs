use std::f32;

pub fn apply_processing(
    signal: &[f32], 
    op: crate::registry::PostProcessOp,
    fs: f32,
    min_freq: Option<f32>,
    _max_freq: Option<f32>,
) -> Vec<f32> {
    match op {
        crate::registry::PostProcessOp::Detrend => {
            let cutoff = min_freq.unwrap_or(0.0);
            detrend(signal, fs, cutoff)
        },
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

/// Calculates the regularization parameter lambda for a desired cutoff frequency.
pub fn detrend_lambda_for_cutoff(fs: f32, cutoff: f32) -> f32 {
    if cutoff <= 1e-5 {
        return 0.0;
    }
    0.075 * (fs / cutoff).powi(2)
}

/// Detrending using Tarvainen-Valtonen (Smoothness Priors) method.
/// 
/// # Arguments
/// * `signal` - Input data
/// * `fs` - Sampling rate
/// * `cutoff` - High-pass cutoff frequency in Hz (frequencies below this are removed)
pub fn detrend(signal: &[f32], fs: f32, cutoff: f32) -> Vec<f32> {
    let n = signal.len();
    if n < 3 {
        return vec![0.0; n];
    }

    let lambda = detrend_lambda_for_cutoff(fs, cutoff);
    detrend_with_lambda(signal, lambda)
}

/// Raw Detrending where Lambda is provided directly.
pub fn detrend_with_lambda(signal: &[f32], lambda: f32) -> Vec<f32> {
    let n = signal.len();
    if n < 3 {
        return vec![0.0; n];
    }
    
    let lambda_sq = lambda * lambda;

    // A is symmetric pentadiagonal.     
    let mut d0 = vec![0.0; n];
    let mut d1 = vec![0.0; n - 1];
    let mut d2 = vec![0.0; n - 2];

    for i in 0..n {
        if i == 0 {
            d0[i] = 1.0 + lambda_sq * 1.0;
            if n > 1 { d1[i] = lambda_sq * -2.0; }
            if n > 2 { d2[i] = lambda_sq * 1.0; }
        } else if i == 1 {
            d0[i] = 1.0 + lambda_sq * 5.0;
            if n > 2 { d1[i] = lambda_sq * -4.0; }
            if n > 3 { d2[i] = lambda_sq * 1.0; }
        } else if i < n - 2 {
            d0[i] = 1.0 + lambda_sq * 6.0;
            d1[i] = lambda_sq * -4.0;
            d2[i] = lambda_sq * 1.0;
        } else if i == n - 2 {
            d0[i] = 1.0 + lambda_sq * 5.0;
            d1[i] = lambda_sq * -2.0;
        } else if i == n - 1 {
            d0[i] = 1.0 + lambda_sq * 1.0;
        }
    }

    let trend = solve_cholesky_banded(&d0, &d1, &d2, signal);

    signal.iter().zip(trend.iter()).map(|(s, t)| s - t).collect()
}

fn solve_cholesky_banded(d0: &[f32], d1: &[f32], d2: &[f32], y: &[f32]) -> Vec<f32> {
    let n = d0.len();
    if n == 0 { return Vec::new(); }
    
    let mut l0 = vec![0.0; n];
    let mut l1 = vec![0.0; n - 1]; 
    let mut l2 = vec![0.0; n - 2]; 

    // 1. Cholesky Decomposition
    for i in 0..n {
        if i >= 2 {
            let val = d2[i-2]; 
            l2[i-2] = val / l0[i-2];
        }
        if i >= 1 {
            let mut sum = 0.0;
            if i >= 2 {
                sum += l2[i-2] * l1[i-2];
            }
            let val = d1[i-1];
            l1[i-1] = (val - sum) / l0[i-1];
        }
        let mut sum_sq = 0.0;
        if i >= 1 { sum_sq += l1[i-1] * l1[i-1]; }
        if i >= 2 { sum_sq += l2[i-2] * l2[i-2]; }
        
        let val = d0[i] - sum_sq;
        if val <= 0.0 {
            return vec![0.0; n]; 
        }
        l0[i] = val.sqrt();
    }

    // 2. Forward Substitution
    let mut z = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        if i >= 1 { sum += l1[i-1] * z[i-1]; }
        if i >= 2 { sum += l2[i-2] * z[i-2]; }
        z[i] = (y[i] - sum) / l0[i];
    }

    // 3. Backward Substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        if i + 1 < n { sum += l1[i] * x[i+1]; } 
        if i + 2 < n { sum += l2[i] * x[i+2]; } 
        x[i] = (z[i] - sum) / l0[i];
    }

    x
}

/// Z-Score normalization (Zero Mean, Unit Variance).
pub fn standardize(signal: &[f32]) -> Vec<f32> {
    if signal.is_empty() { return Vec::new(); }

    let (sum, count) = signal.iter().fold((0.0, 0), |(s, c), &x| {
        if x.is_finite() { (s + x, c + 1) } else { (s, c) }
    });

    if count == 0 { return vec![0.0; signal.len()]; }
    let mean = sum / count as f32;

    let var_sum = signal.iter().fold(0.0, |s, &x| {
        if x.is_finite() { s + (x - mean).powi(2) } else { s }
    });
    
    let std_dev = (var_sum / count as f32).sqrt();

    if std_dev.abs() < 1e-6 {
        return vec![0.0; signal.len()];
    }

    signal.iter()
        .map(|&v| if v.is_finite() { (v - mean) / std_dev } else { 0.0 })
        .collect()
}

// --- UNIT TESTS ---
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ma_impulse_response() {
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
    fn test_detrend_axis_minus_1() {
        let z = vec![0.0, 0.4, 1.2, 2.0, 0.2];
        let lambda = 3.0;
        let result = detrend_with_lambda(&z, lambda);
        
        let expected = vec![-0.29326743, -0.18156859, 0.36271552, 0.99234445, -0.88022395];
        
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "Expected {}, got {}", b, a);
        }
    }

    #[test]
    fn test_detrend_axis_minus_1_row_2() {
        let z = vec![0.0, 0.1, 0.5, 0.3, -0.1];
        let lambda = 3.0;
        
        let result = detrend_with_lambda(&z, lambda);

        let expected = vec![-0.13389946, -0.06970109, 0.309375, 0.12595109, -0.23172554];
        
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "Expected {}, got {}", b, a);
        }
    }

    #[test]
    fn test_detrend_preserves_length() {
        let signal = vec![1.0; 10];
        let res = detrend_with_lambda(&signal, 30.0);
        assert_eq!(res.len(), 10);
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
        let clean = detrend(&signal, 30.0, 0.5);
        let mean = clean.iter().sum::<f32>() / clean.len() as f32;
        assert!(mean.abs() < 0.1, "Mean was {}", mean);
    }

    #[test]
    fn test_standardize_normal() {
        // [10, 20, 30] -> Mean 20, Var sum = 100+0+100=200, Var=200/3=66.66, Std=8.165
        let data = vec![10.0, 20.0, 30.0];
        let res = standardize(&data);
        
        let expected_std = (200.0f32 / 3.0).sqrt();
        assert!((res[0] - (-10.0 / expected_std)).abs() < 1e-5);
        assert!((res[1] - 0.0).abs() < 1e-5);
        assert!((res[2] - (10.0 / expected_std)).abs() < 1e-5);
    }

    #[test]
    fn test_standardize_flatline() {
        // Zero variance should return all zeros, not NaNs/Infs
        let data = vec![5.0, 5.0, 5.0];
        let res = standardize(&data);
        assert_eq!(res, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_standardize_single_element() {
        // Single element has undefined (or zero) variance. Should return 0.
        let data = vec![42.0];
        let res = standardize(&data);
        assert_eq!(res, vec![0.0]);
    }

    #[test]
    fn test_standardize_mixed_nans() {
        // [10, NaN, 30] -> Mean/Std computed on [10, 30] -> Mean 20, Std 10.
        // Result: [(10-20)/10, 0.0, (30-20)/10] -> [-1.0, 0.0, 1.0]
        let data = vec![10.0, f32::NAN, 30.0];
        let res = standardize(&data);
        
        assert!((res[0] - (-1.0)).abs() < 1e-5);
        assert_eq!(res[1], 0.0); // NaNs are replaced by mean (0.0)
        assert!((res[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_standardize_all_nans() {
        let data = vec![f32::NAN, f32::NAN];
        let res = standardize(&data);
        assert_eq!(res, vec![0.0, 0.0]);
    }

    #[test]
    fn test_standardize_high_offset() {
        // Z-score should be invariant to DC offset
        // [1, 2, 3] vs [1001, 1002, 1003] -> Output should be identical
        let normal = vec![1.0, 2.0, 3.0];
        let offset = vec![1001.0, 1002.0, 1003.0];
        
        let res_norm = standardize(&normal);
        let res_off = standardize(&offset);
        
        for (a, b) in res_norm.iter().zip(res_off.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
    
    #[test]
    fn test_standardize_tiny_variance() {
        // Variance below 1e-6 threshold should clamp to zero to avoid explosion
        let data = vec![1.0, 1.0000001, 1.0];
        let res = standardize(&data);
        assert_eq!(res, vec![0.0, 0.0, 0.0]);
    }
}