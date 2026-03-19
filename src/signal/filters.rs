use std::f32;

const DEFAULT_HP_CUTOFF: f32 = 0.5;
const DEFAULT_LP_CUTOFF: f32 = 3.0;

/// Applies a specified post-processing operation to a signal.
///
/// # Arguments
/// * `signal` - The input time-domain signal.
/// * `op` - The post-processing operation to apply from the registry configuration.
/// * `fs` - The sampling frequency in Hz.
/// * `min_freq` - Optional high-pass cutoff frequency in Hz (used for detrending).
/// * `max_freq` - Optional low-pass cutoff frequency in Hz (used for moving average).
///
/// # Returns
/// A new `Vec<f32>` containing the processed signal.
pub fn apply_processing(
    signal: &[f32], 
    op: crate::registry::PostProcessOp,
    fs: f32,
    min_freq: Option<f32>,
    max_freq: Option<f32>,
) -> Vec<f32> {
    match op {
        crate::registry::PostProcessOp::None => signal.to_vec(),
        crate::registry::PostProcessOp::Detrend => {
            let cutoff = min_freq.unwrap_or(DEFAULT_HP_CUTOFF);
            detrend(signal, fs, cutoff)
        },
        crate::registry::PostProcessOp::MovingAverage => {
            let cutoff = max_freq.unwrap_or(DEFAULT_LP_CUTOFF);
            let window = moving_average_window_for_cutoff(fs, cutoff, true);
            moving_average(signal, window)
        },
        crate::registry::PostProcessOp::Standardize => standardize(signal),
        crate::registry::PostProcessOp::MovingAverageStandardize => {
            let cutoff = max_freq.unwrap_or(DEFAULT_LP_CUTOFF);
            let window = moving_average_window_for_cutoff(fs, cutoff, true);
            let smoothed = moving_average(signal, window);
            standardize(&smoothed)
        },
        crate::registry::PostProcessOp::DetrendMovingAverageStandardize => {
            let hp_cutoff = min_freq.unwrap_or(DEFAULT_HP_CUTOFF);
            let detrended = detrend(signal, fs, hp_cutoff);
            let lp_cutoff = max_freq.unwrap_or(DEFAULT_LP_CUTOFF);
            let window = moving_average_window_for_cutoff(fs, lp_cutoff, true);
            let smoothed = moving_average(&detrended, window);
            standardize(&smoothed)
        },
    }
}

/// Calculates the centered moving average of a signal.
///
/// # Arguments
/// * `signal` - Input data.
/// * `window_size` - Number of frames to average (should be odd for perfect centering).
///
/// # Returns
/// A new `Vec<f32>` containing the smoothed signal.
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

/// Estimates the required moving average window size to achieve a specific low-pass cutoff frequency.
///
/// # Arguments
/// * `fs` - Sampling frequency in Hz.
/// * `cutoff_hz` - Desired cutoff frequency in Hz.
/// * `force_odd` - If true, ensures the result is an odd number.
///
/// # Returns
/// The estimated window size in number of frames.
pub fn moving_average_window_for_cutoff(fs: f32, cutoff_hz: f32, force_odd: bool) -> usize {
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
        size_int = size_int.saturating_sub(1);
    }
    
    size_int.max(1)
}

/// Calculates the regularization parameter lambda for a desired high-pass cutoff frequency.
///
/// # Arguments
/// * `fs` - The sampling frequency in Hz.
/// * `cutoff` - The desired high-pass cutoff frequency in Hz.
///
/// # Returns
/// The lambda parameter `f32` to be used in the Tarvainen-Valtonen detrending algorithm.
pub fn detrend_lambda_for_cutoff(fs: f32, cutoff: f32) -> f32 {
    if cutoff <= 1e-5 {
        return 0.0;
    }
    0.075 * (fs / cutoff).powi(2)
}

/// Removes the low-frequency trend from a signal using the Tarvainen-Valtonen (Smoothness Priors) method.
/// 
/// # Arguments
/// * `signal` - Input data.
/// * `fs` - Sampling rate in Hz.
/// * `cutoff` - High-pass cutoff frequency in Hz (frequencies below this are removed).
///
/// # Returns
/// A new `Vec<f32>` containing the detrended signal.
pub fn detrend(signal: &[f32], fs: f32, cutoff: f32) -> Vec<f32> {
    let n = signal.len();
    if n < 3 {
        return vec![0.0; n];
    }

    let lambda = detrend_lambda_for_cutoff(fs, cutoff);
    detrend_with_lambda(signal, lambda)
}

/// Performs detrending using the Tarvainen-Valtonen method with a directly provided lambda.
///
/// # Arguments
/// * `signal` - The input time-domain signal.
/// * `lambda` - The regularization parameter controlling the smoothness of the trend.
///
/// # Returns
/// A new `Vec<f32>` containing the detrended signal.
pub fn detrend_with_lambda(signal: &[f32], lambda: f32) -> Vec<f32> {
    let n = signal.len();
    if n < 3 {
        return vec![0.0; n];
    }
    
    let lambda_sq = lambda * lambda;

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

/// Solves a linear system Ax = y using Cholesky decomposition for a banded symmetric matrix.
///
/// # Arguments
/// * `d0` - The main diagonal of the matrix.
/// * `d1` - The first super/sub-diagonal.
/// * `d2` - The second super/sub-diagonal.
/// * `y` - The right-hand side vector.
///
/// # Returns
/// The solution vector `x` as a `Vec<f32>`.
fn solve_cholesky_banded(d0: &[f32], d1: &[f32], d2: &[f32], y: &[f32]) -> Vec<f32> {
    let n = d0.len();
    if n == 0 { return Vec::new(); }
    
    let mut l0 = vec![0.0; n];
    let mut l1 = vec![0.0; n - 1]; 
    let mut l2 = vec![0.0; n - 2]; 

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

    let mut z = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        if i >= 1 { sum += l1[i-1] * z[i-1]; }
        if i >= 2 { sum += l2[i-2] * z[i-2]; }
        z[i] = (y[i] - sum) / l0[i];
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        if i + 1 < n { sum += l1[i] * x[i+1]; } 
        if i + 2 < n { sum += l2[i] * x[i+2]; } 
        x[i] = (z[i] - sum) / l0[i];
    }

    x
}

/// Normalizes a signal using Z-Score normalization (Zero Mean, Unit Variance).
///
/// # Arguments
/// * `signal` - The input time-domain signal.
///
/// # Returns
/// A new `Vec<f32>` containing the standardized signal.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::PostProcessOp;

    #[test]
    fn test_ma_impulse_response() {
        let data = vec![0.0, 0.0, 10.0, 0.0, 0.0];
        let result = moving_average(&data, 3);
        
        assert_eq!(result.len(), 5);
        assert!((result[2] - 3.333).abs() < 0.01); 
        assert!(result[2] >= result[1] && result[2] >= result[3]);
    }

    #[test]
    fn test_ma_step_response_smoothing() {
        let data = vec![0.0, 0.0, 1.0, 1.0, 1.0];
        let result = moving_average(&data, 3);
        
        assert!(result[1] > 0.0);        
        assert!(result[2] > result[1]);  
        assert!((result[4] - 1.0).abs() < 0.01);  
    }

    #[test]
    fn test_ma_boundary_handling() {
        let data = vec![10.0, 20.0, 30.0];
        let result = moving_average(&data, 3);
        
        assert_eq!(result[0], 15.0);
        assert_eq!(result[1], 20.0);
        assert_eq!(result[2], 25.0);
    }

    #[test]
    fn test_ma_window_larger_than_signal() {
        let data = vec![1.0, 2.0, 3.0];
        let result = moving_average(&data, 10);
        
        assert_eq!(result.len(), 3);
        for x in result {
            assert_eq!(x, 2.0);  
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
        let w_high_cutoff = moving_average_window_for_cutoff(fs, 5.0, true);
        let w_low_cutoff = moving_average_window_for_cutoff(fs, 1.0, true);
        
        assert!(w_low_cutoff > w_high_cutoff, 
            "Lower cutoff (1Hz) should require larger window than high cutoff (5Hz)");
    }

    #[test]
    fn test_estimate_window_fs_relationship() {
        let cutoff = 2.0;
        let w_high_fs = moving_average_window_for_cutoff(60.0, cutoff, true);
        let w_low_fs = moving_average_window_for_cutoff(30.0, cutoff, true);

        assert!(w_high_fs > w_low_fs,
            "Higher sampling rate should require more samples for same cutoff");
    }

    #[test]
    fn test_estimate_window_force_odd() {
        let w = moving_average_window_for_cutoff(30.0, 2.5, true);
        assert_eq!(w % 2, 1, "Window size must be odd");
    }

    #[test]
    fn test_estimate_window_zero_cutoff() {
        let w = moving_average_window_for_cutoff(30.0, 0.0, false);
        assert_eq!(w, 1);  
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
        let data = vec![10.0, 20.0, 30.0];
        let res = standardize(&data);
        
        let expected_std = (200.0f32 / 3.0).sqrt();
        assert!((res[0] - (-10.0 / expected_std)).abs() < 1e-5);
        assert!((res[1] - 0.0).abs() < 1e-5);
        assert!((res[2] - (10.0 / expected_std)).abs() < 1e-5);
    }

    #[test]
    fn test_standardize_flatline() {
        let data = vec![5.0, 5.0, 5.0];
        let res = standardize(&data);
        assert_eq!(res, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_standardize_single_element() {
        let data = vec![42.0];
        let res = standardize(&data);
        assert_eq!(res, vec![0.0]);
    }

    #[test]
    fn test_standardize_mixed_nans() {
        let data = vec![10.0, f32::NAN, 30.0];
        let res = standardize(&data);
        
        assert!((res[0] - (-1.0)).abs() < 1e-5);
        assert_eq!(res[1], 0.0);  
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
        let data = vec![1.0, 1.0000001, 1.0];
        let res = standardize(&data);
        assert_eq!(res, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_pipeline_detrend_ma_std() {
        let fs = 30.0;
        let mut signal = Vec::new();
        for i in 0..100 {
            let t = i as f32 / fs;
            let val = 2.0*t + (2.0 * std::f32::consts::PI * 1.0 * t).sin() + 0.5 * (2.0 * std::f32::consts::PI * 10.0 * t).sin();
            signal.push(val);
        }

        let processed = apply_processing(
            &signal, 
            PostProcessOp::DetrendMovingAverageStandardize, 
            fs, 
            Some(0.5),  
            Some(2.0)   
        );

        let mean = processed.iter().sum::<f32>() / processed.len() as f32;
        let variance = processed.iter().map(|x| x*x).sum::<f32>() / processed.len() as f32;
        assert!(mean.abs() < 1e-5, "Mean not zero");
        assert!((variance - 1.0).abs() < 1e-5, "Variance not one");

        let crossings = processed.windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count();
        
        assert!(crossings >= 4 && crossings <= 10, "Expected ~6 crossings for 1Hz, got {}", crossings);
    }
}