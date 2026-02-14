use rustfft::{FftPlanner, num_complex::Complex};

/// Estimates heart rate using the Periodogram method with zero-padding and parabolic interpolation.
/// 
/// * `signal`: Input signal (e.g. PPG)
/// * `fs`: Sampling frequency
/// * `min_bpm`, `max_bpm`: Search range
/// * `target_res_hz`: Target frequency resolution (e.g. 0.005 Hz) for zero-padding.
/// 
/// Returns (bpm, confidence/snr).
pub fn estimate_rate_periodogram(
    signal: &[f32],
    fs: f32,
    min_bpm: f32,
    max_bpm: f32,
    target_res_hz: f32,
) -> (f32, f32) {
    let len = signal.len();
    if len < 2 {
        return (0.0, 0.0);
    }

    // --- 1. Pre-processing ---
    
    // A. Detrend (Mean Subtraction)
    let mean: f32 = signal.iter().sum::<f32>() / len as f32;
    let detrended: Vec<f32> = signal.iter().map(|x| x - mean).collect();

    // B. Hanning Window to reduce spectral leakage
    let windowed: Vec<f32> = detrended.iter().enumerate().map(|(i, &x)| {
        let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (len - 1) as f32).cos());
        x * w
    }).collect();

    // --- 2. Zero Padding ---

    // Calculate N required for target resolution
    let min_fft_len = (fs / target_res_hz) as usize;
    let fft_len = min_fft_len.max(len).next_power_of_two();
    
    let mut buffer: Vec<Complex<f32>> = Vec::with_capacity(fft_len);
    for x in windowed {
        buffer.push(Complex::new(x, 0.0));
    }
    buffer.resize(fft_len, Complex::new(0.0, 0.0));

    // --- 3. FFT ---
    
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_len);
    fft.process(&mut buffer);

    // --- 4. Peak Finding ---
    
    let num_bins = fft_len / 2;
    let freq_per_bin = fs / fft_len as f32;
    
    let min_bin = (min_bpm / 60.0 / freq_per_bin).floor() as usize;
    let max_bin = (max_bpm / 60.0 / freq_per_bin).ceil() as usize;
    let max_bin = max_bin.min(num_bins - 1); // Clamp to Nyquist

    if min_bin >= max_bin {
        return (0.0, 0.0);
    }

    // Find bin with max energy
    let mut max_energy = -1.0;
    let mut peak_idx = 0;
    let mut total_band_energy = 0.0;

    for k in min_bin..=max_bin {
        let energy = buffer[k].norm_sqr(); // |X[k]|^2
        total_band_energy += energy;
        
        if energy > max_energy {
            max_energy = energy;
            peak_idx = k;
        }
    }

    if max_energy <= 0.0 || peak_idx == 0 || peak_idx >= num_bins - 1 {
        return (0.0, 0.0);
    }

    // --- 5. Refinement (Parabolic Interpolation) ---
    // Use Magnitude (sqrt of Energy) for cleaner interpolation shape
    let y_c = max_energy.sqrt();
    let y_l = buffer[peak_idx - 1].norm();
    let y_r = buffer[peak_idx + 1].norm();

    let denom = 2.0 * (y_l - 2.0 * y_c + y_r);
    let delta = if denom.abs() > 1e-9 {
        (y_l - y_r) / denom
    } else {
        0.0
    };
    
    // Clamp delta to valid range [-0.5, 0.5]
    let delta = delta.max(-0.5).min(0.5);
    
    let refined_bin = peak_idx as f32 + delta;
    let bpm = refined_bin * freq_per_bin * 60.0;

    // --- 6. Confidence (SNR) ---
    // Ratio of Peak Energy (Peak +/- 1 bin) to Total Search Band Energy
    let lobe_energy = max_energy 
                    + buffer[peak_idx - 1].norm_sqr() 
                    + buffer[peak_idx + 1].norm_sqr();
                   
    let confidence = if total_band_energy > 0.0 {
        (lobe_energy / total_band_energy).min(1.0)
    } else {
        0.0
    };

    (bpm, confidence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // --- Test Helpers ---

    struct SignalBuilder {
        fs: f32,
        data: Vec<f32>,
    }

    impl SignalBuilder {
        fn new(duration_sec: f32, fs: f32) -> Self {
            let samples = (duration_sec * fs) as usize;
            Self {
                fs,
                data: vec![0.0; samples],
            }
        }

        fn add_sine(mut self, freq_hz: f32, amplitude: f32) -> Self {
            for i in 0..self.data.len() {
                let t = i as f32 / self.fs;
                self.data[i] += amplitude * (2.0 * PI * freq_hz * t).sin();
            }
            self
        }

        fn add_noise(mut self, amplitude: f32) -> Self {
            // Simple pseudo-random generator to avoid 'rand' dependency in core
            let mut seed: u32 = 12345;
            for x in self.data.iter_mut() {
                seed = (1103515245 * seed + 12345) % 2147483648;
                let noise = (seed as f32 / 2147483648.0) * 2.0 - 1.0; // -1.0 to 1.0
                *x += noise * amplitude;
            }
            self
        }

        fn add_offset(mut self, offset: f32) -> Self {
            for x in self.data.iter_mut() {
                *x += offset;
            }
            self
        }

        fn get(self) -> Vec<f32> {
            self.data
        }
    }

    // --- 1. Accuracy Tests ---

    #[test]
    fn test_pure_sine_accuracy() {
        let fs = 30.0;
        let target_bpm = 72.0;
        let freq = target_bpm / 60.0;

        let signal = SignalBuilder::new(10.0, fs)
            .add_sine(freq, 1.0)
            .get();

        let (bpm, conf) = estimate_rate_periodogram(&signal, fs, 40.0, 200.0, 0.005);

        assert!((bpm - target_bpm).abs() < 0.1, "Expected {}, got {}", target_bpm, bpm);
        assert!(conf > 0.9, "Confidence should be high for pure sine");
    }

    #[test]
    fn test_high_heart_rate() {
        let fs = 30.0;
        let target_bpm = 180.0; // 3 Hz
        
        let signal = SignalBuilder::new(5.0, fs)
            .add_sine(target_bpm / 60.0, 1.0)
            .get();

        let (bpm, _) = estimate_rate_periodogram(&signal, fs, 40.0, 220.0, 0.005);
        assert!((bpm - target_bpm).abs() < 0.5);
    }

    #[test]
    fn test_low_heart_rate() {
        let fs = 30.0;
        let target_bpm = 45.0; // 0.75 Hz
        
        let signal = SignalBuilder::new(10.0, fs)
            .add_sine(target_bpm / 60.0, 1.0)
            .get();

        let (bpm, _) = estimate_rate_periodogram(&signal, fs, 40.0, 220.0, 0.005);
        assert!((bpm - target_bpm).abs() < 0.5);
    }

    // --- 2. Resilience Tests ---

    #[test]
    fn test_dc_offset_invariance() {
        // FFT should ignore constant DC offset (0 Hz bin)
        let fs = 30.0;
        let target_bpm = 60.0;
        
        let signal = SignalBuilder::new(5.0, fs)
            .add_sine(1.0, 0.5) // 0.5 amplitude
            .add_offset(1000.0) // Massive offset
            .get();

        let (bpm, conf) = estimate_rate_periodogram(&signal, fs, 40.0, 200.0, 0.005);

        assert!((bpm - target_bpm).abs() < 0.2);
        assert!(conf > 0.8, "Offset shouldn't ruin confidence");
    }

    #[test]
    fn test_noisy_signal_still_detects_rate() {
        let fs = 30.0;
        let target_bpm = 80.0;
        
        // Signal = 1.0, Noise = 0.5 (SNR 2:1)
        let signal = SignalBuilder::new(8.0, fs)
            .add_sine(target_bpm / 60.0, 1.0)
            .add_noise(0.5) 
            .get();

        let (bpm, conf) = estimate_rate_periodogram(&signal, fs, 40.0, 200.0, 0.005);

        assert!((bpm - target_bpm).abs() < 2.0, "Should still find peak in noise");
        assert!(conf < 0.9, "Confidence should drop with noise");
        assert!(conf > 0.3, "Confidence should not be zero");
    }

    #[test]
    fn test_pure_noise_returns_low_conf() {
        let fs = 30.0;
        let signal = SignalBuilder::new(5.0, fs)
            .add_noise(1.0)
            .get();

        let (_, conf) = estimate_rate_periodogram(&signal, fs, 40.0, 200.0, 0.005);
        
        // It will pick *some* random peak, but confidence should be trash
        assert!(conf < 0.3, "Pure noise should have low confidence");
    }

    // --- 3. Edge Case Tests ---

    #[test]
    fn test_short_signal_padding() {
        // 2 seconds is usually too short for good resolution, 
        // but zero-padding (target_res_hz) should help it approximate.
        let fs = 30.0;
        let target_bpm = 60.0;
        
        let signal = SignalBuilder::new(2.0, fs)
            .add_sine(1.0, 1.0)
            .get();

        let (bpm, _) = estimate_rate_periodogram(&signal, fs, 40.0, 200.0, 0.001);

        // Allow slightly looser tolerance for short signals
        assert!((bpm - target_bpm).abs() < 3.0); 
    }

    #[test]
    fn test_flat_line() {
        let signal = vec![10.0; 100];
        let (bpm, conf) = estimate_rate_periodogram(&signal, 30.0, 40.0, 200.0, 0.005);
        
        assert_eq!(bpm, 0.0);
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn test_empty_input() {
        let signal: Vec<f32> = vec![];
        let (bpm, conf) = estimate_rate_periodogram(&signal, 30.0, 40.0, 200.0, 0.005);
        
        assert_eq!(bpm, 0.0);
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn test_nan_handling() {
        // Ensure it doesn't panic
        let mut signal = vec![0.0; 100];
        signal[50] = f32::NAN; 
        
        // This might return garbage or 0.0, but it MUST NOT panic
        let result = std::panic::catch_unwind(|| {
            estimate_rate_periodogram(&signal, 30.0, 40.0, 200.0, 0.005)
        });
        
        assert!(result.is_ok(), "Function panicked on NaN input");
    }
}