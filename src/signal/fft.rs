use rustfft::{FftPlanner, num_complex::Complex};

/// Represents the Power Spectral Density of a signal.
pub struct PsdResult {
    pub frequencies: Vec<f32>,
    pub power: Vec<f32>,
}

/// Core math engine: Computes the Hanning-windowed periodogram with zero-padding.
pub fn compute_periodogram(signal: &[f32], fs: f32, target_res_hz: f32) -> PsdResult {
    let len = signal.len();
    if len < 2 {
        return PsdResult { frequencies: vec![], power: vec![] };
    }

    // 1. Detrend (Mean Subtraction)
    let mean: f32 = signal.iter().sum::<f32>() / len as f32;
    let detrended: Vec<f32> = signal.iter().map(|x| x - mean).collect();

    // 2. Hanning Window
    let windowed: Vec<f32> = detrended.iter().enumerate().map(|(i, &x)| {
        let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (len - 1) as f32).cos());
        x * w
    }).collect();

    // 3. Zero Padding
    let min_fft_len = (fs / target_res_hz) as usize;
    let fft_len = min_fft_len.max(len).next_power_of_two();
    
    let mut buffer: Vec<Complex<f32>> = Vec::with_capacity(fft_len);
    for x in windowed {
        buffer.push(Complex::new(x, 0.0));
    }
    buffer.resize(fft_len, Complex::new(0.0, 0.0));

    // 4. FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_len);
    fft.process(&mut buffer);

    // 5. Calculate PSD (Power)
    let num_bins = fft_len / 2;
    let freq_per_bin = fs / fft_len as f32;
    
    let mut frequencies = Vec::with_capacity(num_bins);
    let mut power = Vec::with_capacity(num_bins);

    for k in 0..num_bins {
        frequencies.push(k as f32 * freq_per_bin);
        power.push(buffer[k].norm_sqr());
    }

    PsdResult { frequencies, power }
}

/// Estimates heart rate by finding the dominant peak in the periodogram.
pub fn estimate_rate_periodogram(
    signal: &[f32],
    fs: f32,
    min_bpm: f32,
    max_bpm: f32,
    target_res_hz: f32,
) -> (f32, f32) {
    let psd = compute_periodogram(signal, fs, target_res_hz);
    if psd.frequencies.is_empty() {
        return (0.0, 0.0);
    }

    let freq_per_bin = psd.frequencies[1] - psd.frequencies[0];
    let min_bin = (min_bpm / 60.0 / freq_per_bin).floor() as usize;
    let max_bin = (max_bpm / 60.0 / freq_per_bin).ceil() as usize;
    let max_bin = max_bin.min(psd.frequencies.len() - 1);

    if min_bin >= max_bin { return (0.0, 0.0); }

    // Find peak
    let mut max_energy = -1.0;
    let mut peak_idx = 0;
    let mut total_band_energy = 0.0;

    for k in min_bin..=max_bin {
        let energy = psd.power[k];
        total_band_energy += energy;
        if energy > max_energy {
            max_energy = energy;
            peak_idx = k;
        }
    }

    if max_energy <= 0.0 || peak_idx == 0 || peak_idx >= psd.frequencies.len() - 1 {
        return (0.0, 0.0);
    }

    // Refinement (Parabolic Interpolation on Magnitudes)
    let y_c = max_energy.sqrt();
    let y_l = psd.power[peak_idx - 1].sqrt();
    let y_r = psd.power[peak_idx + 1].sqrt();

    let denom = 2.0 * (y_l - 2.0 * y_c + y_r);
    let delta = if denom.abs() > 1e-9 { (y_l - y_r) / denom } else { 0.0 };
    let delta = delta.max(-0.5).min(0.5);
    
    let bpm = (peak_idx as f32 + delta) * freq_per_bin * 60.0;

    // Confidence Calculation
    let search_radius_bpm = 5.0; 
    let search_radius_bins = (search_radius_bpm / 60.0 / freq_per_bin).ceil() as usize;
    let lobe_start = peak_idx.saturating_sub(search_radius_bins).max(min_bin);
    let lobe_end = (peak_idx + search_radius_bins).min(max_bin);
    
    let lobe_energy: f32 = psd.power[lobe_start..=lobe_end].iter().sum();
    let confidence = if total_band_energy > 0.0 { (lobe_energy / total_band_energy).min(1.0) } else { 0.0 };

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
            // Simple pseudo-random generator
            let mut seed: u64 = 12345;
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
        assert!(conf > 0.8, "Confidence should be high for pure sine. Got {}", conf);
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
        // [FIX] Increased duration to 10.0s to match baseline. 
        // Short signals (5s) have wide spectral peaks that lower confidence.
        let fs = 30.0;
        let target_bpm = 60.0;
        
        let signal = SignalBuilder::new(10.0, fs)
            .add_sine(1.0, 0.5) 
            .add_offset(1000.0) // Massive offset
            .get();

        let (bpm, conf) = estimate_rate_periodogram(&signal, fs, 40.0, 200.0, 0.005);

        assert!((bpm - target_bpm).abs() < 0.2);
        assert!(conf > 0.8, "Offset shouldn't ruin confidence. Got {}", conf);
    }

    #[test]
    fn test_noisy_signal_still_detects_rate() {
        let fs = 30.0;
        let target_bpm = 80.0;
        
        // Signal = 1.0, Noise = 0.5
        let signal = SignalBuilder::new(8.0, fs)
            .add_sine(target_bpm / 60.0, 1.0)
            .add_noise(0.5) 
            .get();

        let (bpm, conf) = estimate_rate_periodogram(&signal, fs, 40.0, 200.0, 0.005);

        assert!((bpm - target_bpm).abs() < 2.0, "Should still find peak in noise. Got {}", bpm);
        assert!(conf < 0.9, "Confidence should drop with noise");
        assert!(conf > 0.3, "Confidence should not be zero. Got {}", conf);
    }

    #[test]
    fn test_pure_noise_returns_low_conf() {
        let fs = 30.0;
        let signal = SignalBuilder::new(5.0, fs)
            .add_noise(1.0)
            .get();

        let (_, conf) = estimate_rate_periodogram(&signal, fs, 40.0, 200.0, 0.005);
        
        assert!(conf < 0.4, "Pure noise should have low confidence. Got {}", conf);
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
        
        let result = std::panic::catch_unwind(|| {
            estimate_rate_periodogram(&signal, 30.0, 40.0, 200.0, 0.005)
        });
        
        assert!(result.is_ok(), "Function panicked on NaN input");
    }

    // --- 4. compute_periodogram Tests ---

    #[test]
    fn test_psd_frequency_resolution() {
        let fs = 30.0;
        let target_res = 0.1; // 0.1 Hz per bin
        let signal = vec![0.0; 100];
        
        let psd = compute_periodogram(&signal, fs, target_res);
        
        // Freq per bin = fs / fft_len
        // next_power_of_two of (30 / 0.1 = 300) is 512.
        // Freq per bin = 30 / 512 = 0.0585... which is < 0.1.
        let actual_res = psd.frequencies[1] - psd.frequencies[0];
        assert!(actual_res <= target_res);
        assert_eq!(psd.frequencies.len(), psd.power.len());
    }

    #[test]
    fn test_psd_peak_location() {
        let fs = 20.0;
        let target_hz = 2.0; // 2Hz sine wave
        let signal = SignalBuilder::new(5.0, fs)
            .add_sine(target_hz, 1.0)
            .get();
        
        let psd = compute_periodogram(&signal, fs, 0.01);
        
        // Find index of max power
        let (max_idx, _) = psd.power.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
            
        let peak_freq = psd.frequencies[max_idx];
        
        // Should be very close to 2.0Hz
        assert!((peak_freq - target_hz).abs() < 0.05);
    }

    #[test]
    fn test_psd_energy_scaling() {
        let fs = 10.0;
        let sig1 = SignalBuilder::new(5.0, fs).add_sine(1.0, 1.0).get();
        let sig2 = SignalBuilder::new(5.0, fs).add_sine(1.0, 2.0).get(); // Double amp
        
        let psd1 = compute_periodogram(&sig1, fs, 0.1);
        let psd2 = compute_periodogram(&sig2, fs, 0.1);
        
        let max_p1 = *psd1.power.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_p2 = *psd2.power.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        // Power scales with square of amplitude. 2^2 = 4.
        let ratio = max_p2 / max_p1;
        assert!((ratio - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_psd_dc_removal() {
        let fs = 10.0;
        let signal = SignalBuilder::new(5.0, fs)
            .add_sine(1.0, 1.0)
            .add_offset(100.0) // Large DC offset
            .get();
            
        let psd = compute_periodogram(&signal, fs, 0.1);
        
        // psd.frequencies[0] is the DC bin (0Hz)
        // It should be significantly smaller than the 1Hz signal bin
        let dc_power = psd.power[0];
        let sig_power = *psd.power.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        assert!(dc_power < sig_power * 0.01); 
    }
}