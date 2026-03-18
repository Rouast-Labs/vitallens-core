use rustfft::{FftPlanner, num_complex::Complex};

/// Reusable scratchpad for FFT calculations to minimize memory allocations across continuous frames.
pub struct FftScratch {
    pub windowed: Vec<f32>,
    pub complex_buffer: Vec<Complex<f32>>,
    pub frequencies: Vec<f32>,
    pub power: Vec<f32>,
    pub planner: FftPlanner<f32>,
}

impl Default for FftScratch {
    fn default() -> Self {
        Self {
            windowed: Vec::new(),
            complex_buffer: Vec::new(),
            frequencies: Vec::new(),
            power: Vec::new(),
            planner: FftPlanner::new(),
        }
    }
}

impl Clone for FftScratch {
    fn clone(&self) -> Self {
        Self {
            windowed: self.windowed.clone(),
            complex_buffer: self.complex_buffer.clone(),
            frequencies: self.frequencies.clone(),
            power: self.power.clone(),
            planner: FftPlanner::new(),
        }
    }
}

impl std::fmt::Debug for FftScratch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftScratch")
            .field("windowed_len", &self.windowed.len())
            .field("complex_buffer_len", &self.complex_buffer.len())
            .field("frequencies_len", &self.frequencies.len())
            .field("power_len", &self.power.len())
            .finish_non_exhaustive()
    }
}

impl FftScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Holds the resulting power spectral density from an FFT operation.
pub struct PsdResult {
    pub frequencies: Vec<f32>,
    pub power: Vec<f32>,
}

/// Computes the Power Spectral Density (PSD) of a signal using a periodogram.
///
/// # Arguments
/// * `signal` - The input time-domain signal.
/// * `fs` - The sampling frequency in Hz.
/// * `target_res_hz` - The target frequency resolution in Hz (used to determine zero-padding).
/// * `scratch` - A mutable reference to an `FftScratch` buffer to avoid reallocations.
/// * `apply_window` - If true, applies a Hann window to the signal before FFT to reduce spectral leakage.
pub fn compute_periodogram(
    signal: &[f32], 
    fs: f32, 
    target_res_hz: f32,
    scratch: &mut FftScratch,
    apply_window: bool
) {
    let len = signal.len();
    if len < 2 {
        scratch.frequencies.clear();
        scratch.power.clear();
        return;
    }

    let mean: f32 = signal.iter().sum::<f32>() / len as f32;
    
    scratch.windowed.clear();
    
    if apply_window {
        for (i, &x) in signal.iter().enumerate() {
            let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (len - 1) as f32).cos());
            scratch.windowed.push((x - mean) * w);
        }
    } else {
        for &x in signal.iter() {
            scratch.windowed.push(x - mean);
        }
    }

    let min_fft_len = (fs / target_res_hz) as usize;
    let fft_len = min_fft_len.max(len).next_power_of_two();
    
    scratch.complex_buffer.clear();
    for &x in &scratch.windowed {
        scratch.complex_buffer.push(Complex::new(x, 0.0));
    }
    scratch.complex_buffer.resize(fft_len, Complex::new(0.0, 0.0));

    let fft = scratch.planner.plan_fft_forward(fft_len);
    fft.process(&mut scratch.complex_buffer);

    let num_bins = fft_len / 2;
    let freq_per_bin = fs / fft_len as f32;
    
    scratch.frequencies.clear();
    scratch.power.clear();

    for k in 0..num_bins {
        scratch.frequencies.push(k as f32 * freq_per_bin);
        scratch.power.push(scratch.complex_buffer[k].norm_sqr());
    }
}

/// Estimates the dominant rate (in BPM) within a specific range using a periodogram.
/// Refines the peak location using quadratic interpolation for sub-bin precision.
///
/// # Arguments
/// * `signal` - The input time-domain signal.
/// * `fs` - The sampling frequency in Hz.
/// * `min_bpm` - The minimum allowed rate in Beats Per Minute.
/// * `max_bpm` - The maximum allowed rate in Beats Per Minute.
/// * `target_res_hz` - The target frequency resolution for the FFT.
/// * `scratch_opt` - An optional mutable reference to an `FftScratch` buffer.
///
/// # Returns
/// A tuple of `(rate_in_bpm, confidence_score)`.
pub fn estimate_rate_periodogram(
    signal: &[f32],
    fs: f32,
    min_bpm: f32,
    max_bpm: f32,
    target_res_hz: f32,
    scratch_opt: Option<&mut FftScratch>
) -> (f32, f32) {
    let mut local_scratch;
    let scratch = match scratch_opt {
        Some(s) => s,
        None => {
            local_scratch = FftScratch::new();
            &mut local_scratch
        }
    };

    compute_periodogram(signal, fs, target_res_hz, scratch, true);
    
    if scratch.frequencies.is_empty() {
        return (0.0, 0.0);
    }

    let freqs = &scratch.frequencies;
    let power = &scratch.power;

    let freq_per_bin = freqs[1] - freqs[0];
    let min_bin = (min_bpm / 60.0 / freq_per_bin).floor() as usize;
    let max_bin = (max_bpm / 60.0 / freq_per_bin).ceil() as usize;
    let max_bin = max_bin.min(freqs.len() - 1);

    if min_bin >= max_bin { return (0.0, 0.0); }

    let mut max_energy = -1.0;
    let mut peak_idx = 0;
    let mut total_band_energy = 0.0;

    for k in min_bin..=max_bin {
        let energy = power[k];
        total_band_energy += energy;
        if energy > max_energy {
            max_energy = energy;
            peak_idx = k;
        }
    }

    if max_energy <= 0.0 || peak_idx == 0 || peak_idx >= freqs.len() - 1 {
        return (0.0, 0.0);
    }

    let y_c = max_energy.sqrt();
    let y_l = power[peak_idx - 1].sqrt();
    let y_r = power[peak_idx + 1].sqrt();

    let denom = 2.0 * (y_l - 2.0 * y_c + y_r);
    let delta = if denom.abs() > 1e-9 { (y_l - y_r) / denom } else { 0.0 };
    let delta = delta.max(-0.5).min(0.5);
    
    let bpm = (peak_idx as f32 + delta) * freq_per_bin * 60.0;

    let search_radius_bpm = 5.0; 
    let search_radius_bins = (search_radius_bpm / 60.0 / freq_per_bin).ceil() as usize;
    let lobe_start = peak_idx.saturating_sub(search_radius_bins).max(min_bin);
    let lobe_end = (peak_idx + search_radius_bins).min(max_bin);
    
    let lobe_energy: f32 = power[lobe_start..=lobe_end].iter().sum();
    let confidence = if total_band_energy > 0.0 { (lobe_energy / total_band_energy).min(1.0) } else { 0.0 };

    (bpm, confidence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

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
            let mut seed: u64 = 12345;
            for x in self.data.iter_mut() {
                seed = (1103515245 * seed + 12345) % 2147483648;
                let noise = (seed as f32 / 2147483648.0) * 2.0 - 1.0;
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

    #[test]
    fn test_pure_sine_accuracy() {
        let fs = 30.0;
        let target_bpm = 72.0;
        let freq = target_bpm / 60.0;

        let signal = SignalBuilder::new(10.0, fs)
            .add_sine(freq, 1.0)
            .get();

        let (bpm, conf) = estimate_rate_periodogram(&signal, fs, 40.0, 200.0, 0.005, None);

        assert!((bpm - target_bpm).abs() < 0.1, "Expected {}, got {}", target_bpm, bpm);
        assert!(conf > 0.8, "Confidence should be high for pure sine. Got {}", conf);
    }

    #[test]
    fn test_high_heart_rate() {
        let fs = 30.0;
        let target_bpm = 180.0;
        let signal = SignalBuilder::new(5.0, fs)
            .add_sine(target_bpm / 60.0, 1.0)
            .get();

        let (bpm, _) = estimate_rate_periodogram(&signal, fs, 40.0, 220.0, 0.005, None);
        assert!((bpm - target_bpm).abs() < 0.5);
    }

    #[test]
    fn test_low_heart_rate() {
        let fs = 30.0;
        let target_bpm = 45.0;
        let signal = SignalBuilder::new(10.0, fs)
            .add_sine(target_bpm / 60.0, 1.0)
            .get();

        let (bpm, _) = estimate_rate_periodogram(&signal, fs, 40.0, 220.0, 0.005, None);
        assert!((bpm - target_bpm).abs() < 0.5);
    }

    #[test]
    fn test_dc_offset_invariance() {
        let fs = 30.0;
        let target_bpm = 60.0;
        let signal = SignalBuilder::new(10.0, fs)
            .add_sine(1.0, 0.5) 
            .add_offset(1000.0)
            .get();

        let (bpm, conf) = estimate_rate_periodogram(&signal, fs, 40.0, 200.0, 0.005, None);

        assert!((bpm - target_bpm).abs() < 0.2);
        assert!(conf > 0.8, "Offset shouldn't ruin confidence. Got {}", conf);
    }

    #[test]
    fn test_noisy_signal_still_detects_rate() {
        let fs = 30.0;
        let target_bpm = 80.0;
        let signal = SignalBuilder::new(8.0, fs)
            .add_sine(target_bpm / 60.0, 1.0)
            .add_noise(0.5) 
            .get();

        let (bpm, conf) = estimate_rate_periodogram(&signal, fs, 40.0, 200.0, 0.005, None);

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

        let (_, conf) = estimate_rate_periodogram(&signal, fs, 40.0, 200.0, 0.005, None);
        
        assert!(conf < 0.4, "Pure noise should have low confidence. Got {}", conf);
    }

    #[test]
    fn test_short_signal_padding() {
        let fs = 30.0;
        let target_bpm = 60.0;
        let signal = SignalBuilder::new(2.0, fs)
            .add_sine(1.0, 1.0)
            .get();

        let (bpm, _) = estimate_rate_periodogram(&signal, fs, 40.0, 200.0, 0.001, None);

        assert!((bpm - target_bpm).abs() < 3.0); 
    }

    #[test]
    fn test_flat_line() {
        let signal = vec![10.0; 100];
        let (bpm, conf) = estimate_rate_periodogram(&signal, 30.0, 40.0, 200.0, 0.005, None);
        
        assert_eq!(bpm, 0.0);
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn test_empty_input() {
        let signal: Vec<f32> = vec![];
        let (bpm, conf) = estimate_rate_periodogram(&signal, 30.0, 40.0, 200.0, 0.005, None);
        
        assert_eq!(bpm, 0.0);
        assert_eq!(conf, 0.0);
    }

    #[test]
    fn test_nan_handling() {
        let mut signal = vec![0.0; 100];
        signal[50] = f32::NAN; 
        
        let result = std::panic::catch_unwind(|| {
            estimate_rate_periodogram(&signal, 30.0, 40.0, 200.0, 0.005, None)
        });
        
        assert!(result.is_ok(), "Function panicked on NaN input");
    }

    #[test]
    fn test_psd_frequency_resolution() {
        let fs = 30.0;
        let target_res = 0.1;
        let signal = vec![0.0; 100];
        
        let mut scratch = FftScratch::new();
        compute_periodogram(&signal, fs, target_res, &mut scratch, true);
        
        let actual_res = scratch.frequencies[1] - scratch.frequencies[0];
        assert!(actual_res <= target_res);
        assert_eq!(scratch.frequencies.len(), scratch.power.len());
    }

    #[test]
    fn test_psd_peak_location() {
        let fs = 20.0;
        let target_hz = 2.0;
        let signal = SignalBuilder::new(5.0, fs)
            .add_sine(target_hz, 1.0)
            .get();
        
        let mut scratch = FftScratch::new();
        compute_periodogram(&signal, fs, 0.01, &mut scratch, true);
        
        let (max_idx, _) = scratch.power.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
            
        let peak_freq = scratch.frequencies[max_idx];
        
        assert!((peak_freq - target_hz).abs() < 0.05);
    }

    #[test]
    fn test_psd_energy_scaling() {
        let fs = 10.0;
        let sig1 = SignalBuilder::new(5.0, fs).add_sine(1.0, 1.0).get();
        let sig2 = SignalBuilder::new(5.0, fs).add_sine(1.0, 2.0).get();
        
        let mut scratch = FftScratch::new();
        
        compute_periodogram(&sig1, fs, 0.1, &mut scratch, true);
        let max_p1 = *scratch.power.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        compute_periodogram(&sig2, fs, 0.1, &mut scratch, true);
        let max_p2 = *scratch.power.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        let ratio = max_p2 / max_p1;
        assert!((ratio - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_psd_dc_removal() {
        let fs = 10.0;
        let signal = SignalBuilder::new(5.0, fs)
            .add_sine(1.0, 1.0)
            .add_offset(100.0)
            .get();
            
        let mut scratch = FftScratch::new();
        compute_periodogram(&signal, fs, 0.1, &mut scratch, true);
        
        let dc_power = scratch.power[0];
        let sig_power = *scratch.power.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        assert!(dc_power < sig_power * 0.01); 
    }
}