use crate::signal::peaks::{find_peaks, PeakOptions, SignalBounds};

// TODO: Add ie ratio to jsons (manually verified)
// TODO: ie ratio integration test

pub fn calculate_ie_ratio(signal: &[f32], fs: f32, confidence: &[f32]) -> (f32, f32) {
    let opts = PeakOptions { 
        fs,
        // Bounds: 
        // - Min: 4 BPM (Very slow, deep meditative breathing)
        // - Max: 60 BPM (Hyperventilation / High stress)
        bounds: SignalBounds { 
            min_rate: 4.0, 
            max_rate: 60.0 
        },
        threshold: 1.0,
        ..Default::default() 
    };
    
    // Find Peaks (End-Inspiration)
    let peak_sequences = find_peaks(signal, opts);
    
    // Find Valleys/Troughs (End-Expiration)
    // Invert signal to find valleys using the same peak logic
    let inverted: Vec<f32> = signal.iter().map(|&x| -x).collect();
    let valley_sequences = find_peaks(&inverted, opts);

    if peak_sequences.is_empty() || valley_sequences.is_empty() {
        return (0.0, 0.0);
    }

    // Flatten sequences for simpler processing
    // (We assume the window provided is continuous enough for a ratio calc)
    let peaks = &peak_sequences[0];
    let valleys = &valley_sequences[0];

    // We need at least one full cycle to calculate a ratio
    if peaks.len() < 2 || valleys.len() < 2 { return (0.0, 0.0); }

    let mut insp_times = Vec::new();
    let mut exp_times = Vec::new();
    let mut used_confidences = Vec::new();

    // Logic: A breath cycle is Trough -> Peak (Inspiration) -> Trough (Expiration)
    // We iterate through valleys to find the corresponding peaks
    for i in 0..valleys.len() - 1 {
        let v_start = &valleys[i];
        let v_end = &valleys[i+1];
        
        // Find a peak that sits strictly between these two valleys
        if let Some(p) = peaks.iter().find(|p| p.index > v_start.index && p.index < v_end.index) {
            
            let t_insp = (p.index - v_start.index) as f32 / fs;
            let t_exp = (v_end.index - p.index) as f32 / fs;

            // Basic sanity check to avoid divide-by-zero or negative time
            if t_insp > 0.0 && t_exp > 0.0 {
                insp_times.push(t_insp);
                exp_times.push(t_exp);

                // Track confidence from the key structural points
                if p.index < confidence.len() { used_confidences.push(confidence[p.index]); }
                if v_start.index < confidence.len() { used_confidences.push(confidence[v_start.index]); }
                if v_end.index < confidence.len() { used_confidences.push(confidence[v_end.index]); }
            }
        }
    }

    if insp_times.is_empty() { return (0.0, 0.0); }

    // Calculate Average Duration for Inspiration and Expiration
    let avg_insp: f32 = insp_times.iter().sum::<f32>() / insp_times.len() as f32;
    let avg_exp: f32 = exp_times.iter().sum::<f32>() / exp_times.len() as f32;
    
    let avg_conf = if !used_confidences.is_empty() {
        used_confidences.iter().sum::<f32>() / used_confidences.len() as f32
    } else {
        0.0
    };

    if avg_exp > 0.0 {
        (avg_insp / avg_exp, avg_conf)
    } else {
        (0.0, 0.0)
    }
}

/// Derivative Method (Slope)
/// Best for: Very smooth, synthetic-like model outputs.
/// Logic: Positive slope = Inhale, Negative slope = Exhale.
pub fn calculate_ie_ratio_slope(signal: &[f32], _fs: f32, confidence: &[f32]) -> (f32, f32) {
    if signal.len() < 2 { return (0.0, 0.0); }

    let mut insp_count = 0;
    let mut exp_count = 0;
    let mut total_conf = 0.0;
    let mut conf_samples = 0;

    // We iterate from index 1 and compare to i-1
    for i in 1..signal.len() {
        let diff = signal[i] - signal[i - 1];

        // A tiny epsilon prevents floating point jitter around zero from flipping states
        let epsilon = 1e-5;

        if diff > epsilon {
            insp_count += 1;
        } else if diff < -epsilon {
            exp_count += 1;
        }
        // Note: We ignore flat segments (diff == 0). 
        // In a smooth model output, these are rare (just the exact turning points).

        // Accumulate confidence for the entire window since every point contributes
        if i < confidence.len() {
            total_conf += confidence[i];
            conf_samples += 1;
        }
    }

    if exp_count == 0 { return (0.0, 0.0); }

    // Ratio = Time spent rising / Time spent falling
    let ratio = insp_count as f32 / exp_count as f32;

    let avg_conf = if conf_samples > 0 {
        total_conf / conf_samples as f32
    } else {
        0.0
    };

    (ratio, avg_conf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generates a synthetic respiration signal
    /// insp_ratio: 0.5 means symmetrical (1:1), 0.33 means 1:2 ratio
    fn mock_resp_signal(fs: f32, duration_sec: f32, breath_rate_bpm: f32, insp_ratio: f32) -> Vec<f32> {
        let total_samples = (fs * duration_sec) as usize;
        let mut signal = Vec::with_capacity(total_samples);
        let samples_per_breath = (fs * 60.0 / breath_rate_bpm) as usize;

        for i in 0..total_samples {
            let phase = (i % samples_per_breath) as f32 / samples_per_breath as f32;
            let val = if phase < insp_ratio {
                // Inhalation (Up) - Removed the () here
                phase / insp_ratio
            } else {
                // Exhalation (Down)
                1.0 - ((phase - insp_ratio) / (1.0 - insp_ratio))
            };
            signal.push(val);
        }
        signal
    }

    #[test]
    fn test_ie_ratio_symmetrical() {
        let fs = 10.0;
        // 1:1 ratio (insp_ratio 0.5)
        let signal = mock_resp_signal(fs, 30.0, 12.0, 0.5);
        let confidence = vec![1.0; signal.len()];

        let (ratio, _) = calculate_ie_ratio(&signal, fs, &confidence);
        
        // Expected ~1.0
        assert!((ratio - 1.0).abs() < 0.15, "Expected ratio ~1.0, got {}", ratio);
    }

    #[test]
    fn test_ie_ratio_classic_human() {
        let fs = 20.0;
        // 1:2 ratio (Inhalation is 1/3 of total breath)
        let signal = mock_resp_signal(fs, 60.0, 15.0, 0.333);
        let confidence = vec![1.0; signal.len()];

        let (ratio, _) = calculate_ie_ratio(&signal, fs, &confidence);
        
        // Expected ~0.5
        assert!((ratio - 0.5).abs() < 0.1, "Expected ratio ~0.5, got {}", ratio);
    }

    #[test]
    fn test_ie_ratio_slope_method() {
        let fs = 10.0;
        let signal = mock_resp_signal(fs, 20.0, 12.0, 0.25); // 1:3 ratio
        let confidence = vec![1.0; signal.len()];

        let (ratio, _) = calculate_ie_ratio_slope(&signal, fs, &confidence);
        
        // The slope method counts raw sample directions
        // Expected ~0.33
        assert!((ratio - 0.33).abs() < 0.1, "Slope method expected ~0.33, got {}", ratio);
    }

    #[test]
    fn test_resp_empty_and_flatline() {
        let fs = 10.0;
        let (r1, _) = calculate_ie_ratio(&[], fs, &[]);
        let (r2, _) = calculate_ie_ratio(&vec![0.5; 100], fs, &vec![1.0; 100]);
        
        assert_eq!(r1, 0.0);
        assert_eq!(r2, 0.0);
    }

    #[test]
    fn test_ie_ratio_confidence_weighting() {
        let fs = 10.0;
        let signal = mock_resp_signal(fs, 20.0, 12.0, 0.5);
        // Half the signal is low confidence
        let mut confidence = vec![1.0; signal.len()];
        for i in (signal.len()/2)..signal.len() {
            confidence[i] = 0.2;
        }

        let (_, avg_conf) = calculate_ie_ratio(&signal, fs, &confidence);
        
        assert!(avg_conf < 1.0 && avg_conf > 0.0);
    }

    #[test]
    fn test_compare_methods_clean_signal() {
        let fs = 10.0;
        let insp_ratio = 0.4; // 1:1.5 ratio
        let signal = mock_resp_signal(fs, 30.0, 12.0, insp_ratio);
        let confidence = vec![1.0; signal.len()];

        let (ratio_peak, _) = calculate_ie_ratio(&signal, fs, &confidence);
        let (ratio_slope, _) = calculate_ie_ratio_slope(&signal, fs, &confidence);

        // Peak method: Inhale time = 0.4 * period, Exhale = 0.6 * period. Ratio = 0.66
        // Slope method: Counts samples. 4 up, 6 down. Ratio = 0.66
        assert!((ratio_peak - 0.66).abs() < 0.1);
        assert!((ratio_slope - 0.66).abs() < 0.1);
    }

    #[test]
    fn test_slope_method_noise_sensitivity() {
        let fs = 10.0;
        let mut signal = mock_resp_signal(fs, 20.0, 12.0, 0.5); // Perfect 1:1
        
        // Add "jitter" noise that flips the slope direction constantly
        for i in (0..signal.len()).step_by(2) {
            signal[i] += 0.01;
        }
        
        let confidence = vec![1.0; signal.len()];
        let (ratio_peak, _) = calculate_ie_ratio(&signal, fs, &confidence);
        let (ratio_slope, _) = calculate_ie_ratio_slope(&signal, fs, &confidence);

        // Peak method should still find the macro peaks and return ~1.0
        assert!((ratio_peak - 1.0).abs() < 0.2);
        
        // Slope method will likely be skewed or close to 1.0 depending on jitter,
        // but it's important to verify it doesn't crash or return NaN.
        assert!(ratio_slope.is_finite());
    }

    #[test]
    fn test_slope_method_all_inhale() {
        // Edge case: Signal only goes up
        let signal: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let confidence = vec![1.0; 100];
        
        let (ratio, _) = calculate_ie_ratio_slope(&signal, 10.0, &confidence);
        
        // Should return 0.0 because exp_count is 0
        assert_eq!(ratio, 0.0);
    }

    #[test]
    fn test_ie_ratio_slope_confidence_averaging() {
        let signal = vec![0.0, 1.0, 2.0, 1.0, 0.0];
        let confidence = vec![1.0, 0.8, 0.6, 0.4, 0.2];
        
        let (_, avg_conf) = calculate_ie_ratio_slope(&signal, 10.0, &confidence);
        
        // Average of [0.8, 0.6, 0.4, 0.2] = 0.5
        assert!((avg_conf - 0.5).abs() < 0.01);
    }
}