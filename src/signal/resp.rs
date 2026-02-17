use crate::signal::peaks::{find_cycles, PeakOptions, SignalBounds};

pub fn calculate_ie_ratio(
    signal: &[f32], 
    fs: f32, 
    confidence: &[f32],
    bounds: SignalBounds,
    rate_hint: Option<f32>,
) -> (f32, f32) {
    let opts = PeakOptions { 
        fs,
        bounds,
        avg_rate_hint: rate_hint,
        threshold: 1.2,
        window_cycles: 1.5,
        max_rate_change_per_sec: 0.25,
        refine: true,
        smooth_input: true,
        ..Default::default()
    };

    log::debug!("[Resp] Calculating I:E. Fs: {:.1}, Bounds: [{:.1}-{:.1}], Rate Hint: {:?}", 
        fs, bounds.min_rate, bounds.max_rate, rate_hint);

    // Use the integrated state machine detector
    let cycles = find_cycles(signal, opts);

    if cycles.is_empty() {
        log::warn!("[Resp] No respiration cycles detected.");
        return (0.0, 0.0);
    }

    log::debug!("[Resp] Found {} raw cycles.", cycles.len());

    let mut total_insp_time = 0.0;
    let mut total_exp_time = 0.0;
    let mut total_confidence = 0.0;
    let mut count = 0;

    for (i, cycle) in cycles.iter().enumerate() {
        // Calculate duration based on refined x-coordinates
        let t_insp = (cycle.peak.x - cycle.start_valley.x) / fs;
        let t_exp = (cycle.end_valley.x - cycle.peak.x) / fs;

        // Sanity check: durations must be positive
        if t_insp > 0.0 && t_exp > 0.0 {
            total_insp_time += t_insp;
            total_exp_time += t_exp;

            // Weigh confidence
            let conf = if cycle.peak.index < confidence.len() {
                confidence[cycle.peak.index]
            } else {
                0.0
            };
            total_confidence += conf;
            count += 1;

            log::trace!("[Resp] Cycle {}: StartV={:.1} -> Peak={:.1} -> EndV={:.1} | Insp={:.3}s, Exp={:.3}s, Conf={:.2}",
                i, cycle.start_valley.x, cycle.peak.x, cycle.end_valley.x, t_insp, t_exp, conf);
        } else {
            log::warn!("[Resp] Cycle {} REJECTED: Invalid durations. Insp={:.3}s, Exp={:.3}s", i, t_insp, t_exp);
        }
    }

    if count == 0 {
        log::warn!("[Resp] All detected cycles were invalid (zero duration).");
        return (0.0, 0.0);
    }

    // Calculate averages
    let avg_insp = total_insp_time / count as f32;
    let avg_exp = total_exp_time / count as f32;
    let avg_conf = total_confidence / count as f32;

    log::debug!("[Resp] Final Stats: Avg Insp={:.3}s, Avg Exp={:.3}s, Cycles Used={}", avg_insp, avg_exp, count);

    if avg_exp > 1e-5 {
        let ratio = avg_insp / avg_exp;
        log::info!("[Resp] Result I:E Ratio: {:.3} (Conf: {:.2})", ratio, avg_conf);
        (ratio, avg_conf)
    } else {
        log::warn!("[Resp] Expiration time near zero, returning 0.0");
        (0.0, 0.0)
    }
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
        let signal = mock_resp_signal(fs, 30.0, 12.0, 0.5);
        let confidence = vec![1.0; signal.len()];
        let bounds = SignalBounds { min_rate: 4.0, max_rate: 60.0 };

        let (ratio, _) = calculate_ie_ratio(&signal, fs, &confidence, bounds, None);
        assert!((ratio - 1.0).abs() < 0.15);
    }

    #[test]
    fn test_ie_ratio_classic_human() {
        let fs = 20.0;
        // 1:2 ratio (Inhalation is 1/3 of total breath)
        let signal = mock_resp_signal(fs, 60.0, 15.0, 0.333);
        let confidence = vec![1.0; signal.len()];
        let bounds = SignalBounds { min_rate: 4.0, max_rate: 60.0 };

        let (ratio, _) = calculate_ie_ratio(&signal, fs, &confidence, bounds, None);
        
        // Expected ~0.5
        assert!((ratio - 0.5).abs() < 0.1, "Expected ratio ~0.5, got {}", ratio);
    }

    #[test]
    fn test_resp_empty_and_flatline() {
        let fs = 10.0;
        let bounds = SignalBounds { min_rate: 4.0, max_rate: 60.0 };

        let (r1, _) = calculate_ie_ratio(&[], fs, &[], bounds, None);
        let (r2, _) = calculate_ie_ratio(&vec![0.5; 100], fs, &vec![1.0; 100], bounds, None);
        
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
        let bounds = SignalBounds { min_rate: 4.0, max_rate: 60.0 };

        let (_, avg_conf) = calculate_ie_ratio(&signal, fs, &confidence, bounds, None);
        
        assert!(avg_conf < 1.0 && avg_conf > 0.0);
    }
}