use crate::signal::peaks::{find_peaks, PeakOptions};

pub struct BpResult {
    pub sbp: f32,
    pub dbp: f32,
}

/// Helper to avoid code duplication between SBP and DBP extraction
fn extract_pressure_internal(signal: &[f32], fs: f32, confidence: &[f32], invert: bool) -> (f32, f32) {
    let input = if invert {
        signal.iter().map(|&x| -x).collect::<Vec<f32>>()
    } else {
        signal.to_vec()
    };

    let opts = PeakOptions { fs, threshold: 1.0, ..Default::default() };
    let sequences = find_peaks(&input, opts);
    
    if sequences.is_empty() { return (0.0, 0.0); }
    
    let mut val_sum = 0.0;
    let mut conf_sum = 0.0;
    let mut count = 0;

    for seq in sequences {
        for p in seq {
            val_sum += if invert { -p.y } else { p.y };
            
            if p.index < confidence.len() {
                conf_sum += confidence[p.index];
            }
            count += 1;
        }
    }
    
    if count == 0 { (0.0, 0.0) } else { (val_sum / count as f32, conf_sum / count as f32) }
}

pub fn extract_systolic_pressure(signal: &[f32], fs: f32, confidence: &[f32]) -> (f32, f32) {
    extract_pressure_internal(signal, fs, confidence, false)
}

pub fn extract_diastolic_pressure(signal: &[f32], fs: f32, confidence: &[f32]) -> (f32, f32) {
    extract_pressure_internal(signal, fs, confidence, true)
}

pub fn extract_pulse_pressure(signal: &[f32], fs: f32, confidence: &[f32]) -> (f32, f32) {
    let (sbp, c_sys) = extract_systolic_pressure(signal, fs, confidence);
    let (dbp, c_dia) = extract_diastolic_pressure(signal, fs, confidence);

    if sbp > 0.0 && dbp > 0.0 {
        (sbp - dbp, (c_sys + c_dia) / 2.0)
    } else {
        (0.0, 0.0)
    }
}

pub fn calculate_map_from_signals(sbp: &[f32], dbp: &[f32]) -> (f32, f32) {
    let len = sbp.len().min(dbp.len());
    if len == 0 { return (0.0, 0.0); }

    let mut sum_map = 0.0;
    
    for i in 0..len {
        let val = (sbp[i] + 2.0 * dbp[i]) / 3.0;
        sum_map += val;
    }

    (sum_map / len as f32, 1.0) 
}

pub fn calculate_pp_from_signals(sbp: &[f32], dbp: &[f32]) -> (f32, f32) {
    let len = sbp.len().min(dbp.len());
    if len == 0 { return (0.0, 0.0); }

    let mut sum_pp = 0.0;
    
    for i in 0..len {
        let val = sbp[i] - dbp[i];
        sum_pp += val;
    }

    (sum_pp / len as f32, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn mock_abp_signal(fs: f32, duration_sec: f32, _hr_bpm: f32, sbp: f32, dbp: f32) -> Vec<f32> {
        let mut signal = Vec::new();
        let total_samples = (fs * duration_sec) as usize;
        
        let beat_durations = [
            0.92, 1.05, 0.88, 1.12, 0.98, 1.02, 0.95, 1.08, 0.90, 1.15,
            0.93, 1.01, 0.89, 1.10, 0.97, 1.04, 0.94, 1.11, 0.91, 1.14
        ];
        
        let mut beat_idx = 0;
        
        while signal.len() < total_samples {
            let duration = beat_durations[beat_idx % beat_durations.len()];
            let samples_in_beat = (duration * fs) as usize;
            
            for i in 0..samples_in_beat {
                if signal.len() >= total_samples { break; }
                
                let val = if i < samples_in_beat / 4 {
                    let sub_phase = (i as f32 / (samples_in_beat as f32 * 0.25)) * (PI / 2.0);
                    dbp + (sbp - dbp) * sub_phase.sin()
                } else {
                    let sub_phase = ((i - samples_in_beat / 4) as f32 / (samples_in_beat as f32 * 0.75)) * (PI / 2.0);
                    sbp - (sbp - dbp) * sub_phase.sin().powf(0.5)
                };
                
                let noise = if i % 2 == 0 { 0.05 } else { -0.05 };
                signal.push(val + noise);
            }
            beat_idx += 1;
        }
        signal
    }

    #[test]
    fn test_ideal_sbp_dbp_extraction() {
        let fs = 30.0;
        let signal = mock_abp_signal(fs, 30.0, 60.0, 120.0, 80.0);
        let confidence = vec![1.0; signal.len()];

        let (sbp, sbp_conf) = extract_systolic_pressure(&signal, fs, &confidence);
        let (dbp, dbp_conf) = extract_diastolic_pressure(&signal, fs, &confidence);

        assert!(sbp > 0.0, "Systolic peak not detected. Window calibration likely failed.");
        assert!((sbp - 120.0).abs() < 2.0, "Expected SBP ~120, got {}", sbp);
        assert!((dbp - 80.0).abs() < 2.0, "Expected DBP ~80, got {}", dbp);
        
        assert_eq!(sbp_conf, 1.0);
        assert_eq!(dbp_conf, 1.0);
    }

    #[test]
    fn test_pulse_pressure_calculation() {
        let fs = 30.0;
        let signal = mock_abp_signal(fs, 30.0, 60.0, 130.0, 70.0);
        let confidence = vec![0.8; signal.len()];

        let (pp, pp_conf) = extract_pulse_pressure(&signal, fs, &confidence);

        assert!(pp > 0.0, "Pulse pressure failed: SBP or DBP was not detected.");
        assert!((pp - 60.0).abs() < 5.0, "Expected PP ~60, got {}", pp);
        assert!((pp_conf - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_confidence_selectivity() {
        let fs = 10.0;
        let mut signal = vec![0.0; 30]; 
        signal[5] = 100.0;   
        signal[15] = 100.0;  
        signal[25] = 100.0;  

        let mut confidence = vec![0.1; 30];  
        confidence[5] = 1.0;   
        confidence[15] = 0.5;  
        confidence[25] = 0.9;  

        let (sbp, sbp_conf) = extract_systolic_pressure(&signal, fs, &confidence);

        assert!((sbp - 100.0).abs() < 0.1);
        assert!((sbp_conf - 0.8).abs() < 0.01, "Got {}, expected 0.8", sbp_conf);
    }

    #[test]
    fn test_flatline_returns_zeros() {
        let signal = vec![100.0; 50];  
        let confidence = vec![1.0; 50];
        let fs = 30.0;

        let (sbp, _) = extract_systolic_pressure(&signal, fs, &confidence);
        let (dbp, _) = extract_diastolic_pressure(&signal, fs, &confidence);
        let (pp, _) = extract_pulse_pressure(&signal, fs, &confidence);

        assert_eq!(sbp, 0.0);
        assert_eq!(dbp, 0.0);
        assert_eq!(pp, 0.0);
    }

    #[test]
    fn test_empty_signal() {
        let signal: Vec<f32> = vec![];
        let confidence: Vec<f32> = vec![];
        let fs = 30.0;

        let (sbp, _) = extract_systolic_pressure(&signal, fs, &confidence);
        assert_eq!(sbp, 0.0);
    }

    #[test]
    fn test_missing_confidence_bounds_safety() {
        let fs = 10.0;
        let mut signal = vec![0.0; 20];
        signal[10] = 120.0;  
        
        let confidence = vec![1.0; 5]; 

        let (sbp, sbp_conf) = extract_systolic_pressure(&signal, fs, &confidence);

        assert!((sbp - 120.0).abs() < 1.0);
        assert_eq!(sbp_conf, 0.0);
    }
}