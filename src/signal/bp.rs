use crate::signal::peaks::{find_peaks, PeakOptions};

pub struct BpResult {
    pub sbp: f32,
    pub dbp: f32,
}

pub fn extract_systolic_pressure(signal: &[f32], fs: f32, confidence: &[f32]) -> (f32, f32) {
    let opts = PeakOptions { fs, ..Default::default() };
    let sequences = find_peaks(signal, opts);
    
    if sequences.is_empty() { return (0.0, 0.0); }
    
    let mut val_sum = 0.0;
    let mut conf_sum = 0.0;
    let mut count = 0;

    for seq in sequences {
        for p in seq {
            val_sum += p.y;
            
            if p.index < confidence.len() {
                conf_sum += confidence[p.index];
            } else {
                conf_sum += 0.0; 
            }
            count += 1;
        }
    }
    
    if count == 0 { return (0.0, 0.0); }

    (val_sum / count as f32, conf_sum / count as f32)
}

pub fn extract_diastolic_pressure(signal: &[f32], fs: f32, confidence: &[f32]) -> (f32, f32) {
    let inverted: Vec<f32> = signal.iter().map(|&x| -x).collect();
    let opts = PeakOptions { fs, ..Default::default() };
    let sequences = find_peaks(&inverted, opts);
    
    if sequences.is_empty() { return (0.0, 0.0); }
    
    let mut val_sum = 0.0;
    let mut conf_sum = 0.0;
    let mut count = 0;

    for seq in sequences {
        for p in seq {
            val_sum += -p.y;
            
            if p.index < confidence.len() {
                conf_sum += confidence[p.index];
            } else {
                conf_sum += 0.0;
            }
            count += 1;
        }
    }
    
    if count == 0 { return (0.0, 0.0); }

    (val_sum / count as f32, conf_sum / count as f32)
}