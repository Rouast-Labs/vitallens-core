use crate::signal::peaks::{find_peaks, PeakOptions, Peak};

pub struct BpResult {
    pub sbp: f32,
    pub dbp: f32,
}

pub fn extract_systolic_pressure(signal: &[f32], fs: f32) -> (f32, f32) {
    let opts = PeakOptions { fs, ..Default::default() };
    let sequences = find_peaks(signal, opts);
    
    if sequences.is_empty() { return (0.0, 0.0); }
    
    // explicit types added to satisfy E0282
    let sum: f32 = sequences.iter()
        .flat_map(|s: &Vec<Peak>| s.iter().map(|p: &Peak| p.y))
        .sum();
        
    let count: usize = sequences.iter()
        .map(|s: &Vec<Peak>| s.len())
        .sum();
    
    if count == 0 { return (0.0, 0.0); }

    (sum / count as f32, 0.8)
    // TODO: need to use confs from the detected peaks
}

pub fn extract_diastolic_pressure(signal: &[f32], fs: f32) -> (f32, f32) {
    let inverted: Vec<f32> = signal.iter().map(|&x| -x).collect();
    let opts = PeakOptions { fs, ..Default::default() };
    let sequences = find_peaks(&inverted, opts);
    
    if sequences.is_empty() { return (0.0, 0.0); }
    
    // explicit types added to satisfy E0282
    let sum: f32 = sequences.iter()
        .flat_map(|s: &Vec<Peak>| s.iter().map(|p: &Peak| -p.y)) // negate back to original domain
        .sum();
        
    let count: usize = sequences.iter()
        .map(|s: &Vec<Peak>| s.len())
        .sum();
    
    if count == 0 { return (0.0, 0.0); }

    (sum / count as f32, 0.8)
    // TODO: need to use confs from the detected troughs
}