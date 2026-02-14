use crate::registry::HrvMetric;

pub fn estimate_rate_fft(
    _signal: &[f32], 
    _fs: f32, 
    _min_rate: f32, 
    _max_rate: f32
) -> (f32, f32) {
    (75.0, 0.95) 
}

pub fn estimate_hrv(
    _signal: &[f32], 
    _fs: f32, 
    _metric: HrvMetric, 
    _current_hr: Option<f32>
) -> (f32, f32) {
    (30.0, 0.8)
}

pub fn calculate_average(signal: &[f32]) -> (f32, f32) {
    if signal.is_empty() { return (0.0, 0.0); }
    let sum: f32 = signal.iter().sum();
    (sum / signal.len() as f32, 1.0)
}

pub fn apply_processing(
    signal: &[f32], 
    _op: crate::registry::PostProcessOp
) -> Vec<f32> {
    signal.to_vec()
}