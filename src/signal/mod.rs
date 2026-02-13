pub mod filters;
pub mod fft;
pub mod peaks;

pub fn process_batch(signal: &[f32], _fs: f32) -> f32 {
    // Placeholder: Return average
    if signal.is_empty() { return 0.0; }
    signal.iter().sum::<f32>() / signal.len() as f32
}