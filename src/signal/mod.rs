pub mod filters;
pub mod fft;
pub mod peaks;
pub mod stubs;

// Remove the old dummy process_batch function if it conflicts, 
// or keep it for backward compat until we fully replace it.
// For now, let's leave it but maybe deprecate it.
pub fn process_batch(signal: &[f32], _fs: f32) -> f32 {
    if signal.is_empty() { return 0.0; }
    signal.iter().sum::<f32>() / signal.len() as f32
}