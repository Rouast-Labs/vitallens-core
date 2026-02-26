pub mod filters;
pub mod fft;
pub mod peaks;
pub mod rate;
pub mod hrv;
pub mod bp;
pub mod resp;

pub use hrv::estimate_hrv;
pub use rate::estimate_rate;

/// Calculates the arithmetic mean of a signal.
///
/// # Arguments
/// * `signal` - The input data slice.
///
/// # Returns
/// A tuple of `(mean_value, confidence_score)`.
pub fn calculate_average(signal: &[f32]) -> (f32, f32) {
    if signal.is_empty() { return (0.0, 0.0); }
    let sum: f32 = signal.iter().sum();
    (sum / signal.len() as f32, 1.0)
}