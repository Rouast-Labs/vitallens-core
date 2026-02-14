use crate::signal::{fft, peaks};

#[derive(Debug, Clone, Copy)]
pub struct RateBounds {
    pub min: f32,
    pub max: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RateStrategy {
    Periodogram {
        target_res_hz: f32,
    },
    PeakDetection {
        refine: bool,
        interval_buffer: f32,
    },
}

#[derive(Debug, Clone)]
pub struct RateResult {
    pub value: f32,
    pub confidence: f32,
    pub method: String,
}

/// Generic rate estimator (Signal Agnostic).
pub fn estimate_rate(
    signal: &[f32],
    fs: f32,
    bounds: RateBounds,
    strategy: RateStrategy,
    rate_hint: Option<f32>,
    scratch: Option<&mut fft::FftScratch>,
) -> RateResult {
    match strategy {
        RateStrategy::Periodogram { target_res_hz } => {
            let (val, conf) = fft::estimate_rate_periodogram(
                signal, fs, bounds.min, bounds.max, target_res_hz, scratch
            );
            
            RateResult {
                value: val,
                confidence: conf,
                method: "Periodogram".to_string(),
            }
        },
        RateStrategy::PeakDetection { refine, interval_buffer } => {
            // Configure the Peak Detector using the generic bounds and hint
            let options = peaks::PeakOptions {
                fs,
                avg_rate_hint: rate_hint,
                bounds: peaks::SignalBounds { 
                    min_rate: bounds.min, 
                    max_rate: bounds.max 
                },
                interval_buffer,
                refine,
                // Use intelligent defaults for the rest
                ..Default::default()
            };

            let segments = peaks::find_peaks(signal, options);
            
            if segments.is_empty() {
                return RateResult { value: 0.0, confidence: 0.0, method: "PeakDetection".to_string() };
            }

            // Aggregate intervals from all valid segments
            let mut all_intervals = Vec::new();
            for segment in segments {
                if segment.len() < 2 { continue; }
                for i in 0..segment.len()-1 {
                    let p1 = segment[i];
                    let p2 = segment[i+1];
                    let diff_samples = p2.x - p1.x;
                    let diff_secs = diff_samples / fs;
                    if diff_secs > 0.0 {
                        all_intervals.push(diff_secs);
                    }
                }
            }
            
            if all_intervals.is_empty() {
                return RateResult { value: 0.0, confidence: 0.0, method: "PeakDetection".to_string() };
            }

            // Calculate Statistics
            let sum: f32 = all_intervals.iter().sum();
            let mean_interval = sum / all_intervals.len() as f32;
            
            // Coefficient of Variation (CV) for Confidence
            let variance: f32 = all_intervals.iter()
                .map(|val| (val - mean_interval).powi(2))
                .sum::<f32>() / all_intervals.len() as f32;
            let std_dev = variance.sqrt();
            
            let cv = if mean_interval > 0.0 { std_dev / mean_interval } else { 0.0 };
            
            // Map CV to Confidence (Healthy < 0.1 -> High Conf)
            let confidence = (1.0 - (cv / 0.3)).max(0.0);
            
            let rate = if mean_interval > 0.0 { 60.0 / mean_interval } else { 0.0 };

            RateResult {
                value: rate,
                confidence,
                method: "PeakDetection".to_string(),
            }
        }
    }
}