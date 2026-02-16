use crate::signal::{fft, peaks};
use crate::signal::peaks::Peak;

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
            
            let options = peaks::PeakOptions {
                fs,
                avg_rate_hint: rate_hint,
                bounds: peaks::SignalBounds { 
                    min_rate: bounds.min, 
                    max_rate: bounds.max 
                },
                interval_buffer,
                refine,
                
                ..Default::default()
            };

            let segments = peaks::find_peaks(signal, options);
            
            calculate_from_peaks(segments, fs)
        }
    }
}

fn calculate_from_peaks(segments: Vec<Vec<Peak>>, fs: f32) -> RateResult {
    if segments.is_empty() {
        return RateResult { value: 0.0, confidence: 0.0, method: "PeakDetection".to_string() };
    }

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

    let sum: f32 = all_intervals.iter().sum();
    let mean_interval = sum / all_intervals.len() as f32;
    
    let variance: f32 = all_intervals.iter()
        .map(|val| (val - mean_interval).powi(2))
        .sum::<f32>() / all_intervals.len() as f32;
    let std_dev = variance.sqrt();
    
    let cv = if mean_interval > 0.0 { std_dev / mean_interval } else { 0.0 };
    
    let confidence = (1.0 - (cv / 0.3)).max(0.0);
    let rate = if mean_interval > 0.0 { 60.0 / mean_interval } else { 0.0 };

    RateResult {
        value: rate,
        confidence,
        method: "PeakDetection".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f32) -> Peak {
        Peak { index: x as usize, x, y: 1.0 }
    }

    #[test]
    fn test_logic_perfect_rhythm_returns_high_confidence() {
        let fs = 30.0;
        
        let segments = vec![
            vec![p(0.0), p(30.0), p(60.0)]
        ];

        let result = calculate_from_peaks(segments, fs);

        assert_eq!(result.value, 60.0);
        assert_eq!(result.confidence, 1.0, "Perfect rhythm (CV=0) must have confidence 1.0");
    }

    #[test]
    fn test_logic_irregular_rhythm_returns_low_confidence() {
        let fs = 10.0;
        
        let segments = vec![
            vec![p(0.0), p(15.0), p(25.0)] 
        ];

        let result = calculate_from_peaks(segments, fs);

        assert!((result.value - 48.0).abs() < 0.1);        
        assert!(result.confidence < 0.8, "High CV should lower confidence. Got {}", result.confidence);
        assert!(result.confidence > 0.0, "Confidence should not be zero for moderate irregularity");
    }

    #[test]
    fn test_logic_handles_fragmented_segments() {
        let fs = 1.0;

        let segments = vec![
            vec![p(0.0), p(2.0)],
            vec![p(10.0), p(12.0)]
        ];

        let result = calculate_from_peaks(segments, fs);

        assert_eq!(result.value, 30.0);
        assert_eq!(result.confidence, 1.0, "Logic should ignore gaps between segments");
    }

    #[test]
    fn test_logic_empty_input() {
        let result = calculate_from_peaks(vec![], 30.0);
        assert_eq!(result.value, 0.0);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_logic_single_peak_segment_is_ignored() {
        let fs = 30.0;
        let segments = vec![vec![p(10.0)]]; 

        let result = calculate_from_peaks(segments, fs);
        assert_eq!(result.value, 0.0);
    }
}