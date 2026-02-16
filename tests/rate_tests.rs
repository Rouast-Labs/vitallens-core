use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use serde::Deserialize;
use test_generator::test_resources;

use vitallens_core::signal::{peaks, rate};
use vitallens_core::registry;

// Tolerance for Rate metrics (BPM)
const RATE_TOLERANCE_BPM: f32 = 3.0;

#[derive(Deserialize, Debug)]
struct ReferenceData {
    vital_signs: Vitals,
    fs: f32, 
}

#[derive(Deserialize, Debug)]
struct Vitals {
    ppg_waveform: Option<Waveform>,
    respiratory_waveform: Option<Waveform>,
    heart_rate: Option<ScalarResult>,
    respiratory_rate: Option<ScalarResult>,
}

#[derive(Deserialize, Debug)]
struct Waveform {
    data: Vec<f32>,
    #[allow(dead_code)]
    peak_indices: Option<Vec<usize>>, 
}

#[derive(Deserialize, Debug)]
struct ScalarResult {
    value: f32,
}

#[allow(clippy::too_many_arguments)]
fn verify_rate(
    filename: &str,
    vital_id: &str,
    fs: f32,
    signal: &[f32], 
    ground_truth_rate: f32,
    rate_hint: Option<f32>
) -> Result<(), String> {
    
    // 1. Get Settings from Registry (Single Source of Truth)
    let meta = registry::get_vital_meta(vital_id)
        .unwrap_or_else(|| panic!("Vital ID '{}' not found in registry", vital_id));
    
    // Assuming derivation 0 is the primary rate derivation (Periodogram or Peak)
    let deriv = &meta.derivations[0];

    // Check duration (Skip short files to match Session logic)
    let duration_sec = signal.len() as f32 / fs;
    if duration_sec < deriv.min_window_seconds {
        println!("[{}] {} SKIPPED: Duration {:.1}s < Min Window {:.1}s", 
            filename, vital_id, duration_sec, deriv.min_window_seconds);
        return Ok(());
    }

    // 2. Configure Strategy based on Registry
    // The registry stores the preferred calculation method (e.g. Periodogram vs PeakDetection)
    let (calculated_rate, method_name) = match deriv.method {
        registry::CalculationMethod::Rate(strategy) => {
            let bounds = rate::RateBounds { 
                min: deriv.min_value, 
                max: deriv.max_value 
            };
            
            // Note: For Periodogram, we pass None for scratch to keep test simple (it allocates internally)
            let result = rate::estimate_rate(
                signal, 
                fs, 
                bounds, 
                strategy, 
                rate_hint, 
                None 
            );
            (result.value, result.method)
        },
        _ => panic!("Vital '{}' is not configured as a Rate calculation in registry", vital_id)
    };

    // 3. Assert
    let diff = (calculated_rate - ground_truth_rate).abs();
    
    println!("[{}] {}: Method={}, Calc={:.1}bpm, Ref={:.1}bpm, Diff={:.1}bpm", 
        filename, vital_id, method_name, calculated_rate, ground_truth_rate, diff);

    if diff <= RATE_TOLERANCE_BPM {
        Ok(())
    } else {
        Err(format!(
            "[{}] {} Mismatch. Expected {:.1}, Got {:.1} (Diff {:.1} > {:.1})",
            filename, vital_id, ground_truth_rate, calculated_rate, diff, RATE_TOLERANCE_BPM
        ))
    }
}

// --- Main Test Runner ---

#[test_resources("tests/fixtures/*.json")]
fn test_rate_integrity(resource: &str) {
    let path = Path::new(resource);
    let filename = path.file_name().unwrap().to_str().unwrap();
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);
    
    let ref_data: ReferenceData = serde_json::from_reader(reader)
        .expect("Failed to parse JSON.");
    let fs = ref_data.fs;
    
    let mut failures = Vec::new();

    // 1. Verify Heart Rate
    if let (Some(ppg), Some(hr_ref)) = (&ref_data.vital_signs.ppg_waveform, &ref_data.vital_signs.heart_rate) {
        // Self-referential hint? Ideally for HR we hint with previous HR, but here we test "cold start" 
        // or provide the answer as hint if we want to test "tracking" stability.
        // Let's test "Blind" detection first (None) as it's the hardest case.
        if let Err(e) = verify_rate(
            filename, 
            "heart_rate", 
            fs, 
            &ppg.data, 
            hr_ref.value, 
            None 
        ) {
            failures.push(e);
        }
    }

    // 2. Verify Respiratory Rate
    if let (Some(resp), Some(rr_ref)) = (&ref_data.vital_signs.respiratory_waveform, &ref_data.vital_signs.respiratory_rate) {
        if let Err(e) = verify_rate(
            filename, 
            "respiratory_rate", 
            fs, 
            &resp.data, 
            rr_ref.value, 
            None 
        ) {
            failures.push(e);
        }
    }

    if !failures.is_empty() {
        panic!(
            "\n\nTest failed for file: {:?}\n==================================================\n{}\n", 
            filename,
            failures.join("\n\n")
        );
    }
}