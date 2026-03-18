use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::collections::HashMap;
use serde::Deserialize;
use test_generator::test_resources;

use vitallens_core::signal::rate;
use vitallens_core::registry;

const RATE_TOLERANCE_BPM: f32 = 3.0;

#[derive(Deserialize, Debug)]
struct ReferenceData {
    #[serde(default)]
    waveforms: HashMap<String, Waveform>,
    #[serde(default)]
    vitals: HashMap<String, Vital>,
    fps: f32, 
}

#[derive(Deserialize, Debug)]
struct Waveform {
    data: Vec<f32>,
    #[allow(dead_code)]
    peak_indices: Option<Vec<usize>>, 
}

#[derive(Deserialize, Debug)]
struct Vital {
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
    
    let meta = registry::get_vital_meta(vital_id)
        .unwrap_or_else(|| panic!("Vital ID '{}' not found in registry", vital_id));
    
    let deriv = &meta.derivations[0];

    let duration_sec = signal.len() as f32 / fs;
    if duration_sec < deriv.min_window_seconds {
        println!("[{}] {} SKIPPED: Duration {:.1}s < Min Window {:.1}s", 
            filename, vital_id, duration_sec, deriv.min_window_seconds);
        return Ok(());
    }

    let (calculated_rate, calculated_conf, method_name) = match deriv.method {
        registry::CalculationMethod::Rate(strategy) => {
            let bounds = rate::RateBounds { 
                min: deriv.min_value, 
                max: deriv.max_value 
            };
            
            let result = rate::estimate_rate(
                signal, 
                fs,
                None,
                bounds, 
                strategy, 
                rate_hint, 
                None
            );
            (result.value, result.confidence, result.method)
        },
        _ => panic!("Vital '{}' is not configured as a Rate calculation in registry", vital_id)
    };

    if calculated_conf < 0.0 || calculated_conf > 1.0 {
        return Err(format!(
            "[{}] {} Confidence Out of Bounds. Got {:.4}, Expected [0.0, 1.0]",
            filename, vital_id, calculated_conf
        ));
    }

    let diff = (calculated_rate - ground_truth_rate).abs();
    
    println!("[{}] {}: Method={}, Calc={:.1}bpm (Conf {:.2}), Ref={:.1}bpm, Diff={:.1}bpm", 
        filename, vital_id, method_name, calculated_rate, calculated_conf, ground_truth_rate, diff);

    if diff <= RATE_TOLERANCE_BPM {
        Ok(())
    } else {
        Err(format!(
            "[{}] {} Mismatch. Expected {:.1}, Got {:.1} (Diff {:.1} > {:.1})",
            filename, vital_id, ground_truth_rate, calculated_rate, diff, RATE_TOLERANCE_BPM
        ))
    }
}

#[test_resources("tests/fixtures/*.json")]
fn test_rate_integrity(resource: &str) {
    let path = Path::new(resource);
    let filename = path.file_name().unwrap().to_str().unwrap();
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);
    
    let ref_data: ReferenceData = serde_json::from_reader(reader)
        .expect("Failed to parse JSON.");
    let fs = ref_data.fps;
    
    let mut failures = Vec::new();

    if let (Some(ppg), Some(hr_ref)) = (ref_data.waveforms.get("ppg_waveform"), ref_data.vitals.get("heart_rate")) {
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

    if let (Some(resp), Some(rr_ref)) = (ref_data.waveforms.get("respiratory_waveform"), ref_data.vitals.get("respiratory_rate")) {
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