use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use serde::Deserialize;
use test_generator::test_resources;

use vitallens_core::signal::peaks;
use vitallens_core::registry;

const MATCHING_TOLERANCE_PPG: i32 = 1;
const MATCHING_TOLERANCE_RESP: i32 = 3;

#[derive(Deserialize, Debug)]
struct ReferenceData {
    vital_signs: Vitals,
    fps: f32, 
}

#[derive(Deserialize, Debug)]
struct Vitals {
    ppg_waveform: Option<Waveform>,
    respiratory_waveform: Option<Waveform>,     
    heart_rate: Option<Vital>,
    respiratory_rate: Option<Vital>,
}

#[derive(Deserialize, Debug)]
struct Waveform {
    data: Vec<f32>,
    peak_indices: Option<Vec<usize>>, 
}

#[derive(Deserialize, Debug)]
struct Vital {
    value: f32,
}

#[allow(clippy::too_many_arguments)]
fn verify_peaks(
    filename: &str,
    name: &str,
    fs: f32,
    signal: &[f32], 
    ground_truth: &[usize], 
    matching_tolerance: i32,
    refine: bool,
    threshold: f32,
    window_cycles: f32,
    max_rate_change_per_sec: f32,
    rate_hint: Option<f32>
) -> Result<(), String> {
    
    let vital_id = if name.contains("PPG") {
        "heart_rate"
    } else {
        "respiratory_rate"
    };

    let meta = registry::get_vital_meta(vital_id)
        .unwrap_or_else(|| panic!("Vital ID '{}' not found in registry", vital_id));
    
    let deriv = meta.derivations.first().expect("Vital has no derivations");
    let (min_rate, max_rate) = (deriv.min_value, deriv.max_value);

    let opts = peaks::PeakOptions {
        fs,
        bounds: peaks::SignalBounds { min_rate, max_rate },
        threshold,
        window_cycles,
        refine,
        avg_rate_hint: rate_hint,
        max_rate_change_per_sec,
        smooth_input: true,
        ..Default::default()
    };

    let segments = peaks::find_peaks(signal, opts);
    
    let detected_indices: Vec<usize> = segments.iter()
        .flat_map(|seg| seg.iter().map(|p| p.x.round() as usize))
        .collect();

    let mut missed_peaks = Vec::new();
    for &gt_idx in ground_truth {
        let is_found = detected_indices.iter().any(|&det_idx| {
            (det_idx as i32 - gt_idx as i32).abs() <= matching_tolerance
        });
        if !is_found {
            missed_peaks.push(gt_idx);
        }
    }

    let mut false_positives = Vec::new();
    for &det_idx in &detected_indices {
        let is_valid = ground_truth.iter().any(|&gt_idx| {
            (det_idx as i32 - gt_idx as i32).abs() <= matching_tolerance
        });
        if !is_valid {
            false_positives.push(det_idx);
        }
    }

    let duration_mins = signal.len() as f32 / fs / 60.0;
    
    let (missed_rate, extra_rate) = if name.contains("PPG") {
        if rate_hint.is_some() { (1.0, 0.0) } else { (2.0, 4.0) } 
    } else {
        if rate_hint.is_some() { (1.0, 0.0) } else { (3.0, 10.0) }
    };

    let allowed_misses = (missed_rate * duration_mins).round() as usize;
    let allowed_extras = (extra_rate * duration_mins).round() as usize;

    let passed = missed_peaks.len() <= allowed_misses && false_positives.len() <= allowed_extras;

    println!("\n=== CHECK: [{}] {} ===", filename, name);
    println!("  -> Settings: Range [{:.1}-{:.1}] BPM, Hint: {:?}", min_rate, max_rate, rate_hint);
    println!("  -> Duration: {:.2}s ({:.3} mins)", signal.len() as f32 / fs, duration_mins);
    println!("  -> Limits:   Max Missed: {} (Rate {:.1}), Max Extra: {} (Rate {:.1})", 
             allowed_misses, missed_rate, allowed_extras, extra_rate);
    println!("  -> Ground Truth ({}): {:?}", ground_truth.len(), ground_truth);
    println!("  -> Detected ({}):     {:?}", detected_indices.len(), detected_indices);
    
    if !missed_peaks.is_empty() {
        println!("  [!] Missed (FN): {:?}", missed_peaks);
    }
    if !false_positives.is_empty() {
        println!("  [!] Extras (FP): {:?}", false_positives);
    }

    if passed {
        println!("  [OK] PASSED");
        Ok(())
    } else {
        println!("  [X] FAILED");
        Err(format!(
            "[{}] FAILED.\n    Missed: {} (Allowed: {})\n    Extra: {} (Allowed: {})", 
            name, missed_peaks.len(), allowed_misses, false_positives.len(), allowed_extras
        ))
    }
}

#[test_resources("tests/fixtures/*.json")]
fn test_data_integrity(resource: &str) {
    let path = Path::new(resource);
    let filename = path.file_name().unwrap().to_str().unwrap();
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);
    
    let ref_data: ReferenceData = serde_json::from_reader(reader)
        .expect("Failed to parse JSON. Ensure format matches ReferenceData struct.");
    let fs = ref_data.fps;
    
    let mut failures = Vec::new();

    if let Some(ppg) = ref_data.vital_signs.ppg_waveform {
        if let Some(ground_truth) = ppg.peak_indices {
            if let Err(e) = verify_peaks(
                filename, "PPG-Blind", fs, &ppg.data, &ground_truth, 
                MATCHING_TOLERANCE_PPG, true, 0.5, 2.5, 1.0, None 
            ) {
                failures.push(e);
            }

            if let Some(hr) = ref_data.vital_signs.heart_rate {
                if let Err(e) = verify_peaks(
                    filename, "PPG-Hinted", fs, &ppg.data, &ground_truth, 
                    MATCHING_TOLERANCE_PPG, true, 0.5, 2.5, 1.0, Some(hr.value)
                ) {
                    failures.push(e);
                }
            }
        }
    }

    if let Some(resp) = ref_data.vital_signs.respiratory_waveform {
        if let Some(ground_truth) = resp.peak_indices {
            if let Err(e) = verify_peaks(
                filename, "RESP-Blind", fs, &resp.data, &ground_truth, 
                MATCHING_TOLERANCE_RESP, false, 1.2, 1.5, 0.25, None 
            ) {
                failures.push(e);
            }

            if let Some(rr) = ref_data.vital_signs.respiratory_rate {
                if let Err(e) = verify_peaks(
                    filename, "RESP-Hinted", fs, &resp.data, &ground_truth, 
                    MATCHING_TOLERANCE_RESP, false, 1.2, 1.5, 0.25, Some(rr.value)
                ) {
                    failures.push(e);
                }
            }
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