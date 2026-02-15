use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use serde::Deserialize;
use test_generator::test_resources;

// Import from the library
use vitallens_core::signal::peaks;

// --- 1. CONFIGURATION ---

const STRICT_TOLERANCE: i32 = 1;
const LOOSE_TOLERANCE: i32 = 10;

// --- 2. JSON STRUCTS ---

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
    peak_indices: Option<Vec<usize>>, 
}

#[derive(Deserialize, Debug)]
struct ScalarResult {
    value: f32,
    #[allow(dead_code)]
    confidence: f32,
}

// --- 3. VERIFICATION LOGIC ---

#[allow(clippy::too_many_arguments)]
fn verify_peaks(
    name: &str,
    fs: f32,
    signal: &[f32], 
    ground_truth: &[usize], 
    tolerance: i32,
    refine: bool,
    threshold: f32,
    rate_hint: Option<f32>
) -> Result<(), String> {
    let opts = peaks::PeakOptions {
        fs,
        bounds: peaks::SignalBounds { min_rate: 4.0, max_rate: 220.0 },
        threshold,
        refine,
        avg_rate_hint: rate_hint,
        ..Default::default()
    };

    let segments = peaks::find_peaks(signal, opts);
    
    let detected_indices: Vec<usize> = segments.iter()
        .flat_map(|seg| seg.iter().map(|p| p.x.round() as usize))
        .collect();

    // 1. Check for Missed Peaks (False Negatives)
    let mut missed_peaks = Vec::new();
    for &gt_idx in ground_truth {
        let is_found = detected_indices.iter().any(|&det_idx| {
            (det_idx as i32 - gt_idx as i32).abs() <= tolerance
        });
        if !is_found {
            missed_peaks.push(gt_idx);
        }
    }

    // 2. Check for Extra Peaks (False Positives)
    let mut false_positives = Vec::new();
    for &det_idx in &detected_indices {
        let is_valid = ground_truth.iter().any(|&gt_idx| {
            (det_idx as i32 - gt_idx as i32).abs() <= tolerance
        });
        if !is_valid {
            false_positives.push(det_idx);
        }
    }

    // 3. Return Error String instead of Panicking
    if !missed_peaks.is_empty() || !false_positives.is_empty() {
        return Err(format!(
            "[{}] FAILED (Hint: {:?}).\n    Missed (FN): {:?}\n    Extra (FP): {:?}\n    Full Detected: {:?}", 
            name, rate_hint, missed_peaks, false_positives, detected_indices
        ));
    }

    Ok(())
}

// --- 4. THE GENERATED TEST ---

#[test_resources("tests/fixtures/*.json")]
fn test_data_integrity(resource: &str) {
    let path = Path::new(resource);
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);
    
    let ref_data: ReferenceData = serde_json::from_reader(reader)
        .expect("Failed to parse JSON. Ensure format matches ReferenceData struct.");
    let fs = ref_data.fs;
    
    // COLLECT ALL FAILURES
    let mut failures = Vec::new();

    // --- TEST PPG ---
    if let Some(ppg) = ref_data.vital_signs.ppg_waveform {
        if let Some(ground_truth) = ppg.peak_indices {
            // Case A: Blind
            if let Err(e) = verify_peaks(
                "PPG-Blind", fs, &ppg.data, &ground_truth, 
                STRICT_TOLERANCE, true, 0.5, None 
            ) {
                failures.push(e);
            }

            // Case B: Hinted
            if let Some(hr) = ref_data.vital_signs.heart_rate {
                if let Err(e) = verify_peaks(
                    "PPG-Hinted", fs, &ppg.data, &ground_truth, 
                    STRICT_TOLERANCE, true, 0.5, Some(hr.value)
                ) {
                    failures.push(e);
                }
            }
        }
    }

    // --- TEST RESP ---
    if let Some(resp) = ref_data.vital_signs.respiratory_waveform {
        if let Some(ground_truth) = resp.peak_indices {
            // Case A: Blind
            if let Err(e) = verify_peaks(
                "RESP-Blind", fs, &resp.data, &ground_truth, 
                LOOSE_TOLERANCE, false, 1.0, None 
            ) {
                failures.push(e);
            }

            // Case B: Hinted
            if let Some(rr) = ref_data.vital_signs.respiratory_rate {
                if let Err(e) = verify_peaks(
                    "RESP-Hinted", fs, &resp.data, &ground_truth, 
                    LOOSE_TOLERANCE, false, 1.0, Some(rr.value)
                ) {
                    failures.push(e);
                }
            }
        }
    }

    // REPORT ALL FAILURES AT ONCE
    if !failures.is_empty() {
        panic!(
            "\n\nTest failed for file: {:?}\n==================================================\n{}\n", 
            path.file_name().unwrap(),
            failures.join("\n\n")
        );
    }
}