use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use serde::Deserialize;
use test_generator::test_resources;

use vitallens_core::signal::peaks;

// Tolerance in samples for matching a detected peak to a ground truth peak
const STRICT_TOLERANCE: i32 = 1;
const LOOSE_TOLERANCE: i32 = 10; // TODO: Make stricter when we do moving average pre-det

// --- JSON Data Structures ---

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

// --- Verification Logic ---

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
    let min_rate = if name.contains("RESP") { 4.0 } else { 40.0 };
    let max_rate = if name.contains("RESP") { 60.0 } else { 240.0 };

    let opts = peaks::PeakOptions {
        fs,
        bounds: peaks::SignalBounds { min_rate, max_rate },
        threshold,
        refine,
        avg_rate_hint: rate_hint,
        smooth_input: true,
        ..Default::default()
    };

    let segments = peaks::find_peaks(signal, opts);
    
    let detected_indices: Vec<usize> = segments.iter()
        .flat_map(|seg| seg.iter().map(|p| p.x.round() as usize))
        .collect();

    // 1. Identify Missed Peaks (False Negatives)
    let mut missed_peaks = Vec::new();
    for &gt_idx in ground_truth {
        let is_found = detected_indices.iter().any(|&det_idx| {
            (det_idx as i32 - gt_idx as i32).abs() <= tolerance
        });
        if !is_found {
            missed_peaks.push(gt_idx);
        }
    }

    // 2. Identify Extra Peaks (False Positives)
    let mut false_positives = Vec::new();
    for &det_idx in &detected_indices {
        let is_valid = ground_truth.iter().any(|&gt_idx| {
            (det_idx as i32 - gt_idx as i32).abs() <= tolerance
        });
        if !is_valid {
            false_positives.push(det_idx);
        }
    }

    // 3. Determine Pass/Fail Criteria
    let (allowed_misses, allowed_extras) = if rate_hint.is_some() {
        (0, 0)
    } else {
        (1, 1)
    };
    
    let passed = missed_peaks.len() <= allowed_misses && false_positives.len() <= allowed_extras;

    // --- TEMPORARY LOGGING ---
    println!("\n=== CHECK: {} ===", name);
    println!("  -> Rate Hint: {:?}", rate_hint);
    println!("  -> Threshold: {:.1}, Refine: {}", threshold, refine);
    println!("  -> Ground Truth ({}): {:?}", ground_truth.len(), ground_truth);
    println!("  -> Detected ({}):     {:?}", detected_indices.len(), detected_indices);
    
    if !missed_peaks.is_empty() {
        println!("  [!] Missed (FN): {:?}", missed_peaks);
    }
    if !false_positives.is_empty() {
        println!("  [!] Extras (FP): {:?}", false_positives);
    }

    if passed {
        println!("  [OK] PASSED (Missed: {}/{}, Extra: {}/{})", 
            missed_peaks.len(), allowed_misses, false_positives.len(), allowed_extras);
        Ok(())
    } else {
        println!("  [X] FAILED");
        Err(format!(
            "[{}] FAILED (Hint: {:?}).\n    Missed (FN): {:?} (Allowed: {})\n    Extra (FP): {:?} (Allowed: {})\n    Full Detected: {:?}", 
            name, rate_hint, missed_peaks, allowed_misses, false_positives, allowed_extras, detected_indices
        ))
    }
}

// --- Test Runner ---

#[test_resources("tests/fixtures/*.json")]
fn test_data_integrity(resource: &str) {
    let path = Path::new(resource);
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);
    
    let ref_data: ReferenceData = serde_json::from_reader(reader)
        .expect("Failed to parse JSON. Ensure format matches ReferenceData struct.");
    let fs = ref_data.fs;
    
    // Accumulate failures so we can see all issues in a file at once
    let mut failures = Vec::new();

    // 1. Test PPG
    if let Some(ppg) = ref_data.vital_signs.ppg_waveform {
        if let Some(ground_truth) = ppg.peak_indices {
            // Case A: Blind (No Rate Hint) -> Allows 1 missed peak
            if let Err(e) = verify_peaks(
                "PPG-Blind", fs, &ppg.data, &ground_truth, 
                STRICT_TOLERANCE, true, 0.5, None 
            ) {
                failures.push(e);
            }

            // Case B: Hinted (Known Rate) -> Strict (0 missed peaks)
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

    // 2. Test Respiration
    if let Some(resp) = ref_data.vital_signs.respiratory_waveform {
        if let Some(ground_truth) = resp.peak_indices {
            // Case A: Blind (No Rate Hint) -> Allows 1 missed peak
            if let Err(e) = verify_peaks(
                "RESP-Blind", fs, &resp.data, &ground_truth, 
                LOOSE_TOLERANCE, false, 1.5, None 
            ) {
                failures.push(e);
            }

            // Case B: Hinted (Known Rate) -> Strict (0 missed peaks)
            if let Some(rr) = ref_data.vital_signs.respiratory_rate {
                if let Err(e) = verify_peaks(
                    "RESP-Hinted", fs, &resp.data, &ground_truth, 
                    LOOSE_TOLERANCE, false, 1.5, Some(rr.value)
                ) {
                    failures.push(e);
                }
            }
        }
    }

    // Report results
    if !failures.is_empty() {
        panic!(
            "\n\nTest failed for file: {:?}\n==================================================\n{}\n", 
            path.file_name().unwrap(),
            failures.join("\n\n")
        );
    }
}