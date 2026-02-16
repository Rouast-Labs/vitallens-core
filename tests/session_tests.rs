use std::fs::File;
use std::io::BufReader;
use std::collections::HashMap;
use std::path::Path;
use serde::Deserialize;
use test_generator::test_resources;

use vitallens_core::{Session, ModelConfig, InputChunk, WaveformMode};
use vitallens_core::registry;

// --- Tolerances ---
const TOLERANCE_HR_BPM: f32 = 3.0;
const TOLERANCE_RR_BPM: f32 = 3.0;
const TOLERANCE_SDNN_MS: f32 = 10.0;
const TOLERANCE_RMSSD_MS: f32 = 10.0;
const TOLERANCE_LFHF: f32 = 1.0;

// Consistency tolerance between Incremental and Windowed modes
const CONSISTENCY_TOLERANCE: f32 = 0.5;

// --- Data Structures ---

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
    hrv_sdnn: Option<ScalarResult>,
    hrv_rmssd: Option<ScalarResult>,
    hrv_lfhf: Option<ScalarResult>,
}

#[derive(Deserialize, Debug)]
struct Waveform {
    data: Vec<f32>,
    #[allow(dead_code)]
    confidence: Option<Vec<f32>>,
}

#[derive(Deserialize, Debug)]
struct ScalarResult {
    value: f32,
}

struct TestCase {
    vital_id: String,
    ground_truth: f32,
    tolerance: f32,
    input_signal_key: String,
    input_data: Vec<f32>,
}

// --- Main Test Logic ---

#[test_resources("tests/fixtures/*.json")]
fn test_session_end_to_end(resource: &str) {
    let path = Path::new(resource);
    let filename = path.file_name().unwrap().to_str().unwrap();
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);
    let ref_data: ReferenceData = serde_json::from_reader(reader).expect("Failed to parse JSON");

    println!("\n=== SESSION E2E: {} ===", filename);

    // 1. Setup Test Cases
    let mut cases = Vec::new();
    let duration_sec = if let Some(ppg) = &ref_data.vital_signs.ppg_waveform {
        ppg.data.len() as f32 / ref_data.fs
    } else {
        0.0
    };

    println!("  -> Duration: {:.2}s, Fs: {:.1}Hz", duration_sec, ref_data.fs);

    let mut add_case = |id: &str, gt: Option<&ScalarResult>, signal: Option<&Waveform>, key: &str, tol: f32| {
        if let (Some(g), Some(s)) = (gt, signal) {
            let meta = registry::get_vital_meta(id).unwrap();
            let min_win = meta.derivations[0].min_window_seconds;
            if duration_sec >= min_win {
                cases.push(TestCase {
                    vital_id: id.to_string(),
                    ground_truth: g.value,
                    tolerance: tol,
                    input_signal_key: key.to_string(),
                    input_data: s.data.clone(),
                });
            }
        }
    };

    add_case("heart_rate", ref_data.vital_signs.heart_rate.as_ref(), ref_data.vital_signs.ppg_waveform.as_ref(), "ppg_waveform", TOLERANCE_HR_BPM);
    add_case("respiratory_rate", ref_data.vital_signs.respiratory_rate.as_ref(), ref_data.vital_signs.respiratory_waveform.as_ref(), "respiratory_waveform", TOLERANCE_RR_BPM);
    add_case("hrv_sdnn", ref_data.vital_signs.hrv_sdnn.as_ref(), ref_data.vital_signs.ppg_waveform.as_ref(), "ppg_waveform", TOLERANCE_SDNN_MS);
    add_case("hrv_rmssd", ref_data.vital_signs.hrv_rmssd.as_ref(), ref_data.vital_signs.ppg_waveform.as_ref(), "ppg_waveform", TOLERANCE_RMSSD_MS);
    add_case("hrv_lfhf", ref_data.vital_signs.hrv_lfhf.as_ref(), ref_data.vital_signs.ppg_waveform.as_ref(), "ppg_waveform", TOLERANCE_LFHF);

    if cases.is_empty() {
        println!("  [SKIP] No valid vitals found in file.");
        return;
    }
    
    let active_vitals: Vec<&String> = cases.iter().map(|c| &c.vital_id).collect();
    println!("  -> Active Vitals: {:?}", active_vitals);

    // 2. Run All Modes
    // Global
    let global_results = run_session_extraction(filename, &ref_data, &cases, WaveformMode::Global);
    // Incremental (15 frame chunks)
    let inc_results = run_session_extraction(filename, &ref_data, &cases, WaveformMode::Incremental);
    // Windowed (30 frame chunks)
    let win_results = run_session_extraction(filename, &ref_data, &cases, WaveformMode::Windowed { seconds: 30.0 });

    let mut failures = Vec::new();

    // 3. Assertions
    println!("\n  --- ASSERTIONS [{}] ---", filename);

    for case in &cases {
        let vid = &case.vital_id;
        
        // Check A: Global vs Ground Truth
        if let Some(val) = global_results.get(vid) {
            let diff = (val - case.ground_truth).abs();
            if diff <= case.tolerance {
                println!("  [OK] Global {}: {:.2} (Ref {:.2}, Diff {:.2})", vid, val, case.ground_truth, diff);
            } else {
                let msg = format!("Global {} mismatch: {:.2} vs Ref {:.2} (Diff {:.2} > {:.1})", 
                    vid, val, case.ground_truth, diff, case.tolerance);
                println!("  [X] {}", msg);
                failures.push(msg);
            }
        } else {
            // It is possible Global failed if the file length is exactly at the boundary 
            println!("  [WARN] Global {} produced no result", vid);
        }

        // Check B: Incremental vs Windowed (Consistency Check)
        match (inc_results.get(vid), win_results.get(vid)) {
            (Some(inc), Some(win)) => {
                let diff = (inc - win).abs();
                if diff <= CONSISTENCY_TOLERANCE {
                    println!("  [OK] Consistency {}: Inc {:.2} vs Win {:.2} (Diff {:.2})", vid, inc, win, diff);
                } else {
                    let msg = format!("Consistency {} mismatch: Inc {:.2} vs Win {:.2} (Diff {:.2} > {:.1})", 
                        vid, inc, win, diff, CONSISTENCY_TOLERANCE);
                    println!("  [X] {}", msg);
                    failures.push(msg);
                }
            },
            (None, None) => println!("  [WARN] Neither streaming mode produced result for {}", vid),
            _ => {
                 let msg = format!("Streaming mismatch: One mode produced result for {}, one did not", vid);
                 println!("  [X] {}", msg);
                 failures.push(msg);
            }
        }
    }

    if !failures.is_empty() {
        panic!("Session E2E Failed:\n{}", failures.join("\n"));
    }
}

// Helper: Just runs the session and returns the final values
fn run_session_extraction(
    filename: &str, 
    ref_data: &ReferenceData, 
    cases: &[TestCase], 
    mode: WaveformMode
) -> HashMap<String, f32> {
    
    // Config
    let supported_vitals: Vec<String> = cases.iter().map(|c| c.vital_id.clone()).collect();
    let config = ModelConfig {
        name: format!("{}_{:?}", filename, mode),
        supported_vitals,
        fps_target: ref_data.fs,
        input_size: 30,
        roi_method: "face".to_string(),
    };
    
    let session = Session::new(config);

    // Pre-calculate Global State
    let max_len = cases.iter().map(|c| c.input_data.len()).max().unwrap_or(0);
    let global_timestamps: Vec<f64> = (0..max_len).map(|t| {
        let base = t as f64 / ref_data.fs as f64;
        let jitter = if t % 2 == 0 { 0.001 } else { -0.001 };
        base + jitter
    }).collect();

    // Streaming Logic
    let (chunk_size, step_size) = match mode {
        WaveformMode::Global => (max_len, max_len),
        WaveformMode::Incremental => (15, 12), // Small chunk (0.5s), 3 frame overlap
        WaveformMode::Windowed { .. } => (30, 25), // Standard chunk (1.0s), 5 frame overlap
    };
    
    let mut final_results: HashMap<String, f32> = HashMap::new();
    let mut start = 0;

    // Log Start
    println!("  -> Running Mode: {:?} (Chunk {}, Step {})", mode, chunk_size, step_size);

    while start < max_len {
        let end = (start + chunk_size).min(max_len);
        
        let ts_slice = &global_timestamps[start..end];
        let mut signal_map = HashMap::new();
        let mut confidence_map = HashMap::new();

        for case in cases {
            if start < case.input_data.len() {
                let case_end = (start + chunk_size).min(case.input_data.len());
                if case_end > start {
                    let slice = &case.input_data[start..case_end];
                    signal_map.insert(case.input_signal_key.clone(), slice.to_vec());
                    confidence_map.insert(case.input_signal_key.clone(), vec![1.0; slice.len()]);
                }
            }
        }

        let chunk = InputChunk {
            timestamp: ts_slice.to_vec(),
            signals: signal_map,
            confidences: confidence_map,
            face: None,
        };

        let result = session.process_chunk(chunk, mode.clone());

        for (key, val) in result.signals {
            if let Some(v) = val.value {
                final_results.insert(key, v);
            }
        }

        if end == max_len { break; }
        start += step_size;
    }

    final_results
}