use std::fs::File;
use std::io::BufReader;
use std::collections::HashMap;
use std::path::Path;
use serde::Deserialize;
use test_generator::test_resources;

use vitallens_core::{Session, ModelConfig, InputChunk, WaveformMode};
use vitallens_core::registry;
use vitallens_core::types::FaceInput;

const TOLERANCE_HR_BPM: f32 = 3.0;
const TOLERANCE_RR_BPM: f32 = 3.0;
const TOLERANCE_SDNN_MS: f32 = 10.0;
const TOLERANCE_RMSSD_MS: f32 = 10.0;
const TOLERANCE_IE_RATIO: f32 = 0.15;
const TOLERANCE_PNN50: f32 = 5.0;
const TOLERANCE_LFHF: f32 = 0.5;
const TOLERANCE_SD1SD2: f32 = 0.05;
const TOLERANCE_STRESS_INDEX: f32 = 20.0;

const CONSISTENCY_TOLERANCE: f32 = 0.5;

#[derive(Deserialize, Debug)]
struct FaceRef {
    coordinates: Vec<Vec<f32>>,
    confidence: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct ReferenceData {
    vital_signs: Vitals,
    face: Option<FaceRef>,
    fps: f32, 
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
    hrv_pnn50: Option<ScalarResult>,
    hrv_sd1sd2: Option<ScalarResult>,
    ie_ratio: Option<ScalarResult>,
    stress_index: Option<ScalarResult>,
}

#[derive(Deserialize, Debug)]
struct Waveform {
    data: Vec<f32>,
    confidence: Option<Vec<f32>>,
}

#[derive(Deserialize, Debug)]
struct ScalarResult {
    value: f32,
    confidence: f32,
}

struct TestCase {
    vital_id: String,
    ground_truth: f32,
    gt_confidence: f32,
    tolerance: f32,
    input_signal_key: String,
    input_data: Vec<f32>,
    input_confidence: Vec<f32>,
}

#[test_resources("tests/fixtures/*.json")]
fn test_session(resource: &str) {
    let path = Path::new(resource);
    let filename = path.file_name().unwrap().to_str().unwrap();
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);
    let ref_data: ReferenceData = serde_json::from_reader(reader).expect("Failed to parse JSON");

    println!("\n=== SESSION E2E: {} ===", filename);

    let mut cases = Vec::new();
    let duration_sec = if let Some(ppg) = &ref_data.vital_signs.ppg_waveform {
        ppg.data.len() as f32 / ref_data.fps
    } else {
        0.0
    };

    println!(" -> Duration: {:.2}s, Fs: {:.1}Hz", duration_sec, ref_data.fps);

    let mut add_case = |id: &str, gt: Option<&ScalarResult>, signal: Option<&Waveform>, key: &str, tol: f32| {
        if let (Some(g), Some(s)) = (gt, signal) {
            let meta = registry::get_vital_meta(id).unwrap();
            let min_win = meta.derivations[0].min_window_seconds;
            if duration_sec >= min_win {
                let real_conf = s.confidence.clone().unwrap_or_else(|| vec![1.0; s.data.len()]);

                cases.push(TestCase {
                    vital_id: id.to_string(),
                    ground_truth: g.value,
                    gt_confidence: g.confidence,
                    tolerance: tol,
                    input_signal_key: key.to_string(),
                    input_data: s.data.clone(),
                    input_confidence: real_conf,
                });
            }
        }
    };

    add_case("heart_rate", ref_data.vital_signs.heart_rate.as_ref(), ref_data.vital_signs.ppg_waveform.as_ref(), "ppg_waveform", TOLERANCE_HR_BPM);
    add_case("respiratory_rate", ref_data.vital_signs.respiratory_rate.as_ref(), ref_data.vital_signs.respiratory_waveform.as_ref(), "respiratory_waveform", TOLERANCE_RR_BPM);
    add_case("hrv_sdnn", ref_data.vital_signs.hrv_sdnn.as_ref(), ref_data.vital_signs.ppg_waveform.as_ref(), "ppg_waveform", TOLERANCE_SDNN_MS);
    add_case("hrv_rmssd", ref_data.vital_signs.hrv_rmssd.as_ref(), ref_data.vital_signs.ppg_waveform.as_ref(), "ppg_waveform", TOLERANCE_RMSSD_MS);
    add_case("hrv_lfhf", ref_data.vital_signs.hrv_lfhf.as_ref(), ref_data.vital_signs.ppg_waveform.as_ref(), "ppg_waveform", TOLERANCE_LFHF);
    add_case("hrv_pnn50", ref_data.vital_signs.hrv_pnn50.as_ref(), ref_data.vital_signs.ppg_waveform.as_ref(), "ppg_waveform", TOLERANCE_PNN50);
    add_case("hrv_sd1sd2", ref_data.vital_signs.hrv_sd1sd2.as_ref(), ref_data.vital_signs.ppg_waveform.as_ref(), "ppg_waveform", TOLERANCE_SD1SD2);
    add_case("ie_ratio", ref_data.vital_signs.ie_ratio.as_ref(), ref_data.vital_signs.respiratory_waveform.as_ref(), "respiratory_waveform", TOLERANCE_IE_RATIO);
    add_case("stress_index", ref_data.vital_signs.stress_index.as_ref(), ref_data.vital_signs.ppg_waveform.as_ref(), "ppg_waveform", TOLERANCE_STRESS_INDEX);

    if cases.is_empty() {
        println!(" [SKIP] No valid vitals found in file.");
        return;
    }
    
    let active_vitals: Vec<&String> = cases.iter().map(|c| &c.vital_id).collect();
    println!(" -> Active Vitals: {:?}", active_vitals);

    let (global_results, global_waves) = run_session_extraction(filename, &ref_data, &cases, WaveformMode::Global);
    let (inc_results, inc_waves) = run_session_extraction(filename, &ref_data, &cases, WaveformMode::Incremental);
    let (win_results, win_waves) = run_session_extraction(filename, &ref_data, &cases, WaveformMode::Windowed { seconds: 30.0 });

    let mut failures = Vec::new();

    println!("\n --- ASSERTIONS [{}] ---", filename);

    for case in &cases {
        let vid = &case.vital_id;

        let wave_key = &case.input_signal_key;

        // --- GLOBAL Length Checks ---
        if let Some((data, conf)) = global_waves.get(wave_key) {
            let expected = case.input_data.len();
            assert_eq!(data.len(), expected, "GLOBAL Waveform {} data length mismatch", wave_key);
            assert_eq!(conf.len(), expected, "GLOBAL Waveform {} confidence length mismatch", wave_key);
        }

        // --- INCREMENTAL Length Checks ---
        if let Some((data, conf)) = inc_waves.get(wave_key) {
            assert_eq!(data.len(), case.input_data.len(), "INCREMENTAL total entries mismatch for {}", wave_key);
            assert_eq!(data.len(), conf.len(), "INCREMENTAL {} data/conf length mismatch", wave_key);
        }

        // --- WINDOWED Length Checks ---
        if let Some((data, conf)) = win_waves.get(wave_key) {
            let window_frames = (30.0 * ref_data.fps) as usize;
            let expected_len = window_frames.min(case.input_data.len());
            
            assert!(data.len() >= expected_len, "WINDOWED entries for {} insufficient", wave_key);
            assert_eq!(data.len(), conf.len(), "WINDOWED {} data/conf length mismatch", wave_key);
        }
        
        if let Some((val, conf)) = global_results.get(vid) {
            let val_diff = (val - case.ground_truth).abs();
            let conf_diff = (conf - case.gt_confidence).abs();
            
            if val_diff <= case.tolerance {
                println!(" [OK] Global {}: Val {:.2} (Ref {:.2})", vid, val, case.ground_truth);
            } else {
                let msg = format!("Global {} Value Mismatch: {:.2} vs Ref {:.2} (Diff {:.2})", 
                    vid, val, case.ground_truth, val_diff);
                println!(" [X] {}", msg);
                failures.push(msg);
            }

            if conf_diff <= 0.01 { 
                 println!(" [OK] Global {} Conf: {:.2} (Ref {:.2})", vid, conf, case.gt_confidence);
            } else {
                 let msg = format!("Global {} Conf Mismatch: {:.2} vs Ref {:.2} (Diff {:.2})", 
                    vid, conf, case.gt_confidence, conf_diff);
                 println!(" [X] {}", msg);
                 failures.push(msg);
            }

        } else {
            println!(" [WARN] Global {} produced no result", vid);
        }

        match (inc_results.get(vid), win_results.get(vid)) {
            (Some((inc_val, inc_conf)), Some((win_val, win_conf))) => {
                let val_diff = (inc_val - win_val).abs();
                let conf_diff = (inc_conf - win_conf).abs();

                if val_diff <= CONSISTENCY_TOLERANCE && conf_diff <= CONSISTENCY_TOLERANCE {
                    println!(" [OK] Consistency {}: Match (Val Diff {:.2}, Conf Diff {:.2})", vid, val_diff, conf_diff);
                } else {
                    let msg = format!("Consistency {} mismatch: ValDiff {:.2}, ConfDiff {:.2}", 
                        vid, val_diff, conf_diff);
                    println!(" [X] {}", msg);
                    failures.push(msg);
                }
            },
            (None, None) => println!(" [WARN] Neither streaming mode produced result for {}", vid),
            _ => {
                 let msg = format!("Streaming mismatch: One mode produced result for {}, one did not", vid);
                 println!(" [X] {}", msg);
                 failures.push(msg);
            }
        }
    }

    if !failures.is_empty() {
        panic!("Session E2E Failed:\n{}", failures.join("\n"));
    }
}

fn run_session_extraction(
    filename: &str, 
    ref_data: &ReferenceData, 
    cases: &[TestCase], 
    mode: WaveformMode
) -> (HashMap<String, (f32, f32)>, HashMap<String, (Vec<f32>, Vec<f32>)>) {
    
    let supported_vitals: Vec<String> = cases.iter().map(|c| c.vital_id.clone()).collect();
    let config = ModelConfig {
        name: format!("{}_{:?}", filename, mode),
        supported_vitals,
        fps_target: ref_data.fps,
        input_size: 30,
        roi_method: "face".to_string(),
    };
    
    let session = Session::new(config);

    let max_len = cases.iter().map(|c| c.input_data.len()).max().unwrap_or(0);
    let global_timestamps: Vec<f64> = (0..max_len).map(|t| {
        let base = t as f64 / ref_data.fps as f64;
        let jitter = if t % 2 == 0 { 0.001 } else { -0.001 };
        base + jitter
    }).collect();

    let (chunk_size, step_size) = match mode {
        WaveformMode::Global => (max_len, max_len),
        WaveformMode::Incremental => (15, 12), 
        WaveformMode::Windowed { .. } => (30, 25), 
    };
    
    let mut final_results: HashMap<String, (f32, f32)> = HashMap::new();
    let mut waveform_results: HashMap<String, (Vec<f32>, Vec<f32>)> = HashMap::new();
    let mut start = 0;

    println!(" -> Running Mode: {:?} (Chunk {}, Step {})", mode, chunk_size, step_size);

    while start < max_len {
        let end = (start + chunk_size).min(max_len);
        
        let unique_frames_sent = if start == 0 { end } else { end - (start + chunk_size - step_size) };
        
        let ts_slice = &global_timestamps[start..end];
        let mut signal_map = HashMap::new();
        let mut confidence_map = HashMap::new();

        for case in cases {
            if start < case.input_data.len() {
                let case_end = (start + chunk_size).min(case.input_data.len());
                if case_end > start {
                    let slice_data = &case.input_data[start..case_end];
                    signal_map.insert(case.input_signal_key.clone(), slice_data.to_vec());

                    let slice_conf = &case.input_confidence[start..case_end];
                    confidence_map.insert(case.input_signal_key.clone(), slice_conf.to_vec());
                }
            }
        }

        let face_input = if let Some(face_ref) = &ref_data.face {
            if start < face_ref.coordinates.len() {
                Some(FaceInput {
                    coordinates: face_ref.coordinates[start].clone(),
                    confidence: face_ref.confidence[start],
                })
            } else {
                None
            }
        } else {
            None
        };

        let chunk = InputChunk {
            timestamp: ts_slice.to_vec(),
            signals: signal_map,
            confidences: confidence_map,
            face: face_input,
        };

        let result = session.process_chunk(chunk, mode.clone());

        if ref_data.face.is_some() {
            if let Some(face_res) = &result.face {
                if matches!(mode, WaveformMode::Global) {
                     assert_eq!(
                        face_res.coordinates.len(), 
                        result.timestamp.len(), 
                        "Face result length mismatch in Global mode"
                    );
                }
                
                if let Some(last_coords) = face_res.coordinates.last() {
                     let input_coords = &ref_data.face.as_ref().unwrap().coordinates[start];
                     if !last_coords.is_empty() && !input_coords.is_empty() {
                         assert!((last_coords[0] - input_coords[0]).abs() < 1.0, 
                             "Face coordinate drift detected: Input {:?} vs Output {:?}", input_coords, last_coords);
                     }
                }
            } else {
                panic!("Session failed to return FaceResult despite valid FaceInput provided.");
            }
        }

        for (key, val) in result.signals {
            let entry = waveform_results.entry(key.clone()).or_insert((Vec::new(), Vec::new()));
            
            let avg_conf = if !val.confidence.is_empty() {
                val.confidence.iter().sum::<f32>() / val.confidence.len() as f32
            } else {
                0.0
            };

            if matches!(mode, WaveformMode::Incremental) {
                if !val.data.is_empty() {
                    assert_eq!(
                        val.data.len(), 
                        unique_frames_sent, 
                        "Incremental mode for {} returned {} frames, expected {} (unique frames sent)", 
                        key, val.data.len(), unique_frames_sent
                    );
                    
                    assert_eq!(
                        val.confidence.len(),
                        unique_frames_sent,
                        "Incremental mode for {} returned {} confidence scores, expected {}",
                        key, val.confidence.len(), unique_frames_sent
                    );
                }
                entry.0.extend(val.data);
                entry.1.extend(val.confidence);
            } else {
                entry.0 = val.data;
                entry.1 = val.confidence;
            }

            if let Some(v) = val.value {
                final_results.insert(key, (v, avg_conf));
            }
        }

        if end == max_len { break; }
        start += step_size;
    }

    (final_results, waveform_results)
}