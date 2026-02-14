use std::collections::HashMap;
use vitallens_core::state::session::Session;
use vitallens_core::types::{InputChunk, ModelConfig, WaveformMode, FaceInput};

// --- HELPERS ---

fn mock_config(vitals: Vec<&str>) -> ModelConfig {
    ModelConfig {
        name: "test_model".to_string(),
        supported_vitals: vitals.iter().map(|s| s.to_string()).collect(),
        fps_target: 30.0,
        input_size: 30,
        roi_method: "face".to_string(),
    }
}

fn mock_chunk(
    times: Vec<f64>, 
    signals: Vec<(&str, Vec<f32>)>, 
    face: Option<FaceInput>
) -> InputChunk {
    let mut sig_map = HashMap::new();
    let mut conf_map = HashMap::new();
    
    for (key, data) in signals {
        let len = data.len();
        sig_map.insert(key.to_string(), data);
        conf_map.insert(key.to_string(), vec![1.0; len]);  
    }

    InputChunk {
        timestamp: times,
        signals: sig_map,
        confidences: conf_map,
        face,
    }
}

fn mock_sine(len: usize, fs: f32, freq_hz: f32) -> Vec<f32> {
    (0..len).map(|i| {
        let t = i as f32 / fs;
        (t * 2.0 * std::f32::consts::PI * freq_hz).sin()
    }).collect()
}

// --- 1. CORE STATE & BUFFERING TESTS ---

#[test]
fn st_01_soft_stitching_averages_overlap() {
    let config = mock_config(vec!["ppg_waveform"]);
    let mut session = Session::new(config);

    // Chunk 1: [1.0, 2.0] -> [10, 20]
    let chunk1 = mock_chunk(
        vec![1.0, 2.0], 
        vec![("ppg_waveform", vec![10.0, 20.0])], 
        None
    );
    let _ = session.process_chunk(chunk1, WaveformMode::Complete);

    // Chunk 2: [2.0, 3.0] -> [22, 30]
    // Overlap at t=2.0: Average of 20 and 22 is 21.
    let chunk2 = mock_chunk(
        vec![2.0, 3.0], 
        vec![("ppg_waveform", vec![22.0, 30.0])], 
        None
    );
    let result = session.process_chunk(chunk2, WaveformMode::Complete);

    let ppg = result.signals.get("ppg_waveform").unwrap();
    
    // Total: [1.0, 2.0, 3.0]
    assert_eq!(result.timestamp, vec![1.0, 2.0, 3.0]);
    assert!((ppg.data[1] - 21.0).abs() < 0.001, "Expected 21.0, got {}", ppg.data[1]);
}

#[test]
fn st_02_disjoint_chunks_handle_gaps() {
    let config = mock_config(vec!["ppg_waveform"]);
    let mut session = Session::new(config);

    let chunk1 = mock_chunk(vec![1.0], vec![("ppg_waveform", vec![10.0])], None);
    let _ = session.process_chunk(chunk1, WaveformMode::Complete);

    // Gap between 1.0 and 5.0
    let chunk2 = mock_chunk(vec![5.0], vec![("ppg_waveform", vec![50.0])], None);
    let result = session.process_chunk(chunk2, WaveformMode::Complete);

    assert_eq!(result.timestamp, vec![1.0, 5.0]);
    assert_eq!(result.signals["ppg_waveform"].data, vec![10.0, 50.0]);
}

#[test]
fn st_03_exact_duplicate_chunks_ignored() {
    let config = mock_config(vec!["ppg_waveform"]);
    let mut session = Session::new(config);

    let chunk1 = mock_chunk(vec![1.0, 2.0], vec![("ppg_waveform", vec![10.0, 20.0])], None);
    let _ = session.process_chunk(chunk1, WaveformMode::Complete);

    // Identical chunk
    let chunk2 = mock_chunk(vec![1.0, 2.0], vec![("ppg_waveform", vec![10.0, 20.0])], None);
    let result = session.process_chunk(chunk2, WaveformMode::Complete);

    // Should not append duplicates
    assert_eq!(result.timestamp, vec![1.0, 2.0]);
    
    let ppg = &result.signals["ppg_waveform"].data;
    assert_eq!(ppg[0], 10.0);
    assert_eq!(ppg[1], 20.0);
}

#[test]
fn st_04_nan_handling_in_signal() {
    let config = mock_config(vec!["ppg_waveform"]);
    let mut session = Session::new(config);

    // Initial valid value
    let c1 = mock_chunk(vec![1.0], vec![("ppg_waveform", vec![10.0])], None);
    session.process_chunk(c1, WaveformMode::Complete);

    // Update with NaN
    let c2 = mock_chunk(vec![1.0], vec![("ppg_waveform", vec![f32::NAN])], None);
    let result = session.process_chunk(c2, WaveformMode::Complete);

    // Should treat NaN as 0.0 (or ignore) during merge, resulting in average of 10 and 0 -> 5?
    // Or implementation specific: buffers.rs check `!new_val.is_nan()` skips addition.
    // If skipped: sum=10, count=1 -> 10.
    
    let val = result.signals["ppg_waveform"].data[0];
    assert!(!val.is_nan());
    assert!((val - 10.0).abs() < 0.001);
}

#[test]
fn st_05_pruning_limits_history() {
    let mut config = mock_config(vec!["ppg_waveform"]);
    config.fps_target = 1.0; 
    // max_history = 1.0 * 60.0 = 60 samples
    let mut session = Session::new(config);

    // Feed 100 samples
    let times: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let ppg: Vec<f32> = vec![1.0; 100];
    
    let chunk = mock_chunk(times, vec![("ppg_waveform", ppg)], None);
    let result = session.process_chunk(chunk, WaveformMode::Complete);

    // Should only keep last 60
    assert_eq!(result.timestamp.len(), 60);
    assert_eq!(result.timestamp.first(), Some(&40.0));  
    assert_eq!(result.timestamp.last(), Some(&99.0));
}

// --- Logic / Registry Tests ---

#[test]
fn reg_01_dependency_ordering() {
    // Tests that we calculate Heart Rate (Order 0) before HRV (Order 1)
    // and that HRV is correctly produced.
    let config = mock_config(vec!["ppg_waveform", "hrv_sdnn", "heart_rate"]);
    let mut session = Session::new(config);

    // 10 seconds of data @ 30Hz
    let times: Vec<f64> = (0..300).map(|i| i as f64 / 30.0).collect();
    
    // UPDATED: Use a sine wave (1.2 Hz = 72 BPM) so FFT finds a rate > 40 BPM
    let ppg = mock_sine(300, 30.0, 1.2); 

    let chunk = mock_chunk(times, vec![("ppg_waveform", ppg)], None);
    let result = session.process_chunk(chunk, WaveformMode::Complete);

    // "heart_rate" requires 4s, we gave 10s. Signal is clean sine. Should exist.
    assert!(result.signals.contains_key("heart_rate"));
    // "hrv_sdnn" requires 10s, we gave 10s. Should exist.
    assert!(result.signals.contains_key("hrv_sdnn"));
}

#[test]
fn reg_02_minimum_data_gating() {
    let config = mock_config(vec!["ppg_waveform", "heart_rate"]);
    let mut session = Session::new(config);

    // 2 seconds of data (Req is 4s)
    let times: Vec<f64> = (0..60).map(|i| i as f64 / 30.0).collect();
    let ppg = mock_sine(60, 30.0, 1.2);
    
    let chunk = mock_chunk(times, vec![("ppg_waveform", ppg)], None);
    let result = session.process_chunk(chunk, WaveformMode::Complete);
    
    // Should pass PPG through
    assert!(result.signals.contains_key("ppg_waveform"));
    // Should NOT have HR (insufficient duration)
    assert!(!result.signals.contains_key("heart_rate"));
}

#[test]
fn reg_03_alias_resolution() {
    // "pulse" is an alias for "heart_rate"
    let config = mock_config(vec!["ppg_waveform", "pulse"]);
    let mut session = Session::new(config);

    // 5 seconds of data (Req is 4s)
    let times: Vec<f64> = (0..150).map(|i| i as f64 / 30.0).collect();
    
    // UPDATED: Use sine wave so we get a valid HR
    let ppg = mock_sine(150, 30.0, 1.2);

    let chunk = mock_chunk(times, vec![("ppg_waveform", ppg)], None);
    let result = session.process_chunk(chunk, WaveformMode::Complete);

    // Result should be keyed by the canonical ID "heart_rate", NOT "pulse"
    assert!(result.signals.contains_key("heart_rate"));
    assert!(!result.signals.contains_key("pulse"));
}

#[test]
fn reg_04_provided_scalar_spo2() {
    // SpO2 is provided, not derived from PPG in this context
    let mut config = mock_config(vec!["spo2"]);
    config.fps_target = 1.0;  
    
    let mut session = Session::new(config);

    // 3 samples, window is sufficient for "Average" method
    let chunk = mock_chunk(
        vec![1.0, 2.0, 3.0], 
        vec![("spo2", vec![98.0, 99.0, 98.0])], 
        None
    );
    let result = session.process_chunk(chunk, WaveformMode::Complete);

    let sig = result.signals.get("spo2").unwrap();
    
    assert_eq!(sig.data, vec![98.0, 99.0, 98.0]);
    
    // Check scalar derivation (Average)
    assert!(sig.value.is_some(), "Scalar value missing");
    let val = sig.value.unwrap();
    assert!((val - 98.333).abs() < 0.01);
}

#[test]
fn reg_05_unsupported_vital() {
    let config = mock_config(vec!["blood_pressure"]); // Not in registry
    let mut session = Session::new(config);

    let chunk = mock_chunk(vec![1.0], vec![], None);
    let result = session.process_chunk(chunk, WaveformMode::Complete);

    assert!(!result.signals.contains_key("blood_pressure"));
}

// --- Output/Waveform Mode Tests ---

#[test]
fn out_01_incremental_mode() {
    let config = mock_config(vec!["ppg_waveform"]);
    let mut session = Session::new(config);

    // First push
    let c1 = mock_chunk(vec![1.0], vec![("ppg_waveform", vec![10.0])], None);
    let r1 = session.process_chunk(c1, WaveformMode::Incremental);
    assert_eq!(r1.timestamp, vec![1.0]);

    // Second push
    let c2 = mock_chunk(vec![2.0], vec![("ppg_waveform", vec![20.0])], None);
    let r2 = session.process_chunk(c2, WaveformMode::Incremental);
    
    // Incremental should only return NEW data
    assert_eq!(r2.timestamp, vec![2.0]);
    assert_eq!(r2.signals["ppg_waveform"].data, vec![20.0]);
}

#[test]
fn out_02_windowed_mode() {
    let config = mock_config(vec!["ppg_waveform"]);
    let mut session = Session::new(config);

    // 10 seconds of data
    let times: Vec<f64> = (0..300).map(|i| i as f64 / 30.0).collect();
    let ppg = vec![0.0; 300];
    let chunk = mock_chunk(times, vec![("ppg_waveform", ppg)], None);

    // Request last 2 seconds
    let result = session.process_chunk(chunk, WaveformMode::Windowed(2.0));

    // 2s * 30fps = 60 frames
    assert_eq!(result.timestamp.len(), 60);
    // Should be the end of the signal
    assert!(result.timestamp.last().unwrap() > &9.9);
}

#[test]
fn out_03_complete_mode() {
    let config = mock_config(vec!["ppg_waveform"]);
    let mut session = Session::new(config);

    // First 10s
    let times1: Vec<f64> = (0..300).map(|i| i as f64 / 30.0).collect();
    let ppg1 = vec![1.0; 300];
    let chunk1 = mock_chunk(times1, vec![("ppg_waveform", ppg1)], None);
    let _ = session.process_chunk(chunk1, WaveformMode::Complete);

    // Next 2s
    let times2: Vec<f64> = (300..360).map(|i| i as f64 / 30.0).collect();
    let ppg2 = vec![2.0; 60];
    let chunk2 = mock_chunk(times2, vec![("ppg_waveform", ppg2)], None);
    
    // Return EVERYTHING
    let result = session.process_chunk(chunk2, WaveformMode::Complete);

    // Total 12s = 360 frames
    assert_eq!(result.timestamp.len(), 360);
    assert_eq!(result.timestamp.first(), Some(&0.0));
    assert_eq!(result.timestamp.last(), Some(&11.966666666666667));  
    
    let data = &result.signals["ppg_waveform"].data;
    assert_eq!(data[0], 1.0);     
    assert_eq!(data[359], 2.0);   
}

// --- Face Tests ---

#[test]
fn face_01_sync_with_signal() {
    let config = mock_config(vec!["ppg_waveform"]);
    let mut session = Session::new(config);

    let face = FaceInput {
        coordinates: [10.0, 10.0, 50.0, 50.0],
        confidence: 0.9,
    };

    let chunk = mock_chunk(
        vec![1.0, 2.0], 
        vec![("ppg_waveform", vec![10.0, 20.0])], 
        Some(face)
    );

    let result = session.process_chunk(chunk, WaveformMode::Complete);

    assert!(result.face.is_some());
    let f = result.face.unwrap();
    assert_eq!(f.coordinates.len(), 2);  
    assert_eq!(f.confidence.len(), 2);
    assert_eq!(f.coordinates[0], [10.0, 10.0, 50.0, 50.0]);
}

#[test]
fn face_02_missing_face_data_pads_zeros() {
    let config = mock_config(vec!["ppg_waveform"]);
    let mut session = Session::new(config);

    // Chunk without face data
    let chunk = mock_chunk(
        vec![1.0, 2.0], 
        vec![("ppg_waveform", vec![10.0, 20.0])], 
        None
    );

    let result = session.process_chunk(chunk, WaveformMode::Complete);

    // Should still return face struct, but padded with 0.0
    // (This behavior depends on session.rs `merge_face` logic)
    
    assert!(result.face.is_some());
    let f = result.face.unwrap();
    assert_eq!(f.coordinates.len(), 2);
    assert_eq!(f.coordinates[0], [0.0, 0.0, 0.0, 0.0]);
}