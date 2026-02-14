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
        conf_map.insert(key.to_string(), vec![1.0; len]); // Default 100% confidence
    }

    InputChunk {
        timestamp: times,
        signals: sig_map,
        confidences: conf_map,
        face,
    }
}

// --- 1. CORE STATE & BUFFERING TESTS ---

#[test]
fn st_01_soft_stitching_averages_overlap() {
    let config = mock_config(vec!["ppg_waveform"]);
    let mut session = Session::new(config);

    // Chunk 1: [1.0, 2.0] -> PPG [10, 20]
    let chunk1 = mock_chunk(
        vec![1.0, 2.0], 
        vec![("ppg_waveform", vec![10.0, 20.0])], 
        None
    );
    let _ = session.process_chunk(chunk1, WaveformMode::Complete);

    // Chunk 2: [2.0, 3.0] -> PPG [22, 30] (Overlap at 2.0)
    let chunk2 = mock_chunk(
        vec![2.0, 3.0], 
        vec![("ppg_waveform", vec![22.0, 30.0])], 
        None
    );
    let result = session.process_chunk(chunk2, WaveformMode::Complete);

    let ppg = result.signals.get("ppg_waveform").unwrap();
    
    // Value at 2.0 should be (20 + 22) / 2 = 21
    assert_eq!(result.timestamp, vec![1.0, 2.0, 3.0]);
    assert!((ppg.data[1] - 21.0).abs() < 0.001, "Expected 21.0, got {}", ppg.data[1]);
}

#[test]
fn st_02_disjoint_chunks_handle_gaps() {
    let config = mock_config(vec!["ppg_waveform"]);
    let mut session = Session::new(config);

    let chunk1 = mock_chunk(vec![1.0], vec![("ppg_waveform", vec![10.0])], None);
    let _ = session.process_chunk(chunk1, WaveformMode::Complete);

    // Gap! 1.0 -> 5.0
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

    // Send exact same chunk again
    let chunk2 = mock_chunk(vec![1.0, 2.0], vec![("ppg_waveform", vec![10.0, 20.0])], None);
    let result = session.process_chunk(chunk2, WaveformMode::Complete);

    // Timestamps should not grow
    assert_eq!(result.timestamp, vec![1.0, 2.0]);
    // Values averaged: (10+10)/2 = 10, (20+20)/2 = 20
    let ppg = &result.signals["ppg_waveform"].data;
    assert_eq!(ppg[0], 10.0);
    assert_eq!(ppg[1], 20.0);
}

#[test]
fn st_04_nan_handling_in_signal() {
    let config = mock_config(vec!["ppg_waveform"]);
    let mut session = Session::new(config);

    // Chunk 1: [1.0] -> 10.0
    let c1 = mock_chunk(vec![1.0], vec![("ppg_waveform", vec![10.0])], None);
    session.process_chunk(c1, WaveformMode::Complete);

    // Chunk 2: [1.0] -> NaN (Overlap)
    let c2 = mock_chunk(vec![1.0], vec![("ppg_waveform", vec![f32::NAN])], None);
    let result = session.process_chunk(c2, WaveformMode::Complete);

    // Should ignore NaN, so average is just 10.0 / 1 sample = 10.0
    // (Or 10+0 / 2 if logic treats NaN as 0, but SignalBuffer logic skips NaN addition)
    // Checking SignalBuffer implementation: if !new_val.is_nan() -> adds.
    // So sum=10, count=1. Result=10. Correct.
    let val = result.signals["ppg_waveform"].data[0];
    assert!(!val.is_nan());
    assert!((val - 10.0).abs() < 0.001);
}

#[test]
fn st_05_pruning_limits_history() {
    let mut config = mock_config(vec!["ppg_waveform"]);
    config.fps_target = 1.0; 
    // Session calculates max_history = fps * 60 = 60 frames
    let mut session = Session::new(config);

    // Push 100 frames (0.0 to 99.0)
    let times: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let ppg: Vec<f32> = vec![1.0; 100];
    
    let chunk = mock_chunk(times, vec![("ppg_waveform", ppg)], None);
    let result = session.process_chunk(chunk, WaveformMode::Complete);

    // Should be capped at 60
    assert_eq!(result.timestamp.len(), 60);
    assert_eq!(result.timestamp.first(), Some(&40.0)); // 0..39 dropped
    assert_eq!(result.timestamp.last(), Some(&99.0));
}

// --- 2. REGISTRY & DERIVATION TESTS ---

#[test]
fn reg_01_dependency_ordering() {
    // Request HRV (needs HR) and HR (needs PPG)
    // "hrv_sdnn" order is 1, "heart_rate" is 0. Session should act correctly.
    let config = mock_config(vec!["ppg_waveform", "hrv_sdnn", "heart_rate"]);
    let mut session = Session::new(config);

    // Push enough data for HRV (min 10s). 
    // 300 frames @ 30fps = 10s
    let times: Vec<f64> = (0..300).map(|i| i as f64 / 30.0).collect();
    let ppg: Vec<f32> = vec![0.5; 300]; 

    let chunk = mock_chunk(times, vec![("ppg_waveform", ppg)], None);
    let result = session.process_chunk(chunk, WaveformMode::Complete);

    // Check that we got results (mocks return dummy values)
    assert!(result.signals.contains_key("heart_rate"));
    assert!(result.signals.contains_key("hrv_sdnn"));
}

#[test]
fn reg_02_minimum_data_gating() {
    let config = mock_config(vec!["ppg_waveform", "heart_rate"]);
    let mut session = Session::new(config);

    // HR requires 4.0s (120 frames @ 30fps)
    
    // Push 2.0s (60 frames)
    let times: Vec<f64> = (0..60).map(|i| i as f64 / 30.0).collect();
    let ppg = vec![0.0; 60];
    let chunk = mock_chunk(times, vec![("ppg_waveform", ppg)], None);
    
    let result = session.process_chunk(chunk, WaveformMode::Complete);
    
    // Should have PPG, but NO Heart Rate
    assert!(result.signals.contains_key("ppg_waveform"));
    assert!(!result.signals.contains_key("heart_rate"));
}

#[test]
fn reg_03_alias_resolution() {
    // Request "pulse" (alias for "heart_rate")
    let config = mock_config(vec!["ppg_waveform", "pulse"]);
    let mut session = Session::new(config);

    // Push enough data (5s)
    let times: Vec<f64> = (0..150).map(|i| i as f64 / 30.0).collect();
    let ppg = vec![0.0; 150];
    let chunk = mock_chunk(times, vec![("ppg_waveform", ppg)], None);
    
    let result = session.process_chunk(chunk, WaveformMode::Complete);

    // Should return key "heart_rate" (canonical), NOT "pulse"
    assert!(result.signals.contains_key("heart_rate"));
    assert!(!result.signals.contains_key("pulse"));
}

#[test]
fn reg_04_provided_scalar_spo2() {
    // SpO2 is Provided (Waveform) AND Derived (Average)
    let mut config = mock_config(vec!["spo2"]);
    config.fps_target = 1.0; // Set FPS to 1.0 so 3 frames = 3 seconds > 1.0s min requirement
    
    let mut session = Session::new(config);

    // Push SpO2 waveform [98, 99, 98]
    // At 1.0 FPS, this is 3 seconds of data.
    let chunk = mock_chunk(
        vec![1.0, 2.0, 3.0], 
        vec![("spo2", vec![98.0, 99.0, 98.0])], 
        None
    );
    let result = session.process_chunk(chunk, WaveformMode::Complete);

    let sig = result.signals.get("spo2").unwrap();
    
    // Check Waveform
    assert_eq!(sig.data, vec![98.0, 99.0, 98.0]);
    
    // Check Scalar (Average)
    // (98+99+98)/3 = 98.333
    assert!(sig.value.is_some(), "Scalar value missing - derivation likely skipped due to insufficient data");
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

// --- 3. OUTPUT MODES ---

#[test]
fn out_01_incremental_mode() {
    let config = mock_config(vec!["ppg_waveform"]);
    let mut session = Session::new(config);

    // Push A
    let c1 = mock_chunk(vec![1.0], vec![("ppg_waveform", vec![10.0])], None);
    let r1 = session.process_chunk(c1, WaveformMode::Incremental);
    assert_eq!(r1.timestamp, vec![1.0]);

    // Push B
    let c2 = mock_chunk(vec![2.0], vec![("ppg_waveform", vec![20.0])], None);
    let r2 = session.process_chunk(c2, WaveformMode::Incremental);
    
    // Should ONLY contain 2.0
    assert_eq!(r2.timestamp, vec![2.0]);
    assert_eq!(r2.signals["ppg_waveform"].data, vec![20.0]);
}

#[test]
fn out_02_windowed_mode() {
    let config = mock_config(vec!["ppg_waveform"]);
    let mut session = Session::new(config);

    // Push 0..10s (300 frames)
    let times: Vec<f64> = (0..300).map(|i| i as f64 / 30.0).collect();
    let ppg = vec![0.0; 300];
    let chunk = mock_chunk(times, vec![("ppg_waveform", ppg)], None);

    // Request last 2.0 seconds
    let result = session.process_chunk(chunk, WaveformMode::Windowed(2.0));

    // 2.0s @ 30fps = 60 frames
    assert_eq!(result.timestamp.len(), 60);
    // Last timestamp should be 9.966 (frame 299)
    assert!(result.timestamp.last().unwrap() > &9.9);
}

#[test]
fn out_03_complete_mode() {
    let config = mock_config(vec!["ppg_waveform"]);
    let mut session = Session::new(config);

    // Push Chunk 1 (Time 0..10s)
    let times1: Vec<f64> = (0..300).map(|i| i as f64 / 30.0).collect();
    let ppg1 = vec![1.0; 300];
    let chunk1 = mock_chunk(times1, vec![("ppg_waveform", ppg1)], None);
    let _ = session.process_chunk(chunk1, WaveformMode::Complete);

    // Push Chunk 2 (Time 10..12s)
    let times2: Vec<f64> = (300..360).map(|i| i as f64 / 30.0).collect();
    let ppg2 = vec![2.0; 60];
    let chunk2 = mock_chunk(times2, vec![("ppg_waveform", ppg2)], None);
    
    // Request COMPLETE history
    let result = session.process_chunk(chunk2, WaveformMode::Complete);

    // Should contain ALL 360 frames (12 seconds)
    assert_eq!(result.timestamp.len(), 360);
    assert_eq!(result.timestamp.first(), Some(&0.0));
    assert_eq!(result.timestamp.last(), Some(&11.966666666666667)); // approx 12.0
    
    // Verify data concatenation
    let data = &result.signals["ppg_waveform"].data;
    assert_eq!(data[0], 1.0);   // From Chunk 1
    assert_eq!(data[359], 2.0); // From Chunk 2
}

// --- 4. FACE TRACKING ---

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
    assert_eq!(f.coordinates.len(), 2); // Should match timestamp len
    assert_eq!(f.confidence.len(), 2);
    assert_eq!(f.coordinates[0], [10.0, 10.0, 50.0, 50.0]);
}

#[test]
fn face_02_missing_face_data_pads_zeros() {
    let config = mock_config(vec!["ppg_waveform"]);
    let mut session = Session::new(config);

    // Chunk has signals but NO face
    let chunk = mock_chunk(
        vec![1.0, 2.0], 
        vec![("ppg_waveform", vec![10.0, 20.0])], 
        None
    );

    let result = session.process_chunk(chunk, WaveformMode::Complete);

    // Should produce empty/zero face data aligned with timestamps?
    // Current implementation: if face buffer is populated at all, we return it.
    // If chunk had None, we pushed [0,0,0,0].
    
    assert!(result.face.is_some());
    let f = result.face.unwrap();
    assert_eq!(f.coordinates.len(), 2);
    assert_eq!(f.coordinates[0], [0.0, 0.0, 0.0, 0.0]);
}