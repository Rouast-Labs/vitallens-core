//! Tests for `SessionConfig::hrv_skip_seconds`: skipping the start of a session when deriving HRV.

use std::collections::HashMap;

use vitallens_core::types::SignalInput;
use vitallens_core::{Session, SessionConfig, SessionInput, WaveformMode};

const FPS: f32 = 30.0;

/// Beat times (s): a wandering "settling" stretch first, then regular beats with small variability.
///
/// The settling intervals drift between 0.75 s and 1.25 s. Each step stays within what the HRV
/// outlier filter accepts, so they are not discarded but inflate the SDNN.
fn beat_times(settling_seconds: f64, total_seconds: f64) -> Vec<f64> {
    let mut beats = Vec::new();
    let mut t = 0.3;
    let mut k = 0.0_f64;
    while t < settling_seconds {
        beats.push(t);
        t += 1.0 + 0.25 * (k * std::f64::consts::PI / 3.0).sin();
        k += 1.0;
    }
    let mut jitter = false;
    while t < total_seconds {
        beats.push(t);
        t += if jitter { 1.02 } else { 0.98 };
        jitter = !jitter;
    }
    beats
}

/// A PPG-like waveform: one Gaussian pulse per beat.
fn ppg_signal(beats: &[f64], total_seconds: f64) -> (Vec<f64>, Vec<f32>) {
    let n = (total_seconds * FPS as f64) as usize;
    let timestamps: Vec<f64> = (0..n).map(|i| i as f64 / FPS as f64).collect();
    let sigma = 0.08;
    let data = timestamps
        .iter()
        .map(|&t| {
            beats
                .iter()
                .map(|&b| (-((t - b).powi(2)) / (2.0 * sigma * sigma)).exp())
                .sum::<f64>() as f32
        })
        .collect();
    (timestamps, data)
}

fn config(hrv_skip_seconds: Option<f32>) -> SessionConfig {
    SessionConfig {
        model_name: "vitallens-2.0".to_string(),
        supported_vitals: vec!["heart_rate".to_string(), "hrv_sdnn".to_string()],
        return_waveforms: Some(vec!["ppg_waveform".to_string()]),
        fps_target: FPS,
        input_size: 40,
        n_inputs: 4,
        roi_method: "face".to_string(),
        estimate_rolling_vitals: None,
        hrv_skip_seconds,
    }
}

fn input(timestamps: Vec<f64>, data: Vec<f32>) -> SessionInput {
    let confidence = vec![1.0; data.len()];
    let mut signals = HashMap::new();
    signals.insert("ppg_waveform".to_string(), SignalInput { data, confidence });
    SessionInput { face: None, signals, timestamp: timestamps }
}

fn sdnn(session: &Session, timestamps: Vec<f64>, data: Vec<f32>) -> Option<f32> {
    let result = session.process(input(timestamps, data), WaveformMode::Global);
    result.vitals.get("hrv_sdnn").map(|v| v.value)
}

#[test]
fn skipping_the_start_removes_the_effect_of_unsettled_output() {
    let total = 60.0;
    let beats = beat_times(15.0, total);
    let (ts, data) = ppg_signal(&beats, total);

    let all = sdnn(&Session::new(config(None)), ts.clone(), data.clone())
        .expect("HRV without skipping");
    let skipped = sdnn(&Session::new(config(Some(17.0))), ts, data).expect("HRV with skipping");

    assert!(all > 40.0, "the irregular start should inflate SDNN, got {all}");
    assert!(skipped < 30.0, "SDNN of the regular part should be small, got {skipped}");
}

#[test]
fn no_skip_option_behaves_like_zero_seconds() {
    let total = 60.0;
    let beats = beat_times(10.0, total);
    let (ts, data) = ppg_signal(&beats, total);

    let none = sdnn(&Session::new(config(None)), ts.clone(), data.clone());
    let zero = sdnn(&Session::new(config(Some(0.0))), ts, data);

    assert_eq!(none, zero);
}

#[test]
fn no_hrv_until_enough_data_remains_after_the_skip() {
    // 30 s of data: HRV needs at least 20 s, so skipping 15 s leaves too little.
    let total = 30.0;
    let beats = beat_times(0.0, total);
    let (ts, data) = ppg_signal(&beats, total);

    assert!(sdnn(&Session::new(config(None)), ts.clone(), data.clone()).is_some());
    assert!(sdnn(&Session::new(config(Some(15.0))), ts, data).is_none());
}

#[test]
fn heart_rate_is_not_affected_by_the_skip() {
    let total = 60.0;
    let beats = beat_times(10.0, total);
    let (ts, data) = ppg_signal(&beats, total);

    let hr = |skip: Option<f32>| {
        Session::new(config(skip))
            .process(input(ts.clone(), data.clone()), WaveformMode::Global)
            .vitals
            .get("heart_rate")
            .map(|v| v.value)
    };

    assert_eq!(hr(None), hr(Some(12.0)));
}

#[test]
fn a_reset_starts_a_new_skip_period() {
    let total = 60.0;
    let beats = beat_times(10.0, total);
    let (ts, data) = ppg_signal(&beats, total);

    let session = Session::new(config(Some(12.0)));
    assert!(sdnn(&session, ts.clone(), data.clone()).is_some());

    session.reset();

    // After the reset only 10 s of the regular signal is available, all of it within the skip.
    let later: Vec<f64> = ts.iter().skip(1500).map(|t| t + 100.0).collect();
    let later_data: Vec<f32> = data.iter().skip(1500).cloned().collect();
    let first = later.first().copied().unwrap();
    let short = SessionInput {
        timestamp: later.iter().take(300).cloned().collect(),
        signals: {
            let mut m = HashMap::new();
            m.insert(
                "ppg_waveform".to_string(),
                SignalInput {
                    data: later_data.iter().take(300).cloned().collect(),
                    confidence: vec![1.0; 300],
                },
            );
            m
        },
        face: None,
    };
    let result = session.process(short, WaveformMode::Global);
    assert!(first > 100.0);
    assert!(result.vitals.get("hrv_sdnn").is_none(), "10 s after a reset is within the skip period");
}

#[test]
fn rolling_hrv_starts_only_after_the_skip_period_and_a_minimum_window() {
    let total = 90.0;
    let beats = beat_times(15.0, total);
    let (ts, data) = ppg_signal(&beats, total);

    let mut cfg = config(Some(17.0));
    cfg.estimate_rolling_vitals = Some(true);
    let result = Session::new(cfg).process(input(ts, data), WaveformMode::Global);

    let rolling = result.rolling_vitals.expect("rolling vitals");
    let first_value = rolling["hrv_sdnn"]
        .data
        .iter()
        .position(|v| !v.is_nan())
        .expect("some rolling HRV values");

    // Skip 17 s, then at least a 20 s window: the first value is not before 37 s
    assert!(
        first_value >= (37.0 * FPS as f64) as usize - 1,
        "first rolling HRV value at sample {first_value}"
    );
}
