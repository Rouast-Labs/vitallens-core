#![cfg(feature = "python")]

use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use crate::signal;
use crate::signal::rate::{RateBounds, RateStrategy};
use crate::registry::HrvMetric;
use crate::signal::peaks::{self, PeakOptions, SignalBounds};
use crate::geometry::roi;
use crate::types::{Rect, RoiMethod, FaceDetector};
use crate::types::VitalInfo;
use crate::state::frames::compute_buffer_config;

pub fn register_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate_heart_rate, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_respiratory_rate, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_rate_from_detections, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_hrv_metric, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_hrv_metric_from_detections, m)?)?;
    m.add_function(wrap_pyfunction!(find_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_roi, m)?)?;
    m.add_function(wrap_pyfunction!(get_vital_info, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_buffer_config, m)?)?;
    m.add_function(wrap_pyfunction!(detrend_lambda_for_cutoff, m)?)?;
    m.add_function(wrap_pyfunction!(moving_average_window_for_cutoff, m)?)?;
    Ok(())
}

#[pyfunction]
fn estimate_heart_rate(_py: Python, signal: PyReadonlyArray1<f32>, fs: f32) -> PyResult<(f32, f32)> {
    let s = signal.as_slice()?;
    let bounds = RateBounds { min: 40.0, max: 220.0 }; 
    let strategy = RateStrategy::Periodogram { target_res_hz: 0.005 };
    
    let res = signal::estimate_rate(s, fs, None, bounds, strategy, None, None);
    
    Ok((res.value, res.confidence))
}

#[pyfunction]
fn estimate_respiratory_rate(_py: Python, signal: PyReadonlyArray1<f32>, fs: f32) -> PyResult<(f32, f32)> {
    let s = signal.as_slice()?;
    let bounds = RateBounds { min: 3.0, max: 60.0 };
    let strategy = RateStrategy::Periodogram { target_res_hz: 0.01 };

    let res = signal::estimate_rate(s, fs, None, bounds, strategy, None, None);

    Ok((res.value, res.confidence))
}

#[pyfunction]
fn estimate_hrv_metric(
    _py: Python,
    signal: PyReadonlyArray1<f32>,
    fs: f32,
    metric_name: &str,
    confidence: PyReadonlyArray1<f32>,
    rate_hint: Option<f32>,
) -> PyResult<(f32, f32)> {
    let s = signal.as_slice()?;
    let conf = confidence.as_slice()?;
    
    let metric = match metric_name.to_lowercase().as_str() {
        "sdnn" | "hrv_sdnn" => HrvMetric::Sdnn,
        "rmssd" | "hrv_rmssd" => HrvMetric::Rmssd,
        "lfhf" | "hrv_lfhf" => HrvMetric::LfHf,
        "si" | "stress_index" => HrvMetric::StressIndex,
        "pnn50" | "hrv_pnn50" => HrvMetric::Pnn50,
        "sd1sd2" | "hrv_sd1sd2" => HrvMetric::Sd1Sd2,
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid HRV metric")),
    };

    let bounds = SignalBounds { min_rate: 40.0, max_rate: 220.0 };
    
    let (val, conf_out) = signal::estimate_hrv(
        s, 
        fs, 
        metric, 
        &[], 
        conf, 
        bounds, 
        rate_hint
    );
    
    Ok((val, conf_out))
}

fn py_seqs_to_peaks(sequences: Vec<Vec<f32>>) -> Vec<Vec<crate::signal::peaks::Peak>> {
    sequences.into_iter().map(|seq| {
        seq.into_iter().map(|x| crate::signal::peaks::Peak {
            index: x.floor() as usize,
            x,
            y: 1.0,
        }).collect()
    }).collect()
}

#[pyfunction]
#[pyo3(signature = (sequences, fs, timestamps=None, confidence=None))]
fn estimate_rate_from_detections(
    _py: Python,
    sequences: Vec<Vec<f32>>,
    fs: f32,
    timestamps: Option<Vec<f64>>,
    confidence: Option<Vec<f32>>,
) -> PyResult<(f32, f32)> {
    let peak_segments = py_seqs_to_peaks(sequences);
    let ts_slice = timestamps.as_deref();
    let conf_slice = confidence.as_deref();
    
    let res = crate::signal::rate::estimate_rate_from_detections(&peak_segments, fs, ts_slice, conf_slice);

    Ok((res.value, res.confidence))
}

#[pyfunction]
#[pyo3(signature = (sequences, fs, metric_name, timestamps=None, confidence=None))]
fn estimate_hrv_metric_from_detections(
    _py: Python,
    sequences: Vec<Vec<f32>>,
    fs: f32,
    metric_name: &str,
    timestamps: Option<Vec<f64>>,
    confidence: Option<Vec<f32>>,
) -> PyResult<(f32, f32)> {
    let peak_segments = py_seqs_to_peaks(sequences);
    let ts_slice = timestamps.as_deref();
    let conf_slice = confidence.as_deref();

    let metric = match metric_name.to_lowercase().as_str() {
        "sdnn" | "hrv_sdnn" => HrvMetric::Sdnn,
        "rmssd" | "hrv_rmssd" => HrvMetric::Rmssd,
        "lfhf" | "hrv_lfhf" => HrvMetric::LfHf,
        "si" | "stress_index" => HrvMetric::StressIndex,
        "pnn50" | "hrv_pnn50" => HrvMetric::Pnn50,
        "sd1sd2" | "hrv_sd1sd2" => HrvMetric::Sd1Sd2,
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid HRV metric")),
    };

    let (val, conf_out) = crate::signal::hrv::estimate_hrv_from_detections(
        &peak_segments, fs, metric, ts_slice, conf_slice
    );

    Ok((val, conf_out))
}

#[pyfunction]
#[pyo3(signature = (
    signal, 
    fs, 
    refine, 
    rate_hint=None,
    min_rate=40.0,
    max_rate=220.0,
    detection_threshold=0.45,
    window_cycles=2.5,
    max_rate_change=1.0
))]
#[allow(clippy::too_many_arguments)]
fn find_peaks(
    _py: Python, 
    signal: PyReadonlyArray1<f32>, 
    fs: f32, 
    refine: bool,
    rate_hint: Option<f32>,
    min_rate: f32,
    max_rate: f32,
    detection_threshold: f32,
    window_cycles: f32,
    max_rate_change: f32,
) -> PyResult<Vec<f32>> {
    let s = signal.as_slice()?;
    
    let options = PeakOptions {
        fs,
        bounds: SignalBounds { min_rate, max_rate },
        threshold: detection_threshold,
        window_cycles,
        max_rate_change_per_sec: max_rate_change,
        refine,
        smooth_input: true,
        avg_rate_hint: rate_hint,
        ..Default::default()
    };

    let segments = peaks::find_peaks(s, options);

    let peaks: Vec<f32> = segments.into_iter()
        .flat_map(|seg| seg.into_iter())
        .map(|p| p.x)
        .collect();

    Ok(peaks)
}

#[pyfunction]
#[pyo3(signature = (face, method, detector=FaceDetector::Default, container=None, force_even=false))]
fn calculate_roi(
    _py: Python,
    face: Rect,
    method: RoiMethod,
    detector: FaceDetector,
    container: Option<(f32, f32)>,
    force_even: bool
) -> PyResult<Rect> {
    let (cw, ch) = container.map(|(w, h)| (Some(w), Some(h))).unwrap_or((None, None));    
    Ok(roi::calculate_roi(face, method, detector, cw, ch, force_even))
}

#[pyfunction]
fn get_vital_info(_py: Python, vital_id: &str) -> PyResult<Option<VitalInfo>> {
    Ok(crate::get_vital_info(vital_id.to_string()))
}

#[pyfunction]
#[pyo3(name = "compute_buffer_config")]
fn py_compute_buffer_config(_py: Python, config: &crate::types::SessionConfig) -> PyResult<crate::types::BufferConfig> {
    Ok(compute_buffer_config(config.clone()))
}

#[pyfunction]
fn detrend_lambda_for_cutoff(_py: Python, fs: f32, cutoff: f32) -> PyResult<f32> {
    Ok(crate::signal::filters::detrend_lambda_for_cutoff(fs, cutoff))
}

#[pyfunction]
fn moving_average_window_for_cutoff(_py: Python, fs: f32, cutoff_hz: f32, force_odd: bool) -> PyResult<usize> {
    Ok(crate::signal::filters::moving_average_window_for_cutoff(fs, cutoff_hz, force_odd))
}