#![cfg(feature = "python")]

use pyo3::prelude::*;
use numpy::{PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4, PyUntypedArrayMethods};
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
    m.add_function(wrap_pyfunction!(estimate_rolling_rate_from_detections, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_hrv_metric, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_hrv_metric_from_detections, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_rolling_hrv_metric_from_detections, m)?)?;
    m.add_function(wrap_pyfunction!(find_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_roi, m)?)?;
    m.add_function(wrap_pyfunction!(get_vital_info, m)?)?;
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_buffer_config, m)?)?;
    m.add_function(wrap_pyfunction!(detrend_lambda_for_cutoff, m)?)?;
    m.add_function(wrap_pyfunction!(moving_average_window_for_cutoff, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_frame, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_frames, m)?)?;
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
#[pyo3(signature = (sequences, fs, vital_name, total_frames, timestamps=None, confidence=None))]
fn estimate_rolling_rate_from_detections(
    _py: Python,
    sequences: Vec<Vec<f32>>,
    fs: f32,
    vital_name: &str,
    total_frames: usize,
    timestamps: Option<Vec<f64>>,
    confidence: Option<Vec<f32>>,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let peak_segments = py_seqs_to_peaks(sequences);
    let ts_slice = timestamps.as_deref();
    let conf_slice = confidence.as_deref();
    
    let meta = crate::registry::get_vital_meta(vital_name)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("Unknown vital rate: {}", vital_name)))?;
        
    let deriv = meta.derivations.first()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No derivations found for vital"))?;

    let (rates, confs) = crate::signal::rate::estimate_rolling_rate_from_detections(
        &peak_segments, fs, ts_slice, conf_slice, 
        deriv.min_window_seconds, deriv.preferred_window_seconds, deriv.rolling_stride_seconds, total_frames
    );

    Ok((rates, confs))
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
#[pyo3(signature = (sequences, fs, metric_name, total_frames, timestamps=None, confidence=None))]
fn estimate_rolling_hrv_metric_from_detections(
    _py: Python,
    sequences: Vec<Vec<f32>>,
    fs: f32,
    metric_name: &str,
    total_frames: usize,
    timestamps: Option<Vec<f64>>,
    confidence: Option<Vec<f32>>,
) -> PyResult<(Vec<f32>, Vec<f32>)> {
    let peak_segments = py_seqs_to_peaks(sequences);
    let ts_slice = timestamps.as_deref();
    let conf_slice = confidence.as_deref();

    let meta = crate::registry::get_vital_meta(metric_name)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("Unknown HRV metric: {}", metric_name)))?;
        
    let deriv = meta.derivations.first()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No derivations found for metric"))?;

    let metric = match metric_name.to_lowercase().as_str() {
        "sdnn" | "hrv_sdnn" => HrvMetric::Sdnn,
        "rmssd" | "hrv_rmssd" => HrvMetric::Rmssd,
        "lfhf" | "hrv_lfhf" => HrvMetric::LfHf,
        "si" | "stress_index" => HrvMetric::StressIndex,
        "pnn50" | "hrv_pnn50" => HrvMetric::Pnn50,
        "sd1sd2" | "hrv_sd1sd2" => HrvMetric::Sd1Sd2,
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid HRV metric")),
    };

    let (vals, confs) = crate::signal::hrv::estimate_rolling_hrv_from_detections(
        &peak_segments, fs, metric, ts_slice, conf_slice, 
        deriv.min_window_seconds, deriv.preferred_window_seconds, deriv.rolling_stride_seconds, total_frames
    );

    Ok((vals, confs))
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
fn core_version(_py: Python) -> PyResult<String> {
    Ok(crate::core_version())
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
// ---------------------------------------------------------------------------------------------
// Frame preparation (crop + area downsample to model input), zero copy on the input
// ---------------------------------------------------------------------------------------------

/// Byte strides and pointer of one (h, w, c) uint8 frame view, checked for the given format.
struct FrameView {
    ptr: *const u8,
    len: usize,
    height: u32,
    width: u32,
    row_stride: usize,
    pixel_stride: usize,
}

// The pointer is only read while the owning numpy array is borrowed by the caller.
unsafe impl Send for FrameView {}
unsafe impl Sync for FrameView {}

fn frame_view(ptr: *const u8, shape: &[usize], strides: &[isize], bpp: usize) -> PyResult<FrameView> {
    let err = |m: &str| pyo3::exceptions::PyValueError::new_err(m.to_string());
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    if c < bpp {
        return Err(err("frame has fewer channels than the pixel format needs"));
    }
    if strides.iter().any(|&s| s < 0) || strides[2] != 1 {
        return Err(err("frame must have positive strides and contiguous channels"));
    }
    if h == 0 || w == 0 {
        return Err(err("frame must not be empty"));
    }
    let (rs, ps) = (strides[0] as usize, strides[1] as usize);
    Ok(FrameView {
        ptr,
        len: (h - 1) * rs + (w - 1) * ps + c,
        height: h as u32,
        width: w as u32,
        row_stride: rs,
        pixel_stride: ps,
    })
}

fn prepare_view(
    v: &FrameView,
    format: crate::image::PixelFormat,
    crop: (u32, u32, u32, u32),
    size: (u32, u32),
    orientation: crate::image::Orientation,
    out: &mut [u8],
) -> Result<(), crate::image::FrameError> {
    // SAFETY: `ptr`/`len` describe memory of a numpy array the caller keeps borrowed.
    let data = unsafe { std::slice::from_raw_parts(v.ptr, v.len) };
    let plane = crate::image::Plane { data, row_stride: v.row_stride, pixel_stride: v.pixel_stride };
    crate::image::prepare_frame_into(
        &[plane], v.width, v.height, format, crate::image::YuvMatrix::Bt601Full,
        crate::image::PixelRect { x: crop.0, y: crop.1, width: crop.2, height: crop.3 },
        size.0, size.1, orientation, out,
    )
}

fn to_py_err(e: crate::image::FrameError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}

/// Crops a frame to `crop` = (x, y, width, height), area-averages it to `size` = (width, height),
/// rotates it clockwise by `rotation` degrees and optionally mirrors it.
///
/// `frame` is an (h, w, c) uint8 array ("rgb"/"bgr" need c >= 3, "rgba"/"bgra"/"argb" c >= 4),
/// read in place. Returns an (height, width, 3) uint8 RGB array.
#[pyfunction]
#[pyo3(signature = (frame, crop, size, format="rgb", rotation=0, mirrored=false))]
fn prepare_frame<'py>(
    py: Python<'py>,
    frame: PyReadonlyArray3<'py, u8>,
    crop: (u32, u32, u32, u32),
    size: (u32, u32),
    format: &str,
    rotation: u32,
    mirrored: bool,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let fmt = crate::image::parse_packed_format(format).map_err(to_py_err)?;
    let bpp = if matches!(fmt, crate::image::PixelFormat::Rgb | crate::image::PixelFormat::Bgr) { 3 } else { 4 };
    let view = frame_view(frame.as_array().as_ptr(), frame.shape(), frame.strides(), bpp)?;
    let orientation = crate::image::Orientation { rotation_degrees: rotation, mirrored };
    let mut out = vec![0u8; size.0 as usize * size.1 as usize * 3];
    py.allow_threads(|| prepare_view(&view, fmt, crop, size, orientation, &mut out)).map_err(to_py_err)?;
    let out = numpy::ndarray::Array3::from_shape_vec((size.1 as usize, size.0 as usize, 3), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray3::from_owned_array_bound(py, out))
}

/// Batch version of `prepare_frame` for an (n, h, w, c) video. `crops` is (n, 4) with one
/// (x, y, width, height) per frame, or (1, 4) for the same crop on every frame. Returns
/// (n, height, width, 3).
#[pyfunction]
#[pyo3(signature = (video, crops, size, format="rgb", rotation=0, mirrored=false))]
fn prepare_frames<'py>(
    py: Python<'py>,
    video: PyReadonlyArray4<'py, u8>,
    crops: PyReadonlyArray2<'py, i64>,
    size: (u32, u32),
    format: &str,
    rotation: u32,
    mirrored: bool,
) -> PyResult<Bound<'py, PyArray4<u8>>> {
    let err = |m: &str| pyo3::exceptions::PyValueError::new_err(m.to_string());
    let fmt = crate::image::parse_packed_format(format).map_err(to_py_err)?;
    let bpp = if matches!(fmt, crate::image::PixelFormat::Rgb | crate::image::PixelFormat::Bgr) { 3 } else { 4 };
    let shape = video.shape().to_vec();
    let strides = video.strides().to_vec();
    let n = shape[0];
    if strides[0] < 0 {
        return Err(err("video must have positive strides"));
    }
    let crops = crops.as_array();
    if crops.ncols() != 4 || (crops.nrows() != n && crops.nrows() != 1) {
        return Err(err("crops must have shape (n, 4) or (1, 4)"));
    }
    let mut crop_list = Vec::with_capacity(n);
    for i in 0..n {
        let r = crops.row(if crops.nrows() == 1 { 0 } else { i });
        if r.iter().any(|&v| v < 0 || v > u32::MAX as i64) {
            return Err(err("crop values must be non-negative"));
        }
        crop_list.push((r[0] as u32, r[1] as u32, r[2] as u32, r[3] as u32));
    }
    let base = video.as_array().as_ptr();
    let mut views = Vec::with_capacity(n);
    for i in 0..n {
        // SAFETY: frame i starts i * strides[0] bytes into the borrowed array.
        let ptr = unsafe { base.add(i * strides[0] as usize) };
        views.push(frame_view(ptr, &shape[1..], &strides[1..], bpp)?);
    }
    let orientation = crate::image::Orientation { rotation_degrees: rotation, mirrored };
    let per = size.0 as usize * size.1 as usize * 3;
    let mut out = vec![0u8; n * per];
    py.allow_threads(|| {
        for (i, (v, c)) in views.iter().zip(&crop_list).enumerate() {
            prepare_view(v, fmt, *c, size, orientation, &mut out[i * per..(i + 1) * per])?;
        }
        Ok::<(), crate::image::FrameError>(())
    })
    .map_err(to_py_err)?;
    let out = numpy::ndarray::Array4::from_shape_vec((n, size.1 as usize, size.0 as usize, 3), out)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyArray4::from_owned_array_bound(py, out))
}
