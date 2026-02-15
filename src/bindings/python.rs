#![cfg(feature = "python")]

use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use crate::signal;
use crate::signal::rate::{RateBounds, RateStrategy};
use crate::registry::HrvMetric;
use crate::signal::peaks::SignalBounds;

pub fn register_functions(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate_heart_rate, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_hrv_sdnn, m)?)?;
    Ok(())
}

/// Tier 2: Stateless Heart Rate Estimation
#[pyfunction]
fn estimate_heart_rate(_py: Python, signal: PyReadonlyArray1<f32>, fs: f32) -> PyResult<(f32, f32)> {
    let s = signal.as_slice()?;
    let bounds = RateBounds { min: 40.0, max: 200.0 }; 
    let strategy = RateStrategy::Periodogram { target_res_hz: 0.01 };
    
    let res = signal::estimate_rate(s, fs, bounds, strategy, None, None);
    
    Ok((res.value, res.confidence))
}

/// Tier 2: Stateless SDNN
#[pyfunction]
fn estimate_hrv_sdnn(_py: Python, signal: PyReadonlyArray1<f32>, fs: f32) -> PyResult<(f32, f32)> {
    let s = signal.as_slice()?;
    
    let bounds = SignalBounds { min_rate: 40.0, max_rate: 220.0 };
    let rate_hint = None;

    let (val, conf) = signal::estimate_hrv(
        s, 
        fs, 
        HrvMetric::Sdnn, 
        &[], 
        &[], 
        bounds, 
        rate_hint
    );
    
    Ok((val, conf))
}