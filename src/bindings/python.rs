#![cfg(feature = "python")]

use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use crate::signal;

pub fn register_functions(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_signal_py, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(name = "process_batch")]
fn process_signal_py(_py: Python, signal: PyReadonlyArray1<f32>, fs: f32) -> PyResult<f32> {
    let signal_slice = signal.as_slice()?;
    Ok(signal::process_batch(signal_slice, fs))
}
