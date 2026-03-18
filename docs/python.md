# Python Integration Guide

In Python, `vitallens-core` operates as a native C-extension via PyO3. It is used in two primary ways: high-performance stateful sessions for streaming applications, and stateless, direct mathematical operations for research scripts.

## Get the Artifacts

Build and install the Python bindings directly into your active virtual environment (see [`CONTRIBUTING.md`](../CONTRIBUTING.md) for full setup instructions):

```bash
# Fast local install for development
maturin develop --features python

# Or build the release wheel via Makefile
make build-python
```

## Stateful Usage (Streaming & Apps)

Use the `Session` object to manage overlapping frame buffers, history, and real-time computation in a backend or desktop app.

```python
import vitallens_core as vc

# 1. Configuration
config = vc.SessionConfig(
    model_name="vitallens-2.0",
    supported_vitals=["heart_rate", "hrv_rmssd"],
    fps_target=30.0,
    input_size=100,
    n_inputs=10,
    roi_method="face",
    return_waveforms=["ppg_waveform"]
)
session = vc.Session(config)

# 2. Construct inputs
ppg_signal = vc.SignalInput(
    data=[0.1, 0.2, 0.3], 
    confidence=[1.0, 1.0, 1.0]
)

face_input = vc.FaceInput(
    coordinates=[[0.1, 0.1, 0.5, 0.5], [0.1, 0.1, 0.5, 0.5], [0.1, 0.1, 0.5, 0.5]],
    confidence=[0.9, 0.9, 0.9]
)

input_data = vc.SessionInput(
    face=face_input,
    signals={"ppg_waveform": ppg_signal},
    timestamp=[1.0, 1.033, 1.066]
)

# 3. Process incrementally
result = session.process(input_data, mode="Incremental")

if "heart_rate" in result.vitals:
    print(f"HR: {result.vitals['heart_rate'].value} BPM")
```

## Stateless Usage (Research & Scripts)

The library exposes individual signal processing functions. These take `numpy` arrays directly and are useful for evaluating metrics on complete, pre-extracted signals.

```python
import numpy as np
import vitallens_core as vc

fs = 30.0
signal = np.sin(np.linspace(0, 10 * 2 * np.pi, 300)).astype(np.float32)
confidence = np.ones_like(signal).astype(np.float32)

# Rate estimation
hr_val, hr_conf = vc.estimate_heart_rate(signal, fs)
print(f"Heart Rate: {hr_val:.1f} BPM (Confidence: {hr_conf:.2f})")

# HRV estimation
# Supported metrics: sdnn, rmssd, lfhf, si, pnn50, sd1sd2
sdnn_val, sdnn_conf = vc.estimate_hrv_metric(
    signal, 
    fs, 
    metric_name="sdnn", 
    confidence=confidence, 
    rate_hint=hr_val
)
print(f"SDNN: {sdnn_val:.1f} ms")

# Peak detection
peaks = vc.find_peaks(
    signal=signal,
    fs=fs,
    refine=True,
    rate_hint=hr_val,
    min_rate=40.0,
    max_rate=220.0,
    detection_threshold=0.45,
    window_cycles=2.5,
    max_rate_change=1.0
)
print(f"Detected {len(peaks)} peaks at indices: {peaks[:3]}...")

# Geometry & ROI
face_rect = vc.Rect(100.0, 100.0, 80.0, 120.0)
roi = vc.calculate_roi(
    face=face_rect, 
    method="upper_body", 
    detector="default", 
    container=(1920.0, 1080.0),
    force_even=True
)
print(f"Calculated ROI: x={roi.x}, y={roi.y}, w={roi.width}, h={roi.height}")

# Metadata
meta = vc.get_vital_info("heart_rate")
print(f"Vital ID: {meta.id}, Display Name: {meta.display_name}, Unit: {meta.unit}")
```