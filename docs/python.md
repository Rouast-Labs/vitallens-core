# Python Implementation Guide (PyO3)

In Python, the library is used for two purposes: high-performance stateful sessions in applications, and stateless math in research scripts.

## Stateful Usage (Apps)

Used in the backend or desktop tools to process streaming data.

```python
import vitallens_core as vc

config = vc.SessionConfig(
    supported_vitals=["heart_rate", "hrv_rmssd"],
    fps_target=30.0,
    input_size=100,
    n_inputs=10,
    roi_method="face"
)
session = vc.Session(config)

# Incremental processing
result = session.process_chunk(chunk, mode="Incremental")
```

## Stateless Usage (Research/Jupyter)

Direct access to the DSP engine without the state machine.

```python
import numpy as np
import vitallens_core as vc

# Example: Estimate HR from a raw numpy array
signal = np.random.randn(300).astype(np.float32)
bpm, conf = vc.estimate_heart_rate(signal, fs=30.0)

# Example: Find peaks with refined quadratic interpolation
peaks = vc.find_peaks(signal, fs=30.0, refine=True)
```