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

# Constructing the inputs
ppg_signal = vc.SignalInput(
    data=[0.1, 0.2, 0.3], 
    confidence=[1.0, 1.0, 1.0]
)

face_input = vc.FaceInput(
    coordinates=[[0.1, 0.1, 0.5, 0.5], [0.1, 0.1, 0.5, 0.5], [0.1, 0.1, 0.5, 0.5]],
    confidence=[0.9, 0.9, 0.9]
)

input = vc.SessionInput(
    face=face_input,
    signals={"ppg_waveform": ppg_signal},
    timestamp=[1.0, 1.033, 1.066]
)

# Incremental processing
result = session.process(input, mode="Incremental")
```