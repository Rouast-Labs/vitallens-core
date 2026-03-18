# JavaScript Integration Guide

`vitallens-core` is compiled to WebAssembly (Wasm) via `wasm-pack`. It provides high-performance signal processing for web-based rPPG applications.

## Get the Artifacts

Build the Wasm package locally:

```bash
make build-web
```

This generates the `pkg/` directory containing the `.wasm` binary and the generated JS/TS glue code.

## Integration

### 1. Initialization
In a modern JS environment (e.g., React, Vue, or vanilla ESM), import and initialize the module:

```javascript
import init, { Session, SessionConfig, getVitalInfo } from './pkg/vitallens_core.js';

async function setup() {
    await init(); // Load the Wasm binary
}
```

### 2. Stateful Usage (Streaming)
The `Session` object manages frame buffers and state natively in Wasm.

```javascript
const config = {
    model_name: "vitallens-2.0",
    supported_vitals: ["heart_rate", "respiratory_rate"],
    fps_target: 30.0,
    input_size: 100,
    n_inputs: 5,
    roi_method: "face",
    return_waveforms: ["ppg_waveform"]
};

const session = new Session(config);

// Inside your camera/processing loop
function onInferenceResult(mlOutput) {
    const input = {
        face: {
            coordinates: [[0.1, 0.1, 0.5, 0.5]],
            confidence: [0.98]
        },
        signals: {
            "ppg_waveform": {
                data: mlOutput.ppg,
                confidence: mlOutput.conf
            }
        },
        timestamp: [performance.now() / 1000]
    };

    // Process in "Incremental" mode to get only new frames
    const result = session.processJs(input, "Incremental");

    if (result.vitals.heart_rate) {
        console.log(`HR: ${result.vitals.heart_rate.value.toFixed(1)} BPM`);
    }
}
```

### 3. Utility Functions
Access metadata and geometry helpers directly:

```javascript
import { calculateRoi } from './pkg/vitallens_core.js';

const faceRect = { x: 100, y: 100, width: 80, height: 120 };
const roi = calculateRoi(faceRect, "forehead", "default", 1920, 1080, true);
```

## Data Structures

* **SessionResult**: Returns `waveforms` and `vitals` as Objects (Maps).
* **Performance**: To minimize overhead, ensure input arrays are standard JS `Float32Array` or regular arrays; the Wasm glue layer handles the memory conversion.
