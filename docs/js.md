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
import { calculateRoi, coreVersion } from './pkg/vitallens_core.js';

const faceRect = { x: 100, y: 100, width: 80, height: 120 };
const roi = calculateRoi(faceRect, "forehead", "default", 1920, 1080, true);
const version = coreVersion(); // e.g. "0.4.2"
```

### 4. Frame Preparation

Crops a frame to the ROI, converts it to RGB, area-averages it down to the model input size, then rotates and mirrors it, in one pass. Each output pixel is the exact mean of the source pixels it covers, with partially covered pixels weighted by their covered fraction (like OpenCV `INTER_AREA`); integer arithmetic gives identical bytes on every platform. The crop is in pixels of the raw (unrotated) frame, the output size is the size after rotation, rotation is clockwise (0, 90, 180, 270), then mirroring. The result is packed RGB bytes (`out_width * out_height * 3`, row-major).

Formats: `"rgba"` (canvas `ImageData`), `"bgra"`, `"argb"`, `"rgb"`, `"bgr"` (one plane).

The bytes are copied into wasm memory once per call. In the browser, a GPU path (such as TF.js `avgPool`) can be cheaper than reading pixels back to the CPU.

```javascript
import { prepareFrame } from './pkg/vitallens_core.js';

const img = ctx.getImageData(0, 0, width, height);
const rgb = prepareFrame(
  img.data, width, height, width * 4, "rgba",
  100, 60, 300, 300, // crop x, y, width, height
  40, 40,            // output width, height
  0, false           // rotation (degrees clockwise), mirrored
); // Uint8Array of 40 * 40 * 3
```

## Data Structures

* **SessionResult**: Returns `waveforms` and `vitals` as Objects (Maps).
* **Performance**: To minimize overhead, ensure input arrays are standard JS `Float32Array` or regular arrays; the Wasm glue layer handles the memory conversion.
