# Web/JS Implementation Guide (Wasm)

The Web client (`vitallens.js`) uses WebAssembly (Wasm) to run the `vitallens-core` math engine directly in the browser at near-native speeds. It uses `serde-wasm-bindgen` to seamlessly pass JavaScript objects into Rust.

## Building the Package

To compile the Rust code into WebAssembly and generate the necessary JavaScript/TypeScript glue code, use the automated makefile. 

From the root of `vitallens-core`, run:

```bash
make build-web
```

This compiles the codebase and creates a `pkg/` directory. This directory behaves like a standard NPM package, containing:

- `vitallens_core_bg.wasm` (The binary)
- `vitallens_core.js` (The ES Module interface)
- `vitallens_core.d.ts` (TypeScript types)
- `package.json`

## Integration into your Web Client

Because the `pkg/` folder acts as an NPM package, you can install it directly into your `vitallens.js` project:

```bash
# From your vitallens.js project root:
npm install ../path/to/vitallens-core/pkg
```

*(Alternatively, you can just copy the `pkg/` folder directly to your web server for vanilla HTML/JS usage).*

## Usage Pattern: Browser Processing

Because Wasm handles memory boundaries strictly, we pass plain JSON objects from JS, and the bindings handle the deserialization into Rust structs internally.

```javascript
import init, { Session } from 'vitallens_core';

async function startVitalLens() {
    // 1. Initialize the Wasm memory module
    await init(); 

    // 2. Define your configuration
    const config = {
        model_name: "vitallens-2.0",
        supported_vitals: ["heart_rate", "respiratory_rate"],
        fps_target: 30.0,
        input_size: 100,
        n_inputs: 5,
        roi_method: "face"
    };

    const session = new Session(config);

    // 3. Inside your video loop
    function processFrame(videoTimestamp, ppgArray) {
        const input = {
            timestamp: [videoTimestamp], // Must be an array of numbers
            signals: { 
                "ppg_waveform": { 
                    data: ppgArray, 
                    confidence: [0.95] // Must match length of data
                } 
            },
            face: { 
                coordinates: [[0.1, 0.1, 0.2, 0.2]], // Array of bounding boxes
                confidence: [1.0]                    // Array of confidences
            }
        };

        const result = session.processJs(input, "Incremental");

        if (result.signals.heart_rate && result.signals.heart_rate.value) {
            console.log(`Heart Rate: ${result.signals.heart_rate.value} BPM`);
        }
    }
}

startVitalLens();
```