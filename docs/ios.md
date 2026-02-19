# iOS Implementation Guide (UniFFI)

The iOS client uses `vitallens-core` as a compiled `XCFramework`. It replaces native Swift signal processing and provides a stateful session to manage frame buffers across your camera pipeline.

## Building the Framework

Instead of manually running cargo and `xcodebuild` commands, use the automated makefile to generate the framework and Swift bindings.

From the root of `vitallens-core`, run:

```bash
make build-apple
```

This will output two components:

- **`target/VitaLensCore.xcframework`**: The bundled static libraries for both physical iPhones and Simulators.
- **`bindings/swift/VitalLensCore.swift`**: The UniFFI-generated Swift interface.

## Integration into Xcode

To use the core in your `vitallens-ios` app:

1. **Add the Framework:** Drag and drop `target/VitalLensCore.xcframework` into your Xcode project. Ensure it is added to the **"Frameworks, Libraries, and Embedded Content"** section of your target and set to "Embed & Sign".
2. **Add the Bindings:** Drag `bindings/swift/VitalLensCore.swift` directly into your Xcode project's source tree.
3. **Import:** You can now access the Rust engine natively in your Swift files using `import VitalLensCore`.

## Usage Pattern: Real-time Analysis

In `vitallens-ios`, you typically run a loop using `AVFoundation`. The Core handles the overlapping history, buffering, and DSP.

```swift
import AVFoundation
import VitalLensCore

class VitalLensProcessor {
    private var session: Session

    init() {
        // 1. Initialize the Session
        let config = SessionConfig(
            supportedVitals: ["heart_rate", "hrv_sdnn"],
            fpsTarget: 30.0,
            inputSize: 100,
            nInputs: 5,
            roiMethod: "face"
        )
        self.session = Session(config: config)
    }

    // 2. Process frame chunks
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        // Example: Perform ML Inference (CoreML) to get your raw PPG/Resp signals
        let signals: [String: [Float]] = ["ppg_waveform": [0.12, 0.15, 0.14, 0.11, 0.09]]
        let confidence: [String: [Float]] = ["ppg_waveform": [0.98, 0.98, 0.97, 0.99, 0.98]]
        
        let chunk = InputChunk(
            timestamp: [CACurrentMediaTime(), /* ... */],
            signals: signals,
            confidences: confidence,
            face: FaceInput(coordinates: [0.1, 0.1, 0.5, 0.5], confidence: 0.99)
        )

        // 3. Get smoothed/derived results
        let result = session.processChunk(chunk: chunk, mode: .incremental)
        
        if let hr = result.signals["heart_rate"]?.value {
            print("Current Heart Rate: \(hr) BPM")
            // updateUI(bpm: hr)
        }
    }
}
```