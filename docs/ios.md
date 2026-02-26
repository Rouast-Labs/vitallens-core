# iOS Integration Guide

`vitallens-core` replaces native Swift signal processing and provides a stateful session to manage frame buffers across your camera pipeline natively via UniFFI.

You can integrate this library into an iOS project using Swift Package Manager (recommended) or by building and linking the artifacts manually.

## Swift Package Manager (Recommended)

This is how `vitallens-core` is integrated into the official `vitallens-ios` client. 

`vitallens-core` is structured as a standalone, remote Swift Package. When resolved, SPM clones this repository to retrieve the UniFFI-generated Swift bindings and downloads the pre-compiled `.xcframework.zip` directly from the tagged GitHub release.

This architecture ensures you do not need a Rust toolchain installed; you simply receive a lightweight, pre-built binary that bridges the shared Rust logic into the iOS environment.

To integrate it, add the repository as a remote dependency in your `Package.swift` and link the `VitalLensCore` product to your targets, e.g.:

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/Rouast-Labs/vitallens-core.git", exact: "0.1.0")
],
targets: [
    .target(
        name: "VitalLensInference",
        dependencies: [
            .product(name: "VitalLensCore", package: "vitallens-core")
        ]
    ),
    // ... other targets linking it similarly
]
```

## Manual Integration

If you prefer to compile the core yourself from the Rust source, follow these steps:

**Get the Artifacts:**

Build the Apple artifacts (see [`CONTRIBUTING.md`](../CONTRIBUTING.md) for environment setup):

```bash
make build-apple
```

This generates:

* `target/VitalLensCoreFFI.xcframework` (The compiled library)
* `bindings/swift/VitalLensCore.swift` (The Swift interface)

**Xcode Integration:**

1. Drag `target/VitalLensCoreFFI.xcframework` into your Xcode project. In your target settings under **Frameworks, Libraries, and Embedded Content**, set it to **Embed & Sign**.
2. Drag `bindings/swift/VitalLensCore.swift` directly into your Xcode project's source tree.

## Usage Pattern

In you iOS App, you may be running a loop using `AVFoundation`. The Core handles the overlapping history, buffering, and math.

```swift
import AVFoundation
import VitalLensCore

class VitalLensProcessor {
    private var session: Session

    init() {
        let config = SessionConfig(
            modelName: "vitallens-2.0",
            supportedVitals: ["heart_rate", "hrv_sdnn"],
            fpsTarget: 30.0,
            inputSize: 100,
            nInputs: 5,
            roiMethod: "face",
            returnWaveforms: nil
        )
        self.session = Session(config: config)
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        
        // Example ML inference output
        let ppgData: [Float] = [0.12, 0.15, 0.14, 0.11, 0.09]
        let ppgConf: [Float] = [0.98, 0.98, 0.97, 0.99, 0.98]
        let ppgSignal = SignalInput(data: ppgData, confidence: ppgConf)
        
        // Construct the input
        let input = SessionInput(
            face: FaceInput(
                coordinates: [[0.1, 0.1, 0.5, 0.5], [0.1, 0.1, 0.5, 0.5]], 
                confidence: [0.99, 0.98]
            ),
            signals: ["ppg_waveform": ppgSignal],
            timestamp: [CACurrentMediaTime(), CACurrentMediaTime() + 0.033]
        )

        // Process incrementally
        let result = session.process(input: input, mode: .incremental)
        
        if let hr = result.vitals["heart_rate"]?.value {
            print("Current Heart Rate: \(hr) BPM")
        }
    }
}
```
