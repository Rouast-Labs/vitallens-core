# Kotlin / Android Integration Guide

`vitallens-core` replaces native Kotlin signal processing and provides a stateful session to manage frame buffers across your camera pipeline natively via UniFFI.

You can integrate this library into a Kotlin project from Maven Central (recommended) or by building the artifacts yourself.

## Maven Central (Recommended)

This is how `vitallens-core` is integrated into the official `vitallens-android` client.

Two artifacts are published under `com.rouast`, with the same Kotlin API (package `com.rouast.vitallens.core`):

* `vitallens-core`: Android AAR with the native libraries for `arm64-v8a`, `armeabi-v7a`, `x86_64` and `x86`.
* `vitallens-core-jvm`: plain JVM jar with macOS and Linux native libraries, for host-side unit tests.

No Rust toolchain is needed. The generated bindings call into the native library through JNA, which comes in as a dependency.

```kotlin
// build.gradle.kts
dependencies {
    implementation("com.rouast:vitallens-core:0.4.2")
    testImplementation("com.rouast:vitallens-core-jvm:0.4.2")
}
```

## Manual Integration

If you prefer to compile the core yourself from the Rust source, set up the Android targets, `cargo-ndk` and the NDK (see [`CONTRIBUTING.md`](../CONTRIBUTING.md)), then:

```bash
make build-android   # AAR in kotlin/android/build/outputs/aar/
make build-jvm       # JVM jar for host-side tests
```

The generated Kotlin bindings land in `bindings/kotlin/`.

## Usage Pattern

In your Android app you may be running a CameraX `ImageAnalysis` loop. The Core handles the overlapping history, buffering, and math. `Session` holds native memory: close it when done (it is `AutoCloseable`).

```kotlin
import com.rouast.vitallens.core.*

class VitalLensProcessor : AutoCloseable {
    private val session = Session(
        SessionConfig(
            modelName = "vitallens-2.0",
            supportedVitals = listOf("heart_rate", "hrv_sdnn"),
            returnWaveforms = null,
            fpsTarget = 30f,
            inputSize = 40uL,
            nInputs = 5uL,
            roiMethod = "face",
        )
    )

    fun onInference(timestamps: List<Double>) {
        // Example ML inference output
        val ppg = SignalInput(
            data = listOf(0.12f, 0.15f, 0.14f, 0.11f, 0.09f),
            confidence = listOf(0.98f, 0.98f, 0.97f, 0.99f, 0.98f),
        )
        val input = SessionInput(
            face = FaceInput(
                coordinates = List(5) { listOf(0.1f, 0.1f, 0.5f, 0.5f) },
                confidence = List(5) { 0.99f },
            ),
            signals = mapOf("ppg_waveform" to ppg),
            timestamp = timestamps,
        )

        // Process incrementally
        val result = session.process(input, WaveformMode.Incremental)
        result.vitals["heart_rate"]?.let { println("Current Heart Rate: ${it.value} BPM") }
    }

    override fun close() = session.close()
}
```

## Utility Functions

```kotlin
val roi = calculateRoi(
    face = Rect(x = 100f, y = 100f, width = 80f, height = 120f),
    method = RoiMethod.Face,
    detector = FaceDetector.DEFAULT,
    containerWidth = 1280f,
    containerHeight = 720f,
    forceEven = true,
)
val info = getVitalInfo("heart_rate")
val version = coreVersion() // e.g. "0.4.2"
```

## Frame Preparation

Crops a frame to the ROI, converts it to RGB, area-averages it down to the model input size, then rotates and mirrors it, in one pass. Each output pixel is the exact mean of the source pixels it covers, with partially covered pixels weighted by their covered fraction (like OpenCV `INTER_AREA`); integer arithmetic gives identical bytes on every platform. The crop is in pixels of the raw (unrotated) frame, the output size is the size after rotation, rotation is clockwise (0, 90, 180, 270), then mirroring. The result is packed RGB bytes (`out_width * out_height * 3`, row-major).

Formats: `RGBA`, `BGRA`, `ARGB`, `RGB`, `BGR` (one plane) and `YUV420` (Y, U, V planes with any row and pixel strides, as in `YUV_420_888` and `MediaCodec` output images). Use `YuvMatrix.BT601_FULL` for camera frames and `BT601_LIMITED` or `BT709_LIMITED` for decoded video.

`prepareFrame` reads the planes in place from direct `ByteBuffer`s, without copying the frame: keep the image open until the call returns. Prefer `YUV_420_888` frames over RGBA: they read 1.5 instead of 4 bytes per pixel, and only the crop is read.

```kotlin
// image: an open ImageProxy (CameraX, YUV_420_888) or MediaCodec Image
val rgb: ByteArray = prepareFrame(
    planes = image.planes.map { FramePlaneBuffer(it.buffer, it.rowStride, it.pixelStride) },
    width = image.width, height = image.height,
    format = PixelFormat.YUV420, yuvMatrix = YuvMatrix.BT601_FULL,
    crop = PixelRect(x = 100u, y = 60u, width = 300u, height = 300u),
    outWidth = 40, outHeight = 40,
    orientation = Orientation(rotationDegrees = 270u, mirrored = true),
)
```

`prepareFrame` is hand-written (`kotlin/src`) and calls the native library through JNA direct mapping. The generated `prepareFrameFromAddresses` and `prepareFrameFromPlanes` do the same, but UniFFI passes their records through JNA structures, which costs about 0.8 ms per call on a mid-range phone, more than the frame itself.
