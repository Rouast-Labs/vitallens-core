# VitalLens Core: Project Blueprint

**Status:** Inception / Scaffolding
**Goal:** Create a "Universal Core" for rPPG signal processing that ensures mathematical parity across Python (Backend), iOS (Mobile), and Web (JS/Wasm).

## 1. The "Three-Body" Problem & Solution

We currently maintain three separate implementations of rPPG logic:

1. **Python (`prpy`):** Uses `numpy`/`scipy`. Used for `vitallens-python` client, as well as Training & Backend Inference.
1.1. **Python (`vitallens-python`):** Uses `prpy`. Mostly for file analysis (ex-post) but would be interested in adding real-time later.
3. **iOS (`vitallens-ios`):** Uses Swift `vDSP` / Accelerate. Used for real-time mobile. Can also analyze files.
4. **Web (`vitallens.js`):** Uses TensorFlow.js / JS. Used for browser demos. Can also analyze files.

**The Solution:** `vitallens-core`. A single Rust library that compiles to:

* **Python Extension (`.so`):** Via `PyO3`. (Partly) replaces `prpy` math/integrates in `prpy`.
* **iOS XCFramework (`.a`):** Via `UniFFI`. Replaces `SignalOps.swift`, `VitalsEstimateManager.swift`, `FrameBuffer.swift`, `BufferManager.swift`.
* **WebAssembly (`.wasm`):** Via `wasm-bindgen`. Replaces `physio.ts`, `VitalsEstimateManager.ts`, maybe `FrameBuffer.ts` and `BufferManager.ts`.

## 2. Scope of Responsibilities

The Core is **strictly** a mathematical and state engine. It does **not** handle I/O.

| **IN SCOPE (Rust)** | **OUT OF SCOPE (Keep Native)** |
| --- | --- |
| **Signal Processing:** Detrending, Smoothing, Normalization. | **Camera/Video:** `AVFoundation`, `OpenCV`, `getUserMedia`. |
| **Math:** FFT, Peak Detection, HRV Metrics (SDNN, RMSSD). | **Networking:** API calls, HTTP clients. |
| **State Management:** Rolling circular buffers, windowing logic. | **UI/UX:** Rendering graphs or camera previews. |
| **Business Logic:** "Vital Registry" (Min/Max limits, Units). | **Inference:** Running the ONNX/CoreML model itself. |

## 3. The "Polyglot" Directory Structure

We plan to use a specific structure to support **both** PyO3 (for high-perf NumPy access) and UniFFI (for easy Mobile bindings) without them conflicting.

I need to find out if this is possible as laid out here.

```text
vitallens-core/
├── Cargo.toml                 # Master config (pyo3, uniffi, rustfft, ndarray)
├── uniffi.toml                # Mobile binding config
├── pyproject.toml             # Maturin config for Python wheel
├── src/
│   ├── lib.rs                 # Library Entry Point (Exports modules)
│   │
│   ├── signal/                # [CORE] The Pure Math Engine
│   │   ├── mod.rs
│   │   ├── filters.rs         # Detrending (Tarvainen-Valtonen), Moving Average
│   │   ├── fft.rs             # Periodogram / Welch method
│   │   ├── peaks.rs           # Adaptive Z-Score peak detection
│   │   └── vitals.rs          # HR, RR, SDNN, RMSSD, LF/HF logic
│   │
│   ├── state/                 # [STATEFUL] The Logic Engine
│   │   ├── mod.rs
│   │   └── analyzer.rs        # High-level logic (The "Vitals Estimate Manager")
│   │
│   ├── buffering/             # [STATEFUL] The "Smart Storage"
│   │   ├── mod.rs
│   │   ├── manager.rs         # BufferManager (IoU tracking logic)
│   │   └── buffer.rs          # FrameBuffer (Circular queue with n-inputs overlap)
│   │
│   ├── geometry/              # [CORE] Replaces parts of `prpy.numpy.face`
│   │   ├── rect.rs            # Normalized Rect struct & IoU math 
│   │   └── strategies.rs      # Face box to different strategies calculations (upper body, forehead, ...)
│   │
│   ├── registry.rs            # [METADATA] Single source of truth for Limits/Units
│   │
│   ├── bindings/              # [GLUE] Language Adapters
│   │   ├── mod.rs
│   │   └── python.rs          # PyO3 exports (Direct NumPy access)
│   │
│   └── mobile.rs              # [GLUE] UniFFI exports (Swift/Kotlin interface)

```

## 4. The API Surface

This is the contract the next agent needs to implement.

**Stateless (For Python/Backend):**

```rust
// src/signal/mod.rs
pub fn process_batch(
    signal: &[f32], 
    fs: f32, 
    vital_type: VitalType // Enum: HeartRate, HRV, etc.
) -> VitalResult

```

**Stateful (For Mobile/Web):**

TODO: I need a better name than "Analyzer"

```rust
// src/state/analyzer.rs
pub struct VitalLensAnalyzer {
    // Hidden state
}

impl VitalLensAnalyzer {
    pub fn new(fs: f32) -> Self;
    
    // Push new frame data (e.g. from camera callback)
    pub fn push_frame(&mut self, timestamp: f64, value: f32, confidence: f32);
    
    // Get latest calculated results (e.g. for UI update)
    pub fn get_estimates(&self) -> HashMap<String, VitalResult>;
}

```