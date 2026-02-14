# VitalLens Core: Project Blueprint

**Status:** Implementation / Alpha
**Goal:** A "Universal Core" for rPPG signal processing ensuring mathematical parity across Python (Backend), iOS (Mobile), and Web (JS/Wasm).

## 1. The "Three-Body" Problem & Solution

We maintain three separate implementations of rPPG logic:

1. **Python:** Used for Training (`prpy`) & Client Analysis (`vitallens-python`).
2. **iOS:** Used for real-time mobile (`vitallens-ios`).
3. **Web:** Used for browser demos (`vitallens.js`).

**The Solution:** `vitallens-core`. A single Rust library that compiles to:

* **Python Extension (`.so`):** Via `PyO3`. Replaces `prpy` math and provides both stateful sessions and direct stateless access for research.
* **iOS XCFramework (`.a`):** Via `UniFFI`. Replaces `SignalOps.swift` and `BufferManager.swift`.
* **WebAssembly (`.wasm`):** Via `wasm-bindgen`. Replaces `physio.ts` and `VitalsEstimateManager.ts`.

## 2. Scope of Responsibilities

The Core is **strictly** a mathematical and state engine. It does **not** handle I/O.

| **IN SCOPE (Rust)** | **OUT OF SCOPE (Keep Native)** |
| --- | --- |
| **Signal Processing:** Detrending, Smoothing, Normalization (Z-Score). | **Camera/Video:** `AVFoundation`, `OpenCV`, `getUserMedia`. |
| **Math:** FFT (Periodogram), Adaptive Peak Detection, HRV Metrics (SDNN, RMSSD, LF/HF), Blood Pressure estimation. | **Networking:** API calls, HTTP clients. |
| **State Management:** Rolling circular buffers, timestamp merging, confidence gating. | **UI/UX:** Rendering graphs or camera previews. |
| **Business Logic:** "Vital Registry" (Signal dependencies, Min/Max limits, Units). | **Inference:** Running the ONNX/CoreML model itself. |

## 3. The "Two-Tier" Architecture

To support both real-time apps and offline research, the library exposes two distinct API tiers.

### Tier 1: The Stateful Session (Apps & Real-time)

* **Target:** iOS, Web, Python Apps.
* **Concept:** The `Session` object is the single source of truth. It manages history, stitching, buffering, and emitting results only when enough high-quality data is accumulated.
* **Implementation Pattern:** "Shell & Core". The logic resides in a private `SessionCore`, wrapped by a public `Session` shell using `Mutex` to satisfy FFI thread-safety requirements (Arc/Interior Mutability).

### Tier 2: Stateless Math (Research & Backend)

* **Target:** Python (Jupyter/Training).
* **Concept:** Direct access to low-level signal processing functions without the overhead of a session state machine.
* **Usage:** `vitallens_core.estimate_heart_rate(signal_array, fs)`

## 4. Directory Structure

The structure separates pure math (`signal`), state management (`state`), configuration (`registry`), and language bindings (`bindings`).

```text
vitallens-core/
├── Cargo.toml                 # Configures dependencies (pyo3, uniffi, wasm-bindgen)
├── uniffi.toml                # Mobile binding configuration
├── pyproject.toml             # Maturin config for Python wheel
├── src/
│   ├── lib.rs                 # Library Entry Point (Exports Session & Modules)
│   │
│   ├── signal/                # [TIER 2] Pure Math Engine (Stateless)
│   │   ├── mod.rs
│   │   ├── filters.rs         # Detrending (Tarvainen-Valtonen), Moving Average, Z-Score
│   │   ├── fft.rs             # Welch Periodogram implementation
│   │   ├── peaks.rs           # Adaptive Z-Score peak detection
│   │   ├── rate.rs            # Generic rate estimator (FFT vs Peak strategies)
│   │   ├── hrv.rs             # HRV metrics (SDNN, RMSSD, LF/HF) with interpolation
│   │   └── bp.rs              # Blood Pressure estimation logic
│   │
│   ├── state/                 # [TIER 1] The Logic Engine (Stateful)
│   │   ├── mod.rs
│   │   ├── session.rs         # The "Session" State Machine (Shell & Core pattern)
│   │   └── buffers.rs         # Smart circular buffers with overlap merging
│   │
│   ├── registry.rs            # [METADATA] Single source of truth for Limits, Units, & Dependencies
│   │
│   ├── types.rs               # [DATA] Cross-boundary structs (ModelConfig, InputChunk, SessionResult)
│   │
│   ├── bindings/              # [GLUE] Language Adapters
│   │   ├── mod.rs
│   │   └── python.rs          # PyO3 exports for stateless functions
│   │
│   └── mobile.rs              # [GLUE] UniFFI exports (if needed beyond macros)
```

## 5. API Surface

### 5.1. Stateful Session (iOS, Web, Python App)

This is the primary interface.

**Configuration:**

```rust
// Defined in src/types.rs
pub struct ModelConfig {
    pub name: String,
    pub supported_vitals: Vec<String>, // e.g. ["heart_rate", "hrv_sdnn"]
    pub fps_target: f32,
    pub input_size: u64,
    pub roi_method: String,
}
```

**The Session:**

```rust
// Defined in src/state/session.rs
pub struct Session { ... }

impl Session {
    // Constructor
    // iOS/Python: Session(config)
    // Web: new Session(json_config)
    pub fn new(config: ModelConfig) -> Self;

    // Main Loop
    // Takes a chunk of data (signals, timestamps, face info) and returns calculated results.
    // Handles overlap stitching and buffer management internally.
    pub fn process_chunk(&self, chunk: InputChunk, mode: WaveformMode) -> SessionResult;
}
```

**Output:**

```rust
// Defined in src/types.rs
pub struct SessionResult {
    pub timestamp: Vec<f64>,
    pub signals: HashMap<String, SignalResult>, // Keyed by vital ID (e.g. "heart_rate")
    pub face: Option<FaceResult>,
    pub fps: f32,
    ...
}
```

### 5.2. Stateless Math (Python Research Only)

Exposed via `vitallens_core` module in Python.

```python
# Direct NumPy access
import vitallens_core

# Heart Rate
bpm, conf = vitallens_core.estimate_heart_rate(signal_array, fs)

# HRV
sdnn, _ = vitallens_core.estimate_hrv_sdnn(signal_array, fs)
rmssd, _ = vitallens_core.estimate_hrv_rmssd(signal_array, fs)
```

## 6. Build Targets & Tooling

| Platform | Tooling | Output | Integration |
| --- | --- | --- | --- |
| **Python** | `maturin` + `PyO3` | `.whl` / `.so` | `pip install vitallens_core` |
| **iOS** | `cargo` + `uniffi` | `.a` + `.swift` | XCFramework / Swift Package |
| **Web** | `wasm-pack` + `serde` | `.wasm` + `.js` | NPM Package / ES Module |

## 7. Key Architectural Decisions

1. **"Shell & Core" Pattern:** The `Session` struct exposed to FFI wraps a private `SessionCore` struct inside a `Mutex`. This satisfies UniFFI's thread-safety requirements (`Arc<T>`) while allowing the internal logic to use mutable state (`&mut self`) freely.
2. **Registry-Driven Derivations:** The logic for *how* to calculate a vital (e.g., "Heart Rate comes from PPG using FFT, requires 5s of data") is defined data-structurally in `registry.rs`, not hard-coded in the session logic.
3. **Strict Type Separation:** `src/types.rs` defines clean DTOs (Data Transfer Objects) marked with `Serialize`/`Deserialize` (for JS) and `uniffi::Record`/`pyclass` (for Native/Python). This ensures data passes cleanly across all three language boundaries.
4. **JS Serialization:** WebAssembly bindings use `serde-wasm-bindgen` to pass plain JSON objects, bypassing the strict class limitations of `wasm_bindgen` for complex types like HashMaps.