# VitalLens Core

**Status:** Implementation
**Goal:** A "Universal Core" for rPPG signal processing ensuring mathematical parity across Python (Backend), iOS (Mobile), and Web (JS/Wasm).

## Library Purpose

This library is a single Rust codebase that compiles into three targets:

1.  **Python (`.so`):** Replaces `prpy` math. Built via `maturin`.
2.  **iOS (`.a` / `.swift`):** Replaces `SignalOps.swift`. Built via `cargo` + `uniffi`.
3.  **Web (`.wasm`):** Replaces `physio.ts`. Built via `wasm-pack`.

## Directory Structure

```text
vitallens-core/
├── src/
│   ├── lib.rs                 # Entry point (Feature-gated modules)
│   ├── signal/                # Pure Math (DSP, FFT, Peak Detection)
│   ├── state/                 # Stateful Logic (Buffers, Analyzer)
│   └── bindings/              # Language Glue
│       ├── python.rs          # PyO3 Bindings
│       └── swift/             # Generated Swift code (Output)
├── Cargo.toml                 # Master config
├── Makefile                   # Universal build runner
├── pyproject.toml             # Python build config
└── uniffi.toml                # Mobile binding config
```

## Build Instructions

**Quick Start:**

Run the universal makefile to test all targets:

```bash
make
```

### 1. Python (Local Development)

Builds the Rust core as a Python extension and installs it into your current environment.

```bash
make check-python
# Or alternatively
maturin develop --features python
```

### 2. iOS (Native Mobile)

**Step A: Build Libraries**

Build the static library for both Simulator (x86_64) and Device (arm64).

```bash
make check-ios
```

Or run specific builds:

```
# For Simulator (Intel Mac)
cargo build --release --target x86_64-apple-ios --lib

# For Simulator (Apple Silicon)
cargo build --release --target aarch64-apple-ios-sim --lib

# For Device (iPhone)
cargo build --release --target aarch64-apple-ios --lib
```

**Step B: Generate Swift Bindings**

```bash
# Generates bindings/swift/VitallensCore.swift
cargo run --features=uniffi/cli --bin uniffi-bindgen generate \
    --library target/x86_64-apple-ios/release/libvitallens_core.dylib \
    --language swift \
    --out-dir bindings/swift
```

### 3. Web (Wasm)

Compiles to WebAssembly for use in the browser.

```bash
make check-web
# Or alternatively
wasm-pack build --target web --no-default-features
```

## Architecture Notes

- **Stateless Signal Processing:** `src/signal/` contains pure functions (input -> output) with no side effects.
- **State Management:** `src/state/` handles buffering, windowing, and history.
- **Polyglot Strategy:** We use `#[cfg(feature = "python")]` and `#[cfg(not(target_arch = "wasm32"))]` to prevent platform-specific code (like `libc` or `Python.h`) from breaking builds on other targets.
