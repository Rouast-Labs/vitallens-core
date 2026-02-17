# Contributing & Development Guide

This document serves as a cheat sheet for setting up the development environment, running tests, and building targets for **VitalLens Core**.

## Environment Setup

### Prerequisites

Ensure the following toolchains are installed on your machine (macOS assumed).

**1. Rust & Cargo**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

**2. Python (Virtual Env & Maturin)**

```bash
# Install Python 3.10+
brew install python@3.10

# Install Maturin globally or in venv
pip3 install maturin
```

**3. WebAssembly (wasm-pack)**

```bash
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

**4. iOS Targets (Rust)**

Add the necessary compilation targets for iOS development:

```bash
rustup target add aarch64-apple-ios
rustup target add aarch64-apple-ios-sim
rustup target add x86_64-apple-ios
```

---

## Quick Command Reference (Makefile)

The project includes a `Makefile` to automate common tasks.

| Command | Description |
| --- | --- |
| `make` | Runs all tests and checks builds for Python, iOS, and Web. |
| `make test` | Runs standard Rust unit tests (`cargo test`). |
| `make check-python` | Builds the Python extension wheel (Dry run). |
| `make check-ios` | Compiles the static libraries (`.a`) for iOS. |
| `make check-web` | Compiles the WebAssembly package. |
| `make clean` | Removes all build artifacts (`target/`, `pkg/`, etc.). |

---

## Detailed Workflows

### 🦀 Core Rust Logic

**Run Unit Tests:**

```bash
cargo test
```

**Run Specific Test:**

```bash
cargo test --test session_tests
# Or a specific function
cargo test test_logic_irregular_rhythm
# Show print statements (stdout) during tests:
cargo test -- --nocapture
```

**Format & Lint:**

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
```

---

### 🐍 Python Bindings (PyO3 + Maturin)

**Local Development Install:**

This builds the Rust crate and installs it directly into your current Python environment as a module.

```bash
# Create venv if needed
python3 -m venv .venv
source .venv/bin/activate

# Build & Install
maturin develop --features python
```

**Build Wheel for Release:**

```bash
maturin build --release --features python
```

---

### 🍎 iOS Bindings (UniFFI)

**1. Build Static Libraries:**

```bash
# Device (arm64)
cargo build --release --target aarch64-apple-ios --lib

# Simulator (arm64 - M1/M2/M3)
cargo build --release --target aarch64-apple-ios-sim --lib

# Simulator (x86_64 - Intel)
cargo build --release --target x86_64-apple-ios --lib
```

**2. Generate Swift Bindings:**

This generates the `VitallensCore.swift` file needed by Xcode.

```bash
cargo run --features=uniffi/cli --bin uniffi-bindgen generate \
    --library target/aarch64-apple-ios/release/libvitallens_core.dylib \
    --language swift \
    --out-dir bindings/swift
```

**3. Create XCFramework (Optional Manual Step):**

If you need to bundle the libraries manually:

```bash
xcodebuild -create-xcframework \
    -library target/aarch64-apple-ios/release/libvitallens_core.a \
    -headers bindings/swift \
    -library target/aarch64-apple-ios-sim/release/libvitallens_core.a \
    -headers bindings/swift \
    -output target/VitallensCore.xcframework
```

---

### 🕸️ Web Bindings (Wasm)

**Build for NPM/Bundler:**

```bash
wasm-pack build --target web --no-default-features
```

The output will be generated in the `pkg/` directory, containing the `.wasm` binary and the generated JavaScript glue code.
