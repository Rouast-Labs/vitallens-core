# Contributing & Development Guide

This document serves as a cheat sheet for setting up the development environment, running tests, and building targets for **VitalLens Core**.

## Environment Setup

### Prerequisites

Ensure the following toolchains are installed on macOS.

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

**4. iOS Targets & Tools**

Ensure you have Xcode Command Line Tools installed (required for `lipo` and `xcodebuild`). Then add the necessary Rust compilation targets:

```bash
rustup target add aarch64-apple-ios
rustup target add aarch64-apple-ios-sim
rustup target add x86_64-apple-ios
```

---

## Quick Command Reference (Makefile)

We use a `Makefile` to automate common tasks. To keep development fast, the default `make` command only runs syntax checks and tests. Full optimization and linking are reserved for `make build`.

| Command | Description |
| --- | --- |
| `make` | **(Default)** Runs all fast verifications (`cargo check`) and unit tests. |
| `make build` | Runs full, optimized release builds for Python, iOS, and Web. |
| `make test` | Runs standard Rust unit tests (`cargo test`). |
| `make check-<target>` | Runs `cargo check` for a specific target (`python`, `ios`, `web`) without generating artifacts. |
| `make build-<target>` | Compiles the final release artifacts for a specific target (`python`, `ios`, `web`). |
| `make clean` | Removes all build artifacts (`target/`, `pkg/`, `bindings/swift/`). |

---

## Detailed Workflows

### 🦀 Core Rust Logic

**Run Unit Tests:**

```bash
cargo test
# Show print statements (stdout) during tests:
cargo test -- --nocapture
```

**Run Specific Test:**

```bash
cargo test --test session_tests
# Or a specific function
cargo test test_logic_irregular_rhythm
```

**Format & Lint:**

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
```

---

### 🐍 Python Bindings (PyO3 + Maturin)

**Local Development Install:**

This builds the Rust crate and installs it directly into your current active Python virtual environment for immediate testing.

```bash
maturin develop --features python
```

**Build Release Wheel:**

Use the Makefile to generate optimized `.whl` files in `target/wheels/`.

```bash
make build-python
```

---

### 🍎 iOS Bindings (UniFFI)

Building for iOS requires compiling for physical devices, compiling for Intel/Apple Silicon simulators, merging the simulator binaries using `lipo`, generating Swift headers, and packaging an XCFramework.

**Generate the XCFramework:**

Instead of running these manually, rely on the Makefile to handle the entire pipeline:

```bash
make build-ios
```

*Output: `target/VitallensCore.xcframework` and `bindings/swift/VitalLensCore.swift*`

---

### 🕸️ Web Bindings (Wasm)

**Build for NPM/Bundler:**

Use the Makefile to compile the WebAssembly binary and generate the JavaScript/TypeScript glue code.

```bash
make build-web
```

*Output: The `pkg/` directory, functioning as a standard NPM package.*
