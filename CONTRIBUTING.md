# Contributing & Development Guide

This guide covers setting up the development environment, running tests, and building targets for **vitallens-core**.

## Environment Setup

You need the following toolchains installed (assuming macOS):

**1. Rust & Cargo**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

**2. Python (Virtual Env & Maturin)**

```bash
brew install python@3.10
pip3 install maturin
```

**3. WebAssembly (wasm-pack)**

```bash
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

**4. Apple Targets & Tools**

Ensure Xcode Command Line Tools are installed (required for `lipo` and `xcodebuild`). Add the Rust compilation targets:

```bash
rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios aarch64-apple-darwin x86_64-apple-darwin
```

## Build & Test Commands

We use a `Makefile` to automate tasks.

| Command | Description |
| --- | --- |
| `make check` | Runs syntax checks across all targets and standard Rust unit tests. Fast. |
| `make build` | Runs full, optimized release builds for Python, Apple, and Web. |
| `make test` | Runs standard Rust unit tests (`cargo test`). |
| `make check-<target>` | Runs `cargo check` for `<target>` (`python`, `apple`, `web`). |
| `make build-<target>` | Compiles release artifacts for `<target>`. |
| `make clean` | Removes build artifacts (`target/`, `pkg/`, `bindings/swift/`). |

## Detailed Workflows

### Core Rust Logic

To run unit tests manually:

```bash
cargo test
cargo test -- --nocapture # Show print statements
cargo test --test session_tests # Run specific test suite
cargo test test_logic_irregular_rhythm # Test specific function
```

### Python Bindings (PyO3 + Maturin)

To test Python bindings locally without building a release wheel:

```bash
maturin develop --features python
```

This builds and installs the extension directly into your active virtual environment.

### Apple Bindings (UniFFI)

Running `make build-apple` handles compiling for physical devices, Intel/Apple Silicon simulators, merging binaries via `lipo`, generating Swift headers, and packaging the final XCFramework.

### Web Bindings (Wasm)

Running `make build-web` compiles the Wasm binary and generates JS/TS glue code into the `pkg/` directory.

## Release Workflow

To publish a new version (e.g., to update the Swift Package):

1. **Bump Version:** Update the `version` string in `Cargo.toml`.
2. **Prepare Distribution:** Run `make dist-apple`. This zips the framework and automatically updates `Package.swift` with the new checksum and download URL.
3. **Commit & Tag:**
    ```bash
    git add .
    git commit -m "Release x.y.z"
    git tag x.y.z
    git push origin main --tags
    ```
4. **GitHub Release:** Create a new release on GitHub matching the tag and upload `target/VitalLensCoreFFI.xcframework.zip`.
