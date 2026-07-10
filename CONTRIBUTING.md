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
# For Apple Silicon:
rustup target add wasm32-unknown-unknown
```

**5. Android Targets & Tools**

Add the Rust compilation targets for the four standard Android ABIs, and install `cargo-ndk` for cross-compilation:

```bash
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android i686-linux-android
cargo install cargo-ndk
```

Install the NDK into your existing Android SDK (so it's shared with any local Android app checkouts, rather than living in a separate toolchain path) via `sdkmanager`:

```bash
sdkmanager "ndk;28.2.13676358" --sdk_root="$ANDROID_HOME"
export ANDROID_NDK_HOME="$ANDROID_HOME/ndk/28.2.13676358"
```

You'll also need a JDK to run Gradle (only required for `build-android`/`dist-android`, not `check-android`). Android Studio's bundled JBR works:

```bash
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
```

## Build & Test Commands

We use a `Makefile` to automate tasks.

| Command | Description |
| --- | --- |
| `make check` | Runs syntax checks across all targets and standard Rust unit tests. Fast. |
| `make build` | Runs full, optimized release builds for Python, Apple, and Web. |
| `make test` | Runs standard Rust unit tests (`cargo test`). |
| `make check-<target>` | Runs `cargo check` for `<target>` (`python`, `apple`, `web`, `android`). |
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

### Android Bindings (UniFFI + cargo-ndk)

Running `make build-android` cross-compiles the four standard Android ABIs (`arm64-v8a`, `armeabi-v7a`, `x86_64`, `x86`) via `cargo-ndk`, generates the Kotlin UniFFI bindings, and assembles them into a real `.aar` via a minimal Gradle module in `android/`. `make dist-android` additionally publishes that AAR to your **local** Maven cache (`~/.m2`) — see the note in the Makefile; this does not push anywhere CI or another developer can reach it.

Android builds use the `release-android` Cargo profile (see `Cargo.toml`) instead of the usual `release` profile: `strip = true` removes the ELF symbol table that `uniffi-bindgen` needs to discover the API surface on Linux/Android (Apple's Mach-O format is unaffected by stripping in the same way). This only affects the intermediate AAR's `.so` size — a consuming app's own release build strips native libraries at final packaging as usual.

### Web Bindings (Wasm)

Running `make build-web` compiles the Wasm binary and generates JS/TS glue code into the `pkg/` directory.

## Release Workflow

We use GitHub Actions to automate publishing to npm and PyPI. The Apple XCFramework must be built and attached locally to ensure SPM checksums match the git tag. `build-android`/`dist-android` are intentionally **not** part of this automation yet — `version-*`/`_commit_version` only build and attach the Apple XCFramework, so releasing a new version currently does not produce or publish an updated Android AAR.

1. **Pull Latest:** Ensure your local `main` branch is up to date.

```bash
git checkout main
git pull origin main
```

2. **Bump & Build:** Run the appropriate version command. This automatically bumps `Cargo.toml`, builds the Apple framework, updates `Package.swift`, commits the changes, and tags the release.

```bash
make version-patch # or version-minor, version-major
```

3. **Push:** Push the version commit and the new tag.

```bash
git push origin main --follow-tags
```

4. **Publish Release:** Create the GitHub release and upload the local XCFramework zip. (The `make` command will output the exact command for you to copy-paste).

```bash
gh release create vX.Y.Z target/VitalLensCoreFFI.xcframework.zip --generate-notes
```

Once the release is published, CI will automatically build and publish the cross-platform Python wheels to PyPI and the WebAssembly package to npm.
