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

`ANDROID_NDK_HOME` above is only for `cargo-ndk`. Gradle itself (`build-android`/`dist-android`) needs `ANDROID_HOME` exported too, and a `kotlin/local.properties` file pointing at it (gitignored, not checked in):

```bash
export ANDROID_HOME="$HOME/Library/Android/sdk"
echo "sdk.dir=$ANDROID_HOME" > kotlin/local.properties
```

You'll also need a JDK to run Gradle (required for `build-android`/`dist-android` and `build-jvm`/`dist-jvm`, not `check-android`). Android Studio's bundled JBR works:

```bash
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
```

**6. JVM/Desktop Linux Cross-Compilation (optional)**

Only needed to reproduce the `linux-x86-64` slice of `com.rouast:vitallens-core-jvm` locally (`make build-jvm-native-linux`) — CI builds it natively on Linux, so this is purely a local convenience. Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or another Docker-compatible daemon) running, since macOS has no GNU-compatible linker for glibc-Linux output and [`cross`](https://github.com/cross-rs/cross) builds inside a Linux container instead:

```bash
rustup target add x86_64-unknown-linux-gnu
cargo install cross --git https://github.com/cross-rs/cross
```

## Build & Test Commands

We use a `Makefile` to automate tasks.

| Command | Description |
| --- | --- |
| `make check` | Runs syntax checks across all targets and standard Rust unit tests. Fast. |
| `make build` | Runs full, optimized release builds for Python, Apple, and Web. |
| `make test` | Runs standard Rust unit tests (`cargo test`). |
| `make check-<target>` | Runs `cargo check` for `<target>` (`python`, `apple`, `web`, `android`, `jvm`). |
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

The `kotlin/` directory is a multi-module Gradle build: an aggregator root plus two subprojects, `:android` (the production AAR) and `:jvm` (see below). Running `make build-android` cross-compiles the four standard Android ABIs (`arm64-v8a`, `armeabi-v7a`, `x86_64`, `x86`) via `cargo-ndk`, generates the Kotlin UniFFI bindings, and assembles them into a real `.aar` via `:android`. `make dist-android` additionally publishes that AAR to your **local** Maven cache (`~/.m2`) — see the note in the Makefile; this does not push anywhere CI or another developer can reach it.

Android builds use the `release-android` Cargo profile (see `Cargo.toml`) instead of the usual `release` profile: `strip = true` removes the ELF symbol table that `uniffi-bindgen` needs to discover the API surface on Linux/Android (Apple's Mach-O format is unaffected by stripping in the same way). This only affects the intermediate AAR's `.so` size — a consuming app's own release build strips native libraries at final packaging as usual.

### JVM/Desktop Bindings (UniFFI + JNA)

`:jvm` publishes `com.rouast:vitallens-core-jvm` — a plain jar bundling the same Kotlin bindings plus native libs laid out for JNA's desktop loader, so `testImplementation("com.rouast:vitallens-core-jvm:...")` lets plain `./gradlew test` runs exercise real Rust logic. Test/dev-only; never use it as a production Android dependency.

`make build-jvm`/`dist-jvm` build only the `darwin-aarch64`/`darwin-x86-64` slices (a macOS dev machine's own targets). The published jar also carries a `linux-x86-64` slice, which CI builds natively on `ubuntu-latest` and merges in (`.github/workflows/release.yml`'s `jvm-native-linux`/`jvm-release` jobs). `make build-jvm-native-linux` reproduces that slice locally via `cross`/Docker — see setup step 6 below. First run pulls a multi-GB image and compiles under x86_64 emulation on Apple Silicon, so it's noticeably slower than the native darwin builds.

### Web Bindings (Wasm)

Running `make build-web` compiles the Wasm binary and generates JS/TS glue code into the `pkg/` directory.

## Release Workflow

We use GitHub Actions to automate publishing to npm, PyPI, and Maven Central (the Android AAR and JVM jar, each gated behind a manual approval). The Apple XCFramework must be built and attached locally to ensure SPM checksums match the git tag — `version-*`/`_commit_version` only build and attach that.

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
