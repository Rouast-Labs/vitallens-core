.PHONY: default check test check-python check-apple check-web check-android check-jvm build build-python build-apple build-web build-android build-jvm build-jvm-native-macos build-jvm-native-linux bindings-kotlin-jvm dist-apple dist-web dist-python dist-android dist-jvm version-patch version-minor version-major _commit_version clean

# Used by build-jvm-native-linux below to pick a native `cargo build` on a
# real Linux host (CI's ubuntu-latest runner) vs. a Docker-based cross build
# everywhere else (e.g. a macOS dev machine, which has no GNU-compatible
# linker for this target).
UNAME_S := $(shell uname -s)

default: check

# ==========================================
# MINIMAL TIME-INTENSIVE CHECKS
# ==========================================

check: test check-python check-apple check-web check-android check-jvm
	@echo "✅ All fast checks passed!"

test:
	@echo "🧪 Running Cargo Tests..."
	cargo test

check-python:
	@echo "🐍 Checking Python compilation..."
	cargo check --features python

check-apple:
	@echo "🍎 Checking Apple compilation..."
	cargo check --target x86_64-apple-ios --lib
	cargo check --target aarch64-apple-ios-sim --lib
	cargo check --target aarch64-apple-ios --lib
	cargo check --target aarch64-apple-darwin --lib
	cargo check --target x86_64-apple-darwin --lib

check-web:
	@echo "🕸️ Checking Wasm compilation..."
	cargo check --target wasm32-unknown-unknown --no-default-features

check-android:
	@echo "🤖 Checking Android compilation..."
	cargo check --target aarch64-linux-android --lib
	cargo check --target armv7-linux-androideabi --lib
	cargo check --target x86_64-linux-android --lib
	cargo check --target i686-linux-android --lib

check-jvm:
	@echo "☕ Checking JVM/Desktop (Linux) compilation..."
	# Only the linux-x86-64 slice needs checking here — the macOS darwin
	# targets are already covered by check-apple above. `cargo check` never
	# links, so this passes even without a working Linux cross-linker (unlike
	# `make build-jvm-native-linux`, which genuinely needs a real Linux host).
	cargo check --target x86_64-unknown-linux-gnu --lib

# ==========================================
# FULL BUILDS
# ==========================================

build: build-python build-apple build-web build-android build-jvm
	@echo "✅ All release builds complete!"

build-python:
	@echo "🐍 Building Python Wheel..."
	maturin build --release --features python
	@echo "✅ Python Wheel built in target/wheels/"

build-apple:
	@echo "🍎 Building Apple Static Libraries..."
	# iOS Device
	cargo build --release --target aarch64-apple-ios --lib
	# iOS Simulator
	cargo build --release --target x86_64-apple-ios --lib
	cargo build --release --target aarch64-apple-ios-sim --lib
	# macOS
	cargo build --release --target aarch64-apple-darwin --lib
	cargo build --release --target x86_64-apple-darwin --lib
	@echo "🔗 Generating Swift Bindings..."
	cargo run --features=uniffi/cli --bin uniffi-bindgen generate \
		--library target/aarch64-apple-ios/release/libvitallens_core.dylib \
		--language swift \
		--out-dir bindings/swift
	@echo "📦 Fixing Modulemap for SPM..."
	mkdir -p bindings/headers
	cp bindings/swift/*FFI.h bindings/headers/
	cp bindings/swift/*.modulemap bindings/headers/module.modulemap
	@echo "📦 Merging Simulator Binaries (lipo)..."
	mkdir -p target/ios-sim
	lipo -create target/x86_64-apple-ios/release/libvitallens_core.a \
	             target/aarch64-apple-ios-sim/release/libvitallens_core.a \
	     -output target/ios-sim/libvitallens_core.a
	@echo "📦 Merging macOS Binaries (lipo)..."
	mkdir -p target/macos
	lipo -create target/x86_64-apple-darwin/release/libvitallens_core.a \
	             target/aarch64-apple-darwin/release/libvitallens_core.a \
	     -output target/macos/libvitallens_core.a
	@echo "📦 Creating XCFramework..."
	rm -rf target/VitalLensCoreFFI.xcframework
	xcodebuild -create-xcframework \
		-library target/aarch64-apple-ios/release/libvitallens_core.a \
		-headers bindings/headers \
		-library target/ios-sim/libvitallens_core.a \
		-headers bindings/headers \
		-library target/macos/libvitallens_core.a \
		-headers bindings/headers \
		-output target/VitalLensCoreFFI.xcframework
	@echo "✅ Apple Build Complete! -> target/VitalLensCoreFFI.xcframework"

build-web:
	@echo "🕸️ Building Wasm Package..."
	wasm-pack build --target web --no-default-features
	@echo "✅ Web Build Complete! -> pkg/"

build-android:
	@echo "🤖 Building Android Native Libraries..."
	# Uses the `release-android` profile (see Cargo.toml) instead of --release:
	# uniffi-bindgen needs the ELF symbol table that `strip = true` removes.
	cargo ndk -t arm64-v8a -t armeabi-v7a -t x86_64 -t x86 -o kotlin/android/jniLibs build --profile release-android --lib
	@echo "🔗 Generating Kotlin Bindings..."
	cargo run --features=uniffi/cli --bin uniffi-bindgen generate \
		--library target/aarch64-linux-android/release-android/libvitallens_core.so \
		--language kotlin \
		--out-dir bindings/kotlin
	@echo "📦 Assembling AAR..."
	cd kotlin && ./gradlew :android:assembleRelease
	@echo "✅ Android Build Complete! -> kotlin/android/build/outputs/aar/"

# ==========================================
# JVM / Desktop (test/dev-only, com.rouast:vitallens-core-jvm)
#
# Split into per-OS native pieces because the published jar bundles BOTH a
# macOS and a Linux native slice (vitallens-android's real consumer is Linux
# CI). CI builds each slice on its own runner (macos-latest / ubuntu-latest)
# and merges them before packaging (see .github/workflows/release.yml's
# jvm-release job). Locally, build-jvm-native-linux uses `cross` (Docker) to
# reproduce the Linux slice on a macOS dev machine too — see CONTRIBUTING.md
# for one-time setup (Docker + `cross`). `build-jvm` below is the local,
# macOS-only convenience path: darwin slices + bindings + jar, everything a
# Mac dev machine can produce without Docker.
# ==========================================

build-jvm-native-macos:
	@echo "☕ Building JVM/Desktop Native Library (macOS slices)..."
	cargo build --release --target aarch64-apple-darwin --lib
	cargo build --release --target x86_64-apple-darwin --lib
	@echo "📦 Staging native libraries in JNA's desktop resource layout..."
	mkdir -p kotlin/jvm/nativeLibs/darwin-aarch64 kotlin/jvm/nativeLibs/darwin-x86-64
	cp target/aarch64-apple-darwin/release/libvitallens_core.dylib kotlin/jvm/nativeLibs/darwin-aarch64/
	cp target/x86_64-apple-darwin/release/libvitallens_core.dylib kotlin/jvm/nativeLibs/darwin-x86-64/

build-jvm-native-linux:
	@echo "☕ Building JVM/Desktop Native Library (Linux slice)..."
	# Plain `release` profile is fine here (unlike build-android's
	# release-android): uniffi-bindgen never reads this .so — bindings are
	# always generated from a macOS dylib (see bindings-kotlin-jvm) — and
	# `strip = true` only removes .symtab, not the .dynsym exports JNA
	# actually calls at runtime.
	#
	# On a real Linux host (CI's ubuntu-latest runner) this target is just
	# the native host triple, so a plain `cargo build` links directly. On
	# macOS there's no GNU-compatible linker for glibc-Linux output, so we
	# run the same build inside a `cross` (Docker) container instead — same
	# `--target`/`--profile` flags, same output path under target/. See
	# CONTRIBUTING.md for one-time `cross`/Docker setup.
ifeq ($(UNAME_S),Linux)
	cargo build --release --target x86_64-unknown-linux-gnu --lib
else
	cross build --release --target x86_64-unknown-linux-gnu --lib
endif
	@echo "📦 Staging native library in JNA's desktop resource layout..."
	mkdir -p kotlin/jvm/nativeLibs/linux-x86-64
	cp target/x86_64-unknown-linux-gnu/release/libvitallens_core.so kotlin/jvm/nativeLibs/linux-x86-64/

bindings-kotlin-jvm:
	@echo "🔗 Generating Kotlin Bindings..."
	# Reuses the exact same bindgen invocation as build-android — bindings
	# are identical across native targets, this just avoids requiring the
	# Android NDK toolchain to produce them. Mach-O keeps the UNIFFI_META_*
	# marker symbols in the exports trie even under `strip = true` (unlike
	# the stripped-ELF case handled by the release-android profile), so the
	# plain `release` profile works fine here. Requires build-jvm-native-macos
	# to have run first (reads its aarch64-apple-darwin dylib).
	cargo run --features=uniffi/cli --bin uniffi-bindgen generate \
		--library target/aarch64-apple-darwin/release/libvitallens_core.dylib \
		--language kotlin \
		--out-dir bindings/kotlin

# Local, macOS-only convenience target: darwin native slices + bindings + jar.
# The published jar additionally carries a linux-x86-64 slice, assembled only
# in CI (see the note above) — this is enough for local `dist-jvm` testing on
# a Mac dev machine.
build-jvm: build-jvm-native-macos bindings-kotlin-jvm
	@echo "📦 Assembling JVM Jar..."
	cd kotlin && ./gradlew :jvm:jar
	@echo "✅ JVM Build Complete! -> kotlin/jvm/build/libs/"

# ==========================================
# DISTRIBUTION TARGETS
# ==========================================

dist-apple: build-apple
	@echo "📦 Zipping XCFramework..."
	@cd target && rm -f VitalLensCoreFFI.xcframework.zip && zip -yr VitalLensCoreFFI.xcframework.zip VitalLensCoreFFI.xcframework
	@CHECKSUM=$$(swift package compute-checksum target/VitalLensCoreFFI.xcframework.zip); \
	VERSION=$$(grep '^version =' Cargo.toml | head -n 1 | cut -d '"' -f 2); \
	echo "📝 Updating Package.swift: Version v$$VERSION, Checksum $$CHECKSUM"; \
	sed -i '' "s|url: \".*releases/download/.*/VitalLensCoreFFI.xcframework.zip\"|url: \"https://github.com/Rouast-Labs/vitallens-core/releases/download/v$$VERSION/VitalLensCoreFFI.xcframework.zip\"|" Package.swift; \
	sed -i '' "s|checksum: \".*\"|checksum: \"$$CHECKSUM\"|" Package.swift
	@echo "✅ Package.swift updated."

dist-web: build-web
	@echo "📦 Publishing WebAssembly to npm..."
	cd pkg && npm publish && cd ..
	@echo "✅ Published to npm."

dist-python: build-python
	@echo "🐍 Publishing Python Wheel to PyPI..."
	maturin publish --release --features python
	@echo "✅ Published to PyPI."

# NOTE: unlike dist-web (npm) and dist-python (PyPI), dist-android does NOT
# publish to a remote registry reachable by CI or other developers — it only
# installs the AAR into the LOCAL Maven cache (~/.m2) for local verification.
# The real remote Maven Central publish is release automation, done in
# .github/workflows/release.yml's android-release job instead.
dist-android: build-android
	@echo "📦 Publishing AAR to local Maven cache (~/.m2)..."
	cd kotlin && ./gradlew :android:publishToMavenLocal
	@echo "✅ Published to LOCAL Maven cache (~/.m2) only — not a remote registry."

# NOTE: same local-only convention as dist-android above — the real remote
# Maven Central publish is done in .github/workflows/release.yml's
# jvm-release job, which merges this macOS-only local build with a
# separately-built Linux slice before publishing (see build-jvm's note).
dist-jvm: build-jvm
	@echo "📦 Publishing JVM Jar to local Maven cache (~/.m2)..."
	cd kotlin && ./gradlew :jvm:publishToMavenLocal
	@echo "✅ Published to LOCAL Maven cache (~/.m2) only — not a remote registry."

# ==========================================
# VERSIONING WORKFLOW
# ==========================================

version-patch:
	@cargo install cargo-edit > /dev/null 2>&1 || true
	@cargo set-version --bump patch
	@$(MAKE) _commit_version

version-minor:
	@cargo install cargo-edit > /dev/null 2>&1 || true
	@cargo set-version --bump minor
	@$(MAKE) _commit_version

version-major:
	@cargo install cargo-edit > /dev/null 2>&1 || true
	@cargo set-version --bump major
	@$(MAKE) _commit_version

_commit_version: dist-apple
	@VERSION=$$(grep '^version =' Cargo.toml | head -n 1 | cut -d '"' -f 2); \
	git add Cargo.toml Cargo.lock bindings/swift/VitalLensCore.swift Package.swift; \
	git commit -S -m "v$$VERSION"; \
	git tag -a v$$VERSION -m "Release v$$VERSION"; \
	echo "\n✅ Version bumped to v$$VERSION, framework packaged, and commit/tag created."; \
	echo "🚀 Next steps:"; \
	echo "   1. git push origin main --follow-tags"; \
	echo "   2. gh release create v$$VERSION target/VitalLensCoreFFI.xcframework.zip --generate-notes"

# ==========================================
# CLEANUP
# ==========================================

clean:
	cargo clean
	rm -rf target/
	rm -rf pkg/
	# Removes only gitignored/generated content under bindings/ and kotlin/ —
	# leaves bindings/swift/VitalLensCore.swift alone: it's checked into git
	# because SPM resolves this repo directly as its source (Package.swift
	# points `path:` at it), not a downloaded artifact. build-apple always
	# regenerates it fresh anyway.
	git clean -fdX -- bindings/ kotlin/