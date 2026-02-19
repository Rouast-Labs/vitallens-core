.PHONY: default check test check-python check-ios check-web build build-python build-ios build-web clean

default: check

# ==========================================
# MINIMAL TIME-INTENSIVE CHECKS
# ==========================================

check: test check-python check-ios check-web
	@echo "✅ All fast checks passed!"

test:
	@echo "🧪 Running Cargo Tests..."
	cargo test

check-python:
	@echo "🐍 Checking Python compilation..."
	cargo check --features python

check-ios:
	@echo "🍎 Checking iOS compilation..."
	cargo check --target x86_64-apple-ios --lib
	cargo check --target aarch64-apple-ios-sim --lib
	cargo check --target aarch64-apple-ios --lib

check-web:
	@echo "🕸️ Checking Wasm compilation..."
	cargo check --target wasm32-unknown-unknown --no-default-features

# ==========================================
# FULL BUILDS
# ==========================================

build: build-python build-ios build-web
	@echo "✅ All release builds complete!"

build-python:
	@echo "🐍 Building Python Wheel..."
	maturin build --release --features python
	@echo "✅ Python Wheel built in target/wheels/"

build-ios:
	@echo "🍎 Building iOS Static Libraries..."
	cargo build --release --target x86_64-apple-ios --lib
	cargo build --release --target aarch64-apple-ios-sim --lib
	cargo build --release --target aarch64-apple-ios --lib
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
	@echo "📦 Creating XCFramework..."
	rm -rf target/VitalLensCoreFFI.xcframework
	xcodebuild -create-xcframework \
		-library target/aarch64-apple-ios/release/libvitallens_core.a \
		-headers bindings/headers \
		-library target/ios-sim/libvitallens_core.a \
		-headers bindings/headers \
		-output target/VitalLensCoreFFI.xcframework
	@echo "✅ iOS Build Complete! -> target/VitalLensCoreFFI.xcframework"

build-web:
	@echo "🕸️ Building Wasm Package..."
	wasm-pack build --target web --no-default-features
	@echo "✅ Web Build Complete! -> pkg/"


# ==========================================
# UTILITIES
# ==========================================

clean:
	cargo clean
	rm -rf target/
	rm -rf pkg/
	rm -rf bindings/swift/