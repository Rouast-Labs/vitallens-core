.PHONY: default check test check-python check-apple check-web build build-python build-apple build-web dist-apple dist-web dist-python version-patch version-minor version-major _commit_version clean

default: check

# ==========================================
# MINIMAL TIME-INTENSIVE CHECKS
# ==========================================

check: test check-python check-apple check-web
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

# ==========================================
# FULL BUILDS
# ==========================================

build: build-python build-apple build-web
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

# ==========================================
# DISTRIBUTION TARGETS
# ==========================================

dist-apple: build-apple
	@echo "📦 Zipping XCFramework..."
	@cd target && rm -f VitalLensCoreFFI.xcframework.zip && zip -yr VitalLensCoreFFI.xcframework.zip VitalLensCoreFFI.xcframework
	$(eval CHECKSUM=$(shell swift package compute-checksum target/VitalLensCoreFFI.xcframework.zip))
	$(eval VERSION=$(shell grep '^version =' Cargo.toml | head -n 1 | cut -d '"' -f 2))
	@echo "📝 Updating Package.swift: Version v$(VERSION), Checksum $(CHECKSUM)"
	@sed -i '' 's|url: ".*releases/download/.*/VitalLensCoreFFI.xcframework.zip"|url: "https://github.com/Rouast-Labs/vitallens-core/releases/download/v$(VERSION)/VitalLensCoreFFI.xcframework.zip"|' Package.swift
	@sed -i '' 's|checksum: ".*"|checksum: "$(CHECKSUM)"|' Package.swift
	@echo "✅ Package.swift updated."

dist-web: build-web
	@echo "📦 Publishing WebAssembly to npm..."
	cd pkg && npm publish && cd ..
	@echo "✅ Published to npm."

dist-python: build-python
	@echo "🐍 Publishing Python Wheel to PyPI..."
	maturin publish --release --features python
	@echo "✅ Published to PyPI."

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
	$(eval VERSION=$(shell grep '^version =' Cargo.toml | head -n 1 | cut -d '"' -f 2))
	@git add Cargo.toml bindings/swift/VitalLensCore.swift Package.swift
	@git commit -S -m "v$(VERSION)"
	@git tag v$(VERSION)
	@echo "\n✅ Version bumped to v$(VERSION), framework packaged, and commit/tag created."
	@echo "🚀 Next steps:"
	@echo "   1. git push origin main --follow-tags"
	@echo "   2. gh release create v$(VERSION) target/VitalLensCoreFFI.xcframework.zip --generate-notes"

# ==========================================
# CLEANUP
# ==========================================

clean:
	cargo clean
	rm -rf target/
	rm -rf pkg/
	rm -rf bindings/