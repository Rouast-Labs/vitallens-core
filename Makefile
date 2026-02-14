# Makefile for VitalLens Core

.PHONY: all test check-python check-ios check-web clean

# default rule: runs everything
all: test check-python check-ios check-web
	@echo "✅ All checks passed!"

# 1. Standard Rust Tests
test:
	@echo "🧪 Running Cargo Tests..."
	cargo test

# 2. Python Bindings Check
check-python:
	@echo "🐍 Checking Python Build..."
	# We use --dry-run or just build in release mode to ensure it compiles
	# 'maturin develop' installs to current venv, which might be intrusive.
	# 'maturin build' is safer for a check.
	maturin build --features python

# 3. iOS Target Check
check-ios:
	@echo "🍎 Checking iOS Build..."
	cargo build --release --target x86_64-apple-ios --lib

# 4. WebAssembly Check
check-web:
	@echo "🕸️ Checking Wasm Build..."
	wasm-pack build --target web --no-default-features

# Clean up artifacts
clean:
	cargo clean
	rm -rf target/
	rm -rf pkg/
