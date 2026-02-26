<div align="center">
  <h1>vitallens-core</h1>
  <p align="center">
    <p>The universal Rust engine powering the VitalLens ecosystem.</p>
  </p>

[![Tests](https://github.com/Rouast-Labs/vitallens-core/actions/workflows/ci.yml/badge.svg)](https://github.com/Rouast-Labs/vitallens-core/actions/workflows/ci.yml)
[![Website](https://img.shields.io/badge/Website-rouast.com/api-blue.svg)](https://www.rouast.com/api/)
[![Documentation](https://img.shields.io/badge/Docs-docs.rouast.com-blue.svg)](https://docs.rouast.com/)

</div>

`vitallens-core` is the shared internal signal processing library for [**VitalLens**](https://www.rouast.com/api/). It provides a single Rust codebase that guarantees mathematical parity and high performance across all our client platforms and SDKs.

## Targets

This core compiles directly into the native formats required by our clients:

1. **Apple (iOS/macOS):** Compiled as an `.xcframework` with Swift bindings via UniFFI.
2. **Web (JavaScript/TypeScript):** Compiled to WebAssembly (`.wasm`) via `wasm-pack`.
3. **Python:** Compiled as a native Python extension (`.so`) via PyO3 and Maturin.

## Integration Guides

If you are looking to integrate this core into a specific environment, refer to the dedicated guides:

- [🍎 iOS / Swift Implementation Guide](docs/ios.md)
- [🕸️ Web / JS Implementation Guide](docs/js.md)
- [🐍 Python Implementation Guide](docs/python.md)

## Development

Prerequisites include Rust, Python 3.10+, `wasm-pack`, and the necessary Apple targets. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full setup instructions.

We use a universal `Makefile` to handle building and testing.

```bash
# Run all fast verifications and unit tests
make check

# Build release artifacts for all targets
make build
```

## Disclaimer

The estimates provided by this software are for general wellness purposes only and are not intended for medical use. Always consult with a doctor for any health concerns or for medically precise measurements.

## License

This project is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).
