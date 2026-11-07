// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "VitalLensCore",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "VitalLensCore",
            targets: ["VitalLensCore"]
        )
    ],
    targets: [
        .target(
            name: "VitalLensCore",
            dependencies: ["VitalLensCoreFFI"],
            path: "bindings/swift" 
        ),
        .binaryTarget(
            name: "VitalLensCoreFFI",
            url: "https://github.com/Rouast-Labs/vitallens-core/releases/download/v0.5.1/VitalLensCoreFFI.xcframework.zip",
            checksum: "4d04fe933ef11bfa2d86c3b915065c8dffc0ade4b4ae06b8f6e2f4fe1683c20b"
        )
    ]
)