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
            url: "https://github.com/Rouast-Labs/vitallens-core/releases/download/v0.2.4/VitalLensCoreFFI.xcframework.zip",
            checksum: "925d1a53bb745a5c193b0060bc15300bea3f2de3abd8949b3cd7246f11cde69e"
        )
    ]
)