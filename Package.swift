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
            url: "https://github.com/Rouast-Labs/vitallens-core/releases/download/v0.2.1/VitalLensCoreFFI.xcframework.zip",
            checksum: "9a80d8f18d558c8a92d513978d598f716453f14f39067646a987b08efd3a0b42"
        )
    ]
)