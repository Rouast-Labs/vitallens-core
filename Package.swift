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
            url: "https://github.com/Rouast-Labs/vitallens-core/releases/download/v0.1.0/VitalLensCoreFFI.xcframework.zip",
            checksum: "10e83488be377140c03c5f09d80a009b3ebfd3965fab7224848b1038c7f0eec0"
        )
    ]
)