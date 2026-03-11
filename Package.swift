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
            url: "https://github.com/Rouast-Labs/vitallens-core/releases/download/v0.2.2/VitalLensCoreFFI.xcframework.zip",
            checksum: "8b04ef6501716899c59a241aa0f4bb8a64fb6977b5305fb53a5882cd1e3544eb"
        )
    ]
)