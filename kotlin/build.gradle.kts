// ==========================================
// Aggregator root — publishes nothing itself.
//
// This project only centralizes plugin *versions* (`apply false`) so both
// subprojects can apply them unversioned:
//   :android -> com.rouast:vitallens-core     (production Android AAR)
//   :jvm     -> com.rouast:vitallens-core-jvm (test/dev-only desktop JVM jar)
//
// Declaring a version here and re-requesting it (even at the same version)
// from a subproject fails, because Gradle multi-project builds share one
// plugin classpath — see the individual subproject build files for where
// each plugin is actually applied.
// ==========================================

plugins {
    id("com.android.library") version "9.1.1" apply false
    id("org.jetbrains.kotlin.jvm") version "2.4.0" apply false
    id("com.vanniktech.maven.publish") version "0.37.0" apply false
}
