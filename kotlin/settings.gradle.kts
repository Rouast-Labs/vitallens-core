pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "vitallens-core"

// This root project only aggregates shared plugin-version declarations (see
// ../build.gradle.kts) and publishes nothing itself. The two publishable
// artifacts are peer subprojects:
//   :android -> com.rouast:vitallens-core     (production Android AAR)
//   :jvm     -> com.rouast:vitallens-core-jvm (test/dev-only desktop JVM jar)
include(":android", ":jvm")
