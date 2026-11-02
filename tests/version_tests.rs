use vitallens_core::core_version;

#[test]
fn core_version_is_the_crate_version() {
    let v = core_version();
    assert_eq!(v, env!("CARGO_PKG_VERSION"));
    let parts: Vec<&str> = v.split('.').collect();
    assert_eq!(parts.len(), 3);
    assert!(parts.iter().all(|p| p.parse::<u32>().is_ok()));
}
