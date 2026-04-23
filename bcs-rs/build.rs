fn main() {
    if std::env::var("TARGET").is_ok_and(|t| t.contains("windows")) {
        println!("cargo:rustc-link-lib=advapi32");
    }
}
