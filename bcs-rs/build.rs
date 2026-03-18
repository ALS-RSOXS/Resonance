fn main() {
    if std::env::var("TARGET").map_or(false, |t| t.contains("windows")) {
        println!("cargo:rustc-link-lib=advapi32");
    }
}
