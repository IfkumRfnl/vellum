fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // The window and taskbar icon, and the name Windows shows for the executable.
    #[cfg(windows)]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        println!("cargo:rerun-if-changed=assets/vellum.ico");
        winresource::WindowsResource::new()
            .set_icon("assets/vellum.ico")
            .set("FileDescription", "Vellum")
            .compile()
            .expect("compile Vellum window icon");
    }
}
