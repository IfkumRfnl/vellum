//! A static Vulkan build of transcribe.cpp links the Vulkan loader's import library, which the
//! SDK keeps in its own `Lib` folder.

fn main() {
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && let Some(sdk) = std::env::var_os("VULKAN_SDK")
    {
        let lib = std::path::Path::new(&sdk).join("Lib");
        println!("cargo:rustc-link-search=native={}", lib.display());
    }
}
