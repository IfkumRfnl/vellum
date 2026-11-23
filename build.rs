use std::{collections::BTreeSet, env, fs, io, path::PathBuf};

fn main() -> io::Result<()> {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        stage_linux_speech_libraries()?;
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/vellum-libs:$ORIGIN/../vellum-libs");
    }
    #[cfg(windows)]
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        println!("cargo:rustc-link-lib=delayimp");
        println!("cargo:rustc-link-arg=/DELAYLOAD:sherpa-onnx-c-api.dll");
        println!("cargo:rerun-if-changed=assets/vellum.ico");
        winresource::WindowsResource::new()
            .set_icon("assets/vellum.ico")
            .set("FileDescription", "Vellum")
            .compile()
            .expect("compile Vellum window icon");
    }
    Ok(())
}

fn stage_linux_speech_libraries() -> io::Result<()> {
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    let profile = out.ancestors().nth(3).expect("Cargo profile directory");
    let dest = profile.join("vellum-libs");
    fs::create_dir_all(&dest)?;
    // Some native installs put the backend modules in a separate directory.
    // The runtime discovers them alongside libtranscribe in the final bundle.
    let sources = [
        "DEP_TRANSCRIBE_CPP_RUNTIME_DIR",
        "DEP_TRANSCRIBE_CPP_MODULE_DIR",
    ]
    .map(|key| PathBuf::from(env::var_os(key).expect(key)));
    for dir in BTreeSet::from(sources) {
        println!("cargo:rerun-if-changed={}", dir.display());
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let filename = name.to_string_lossy();
            if filename.ends_with(".so") || filename.contains(".so.") {
                // A running reader may still have the previous library mapped.
                let staging = dest.join(format!(".{filename}.tmp"));
                fs::copy(entry.path(), &staging)?;
                fs::rename(staging, dest.join(&name))?;
            }
        }
    }
    Ok(())
}
