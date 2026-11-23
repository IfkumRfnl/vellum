use std::{
    collections::BTreeSet,
    env, fs, io,
    path::{Path, PathBuf},
};

fn main() -> io::Result<()> {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        stage_speech_libraries(&profile_dir().join("vellum-libs"), |name| {
            name.ends_with(".so") || name.contains(".so.")
        })?;
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/vellum-libs:$ORIGIN/../vellum-libs");
    }
    #[cfg(windows)]
    if target_os == "windows" {
        // Windows finds DLLs beside the executable; tests run from deps/.
        let profile = profile_dir();
        for dest in [profile.clone(), profile.join("deps")] {
            stage_speech_libraries(&dest, |name| name.ends_with(".dll"))?;
        }
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

/// The Cargo profile directory (`target/release`) the executable lands in: the nearest
/// ancestor of `OUT_DIR` named after the profile, whatever layout the build directory has.
fn profile_dir() -> PathBuf {
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    let profile = env::var("PROFILE").expect("Cargo PROFILE");
    out.ancestors()
        .find(|dir| dir.file_name().is_some_and(|name| name == profile.as_str()))
        .expect("Cargo profile directory")
        .to_path_buf()
}

/// Copies the speech engine's shared libraries to `dest`. A static build has none.
fn stage_speech_libraries(dest: &Path, wanted: impl Fn(&str) -> bool) -> io::Result<()> {
    // Some native installs put the backend modules in a separate directory.
    let sources: BTreeSet<PathBuf> = [
        "DEP_TRANSCRIBE_CPP_RUNTIME_DIR",
        "DEP_TRANSCRIBE_CPP_MODULE_DIR",
    ]
    .into_iter()
    .filter_map(env::var_os)
    .map(PathBuf::from)
    .collect();
    if sources.is_empty() {
        return Ok(());
    }
    fs::create_dir_all(dest)?;
    for dir in sources {
        println!("cargo:rerun-if-changed={}", dir.display());
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let filename = name.to_string_lossy();
            if !wanted(&filename) {
                continue;
            }
            // A running Vellum may still have the previous library loaded: copy beside it, then
            // swap it in. Windows can't replace a loaded DLL, so keep the old one and say so.
            let staging = dest.join(format!(".{filename}.tmp"));
            fs::copy(entry.path(), &staging)?;
            if let Err(error) = fs::rename(&staging, dest.join(&name)) {
                let _ = fs::remove_file(&staging);
                println!("cargo:warning=could not update {filename} (is Vellum running?): {error}");
            }
        }
    }
    Ok(())
}
